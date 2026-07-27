"""Diagonal state-space model (S4D) backbone (Phase 4).

Implements the diagonal SSM parameterization of Gu, Gupta, Goel, Ré (2022),
"On the Parameterization and Initialization of Diagonal State Space Models",
NeurIPS 2022 (arXiv:2206.11893). S4D — not full S4 — is the standard object
of study in the SSM quantization literature (e.g. Quamba2, arXiv:2503.22879),
which motivates it as the backbone for a quantization-focused comparison.

Core contract of ``S4Layer`` (the previous codebase violated both):
  - The recurrence is LINEAR — no nonlinearity inside the scan. GELU lives
    between layers, in the block wrapper.
  - B and C are both trainable and both on the compute path (the old code
    had a dead ``self.C``).

Parameterization per layer (H = d_model channels; ``d_state`` is the
EFFECTIVE REAL SSM state size per channel). Following §3.3 ("Conjugate
Symmetry") and Listing 1 of the paper, only ``N_half = d_state // 2``
independent complex eigenvalues are stored; their conjugates are implicit,
so the real state capacity is ``d_state``. ``d_state`` must be even.
  - Continuous-time diagonal ``A = -exp(log_A_real) + 1j * A_imag`` so
    eigenvalues always have negative real part (stable by construction).
  - Learnable per-channel timescale ``dt = exp(log_dt)``; ZOH
    discretization ``dA = exp(dt * A)``, ``dB = (dA - 1) / A * B`` maps the
    stable continuous system to a stable discrete one (|dA| < 1).
  - Real trainable B, COMPLEX trainable C (stored as ``C_real``/``C_imag``);
    ``y_t = 2 · Re(Σ_n C_n · h_{t,n})``, the conjugate-pair sum.

Init (arXiv:2206.11893 §4): Re(A) = -1/2 constant (S4D-Lin, Eq. 9),
Im(A) = π·n. ``dt`` log-uniform over [1e-3, 1e-1].

Conditioning on y_module (continuous scalar — never embedded as an ID):
  - 'concat':   project y_module, concat to the pooled representation.
  - 'modulate': FiLM-style per-layer, per-channel γ/β on A, B, C, e.g.
    ``A_mod = A * (1 + γ_A) + β_A``. The modulation net is bias-free with an
    odd activation, so ``y_module = 0`` yields γ = β = 0 exactly and the
    model reduces to plain S4 (the ``1 +`` centering; verified in tests).
    γ_A, β_A act on Im(A) only — Re(A) is fixed to preserve unconditional
    stability under conditioning.
  - 'modulate' + ``film_bias=True`` (registered as ``s4_modulate_biased``):
    identical to 'modulate' except the FiLM net's OUTPUT projection carries a
    bias, so the γ/β it emits are no longer an odd function of y_module. The
    bias-free 'modulate' net is provably odd — measured in production,
    ``corr(γ_A(+8), -γ_A(-8)) = 1.0000`` — which forces the conditioning to be
    exactly antisymmetric about the detector midplane. If the true
    y_module-to-charge-dynamics relationship is not antisymmetric, that is a
    hard constraint rather than an inductive bias, and s4_modulate did in fact
    underperform s4_concat on real data (MSE 11.31 vs 10.23). The bias buys
    that expressiveness at the cost of the exact-identity-at-zero property:
    y_module = 0 now yields γ = β = <learned bias>, not 0, so the biased
    variant does NOT reduce to plain S4 at the midplane. That is the intended
    trade-off of this ablation, not a bug.
  - 'modulate' + ``modulate_decay=True`` (registered as ``s4_modulate_full``):
    the original odd, bias-free FiLM net — NOT combined with ``film_bias`` —
    extended with a seventh emitted parameter per layer/channel, γ_A_re, that
    modulates Re(A) as well as Im(A). The gate is multiplicative and bounded,
    ``Re(A_mod) = Re(A) * (0.5 + sigmoid(γ_A_re))``, so the multiplier lies in
    (0.5, 1.5): strictly positive for every real γ_A_re, hence the sign of
    Re(A) is preserved and ``Re(A_mod) < 0`` holds unconditionally. That is
    why an additive shift is NOT used here — an unconstrained β on Re(A) could
    drive it positive and destabilize the recurrence, which is exactly the risk
    that motivated fixing Re(A) in 'modulate' in the first place. Because the
    net stays odd, γ_A_re(0) = 0 and the gate becomes ``0.5 + sigmoid(0) = 1``,
    so y_module = 0 still reduces this variant to plain S4 exactly — the
    identity property s4_modulate has and s4_modulate_biased gives up.

    This variant tests whether the Re(A) restriction, necessary for stability
    but never free, was ALSO costing expressiveness. It follows the pattern of
    the ablations above: on real data each successive relaxation of the
    modulation constraints moved performance toward s4_concat
    (s4 12.21 -> s4_modulate 11.31 -> s4_modulate_biased 11.01, vs
    s4_concat 10.23), with s4_modulate_biased relaxing the oddness constraint
    and this variant relaxing the fixed-decay constraint. The two relaxations
    are deliberately kept separate so each is measured in isolation.
    ``s4_modulate_full`` is the final variant in this line of investigation;
    no further modulation ablations are planned.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Data.config import ACTIVE_TARGET_DIM
from Models.common.registry import register_model

default_config = {
    'd_model': 128,
    'd_state': 64,
    'n_layers': 4,
    'dropout': 0.1,
    'conditioning': 'none',     # 'none' | 'concat' | 'modulate'
    'y_module_hidden': 32,
    # Like 'conditioning', these are fixed by the registered variant and any
    # value passed in a user config is overwritten by _build. They live in
    # default_config only so _merge's unknown-key rejection accepts them.
    'film_bias': False,
    'modulate_decay': False,
}

_T, _HH, _WW = 80, 13, 21
_DT_MIN, _DT_MAX = 1e-3, 1e-1


def _decay_gate(gA_re):
    """Bounded multiplier for Re(A): ``0.5 + sigmoid(x)``, always in (0.5, 1.5).

    In float32 the realized range is the closed [0.5, 1.5] — sigmoid saturates
    to exactly 0.0/1.0 for large |x|, including ±inf — but both endpoints are
    strictly positive, so multiplying Re(A) by this can never change Re(A)'s
    sign. That is the whole stability argument for ``s4_modulate_full``; it is
    a named function so the tests can assert the bound against the exact
    expression the forward pass uses.
    """
    return 0.5 + torch.sigmoid(gA_re)


class S4Layer(nn.Module):
    """One diagonal SSM: (B, T, H) -> (B, T, H), linear in the input.

    A and B are per-channel (untied), matching Gu et al.'s main-text
    parameterization (Table 3b) rather than the tied variant in Appendix B.

    The state axis holds ``N_half = d_state // 2`` complex modes; the
    conjugate half is implicit. The output is ``2 * Re(C * h)`` summed over
    the state axis, which implements the conjugate-pair sum that guarantees
    a real output.

    ``mod``, if given, is a 6-tuple (γ_A, β_A, γ_B, β_B, γ_C, β_C) of (B, H)
    tensors applied FiLM-style to the continuous-time A and to B, C before
    discretization; broadcast over the state dimension N_half. γ_A/β_A act on
    Im(A) only.

    ``modulate_decay=True`` makes ``mod`` a 7-tuple, the extra element being
    γ_A_re, a bounded multiplicative gate on Re(A) (see module docstring).
    The arity of ``mod`` is dispatched on this explicit flag rather than on
    ``len(mod)``: implicit length dispatch would silently change behaviour if
    a caller ever emitted the wrong tuple width, and the flag makes the active
    mode visible at the call site instead of inferred from the data.
    """

    def __init__(self, d_model, d_state):
        super().__init__()
        assert d_state % 2 == 0, \
            f"d_state must be even for conjugate-pair parameterization; got {d_state}"
        H, N_half = d_model, d_state // 2
        self.log_dt = nn.Parameter(
            torch.empty(H).uniform_(math.log(_DT_MIN), math.log(_DT_MAX)))
        # Re(A) = -1/2 constant (S4D-Lin, Gu et al. 2022 Eq. 9)
        self.log_A_real = nn.Parameter(
            torch.full((H, N_half), math.log(0.5)))
        # Im(A) = pi * n for n = 0, ..., N_half - 1  (S4D-Lin)
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(N_half, dtype=torch.float32).repeat(H, 1))
        self.B = nn.Parameter(torch.ones(H, N_half))
        # C is complex: two real parameters. Init scale matches the previous
        # 1/sqrt(N) for output variance; N here means d_state, not N_half.
        c_scale = 1.0 / math.sqrt(d_state)
        self.C_real = nn.Parameter(torch.randn(H, N_half) * c_scale)
        self.C_imag = nn.Parameter(torch.randn(H, N_half) * c_scale)

    def forward(self, u, mod=None, modulate_decay=False):
        u = u.float()  # recurrence always in fp32/complex64, even under AMP
        Bsz, T, H = u.shape
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)  # (H, N_half)
        Bp = self.B
        C_complex = torch.complex(self.C_real, self.C_imag)          # (H, N_half)
        if mod is not None:
            if modulate_decay:
                gA, bA, gB, bB, gC, bC, gA_re = (
                    m.float().unsqueeze(-1) for m in mod)
            else:
                gA, bA, gB, bB, gC, bC = (m.float().unsqueeze(-1) for m in mod)
                gA_re = None
            A_im_mod = A.imag * (1.0 + gA) + bA
            if gA_re is None:
                # Re(A) fixed — the stability-preserving default
                A_re_mod = A.real
            else:
                # Bounded multiplicative gate in (0.5, 1.5): strictly
                # positive, so sign(Re(A)) — always negative, since
                # Re(A) = -exp(...) — survives any value the FiLM net emits.
                # An additive shift could not make that guarantee.
                A_re_mod = A.real * _decay_gate(gA_re)
            A = torch.complex(A_re_mod, A_im_mod)
            Bp = Bp * (1.0 + gB) + bB
            C_complex = C_complex * (1.0 + gC) + bC   # bC broadcasts as real shift
        dt = torch.exp(self.log_dt).unsqueeze(-1)        # (H, 1)
        z = A * dt                                       # dt * A
        dA = torch.polar(torch.exp(z.real), z.imag)      # exp(dt*A), |dA|<1
        dB = (dA - 1.0) / z * (dt * Bp)                  # ZOH input matrix

        h = torch.zeros(Bsz, H, A.shape[-1], dtype=dA.dtype, device=u.device)
        ys = []
        for t in range(T):
            h = dA * h + dB * u[:, t].unsqueeze(-1)      # linear — no tanh!
            ys.append(2.0 * (C_complex * h).real.sum(-1))  # conjugate-pair sum
        return torch.stack(ys, dim=1)                    # (B, T, H)


class FrameEncoder(nn.Module):
    """Weight-shared 2-layer strided CNN per frame: (B, 80, 13, 21) -> (B, 80, D)."""

    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 13x21 -> 7x11
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x11 -> 4x6
            nn.GELU(),
        )
        self.proj = nn.Linear(64 * 4 * 6, d_model)

    def forward(self, X):
        B = X.shape[0]
        z = self.convs(X.reshape(B * _T, 1, _HH, _WW))
        return self.proj(z.reshape(B, _T, -1))


class FiLMNet(nn.Module):
    """y_module (B,) -> per-layer 6-tuples of (B, H) γ/β for A, B, C.

    Bias-free Linear layers with an odd activation (tanh) so that
    y_module = 0 maps to exactly zero modulation — required for the
    ``1 + γ`` centering to reduce to plain S4.

    ``film_bias=True`` (default False, which reproduces the above exactly)
    adds a bias to the OUTPUT projection only. The hidden layer stays
    bias-free, so the hidden code is still ``tanh(W y)`` and still zero at
    y = 0, but the output becomes ``W2 tanh(W1 y) + b`` — no longer odd, and
    no longer zero at y = 0. See the module docstring for the motivation.

    ``modulate_decay=True`` (default False) widens the output projection to
    emit a SEVENTH parameter per layer/channel, γ_A_re, the Re(A) gate. The
    tuples yielded per layer are then 7 long, with γ_A_re last so the leading
    six keep their meaning. The two flags are independent, but the registered
    ``s4_modulate_full`` variant deliberately uses ``modulate_decay=True``
    with ``film_bias=False``: the Re(A) gate is tested against the original
    odd, bias-free net so that this ablation isolates the decay relaxation.
    """

    def __init__(self, hidden, n_layers, d_model, film_bias=False,
                 modulate_decay=False):
        super().__init__()
        self.n_layers, self.d_model = n_layers, d_model
        self.n_params = 7 if modulate_decay else 6
        self.net = nn.Sequential(
            nn.Linear(1, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, n_layers * self.n_params * d_model,
                      bias=film_bias),
        )

    def forward(self, y_module):
        out = self.net(y_module.float().unsqueeze(1))
        out = out.view(-1, self.n_layers, self.n_params, self.d_model)
        return [tuple(out[:, l, k] for k in range(self.n_params))
                for l in range(self.n_layers)]


class S4Backbone(nn.Module):
    """Frame encoder -> n_layers S4 blocks -> mean pool over T -> head.

    Block structure: S4Layer -> GELU -> Dropout -> residual -> LayerNorm.
    Conditioning modules are constructed LAST so that, at a fixed seed,
    the shared weights of 'none' and 'modulate' models are identical.
    """

    def __init__(self, d_model, d_state, n_layers, dropout, conditioning,
                 y_module_hidden, film_bias=False, modulate_decay=False):
        super().__init__()
        if conditioning not in ('none', 'concat', 'modulate'):
            raise ValueError(f"conditioning must be 'none' | 'concat' | "
                             f"'modulate', got '{conditioning}'")
        self.conditioning = conditioning
        # only meaningful under 'modulate'; kept False elsewhere so the flag
        # passed to S4Layer.forward is never ambiguous
        self.modulate_decay = modulate_decay and conditioning == 'modulate'
        self.encoder = FrameEncoder(d_model)
        self.layers = nn.ModuleList(
            S4Layer(d_model, d_state) for _ in range(n_layers))
        self.norms = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        head_in = d_model + (y_module_hidden if conditioning == 'concat' else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, ACTIVE_TARGET_DIM),
        )
        # conditioning modules last — see class docstring
        self.y_proj = (nn.Linear(1, y_module_hidden)
                       if conditioning == 'concat' else None)
        self.film = (FiLMNet(y_module_hidden, n_layers, d_model, film_bias,
                             self.modulate_decay)
                     if conditioning == 'modulate' else None)

    def forward(self, X, y_module):
        x = self.encoder(X)
        mods = (self.film(y_module) if self.film is not None
                else [None] * len(self.layers))
        for layer, norm, mod in zip(self.layers, self.norms, mods):
            y = self.dropout(F.gelu(layer(x, mod, self.modulate_decay)))
            x = norm(x + y)
        pooled = x.mean(dim=1)                           # (B, d_model)
        if self.conditioning == 'concat':
            y_feat = F.gelu(self.y_proj(y_module.to(pooled.dtype).unsqueeze(1)))
            pooled = torch.cat([pooled, y_feat], dim=1)
        return self.head(pooled)


def _merge(config):
    cfg = {**default_config, **(config or {})}
    unknown = set(cfg) - set(default_config)
    if unknown:
        raise ValueError(f"unknown s4 config keys: {sorted(unknown)}")
    return cfg


def _build(config, conditioning, film_bias=False, modulate_decay=False):
    cfg = _merge(config)
    cfg['conditioning'] = conditioning
    cfg['film_bias'] = film_bias
    cfg['modulate_decay'] = modulate_decay
    return S4Backbone(cfg['d_model'], cfg['d_state'], cfg['n_layers'],
                      cfg['dropout'], cfg['conditioning'],
                      cfg['y_module_hidden'], cfg['film_bias'],
                      cfg['modulate_decay'])


@register_model('s4')
def build_s4(config):
    return _build(config, conditioning='none')


@register_model('s4_concat')
def build_s4_concat(config):
    return _build(config, conditioning='concat')


@register_model('s4_modulate')
def build_s4_modulate(config):
    return _build(config, conditioning='modulate')


@register_model('s4_modulate_biased')
def build_s4_modulate_biased(config):
    """'modulate' with a biased FiLM output projection — relaxed oddness.

    Identical to ``s4_modulate`` in every respect except that the FiLM net's
    final Linear carries a bias, so γ/β are no longer an odd function of
    y_module and are no longer exactly zero at y_module = 0. The biased
    variant therefore does not reduce to plain S4 at the midplane; see the
    module docstring for why that property is deliberately given up here.
    """
    return _build(config, conditioning='modulate', film_bias=True)


@register_model('s4_modulate_full')
def build_s4_modulate_full(config):
    """'modulate' extended to Re(A) — the decay rate — as well as Im(A).

    Identical to ``s4_modulate`` (odd, bias-free FiLM net; ``film_bias`` is
    deliberately NOT enabled here) except that the net emits a seventh
    parameter γ_A_re per layer/channel, gating Re(A) by
    ``0.5 + sigmoid(γ_A_re)`` ∈ (0.5, 1.5). The multiplier is strictly
    positive for every real input, so Re(A) keeps its sign and the recurrence
    stays stable no matter what the net learns — the guarantee an additive
    shift on Re(A) could not give, which is why 'modulate' froze Re(A) at all.
    Oddness also means γ_A_re(0) = 0, so the gate is exactly 1 and this
    variant still reduces to plain S4 at y_module = 0.
    """
    return _build(config, conditioning='modulate', modulate_decay=True)
