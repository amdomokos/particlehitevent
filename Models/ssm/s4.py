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

Parameterization per layer (H = d_model channels, N = d_state per channel):
  - Continuous-time diagonal ``A = -exp(log_A_real) + 1j * A_imag`` so
    eigenvalues always have negative real part (stable by construction).
  - Learnable per-channel timescale ``dt = exp(log_dt)``; ZOH
    discretization ``dA = exp(dt * A)``, ``dB = (dA - 1) / A * B`` maps the
    stable continuous system to a stable discrete one (|dA| < 1).
  - Real trainable B and C; ``y_t = Re(C · h_t)``.

Init (arXiv:2206.11893 §4): Re(A) = -1/2 (constant, S4D-Lin), Im(A) = π·n.
``dt`` log-uniform over [1e-3, 1e-1].

Conditioning on y_module (continuous scalar — never embedded as an ID):
  - 'concat':   project y_module, concat to the pooled representation.
  - 'modulate': FiLM-style per-layer, per-channel γ/β on A, B, C, e.g.
    ``A_mod = A * (1 + γ_A) + β_A``. The modulation net is bias-free with an
    odd activation, so ``y_module = 0`` yields γ = β = 0 exactly and the
    model reduces to plain S4 (the ``1 +`` centering; verified in tests).
    γ_A, β_A act on Im(A) only — Re(A) is fixed to preserve unconditional
    stability under conditioning.
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
}

_T, _HH, _WW = 80, 13, 21
_DT_MIN, _DT_MAX = 1e-3, 1e-1


class S4Layer(nn.Module):
    """One diagonal SSM: (B, T, H) -> (B, T, H), linear in the input.

    ``mod``, if given, is a 6-tuple (γ_A, β_A, γ_B, β_B, γ_C, β_C) of (B, H)
    tensors applied FiLM-style to the continuous-time A and to B, C before
    discretization; broadcast over the state dimension N.
    """

    def __init__(self, d_model, d_state):
        super().__init__()
        H, N = d_model, d_state
        self.log_dt = nn.Parameter(
            torch.empty(H).uniform_(math.log(_DT_MIN), math.log(_DT_MAX)))
        self.log_A_real = nn.Parameter(
            torch.full((H, N), math.log(0.5)))
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(N, dtype=torch.float32).repeat(H, 1))
        self.B = nn.Parameter(torch.ones(H, N))
        self.C = nn.Parameter(torch.randn(H, N) / math.sqrt(N))

    def forward(self, u, mod=None):
        u = u.float()  # recurrence always in fp32/complex64, even under AMP
        Bsz, T, H = u.shape
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)  # (H, N)
        Bp, Cp = self.B, self.C
        if mod is not None:
            gA, bA, gB, bB, gC, bC = (m.float().unsqueeze(-1) for m in mod)
            A_im_mod = A.imag * (1.0 + gA) + bA
            A = torch.complex(A.real, A_im_mod)  # Re(A) fixed for stability
            Bp = Bp * (1.0 + gB) + bB
            Cp = Cp * (1.0 + gC) + bC
        dt = torch.exp(self.log_dt).unsqueeze(-1)        # (H, 1)
        z = A * dt                                       # dt * A
        dA = torch.polar(torch.exp(z.real), z.imag)      # exp(dt*A), |dA|<1
        dB = (dA - 1.0) / z * (dt * Bp)                  # ZOH input matrix

        h = torch.zeros(Bsz, H, A.shape[-1], dtype=dA.dtype, device=u.device)
        ys = []
        for t in range(T):
            h = dA * h + dB * u[:, t].unsqueeze(-1)      # linear — no tanh!
            ys.append((h.real * Cp).sum(-1))             # Re(C·h), C real
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
    """

    def __init__(self, hidden, n_layers, d_model):
        super().__init__()
        self.n_layers, self.d_model = n_layers, d_model
        self.net = nn.Sequential(
            nn.Linear(1, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, n_layers * 6 * d_model, bias=False),
        )

    def forward(self, y_module):
        out = self.net(y_module.float().unsqueeze(1))
        out = out.view(-1, self.n_layers, 6, self.d_model)
        return [tuple(out[:, l, k] for k in range(6))
                for l in range(self.n_layers)]


class S4Backbone(nn.Module):
    """Frame encoder -> n_layers S4 blocks -> mean pool over T -> head.

    Block structure: S4Layer -> GELU -> Dropout -> residual -> LayerNorm.
    Conditioning modules are constructed LAST so that, at a fixed seed,
    the shared weights of 'none' and 'modulate' models are identical.
    """

    def __init__(self, d_model, d_state, n_layers, dropout, conditioning,
                 y_module_hidden):
        super().__init__()
        if conditioning not in ('none', 'concat', 'modulate'):
            raise ValueError(f"conditioning must be 'none' | 'concat' | "
                             f"'modulate', got '{conditioning}'")
        self.conditioning = conditioning
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
        self.film = (FiLMNet(y_module_hidden, n_layers, d_model)
                     if conditioning == 'modulate' else None)

    def forward(self, X, y_module):
        x = self.encoder(X)
        mods = (self.film(y_module) if self.film is not None
                else [None] * len(self.layers))
        for layer, norm, mod in zip(self.layers, self.norms, mods):
            y = self.dropout(F.gelu(layer(x, mod)))
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


def _build(config, conditioning):
    cfg = _merge(config)
    cfg['conditioning'] = conditioning
    return S4Backbone(cfg['d_model'], cfg['d_state'], cfg['n_layers'],
                      cfg['dropout'], cfg['conditioning'],
                      cfg['y_module_hidden'])


@register_model('s4')
def build_s4(config):
    return _build(config, conditioning='none')


@register_model('s4_concat')
def build_s4_concat(config):
    return _build(config, conditioning='concat')


@register_model('s4_modulate')
def build_s4_modulate(config):
    return _build(config, conditioning='modulate')
