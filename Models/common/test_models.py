"""Model-contract tests (Phase 4) — every registered model, one contract.

Parametrized over all nine registered names with tiny configs (legal
overrides of each model's ``default_config``) so the suite stays fast.
S4-specific tests pin down the FiLM centering and the linearity of the
recurrence — the two properties the legacy implementation got wrong.

Run from the repo root:
    pytest Models/common/test_models.py
"""
import json

import pytest
import torch
from torch.utils.data import Dataset

import Models.models_import_all  # noqa: F401 — populates registry
from Data.config import RAW_TARGET_DIM
from Models.common.engine import RunConfig, fit
from Models.common.metrics import count_parameters
from Models.common.registry import build_model, list_models
from Models.rnn.rnn import RNNRegressor
from Models.ssm.s4 import FiLMNet, S4Layer, _decay_gate

EXPECTED_MODELS = ['cnn', 'cnn_concat', 'gru', 'mlp',
                   's4', 's4_concat', 's4_modulate', 's4_modulate_biased',
                   's4_modulate_full']

_TINY_S4 = {'d_model': 16, 'd_state': 8, 'n_layers': 2, 'dropout': 0.0,
            'y_module_hidden': 8}
TINY_CONFIGS = {
    'mlp': {'hidden_dims': [16, 8], 'dropout': 0.0},
    'cnn': {'conv_channels': [4, 8], 'head_hidden': 8, 'dropout': 0.0},
    'cnn_concat': {'conv_channels': [4, 8], 'head_hidden': 8, 'dropout': 0.0},
    'gru': {'hidden_size': 16, 'num_layers': 1, 'dropout': 0.0,
            'frame_encoder_channels': [4, 8]},
    's4': _TINY_S4,
    's4_concat': _TINY_S4,
    's4_modulate': _TINY_S4,
    's4_modulate_biased': _TINY_S4,
    's4_modulate_full': _TINY_S4,
}


def make_batch(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 80, 13, 21, generator=g),
            torch.randn(n, generator=g))


def build_tiny(name, seed=0):
    torch.manual_seed(seed)
    return build_model(name, TINY_CONFIGS[name])


def test_registry_contents():
    # subset, not equality: test_engine.py registers test-only models when
    # the whole directory runs in one pytest session
    assert set(EXPECTED_MODELS) <= set(list_models())


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_build_returns_module(name):
    assert isinstance(build_tiny(name), torch.nn.Module)


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_forward_shape_and_dtype(name):
    model = build_tiny(name).eval()
    X, y_module = make_batch()
    out = model(X, y_module)
    assert out.shape == (2, 5)
    assert out.dtype == X.dtype


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_all_params_receive_grad(name):
    model = build_tiny(name).train()
    X, y_module = make_batch()
    model(X, y_module).sum().backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]
    assert dead == [], f"parameters with no grad (dead): {dead}"


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_param_count_positive(name):
    assert count_parameters(build_tiny(name)) > 0


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_determinism_same_seed_same_output(name):
    X, y_module = make_batch()
    out1 = build_tiny(name, seed=7).eval()(X, y_module)
    out2 = build_tiny(name, seed=7).eval()(X, y_module)
    assert torch.equal(out1, out2)


# --- extra coverage for config paths not hit by the registered names -------

def test_cnn_attention_pool():
    cfg = {**TINY_CONFIGS['cnn'], 'temporal_pool': 'attention'}
    torch.manual_seed(0)
    model = build_model('cnn', cfg).train()
    X, y_module = make_batch()
    out = model(X, y_module)
    assert out.shape == (2, 5)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_rnn_class_supports_lstm():
    torch.manual_seed(0)
    model = RNNRegressor(cell='lstm', hidden_size=16, num_layers=1,
                         bidirectional=False, dropout=0.0, use_y_module=True,
                         frame_encoder_channels=[4, 8]).eval()
    X, y_module = make_batch()
    assert model(X, y_module).shape == (2, 5)


# --- S4-specific -----------------------------------------------------------

def test_s4_modulate_at_zero_matches_none():
    """FiLM centering: y_module=0 must reduce 'modulate' to plain S4."""
    X, _ = make_batch(n=4)
    zero = torch.zeros(4)
    out_none = build_tiny('s4', seed=3).eval()(X, zero)
    out_mod = build_tiny('s4_modulate', seed=3).eval()(X, zero)
    assert torch.allclose(out_none, out_mod, atol=1e-5), (
        f"max abs diff {(out_none - out_mod).abs().max().item()}"
    )


def test_s4_modulate_responds_to_y_module():
    model = build_tiny('s4_modulate', seed=3).eval()
    X, _ = make_batch(n=4)
    out_a = model(X, torch.full((4,), -4.0))
    out_b = model(X, torch.full((4,), 4.0))
    assert (out_a - out_b).norm().item() > 1e-3


def test_s4_modulate_film_output_layer_has_no_bias():
    """Regression: the shipped 'modulate' variant stays bias-free."""
    film = build_tiny('s4_modulate', seed=3).film
    assert film.net[-1].bias is None
    assert film.net[0].bias is None


def test_s4_modulate_shared_weights_match_plain_s4():
    """Regression: adding film_bias must not perturb construction order.

    Conditioning modules are built last, so at a fixed seed every parameter
    'modulate' shares with plain S4 must still be bit-identical.
    """
    plain = dict(build_tiny('s4', seed=11).named_parameters())
    mod = dict(build_tiny('s4_modulate', seed=11).named_parameters())
    assert set(plain) <= set(mod)
    for name, p in plain.items():
        assert torch.equal(p, mod[name]), f"parameter '{name}' changed"


def test_s4_modulate_film_is_exactly_odd():
    """Baseline for the ablation: bias-free FiLM is an odd function of y."""
    film = build_tiny('s4_modulate', seed=5).film
    pos = film(torch.tensor([8.0]))[0][0]
    neg = film(torch.tensor([-8.0]))[0][0]
    assert torch.allclose(pos, -neg, atol=1e-6), (
        f"max abs dev from oddness {(pos + neg).abs().max().item()}"
    )


def test_s4_modulate_biased_film_output_layer_has_bias():
    film = build_tiny('s4_modulate_biased', seed=3).film
    assert film.net[-1].bias is not None
    assert film.net[0].bias is None, "hidden layer must stay bias-free"


def test_s4_modulate_biased_film_is_not_odd():
    """The point of the variant: gamma_A(+8) != -gamma_A(-8)."""
    film = build_tiny('s4_modulate_biased', seed=5).film
    pos = film(torch.tensor([8.0]))[0][0]
    neg = film(torch.tensor([-8.0]))[0][0]
    assert not torch.allclose(pos, -neg, atol=1e-6)
    assert (pos + neg).abs().max().item() > 1e-4


def test_s4_modulate_biased_nonzero_at_y_zero():
    """y_module = 0 now emits the learned bias, not exact zero.

    The bias is initialized nonzero by nn.Linear, but pin the mechanism
    down independently of that init by setting a known constant.
    """
    torch.manual_seed(0)
    film = FiLMNet(hidden=8, n_layers=2, d_model=16, film_bias=True)
    with torch.no_grad():
        film.net[-1].bias.fill_(0.25)
    gA = film(torch.zeros(3))[0][0]
    assert torch.allclose(gA, torch.full_like(gA, 0.25))
    # and the unbiased net is still exactly zero there
    torch.manual_seed(0)
    plain = FiLMNet(hidden=8, n_layers=2, d_model=16)
    assert torch.count_nonzero(plain(torch.zeros(3))[0][0]) == 0


def test_s4_modulate_biased_film_bias_receives_grad():
    model = build_tiny('s4_modulate_biased', seed=3).train()
    X, y_module = make_batch()
    model(X, y_module).sum().backward()
    bias = model.film.net[-1].bias
    assert bias.grad is not None
    assert bias.grad.abs().sum().item() > 0


def test_s4_modulate_biased_differs_from_modulate():
    X, y_module = make_batch(n=4)
    out_mod = build_tiny('s4_modulate', seed=3).eval()(X, y_module)
    out_biased = build_tiny('s4_modulate_biased', seed=3).eval()(X, y_module)
    assert (out_mod - out_biased).abs().max().item() > 1e-5


# --- s4_modulate_full: Re(A) decay-rate modulation -------------------------
#
# The four variants above must not move. These constants were captured from
# the implementation at commit 8e457c0, BEFORE the Re(A) gate was added, via
# build_tiny(name, seed=0).eval()(*make_batch()). They are the regression
# anchor for this change: if any of them drifts, s4_modulate_full has leaked
# into a variant it must not touch.
_PRE_CHANGE_OUTPUTS = {
    's4': [0.27322953939437866, -0.01677786558866501, -0.2651069760322571,
           -0.17143961787223816, -0.17656680941581726, 0.32115015387535095,
           -0.041607558727264404, -0.27890390157699585, -0.0959257110953331,
           -0.15190786123275757],
    's4_concat': [-0.0323086678981781, 0.4466997981071472,
                  -0.18785545229911804, 0.03187023103237152,
                  0.06519338488578796, -0.20594936609268188,
                  0.36489662528038025, -0.2707725465297699,
                  -0.07448975741863251, -0.12941953539848328],
    's4_modulate': [0.27829089760780334, -0.017063148319721222,
                    -0.26318418979644775, -0.16899126768112183,
                    -0.17725223302841187, 0.34435874223709106,
                    -0.042595863342285156, -0.27536964416503906,
                    -0.08965548872947693, -0.16194382309913635],
    's4_modulate_biased': [0.19297868013381958, 0.012519896030426025,
                           -0.2630693316459656, -0.17861157655715942,
                           -0.12337416410446167, 0.22424928843975067,
                           0.00147198885679245, -0.278859943151474,
                           -0.12371826171875, -0.09810319542884827],
}


@pytest.mark.parametrize('name', sorted(_PRE_CHANGE_OUTPUTS))
def test_s4_existing_variants_unchanged_by_decay_gate(name):
    """THE regression test: adding s4_modulate_full moved nothing else."""
    X, y_module = make_batch()
    with torch.no_grad():
        out = build_tiny(name, seed=0).eval()(X, y_module).flatten()
    expected = torch.tensor(_PRE_CHANGE_OUTPUTS[name])
    assert torch.equal(out, expected), (
        f"'{name}' drifted from its pre-change output; max abs diff "
        f"{(out - expected).abs().max().item()}"
    )


def test_s4_modulate_full_builds_and_forward_shape():
    model = build_tiny('s4_modulate_full', seed=3).eval()
    X, y_module = make_batch()
    assert model(X, y_module).shape == (2, 5)


def test_s4_modulate_full_film_emits_seven_params():
    """Seventh element is gamma_A_re; the leading six keep their meaning."""
    full = build_tiny('s4_modulate_full', seed=3)
    assert full.modulate_decay is True
    assert full.film.n_params == 7
    assert all(len(t) == 7 for t in full.film(torch.zeros(2)))
    # the variants that must stay on the 6-tuple path
    for name in ('s4_modulate', 's4_modulate_biased'):
        film = build_tiny(name, seed=3).film
        assert film.n_params == 6
        assert all(len(t) == 6 for t in film(torch.zeros(2)))


def test_s4_modulate_full_film_stays_bias_free():
    """This ablation isolates Re(A); it must NOT also relax oddness."""
    film = build_tiny('s4_modulate_full', seed=3).film
    assert film.net[0].bias is None
    assert film.net[-1].bias is None
    pos = film(torch.tensor([8.0]))[0][6]
    neg = film(torch.tensor([-8.0]))[0][6]
    assert torch.allclose(pos, -neg, atol=1e-6), "gamma_A_re must stay odd"


def test_decay_gate_range():
    """0.5 + sigmoid(x) is bounded and strictly positive for any real x.

    Mathematically the gate is in the OPEN interval (0.5, 1.5); in float32,
    sigmoid saturates to exactly 0.0/1.0 once |x| is large, so the realized
    range is the CLOSED [0.5, 1.5]. Both endpoints are strictly positive,
    which is all the stability argument needs — assert the honest bound.
    """
    saturating = torch.tensor([-1e30, -1e4, 1e4, 1e30,
                               float('inf'), float('-inf')])
    moderate = torch.tensor([-8., -1., 0., 1., 8.])
    for gate in (_decay_gate(saturating), _decay_gate(moderate)):
        assert (gate >= 0.5).all() and (gate <= 1.5).all()
        assert (gate > 0).all(), "positivity is the entire stability argument"
    # strictly inside the open interval wherever sigmoid has not saturated
    inner = _decay_gate(moderate)
    assert (inner > 0.5).all() and (inner < 1.5).all()
    assert torch.equal(_decay_gate(torch.zeros(1)), torch.ones(1))


def _re_a_mod(layer, film, y_module, l):
    """Re(A_mod) for layer index ``l``, exactly as S4Layer.forward computes it."""
    A_real = -torch.exp(layer.log_A_real)                    # (H, N_half)
    gA_re = film(y_module)[l][6].float().unsqueeze(-1)       # (B, H, 1)
    return A_real * _decay_gate(gA_re)                       # (B, H, N_half)


def test_s4_modulate_full_re_a_stays_negative_under_adversarial_weights():
    """The safety claim, stress-tested: Re(A_mod) < 0 no matter what.

    Extreme y_module AND extreme FiLM weights — far outside anything training
    would produce — because the whole point of the multiplicative gate is that
    stability is guaranteed by construction, not by the optimizer behaving.
    """
    model = build_tiny('s4_modulate_full', seed=3)
    film = model.film
    with torch.no_grad():
        for m in film.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.uniform_(m.weight, -1000.0, 1000.0)
        # also stress the layers' own decay params away from the -1/2 init
        for layer in model.layers:
            torch.nn.init.uniform_(layer.log_A_real, -20.0, 20.0)

    y_module = torch.linspace(-100.0, 100.0, 50)
    with torch.no_grad():
        for l, layer in enumerate(model.layers):
            A_re_mod = _re_a_mod(layer, film, y_module, l)
            assert torch.isfinite(A_re_mod).all(), f"layer {l}: non-finite Re(A)"
            assert (A_re_mod < 0).all(), (
                f"layer {l}: Re(A_mod) reached {A_re_mod.max().item()} >= 0 — "
                f"the recurrence would be unstable"
            )

    # and end to end: an unstable Re(A) would blow the 80-step scan up
    X, _ = make_batch(n=4)
    with torch.no_grad():
        out = model.eval()(X, torch.full((4,), 100.0))
    assert torch.isfinite(out).all()


def test_s4_modulate_full_re_a_responds_to_y_module():
    """The gate is wired to learnable params, not a no-op."""
    model = build_tiny('s4_modulate_full', seed=3)
    layer, film = model.layers[0], model.film
    with torch.no_grad():
        pos = _re_a_mod(layer, film, torch.tensor([8.0]), 0)
        neg = _re_a_mod(layer, film, torch.tensor([-8.0]), 0)
    assert (pos - neg).norm().item() > 1e-3, (
        f"Re(A_mod) barely moved between y=+8 and y=-8: "
        f"{(pos - neg).norm().item()}"
    )


def test_s4_modulate_full_at_zero_matches_none():
    """Oddness is preserved, so the identity-at-midplane property survives.

    gamma_A_re(0) = 0 -> gate = 0.5 + sigmoid(0) = 1 -> Re(A_mod) = Re(A).
    """
    model = build_tiny('s4_modulate_full', seed=3)
    layer, film = model.layers[0], model.film
    with torch.no_grad():
        A_re_mod = _re_a_mod(layer, film, torch.zeros(4), 0)
        assert torch.equal(film(torch.zeros(4))[0][6],
                           torch.zeros(4, film.d_model))
        assert torch.allclose(A_re_mod, -torch.exp(layer.log_A_real))

    X, _ = make_batch(n=4)
    zero = torch.zeros(4)
    out_none = build_tiny('s4', seed=3).eval()(X, zero)
    out_full = model.eval()(X, zero)
    assert torch.allclose(out_none, out_full, atol=1e-5), (
        f"max abs diff {(out_none - out_full).abs().max().item()}"
    )


def test_s4_modulate_full_gamma_a_re_receives_grad():
    """The gamma_A_re slice of the output projection is on the compute path."""
    model = build_tiny('s4_modulate_full', seed=3).train()
    X, y_module = make_batch()
    model(X, y_module).sum().backward()
    w = model.film.net[-1].weight
    assert w.grad is not None
    n_layers, d_model = model.film.n_layers, model.film.d_model
    # rows are laid out as (layer, param_index, channel); gamma_A_re is k = 6
    rows = [l * 7 * d_model + 6 * d_model + j
            for l in range(n_layers) for j in range(d_model)]
    gA_re_grad = w.grad[rows]
    assert gA_re_grad.abs().sum().item() > 0, "gamma_A_re rows got zero grad"


def test_s4_modulate_full_differs_from_modulate():
    X, y_module = make_batch(n=4)
    out_mod = build_tiny('s4_modulate', seed=3).eval()(X, y_module)
    out_full = build_tiny('s4_modulate_full', seed=3).eval()(X, y_module)
    assert (out_mod - out_full).abs().max().item() > 1e-5


def test_s4_modulate_full_shared_weights_match_plain_s4():
    """Conditioning built last: the wider FiLM net must not shift the seed."""
    plain = dict(build_tiny('s4', seed=11).named_parameters())
    full = dict(build_tiny('s4_modulate_full', seed=11).named_parameters())
    assert set(plain) <= set(full)
    for name, p in plain.items():
        assert torch.equal(p, full[name]), f"parameter '{name}' changed"


def test_s4_layer_ignores_seventh_param_when_flag_off():
    """Explicit-flag dispatch: arity comes from the flag, never from len(mod).

    A 7-tuple with modulate_decay=False is a caller bug and must fail loudly
    rather than silently modulating Re(A) — the failure mode that implicit
    tuple-length dispatch would have hidden.
    """
    torch.manual_seed(0)
    layer = S4Layer(d_model=8, d_state=4).eval()
    u = torch.randn(2, 80, 8)
    six = tuple(torch.zeros(2, 8) for _ in range(6))
    seven = six + (torch.zeros(2, 8),)
    assert torch.allclose(layer(u, six), layer(u), atol=1e-6)
    with pytest.raises(ValueError):
        layer(u, seven)
    with pytest.raises(ValueError):
        layer(u, six, modulate_decay=True)


def test_s4_layer_is_linear():
    """Doubling the input doubles the output — no nonlinearity in the scan."""
    torch.manual_seed(0)
    layer = S4Layer(d_model=8, d_state=4).eval()
    u = torch.randn(2, 80, 8)
    y1 = layer(u)
    y2 = layer(2.0 * u)
    assert torch.allclose(y2, 2.0 * y1, atol=1e-5), (
        f"max abs dev from linearity "
        f"{(y2 - 2.0 * y1).abs().max().item()}"
    )


# --- engine integration ----------------------------------------------------

class SyntheticDataset(Dataset):
    """Emits (X (80,13,21), y_module scalar, Y (6,)) like PixelClusterDataset."""

    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 80, 13, 21, generator=g)
        self.y_module = torch.randn(n, generator=g)
        self.Y = torch.randn(n, RAW_TARGET_DIM, generator=g)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y_module[i], self.Y[i]


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_fit_two_epochs_produces_artifacts(name, tmp_path):
    cfg = RunConfig(
        model_name=name,
        model_config=TINY_CONFIGS[name],
        checkpoint_dir=str(tmp_path / 'ckpt'),
        batch_size=16,
        max_epochs=2,
        num_workers=0,
        early_stop_patience=0,
        data_dir=str(tmp_path),  # unused: datasets are injected
    )
    fit(cfg, SyntheticDataset(48, seed=0), SyntheticDataset(32, seed=1),
        SyntheticDataset(32, seed=2))
    ckpt_dir = tmp_path / 'ckpt'
    assert (ckpt_dir / 'best.pt').exists()
    assert (ckpt_dir / 'latest.pt').exists()
    with open(ckpt_dir / 'final_report.json') as f:
        report = json.load(f)
    assert len(report['table_row']['per_target_mse']) == 5
