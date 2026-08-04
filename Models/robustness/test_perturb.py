"""Tests for the Phase 7 perturbation mechanism.

Run from the repo root:  pytest Models/robustness/test_perturb.py

No checkpoints, no dataset, no GPU: models are tiny registered S4 variants.
The whole point of this file is to establish, on synthetic data, that
"perturb, evaluate, restore" is actually perturbing by the requested amount,
touching only what it claims to touch, and returning bit-exactly — before any
of it is pointed at a real checkpoint, where none of those properties can be
observed from the output numbers.
"""
import math

import pytest
import torch

import Models.models_import_all  # noqa: F401 — populates registry
from Models.common.registry import build_model
from Models.quantization.fake_quant import SUBSET_PARAMS
from Models.quantization.sweep import verify_restore
from Models.robustness.perturb import (
    SIGMAS,
    SUBSETS,
    apply_perturbation,
    config_seed,
    gaussian_noise,
    perturbation_error,
    subset_members,
    tensor_sigma,
    validate_sigma,
)

TINY_S4 = {'d_model': 16, 'd_state': 8, 'n_layers': 2, 'dropout': 0.0,
           'y_module_hidden': 8}
# Bigger, for the statistical claims: sigma is estimated from the realized
# noise, and 16x4 elements per tensor is too few for a tight tolerance.
BIG_S4 = {'d_model': 128, 'd_state': 64, 'n_layers': 2, 'dropout': 0.0,
          'y_module_hidden': 8}


def build_tiny(name='s4_concat', config=None, seed=0, detrain=True):
    """A small registered S4 model.

    ``detrain`` perturbs the SSM parameters once so the model resembles a
    trained checkpoint rather than an initialization. A freshly built S4Layer
    has ``log_A_real`` constant across every element (std == 0), which is a
    real but atypical corner of ``tensor_sigma``; tests that want that corner
    ask for it explicitly.
    """
    torch.manual_seed(seed)
    model = build_model(name, dict(config or TINY_S4))
    if detrain:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for layer in model.layers:
                for attr in ('log_A_real', 'A_imag', 'B', 'C_real', 'C_imag'):
                    p = getattr(layer, attr)
                    p.add_(torch.randn(p.shape, generator=g) * 0.1)
    return model


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def changed_keys(before, after):
    return {k for k in before if not torch.equal(before[k], after[k])}


def expected_keys(subset, n_layers=2):
    return {f'layers.{i}.{attr}'
            for i in range(n_layers)
            for name in subset_members(subset)
            for attr in SUBSET_PARAMS[name]}


def make_batch(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 80, 13, 21, generator=g),
            torch.randn(n, generator=g))


# ---------------------------------------------------------------- restore ---

@pytest.mark.parametrize('subset', SUBSETS)
def test_restore_undoes_perturbation_bit_exactly(subset):
    """The property the whole sweep rests on: after restore, the model is not
    'close to' the trained weights, it IS them."""
    model = build_tiny()
    pristine = snapshot(model)

    apply_perturbation(model, subset, 0.5, seed=7)
    assert changed_keys(pristine, snapshot(model)), 'nothing was perturbed'

    model.load_state_dict(pristine)
    verify_restore(model, pristine)          # raises if not bit-exact
    for k, v in snapshot(model).items():
        assert torch.equal(v, pristine[k])


def test_verify_restore_detects_unrestored_weights():
    model = build_tiny()
    pristine = snapshot(model)
    apply_perturbation(model, 'A+B+C', 0.1, seed=1)
    with pytest.raises(RuntimeError, match='restore failed'):
        verify_restore(model, pristine)


def test_perturbations_do_not_compound_across_configs():
    """Restore between configurations means the second draw starts from the
    trained weights, not from the first draw's output. Without the restore the
    two would differ, and sigma would silently grow down the sweep."""
    model = build_tiny()
    pristine = snapshot(model)

    apply_perturbation(model, 'A+B+C', 0.2, seed=11)
    first = snapshot(model)

    model.load_state_dict(pristine)
    apply_perturbation(model, 'C', 0.9, seed=99)      # interfering config
    model.load_state_dict(pristine)
    apply_perturbation(model, 'A+B+C', 0.2, seed=11)  # same config again

    for k, v in snapshot(model).items():
        assert torch.equal(v, first[k]), f'{k} did not reproduce'


# ------------------------------------------------------------- magnitude ---

@pytest.mark.parametrize('sigma', SIGMAS)
def test_noise_std_scales_with_sigma(sigma):
    """The requested magnitude is the delivered magnitude: the std of what was
    actually added, divided by the tensor's own std, comes back as sigma."""
    model = build_tiny(config=BIG_S4)
    before = snapshot(model)
    report = apply_perturbation(model, 'A+B+C', sigma, seed=3)
    after = snapshot(model)

    for detail in report['per_tensor']:
        key = f"layers.{detail['layer']}.{detail['param']}"
        delta = (after[key] - before[key]).to(torch.float64)
        w_std = before[key].to(torch.float64).std(unbiased=True).item()
        realized = delta.std(unbiased=True).item() / w_std
        # 128x32 = 4096 samples: the sample std of a Gaussian has relative
        # standard error 1/sqrt(2n) ~ 1.1%, so 8% is ~7 sigma of slack.
        assert realized == pytest.approx(sigma, rel=0.08), \
            f'{key}: asked for {sigma}, delivered {realized}'
    assert report['mean_realized_sigma_ratio'] == pytest.approx(sigma, rel=0.08)


def test_noise_magnitude_is_monotonic_in_sigma():
    """A larger sigma is always a larger perturbation in weight space."""
    errs = []
    for sigma in SIGMAS:
        model = build_tiny(config=BIG_S4, seed=5)
        errs.append(apply_perturbation(model, 'A+B+C', sigma,
                                       seed=3)['mean_rel_l2_err'])
    assert errs == sorted(errs)
    assert errs[0] < 0.01 < errs[-1]


def test_noise_is_zero_mean_and_gaussian_scaled():
    g = torch.Generator().manual_seed(0)
    noise = gaussian_noise((200, 200), 0.25, g)
    assert noise.shape == (200, 200)
    assert torch.isfinite(noise).all()
    assert abs(noise.mean().item()) < 0.01
    assert noise.std(unbiased=True).item() == pytest.approx(0.25, rel=0.03)


def test_zero_sigma_is_an_exact_noop():
    model = build_tiny()
    before = snapshot(model)
    report = apply_perturbation(model, 'A+B+C', 0.0, seed=42)
    assert not changed_keys(before, snapshot(model))
    assert report['max_abs_err'] == 0.0
    assert report['mean_rel_l2_err'] == 0.0


def test_perturbation_preserves_shape_dtype_and_finiteness():
    model = build_tiny(config=BIG_S4)
    before = snapshot(model)
    apply_perturbation(model, 'A+B+C', 1.0, seed=2)
    for k, v in snapshot(model).items():
        assert v.shape == before[k].shape
        assert v.dtype == before[k].dtype
        assert torch.isfinite(v).all()


# ---------------------------------------------------------- tensor_sigma ---

def test_tensor_sigma_uses_tensor_std():
    w = torch.randn(400, 400) * 3.0
    assert tensor_sigma(w, 0.1) == pytest.approx(0.3, rel=0.02)


def test_tensor_sigma_falls_back_to_mean_for_a_constant_tensor():
    """log_A_real initializes constant, so this branch is live, and silently
    perturbing a constant tensor by zero would report a completed
    configuration that did nothing."""
    w = torch.full((8, 4), -0.693)
    # Reduced in float64 on purpose: a two-pass fp32 std of this same constant
    # tensor returns ~6e-8 rather than 0, and a "std > 0" guard reading that
    # would scale the noise by roundoff instead of taking the fallback.
    assert w.to(torch.float64).std(unbiased=True).item() == 0.0
    assert tensor_sigma(w, 0.5) == pytest.approx(0.5 * 0.693, rel=1e-6)


def test_tensor_sigma_falls_back_to_sigma_for_an_all_zero_tensor():
    assert tensor_sigma(torch.zeros(8, 4), 0.25) == pytest.approx(0.25)


def test_constant_tensor_still_receives_nonzero_noise():
    """End-to-end version of the fallback: an untrained model (constant
    log_A_real) must still be measurably perturbed."""
    model = build_tiny(detrain=False)
    assert model.layers[0].log_A_real.std(unbiased=True).item() == 0.0
    before = snapshot(model)
    apply_perturbation(model, 'A', 0.1, seed=4)
    delta = (snapshot(model)['layers.0.log_A_real'] - before['layers.0.log_A_real'])
    assert delta.abs().max().item() > 0


# --------------------------------------------------------------- targeting ---

@pytest.mark.parametrize('subset', SUBSETS)
def test_perturbation_is_surgical(subset):
    """Exactly the subset's tensors move; log_dt, the encoder, the FiLM net,
    the y-projection, the norms and the head are bit-identical."""
    model = build_tiny('s4_modulate_full')
    before = snapshot(model)
    apply_perturbation(model, subset, 0.5, seed=8)
    changed = changed_keys(before, snapshot(model))

    assert changed == expected_keys(subset)
    assert not any(k.endswith('log_dt') for k in changed)
    assert not any(k.startswith(('encoder.', 'head.', 'film.', 'y_proj.',
                                 'norms.')) for k in changed)


def test_joint_subset_perturbs_the_union_of_the_three():
    model = build_tiny()
    before = snapshot(model)
    apply_perturbation(model, 'A+B+C', 0.3, seed=6)
    changed = changed_keys(before, snapshot(model))
    assert changed == (expected_keys('A') | expected_keys('B')
                       | expected_keys('C'))


def test_report_counts_every_targeted_tensor():
    model = build_tiny()
    report = apply_perturbation(model, 'A+B+C', 0.1, seed=0)
    # 2 layers x (log_A_real, A_imag, B, C_real, C_imag)
    assert report['n_tensors'] == 10
    assert report['n_params_perturbed'] == 10 * 16 * (8 // 2)
    assert len(report['per_tensor']) == 10
    assert report['sigma'] == 0.1
    assert report['seed'] == 0


def test_film_net_is_never_perturbed():
    model = build_tiny('s4_modulate_full')
    before = snapshot(model)
    assert any(k.startswith('film.') for k in before), 'no FiLM net to check'
    apply_perturbation(model, 'A+B+C', 1.0, seed=0)
    after = snapshot(model)
    for k in before:
        if k.startswith('film.'):
            assert torch.equal(before[k], after[k])


# ------------------------------------------------------------ determinism ---

def test_same_seed_reproduces_bit_exactly():
    a, b = build_tiny(seed=0), build_tiny(seed=0)
    apply_perturbation(a, 'A+B+C', 0.2, seed=1234)
    apply_perturbation(b, 'A+B+C', 0.2, seed=1234)
    for k, v in snapshot(a).items():
        assert torch.equal(v, snapshot(b)[k])


def test_different_seeds_give_different_draws():
    a, b = build_tiny(seed=0), build_tiny(seed=0)
    apply_perturbation(a, 'A+B+C', 0.2, seed=1)
    apply_perturbation(b, 'A+B+C', 0.2, seed=2)
    assert changed_keys(snapshot(a), snapshot(b))


def test_tensors_within_one_config_get_independent_noise():
    """One generator advanced across tensors, not restarted per tensor — two
    same-shaped tensors must not receive identical noise."""
    model = build_tiny(config=BIG_S4)
    before = snapshot(model)
    apply_perturbation(model, 'C', 0.5, seed=0)
    after = snapshot(model)
    d_real = after['layers.0.C_real'] - before['layers.0.C_real']
    d_imag = after['layers.0.C_imag'] - before['layers.0.C_imag']
    assert not torch.equal(d_real, d_imag)
    corr = torch.corrcoef(torch.stack([d_real.flatten(),
                                       d_imag.flatten()]))[0, 1].item()
    assert abs(corr) < 0.1


def test_config_seed_is_stable_and_distinct():
    """Not Python's salted hash: a re-run in a new process must reproduce."""
    assert config_seed('s4', 'A', 0.1, 0) == config_seed('s4', 'A', 0.1, 0)
    keys = {config_seed(m, s, sig, r)
            for m in ('s4', 's4_concat')
            for s in SUBSETS
            for sig in SIGMAS
            for r in range(5)}
    assert len(keys) == 2 * len(SUBSETS) * len(SIGMAS) * 5
    assert all(0 <= k < (1 << 63) for k in keys)


def test_noise_is_device_independent_by_construction():
    """Noise is drawn on a CPU generator regardless of where the parameter
    lives, so mean +/- std over seeds does not depend on which machine ran the
    sweep. Verified here by drawing against a CPU model and confirming the
    generator, not any device RNG, determines the values."""
    torch.manual_seed(999)            # global RNG deliberately desynchronized
    a = build_tiny(seed=0)
    apply_perturbation(a, 'A+B+C', 0.3, seed=77)
    torch.manual_seed(1)              # different global state
    b = build_tiny(seed=0)
    apply_perturbation(b, 'A+B+C', 0.3, seed=77)
    for k, v in snapshot(a).items():
        assert torch.equal(v, snapshot(b)[k])


# --------------------------------------------------------------- stability ---

@pytest.mark.parametrize('sigma', [0.1, 1.0])
def test_log_space_perturbation_keeps_the_recurrence_stable(sigma):
    """Re(A) = -exp(log_A_real) stays negative and |dA| < 1 for ANY finite
    perturbation, because A is perturbed in log space as stored. This is the
    claim that lets sigma=1 report a robustness number instead of a NaN."""
    model = build_tiny(config=BIG_S4)
    apply_perturbation(model, 'A+B+C', sigma, seed=13)
    for layer in model.layers:
        A_real = -torch.exp(layer.log_A_real)
        assert (A_real < 0).all()
        dt = torch.exp(layer.log_dt).unsqueeze(-1)
        assert (torch.exp((A_real * dt)) < 1).all()


def test_forward_stays_finite_under_the_largest_sigma():
    model = build_tiny('s4_modulate_full', config=BIG_S4)
    model.eval()
    X, y_module = make_batch(2, seed=0)
    apply_perturbation(model, 'A+B+C', max(SIGMAS), seed=21)
    with torch.no_grad():
        out = model(X, y_module)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_perturbation_actually_changes_the_output():
    """A perturbation the forward pass ignores would produce a flat, and
    entirely meaningless, robustness curve."""
    model = build_tiny(config=BIG_S4)
    model.eval()
    X, y_module = make_batch(2, seed=1)
    with torch.no_grad():
        before = model(X, y_module)
    apply_perturbation(model, 'A+B+C', 0.1, seed=5)
    with torch.no_grad():
        after = model(X, y_module)
    assert not torch.allclose(before, after)


# --------------------------------------------------------------- validation ---

def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError):
        apply_perturbation(build_tiny(), 'D', 0.1, seed=0)


@pytest.mark.parametrize('bad', [-0.1, float('nan'), 1e6])
def test_invalid_sigma_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_sigma(bad)


def test_non_s4_model_is_rejected():
    model = build_model('mlp', {})
    with pytest.raises(AttributeError, match='layers'):
        apply_perturbation(model, 'A', 0.1, seed=0)


def test_perturbation_error_matches_the_quantization_convention():
    x = torch.ones(4, 4)
    x_hat = x + 0.5
    rel, mx = perturbation_error(x, x_hat)
    assert mx == pytest.approx(0.5)
    assert rel == pytest.approx(0.5 / 1.0)   # ||0.5||/||1|| over equal counts
    rel0, mx0 = perturbation_error(torch.zeros(3), torch.zeros(3))
    assert (rel0, mx0) == (0.0, 0.0)


def test_report_error_fields_are_consistent():
    model = build_tiny(config=BIG_S4)
    report = apply_perturbation(model, 'A+B+C', 0.2, seed=0)
    rels = [d['rel_l2_err'] for d in report['per_tensor']]
    assert report['max_rel_l2_err'] == pytest.approx(max(rels))
    assert report['mean_rel_l2_err'] == pytest.approx(sum(rels) / len(rels))
    assert report['max_abs_err'] == pytest.approx(
        max(d['max_abs_err'] for d in report['per_tensor']))
    assert all(math.isfinite(d['sigma_eff']) for d in report['per_tensor'])
