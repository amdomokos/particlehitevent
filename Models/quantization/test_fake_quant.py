"""Tests for the Phase 6 fake-quantization math and its model targeting.

The point of this file is to catch a quantization-math bug in 30 seconds
rather than after 64 real checkpoint evaluations. It pins, in order:
  - that the round-trip actually reduces precision (level counting), and that
    error grows monotonically as bits fall;
  - that per-channel genuinely beats per-tensor on a tensor whose scale varies
    per channel, which is the whole reason granularity is a sweep axis;
  - that subset targeting is surgical — quantizing 'A' leaves B, C, log_dt,
    the encoder, the head, and the FiLM net bit-identical;
  - that 2-bit quantization of A cannot destabilize the recurrence, the
    stability argument behind quantizing A in log-space.

Run from the repo root:
    pytest Models/quantization/test_fake_quant.py
"""
import math

import pytest
import torch

import Models.models_import_all  # noqa: F401 — populates registry
from Models.common.registry import build_model
from Models.quantization.fake_quant import (
    BITS,
    CHANNEL_AXIS,
    GRANULARITIES,
    SUBSET_PARAMS,
    SUBSETS,
    apply_fake_quant,
    fake_quantize,
    iter_ssm_params,
    n_levels,
    quantization_error,
    subset_members,
)

TINY_S4 = {'d_model': 16, 'd_state': 8, 'n_layers': 2, 'dropout': 0.0,
           'y_module_hidden': 8}


def randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------
# core round-trip behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_shape_dtype_preserved(bits, granularity):
    x = randn(16, 32)
    q = fake_quantize(x, bits, granularity)
    assert q.shape == x.shape
    assert q.dtype == x.dtype
    assert torch.isfinite(q).all()


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_input_not_mutated(bits, granularity):
    x = randn(16, 32)
    before = x.clone()
    fake_quantize(x, bits, granularity)
    assert torch.equal(x, before)


@pytest.mark.parametrize('bits', BITS)
def test_per_tensor_level_count(bits):
    """The defining property: no more than 2**bits distinct values survive."""
    x = randn(16, 32)                       # 512 distinct values going in
    q = fake_quantize(x, bits, 'per_tensor')
    assert q.unique().numel() <= n_levels(bits)
    # and the reduction is real, not vacuous
    assert q.unique().numel() < x.unique().numel()


@pytest.mark.parametrize('bits', BITS)
def test_per_channel_level_count_is_per_row(bits):
    """Per-channel gets 2**bits levels PER channel, not per tensor."""
    x = randn(8, 64)
    q = fake_quantize(x, bits, 'per_channel')
    for row in range(x.shape[CHANNEL_AXIS]):
        assert q[row].unique().numel() <= n_levels(bits)
    if bits <= 4:
        # each row spends its own budget, so the tensor total exceeds one
        # row's budget — the observable difference from per-tensor
        assert q.unique().numel() > n_levels(bits)


@pytest.mark.parametrize('granularity', GRANULARITIES)
def test_error_grows_monotonically_as_bits_fall(granularity):
    x = randn(32, 64)
    errs = [quantization_error(x, fake_quantize(x, b, granularity))[0]
            for b in (8, 6, 4, 2)]
    assert errs == sorted(errs), f"error not monotonic in bits: {errs}"
    # 8-bit is a faithful round-trip; 2-bit is visibly destructive
    assert errs[0] < 0.01, f"8-bit rel L2 err {errs[0]:.4g} unexpectedly large"
    assert errs[-1] > 0.10, f"2-bit rel L2 err {errs[-1]:.4g} suspiciously small"
    assert errs[-1] > 10 * errs[0]


@pytest.mark.parametrize('granularity', GRANULARITIES)
def test_high_bitwidth_is_near_lossless(granularity):
    """The sanity anchor for the sweep: at 16 bits the round-trip is
    effectively the identity, so a 16-bit run must reproduce FP32 metrics."""
    x = randn(32, 64)
    rel, _ = quantization_error(x, fake_quantize(x, 16, granularity))
    assert rel < 1e-4, f"16-bit rel L2 err {rel:.4g} is not near-lossless"


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_error_bounded_by_half_step(bits, granularity):
    """Uniform quantization cannot err by more than half a step anywhere."""
    x = randn(16, 32)
    q = fake_quantize(x, bits, granularity)
    step = (x.max() - x.min()).item() / (n_levels(bits) - 1)
    assert (q - x).abs().max().item() <= step / 2 + 1e-6


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_no_range_expansion(bits, granularity):
    """Quantized values stay inside the original observed range (up to
    zero-point rounding), so quantization cannot invent outliers."""
    x = randn(16, 32)
    q = fake_quantize(x, bits, granularity)
    step = (x.max() - x.min()).item() / (n_levels(bits) - 1)
    assert q.min().item() >= x.min().item() - step
    assert q.max().item() <= x.max().item() + step


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_idempotent(bits, granularity):
    """The round-trip is a projection onto the b-bit grid, so re-applying it
    must not move values or add levels. A scale computed from the wrong
    statistics (e.g. re-observing a range the previous pass had shifted) would
    drift by a fraction of a step on the second pass and multiply the level
    count. Tolerance is float ULP, not a quantization step: recomputing
    (hi - lo) / n per channel in float32 can differ in the last bit.
    """
    x = randn(16, 32)
    once = fake_quantize(x, bits, granularity)
    twice = fake_quantize(once, bits, granularity)
    step = (x.max() - x.min()).item() / (n_levels(bits) - 1)
    drift = (twice - once).abs().max().item()
    assert drift < 1e-6 * max(1.0, step), (
        f"second pass moved values by {drift:.3e} (step {step:.3e})")
    assert twice.unique().numel() == once.unique().numel()


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_deterministic(bits, granularity):
    x = randn(16, 32)
    assert torch.equal(fake_quantize(x, bits, granularity),
                       fake_quantize(x, bits, granularity))


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_zero_is_exactly_representable(bits, granularity):
    """The reason for an integer zero-point: a tensor containing exact zeros
    must round-trip them exactly, at any bit-width."""
    x = randn(8, 16)
    x[0, 0] = 0.0
    x[3, 7] = 0.0
    q = fake_quantize(x, bits, granularity)
    assert q[0, 0].item() == 0.0
    assert q[3, 7].item() == 0.0


# --------------------------------------------------------------------------
# degenerate ranges — log_A_real is initialized to a constant, so this is a
# real code path, not a hypothetical
# --------------------------------------------------------------------------

@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_constant_tensor_round_trips_exactly(bits, granularity):
    x = torch.full((16, 32), math.log(0.5))
    q = fake_quantize(x, bits, granularity)
    assert torch.equal(q, x), "zero-range tensor must pass through untouched"


@pytest.mark.parametrize('bits', BITS)
def test_constant_channel_round_trips_exactly(bits):
    """One degenerate channel must not poison its neighbours, or vice versa."""
    x = randn(4, 32)
    x[1] = 2.5                              # constant channel
    q = fake_quantize(x, bits, 'per_channel')
    assert torch.equal(q[1], x[1])
    assert not torch.equal(q[0], x[0])      # normal channels still quantized


# --------------------------------------------------------------------------
# granularity actually matters — the reason it is a sweep axis
# --------------------------------------------------------------------------

@pytest.mark.parametrize('bits', (8, 4))
def test_per_channel_beats_per_tensor_on_varying_scale(bits):
    """A tensor whose per-channel dynamic ranges differ by 1000x is exactly the
    case per-channel exists for: one global scale is set by the loudest channel
    and annihilates the quiet ones."""
    base = randn(4, 64, seed=1)
    x = base * torch.tensor([1.0, 10.0, 100.0, 1000.0]).unsqueeze(1)

    rel_t, _ = quantization_error(x, fake_quantize(x, bits, 'per_tensor'))
    rel_c, _ = quantization_error(x, fake_quantize(x, bits, 'per_channel'))
    assert rel_c < rel_t, f"per_channel {rel_c:.4g} !< per_tensor {rel_t:.4g}"

    # The mechanism, not just the aggregate: per-channel error is proportional
    # to each channel's own range, so relative error is roughly uniform across
    # channels, whereas per-tensor error is catastrophic on the quiet channel
    # and negligible on the loud one.
    qt = fake_quantize(x, bits, 'per_tensor')
    qc = fake_quantize(x, bits, 'per_channel')
    per_row_t = [quantization_error(x[i], qt[i])[0] for i in range(4)]
    per_row_c = [quantization_error(x[i], qc[i])[0] for i in range(4)]
    # The quiet channel is annihilated outright — every value of it falls
    # inside a single step of the global grid, so relative error saturates
    # near 1.0 — while the loud channel that set the scale is unharmed.
    assert per_row_t[0] > 0.5, (
        f"expected per-tensor to wreck the quiet channel: {per_row_t}")
    assert per_row_t[0] > 10 * per_row_t[3], (
        f"expected per-tensor error to fall with channel scale: {per_row_t}")
    assert max(per_row_c) < 5 * min(per_row_c), (
        f"expected per-channel error to be scale-invariant: {per_row_c}")


def test_per_channel_equals_per_tensor_when_scales_match():
    """With identical per-channel ranges the two granularities must agree —
    a guard against per-channel accidentally reducing over the wrong axis."""
    x = torch.stack([torch.linspace(-1, 1, 33) for _ in range(5)])
    assert torch.allclose(fake_quantize(x, 6, 'per_tensor'),
                          fake_quantize(x, 6, 'per_channel'))


def test_per_channel_reduces_over_the_state_axis():
    """Explicitly pin the axis convention: channel = axis 0 = d_model. If the
    reduction were transposed, a tensor with per-COLUMN varying scale would be
    the one per-channel helps, and this asserts it is not."""
    x = torch.zeros(4, 6)
    x[0] = 1000.0 * torch.linspace(-1, 1, 6)
    x[1:] = 0.001 * torch.linspace(-1, 1, 6)

    # row 0's huge range must not leak into rows 1..3
    rel_quiet, _ = quantization_error(x[1:], fake_quantize(x, 8, 'per_channel')[1:])
    assert rel_quiet < 0.05, (
        f"quiet rows degraded by {rel_quiet:.4g} — per-channel appears to be "
        f"sharing a scale across axis {CHANNEL_AXIS}")
    # contrast: sharing one scale (which is what a transposed reduction would
    # effectively do to these rows) annihilates them
    rel_shared, _ = quantization_error(x[1:], fake_quantize(x, 8, 'per_tensor')[1:])
    assert rel_shared > 0.9, (
        f"per-tensor should annihilate the quiet rows, got {rel_shared:.4g} — "
        f"the test's contrast is not discriminating")


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_rejects_complex_input():
    z = torch.complex(randn(4, 4), randn(4, 4, seed=1))
    with pytest.raises(TypeError, match='complex'):
        fake_quantize(z, 8, 'per_tensor')


def test_rejects_integer_input():
    with pytest.raises(TypeError, match='float'):
        fake_quantize(torch.ones(4, 4, dtype=torch.int32), 8, 'per_tensor')


@pytest.mark.parametrize('bits', (0, 1, 17, 32, -8))
def test_rejects_out_of_range_bits(bits):
    with pytest.raises(ValueError, match='bits'):
        fake_quantize(randn(4, 4), bits, 'per_tensor')


@pytest.mark.parametrize('bits', (8.0, '8', None))
def test_rejects_non_integer_bits(bits):
    with pytest.raises(TypeError, match='bits'):
        fake_quantize(randn(4, 4), bits, 'per_tensor')


def test_rejects_unknown_granularity():
    with pytest.raises(ValueError, match='granularity'):
        fake_quantize(randn(4, 4), 8, 'per_group')


def test_per_channel_rejects_1d():
    with pytest.raises(ValueError, match='channel axis'):
        fake_quantize(randn(16), 8, 'per_channel')


def test_rejects_unknown_subset():
    with pytest.raises(ValueError, match='subset'):
        subset_members('D')


# --------------------------------------------------------------------------
# model targeting
# --------------------------------------------------------------------------

def _clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _changed_keys(before, model):
    after = model.state_dict()
    return {k for k, v in before.items() if not torch.equal(v, after[k])}


@pytest.mark.parametrize('model_name', ['s4_concat', 's4_modulate_full'])
def test_iter_ssm_params_covers_every_layer(model_name):
    model = build_model(model_name, TINY_S4)
    found = list(iter_ssm_params(model))
    expected = TINY_S4['n_layers'] * sum(len(v) for v in SUBSET_PARAMS.values())
    assert len(found) == expected == 10
    assert {s for _, s, _, _ in found} == {'A', 'B', 'C'}
    for _, _, _, param in found:
        assert tuple(param.shape) == (TINY_S4['d_model'],
                                      TINY_S4['d_state'] // 2)


@pytest.mark.parametrize('model_name', ['s4_concat', 's4_modulate_full'])
@pytest.mark.parametrize('subset', SUBSETS)
def test_subset_targeting_is_surgical(model_name, subset):
    """Quantizing one subset must leave every other parameter bit-identical —
    including log_dt, which is deliberately never quantized."""
    model = build_model(model_name, TINY_S4)
    # break the constant init so A is genuinely quantizable, not degenerate
    with torch.no_grad():
        for _, _, _, param in iter_ssm_params(model):
            param.add_(randn(*param.shape, seed=7) * 0.1)
    before = _clone_state(model)

    apply_fake_quant(model, subset, bits=4, granularity='per_tensor')
    changed = _changed_keys(before, model)

    expected = {
        f'layers.{i}.{attr}'
        for i in range(TINY_S4['n_layers'])
        for name in subset_members(subset)
        for attr in SUBSET_PARAMS[name]
    }
    assert changed == expected, (
        f"unexpected difference: extra={sorted(changed - expected)}, "
        f"missing={sorted(expected - changed)}")
    assert not any(k.endswith('log_dt') for k in changed)
    assert not any(k.startswith(('encoder.', 'head.', 'film.', 'y_proj.',
                                 'norms.')) for k in changed)


def test_joint_subset_is_the_union_of_the_three():
    model_joint = build_model('s4_concat', TINY_S4)
    torch.manual_seed(0)
    base = _clone_state(model_joint)

    model_seq = build_model('s4_concat', TINY_S4)
    model_seq.load_state_dict(base)

    apply_fake_quant(model_joint, 'A+B+C', 6, 'per_channel')
    for name in ('A', 'B', 'C'):
        apply_fake_quant(model_seq, name, 6, 'per_channel')

    for k, v in model_joint.state_dict().items():
        assert torch.equal(v, model_seq.state_dict()[k]), f"mismatch in {k}"


def test_film_net_is_never_quantized():
    """s4_modulate_full's FiLM net modulates A/B/C at runtime, but its weights
    are not part of the matrices under study."""
    model = build_model('s4_modulate_full', TINY_S4)
    before = _clone_state(model)
    apply_fake_quant(model, 'A+B+C', 2, 'per_tensor')
    changed = _changed_keys(before, model)
    assert changed, "nothing was quantized"
    assert not [k for k in changed if k.startswith('film.')]


@pytest.mark.parametrize('granularity', GRANULARITIES)
@pytest.mark.parametrize('bits', BITS)
def test_report_counts_and_levels(bits, granularity):
    model = build_model('s4_concat', TINY_S4)
    with torch.no_grad():
        for _, _, _, param in iter_ssm_params(model):
            param.add_(randn(*param.shape, seed=3) * 0.1)
    rep = apply_fake_quant(model, 'A+B+C', bits, granularity)

    n_tensors = TINY_S4['n_layers'] * 5
    per_tensor_numel = TINY_S4['d_model'] * (TINY_S4['d_state'] // 2)
    assert rep['n_tensors'] == n_tensors
    assert rep['n_params_quantized'] == n_tensors * per_tensor_numel
    assert rep['mean_rel_l2_err'] > 0
    if granularity == 'per_tensor':
        assert all(d['n_unique'] <= n_levels(bits) for d in rep['per_tensor'])


def test_report_error_grows_as_bits_fall():
    errs = {}
    for bits in BITS:
        model = build_model('s4_concat', TINY_S4)
        torch.manual_seed(0)
        with torch.no_grad():
            for _, _, _, param in iter_ssm_params(model):
                param.add_(randn(*param.shape, seed=3) * 0.1)
        errs[bits] = apply_fake_quant(
            model, 'A+B+C', bits, 'per_tensor')['mean_rel_l2_err']
    assert errs[8] < errs[6] < errs[4] < errs[2]


def test_apply_rejects_non_s4_model():
    model = build_model('mlp', {'hidden_dims': [8], 'dropout': 0.0})
    with pytest.raises(AttributeError, match='layers'):
        apply_fake_quant(model, 'A', 8, 'per_tensor')


# --------------------------------------------------------------------------
# stability: the reason A is quantized in log-space
# --------------------------------------------------------------------------

@pytest.mark.parametrize('model_name', ['s4_concat', 's4_modulate_full'])
def test_two_bit_quantization_cannot_destabilize_the_recurrence(model_name):
    """Re(A) = -exp(log_A_real) < 0 for any finite quantized log_A_real, so
    even 2-bit A leaves |dA| < 1 and the forward pass finite."""
    model = build_model(model_name, TINY_S4).eval()
    with torch.no_grad():
        for _, _, _, param in iter_ssm_params(model):
            param.add_(randn(*param.shape, seed=5) * 0.5)
    apply_fake_quant(model, 'A+B+C', 2, 'per_tensor')

    for layer in model.layers:
        A_real = -torch.exp(layer.log_A_real)
        assert (A_real < 0).all(), "quantized Re(A) lost its sign"
        dA = torch.polar(torch.exp((A_real * torch.exp(layer.log_dt)
                                    .unsqueeze(-1))), torch.zeros_like(A_real))
        assert (dA.abs() < 1.0).all(), "quantized A produced |dA| >= 1"

    X = randn(2, 80, 13, 21, seed=11)
    y_module = randn(2, seed=12)
    with torch.no_grad():
        out = model(X, y_module)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all(), "2-bit A produced non-finite outputs"
