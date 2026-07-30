"""Simulated (fake) quantization of the S4 state-space matrices (Phase 6).

"Fake" quantization means quantize-then-immediately-dequantize: the tensor is
snapped onto the grid a b-bit integer representation could express, but is
stored and computed with in float32. That isolates the *precision loss* of
low-bit storage from every other effect of a real integer kernel (integer
accumulation, requantization between ops, kernel-specific rounding), which is
what a bit-width sensitivity study wants to measure. It is the standard PTQ
methodology in the SSM quantization literature this study compares against.

Scheme: ASYMMETRIC (affine) uniform quantization over the observed min/max.

    s  = (max - min) / (2**b - 1)
    z  = round(-min / s)                      # integer zero-point
    q  = clamp(round(x / s) + z, 0, 2**b - 1)
    x^ = (q - z) * s

Asymmetric rather than symmetric because the tensors under study are
emphatically not zero-centered, and a symmetric range would spend most of its
levels on empty space:

  - ``log_A_real`` is initialized to ``log(0.5) ~ -0.693`` and trains in a
    narrow band around it — never near zero.
  - ``B`` is initialized to all-ones.
  - ``A_imag`` is initialized to ``pi * n`` for n = 0..N_half-1, i.e.
    [0, 97.4] for d_state=64 — strictly non-negative and spanning two orders
    of magnitude.

At 2 bits (4 levels) a symmetric range on ``A_imag`` would leave a single
usable level for the whole populated interval, so the resulting degradation
would be a property of the scheme, not of the model. Asymmetric min/max is
also PyTorch's own PTQ default (``MinMaxObserver`` with an affine qscheme), so
it is the comparable choice. ``C_real``/``C_imag`` are near zero-centered and
are close to indifferent to the choice.

Because the zero-point is an integer, exact zero is always representable. The
observed min/max are covered end to end, up to that zero-point rounding.

Complex parameters. ``S4Layer`` already stores its complex quantities as pairs
of real ``nn.Parameter``s (``A_imag`` alongside the log-magnitude
``log_A_real``; ``C_real``/``C_imag``), reassembled by ``torch.complex`` in the
forward pass. Each real component is therefore quantized as an independent
tensor with its own scale and zero-point. That is not merely convenient, it is
the only defensible choice numerically: A's stored real part sits near -0.69 in
log-space while its imaginary part spans [0, 97.4], so a shared scale would
quantize one of the two into a single level.

A is quantized in LOG-SPACE, as stored. This is deliberate and load-bearing
for stability: ``Re(A) = -exp(log_A_real)`` is strictly negative for *any*
finite value ``log_A_real`` takes, so no bit-width, however low, can flip the
sign of the decay rate and diverge the recurrence. Quantizing ``Re(A)``
directly would put values near zero one rounding step away from crossing it,
and a 2-bit run would then report a stability blowup rather than a precision
effect.

``log_dt`` is NOT quantized by any subset. It is a per-channel timescale
vector, not one of the three matrices under study; the proposal's Table 4 axis
is A / B / C.
"""
import torch

# Sweep axes. Exported so the CLI and the table builders agree by construction.
BITS = (8, 6, 4, 2)
GRANULARITIES = ('per_tensor', 'per_channel')
SUBSETS = ('A', 'B', 'C', 'A+B+C')
JOINT_SUBSET = 'A+B+C'

# Matrix subset -> the S4Layer parameter attributes that constitute it.
SUBSET_PARAMS = {
    'A': ('log_A_real', 'A_imag'),
    'B': ('B',),
    'C': ('C_real', 'C_imag'),
}

# Every quantizable tensor is (H, N_half) with H == d_model. Axis 0 is
# therefore the channel axis, and per-channel quantization reduces over the
# state axis to yield H independent scales. Asserted against the live tensor
# shapes in ``iter_ssm_params`` so a future reparameterization cannot silently
# transpose this.
CHANNEL_AXIS = 0

_MIN_BITS, _MAX_BITS = 2, 16


def n_levels(bits):
    """Number of representable integer levels at ``bits`` bits: 2**bits."""
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(f"bits must be an int, got {type(bits).__name__}")
    if not _MIN_BITS <= bits <= _MAX_BITS:
        raise ValueError(
            f"bits must be in [{_MIN_BITS}, {_MAX_BITS}], got {bits}")
    return 2 ** bits


def fake_quantize(x, bits, granularity='per_tensor'):
    """Asymmetric uniform quantize -> dequantize round-trip of ``x``.

    Args:
        x:           real-valued tensor. Complex input is rejected — callers
                     quantize the real and imaginary components separately
                     (see the module docstring).
        bits:        2..16.
        granularity: 'per_tensor' (one scale over every element) or
                     'per_channel' (one scale per index along CHANNEL_AXIS,
                     reducing over all other axes).

    Returns:
        A new tensor, same shape and dtype as ``x``, holding only values on the
        b-bit grid. ``x`` is not modified.

    The round-trip is a projection onto that grid, hence idempotent: quantizing
    an already-quantized tensor at the same settings is a no-op (pinned by a
    test).
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    if x.is_complex():
        raise TypeError(
            "fake_quantize does not accept complex tensors; quantize the real "
            "and imaginary components independently (see module docstring)")
    if not x.is_floating_point():
        raise TypeError(f"fake_quantize expects a float tensor, got {x.dtype}")
    n = n_levels(bits) - 1               # max integer code

    xf = x.detach().to(torch.float32)
    if granularity == 'per_tensor':
        lo, hi = xf.min(), xf.max()
    else:
        if xf.dim() < 2:
            raise ValueError(
                f"per-channel quantization needs a >=2D tensor to have a "
                f"channel axis; got shape {tuple(xf.shape)}")
        reduce_dims = [d for d in range(xf.dim()) if d != CHANNEL_AXIS]
        lo = xf.amin(dim=reduce_dims, keepdim=True)
        hi = xf.amax(dim=reduce_dims, keepdim=True)

    scale = (hi - lo) / n
    # A constant tensor (or constant channel) has zero range, and s = 0 would
    # divide by zero. Its single value is trivially representable at any
    # bit-width, so those entries are passed through untouched. This is not a
    # theoretical case: log_A_real is initialized to a constant across every
    # element, so a freshly-built model hits it on every channel.
    degenerate = scale == 0
    scale = torch.where(degenerate, torch.ones_like(scale), scale)

    zero_point = torch.round(-lo / scale)
    q = torch.clamp(torch.round(xf / scale) + zero_point, 0, n)
    out = (q - zero_point) * scale
    out = torch.where(degenerate.expand_as(xf), xf, out)
    return out.to(dtype=x.dtype)


def quantization_error(x, x_hat):
    """-> (relative L2 error, max absolute error) between a tensor and its
    quantized round-trip. Relative L2 is ||x^ - x||_2 / ||x||_2, or the bare
    absolute L2 if ``x`` is all zeros."""
    diff = (x_hat.detach().to(torch.float64) - x.detach().to(torch.float64))
    denom = x.detach().to(torch.float64).norm().item()
    l2 = diff.norm().item()
    return (l2 / denom if denom > 0 else l2), diff.abs().max().item()


def iter_ssm_params(model):
    """Yield ``(layer_index, subset, attr_name, parameter)`` for every
    quantizable tensor of every S4 layer in ``model``, in a stable order.

    Raises if the model has no ``layers`` ModuleList, if a layer is missing an
    expected parameter, or if a parameter is not (H, N_half) — all three mean
    the checkpoint's architecture has drifted from what this study assumes, and
    silently quantizing whatever is there instead would produce numbers nobody
    could interpret.
    """
    layers = getattr(model, 'layers', None)
    if layers is None or len(layers) == 0:
        raise AttributeError(
            f"{type(model).__name__} has no non-empty '.layers' ModuleList; "
            f"this harness targets the S4 family (Models/ssm/s4.py)")
    for i, layer in enumerate(layers):
        for subset in ('A', 'B', 'C'):
            for attr in SUBSET_PARAMS[subset]:
                param = getattr(layer, attr, None)
                if not isinstance(param, torch.nn.Parameter):
                    raise AttributeError(
                        f"layer {i} ({type(layer).__name__}) has no "
                        f"nn.Parameter '{attr}'; expected the S4Layer "
                        f"parameterization from Models/ssm/s4.py")
                if param.dim() != 2:
                    raise ValueError(
                        f"layer {i} parameter '{attr}' has shape "
                        f"{tuple(param.shape)}; this harness assumes "
                        f"(H, N_half) so that axis {CHANNEL_AXIS} is the "
                        f"d_model channel axis")
                yield i, subset, attr, param


def subset_members(subset):
    """-> tuple of the individual matrix names a subset expands to."""
    if subset == JOINT_SUBSET:
        return ('A', 'B', 'C')
    if subset in SUBSET_PARAMS:
        return (subset,)
    raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")


def apply_fake_quant(model, subset, bits, granularity):
    """Fake-quantize ``subset``'s parameters of every S4 layer, IN PLACE.

    Only the named subset is perturbed; every other parameter — including the
    other two matrices, ``log_dt``, the frame encoder, the FiLM/conditioning
    net, and the head — is left bit-identical at FP32. ``A+B+C`` quantizes all
    three simultaneously at the same bit-width and granularity.

    Returns a report dict for the run log:
        n_tensors, n_params_quantized, mean_rel_l2_err, max_rel_l2_err,
        max_abs_err, per_tensor (list of {layer, subset, param, rel_l2_err,
        max_abs_err, n_unique})

    ``n_unique`` is the number of distinct values surviving in the tensor. It
    is the cheap independent witness that the bit-width actually took effect:
    it can never exceed 2**bits under per-tensor granularity.
    """
    wanted = subset_members(subset)
    details, errs = [], []
    with torch.no_grad():
        for idx, name, attr, param in iter_ssm_params(model):
            if name not in wanted:
                continue
            before = param.detach().clone()
            param.copy_(fake_quantize(param, bits, granularity))
            rel, mx = quantization_error(before, param)
            errs.append((rel, mx, param.numel()))
            details.append({
                'layer': idx, 'subset': name, 'param': attr,
                'shape': list(param.shape),
                'rel_l2_err': rel, 'max_abs_err': mx,
                'n_unique': int(param.detach().unique().numel()),
            })
    if not details:
        raise RuntimeError(
            f"subset {subset!r} matched no parameters in "
            f"{type(model).__name__} — refusing to report an 'evaluation' that "
            f"quantized nothing")
    return {
        'n_tensors': len(details),
        'n_params_quantized': sum(e[2] for e in errs),
        'mean_rel_l2_err': sum(e[0] for e in errs) / len(errs),
        'max_rel_l2_err': max(e[0] for e in errs),
        'max_abs_err': max(e[1] for e in errs),
        'per_tensor': details,
    }
