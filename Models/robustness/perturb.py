"""Gaussian weight perturbation of the S4 state-space matrices (Phase 7).

The proposal (Section 7.4) writes the perturbation as ``theta' = theta + eps``
with ``eps ~ N(0, sigma^2)``, i.e. one absolute noise scale shared by every
weight. This module implements the RELATIVE form instead:

    eps_ij ~ N(0, (sigma * std(W))^2)      for each tensor W independently

so ``sigma`` is a dimensionless fraction of each tensor's own spread rather
than an absolute magnitude. That deviation is deliberate and load-bearing.

Why relative. The S4Layer parameters differ in scale by three orders of
magnitude at initialization and stay far apart after training:

  - ``A_imag``    is ``pi * n`` for n = 0..N_half-1, i.e. [0, 97.4] at d_state=64
  - ``B``         is initialized to all-ones
  - ``log_A_real`` sits in a narrow band around ``log(0.5) = -0.693``
  - ``C_real``/``C_imag`` are ``randn / sqrt(d_state)``, so std ~ 0.125

Under a single absolute sigma, sigma=0.1 is a 100% perturbation of C and a
0.1% perturbation of A_imag. A per-matrix sensitivity comparison run that way
would rank the matrices by how small their weights happen to be, not by how
much the model depends on them — it would measure the parameterization, not
the model. The same argument applies across checkpoints: two models whose
weight norms drifted apart during training would not be comparable at equal
absolute sigma. Relative sigma removes both confounds, at the cost of no
longer being the proposal's literal formula. Absolute results can be
recovered from ``sigma_eff`` and ``w_std``, both recorded per tensor in the
report.

A is perturbed in LOG-SPACE, as stored. ``Re(A) = -exp(log_A_real)`` is
strictly negative for any finite value ``log_A_real`` takes, so no sigma,
however large, can flip the sign of the decay rate and diverge the
recurrence. Perturbing ``Re(A)`` directly would let large sigma cross zero and
the study would then report a stability blowup rather than a robustness curve.
This mirrors the same choice, for the same reason, in
``Models/quantization/fake_quant.py``.

``log_dt`` is NOT perturbed by any subset. It is a per-channel timescale
vector, not one of the three matrices the proposal's axis names.

Determinism. Noise is always drawn from a CPU ``torch.Generator`` seeded by a
stable hash of the configuration, then moved to the parameter's device. CUDA,
XPU and CPU RNGs produce different streams from the same seed, and a study
that reports mean +/- std over seeds must not have its "seeds" depend on which
machine ran the sweep. Seeding from the configuration rather than from a
counter also means any single configuration can be re-run in isolation and
reproduce bit-exactly, independent of sweep order.
"""
import hashlib

import torch

# Reused, not re-derived: subset -> S4Layer attribute mapping, the tensor
# iterator with its architecture assertions, and the subset vocabulary. Phase 6
# already validated all of it against the live model.
from Models.quantization.fake_quant import (  # noqa: F401 — SUBSETS re-exported
    JOINT_SUBSET,
    SUBSET_PARAMS,
    SUBSETS,
    iter_ssm_params,
    subset_members,
)

# Log-spaced from negligible to destructive, ~half a decade apart. At the low
# end the perturbation is far below the weights' own trained precision; at
# sigma=1 the noise is as large as the weights themselves and the model is
# expected to be unusable. Seven points is enough to read a slope off a log-log
# plot without spending inference time on redundant resolution.
SIGMAS = (0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1.0)

_MAX_SIGMA = 100.0


def validate_sigma(sigma):
    """-> float. Rejects negatives, NaN, and absurd magnitudes."""
    s = float(sigma)
    if s != s:
        raise ValueError('sigma must not be NaN')
    if s < 0:
        raise ValueError(f'sigma must be >= 0, got {s}')
    if s > _MAX_SIGMA:
        raise ValueError(f'sigma must be <= {_MAX_SIGMA}, got {s}')
    return s


def tensor_sigma(w, sigma):
    """-> the absolute noise std to use for tensor ``w`` at relative ``sigma``.

    ``sigma * std(w)`` normally. A constant tensor has zero std, and scaling
    noise by zero would silently perturb nothing while the sweep reported a
    completed configuration — so two fallbacks apply, in order:

      1. ``sigma * |mean(w)|``, the only other scale the tensor offers;
      2. ``sigma``, for an all-zero tensor, which has no scale at all.

    This is not a theoretical branch: ``log_A_real`` is initialized to a
    constant across every element, so a freshly built (untrained) model takes
    fallback 1 on every layer. A trained checkpoint does not, but the tests
    build untrained models.
    """
    wf = w.detach().to(torch.float64)
    std = wf.std(unbiased=True).item() if wf.numel() > 1 else 0.0
    if std > 0:
        return sigma * std
    mean_abs = wf.mean().abs().item()
    if mean_abs > 0:
        return sigma * mean_abs
    return sigma


def config_seed(model_name, subset, sigma, repeat):
    """-> a stable 63-bit seed derived from the configuration itself.

    Python's ``hash`` is salted per process and would make a re-run of one
    configuration draw different noise, so blake2b over the formatted key is
    used instead. ``{sigma:.10g}`` is the same formatting the sweep's config
    key uses, so the seed and the key can never disagree about which
    configuration this is.
    """
    key = f"{model_name}|{subset}|{sigma:.10g}|{repeat}"
    digest = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') & ((1 << 63) - 1)


def gaussian_noise(shape, sigma_eff, generator, dtype=torch.float32):
    """-> a CPU tensor of ``N(0, sigma_eff^2)`` samples with the given shape.

    Drawn on CPU by design (see the module docstring on determinism); the
    caller moves it to the parameter's device.
    """
    if sigma_eff == 0:
        return torch.zeros(shape, dtype=dtype)
    return torch.randn(shape, generator=generator, dtype=dtype) * sigma_eff


def perturbation_error(x, x_hat):
    """-> (relative L2 error, max absolute error), same convention as
    ``fake_quant.quantization_error`` so the two studies' weight-space error
    columns mean the same thing."""
    a = x.detach().to(torch.float64)
    b = x_hat.detach().to(torch.float64)
    diff = b - a
    denom = a.norm().item()
    l2 = diff.norm().item()
    return (l2 / denom if denom > 0 else l2), diff.abs().max().item()


def apply_perturbation(model, subset, sigma, seed):
    """Add Gaussian noise to ``subset``'s parameters of every S4 layer, IN PLACE.

    Only the named subset is perturbed; every other parameter — the other two
    matrices, ``log_dt``, the frame encoder, the FiLM/conditioning net, the
    y-projection, the norms and the head — is left bit-identical. ``A+B+C``
    perturbs all three simultaneously at the same relative sigma, each tensor
    with independent noise.

    Args:
        model:  an S4-family model (has a non-empty ``.layers`` ModuleList).
        subset: one of ``SUBSETS`` — 'A', 'B', 'C', 'A+B+C'.
        sigma:  relative noise scale, >= 0. ``sigma == 0`` is an exact no-op.
        seed:   int; the same seed reproduces the same noise bit-exactly.

    Returns a report dict:
        n_tensors, n_params_perturbed, sigma, seed, mean_rel_l2_err,
        max_rel_l2_err, max_abs_err, mean_realized_sigma_ratio,
        per_tensor: [{layer, subset, param, shape, w_std, sigma_eff,
                      realized_noise_std, realized_sigma_ratio,
                      rel_l2_err, max_abs_err}]

    ``realized_noise_std`` is the measured std of what was actually added, and
    ``realized_sigma_ratio`` is that divided by ``std(W)`` — which should come
    back as ``sigma``. It is the cheap independent witness that the requested
    magnitude took effect, playing the same role ``n_unique`` plays in the
    quantization report.
    """
    sigma = validate_sigma(sigma)
    wanted = subset_members(subset)   # raises on an unknown subset
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))

    details, errs = [], []
    with torch.no_grad():
        for idx, name, attr, param in iter_ssm_params(model):
            if name not in wanted:
                continue
            before = param.detach().clone()
            w_std = before.to(torch.float64).std(unbiased=True).item()
            sigma_eff = tensor_sigma(before, sigma)
            # One generator advanced across tensors: each tensor gets an
            # independent draw, and the whole configuration is reproducible.
            noise = gaussian_noise(tuple(param.shape), sigma_eff, generator)
            param.add_(noise.to(device=param.device, dtype=param.dtype))

            realized = noise.to(torch.float64).std(unbiased=True).item() \
                if noise.numel() > 1 else 0.0
            rel, mx = perturbation_error(before, param)
            errs.append((rel, mx, param.numel()))
            details.append({
                'layer': idx, 'subset': name, 'param': attr,
                'shape': list(param.shape),
                'w_std': w_std,
                'sigma_eff': sigma_eff,
                'realized_noise_std': realized,
                'realized_sigma_ratio': (realized / w_std) if w_std > 0 else None,
                'rel_l2_err': rel, 'max_abs_err': mx,
            })

    if not details:
        raise RuntimeError(
            f"subset {subset!r} matched no parameters in "
            f"{type(model).__name__} — refusing to report an 'evaluation' that "
            f"perturbed nothing")

    ratios = [d['realized_sigma_ratio'] for d in details
              if d['realized_sigma_ratio'] is not None]
    return {
        'n_tensors': len(details),
        'n_params_perturbed': sum(e[2] for e in errs),
        'sigma': sigma,
        'seed': int(seed),
        'mean_rel_l2_err': sum(e[0] for e in errs) / len(errs),
        'max_rel_l2_err': max(e[0] for e in errs),
        'max_abs_err': max(e[1] for e in errs),
        'mean_realized_sigma_ratio': (sum(ratios) / len(ratios)) if ratios
                                     else None,
        'per_tensor': details,
    }
