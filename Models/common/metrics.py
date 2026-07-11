"""Shared evaluation metrics — single source of truth for every table number.

Every cell of the paper's Table 2/3/4 is computed by the functions in this
file, called with the same signature by every model's evaluation script, so
rows are directly comparable by construction.

All functions are pure: no state, no checkpoint I/O, no dataset access.
Trainers accumulate ``all_preds`` / ``all_targets`` arrays over the test set
and pass them here. Never compute table numbers from a running average like
``total_loss / len(loader)`` — that mis-weights the final partial batch.

Inputs may be ``torch.Tensor`` or ``numpy.ndarray``; everything is converted
to float64 numpy internally so R^2 doesn't lose precision on well-fit targets
where SS_res approaches machine epsilon. NaN/Inf values are NOT filtered —
they propagate into the outputs, where they are visible rather than silently
averaged away.

Spaces: the dataset returns positions z-scored and direction components
unit-normalized (already physical). ``compute_table_row(denorm=True)``
converts position columns back to physical units via ``norm_stats.json``;
direction columns are never denormalized.
"""
import functools
import json

import numpy as np
import torch

from Data.config import (
    ACTIVE_TARGET_DIM,
    ACTIVE_TARGET_NAMES,
    DIRECTION_SLICE,
    NORM_STATS_PATH,
    POSITION_SLICE,
)


def _to_f64(x):
    """Coerce a torch tensor or numpy array to a float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _validate_pair(pred, target):
    pred, target = _to_f64(pred), _to_f64(target)
    assert pred.shape == target.shape, (
        f"pred/target shape mismatch: {pred.shape} vs {target.shape}"
    )
    assert pred.ndim == 2 and pred.shape[1] == ACTIVE_TARGET_DIM, (
        f"expected (N, {ACTIVE_TARGET_DIM}) arrays in ACTIVE_TARGET_NAMES "
        f"order, got {pred.shape}"
    )
    return pred, target


def per_target_mse(pred, target):
    """(B,5),(B,5) -> (5,) tensor, mean squared error per target."""
    pred, target = _validate_pair(pred, target)
    return torch.from_numpy(((pred - target) ** 2).mean(axis=0))


def per_target_rmse(pred, target):
    """(B,5),(B,5) -> (5,) tensor, sqrt of per_target_mse."""
    return per_target_mse(pred, target).sqrt()


def per_target_mae(pred, target):
    """(B,5),(B,5) -> (5,) tensor, mean absolute error per target."""
    pred, target = _validate_pair(pred, target)
    return torch.from_numpy(np.abs(pred - target).mean(axis=0))


def per_target_r2(pred, target, target_var=None):
    """Coefficient of determination per target, 1 - SS_res / SS_tot.

    SS_tot uses the variance of the *passed* target tensor (i.e. the
    evaluation set) unless ``target_var`` overrides it. Deliberately NOT the
    frozen training-set variance from target_stats.json: R^2 is an
    evaluation-space quantity, and using train variance would mix
    distributions. Excludes z_entry by construction (it's not in the input
    tensors).

    Returns (5,) tensor.
    """
    pred, target = _validate_pair(pred, target)
    ss_res = ((pred - target) ** 2).mean(axis=0)
    if target_var is None:
        target_var = target.var(axis=0)
    else:
        target_var = _to_f64(target_var)
    return torch.from_numpy(1.0 - ss_res / target_var)


def aggregate_mse(pred, target, weights=None):
    """Scalar. If weights is None, unweighted mean of per-target MSE.
    If weights supplied, weighted sum matching WeightedMSELoss.
    Always report BOTH in tables — the unweighted number is the physical
    quantity, the weighted number is what the model was optimizing.
    """
    pt = per_target_mse(pred, target)
    if weights is None:
        return pt.mean().item()
    w = _to_f64(weights).flatten()
    assert w.shape == (ACTIVE_TARGET_DIM,), (
        f"weights must have shape ({ACTIVE_TARGET_DIM},), got {w.shape}"
    )
    return float((w * pt.numpy()).sum())


def direction_norm_stats(pred_dir):
    """(B,3) -> dict with keys: 'mean_norm', 'std_norm', 'mean_dev_from_unit'.

    For diagnosing how close the 3-DOF direction predictions are to the
    unit-norm constraint that the ground truth satisfies exactly.
    'mean_dev_from_unit' is mean(| ||n|| - 1 |).
    """
    pred_dir = _to_f64(pred_dir)
    assert pred_dir.ndim == 2 and pred_dir.shape[1] == 3, (
        f"expected (B, 3) direction array, got {pred_dir.shape}"
    )
    norms = np.linalg.norm(pred_dir, axis=1)
    return {
        'mean_norm': float(norms.mean()),
        'std_norm': float(norms.std()),
        'mean_dev_from_unit': float(np.abs(norms - 1.0).mean()),
    }


def count_parameters(model):
    """int, trainable parameter count. Reported in Table 2's 'Params' column."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@functools.lru_cache(maxsize=4)
def _load_position_norm(path):
    """(pos_mean, pos_std) for the 2 active position targets, from norm_stats.

    norm_stats.json stores length-3 pos_mean/pos_std for the raw (x, y, z)
    entry positions; z_entry is constant and excluded from the active
    targets, so only the first two components apply here.
    """
    with open(path, 'r') as f:
        stats = json.load(f)
    pos_mean = np.asarray(stats['pos_mean'][:2], dtype=np.float64)
    pos_std = np.asarray(stats['pos_std'][:2], dtype=np.float64)
    return pos_mean, pos_std


def compute_table_row(all_preds, all_targets, model_name, model,
                      denorm=True, norm_stats_path=NORM_STATS_PATH):
    """Compute the row for the paper's main results table (Table 2).

    Args:
        all_preds:   (N, 5) tensor or ndarray, model outputs over the test set
        all_targets: (N, 5) tensor or ndarray, ground truth (same order)
        model_name:  string label for the row
        model:       nn.Module (for param count)
        denorm:      if True, report position metrics (x_entry, y_entry)
                     in physical units by loading pos_mean/pos_std from
                     norm_stats.json. Direction components are already
                     physical and are never denormalized.

    Returns:
        A dict with keys:
            'model', 'mse', 'rmse', 'mae', 'r2', 'nz_mae', 'ny_mae',
            'per_target_mse', 'per_target_rmse', 'per_target_mae',
            'per_target_r2', 'direction_norm_mean_dev', 'params',
            'space' ('normalized' or 'physical'),
            'n_test_samples'

    The scalar 'mse'/'rmse'/'mae' are unweighted means over the 5 active
    targets; 'r2' is the unweighted mean per-target R^2. R^2 is invariant
    under the per-target affine denormalization, so it is identical in both
    spaces. 'nz_mae' and 'ny_mae' are surfaced as top-level scalars because
    n_z is the target quantization hurts most and n_y is the hardest target
    overall.
    """
    pred, target = _validate_pair(all_preds, all_targets)

    if denorm:
        pos_mean, pos_std = _load_position_norm(norm_stats_path)
        pred = pred.copy()
        target = target.copy()
        pred[:, POSITION_SLICE] = pred[:, POSITION_SLICE] * pos_std + pos_mean
        target[:, POSITION_SLICE] = (
            target[:, POSITION_SLICE] * pos_std + pos_mean
        )

    pt_mse = per_target_mse(pred, target)
    pt_rmse = pt_mse.sqrt()
    pt_mae = per_target_mae(pred, target)
    pt_r2 = per_target_r2(pred, target)
    dir_stats = direction_norm_stats(pred[:, DIRECTION_SLICE])

    ny_idx = ACTIVE_TARGET_NAMES.index('n_y')
    nz_idx = ACTIVE_TARGET_NAMES.index('n_z')

    return {
        'model': model_name,
        'mse': pt_mse.mean().item(),
        'rmse': pt_rmse.mean().item(),
        'mae': pt_mae.mean().item(),
        'r2': pt_r2.mean().item(),
        'nz_mae': pt_mae[nz_idx].item(),
        'ny_mae': pt_mae[ny_idx].item(),
        'per_target_mse': pt_mse.tolist(),
        'per_target_rmse': pt_rmse.tolist(),
        'per_target_mae': pt_mae.tolist(),
        'per_target_r2': pt_r2.tolist(),
        'direction_norm_mean_dev': dir_stats['mean_dev_from_unit'],
        'params': count_parameters(model),
        'space': 'physical' if denorm else 'normalized',
        'n_test_samples': pred.shape[0],
    }
