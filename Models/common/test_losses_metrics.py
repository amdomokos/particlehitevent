"""Contract tests for Models/common/losses.py and metrics.py.

Run from the repo root (relative artifact paths):
    pytest Models/common/test_losses_metrics.py
"""
import json

import numpy as np
import pytest
import torch

from Data.config import ACTIVE_TARGET_DIM, NORM_STATS_PATH, load_target_stats
from Models.common.losses import WeightedMSELoss, build_default_loss
from Models.common.metrics import (
    compute_table_row,
    direction_norm_stats,
    per_target_mae,
    per_target_mse,
    per_target_r2,
)

UNIT_WEIGHTS = [1.0] * ACTIVE_TARGET_DIM


def test_zero_pred_zero_target():
    loss_fn = WeightedMSELoss(UNIT_WEIGHTS)
    loss, per_target = loss_fn(torch.zeros(8, 5), torch.zeros(8, 5))
    assert loss.item() == 0.0
    assert per_target.tolist() == [0.0] * 5


def test_constant_offset():
    stats = load_target_stats()
    w = torch.tensor(stats['weights'])
    loss_fn = WeightedMSELoss(w)
    offset = 0.5
    loss, per_target = loss_fn(torch.zeros(16, 5), torch.full((16, 5), offset))
    assert torch.allclose(per_target, torch.full((5,), offset ** 2))
    expected = (w * offset ** 2).sum()
    assert torch.allclose(loss, expected)


def test_dir_penalty_zero_on_unit_norm():
    loss_fn = WeightedMSELoss(UNIT_WEIGHTS, lambda_dir=1.0)
    pred = torch.zeros(4, 5)
    pred[:, 2] = 1.0  # unit-norm direction (1, 0, 0)
    loss_with, _ = loss_fn(pred, pred.clone())
    assert loss_with.item() == pytest.approx(0.0)


def test_dir_penalty_on_double_unit_norm():
    # ||n||^2 = 4 -> penalty (4 - 1)^2 = 9
    pred = torch.zeros(4, 5)
    pred[:, 2] = 2.0
    target = pred.clone()
    base, _ = WeightedMSELoss(UNIT_WEIGHTS, lambda_dir=0.0)(pred, target)
    pen, _ = WeightedMSELoss(UNIT_WEIGHTS, lambda_dir=1.0)(pred, target)
    assert (pen - base).item() == pytest.approx(9.0)


def test_batch_size_independence():
    torch.manual_seed(0)
    pred, target = torch.randn(64, 5), torch.randn(64, 5)
    loss_fn = build_default_loss()
    full, _ = loss_fn(pred, target)
    # Same distributional data at half batch: use identical halves so the
    # batch-mean is exactly the same value.
    doubled = torch.cat([pred, pred]), torch.cat([target, target])
    twice, _ = loss_fn(*doubled)
    # relative tolerance: the reduction runs in float32, so absolute drift
    # scales with loss magnitude
    assert full.item() == pytest.approx(twice.item(), rel=1e-6)


def test_wrong_weight_length_raises():
    with pytest.raises(ValueError):
        WeightedMSELoss([1.0, 1.0, 1.0])


def test_non_mean_one_weights_raise():
    with pytest.raises(ValueError):
        WeightedMSELoss([2.0] * 5)  # sums to 10, mean 2


def test_r2_perfect_prediction():
    torch.manual_seed(1)
    target = torch.randn(100, 5)
    r2 = per_target_r2(target, target)
    assert torch.allclose(r2, torch.ones(5, dtype=torch.float64))


def test_direction_norm_stats_unit_inputs():
    torch.manual_seed(2)
    d = torch.randn(50, 3, dtype=torch.float64)
    d = d / d.norm(dim=1, keepdim=True)
    stats = direction_norm_stats(d)
    assert stats['mean_norm'] == pytest.approx(1.0, abs=1e-12)
    assert stats['mean_dev_from_unit'] == pytest.approx(0.0, abs=1e-12)


def test_table_row_denorm_scales_positions_only():
    torch.manual_seed(3)
    target = torch.randn(200, 5)
    pred = target + 0.1 * torch.randn(200, 5)
    model = torch.nn.Linear(3, 5)

    norm_row = compute_table_row(pred, target, 'toy', model, denorm=False)
    phys_row = compute_table_row(pred, target, 'toy', model, denorm=True)

    with open(NORM_STATS_PATH) as f:
        pos_std = np.asarray(json.load(f)['pos_std'][:2])

    assert norm_row['space'] == 'normalized'
    assert phys_row['space'] == 'physical'
    for i, std in enumerate(pos_std):
        assert phys_row['per_target_mae'][i] == pytest.approx(
            norm_row['per_target_mae'][i] * std)
        assert phys_row['per_target_mse'][i] == pytest.approx(
            norm_row['per_target_mse'][i] * std ** 2)
    # direction metrics identical across spaces
    assert phys_row['per_target_mae'][2:] == pytest.approx(
        norm_row['per_target_mae'][2:])
    assert phys_row['direction_norm_mean_dev'] == pytest.approx(
        norm_row['direction_norm_mean_dev'])
    # R^2 is affine-invariant per target
    assert phys_row['per_target_r2'] == pytest.approx(norm_row['per_target_r2'])
    assert phys_row['params'] == sum(p.numel() for p in model.parameters())


def test_metrics_hand_computation():
    pred = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                         [3.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.zeros(2, 5)
    assert per_target_mse(pred, target)[0].item() == pytest.approx(5.0)
    assert per_target_mae(pred, target)[0].item() == pytest.approx(2.0)
