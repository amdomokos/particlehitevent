"""Shared training loss for every model in the comparison sweep (Phase 2).

Single source of truth for the training objective. Every model (MLP, CNN,
GRU, S4, target-aware SSM, quantized SSM) constructs its loss through
``build_default_loss()`` so that all rows of the paper's results tables are
optimized against byte-identical code and the frozen weights from
``preprocessed_data/target_stats.json``. Do not copy-paste variants of this
loss into model directories — that is exactly the drift this module exists
to prevent.

Operates in the same space the dataset returns: positions z-scored,
direction components unit-normalized. Physical-unit reporting is a
metrics-module concern (``Models/common/metrics.py``).
"""
import torch
import torch.nn as nn

from Data.config import ACTIVE_TARGET_DIM, DIRECTION_SLICE, load_target_stats


class WeightedMSELoss(nn.Module):
    """Weighted MSE over the 5 active targets, with optional direction penalty.

    Reduces over the batch first, then weights and sums over targets, so:
      - the returned scalar loss is batch-size-independent
      - the returned per-target vector is free for logging / metrics reuse

    Weights are registered as a buffer, so:
      - .to(device) moves them
      - state_dict() carries them, so every saved checkpoint documents
        the exact loss it was trained under

    The direction penalty uses (||n||^2 - 1)^2, not (||n|| - 1)^2, to avoid
    the sqrt gradient singularity at the origin. Same minimizer, safer
    gradients.

    Inputs to forward():
        pred:   (B, 5) — predictions in ACTIVE_TARGET_NAMES order
        target: (B, 5) — ground truth in ACTIVE_TARGET_NAMES order

    Returns:
        loss:       scalar tensor
        per_target: (5,) tensor of per-target MSE, detached
    """

    def __init__(self, weights, lambda_dir=0.0):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32).flatten()
        if w.numel() != ACTIVE_TARGET_DIM:
            raise ValueError(
                f"weights must have length {ACTIVE_TARGET_DIM} "
                f"(ACTIVE_TARGET_DIM), got {w.numel()}; load them via "
                f"Data.config.load_target_stats()['weights'] or use "
                f"build_default_loss()"
            )
        if abs(w.sum().item() - ACTIVE_TARGET_DIM) >= 1e-3:
            raise ValueError(
                f"weights must be mean-1 normalized (sum == "
                f"{ACTIVE_TARGET_DIM}), got sum={w.sum().item():.6f}; load "
                f"them via Data.config.load_target_stats()['weights'] rather "
                f"than passing arbitrary values"
            )
        self.register_buffer('w', w)
        self.lambda_dir = float(lambda_dir)

    def forward(self, pred, target):
        assert pred.shape == target.shape, (
            f"pred/target shape mismatch: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
        assert pred.shape[-1] == ACTIVE_TARGET_DIM, (
            f"expected last dim {ACTIVE_TARGET_DIM} (active targets), got "
            f"{pred.shape[-1]}"
        )

        per_target = ((pred - target) ** 2).mean(dim=0)
        loss = (self.w * per_target).sum()

        if self.lambda_dir > 0:
            pred_dir = pred[:, DIRECTION_SLICE]
            n_sq = (pred_dir ** 2).sum(dim=1)
            dir_penalty = ((n_sq - 1.0) ** 2).mean()
            loss = loss + self.lambda_dir * dir_penalty

        return loss, per_target.detach()

    def extra_repr(self):
        w = [round(v, 4) for v in self.w.tolist()]
        return f"weights={w}, lambda_dir={self.lambda_dir}"


def build_default_loss(lambda_dir=0.0):
    """Construct a WeightedMSELoss from the frozen target_stats.json.

    This is the canonical loss factory used by every training script.
    Reads weights via Data.config.load_target_stats() so drift between
    the JSON and the loss cannot occur silently.
    """
    stats = load_target_stats()
    return WeightedMSELoss(weights=stats['weights'], lambda_dir=lambda_dir)
