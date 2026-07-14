"""MLP baseline (Phase 4) — flatten-everything reference point for Table 2.

Uses ReLU (not GELU): the MLP is a quantization-sweep candidate, and ReLU
quantizes materially cleaner while being within noise of GELU at FP32 for
this architecture class (arXiv:2209.06383; arXiv:2004.09602).
"""
import torch
import torch.nn as nn

from Data.config import ACTIVE_TARGET_DIM
from Models.common.registry import register_model

default_config = {
    'hidden_dims': [512, 256, 128],
    'dropout': 0.1,
    'use_y_module': True,
}

_INPUT_DIM = 80 * 13 * 21  # (T, H, W) flattened


class MLPRegressor(nn.Module):
    """Flatten X to (B, 21840), optionally concat y_module, MLP to (B, 5)."""

    def __init__(self, hidden_dims, dropout, use_y_module):
        super().__init__()
        self.use_y_module = use_y_module
        in_dim = _INPUT_DIM + (1 if use_y_module else 0)
        blocks = []
        for h in hidden_dims:
            blocks += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.body = nn.Sequential(*blocks)
        self.out = nn.Linear(in_dim, ACTIVE_TARGET_DIM)

    def forward(self, X, y_module):
        z = X.flatten(1)
        if self.use_y_module:
            z = torch.cat([z, y_module.to(z.dtype).unsqueeze(1)], dim=1)
        return self.out(self.body(z))


def _merge(config):
    cfg = {**default_config, **(config or {})}
    unknown = set(cfg) - set(default_config)
    if unknown:
        raise ValueError(f"unknown mlp config keys: {sorted(unknown)}")
    return cfg


@register_model('mlp')
def build_mlp(config):
    cfg = _merge(config)
    return MLPRegressor(cfg['hidden_dims'], cfg['dropout'],
                        cfg['use_y_module'])
