"""Per-frame CNN baseline (Phase 4) — spatial convs, temporal pooling.

Uses ReLU (not GELU): the CNN is a quantization-sweep candidate, and ReLU
quantizes materially cleaner while being within noise of GELU at FP32 for
this architecture class (arXiv:2209.06383; arXiv:2004.09602).
"""
import torch
import torch.nn as nn

from Data.config import ACTIVE_TARGET_DIM
from Models.common.registry import register_model

default_config = {
    'conv_channels': [32, 64, 128],
    'kernel_size': 3,
    'temporal_pool': 'mean',    # 'mean' | 'attention'
    'head_hidden': 128,
    'dropout': 0.1,
    'use_y_module': False,
}

_T, _H, _W = 80, 13, 21


class AttentionPool(nn.Module):
    """Learned single-head pooling over the time axis: (B, T, D) -> (B, D)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, frames):
        attn = torch.softmax(self.score(frames), dim=1)  # (B, T, 1)
        return (attn * frames).sum(dim=1)


class CNNRegressor(nn.Module):
    """Conv2d stack per frame (13x21 preserved), temporal pool, MLP head.

    (B, 80, 13, 21) -> (B*80, 1, 13, 21) -> convs -> (B, 80, D) with
    D = conv_channels[-1] * 13 * 21 -> pool over T -> head -> (B, 5).
    """

    def __init__(self, conv_channels, kernel_size, temporal_pool,
                 head_hidden, dropout, use_y_module):
        super().__init__()
        if temporal_pool not in ('mean', 'attention'):
            raise ValueError(f"temporal_pool must be 'mean' or 'attention', "
                             f"got '{temporal_pool}'")
        self.temporal_pool = temporal_pool
        self.use_y_module = use_y_module

        convs, in_ch = [], 1
        pad = kernel_size // 2  # preserve 13x21 at every conv
        for ch in conv_channels:
            convs += [nn.Conv2d(in_ch, ch, kernel_size, padding=pad),
                      nn.ReLU()]
            in_ch = ch
        self.convs = nn.Sequential(*convs)
        frame_dim = conv_channels[-1] * _H * _W

        self.pool = (AttentionPool(frame_dim)
                     if temporal_pool == 'attention' else None)
        head_in = frame_dim + (1 if use_y_module else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, ACTIVE_TARGET_DIM),
        )

    def forward(self, X, y_module):
        B = X.shape[0]
        z = self.convs(X.reshape(B * _T, 1, _H, _W))
        frames = z.reshape(B, _T, -1)                       # (B, 80, D)
        pooled = (self.pool(frames) if self.pool is not None
                  else frames.mean(dim=1))                  # (B, D)
        if self.use_y_module:
            pooled = torch.cat(
                [pooled, y_module.to(pooled.dtype).unsqueeze(1)], dim=1)
        return self.head(pooled)


def _merge(config):
    cfg = {**default_config, **(config or {})}
    unknown = set(cfg) - set(default_config)
    if unknown:
        raise ValueError(f"unknown cnn config keys: {sorted(unknown)}")
    return cfg


def _build(config, use_y_module):
    cfg = _merge(config)
    cfg['use_y_module'] = use_y_module
    return CNNRegressor(cfg['conv_channels'], cfg['kernel_size'],
                        cfg['temporal_pool'], cfg['head_hidden'],
                        cfg['dropout'], cfg['use_y_module'])


@register_model('cnn')
def build_cnn(config):
    return _build(config, use_y_module=False)


@register_model('cnn_concat')
def build_cnn_concat(config):
    return _build(config, use_y_module=True)
