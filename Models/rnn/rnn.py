"""Recurrent baseline (Phase 4) — weight-shared frame CNN + GRU/LSTM.

Encoder uses ReLU (matches the CNN baseline for row-comparability).
Head uses GELU: the RNN is not a quantization-sweep candidate, so
the smoother activation is preferred for optimization.
"""
import torch
import torch.nn as nn

from Data.config import ACTIVE_TARGET_DIM
from Models.common.registry import register_model

default_config = {
    'cell': 'gru',              # 'gru' | 'lstm'
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': False,
    'dropout': 0.1,
    'use_y_module': True,
    'frame_encoder_channels': [32, 64],
}

_T, _H, _W = 80, 13, 21


class FrameEncoder(nn.Module):
    """Weight-shared 2-layer strided CNN per frame: (B, 80, 13, 21) -> (B, 80, D)."""

    def __init__(self, channels, out_dim):
        super().__init__()
        c1, c2 = channels
        self.convs = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1),   # 13x21 -> 7x11
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),  # 7x11 -> 4x6
            nn.ReLU(),
        )
        self.proj = nn.Linear(c2 * 4 * 6, out_dim)

    def forward(self, X):
        B = X.shape[0]
        z = self.convs(X.reshape(B * _T, 1, _H, _W))
        return self.proj(z.reshape(B, _T, -1))


class RNNRegressor(nn.Module):
    """Frame encoder -> GRU/LSTM over T=80 -> last hidden state -> head."""

    def __init__(self, cell, hidden_size, num_layers, bidirectional,
                 dropout, use_y_module, frame_encoder_channels):
        super().__init__()
        if cell not in ('gru', 'lstm'):
            raise ValueError(f"cell must be 'gru' or 'lstm', got '{cell}'")
        self.use_y_module = use_y_module
        self.bidirectional = bidirectional
        self.encoder = FrameEncoder(frame_encoder_channels, hidden_size)
        rnn_cls = nn.GRU if cell == 'gru' else nn.LSTM
        self.rnn = rnn_cls(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat = hidden_size * (2 if bidirectional else 1)
        feat += 1 if use_y_module else 0
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, ACTIVE_TARGET_DIM),
        )

    def forward(self, X, y_module):
        frames = self.encoder(X)
        _, state = self.rnn(frames)
        h_n = state[0] if isinstance(state, tuple) else state  # LSTM: (h, c)
        # h_n: (num_layers * num_directions, B, hidden). Last layer's state(s).
        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feat = h_n[-1]
        if self.use_y_module:
            feat = torch.cat(
                [feat, y_module.to(feat.dtype).unsqueeze(1)], dim=1)
        return self.head(feat)


def _merge(config):
    cfg = {**default_config, **(config or {})}
    unknown = set(cfg) - set(default_config)
    if unknown:
        raise ValueError(f"unknown rnn config keys: {sorted(unknown)}")
    return cfg


@register_model('gru')
def build_gru(config):
    cfg = _merge(config)
    cfg['cell'] = 'gru'
    cfg['use_y_module'] = True
    return RNNRegressor(cfg['cell'], cfg['hidden_size'], cfg['num_layers'],
                        cfg['bidirectional'], cfg['dropout'],
                        cfg['use_y_module'], cfg['frame_encoder_channels'])
