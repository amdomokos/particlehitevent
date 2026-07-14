"""Model-contract tests (Phase 4) — every registered model, one contract.

Parametrized over all seven registered names with tiny configs (legal
overrides of each model's ``default_config``) so the suite stays fast.
S4-specific tests pin down the FiLM centering and the linearity of the
recurrence — the two properties the legacy implementation got wrong.

Run from the repo root:
    pytest Models/common/test_models.py
"""
import json

import pytest
import torch
from torch.utils.data import Dataset

import Models.models_import_all  # noqa: F401 — populates registry
from Data.config import RAW_TARGET_DIM
from Models.common.engine import RunConfig, fit
from Models.common.metrics import count_parameters
from Models.common.registry import build_model, list_models
from Models.rnn.rnn import RNNRegressor
from Models.ssm.s4 import S4Layer

EXPECTED_MODELS = ['cnn', 'cnn_concat', 'gru', 'mlp',
                   's4', 's4_concat', 's4_modulate']

_TINY_S4 = {'d_model': 16, 'd_state': 8, 'n_layers': 2, 'dropout': 0.0,
            'y_module_hidden': 8}
TINY_CONFIGS = {
    'mlp': {'hidden_dims': [16, 8], 'dropout': 0.0},
    'cnn': {'conv_channels': [4, 8], 'head_hidden': 8, 'dropout': 0.0},
    'cnn_concat': {'conv_channels': [4, 8], 'head_hidden': 8, 'dropout': 0.0},
    'gru': {'hidden_size': 16, 'num_layers': 1, 'dropout': 0.0,
            'frame_encoder_channels': [4, 8]},
    's4': _TINY_S4,
    's4_concat': _TINY_S4,
    's4_modulate': _TINY_S4,
}


def make_batch(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 80, 13, 21, generator=g),
            torch.randn(n, generator=g))


def build_tiny(name, seed=0):
    torch.manual_seed(seed)
    return build_model(name, TINY_CONFIGS[name])


def test_registry_contents():
    # subset, not equality: test_engine.py registers test-only models when
    # the whole directory runs in one pytest session
    assert set(EXPECTED_MODELS) <= set(list_models())


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_build_returns_module(name):
    assert isinstance(build_tiny(name), torch.nn.Module)


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_forward_shape_and_dtype(name):
    model = build_tiny(name).eval()
    X, y_module = make_batch()
    out = model(X, y_module)
    assert out.shape == (2, 5)
    assert out.dtype == X.dtype


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_all_params_receive_grad(name):
    model = build_tiny(name).train()
    X, y_module = make_batch()
    model(X, y_module).sum().backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]
    assert dead == [], f"parameters with no grad (dead): {dead}"


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_param_count_positive(name):
    assert count_parameters(build_tiny(name)) > 0


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_determinism_same_seed_same_output(name):
    X, y_module = make_batch()
    out1 = build_tiny(name, seed=7).eval()(X, y_module)
    out2 = build_tiny(name, seed=7).eval()(X, y_module)
    assert torch.equal(out1, out2)


# --- extra coverage for config paths not hit by the registered names -------

def test_cnn_attention_pool():
    cfg = {**TINY_CONFIGS['cnn'], 'temporal_pool': 'attention'}
    torch.manual_seed(0)
    model = build_model('cnn', cfg).train()
    X, y_module = make_batch()
    out = model(X, y_module)
    assert out.shape == (2, 5)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_rnn_class_supports_lstm():
    torch.manual_seed(0)
    model = RNNRegressor(cell='lstm', hidden_size=16, num_layers=1,
                         bidirectional=False, dropout=0.0, use_y_module=True,
                         frame_encoder_channels=[4, 8]).eval()
    X, y_module = make_batch()
    assert model(X, y_module).shape == (2, 5)


# --- S4-specific -----------------------------------------------------------

def test_s4_modulate_at_zero_matches_none():
    """FiLM centering: y_module=0 must reduce 'modulate' to plain S4."""
    X, _ = make_batch(n=4)
    zero = torch.zeros(4)
    out_none = build_tiny('s4', seed=3).eval()(X, zero)
    out_mod = build_tiny('s4_modulate', seed=3).eval()(X, zero)
    assert torch.allclose(out_none, out_mod, atol=1e-5), (
        f"max abs diff {(out_none - out_mod).abs().max().item()}"
    )


def test_s4_modulate_responds_to_y_module():
    model = build_tiny('s4_modulate', seed=3).eval()
    X, _ = make_batch(n=4)
    out_a = model(X, torch.full((4,), -4.0))
    out_b = model(X, torch.full((4,), 4.0))
    assert (out_a - out_b).norm().item() > 1e-3


def test_s4_layer_is_linear():
    """Doubling the input doubles the output — no nonlinearity in the scan."""
    torch.manual_seed(0)
    layer = S4Layer(d_model=8, d_state=4).eval()
    u = torch.randn(2, 80, 8)
    y1 = layer(u)
    y2 = layer(2.0 * u)
    assert torch.allclose(y2, 2.0 * y1, atol=1e-5), (
        f"max abs dev from linearity "
        f"{(y2 - 2.0 * y1).abs().max().item()}"
    )


# --- engine integration ----------------------------------------------------

class SyntheticDataset(Dataset):
    """Emits (X (80,13,21), y_module scalar, Y (6,)) like PixelClusterDataset."""

    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 80, 13, 21, generator=g)
        self.y_module = torch.randn(n, generator=g)
        self.Y = torch.randn(n, RAW_TARGET_DIM, generator=g)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y_module[i], self.Y[i]


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_fit_two_epochs_produces_artifacts(name, tmp_path):
    cfg = RunConfig(
        model_name=name,
        model_config=TINY_CONFIGS[name],
        checkpoint_dir=str(tmp_path / 'ckpt'),
        batch_size=16,
        max_epochs=2,
        num_workers=0,
        early_stop_patience=0,
        data_dir=str(tmp_path),  # unused: datasets are injected
    )
    fit(cfg, SyntheticDataset(48, seed=0), SyntheticDataset(32, seed=1),
        SyntheticDataset(32, seed=2))
    ckpt_dir = tmp_path / 'ckpt'
    assert (ckpt_dir / 'best.pt').exists()
    assert (ckpt_dir / 'latest.pt').exists()
    with open(ckpt_dir / 'final_report.json') as f:
        report = json.load(f)
    assert len(report['table_row']['per_target_mse']) == 5
