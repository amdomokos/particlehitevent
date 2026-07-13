"""Engine-mechanics tests (Phase 3) — no real model, no chunk files.

Synthetic datasets emit the ``(X, y_module, Y)`` tuple contract of
PixelClusterDataset; trivial models are registered under test-only names.
The frozen JSON artifacts (target_stats.json / norm_stats.json) ARE read —
they are checked into git and required by the loss/metrics contract.

Run from the repo root:
    pytest Models/common/test_engine.py
"""
import json
import os

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from Data.config import ACTIVE_INDICES, RAW_TARGET_DIM
from Models import train as train_mod
from Models.common.engine import (
    RunConfig,
    RunState,
    evaluate,
    fit,
    load_checkpoint,
    save_checkpoint,
    smoke_test,
)
from Models.common.losses import build_default_loss
from Models.common.registry import build_model, list_models, register_model

FIXED_OUTPUT = [0.1, -0.2, 0.3, -0.4, 0.5]


class SyntheticDataset(Dataset):
    """Emits (X (80,13,21), y_module scalar, Y (6,)) like PixelClusterDataset."""

    def __init__(self, n=96, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 80, 13, 21, generator=g)
        self.y_module = torch.randn(n, generator=g)
        self.Y = torch.randn(n, RAW_TARGET_DIM, generator=g)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y_module[i], self.Y[i]


class TinyModel(torch.nn.Module):
    """Linear head over (mean(X), y_module) -> (B, 5)."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 5)

    def forward(self, X, y_module):
        feats = torch.stack([X.mean(dim=(1, 2, 3)), y_module.float()], dim=1)
        return self.fc(feats)


class FixedModel(torch.nn.Module):
    """Returns a constant (B, 5) row regardless of input."""

    def __init__(self, value):
        super().__init__()
        self.register_buffer('out', torch.tensor(value, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(1))  # so AdamW has params

    def forward(self, X, y_module):
        return self.out.expand(X.shape[0], 5) + 0.0 * self.bias


if 'test-tiny' not in list_models():
    @register_model('test-tiny')
    def _build_tiny(config):
        return TinyModel()

    @register_model('test-fixed')
    def _build_fixed(config):
        return FixedModel(config.get('value', FIXED_OUTPUT))


def make_cfg(tmp_path, **overrides):
    kw = dict(
        model_name='test-tiny',
        checkpoint_dir=str(tmp_path / 'ckpt'),
        batch_size=16,
        max_epochs=2,
        num_workers=0,
        precision='auto',
        early_stop_patience=0,
        data_dir=str(tmp_path),  # unused: datasets are injected
    )
    kw.update(overrides)
    return RunConfig(**kw)


@pytest.fixture
def datasets():
    return (SyntheticDataset(96, seed=0), SyntheticDataset(48, seed=1),
            SyntheticDataset(48, seed=2))


def test_smoke_test_leaves_no_checkpoints(tmp_path, datasets):
    train_ds, val_ds, _ = datasets
    cfg = make_cfg(tmp_path, max_epochs=1)
    smoke_test(cfg, train_dataset=train_ds, val_dataset=val_ds)
    ckpt_dir = tmp_path / 'ckpt'
    leftover = list(ckpt_dir.glob('*')) if ckpt_dir.exists() else []
    assert leftover == [], f"smoke test left files behind: {leftover}"


def test_fit_produces_artifacts(tmp_path, datasets):
    train_ds, val_ds, test_ds = datasets
    cfg = make_cfg(tmp_path)
    summary = fit(cfg, train_ds, val_ds, test_ds)
    ckpt_dir = tmp_path / 'ckpt'
    assert (ckpt_dir / 'best.pt').exists()
    assert (ckpt_dir / 'latest.pt').exists()
    assert (ckpt_dir / 'final_report.json').exists()
    assert summary['epochs_trained'] == 2
    with open(ckpt_dir / 'final_report.json') as f:
        report = json.load(f)
    row = report['table_row']
    assert row['model'] == 'test-tiny'
    assert len(row['per_target_mse']) == 5
    assert report['metadata']['precision'] in ('bf16', 'fp16', 'fp32')
    assert len(report['history']['train_loss']) == 2


def test_resume_restores_state(tmp_path, datasets):
    train_ds, val_ds, test_ds = datasets
    cfg1 = make_cfg(tmp_path, max_epochs=1)
    fit(cfg1, train_ds, val_ds, test_ds)
    latest = tmp_path / 'ckpt' / 'latest.pt'
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 1
    assert ckpt['step'] == 6  # 96 samples / batch 16 = 6 optimizer steps
    # max_epochs is NOT a material field — resume with more epochs is fine
    cfg2 = make_cfg(tmp_path, max_epochs=3)
    summary = fit(cfg2, train_ds, val_ds, test_ds, resume=True)
    assert summary['resumed_from_epoch'] == 1
    assert summary['epochs_trained'] == 3


def test_resume_rejects_material_diff(tmp_path, datasets):
    train_ds, val_ds, test_ds = datasets
    fit(make_cfg(tmp_path, max_epochs=1), train_ds, val_ds, test_ds)
    cfg_bad = make_cfg(tmp_path, max_epochs=2, lr=5e-4)
    with pytest.raises(RuntimeError, match='lr'):
        fit(cfg_bad, train_ds, val_ds, test_ds, resume=True)


def test_checkpoint_metadata(tmp_path, datasets):
    train_ds, val_ds, test_ds = datasets
    cfg = make_cfg(tmp_path, max_epochs=1)
    fit(cfg, train_ds, val_ds, test_ds)
    ckpt = torch.load(tmp_path / 'ckpt' / 'best.pt', map_location='cpu',
                      weights_only=False)
    md = ckpt['metadata']
    for key in ('git_commit', 'gpu', 'precision', 'target_stats_hash',
                'norm_stats_hash', 'seed', 'timestamp', 'torch_version',
                'loss_weights'):
        assert key in md, f"metadata missing '{key}'"
    assert md['precision'] != 'auto'
    assert len(md['loss_weights']) == 5
    assert len(md['target_stats_hash']) == 64
    assert ckpt['run_config']['model_name'] == 'test-tiny'


def test_evaluate_accumulates_correctly(datasets):
    _, val_ds, _ = datasets
    model = build_model('test-fixed', {'value': FIXED_OUTPUT})
    loader = DataLoader(val_ds, batch_size=13)  # deliberately uneven batches
    loss = build_default_loss()
    res = evaluate(model, loader, loss, torch.device('cpu'), None)
    Y_active = val_ds.Y[:, list(ACTIVE_INDICES)]
    expected = ((torch.tensor(FIXED_OUTPUT) - Y_active) ** 2).mean(dim=0)
    assert torch.allclose(res['per_target_mse'].float(), expected, atol=1e-6)
    assert res['n_samples'] == len(val_ds)
    assert res['aggregate_mse_unweighted'] == pytest.approx(
        expected.mean().item(), rel=1e-6)


def test_save_load_checkpoint_roundtrip(tmp_path):
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = make_cfg(tmp_path)
    path = str(tmp_path / 'rt.pt')
    save_checkpoint(path, model, opt, None, None,
                    RunState(epoch=3, step=42, best_val_mse=0.5), cfg,
                    {'precision': 'fp32'})
    twin = TinyModel()
    ckpt = load_checkpoint(path, twin)
    assert ckpt['epoch'] == 3 and ckpt['step'] == 42
    assert ckpt['best_val_mse'] == 0.5
    for k, v in model.state_dict().items():
        assert torch.equal(v, twin.state_dict()[k])


def test_train_cli_smoke_exits_zero(tmp_path, datasets):
    train_ds, val_ds, _ = datasets
    rc = train_mod.main(
        ['--model', 'test-tiny', '--smoke',
         '--checkpoint-dir', str(tmp_path / 'ckpt'),
         '--batch-size', '16', '--num-workers', '0'],
        train_dataset=train_ds, val_dataset=val_ds)
    assert rc == 0
