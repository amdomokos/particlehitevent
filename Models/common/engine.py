"""Shared training engine (Phase 3) — every model trains through this file.

One trainer for all six architectures (MLP, CNN, GRU, S4, target-aware SSM,
quantized SSM) so every row of the paper's tables is produced by identical
optimizer, scheduler, AMP, seeding, checkpointing, and evaluation code.
Models plug in via ``Models.common.registry`` and only ever see
``(X, y_module) -> (B, 5)``; the 6->5 active-target selection happens here,
once, at the batch boundary.

Design decisions the docstrings below rely on:
  - "val MSE" for early stopping / best-checkpoint selection is the
    *weighted* aggregate MSE (same weights as the training loss), so model
    selection matches the optimization objective. Tables report both
    weighted and unweighted.
  - Evaluation accumulates all predictions/targets and computes metrics
    once at the end — never a running mean over batches, which would
    mis-weight the final partial batch.
  - Checkpoints are written atomically (tmp file + os.replace) so a pod
    killed mid-save cannot corrupt latest.pt.
  - A degenerate split (0 train or 0 val samples) raises immediately —
    fast-fail beats silently training without validation.
"""
import contextlib
import copy
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256

import torch
from torch.utils.data import DataLoader, Subset

from Data.config import (
    ACTIVE_INDICES,
    ACTIVE_TARGET_DIM,
    DIRECTION_SLICE,
    NORM_STATS_PATH,
    SEED,
    TARGET_STATS_PATH,
    load_target_stats,
)
from Models.common.device import get_amp_dtype, get_device, gpu_summary
from Models.common.losses import build_default_loss
from Models.common.metrics import (
    aggregate_mse,
    compute_table_row,
    direction_norm_stats,
    per_target_mae,
    per_target_mse,
)
from Models.common.registry import build_model
from Models.common.seeding import seed_everything, worker_init_fn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ACTIVE_IDX = list(ACTIVE_INDICES)
_SMOKE_SUBSET = 256

# RunConfig fields that must match between a resumed checkpoint and the
# current run. max_epochs / patience / workers / etc. may differ freely.
_RESUME_MATERIAL_FIELDS = (
    'model_name', 'model_config', 'batch_size', 'lr', 'seed',
    'lambda_dir', 'data_dir',
)


@dataclass
class RunConfig:
    model_name: str
    checkpoint_dir: str            # must be persistent storage on RunPod
    model_config: dict = field(default_factory=dict)
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    lambda_dir: float = 0.01
    precision: str = 'auto'        # 'auto' | 'bf16' | 'fp16' | 'fp32'
    seed: int = SEED
    strict_deterministic: bool = False
    num_workers: int = 4
    early_stop_patience: int = 10  # <= 0 disables early stopping
    grad_clip: float = 1.0         # 0 disables
    data_dir: str = os.path.join(_REPO_ROOT, 'preprocessed_data')
    smoke: bool = False


@dataclass
class RunState:
    epoch: int = 0                 # completed epochs
    step: int = 0                  # optimizer steps
    best_val_mse: float = math.inf


def _resolve_precision(precision, device):
    """-> (amp_dtype or None, resolved name). Never returns 'auto'."""
    names = {torch.bfloat16: 'bf16', torch.float16: 'fp16', None: 'fp32'}
    if precision == 'auto':
        dtype = get_amp_dtype(device)
        return dtype, names[dtype]
    if precision == 'fp32':
        return None, 'fp32'
    if precision == 'bf16':
        if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this GPU does not support it")
        return torch.bfloat16, 'bf16'
    if precision == 'fp16':
        if device.type == 'cpu':
            raise RuntimeError("fp16 autocast is not supported on CPU; use bf16 or fp32")
        return torch.float16, 'fp16'
    raise ValueError(f"precision must be auto|bf16|fp16|fp32, got '{precision}'")


def _autocast(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _batch_to_device(batch, device):
    """Move a (X, y_module, Y) batch and select the 5 active target columns.

    The single place the 6-column raw Y is narrowed — no model ever sees it.
    """
    X, y_module, Y = batch
    return X.to(device), y_module.to(device), Y[:, _ACTIVE_IDX].to(device)


def _check_pred(pred):
    assert pred.shape[-1] == ACTIVE_TARGET_DIM, (
        f"model must output (B, {ACTIVE_TARGET_DIM}) in ACTIVE_TARGET_NAMES "
        f"order, got shape {tuple(pred.shape)}"
    )


def _make_loader(dataset, cfg, shuffle):
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
    )


def _fmt_vec(t):
    return '[' + ', '.join(f'{v:.4g}' for v in t.tolist()) + ']'


def _git_info():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=_REPO_ROOT, text=True).strip())
        return commit, dirty
    except Exception:
        return 'unknown', None


def _sha256_file(path):
    with open(path, 'rb') as f:
        return sha256(f.read()).hexdigest()


def _build_metadata(cfg, device, extra_metadata=None):
    commit, dirty = _git_info()
    return {
        'seed': cfg.seed,
        'git_commit': commit,
        'git_dirty': dirty,
        'gpu': gpu_summary(device),
        'precision': cfg.precision,  # overridden with resolved value below
        'target_stats_hash': _sha256_file(TARGET_STATS_PATH),
        'norm_stats_hash': _sha256_file(NORM_STATS_PATH),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'torch_version': torch.__version__,
        'loss_weights': load_target_stats()['weights'],
        'lambda_dir': cfg.lambda_dir,
        **(extra_metadata or {}),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, run_state,
                    cfg, extra_metadata=None):
    """Save a full-resume checkpoint with reproducibility metadata.

    ``fit()`` always passes the *resolved* precision (never 'auto') via
    ``extra_metadata``, which overrides the cfg value in the metadata block.
    Written atomically: tmp file then ``os.replace``.
    """
    params = list(model.parameters())
    device = params[0].device if params else torch.device('cpu')
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': (scaler.state_dict()
                              if scaler is not None and scaler.is_enabled()
                              else None),
        'model_config': cfg.model_config,
        'model_name': cfg.model_name,
        'run_config': asdict(cfg),
        'epoch': run_state.epoch,
        'step': run_state.step,
        'best_val_mse': run_state.best_val_mse,
        'metadata': _build_metadata(cfg, device, extra_metadata),
    }
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load a checkpoint. Model-only if optimizer/scheduler/scaler are None
    (evaluation); full resume if provided. Returns the checkpoint dict.

    Optimizer state tensors are moved to the model's parameter device by
    torch's ``load_state_dict``, so call this after ``model.to(device)``.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and ckpt['scaler_state_dict'] is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt


def _check_resume_config(saved_run_config, cfg):
    diffs = []
    current = asdict(cfg)
    for k in _RESUME_MATERIAL_FIELDS:
        if saved_run_config.get(k) != current[k]:
            diffs.append(f"  {k}: checkpoint={saved_run_config.get(k)!r} "
                         f"current={current[k]!r}")
    if diffs:
        msg = ("Refusing to resume: checkpoint run_config differs in "
               "fields that affect training:\n" + '\n'.join(diffs))
        print(msg)
        raise RuntimeError(msg)


def _collect_predictions(model, loader, device, amp_dtype):
    """-> (preds, targets), each (N, 5) float32 on CPU, accumulated over the
    whole loader. float64 conversion happens inside the metrics module."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            X, y_module, Y_active = _batch_to_device(batch, device)
            with _autocast(device, amp_dtype):
                pred = model(X, y_module)
            _check_pred(pred)
            preds.append(pred.float().cpu())
            targets.append(Y_active.float().cpu())
    return torch.cat(preds), torch.cat(targets)


def evaluate(model, loader, loss, device, amp_dtype):
    """One evaluation pass; metrics computed once over all accumulated
    predictions (see module docstring on partial-batch weighting)."""
    preds, targets = _collect_predictions(model, loader, device, amp_dtype)
    w = loss.w.detach().cpu()
    return {
        'per_target_mse': per_target_mse(preds, targets),
        'aggregate_mse_unweighted': aggregate_mse(preds, targets),
        'aggregate_mse_weighted': aggregate_mse(preds, targets, weights=w),
        'per_target_mae': per_target_mae(preds, targets),
        'direction_norm_stats': direction_norm_stats(preds[:, DIRECTION_SLICE]),
        'n_samples': preds.shape[0],
    }


def _train_epoch(model, loader, loss, optimizer, scaler, device, amp_dtype,
                 cfg, run_state, label):
    """One training epoch. Returns (mean loss, per-target train MSE vector).

    The per-target vector is a sample-weighted mean of the loss's per-batch
    vectors — fine for logging (table numbers come from evaluate())."""
    model.train()
    total, n = 0.0, 0
    pt_sum = torch.zeros(ACTIVE_TARGET_DIM, dtype=torch.float64)
    for i, batch in enumerate(loader):
        X, y_module, Y_active = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            pred = model(X, y_module)
        _check_pred(pred)
        # loss in fp32 regardless of autocast dtype
        loss_val, per_target = loss(pred.float(), Y_active.float())
        scaler.scale(loss_val).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        run_state.step += 1
        b = X.shape[0]
        total += loss_val.item() * b
        n += b
        pt_sum += per_target.double().cpu() * b
        if (i + 1) % 50 == 0:
            print(f"  [{label}] batch {i + 1}/{len(loader)} loss={total / n:.4f}")
    return total / n, pt_sum / n


def _fast_fail_check(model, loss, train_loader, device, amp_dtype, cfg,
                     resolved_precision):
    """One forward+backward+step on a throwaway copy of the model, plus a
    checkpoint write/delete, before committing to a full (paid) run. The
    copy leaves the real model, optimizer, and RNG-visible state untouched
    whether the run is fresh or resumed."""
    probe = copy.deepcopy(model)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    probe_scaler = torch.amp.GradScaler(device.type,
                                        enabled=(resolved_precision == 'fp16'))
    batch = next(iter(train_loader))
    X, y_module, Y_active = _batch_to_device(batch, device)
    with _autocast(device, amp_dtype):
        pred = probe(X, y_module)
    _check_pred(pred)
    loss_val, _ = loss(pred.float(), Y_active.float())
    if not torch.isfinite(loss_val):
        raise RuntimeError(f"preflight: non-finite loss {loss_val.item()} on first batch")
    if loss_val.item() <= 0:
        raise RuntimeError(f"preflight: loss {loss_val.item()} <= 0 on first batch")
    probe_scaler.scale(loss_val).backward()
    probe_scaler.step(probe_opt)
    probe_scaler.update()

    probe_path = os.path.join(cfg.checkpoint_dir, 'smoke_ok.tmp')
    save_checkpoint(probe_path, probe, probe_opt, None, probe_scaler,
                    RunState(), cfg, {'precision': resolved_precision})
    if not os.path.exists(probe_path):
        raise RuntimeError(f"preflight: checkpoint write to {probe_path} failed")
    os.remove(probe_path)
    del probe, probe_opt
    print(f"[preflight] OK — loss={loss_val.item():.4f}, "
          f"checkpoint dir {cfg.checkpoint_dir} writable")


def fit(cfg, train_dataset=None, val_dataset=None, test_dataset=None,
        resume=False):
    """Train one model end to end; returns a JSON-serializable summary dict.

    Datasets may be injected (tests, custom subsets); by default they are
    built as ``PixelClusterDataset(cfg.data_dir, split)`` with the canonical
    chunk split — val/test chunks never reach the training loader.
    """
    seed_everything(cfg.seed, strict=cfg.strict_deterministic)
    device = get_device()
    amp_dtype, precision = _resolve_precision(cfg.precision, device)
    print(f"[precision] {precision} (requested '{cfg.precision}')")

    if train_dataset is None or val_dataset is None:
        from Data.dataset_v2 import PixelClusterDataset
        train_dataset = train_dataset or PixelClusterDataset(cfg.data_dir, 'train')
        val_dataset = val_dataset or PixelClusterDataset(cfg.data_dir, 'val')
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"degenerate split: {len(train_dataset)} train / "
            f"{len(val_dataset)} val samples — check data_dir and split ratios"
        )

    model = build_model(cfg.model_name, cfg.model_config).to(device)
    loss = build_default_loss(lambda_dir=cfg.lambda_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs) if cfg.max_epochs > 1 else None)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == 'fp16'))

    if cfg.smoke:
        banner = '=' * 24
        print(f"\n{banner} [SMOKE] one tiny epoch, no checkpoints {banner}")
        sub_train = Subset(train_dataset,
                           range(min(_SMOKE_SUBSET, len(train_dataset))))
        sub_val = Subset(val_dataset,
                         range(min(_SMOKE_SUBSET, len(val_dataset))))
        run_state = RunState()
        train_loss, pt_train = _train_epoch(
            model, _make_loader(sub_train, cfg, shuffle=True), loss,
            optimizer, scaler, device, amp_dtype, cfg, run_state, 'smoke')
        val = evaluate(model, _make_loader(sub_val, cfg, shuffle=False),
                       loss, device, amp_dtype)
        print(f"[SMOKE] train_loss={train_loss:.4f} "
              f"val_mse={val['aggregate_mse_weighted']:.4f} "
              f"per_target_mse={_fmt_vec(val['per_target_mse'])}")
        print(f"{banner} [SMOKE] done {banner}\n")
        return {
            'smoke': True,
            'train_loss': train_loss,
            'train_per_target_mse': pt_train.tolist(),
            'val_mse_weighted': val['aggregate_mse_weighted'],
            'val_per_target_mse': val['per_target_mse'].tolist(),
            'n_train_samples': len(sub_train),
            'precision': precision,
            'device': device.type,
        }

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loader = _make_loader(train_dataset, cfg, shuffle=True)
    val_loader = _make_loader(val_dataset, cfg, shuffle=False)
    best_path = os.path.join(cfg.checkpoint_dir, 'best.pt')
    latest_path = os.path.join(cfg.checkpoint_dir, 'latest.pt')
    meta_extra = {'precision': precision}

    run_state = RunState()
    start_epoch = 0
    if resume:
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"--resume requested but {latest_path} not found")
        ckpt = load_checkpoint(latest_path, model, optimizer, scheduler, scaler)
        _check_resume_config(ckpt['run_config'], cfg)
        run_state = RunState(epoch=ckpt['epoch'], step=ckpt['step'],
                             best_val_mse=ckpt['best_val_mse'])
        start_epoch = ckpt['epoch']
        print(f"[resume] restored epoch={start_epoch} step={run_state.step} "
              f"best_val_mse={run_state.best_val_mse:.6f} from {latest_path}")

    _fast_fail_check(model, loss, train_loader, device, amp_dtype, cfg, precision)

    history = {'train_loss': [], 'val_mse': [], 'val_per_target_mse': []}
    epochs_since_improve = 0
    try:
        for epoch in range(start_epoch, cfg.max_epochs):
            t0 = time.time()
            label = f"ep {epoch + 1:02d}/{cfg.max_epochs}"
            train_loss, _ = _train_epoch(model, train_loader, loss, optimizer,
                                         scaler, device, amp_dtype, cfg,
                                         run_state, label)
            val = evaluate(model, val_loader, loss, device, amp_dtype)
            val_mse = val['aggregate_mse_weighted']
            run_state.epoch = epoch + 1
            history['train_loss'].append(train_loss)
            history['val_mse'].append(val_mse)
            history['val_per_target_mse'].append(val['per_target_mse'].tolist())

            improved = val_mse < run_state.best_val_mse
            if improved:
                run_state.best_val_mse = val_mse
                epochs_since_improve = 0
                save_checkpoint(best_path, model, optimizer, scheduler, scaler,
                                run_state, cfg, meta_extra)
            else:
                epochs_since_improve += 1
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler,
                            run_state, cfg, meta_extra)
            if scheduler is not None:
                scheduler.step()

            print(f"[{label}] train_loss={train_loss:.4f} val_mse={val_mse:.4f} "
                  f"per_target_mse={_fmt_vec(val['per_target_mse'])} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"t={time.time() - t0:.1f}s"
                  + (' *best*' if improved else ''))

            if 0 < cfg.early_stop_patience <= epochs_since_improve:
                print(f"[early-stop] no val improvement in "
                      f"{epochs_since_improve} epochs")
                break
    except KeyboardInterrupt:
        interrupted = os.path.join(cfg.checkpoint_dir, 'interrupted.pt')
        save_checkpoint(interrupted, model, optimizer, scheduler, scaler,
                        run_state, cfg, meta_extra)
        print(f"[interrupted] state saved to {interrupted}")
        raise

    # Final: test-set evaluation with the best-val weights.
    load_checkpoint(best_path, model)
    if test_dataset is None:
        from Data.dataset_v2 import PixelClusterDataset
        test_dataset = PixelClusterDataset(cfg.data_dir, 'test')
    test_loader = _make_loader(test_dataset, cfg, shuffle=False)
    preds, targets = _collect_predictions(model, test_loader, device, amp_dtype)
    table_row = compute_table_row(preds, targets, cfg.model_name, model)
    report = {
        'table_row': table_row,
        'test_aggregate_mse_weighted': aggregate_mse(
            preds, targets, weights=loss.w.detach().cpu()),
        'best_val_mse': run_state.best_val_mse,
        'epochs_trained': run_state.epoch,
        'run_config': asdict(cfg),
        'metadata': _build_metadata(cfg, device, meta_extra),
        'history': history,
    }
    # Atomic like the .pt files: Phase 5 aggregates final_report.json across
    # all six models, and one partially-written JSON breaks the aggregation.
    report_path = os.path.join(cfg.checkpoint_dir, 'final_report.json')
    tmp = report_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, report_path)
    print(f"[done] test mse={table_row['mse']:.6f} r2={table_row['r2']:.4f} "
          f"report={report_path}")

    try:  # quick-look artifacts only; a plotting failure must not fail the run
        from Models.common.plotting import plot_loss_curves, plot_parity
        if history['train_loss']:
            plot_loss_curves(history,
                             os.path.join(cfg.checkpoint_dir, 'loss_curves.png'))
        plot_parity(preds, targets,
                    os.path.join(cfg.checkpoint_dir, 'parity.png'),
                    space='normalized')
    except Exception as e:
        print(f"[warn] diagnostic plotting failed: {e}")

    return {
        'model_name': cfg.model_name,
        'best_val_mse': run_state.best_val_mse,
        'epochs_trained': run_state.epoch,
        'resumed_from_epoch': start_epoch if resume else None,
        'test_table_row': table_row,
        'checkpoint_dir': cfg.checkpoint_dir,
        'precision': precision,
        'device': device.type,
    }


def smoke_test(cfg, train_dataset=None, val_dataset=None):
    """First thing a RunPod session runs: one tiny epoch (fit with
    smoke=True) plus a checkpoint write/read round-trip to
    ``cfg.checkpoint_dir``. Leaves no files behind."""
    import dataclasses
    smoke_cfg = dataclasses.replace(cfg, smoke=True)
    fit(smoke_cfg, train_dataset=train_dataset, val_dataset=val_dataset)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    model = build_model(cfg.model_name, cfg.model_config)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    path = os.path.join(cfg.checkpoint_dir, '_smoke_roundtrip.pt')
    try:
        save_checkpoint(path, model, opt, None, None, RunState(), cfg,
                        {'precision': 'fp32', 'smoke': True})
        twin = build_model(cfg.model_name, cfg.model_config)
        ckpt = load_checkpoint(path, twin)
        for k, v in model.state_dict().items():
            if not torch.equal(v, twin.state_dict()[k]):
                raise RuntimeError(f"checkpoint round-trip mismatch in '{k}'")
        for key in ('git_commit', 'gpu', 'precision',
                    'target_stats_hash', 'norm_stats_hash'):
            if key not in ckpt['metadata']:
                raise RuntimeError(f"checkpoint metadata missing '{key}'")
    finally:
        if os.path.exists(path):
            os.remove(path)
    print(f"[SMOKE] checkpoint round-trip to {cfg.checkpoint_dir} OK — "
          f"pod is good for a full run")
