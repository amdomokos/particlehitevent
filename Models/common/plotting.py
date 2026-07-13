"""Quick-look training diagnostics (Phase 3). Not the paper's figures.

Written by the engine into ``cfg.checkpoint_dir`` after training so a run's
health is inspectable without loading the checkpoint. Agg backend — RunPod
pods have no display server.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Data.config import ACTIVE_TARGET_DIM, ACTIVE_TARGET_NAMES


def _to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot_loss_curves(history, out_path):
    """2-panel PNG: (top) train_loss + val_mse per epoch; (bottom) per-target
    val MSE per epoch, one line per active target.

    ``history`` keys: 'train_loss', 'val_mse' (lists of floats),
    'val_per_target_mse' (list of length-5 lists).
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)
    per_target = np.asarray(history['val_per_target_mse'])  # (E, 5)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_top.plot(epochs, history['train_loss'], label='train loss', marker='o')
    ax_top.plot(epochs, history['val_mse'], label='val MSE (weighted)', marker='s')
    ax_top.set_ylabel('loss / MSE')
    ax_top.set_yscale('log')
    ax_top.legend()
    ax_top.set_title('Training curves')

    for j, name in enumerate(ACTIVE_TARGET_NAMES):
        ax_bot.plot(epochs, per_target[:, j], label=name, marker='.')
    ax_bot.set_xlabel('epoch')
    ax_bot.set_ylabel('val MSE per target')
    ax_bot.set_yscale('log')
    ax_bot.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_parity(preds, targets, out_path, space='normalized'):
    """5-panel predicted-vs-true scatter, one panel per active target.

    Axes are in whatever space the caller passes (no denormalization here);
    ``space`` is recorded in the figure title.
    """
    preds, targets = _to_np(preds), _to_np(targets)
    assert preds.shape == targets.shape and preds.shape[1] == ACTIVE_TARGET_DIM

    fig, axes = plt.subplots(1, ACTIVE_TARGET_DIM, figsize=(4 * ACTIVE_TARGET_DIM, 4))
    for j, (ax, name) in enumerate(zip(axes, ACTIVE_TARGET_NAMES)):
        t, p = targets[:, j], preds[:, j]
        ax.scatter(t, p, s=2, alpha=0.3, rasterized=True)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_title(name)
        ax.set_xlabel('true')
        if j == 0:
            ax.set_ylabel('predicted')
    fig.suptitle(f'Parity plots ({space} space, n={preds.shape[0]})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
