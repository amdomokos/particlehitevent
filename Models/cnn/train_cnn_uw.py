"""CNN training with Kendall & Gal homoscedastic uncertainty weighting.

Instead of hand-tuning per-target loss weights, a learnable log-variance s_i per
target balances the targets automatically in a single run:

    L = sum_i [ exp(-s_i) * MSE_i + s_i ]

The constant z_entry target (index 2, var = 0) is excluded — its zero error would
drive s_2 -> -inf. We train/evaluate on the 5 physically meaningful targets.

Uses fewer epochs than the baseline for fast iteration. Baseline full run is
preserved in logs/cnn; this writes to logs/cnn_uw.

Run on the Arc iGPU:
    .venv-xpu/Scripts/python.exe -m Models.cnn.train_cnn_uw
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Data.dataset_v2 import PixelClusterDataset
from Models.cnn.cnn import CNN
from Models.cnn.train_cnn import get_device, plot_predictions, plot_loss_curve, _HAS_IPEX, ipex
import logging
import os
import sys
import numpy as np

# Exclude constant z_entry (index 2); train on the 5 meaningful targets.
ACTIVE = [0, 1, 3, 4, 5]
TARGET_NAMES = ['x_entry', 'y_entry', 'n_x', 'n_y', 'n_z']


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic (Kendall & Gal) multi-task loss with learned log-variances."""

    def __init__(self, n_targets):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_targets))

    def forward(self, preds, targets):
        mse_per = ((preds - targets) ** 2).mean(dim=0)        # (n_targets,)
        precision = torch.exp(-self.log_vars)
        loss = (precision * mse_per + self.log_vars).sum()
        return loss, mse_per.detach()

    def weights(self):
        # exp(-s_i): the effective per-target weight the model has learned.
        return torch.exp(-self.log_vars).detach().cpu().numpy()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    params = list(model.parameters()) + list(criterion.parameters())
    for X, y_module, Y in loader:
        X, Y = X.to(device), Y.to(device)
        Ya = Y[:, ACTIVE]
        optimizer.zero_grad()
        preds = model(X)[:, ACTIVE]
        loss, _ = criterion(preds, Ya)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y_module, Y in loader:
            X, Y = X.to(device), Y.to(device)
            Ya = Y[:, ACTIVE]
            preds = model(X)[:, ACTIVE]
            loss, _ = criterion(preds, Ya)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Ya.cpu().numpy())
    avg_loss = total_loss / len(loader)
    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    per_target_mse = np.mean((preds_arr - targets_arr) ** 2, axis=0)
    return avg_loss, preds_arr, targets_arr, per_target_mse


def main():
    device = get_device()

    BATCH_SIZE = 64
    NUM_EPOCHS = 8          # lower epochs for fast iteration
    LR = 1e-3
    DATA_DIR = 'preprocessed_data'
    LOG_DIR = 'logs/cnn_uw'
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'train_cnn_uw.log'), mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    if device.type == 'xpu':
        logging.info(f"Using device: {device} ({torch.xpu.get_device_name(0)})")
    else:
        logging.info(f"Using device: {device}")
    logging.info(f"Active targets: {TARGET_NAMES} (z_entry excluded)")

    logging.info("Loading datasets...")
    train_dataset = PixelClusterDataset(DATA_DIR, split='train')
    val_dataset = PixelClusterDataset(DATA_DIR, split='val')
    test_dataset = PixelClusterDataset(DATA_DIR, split='test')
    logging.info(f"Sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN().to(device)
    criterion = UncertaintyWeightedLoss(len(ACTIVE)).to(device)
    # Learned log-variances are parameters too — include them in the optimizer.
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=LR)
    if _HAS_IPEX and device.type in ('xpu', 'cpu'):
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        logging.info("Applied ipex.optimize.")
    logging.info("Model, uncertainty-weighted loss, and optimizer initialized.")

    train_losses, val_losses = [], []
    best_val_mse = float('inf')
    checkpoint_path = os.path.join(LOG_DIR, 'best_model.pt')

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets, val_per_target_mse = evaluate(model, val_loader, criterion, device)
        val_mse_mean = float(val_per_target_mse.mean())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        w = criterion.weights()
        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} — Train UW: {train_loss:.4f}, Val UW: {val_loss:.4f}, Val mean MSE (5 targets): {val_mse_mean:.5f}")
        logging.info(f"  Per-target Val MSE: {dict(zip(TARGET_NAMES, np.round(val_per_target_mse, 5).tolist()))}")
        logging.info(f"  Learned weights exp(-s): {dict(zip(TARGET_NAMES, np.round(w, 3).tolist()))}")

        # Checkpoint by the actual metric we care about (mean per-target MSE),
        # not the UW loss (which includes the s_i terms and can go negative).
        if val_mse_mean < best_val_mse:
            best_val_mse = val_mse_mean
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'log_vars': criterion.log_vars.detach().cpu()}, checkpoint_path)
            logging.info(f"  New best val mean MSE: {best_val_mse:.5f} (checkpoint saved)")

    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(LOG_DIR, 'loss_curve.png'))

    logging.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_preds, test_targets, test_per_target_mse = evaluate(model, test_loader, criterion, device)
    logging.info(f"Test mean MSE (5 targets): {float(test_per_target_mse.mean()):.5f}")
    logging.info(f"Per-target Test MSE: {dict(zip(TARGET_NAMES, np.round(test_per_target_mse, 5).tolist()))}")

    plot_predictions(test_preds, test_targets, save_path=os.path.join(LOG_DIR, 'test_pred_vs_target.png'))
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
