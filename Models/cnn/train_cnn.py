import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Data.dataset_v2 import PixelClusterDataset
from Models.cnn.cnn import CNN
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np

# Intel Extension for PyTorch enables the Arc 140V iGPU (XPU) and applies
# graph/operator optimizations. Optional: absent on CPU-only installs, in which
# case training falls back to CPU transparently.
try:
    import intel_extension_for_pytorch as ipex
    _HAS_IPEX = True
except ImportError:
    ipex = None
    _HAS_IPEX = False


def get_device():
    """Prefer the Intel Arc iGPU (XPU), then CUDA, then CPU."""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y_module, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y_module, Y in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)
            loss = criterion(preds, Y)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    per_target_mse = np.mean((preds_arr - targets_arr) ** 2, axis=0)
    return avg_loss, preds_arr, targets_arr, per_target_mse


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def plot_predictions(preds, targets, save_path):
    n_targets = preds.shape[1]
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()
    for i in range(n_targets):
        t = targets[:200, i]
        p = preds[:200, i]
        axs[i].scatter(t, p, alpha=0.4, s=10)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        axs[i].plot(lims, lims, 'r--', linewidth=1)
        axs[i].set_xlabel('True')
        axs[i].set_ylabel('Pred')
        axs[i].set_title(f'Target {i}')
        axs[i].grid(True)
    for i in range(n_targets, len(axs)):
        axs[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved prediction plot to {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('CNN Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved loss curve to {save_path}")


def main():
    # 1. setup device
    device = get_device()

    # 2. hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LR = 1e-3
    DATA_DIR = 'preprocessed_data'
    LOG_DIR = 'logs/cnn'
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'train_cnn.log'), mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    if device.type == 'xpu':
        logging.info(f"Using device: {device} ({torch.xpu.get_device_name(0)})")
    else:
        logging.info(f"Using device: {device}")
    logging.info(f"IPEX available: {_HAS_IPEX}")

    # 3. load datasets and dataloaders
    logging.info("Loading datasets...")
    train_dataset = PixelClusterDataset(DATA_DIR, split='train')
    val_dataset = PixelClusterDataset(DATA_DIR, split='val')
    test_dataset = PixelClusterDataset(DATA_DIR, split='test')
    logging.info(f"Dataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. instantiate model, criterion, optimizer
    model = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Apply IPEX graph/operator optimizations on Intel hardware (XPU iGPU or CPU).
    if _HAS_IPEX and device.type in ('xpu', 'cpu'):
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        logging.info("Applied ipex.optimize to model and optimizer.")

    logging.info("Model, criterion, and optimizer initialized.")

    # 5. training loop over epochs
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(LOG_DIR, 'best_model.pt')

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets, val_per_target_mse = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} — Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        logging.info(f"  Per-target Val MSE: {np.round(val_per_target_mse, 4).tolist()}")

        # save checkpoint if best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            logging.info(f"  New best val loss: {best_val_loss:.4f}")

        if epoch % 5 == 0:
            plot_predictions(val_preds, val_targets, save_path=os.path.join(LOG_DIR, f'val_pred_epoch{epoch}.png'))

    # 6. save plots
    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(LOG_DIR, 'loss_curve.png'))

    # 7. final evaluation on test set using best checkpoint
    logging.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_preds, test_targets, test_per_target_mse = evaluate(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Per-target Test MSE: {np.round(test_per_target_mse, 4).tolist()}")

    plot_predictions(test_preds, test_targets, save_path=os.path.join(LOG_DIR, 'test_pred_vs_target.png'))
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
