# import argparse
# from Data.dataset import PixelClusterDataset
# from Data.visualize_dataset import plot_sample_distribution, plot_target_distribution, plot_all_target_distributions_across_files
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# def main():
#     parser = argparse.ArgumentParser(description="Lazy-loaded pixel dataset visualizer")
#     parser.add_argument("--data_dir", type=str, default="training_data")
#     parser.add_argument("--visualize", action="store_true")
#     parser.add_argument("--file_index", type=int, default=0)
#     parser.add_argument("--save_dir", type=str, help="Directory to save plots (optional)")
#     parser.add_argument("--plot_all_targets", action="store_true", help="Plot all 9 target variables")



#     args = parser.parse_args()

#     print("[DEBUG] Loading dataset...")
#     dataset = PixelClusterDataset(args.data_dir)

#     # if args.visualize:
#     #     print("[DEBUG] Visualizing cluster distribution...")
#     #     plot_sample_distribution(dataset)
#     #     print("[DEBUG] Visualizing target values for file...")
#     #     plot_target_distribution(dataset, file_index=args.file_index)

#     if args.visualize:
#         print("[DEBUG] Visualizing cluster distribution...")
#         plot_sample_distribution(dataset, save_dir=args.save_dir)
#         print("[DEBUG] Visualizing target values for file...")
#         plot_target_distribution(dataset, file_index=args.file_index, save_dir=args.save_dir)

#     if args.plot_all_targets:
#         print("[DEBUG] Plotting all target variables...")
#         plot_all_target_distributions_across_files(dataset, num_files=10, save_dir=args.save_dir)

# if __name__ == "__main__":
#     main()


import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.dataset import PixelClusterDataset
from Data.download import download_and_extract_data
from Models.s4network import S4PredictionNetwork
from Models.training import train_model, test_model, save_predictions

def create_url_file(start_num, end_num, base_url, output_file="file_links.txt"):
    """Create a file with download URLs for the dataset."""
    with open(output_file, "w") as f:
        for i in range(start_num, end_num + 1):
            url = f"{base_url}{i}.out.gz"
            f.write(url + "\n")
    print(f"{output_file} has been created with all URLs.")

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, output_dir="results"):
    """Plot training and testing metrics over epochs."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.create_url_file:
        create_url_file(args.start_num, args.end_num, args.base_url, args.url_file)

    if args.download_data:
        print("Downloading and extracting data...")
        download_and_extract_data(
            start_num=args.start_num,
            end_num=args.end_num,
            base_url=args.base_url,
            output_dir=args.data_dir
        )

    print("Loading dataset...")
    dataset = PixelClusterDataset(args.data_dir, sequence_length=args.sequence_length, device=device)

    print(f"Total events found: {len(dataset)}")
    print("Clusters per file:")
    file_event_counts = {}
    for file_path, _ in dataset.index_map:
        fname = os.path.basename(file_path)
        file_event_counts[fname] = file_event_counts.get(fname, 0) + 1
    for fname, count in file_event_counts.items():
        print(f"  {fname}: {count} clusters")

    train_size = int(args.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Inspecting one training batch...")
    for inputs, targets in train_loader:
        print(f"Training batch - Inputs shape: {inputs.shape}")
        print(f"Training batch - Targets shape: {targets.shape}")
        break

    print("Initializing model...")
    model = S4PredictionNetwork(args.hidden_dim, args.output_dim, height=args.height, width=args.width)
    model.to(device)

    print("Starting training...")
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, 
        train_loader, 
        test_loader,
        num_epochs=args.epochs, 
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        device=device
    )

    print("Evaluating final test loss and accuracy...")
    final_test_loss, final_test_accuracy = test_model(model, test_loader, device, return_accuracy=True)
    print(f"Final Test Loss (MSE): {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

    print("Saving predictions...")
    save_predictions(model, test_loader, os.path.join(args.output_dir, "predictions.csv"), device)

    print("Plotting training curves...")
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, args.output_dir)

    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Model saved to {os.path.join(args.output_dir, 'model.pt')}")

    print("Saving state-space matrices...")
    model.s4_layer.save_matrices()

    try:
        from utils.visualize_matrices import plot_all_matrices
        plot_all_matrices()
    except ImportError:
        print("Visualization module not found. Skipping matrix plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Cluster Prediction with S4 Network")

    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--create_url_file", action="store_true", help="Create URL file for data download")
    parser.add_argument("--url_file", type=str, default="file_links.txt", help="File to store URLs")
    parser.add_argument("--download_data", action="store_true", help="Download and extract data files")
    parser.add_argument("--start_num", type=int, default=16401, help="Start number for data download")
    parser.add_argument("--end_num", type=int, default=16480, help="End number for data download")
    parser.add_argument("--base_url", type=str, default="https://cernbox.cern.ch/remote.php/dav/public-files/wJhzxMJq6gmbhE7/pixel_clusters_d", help="Base URL for data download")

    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length for input data")
    parser.add_argument("--hidden_dim", type=int, default=500, help="Hidden dimension for model")
    parser.add_argument("--output_dim", type=int, default=9, help="Output dimension for model")
    parser.add_argument("--height", type=int, default=13, help="Height of input image")
    parser.add_argument("--width", type=int, default=21, help="Width of input image")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data to use for training")

    args = parser.parse_args()
    main(args)
