# # import os
# # import argparse
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from dataset import PixelClusterDataset
# # import torch
# # import sys
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# # def plot_cluster_distribution(dataset: PixelClusterDataset):
# #     """Plot number of clusters (events) per file."""
# #     plt.figure(figsize=(12, 6))
# #     plt.bar(range(len(dataset.file_cluster_counts)), dataset.file_cluster_counts)
# #     plt.xlabel('File Index')
# #     plt.ylabel('Number of Clusters')
# #     plt.title('Cluster Distribution Across Files')
# #     plt.tight_layout()
# #     plt.show()


# # def plot_target_distribution(dataset: PixelClusterDataset, file_index: int = 0):
# #     """Plot unnormalized target values for a specific file (first 100 clusters in that file)."""
# #     start_idx = file_index * 100
# #     end_idx = min(start_idx + 100, len(dataset.all_targets))
# #     file_targets = dataset.all_targets[start_idx:end_idx]

# #     if file_targets.shape[0] == 0:
# #         print(f"No targets found for file index {file_index}")
# #         return

# #     plt.figure(figsize=(12, 6))
# #     for i in range(file_targets.shape[1]):
# #         plt.plot(file_targets[:, i], label=f'Target {i+1}')

# #     plt.xlabel('Cluster Index')
# #     plt.ylabel('Target Value (before normalization)')
# #     plt.title(f'Target Values for File {dataset.files[file_index]}')
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()


# # def plot_all_target_distributions(dataset: PixelClusterDataset):
# #     """Plot each of the 9 target variable distributions across all events (before normalization)."""
# #     if dataset.all_targets.shape[1] != 9:
# #         print("Expected 9 target variables for this plot.")
# #         return

# #     plt.figure(figsize=(18, 10))
# #     for i in range(9):
# #         plt.subplot(3, 3, i + 1)
# #         plt.plot(dataset.all_targets[:, i])
# #         plt.title(f"Target {i + 1}")
# #         plt.xlabel("Event Index")
# #         plt.ylabel("Value")
# #     plt.tight_layout()
# #     plt.suptitle("Target Value Distributions Across All Events", y=1.02)
# #     plt.show()


# # def main(data_dir: str):
# #     print("Loading dataset...")
# #     dataset = PixelClusterDataset(data_dir=data_dir)

# #     print("Plotting cluster distribution...")
# #     plot_cluster_distribution(dataset)

# #     print("Plotting target distributions...")
# #     plot_target_distribution(dataset, file_index=0)

# #     print("Plotting all target distributions...")
# #     plot_all_target_distributions(dataset)


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Visualize pixel cluster dataset")
# #     parser.add_argument("--data_dir", type=str, required=True, help="Path to directory with .out files")
# #     args = parser.parse_args()

# #     main(args.data_dir)


# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from Data.dataset import PixelClusterDataset
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# def _maybe_save(fig, filename, save_dir=None):
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         filepath = os.path.join(save_dir, filename)
#         fig.savefig(filepath, bbox_inches='tight')
#         print(f"[DEBUG] Plot saved to {filepath}")
#     else:
#         plt.show()

# def plot_sample_distribution(dataset, max_files=80, save_dir=None):
#     print("[DEBUG] Counting clusters per file...")
#     counts = {}
#     for file_path, _, _ in dataset.cluster_index:
#         fname = os.path.basename(file_path)
#         counts[fname] = counts.get(fname, 0) + 1

#     files = list(counts.keys())[:max_files]
#     values = [counts[f] for f in files]

#     fig = plt.figure(figsize=(12, 6))
#     plt.bar(range(len(files)), values)
#     plt.xticks(range(len(files)), files, rotation=90)
#     plt.xlabel('File Name')
#     plt.ylabel('Number of Clusters')
#     plt.title('Cluster Count per File')
#     plt.tight_layout()
#     _maybe_save(fig, "cluster_distribution.png", save_dir)


# def plot_target_distribution(dataset, file_index=0, save_dir=None):
#     print(f"[DEBUG] Plotting targets for file index {file_index}...")

#     file_clusters = [i for i, (fp, _, _) in enumerate(dataset.cluster_index)
#                      if dataset.files[file_index] in fp]

#     if not file_clusters:
#         print("[WARN] No clusters found for the given file index.")
#         return

#     targets = []
#     for i in file_clusters[:100]:
#         _, t = dataset[i]
#         targets.append(t.numpy())

#     if not targets:
#         print("[WARN] No valid targets extracted.")
#         return

#     targets = np.stack(targets)
#     fig = plt.figure(figsize=(12, 6))
#     for i in range(targets.shape[1]):
#         plt.plot(targets[:, i], label=f'Target {i+1}')
#     plt.xlabel("Cluster Index")
#     plt.ylabel("Target Value")
#     plt.title("Target Distribution for One File")
#     plt.legend()
#     plt.tight_layout()
#     _maybe_save(fig, f"target_distribution_file_{file_index}.png", save_dir)

# # def plot_all_target_distributions_across_files(dataset, num_files=10, save_dir=None):
# #     """
# #     Plot all 9 target variables in one 3x3 subplot grid across clusters from the first `num_files` files.
# #     Saves a single PNG file if save_dir is provided.
# #     """
# #     print(f"[DEBUG] Plotting combined 9-target distribution from first {num_files} files...")

# #     # Step 1: Select clusters from first N files
# #     selected_clusters = [
# #         i for i, (fp, _, _) in enumerate(dataset.cluster_index)
# #         if os.path.basename(fp) in dataset.files[:num_files]
# #     ]

# #     all_targets = []
# #     for i in selected_clusters:
# #         try:
# #             _, t = dataset[i]
# #             all_targets.append(t.numpy())
# #         except Exception as e:
# #             print(f"[WARN] Skipping cluster {i}: {e}")

# #     if not all_targets:
# #         print("[WARN] No targets found to plot.")
# #         return

# #     all_targets = np.stack(all_targets)
# #     num_targets = all_targets.shape[1]

# #     # Step 2: Create 3x3 subplot
# #     fig, axes = plt.subplots(3, 3, figsize=(15, 10))
# #     fig.suptitle(f"Target Variable Distributions (First {num_files} Files)", fontsize=16)

# #     for i in range(num_targets):
# #         ax = axes[i // 3, i % 3]
# #         ax.plot(all_targets[:, i], label=f'Target {i+1}')
# #         ax.set_title(f'Target {i+1}')
# #         ax.set_xlabel('Cluster Index')
# #         ax.set_ylabel('Value')
# #         ax.grid(True)

# #     plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

# #     # Save or show
# #     if save_dir:
# #         os.makedirs(save_dir, exist_ok=True)
# #         save_path = os.path.join(save_dir, "all_targets_distribution.png")
# #         fig.savefig(save_path, bbox_inches='tight')
# #         print(f"[DEBUG] Saved: {save_path}")
# #     else:
# #         plt.show()

# def plot_all_target_distributions_across_files(dataset, num_files=10, save_dir=None):
#     """
#     Plot each of the 9 target variables across all clusters from the first `num_files` files.
#     Saves each target plot as a separate PNG if save_dir is provided.
#     """
#     print(f"[DEBUG] Plotting target variable distributions for first {num_files} files...")

#     # Step 1: Collect all target tensors from the first N files
#     selected_clusters = [
#         i for i, (fp, _, _) in enumerate(dataset.cluster_index)
#         if os.path.basename(fp) in dataset.files[:num_files]
#     ]

#     all_targets = []
#     for i in selected_clusters:
#         try:
#             _, t = dataset[i]
#             all_targets.append(t.numpy())
#         except Exception as e:
#             print(f"[WARN] Skipping cluster {i}: {e}")

#     if not all_targets:
#         print("[WARN] No targets found to plot.")
#         return

#     all_targets = np.stack(all_targets)
#     num_targets = all_targets.shape[1]

#     print(f"[DEBUG] Total clusters used: {len(all_targets)}")
#     print(f"[DEBUG] Target shape per cluster: {all_targets.shape[1]}")

#     for i in range(num_targets):
#         fig = plt.figure(figsize=(10, 4))
#         plt.plot(all_targets[:, i], label=f'Target {i+1}')
#         plt.title(f"Target {i+1} Across First {num_files} Files")
#         plt.xlabel("Cluster Index")
#         plt.ylabel("Target Value")
#         plt.grid(True)
#         plt.tight_layout()

#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             filename = os.path.join(save_dir, f"target_{i+1}_distribution.png")
#             fig.savefig(filename, bbox_inches='tight')
#             print(f"[DEBUG] Saved: {filename}")
#         else:
#             plt.show()


# visualize_dataset_stats.py
import os
import torch
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def count_events_in_raw(data_dir):
    count = 0
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("pixel_clusters") and fname.endswith(".out"):
            with open(os.path.join(data_dir, fname), 'r') as f:
                lines = f.readlines()
                count += sum(1 for line in lines if line.strip() == "<cluster>")
    return count

def count_events_in_chunks(preprocessed_dir):
    total = 0
    for fname in sorted(os.listdir(preprocessed_dir)):
        if fname.endswith(".pt") and fname.startswith("chunk_"):
            data = torch.load(os.path.join(preprocessed_dir, fname), map_location='cpu')
            total += data['inputs'].shape[0]
    return total

def parse_targets_from_file(filepath, target_dim=9):
    with open(filepath, "r") as f:
        lines = f.readlines()

    targets = []
    cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]
    for idx in cluster_indices:
        try:
            target_line = lines[idx + 1].strip()
            target = np.array(list(map(float, target_line.split())), dtype=np.float32)
            if target.shape[0] == target_dim:
                targets.append(target)
        except Exception:
            continue
    return np.array(targets)

def compute_means_per_file(data_dir, target_dim=9):
    all_means = []
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".out") and f.startswith("pixel_clusters")
    ])

    for file in tqdm(files, desc="Computing means per file"):
        targets = parse_targets_from_file(file, target_dim=target_dim)
        if targets.size == 0:
            continue
        mean = targets.mean(axis=0)
        all_means.append(mean)

    return np.array(all_means), files

def plot_target_histograms(target_means, save_path="logs/target_mean_histograms.png", csv_path="logs/target_mean_summary.csv"):
    num_targets = target_means.shape[1]
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(num_targets):
        axs[i].plot(range(1, len(target_means) + 1), target_means[:, i], marker='o', linestyle='-')
        axs[i].set_title(f"Target {i} Mean per File")
        axs[i].set_xlabel("File Index (1-based)")
        axs[i].set_ylabel("Mean Value")
        axs[i].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Saved per-file target mean plot to {save_path}")

    # Optional: log table-style summary of mean ranges
    for i in range(num_targets):
        mean_val = target_means[:, i].mean()
        min_val = target_means[:, i].min()
        max_val = target_means[:, i].max()
        logging.info(f"Target {i}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
        # Save CSV summary of target mean stats
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Target Index", "Mean of Means", "Min", "Max"])
        for i in range(num_targets):
            mean_val = target_means[:, i].mean()
            min_val = target_means[:, i].min()
            max_val = target_means[:, i].max()
            writer.writerow([i, f"{mean_val:.4f}", f"{min_val:.4f}", f"{max_val:.4f}"])
    logging.info(f"Saved target summary CSV to {csv_path}")
    logging.info(f"Saved histogram plot to {save_path}")

def compute_means_from_chunks(preprocessed_dir, target_dim=9):
    means = []
    chunk_files = sorted([
        os.path.join(preprocessed_dir, f)
        for f in os.listdir(preprocessed_dir)
        if f.startswith("chunk_") and f.endswith(".pt")
    ])

    for path in tqdm(chunk_files, desc="Computing means from normalized chunks"):
        data = torch.load(path, map_location='cpu')
        targets = data['targets'].numpy()
        mean = targets.mean(axis=0)
        means.append(mean)

    return np.array(means), chunk_files

def main():
    raw_data_dir = "training_data"
    processed_data_dir = "preprocessed_data"

    logging.info("Counting raw events in .out files...")
    raw_event_count = count_events_in_raw(raw_data_dir)
    logging.info(f"Total raw events found: {raw_event_count}")

    logging.info("Counting normalized events in .pt chunks...")
    normalized_event_count = count_events_in_chunks(processed_data_dir)
    logging.info(f"Total normalized events found: {normalized_event_count}")

    lost = raw_event_count - normalized_event_count
    if lost == 0:
        logging.info("✅ No data lost during preprocessing.")
    else:
        logging.warning(f"⚠️ Lost {lost} events during preprocessing.")

    logging.info("Generating per-target mean histograms across files...")
    target_means, files = compute_means_per_file(raw_data_dir)
    logging.info(f"Processed {len(files)} files, target mean shape: {target_means.shape}")
    plot_target_histograms(target_means)

    logging.info("Generating per-target mean histograms from normalized data...")
    norm_means, chunk_files = compute_means_from_chunks(processed_data_dir)
    logging.info(f"Processed {len(chunk_files)} chunks, normalized target mean shape: {norm_means.shape}")
    plot_target_histograms(norm_means, save_path="logs/norm_target_mean_plot.png", csv_path="logs/norm_target_mean_summary.csv")

if __name__ == "__main__":
    main()
