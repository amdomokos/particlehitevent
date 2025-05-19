import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "training_data"
max_files = 50  # change if you want more
cluster_lengths = []

files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.startswith("pixel_clusters") and f.endswith(".out")
])[:max_files]

for file_path in files:
    with open(file_path, "r") as f:
        lines = f.readlines()

    cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]
    print("Cluster indices: ", cluster_indices)
    for cluster_idx in cluster_indices:
        # Count number of <time slice> blocks until the next <cluster> or end
        length = 0
        for i in range(cluster_idx + 2, len(lines)):
            if lines[i].strip() == "<cluster>":
                break
            if lines[i].startswith("<time slice"):
                length += 1
        cluster_lengths.append(length)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(cluster_lengths, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Number of Time Slices per Cluster")
plt.ylabel("Frequency")
plt.title("Distribution of Time Slice Counts per Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/time_slice_distribution.png")
plt.show()

# Print basic stats
print(f"Total clusters analyzed: {len(cluster_lengths)}")
print(f"Min: {min(cluster_lengths)}, Max: {max(cluster_lengths)}")
print(f"Mean: {np.mean(cluster_lengths):.2f}, Median: {np.median(cluster_lengths)}")
print(f"90th percentile: {np.percentile(cluster_lengths, 90)}")
