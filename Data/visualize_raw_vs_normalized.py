import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

RAW_DATA_DIR = "training_data"
STATS_FILE = "preprocessed_data/norm_stats.json"
SAVE_DIR = "logs/dynamic_pixel_comparison"
HEIGHT, WIDTH = 13, 21
SEQUENCE_LENGTH = 80
TOP_K = 3
NUM_EVENTS = 3

os.makedirs(SAVE_DIR, exist_ok=True)

with open(STATS_FILE, "r") as f:
    stats = json.load(f)
input_min = stats["input_min"]
input_max = stats["input_max"]

def parse_raw_event(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]
    events = []
    for idx in cluster_indices:
        try:
            frames = []
            end_idx = next((i for i in range(idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))
            i = idx + 2
            while i < end_idx:
                if lines[i].startswith("<time slice"):
                    frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(HEIGHT)]
                    frames.append(np.array(frame).flatten())
                    i += HEIGHT + 1
                else:
                    i += 1
            if len(frames) == SEQUENCE_LENGTH:
                events.append(np.array(frames))
        except Exception:
            continue
    return events

def plot_comparison(raw_event, event_id):
    raw_x = raw_event.T
    norm_x = ((raw_event - input_min) / (input_max - input_min + 1e-8)).T
    variation = raw_x.max(axis=1) - raw_x.min(axis=1)
    top_indices = variation.argsort()[-TOP_K:][::-1]

    fig, axs = plt.subplots(TOP_K, 2, figsize=(12, 3 * TOP_K))
    for i, idx in enumerate(top_indices):
        axs[i, 0].plot(raw_x[idx])
        axs[i, 0].set_title(f"Raw Pixel {idx}")
        axs[i, 1].plot(norm_x[idx])
        axs[i, 1].set_title(f"Normalized Pixel {idx}")
        for j in range(2): axs[i, j].grid(True)
    for ax in axs[-1, :]:
        ax.set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"raw_vs_normalized_event_{event_id}.png"))
    plt.close()

files = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR) if f.endswith(".out")]
random.shuffle(files)

count = 0
for file in files:
    events = parse_raw_event(file)
    for ev in events:
        plot_comparison(ev, count)
        count += 1
        if count >= NUM_EVENTS:
            break
    if count >= NUM_EVENTS:
        break

print(f"Saved raw/normalized plots in: {SAVE_DIR}")
