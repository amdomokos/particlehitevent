import os, json, torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATA_DIR = "training_data"
SAVE_DIR = "preprocessed_data"
SEQUENCE_LENGTH = 80
INPUT_SHAPE = (13, 21)
RAW_TARGET_DIM = 9
TARGET_DIM = 6
CHUNK_SIZE = 20000
LOG_FILE = os.path.join(SAVE_DIR, "preprocess_debug.log")

os.makedirs(SAVE_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def parse_file(file_path):
    file_inputs, file_targets = [], []
    error_count = 0
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]

        for cluster_idx in cluster_indices:
            try:
                target_line = lines[cluster_idx + 1].strip()
                target = np.array(list(map(float, target_line.split())), dtype=np.float32)
                if len(target) != RAW_TARGET_DIM:
                    raise ValueError(f"Target length {len(target)} != {TARGET_DIM}")

                end_idx = next((i for i in range(cluster_idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))
                frames = []
                i = cluster_idx + 2
                while i < end_idx:
                    if lines[i].startswith("<time slice"):
                        try:
                            frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
                            frames.append(np.array(frame))
                            i += 14
                        except (IndexError, ValueError):
                            break
                    else:
                        i += 1

                if len(frames) != SEQUENCE_LENGTH:
                    raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {len(frames)}")

                input_tensor = np.array(frames, dtype=np.float32)
                file_inputs.append(input_tensor)
                file_targets.append(target)

            except Exception as e:
                error_count += 1
                log(f"[WARN] Skipped cluster in {os.path.basename(file_path)}: {e}")

        return file_inputs, file_targets, os.path.basename(file_path), error_count

    except Exception as e:
        log(f"[ERROR] Failed to parse file {file_path}: {e}")
        return [], [], os.path.basename(file_path), -1
    
def compute_stats(results):
    input_sum    = np.zeros(273, dtype=np.float64)
    input_sum_sq = np.zeros(273, dtype=np.float64)
    target_sum    = np.zeros(3, dtype=np.float64)
    target_sum_sq = np.zeros(3, dtype=np.float64)
    n_inputs  = 0
    n_targets = 0

    for file_inputs, file_targets, _, error_count in results:
        if error_count == -1:
            continue
        for arr in file_inputs:
            flat = arr.reshape(80, 273).astype(np.float64)
            input_sum    += flat.sum(axis=0)
            input_sum_sq += (flat ** 2).sum(axis=0)
            n_inputs += 80
        for target in file_targets:
            t = target[:3].astype(np.float64)
            target_sum    += t
            target_sum_sq += t ** 2
            n_targets += 1

    input_mean = input_sum / n_inputs
    input_std  = np.sqrt(np.maximum(input_sum_sq / n_inputs - input_mean ** 2, 0.0))
    input_std  = np.maximum(input_std, 1e-8)

    target_mean = target_sum / n_targets
    target_std  = np.sqrt(np.maximum(target_sum_sq / n_targets - target_mean ** 2, 0.0))
    target_std  = np.maximum(target_std, 1e-8)

    stats = {
        "input_mean":   input_mean.tolist(),
        "input_std":    input_std.tolist(),
        "pos_mean":  target_mean.tolist(),
        "pos_std":   target_std.tolist(),
    }
    with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f)

    return (
        input_mean.astype(np.float32),
        input_std.astype(np.float32),
        target_mean.astype(np.float32),
        target_std.astype(np.float32),
    )
    
def _save_chunk(input_buf, target_buf, ymodule_buf, idx):
    X = torch.tensor(np.array(input_buf), dtype=torch.float32)
    Y = torch.tensor(np.array(target_buf), dtype=torch.float32)
    y_module = torch.tensor(np.array(ymodule_buf), dtype=torch.float32)
    path = os.path.join(SAVE_DIR, f"chunk_{idx:04d}.pt")
    torch.save({    "X": X, "y_module": y_module, "Y": Y}, path)
    log(f"[INFO] Saved chunk {idx} ({len(input_buf)} samples) -> {path}")

def normalize_and_save_chunks(results, input_mean, input_std, target_mean, target_std):
    dead_pixel_mask = (input_std <= 1e-8)  # [273] boolean mask
    input_buf, target_buf, ymodule_buf = [], [], []
    chunk_idx = 0

    for file_inputs, file_targets, _, error_count in results:
        if error_count == -1:
            continue
        for x, y in zip(file_inputs, file_targets):
            x_flat = x.reshape(80, 273)
            raw_zero_mask = (x_flat == 0.0)          
            x_norm = (x_flat - input_mean) / input_std
            x_norm[:, dead_pixel_mask] = 0.0        
            x_norm[raw_zero_mask] = 0.0              
            x_norm = x_norm.reshape(80, 13, 21)

            y_pos = y[:3].astype(np.float32)
            y_pos_norm = (y_pos - target_mean) / target_std

            y_dir = y[3:6].astype(np.float32)
            norm = np.linalg.norm(y_dir)
            y_dir_unit = y_dir / (norm + 1e-8)
            y_norm = np.concatenate([y_pos_norm, y_dir_unit])
            
            y_module = np.float32(y[7])

            input_buf.append(x_norm)
            target_buf.append(y_norm)
            ymodule_buf.append(y_module)

            if len(input_buf) >= CHUNK_SIZE:
                _save_chunk(input_buf, target_buf, ymodule_buf, chunk_idx)
                input_buf, target_buf, ymodule_buf = [], [], []
                chunk_idx += 1

    if input_buf:
        _save_chunk(input_buf, target_buf, ymodule_buf, chunk_idx)
        chunk_idx += 1

    log(f"[INFO] Saved {chunk_idx} chunks to {SAVE_DIR}")

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith("pixel_clusters") and f.endswith(".out")
    ])

    log(f"[INFO] Found {len(files)} files. Parsing with {cpu_count()} workers...")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_file, files), total=len(files)))

    total_clusters = sum(len(r[0]) for r in results)
    total_errors = sum(r[3] for r in results if r[3] > 0)
    corrupted_files = [r[2] for r in results if r[3] == -1]

    log(f"[INFO] Total clusters: {total_clusters}")
    log(f"[INFO] Skipped: {total_errors}")
    log(f"[INFO] Corrupted files: {corrupted_files}")

    if total_clusters == 0:
        log("[FATAL] No valid data. Aborting.")
        return

    input_mean, input_std, target_mean, target_std = compute_stats(results)
    normalize_and_save_chunks(results, input_mean, input_std, target_mean, target_std)
    log(f"[SUCCESS] Preprocessing complete. Chunks saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()
        
    
