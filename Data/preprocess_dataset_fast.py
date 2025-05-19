# import os, json, torch
# import numpy as np
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# DATA_DIR = "training_data"
# SAVE_DIR = "preprocessed_data"
# SEQUENCE_LENGTH = 80
# INPUT_DIM = 273
# TARGET_DIM = 9
# LOG_FILE = os.path.join(SAVE_DIR, "preprocess_debug.log")

# os.makedirs(SAVE_DIR, exist_ok=True)

# def log(msg):
#     with open(LOG_FILE, "a") as f:
#         f.write(msg + "\n")
#     print(msg)

# def parse_file(file_path):
#     file_inputs, file_targets = [], []
#     error_count = 0
#     try:
#         with open(file_path, "r") as f:
#             lines = f.readlines()

#         cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]

#         for cluster_idx in cluster_indices:
#             try:
#                 target_line = lines[cluster_idx + 1].strip()
#                 target = np.array(list(map(float, target_line.split())), dtype=np.float32)
#                 if len(target) != TARGET_DIM:
#                     raise ValueError(f"Target length {len(target)} != {TARGET_DIM}")

#                 frames = []
#                 # for i, line in enumerate(lines):
#                 #     if line.startswith("<time slice"):
#                 #         frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                 #         frames.append(np.array(frame).flatten())

#                 # Get end index of the current cluster (start of next cluster or end of file)
#                 end_idx = next((i for i in range(cluster_idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))

#                 frames = []
#                 i = cluster_idx + 2  # start just after the target line
#                 while i < end_idx:
#                     if lines[i].startswith("<time slice"):
#                         try:
#                             frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                             frames.append(np.array(frame).flatten())
#                             i += 14  # skip current slice + 13 rows
#                         except (IndexError, ValueError):
#                             break
#                     else:
#                         i += 1


#                 if len(frames) != SEQUENCE_LENGTH:
#                     raise ValueError(f"Expected 80 frames, got {len(frames)}")

#                 input_tensor = np.array(frames, dtype=np.float32)
#                 file_inputs.append(input_tensor)
#                 file_targets.append(target)

#             except Exception as e:
#                 error_count += 1
#                 log(f"[WARN] Skipped cluster in {os.path.basename(file_path)}: {e}")

#         return file_inputs, file_targets, os.path.basename(file_path), error_count

#     except Exception as e:
#         log(f"[ERROR] Failed to parse file {file_path}: {e}")
#         return [], [], os.path.basename(file_path), -1

# def main():
#     if os.path.exists(LOG_FILE):
#         os.remove(LOG_FILE)

#     files = sorted([
#         os.path.join(DATA_DIR, f)
#         for f in os.listdir(DATA_DIR)
#         if f.startswith("pixel_clusters") and f.endswith(".out")
#     ])

#     log(f"[INFO] Found {len(files)} files. Parsing with {cpu_count()} workers...")

#     all_inputs, all_targets = [], []
#     total_errors = 0
#     corrupted_files = []

#     with Pool(cpu_count()) as pool:
#         results = list(tqdm(pool.imap(parse_file, files), total=len(files)))

#     for inputs, targets, fname, err_count in results:
#         all_inputs.extend(inputs)
#         all_targets.extend(targets)
#         if err_count == -1:
#             corrupted_files.append(fname)
#         else:
#             total_errors += err_count

#     log(f"[INFO] Total clusters processed: {len(all_inputs)}")
#     log(f"[INFO] Total clusters skipped due to errors: {total_errors}")
#     log(f"[INFO] Corrupted or unreadable files: {corrupted_files}")

#     if not all_inputs or not all_targets:
#         log("[FATAL] No valid data extracted. Aborting.")
#         return

#     # Convert to CPU tensors for normalization
#     inputs_tensor = torch.tensor(np.stack(all_inputs), dtype=torch.float32)
#     targets_tensor = torch.tensor(np.stack(all_targets), dtype=torch.float32)

#     # Normalize globally
#     input_min, input_max = inputs_tensor.min(), inputs_tensor.max()
#     target_min, target_max = targets_tensor.min(0).values, targets_tensor.max(0).values

#     norm_inputs = (inputs_tensor - input_min) / (input_max - input_min + 1e-8)
#     norm_targets = (targets_tensor - target_min) / (target_max - target_min + 1e-8)

#     # Save stats for later inverse normalization
#     with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
#         json.dump({
#             "input_min": float(input_min),
#             "input_max": float(input_max),
#             "target_min": target_min.tolist(),
#             "target_max": target_max.tolist()
#         }, f)
#     log("[INFO] Saved normalization stats")

#     # Save dataset in chunks
#     chunk_size = 20000
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     total_samples = norm_inputs.shape[0]

#     for i in range(0, total_samples, chunk_size):
#         chunk_inputs = norm_inputs[i:i + chunk_size]
#         chunk_targets = norm_targets[i:i + chunk_size]
#         torch.save(
#             {'inputs': chunk_inputs, 'targets': chunk_targets},
#             os.path.join(SAVE_DIR, f"chunk_{i // chunk_size:03}.pt")
#         )
#         log(f"[INFO] Saved chunk {i // chunk_size:03} with {len(chunk_inputs)} samples")

#     log(f"[SUCCESS] Preprocessing complete. All chunks saved in {SAVE_DIR}")


# if __name__ == "__main__":
#     main()


import os, json, torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATA_DIR = "training_data"
SAVE_DIR = "preprocessed_data"
SEQUENCE_LENGTH = 80
INPUT_DIM = 273
TARGET_DIM = 9
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
                if len(target) != TARGET_DIM:
                    raise ValueError(f"Target length {len(target)} != {TARGET_DIM}")

                end_idx = next((i for i in range(cluster_idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))
                frames = []
                i = cluster_idx + 2
                while i < end_idx:
                    if lines[i].startswith("<time slice"):
                        try:
                            frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
                            frames.append(np.array(frame).flatten())
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
    input_min = float("inf")
    input_max = float("-inf")
    target_min = np.full(TARGET_DIM, float("inf"))
    target_max = np.full(TARGET_DIM, float("-inf"))

    for inputs, targets, _, _ in results:
        for x in inputs:
            input_min = min(input_min, np.min(x))   # global across time/space
            input_max = max(input_max, np.max(x))
        for y in targets:
            target_min = np.minimum(target_min, y)  # per-feature min
            target_max = np.maximum(target_max, y)  # per-feature max

    stats = {
        "input_min": float(input_min),
        "input_max": float(input_max),
        "target_min": target_min.tolist(),
        "target_max": target_max.tolist()
    }

    with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f)

    log("[INFO] Saved normalization stats")
    return input_min, input_max, target_min, target_max


def normalize_and_save_chunks(results, input_min, input_max, target_min, target_max):
    buffer_inputs, buffer_targets = [], []
    chunk_id = 0

    for inputs, targets, _, _ in tqdm(results, desc="Normalizing and saving"):
        for x, y in zip(inputs, targets):
            # Global normalization for input
            norm_x = (x - input_min) / (input_max - input_min + 1e-8)

            # Per-feature normalization for target
            norm_y = (y - target_min) / (target_max - target_min + 1e-8)

            buffer_inputs.append(torch.tensor(norm_x, dtype=torch.float32))
            buffer_targets.append(torch.tensor(norm_y, dtype=torch.float32))

            if len(buffer_inputs) >= CHUNK_SIZE:
                torch.save({
                    'inputs': torch.stack(buffer_inputs),
                    'targets': torch.stack(buffer_targets)
                }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
                log(f"[INFO] Saved chunk {chunk_id:03} with {len(buffer_inputs)} samples")
                buffer_inputs, buffer_targets = [], []
                chunk_id += 1

    if buffer_inputs:
        torch.save({
            'inputs': torch.stack(buffer_inputs),
            'targets': torch.stack(buffer_targets)
        }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
        log(f"[INFO] Saved final chunk {chunk_id:03} with {len(buffer_inputs)} samples")

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

    log(f"[INFO] Total clusters processed: {total_clusters}")
    log(f"[INFO] Total clusters skipped due to errors: {total_errors}")
    log(f"[INFO] Corrupted or unreadable files: {corrupted_files}")

    if total_clusters == 0:
        log("[FATAL] No valid data extracted. Aborting.")
        return

    # Step 1: compute stats
    input_min, input_max, target_min, target_max = compute_stats(results)

    # Step 2: normalize and save
    normalize_and_save_chunks(results, input_min, input_max, np.array(target_min), np.array(target_max))

    log(f"[SUCCESS] Preprocessing complete. All chunks saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()
