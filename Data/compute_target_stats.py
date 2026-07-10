"""Compute and freeze per-target training statistics for loss weighting.

WHAT: Reads the preprocessed ``chunk_*.pt`` files, selects the *training*
chunks with the shared ``split_chunk_indices`` split, and accumulates a
numerically stable single-pass mean/variance for each of the 6 target columns
of ``Y`` over all training samples. From the 5 *active* targets it derives
inverse-variance loss weights, renormalized to mean 1, and writes everything to
``preprocessed_data/target_stats.json``.

WHY: The targets live in two differently-scaled subspaces -- position
components (indices 0, 1) were z-scored in preprocessing while direction
components (indices 3, 4, 5) were unit-normalized -- so their raw MSE
contributions are wildly unequal (n_y's variance is ~230x smaller than a
position component's). Inverse-variance weights equalize each target's
contribution to the loss. z_entry (index 2) is a hardcoded constant (100.0) with
zero variance after z-scoring; it is recorded but flagged inactive and excluded
from weighting so it cannot produce inf/NaN weights.

WHEN TO RE-RUN: Only when preprocessing is re-run (new chunks / new norm stats).
This artifact is read by every training run and must be computed once and frozen.

CITATION: fixed-weight (homoscedastic uncertainty) multi-task weighting form of
Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics", CVPR 2018.
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import tqdm

# Run as ``python -m Data.compute_target_stats`` from the project root; also
# tolerate direct invocation by putting the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.config import SEED, TRAIN_RATIO, VAL_RATIO, split_chunk_indices

# Raw storage layout of Y: 6 columns, only 5 active. Index 2 (z_entry) is a
# hardcoded constant in the raw data and is identically 0 after z-scoring.
TARGET_NAMES = ["x_entry", "y_entry", "z_entry", "n_x", "n_y", "n_z"]
ACTIVE_INDICES = [0, 1, 3, 4, 5]
INACTIVE_INDICES = [2]
VAR_FLOOR = 1e-12  # active target with variance below this is a pipeline error


def log(level, msg):
    print(f"[{level}] {msg}")


def git_info():
    """Return (commit_hash, dirty_bool). Never raises; degrades to 'unknown'."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown", False
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
        dirty = bool(porcelain.strip())
    except Exception:
        dirty = False
    return commit, dirty


def summarize_existing(path):
    """Print a short summary of an already-present target_stats.json."""
    try:
        with open(path) as f:
            existing = json.load(f)
    except Exception as e:
        log("WARN", f"Could not read existing {path}: {e}")
        return
    log("INFO", f"Existing {path}:")
    log("INFO", f"    computed_at     = {existing.get('computed_at')}")
    log("INFO", f"    git_commit      = {existing.get('git_commit')}")
    log("INFO", f"    n_train_samples = {existing.get('n_train_samples')}")


def main():
    parser = argparse.ArgumentParser(
        description="Freeze per-target training statistics and inverse-variance "
                    "loss weights into preprocessed_data/target_stats.json.")
    parser.add_argument("--data-dir", default="preprocessed_data")
    parser.add_argument("--output",
                        default="preprocessed_data/target_stats.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing target_stats.json.")
    args = parser.parse_args()

    # Drift is the failure mode this whole design exists to prevent: loudly warn
    # if a CLI override diverges from the Data.config values preprocessing used.
    if args.seed != SEED:
        log("WARN", f"--seed={args.seed} DIFFERS from Data.config SEED={SEED}. "
                    f"Stats will not match the preprocessing split unless "
                    f"preprocessing used the same override.")
    if args.train_ratio != TRAIN_RATIO:
        log("WARN", f"--train-ratio={args.train_ratio} DIFFERS from "
                    f"Data.config TRAIN_RATIO={TRAIN_RATIO}.")
    if args.val_ratio != VAL_RATIO:
        log("WARN", f"--val-ratio={args.val_ratio} DIFFERS from "
                    f"Data.config VAL_RATIO={VAL_RATIO}.")

    if os.path.exists(args.output) and not args.force:
        log("INFO", f"{args.output} already exists; not overwriting.")
        summarize_existing(args.output)
        log("ERROR", "Re-run with --force to regenerate.")
        sys.exit(1)

    # --- Enumerate chunks and reconstruct the training split -----------------
    import glob
    chunk_paths = sorted(glob.glob(os.path.join(args.data_dir, "chunk_*.pt")))
    if not chunk_paths:
        log("ERROR", f"No chunk_*.pt files found in {args.data_dir}")
        sys.exit(1)
    n_chunks = len(chunk_paths)

    train_ids = split_chunk_indices(
        n_chunks, "train", seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_ids_sorted = sorted(train_ids)

    log("INFO", f"seed={args.seed} train_ratio={args.train_ratio} "
                f"val_ratio={args.val_ratio}")
    log("INFO", f"{n_chunks} chunks total; {len(train_ids)} assigned to train: "
                f"{train_ids_sorted}")

    # --- Single-pass accumulators over all training samples ------------------
    # float64 accumulators regardless of on-disk dtype, to avoid catastrophic
    # cancellation when squaring large-magnitude values.
    n_cols = len(TARGET_NAMES)
    col_sum = np.zeros(n_cols, dtype=np.float64)
    col_sum_sq = np.zeros(n_cols, dtype=np.float64)
    col_min = np.full(n_cols, np.inf, dtype=np.float64)
    col_max = np.full(n_cols, -np.inf, dtype=np.float64)
    count = 0
    chunk_dtype = None

    # One chunk mmap open at a time: holding many open has caused segfaults here.
    for path in tqdm(
            [chunk_paths[i] for i in train_ids_sorted],
            desc="[INFO] accumulating Y over train chunks"):
        chunk = torch.load(path, weights_only=True, mmap=True)
        Y = chunk["Y"]
        if chunk_dtype is None:
            chunk_dtype = str(Y.dtype)
        y = Y.numpy().astype(np.float64)  # small [N, 6] tensor -> own array
        col_sum += y.sum(axis=0)
        col_sum_sq += (y ** 2).sum(axis=0)
        col_min = np.minimum(col_min, y.min(axis=0))
        col_max = np.maximum(col_max, y.max(axis=0))
        count += y.shape[0]
        del y, Y, chunk  # release the mmap before opening the next chunk

    if count == 0:
        log("ERROR", "No training samples found.")
        sys.exit(1)

    mean = col_sum / count
    var = np.maximum(col_sum_sq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    log("INFO", f"Accumulated {count} training samples across "
                f"{len(train_ids_sorted)} chunks; Y dtype={chunk_dtype}")

    # --- Guard: any active target with ~zero variance is a pipeline error ----
    for i in ACTIVE_INDICES:
        if var[i] < VAR_FLOOR:
            log("ERROR", f"Active target '{TARGET_NAMES[i]}' (index {i}) has "
                         f"variance {var[i]:.3e} < {VAR_FLOOR:.0e}. A constant "
                         f"target has likely been reintroduced upstream. "
                         f"Refusing to compute weights.")
            sys.exit(1)

    # --- Inverse-variance weights, renormalized to mean 1 (float64) ----------
    raw_w = np.array([1.0 / var[i] for i in ACTIVE_INDICES], dtype=np.float64)
    n_active = len(ACTIVE_INDICES)
    weights = raw_w * (n_active / raw_w.sum())
    mean_w = float(weights.mean())
    if abs(mean_w - 1.0) > 1e-6:
        log("ERROR", f"Weight renormalization failed: mean(w)={mean_w} != 1.")
        sys.exit(1)
    log("INFO", f"Weight renormalization check: mean(w)={mean_w:.12f} (== 1)")

    # --- Verification table --------------------------------------------------
    print()
    print(f"{'target':<9}{'mean':>12}{'var':>12}{'std':>12}"
          f"{'1/var':>12}{'weight(m=1)':>14}")
    for k, i in enumerate(ACTIVE_INDICES):
        print(f"{TARGET_NAMES[i]:<9}{mean[i]:>12.6f}{var[i]:>12.6f}"
              f"{std[i]:>12.6f}{1.0 / var[i]:>12.4f}{weights[k]:>14.6f}")
    print()

    w_sum = float(weights.sum())
    print(f"sum(weights)     = {w_sum:.8f}  (target {float(n_active)}, "
          f"|err|={abs(w_sum - n_active):.2e})")
    assert abs(w_sum - n_active) < 1e-6, "sum(weights) != n_active"
    print(f"n_train_samples  = {count}")
    print(f"train chunk ids  = {train_ids_sorted}")

    # Sanity assertion: n_y must be the hardest (largest weight). If this fails
    # the data pipeline changed in a way that invalidates the plan -- hard error.
    w = {TARGET_NAMES[i]: weights[k] for k, i in enumerate(ACTIVE_INDICES)}
    if not (w["n_y"] > w["n_x"] and w["n_y"] > w["x_entry"]):
        log("ERROR", f"Sanity assertion FAILED: expected w[n_y] to dominate but "
                     f"got n_y={w['n_y']:.4f}, n_x={w['n_x']:.4f}, "
                     f"x_entry={w['x_entry']:.4f}. The pipeline changed; "
                     f"re-examine the plan before regenerating. Not writing JSON.")
        sys.exit(1)
    print(f"sanity           = OK (w[n_y]={w['n_y']:.4f} > w[n_x]={w['n_x']:.4f}, "
          f"w[x_entry]={w['x_entry']:.4f})")
    print()

    # --- n_z metadata (for the future 2-DOF-head ablation) -------------------
    nz = 5
    sign_consistent = bool(col_min[nz] * col_max[nz] > 0) or bool(
        col_min[nz] == col_max[nz])
    n_z_metadata = {
        "min": float(col_min[nz]),
        "max": float(col_max[nz]),
        "sign_consistent": sign_consistent,
        "abs_min": float(min(abs(col_min[nz]), abs(col_max[nz]))),
    }

    # --- Assemble JSON -------------------------------------------------------
    commit, dirty = git_info()
    if dirty:
        log("WARN", "Working tree is dirty; recording dirty=true in artifact.")

    per_target = []
    for i in range(n_cols):
        per_target.append({
            "name": TARGET_NAMES[i],
            "active": i in ACTIVE_INDICES,
            "mean": float(mean[i]),
            "var": float(var[i]),
            "std": float(std[i]),
            "min": float(col_min[i]),
            "max": float(col_max[i]),
        })

    out = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "n_chunks_total": n_chunks,
        "n_train_chunks": len(train_ids_sorted),
        "train_chunk_ids": train_ids_sorted,
        "n_train_samples": count,
        "targets": per_target,
        "active_indices": ACTIVE_INDICES,
        # float32 for the training loop to consume.
        "weights": [float(np.float32(x)) for x in weights],
        "weight_normalization": "mean_1",
        "weight_mean": mean_w,
        "n_z_metadata": n_z_metadata,
        "git_commit": commit,
        "dirty": dirty,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": os.path.abspath(args.data_dir),
        "chunk_dtype": chunk_dtype,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log("INFO", f"Wrote {args.output}")


if __name__ == "__main__":
    main()
