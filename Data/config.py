"""Lightweight shared configuration for the preprocessing + data pipeline.

Single source of truth for values that MUST agree between preprocessing
(``Data/preprocess_v2.py``) and data loading (``Data/dataset_v2.py``). The most
important is the split ``SEED``: preprocessing computes normalization statistics
over the *training* chunks only, and the dataset reconstructs that same split at
load time. If the two ever disagreed, validation/test data would leak into the
training statistics.

Override the seed at runtime without touching code via the ``PHE_SEED``
environment variable, e.g.::

    PHE_SEED=7 python Data/preprocess_v2.py
"""
import json
import os
import random

import torch

# Global RNG seed for the chunk-level train/val/test split. Read here once so no
# module hardcodes its own literal; override with the PHE_SEED env var.
SEED = int(os.environ.get("PHE_SEED", "42"))

# Split fractions. Test is the remainder: 1 - TRAIN_RATIO - VAL_RATIO.
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

# Samples per saved chunk_*.pt file.
CHUNK_SIZE = 20000


def split_chunk_indices(n_chunks, split, seed=SEED,
                        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Chunk indices assigned to ``split`` ('train' | 'val' | 'test').

    Deterministic given ``n_chunks`` and ``seed``. Shared by preprocessing (to
    select which chunks feed stat computation) and the dataset (to select a
    split's chunks) so the two can never drift apart.
    """
    assert split in ('train', 'val', 'test')
    perm = torch.randperm(
        n_chunks, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    n_train = int(n_chunks * train_ratio)
    n_val = int(n_chunks * val_ratio)
    if split == 'train':
        return perm[:n_train]
    if split == 'val':
        return perm[n_train:n_train + n_val]
    return perm[n_train + n_val:]


def seed_everything(seed=SEED):
    """Seed Python, NumPy, and torch RNGs from the one global ``SEED``.

    Call once at the top of a training script so model init and DataLoader
    shuffling are reproducible and driven by ``PHE_SEED`` — no per-script seed
    literals. Returns a seeded ``torch.Generator`` for callers that want to pass
    one to ``DataLoader(generator=...)`` explicitly. Seeding torch's default
    generator already makes ``shuffle=True`` deterministic without it.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


# --- Target layout (frozen experiment-design constants; Phase 0/1) ---------
#
# These describe the fixed target layout established by Phase 0's data
# inspection and recorded in ``preprocessed_data/target_stats.json``. Every
# downstream file (loss, metrics, output heads, evaluation, plotting) reads
# them from here so the "5 active targets, not 6" fact lives in exactly one
# place. Changing any of them WITHOUT re-running preprocessing and
# ``python -m Data.compute_target_stats`` will silently corrupt training:
# the frozen weights and per-target stats would no longer line up with the
# ordering the code assumes.

# Raw Y-tensor column layout as stored in chunk_*.pt files.
RAW_TARGET_NAMES = ('x_entry', 'y_entry', 'z_entry', 'n_x', 'n_y', 'n_z')
RAW_TARGET_DIM = 6

# Active targets used for loss, metrics, and reporting.
# z_entry (index 2) is a preprocessing-time constant and is excluded.
# Order here IS the canonical ordering used by every downstream tensor
# (weights vector, per-target loss vector, metrics dict, output head).
ACTIVE_INDICES = (0, 1, 3, 4, 5)
ACTIVE_TARGET_NAMES = ('x_entry', 'y_entry', 'n_x', 'n_y', 'n_z')
ACTIVE_TARGET_DIM = 5

# Positional vs directional slicing within the ACTIVE_TARGET ordering
# (not the raw ordering) — used by the direction-penalty term and by
# any future physics-aware output head.
POSITION_SLICE = slice(0, 2)   # x_entry, y_entry
DIRECTION_SLICE = slice(2, 5)  # n_x, n_y, n_z

# Physics facts recorded from Phase 0; documented here for downstream files
# so they don't re-derive them or make assumptions inconsistent with the data.
N_Z_SIGN = -1                  # n_z is always negative in the training split
N_Z_ABS_MIN = 0.099            # rounded down from the measured 0.09935 floor,
                               # conservative bound for future 2-DOF-head safety

# y_module is continuous (~all-unique floats in ±8.1), NOT categorical.
# Future model conditioning must project as a scalar, not embed as an ID.
Y_MODULE_IS_CATEGORICAL = False
Y_MODULE_RANGE = (-8.1, 8.1)   # approximate; recorded for sanity-check use

# Path to the frozen target statistics file. Downstream code loads weights
# and per-target stats from here; do not recompute or hardcode.
#
# Anchor stats-file paths to the repo root (the parent of the Data/
# directory containing this file), so paths resolve correctly regardless
# of the process's current working directory — important for cloud
# environments (RunPod), subprocess launches, and notebook kernels
# started from arbitrary directories.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_STATS_PATH = os.path.join(_REPO_ROOT, 'preprocessed_data', 'target_stats.json')
NORM_STATS_PATH = os.path.join(_REPO_ROOT, 'preprocessed_data', 'norm_stats.json')


def load_target_stats(path=TARGET_STATS_PATH):
    """Load the frozen target statistics artifact.

    Returns a dict with keys including 'weights' (list of float32,
    len == ACTIVE_TARGET_DIM, mean == 1), per-target column stats
    (mean/var/std/min/max), 'active_indices', and reproducibility
    metadata (seed, git_commit, n_train_samples, etc.).

    Raises FileNotFoundError with an actionable message pointing at
    Data/compute_target_stats.py if the file is missing, and RuntimeError
    if the artifact has drifted from config.py's frozen constants.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"target_stats.json not found at {path}. "
            f"Run: python -m Data.compute_target_stats"
        )
    with open(path, "r") as f:
        stats = json.load(f)

    if len(stats['weights']) != ACTIVE_TARGET_DIM:
        raise RuntimeError(
            f"config.py says ACTIVE_TARGET_DIM={ACTIVE_TARGET_DIM}, but "
            f"target_stats.json has {len(stats['weights'])} weights; "
            f"regenerate with python -m Data.compute_target_stats --force"
        )
    if stats['active_indices'] != list(ACTIVE_INDICES):
        raise RuntimeError(
            f"config.py says ACTIVE_INDICES={ACTIVE_INDICES}, but "
            f"target_stats.json says active_indices={stats['active_indices']}; "
            f"regenerate with python -m Data.compute_target_stats --force"
        )
    if abs(sum(stats['weights']) - ACTIVE_TARGET_DIM) >= 1e-4:
        raise RuntimeError(
            f"target_stats.json weights sum to {sum(stats['weights'])}, "
            f"expected {ACTIVE_TARGET_DIM} (mean_1 normalization); "
            f"regenerate with python -m Data.compute_target_stats --force"
        )

    return stats


# Sanity: raw layout self-consistency (cheap, runs on every import).
assert len(RAW_TARGET_NAMES) == RAW_TARGET_DIM
assert len(ACTIVE_INDICES) == ACTIVE_TARGET_DIM == len(ACTIVE_TARGET_NAMES)
assert all(RAW_TARGET_NAMES[i] == ACTIVE_TARGET_NAMES[j]
           for j, i in enumerate(ACTIVE_INDICES))
assert 2 not in ACTIVE_INDICES, "z_entry (index 2) must remain excluded"
