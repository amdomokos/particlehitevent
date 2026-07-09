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
