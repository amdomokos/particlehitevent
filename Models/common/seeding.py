"""Seeding utilities for the shared training engine (Phase 3).

The engine seeds through this module (not ``Data.config.seed_everything``,
which predates it and stays for the preprocessing scripts) because training
additionally needs XPU seeding, the opt-in strict-determinism cuDNN flags,
and a DataLoader ``worker_init_fn``.
"""
import random

import numpy as np
import torch


def seed_everything(seed, strict=False):
    """Seed Python, numpy, and torch (CPU + all CUDA/XPU devices).

    ``strict=True`` additionally sets ``cudnn.deterministic = True`` and
    ``cudnn.benchmark = False``. Deterministic cuDNN kernels are slow, so
    the default is best-effort reproducibility without them. Bitwise
    reproducibility across GPU architectures is not promised either way —
    the checkpoint metadata records the GPU for exactly that reason.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[seed] seed={seed} strict_deterministic={strict}")


def worker_init_fn(worker_id):
    """Reseed numpy/random in each DataLoader worker deterministically.

    torch already gives each worker ``base_seed + worker_id`` as its initial
    seed; this propagates that to the other RNGs so workers never share
    correlated random state. No augmentations exist today — wired in anyway
    so the engine's DataLoader construction is future-proof.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
