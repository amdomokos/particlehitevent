import os
import glob
import torch
from torch.utils.data import Dataset

from Data.config import SEED, TRAIN_RATIO, VAL_RATIO, split_chunk_indices


class PixelClusterDataset(Dataset):
    """Pixel-cluster dataset backed by preprocessed ``chunk_*.pt`` files.

    Chunks are memory-mapped lazily rather than loaded fully into RAM. Only the
    samples actually indexed are paged in by the OS (reclaimable page cache), so
    peak memory stays bounded regardless of total dataset size — important since
    the full set is ~35 GB across 20 chunks and previously every chunk of a split
    was held resident in memory.
    """

    def __init__(self, data_dir, split='train', train_ratio=TRAIN_RATIO,
                 val_ratio=VAL_RATIO, seed=SEED):
        assert split in ('train', 'val', 'test')
        self.data_dir = data_dir

        chunk_paths = sorted(glob.glob(os.path.join(data_dir, 'chunk_*.pt')))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk_*.pt files found in {data_dir}")

        # Same split logic (and seed) that preprocessing used to pick the
        # training chunks for stat computation — kept in one place in Data.config
        # so training stats and this split can't drift apart.
        selected = split_chunk_indices(
            len(chunk_paths), split, seed=seed,
            train_ratio=train_ratio, val_ratio=val_ratio,
        )

        self.chunk_paths = [chunk_paths[i] for i in selected]

        # Read only each chunk's sample count to build the global index map.
        # mmap=True maps the file without reading tensor data into RAM, so this
        # touches metadata only; the mapping is released immediately after.
        self.index_map = []
        for chunk_id, path in enumerate(self.chunk_paths):
            chunk = torch.load(path, weights_only=True, mmap=True)
            n_samples = chunk['X'].shape[0]
            for sample_id in range(n_samples):
                self.index_map.append((chunk_id, sample_id))
            del chunk  # reopened lazily on first access

        # Per-process cache of open memory-mapped chunks. Excluded from pickling
        # so each DataLoader worker opens its own mappings (see __getstate__).
        self._chunk_cache = {}

    def _get_chunk(self, chunk_id):
        chunk = self._chunk_cache.get(chunk_id)
        if chunk is None:
            chunk = torch.load(self.chunk_paths[chunk_id], weights_only=True, mmap=True)
            self._chunk_cache[chunk_id] = chunk
        return chunk

    def __getstate__(self):
        # Don't pickle open memory maps to worker processes; they reopen lazily.
        state = self.__dict__.copy()
        state['_chunk_cache'] = {}
        return state

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_id, sample_id = self.index_map[idx]
        chunk = self._get_chunk(chunk_id)
        X = chunk['X'][sample_id]
        y_module = chunk['y_module'][sample_id]
        Y = chunk['Y'][sample_id]
        return X, y_module, Y
