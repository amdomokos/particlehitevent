import os
import glob
import torch
from torch.utils.data import Dataset


class PixelClusterDataset(Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.7, val_ratio=0.1, seed=42):
        assert split in ('train', 'val', 'test')
        self.data_dir = data_dir

        chunk_paths = sorted(glob.glob(os.path.join(data_dir, 'chunk_*.pt')))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk_*.pt files found in {data_dir}")

        n = len(chunk_paths)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            selected = indices[:n_train]
        elif split == 'val':
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        self.chunk_paths = [chunk_paths[i] for i in selected]

        self.index_map = []
        self.chunks = []
        for chunk_id, path in enumerate(self.chunk_paths):
            chunk = torch.load(path, weights_only=True)
            self.chunks.append(chunk)
            n_samples = chunk['X'].shape[0]
            for sample_id in range(n_samples):
                self.index_map.append((chunk_id, sample_id))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_id, sample_id = self.index_map[idx]
        chunk = self.chunks[chunk_id]
        X = chunk['X'][sample_id]
        y_module = chunk['y_module'][sample_id]
        Y = chunk['Y'][sample_id]
        return X, y_module, Y
