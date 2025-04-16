# # import os
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # import sys
# # import matplotlib.pyplot as plt
# # # Add the parent directory to the path so modules can be found
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # class PixelClusterDataset(Dataset):
# #     def __init__(self, data_dir, sequence_length=100, input_dim=273):
# #         self.data_dir = data_dir
# #         self.sequence_length = sequence_length
# #         self.input_dim = input_dim
# #         self.files = sorted([f for f in os.listdir(data_dir) if f.startswith("pixel_clusters") and f.endswith(".out")])
# #         self.input_data = []
# #         self.target_data = []

# #         # Prepare the dataset by reading each file
# #         self._prepare_data()

# #         # Normalize the data
# #         self._normalize_data()

# #     def _prepare_data(self):
# #         file_cluster_counts = []  # Store cluster counts per file
# #         all_targets = []  # Store all target values for visualization
        
# #         for file_name in self.files:
# #             file_path = os.path.join(self.data_dir, file_name)
# #             with open(file_path, 'r') as f:
# #                 lines = f.readlines()

# #             clusters = []
# #             cluster_indices = []

# #             # Identify all <cluster> positions
# #             for i, line in enumerate(lines):
# #                 if line.strip() == "<cluster>":
# #                     clusters.append(i)
# #                     cluster_indices.append(i)

# #             file_cluster_counts.append(len(clusters))
# #             event_count = 0
# #             for idx in range(len(cluster_indices)):
# #                 target_line_index = cluster_indices[idx] + 1
# #                 # Handle last cluster by going to the end of the file
# #                 next_cluster_index = cluster_indices[idx + 1] if idx + 1 < len(cluster_indices) else len(lines)

# #                 # Extract target values
# #                 try:
# #                     target_values = list(map(float, lines[target_line_index].strip().split()))
# #                     all_targets.append(target_values)
# #                 except Exception as e:
# #                     print(f"Skipping invalid target in {file_name} at line {target_line_index}: {e}")
# #                     continue

# #                 # Extract time slices between current <cluster> and next <cluster>
# #                 frames = []
# #                 i = target_line_index + 1
# #                 while i < next_cluster_index:
# #                     if lines[i].startswith("<time slice"):
# #                         try:
# #                             frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
# #                             flat_frame = np.array(frame).flatten()  # shape: (273,)
# #                             frames.append(flat_frame)
# #                             i += 14  # 1 line for <time slice>, 13 for matrix
# #                         except Exception as e:
# #                             print(f"Error parsing frame in {file_name} at line {i}: {e}")
# #                             i += 1
# #                     else:
# #                         i += 1

# #                 if len(frames) == 0:
# #                     continue

# #                 # Pad or truncate sequence to fixed length
# #                 if len(frames) > self.sequence_length:
# #                     frames = frames[:self.sequence_length]
# #                 else:
# #                     padding = np.zeros((self.sequence_length - len(frames), self.input_dim))
# #                     frames = np.vstack((frames, padding)) if len(frames) > 0 else padding

# #                 self.input_data.append(frames)
# #                 self.target_data.append(target_values)
# #                 event_count += 1

# #             print(f"{file_name}: {event_count} events found")
        
# #         if event_count != len(cluster_indices):
# #             print(f"Warning: Only parsed {event_count}/{len(cluster_indices)} clusters in {file_name}")

# #         print("assigning to self")
# #         self.input_data = np.array(self.input_data)
# #         self.target_data = np.array(self.target_data)
# #         self.file_cluster_counts = file_cluster_counts
# #         self.all_targets = np.array(all_targets)

# #         print(f"Total events: {len(self.input_data)}")
# #         print(f"Input shape: {self.input_data.shape}")
# #         print(f"Target shape: {self.target_data.shape}")

# #     def _normalize_data(self):
# #         print("normalizing data")
# #         input_min = np.min(self.input_data)
# #         input_max = np.max(self.input_data)
# #         if input_max > input_min:
# #             self.input_data = (self.input_data - input_min) / (input_max - input_min)

# #         target_min = np.min(self.target_data)
# #         target_max = np.max(self.target_data)
# #         if target_max > target_min:
# #             self.target_data = (self.target_data - target_min) / (target_max - target_min)

# #         self.input_norm = {"min": input_min, "max": input_max}
# #         self.target_norm = {"min": target_min, "max": target_max}

# #     def __len__(self):
# #         return len(self.input_data)

# #     def __getitem__(self, idx):
# #         return torch.tensor(self.input_data[idx], dtype=torch.float32), torch.tensor(self.target_data[idx], dtype=torch.float32)

# #     def visualize_cluster_distribution(self):
# #         print("visualizing cluster distribution")
# #         """Visualize the distribution of clusters across files"""
# #         plt.figure(figsize=(12, 6))
# #         plt.bar(range(len(self.file_cluster_counts)), self.file_cluster_counts)
# #         plt.xlabel('File Index')
# #         plt.ylabel('Number of Clusters')
# #         plt.title('Cluster Distribution Across Files')
# #         plt.show()

# #     def visualize_target_distribution(self, file_index=0):
# #         print("visualizing target distribution")
# #         """Visualize target values for clusters in a specific file"""
# #         if file_index >= len(self.files):
# #             print(f"File index {file_index} out of range. Total files: {len(self.files)}")
# #             return

# #         # Get target values for the specified file
# #         file_targets = self.all_targets[file_index*100:(file_index+1)*100]  # Assuming roughly 100 clusters per file
        
# #         plt.figure(figsize=(12, 6))
# #         for i in range(file_targets.shape[1]):  # Plot each target dimension
# #             plt.plot(file_targets[:, i], label=f'Target {i+1}')
        
# #         plt.xlabel('Cluster Index')
# #         plt.ylabel('Target Value')
# #         plt.title(f'Target Values Distribution for File {self.files[file_index]}')
# #         plt.legend()
# #         plt.show()


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# class PixelClusterDataset(Dataset):
#     def __init__(self, data_dir, sequence_length=100, input_dim=273):
#         self.data_dir = data_dir
#         self.sequence_length = sequence_length
#         self.input_dim = input_dim
#         self.files = sorted([f for f in os.listdir(data_dir) if f.startswith("pixel_clusters") and f.endswith(".out")])
#         self.cluster_index = []  # stores (file_path, target_line_index, cluster_end_index)

#         print("[DEBUG] Indexing clusters across files...")
#         self._index_clusters()
#         print(f"[DEBUG] Total events indexed: {len(self.cluster_index)}")

#     def _index_clusters(self):
#         for file_name in self.files:
#             file_path = os.path.join(self.data_dir, file_name)
#             try:
#                 with open(file_path, 'r') as f:
#                     lines = f.readlines()
#             except Exception as e:
#                 print(f"[ERROR] Could not read {file_path}: {e}")
#                 continue

#             cluster_start_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]
#             cluster_end_indices = cluster_start_indices[1:] + [len(lines)]

#             for start, end in zip(cluster_start_indices, cluster_end_indices):
#                 target_line_index = start + 1
#                 self.cluster_index.append((file_path, target_line_index, end))

#     def __len__(self):
#         return len(self.cluster_index)

#     def __getitem__(self, idx):
#         file_path, target_line_index, end_index = self.cluster_index[idx]

#         try:
#             with open(file_path, 'r') as f:
#                 lines = f.readlines()[target_line_index:end_index]
#         except Exception as e:
#             raise RuntimeError(f"[ERROR] Reading lines {target_line_index}–{end_index} in {file_path}: {e}")

#         try:
#             target = list(map(float, lines[0].strip().split()))
#         except Exception as e:
#             raise ValueError(f"[ERROR] Invalid target at {file_path}, line {target_line_index}: {e}")

#         frames = []
#         i = 1
#         while i < len(lines):
#             if lines[i].startswith("<time slice"):
#                 try:
#                     frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                     flat_frame = np.array(frame).flatten()
#                     frames.append(flat_frame)
#                     i += 14
#                 except Exception as e:
#                     print(f"[WARN] Skipping bad frame at {file_path}, line {target_line_index + i}: {e}")
#                     i += 1
#             else:
#                 i += 1

#         if len(frames) > self.sequence_length:
#             frames = frames[:self.sequence_length]
#         else:
#             padding = np.zeros((self.sequence_length - len(frames), self.input_dim))
#             frames = np.vstack((frames, padding)) if len(frames) > 0 else padding

#         input_tensor = torch.tensor(frames, dtype=torch.float32)
#         target_tensor = torch.tensor(target, dtype=torch.float32)
#         return input_tensor, target_tensor


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import json
# import sys
# from tqdm import tqdm
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# class PixelClusterDataset(Dataset):
#     def __init__(self, data_dir, sequence_length=100, input_dim=273, device='cpu', normalize=True):
#         self.data_dir = data_dir
#         self.sequence_length = sequence_length
#         self.input_dim = input_dim
#         self.device = device
#         self.normalize = normalize

#         self.norm_file = os.path.join(self.data_dir, "normalization_stats.npz")

#         # Gather all relevant files
#         self.files = sorted([
#             os.path.join(self.data_dir, f)
#             for f in os.listdir(self.data_dir)
#             if f.startswith("pixel_clusters") and f.endswith(".out")
#         ])

#         # Build index mapping: list of tuples (file_path, cluster_index)
#         self.index_map = self._build_index()

#         # Load or compute normalization parameters
#         if os.path.exists(self.norm_file):
#             print("Loading normalization parameters from file...")
#             with open(self.norm_file, 'r') as f:
#                 norm_params = json.load(f)
#                 self.input_min = norm_params['input_min']
#                 self.input_max = norm_params['input_max']
#                 self.target_min = norm_params['target_min']
#                 self.target_max = norm_params['target_max']
#         else:
#             self._compute_normalization_params()
#             print("Normalization parameters computed and saved.")
#             # Save normalization parameters
#             norm_params = {
#                 'input_min': self.input_min,
#                 'input_max': self.input_max,
#                 'target_min': self.target_min,
#                 'target_max': self.target_max
#             }
#             with open(self.norm_file, 'w') as f:
#                 json.dump(norm_params, f)

                

#     def _build_index(self):
#         index_map = []
#         for file_path in self.files:
#             with open(file_path, 'r') as f:
#                 lines = f.readlines()
#             cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]
#             for idx in cluster_indices:
#                 index_map.append((file_path, idx))
#         return index_map

#     def _compute_normalization_params(self):
#         print("Computing normalization parameters...")
#         input_mins = []
#         input_maxs = []
#         target_mins = []
#         target_maxs = []

#         for file_path, cluster_idx in tqdm(self.index_map, desc="Processing events"):
#             with open(file_path, 'r') as f:
#                 lines = f.readlines()

#             # Extract target values
#             target_line = lines[cluster_idx + 1].strip()
#             target_values = list(map(float, target_line.split()))
#             target_mins.append(np.min(target_values))
#             target_maxs.append(np.max(target_values))

#             # Extract frames
#             frames = []
#             for i, line in enumerate(lines):
#                 if line.startswith("<time slice"):
#                     try:
#                         frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                         frames.append(np.array(frame).flatten())
#                     except (IndexError, ValueError):
#                         continue

#             if frames:
#                 frames = np.array(frames)
#                 input_mins.append(np.min(frames))
#                 input_maxs.append(np.max(frames))

#         self.input_min = float(np.min(input_mins)) if input_mins else 0.0
#         self.input_max = float(np.max(input_maxs)) if input_maxs else 1.0
#         self.target_min = float(np.min(target_mins)) if target_mins else 0.0
#         self.target_max = float(np.max(target_maxs)) if target_maxs else 1.0

#         np.savez(self.norm_file,
#                 input_min=self.input_min,
#                 input_max=self.input_max,
#                 target_min=self.target_min,
#                 target_max=self.target_max)

#         print(f"Normalization parameters computed and saved to {self.norm_file}.")


#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, idx):
#         file_path, cluster_idx = self.index_map[idx]
#         with open(file_path, 'r') as f:
#             lines = f.readlines()

#         # Extract target values
#         target_line = lines[cluster_idx + 1].strip()
#         target_values = np.array(list(map(float, target_line.split())), dtype=np.float32)

#         # Extract frames
#         frames = []
#         for i, line in enumerate(lines):
#             if line.startswith("<time slice"):
#                 try:
#                     frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                     frames.append(np.array(frame).flatten())
#                 except (IndexError, ValueError):
#                     continue

#         # Handle varying sequence lengths
#         if len(frames) > self.sequence_length:
#             frames = frames[:self.sequence_length]
#         else:
#             padding = [np.zeros(self.input_dim, dtype=np.float32) for _ in range(self.sequence_length - len(frames))]
#             frames.extend(padding)

#         input_data = np.array(frames, dtype=np.float32)

#         # Normalize input and target
#         input_data = (input_data - self.input_min) / (self.input_max - self.input_min + 1e-8)
#         print("Input data shape:", input_data.shape)
#         target_values = (target_values - self.target_min) / (self.target_max - self.target_min + 1e-8)
#         print("Target values shape:", target_values.shape)
#         # Convert to torch and move to device
#         input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
#         target_tensor = torch.tensor(target_values, dtype=torch.float32).to(self.device)
#         print("Input tensor shape:", input_tensor.shape)
#         print("Target tensor shape:", target_tensor.shape)

#         return input_tensor, target_tensor


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm

class PixelClusterDataset(Dataset):
    def __init__(self, data_dir, sequence_length=100, input_dim=273, device='cpu', normalize=True):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.device = device
        self.normalize = normalize

        self.norm_file = os.path.join(self.data_dir, "normalization_stats.json")

        # Gather all relevant files
        self.files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith("pixel_clusters") and f.endswith(".out")
        ])

        # Build index mapping: list of tuples (file_path, cluster_index)
        self.index_map = self._build_index()

        # Load or compute normalization parameters
        if os.path.exists(self.norm_file):
            print("Loading normalization parameters from file...")
            with open(self.norm_file, 'r') as f:
                norm_params = json.load(f)
                self.input_min = np.array(norm_params['input_min'])
                self.input_max = np.array(norm_params['input_max'])
                self.target_min = np.array(norm_params['target_min'])
                self.target_max = np.array(norm_params['target_max'])
        else:
            self._compute_normalization_params(sample_size=10000)

    def _build_index(self):
        index_map = []
        for file_path in self.files:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]
            for idx in cluster_indices:
                index_map.append((file_path, idx))
        return index_map

    def _compute_normalization_params(self, sample_size=10000):
        print("Computing normalization parameters with sampling...")
        import random

        input_min = None
        input_max = None
        target_min = None
        target_max = None

        sampled_indices = random.sample(self.index_map, min(sample_size, len(self.index_map)))

        for file_path, cluster_idx in tqdm(sampled_indices, desc="Sampling events"):
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Extract target values
            target_line = lines[cluster_idx + 1].strip()
            target_values = np.array(list(map(float, target_line.split())), dtype=np.float32)
            target_min = target_values if target_min is None else np.minimum(target_min, target_values)
            target_max = target_values if target_max is None else np.maximum(target_max, target_values)

            # Extract frames
            frames = []
            for i, line in enumerate(lines):
                if line.startswith("<time slice"):
                    try:
                        frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
                        frames.append(np.array(frame).flatten())
                    except (IndexError, ValueError):
                        continue

            if frames:
                frames = np.array(frames, dtype=np.float32)
                frame_min = np.min(frames, axis=0)
                frame_max = np.max(frames, axis=0)
                input_min = frame_min if input_min is None else np.minimum(input_min, frame_min)
                input_max = frame_max if input_max is None else np.maximum(input_max, frame_max)

        self.input_min = input_min if input_min is not None else 0.0
        self.input_max = input_max if input_max is not None else 1.0
        self.target_min = target_min if target_min is not None else 0.0
        self.target_max = target_max if target_max is not None else 1.0

        # Save stats
        norm_params = {
            'input_min': self.input_min.tolist(),
            'input_max': self.input_max.tolist(),
            'target_min': self.target_min.tolist(),
            'target_max': self.target_max.tolist()
        }
        with open(self.norm_file, 'w') as f:
            json.dump(norm_params, f)
        print(f"Saved normalization stats to {self.norm_file}.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, cluster_idx = self.index_map[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract target values
        target_line = lines[cluster_idx + 1].strip()
        target_values = np.array(list(map(float, target_line.split())), dtype=np.float32)

        # Extract frames
        frames = []
        for i, line in enumerate(lines):
            if line.startswith("<time slice"):
                try:
                    frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
                    frames.append(np.array(frame).flatten())
                except (IndexError, ValueError):
                    continue

        # Handle varying sequence lengths
        if len(frames) > self.sequence_length:
            frames = frames[:self.sequence_length]
        else:
            padding = [np.zeros(self.input_dim, dtype=np.float32) for _ in range(self.sequence_length - len(frames))]
            frames.extend(padding)

        input_data = np.array(frames, dtype=np.float32)

        # Normalize input and target
        input_data = (input_data - self.input_min) / (self.input_max - self.input_min + 1e-8)
        target_values = (target_values - self.target_min) / (self.target_max - self.target_min + 1e-8)

        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_values, dtype=torch.float32).to(self.device)

        return input_tensor, target_tensor
