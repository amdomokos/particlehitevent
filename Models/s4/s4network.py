# import torch
# import torch.nn as nn
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# class ConstrainedS4Layer(nn.Module):
#     def __init__(self, height=13, width=21):
#         super(ConstrainedS4Layer, self).__init__()
#         self.height = height
#         self.width = width
#         self.hidden_dim = height * width

#         # State-space matrices
#         self.A = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
#         self.B = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
#         self.C = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)

#         # Define and initialize masks
#         A_mask, B_mask, C_mask, peripheral_indices = self._initialize_nearest_neighbor_constraints()

#         # Register masks and indices as buffers
#         self.register_buffer("A_mask", A_mask)
#         self.register_buffer("B_mask", B_mask)
#         self.register_buffer("C_mask", C_mask)
#         self.register_buffer("peripheral_indices", peripheral_indices)

#     def _initialize_nearest_neighbor_constraints(self):
#         print("Initializing nearest-neighbor constraints...")
#         """Sets the mask for nearest-neighbor constraints on matrices A, B, and C."""
#         A_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
#         B_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
#         C_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
#         peripheral_indices = []

#         for i in range(self.height):
#             for j in range(self.width):
#                 idx = i * self.width + j  # Flattened index for each pixel

#                 neighbors = [
#                     (i - 1) * self.width + j if i > 0 else None,         # Top
#                     (i + 1) * self.width + j if i < self.height - 1 else None,  # Bottom
#                     i * self.width + (j - 1) if j > 0 else None,         # Left
#                     i * self.width + (j + 1) if j < self.width - 1 else None    # Right
#                 ]

#                 neighbors = [n for n in neighbors if n is not None]  # Filter out None entries
#                 A_mask[idx, neighbors] = 1  # Allow connections to neighbors in A
#                 B_mask[idx, idx] = 1  # Self-connection for B

#                 # Only apply C_mask for peripheral pixels
#                 if i in {0, self.height - 1} or j in {0, self.width - 1}:
#                     C_mask[idx, idx] = 1  # Output connections only for peripheral positions
#                     peripheral_indices.append(idx)

#         return A_mask, B_mask, C_mask, torch.tensor(peripheral_indices)

#     def forward(self, x):
#         masked_A = self.A * self.A_mask
#         masked_B = self.B * self.B_mask

#         batch_size, sequence_length, _ = x.shape
#         outputs = []

#         state = torch.zeros(batch_size, self.hidden_dim, device=x.device)

#         for t in range(sequence_length):
#             u_t = x[:, t, :]
#             state = torch.tanh(state @ masked_A.T + u_t @ masked_B.T)

#             if t == sequence_length - 1:
#                 y_t = state[:, self.peripheral_indices]
#                 outputs.append(y_t)

#         outputs = torch.cat(outputs, dim=1)
#         return outputs

#     def save_matrices(self, output_dir="results/matrices"):
#         os.makedirs(output_dir, exist_ok=True)
#         torch.save(self.A.detach().cpu(), os.path.join(output_dir, "A_matrix.pt"))
#         torch.save(self.B.detach().cpu(), os.path.join(output_dir, "B_matrix.pt"))
#         torch.save(self.C.detach().cpu(), os.path.join(output_dir, "C_matrix.pt"))
#         print(f"Matrices saved to {output_dir}")

# class S4PredictionNetwork(nn.Module):
#     def __init__(self, hidden_output_dim, final_output_dim, height=13, width=21):
#         super(S4PredictionNetwork, self).__init__()

#         self.s4_layer = ConstrainedS4Layer(height, width)
#         s4_output_dim = self.s4_layer.peripheral_indices.shape[0]

#         self.fc1 = nn.Linear(s4_output_dim, hidden_output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(hidden_output_dim, final_output_dim)

#     def forward(self, x):
#         s4_output = self.s4_layer(x)
#         hidden = self.relu(self.fc1(s4_output))
#         hidden = self.dropout(hidden)
#         prediction = self.fc2(hidden)
#         return prediction


# import torch
# import torch.nn as nn
# import os

# class ConstrainedS4Layer(nn.Module):
#     def __init__(self, height=13, width=21):
#         super(ConstrainedS4Layer, self).__init__()
#         self.height = height
#         self.width = width
#         self.hidden_dim = height * width

#         self.A = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
#         self.B = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
#         self.C = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)

#         A_mask, B_mask, C_mask, peripheral_indices = self._initialize_nearest_neighbor_constraints()
#         self.register_buffer("A_mask", A_mask)
#         self.register_buffer("B_mask", B_mask)
#         self.register_buffer("C_mask", C_mask)
#         self.register_buffer("peripheral_indices", peripheral_indices)

#     def _initialize_nearest_neighbor_constraints(self):
#         A_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
#         B_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
#         C_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
#         peripheral_indices = []

#         for i in range(self.height):
#             for j in range(self.width):
#                 idx = i * self.width + j
#                 neighbors = [
#                     (i - 1) * self.width + j if i > 0 else None,
#                     (i + 1) * self.width + j if i < self.height - 1 else None,
#                     i * self.width + (j - 1) if j > 0 else None,
#                     i * self.width + (j + 1) if j < self.width - 1 else None
#                 ]
#                 neighbors = [n for n in neighbors if n is not None]
#                 A_mask[idx, neighbors] = 1
#                 B_mask[idx, idx] = 1
#                 if i in {0, self.height - 1} or j in {0, self.width - 1}:
#                     C_mask[idx, idx] = 1
#                     peripheral_indices.append(idx)

#         return A_mask, B_mask, C_mask, torch.tensor(peripheral_indices)

#     def forward(self, x):
#         masked_A = self.A * self.A_mask
#         masked_B = self.B * self.B_mask
#         batch_size, sequence_length, _ = x.shape
#         state = torch.zeros(batch_size, self.hidden_dim, device=x.device)

#         for t in range(sequence_length):
#             u_t = x[:, t, :]
#             state = torch.tanh(state @ masked_A.T + u_t @ masked_B.T)

#         y_t = state[:, self.peripheral_indices]
#         return y_t

#     def save_matrices(self, output_dir="results/matrices"):
#         os.makedirs(output_dir, exist_ok=True)
#         torch.save(self.A.detach().cpu(), os.path.join(output_dir, "A_matrix.pt"))
#         torch.save(self.B.detach().cpu(), os.path.join(output_dir, "B_matrix.pt"))
#         torch.save(self.C.detach().cpu(), os.path.join(output_dir, "C_matrix.pt"))
#         print(f"[INFO] Matrices saved to {output_dir}")


# class S4PredictionNetwork(nn.Module):
#     def __init__(self, hidden_output_dim, final_output_dim, height=13, width=21):
#         super(S4PredictionNetwork, self).__init__()
#         self.s4_layer = ConstrainedS4Layer(height, width)
#         s4_output_dim = self.s4_layer.peripheral_indices.shape[0]

#         self.fc1 = nn.Linear(s4_output_dim, hidden_output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(hidden_output_dim, final_output_dim)

#     def forward(self, x):
#         x = self.s4_layer(x)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         return self.fc2(x)


import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

class ConstrainedS4Layer(nn.Module):
    def __init__(self, height=13, width=21):
        super().__init__()
        self.height = height
        self.width = width
        self.hidden_dim = height * width
        logging.info(f"Creating S4 layer with grid {height}x{width} ({self.hidden_dim} hidden units)")

        self.A = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)

        A_mask, B_mask, C_mask, peripheral_indices = self._initialize_nearest_neighbor_constraints()

        self.register_buffer("A_mask", A_mask)
        self.register_buffer("B_mask", B_mask)
        self.register_buffer("C_mask", C_mask)
        self.register_buffer("peripheral_indices", peripheral_indices)
        logging.info(f"Peripheral indices count: {peripheral_indices.shape[0]}")

    def _initialize_nearest_neighbor_constraints(self):
        logging.info("Initializing nearest-neighbor constraints...")
        A_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
        B_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
        C_mask = torch.zeros(self.hidden_dim, self.hidden_dim)
        peripheral_indices = []

        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                neighbors = [
                    (i - 1) * self.width + j if i > 0 else None,
                    (i + 1) * self.width + j if i < self.height - 1 else None,
                    i * self.width + (j - 1) if j > 0 else None,
                    i * self.width + (j + 1) if j < self.width - 1 else None
                ]
                neighbors = [n for n in neighbors if n is not None]
                A_mask[idx, neighbors] = 1
                B_mask[idx, idx] = 1
                if i in {0, self.height - 1} or j in {0, self.width - 1}:
                    C_mask[idx, idx] = 1
                    peripheral_indices.append(idx)

        logging.info(f"Mask init complete. Peripheral count: {len(peripheral_indices)}")
        return A_mask, B_mask, C_mask, torch.tensor(peripheral_indices)

    def forward(self, x):
        logging.debug(f"S4Layer forward input shape: {x.shape}")
        masked_A = self.A * self.A_mask
        masked_B = self.B * self.B_mask

        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            u_t = x[:, t, :]
            state = torch.tanh(state @ masked_A.T + u_t @ masked_B.T)
            if t == 0 or t == seq_len - 1:
                logging.debug(f"Step {t}: state.mean() = {state.mean().item():.4f}")

        y_t = state[:, self.peripheral_indices]
        logging.debug(f"Output shape: {y_t.shape}")
        return y_t

    def save_matrices(self, output_dir="results/matrices"):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.A.detach().cpu(), os.path.join(output_dir, "A_matrix.pt"))
        torch.save(self.B.detach().cpu(), os.path.join(output_dir, "B_matrix.pt"))
        torch.save(self.C.detach().cpu(), os.path.join(output_dir, "C_matrix.pt"))
        logging.info(f"Matrices saved to {output_dir}")

class S4PredictionNetwork(nn.Module):
    def __init__(self, hidden_output_dim, final_output_dim, height=13, width=21):
        super().__init__()
        logging.info(f"Initializing S4PredictionNetwork with hidden_output_dim={hidden_output_dim}, final_output_dim={final_output_dim}")
        self.s4_layer = ConstrainedS4Layer(height, width)
        s4_output_dim = self.s4_layer.peripheral_indices.shape[0]

        self.fc1 = nn.Linear(s4_output_dim, hidden_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_output_dim, final_output_dim)

    def forward(self, x):
        logging.debug(f"PredictionNetwork input shape: {x.shape}")
        s4_output = self.s4_layer(x)
        hidden = self.relu(self.fc1(s4_output))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        logging.debug(f"Prediction output shape: {output.shape}")
        return output
