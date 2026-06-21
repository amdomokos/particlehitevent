# Models/visualize_matrices.py
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_matrix(matrix, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_and_plot_matrices(model, output_dir="results/matrices"):
    os.makedirs(output_dir, exist_ok=True)

    # Extract the S4 layer
    s4 = model.s4_layer

    # Detach and mask
    A = (s4.A * s4.A_mask).detach().cpu().numpy()
    B = (s4.B * s4.B_mask).detach().cpu().numpy()
    C = (s4.C * s4.C_mask).detach().cpu().numpy()

    # Save raw matrices
    torch.save(s4.A.detach().cpu(), os.path.join(output_dir, "A_matrix.pt"))
    torch.save(s4.B.detach().cpu(), os.path.join(output_dir, "B_matrix.pt"))
    torch.save(s4.C.detach().cpu(), os.path.join(output_dir, "C_matrix.pt"))

    # Save plots
    plot_matrix(A, "A Matrix (Masked)", os.path.join(output_dir, "A_matrix.png"))
    plot_matrix(B, "B Matrix (Masked)", os.path.join(output_dir, "B_matrix.png"))
    plot_matrix(C, "C Matrix (Masked)", os.path.join(output_dir, "C_matrix.png"))
