import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input: (batch, 80, 13, 21) — 80 time slices as channels, 13x21 pixel grid
        self.conv1 = nn.Conv2d(80, 64, kernel_size=3, padding=1)   # -> (64, 13, 21)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # -> (128, 6, 10) after pool
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 10, 256)
        self.fc2 = nn.Linear(256, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))   # (batch, 64, 13, 21)
        x = self.pool(x)               # (batch, 64, 6, 10)
        x = self.relu(self.conv2(x))   # (batch, 128, 6, 10)
        x = x.flatten(start_dim=1)     # (batch, 7680)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
