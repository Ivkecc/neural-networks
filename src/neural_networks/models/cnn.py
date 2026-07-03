#!/usr/bin/env python3

import torch
from torch import nn

from neural_networks.config.config import (
    INPUT_CHANNELS,
    NUM_CLASSES,
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=INPUT_CHANNELS,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ulaz dolazi kao [batch, 128, 6]
        # Conv1d očekuje [batch, kanali, dužina] = [batch, 6, 128]
        x = x.permute(0, 2, 1)

        x = self.features(x)
        logits = self.classifier(x)

        return logits