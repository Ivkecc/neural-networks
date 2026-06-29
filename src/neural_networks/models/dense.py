#!/usr/bin/env python3

import torch
from torch import nn

from neural_networks.config.config import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    NUM_CLASSES,
)


class DenseNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(INPUT_CHANNELS * INPUT_LENGTH, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.network(x)
        return logits
