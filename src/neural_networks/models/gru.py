#!/usr/bin/env python3

import torch
from torch import nn

from neural_networks.config.config import INPUT_CHANNELS, NUM_CLASSES


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_CHANNELS,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(x)

        # bidirectional=True:
        # hidden[-2] = forward smjer
        # hidden[-1] = backward smjer
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        logits = self.classifier(last_hidden)
        return logits