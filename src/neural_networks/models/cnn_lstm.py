#!/usr/bin/env python3

import torch
from torch import nn

from neural_networks.config.config import INPUT_CHANNELS, NUM_CLASSES


class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, 128, 6] -> [batch, 6, 128]
        x = x.permute(0, 2, 1)

        # [batch, 6, 128] -> [batch, 128, 64]
        x = self.cnn(x)

        # [batch, 128, 64] -> [batch, 64, 128]
        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)

        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        logits = self.classifier(last_hidden)
        return logits