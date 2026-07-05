from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HARDataset(Dataset):
    def __init__(self, root: str, split: str):
        super().__init__()

        self.root = Path(root)
        self.split = split

        split_path = self.root / self.split
        signals_path = split_path / "Inertial Signals"
        label_file = split_path / f"y_{self.split}.txt"

        channels = (
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
        )

        sensor_data = []

        for channel in channels:
            filename = f"{channel}_{self.split}.txt"
            file_path = signals_path / filename

            data = np.loadtxt(file_path)
            data = torch.from_numpy(data).float()

            sensor_data.append(data)

        self.data = torch.stack(sensor_data).permute(1, 2, 0)

        labels = np.loadtxt(label_file)
        self.labels = torch.from_numpy(labels).long() - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
