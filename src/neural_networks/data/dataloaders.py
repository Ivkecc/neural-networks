from torch.utils.data import DataLoader

from neural_networks.config.config import BATCH_SIZE, DATASET_ROOT
from neural_networks.datasets.har_dataset import HARDataset


def create_dataloaders(batch_size: int = BATCH_SIZE):
    train_dataset = HARDataset(
        root=DATASET_ROOT,
        split="train",
    )

    test_dataset = HARDataset(
        root=DATASET_ROOT,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
