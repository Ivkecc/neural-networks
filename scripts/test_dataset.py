from neural_networks.config.config import DATASET_ROOT
from neural_networks.datasets.har_dataset import HARDataset

dataset = HARDataset(
    root=DATASET_ROOT,
    split="train",
)

print(len(dataset))

X, y = dataset[0]

print(X.shape)
print(y)
