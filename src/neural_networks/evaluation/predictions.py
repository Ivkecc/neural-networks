import torch

from torch import nn
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            logits = model(X)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions)
            all_labels.append(y)
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    return all_predictions, all_labels
    