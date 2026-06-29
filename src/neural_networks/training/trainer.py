import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from neural_networks.config.config import EPOCHS, LEARNING_RATE, CLASS_NAMES
from neural_networks.data.dataloaders import create_dataloaders
from neural_networks.models.dense import DenseNN
from neural_networks.visualization.plots import plot_training_history
from neural_networks.training.checkpoint import save_checkpoint
from neural_networks.training.early_stopping import EarlyStopping
from neural_networks.evaluation.predictions import collect_predictions
from neural_networks.evaluation.confusion_matrix import plot_confusion_matrix


def train_one_epoch(
        model: nn.Module, 
        train_loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer):
    
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for X, y in train_loader:
        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)
        
        correct_predictions += (predictions == y).sum().item()

        total_samples += y.size(0)

    average_loss = total_loss / len(train_loader)

    accuracy = correct_predictions / total_samples
    
    return average_loss, accuracy


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():

        for X, y in test_loader:
            logits = model(X)

            loss = criterion(logits, y)

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct_predictions += (predictions == y).sum().item()

            total_samples += y.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def train():
    train_loader, test_loader = create_dataloaders()

    model = DenseNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_dir = Path("checkpoints")

    best_model_path = checkpoint_dir / "dense_best.pt"

    early_stopper = EarlyStopping(patience=5)  

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer)

        test_loss, test_accuracy = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        should_stop = early_stopper.step(test_loss)

        if early_stopper.best_loss == test_loss:
            save_checkpoint(model, best_model_path)
            print(f"Saved best model to {best_model_path}")

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f} | "
        )

        if should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    plot_training_history(
        train_losses=train_losses,
        test_losses=test_losses,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
    )

    predictions, labels = collect_predictions(
        model=model,
        data_loader=test_loader,
    )

    plot_confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_names=CLASS_NAMES
    )


if __name__ == "__main__":
    train()
