from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
):

    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    #---------Loss plot--------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss",
    )

    plt.plot(
        epochs,
        test_losses,
        label="Test loss",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training and Test Loss")

    plt.legend()

    plt.savefig(output_dir / "dense_loss.png")

    plt.close()

    #---------Accuracy plot--------

    plt.figure(figsize=(8,5))

    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")

    plt.legend()

    plt.savefig(output_dir / "dense_accuracy.png")
    plt.close()