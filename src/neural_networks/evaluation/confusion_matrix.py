from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    labels,
    predictions,
    class_names,
    model_type,
):
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, predictions,)

    plt.figure(figsize=(8, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
      output_dir / f"{model_type}_confusion_matrix.png",
      bbox_inches="tight"
    )

    plt.close()