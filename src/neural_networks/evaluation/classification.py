from sklearn.metrics import classification_report


def print_classification_report(
    labels,
    predictions,
    class_names,
):
    print("\nClassification Report")

    print(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            digits=4,
        )
    )