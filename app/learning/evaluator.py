import numpy as np

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_predictions(
    predictions,
    labels,
):

    predicted_ids = np.argmax(
        predictions,
        axis=-1,
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predicted_ids,
        labels,
    ):

        current_predictions = []
        current_labels = []

        for pred, true in zip(
            prediction,
            label,
        ):

            if true == -100:
                continue

            current_predictions.append(
                str(pred)
            )

            current_labels.append(
                str(true)
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    return {
        "precision": precision_score(
            true_labels,
            true_predictions,
        ),
        "recall": recall_score(
            true_labels,
            true_predictions,
        ),
        "f1": f1_score(
            true_labels,
            true_predictions,
        ),
    }