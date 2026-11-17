"""Score seven activities with a separate column for unavailable predictions."""

import numpy as np


def confusion(y, prediction):
    """Column 7 means unavailable; it is never an eighth true activity."""
    y, prediction = np.asarray(y, int), np.asarray(prediction, int)
    if y.shape != prediction.shape or not len(y) or not ((y >= 0) & (y < 7)).all():
        raise ValueError("Invalid true labels")
    if not ((prediction >= 0) & (prediction <= 7)).all():
        raise ValueError("Invalid predictions")
    matrix = np.zeros((7, 8), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    return matrix


def metrics(y, prediction):
    matrix = confusion(y, prediction)
    correct = np.diag(matrix[:, :7])
    support, predicted = matrix.sum(1), matrix[:, :7].sum(0)
    precision = np.divide(correct, predicted, out=np.zeros(7), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(7), where=support > 0)
    f1 = np.divide(
        2 * correct,
        support + predicted,
        out=np.zeros(7),
        where=(support + predicted) > 0,
    )
    return {
        "accuracy": float(correct.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion": matrix,
    }
