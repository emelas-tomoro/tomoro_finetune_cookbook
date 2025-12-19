from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    # [[TN, FP],
    #  [FN, TP]]
    confusion_matrix: list[list[int]]


def compute_binary_metrics(*, y_true: list[bool], y_pred: list[bool]) -> BinaryClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")

    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    cm = confusion_matrix(yt, yp).tolist()
    return BinaryClassificationMetrics(
        accuracy=float(accuracy_score(yt, yp)),
        precision=float(precision_score(yt, yp)),
        recall=float(recall_score(yt, yp)),
        f1=float(f1_score(yt, yp)),
        confusion_matrix=[[int(x) for x in row] for row in cm],
    )


