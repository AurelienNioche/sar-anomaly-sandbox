import numpy as np
from sklearn.metrics import f1_score, roc_curve


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the ROC-curve threshold that maximises the F1 score."""
    _, _, thresholds = roc_curve(y_true, y_score)
    best_thresh, best_f1 = float(thresholds[0]), 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh
