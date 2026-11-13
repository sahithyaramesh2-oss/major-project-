"""Presentation-attack detection metrics with spoof-score orientation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


def equal_error_rate(y_true: np.ndarray, spoof_scores: np.ndarray) -> float:
    """Interpolate threshold-swept EER; scores must increase toward spoof."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(spoof_scores, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        return float("nan")
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    apcer, bpcer = [], []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        apcer.append(float(np.mean(predictions[y == 1] == 0)))
        bpcer.append(float(np.mean(predictions[y == 0] == 1)))
    differences = np.asarray(apcer) - np.asarray(bpcer)
    crossings = np.flatnonzero(differences[:-1] * differences[1:] <= 0)
    if not len(crossings):
        return float(np.mean([apcer[int(np.argmin(np.abs(differences)))], bpcer[int(np.argmin(np.abs(differences)))]]))
    index = int(crossings[0])
    if differences[index] == 0:
        return float(apcer[index])
    if differences[index + 1] == 0:
        return float(apcer[index + 1])
    fraction = abs(differences[index]) / (abs(differences[index]) + abs(differences[index + 1]))
    return float(apcer[index] + fraction * (apcer[index + 1] - apcer[index]))


def compute_metrics(y_true: np.ndarray, spoof_scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute test metrics with class 0 genuine and class 1 spoof."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(spoof_scores, dtype=float)
    prediction = (scores >= threshold).astype(int)
    genuine = y == 0
    attacks = y == 1
    apcer = float(np.mean(prediction[attacks] == 0)) if attacks.any() else float("nan")
    bpcer = float(np.mean(prediction[genuine] == 1)) if genuine.any() else float("nan")
    both_classes = genuine.any() and attacks.any()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "apcer": apcer,
        "bpcer": bpcer,
        "far": apcer,
        "frr": bpcer,
        "acer": (apcer + bpcer) / 2 if both_classes else float("nan"),
        "eer": equal_error_rate(y, scores),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else float("nan"),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def validation_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose an ACER-minimizing threshold using validation scores only."""
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        return 0.5
    candidates = np.r_[np.min(values) - 1e-9, np.unique(values), np.max(values) + 1e-9]
    best_threshold = 0.5
    best_acer = float("inf")
    for candidate in candidates:
        predictions = values >= candidate
        apcer = np.mean(predictions[y == 0])
        bpcer = np.mean(~predictions[y == 1])
        acer = (apcer + bpcer) / 2
        if acer < best_acer:
            best_acer, best_threshold = float(acer), float(candidate)
    return best_threshold


def roc_points(y_true: np.ndarray, scores: np.ndarray):
    y = np.asarray(y_true, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        return None
    false_accept, true_accept, thresholds = roc_curve(y, scores)
    return false_accept, true_accept, thresholds