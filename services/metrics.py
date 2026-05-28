"""
Shared metric catalogue for the MLOps platform.

All metrics take (y_true, y_score) where:
  y_true  - ground truth labels (0/1 int array)
  y_score - predicted probabilities [0, 1] (continuous float array)
  y_pred  - optional pre-thresholded predictions; computed from y_score+threshold if None

Adding a new metric:
  1. Add key → display name to REGISTRY
  2. Add computation in compute_all_metrics()
  3. That's it - DAG comparison, monitoring, and UI pick it up automatically.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ordered registry: key → human-readable display name
REGISTRY: dict[str, str] = {
    "roc_auc":   "ROC-AUC",
    "pr_auc":    "PR-AUC",
    "f1":        "F1",
    "precision": "Precision",
    "recall":    "Recall",
    "accuracy":  "Accuracy",
}

METRIC_KEYS = list(REGISTRY.keys())

DEFAULT_PRIMARY: str = os.getenv("DEFAULT_PRIMARY_METRIC", "roc_auc")


def display_name(key: str) -> str:
    """Return human-readable label for a metric key."""
    return REGISTRY.get(key, key.upper())


def compute_all_metrics(
    y_true,
    y_score,
    y_pred=None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute all registered metrics.

    Parameters
    ----------
    y_true    : Ground truth labels (0/1).
    y_score   : Predicted probabilities (0–1).
    y_pred    : Pre-thresholded predictions. If None, derived from y_score >= threshold.
    threshold : Decision threshold applied when y_pred is None (default 0.5).

    Returns
    -------
    dict mapping metric key → rounded float value.
    """
    y_true  = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two classes to compute metrics.")

    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_score)), 6),
        "pr_auc":    round(float(average_precision_score(y_true, y_score)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
    }


def primary_value(metrics: dict[str, float], primary_metric: str = DEFAULT_PRIMARY) -> float:
    """Extract the value of the primary metric from a metrics dict."""
    return metrics.get(primary_metric, metrics.get(DEFAULT_PRIMARY, 0.0))
