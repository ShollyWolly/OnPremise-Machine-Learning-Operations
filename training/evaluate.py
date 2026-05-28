"""
Model Evaluation Helpers
========================

Used by train.py and the retraining DAG to assess model quality
before promoting to Production.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)

PROMOTION_THRESHOLDS = {
    "roc_auc": float(os.getenv("MIN_ROC_AUC", 0.51)),
    "f1_score": 0.35,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute full classification metrics dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_score": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_score": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def passes_promotion_gate(metrics: dict, baseline_metrics: dict | None = None) -> tuple[bool, str]:
    """
    Determine if a candidate model should be promoted to Production.

    Rules:
    1. Must meet absolute thresholds in PROMOTION_THRESHOLDS
    2. If baseline_metrics provided, must not be worse by more than 2% on ROC-AUC
    """
    reasons = []

    for metric, threshold in PROMOTION_THRESHOLDS.items():
        if metrics.get(metric, 0) < threshold:
            reasons.append(
                f"{metric}={metrics[metric]:.4f} < required {threshold}"
            )

    if baseline_metrics:
        delta = metrics.get("roc_auc", 0) - baseline_metrics.get("roc_auc", 0)
        if delta < -0.02:
            reasons.append(
                f"Regression vs production: roc_auc dropped by {abs(delta):.4f}"
            )

    passed = not bool(reasons)
    reason_str = "All gates passed" if passed else "; ".join(reasons)

    log.info("Promotion gate: passed=%s - %s", passed, reason_str)
    return passed, reason_str
