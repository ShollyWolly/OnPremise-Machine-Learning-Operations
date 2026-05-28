"""Tests for training/evaluate.py - promotion gate and metric computation."""

import numpy as np
import pytest

from evaluate import PROMOTION_THRESHOLDS, compute_metrics, passes_promotion_gate


class TestComputeMetrics:
    def test_returns_all_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in result
        assert "f1_score" in result
        assert "precision_score" in result
        assert "recall_score" in result
        assert "roc_auc" in result

    def test_perfect_predictions_all_ones(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result["accuracy"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["roc_auc"] == 1.0

    def test_all_values_are_floats(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.6, 0.4])
        result = compute_metrics(y_true, y_pred, y_prob)
        for v in result.values():
            assert isinstance(v, float)

    def test_wrong_predictions_low_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result["f1_score"] == 0.0
        assert result["accuracy"] == 0.0


class TestPassesPromotionGate:
    def _good_metrics(self):
        return {"roc_auc": 0.85, "f1_score": 0.70, "accuracy": 0.82}

    def _bad_roc_metrics(self):
        return {"roc_auc": 0.40, "f1_score": 0.70, "accuracy": 0.75}

    def _bad_f1_metrics(self):
        return {"roc_auc": 0.80, "f1_score": 0.20, "accuracy": 0.75}

    def test_good_model_passes(self):
        passed, reason = passes_promotion_gate(self._good_metrics())
        assert passed
        assert "passed" in reason.lower()

    def test_returns_tuple_of_bool_and_str(self):
        result = passes_promotion_gate(self._good_metrics())
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_low_roc_auc_fails(self):
        passed, reason = passes_promotion_gate(self._bad_roc_metrics())
        assert not passed
        assert "roc_auc" in reason

    def test_low_f1_fails(self):
        passed, reason = passes_promotion_gate(self._bad_f1_metrics())
        assert not passed
        assert "f1_score" in reason

    def test_multiple_failures_reported_together(self):
        bad = {"roc_auc": 0.40, "f1_score": 0.10}
        passed, reason = passes_promotion_gate(bad)
        assert not passed
        # Both failures should appear in reason
        assert "roc_auc" in reason
        assert "f1_score" in reason

    def test_regression_vs_baseline_fails_when_drop_large(self):
        candidate = {"roc_auc": 0.70, "f1_score": 0.60}
        baseline = {"roc_auc": 0.75}
        passed, reason = passes_promotion_gate(candidate, baseline_metrics=baseline)
        assert not passed
        assert "Regression" in reason

    def test_small_drop_vs_baseline_passes(self):
        candidate = {"roc_auc": 0.74, "f1_score": 0.60}
        baseline = {"roc_auc": 0.75}
        passed, _ = passes_promotion_gate(candidate, baseline_metrics=baseline)
        assert passed

    def test_improvement_over_baseline_passes(self):
        candidate = {"roc_auc": 0.80, "f1_score": 0.65}
        baseline = {"roc_auc": 0.75}
        passed, _ = passes_promotion_gate(candidate, baseline_metrics=baseline)
        assert passed

    def test_no_baseline_only_absolute_thresholds_apply(self):
        good = {"roc_auc": 0.80, "f1_score": 0.60}
        passed, _ = passes_promotion_gate(good, baseline_metrics=None)
        assert passed

    def test_thresholds_dict_has_expected_keys(self):
        assert "roc_auc" in PROMOTION_THRESHOLDS
        assert "f1_score" in PROMOTION_THRESHOLDS
