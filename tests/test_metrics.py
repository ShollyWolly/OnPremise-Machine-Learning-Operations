"""Tests for services/metrics.py - shared metric catalogue."""

import numpy as np
import pytest

from metrics import (
    DEFAULT_PRIMARY,
    METRIC_KEYS,
    REGISTRY,
    compute_all_metrics,
    display_name,
    primary_value,
)


class TestRegistry:
    def test_registry_has_six_metrics(self):
        assert len(REGISTRY) == 6

    def test_all_expected_keys_present(self):
        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"):
            assert key in REGISTRY

    def test_metric_keys_matches_registry(self):
        assert set(METRIC_KEYS) == set(REGISTRY.keys())

    def test_display_name_known_key(self):
        assert display_name("roc_auc") == "ROC-AUC"
        assert display_name("f1") == "F1"
        assert display_name("accuracy") == "Accuracy"

    def test_display_name_unknown_key_uppercases(self):
        assert display_name("custom_metric") == "CUSTOM_METRIC"


class TestComputeAllMetrics:
    def test_returns_all_six_keys(self, perfect_binary_labels):
        y_true, y_score = perfect_binary_labels
        result = compute_all_metrics(y_true, y_score)
        assert set(result.keys()) == set(METRIC_KEYS)

    def test_perfect_model_roc_auc_is_one(self, perfect_binary_labels):
        y_true, y_score = perfect_binary_labels
        result = compute_all_metrics(y_true, y_score)
        assert result["roc_auc"] == 1.0

    def test_perfect_model_accuracy_is_one(self, perfect_binary_labels):
        y_true, y_score = perfect_binary_labels
        result = compute_all_metrics(y_true, y_score)
        assert result["accuracy"] == 1.0

    def test_metrics_are_bounded_0_to_1(self, balanced_labels):
        y_true, y_score = balanced_labels
        result = compute_all_metrics(y_true, y_score)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_all_zeros_prediction_yields_zero_recall(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_score = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        result = compute_all_metrics(y_true, y_score, threshold=0.5)
        assert result["recall"] == 0.0

    def test_custom_threshold_changes_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.4, 0.6, 0.7])
        result_default = compute_all_metrics(y_true, y_score, threshold=0.5)
        result_low = compute_all_metrics(y_true, y_score, threshold=0.2)
        # at threshold=0.2 all positives are caught (recall=1) but precision drops
        assert result_low["recall"] == 1.0
        assert result_low["precision"] < 1.0
        assert result_default["f1"] != result_low["f1"]

    def test_pre_supplied_y_pred_used_directly(self):
        y_true = np.array([0, 1, 1])
        y_score = np.array([0.6, 0.6, 0.6])
        y_pred = np.array([0, 1, 0])
        result = compute_all_metrics(y_true, y_score, y_pred=y_pred)
        # recall should be 0.5 (1 of 2 positives caught)
        assert result["recall"] == pytest.approx(0.5)

    def test_values_are_rounded_to_six_decimals(self, balanced_labels):
        y_true, y_score = balanced_labels
        result = compute_all_metrics(y_true, y_score)
        for val in result.values():
            assert val == round(val, 6)

    def test_accepts_list_input(self):
        result = compute_all_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        assert result["roc_auc"] == 1.0

    def test_single_class_precision_recall_zero_division_safe(self):
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        # roc_auc undefined for single class - sklearn raises, so we expect ValueError
        with pytest.raises(ValueError):
            compute_all_metrics(y_true, y_score)


class TestPrimaryValue:
    def test_extracts_named_metric(self):
        metrics = {"roc_auc": 0.85, "f1": 0.70}
        assert primary_value(metrics, "f1") == 0.70

    def test_falls_back_to_default_primary(self):
        metrics = {"roc_auc": 0.85, "f1": 0.70}
        # when primary_metric key not present, falls back to DEFAULT_PRIMARY
        val = primary_value(metrics, "nonexistent_key")
        assert val == metrics.get(DEFAULT_PRIMARY, 0.0)

    def test_returns_zero_when_both_missing(self):
        assert primary_value({}, "missing") == 0.0
