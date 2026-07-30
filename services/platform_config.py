"""Shared constants for the MLOps platform.

Contains values that are not environment-specific and are referenced across
multiple modules — centralised here to eliminate duplication.
"""

import uuid

FEATURE_COLUMNS: list[str] = [
    "age",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "employment_length_years",
    "home_ownership_encoded",
    "debt_to_income_ratio",
    "num_credit_lines",
    "payment_history_score",
]

TARGET_COLUMN = "default_flag"

# MLflow registry / experiment names
MODEL_REGISTRY_NAME_DEFAULT = "credit-risk-classifier"
TRAINING_EXPERIMENT = "credit-risk-training"
MONITORING_EXPERIMENT_HARD = "monitoring_hard"
MONITORING_EXPERIMENT_DRIFT = "monitoring_drift"
MONITORING_EXPERIMENT_SHAP = "monitoring_shap"

# Decision / quality-gate defaults (override via env vars in production)
DECISION_THRESHOLD_DEFAULT: float = 0.5
MIN_ROC_AUC_DEFAULT: float = 0.70

MAX_INFERENCE_BATCH_SIZE: int = 1000

# Evidently self-hosted UI (docker-compose service 'evidently-ui', port 8000).
# Project ids are derived deterministically from a fixed name so every monitoring
# run resolves to the same project without needing to persist the id anywhere.
EVIDENTLY_UUID_NAMESPACE = uuid.UUID("c9a1f9d4-6f1e-4b8a-9e5a-3f6a2b1c7d8e")
EVIDENTLY_DRIFT_PROJECT_NAME = "credit-risk-data-drift"
EVIDENTLY_HARD_PROJECT_NAME = "credit-risk-hard-metrics"
EVIDENTLY_DRIFT_PROJECT_ID = uuid.uuid5(EVIDENTLY_UUID_NAMESPACE, EVIDENTLY_DRIFT_PROJECT_NAME)
EVIDENTLY_HARD_PROJECT_ID = uuid.uuid5(EVIDENTLY_UUID_NAMESPACE, EVIDENTLY_HARD_PROJECT_NAME)
