"""Shared constants for the MLOps platform.

Contains values that are not environment-specific and are referenced across
multiple modules — centralised here to eliminate duplication.
"""

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
