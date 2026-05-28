"""
Distribution parameters for synthetic loan application data generation.

STABLE_PARAMS defines the baseline (no drift) distributions.
Each parameter can be shifted by drift_factor in generator.py.
"""

STABLE_PARAMS = {
    "age": {
        "dist": "normal",
        "mean": 42.0,
        "std": 12.0,
        "clip_min": 18,
        "clip_max": 75,
        "dtype": "int",
    },
    "annual_income": {
        "dist": "lognormal",
        "mean": 11.0,   # log-space mean → e^11 ≈ 60k
        "std": 0.5,
        "clip_min": 15_000,
        "clip_max": 500_000,
        "dtype": "float",
    },
    "credit_score": {
        "dist": "normal",
        "mean": 680.0,
        "std": 50.0,
        "clip_min": 300,
        "clip_max": 850,
        "dtype": "int",
    },
    "loan_amount": {
        "dist": "lognormal",
        "mean": 10.0,   # log-space mean → e^10 ≈ 22k
        "std": 0.6,
        "clip_min": 1_000,
        "clip_max": 200_000,
        "dtype": "float",
    },
    "loan_term_months": {
        "dist": "choice",
        "choices": [12, 24, 36, 48, 60, 72, 84],
        "weights": [0.05, 0.15, 0.30, 0.20, 0.20, 0.07, 0.03],
    },
    "employment_length_years": {
        "dist": "gamma",
        "shape": 2.0,
        "scale": 3.0,
        "clip_min": 0.0,
        "clip_max": 40.0,
        "dtype": "float",
    },
    "home_ownership": {
        "dist": "choice",
        "choices": ["RENT", "MORTGAGE", "OWN", "OTHER"],
        "weights": [0.40, 0.35, 0.20, 0.05],
    },
    "num_credit_lines": {
        "dist": "poisson",
        "lam": 8.0,
        "clip_min": 1,
        "clip_max": 40,
        "dtype": "int",
    },
    "payment_history_score": {
        "dist": "beta",
        "a": 5.0,
        "b": 2.0,
        "scale": 100.0,   # output = beta_sample * scale
        "clip_min": 0.0,
        "clip_max": 100.0,
        "dtype": "float",
    },
}

# How drift_factor modifies distributions.
# drift_factor=0.0 → stable params. drift_factor=1.0 → maximum shift.
DRIFT_DELTAS = {
    # Lower credit scores: mean drops from 680 → 580 at max drift
    "credit_score": {"mean_delta": -100.0, "std_delta": 20.0},
    # Lower income: log-mean drops slightly (income falls ~18% at max drift)
    "annual_income": {"mean_delta": -0.2, "std_delta": 0.15},
    # More debt: loan amounts increase
    "loan_amount": {"mean_delta": 0.3, "std_delta": 0.2},
    # Shorter employment (riskier applicants)
    "employment_length_years": {"shape_delta": -0.5, "scale_delta": -0.5},
    # Worse payment history
    "payment_history_score": {"a_delta": -2.0},
    # Shift tenure mix toward RENT (less stable)
    "home_ownership": {
        "weights_drift": [0.55, 0.25, 0.15, 0.05]  # at max drift
    },
}

# Logistic regression coefficients used to derive default_flag from features.
# These are fixed (stable across modes) — only covariate drift, not concept drift.
# Coefficients are intentionally strong so the synthetic data produces clear
# class separation (theoretical AUC ~0.85+), ensuring the model comfortably
# exceeds the promotion threshold (MIN_ROC_AUC) on every stable inference run.
DEFAULT_COEFFICIENTS = {
    "intercept": 13.2,            # tuned so average applicant has ~20% default rate
    "credit_score": -0.015,       # higher score → lower default risk
    "annual_income": -0.000009,   # higher income → lower default risk
    "loan_amount": 0.000009,      # higher loan → higher default risk
    "debt_to_income_ratio": 2.4,  # high DTI → higher default risk
    "payment_history_score": -0.06,
    "employment_length_years": -0.15,
}
