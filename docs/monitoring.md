# Monitoring Guide

## Overview

Three independent monitoring DAGs run against any completed batch inference run. They can be triggered in any order, at any time, against any `run_index`.

| DAG | Module | Trigger |
|---|---|---|
| `dag_04a_monitor_hard` | `services/monitoring/hard_metrics.py` | Auto after dag_03; or manual |
| `dag_04b_monitor_drift` | `services/monitoring/drift.py` | Manual |
| `dag_04c_monitor_shap` | `services/monitoring/shap_monitor.py` | Manual |

---

## 1. Hard Metrics (dag_04a)

### Metrics Computed

All six metrics are computed from `services/metrics.py`, the single source of truth shared across DAGs, monitoring scripts, and UIs.

| Key | Display Name | Notes |
|---|---|---|
| `roc_auc` | ROC-AUC | Ranking quality; insensitive to class imbalance |
| `pr_auc` | PR-AUC | Area under precision-recall curve; better for rare positive class |
| `f1` | F1 | Harmonic mean of precision and recall |
| `precision` | Precision | TP / (TP + FP) |
| `recall` | Recall | TP / (TP + FN) |
| `accuracy` | Accuracy | Overall correct prediction rate |

### Primary Metric

The **primary metric** is the single metric used for the retraining threshold check. It is read from the `primary_metric` tag on the current production model version in MLflow. Falls back to the `DEFAULT_PRIMARY_METRIC` environment variable (`roc_auc` by default).

Retraining triggers when: `primary_metric_score < MIN_ROC_AUC` (env var default: 0.60)

### Ground Truth Timing

The data generator sets `created_at = NOW() - 1 day`. Ground truth (`actual_default_flag`) is populated from `dwh_clean.cleaned_features` via a COALESCE join, since data is synthetic, it is immediately available. In a production deployment this would reflect a real outcome delay (e.g. loan repayment status known 30+ days later).

### Storage

- `dwh_monitoring_hard.results`, one row per `run_index`, unique constraint
- MLflow experiment `monitoring_hard`, all 6 metrics as MLflow metrics
- `/data/monitoring/hard/run_NNNNN.parquet`

---

## 2. Data Drift Detection (dag_04b)

### Methodology

Kolmogorov-Smirnov test applied independently to each of the 10 model input features.

- **Reference dataset**: `reference_data.parquet` stored in the production model's MLflow artifact store, logged at training time by `train.py`
- **Current dataset**: feature values from `dwh_history.prediction_ground_truth` for the specified `run_index`
- **Statistical test**: two-sample KS test per feature (p-value threshold: 0.05)

### Drift Score

`drift_score` = fraction of features with statistically significant drift.

| Score Range | Interpretation |
|---|---|
| 0.00 – 0.10 | No drift, monitor normally |
| 0.10 – 0.20 | Mild drift, watch closely |
| 0.20 – 0.40 | Moderate drift, consider retraining |
| > 0.40 | Severe drift, investigate data source |

Threshold for flagging: `drift_score > MAX_DRIFT_SCORE` (env var default: 0.20)

### What Drift Looks Like

In **drift mode** (`drift_factor=0.5`):
- `credit_score` mean drops (worse applicants entering)
- `annual_income` variance increases
- `loan_amount` shifts upward
- `payment_history_score` degrades slightly

Only covariate drift is simulated. The target relationship (what causes default) stays constant.

### Storage

- `dwh_monitoring_drift.results`, one row per `run_index`
  - `drift_detected` BOOLEAN
  - `drift_score` FLOAT (fraction of drifted features)
  - `num_drifted_features` INT
  - `drifted_feature_names` TEXT (comma-separated)
- MLflow experiment `monitoring_drift`
- `/data/monitoring/drift/run_NNNNN.parquet`

---

## 3. SHAP Explainability (dag_04c)

### What Is Computed

Per-feature SHAP values for every record in the specified batch run.

| Output | Description |
|---|---|
| `feature_importances` JSONB | Mean |SHAP| per feature, aggregate importance |
| `top_feature` TEXT | Feature with highest mean |SHAP| |
| `customer_shap_values` | Per-record: full SHAP vector + base_value + predicted_probability |
| MLflow artifact: beeswarm plot | Directional feature impact distribution |
| MLflow artifact: bar chart | Mean |SHAP| per feature |

### Explainer Selection

1. `shap.TreeExplainer`, fast, exact; used for RandomForest and XGBoost
2. `shap.KernelExplainer`, model-agnostic fallback; slower, samples background

### Interpreting SHAP Values

- **Positive SHAP** → feature pushed prediction toward default (higher probability)
- **Negative SHAP** → feature pushed prediction away from default
- **Mean |SHAP|** → overall importance regardless of direction

Expected top features by importance (simulation):
1. `debt_to_income_ratio` (positive, higher DTI = more default)
2. `payment_history_score` (negative, lower score = more default)
3. `credit_score` (negative, lower score = more default)
4. `annual_income` (negative, lower income = more default)

### Customer Waterfall

Each row in `dwh_monitoring_shap.customer_shap_values` contains a `customer_id` (deterministic UUID5, stable across runs). The Dashboard Explainability tab uses this to render a per-customer waterfall chart, select any customer_id to see exactly which features pushed their default probability up or down.

### Storage

- `dwh_monitoring_shap.results`, aggregate per run_index (`feature_importances` JSONB)
- `dwh_monitoring_shap.customer_shap_values`, per-record SHAP vectors
- MLflow experiment `monitoring_shap`, beeswarm + bar chart logged as artifacts
- `/data/monitoring/shap/run_NNNNN.parquet`

---

## Retraining Decision

Retraining is triggered in `hard_metrics.py` if:

```python
retraining_triggered = primary_metric_score < MIN_ROC_AUC
```

The primary metric is determined at runtime from the production model's `primary_metric` tag. This means a model optimised for F1 will retrain when F1 drops below the threshold, not when ROC-AUC drops.

Drift alone does **not** trigger retraining directly, it is informational. Only hard metric degradation triggers dag_05 automatically.

### Threshold Tuning

| Env Var | Default | Guidance |
|---|---|---|
| `MIN_ROC_AUC` | `0.60` | Lower → less frequent retraining. Raise if model quality is critical. |
| `MAX_DRIFT_SCORE` | `0.20` | Fraction of features that must drift before flagging. |
| `MAX_DRIFT_FEATURE_FRACTION` | `0.30` | Secondary threshold (feature fraction). |

---

## Viewing Results

### Monitoring Dashboard (http://localhost:8502)

- **Hard Metrics** tab: metric cards (★ = primary metric), ROC/PR curves, confusion matrix, threshold sensitivity, trend chart across all runs
- **Data Drift** tab: per-feature distribution overlays, KS statistics table, drift score trend
- **Explainability** tab: beeswarm plot, mean |SHAP| bar, customer waterfall selector

### MLflow UI (http://localhost:5000)

- Experiment `monitoring_hard` → metric comparisons across runs
- Experiment `monitoring_drift` → drift score history
- Experiment `monitoring_shap` → SHAP beeswarm/bar artifacts per run

### Direct SQL

```sql
-- Latest hard metric results
SELECT run_index, roc_auc, pr_auc, f1_score, primary_metric, retraining_triggered, evaluated_at
FROM dwh_monitoring_hard.results
ORDER BY run_index DESC
LIMIT 10;

-- Drift history
SELECT run_index, drift_score, drift_detected, drifted_feature_names, evaluated_at
FROM dwh_monitoring_drift.results
ORDER BY run_index DESC
LIMIT 10;

-- Retraining audit trail
SELECT retrained_at, trigger_reason, old_model_version, new_model_version, promotion_succeeded
FROM dwh_history.retraining_log
ORDER BY retrained_at DESC;

-- Top SHAP features for a run
SELECT run_index, top_feature, feature_importances
FROM dwh_monitoring_shap.results
ORDER BY run_index DESC
LIMIT 5;

-- Per-customer SHAP for a specific run
SELECT customer_id, shap_values, base_value, predicted_probability
FROM dwh_monitoring_shap.customer_shap_values
WHERE run_index = <N>
ORDER BY predicted_probability DESC
LIMIT 20;
```
