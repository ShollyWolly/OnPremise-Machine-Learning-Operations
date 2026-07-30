# Monitoring Guide

## Overview

Three independent monitoring DAGs run against any completed batch inference run. They can be triggered in any order, at any time, against any `run_index`.

| DAG | Module | Trigger |
|---|---|---|
| `dag_04a_monitor_hard` | `services/monitoring/hard_metrics.py` | Auto after dag_03; or manual |
| `dag_04b_monitor_drift` | `services/monitoring/data_drift.py` | Manual |
| `dag_04c_monitor_shap` | `services/monitoring/shap_explainability.py` | Manual |

---

## 1. Hard Metrics (dag_04a)

### Metrics Computed

`accuracy`, `precision`, `recall`, `f1`, and `roc_auc` come from Evidently's `ClassificationPreset`
(`services/monitoring/hard_metrics.py::compute_metrics`), which also generates the confusion matrix,
ROC curve, and PR curve viewable in the Evidently UI. `pr_auc` is kept from `services/metrics.py`
(the single source of truth for the metric registry) since Evidently 0.4.x doesn't expose a scalar
PR-AUC directly.

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

- `dwh_monitoring_hard.results`, one row per `run_index`, unique constraint (includes `evidently_snapshot_id`)
- MLflow experiment `monitoring_hard`, all 6 metrics as MLflow metrics
- `/data/monitoring/hard/run_NNNNN.parquet`
- Evidently Workspace snapshot (project `credit-risk-hard-metrics`), viewable at http://localhost:8000

---

## 2. Data Drift Detection (dag_04b)

### Methodology

Evidently's `DataDriftPreset` applied to the 10 model input features (`services/monitoring/data_drift.py::compute_drift`).

- **Reference dataset**: `reference_data.parquet` stored in the production model's MLflow artifact store, logged at training time by `train.py`
- **Current dataset**: feature values from `dwh_history.prediction_ground_truth` for the specified `run_index`
- **Per-feature test**: Evidently auto-selects a statistical test per column (KS test for continuous numeric features by default)
- **Dataset-level cutoff**: `DataDriftPreset(drift_share=MAX_DRIFT_FEATURE_FRACTION)` — dataset is flagged as drifted once this share of columns drift

### Drift Score

`drift_score` = fraction of features with statistically significant drift (Evidently's `share_of_drifted_columns`).

| Score Range | Interpretation |
|---|---|
| 0.00 – 0.10 | No drift, monitor normally |
| 0.10 – 0.20 | Mild drift, watch closely |
| 0.20 – 0.40 | Moderate drift, consider retraining |
| > 0.40 | Severe drift, investigate data source |

Threshold for flagging: `drift_score >= MAX_DRIFT_FEATURE_FRACTION` (env var default: 0.30). `dag_04b` now cascades to `dag_05_retraining` when this fires, the same way `dag_04a` does for hard metrics.

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
  - `evidently_snapshot_id` VARCHAR(36) — id of the report snapshot pushed to the Evidently UI
- MLflow experiment `monitoring_drift`
- `/data/monitoring/drift/run_NNNNN.parquet`
- Evidently Workspace snapshot (project `credit-risk-data-drift`), viewable at http://localhost:8000

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

- **Hard Metrics** tab: metric cards (★ = primary metric), trend chart across all runs, embedded Evidently classification report (confusion matrix, ROC/PR curves, quality metrics)
- **Data Drift** tab: summary cards, dataset statistics, drift score trend, embedded Evidently drift report (per-feature distribution comparison)
- **Explainability** tab: beeswarm plot, mean |SHAP| bar, customer waterfall selector

### Evidently UI (http://localhost:8000)

Self-hosted Evidently report viewer (docker-compose service `evidently-ui`), fed by a shared
Workspace directory (`/data/monitoring/evidently_workspace`) that `hard_metrics.py` and
`data_drift.py` push snapshots to on every run. Two projects: `credit-risk-data-drift` and
`credit-risk-hard-metrics`. The dashboard embeds the exact run's report via
`/projects/{project_id}/reports/{snapshot_id}`; the same URL works standalone in a browser.

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
