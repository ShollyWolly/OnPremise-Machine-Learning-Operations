# Pipeline Documentation

## DAG Overview

```
dag_00_deploy_model          (manual — promote any model version)

dag_03_batch_inference       (manual)
       │  auto-triggers on completion
       ▼
dag_04a_monitor_hard         (event-driven + manual)
       │  conditional: primary metric below threshold
       ▼
dag_05_retraining            (event-driven + manual)
       │  post-retrain check
       ▼
dag_04a_monitor_hard         (skip_retrain_trigger=True — verify new model)

dag_04b_monitor_drift        (manual — independent of dag_04a)
dag_04c_monitor_shap         (manual — independent of dag_04a)

dag_06_challenger_comparison (manual — triggered from Control Panel Challenger tab)
```

All DAGs can also be triggered from the Airflow UI or the Control Panel.

---

## DAG 00 — Deploy Model

**File**: `airflow/dags/dag_00_deploy_model.py`  
**Schedule**: None (manual)  
**Purpose**: Promote a specific MLflow model version to the `production` alias and hot-reload Flask.

### Airflow Params

| Param | Default | Description |
|---|---|---|
| `model_name` | `credit-risk-classifier` | MLflow registered model name |
| `model_version` | `""` | Version number to promote (empty = current production) |
| `model_stage` | `production` | Target alias |

### Task Graph
```
promote_to_stage  →  reload_flask_api  →  verify_deployment
```

---

## DAG 03 — Batch Inference

**File**: `airflow/dags/dag_03_batch_inference.py`  
**Schedule**: None (manual)  
**Purpose**: Self-contained pipeline from data generation through model scoring. Registers a run_index and auto-triggers hard metric monitoring on completion.

### Airflow Params

| Param | Default | Description |
|---|---|---|
| `drift_factor` | `0.0` | Covariate shift intensity [0.0–1.0] |
| `n_records` | `1000` | Records to generate for this run |

### Task Graph
```
register_run
    ↓
generate_data    (generator.py --mode stable|drift --drift-factor N)
    ↓
process_data     (pipeline.py — clean, encode, validate)
    ↓
check_flask_health
    ↓
run_inference    (inference.py --run-index N — calls /predict, writes dwh_history)
    ↓
trigger_monitoring  (TriggerDagRunOperator → dag_04a with run_index conf)
```

### Outputs
- `dwh_history.run_registry` — one row, SERIAL `run_index`
- `dwh_history.prediction_ground_truth` — one row per record, includes all features + score + `actual_default_flag`
- `dwh_predictions.batch_predictions` — legacy copy, one row per record
- `/data/raw/*.parquet`, `/data/processed/*.parquet`, `/data/predictions/*.parquet`

### Failure Modes
- Flask unhealthy → `check_flask_health` fails; ensure a model has the `production` alias
- No Production model → promote one via Control Panel Deploy tab or dag_00

---

## DAG 04a — Hard Metrics Monitor

**File**: `airflow/dags/dag_04a_monitor_hard.py`  
**Schedule**: None (triggered by dag_03 and dag_05, or manually)  
**Purpose**: Compute all 6 classification metrics for a specific run against labelled ground truth. Triggers retraining if the primary metric is below threshold.

### Conf

| Key | Type | Description |
|---|---|---|
| `run_index` | int or null | Which batch run to evaluate (null = latest) |
| `skip_retrain_trigger` | bool | If True, never triggers dag_05 (prevents post-retrain loop) |

### Task Graph
```
run_hard_metrics  (hard_metrics.py --run-index N)
    ↓
parse_result
    ↓
decide_retrain  (BranchPythonOperator)
  ↙                    ↘
trigger_retraining   no_retraining_needed
(dag_05)             (EmptyOperator)
```

### Retraining Threshold

Primary metric is read from the production model's MLflow tag (`primary_metric`). Falls back to `DEFAULT_PRIMARY_METRIC` env var (`roc_auc`).

| Env Var | Default | Condition |
|---|---|---|
| `MIN_ROC_AUC` | `0.60` | Trigger retraining if primary metric score < this value |

### Outputs
- `dwh_monitoring_hard.results` — all 6 metrics + `primary_metric` + `retraining_triggered`
- MLflow experiment `monitoring_hard`
- `/data/monitoring/hard/run_NNNNN.parquet`

---

## DAG 04b — Data Drift Monitor

**File**: `airflow/dags/dag_04b_monitor_drift.py`  
**Schedule**: None (manual)  
**Purpose**: KS drift analysis comparing current batch feature distributions to the training reference distribution.

### Conf

| Key | Type | Description |
|---|---|---|
| `run_index` | int or null | Which batch run to analyse |

### What drift.py does
1. Loads current batch features from `dwh_history.prediction_ground_truth`
2. Loads reference distribution from MLflow artifact `reference_data.parquet` (logged at training time)
3. Runs Kolmogorov-Smirnov test per feature
4. `drift_score` = fraction of features with statistically significant drift
5. `drift_detected = True` if `drift_score > MAX_DRIFT_SCORE`

### Outputs
- `dwh_monitoring_drift.results` — `drift_detected`, `drift_score`, `num_drifted_features`, `drifted_feature_names`
- MLflow experiment `monitoring_drift`
- `/data/monitoring/drift/run_NNNNN.parquet`

---

## DAG 04c — SHAP Explainability

**File**: `airflow/dags/dag_04c_monitor_shap.py`  
**Schedule**: None (manual)  
**Purpose**: Compute SHAP values for all records in a run. Stores aggregate and per-customer values for the Dashboard waterfall view.

### What shap_monitor.py does
1. Loads features + customer_id from `dwh_history.prediction_ground_truth`
2. Loads Production model from MLflow registry
3. Tries `shap.TreeExplainer`; falls back to `shap.KernelExplainer`
4. Computes SHAP matrix (one value per feature per record)
5. Logs beeswarm plot and mean |SHAP| bar chart as MLflow artifacts
6. Writes aggregate `feature_importances` JSONB to `dwh_monitoring_shap.results`
7. Writes per-record SHAP vectors to `dwh_monitoring_shap.customer_shap_values`

### Customer SHAP
Each row in `customer_shap_values` contains:
- `customer_id` — deterministic UUID5 (stable across runs for the same customer)
- `shap_values` — JSONB: `{feature: shap_value, ...}` for that record
- `base_value` — model's expected output (log-odds baseline)
- `predicted_probability` — actual model output for this record

### Outputs
- `dwh_monitoring_shap.results` (aggregate per run)
- `dwh_monitoring_shap.customer_shap_values` (per-record SHAP)
- MLflow experiment `monitoring_shap` — beeswarm + bar chart artifacts
- `/data/monitoring/shap/run_NNNNN.parquet`

---

## DAG 05 — Retraining

**File**: `airflow/dags/dag_05_retraining.py`  
**Schedule**: None (event-driven via dag_04a or manual)  
**Purpose**: Retrain on all available clean data, validate quality gate, promote to Production, reload Flask.

### Task Graph
```
validate_data_available
    ↓
retrain_model  (train.py --promote)
    ↓
parse_training_result  (fails DAG if quality gate not passed)
    ↓
verify_promotion
    ↓
restart_flask_api  (POST /reload)
    ↓
log_retraining_event  (dwh_history.retraining_log)
    ↓
trigger_post_retrain_metrics  (dag_04a with skip_retrain_trigger=True)
```

### Promotion Gate (training/train.py)

| Metric | Threshold |
|---|---|
| `roc_auc` | ≥ 0.60 |
| `f1` | ≥ 0.35 |

If the gate fails, the DAG fails and the current Production model remains unchanged.

### Inputs / Outputs
- **Input**: `dwh_clean.cleaned_features` — **all available records** (no time window filter)
- **Artifact**: `reference_data.parquet` logged to MLflow run (training features, used by drift monitor)
- **Output**: new MLflow model version with `production` alias; `dwh_history.retraining_log` row
- **Post-retrain**: triggers dag_04a with `skip_retrain_trigger=True` to verify new model quality

---

## DAG 06 — Challenger Comparison

**File**: `airflow/dags/dag_06_challenger_comparison.py`  
**Schedule**: None (manual — triggered from Control Panel Challenger tab)  
**Purpose**: Compare a challenger model (logged to MLflow) against the current Production model using StratifiedKFold cross-validation on recent labelled ground truth.

### Conf

| Key | Type | Description |
|---|---|---|
| `challenger_run_id` | string | MLflow run ID of the challenger (must have model logged at `model/`) |
| `force_deploy` | bool | If True, promote challenger regardless of CV result |

### Task Graph
```
load_and_score
    ↓
compare_models  (StratifiedKFold CV, log to MLflow challenger_experiments)
    ↓
decide_promotion  (BranchPythonOperator)
  ↙                        ↘
promote_challenger       skip_promotion
(reload Flask + DB log)  (EmptyOperator)
    ↘                        ↙
           log_comparison  (trigger_rule=ALL_DONE)
```

### Comparison Logic
- Loads latest 2000 labelled rows from `dwh_history.prediction_ground_truth`
- Scores both Production and Challenger models on the same feature matrix
- Runs `CHALLENGER_CV_FOLDS`-fold StratifiedKFold CV on the **primary metric**
- Primary metric read from challenger MLflow run tag; falls back to `DEFAULT_PRIMARY_METRIC`
- Challenger wins if: `challenger_cv_mean - prod_cv_mean > CHALLENGER_CV_MARGIN`
- On promotion: logs eval data as `reference_data.parquet` to challenger's MLflow run

### Env Vars

| Var | Default | Description |
|---|---|---|
| `CHALLENGER_CV_FOLDS` | `5` | Number of CV folds |
| `CHALLENGER_CV_MARGIN` | `0.05` | Minimum improvement required (absolute, on primary metric) |

### Outputs
- `dwh_challenger.comparison_log` — CV scores, `challenger_wins`, `force_deploy`, `promoted`
- MLflow experiment `challenger_experiments` — comparison metrics logged
- On promotion: Flask reloaded, new version gets `production` alias
