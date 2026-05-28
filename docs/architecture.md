# System Architecture

## Overview

End-to-end MLOps platform for binary credit risk classification (loan default prediction). All components run in Docker containers on a shared `mlops-net` bridge network. Apache Airflow orchestrates all pipeline DAGs. Two Streamlit apps provide operational UIs. A containerized JupyterLab environment is pre-wired for prototyping and EDA.

---

## High-Level Component Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACES                                  │
│                                                                           │
│  Control Panel :8501          Monitoring Dashboard :8502                  │
│  ┌─────────────────────┐      ┌──────────────────────────┐               │
│  │ Pipelines           │      │ Data Drift               │               │
│  │  └ Batch Inference  │      │ Hard Metrics             │               │
│  │ Monitor             │      │ Explainability (SHAP)    │               │
│  │ Deploy Model        │      └──────────────────────────┘               │
│  │ Challenger          │                                                  │
│  │ DAG Logs            │      JupyterLab :8888                           │
│  └─────────────────────┘      ┌──────────────────────────┐               │
│                                │ 01_challenger_template   │               │
│                                │ 02_eda_and_feature_...   │               │
│                                │ 03_production_model_...  │               │
│                                └──────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
          │ trigger DAGs / read metrics          │ log experiments / read data
          ▼                                      ▼
┌─────────────────────┐              ┌───────────────────────┐
│  Apache Airflow     │              │  MLflow :5000          │
│  :8080              │◄────────────►│  Experiments           │
│  Webserver          │  register /  │  Model Registry        │
│  Scheduler          │  load model  │  Artifacts             │
└──────────┬──────────┘              └──────────┬────────────┘
           │ run scripts                         │ load model
           ▼                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────┐
│  PostgreSQL :5432                │   │  Flask API :5001     │
│  dwh_raw · dwh_clean            │   │  POST /predict       │
│  dwh_history                    │◄──│  POST /reload        │
│  dwh_monitoring_hard            │   │  GET  /health        │
│  dwh_monitoring_drift           │   │  GET  /model-info    │
│  dwh_monitoring_shap            │   └─────────────────────┘
│  dwh_challenger                 │
│  dwh_predictions (legacy)       │
│  airflow · mlflow               │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Bind-Mounted File System       │
│  /data/raw                      │
│  /data/processed                │
│  /data/predictions              │
│  /data/monitoring               │
│  /mlflow-artifacts              │
└─────────────────────────────────┘
```

---

## Data Flow

```
  [1] BATCH INFERENCE PIPELINE  (dag_03_batch_inference)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  data_generator  →  processing_pipeline  →  batch_inference              │
  │                                                                           │
  │  generator.py --mode stable|drift --drift-factor 0.0–1.0                 │
  │  pipeline.py   (clean, encode, validate)                                 │
  │  inference.py  --run-index N  (calls Flask /predict, writes history)     │
  └──────────────────────────────────────────────────────────────────────────┘
           │  writes to:
           ├──► /data/raw/*.parquet            dwh_raw.raw_loan_applications
           ├──► /data/processed/*.parquet      dwh_clean.cleaned_features
           ├──► /data/predictions/*.parquet
           ├──► dwh_predictions.batch_predictions  (legacy, per-record)
           └──► dwh_history.run_registry           (SERIAL run_index)
                dwh_history.prediction_ground_truth (features + score + actual)

  [2] MODEL SERVING  (flask-api)
  ┌──────────────────┐
  │  gunicorn+Flask  │  loads Production model from MLflow at startup
  └────────┬─────────┘
           ├── POST /predict      → default_probability, predicted label
           ├── GET  /health       → liveness check
           ├── GET  /model-info   → current model version + run_id
           └── POST /reload       → hot-swap to latest Production alias

  [3] MONITORING  (three independent DAGs, any order, any run_index)
  ┌───────────────────────────────────────────────────────────────────┐
  │  dag_04a  hard_metrics.py    → ROC-AUC, PR-AUC, F1, Prec, Rec, Acc│
  │  dag_04b  drift.py           → KS test per feature (10 features)   │
  │  dag_04c  shap_monitor.py    → SHAP values per record + per feature │
  └───────────────────────────────────────────────────────────────────┘
           writes to:
           ├──► dwh_monitoring_hard.results
           ├──► dwh_monitoring_drift.results
           ├──► dwh_monitoring_shap.results
           ├──► dwh_monitoring_shap.customer_shap_values
           ├──► MLflow experiments: monitoring_hard / monitoring_drift / monitoring_shap
           └──► /data/monitoring/{hard,drift,shap}/*.parquet

  [4] RETRAINING  (dag_05, triggered by dag_04a when primary metric below threshold)
  ┌────────────────────────────────────────────────────────────────┐
  │  training/train.py, hyperparameter search, trains on ALL data  │
  │  Triggered when: primary_metric < MIN_ROC_AUC (default 0.60)   │
  └────────────────────────────────────────────────────────────────┘
           on success:
           ├──► MLflow: new model version registered + 'production' alias set
           ├──► reference_data.parquet logged as artifact (training set features)
           ├──► Flask /reload hot-swaps the model
           └──► dwh_history.retraining_log entry written

  [5] CHALLENGER COMPARISON  (dag_06, manual trigger from Control Panel)
  ┌────────────────────────────────────────────────────────────────┐
  │  StratifiedKFold CV (5 folds) on latest labelled ground truth  │
  │  Challenger must beat Production by CHALLENGER_CV_MARGIN        │
  │  force_deploy=True promotes regardless of comparison result     │
  └────────────────────────────────────────────────────────────────┘
           on promotion:
           ├──► MLflow: challenger run registered as new version
           ├──► 'production' alias reassigned to new version
           ├──► reference_data.parquet logged to challenger's run (eval data)
           ├──► Flask /reload
           └──► dwh_challenger.comparison_log entry written
```

---

## Services

| Container | Image / Build | Port | Role |
|---|---|---|---|
| `mlops-postgres` | `postgres:15` | 5432 | All relational storage |
| `mlops-mlflow` | `./infrastructure/mlflow` | 5000 | Tracking server + model registry + artifact store |
| `mlops-platform-init` | `./airflow` | - | One-shot bootstrap: Airflow DB, admin user, initial model |
| `mlops-airflow-webserver` | `./airflow` | 8080 | Airflow UI |
| `mlops-airflow-scheduler` | `./airflow` | - | DAG scheduling |
| `mlops-flask-api` | `./services/model_serving` | 5001 | Model serving endpoint (gunicorn + Flask) |
| `mlops-streamlit-ui` | `./services/streamlit_ui` | 8501 | Control Panel |
| `mlops-streamlit-dashboard` | `./services/streamlit_ui` | 8502 | Monitoring Dashboard |
| `mlops-jupyter` | `./infrastructure/jupyter` | 8888 | JupyterLab, EDA + challenger prototyping |

Pipeline scripts (data_generator, processing_pipeline, batch_inference, monitoring modules) run as Airflow BashOperator tasks, they are NOT long-running services.

---

## PostgreSQL Schemas

```
mlops (database)
├── dwh_raw                    raw generated loan applications
│   └── raw_loan_applications
├── dwh_clean                  feature-engineered records (model input)
│   └── cleaned_features
├── dwh_history                batch run tracking + prediction + ground truth
│   ├── run_registry           SERIAL run_index, one row per batch run
│   ├── prediction_ground_truth  features, score, actual_default_flag per record
│   └── retraining_log         audit trail of every retraining event
├── dwh_monitoring_hard        classification metrics per run
│   └── results                all 6 metrics + primary_metric + retraining_triggered
├── dwh_monitoring_drift       feature drift per run
│   └── results                drift_score, drifted_feature_names, KS stats
├── dwh_monitoring_shap        SHAP explainability per run
│   ├── results                aggregate: feature_importances JSONB, top_feature
│   └── customer_shap_values   per-record: shap_values JSONB, base_value
├── dwh_challenger             challenger comparison audit
│   └── comparison_log         CV scores, promoted flag, force_deploy
├── dwh_predictions            batch inference output (legacy, still populated)
│   └── batch_predictions
├── airflow                    Airflow backend (auto-managed)
└── mlflow                     MLflow backend (auto-managed)
```

---

## MLflow Organization

```
MLflow Tracking Server (http://localhost:5000)
├── Experiments
│   ├── credit-risk-training      training runs (train.py)
│   ├── challenger_experiments    challenger prototype runs (from notebooks)
│   ├── monitoring_hard           hard metric monitoring runs
│   ├── monitoring_drift          drift detection runs
│   └── monitoring_shap           SHAP explainability runs
└── Models (Registry)
    └── credit-risk-classifier
        ├── versions               all trained/registered versions
        ├── tag: primary_metric    which metric this version optimises
        ├── tag: stage             'production' | 'archived'
        └── alias: production      currently served by Flask API
```

No `staging` alias exists. The registry uses `production` alias + `stage` tag for lineage.

---

## Streamlit Applications

### Control Panel (:8501)

| Tab | Function |
|---|---|
| **Pipelines** | Trigger dag_03 (batch inference). Configure drift_factor and n_records. Shows run status. |
| **Monitor** | Trigger dag_04a / 04b / 04c individually. Shows DAG run state with auto-refresh. |
| **Deploy Model** | Select any MLflow model version, assign `production` alias, reload Flask (calls dag_00). |
| **Challenger** | Paste a JupyterLab run_id. Primary metric auto-detected from MLflow tag. Trigger dag_06. Shows recent comparison log. |
| **DAG Logs** | Browse recent runs across all DAGs. Fetch task-level logs via Airflow REST API. |

### Monitoring Dashboard (:8502)

| Tab | Function |
|---|---|
| **Data Drift** | Feature distribution comparison (reference vs current). KS statistics per feature. Drift score trend. |
| **Hard Metrics** | All 6 metric cards (★ primary metric highlighted). ROC, PR, calibration curves. Confusion matrix. Threshold sensitivity. Metric trend across runs. |
| **Explainability** | SHAP beeswarm + mean |SHAP| bar chart. Customer waterfall: select customer_id to see per-feature contribution. |

---

## Airflow DAGs

| DAG | Schedule | Purpose |
|---|---|---|
| `dag_00_deploy_model` | manual | Promote model version → assign alias → reload Flask |
| `dag_03_batch_inference` | manual | Full pipeline: generate → process → score → write history → trigger dag_04a |
| `dag_04a_monitor_hard` | event-driven / manual | Compute all 6 metrics; auto-triggers dag_05 if primary metric below threshold |
| `dag_04b_monitor_drift` | manual | KS drift analysis vs training reference distribution |
| `dag_04c_monitor_shap` | manual | SHAP values, aggregate + per-customer (waterfall source) |
| `dag_05_retraining` | event-driven / manual | Hyperparameter search → train on ALL data → quality gate → promote → reload |
| `dag_06_challenger_comparison` | manual | StratifiedKFold CV challenger vs production; promote if wins or force_deploy=True |

---

## Customer Identity

Each generated record receives a **deterministic customer UUID** computed with `uuid.uuid5` from a pool of 10,000 synthetic customer identifiers. The same customer can appear across multiple batch runs, enabling longitudinal tracking in the Dashboard waterfall view (per-customer SHAP across time).

---

## Drift Simulation

Only **covariate drift** is simulated (feature distributions shift). The target relationship (the logistic formula that determines default_flag) stays constant. This mirrors real credit risk scenarios where macroeconomic conditions change applicant profiles while default mechanics remain stable.

The `drift_factor` parameter [0.0–1.0] controls shift intensity:
- `0.0`, stable distribution (training reference)
- `0.5`, moderate shift: credit_score mean drops, DTI increases, income variance grows
- `1.0`, severe shift: most features significantly displaced

---

## Reference Dataset

At training time, `train.py` logs the training feature matrix as `reference_data.parquet` to the MLflow run artifact store. Drift monitoring loads this file to compare against the current batch distribution. When a challenger is promoted via dag_06, the evaluation ground truth used during CV is logged as its `reference_data.parquet`.
