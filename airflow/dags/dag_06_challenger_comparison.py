"""
DAG 06 — Challenger Model Comparison
======================================

Compares a prototype/challenger model (identified by an MLflow run_id) against
the currently deployed Production model. Both are scored on the same held-out
ground truth data from dwh_history.prediction_ground_truth.

The comparison always runs (for transparency/logging). The force_deploy flag
controls whether promotion happens regardless of outcome.

Conf keys:
  challenger_run_id  (str, required) — MLflow run_id with a model logged under model/
  force_deploy       (bool, default false) — promote even if challenger loses
  primary_metric     (str, default "roc_auc") — metric used for comparison
                     one of: roc_auc, pr_auc, f1, precision, recall, accuracy

Tasks:
  load_and_score → compare_models → decide_promotion
                                         ↓               ↓
                                  promote_challenger  skip_promotion
                                         ↓               ↓
                                        log_comparison (ALL_DONE)
"""

import os
import sys
import uuid
from datetime import datetime, timedelta

# Shared metrics module (mounted at /opt/mlops/services in Airflow containers)
sys.path.insert(0, "/opt/mlops/services")

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from metrics import DEFAULT_PRIMARY, REGISTRY, compute_all_metrics, display_name  # noqa: E402

FEATURE_COLUMNS = [
    "age", "annual_income", "credit_score", "loan_amount",
    "loan_term_months", "employment_length_years", "home_ownership_encoded",
    "debt_to_income_ratio", "num_credit_lines", "payment_history_score",
]

EVAL_LIMIT = 2000

default_args = {
    "owner": "mlops",
    "retries": 0,
    "email_on_failure": False,
}


def _load_and_score(ti, **context):
    import mlflow
    import numpy as np
    import pandas as pd
    import psycopg2

    conf = context["dag_run"].conf or {}
    challenger_run_id = conf.get("challenger_run_id", "").strip()
    if not challenger_run_id:
        raise ValueError("conf.challenger_run_id is required")

    primary_metric = conf.get("primary_metric", DEFAULT_PRIMARY)
    if primary_metric not in REGISTRY:
        print(f"Warning: unknown primary_metric '{primary_metric}', falling back to {DEFAULT_PRIMARY}")
        primary_metric = DEFAULT_PRIMARY
    print(f"Primary metric: {primary_metric} ({display_name(primary_metric)})")

    model_name = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Load production model
    prod_version = client.get_model_version_by_alias(model_name, "production")
    prod_run_id = prod_version.run_id
    prod_model = mlflow.pyfunc.load_model(f"runs:/{prod_run_id}/model")
    print(f"Production model: version {prod_version.version} (run {prod_run_id})")

    # Load challenger model
    challenger_model = mlflow.pyfunc.load_model(f"runs:/{challenger_run_id}/model")
    print(f"Challenger model loaded from run {challenger_run_id}")

    # Fetch eval data — two-stage fallback for resilience on fresh stacks
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )

    # Stage 1: prediction_ground_truth rows, label from actual_default_flag OR
    # cleaned_features.default_flag (handles case where monitoring hasn't run yet)
    feat_cols_sql = ", ".join(f"p.{c}" for c in FEATURE_COLUMNS)
    stage1_query = f"""
        SELECT {feat_cols_sql},
               COALESCE(p.actual_default_flag, c.default_flag) AS actual_default_flag
        FROM dwh_history.prediction_ground_truth p
        LEFT JOIN dwh_clean.cleaned_features c ON p.record_id = c.record_id
        WHERE COALESCE(p.actual_default_flag, c.default_flag) IS NOT NULL
        ORDER BY p.predicted_at DESC
        LIMIT {EVAL_LIMIT}
    """
    with conn:
        eval_df = pd.read_sql(stage1_query, conn)

    data_source = "prediction_ground_truth"

    # Stage 2: no batch inference at all — fall back to cleaned_features holdout
    if eval_df.empty:
        print("No prediction_ground_truth rows found — falling back to cleaned_features holdout split")
        stage2_query = f"""
            SELECT {", ".join(FEATURE_COLUMNS)}, default_flag AS actual_default_flag
            FROM dwh_clean.cleaned_features
            WHERE default_flag IS NOT NULL
            ORDER BY created_at
        """
        with conn:
            full_df = pd.read_sql(stage2_query, conn)

        if full_df.empty:
            conn.close()
            raise ValueError("No training data in cleaned_features — cannot evaluate models")

        # Use last 20% as holdout (same ordering as training: ORDER BY created_at)
        holdout_start = int(len(full_df) * 0.8)
        eval_df = full_df.iloc[holdout_start:].reset_index(drop=True)
        data_source = "cleaned_features_holdout"

    conn.close()
    print(f"Eval data source: {data_source} | {len(eval_df)} records")

    # Log eval feature set as reference_data.parquet on the challenger's run so that
    # if the challenger is promoted, the drift detector has a reference distribution.
    import tempfile
    _ref_tmp = os.path.join(tempfile.gettempdir(), "reference_data.parquet")
    eval_df[FEATURE_COLUMNS].to_parquet(_ref_tmp, index=False, engine="pyarrow")
    with mlflow.start_run(run_id=challenger_run_id):
        mlflow.log_artifact(_ref_tmp)  # logged at root as reference_data.parquet
    print(f"Logged reference_data.parquet ({len(eval_df)} rows) to challenger run {challenger_run_id}")

    from sklearn.model_selection import StratifiedKFold

    n_folds  = int(os.getenv("CHALLENGER_CV_FOLDS", 5))
    cv_margin = float(os.getenv("CHALLENGER_CV_MARGIN", 0.05))

    X_eval = eval_df[FEATURE_COLUMNS]
    y_true = eval_df["actual_default_flag"].values.astype(int)
    print(f"Eval pool: {len(eval_df)} records, default rate {y_true.mean():.3f}")
    print(f"CV strategy: StratifiedKFold n_splits={n_folds}, win margin={cv_margin:+.0%}")

    def _predict_proba(model, X):
        pred = model.predict(X)
        arr = pred.values if hasattr(pred, "values") else np.array(pred)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 1].astype(float)
        arr = arr.astype(float)
        if arr.max() <= 1.0:
            return arr
        raise ValueError("Cannot extract probabilities from model output")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    prod_fold_primary, chal_fold_primary = [], []
    prod_fold_roc,     chal_fold_roc     = [], []

    for fold_i, (_, val_idx) in enumerate(skf.split(X_eval, y_true)):
        X_fold = X_eval.iloc[val_idx]
        y_fold = y_true[val_idx]

        prod_proba = _predict_proba(prod_model, X_fold)
        chal_proba = _predict_proba(challenger_model, X_fold)

        pm = compute_all_metrics(y_fold, prod_proba)
        cm = compute_all_metrics(y_fold, chal_proba)

        prod_fold_primary.append(pm[primary_metric])
        chal_fold_primary.append(cm[primary_metric])
        prod_fold_roc.append(pm["roc_auc"])
        chal_fold_roc.append(cm["roc_auc"])

        metric_lbl = display_name(primary_metric)
        print(
            f"  Fold {fold_i + 1}/{n_folds}  "
            f"prod {metric_lbl}={pm[primary_metric]:.4f}  "
            f"chal {metric_lbl}={cm[primary_metric]:.4f}  "
            f"delta={cm[primary_metric] - pm[primary_metric]:+.4f}"
        )

    prod_avg_primary = float(np.mean(prod_fold_primary))
    chal_avg_primary = float(np.mean(chal_fold_primary))
    prod_std_primary = float(np.std(prod_fold_primary))
    chal_std_primary = float(np.std(chal_fold_primary))
    prod_avg_roc     = float(np.mean(prod_fold_roc))
    chal_avg_roc     = float(np.mean(chal_fold_roc))

    delta = chal_avg_primary - prod_avg_primary
    metric_lbl = display_name(primary_metric)
    print(
        f"\nCV summary [{metric_lbl}]:\n"
        f"  Production:  avg={prod_avg_primary:.4f}  std={prod_std_primary:.4f}\n"
        f"  Challenger:  avg={chal_avg_primary:.4f}  std={chal_std_primary:.4f}\n"
        f"  Delta:       {delta:+.4f}  (need > {cv_margin:+.4f} to win)"
    )

    ti.xcom_push(key="prod_roc_auc",             value=prod_avg_roc)
    ti.xcom_push(key="challenger_roc_auc",       value=chal_avg_roc)
    ti.xcom_push(key="prod_primary_score",       value=prod_avg_primary)
    ti.xcom_push(key="challenger_primary_score", value=chal_avg_primary)
    ti.xcom_push(key="prod_cv_std",              value=prod_std_primary)
    ti.xcom_push(key="challenger_cv_std",        value=chal_std_primary)
    ti.xcom_push(key="cv_folds",                 value=n_folds)
    ti.xcom_push(key="cv_margin",                value=cv_margin)
    ti.xcom_push(key="primary_metric",           value=primary_metric)
    ti.xcom_push(key="n_eval_records",           value=len(eval_df))
    ti.xcom_push(key="prod_run_id",              value=prod_run_id)
    ti.xcom_push(key="prod_version",             value=prod_version.version)
    ti.xcom_push(key="challenger_run_id",        value=challenger_run_id)
    ti.xcom_push(key="data_source",              value=data_source)


def _compare_models(ti, **context):
    import mlflow

    conf = context["dag_run"].conf or {}
    force_deploy = bool(conf.get("force_deploy", False))

    primary_metric    = ti.xcom_pull(task_ids="load_and_score", key="primary_metric")
    prod_primary      = ti.xcom_pull(task_ids="load_and_score", key="prod_primary_score")
    chal_primary      = ti.xcom_pull(task_ids="load_and_score", key="challenger_primary_score")
    prod_cv_std       = ti.xcom_pull(task_ids="load_and_score", key="prod_cv_std")
    chal_cv_std       = ti.xcom_pull(task_ids="load_and_score", key="challenger_cv_std")
    cv_folds          = ti.xcom_pull(task_ids="load_and_score", key="cv_folds")
    cv_margin         = ti.xcom_pull(task_ids="load_and_score", key="cv_margin")
    prod_roc          = ti.xcom_pull(task_ids="load_and_score", key="prod_roc_auc")
    chal_roc          = ti.xcom_pull(task_ids="load_and_score", key="challenger_roc_auc")
    n_records         = ti.xcom_pull(task_ids="load_and_score", key="n_eval_records")
    challenger_run_id = ti.xcom_pull(task_ids="load_and_score", key="challenger_run_id")

    delta = chal_primary - prod_primary
    # Challenger wins only if it beats production by more than the configured margin
    challenger_wins = delta > cv_margin
    metric_label = display_name(primary_metric)

    print(
        f"Comparison [{metric_label}] {cv_folds}-fold CV:\n"
        f"  Production:  avg={prod_primary:.4f}  std={prod_cv_std:.4f}\n"
        f"  Challenger:  avg={chal_primary:.4f}  std={chal_cv_std:.4f}\n"
        f"  Delta:       {delta:+.4f}  (margin required: +{cv_margin:.4f})\n"
        f"  Result:      {'WINS' if challenger_wins else 'LOSES'} | force_deploy={force_deploy}"
    )

    # Log comparison run to MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("challenger_experiments")
    with mlflow.start_run(run_name=f"challenger_vs_prod_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_metrics({
            f"prod_{primary_metric}_avg":  prod_primary,
            f"prod_{primary_metric}_std":  prod_cv_std,
            f"chal_{primary_metric}_avg":  chal_primary,
            f"chal_{primary_metric}_std":  chal_cv_std,
            f"delta_{primary_metric}":     delta,
            "prod_roc_auc_avg":            prod_roc,
            "chal_roc_auc_avg":            chal_roc,
            "cv_margin":                   cv_margin,
        })
        mlflow.log_params({
            "challenger_run_id": challenger_run_id,
            "primary_metric":    primary_metric,
            "cv_folds":          cv_folds,
            "cv_margin":         cv_margin,
            "force_deploy":      str(force_deploy),
            "challenger_wins":   str(challenger_wins),
            "n_eval_records":    n_records,
        })
        mlflow.set_tag("primary_metric", primary_metric)

    ti.xcom_push(key="challenger_wins", value=challenger_wins)


def _decide_promotion(ti, **context):
    conf = context["dag_run"].conf or {}
    force_deploy = bool(conf.get("force_deploy", False))
    challenger_wins = ti.xcom_pull(task_ids="compare_models", key="challenger_wins")

    if challenger_wins or force_deploy:
        return "promote_challenger"
    return "skip_promotion"


def _promote_challenger(ti, **context):
    import mlflow
    import requests

    challenger_run_id    = ti.xcom_pull(task_ids="load_and_score", key="challenger_run_id")
    current_prod_version = ti.xcom_pull(task_ids="load_and_score", key="prod_version")
    primary_metric       = ti.xcom_pull(task_ids="load_and_score", key="primary_metric") or DEFAULT_PRIMARY

    model_name = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Archive current production version
    try:
        client.set_model_version_tag(model_name, current_prod_version, "stage", "archived")
        print(f"Archived previous production version {current_prod_version}")
    except Exception as exc:
        print(f"Warning: could not archive previous version: {exc}")

    # Register challenger model as new version
    model_uri = f"runs:/{challenger_run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    new_version = mv.version
    print(f"Registered challenger as version {new_version}")

    # Wait briefly for registry to settle
    import time
    time.sleep(3)

    # Promote to production alias and store primary_metric tag
    client.set_registered_model_alias(model_name, "production", new_version)
    client.set_model_version_tag(model_name, new_version, "stage", "production")
    client.set_model_version_tag(model_name, new_version, "primary_metric", primary_metric)
    print(f"Promoted version {new_version} to production alias (primary_metric={primary_metric})")

    # Reload Flask API
    flask_endpoint = os.getenv("FLASK_ENDPOINT", "http://flask-api:5001")
    try:
        resp = requests.post(f"{flask_endpoint}/reload", timeout=30)
        if resp.status_code == 200:
            print("Flask API reloaded successfully")
        else:
            print(f"Flask reload returned {resp.status_code}")
    except Exception as exc:
        print(f"Warning: Flask reload failed: {exc}")

    ti.xcom_push(key="new_version", value=new_version)
    ti.xcom_push(key="promoted", value=True)


def _log_comparison(ti, **context):
    import psycopg2

    conf = context["dag_run"].conf or {}
    force_deploy = bool(conf.get("force_deploy", False))

    primary_metric    = ti.xcom_pull(task_ids="load_and_score", key="primary_metric") or DEFAULT_PRIMARY
    prod_roc          = ti.xcom_pull(task_ids="load_and_score", key="prod_roc_auc")
    chal_roc          = ti.xcom_pull(task_ids="load_and_score", key="challenger_roc_auc")
    prod_primary      = ti.xcom_pull(task_ids="load_and_score", key="prod_primary_score")
    chal_primary      = ti.xcom_pull(task_ids="load_and_score", key="challenger_primary_score")
    prod_cv_std       = ti.xcom_pull(task_ids="load_and_score", key="prod_cv_std")
    chal_cv_std       = ti.xcom_pull(task_ids="load_and_score", key="challenger_cv_std")
    cv_folds          = ti.xcom_pull(task_ids="load_and_score", key="cv_folds")
    cv_margin         = ti.xcom_pull(task_ids="load_and_score", key="cv_margin")
    n_records         = ti.xcom_pull(task_ids="load_and_score", key="n_eval_records")
    challenger_run_id = ti.xcom_pull(task_ids="load_and_score", key="challenger_run_id")
    data_source       = ti.xcom_pull(task_ids="load_and_score", key="data_source") or "unknown"
    challenger_wins   = ti.xcom_pull(task_ids="compare_models", key="challenger_wins")
    promoted          = ti.xcom_pull(task_ids="promote_challenger", key="promoted") or False

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dwh_challenger.comparison_log
                (id, challenger_run_id, primary_metric,
                 prod_roc_auc, challenger_roc_auc,
                 prod_primary_score, challenger_primary_score,
                 prod_cv_std, challenger_cv_std, cv_folds, cv_margin,
                 challenger_wins, force_deploy, promoted, eval_records, compared_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    challenger_run_id, primary_metric,
                    prod_roc, chal_roc,
                    prod_primary, chal_primary,
                    prod_cv_std, chal_cv_std, cv_folds, cv_margin,
                    challenger_wins, force_deploy, promoted,
                    n_records, datetime.utcnow(),
                ),
            )
    conn.close()
    delta = (chal_primary or 0) - (prod_primary or 0)
    print(
        f"Comparison logged: {primary_metric} {cv_folds}-fold CV | "
        f"prod_avg={prod_primary:.4f}±{prod_cv_std:.4f} "
        f"chal_avg={chal_primary:.4f}±{chal_cv_std:.4f} "
        f"delta={delta:+.4f} margin={cv_margin:.4f} "
        f"wins={challenger_wins} promoted={promoted} source={data_source}"
    )


with DAG(
    dag_id="dag_06_challenger_comparison",
    description="Compare prototype challenger model vs Production; optionally promote",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["mlops", "challenger", "prototyping"],
    params={
        "challenger_run_id": "",
        "force_deploy":      False,
        "primary_metric":    DEFAULT_PRIMARY,
    },
) as dag:

    load_and_score = PythonOperator(
        task_id="load_and_score",
        python_callable=_load_and_score,
        doc_md=(
            "Loads both production and challenger models from MLflow. "
            "Scores both on the same held-out ground truth batch data. "
            "Computes ROC-AUC for each."
        ),
    )

    compare_models = PythonOperator(
        task_id="compare_models",
        python_callable=_compare_models,
        doc_md="Compares metrics, logs comparison run to MLflow challenger_experiments.",
    )

    decide_promotion = BranchPythonOperator(
        task_id="decide_promotion",
        python_callable=_decide_promotion,
        doc_md="Promotes if challenger_wins OR force_deploy=True.",
    )

    promote_challenger = PythonOperator(
        task_id="promote_challenger",
        python_callable=_promote_challenger,
        doc_md=(
            "Registers challenger run as new model version, "
            "sets production alias, reloads Flask API."
        ),
    )

    skip_promotion = EmptyOperator(
        task_id="skip_promotion",
        doc_md="Challenger did not win and force_deploy=False — production unchanged.",
    )

    log_comparison = PythonOperator(
        task_id="log_comparison",
        python_callable=_log_comparison,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Writes comparison result to dwh_challenger.comparison_log regardless of outcome.",
    )

    load_and_score >> compare_models >> decide_promotion
    decide_promotion >> [promote_challenger, skip_promotion]
    promote_challenger >> log_comparison
    skip_promotion >> log_comparison
