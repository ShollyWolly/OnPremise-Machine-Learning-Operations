"""
DAG 05 — Retraining Pipeline
==============================

Triggered by dag_04_monitoring (or manually).
Retrains the model on fresh data, evaluates it against the current production
model, and promotes it to Production if it passes the quality gate.
On success, the Flask API is restarted to load the new model.

This DAG is NOT on a schedule — it is event-driven.

Tasks:
  validate_data_available → checks dwh_clean has enough records
  retrain_model           → runs training/train.py with --promote flag
  parse_training_result   → extracts XCom from train.py JSON output
  verify_promotion        → confirms Production model version changed in registry
  restart_flask_api       → signals Flask to reload model
  log_retraining_event    → writes to dwh_monitoring.retraining_log

On failure: model stays at current Production version, alert logged.
"""

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

TRAINING_PATH = "/opt/mlops/training"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

DB_ENV = {
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "postgres"),
    "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
    "POSTGRES_DB": os.getenv("POSTGRES_DB", "mlops"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "mlops_user"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MODEL_REGISTRY_NAME": os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier"),
}

TRAINING_RUNTIME_ENV = {
    "TRAINING_SEARCH_N_JOBS": "1",
    "TRAINING_MODEL_N_JOBS": "1",
    "XGBOOST_N_JOBS": "1",
    "RF_N_JOBS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JOBLIB_TEMP_FOLDER": "/tmp",
    "GIT_PYTHON_REFRESH": "quiet",
    "PYTHONUNBUFFERED": "1",
}

TRAINING_ENV = {
    **DB_ENV,
    **TRAINING_RUNTIME_ENV,
}


def _validate_data(**context):
    """Ensure there are enough training records in dwh_clean."""
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM dwh_clean.cleaned_features")
            count = cur.fetchone()[0]

    MIN_RECORDS = 500
    if count < MIN_RECORDS:
        raise ValueError(
            f"Insufficient training data: {count} records (need >= {MIN_RECORDS}). "
            "Generate more data with DAG 01 first."
        )
    print(f"Data validation passed: {count} records available for training")


def _parse_training_result(ti, **context):
    """Extract training results from train.py JSON stdout (captured as XCom)."""
    raw = ti.xcom_pull(task_ids="retrain_model")
    if not raw:
        raise ValueError("No output from train.py")

    lines = raw.strip().splitlines()
    result = json.loads(lines[-1])

    ti.xcom_push(key="promoted", value=result.get("promoted", False))
    ti.xcom_push(key="model_version", value=result.get("model_version"))
    ti.xcom_push(key="roc_auc", value=result.get("test_metrics", {}).get("roc_auc"))
    ti.xcom_push(key="promotion_reason", value=result.get("promotion_reason"))
    ti.xcom_push(key="run_id", value=result.get("run_id"))

    if not result.get("promoted"):
        raise ValueError(
            f"Model did NOT pass promotion gate: {result.get('promotion_reason')}. "
            "Current Production model unchanged."
        )

    print(f"Training result: {json.dumps(result, indent=2)}")


def _verify_promotion(ti, **context):
    """Confirm MLflow registry now has the new version with 'production' alias."""
    import mlflow

    model_name = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.MlflowClient()

    try:
        version_info = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        raise RuntimeError(f"No 'production' alias found in registry for '{model_name}'")

    new_version = version_info.version
    expected_version = ti.xcom_pull(task_ids="parse_training_result", key="model_version")

    if str(new_version) != str(expected_version):
        raise RuntimeError(
            f"Version mismatch: registry shows {new_version}, expected {expected_version}"
        )

    print(f"Promotion verified: '{model_name}' version {new_version} has alias 'production'")
    ti.xcom_push(key="verified_version", value=new_version)


def _restart_flask(**context):
    """
    Signal Flask API to reload the model.

    Strategy: POST to /reload if endpoint supports it, otherwise the Flask
    container must be restarted. In production use Docker SDK or k8s rollout.
    Here we use a graceful approach: check if /reload exists, else log warning.
    """
    import requests

    endpoint = os.getenv("FLASK_ENDPOINT", "http://flask-api:5001")

    try:
        # Attempt graceful reload endpoint (add to Flask if needed)
        resp = requests.post(f"{endpoint}/reload", timeout=30)
        if resp.status_code == 200:
            print("Flask API reloaded model gracefully.")
            return
    except Exception:
        pass

    # Fallback: verify the current /model-info reflects the new version
    try:
        ti = context["ti"]
        expected = ti.xcom_pull(task_ids="parse_training_result", key="model_version")
        resp = requests.get(f"{endpoint}/model-info", timeout=10)
        info = resp.json()
        current = str(info.get("model_version", ""))
        if current == str(expected):
            print(f"Flask already serving version {current} (no restart needed).")
        else:
            print(
                f"WARNING: Flask serves version {current}, expected {expected}. "
                "Restart flask-api container manually or redeploy."
            )
    except Exception as exc:
        print(f"Could not verify Flask model version: {exc}")


def _log_retraining_event(ti, **context):
    """Write retraining outcome to dwh_history.retraining_log."""
    import uuid
    import psycopg2

    dag_run_conf = context["dag_run"].conf or {}
    triggered_by = dag_run_conf.get("triggered_by", "manual")
    trigger_roc_auc = dag_run_conf.get("roc_auc")
    trigger_reason = dag_run_conf.get("reason", triggered_by)

    model_version = ti.xcom_pull(task_ids="parse_training_result", key="model_version")
    roc_auc = ti.xcom_pull(task_ids="parse_training_result", key="roc_auc")
    promoted = ti.xcom_pull(task_ids="parse_training_result", key="promoted")

    record = (
        str(uuid.uuid4()),
        trigger_reason,
        float(trigger_roc_auc) if trigger_roc_auc else None,
        str(model_version) if model_version else None,
        float(roc_auc) if roc_auc else None,
        bool(promoted),
        datetime.utcnow(),
    )

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
                INSERT INTO dwh_history.retraining_log
                (log_id, trigger_reason, trigger_roc_auc,
                 new_model_version, new_model_roc_auc, promotion_succeeded, retrained_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                record,
            )
        conn.commit()

    print("Retraining event logged to dwh_history.retraining_log")


with DAG(
    dag_id="dag_05_retraining",
    description="Retrain model on fresh data; promote to Production if quality gates pass",
    schedule_interval=None,  # event-driven, triggered by dag_04 or manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["mlops", "training", "retraining"],
) as dag:

    validate_data = PythonOperator(
        task_id="validate_data_available",
        python_callable=_validate_data,
        doc_md="Checks dwh_clean has >= 500 records before training.",
    )

    retrain = BashOperator(
        task_id="retrain_model",
        bash_command=(
            "cd {path} && "
            "python train.py "
            "--model xgboost "
            "--promote"
        ).format(path=TRAINING_PATH),
        env=TRAINING_ENV,
        append_env=True,
        do_xcom_push=True,
        pool="training_pool",
        doc_md=(
            "Runs training/train.py with --promote flag. "
            "Logs to MLflow and promotes to Production if quality gates pass. "
            "Outputs JSON result to stdout (captured as XCom)."
        ),
    )

    parse_result = PythonOperator(
        task_id="parse_training_result",
        python_callable=_parse_training_result,
    )

    verify_promotion = PythonOperator(
        task_id="verify_promotion",
        python_callable=_verify_promotion,
        doc_md="Confirms MLflow registry has the new version as Production.",
    )

    restart_flask = PythonOperator(
        task_id="restart_flask_api",
        python_callable=_restart_flask,
        doc_md="Signals Flask API to reload the new Production model.",
    )

    log_event = PythonOperator(
        task_id="log_retraining_event",
        python_callable=_log_retraining_event,
        doc_md="Writes retraining outcome to dwh_monitoring.retraining_log for audit trail.",
    )

    trigger_post_retrain_metrics = TriggerDagRunOperator(
        task_id="trigger_post_retrain_metrics",
        trigger_dag_id="dag_04a_monitor_hard",
        conf={
            "triggered_by": "dag_05_retraining",
            "skip_retrain_trigger": True,
        },
        doc_md=(
            "Triggers hard metrics on the latest inference run after retraining. "
            "skip_retrain_trigger=True prevents infinite retraining loops."
        ),
    )

    validate_data >> retrain >> parse_result >> verify_promotion >> restart_flask >> log_event >> trigger_post_retrain_metrics
