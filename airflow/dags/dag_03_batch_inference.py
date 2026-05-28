"""
DAG 03 - Batch Inference (self-contained, manually triggerable)
================================================================
Fully self-contained pipeline: generates data, processes it, runs inference,
then cascades to hard-metrics monitoring.

Can be triggered as many times as desired from the Airflow UI.
Each run gets a unique sequential run_index from dwh_history.run_registry.

Params (set when triggering from UI):
  drift_factor: float [0.0–1.0] - covariate shift applied to generated data
  n_records:    int             - number of records to generate per run

Tasks:
  register_run        → INSERT into run_registry, get run_index (SERIAL)
  generate_data       → data_generator/generator.py
  process_data        → processing_pipeline/pipeline.py
  check_flask_health  → GET /health on Flask API
  run_inference       → batch_inference/inference.py --run-index <run_index>
  trigger_monitoring  → trigger dag_04a_monitor_hard with run_index
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

GENERATOR_PATH = "/opt/mlops/services/data_generator"
PROCESSING_PATH = "/opt/mlops/services/processing_pipeline"
INFERENCE_PATH = "/opt/mlops/services/batch_inference"
FLASK_ENDPOINT = os.getenv("FLASK_ENDPOINT", "http://flask-api:5001")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
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
    "DATA_RAW_PATH": "/data/raw",
    "DATA_PROCESSED_PATH": "/data/processed",
    "DATA_PREDICTIONS_PATH": "/data/predictions",
    "FLASK_ENDPOINT": FLASK_ENDPOINT,
    "DECISION_THRESHOLD": os.getenv("DECISION_THRESHOLD", "0.5"),
}


def _register_run(ti, **context):
    """Insert a pending entry into run_registry and return the SERIAL run_index."""
    import psycopg2

    params = context["params"]
    drift_factor = float(params.get("drift_factor", 0.0))
    n_records = int(params.get("n_records", 1000))

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
                INSERT INTO dwh_history.run_registry
                    (drift_factor, n_records_requested, status)
                VALUES (%s, %s, 'running')
                RETURNING run_index
                """,
                (drift_factor, n_records),
            )
            run_index = cur.fetchone()[0]
        conn.commit()

    ti.xcom_push(key="run_index", value=run_index)
    print(f"Registered run_index={run_index} (drift_factor={drift_factor}, n_records={n_records})")
    return run_index


def _check_flask_health(**context):
    import requests
    url = f"{FLASK_ENDPOINT}/health"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        print(f"Flask API healthy: {resp.json()}")
    except Exception as exc:
        raise RuntimeError(f"Flask API not reachable at {url}: {exc}")


def _trigger_monitoring(ti, **context):
    """Trigger dag_04a_monitor_hard with the run_index from this run."""
    from airflow.api.client.local_client import Client

    run_index = ti.xcom_pull(task_ids="register_run", key="run_index")
    client = Client(None, None)
    client.trigger_dag(
        dag_id="dag_04a_monitor_hard",
        run_id=f"triggered_by_dag_03_run_{run_index}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
        conf={"run_index": run_index, "skip_retrain_trigger": False},
    )
    print(f"Triggered dag_04a_monitor_hard for run_index={run_index}")


with DAG(
    dag_id="dag_03_batch_inference",
    description="Self-contained: generate → process → infer → monitor. Trigger from UI.",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    params={
        "drift_factor": Param(
            0.0,
            type="number",
            minimum=0.0,
            maximum=1.0,
            description="Covariate drift applied to generated data (0=stable, 1=max drift)",
        ),
        "n_records": Param(
            1000,
            type="integer",
            minimum=10,
            description="Number of synthetic records to generate",
        ),
    },
    tags=["mlops", "inference"],
) as dag:

    register_run = PythonOperator(
        task_id="register_run",
        python_callable=_register_run,
        doc_md="Creates a run_registry entry and obtains a sequential run_index.",
    )

    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=(
            "cd {path} && python generator.py "
            "--mode {{{{ 'drift' if params.drift_factor > 0 else 'stable' }}}} "
            "--drift-factor {{{{ params.drift_factor }}}} "
            "--n-records {{{{ params.n_records }}}}"
        ).format(path=GENERATOR_PATH),
        env={**DB_ENV},
        doc_md="Generates synthetic loan application data with optional covariate drift.",
    )

    process_data = BashOperator(
        task_id="process_data",
        bash_command=f"cd {PROCESSING_PATH} && python pipeline.py",
        env={**DB_ENV},
        doc_md="Ingests raw data, cleans and encodes features, writes to dwh_clean.",
    )

    check_health = PythonOperator(
        task_id="check_flask_health",
        python_callable=_check_flask_health,
        doc_md="Pings Flask /health. Fails fast if the model serving API is down.",
    )

    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=(
            "cd {path} && python inference.py "
            "--run-index {{{{ ti.xcom_pull(task_ids='register_run', key='run_index') }}}} "
            "--processed-dir /data/processed "
            "--predictions-dir /data/predictions "
            "--endpoint {endpoint} "
            "--batch-size 200 "
            "--threshold {{{{ var.value.get('DECISION_THRESHOLD', '0.5') }}}}"
        ).format(path=INFERENCE_PATH, endpoint=FLASK_ENDPOINT),
        env={**DB_ENV},
        do_xcom_push=True,
        doc_md=(
            "Reads latest cleaned parquet, calls Flask /predict in batches, "
            "writes to dwh_history.prediction_ground_truth."
        ),
    )

    trigger_monitoring = PythonOperator(
        task_id="trigger_monitoring",
        python_callable=_trigger_monitoring,
        doc_md="Triggers dag_04a_monitor_hard with the current run_index.",
    )

    register_run >> generate_data >> process_data >> check_health >> run_inference >> trigger_monitoring
