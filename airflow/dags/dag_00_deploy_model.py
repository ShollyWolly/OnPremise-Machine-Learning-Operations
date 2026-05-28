"""
DAG 00 - Deploy Model
======================
Manually triggered from Airflow UI. Assigns the 'production' alias to a model
version in the MLflow registry and hot-reloads the Flask serving API.

Params (configurable from Trigger DAG UI):
  model_name:    Registry name  (default: credit-risk-classifier)
  model_version: Specific version number to promote; empty = version with 'staging' alias
  model_stage:   Target alias   (default: production)

Tasks:
  promote_to_stage  → set alias on model version in MLflow registry
  reload_flask_api  → POST /reload to Flask API
  verify_deployment → GET /model-info to confirm new version is live
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


def _promote_to_stage(ti, **context):
    import mlflow

    params = context["params"]
    model_name = params.get("model_name", "credit-risk-classifier")
    model_version = str(params.get("model_version", "")).strip()
    target_alias = params.get("model_stage", "production").lower()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.MlflowClient()

    if model_version:
        version_to_promote = model_version
        print(f"Promoting specific version: {model_name} v{version_to_promote}")
    else:
        try:
            staging_version = client.get_model_version_by_alias(model_name, "staging")
            version_to_promote = staging_version.version
            print(f"Promoting version with 'staging' alias: {model_name} v{version_to_promote}")
        except Exception:
            raise RuntimeError(f"No 'staging' alias found for '{model_name}'. "
                               "Train and register a model first.")

    # Tag current version under target alias as archived
    try:
        current = client.get_model_version_by_alias(model_name, target_alias)
        if str(current.version) != str(version_to_promote):
            client.set_model_version_tag(model_name, current.version, "stage", "archived")
            print(f"Archived: {model_name} v{current.version}")
    except Exception:
        pass

    client.set_registered_model_alias(model_name, target_alias, version_to_promote)
    client.set_model_version_tag(model_name, version_to_promote, "stage", target_alias)
    ti.xcom_push(key="deployed_version", value=str(version_to_promote))
    ti.xcom_push(key="model_name", value=model_name)
    print(f"Promoted '{model_name}' v{version_to_promote} → alias:{target_alias}")


def _reload_flask_api(ti, **context):
    import requests

    endpoint = os.getenv("FLASK_ENDPOINT", "http://flask-api:5001")
    resp = requests.post(f"{endpoint}/reload", timeout=30)
    resp.raise_for_status()
    result = resp.json()
    print(f"Flask reloaded: {result}")
    ti.xcom_push(key="flask_model_version", value=result.get("model_version"))


def _verify_deployment(ti, **context):
    import requests

    endpoint = os.getenv("FLASK_ENDPOINT", "http://flask-api:5001")
    expected_version = ti.xcom_pull(task_ids="promote_to_stage", key="deployed_version")

    resp = requests.get(f"{endpoint}/model-info", timeout=10)
    resp.raise_for_status()
    info = resp.json()
    current_version = str(info.get("model_version", ""))

    if current_version != str(expected_version):
        raise RuntimeError(
            f"Version mismatch: Flask serves v{current_version}, expected v{expected_version}. "
            "Try restarting the flask-api container."
        )

    print(f"Deployment verified: Flask serving '{info['model_name']}' "
          f"v{current_version} (alias:{info.get('model_alias', 'production')})")


with DAG(
    dag_id="dag_00_deploy_model",
    description="Promote a model version to Production and reload the Flask serving API",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    params={
        "model_name": Param(
            "credit-risk-classifier",
            type="string",
            description="MLflow model registry name",
        ),
        "model_version": Param(
            "",
            type="string",
            description="Version number to promote. Leave empty to use version with 'staging' alias.",
        ),
        "model_stage": Param(
            "production",
            type="string",
            enum=["production", "staging"],
            description="Target alias to assign",
        ),
    },
    tags=["mlops", "deployment"],
) as dag:

    promote = PythonOperator(
        task_id="promote_to_stage",
        python_callable=_promote_to_stage,
        doc_md="Assigns the target alias to the specified (or 'staging' alias) model version.",
    )

    reload_flask = PythonOperator(
        task_id="reload_flask_api",
        python_callable=_reload_flask_api,
        doc_md="POSTs to /reload - Flask reloads the new Production model in-process.",
    )

    verify = PythonOperator(
        task_id="verify_deployment",
        python_callable=_verify_deployment,
        doc_md="Confirms Flask /model-info returns the expected new version.",
    )

    promote >> reload_flask >> verify
