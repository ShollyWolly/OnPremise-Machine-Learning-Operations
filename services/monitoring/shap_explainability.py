"""
Monitoring Module 3: SHAP Explainability
==========================================
Computes SHAP values for the current batch using the Production model.

Outputs:
  - MLflow artifacts: beeswarm plot, bar chart
  - Parquet (aggregate means): /data/monitoring/shap/run_NNNNN.parquet
  - Parquet (per-record SHAP): /data/monitoring/shap/run_NNNNN_customer_shap.parquet
  - DB (aggregate): dwh_monitoring_shap.results
  - DB (per-record): dwh_monitoring_shap.customer_shap_values

Usage:
    python shap_explainability.py --run-index 42
    python shap_explainability.py --batch-run-id <uuid>
"""

import argparse
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import shap
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MONITORING_EXPERIMENT = "monitoring_shap"

FEATURE_COLUMNS = [
    "age", "annual_income", "credit_score", "loan_amount",
    "loan_term_months", "employment_length_years", "home_ownership_encoded",
    "debt_to_income_ratio", "num_credit_lines", "payment_history_score",
]


def _db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def load_features(run_index: int | None, batch_run_id: str | None) -> pd.DataFrame:
    if run_index is not None:
        where = "WHERE run_index = %(run_index)s"
        params = {"run_index": run_index}
    elif batch_run_id:
        where = "WHERE batch_run_id = %(batch_run_id)s"
        params = {"batch_run_id": batch_run_id}
    else:
        where = "WHERE run_index = (SELECT MAX(run_index) FROM dwh_history.prediction_ground_truth)"
        params = {}

    query = f"""
        SELECT run_index, batch_run_id, record_id, customer_id,
               {", ".join(FEATURE_COLUMNS)},
               default_probability, actual_default_flag
        FROM dwh_history.prediction_ground_truth
        {where}
    """
    with _db_conn() as conn:
        df = pd.read_sql(query, conn, params=params)
    log.info("Loaded %d rows for SHAP analysis", len(df))
    return df


def _get_underlying_model(mlflow_pyfunc_model):
    impl = mlflow_pyfunc_model._model_impl
    if hasattr(impl, "sklearn_model"):
        return impl.sklearn_model
    if hasattr(impl, "xgb_model"):
        return impl.xgb_model
    if hasattr(impl, "_model"):
        return impl._model
    return impl


def compute_shap(model, data_df: pd.DataFrame) -> dict:
    X = data_df[FEATURE_COLUMNS].values
    X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    underlying = _get_underlying_model(model)

    explainer = None
    explainer_type = None
    shap_values = None
    base_value = 0.0

    try:
        explainer = shap.TreeExplainer(underlying)
        expl_out = explainer(X_df)
        # expl_out.values shape: (n, features) or (n, features, 2) for binary
        sv = expl_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_values = sv
        base_value = float(np.mean(expl_out.base_values[:, 1]) if expl_out.base_values.ndim == 2
                           else np.mean(expl_out.base_values))
        explainer_type = "TreeExplainer"
    except Exception as tree_err:
        log.warning("TreeExplainer failed (%s); falling back to KernelExplainer", tree_err)
        background = shap.sample(X_df, min(100, len(X_df)))

        def _predict_proba(data_array):
            df = pd.DataFrame(data_array, columns=FEATURE_COLUMNS)
            raw = model.predict(df)
            if raw.ndim == 1:
                return np.column_stack([1 - raw, raw])
            return raw

        explainer = shap.KernelExplainer(_predict_proba, background)
        sv_list = explainer.shap_values(X[:50], nsamples=50)
        if isinstance(sv_list, list):
            shap_values = sv_list[1]
        else:
            shap_values = sv_list
        base_value = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)
        explainer_type = "KernelExplainer"

    mean_abs = {feat: float(np.abs(shap_values[:, i]).mean())
                for i, feat in enumerate(FEATURE_COLUMNS)}
    top_feature = max(mean_abs, key=mean_abs.get)
    log.info("SHAP via %s. Top feature: %s (%.4f)", explainer_type, top_feature, mean_abs[top_feature])

    return {
        "shap_values": shap_values,      # ndarray (n, features)
        "mean_abs_shap": mean_abs,
        "explainer_type": explainer_type,
        "top_feature": top_feature,
        "X": X,
        "X_df": X_df,
        "base_value": base_value,
        "explainer": explainer,
    }


def _make_beeswarm_plot(shap_result: dict, run_index: int) -> str:
    """Directional summary dot plot: each dot is one record, coloured by feature value."""
    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X_df"],
        feature_names=FEATURE_COLUMNS,
        show=False,
        max_display=10,
    )
    plt.title(f"SHAP Summary (Beeswarm) - Run {run_index:05d}")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix="_beeswarm.png", delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close("all")
    return tmp.name


def _make_bar_chart(shap_result: dict, run_index: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_items = sorted(shap_result["mean_abs_shap"].items(), key=lambda x: x[1], reverse=True)
    feats = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    ax.barh(feats[::-1], vals[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) - Run {run_index:05d}")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix="_bar.png", delete=False)
    fig.savefig(tmp.name, dpi=150)
    plt.close(fig)
    return tmp.name


def log_to_mlflow(shap_result: dict, run_index: int, batch_run_id: str) -> str:
    with mlflow.start_run(run_name=f"shap_run_{run_index:05d}") as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("run_index", run_index)
        mlflow.log_param("batch_run_id", batch_run_id)
        mlflow.log_param("explainer_type", shap_result["explainer_type"])
        mlflow.log_param("top_feature", shap_result["top_feature"])
        for feat, val in shap_result["mean_abs_shap"].items():
            mlflow.log_metric(f"shap_mean_abs_{feat}", val)

        bar_path = _make_bar_chart(shap_result, run_index)
        mlflow.log_artifact(bar_path, artifact_path="shap")

        beeswarm_path = _make_beeswarm_plot(shap_result, run_index)
        mlflow.log_artifact(beeswarm_path, artifact_path="shap")

    log.info("SHAP logged to MLflow run %s", mlflow_run_id)
    return mlflow_run_id


def write_parquet_aggregate(shap_result: dict, run_index: int, batch_run_id: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_index:05d}.parquet"
    rows = [
        {"run_index": run_index, "batch_run_id": batch_run_id,
         "feature": feat, "mean_abs_shap": shap_result["mean_abs_shap"][feat]}
        for feat in FEATURE_COLUMNS
    ]
    pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow")
    log.info("SHAP aggregate parquet → %s", path)
    return str(path)


def write_parquet_per_record(data_df: pd.DataFrame, shap_result: dict,
                             run_index: int, batch_run_id: str, output_dir: str) -> str:
    """One row per prediction: customer_id, record_id, shap per feature, base_value, prediction."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_index:05d}_customer_shap.parquet"

    n = len(shap_result["shap_values"])
    rows_df = data_df[["customer_id", "record_id", "default_probability"]].iloc[:n].copy().reset_index(drop=True)
    rows_df["run_index"] = run_index
    rows_df["batch_run_id"] = batch_run_id
    rows_df["base_value"] = shap_result["base_value"]
    for i, feat in enumerate(FEATURE_COLUMNS):
        rows_df[f"shap_{feat}"] = shap_result["shap_values"][:, i]

    rows_df.to_parquet(path, index=False, engine="pyarrow")
    log.info("SHAP per-record parquet → %s (%d rows)", path, len(rows_df))
    return str(path)


def write_db_aggregate(run_index: int, batch_run_id: str, shap_result: dict,
                       n_records: int, mlflow_run_id: str, parquet_path: str) -> None:
    record = (
        str(uuid.uuid4()), run_index, batch_run_id,
        shap_result["explainer_type"], shap_result["top_feature"],
        json.dumps(shap_result["mean_abs_shap"]),
        n_records, mlflow_run_id, parquet_path, datetime.utcnow(),
    )
    insert_sql = """
        INSERT INTO dwh_monitoring_shap.results
        (id, run_index, batch_run_id, explainer_type, top_feature,
         feature_importances, n_records, mlflow_run_id, parquet_path, evaluated_at)
        VALUES %s
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dwh_monitoring_shap.results WHERE run_index = %s",
                (run_index,),
            )
            psycopg2.extras.execute_values(cur, insert_sql, [record])
        conn.commit()
    log.info("SHAP aggregate upserted to dwh_monitoring_shap.results")


def write_db_per_record(data_df: pd.DataFrame, shap_result: dict,
                        run_index: int, batch_run_id: str) -> None:
    n = len(shap_result["shap_values"])
    subset = data_df.iloc[:n].reset_index(drop=True)
    records = []
    for idx in range(n):
        shap_dict = {feat: float(shap_result["shap_values"][idx, i])
                     for i, feat in enumerate(FEATURE_COLUMNS)}
        records.append((
            run_index,
            batch_run_id,
            str(subset.at[idx, "customer_id"]),
            str(subset.at[idx, "record_id"]) if pd.notna(subset.at[idx, "record_id"]) else None,
            json.dumps(shap_dict),
            float(shap_result["base_value"]),
            float(subset.at[idx, "default_probability"]) if pd.notna(subset.at[idx, "default_probability"]) else None,
            datetime.utcnow(),
        ))

    insert_sql = """
        INSERT INTO dwh_monitoring_shap.customer_shap_values
        (run_index, batch_run_id, customer_id, record_id,
         shap_values, base_value, predicted_probability, computed_at)
        VALUES %s
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dwh_monitoring_shap.customer_shap_values WHERE run_index = %s",
                (run_index,),
            )
            psycopg2.extras.execute_values(cur, insert_sql, records, page_size=500)
        conn.commit()
    log.info("Per-record SHAP upserted to dwh_monitoring_shap.customer_shap_values (%d rows)", len(records))


def run_shap_explainability(run_index: int | None = None, batch_run_id: str | None = None) -> dict:
    log.info("=== SHAP Explainability START (run_index=%s) ===", run_index)

    model_name = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MONITORING_EXPERIMENT)

    data_df = load_features(run_index, batch_run_id)
    if data_df.empty:
        result = {"run_index": run_index, "top_feature": None, "reason": "no_data"}
        print(json.dumps(result))
        return result

    actual_run_index = int(data_df["run_index"].iloc[0])
    actual_batch_run_id = data_df["batch_run_id"].iloc[0]

    client = mlflow.MlflowClient()
    try:
        client.get_model_version_by_alias(model_name, "production")
    except Exception:
        raise RuntimeError(f"No 'production' alias found for '{model_name}'")
    prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

    shap_result = compute_shap(prod_model, data_df)

    output_dir = os.getenv("DATA_MONITORING_SHAP_PATH", "/data/monitoring/shap")
    parquet_path = write_parquet_aggregate(shap_result, actual_run_index, actual_batch_run_id, output_dir)
    write_parquet_per_record(data_df, shap_result, actual_run_index, actual_batch_run_id, output_dir)
    mlflow_run_id = log_to_mlflow(shap_result, actual_run_index, actual_batch_run_id)
    write_db_aggregate(actual_run_index, actual_batch_run_id, shap_result,
                       len(data_df), mlflow_run_id, parquet_path)
    write_db_per_record(data_df, shap_result, actual_run_index, actual_batch_run_id)

    result = {
        "run_index": actual_run_index,
        "batch_run_id": actual_batch_run_id,
        "explainer_type": shap_result["explainer_type"],
        "top_feature": shap_result["top_feature"],
        "feature_importances": shap_result["mean_abs_shap"],
        "n_records": len(data_df),
    }
    log.info("=== SHAP Explainability COMPLETE ===")
    print(json.dumps(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explainability monitoring module")
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--batch-run-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_explainability(run_index=args.run_index, batch_run_id=args.batch_run_id)
