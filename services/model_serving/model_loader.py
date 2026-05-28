"""
MLflow Model Loader
===================

Fetches the champion model from MLflow registry at startup.
Exposes load_model() which returns (model, version_info).
"""

import logging
import os

import mlflow
from mlflow import MlflowClient

log = logging.getLogger(__name__)


def load_model():
    """
    Load the model with the 'production' alias from MLflow registry.

    Returns:
        tuple: (pyfunc_model, model_version_info)

    Raises:
        RuntimeError: if no model with the 'production' alias exists.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
    alias = os.getenv("MODEL_ALIAS", "production")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        version_info = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        raise RuntimeError(
            f"No model '{model_name}' with alias '{alias}'. "
            "Train and promote a model first."
        )

    model_uri = f"models:/{model_name}@{alias}"

    log.info(
        "Loading model '%s' alias='%s' version=%s from %s",
        model_name, alias, version_info.version, model_uri,
    )
    model = mlflow.pyfunc.load_model(model_uri)
    log.info("Model loaded successfully.")
    return model, version_info
