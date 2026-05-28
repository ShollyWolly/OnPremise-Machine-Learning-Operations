"""
MLOps Credit Risk - Flask Model Serving API
============================================

Endpoints:
    GET  /health       - liveness probe
    GET  /model-info   - metadata about the currently loaded model
    POST /predict      - single or batch prediction

    POST /predict payload (JSON):
        {
          "records": [
            {
              "customer_id": "...",       (optional, for tracking)
              "age": 35,
              "annual_income": 65000.0,
              "credit_score": 720,
              "loan_amount": 15000.0,
              "loan_term_months": 36,
              "employment_length_years": 5.0,
              "home_ownership_encoded": 1,
              "debt_to_income_ratio": 0.23,
              "num_credit_lines": 8,
              "payment_history_score": 85.0
            }
          ],
          "threshold": 0.5    (optional, overrides env default)
        }

    Response:
        {
          "predictions": [
            {
              "customer_id": "...",
              "default_probability": 0.12,
              "default_flag_predicted": 0,
              "threshold_used": 0.5
            }
          ],
          "model_name": "credit-risk-classifier",
          "model_version": "3",
          "model_alias": "production"
        }
"""

import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from model_loader import load_model

_SERVICES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SERVICES_ROOT not in sys.path:
    sys.path.insert(0, _SERVICES_ROOT)

from platform_config import FEATURE_COLUMNS  # noqa: E402
from schema import INFERENCE_FEATURE_SCHEMA  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# Model loaded once at startup
_model = None
_version_info = None


def get_model():
    global _model, _version_info
    if _model is None:
        _model, _version_info = load_model()
    return _model, _version_info


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    _, vi = get_model()
    return jsonify({
        "model_name": os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier"),
        "model_version": vi.version,
        "model_alias": os.getenv("MODEL_ALIAS", "production"),
        "run_id": vi.run_id,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    if not body or "records" not in body:
        return jsonify({"error": "Request body must contain 'records' list"}), 400

    records = body["records"]
    if not records:
        return jsonify({"error": "'records' list is empty"}), 400

    threshold = float(body.get("threshold", os.getenv("DECISION_THRESHOLD", 0.5)))

    df = pd.DataFrame(records)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing feature columns: {missing}"}), 400

    try:
        INFERENCE_FEATURE_SCHEMA.validate(df[FEATURE_COLUMNS])
    except Exception as exc:
        return jsonify({"error": f"Input validation failed: {exc}"}), 422

    model, vi = get_model()

    try:
        probabilities = model.predict(df[FEATURE_COLUMNS])
    except Exception as exc:
        log.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500

    # pyfunc models can return probabilities or class labels depending on flavor.
    # For sklearn classifiers logged with mlflow.sklearn, predict returns classes.
    # We need probabilities - re-run through the underlying model if needed.
    if hasattr(model._model_impl, "predict_proba"):
        proba = model._model_impl.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    else:
        proba = probabilities.astype(float)

    predictions = []
    for i, (prob, row) in enumerate(zip(proba, records)):
        predictions.append({
            "customer_id": row.get("customer_id", f"record_{i}"),
            "default_probability": round(float(prob), 6),
            "default_flag_predicted": int(prob >= threshold),
            "threshold_used": threshold,
        })

    return jsonify({
        "predictions": predictions,
        "model_name": os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier"),
        "model_version": vi.version,
        "model_alias": os.getenv("MODEL_ALIAS", "production"),
    }), 200


@app.route("/reload", methods=["POST"])
def reload_model():
    global _model, _version_info
    try:
        _model, _version_info = load_model()
        alias = os.getenv("MODEL_ALIAS", "production")
        log.info("Model reloaded: version=%s alias=%s", _version_info.version, alias)
        return jsonify({
            "status": "reloaded",
            "model_version": _version_info.version,
            "model_alias": alias,
        }), 200
    except Exception as exc:
        log.exception("Model reload failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    # Pre-load model on startup so first request is fast
    get_model()
    app.run(host="0.0.0.0", port=port)
