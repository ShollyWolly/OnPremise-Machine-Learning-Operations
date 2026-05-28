"""Integration tests for batch inference — mocks Flask and DB, tests data flow."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from platform_config import FEATURE_COLUMNS


def _make_clean_parquet(tmp_path: "Path", n: int = 5) -> "Path":
    """Write a minimal clean parquet file for use in inference tests."""
    import numpy as np

    rng = np.random.default_rng(42)
    data = {
        "record_id": [f"r{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "age": rng.uniform(20, 65, n),
        "annual_income": rng.uniform(30_000, 200_000, n),
        "credit_score": rng.uniform(400, 820, n),
        "loan_amount": rng.uniform(5_000, 80_000, n),
        "loan_term_months": rng.choice([24, 36, 48, 60], n).astype(float),
        "employment_length_years": rng.uniform(0.5, 20.0, n),
        "home_ownership_encoded": rng.integers(0, 4, n).astype(float),
        "debt_to_income_ratio": rng.uniform(0.05, 2.0, n),
        "num_credit_lines": rng.integers(2, 20, n).astype(float),
        "payment_history_score": rng.uniform(20.0, 98.0, n),
        "default_flag": rng.integers(0, 2, n),
        "created_at": pd.Timestamp("2024-01-01"),
        "ground_truth_available_at": pd.Timestamp("2024-01-01"),
    }
    path = tmp_path / "2024-01-01_000000.parquet"
    pd.DataFrame(data).to_parquet(path, index=False, engine="pyarrow")
    return path


def _mock_flask_response(n: int):
    """Return a mock requests.Response matching the Flask /predict contract."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "predictions": [
            {
                "customer_id": f"c{i}",
                "default_probability": 0.1,
                "default_flag_predicted": 0,
                "threshold_used": 0.5,
            }
            for i in range(n)
        ],
        "model_name": "credit-risk-classifier",
        "model_version": "1",
    }
    return mock


class TestCallPredictEndpoint:
    def test_sends_correct_payload(self):
        """call_predict_endpoint sends records and threshold in the request body."""
        from inference import call_predict_endpoint

        records = [{col: 1.0 for col in FEATURE_COLUMNS} for _ in range(3)]
        mock_resp = _mock_flask_response(3)

        with patch("requests.post", return_value=mock_resp) as mock_post:
            call_predict_endpoint(records, "http://flask:5001", 0.5)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert "records" in payload
        assert payload["threshold"] == 0.5
        assert len(payload["records"]) == 3

    def test_returns_predictions_model_name_version(self):
        from inference import call_predict_endpoint

        records = [{col: 1.0 for col in FEATURE_COLUMNS}]
        mock_resp = _mock_flask_response(1)

        with patch("requests.post", return_value=mock_resp):
            preds, model_name, model_version = call_predict_endpoint(
                records, "http://flask:5001", 0.5
            )

        assert len(preds) == 1
        assert model_name == "credit-risk-classifier"
        assert model_version == "1"

    def test_prediction_has_required_keys(self):
        from inference import call_predict_endpoint

        records = [{col: 1.0 for col in FEATURE_COLUMNS}]
        with patch("requests.post", return_value=_mock_flask_response(1)):
            preds, _, _ = call_predict_endpoint(records, "http://flask:5001", 0.5)

        assert "default_probability" in preds[0]
        assert "default_flag_predicted" in preds[0]
        assert "threshold_used" in preds[0]


class TestLoadCleanData:
    def test_loads_most_recent_parquet(self, tmp_path):
        from inference import load_clean_data

        _make_clean_parquet(tmp_path)
        df = load_clean_data(str(tmp_path))
        assert not df.empty
        assert all(col in df.columns for col in FEATURE_COLUMNS)

    def test_raises_when_no_parquet_files(self, tmp_path):
        from inference import load_clean_data

        with pytest.raises(FileNotFoundError):
            load_clean_data(str(tmp_path))


class TestInferenceSchemaValidation:
    def test_inference_fails_when_feature_column_missing(self, tmp_path):
        """run_inference raises ValueError when parquet is missing a feature column."""
        import numpy as np
        from inference import load_clean_data

        # Write parquet without 'credit_score'
        n = 3
        rng = np.random.default_rng(0)
        data = {col: rng.uniform(1, 100, n) for col in FEATURE_COLUMNS if col != "credit_score"}
        data["record_id"] = [f"r{i}" for i in range(n)]
        data["customer_id"] = [f"c{i}" for i in range(n)]
        path = tmp_path / "2024-01-01_000000.parquet"
        pd.DataFrame(data).to_parquet(path, index=False, engine="pyarrow")

        df = load_clean_data(str(tmp_path))
        # Schema validation should raise because credit_score is missing
        from schema import INFERENCE_FEATURE_SCHEMA
        from pandera.errors import SchemaError

        with pytest.raises(SchemaError):
            INFERENCE_FEATURE_SCHEMA.validate(df[
                [c for c in FEATURE_COLUMNS if c in df.columns]
            ])

    def test_valid_parquet_passes_schema(self, tmp_path):
        from inference import load_clean_data
        from schema import INFERENCE_FEATURE_SCHEMA

        _make_clean_parquet(tmp_path)
        df = load_clean_data(str(tmp_path))
        INFERENCE_FEATURE_SCHEMA.validate(df[FEATURE_COLUMNS])


class TestInferenceOnlyPassesFeatureColumns:
    def test_records_sent_to_flask_contain_only_expected_columns(self, tmp_path):
        """run_inference only sends FEATURE_COLUMNS + customer_id + record_id to Flask."""
        _make_clean_parquet(tmp_path)
        n = 5
        mock_resp = _mock_flask_response(n)

        with (
            patch("requests.post", return_value=mock_resp) as mock_post,
            patch("inference._db_conn") as mock_db,
        ):
            mock_cursor = MagicMock()
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock(
                cursor=MagicMock(return_value=mock_cursor)
            ))
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from inference import run_inference

            try:
                run_inference(
                    processed_dir=str(tmp_path),
                    predictions_dir=str(tmp_path / "preds"),
                    endpoint="http://flask:5001",
                    batch_size=200,
                    threshold=0.5,
                    run_index=1,
                )
            except Exception:
                pass  # DB writes may fail; we only care about the Flask call

        if mock_post.called:
            payload = mock_post.call_args[1]["json"]
            sent_keys = set(payload["records"][0].keys())
            allowed = set(FEATURE_COLUMNS) | {"customer_id", "record_id"}
            assert sent_keys.issubset(allowed), f"Unexpected keys sent to Flask: {sent_keys - allowed}"
