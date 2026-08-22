"""
Run with:  pytest -v

These tests don't retrain anything — they exercise the saved artifacts, which
is exactly what would break silently in production (missing file, schema
drift between training and serving, preprocessor/model mismatch, etc).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from utils import ALL_COLS, load_artifacts, predict_price, validate_input

VALID_PAYLOAD = {
    "device_type": "Laptop", "brand": "Dell", "os": "Windows",
    "form_factor": "Ultrabook", "release_year": 2024, "warranty_months": 24,
    "cpu_brand": "Intel", "cpu_tier": 4, "cpu_cores": 8, "cpu_threads": 16,
    "cpu_base_ghz": 2.6, "cpu_boost_ghz": 4.2, "gpu_brand": "NVIDIA",
    "gpu_tier": 3, "vram_gb": 6, "gpu_generation": 40, "gpu_suffix": "None",
    "ram_gb": 16, "storage_type": "SSD", "storage_gb": 512,
    "storage_drive_count": 1, "display_type": "IPS", "display_size_in": 15.6,
    "resolution_pixels": 2073600, "refresh_hz": 144, "battery_wh": 56,
    "charger_watts": 65, "psu_watts": 0, "weight_kg": 1.8, "wifi": "Wi-Fi 6",
    "bluetooth": 5.2,
}


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_artifacts_load(artifacts):
    assert artifacts["model"] is not None
    assert artifacts["preprocessor"] is not None
    assert set(artifacts["feature_meta"]["numerical"]) | set(artifacts["feature_meta"]["categorical"])


def test_valid_payload_has_no_errors(artifacts):
    errors = validate_input(VALID_PAYLOAD, artifacts["feature_meta"])
    assert errors == []


def test_missing_field_is_rejected(artifacts):
    bad = VALID_PAYLOAD.copy()
    del bad["ram_gb"]
    errors = validate_input(bad, artifacts["feature_meta"])
    assert any("ram_gb" in e for e in errors)


def test_unknown_category_is_rejected(artifacts):
    bad = VALID_PAYLOAD.copy()
    bad["brand"] = "TotallyMadeUpBrand"
    errors = validate_input(bad, artifacts["feature_meta"])
    assert any("brand" in e for e in errors)


def test_out_of_range_numeric_is_rejected(artifacts):
    bad = VALID_PAYLOAD.copy()
    bad["cpu_cores"] = 100_000
    errors = validate_input(bad, artifacts["feature_meta"])
    assert any("cpu_cores" in e for e in errors)


def test_prediction_is_positive_and_reasonable(artifacts):
    result = predict_price(
        VALID_PAYLOAD, artifacts["model"], artifacts["preprocessor"],
        artifacts["feature_meta"], log_request=False,
    )
    assert result["predicted_price"] > 0
    # sanity band, not a strict spec: a mid-range 2024 laptop shouldn't price
    # like a supercomputer or come back as $0
    assert 100 < result["predicted_price"] < 20000


def test_prediction_interval_brackets_point_estimate(artifacts):
    result = predict_price(
        VALID_PAYLOAD, artifacts["model"], artifacts["preprocessor"],
        artifacts["feature_meta"], residual_quantiles=artifacts["residual_quantiles"],
        log_request=False,
    )
    assert result["price_low"] is not None
    assert result["price_high"] is not None
    assert result["price_low"] < result["predicted_price"] < result["price_high"]
    assert result["interval_confidence"] == 0.80


def test_prediction_without_quantiles_has_no_interval(artifacts):
    result = predict_price(
        VALID_PAYLOAD, artifacts["model"], artifacts["preprocessor"],
        artifacts["feature_meta"], residual_quantiles=None, log_request=False,
    )
    assert result["price_low"] is None
    assert result["price_high"] is None


def test_all_cols_covers_payload_schema():
    assert set(ALL_COLS) == set(VALID_PAYLOAD.keys())


def test_api_health_and_predict():
    from api import app  # imported here so a broken api.py doesn't break other tests

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_price"] > 0

        bad_resp = client.post("/predict", json={**VALID_PAYLOAD, "cpu_cores": -5})
        assert bad_resp.status_code == 422
