"""
Shared utilities for the Laptop Price Prediction service.

Both the Streamlit UI (app.py) and the FastAPI service (api.py) import from
here so there is exactly ONE place that knows how to load the model, build a
feature row, run a prediction, and log it. Duplicate logic between a notebook,
a UI, and an API is the #1 cause of "works in the demo, wrong in prod" bugs.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "laptop_price_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "laptop_price_preprocessor.pkl"
FEATURE_META_PATH = ARTIFACTS_DIR / "feature_meta.json"
COMPARISON_PATH = ARTIFACTS_DIR / "comparison_results.csv"
RESIDUAL_QUANTILES_PATH = ARTIFACTS_DIR / "residual_quantiles.json"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.csv"

NUMERICAL_COLS = [
    "release_year", "cpu_cores", "cpu_threads", "cpu_base_ghz", "cpu_boost_ghz",
    "vram_gb", "ram_gb", "storage_gb", "storage_drive_count", "display_size_in",
    "refresh_hz", "battery_wh", "charger_watts", "psu_watts", "weight_kg",
    "warranty_months", "gpu_generation", "resolution_pixels",
]

CATEGORICAL_COLS = [
    "device_type", "brand", "os", "form_factor", "cpu_brand", "cpu_tier",
    "gpu_brand", "gpu_tier", "storage_type", "display_type", "gpu_suffix",
    "wifi", "bluetooth",
]

ALL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# --------------------------------------------------------------------------- #
# Logging setup (application logs, not the prediction audit trail)
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("laptop_price")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class ModelLoadError(RuntimeError):
    """Raised when the model or preprocessor artifacts can't be loaded."""


class PredictionError(RuntimeError):
    """Raised when a prediction request can't be completed."""


# --------------------------------------------------------------------------- #
# Artifact loading (cached by caller — Streamlit uses st.cache_resource,
# FastAPI loads once at startup)
# --------------------------------------------------------------------------- #

def load_artifacts() -> dict[str, Any]:
    """Load model, preprocessor, and feature metadata from disk.

    Raises ModelLoadError with a clear message instead of letting a raw
    FileNotFoundError / joblib error bubble up to the user.
    """
    missing = [p for p in (MODEL_PATH, PREPROCESSOR_PATH, FEATURE_META_PATH) if not p.exists()]
    if missing:
        raise ModelLoadError(
            "Missing required artifact(s): "
            + ", ".join(str(p) for p in missing)
            + ". Re-run the notebook's save step or copy the .pkl files into artifacts/."
        )

    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as exc:  # noqa: BLE001 - we want to wrap *any* load failure
        raise ModelLoadError(f"Failed to load model/preprocessor: {exc}") from exc

    with open(FEATURE_META_PATH) as f:
        feature_meta = json.load(f)

    comparison_df = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else None

    residual_quantiles = None
    if RESIDUAL_QUANTILES_PATH.exists():
        with open(RESIDUAL_QUANTILES_PATH) as f:
            residual_quantiles = json.load(f)
    else:
        logger.warning(
            "No residual_quantiles.json found — predictions will not include an interval. "
            "Regenerate it from held-out test residuals (see README)."
        )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_meta": feature_meta,
        "comparison_df": comparison_df,
        "residual_quantiles": residual_quantiles,
        "model_version": _model_version_string(),
    }


def _model_version_string() -> str:
    """Best-effort version tag: file mtime of the model artifact.

    This is intentionally simple. If you want real versioning, write a
    artifacts/model_card.json at save time in the notebook with
    {"model_name": ..., "trained_at": ..., "metrics": {...}} and read that
    here instead.
    """
    if MODEL_PATH.exists():
        ts = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        return ts.strftime("%Y-%m-%d")
    return "unknown"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def validate_input(payload: dict[str, Any], feature_meta: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid).

    Checks: all required fields present, numerics within a sane range of the
    training distribution (a few % beyond min/max is allowed so the app isn't
    brittle, but wildly out-of-range values are rejected before they hit the
    model), categoricals are known values (OneHotEncoder would silently
    ignore-unknown, which hides typos — better to fail loudly here).
    """
    errors: list[str] = []

    for col in ALL_COLS:
        if col not in payload or payload[col] in (None, ""):
            errors.append(f"'{col}' is required.")

    for col, bounds in feature_meta.get("numerical", {}).items():
        if col not in payload or payload[col] in (None, ""):
            continue
        try:
            val = float(payload[col])
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a number.")
            continue
        lo, hi = bounds["min"], bounds["max"]
        margin = (hi - lo) * 0.25 if hi > lo else 1.0
        if val < lo - margin or val > hi + margin:
            errors.append(
                f"'{col}' = {val} is far outside the training range "
                f"[{lo}, {hi}]; prediction would be an unreliable extrapolation."
            )

    for col, allowed in feature_meta.get("categorical", {}).items():
        if col not in payload or payload[col] in (None, ""):
            continue
        # bluetooth/cpu_tier/gpu_tier are numeric-looking categoricals
        val = payload[col]
        allowed_str = [str(a) for a in allowed]
        if str(val) not in allowed_str:
            errors.append(f"'{col}' = {val!r} is not one of the known values {allowed}.")

    return errors


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict_price(
    payload: dict[str, Any],
    model,
    preprocessor,
    feature_meta: dict[str, Any],
    *,
    residual_quantiles: dict[str, float] | None = None,
    log_request: bool = True,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Run one prediction. Returns dict with price, interval, latency, and request id.

    payload keys must match ALL_COLS. Caller (UI/API layer) is responsible
    for calling validate_input() first if it wants pre-validation errors
    surfaced separately from prediction errors.

    If residual_quantiles is provided (p10/p90 of true-minus-predicted log
    price on held-out test data — see artifacts/residual_quantiles.json),
    the result includes an 80% prediction interval [price_low, price_high].
    This is an empirical interval from actual model error, not a formula —
    it widens/narrows automatically if the model gets better or worse.
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    start = time.perf_counter()

    try:
        row = pd.DataFrame([{col: payload[col] for col in ALL_COLS}])
        for col in NUMERICAL_COLS:
            row[col] = pd.to_numeric(row[col])

        X = preprocessor.transform(row)
        log_pred = model.predict(X)[0]
        price = float(np.expm1(log_pred))
        price = max(price, 0.0)  # a negative predicted price is a bug, not an answer

        price_low = price_high = None
        if residual_quantiles:
            price_low = max(float(np.expm1(log_pred + residual_quantiles["p10"])), 0.0)
            price_high = float(np.expm1(log_pred + residual_quantiles["p90"]))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed for request_id=%s", request_id)
        raise PredictionError(f"Could not generate a prediction: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    result = {
        "request_id": request_id,
        "predicted_price": round(price, 2),
        "price_low": round(price_low, 2) if price_low is not None else None,
        "price_high": round(price_high, 2) if price_high is not None else None,
        "interval_confidence": 0.80 if residual_quantiles else None,
        "latency_ms": round(latency_ms, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if log_request:
        _log_prediction(payload, result)

    return result


def predict_batch(
    df: pd.DataFrame,
    model,
    preprocessor,
    feature_meta: dict[str, Any],
    residual_quantiles: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Vectorized batch prediction for CSV upload. Returns df with new columns:
    predicted_price (+ price_low/price_high if residual_quantiles given) and,
    per-row, any validation error (row is skipped/NaN'd rather than crashing
    the whole batch).
    """
    out = df.copy()
    out["predicted_price"] = np.nan
    out["price_low"] = np.nan
    out["price_high"] = np.nan
    out["error"] = ""

    missing_cols = [c for c in ALL_COLS if c not in out.columns]
    if missing_cols:
        raise PredictionError(f"Uploaded file is missing required column(s): {missing_cols}")

    for idx, row in out.iterrows():
        payload = row[ALL_COLS].to_dict()
        errors = validate_input(payload, feature_meta)
        if errors:
            out.at[idx, "error"] = "; ".join(errors)
            continue
        try:
            result = predict_price(
                payload, model, preprocessor, feature_meta,
                residual_quantiles=residual_quantiles, log_request=False,
            )
            out.at[idx, "predicted_price"] = result["predicted_price"]
            out.at[idx, "price_low"] = result["price_low"]
            out.at[idx, "price_high"] = result["price_high"]
        except PredictionError as exc:
            out.at[idx, "error"] = str(exc)

    return out


def _log_prediction(payload: dict[str, Any], result: dict[str, Any]) -> None:
    """Append one row to logs/predictions.csv (the prediction audit trail).

    This is the cheapest possible version of "monitoring": every prediction
    is recorded with its inputs and output so you can later check for input
    drift (are people sending specs way outside training range?) or just
    demo that you thought about observability at all.
    """
    record = {**payload, **result}
    write_header = not PREDICTION_LOG_PATH.exists()
    pd.DataFrame([record]).to_csv(
        PREDICTION_LOG_PATH, mode="a", header=write_header, index=False
    )
