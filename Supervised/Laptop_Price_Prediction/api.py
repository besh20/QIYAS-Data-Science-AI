"""
FastAPI service for the Laptop Price Predictor.

Run locally:
    uvicorn api:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive Swagger docs.

This exists alongside the Streamlit app because a portfolio project that only
has a UI reads as "a demo"; one that also exposes a versioned, validated REST
API reads as "something another service could actually call."
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import (
    ModelLoadError,
    PredictionError,
    load_artifacts,
    predict_price,
    validate_input,
)

_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _state.update(load_artifacts())
    except ModelLoadError as exc:
        # Fail fast and loud at startup rather than 500-ing on first request.
        raise RuntimeError(str(exc)) from exc
    yield
    _state.clear()


app = FastAPI(
    title="Laptop Price Predictor API",
    version="1.0.0",
    description="Predicts device price (USD) from hardware/software specs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to known origins before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class DeviceSpecs(BaseModel):
    device_type: Literal["Desktop", "Laptop"]
    brand: str
    os: str
    form_factor: str
    release_year: int = Field(ge=2000, le=2100)
    warranty_months: int = Field(ge=0)
    cpu_brand: str
    cpu_tier: int = Field(ge=1, le=6)
    cpu_cores: int = Field(ge=1)
    cpu_threads: int = Field(ge=1)
    cpu_base_ghz: float = Field(gt=0)
    cpu_boost_ghz: float = Field(gt=0)
    gpu_brand: str
    gpu_tier: int = Field(ge=1, le=6)
    vram_gb: int = Field(ge=0)
    gpu_generation: int = Field(ge=0)
    gpu_suffix: str = "None"
    ram_gb: int = Field(gt=0)
    storage_type: str
    storage_gb: int = Field(gt=0)
    storage_drive_count: int = Field(ge=1)
    display_type: str
    display_size_in: float = Field(gt=0)
    resolution_pixels: int = Field(gt=0)
    refresh_hz: int = Field(gt=0)
    battery_wh: int = Field(ge=0)
    charger_watts: int = Field(ge=0)
    psu_watts: int = Field(ge=0)
    weight_kg: float = Field(gt=0)
    wifi: str
    bluetooth: float

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    request_id: str
    predicted_price: float
    price_low: float | None = None
    price_high: float | None = None
    interval_confidence: float | None = None
    latency_ms: float
    timestamp: str
    model_version: str


@app.get("/health")
def health():
    """Liveness/readiness probe. Returns 200 only if the model is loaded."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": _state["model_version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(specs: DeviceSpecs):
    payload = specs.model_dump()

    errors = validate_input(payload, _state["feature_meta"])
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    try:
        result = predict_price(
            payload, _state["model"], _state["preprocessor"], _state["feature_meta"],
            residual_quantiles=_state["residual_quantiles"],
        )
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {**result, "model_version": _state["model_version"]}
