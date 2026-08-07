"""FastAPI inference contract for the locked FiberGuard XGBoost model."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
import mlflow
import mlflow.xgboost
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts" / "production_manifest.json"
REPLAY_PATH = ROOT / "artifacts" / "historical_replay.json"
STATIC_INDEX_PATH = ROOT / "src" / "fiberguard" / "static" / "index.html"
LABELS = {0: "healthy/no-failure", 1: "ECL", 2: "EDFA", 3: "NLI"}


class TelemetryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lp_length_km: float = Field(description="Lightpath length in kilometres")
    laser_current_ma: float = Field(description="ECL operating current in milliamps")
    lp_power_dbm: float = Field(description="Received lightpath optical power in dBm")
    osnr_db: float = Field(description="Optical signal-to-noise ratio in dB")
    ber_db: float = Field(description="BER measurement in dB")

    def as_array(self) -> np.ndarray:
        return np.asarray([[
            self.lp_length_km,
            self.laser_current_ma,
            self.lp_power_dbm,
            self.osnr_db,
            self.ber_db,
        ]], dtype=np.float32)


def _psi(actual: np.ndarray, edges: np.ndarray, expected: np.ndarray) -> float:
    counts, _ = np.histogram(actual, bins=edges)
    observed = counts / max(counts.sum(), 1)
    expected = np.clip(expected, 1e-6, None)
    observed = np.clip(observed, 1e-6, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def build_psi_reference(x_train: np.ndarray, feature_names: list[str], bins: int = 10) -> dict:
    reference = {"method": "PSI with 10 training-quantile bins", "warning_threshold": 0.20, "features": {}}
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for index, feature in enumerate(feature_names):
        internal = np.unique(np.quantile(x_train[:, index], quantiles))
        edges = np.concatenate(([-np.inf], internal, [np.inf]))
        counts, _ = np.histogram(x_train[:, index], bins=edges)
        reference["features"][feature] = {
            "edges": [None if not np.isfinite(value) else float(value) for value in edges],
            "expected_proportions": (counts / counts.sum()).tolist(),
        }
    return reference


def drift_from_replay(reference: dict, replay_path: Path = REPLAY_PATH) -> dict:
    if not replay_path.exists():
        return {"method": reference["method"], "status": "warning", "reason": "no historical replay window", "per_feature": {}}
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    rows = replay["records"]
    values = np.asarray([[row["input"][key] for key in (
        "lp_length_km", "laser_current_ma", "lp_power_dbm", "osnr_db", "ber_db"
    )] for row in rows], dtype=np.float64)
    per_feature = {}
    for index, (feature, item) in enumerate(reference["features"].items()):
        edges = np.asarray([-np.inf if value is None and edge_index == 0 else np.inf if value is None else value for edge_index, value in enumerate(item["edges"])])
        per_feature[feature] = _psi(values[:, index], edges, np.asarray(item["expected_proportions"]))
    threshold = float(reference["warning_threshold"])
    return {
        "method": reference["method"],
        "reference": "C2 training-lightpath rows",
        "window": "bounded held-out historical telemetry replay",
        "window_rows": len(rows),
        "warning_threshold": threshold,
        "per_feature": per_feature,
        "status": "warning" if any(value >= threshold for value in per_feature.values()) else "stable",
    }


class InferenceEngine:
    def __init__(self, manifest_path: Path = MANIFEST_PATH):
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mlflow.set_tracking_uri(self.manifest["mlflow_tracking_uri"])
        self.model = mlflow.xgboost.load_model(self.manifest["registered_model_uri"])
        self.threshold = float(self.manifest["raw_review_threshold"])
        self.version = str(self.manifest["model_version"])

    def predict(self, telemetry: TelemetryInput) -> dict[str, Any]:
        started = time.perf_counter()
        probabilities = self.model.predict_proba(telemetry.as_array())[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        return {
            "predicted_state": LABELS[prediction],
            "confidence": confidence,
            "decision": "review" if confidence < self.threshold else "confident",
            "model_version": self.version,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


def create_app() -> FastAPI:
    engine = InferenceEngine()
    reference = json.loads((ROOT / "artifacts" / "drift_reference.json").read_text(encoding="utf-8"))
    application = FastAPI(title="FiberGuard", version=engine.version)

    @application.get("/", include_in_schema=False)
    async def operator_view() -> FileResponse:
        return FileResponse(STATIC_INDEX_PATH, media_type="text/html")

    @application.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": "fiberguard-fault-classifier", "model_version": engine.version}

    @application.post("/predict")
    async def predict(telemetry: TelemetryInput) -> dict:
        return engine.predict(telemetry)

    @application.get("/drift")
    async def drift() -> dict:
        return drift_from_replay(reference)

    @application.get("/replay/example")
    async def replay_example() -> dict:
        replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
        examples = []
        for label in range(4):
            label_examples = [record for record in replay["records"] if record["actual_label"] == label]
            examples.extend(label_examples[:2])
        return {
            "description": "Historical telemetry replay; not live carrier traffic",
            "records": examples,
        }

    return application


app = create_app()
