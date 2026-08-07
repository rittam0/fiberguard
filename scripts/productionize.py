#!/usr/bin/env python3
"""Train once, track, register, and freeze the locked C2 model."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
import numpy as np  # noqa: E402

from fiberguard.baselines import models  # noqa: E402
from fiberguard.data import discover_raw_files  # noqa: E402
from fiberguard.entities import scan_and_collect  # noqa: E402
from fiberguard.reliability import probability_metrics, select_review_threshold  # noqa: E402

MODEL_NAME = "fiberguard-fault-classifier"
FEATURES = ["LP length (km)", "Laser current (mA)", "LP power (dBm)", "OSNR (dB)", "BER (dB)"]
RAW_DIR = ROOT / "data" / "raw" / "Optical network soft failure dataset"
MLFLOW_DB = ROOT / "artifacts" / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "artifacts" / "mlflow-artifacts"
MANIFEST_PATH = ROOT / "artifacts" / "production_manifest.json"
DRIFT_PATH = ROOT / "artifacts" / "drift_reference.json"


def build_psi_reference(x_train, feature_names, bins=10):
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


def main() -> None:
    tracking_uri = f"sqlite:///{MLFLOW_DB.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("FiberGuard production model")
    if experiment is None:
        experiment_id = client.create_experiment(
            "FiberGuard production model", artifact_location=MLFLOW_ARTIFACTS.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    split_manifest = json.loads((ROOT / "artifacts" / "split_manifest.json").read_text(encoding="utf-8"))
    split = split_manifest["splits"]
    train_roles = {entity: "train" for entity in split["train"]}
    train_roles.update({entity: "validation" for entity in split["validation"]})
    test_roles = {entity: "final_test" for entity in split["final_test"]}
    paths = discover_raw_files(RAW_DIR)
    train_scan = scan_and_collect(paths["train"], 900, train_roles)
    test_scan = scan_and_collect(paths["test"], 300, test_roles)
    x_train, y_train = train_scan.arrays["train"]
    x_validation, y_validation = train_scan.arrays["validation"]
    x_test, y_test = test_scan.arrays["final_test"]

    model = models(seed=42)["xgboost"]
    started = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started
    validation_probabilities = model.predict_proba(x_validation)
    test_probabilities = model.predict_proba(x_test)
    validation_metrics = probability_metrics(y_validation, validation_probabilities)
    test_metrics = probability_metrics(y_test, test_probabilities)
    threshold, threshold_evidence = select_review_threshold(y_validation, validation_probabilities)

    drift_reference = build_psi_reference(x_train, FEATURES)
    DRIFT_PATH.write_text(json.dumps(drift_reference, indent=2) + "\n", encoding="utf-8")

    with mlflow.start_run(run_name="locked-xgboost-raw-features") as run:
        run_id = run.info.run_id
        mlflow.set_tags({
            "model_type": "XGBClassifier",
            "model_name": MODEL_NAME,
            "split_methodology": "C2-seed-42-publisher-order-lightpath-entity-safe",
            "calibration": "uncalibrated; sigmoid tested and rejected in C3",
        })
        mlflow.log_params({
            "training_rows": len(y_train),
            "features": "|".join(FEATURES),
            "split_seed": split_manifest["seed"],
            **{f"xgb_{key}": value for key, value in model.get_params().items() if value is not None},
        })
        tracked_metrics = {
            "validation_macro_f1": validation_metrics["macro_f1"],
            "final_test_macro_f1": test_metrics["macro_f1"],
            "final_test_log_loss": test_metrics["log_loss"],
            "final_test_brier_score": test_metrics["multiclass_brier_score"],
            "training_duration_seconds": train_seconds,
        }
        mlflow.log_metrics(tracked_metrics)
        mlflow.log_artifact(ROOT / "artifacts" / "split_manifest.json", artifact_path="evaluation")
        signature = infer_signature(x_train[:20], model.predict_proba(x_train[:20]))
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=x_train[:2],
        )

    versions = [item for item in client.search_model_versions(f"name='{MODEL_NAME}'") if item.run_id == run_id]
    if len(versions) != 1:
        raise RuntimeError(f"expected one registered version for run {run_id}, found {len(versions)}")
    version = str(versions[0].version)
    artifact_uri = f"runs:/{run_id}/model"
    logged_model_id = model_info.model_uri.rsplit("/", 1)[-1]
    local_artifact = MLFLOW_ARTIFACTS / "models" / logged_model_id / "artifacts"
    if not (local_artifact / "MLmodel").exists():
        raise RuntimeError(f"durable MLflow model artifact not found at {local_artifact}")
    manifest = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "run_id": run_id,
        "mlflow_tracking_uri": tracking_uri,
        "registered_model_uri": f"models:/{MODEL_NAME}/{version}",
        "model_artifact_uri": artifact_uri,
        "local_model_artifact": str(local_artifact),
        "features": FEATURES,
        "raw_review_threshold": threshold,
        "review_threshold_source": "C2 validation rows, raw XGBoost maximum probabilities only",
        "threshold_validation_evidence": threshold_evidence,
        "tracked_metrics": tracked_metrics,
        "model_info_uri": model_info.model_uri,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
