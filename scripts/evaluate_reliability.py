#!/usr/bin/env python3
"""Run the single C3 feature-ablation and reliability experiment."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from fiberguard.baselines import models  # noqa: E402
from fiberguard.data import discover_raw_files  # noqa: E402
from fiberguard.entities import scan_and_collect  # noqa: E402
from fiberguard.reliability import (  # noqa: E402
    abstention_metrics,
    probability_metrics,
    reliability_bins,
    select_review_threshold,
)

FEATURES = ["LP length (km)", "Laser current (mA)", "LP power (dBm)", "OSNR (dB)", "BER (dB)"]
RAW_DIR = ROOT / "data" / "raw" / "Optical network soft failure dataset"
SPLIT_PATH = ROOT / "artifacts" / "split_manifest.json"
ABLATION_PATH = ROOT / "artifacts" / "feature_ablation.json"
CALIBRATION_PATH = ROOT / "artifacts" / "calibration_metrics.json"
CURVE_PATH = ROOT / "artifacts" / "reliability_curve.png"


def new_xgboost():
    return models(seed=42)["xgboost"]


def fit_ablation(x_train, y_train, x_test, y_test, indices):
    model = new_xgboost()
    started = time.perf_counter()
    model.fit(x_train[:, indices], y_train)
    train_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(x_test[:, indices])
    macro_f1 = float(f1_score(y_test, probabilities.argmax(axis=1), labels=[0, 1, 2, 3], average="macro"))
    return model, probabilities, {"features": [FEATURES[i] for i in indices], "final_test_macro_f1": macro_f1, "train_seconds": train_seconds}


def plot_reliability(y_validation, validation_sets, y_test, test_sets):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, title, targets, sets in (
        (axes[0], "Validation", y_validation, validation_sets),
        (axes[1], "Final test", y_test, test_sets),
    ):
        axis.plot([0, 1], [0, 1], "--", color="0.5", label="ideal")
        for label, probabilities in sets.items():
            points = reliability_bins(targets, probabilities)
            axis.plot([p["mean_confidence"] for p in points], [p["accuracy"] for p in points], marker="o", label=label)
        axis.set(title=title, xlabel="Mean maximum probability", ylabel="Observed accuracy", xlim=(0.45, 1.0), ylim=(0.45, 1.0))
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("FiberGuard XGBoost reliability (15 equal-width bins)")
    figure.savefig(CURVE_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split = manifest["splits"]
    train_roles = {entity: "train" for entity in split["train"]}
    train_roles.update({entity: "validation" for entity in split["validation"]})
    test_roles = {entity: "final_test" for entity in split["final_test"]}
    paths = discover_raw_files(RAW_DIR)
    train_scan = scan_and_collect(paths["train"], 900, train_roles)
    test_scan = scan_and_collect(paths["test"], 300, test_roles)
    x_train, y_train = train_scan.arrays["train"]
    x_validation, y_validation = train_scan.arrays["validation"]
    x_test, y_test = test_scan.arrays["final_test"]

    all_indices = list(range(5))
    all_model, uncalibrated_test, all_result = fit_ablation(x_train, y_train, x_test, y_test, all_indices)
    uncalibrated_validation = all_model.predict_proba(x_validation)
    print("all_features", all_result)

    singles = {}
    for index, feature in enumerate(FEATURES):
        model, _, result = fit_ablation(x_train, y_train, x_test, y_test, [index])
        singles[feature] = result
        print("single", feature, result)
        del model
        gc.collect()

    leave_one_out = {}
    for omitted, feature in enumerate(FEATURES):
        indices = [index for index in all_indices if index != omitted]
        model, _, result = fit_ablation(x_train, y_train, x_test, y_test, indices)
        leave_one_out[feature] = result
        print("leave_one_out", feature, result)
        del model
        gc.collect()

    ablation = {
        "split_seed": manifest["seed"],
        "xgboost_configuration": all_model.get_params(),
        "all_features": all_result,
        "single_feature": singles,
        "leave_one_feature_out": leave_one_out,
    }
    ABLATION_PATH.write_text(json.dumps(ablation, indent=2) + "\n", encoding="utf-8")

    calibrated = CalibratedClassifierCV(FrozenEstimator(all_model), method="sigmoid")
    calibrated.fit(x_validation, y_validation)
    calibrated_validation = calibrated.predict_proba(x_validation)
    calibrated_test = calibrated.predict_proba(x_test)
    threshold, validation_selection = select_review_threshold(y_validation, calibrated_validation)
    calibration = {
        "method": "sigmoid via CalibratedClassifierCV(FrozenEstimator), fitted on validation only",
        "uncalibrated": {
            "validation": probability_metrics(y_validation, uncalibrated_validation),
            "final_test": probability_metrics(y_test, uncalibrated_test),
        },
        "calibrated": {
            "validation": probability_metrics(y_validation, calibrated_validation),
            "final_test": probability_metrics(y_test, calibrated_test),
        },
        "threshold_selection": validation_selection,
        "final_test_abstention": abstention_metrics(y_test, calibrated_test, threshold),
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    plot_reliability(
        y_validation,
        {"uncalibrated": uncalibrated_validation, "sigmoid": calibrated_validation},
        y_test,
        {"uncalibrated": uncalibrated_test, "sigmoid": calibrated_test},
    )
    print("calibration", json.dumps(calibration))


if __name__ == "__main__":
    main()
