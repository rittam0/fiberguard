#!/usr/bin/env python3
"""Run the single C2 entity-safe raw-feature baseline experiment."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402

from fiberguard.baselines import evaluate, fit_and_measure, models  # noqa: E402
from fiberguard.data import discover_raw_files  # noqa: E402
from fiberguard.entities import scan_and_collect, validate_paired_scans  # noqa: E402
from fiberguard.split import assert_disjoint, split_lightpaths  # noqa: E402

SEED = 42
RAW_DIR = ROOT / "data" / "raw" / "Optical network soft failure dataset"
SPLIT_PATH = ROOT / "artifacts" / "split_manifest.json"
METRICS_PATH = ROOT / "artifacts" / "baseline_metrics.json"


def counts(y: np.ndarray) -> dict[str, int]:
    values = Counter(int(value) for value in y)
    return {str(label): values[label] for label in range(4)}


def main() -> None:
    paths = discover_raw_files(RAW_DIR)
    split = split_lightpaths(seed=SEED)
    assert_disjoint(split)
    train_roles = {entity: "train" for entity in split["train"]}
    train_roles.update({entity: "validation" for entity in split["validation"]})
    test_roles = {entity: "final_test" for entity in split["final_test"]}

    train_scan = scan_and_collect(paths["train"], 900, train_roles)
    test_scan = scan_and_collect(paths["test"], 300, test_roles)
    evidence = validate_paired_scans(train_scan, test_scan)

    x_train, y_train = train_scan.arrays["train"]
    x_validation, y_validation = train_scan.arrays["validation"]
    x_test, y_test = test_scan.arrays["final_test"]
    class_counts = {
        "train": counts(y_train),
        "validation": counts(y_validation),
        "final_test": counts(y_test),
    }
    if any(len(set(value.values())) != 1 for value in class_counts.values()):
        raise RuntimeError(f"entity split is not label-balanced: {class_counts}")

    manifest = {
        "seed": SEED,
        "identity_source": "reconstructed from publisher order after strict validation; not present in raw schema",
        "mapping": evidence["mapping"],
        "splits": split,
        "lightpath_counts": {name: len(ids) for name, ids in split.items()},
        "row_sources": {"train": "publisher training file", "validation": "publisher training file", "final_test": "publisher test file"},
        "row_counts": {"train": len(y_train), "validation": len(y_validation), "final_test": len(y_test)},
        "class_counts": class_counts,
        "entity_overlap": 0,
        "entity_validation": evidence,
    }
    SPLIT_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logreg_stride = 10
    x_logreg, y_logreg = x_train[::logreg_stride], y_train[::logreg_stride]
    results = {}
    for name, model in models(SEED).items():
        fit_x, fit_y = (x_logreg, y_logreg) if name == "logistic_regression" else (x_train, y_train)
        fitted, train_seconds = fit_and_measure(model, fit_x, fit_y)
        results[name] = {
            "train_rows_used": len(fit_y),
            "train_seconds": train_seconds,
            "validation": evaluate(fitted, x_validation, y_validation),
            "final_test": evaluate(fitted, x_test, y_test),
        }
        print(name, json.dumps({"train_seconds": train_seconds, "validation_macro_f1": results[name]["validation"]["macro_f1"], "test_macro_f1": results[name]["final_test"]["macro_f1"]}))

    best = max(results, key=lambda name: results[name]["validation"]["macro_f1"])
    metrics = {
        "experiment": "C2 raw-feature entity-safe baselines",
        "seed": SEED,
        "features": ["LP length (km)", "Laser current (mA)", "LP power (dBm)", "OSNR (dB)", "BER (dB)"],
        "excluded_inputs": ["Time stamp", "lightpath_id", "block_id"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "training_rows_available": len(y_train),
        "logistic_regression_subsample": {"method": "every 10th row from eligible training-lightpath rows only", "rows": len(y_logreg), "validation_or_test_subsampled": False},
        "models": results,
        "best_raw_model_by_validation_macro_f1": best,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
