#!/usr/bin/env python3
"""Bounded historical telemetry replay through the production inference path."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from fiberguard.api import InferenceEngine, TelemetryInput  # noqa: E402
from fiberguard.data import discover_raw_files, iter_rows  # noqa: E402

SAMPLES_PER_CONDITION = 10
BLOCK_ROWS = 300
OUTPUT = ROOT / "artifacts" / "historical_replay.json"
RAW_DIR = ROOT / "data" / "raw" / "Optical network soft failure dataset"


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def main() -> None:
    split = json.loads((ROOT / "artifacts" / "split_manifest.json").read_text(encoding="utf-8"))["splits"]
    final_ids = set(split["final_test"])
    path = discover_raw_files(RAW_DIR)["test"]
    engine = InferenceEngine()
    records = []
    # Fixed, evenly spaced positions cover each condition without disturbing
    # the publisher's within-condition sequence order.
    selected_positions = set(np.linspace(0, BLOCK_ROWS - 1, SAMPLES_PER_CONDITION, dtype=int).tolist())
    for row_index, row in enumerate(iter_rows(path)):
        block_id, within_block = divmod(row_index, BLOCK_ROWS)
        lightpath_id = block_id // 4
        if lightpath_id not in final_ids or within_block not in selected_positions:
            continue
        payload = TelemetryInput(
            lp_length_km=row[1], laser_current_ma=row[2], lp_power_dbm=row[3], osnr_db=row[4], ber_db=row[5]
        )
        prediction = engine.predict(payload)
        records.append({
            "lightpath_id": lightpath_id,
            "sequence_index": row[0],
            "input": payload.model_dump(),
            **prediction,
            "actual_label": row[6],
        })
    expected_rows = len(final_ids) * 4 * SAMPLES_PER_CONDITION
    if len(records) != expected_rows:
        raise RuntimeError(f"found {len(records)} held-out replay rows, expected {expected_rows}")
    actual_names = {0: "healthy/no-failure", 1: "ECL", 2: "EDFA", 3: "NLI"}
    correct = [record["predicted_state"] == actual_names[record["actual_label"]] for record in records]
    latencies = [record["latency_ms"] for record in records]
    summary = {
        "description": "HISTORICAL TELEMETRY REPLAY; not live or production traffic",
        "rows": len(records),
        "lightpaths": len({record["lightpath_id"] for record in records}),
        "class_counts": {
            actual_names[label]: sum(record["actual_label"] == label for record in records)
            for label in actual_names
        },
        "selection": (
            f"{SAMPLES_PER_CONDITION} fixed evenly spaced observations from each of four ordered "
            "condition blocks for every final-test lightpath"
        ),
        "accuracy": statistics.fmean(correct),
        "review_rate": statistics.fmean(record["decision"] == "review" for record in records),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
