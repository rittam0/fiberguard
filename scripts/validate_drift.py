#!/usr/bin/env python3
"""Validate PSI on representative replay and one controlled feature shift."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiberguard.api import drift_from_replay  # noqa: E402

REPLAY_PATH = ROOT / "artifacts" / "historical_replay.json"
REFERENCE_PATH = ROOT / "artifacts" / "drift_reference.json"
OUTPUT_PATH = ROOT / "artifacts" / "drift_validation.json"
SHIFTED_FIELD = "lp_power_dbm"
SHIFTED_FEATURE = "LP power (dBm)"
SHIFT_DBM = 2.0


def main() -> None:
    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    records = replay["records"]
    class_names = {0: "healthy/no-failure", 1: "ECL", 2: "EDFA", 3: "NLI"}

    base = drift_from_replay(reference, REPLAY_PATH)
    shifted = deepcopy(replay)
    for record in shifted["records"]:
        record["input"][SHIFTED_FIELD] += SHIFT_DBM

    with tempfile.TemporaryDirectory(prefix="fiberguard-drift-") as directory:
        shifted_path = Path(directory) / "controlled_shift.json"
        shifted_path.write_text(json.dumps(shifted), encoding="utf-8")
        shifted_result = drift_from_replay(reference, shifted_path)

    before = base["per_feature"][SHIFTED_FEATURE]
    after = shifted_result["per_feature"][SHIFTED_FEATURE]
    materially_higher = after > before and after - before >= 0.20
    conclusion = (
        "The representative held-out replay provides the lower-drift baseline; the controlled "
        f"+{SHIFT_DBM:.1f} dBm optical-power shift increases that feature's PSI from "
        f"{before:.6f} to {after:.6f} and "
        + ("triggers the configured warning." if shifted_result["status"] == "warning" else "does not trigger the configured warning.")
    )
    result = {
        "name": "C5 representative replay and controlled drift validation",
        "representative_sample": {
            "source": "held-out final-test lightpaths from the publisher test file",
            "training_or_threshold_tuning_use": False,
            "rows": len(records),
            "lightpaths": len({record["lightpath_id"] for record in records}),
            "class_counts": {
                class_names[label]: sum(record["actual_label"] == label for record in records)
                for label in class_names
            },
            "selection": replay["summary"]["selection"],
            "sequence_order_preserved": True,
        },
        "base_drift_result": base,
        "controlled_drift_injection": {
            "name": "CONTROLLED DRIFT INJECTION",
            "feature": SHIFTED_FEATURE,
            "description": (
                f"Added a deterministic +{SHIFT_DBM:.1f} dBm to optical power on a copy of every "
                "representative replay row; labels were unchanged."
            ),
            "labels_modified": False,
            "result": shifted_result,
            "feature_psi_before": before,
            "feature_psi_after": after,
            "feature_psi_increase": after - before,
            "materially_higher": materially_higher,
        },
        "conclusion": conclusion,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
