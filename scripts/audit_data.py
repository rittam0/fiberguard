#!/usr/bin/env python3
"""One-pass streaming audit of the FiberGuard publisher files."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiberguard.data import EXPECTED_HEADER, discover_raw_files, read_preamble  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "Optical network soft failure dataset"
OUTPUT = ROOT / "artifacts" / "data_audit.json"
LABELS = {0: "healthy/no-failure", 1: "ECL", 2: "EDFA", 3: "NLI"}


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def result(self) -> dict[str, float]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std_population": math.sqrt(self.m2 / self.n),
        }


def audit_file(path: Path) -> dict:
    metadata, header = read_preamble(path)
    stats = [RunningStats() for _ in header]
    missing = [0] * len(header)
    nonfinite = [0] * len(header)
    labels: Counter[int] = Counter()
    first_rows: list[list[int | float]] = []
    fingerprints: set[int] = set()
    duplicates = 0
    rows = 0
    prior_timestamp: int | None = None
    prior_label: int | None = None
    prior_length: float | None = None
    run_length = 0
    run_lengths: Counter[int] = Counter()
    run_keys: Counter[str] = Counter()
    timestamp_resets = 0
    reset_from_to: Counter[str] = Counter()
    timestamp_step_counts: Counter[int] = Counter()

    with path.open("r", encoding="ascii", newline=None) as handle:
        next(handle)
        next(handle)
        for line_number, raw_line in enumerate(handle, start=3):
            fields = raw_line.split()
            if len(fields) != 7:
                raise ValueError(f"{path}:{line_number}: expected 7 fields, found {len(fields)}")
            for index, token in enumerate(fields):
                if token.lower() in {"", "na", "nan", "null", "none"}:
                    missing[index] += 1
            try:
                row = [int(fields[0]), *(float(v) for v in fields[1:6]), int(fields[6])]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric row") from exc
            if row[6] not in LABELS:
                raise ValueError(f"{path}:{line_number}: invalid label {row[6]}")
            rows += 1
            if len(first_rows) < 10:
                first_rows.append(row)
            labels[row[6]] += 1
            for index, value in enumerate(row):
                if math.isfinite(value):
                    stats[index].add(value)
                else:
                    nonfinite[index] += 1

            digest = hashlib.blake2b(raw_line.rstrip(b"\r\n") if isinstance(raw_line, bytes) else raw_line.rstrip("\r\n").encode("ascii"), digest_size=8).digest()
            fingerprint = int.from_bytes(digest)
            if fingerprint in fingerprints:
                duplicates += 1
            else:
                fingerprints.add(fingerprint)

            timestamp, length, label = row[0], row[1], row[6]
            if prior_timestamp is not None:
                step = timestamp - prior_timestamp
                timestamp_step_counts[step] += 1
                if timestamp <= prior_timestamp:
                    timestamp_resets += 1
                    reset_from_to[f"{prior_timestamp}->{timestamp}"] += 1
            key = (length, label)
            prior_key = (prior_length, prior_label)
            if prior_label is None or key == prior_key:
                run_length += 1
            else:
                run_lengths[run_length] += 1
                run_keys[f"length={prior_length:g},label={prior_label}"] += 1
                run_length = 1
            prior_timestamp, prior_length, prior_label = timestamp, length, label

    if run_length:
        run_lengths[run_length] += 1
        run_keys[f"length={prior_length:g},label={prior_label}"] += 1
    return {
        "metadata": metadata,
        "header": list(header),
        "rows": rows,
        "first_10_rows": first_rows,
        "dtypes": ["int64", "float64", "float64", "float64", "float64", "float64", "int64"],
        "class_counts": {str(key): labels[key] for key in sorted(labels)},
        "numeric_summary": {name: stat.result() for name, stat in zip(header, stats)},
        "missing_values": {name: count for name, count in zip(header, missing)},
        "nonfinite_values": {name: count for name, count in zip(header, nonfinite)},
        "duplicate_rows": duplicates,
        "order_evidence": {
            "timestamp_resets": timestamp_resets,
            "reset_transitions": dict(reset_from_to),
            "timestamp_step_counts": {str(k): v for k, v in sorted(timestamp_step_counts.items())},
            "contiguous_length_label_run_lengths": {str(k): v for k, v in sorted(run_lengths.items())},
            "contiguous_runs": sum(run_lengths.values()),
            "distinct_length_label_keys": len(run_keys),
        },
    }


def main() -> None:
    files = discover_raw_files(RAW_DIR)
    audited = {split: audit_file(path) for split, path in files.items()}
    report = {
        "files": {split: {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size} for split, path in files.items()},
        "rows": {split: item["rows"] for split, item in audited.items()},
        "columns": {"count": 7, "names": list(EXPECTED_HEADER), "delimiter": "whitespace", "header": True, "metadata_lines_before_header": 1},
        "schema": {"dtypes": dict(zip(EXPECTED_HEADER, audited["train"]["dtypes"])), "telemetry_columns": list(EXPECTED_HEADER[1:6]), "target_column": EXPECTED_HEADER[6]},
        "label_encoding": {"column": "Failure type", "storage": "integer", "mapping": {str(k): v for k, v in LABELS.items()}, "source": audited["train"]["metadata"]},
        "entity_encoding": {"lightpath_id_explicit": False, "evidence": "No entity-ID column exists. LP length repeats (only 1,440 distinct length-label keys across 3,024 blocks) and is not a unique identifier."},
        "order_encoding": {"timestamp_explicit": True, "column": "Time stamp", "scope": "within-block sample index; not a global timestamp", "likely_block_structure": "756 ordered lightpaths, each represented by four consecutive condition blocks in label order 0,1,2,3; each train block has 900 rows and each test block 300 rows", "identity_status": "The four-block grouping and correspondence between train/test are implied solely by row order.", "evidence_by_file": {k: v["order_evidence"] for k, v in audited.items()}},
        "class_distribution": {k: v["class_counts"] for k, v in audited.items()},
        "numeric_summary": {k: v["numeric_summary"] for k, v in audited.items()},
        "missing_values": {k: v["missing_values"] for k, v in audited.items()},
        "nonfinite_values": {k: v["nonfinite_values"] for k, v in audited.items()},
        "duplicate_rows": {k: v["duplicate_rows"] for k, v in audited.items()},
        "assumptions": [
            "A future reconstruction would have to assume each timestamp reset starts a new lightpath-condition block.",
            "It would have to assume publisher row ordering is stable and complete within every block.",
            "It would have to assume blocks can be associated across the four labels and train/test splits; the raw files do not encode that association."
        ],
        "warnings": [
            "Do not use LP length as lightpath identity: it is a measurement and may not be unique.",
            "Time stamp is a within-block sample index, not an absolute time value.",
            "Any inferred entity identity is vulnerable to row reordering, missing blocks, or undocumented publisher ordering."
        ],
        "first_10_rows": {k: v["first_10_rows"] for k, v in audited.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "class_distribution": report["class_distribution"], "duplicate_rows": report["duplicate_rows"], "order_encoding": report["order_encoding"]}, indent=2))


if __name__ == "__main__":
    main()
