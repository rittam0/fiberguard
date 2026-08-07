"""Validation of publisher-order-derived FiberGuard entity identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .data import iter_rows


class EntityValidationError(ValueError):
    """Raised when publisher ordering does not support entity reconstruction."""


@dataclass(frozen=True)
class BlockSummary:
    block_id: int
    lightpath_id: int
    label: int
    rows: int
    lp_length_km: float


@dataclass
class ScanResult:
    blocks: list[BlockSummary]
    arrays: dict[str, tuple[np.ndarray, np.ndarray]]


def scan_and_collect(
    path: str | Path,
    expected_block_rows: int,
    role_by_lightpath: Mapping[int, str] | None = None,
    expected_lightpaths: int = 756,
) -> ScanResult:
    """Validate block mechanics while collecting only requested entity roles."""
    role_by_lightpath = role_by_lightpath or {}
    expected_total = expected_lightpaths * 4 * expected_block_rows
    role_counts: dict[str, int] = {}
    for role in role_by_lightpath.values():
        role_counts[role] = role_counts.get(role, 0) + 4 * expected_block_rows
    arrays = {
        role: (np.empty((count, 5), dtype=np.float32), np.empty(count, dtype=np.int8))
        for role, count in role_counts.items()
    }
    offsets = {role: 0 for role in arrays}
    blocks: list[BlockSummary] = []
    current_label: int | None = None
    current_length: float | None = None
    current_rows = 0

    for row_index, row in enumerate(iter_rows(path)):
        if row_index >= expected_total:
            raise EntityValidationError(f"{path}: more than {expected_total} data rows")
        block_id, within_block = divmod(row_index, expected_block_rows)
        lightpath_id = block_id // 4
        timestamp, lp_length, *_, label = row
        expected_timestamp = within_block + 1
        if timestamp != expected_timestamp:
            raise EntityValidationError(
                f"{path}: block {block_id} row {within_block} has timestamp {timestamp}, expected {expected_timestamp}"
            )
        if within_block == 0:
            if current_label is not None:
                blocks.append(BlockSummary(block_id - 1, (block_id - 1) // 4, current_label, current_rows, current_length))
            current_label = label
            current_length = lp_length
            current_rows = 0
        elif label != current_label:
            raise EntityValidationError(f"{path}: label changes inside block {block_id}")
        elif lp_length != current_length:
            raise EntityValidationError(f"{path}: LP length changes inside block {block_id}")
        current_rows += 1

        role = role_by_lightpath.get(lightpath_id)
        if role is not None:
            offset = offsets[role]
            x, y = arrays[role]
            x[offset] = row[1:6]
            y[offset] = label
            offsets[role] = offset + 1

    actual_rows = sum(block.rows for block in blocks) + current_rows
    if actual_rows != expected_total:
        raise EntityValidationError(f"{path}: found {actual_rows} rows, expected {expected_total}")
    if current_label is not None:
        last_id = expected_lightpaths * 4 - 1
        blocks.append(BlockSummary(last_id, last_id // 4, current_label, current_rows, current_length))
    for role, offset in offsets.items():
        if offset != len(arrays[role][1]):
            raise EntityValidationError(f"{path}: collected {offset} {role} rows, expected {len(arrays[role][1])}")
    return ScanResult(blocks, arrays)


def validate_paired_scans(
    train: ScanResult,
    publisher_test: ScanResult,
    train_block_rows: int = 900,
    test_block_rows: int = 300,
    expected_lightpaths: int = 756,
) -> dict:
    """Strictly validate four-condition groups and cross-file positional links."""
    expected_blocks = expected_lightpaths * 4
    failures: list[str] = []
    if len(train.blocks) != expected_blocks or len(publisher_test.blocks) != expected_blocks:
        failures.append(f"block counts are train={len(train.blocks)}, test={len(publisher_test.blocks)}, expected={expected_blocks}")

    for name, scan, expected_rows in (("train", train, train_block_rows), ("publisher_test", publisher_test, test_block_rows)):
        for lightpath_id in range(expected_lightpaths):
            group = scan.blocks[lightpath_id * 4 : lightpath_id * 4 + 4]
            if len(group) != 4:
                failures.append(f"{name} lightpath {lightpath_id}: found {len(group)} blocks")
                continue
            labels = [block.label for block in group]
            if labels != [0, 1, 2, 3]:
                failures.append(f"{name} lightpath {lightpath_id}: labels={labels}")
            if any(block.rows != expected_rows for block in group):
                failures.append(f"{name} lightpath {lightpath_id}: block lengths={[b.rows for b in group]}")
            lengths = {block.lp_length_km for block in group}
            if len(lengths) != 1:
                failures.append(f"{name} lightpath {lightpath_id}: physical signatures={sorted(lengths)}")

    positional_matches = 0
    if len(train.blocks) == len(publisher_test.blocks) == expected_blocks:
        for lightpath_id in range(expected_lightpaths):
            train_length = train.blocks[lightpath_id * 4].lp_length_km
            test_length = publisher_test.blocks[lightpath_id * 4].lp_length_km
            if train_length == test_length:
                positional_matches += 1
            else:
                failures.append(f"lightpath position {lightpath_id}: train length {train_length} != test length {test_length}")

    unique_lengths = len({train.blocks[i * 4].lp_length_km for i in range(expected_lightpaths)}) if len(train.blocks) == expected_blocks else 0
    evidence = {
        "valid": not failures,
        "mapping": "block_id = zero-based contiguous timestamp-reset block; lightpath_id = block_id // 4",
        "identity_source": "derived from publisher file ordering; not stored in raw data",
        "lightpaths_checked": expected_lightpaths,
        "blocks_per_file": expected_blocks,
        "blocks_per_lightpath": 4,
        "required_label_order": [0, 1, 2, 3],
        "train_block_rows": train_block_rows,
        "publisher_test_block_rows": test_block_rows,
        "internally_constant_lp_length_blocks_train": sum(len({b.lp_length_km}) == 1 for b in train.blocks),
        "internally_constant_lp_length_blocks_test": sum(len({b.lp_length_km}) == 1 for b in publisher_test.blocks),
        "four_condition_signature_matches_train": sum(len({b.lp_length_km for b in train.blocks[i*4:i*4+4]}) == 1 for i in range(expected_lightpaths)),
        "four_condition_signature_matches_test": sum(len({b.lp_length_km for b in publisher_test.blocks[i*4:i*4+4]}) == 1 for i in range(expected_lightpaths)),
        "cross_file_positional_signature_matches": positional_matches,
        "unique_lp_lengths": unique_lengths,
        "failures": failures[:20],
    }
    if failures:
        raise EntityValidationError("entity reconstruction failed: " + "; ".join(failures[:3]))
    return evidence
