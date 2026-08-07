"""Deterministic entity-safe splitting."""

from __future__ import annotations

import random


def split_lightpaths(count: int = 756, seed: int = 42) -> dict[str, list[int]]:
    ids = list(range(count))
    random.Random(seed).shuffle(ids)
    train_end = round(count * 0.70)
    validation_end = train_end + round(count * 0.15)
    return {
        "train": sorted(ids[:train_end]),
        "validation": sorted(ids[train_end:validation_end]),
        "final_test": sorted(ids[validation_end:]),
    }


def assert_disjoint(split: dict[str, list[int]], expected_count: int = 756) -> None:
    sets = {name: set(values) for name, values in split.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"entity overlap between {left} and {right}: {sorted(overlap)[:5]}")
    union = set().union(*sets.values())
    if union != set(range(expected_count)):
        raise ValueError("split does not cover each lightpath exactly once")
