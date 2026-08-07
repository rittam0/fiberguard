"""Reliable, streaming access to the FiberGuard raw text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Iterator, Sequence

EXPECTED_HEADER = (
    "Time stamp",
    "LP length (km)",
    "Laser current (mA)",
    "LP power (dBm)",
    "OSNR (dB)",
    "BER (dB)",
    "Failure type",
)
EXPECTED_METADATA_PREFIX = "failure_description ="


@dataclass(frozen=True)
class RawSchema:
    columns: tuple[str, ...] = EXPECTED_HEADER
    delimiter: str = "whitespace"
    metadata_lines: int = 1
    header_lines: int = 1


def discover_raw_files(raw_dir: str | Path) -> dict[str, Path]:
    """Find exactly one publisher train file and one publisher test file."""
    directory = Path(raw_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {directory}")
    result: dict[str, Path] = {}
    for split in ("train", "test"):
        matches = sorted(directory.glob(f"*_{split}_*.txt"))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {split} .txt file in {directory}, found {len(matches)}"
            )
        result[split] = matches[0]
    return result


def read_preamble(path: str | Path) -> tuple[str, tuple[str, ...]]:
    """Read and validate the metadata line and quoted whitespace header."""
    with Path(path).open("r", encoding="ascii", newline=None) as handle:
        metadata = handle.readline().strip()
        header_line = handle.readline().strip()
    if not metadata.startswith(EXPECTED_METADATA_PREFIX):
        raise ValueError(f"unexpected metadata line in {path!s}: {metadata!r}")
    try:
        header = tuple(shlex.split(header_line))
    except ValueError as exc:
        raise ValueError(f"invalid quoted header in {path!s}") from exc
    if header != EXPECTED_HEADER:
        raise ValueError(f"schema mismatch in {path!s}: {header!r}")
    return metadata, header


def _parse_fields(fields: Sequence[str], path: Path, line_number: int) -> tuple[int, float, float, float, float, float, int]:
    if len(fields) != len(EXPECTED_HEADER):
        raise ValueError(
            f"{path}:{line_number}: expected 7 whitespace-delimited fields, found {len(fields)}"
        )
    try:
        timestamp = int(fields[0])
        values = tuple(float(value) for value in fields[1:6])
        label = int(fields[6])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
    if label not in (0, 1, 2, 3):
        raise ValueError(f"{path}:{line_number}: label outside 0..3: {label}")
    return (timestamp, *values, label)


def iter_rows(path: str | Path) -> Iterator[tuple[int, float, float, float, float, float, int]]:
    """Yield typed rows without loading the file into memory."""
    source = Path(path)
    read_preamble(source)
    with source.open("r", encoding="ascii", newline=None) as handle:
        next(handle)
        next(handle)
        for line_number, line in enumerate(handle, start=3):
            if not line.strip():
                raise ValueError(f"{source}:{line_number}: blank data row")
            yield _parse_fields(line.split(), source, line_number)


def iter_chunks(path: str | Path, chunk_size: int = 100_000) -> Iterator[list[tuple[int, float, float, float, float, float, int]]]:
    """Yield bounded row chunks; callers control the memory/performance tradeoff."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk = []
    for row in iter_rows(path):
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
