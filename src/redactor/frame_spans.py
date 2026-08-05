from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "frame_span.v1"


def build_frame_span_report(
    frame_indices: Iterable[int],
    *,
    source_frame_count: int | None,
) -> dict[str, Any]:
    """Summarize which source frames have processing records."""
    unique_indices = sorted({int(index) for index in frame_indices})
    first_processed = unique_indices[0] if unique_indices else None
    last_processed = unique_indices[-1] if unique_indices else None
    processed_span_count = 0
    if first_processed is not None and last_processed is not None:
        processed_span_count = last_processed - first_processed + 1
    missing_ranges = missing_processed_ranges(
        unique_indices,
        source_frame_count=source_frame_count,
    )
    full_source_processed = (
        source_frame_count is not None
        and source_frame_count >= 0
        and len(unique_indices) == source_frame_count
        and first_processed == 0
        and last_processed == source_frame_count - 1
        and not missing_ranges
    )
    return {
        "first_processed_frame": first_processed,
        "full_source_processed": full_source_processed,
        "last_processed_frame": last_processed,
        "missing_processed_ranges": [
            {"frame_start": start, "frame_end": end, "length": end - start + 1}
            for start, end in missing_ranges
        ],
        "missing_source_frame_count": sum(end - start + 1 for start, end in missing_ranges),
        "processed_frame_count": len(unique_indices),
        "processed_frame_span_count": processed_span_count,
        "record_frame_count": len(unique_indices),
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": source_frame_count,
    }


def missing_processed_ranges(
    frame_indices: Iterable[int],
    *,
    source_frame_count: int | None,
) -> list[tuple[int, int]]:
    if source_frame_count is None:
        return []
    end_frame = int(source_frame_count) - 1
    if end_frame < 0:
        return []
    present = {int(index) for index in frame_indices if 0 <= int(index) <= end_frame}
    ranges: list[tuple[int, int]] = []
    range_start: int | None = None
    previous_missing: int | None = None
    for frame_index in range(end_frame + 1):
        if frame_index in present:
            if range_start is not None and previous_missing is not None:
                ranges.append((range_start, previous_missing))
                range_start = None
                previous_missing = None
            continue
        if range_start is None:
            range_start = frame_index
        previous_missing = frame_index
    if range_start is not None and previous_missing is not None:
        ranges.append((range_start, previous_missing))
    return ranges


def jsonl_frame_indices(path: Path) -> list[int]:
    indices: list[int] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            indices.append(int(payload["frame_index"]))
    return indices
