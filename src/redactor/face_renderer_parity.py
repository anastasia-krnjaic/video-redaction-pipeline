from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2 as cv

CASE_SCHEMA_VERSION = "face_shared_renderer_parity.v2"
SUMMARY_SCHEMA_VERSION = "face_shared_renderer_parity_summary.v2"

CLASSIFICATION_PASS = "pass"
CLASSIFICATION_BASELINE_DEFECT = "baseline_defect"
CLASSIFICATION_SHARED_RENDERER_DEFECT = "shared_renderer_defect"
CLASSIFICATION_COMMON_PIPELINE_DEFECT = "common_pipeline_defect"
CLASSIFICATION_INTENTIONAL_SPAN = "intentional_span"
CLASSIFICATION_BLOCKED = "blocked"

STATUS_PASS = "pass"
STATUS_PASS_WITH_DOCUMENTED_BASELINE_DEFECT = "pass_with_documented_baseline_defect"
STATUS_FAIL = "fail"
STATUS_BLOCKED = "blocked"

TIMESTAMP_TOLERANCE_SECONDS = 0.01


@dataclass(frozen=True)
class SequentialDecodeSummary:
    path: str
    frame_count: int
    first_timestamp: float | None
    last_timestamp: float | None


@dataclass(frozen=True)
class RawRenderComparison:
    passed: bool
    frame_count_compared: int
    first_difference_frame: int | None
    max_abs_diff: int | None


@dataclass(frozen=True)
class DecodedOutputComparison:
    frames_compared: int
    first_difference_frame: int | None
    max_abs_diff: int | None
    outputs_ended_together: bool


def decode_video_summary(path: Path, *, display_path: str | None = None) -> SequentialDecodeSummary:
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")

    frame_count = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            timestamp = _timestamp_after_read(capture)
            if frame_count == 0:
                first_timestamp = timestamp
            last_timestamp = timestamp
            frame_count += 1
    finally:
        capture.release()

    return SequentialDecodeSummary(
        path=display_path or path.as_posix(),
        frame_count=frame_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
    )


def compare_decoded_outputs(left_path: Path, right_path: Path) -> DecodedOutputComparison:
    left_capture = cv.VideoCapture(str(left_path))
    right_capture = cv.VideoCapture(str(right_path))
    if not left_capture.isOpened():
        raise ValueError(f"could not open video: {left_path}")
    if not right_capture.isOpened():
        left_capture.release()
        raise ValueError(f"could not open video: {right_path}")

    frames_compared = 0
    first_difference_frame: int | None = None
    max_abs_diff: int | None = None
    try:
        while True:
            left_ok, left_frame = left_capture.read()
            right_ok, right_frame = right_capture.read()
            if not left_ok or not right_ok:
                if left_ok != right_ok and first_difference_frame is None:
                    first_difference_frame = frames_compared
                return DecodedOutputComparison(
                    frames_compared=frames_compared,
                    first_difference_frame=first_difference_frame,
                    max_abs_diff=max_abs_diff,
                    outputs_ended_together=left_ok == right_ok,
                )

            if left_frame.shape != right_frame.shape:
                frame_max_abs_diff = 255
                if first_difference_frame is None:
                    first_difference_frame = frames_compared
            else:
                frame_max_abs_diff = int(cv.absdiff(left_frame, right_frame).max())
            max_abs_diff = (
                frame_max_abs_diff
                if max_abs_diff is None
                else max(max_abs_diff, frame_max_abs_diff)
            )
            frames_compared += 1
    finally:
        left_capture.release()
        right_capture.release()


def build_case_report(
    *,
    case_id: str,
    source_video: str,
    frozen_output_video: str,
    shared_output_video: str,
    source_summary: SequentialDecodeSummary,
    frozen_summary: SequentialDecodeSummary,
    shared_summary: SequentialDecodeSummary,
    raw_render_comparison: RawRenderComparison,
    decoded_output_comparison: DecodedOutputComparison,
    media_validation_passed: bool,
    rendered_region_validation_passed: bool,
) -> dict[str, Any]:
    legacy_source_span_preserved = _source_span_preserved(source_summary, frozen_summary)
    shared_source_span_preserved = _source_span_preserved(source_summary, shared_summary)
    exact_encoded_output_parity_passed = (
        frozen_summary.frame_count == shared_summary.frame_count
        and decoded_output_comparison.outputs_ended_together
        and _timestamps_match(frozen_summary.first_timestamp, shared_summary.first_timestamp)
        and _timestamps_match(frozen_summary.last_timestamp, shared_summary.last_timestamp)
    )
    classification = _classify_case(
        raw_render_comparison=raw_render_comparison,
        media_validation_passed=media_validation_passed,
        rendered_region_validation_passed=rendered_region_validation_passed,
        legacy_source_span_preserved=legacy_source_span_preserved,
        shared_source_span_preserved=shared_source_span_preserved,
    )
    blockers = _case_blockers(
        classification=classification,
        raw_render_comparison=raw_render_comparison,
        media_validation_passed=media_validation_passed,
        rendered_region_validation_passed=rendered_region_validation_passed,
        shared_source_span_preserved=shared_source_span_preserved,
    )
    shared_renderer_migration_safe = (
        classification in {CLASSIFICATION_PASS, CLASSIFICATION_BASELINE_DEFECT}
        and shared_source_span_preserved
        and raw_render_comparison.passed
        and media_validation_passed
        and rendered_region_validation_passed
    )

    return {
        "blockers": blockers,
        "case_id": case_id,
        "classification": classification,
        "decoded_first_difference_frame": decoded_output_comparison.first_difference_frame,
        "decoded_frames_compared": decoded_output_comparison.frames_compared,
        "decoded_max_abs_diff": decoded_output_comparison.max_abs_diff,
        "decoded_outputs_ended_together": decoded_output_comparison.outputs_ended_together,
        "exact_encoded_output_parity_passed": exact_encoded_output_parity_passed,
        "frozen_first_timestamp": frozen_summary.first_timestamp,
        "frozen_last_timestamp": frozen_summary.last_timestamp,
        "frozen_output_video": frozen_output_video,
        "frozen_sequential_decoded_frame_count": frozen_summary.frame_count,
        "legacy_source_span_preserved": legacy_source_span_preserved,
        "media_validation_passed": media_validation_passed,
        "notes": _case_notes(
            classification=classification,
            exact_encoded_output_parity_passed=exact_encoded_output_parity_passed,
            legacy_source_span_preserved=legacy_source_span_preserved,
            shared_source_span_preserved=shared_source_span_preserved,
        ),
        "raw_render_first_difference_frame": raw_render_comparison.first_difference_frame,
        "raw_render_frame_count_compared": raw_render_comparison.frame_count_compared,
        "raw_render_max_abs_diff": raw_render_comparison.max_abs_diff,
        "raw_render_parity_passed": raw_render_comparison.passed,
        "rendered_region_validation_passed": rendered_region_validation_passed,
        "schema_version": CASE_SCHEMA_VERSION,
        "shared_first_timestamp": shared_summary.first_timestamp,
        "shared_last_timestamp": shared_summary.last_timestamp,
        "shared_output_video": shared_output_video,
        "shared_renderer_migration_safe": shared_renderer_migration_safe,
        "shared_sequential_decoded_frame_count": shared_summary.frame_count,
        "shared_source_span_preserved": shared_source_span_preserved,
        "source_first_timestamp": source_summary.first_timestamp,
        "source_last_timestamp": source_summary.last_timestamp,
        "source_sequential_decoded_frame_count": source_summary.frame_count,
        "source_video": source_video,
    }


def build_summary_report(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_defect_detected = any(
        case["classification"] == CLASSIFICATION_BASELINE_DEFECT for case in cases.values()
    )
    all_shared_source_spans_preserved = all(
        case["shared_source_span_preserved"] for case in cases.values()
    )
    exact_encoded_output_parity_for_all_cases = all(
        case["exact_encoded_output_parity_passed"] for case in cases.values()
    )
    shared_renderer_migration_safe = all(
        case["shared_renderer_migration_safe"] for case in cases.values()
    )
    status = _summary_status(
        cases,
        baseline_defect_detected=baseline_defect_detected,
        shared_renderer_migration_safe=shared_renderer_migration_safe,
    )

    return {
        "all_shared_source_spans_preserved": all_shared_source_spans_preserved,
        "baseline_defect_detected": baseline_defect_detected,
        "case_count": len(cases),
        "cases": cases,
        "exact_encoded_output_parity_for_all_cases": exact_encoded_output_parity_for_all_cases,
        "generated_at_utc": _utc_timestamp(),
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "shared_renderer_migration_safe": shared_renderer_migration_safe,
        "status": status,
    }


def write_report_from_v1(
    *,
    v1_report_path: Path,
    output_dir: Path,
    repo_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    cases: dict[str, dict[str, Any]] = {}
    for case_id, case_payload in v1_report.get("cases", {}).items():
        cases[case_id] = _case_report_from_v1_case(case_id, case_payload, repo_root=repo_root)

    report = build_summary_report(cases)
    write_json_atomic(output_dir / "face_renderer_parity_report.json", report)
    write_text_atomic(output_dir / "face_renderer_parity_report.md", markdown_report(report))
    return report


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Face Shared-Renderer Parity Report v2",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Status: `{report['status']}`",
        "- Shared renderer migration safe: "
        f"`{str(report['shared_renderer_migration_safe']).lower()}`",
        "- Exact encoded-output parity for all cases: "
        f"`{str(report['exact_encoded_output_parity_for_all_cases']).lower()}`",
        f"- Baseline defect detected: `{str(report['baseline_defect_detected']).lower()}`",
        "",
        "| Case | Classification | Raw render | Exact final file | "
        "Legacy span | Shared span | Migration safe |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case_id, case in report["cases"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    case_id,
                    f"`{case['classification']}`",
                    _bool_text(case["raw_render_parity_passed"]),
                    _bool_text(case["exact_encoded_output_parity_passed"]),
                    _bool_text(case["legacy_source_span_preserved"]),
                    _bool_text(case["shared_source_span_preserved"]),
                    _bool_text(case["shared_renderer_migration_safe"]),
                ]
            )
            + " |"
        )
    lines.extend([""])
    for case_id, case in report["cases"].items():
        lines.extend(
            [
                f"## {case_id}",
                "",
                f"- Source decoded frames: `{case['source_sequential_decoded_frame_count']}`",
                f"- Frozen decoded frames: `{case['frozen_sequential_decoded_frame_count']}`",
                f"- Shared decoded frames: `{case['shared_sequential_decoded_frame_count']}`",
                f"- Decoded outputs ended together: "
                f"`{str(case['decoded_outputs_ended_together']).lower()}`",
                f"- Decoded frames compared: `{case['decoded_frames_compared']}`",
                f"- Raw render frames compared: `{case['raw_render_frame_count_compared']}`",
                "",
            ]
        )
        for note in case["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _case_report_from_v1_case(
    case_id: str,
    case_payload: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    source_video = str(case_payload["source_video"])
    frozen_output_video = str(case_payload["frozen_output_video"])
    shared_output_video = str(case_payload["shared_output_video"])
    source_path = _resolve_report_path(repo_root, source_video)
    frozen_path = _resolve_report_path(repo_root, frozen_output_video)
    shared_path = _resolve_report_path(repo_root, shared_output_video)
    raw_payload = case_payload["raw_frame_render_comparison"]
    decoded_output_comparison = compare_decoded_outputs(frozen_path, shared_path)

    return build_case_report(
        case_id=case_id,
        source_video=source_video,
        frozen_output_video=frozen_output_video,
        shared_output_video=shared_output_video,
        source_summary=decode_video_summary(source_path, display_path=source_video),
        frozen_summary=decode_video_summary(frozen_path, display_path=frozen_output_video),
        shared_summary=decode_video_summary(shared_path, display_path=shared_output_video),
        raw_render_comparison=RawRenderComparison(
            passed=bool(raw_payload["exact_frame_render_parity"]),
            frame_count_compared=int(raw_payload["frame_count_compared"]),
            first_difference_frame=_optional_int(raw_payload.get("first_difference_frame")),
            max_abs_diff=_optional_int(raw_payload.get("max_abs_diff")),
        ),
        decoded_output_comparison=decoded_output_comparison,
        media_validation_passed=bool(case_payload.get("media_validation_passed")),
        rendered_region_validation_passed=bool(
            case_payload.get("rendered_region_validation_passed")
        ),
    )


def _classify_case(
    *,
    raw_render_comparison: RawRenderComparison,
    media_validation_passed: bool,
    rendered_region_validation_passed: bool,
    legacy_source_span_preserved: bool,
    shared_source_span_preserved: bool,
) -> str:
    if not shared_source_span_preserved and not legacy_source_span_preserved:
        return CLASSIFICATION_COMMON_PIPELINE_DEFECT
    if not shared_source_span_preserved:
        return CLASSIFICATION_SHARED_RENDERER_DEFECT
    if not legacy_source_span_preserved:
        return CLASSIFICATION_BASELINE_DEFECT
    if not raw_render_comparison.passed:
        return CLASSIFICATION_SHARED_RENDERER_DEFECT
    if not media_validation_passed or not rendered_region_validation_passed:
        return CLASSIFICATION_BLOCKED
    return CLASSIFICATION_PASS


def _case_blockers(
    *,
    classification: str,
    raw_render_comparison: RawRenderComparison,
    media_validation_passed: bool,
    rendered_region_validation_passed: bool,
    shared_source_span_preserved: bool,
) -> list[str]:
    blockers: list[str] = []
    if classification in {
        CLASSIFICATION_SHARED_RENDERER_DEFECT,
        CLASSIFICATION_COMMON_PIPELINE_DEFECT,
    }:
        blockers.append(classification)
    if not shared_source_span_preserved:
        blockers.append("shared_source_span_not_preserved")
    if not raw_render_comparison.passed:
        blockers.append("raw_render_parity_failed")
    if not media_validation_passed:
        blockers.append("media_validation_failed")
    if not rendered_region_validation_passed:
        blockers.append("rendered_region_validation_failed")
    return blockers


def _case_notes(
    *,
    classification: str,
    exact_encoded_output_parity_passed: bool,
    legacy_source_span_preserved: bool,
    shared_source_span_preserved: bool,
) -> list[str]:
    notes: list[str] = []
    if classification == CLASSIFICATION_BASELINE_DEFECT:
        notes.append("Legacy frozen output does not preserve the source span; shared output does.")
    if not exact_encoded_output_parity_passed:
        notes.append("Final encoded outputs are not exact span/count parity.")
    if legacy_source_span_preserved and shared_source_span_preserved:
        notes.append("Both final outputs preserve the source span.")
    return notes


def _source_span_preserved(
    source: SequentialDecodeSummary,
    output: SequentialDecodeSummary,
) -> bool:
    return (
        source.frame_count == output.frame_count
        and _timestamps_match(source.first_timestamp, output.first_timestamp)
        and _timestamps_match(source.last_timestamp, output.last_timestamp)
    )


def _summary_status(
    cases: dict[str, dict[str, Any]],
    *,
    baseline_defect_detected: bool,
    shared_renderer_migration_safe: bool,
) -> str:
    if any(case["classification"] == CLASSIFICATION_BLOCKED for case in cases.values()):
        return STATUS_BLOCKED
    if not shared_renderer_migration_safe:
        return STATUS_FAIL
    if baseline_defect_detected:
        return STATUS_PASS_WITH_DOCUMENTED_BASELINE_DEFECT
    return STATUS_PASS


def _timestamps_match(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= TIMESTAMP_TOLERANCE_SECONDS


def _timestamp_after_read(capture: cv.VideoCapture) -> float | None:
    value = capture.get(cv.CAP_PROP_POS_MSEC)
    if not math.isfinite(value):
        return None
    return round(value / 1000.0, 6)


def _resolve_report_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _bool_text(value: bool) -> str:
    return f"`{str(value).lower()}`"


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
