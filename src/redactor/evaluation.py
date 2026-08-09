from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

import cv2 as cv
import numpy as np

from redactor import media
from redactor.privacy_records import CATEGORY_FACE, CATEGORY_PLATE

SCHEMA_VERSION = "1.0"

MODE_DETECTOR = "detector"
MODE_AUTOMATIC = "automatic"
MODE_REVIEWED_FINAL = "reviewed_final"

STATUS_DETECTOR_EVALUATED = "detector_evaluated"
STATUS_REVIEW_REQUIRED = "review_required"
STATUS_PASSED_AUTOMATIC = "passed_automatic"
STATUS_PASSED_REVIEWED_FINAL = "passed_reviewed_final"
STATUS_FAILED_PRIVACY_ACCEPTANCE = "failed_privacy_acceptance"
STATUS_FAILED_INPUT_VALIDATION = "failed_input_validation"
STATUS_NOT_SCORABLE = "not_scorable"

VALID_POLICIES = {"redact", "keep", "ignore"}
VALID_OBJECT_TYPES = {"face", "license_plate"}
VALID_MODES = {MODE_DETECTOR, MODE_AUTOMATIC, MODE_REVIEWED_FINAL}

AUTOMATIC_PROVENANCES = {"detector", "ordinary_tracker"}
REVIEWED_PROVENANCE_MARKERS = ("preserved", "reviewed", "manual")
SYNTHETIC_PROVENANCES = {
    "bidirectional_privacy_bridge",
    "forward_only_bridge",
    "backward_only_bridge",
    "interpolation",
    "conservative_fallback",
}

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PrivacyRegion:
    frame_index: int
    bbox_xyxy: Box | None
    provenance: str
    state: str
    target_track_ids: tuple[int, ...]
    source: dict[str, Any]

    @property
    def is_covered(self) -> bool:
        return self.state == "covered" and self.bbox_xyxy is not None


@dataclass(frozen=True)
class TrackBox:
    frame_index: int
    track_id: int
    bbox_xyxy: Box
    provenance: str
    score: float | None
    candidate_id: str | None
    quality_status: str


@dataclass(frozen=True)
class EvaluationRequest:
    ground_truth_path: Path
    clip_id: str
    mode: str
    output_dir: Path
    source_video_path: Path | None = None
    detections_path: Path | None = None
    adjudication_path: Path | None = None
    tracks_path: Path | None = None
    privacy_regions_path: Path | None = None
    interval_decisions_path: Path | None = None
    preserved_regions_path: Path | None = None
    rendered_video_path: Path | None = None
    render_manifest_path: Path | None = None
    render_validation_path: Path | None = None
    detector_threshold: float = 0.30
    overwrite: bool = False


class CategoryEvaluationAdapter(Protocol):
    category: str

    def run(self, request: EvaluationRequest) -> dict[str, Any]: ...


class FaceEvaluationAdapter:
    category = CATEGORY_FACE

    def run(self, request: EvaluationRequest) -> dict[str, Any]:
        if request.source_video_path is None:
            raise ValueError("face evaluation requires source_video_path")
        if request.detections_path is None:
            raise ValueError("face evaluation requires detections_path")
        return _run_face_evaluation_impl(
            ground_truth_path=request.ground_truth_path,
            clip_id=request.clip_id,
            mode=request.mode,
            source_video_path=request.source_video_path,
            detections_path=request.detections_path,
            output_dir=request.output_dir,
            adjudication_path=request.adjudication_path,
            tracks_path=request.tracks_path,
            privacy_regions_path=request.privacy_regions_path,
            interval_decisions_path=request.interval_decisions_path,
            preserved_regions_path=request.preserved_regions_path,
            rendered_video_path=request.rendered_video_path,
            render_manifest_path=request.render_manifest_path,
            render_validation_path=request.render_validation_path,
            detector_threshold=request.detector_threshold,
            overwrite=request.overwrite,
        )


class PlateEvaluationAdapter:
    category = CATEGORY_PLATE

    def run(self, request: EvaluationRequest) -> dict[str, Any]:
        ensure_output_dir(request.output_dir, overwrite=request.overwrite)
        capability = plate_evaluation_capability(
            ground_truth_path=request.ground_truth_path,
            clip_id=request.clip_id,
        )
        metrics = {
            "acceptance": {
                "blocking_reasons": capability["blockers"],
                "final_status": STATUS_NOT_SCORABLE,
            },
            "capability": capability,
            "category": CATEGORY_PLATE,
            "clip_id": request.clip_id,
            "detector": {
                "detector_scored_object_frames": None,
                "frames_with_valid_detection": None,
                "object_frame_recall": None,
            },
            "evaluation_mode": request.mode,
            "privacy": {
                "covered_privacy_frames": None,
                "privacy_coverage": None,
                "privacy_required_frames": None,
            },
            "publishable_metrics": False,
            "schema_version": SCHEMA_VERSION,
        }
        write_json(request.output_dir / "metrics.json", metrics)
        (request.output_dir / "report.md").write_text(
            plate_capability_report_markdown(capability),
            encoding="utf-8",
        )
        return metrics


def run_category_evaluation(category: str, request: EvaluationRequest) -> dict[str, Any]:
    adapters: dict[str, CategoryEvaluationAdapter] = {
        CATEGORY_FACE: FaceEvaluationAdapter(),
        CATEGORY_PLATE: PlateEvaluationAdapter(),
    }
    adapter = adapters.get(category)
    if adapter is None:
        raise ValueError(f"unsupported evaluation category: {category}")
    return adapter.run(request)


def run_face_evaluation(
    *,
    ground_truth_path: Path,
    clip_id: str,
    mode: str,
    source_video_path: Path,
    detections_path: Path,
    output_dir: Path,
    adjudication_path: Path | None = None,
    tracks_path: Path | None = None,
    privacy_regions_path: Path | None = None,
    interval_decisions_path: Path | None = None,
    preserved_regions_path: Path | None = None,
    rendered_video_path: Path | None = None,
    render_manifest_path: Path | None = None,
    render_validation_path: Path | None = None,
    detector_threshold: float = 0.30,
    overwrite: bool = False,
) -> dict[str, Any]:
    return run_category_evaluation(
        CATEGORY_FACE,
        EvaluationRequest(
            adjudication_path=adjudication_path,
            clip_id=clip_id,
            detector_threshold=detector_threshold,
            detections_path=detections_path,
            ground_truth_path=ground_truth_path,
            interval_decisions_path=interval_decisions_path,
            mode=mode,
            output_dir=output_dir,
            overwrite=overwrite,
            preserved_regions_path=preserved_regions_path,
            privacy_regions_path=privacy_regions_path,
            render_manifest_path=render_manifest_path,
            render_validation_path=render_validation_path,
            rendered_video_path=rendered_video_path,
            source_video_path=source_video_path,
            tracks_path=tracks_path,
        ),
    )


def run_plate_evaluation(
    *,
    ground_truth_path: Path,
    clip_id: str,
    output_dir: Path,
    mode: str = "capability",
    overwrite: bool = False,
) -> dict[str, Any]:
    return run_category_evaluation(
        CATEGORY_PLATE,
        EvaluationRequest(
            clip_id=clip_id,
            ground_truth_path=ground_truth_path,
            mode=mode,
            output_dir=output_dir,
            overwrite=overwrite,
        ),
    )


def _run_face_evaluation_impl(
    *,
    ground_truth_path: Path,
    clip_id: str,
    mode: str,
    source_video_path: Path,
    detections_path: Path,
    output_dir: Path,
    adjudication_path: Path | None = None,
    tracks_path: Path | None = None,
    privacy_regions_path: Path | None = None,
    interval_decisions_path: Path | None = None,
    preserved_regions_path: Path | None = None,
    rendered_video_path: Path | None = None,
    render_manifest_path: Path | None = None,
    render_validation_path: Path | None = None,
    detector_threshold: float = 0.30,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    ensure_output_dir(output_dir, overwrite=overwrite)

    ground_truth = load_ground_truth(ground_truth_path)
    validation_errors = validate_ground_truth_document(
        ground_truth,
        source_video_paths={clip_id: source_video_path},
    )
    if validation_errors:
        payload = failed_validation_payload(ground_truth_path, clip_id, mode, validation_errors)
        write_evaluation_outputs(output_dir, payload)
        raise ValueError("ground-truth validation failed: " + "; ".join(validation_errors))

    clip = clip_by_id(ground_truth, clip_id)
    face_objects = [obj for obj in clip["objects"] if obj["object_type"] == "face"]
    detections = load_detections(detections_path)
    adjudication = load_adjudication(adjudication_path) if adjudication_path else {}
    tracks = load_tracks(tracks_path) if tracks_path else {}
    intervals = load_interval_decisions(interval_decisions_path)
    privacy_regions = load_privacy_regions(privacy_regions_path) if privacy_regions_path else {}
    render_manifest = load_json(render_manifest_path) if render_manifest_path else None
    render_validation = load_json(render_validation_path) if render_validation_path else None

    object_track_ids = infer_object_track_ids(face_objects, adjudication, tracks, clip_id)
    object_metrics: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for obj in face_objects:
        object_id = str(obj["object_id"])
        detector_metrics, detector_failures = detector_metrics_for_object(
            clip_id=clip_id,
            obj=obj,
            detections=detections,
            adjudication=adjudication,
            detector_threshold=detector_threshold,
        )
        tracking_metrics = tracking_metrics_for_object(
            obj=obj,
            tracks=tracks,
            track_ids=object_track_ids.get(object_id, set()),
        )
        privacy_metrics, privacy_failures = privacy_metrics_for_object(
            clip_id=clip_id,
            obj=obj,
            mode=mode,
            privacy_regions=privacy_regions,
            intervals=intervals,
            object_track_ids=object_track_ids,
            tracks=tracks,
        )
        stability_metrics = stability_metrics_for_object(
            obj=obj,
            tracks=tracks,
            privacy_regions=privacy_regions,
            object_track_ids=object_track_ids.get(object_id, set()),
            use_privacy_regions=bool(privacy_regions) and obj["policy"] == "redact",
        )
        object_metrics[object_id] = {
            "detector": detector_metrics,
            "ground_truth": object_ground_truth_summary(obj),
            "object_id": object_id,
            "object_type": obj["object_type"],
            "policy": obj["policy"],
            "privacy": privacy_metrics,
            "stability": stability_metrics,
            "tracking": tracking_metrics,
        }
        failures.extend(detector_failures)
        failures.extend(privacy_failures)

    keep_failures = keep_person_false_redaction_failures(
        clip_id=clip_id,
        objects=face_objects,
        privacy_regions=privacy_regions,
        object_track_ids=object_track_ids,
        tracks=tracks,
    )
    failures.extend(keep_failures)

    render_metrics, render_failures = render_metrics_for_run(
        mode=mode,
        source_video_path=source_video_path,
        privacy_regions_path=privacy_regions_path,
        rendered_video_path=rendered_video_path,
        render_manifest=render_manifest,
        render_validation=render_validation,
        accepted_region_count=sum(
            metrics["privacy"]["covered_frame_count"]
            for metrics in object_metrics.values()
            if metrics["policy"] == "redact"
        ),
    )
    failures.extend(render_failures)

    acceptance = acceptance_for_run(mode, object_metrics, intervals, failures, render_metrics)
    metrics = aggregate_metrics(
        clip=clip,
        mode=mode,
        detector_threshold=detector_threshold,
        object_metrics=object_metrics,
        intervals=intervals,
        failures=failures,
        render_metrics=render_metrics,
        acceptance=acceptance,
        elapsed_seconds=time.perf_counter() - started,
        paths={
            "adjudication": adjudication_path,
            "detections": detections_path,
            "ground_truth": ground_truth_path,
            "interval_decisions": interval_decisions_path,
            "preserved_regions": preserved_regions_path,
            "privacy_regions": privacy_regions_path,
            "render_manifest": render_manifest_path,
            "render_validation": render_validation_path,
            "rendered_video": rendered_video_path,
            "source_video": source_video_path,
            "tracks": tracks_path,
        },
    )
    visual_paths = write_failure_visuals(
        source_video_path=source_video_path,
        output_dir=output_dir,
        failures=failures,
        privacy_regions=privacy_regions,
        source_frame_count=int(clip["source_frame_count"]),
        overwrite=overwrite,
    )
    attach_visual_paths(failures, visual_paths)

    payload = {
        "failures": {
            "failures": failures,
            "failure_count": len(failures),
            "schema_version": SCHEMA_VERSION,
        },
        "interval_decisions": interval_decision_summary(intervals),
        "metrics": metrics,
        "object_metrics": {
            "objects": object_metrics,
            "schema_version": SCHEMA_VERSION,
        },
        "render_validation": render_metrics,
        "report": report_markdown(metrics, object_metrics, failures, render_metrics),
        "summary_rows": summary_rows(metrics, object_metrics),
    }
    write_evaluation_outputs(output_dir, payload)
    return metrics


def load_ground_truth(path: Path) -> dict[str, Any]:
    return load_json(path)


def validate_ground_truth_file(
    path: Path,
    *,
    source_video_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    ground_truth = load_ground_truth(path)
    errors = validate_ground_truth_document(ground_truth, source_video_paths=source_video_paths)
    return {
        "error_count": len(errors),
        "errors": errors,
        "ground_truth_id": ground_truth.get("ground_truth_id"),
        "schema_version": ground_truth.get("schema_version"),
        "status": "valid" if not errors else "invalid",
    }


def plate_evaluation_capability(*, ground_truth_path: Path, clip_id: str) -> dict[str, Any]:
    ground_truth = load_ground_truth(ground_truth_path)
    validation_errors = validate_ground_truth_document(ground_truth)
    clip = next(
        (
            candidate
            for candidate in ground_truth.get("clips", [])
            if candidate.get("clip_id") == clip_id
        ),
        None,
    )
    blockers: list[str] = []
    if validation_errors:
        blockers.append("ground_truth_schema_validation_failed")
    if clip is None:
        blockers.append("clip_id_not_found")
        plate_objects: list[dict[str, Any]] = []
    else:
        plate_objects = [
            obj for obj in clip.get("objects", []) if obj.get("object_type") == CATEGORY_PLATE
        ]
    objects_with_dense_boxes = [
        obj for obj in plate_objects if obj.get("per_frame_boxes") or obj.get("keyframe_boxes")
    ]
    if not plate_objects:
        blockers.append("machine_readable_ground_truth_has_no_license_plate_objects")
    elif not objects_with_dense_boxes:
        blockers.append("license_plate_objects_lack_tight_scoring_boxes")

    annotation_scope = ground_truth.get("annotation_scope") or {}
    can_score = not blockers and len(objects_with_dense_boxes) == len(plate_objects)
    return {
        "annotation_scope": {
            "box_truth_level": annotation_scope.get("box_truth_level"),
            "keyframe_boxes_available": annotation_scope.get("keyframe_boxes_available"),
            "per_frame_boxes_available": annotation_scope.get("per_frame_boxes_available"),
        },
        "blockers": blockers,
        "can_publish_detector_metrics": can_score,
        "can_publish_privacy_metrics": can_score,
        "category": CATEGORY_PLATE,
        "clip_id": clip_id,
        "ground_truth_id": ground_truth.get("ground_truth_id"),
        "license_plate_object_count": len(plate_objects),
        "object_ids": [str(obj.get("object_id")) for obj in plate_objects],
        "objects_with_tight_boxes": len(objects_with_dense_boxes),
        "publishable_metrics": can_score,
        "schema_version": "plate_evaluation_capability.v1",
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }


def plate_capability_report_markdown(capability: dict[str, Any]) -> str:
    blockers = capability.get("blockers") or []
    lines = [
        "# License Plate Evaluation Capability",
        "",
        f"- Clip: `{capability.get('clip_id')}`",
        f"- Ground truth: `{capability.get('ground_truth_id')}`",
        f"- Publishable metrics: `{capability.get('publishable_metrics')}`",
        f"- License-plate objects: `{capability.get('license_plate_object_count')}`",
        f"- Objects with tight boxes: `{capability.get('objects_with_tight_boxes')}`",
        "",
        "## Status",
        "",
    ]
    if blockers:
        lines.append("Plate detector/privacy scores were not produced.")
        lines.append("")
        lines.append("## Blockers")
        lines.append("")
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("The machine-readable ground truth contains the fields needed for scoring.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Range-level or search ROI annotations are not treated as tight plate boxes.",
            "This report is a capability record, not an evaluation scorecard.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_ground_truth_document(
    ground_truth: dict[str, Any],
    *,
    source_video_paths: dict[str, Path] | None = None,
) -> list[str]:
    errors: list[str] = []
    if ground_truth.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not ground_truth.get("ground_truth_id"):
        errors.append("missing ground_truth_id")
    clips = ground_truth.get("clips")
    if not isinstance(clips, list) or not clips:
        return [*errors, "clips must be a non-empty list"]

    clip_ids: set[str] = set()
    object_ids: set[str] = set()
    source_video_paths = source_video_paths or {}
    for clip in clips:
        clip_id = str(clip.get("clip_id") or "")
        if not clip_id:
            errors.append("clip missing clip_id")
            continue
        if clip_id in clip_ids:
            errors.append(f"duplicate clip_id: {clip_id}")
        clip_ids.add(clip_id)
        frame_count = as_int(clip.get("source_frame_count"), default=-1)
        if frame_count <= 0:
            errors.append(f"{clip_id}: source_frame_count must be positive")
        validate_range_field(
            errors,
            clip_id,
            "evaluated_frame_range",
            clip.get("evaluated_frame_range"),
            frame_count,
            single=True,
        )
        source = clip.get("source_video") or {}
        if not source.get("sha256"):
            errors.append(f"{clip_id}: source_video.sha256 is required")
        if local_path := source_video_paths.get(clip_id):
            if not local_path.exists():
                errors.append(f"{clip_id}: source video does not exist: {local_path}")
            elif media.sha256_file(local_path) != source.get("sha256"):
                errors.append(f"{clip_id}: source hash mismatch")
        objects = clip.get("objects")
        if not isinstance(objects, list) or not objects:
            errors.append(f"{clip_id}: objects must be a non-empty list")
            continue
        for obj in objects:
            object_id = str(obj.get("object_id") or "")
            if not object_id:
                errors.append(f"{clip_id}: object missing object_id")
                continue
            if object_id in object_ids:
                errors.append(f"duplicate object_id: {object_id}")
            object_ids.add(object_id)
            if obj.get("object_type") not in VALID_OBJECT_TYPES:
                errors.append(f"{object_id}: invalid object_type")
            if obj.get("policy") not in VALID_POLICIES:
                errors.append(f"{object_id}: invalid policy")
            for field in (
                "detector_scored_ranges",
                "detector_ignore_ranges",
                "privacy_required_ranges",
                "keep_clear_ranges",
            ):
                validate_range_field(errors, object_id, field, obj.get(field), frame_count)
            if overlapping_frames(
                obj.get("detector_scored_ranges", []), obj.get("detector_ignore_ranges", [])
            ):
                errors.append(f"{object_id}: detector_scored_ranges overlap detector_ignore_ranges")
            if overlapping_frames(
                obj.get("privacy_required_ranges", []), obj.get("keep_clear_ranges", [])
            ):
                errors.append(f"{object_id}: privacy_required_ranges overlap keep_clear_ranges")
            for box_field in ("reference_roi_xyxy",):
                if box_field in obj and obj[box_field] is not None:
                    validate_box(errors, object_id, box_field, obj[box_field])
            validate_box_records(
                errors, object_id, "per_frame_boxes", obj.get("per_frame_boxes", [])
            )
            validate_box_records(errors, object_id, "keyframe_boxes", obj.get("keyframe_boxes", []))
            if not obj.get("annotation_provenance"):
                errors.append(f"{object_id}: annotation_provenance is required")
            if not obj.get("review_status"):
                errors.append(f"{object_id}: review_status is required")
    return errors


def validate_range_field(
    errors: list[str],
    owner: str,
    field: str,
    ranges: Any,
    frame_count: int,
    *,
    single: bool = False,
) -> None:
    if ranges is None:
        errors.append(f"{owner}: missing {field}")
        return
    check_ranges = [ranges] if single else ranges
    if not isinstance(check_ranges, list):
        errors.append(f"{owner}: {field} must be a list")
        return
    for item in check_ranges:
        if not isinstance(item, list | tuple) or len(item) != 2:
            errors.append(f"{owner}: {field} entries must be [start, end]")
            continue
        start, end = as_int(item[0]), as_int(item[1])
        if start is None or end is None:
            errors.append(f"{owner}: {field} entries must be integers")
            continue
        if start > end:
            errors.append(f"{owner}: {field} has start > end")
        if start < 0 or end >= frame_count:
            errors.append(f"{owner}: {field} outside source frame bounds")


def validate_box_records(errors: list[str], owner: str, field: str, records: Any) -> None:
    if not isinstance(records, list):
        errors.append(f"{owner}: {field} must be a list")
        return
    for record in records:
        if not isinstance(record, dict):
            errors.append(f"{owner}: {field} entries must be objects")
            continue
        validate_box(errors, owner, field, record.get("bbox_xyxy"))


def validate_box(errors: list[str], owner: str, field: str, values: Any) -> None:
    box = as_box(values)
    if box is None:
        errors.append(f"{owner}: {field} must be xyxy coordinates")
        return
    if box[2] <= box[0] or box[3] <= box[1]:
        errors.append(f"{owner}: {field} must have positive area")


def failed_validation_payload(
    ground_truth_path: Path,
    clip_id: str,
    mode: str,
    errors: list[str],
) -> dict[str, Any]:
    metrics = {
        "acceptance": {
            "blocking_reasons": ["input_validation_errors"],
            "final_status": STATUS_FAILED_INPUT_VALIDATION,
        },
        "clip_id": clip_id,
        "evaluation_mode": mode,
        "ground_truth": ground_truth_path.as_posix(),
        "schema_version": SCHEMA_VERSION,
    }
    failures = [
        {
            "clip": clip_id,
            "evaluated_stage": "input_validation",
            "failure_reason": error,
            "frame": None,
            "object_id": None,
            "observed_provenance": None,
            "policy": None,
            "review_status": "failed_input_validation",
            "visual_evidence": None,
        }
        for error in errors
    ]
    return {
        "failures": {"failures": failures, "failure_count": len(failures)},
        "interval_decisions": {"interval_decisions": [], "schema_version": SCHEMA_VERSION},
        "metrics": metrics,
        "object_metrics": {"objects": {}, "schema_version": SCHEMA_VERSION},
        "render_validation": {"not_applicable": True},
        "report": report_markdown(metrics, {}, failures, {"not_applicable": True}),
        "summary_rows": [],
    }


def detector_metrics_for_object(
    *,
    clip_id: str,
    obj: dict[str, Any],
    detections: dict[int, list[dict[str, Any]]],
    adjudication: dict[tuple[str, str, int], dict[str, Any]],
    detector_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    object_id = str(obj["object_id"])
    frames = sorted(expand_ranges(obj.get("detector_scored_ranges", [])))
    hits: set[int] = set()
    misses: list[int] = []
    selected_scores: list[float] = []
    failures: list[dict[str, Any]] = []
    selected_candidate_ids: set[str] = set()

    for frame in frames:
        decision = adjudication.get((clip_id, object_id, frame))
        if not decision or decision.get("choice") != "candidate":
            misses.append(frame)
            failures.append(
                failure(
                    clip_id,
                    object_id,
                    frame,
                    obj["policy"],
                    "detector",
                    "none",
                    "detector_scored_frame_without_adjudicated_candidate",
                    obj.get("review_status"),
                )
            )
            continue
        candidate_id = str(decision["candidate_id"])
        selected_candidate_ids.add(candidate_id)
        source_detection = detection_by_candidate(detections.get(frame, []), candidate_id)
        if source_detection is None:
            misses.append(frame)
            failures.append(
                failure(
                    clip_id,
                    object_id,
                    frame,
                    obj["policy"],
                    "detector",
                    "missing_candidate",
                    "adjudicated_candidate_missing_from_detector_output",
                    obj.get("review_status"),
                )
            )
            continue
        score = float(source_detection.get("score", decision.get("score", 0.0)))
        if score < detector_threshold:
            misses.append(frame)
            failures.append(
                failure(
                    clip_id,
                    object_id,
                    frame,
                    obj["policy"],
                    "detector",
                    "below_threshold",
                    "adjudicated_candidate_below_detector_threshold",
                    obj.get("review_status"),
                )
            )
            continue
        hits.add(frame)
        selected_scores.append(score)

    evaluated_detection_counts = [
        len(
            [
                det
                for det in detections.get(frame, [])
                if float(det.get("score", 0.0)) >= detector_threshold
            ]
        )
        for frame in frames
    ]
    unmatched = sum(
        1
        for frame in frames
        for det in detections.get(frame, [])
        if float(det.get("score", 0.0)) >= detector_threshold
        and det.get("candidate_id") not in selected_candidate_ids
    )
    return (
        {
            "detector_scored_object_frames": len(frames),
            "detection_truth_source": "corrected_adas_adjudication"
            if adjudication
            else "unavailable",
            "duplicate_detections_where_measurable": sum(
                1 for count in evaluated_detection_counts if count > 1
            ),
            "frames_with_valid_detection": len(hits),
            "missed_detector_scored_frames": misses,
            "missed_detector_scored_frame_count": len(misses),
            "longest_consecutive_missed_run": longest_run(misses),
            "object_frame_recall": ratio(len(hits), len(frames)),
            "raw_detections_per_evaluated_frame": distribution(evaluated_detection_counts),
            "score_distribution": distribution(selected_scores),
            "selected_candidate_count": len(selected_candidate_ids),
            "threshold": detector_threshold,
            "unmatched_thresholded_proposals": unmatched,
        },
        failures,
    )


def tracking_metrics_for_object(
    *,
    obj: dict[str, Any],
    tracks: dict[int, list[TrackBox]],
    track_ids: set[int],
) -> dict[str, Any]:
    relevant_frames = relevant_tracking_frames(obj)
    boxes_by_frame = {
        frame: [box for box in tracks.get(frame, []) if box.track_id in track_ids]
        for frame in relevant_frames
    }
    covered_frames = sorted(frame for frame, boxes in boxes_by_frame.items() if boxes)
    provenances = [box.provenance for boxes in boxes_by_frame.values() for box in boxes]
    starts = sorted(
        (
            min(
                frame
                for frame, boxes in boxes_by_frame.items()
                if any(b.track_id == tid for b in boxes)
            ),
            tid,
        )
        for tid in track_ids
        if any(any(b.track_id == tid for b in boxes) for boxes in boxes_by_frame.values())
    )
    ends = sorted(
        (
            max(
                frame
                for frame, boxes in boxes_by_frame.items()
                if any(b.track_id == tid for b in boxes)
            ),
            tid,
        )
        for tid in track_ids
        if any(any(b.track_id == tid for b in boxes) for boxes in boxes_by_frame.values())
    )
    uncovered = sorted(set(relevant_frames) - set(covered_frames))
    return {
        "box_level_coverage_measured": False,
        "fragment_track_ids": sorted(track_ids),
        "interpolated_frames": count_frames_with_provenance(boxes_by_frame, {"interpolated"}),
        "longest_tracking_gap": longest_run(uncovered),
        "observed_frames": count_frames_with_prefix(boxes_by_frame, "observed"),
        "predicted_frames": count_frames_with_provenance(boxes_by_frame, {"predicted"}),
        "provenance_counts": count_values(provenances),
        "tracked_visible_frames": len(covered_frames),
        "tracking_coverage": ratio(len(covered_frames), len(relevant_frames)),
        "tracking_coverage_denominator": len(relevant_frames),
        "track_ends": [{"track_id": tid, "frame": frame} for frame, tid in ends],
        "track_fragmentation": len(track_ids),
        "track_starts": [{"track_id": tid, "frame": frame} for frame, tid in starts],
    }


def privacy_metrics_for_object(
    *,
    clip_id: str,
    obj: dict[str, Any],
    mode: str,
    privacy_regions: dict[int, list[PrivacyRegion]],
    intervals: list[dict[str, Any]],
    object_track_ids: dict[str, set[int]],
    tracks: dict[int, list[TrackBox]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    object_id = str(obj["object_id"])
    if mode == MODE_DETECTOR:
        return (
            {
                "accepted_after_refinement_intervals": 0,
                "accepted_automatic_intervals": 0,
                "automatic_privacy_coverage": None,
                "covered_frame_count": 0,
                "final_privacy_coverage": None,
                "not_applicable": True,
                "privacy_required_frames": 0,
                "provenance_counts": {},
                "rejected_intervals": 0,
                "reviewed_correction_frame_count": 0,
                "unresolved_privacy_frames": 0,
            },
            [],
        )
    if obj["policy"] != "redact":
        return (
            {
                "accepted_after_refinement_intervals": 0,
                "accepted_automatic_intervals": 0,
                "automatic_privacy_coverage": None,
                "covered_frame_count": 0,
                "final_privacy_coverage": None,
                "privacy_required_frames": 0,
                "provenance_counts": {},
                "rejected_intervals": 0,
                "reviewed_correction_frame_count": 0,
                "unresolved_privacy_frames": 0,
            },
            [],
        )

    required_frames = sorted(expand_ranges(obj.get("privacy_required_ranges", [])))
    detector_scored_frames = set(expand_ranges(obj.get("detector_scored_ranges", [])))
    object_regions = object_privacy_regions(
        obj,
        privacy_regions,
        object_track_ids.get(object_id, set()),
        all_redact_object_ids=[object_id],
    )
    covered: dict[int, PrivacyRegion] = {}
    unresolved_input_frames: set[int] = set()
    invalid_boxes: list[int] = []
    for frame in required_frames:
        candidates = object_regions.get(frame, [])
        if not candidates:
            continue
        accepted_candidates = [
            region
            for region in candidates
            if privacy_region_counts_for_mode(region, frame, mode, intervals)
        ]
        if accepted_candidates:
            covered[frame] = accepted_candidates[0]
        if any(region.state != "covered" for region in candidates):
            unresolved_input_frames.add(frame)
        for region in candidates:
            if region.bbox_xyxy is not None and not valid_box(region.bbox_xyxy):
                invalid_boxes.append(frame)

    covered_frames = set(covered)
    uncovered_frames = sorted(set(required_frames) - covered_frames)
    visible_uncovered = sorted(detector_scored_frames & set(uncovered_frames))
    failures = [
        failure(
            clip_id,
            object_id,
            frame,
            obj["policy"],
            "privacy",
            "none",
            "privacy_required_frame_uncovered",
            obj.get("review_status"),
        )
        for frame in uncovered_frames
    ]
    failures.extend(
        failure(
            clip_id,
            object_id,
            frame,
            obj["policy"],
            "privacy",
            "invalid_box",
            "invalid_privacy_box",
            obj.get("review_status"),
        )
        for frame in invalid_boxes
    )
    provenance_counts = count_values(region.provenance for region in covered.values())
    interval_counts = count_interval_statuses(intervals)
    return (
        {
            "accepted_after_refinement_intervals": interval_counts.get(
                "accepted_after_refinement", 0
            ),
            "accepted_automatic_intervals": interval_counts.get("accepted_automatically", 0),
            "automatically_generated_frame_count": sum(
                count
                for provenance, count in provenance_counts.items()
                if provenance in SYNTHETIC_PROVENANCES
            ),
            "automatic_privacy_coverage": ratio(len(covered_frames), len(required_frames))
            if mode == MODE_AUTOMATIC
            else None,
            "covered_frame_count": len(covered_frames),
            "detector_scored_visible_uncovered_frames": visible_uncovered,
            "detector_scored_visible_uncovered_frame_count": len(visible_uncovered),
            "final_privacy_coverage": ratio(len(covered_frames), len(required_frames))
            if mode == MODE_REVIEWED_FINAL
            else None,
            "invalid_box_count": len(invalid_boxes),
            "needs_review_intervals": interval_counts.get("needs_review", 0),
            "preserved_or_manual_frame_count": sum(
                count
                for provenance, count in provenance_counts.items()
                if is_reviewed_provenance(provenance)
            ),
            "privacy_required_frames": len(required_frames),
            "provenance_counts": provenance_counts,
            "rejected_intervals": interval_counts.get("rejected", 0),
            "reviewed_correction_frame_count": sum(
                count
                for provenance, count in provenance_counts.items()
                if is_reviewed_provenance(provenance)
            ),
            "uncovered_privacy_frames": uncovered_frames,
            "uncovered_privacy_frame_count": len(uncovered_frames),
            "unresolved_input_privacy_frames": sorted(unresolved_input_frames),
            "unresolved_input_privacy_frame_count": len(unresolved_input_frames),
        },
        failures,
    )


def stability_metrics_for_object(
    *,
    obj: dict[str, Any],
    tracks: dict[int, list[TrackBox]],
    privacy_regions: dict[int, list[PrivacyRegion]],
    object_track_ids: set[int],
    use_privacy_regions: bool,
) -> dict[str, Any]:
    frames = relevant_tracking_frames(obj)
    boxes: list[tuple[int, Box, str]] = []
    if use_privacy_regions:
        for frame in frames:
            candidates = [region for region in privacy_regions.get(frame, []) if region.is_covered]
            if candidates:
                boxes.append((frame, candidates[0].bbox_xyxy, candidates[0].provenance))
    else:
        for frame in frames:
            candidates = [box for box in tracks.get(frame, []) if box.track_id in object_track_ids]
            if candidates:
                boxes.append((frame, candidates[0].bbox_xyxy, candidates[0].provenance))
    jumps: list[float] = []
    normalized_jumps: list[float] = []
    scale_changes: list[float] = []
    area_changes: list[float] = []
    provenance_transitions = 0
    flicker_events = 0
    covered_frames = {frame for frame, _, _ in boxes}
    previous_covered = None
    for frame in frames:
        covered = frame in covered_frames
        if previous_covered is not None and covered != previous_covered:
            flicker_events += 1
        previous_covered = covered
    for previous, current in zip(boxes, boxes[1:]):
        if current[0] != previous[0] + 1:
            continue
        prev_box = previous[1]
        curr_box = current[1]
        center_jump = distance(center(prev_box), center(curr_box))
        jumps.append(center_jump)
        normalized_jumps.append(center_jump / max(diagonal(prev_box), 1.0))
        prev_area = area(prev_box)
        curr_area = area(curr_box)
        area_changes.append(abs(curr_area - prev_area))
        scale_changes.append(abs(math.log(max(curr_area, 1.0) / max(prev_area, 1.0))))
        if previous[2] != current[2]:
            provenance_transitions += 1
    return {
        "box_area_change_max": round(max(area_changes), 6) if area_changes else 0.0,
        "center_jump_max": round(max(jumps), 6) if jumps else 0.0,
        "flicker_events": flicker_events,
        "normalized_center_jump_max": round(max(normalized_jumps), 6) if normalized_jumps else 0.0,
        "scale_jump_max": round(max(scale_changes), 6) if scale_changes else 0.0,
        "suspicious_provenance_transitions": provenance_transitions,
        "track_fragmentation": len(object_track_ids),
    }


def render_metrics_for_run(
    *,
    mode: str,
    source_video_path: Path,
    privacy_regions_path: Path | None,
    rendered_video_path: Path | None,
    render_manifest: dict[str, Any] | None,
    render_validation: dict[str, Any] | None,
    accepted_region_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if render_manifest is None:
        return ({"not_applicable": True, "reason": "render manifest not supplied"}, [])
    failures: list[dict[str, Any]] = []
    media_validation = (render_validation or {}).get("media_validation")
    source_hash = media.sha256_file(source_video_path)
    manifest_consistent = True
    if render_manifest.get("source_video_hash") != source_hash:
        manifest_consistent = False
        failures.append(
            failure(
                None, None, None, None, "render", "manifest", "source_video_hash_mismatch", None
            )
        )
    if privacy_regions_path is not None:
        privacy_hash = media.sha256_file(privacy_regions_path)
        if render_manifest.get("privacy_region_input_hash") != privacy_hash:
            manifest_consistent = False
            failures.append(
                failure(
                    None,
                    None,
                    None,
                    None,
                    "render",
                    "manifest",
                    "privacy_region_input_hash_mismatch",
                    None,
                )
            )
    if rendered_video_path is not None:
        rendered_hash = media.sha256_file(rendered_video_path)
        if render_manifest.get("output_video_hash") != rendered_hash:
            manifest_consistent = False
            failures.append(
                failure(
                    None, None, None, None, "render", "manifest", "output_video_hash_mismatch", None
                )
            )
    media_validation_passed = (
        None if media_validation is None else bool(media_validation.get("passed"))
    )
    if mode == MODE_REVIEWED_FINAL and render_manifest.get("final_approval_status") is not True:
        failures.append(
            failure(None, None, None, None, "render", "manifest", "final_approval_false", None)
        )
    if media_validation_passed is False:
        failures.append(
            failure(None, None, None, None, "render", "ffprobe", "media_validation_failed", None)
        )
    rendered_count = as_int(render_manifest.get("rendered_region_count"), default=0)
    if mode == MODE_REVIEWED_FINAL and rendered_count != accepted_region_count:
        failures.append(
            failure(
                None, None, None, None, "render", "manifest", "rendered_region_count_mismatch", None
            )
        )
    return (
        {
            "actual_pixel_privacy_verification": "not_measured",
            "audio_preservation": None
            if media_validation is None
            else media_validation.get("audio_ok"),
            "expected_regions_rendered": rendered_count == accepted_region_count,
            "expected_region_count": accepted_region_count,
            "frames_containing_expected_rendering": None,
            "manifest_consistent": manifest_consistent,
            "media_validation": media_validation,
            "media_validation_passed": media_validation_passed,
            "missing_rendered_regions_where_measurable": None
            if rendered_count == accepted_region_count
            else abs(rendered_count - accepted_region_count),
            "output_hash": render_manifest.get("output_video_hash"),
            "rendered_region_count": rendered_count,
            "rendering_mode": render_manifest.get("rendering_mode"),
        },
        failures,
    )


def acceptance_for_run(
    mode: str,
    object_metrics: dict[str, dict[str, Any]],
    intervals: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    render_metrics: dict[str, Any],
) -> dict[str, Any]:
    blocking_reasons: set[str] = set()
    interval_counts = count_interval_statuses(intervals)
    if mode == MODE_DETECTOR:
        return {"blocking_reasons": [], "final_status": STATUS_DETECTOR_EVALUATED}
    if interval_counts.get("needs_review", 0):
        blocking_reasons.add("needs_review_intervals")
    if interval_counts.get("rejected", 0):
        blocking_reasons.add("rejected_intervals")
    for metrics in object_metrics.values():
        privacy = metrics["privacy"]
        if privacy.get("uncovered_privacy_frame_count", 0):
            blocking_reasons.add("unresolved_privacy_frames")
        if privacy.get("detector_scored_visible_uncovered_frame_count", 0):
            blocking_reasons.add("uncovered_detector_scored_visible_frames")
        if privacy.get("invalid_box_count", 0):
            blocking_reasons.add("invalid_boxes")
    if any(item["failure_reason"] == "keep_person_false_redaction_overlap" for item in failures):
        blocking_reasons.add("keep_person_false_redaction")
    if any(item["failure_reason"] == "identity_switch_event" for item in failures):
        blocking_reasons.add("identity_switch_events")
    if mode == MODE_REVIEWED_FINAL:
        if render_metrics.get("manifest_consistent") is False:
            blocking_reasons.add("render_manifest_inconsistent")
        if render_metrics.get("media_validation_passed") is False:
            blocking_reasons.add("render_media_validation_failed")
        if render_metrics.get("expected_regions_rendered") is False:
            blocking_reasons.add("rendered_region_count_mismatch")
        final_status = (
            STATUS_PASSED_REVIEWED_FINAL
            if not blocking_reasons
            else STATUS_FAILED_PRIVACY_ACCEPTANCE
        )
    else:
        final_status = STATUS_PASSED_AUTOMATIC if not blocking_reasons else STATUS_REVIEW_REQUIRED
    return {"blocking_reasons": sorted(blocking_reasons), "final_status": final_status}


def aggregate_metrics(
    *,
    clip: dict[str, Any],
    mode: str,
    detector_threshold: float,
    object_metrics: dict[str, dict[str, Any]],
    intervals: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    render_metrics: dict[str, Any],
    acceptance: dict[str, Any],
    elapsed_seconds: float,
    paths: dict[str, Path | None],
) -> dict[str, Any]:
    detector_denominator = sum(
        item["detector"]["detector_scored_object_frames"] for item in object_metrics.values()
    )
    detector_hits = sum(
        item["detector"]["frames_with_valid_detection"] for item in object_metrics.values()
    )
    privacy_denominator = sum(
        item["privacy"].get("privacy_required_frames", 0)
        for item in object_metrics.values()
        if item["policy"] == "redact"
    )
    privacy_hits = sum(
        item["privacy"].get("covered_frame_count", 0)
        for item in object_metrics.values()
        if item["policy"] == "redact"
    )
    return {
        "acceptance": acceptance,
        "annotation_limitations": {
            "box_level_or_iou_coverage": "not_measured_without_per_frame_ground_truth_boxes",
            "pixel_level_rendered_privacy": "not_measured",
            "range_level_truth": True,
            "visual_review_evidence": (
                "referenced_when supplied; not equivalent to automated pixel verification"
            ),
        },
        "clip_id": clip["clip_id"],
        "detector": {
            "detector_scored_object_frames": detector_denominator,
            "frames_with_valid_detection": detector_hits,
            "object_frame_recall": ratio(detector_hits, detector_denominator),
            "threshold": detector_threshold,
        },
        "evaluation_mode": mode,
        "failure_count": len(failures),
        "interval_decisions": interval_decision_summary(intervals)["summary"],
        "inputs": {
            key: None if value is None else value.as_posix() for key, value in paths.items()
        },
        "privacy": {
            "covered_privacy_frames": privacy_hits,
            "privacy_required_frames": privacy_denominator,
            "privacy_coverage": ratio(privacy_hits, privacy_denominator),
        },
        "render": render_metrics,
        "runtime": {
            "elapsed_seconds": round(elapsed_seconds, 6),
            "processing_fps": round(int(clip["source_frame_count"]) / elapsed_seconds, 6)
            if elapsed_seconds > 0
            else 0.0,
        },
        "schema_version": SCHEMA_VERSION,
        "source_video": clip["source_video"],
        "source_video_frame_count": clip["source_frame_count"],
    }


def write_evaluation_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "metrics.json", payload["metrics"])
    write_json(output_dir / "object_metrics.json", payload["object_metrics"])
    write_json(output_dir / "interval_decisions.json", payload["interval_decisions"])
    write_json(output_dir / "failures.json", payload["failures"])
    write_json(output_dir / "render_validation.json", payload["render_validation"])
    (output_dir / "report.md").write_text(payload["report"], encoding="utf-8")
    write_summary_table(output_dir / "summary_table.csv", payload["summary_rows"])


def report_markdown(
    metrics: dict[str, Any],
    object_metrics: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
    render_metrics: dict[str, Any],
) -> str:
    lines = [
        "# Face Pipeline Evaluation Report",
        "",
        f"- Clip: `{metrics.get('clip_id')}`",
        f"- Mode: `{metrics.get('evaluation_mode')}`",
        f"- Final status: `{metrics.get('acceptance', {}).get('final_status')}`",
        f"- Blocking reasons: `{metrics.get('acceptance', {}).get('blocking_reasons', [])}`",
        "",
        "## Summary",
        "",
        "| Object | Policy | Detector recall | Privacy coverage | Tracking coverage |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for object_id, item in object_metrics.items():
        privacy_ratio = item["privacy"].get("final_privacy_coverage")
        if privacy_ratio is None:
            privacy_ratio = item["privacy"].get("automatic_privacy_coverage")
        lines.append(
            "| "
            f"`{object_id}` | `{item['policy']}` | "
            f"{format_ratio(item['detector']['object_frame_recall'])} | "
            f"{format_ratio(privacy_ratio)} | "
            f"{format_ratio(item['tracking']['tracking_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Rendering",
            "",
            "- Pixel-level rendered privacy verification: "
            f"`{render_metrics.get('actual_pixel_privacy_verification', 'not_applicable')}`",
            f"- Render manifest consistent: `{render_metrics.get('manifest_consistent')}`",
            f"- Media validation passed: `{render_metrics.get('media_validation_passed')}`",
            "",
            "## Failures",
            "",
        ]
    )
    if not failures:
        lines.append("No blocking failures were recorded for this mode.")
    else:
        lines.append("| Stage | Object | Frame/interval | Reason | Evidence |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in failures[:80]:
            frame_text = (
                f"{item.get('frame_start')}-{item.get('frame_end')}"
                if item.get("frame_start") is not None
                else str(item.get("frame"))
            )
            evidence = item.get("visual_evidence") or ""
            lines.append(
                f"| `{item.get('evaluated_stage')}` | `{item.get('object_id')}` | "
                f"`{frame_text}` | `{item.get('failure_reason')}` | `{evidence}` |"
            )
        if len(failures) > 80:
            lines.append(
                f"\nAdditional failures omitted from report table: `{len(failures) - 80}`."
            )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "This report distinguishes raw detector recall, ordinary tracking, "
            "synthetic or preserved privacy coverage, and rendering manifest/media checks. "
            "It does not make a universal zero-leak, perfect-recall, or "
            "production-readiness claim.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_rows(
    metrics: dict[str, Any], object_metrics: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = [
        {
            "clip_id": metrics.get("clip_id"),
            "final_status": metrics.get("acceptance", {}).get("final_status"),
            "metric": "detector_object_frame_recall",
            "mode": metrics.get("evaluation_mode"),
            "object_id": "all",
            "value": metrics.get("detector", {}).get("object_frame_recall"),
        },
        {
            "clip_id": metrics.get("clip_id"),
            "final_status": metrics.get("acceptance", {}).get("final_status"),
            "metric": "privacy_coverage",
            "mode": metrics.get("evaluation_mode"),
            "object_id": "all_redact",
            "value": metrics.get("privacy", {}).get("privacy_coverage"),
        },
    ]
    for object_id, item in object_metrics.items():
        rows.append(
            {
                "clip_id": metrics.get("clip_id"),
                "final_status": metrics.get("acceptance", {}).get("final_status"),
                "metric": "detector_recall",
                "mode": metrics.get("evaluation_mode"),
                "object_id": object_id,
                "value": item["detector"]["object_frame_recall"],
            }
        )
        if item["policy"] == "redact":
            rows.append(
                {
                    "clip_id": metrics.get("clip_id"),
                    "final_status": metrics.get("acceptance", {}).get("final_status"),
                    "metric": "privacy_coverage",
                    "mode": metrics.get("evaluation_mode"),
                    "object_id": object_id,
                    "value": item["privacy"].get("final_privacy_coverage")
                    or item["privacy"].get("automatic_privacy_coverage"),
                }
            )
    return rows


def write_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["clip_id", "mode", "final_status", "object_id", "metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_visuals(
    *,
    source_video_path: Path,
    output_dir: Path,
    failures: list[dict[str, Any]],
    privacy_regions: dict[int, list[PrivacyRegion]],
    source_frame_count: int,
    overwrite: bool,
) -> dict[str, str]:
    if not failures:
        return {}
    visual_dir = output_dir / "visual_evidence"
    visual_dir.mkdir(parents=True, exist_ok=True)
    frame_samples = []
    for item in failures:
        frame = item.get("frame")
        if frame is None:
            frame = item.get("frame_start")
        if frame is not None:
            frame_samples.append(max(0, min(source_frame_count - 1, int(frame))))
    selected_frames = sorted(dict.fromkeys(frame_samples))[:24]
    paths: dict[str, str] = {}
    if selected_frames:
        sheet_path = visual_dir / "failure_contact_sheet.jpg"
        create_failure_contact_sheet(
            source_video_path,
            sheet_path,
            selected_frames,
            privacy_regions,
            overwrite=overwrite,
        )
        paths["contact_sheet"] = sheet_path.relative_to(output_dir).as_posix()
    interval_failures = [
        item
        for item in failures
        if item.get("frame_start") is not None and item.get("frame_end") is not None
    ][:4]
    for index, item in enumerate(interval_failures, start=1):
        start = max(0, int(item["frame_start"]) - 10)
        end = min(source_frame_count - 1, int(item["frame_end"]) + 10)
        clip_path = visual_dir / f"failure_clip_{index:02d}_{start}_{end}.mp4"
        create_failure_overlay_clip(
            source_video_path,
            clip_path,
            start,
            end,
            privacy_regions,
            overwrite=overwrite,
        )
        paths[f"clip_{index}"] = clip_path.relative_to(output_dir).as_posix()
    return paths


def attach_visual_paths(failures: list[dict[str, Any]], visual_paths: dict[str, str]) -> None:
    if not visual_paths:
        return
    default = visual_paths.get("contact_sheet")
    for item in failures:
        if item.get("visual_evidence") is None:
            item["visual_evidence"] = default
        if item.get("frame_start") is not None:
            item["visual_evidence"] = next(
                (value for key, value in visual_paths.items() if key.startswith("clip_")),
                default,
            )


def create_failure_contact_sheet(
    source_video_path: Path,
    output_path: Path,
    frames: list[int],
    privacy_regions: dict[int, list[PrivacyRegion]],
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    capture = cv.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source_video_path}")
    tiles: list[np.ndarray] = []
    try:
        for frame_index in frames:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            tiles.append(
                resize_for_sheet(draw_failure_overlay(frame, frame_index, privacy_regions))
            )
    finally:
        capture.release()
    if not tiles:
        return
    columns = min(4, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    cv.imwrite(str(output_path), sheet)


def create_failure_overlay_clip(
    source_video_path: Path,
    output_path: Path,
    start: int,
    end: int,
    privacy_regions: dict[int, list[PrivacyRegion]],
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    info = media.probe_media(source_video_path)
    fps = rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    capture = cv.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source_video_path}")
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info.video_stream.width, info.video_stream.height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not write video: {output_path}")
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, start)
        frame_index = start
        while frame_index <= end:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(draw_failure_overlay(frame, frame_index, privacy_regions))
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def draw_failure_overlay(
    frame: np.ndarray,
    frame_index: int,
    privacy_regions: dict[int, list[PrivacyRegion]],
) -> np.ndarray:
    output = frame.copy()
    cv.putText(
        output,
        f"frame {frame_index}",
        (18, 36),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )
    for region in privacy_regions.get(frame_index, []):
        if region.bbox_xyxy is None:
            continue
        color = (255, 255, 0) if region.is_covered else (0, 0, 255)
        x1, y1, x2, y2 = region.bbox_xyxy
        cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            output,
            region.provenance[:24],
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv.LINE_AA,
        )
    return output


def resize_for_sheet(frame: np.ndarray, width: int = 480) -> np.ndarray:
    scale = width / frame.shape[1]
    return cv.resize(frame, (width, round(frame.shape[0] * scale)), interpolation=cv.INTER_AREA)


def failure(
    clip_id: str | None,
    object_id: str | None,
    frame: int | None,
    policy: str | None,
    stage: str,
    provenance: str | None,
    reason: str,
    review_status: str | None,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> dict[str, Any]:
    return {
        "clip": clip_id,
        "evaluated_stage": stage,
        "expected_policy": policy,
        "failure_reason": reason,
        "frame": frame,
        "frame_end": frame_end,
        "frame_start": frame_start,
        "object_id": object_id,
        "observed_provenance": provenance,
        "review_status": review_status,
        "visual_evidence": None,
    }


def keep_person_false_redaction_failures(
    *,
    clip_id: str,
    objects: list[dict[str, Any]],
    privacy_regions: dict[int, list[PrivacyRegion]],
    object_track_ids: dict[str, set[int]],
    tracks: dict[int, list[TrackBox]],
) -> list[dict[str, Any]]:
    keep_objects = [obj for obj in objects if obj["policy"] == "keep"]
    failures: list[dict[str, Any]] = []
    for obj in keep_objects:
        keep_frames = sorted(expand_ranges(obj.get("keep_clear_ranges", [])))
        keep_track_ids = object_track_ids.get(str(obj["object_id"]), set())
        for frame in keep_frames:
            keep_boxes = [box for box in tracks.get(frame, []) if box.track_id in keep_track_ids]
            if not keep_boxes:
                continue
            for region in privacy_regions.get(frame, []):
                if not region.is_covered:
                    continue
                if any(
                    overlap_fraction(region.bbox_xyxy, box.bbox_xyxy) > 0.01 for box in keep_boxes
                ):
                    failures.append(
                        failure(
                            clip_id,
                            str(obj["object_id"]),
                            frame,
                            obj["policy"],
                            "privacy",
                            region.provenance,
                            "keep_person_false_redaction_overlap",
                            obj.get("review_status"),
                        )
                    )
    return failures


def object_privacy_regions(
    obj: dict[str, Any],
    privacy_regions: dict[int, list[PrivacyRegion]],
    object_track_ids: set[int],
    *,
    all_redact_object_ids: list[str],
) -> dict[int, list[PrivacyRegion]]:
    object_id = str(obj["object_id"])
    mapped: dict[int, list[PrivacyRegion]] = {}
    for frame, regions in privacy_regions.items():
        for region in regions:
            if region.target_track_ids and object_track_ids:
                if not set(region.target_track_ids) & object_track_ids:
                    continue
            elif len(all_redact_object_ids) != 1 or object_id not in all_redact_object_ids:
                continue
            mapped.setdefault(frame, []).append(region)
    return mapped


def privacy_region_counts_for_mode(
    region: PrivacyRegion,
    frame: int,
    mode: str,
    intervals: list[dict[str, Any]],
) -> bool:
    if not region.is_covered:
        return False
    if mode == MODE_REVIEWED_FINAL:
        return True
    if mode != MODE_AUTOMATIC:
        return False
    if is_reviewed_provenance(region.provenance):
        return False
    if region.provenance in AUTOMATIC_PROVENANCES:
        return True
    interval = interval_for_frame(intervals, frame)
    if interval is None:
        return region.provenance not in SYNTHETIC_PROVENANCES
    return interval.get("status") in {"accepted_automatically", "accepted_after_refinement"}


def infer_object_track_ids(
    objects: list[dict[str, Any]],
    adjudication: dict[tuple[str, str, int], dict[str, Any]],
    tracks: dict[int, list[TrackBox]],
    clip_id: str,
) -> dict[str, set[int]]:
    candidate_to_track_ids: dict[str, set[int]] = {}
    for boxes in tracks.values():
        for box in boxes:
            if box.candidate_id:
                candidate_to_track_ids.setdefault(box.candidate_id, set()).add(box.track_id)
    result = {str(obj["object_id"]): set() for obj in objects}
    for obj in objects:
        object_id = str(obj["object_id"])
        for frame in relevant_tracking_frames(obj):
            decision = adjudication.get((clip_id, object_id, frame))
            if not decision or decision.get("choice") != "candidate":
                continue
            result[object_id].update(
                candidate_to_track_ids.get(str(decision["candidate_id"]), set())
            )
    return result


def load_detections(path: Path) -> dict[int, list[dict[str, Any]]]:
    records: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["frame_index"])] = [
                dict(item) for item in record.get("detections", [])
            ]
    return records


def load_adjudication(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = load_json(path)
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.get("items", []):
        decision = dict(item.get("decision", {}))
        result[(str(item["clip"]), str(item["target_id"]), int(item["frame"]))] = decision
    return result


def load_tracks(path: Path | None) -> dict[int, list[TrackBox]]:
    if path is None:
        return {}
    records: dict[int, list[TrackBox]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame_index"])
            boxes: list[TrackBox] = []
            for item in record.get("track_boxes", []):
                bbox = as_box(item.get("bbox_xyxy"))
                if bbox is None:
                    continue
                boxes.append(
                    TrackBox(
                        frame_index=frame,
                        track_id=int(item["track_id"]),
                        bbox_xyxy=bbox,
                        provenance=str(item.get("provenance") or "unknown"),
                        score=None if item.get("score") is None else float(item["score"]),
                        candidate_id=item.get("candidate_id"),
                        quality_status=str(item.get("quality_status") or "unknown"),
                    )
                )
            records[frame] = boxes
    return records


def load_privacy_regions(path: Path | None) -> dict[int, list[PrivacyRegion]]:
    if path is None:
        return {}
    records: dict[int, list[PrivacyRegion]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame_index"])
            raw_regions = record.get("privacy_regions")
            if raw_regions is None:
                raw_regions = [record]
            regions: list[PrivacyRegion] = []
            for item in raw_regions:
                bbox = as_box(item.get("bbox_xyxy"))
                regions.append(
                    PrivacyRegion(
                        frame_index=frame,
                        bbox_xyxy=bbox,
                        provenance=str(item.get("provenance") or item.get("method") or "unknown"),
                        state=str(item.get("state") or ("covered" if bbox else "unresolved")),
                        target_track_ids=tuple(
                            int(value) for value in item.get("target_track_ids", [])
                        ),
                        source=dict(item),
                    )
                )
            records[frame] = regions
    return records


def load_interval_decisions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load_json(path)
    return [dict(item) for item in payload.get("interval_decisions", [])]


def interval_decision_summary(intervals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "interval_decisions": intervals,
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "count": len(intervals),
            "status_counts": count_interval_statuses(intervals),
            "strategy_counts": count_values(str(item.get("strategy")) for item in intervals),
        },
    }


def clip_by_id(ground_truth: dict[str, Any], clip_id: str) -> dict[str, Any]:
    for clip in ground_truth.get("clips", []):
        if clip.get("clip_id") == clip_id:
            return clip
    raise ValueError(f"clip_id not found in ground truth: {clip_id}")


def object_ground_truth_summary(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector_ignore_frames": len(expand_ranges(obj.get("detector_ignore_ranges", []))),
        "detector_scored_frames": len(expand_ranges(obj.get("detector_scored_ranges", []))),
        "keep_clear_frames": len(expand_ranges(obj.get("keep_clear_ranges", []))),
        "keyframe_boxes_available": bool(obj.get("keyframe_boxes")),
        "per_frame_boxes_available": bool(obj.get("per_frame_boxes")),
        "privacy_required_frames": len(expand_ranges(obj.get("privacy_required_ranges", []))),
        "reference_roi_xyxy": obj.get("reference_roi_xyxy"),
    }


def relevant_tracking_frames(obj: dict[str, Any]) -> list[int]:
    ranges = (
        obj.get("privacy_required_ranges")
        or obj.get("keep_clear_ranges")
        or obj.get("detector_scored_ranges", [])
    )
    return sorted(expand_ranges(ranges))


def detection_by_candidate(
    detections: list[dict[str, Any]], candidate_id: str
) -> dict[str, Any] | None:
    return next((item for item in detections if item.get("candidate_id") == candidate_id), None)


def count_interval_statuses(intervals: list[dict[str, Any]]) -> dict[str, int]:
    return count_values(str(item.get("status") or "unknown") for item in intervals)


def interval_for_frame(intervals: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in intervals
            if as_int(item.get("frame_start"), default=-1)
            <= frame
            <= as_int(item.get("frame_end"), default=-2)
        ),
        None,
    )


def expand_ranges(ranges: Any) -> set[int]:
    frames: set[int] = set()
    for start, end in ranges or []:
        frames.update(range(int(start), int(end) + 1))
    return frames


def overlapping_frames(left: Any, right: Any) -> bool:
    return bool(expand_ranges(left) & expand_ranges(right))


def longest_run(frames: list[int] | set[int]) -> int:
    ordered = sorted(frames)
    if not ordered:
        return 0
    best = current = 1
    previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
        previous = frame
    return max(best, current)


def distribution(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "max": None, "mean": None, "min": None}
    return {
        "count": len(values),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
        "min": round(min(values), 6),
    }


def count_frames_with_prefix(boxes_by_frame: dict[int, list[TrackBox]], prefix: str) -> int:
    return sum(
        1
        for boxes in boxes_by_frame.values()
        if any(box.provenance.startswith(prefix) for box in boxes)
    )


def count_frames_with_provenance(
    boxes_by_frame: dict[int, list[TrackBox]],
    provenances: set[str],
) -> int:
    return sum(
        1
        for boxes in boxes_by_frame.values()
        if any(box.provenance in provenances for box in boxes)
    )


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def is_reviewed_provenance(provenance: str) -> bool:
    lowered = provenance.lower()
    return any(marker in lowered for marker in REVIEWED_PROVENANCE_MARKERS)


def valid_box(box: Box) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def as_box(values: Any) -> Box | None:
    if not isinstance(values, list | tuple) or len(values) != 4:
        return None
    try:
        return tuple(int(round(float(value))) for value in values)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def diagonal(box: Box) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def area(box: Box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap_fraction(left: Box, right: Box) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = area((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else 0.0
    return intersection / max(area(left), 1.0)


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def format_ratio(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def rational_to_float(value: str | None) -> float | None:
    if value is None or value == "0/0":
        return None
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    return None if denominator_value == 0 else float(numerator) / denominator_value


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
