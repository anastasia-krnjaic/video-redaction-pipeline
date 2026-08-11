from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import webbrowser
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import cv2 as cv

from redactor import media

SCHEMA_VERSION = "plate_annotation_package.v1"
STATE_SCHEMA_VERSION = "plate_annotation_state.v2"
EXPORT_SCHEMA_VERSION = "plate_annotation_review_export.v1"
GENERATOR_VERSION = "plate_annotation_package_builder.v1"
PACKAGE_IDENTITY_SCHEMA_VERSION = "plate_annotation_package_identity.v1"
PHASE_A_CERTIFIED_CONFIG_VERSION = "phase_a_plate_annotation_config.v1"
CUSTOM_CERTIFIED_CONFIG_VERSION = "custom_plate_annotation_config.v1"
ALLOWED_CERTIFIED_CONFIG_VERSIONS = {
    CUSTOM_CERTIFIED_CONFIG_VERSION,
    PHASE_A_CERTIFIED_CONFIG_VERSION,
}

DEFAULT_OUTPUT_ROOT = Path("artifacts/plate_ground_truth_annotation_v1")
DEFAULT_DAY_VIDEO = Path("artifacts/eval_day_plate.mp4")
DEFAULT_NIGHT_VIDEO = Path("artifacts/eval_night_selective.mp4")
DEFAULT_DRAFT_PATH = Path(
    "artifacts/license_plate_detector_finalists_v2_1/annotation_review/"
    "all_visible_plate_annotations.json"
)
DEFAULT_DRAFT_MAPPING_AUDIT = Path(
    "artifacts/license_plate_detector_finalists_v2_1/clip_mapping_audit.json"
)
EXCLUDED_CLOSE_VIDEO = Path("artifacts/eval_day_plate_close.mp4")

DAY_VIDEO_SHA256 = "63d41905e2b6c65a3cdc72aa73ee31668fe1c1b4fdfefa2c5688cd38c90d1097"
NIGHT_VIDEO_SHA256 = "0b55aa3a497941711ac8634a049c557637e20928e8d3bd8b63e5b030b6b272e0"

STATUS_UNREVIEWED = "unreviewed"
STATUS_REVIEWED_BOX = "reviewed_box"
STATUS_NEEDS_RECHECK = "needs_recheck"
STATUS_TEMPORAL_GT_CONFLICT = "temporal_gt_conflict"
ALLOWED_MAIN_STATUSES = {
    STATUS_NEEDS_RECHECK,
    STATUS_REVIEWED_BOX,
    STATUS_TEMPORAL_GT_CONFLICT,
    STATUS_UNREVIEWED,
}

SECONDARY_ADDITIONAL = "additional_localizable_plate"
SECONDARY_IGNORE_SMALL = "ignore_small"
SECONDARY_AMBIGUOUS = "ambiguous_review_required"
ALLOWED_SECONDARY_KINDS = {SECONDARY_ADDITIONAL, SECONDARY_IGNORE_SMALL, SECONDARY_AMBIGUOUS}

FRAME_ASSET_NAME_TEMPLATE = "{benchmark}_{parent_frame:06d}_{kind}.{extension}"
MAX_SAVE_PAYLOAD_BYTES = 1024 * 1024


class StaleStateRevisionError(ValueError):
    def __init__(self, *, current_revision: int, expected_revision: int) -> None:
        super().__init__(
            f"stale state revision: expected {expected_revision}, current {current_revision}"
        )
        self.current_revision = current_revision
        self.expected_revision = expected_revision


@dataclass(frozen=True)
class PlateTargetConfig:
    benchmark_id: str
    video_path: Path
    expected_sha256: str | None
    total_video_frames: int | None
    parent_frame_start: int
    parent_frame_end: int
    dense_frame_offset: int
    target_id: str
    object_type: str
    action: str
    broad_roi_xyxy: tuple[int, int, int, int]
    source_video_label: str
    source_interval: str
    expected_width: int
    expected_height: int
    expected_fps: str | None = "30000/1001"

    @property
    def mandatory_frame_count(self) -> int:
        return self.parent_frame_end - self.parent_frame_start + 1

    @property
    def dense_frame_start(self) -> int:
        return self.parent_frame_start + self.dense_frame_offset

    @property
    def dense_frame_end(self) -> int:
        return self.parent_frame_end + self.dense_frame_offset

    def with_video_path(self, video_path: Path) -> PlateTargetConfig:
        return replace(self, video_path=video_path)


DEFAULT_TARGETS = (
    PlateTargetConfig(
        benchmark_id="day_plate",
        video_path=DEFAULT_DAY_VIDEO,
        expected_sha256=DAY_VIDEO_SHA256,
        total_video_frames=570,
        parent_frame_start=240,
        parent_frame_end=479,
        dense_frame_offset=-240,
        target_id="plate_day_01",
        object_type="license_plate",
        action="redact",
        broad_roi_xyxy=(650, 380, 1060, 570),
        source_video_label="bodycam_sample.mp4",
        source_interval="00:01:41.000-00:01:49.000",
        expected_width=1920,
        expected_height=1080,
    ),
    PlateTargetConfig(
        benchmark_id="night_selective",
        video_path=DEFAULT_NIGHT_VIDEO,
        expected_sha256=NIGHT_VIDEO_SHA256,
        total_video_frames=1589,
        parent_frame_start=420,
        parent_frame_end=779,
        dense_frame_offset=-420,
        target_id="plate_night_01",
        object_type="license_plate",
        action="redact",
        broad_roi_xyxy=(60, 560, 500, 900),
        source_video_label="night_candidate.mp4",
        source_interval="00:04:56.000-00:05:08.000",
        expected_width=1920,
        expected_height=1080,
    ),
)


def default_plate_targets(
    *,
    day_video_path: Path = DEFAULT_DAY_VIDEO,
    night_video_path: Path = DEFAULT_NIGHT_VIDEO,
) -> tuple[PlateTargetConfig, PlateTargetConfig]:
    day, night = DEFAULT_TARGETS
    return day.with_video_path(day_video_path), night.with_video_path(night_video_path)


def build_plate_annotation_package(
    *,
    day_video_path: Path = DEFAULT_DAY_VIDEO,
    night_video_path: Path = DEFAULT_NIGHT_VIDEO,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
    include_drafts: bool = True,
    draft_path: Path = DEFAULT_DRAFT_PATH,
    draft_mapping_audit_path: Path = DEFAULT_DRAFT_MAPPING_AUDIT,
    run_id: str | None = None,
    verify_git_ignore: bool = True,
    repo_root: Path | None = None,
    targets: tuple[PlateTargetConfig, ...] | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or Path.cwd()).resolve()
    output_root = output_root.resolve()
    if verify_git_ignore:
        assert_annotation_root_is_ignored(output_root, repo_root)

    if targets is None:
        target_configs = default_plate_targets(
            day_video_path=day_video_path,
            night_video_path=night_video_path,
        )
        certified_config_version = PHASE_A_CERTIFIED_CONFIG_VERSION
    else:
        target_configs = targets
        certified_config_version = CUSTOM_CERTIFIED_CONFIG_VERSION
    run_name = run_id or unique_run_id(output_root)
    package_root = output_root / run_name
    temp_root = output_root / f".{run_name}.{uuid4().hex}.tmp"
    if package_root.exists() and any(package_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{package_root} already exists; pass --overwrite to replace it")
    if temp_root.exists():
        shutil.rmtree(temp_root)

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        _build_into_temp_root(
            package_root=temp_root,
            final_package_root=package_root,
            targets=target_configs,
            certified_config_version=certified_config_version,
            include_drafts=include_drafts,
            draft_path=draft_path,
            draft_mapping_audit_path=draft_mapping_audit_path,
            repo_root=repo_root,
        )
        if package_root.exists() and overwrite:
            shutil.rmtree(package_root)
        os.replace(temp_root, package_root)
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise

    validation = validate_plate_annotation_package(package_root)
    if not validation["passed"]:
        raise ValueError("; ".join(validation["errors"]))
    return summarize_plate_annotation_package(package_root) | {
        "package_root": package_root.as_posix(),
        "validation": validation,
    }


def validate_plate_annotation_package(
    package_root: Path,
    *,
    level: str = "structural",
) -> dict[str, Any]:
    if level not in {"structural", "completed_export"}:
        raise ValueError(f"unsupported validation level: {level}")
    package_root = package_root.resolve()
    errors: list[str] = []
    try:
        manifest = read_json(package_root / "package_manifest.json")
        review_data = read_json(package_root / "review_data.json")
        identity = read_json(package_root / "package_identity.json")
        state = read_json(package_root / "annotations.draft.json")
        progress = read_json(package_root / "progress.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"errors": [str(exc)], "passed": False, "schema_version": SCHEMA_VERSION}

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("package_manifest.json schema_version mismatch")
    if review_data.get("schema_version") != "plate_annotation_review_data.v1":
        errors.append("review_data.json schema_version mismatch")
    if identity.get("schema_version") != PACKAGE_IDENTITY_SCHEMA_VERSION:
        errors.append("package_identity.json schema_version mismatch")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        errors.append("annotations.draft.json schema_version mismatch")
    if not isinstance(state.get("state_revision"), int) or state.get("state_revision", -1) < 0:
        errors.append("annotations.draft.json state_revision must be a non-negative integer")

    frame_records = review_data.get("frames")
    if not isinstance(frame_records, list):
        errors.append("review_data.json frames must be a list")
        frame_records = []
    state_frames = state.get("frames")
    if not isinstance(state_frames, dict):
        errors.append("annotations.draft.json frames must be an object")
        state_frames = {}

    package_id = manifest.get("package_id")
    if (
        package_id != state.get("package_id")
        or package_id != review_data.get("package_id")
        or package_id != identity.get("package_id")
    ):
        errors.append("package_id mismatch")
    if manifest.get("generator", {}).get("version") != GENERATOR_VERSION:
        errors.append("package generator version mismatch")
    if identity.get("generator_version") != GENERATOR_VERSION:
        errors.append("package identity generator version mismatch")

    certified_config_version = manifest.get("certified_config_version")
    if certified_config_version not in ALLOWED_CERTIFIED_CONFIG_VERSIONS:
        errors.append(f"unknown certified_config_version: {certified_config_version}")
    if identity.get("certified_config_version") != certified_config_version:
        errors.append("certified_config_version mismatch between manifest and identity")
    if manifest.get("package_identity_path") != "package_identity.json":
        errors.append("package_identity_path mismatch")
    if identity.get("package_manifest_sha256") != media.sha256_file(
        package_root / "package_manifest.json"
    ):
        errors.append("package manifest identity hash mismatch")
    if identity.get("review_data_sha256") != media.sha256_file(package_root / "review_data.json"):
        errors.append("review data identity hash mismatch")

    identity_targets = identity.get("certified_targets")
    if not isinstance(identity_targets, dict):
        errors.append("package identity certified_targets must be an object")
        identity_targets = {}
    if identity.get("certified_target_config_sha256") != stable_json_sha256(identity_targets):
        errors.append("certified target config identity hash mismatch")
    if certified_config_version == PHASE_A_CERTIFIED_CONFIG_VERSION:
        phase_a_targets = phase_a_certified_target_identities()
        if identity_targets != phase_a_targets:
            errors.append("Phase A certified target identity mismatch")
        authoritative_targets = phase_a_targets
    else:
        authoritative_targets = identity_targets

    target_records = manifest.get("targets")
    if not isinstance(target_records, dict):
        errors.append("manifest targets must be an object")
        target_records = {}
    expected_total = sum(
        max(int(target.get("expected_mandatory_count", -1)), 0)
        for target in authoritative_targets.values()
        if isinstance(target, dict)
    )
    if manifest.get("expected_mandatory_total") != expected_total:
        errors.append("expected_mandatory_total mismatch")
    if set(target_records) != set(authoritative_targets):
        errors.append("manifest target set does not match certified target set")

    benchmark_requirements: dict[str, dict[str, Any]] = {}
    for benchmark_id, requirement in sorted(authoritative_targets.items()):
        if not isinstance(requirement, dict):
            errors.append(f"{benchmark_id} certified target must be an object")
            continue
        manifest_target = target_records.get(benchmark_id)
        if not isinstance(manifest_target, dict):
            errors.append(f"{benchmark_id} missing from manifest targets")
            continue
        _validate_manifest_target(
            benchmark_id=benchmark_id,
            manifest_target=manifest_target,
            requirement=requirement,
            errors=errors,
        )
        benchmark_requirements[benchmark_id] = requirement
        video_record = requirement.get("evaluation_video") or {}
        video_path = resolve_evaluation_video_path(
            package_root,
            str(video_record.get("path") or ""),
        )
        if not video_path.exists():
            errors.append(f"{benchmark_id} evaluation video missing: {video_record.get('path')}")
        else:
            if media.sha256_file(video_path) != video_record.get("sha256"):
                errors.append(f"{benchmark_id} evaluation video hash mismatch")
            _validate_actual_video_metadata(
                benchmark_id=benchmark_id,
                video_path=video_path,
                requirement=requirement,
                errors=errors,
            )

    target_video_paths = {
        str(target.get("evaluation_video", {}).get("path") or "")
        for target in target_records.values()
        if isinstance(target, dict)
    }
    if EXCLUDED_CLOSE_VIDEO.as_posix() in target_video_paths:
        errors.append("excluded close-plate clip is referenced as a target video")

    expected_frames = expected_mandatory_frames(benchmark_requirements)
    seen_frame_ids: set[str] = set()
    seen_by_benchmark: dict[str, set[tuple[int, int]]] = {}
    for index, frame in enumerate(frame_records):
        frame_id = str(frame.get("frame_id") or "")
        if not frame_id:
            errors.append(f"frames[{index}] missing frame_id")
            continue
        if frame_id in seen_frame_ids:
            errors.append(f"duplicate frame_id: {frame_id}")
        seen_frame_ids.add(frame_id)
        benchmark_id = str(frame.get("benchmark_id") or "")
        expected_frame = expected_frames.get(frame_id)
        required = benchmark_requirements.get(benchmark_id)
        if required is None:
            errors.append(f"{frame_id} unknown benchmark_id: {benchmark_id}")
            continue
        if expected_frame is None:
            errors.append(f"{frame_id} is not a certified mandatory frame")
            continue
        parent_frame = frame.get("parent_frame")
        dense_frame = frame.get("dense_frame")
        if not isinstance(parent_frame, int) or not isinstance(dense_frame, int):
            errors.append(f"{frame_id} parent/dense frames must be integers")
            continue
        if parent_frame != expected_frame["parent_frame"]:
            errors.append(f"{frame_id} parent_frame mismatch")
        if dense_frame != expected_frame["dense_frame"]:
            errors.append(f"{frame_id} frame mapping mismatch")
        if frame.get("target_id") != required["target_id"]:
            errors.append(f"{frame_id} target_id mismatch")
        if frame.get("object_type") != required["object_type"]:
            errors.append(f"{frame_id} object_type mismatch")
        if frame.get("action") != required["action"]:
            errors.append(f"{frame_id} action mismatch")
        if frame.get("frame_width") != required["evaluation_video"]["width"]:
            errors.append(f"{frame_id} frame_width mismatch")
        if frame.get("frame_height") != required["evaluation_video"]["height"]:
            errors.append(f"{frame_id} frame_height mismatch")
        key = (parent_frame, dense_frame)
        seen_for_benchmark = seen_by_benchmark.setdefault(benchmark_id, set())
        if key in seen_for_benchmark:
            errors.append(f"duplicate frame mapping for {benchmark_id}: {key}")
        seen_for_benchmark.add(key)
        for asset_key in ("crop", "context"):
            asset = frame.get(asset_key)
            if not isinstance(asset, dict):
                errors.append(f"{frame_id} missing {asset_key} asset")
                continue
            path = safe_package_join(package_root, str(asset.get("path") or ""))
            if not path.exists():
                errors.append(f"{frame_id} missing {asset_key} asset file")
                continue
            expected_sha = asset.get("sha256")
            if expected_sha and media.sha256_file(path) != expected_sha:
                errors.append(f"{frame_id} {asset_key} asset hash mismatch")
        crop = frame.get("crop") if isinstance(frame.get("crop"), dict) else {}
        roi = required.get("broad_search_roi_xyxy") or []
        if not isinstance(roi, list) or len(roi) != 4:
            errors.append(f"{frame_id} certified ROI malformed")
        else:
            if crop.get("roi_provenance") != "qa_search_only":
                errors.append(f"{frame_id} crop ROI is not marked qa_search_only")
            if crop.get("origin_xy") != roi[:2]:
                errors.append(f"{frame_id} crop origin mismatch")
            if crop.get("width") != roi[2] - roi[0] or crop.get("height") != roi[3] - roi[1]:
                errors.append(f"{frame_id} crop dimensions mismatch")
        state_record = state_frames.get(frame_id)
        if not isinstance(state_record, dict):
            errors.append(f"{frame_id} missing annotation state")
            continue
        _validate_state_identity(
            state_record=state_record,
            frame=frame,
            requirement=required,
            errors=errors,
            context=frame_id,
        )
        _validate_state_record(
            state_record,
            width=int(required["evaluation_video"]["width"]),
            height=int(required["evaluation_video"]["height"]),
            errors=errors,
            context=frame_id,
        )

    for benchmark_id, required in sorted(benchmark_requirements.items()):
        count = len(seen_by_benchmark.get(benchmark_id, set()))
        if count != required["expected_mandatory_count"]:
            errors.append(
                f"{benchmark_id} expected {required['expected_mandatory_count']} mandatory frames, got {count}"
            )
    if len(seen_frame_ids) != expected_total:
        errors.append(f"expected {expected_total} total mandatory frames, got {len(seen_frame_ids)}")
    missing_frames = set(expected_frames) - seen_frame_ids
    if missing_frames:
        errors.append(f"missing mandatory frame records for {len(missing_frames)} frames")
    missing_state = seen_frame_ids - set(state_frames)
    extra_state = set(state_frames) - seen_frame_ids
    if missing_state:
        errors.append(f"missing annotation state for {len(missing_state)} frames")
    if extra_state:
        errors.append(f"unexpected annotation state for {len(extra_state)} frames")

    computed_progress = compute_progress(review_data, state)
    if progress != computed_progress:
        errors.append("progress.json does not match annotation state")
    completion_errors = completed_export_completion_errors(computed_progress)
    gate_errors = list(errors) + completion_errors
    if computed_progress["final_export_ready"] and gate_errors:
        errors.append("progress reports final_export_ready while completed export validation fails")
    if level == "completed_export":
        errors.extend(completion_errors)

    return {
        "errors": errors,
        "export_gate_errors": gate_errors,
        "level": level,
        "passed": not errors,
        "progress": computed_progress,
        "schema_version": SCHEMA_VERSION,
    }


def phase_a_certified_target_identities() -> dict[str, dict[str, Any]]:
    return {
        target.benchmark_id: {
            "action": target.action,
            "benchmark_id": target.benchmark_id,
            "broad_search_roi_xyxy": list(target.broad_roi_xyxy),
            "dense_frame_interval": [target.dense_frame_start, target.dense_frame_end],
            "evaluation_video": {
                "fps": target.expected_fps,
                "frame_count": target.total_video_frames,
                "height": target.expected_height,
                "path": target.video_path.as_posix(),
                "sha256": target.expected_sha256,
                "width": target.expected_width,
            },
            "expected_mandatory_count": target.mandatory_frame_count,
            "frozen_parent_frame_interval": [target.parent_frame_start, target.parent_frame_end],
            "object_type": target.object_type,
            "parent_to_dense_offset": target.dense_frame_offset,
            "roi_provenance": "qa_search_only",
            "source_interval": target.source_interval,
            "source_video_label": target.source_video_label,
            "target_id": target.target_id,
        }
        for target in DEFAULT_TARGETS
    }


def resolve_evaluation_video_path(package_root: Path, source_path: str) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    candidates = [
        package_root.parent.parent / path,
        package_root.parent.parent.parent / path,
        Path.cwd() / path,
        path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _validate_actual_video_metadata(
    *,
    benchmark_id: str,
    video_path: Path,
    requirement: dict[str, Any],
    errors: list[str],
) -> None:
    capture = cv.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            errors.append(f"{benchmark_id} evaluation video could not be opened")
            return
        video_record = requirement.get("evaluation_video") or {}
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        if frame_count != video_record.get("frame_count"):
            errors.append(f"{benchmark_id} evaluation video frame_count mismatch")
        if width != video_record.get("width") or height != video_record.get("height"):
            errors.append(f"{benchmark_id} evaluation video resolution mismatch")
    finally:
        capture.release()


def _validate_manifest_target(
    *,
    benchmark_id: str,
    manifest_target: dict[str, Any],
    requirement: dict[str, Any],
    errors: list[str],
) -> None:
    field_pairs = (
        ("action", "action"),
        ("broad_search_roi_xyxy", "broad_search_roi_xyxy"),
        ("dense_frame_interval", "dense_frame_interval"),
        ("expected_mandatory_count", "expected_mandatory_count"),
        ("frozen_parent_frame_interval", "frozen_parent_frame_interval"),
        ("object_type", "object_type"),
        ("roi_provenance", "roi_provenance"),
        ("source_interval", "source_interval"),
        ("source_video_label", "source_video_label"),
        ("target_id", "target_id"),
    )
    for manifest_key, requirement_key in field_pairs:
        if manifest_target.get(manifest_key) != requirement.get(requirement_key):
            errors.append(f"{benchmark_id} manifest {manifest_key} mismatch")
    mapping = manifest_target.get("parent_to_dense_mapping") or {}
    if mapping.get("offset") != requirement.get("parent_to_dense_offset"):
        errors.append(f"{benchmark_id} manifest parent_to_dense_mapping offset mismatch")
    manifest_video = manifest_target.get("evaluation_video") or {}
    required_video = requirement.get("evaluation_video") or {}
    for key in ("fps", "frame_count", "height", "path", "sha256", "width"):
        if manifest_video.get(key) != required_video.get(key):
            errors.append(f"{benchmark_id} manifest evaluation_video {key} mismatch")


def expected_mandatory_frames(
    benchmark_requirements: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for benchmark_id, requirement in benchmark_requirements.items():
        parent_start, parent_end = requirement["frozen_parent_frame_interval"]
        offset = requirement["parent_to_dense_offset"]
        for parent_frame in range(parent_start, parent_end + 1):
            frame_id = f"{benchmark_id}_{parent_frame:06d}"
            expected[frame_id] = {
                "benchmark_id": benchmark_id,
                "dense_frame": parent_frame + offset,
                "parent_frame": parent_frame,
            }
    return expected


def _validate_state_identity(
    *,
    state_record: dict[str, Any],
    frame: dict[str, Any],
    requirement: dict[str, Any],
    errors: list[str],
    context: str,
) -> None:
    expected_fields = {
        "benchmark_id": frame.get("benchmark_id"),
        "dense_frame": frame.get("dense_frame"),
        "frame_id": frame.get("frame_id"),
        "parent_frame": frame.get("parent_frame"),
        "target_id": requirement.get("target_id"),
    }
    for key, expected_value in expected_fields.items():
        if state_record.get(key) != expected_value:
            errors.append(f"{context}: state {key} mismatch")


def completed_export_completion_errors(progress: dict[str, Any]) -> list[str]:
    total = progress.get("total") or {}
    errors: list[str] = []
    if total.get("reviewed_box") != total.get("mandatory"):
        errors.append("not all mandatory frames have human-reviewed boxes")
    if total.get("unresolved_temporal_gt_conflict"):
        errors.append("unresolved temporal_gt_conflict exists")
    if total.get("unresolved_ambiguous_secondary"):
        errors.append("unresolved ambiguous secondary instance exists")
    return errors


def summarize_plate_annotation_package(package_root: Path) -> dict[str, Any]:
    package_root = package_root.resolve()
    manifest = read_json(package_root / "package_manifest.json")
    progress = read_json(package_root / "progress.json")
    state = read_json(package_root / "annotations.draft.json")
    draft_count = sum(1 for record in state["frames"].values() if record["draft_available"])
    return {
        "draft_count": draft_count,
        "package_id": manifest["package_id"],
        "package_root": package_root.as_posix(),
        "progress": progress,
        "schema_version": SCHEMA_VERSION,
        "targets": manifest["targets"],
    }


def require_expected_state_revision(state: dict[str, Any], expected_state_revision: Any) -> int:
    current_revision = state.get("state_revision")
    if not isinstance(current_revision, int) or current_revision < 0:
        raise ValueError("state_revision must be a non-negative integer")
    if (
        not isinstance(expected_state_revision, int)
        or isinstance(expected_state_revision, bool)
        or expected_state_revision < 0
    ):
        raise ValueError("expected_state_revision is required")
    if expected_state_revision != current_revision:
        raise StaleStateRevisionError(
            current_revision=current_revision,
            expected_revision=expected_state_revision,
        )
    return current_revision


def next_state_revision(state: dict[str, Any]) -> int:
    revision = state.get("state_revision")
    if not isinstance(revision, int) or revision < 0:
        raise ValueError("state_revision must be a non-negative integer")
    state["state_revision"] = revision + 1
    return state["state_revision"]


def save_frame_annotation(
    package_root: Path,
    frame_id: str,
    payload: dict[str, Any],
    *,
    expected_state_revision: int | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    review_data = read_json(package_root / "review_data.json")
    state = read_json(package_root / "annotations.draft.json")
    previous_revision = require_expected_state_revision(state, expected_state_revision)
    frame_index = {frame["frame_id"]: frame for frame in review_data["frames"]}
    if frame_id not in frame_index:
        raise ValueError(f"unknown frame_id: {frame_id}")
    frame = frame_index[frame_id]
    width = int(frame["frame_width"])
    height = int(frame["frame_height"])
    existing = dict(state["frames"][frame_id])
    status = str(payload.get("annotation_status", existing["annotation_status"]))
    if payload.get("temporal_gt_conflict") is True:
        status = STATUS_TEMPORAL_GT_CONFLICT
    if status not in ALLOWED_MAIN_STATUSES:
        raise ValueError(f"unsupported annotation_status: {status}")
    blocked_target_flags = {"detector_ignore", "invisible"}
    if status in blocked_target_flags or payload.get("visibility") in blocked_target_flags:
        raise ValueError("target frames cannot be marked invisible or detector_ignore in this phase")

    now_utc = now_utc or media.utc_timestamp()
    review_note = str(payload.get("review_note", "")).strip()
    conflicts = normalize_conflicts(existing.get("conflicts", []))
    unresolved_temporal = [
        conflict
        for conflict in conflicts
        if conflict["type"] == "temporal_gt_conflict" and not conflict["resolved"]
    ]
    temporal_transition = "none"
    if status == STATUS_REVIEWED_BOX:
        human_bbox = validate_bbox_xyxy(payload.get("human_bbox"), width=width, height=height)
        human_reviewed = True
        if unresolved_temporal:
            temporal_transition = "resolved"
            for conflict in unresolved_temporal:
                conflict["resolved"] = True
                conflict["resolved_at_utc"] = now_utc
                conflict["resolution"] = STATUS_REVIEWED_BOX
                conflict["resolved_human_bbox"] = human_bbox
    elif status == STATUS_NEEDS_RECHECK:
        if unresolved_temporal:
            raise ValueError(
                "temporal_gt_conflict can only be resolved by reviewed_box or kept as temporal_gt_conflict"
            )
        if not review_note:
            raise ValueError("needs_recheck requires review_note")
        human_bbox = None
        human_reviewed = True
    elif status == STATUS_TEMPORAL_GT_CONFLICT:
        if not review_note:
            raise ValueError("temporal_gt_conflict requires review_note")
        human_bbox = None
        if payload.get("human_bbox") is not None:
            human_bbox = validate_bbox_xyxy(payload.get("human_bbox"), width=width, height=height)
        human_reviewed = False
        if unresolved_temporal:
            unresolved_temporal[-1]["note"] = review_note
            unresolved_temporal[-1]["updated_at_utc"] = now_utc
            temporal_transition = "updated"
        else:
            conflicts.append(
                {
                    "created_at_utc": now_utc,
                    "note": review_note,
                    "resolved": False,
                    "type": "temporal_gt_conflict",
                }
            )
            temporal_transition = "created"
    else:
        if unresolved_temporal:
            raise ValueError(
                "temporal_gt_conflict can only be resolved by reviewed_box or kept as temporal_gt_conflict"
            )
        human_bbox = None
        human_reviewed = False

    secondary_instances = normalize_secondary_instances(
        payload.get("secondary_instances", existing.get("secondary_instances", [])),
        frame_id=frame_id,
        width=width,
        height=height,
    )

    updated = existing | {
        "annotation_status": status,
        "conflicts": conflicts,
        "draft_shown": bool(payload.get("draft_shown", existing.get("draft_shown", False))),
        "human_bbox": human_bbox,
        "human_reviewed": human_reviewed,
        "review_note": review_note,
        "reviewed_at": now_utc if human_reviewed else None,
        "secondary_instances": secondary_instances,
    }
    state_errors: list[str] = []
    _validate_state_record(updated, width=width, height=height, errors=state_errors, context=frame_id)
    if state_errors:
        raise ValueError("; ".join(state_errors))
    state["frames"][frame_id] = updated
    new_revision = next_state_revision(state)
    progress = compute_progress(review_data, state)
    atomic_write_json(package_root / "annotations.draft.json", state)
    atomic_write_json(package_root / "progress.json", progress)
    append_event(
        package_root / "change_log.jsonl",
        {
            "action": frame.get("action"),
            "annotation_status": status,
            "benchmark_id": frame.get("benchmark_id"),
            "dense_frame": frame.get("dense_frame"),
            "event": "save_annotation",
            "frame_id": frame_id,
            "human_bbox": human_bbox,
            "new_state_revision": new_revision,
            "new_status": status,
            "parent_frame": frame.get("parent_frame"),
            "previous_state_revision": previous_revision,
            "previous_status": existing.get("annotation_status"),
            "recorded_at_utc": now_utc,
            "review_note": review_note,
            "schema_version": "plate_annotation_change.v1",
            "semantic_state_hash": stable_json_sha256(updated),
            "target_id": frame.get("target_id"),
            "temporal_gt_conflict_transition": temporal_transition,
        },
    )
    return {"frame": updated, "progress": progress, "state_revision": new_revision}


def mark_draft_shown(
    package_root: Path,
    frame_id: str,
    *,
    expected_state_revision: int | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    review_data = read_json(package_root / "review_data.json")
    state = read_json(package_root / "annotations.draft.json")
    previous_revision = require_expected_state_revision(state, expected_state_revision)
    if frame_id not in state["frames"]:
        raise ValueError(f"unknown frame_id: {frame_id}")
    record = dict(state["frames"][frame_id])
    if record["draft_available"]:
        record["draft_shown"] = True
        state["frames"][frame_id] = record
        new_revision = next_state_revision(state)
        progress = compute_progress(review_data, state)
        atomic_write_json(package_root / "annotations.draft.json", state)
        atomic_write_json(package_root / "progress.json", progress)
        append_event(
            package_root / "change_log.jsonl",
            {
                "benchmark_id": record.get("benchmark_id"),
                "dense_frame": record.get("dense_frame"),
                "event": "draft_shown",
                "frame_id": frame_id,
                "new_state_revision": new_revision,
                "parent_frame": record.get("parent_frame"),
                "previous_state_revision": previous_revision,
                "recorded_at_utc": now_utc or media.utc_timestamp(),
                "schema_version": "plate_annotation_change.v1",
                "semantic_state_hash": stable_json_sha256(record),
            },
        )
    else:
        new_revision = previous_revision
    return {
        "frame": record,
        "progress": compute_progress(review_data, state),
        "state_revision": new_revision,
    }


def export_plate_annotation_review(
    package_root: Path,
    *,
    final: bool = False,
    output_path: Path | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    if final:
        validation = validate_plate_annotation_package(package_root, level="completed_export")
        if not validation["passed"]:
            raise ValueError("; ".join(validation["errors"]))
    manifest = read_json(package_root / "package_manifest.json")
    review_data = read_json(package_root / "review_data.json")
    state = read_json(package_root / "annotations.draft.json")
    progress = compute_progress(review_data, state)
    if final:
        export_kind = "final_review"
        default_name = "plate_annotation_review.export.json"
    else:
        export_kind = "progress_review"
        default_name = "plate_annotation_review.progress.json"

    records = [
        {
            "annotation_provenance": {
                "draft_available": state["frames"][frame["frame_id"]]["draft_available"],
                "draft_model": state["frames"][frame["frame_id"]]["draft_model"],
                "draft_shown": state["frames"][frame["frame_id"]]["draft_shown"],
                "draft_source": state["frames"][frame["frame_id"]]["draft_source"],
                "human_reviewed": state["frames"][frame["frame_id"]]["human_reviewed"],
                "reviewed_at": state["frames"][frame["frame_id"]]["reviewed_at"],
            },
            "benchmark_id": frame["benchmark_id"],
            "conflicts": state["frames"][frame["frame_id"]]["conflicts"],
            "dense_frame": frame["dense_frame"],
            "frame_id": frame["frame_id"],
            "human_bbox": state["frames"][frame["frame_id"]]["human_bbox"],
            "parent_frame": frame["parent_frame"],
            "review_note": state["frames"][frame["frame_id"]]["review_note"],
            "secondary_instances": state["frames"][frame["frame_id"]]["secondary_instances"],
            "target_id": frame["target_id"],
        }
        for frame in review_data["frames"]
    ]
    payload = {
        "evaluation_sources": {
            benchmark_id: {
                "evaluation_video": target["evaluation_video"],
                "frozen_parent_frame_interval": target["frozen_parent_frame_interval"],
                "source_interval": target["source_interval"],
                "source_video_label": target["source_video_label"],
            }
            for benchmark_id, target in sorted(manifest["targets"].items())
        },
        "export_kind": export_kind,
        "frame_mappings": [
            {
                "benchmark_id": frame["benchmark_id"],
                "dense_frame": frame["dense_frame"],
                "frame_id": frame["frame_id"],
                "parent_frame": frame["parent_frame"],
                "target_id": frame["target_id"],
            }
            for frame in review_data["frames"]
        ],
        "mandatory_frames": records,
        "package_id": manifest["package_id"],
        "package_manifest_sha256": media.sha256_file(package_root / "package_manifest.json"),
        "reviewer_completion_state": progress["reviewer_completion_state"],
        "schema_version": EXPORT_SCHEMA_VERSION,
    }
    payload["semantic_content_hash"] = semantic_content_hash(payload)
    payload["exported_at_utc"] = now_utc or media.utc_timestamp()
    output_path = output_path or (package_root / default_name)
    atomic_write_json(output_path, payload)
    return payload


def compute_progress(review_data: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    frame_records = review_data.get("frames") or []
    state_frames = state.get("frames") or {}
    by_benchmark: dict[str, dict[str, int]] = {}
    total = {
        "additional_localizable_plate": 0,
        "ambiguous_review_required": 0,
        "draft_shown": 0,
        "human_reviewed": 0,
        "ignore_small": 0,
        "mandatory": len(frame_records),
        "needs_recheck": 0,
        "reviewed_box": 0,
        "temporal_gt_conflict": 0,
        "unreviewed": 0,
        "unresolved_ambiguous_secondary": 0,
        "unresolved_temporal_gt_conflict": 0,
    }
    for frame in frame_records:
        benchmark = frame["benchmark_id"]
        row = by_benchmark.setdefault(
            benchmark,
            {
                "additional_localizable_plate": 0,
                "ambiguous_review_required": 0,
                "draft_shown": 0,
                "human_reviewed": 0,
                "ignore_small": 0,
                "mandatory": 0,
                "needs_recheck": 0,
                "reviewed_box": 0,
                "temporal_gt_conflict": 0,
                "unreviewed": 0,
                "unresolved_ambiguous_secondary": 0,
                "unresolved_temporal_gt_conflict": 0,
            },
        )
        row["mandatory"] += 1
        record = state_frames.get(frame["frame_id"], {})
        status = record.get("annotation_status", STATUS_UNREVIEWED)
        if status == STATUS_REVIEWED_BOX:
            row["reviewed_box"] += 1
            total["reviewed_box"] += 1
        elif status == STATUS_NEEDS_RECHECK:
            row["needs_recheck"] += 1
            total["needs_recheck"] += 1
        elif status == STATUS_TEMPORAL_GT_CONFLICT:
            row["temporal_gt_conflict"] += 1
            row["unresolved_temporal_gt_conflict"] += 1
            total["temporal_gt_conflict"] += 1
            total["unresolved_temporal_gt_conflict"] += 1
        else:
            row["unreviewed"] += 1
            total["unreviewed"] += 1
        if record.get("human_reviewed"):
            row["human_reviewed"] += 1
            total["human_reviewed"] += 1
        if record.get("draft_shown"):
            row["draft_shown"] += 1
            total["draft_shown"] += 1
        if status != STATUS_TEMPORAL_GT_CONFLICT:
            for conflict in record.get("conflicts") or []:
                if conflict.get("type") != "temporal_gt_conflict" or conflict.get("resolved"):
                    continue
                row["temporal_gt_conflict"] += 1
                total["temporal_gt_conflict"] += 1
                row["unresolved_temporal_gt_conflict"] += 1
                total["unresolved_temporal_gt_conflict"] += 1
        for secondary in record.get("secondary_instances") or []:
            kind = secondary.get("kind")
            if kind in {SECONDARY_ADDITIONAL, SECONDARY_IGNORE_SMALL, SECONDARY_AMBIGUOUS}:
                row[kind] += 1
                total[kind] += 1
            if kind == SECONDARY_AMBIGUOUS and not secondary.get("resolved"):
                row["unresolved_ambiguous_secondary"] += 1
                total["unresolved_ambiguous_secondary"] += 1
    ready = (
        total["reviewed_box"] == total["mandatory"]
        and total["mandatory"] > 0
        and total["unresolved_temporal_gt_conflict"] == 0
        and total["unresolved_ambiguous_secondary"] == 0
    )
    return {
        "by_benchmark": by_benchmark,
        "final_export_ready": ready,
        "reviewer_completion_state": {
            "all_mandatory_boxes_reviewed": total["reviewed_box"] == total["mandatory"],
            "final_export_ready": ready,
            "state": "ready_for_final_export" if ready else "in_progress",
            "unresolved_ambiguous_secondary": total["unresolved_ambiguous_secondary"],
            "unresolved_temporal_gt_conflict": total["unresolved_temporal_gt_conflict"],
        },
        "schema_version": "plate_annotation_progress.v1",
        "total": total,
    }


def final_export_gate_errors(
    package_root: Path,
    manifest: dict[str, Any],
    review_data: dict[str, Any],
    state: dict[str, Any],
) -> list[str]:
    _ = (manifest, review_data, state)
    return validate_plate_annotation_package(package_root, level="completed_export")["errors"]


def validate_bbox_xyxy(value: Any, *, width: int, height: int) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    coords = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(item):
            raise ValueError("bbox coordinates must be finite numbers")
        if int(item) != item:
            raise ValueError("bbox coordinates must be integers")
        coords.append(int(item))
    x1, y1, x2, y2 = coords
    if x1 < 0 or y1 < 0:
        raise ValueError("bbox cannot be negative")
    if x2 > width or y2 > height:
        raise ValueError("bbox exceeds frame bounds")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
    return coords


def crop_box_to_original(
    crop_bbox_xyxy: list[int] | tuple[int, int, int, int],
    crop_origin_xy: list[int] | tuple[int, int],
    *,
    crop_width: int | None = None,
    crop_height: int | None = None,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    ox, oy = int(crop_origin_xy[0]), int(crop_origin_xy[1])
    crop_bbox = validate_bbox_xyxy(
        crop_bbox_xyxy,
        width=crop_width or frame_width - ox,
        height=crop_height or frame_height - oy,
    )
    return validate_bbox_xyxy(
        [crop_bbox[0] + ox, crop_bbox[1] + oy, crop_bbox[2] + ox, crop_bbox[3] + oy],
        width=frame_width,
        height=frame_height,
    )


def original_box_to_crop(
    original_bbox_xyxy: list[int] | tuple[int, int, int, int],
    crop_origin_xy: list[int] | tuple[int, int],
    *,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    original_bbox = validate_bbox_xyxy(
        original_bbox_xyxy,
        width=frame_width,
        height=frame_height,
    )
    ox, oy = int(crop_origin_xy[0]), int(crop_origin_xy[1])
    return validate_bbox_xyxy(
        [original_bbox[0] - ox, original_bbox[1] - oy, original_bbox[2] - ox, original_bbox[3] - oy],
        width=crop_width,
        height=crop_height,
    )


def normalize_secondary_instances(
    value: Any,
    *,
    frame_id: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("secondary_instances must be a list")
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("secondary instance must be an object")
        kind = str(item.get("kind") or "")
        if kind not in ALLOWED_SECONDARY_KINDS:
            raise ValueError(f"unsupported secondary instance kind: {kind}")
        note = str(item.get("note") or item.get("reason") or "").strip()
        bbox_value = item.get("bbox_xyxy")
        bbox = None
        if bbox_value is not None:
            bbox = validate_bbox_xyxy(bbox_value, width=width, height=height)
        if kind == SECONDARY_ADDITIONAL and bbox is None:
            raise ValueError("additional_localizable_plate requires bbox_xyxy")
        if kind in {SECONDARY_IGNORE_SMALL, SECONDARY_AMBIGUOUS} and not note:
            raise ValueError(f"{kind} requires a note or reason")
        resolved = bool(item.get("resolved", False))
        normalized_item = {
            "bbox_xyxy": bbox,
            "kind": kind,
            "note": note,
            "provisional_instance_id": provisional_secondary_id(
                frame_id=frame_id,
                kind=kind,
                bbox_xyxy=bbox,
                note=note,
                index=index,
            ),
            "resolved": resolved,
        }
        normalized.append(normalized_item)
    return normalized


def provisional_secondary_id(
    *,
    frame_id: str,
    kind: str,
    bbox_xyxy: list[int] | None,
    note: str,
    index: int,
) -> str:
    digest = stable_json_sha256(
        {
            "bbox_xyxy": bbox_xyxy,
            "frame_id": frame_id,
            "index": index,
            "kind": kind,
            "note": note,
        }
    )
    return f"secondary_{digest[:16]}"


def normalize_conflicts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("conflicts must be a list")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("conflict must be an object")
        conflict_type = str(item.get("type") or "")
        if conflict_type != "temporal_gt_conflict":
            raise ValueError(f"unsupported conflict type: {conflict_type}")
        normalized_item = {
            "note": str(item.get("note") or "").strip(),
            "resolved": bool(item.get("resolved", False)),
            "type": conflict_type,
        }
        for key in (
            "created_at_utc",
            "resolution",
            "resolved_at_utc",
            "resolved_human_bbox",
            "updated_at_utc",
        ):
            if key in item:
                normalized_item[key] = item[key]
        normalized.append(normalized_item)
    return normalized


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def semantic_content_hash(payload: dict[str, Any]) -> str:
    semantic_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"exported_at_utc", "semantic_content_hash"}
    }
    for record in semantic_payload.get("mandatory_frames", []):
        provenance = record.get("annotation_provenance") or {}
        provenance.pop("reviewed_at", None)
    return stable_json_sha256(semantic_payload)


def unique_run_id(output_root: Path) -> str:
    base = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    candidate = base
    suffix = 1
    while (output_root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def assert_annotation_root_is_ignored(output_root: Path, repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    try:
        rel_path = output_root.relative_to(repo_root)
    except ValueError:
        return
    result = subprocess.run(
        ["git", "check-ignore", "-v", f"{rel_path.as_posix().rstrip('/')}/"],
        capture_output=True,
        cwd=repo_root,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"annotation output root is not ignored by git: {rel_path.as_posix()}")


def safe_package_join(package_root: Path, relative_path: str) -> Path:
    if relative_path.startswith("/"):
        raise ValueError("absolute paths are not allowed")
    candidate = (package_root / relative_path).resolve()
    root = package_root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("path traversal is not allowed")
    return candidate


def _build_into_temp_root(
    *,
    package_root: Path,
    final_package_root: Path,
    targets: tuple[PlateTargetConfig, ...],
    certified_config_version: str,
    include_drafts: bool,
    draft_path: Path,
    draft_mapping_audit_path: Path,
    repo_root: Path,
) -> None:
    for child in ("assets", "frames/day_plate", "frames/night_selective"):
        (package_root / child).mkdir(parents=True, exist_ok=True)
    git_commit = git_commit_for(repo_root)
    media_records = {target.benchmark_id: probe_and_validate_target(target, repo_root) for target in targets}
    draft_import = import_drafts(
        targets=targets,
        media_records=media_records,
        draft_path=draft_path,
        draft_mapping_audit_path=draft_mapping_audit_path,
        include_drafts=include_drafts,
        repo_root=repo_root,
    )
    frame_records = []
    for target in targets:
        frame_records.extend(
            extract_target_frames(
                package_root=package_root,
                target=target,
                media_record=media_records[target.benchmark_id],
            )
        )
    package_id = final_package_root.name
    manifest = build_manifest(
        package_id=package_id,
        targets=targets,
        media_records=media_records,
        draft_import=draft_import,
        git_commit=git_commit,
        repo_root=repo_root,
        certified_config_version=certified_config_version,
    )
    review_data = {
        "frame_count": len(frame_records),
        "frames": frame_records,
        "package_id": package_id,
        "schema_version": "plate_annotation_review_data.v1",
    }
    state = initial_state_for_frames(
        package_id=package_id,
        frame_records=frame_records,
        draft_import=draft_import,
    )
    progress = compute_progress(review_data, state)
    write_json(package_root / "package_manifest.json", manifest)
    write_json(package_root / "review_data.json", review_data)
    write_json(
        package_root / "package_identity.json",
        build_package_identity(
            package_root=package_root,
            package_id=package_id,
            targets=targets,
            media_records=media_records,
            certified_config_version=certified_config_version,
            repo_root=repo_root,
        ),
    )
    write_json(package_root / "annotations.draft.json", state)
    write_json(package_root / "progress.json", progress)
    write_json(package_root / "draft_import_audit.json", draft_import)
    write_text(package_root / "change_log.jsonl", "")
    write_text(package_root / "instructions.md", instructions_markdown(targets))
    write_text(package_root / "report.md", report_markdown(manifest, progress, draft_import))
    write_text(package_root / "index.html", index_html())
    write_text(package_root / "assets/style.css", style_css())
    write_text(package_root / "assets/app.js", app_js())


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def git_commit_for(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        cwd=repo_root,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def probe_and_validate_target(target: PlateTargetConfig, repo_root: Path) -> dict[str, Any]:
    if not target.video_path.exists():
        raise FileNotFoundError(target.video_path)
    info = media.probe_media(target.video_path, cwd=repo_root).to_dict()
    stream = info["video_stream"]
    frame_count = int(stream.get("frame_count") or 0)
    if frame_count == 0:
        frame_count = opencv_frame_count(target.video_path)
    if target.expected_sha256 and info["sha256"] != target.expected_sha256:
        raise ValueError(f"{target.benchmark_id} video hash mismatch")
    if target.total_video_frames is not None and frame_count != target.total_video_frames:
        raise ValueError(
            f"{target.benchmark_id} expected {target.total_video_frames} video frames, got {frame_count}"
        )
    if stream["width"] != target.expected_width or stream["height"] != target.expected_height:
        raise ValueError(
            f"{target.benchmark_id} expected "
            f"{target.expected_width}x{target.expected_height}, got {stream['width']}x{stream['height']}"
        )
    if frame_count <= target.parent_frame_end:
        raise ValueError(f"{target.benchmark_id} video is shorter than the frozen parent interval")
    return {
        "audio_present": info["audio_stream"] is not None,
        "duration_seconds": info["duration_seconds"],
        "fps": stream.get("average_frame_rate") or stream.get("nominal_frame_rate"),
        "frame_count": frame_count,
        "height": stream["height"],
        "path": media.portable_source_path(target.video_path, repo_root),
        "sha256": info["sha256"],
        "size_bytes": info["file_size_bytes"],
        "width": stream["width"],
    }


def opencv_frame_count(video_path: Path) -> int:
    capture = cv.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        return int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def extract_target_frames(
    *,
    package_root: Path,
    target: PlateTargetConfig,
    media_record: dict[str, Any],
) -> list[dict[str, Any]]:
    capture = cv.VideoCapture(str(target.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {target.video_path}")
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, target.parent_frame_start)
        frame_records = []
        for parent_frame in range(target.parent_frame_start, target.parent_frame_end + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not read frame {parent_frame} from {target.video_path}")
            frame_records.append(
                write_frame_assets(
                    package_root=package_root,
                    target=target,
                    media_record=media_record,
                    parent_frame=parent_frame,
                    frame=frame,
                )
            )
        return frame_records
    finally:
        capture.release()


def write_frame_assets(
    *,
    package_root: Path,
    target: PlateTargetConfig,
    media_record: dict[str, Any],
    parent_frame: int,
    frame: Any,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    if width != target.expected_width or height != target.expected_height:
        raise ValueError(f"{target.benchmark_id} frame geometry mismatch")
    x1, y1, x2, y2 = target.broad_roi_xyxy
    crop = frame[y1:y2, x1:x2]
    benchmark_dir = package_root / "frames" / target.benchmark_id
    crop_rel = (
        Path("frames")
        / target.benchmark_id
        / FRAME_ASSET_NAME_TEMPLATE.format(
            benchmark=target.benchmark_id,
            extension="png",
            kind="crop",
            parent_frame=parent_frame,
        )
    )
    context_rel = (
        Path("frames")
        / target.benchmark_id
        / FRAME_ASSET_NAME_TEMPLATE.format(
            benchmark=target.benchmark_id,
            extension="jpg",
            kind="context",
            parent_frame=parent_frame,
        )
    )
    crop_path = package_root / crop_rel
    context_path = package_root / context_rel
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(crop_path), crop, [cv.IMWRITE_PNG_COMPRESSION, 3]):
        raise ValueError(f"could not write crop frame: {crop_path}")
    context = context_image(frame)
    if not cv.imwrite(str(context_path), context, [cv.IMWRITE_JPEG_QUALITY, 88]):
        raise ValueError(f"could not write context frame: {context_path}")
    dense_frame = parent_frame + target.dense_frame_offset
    fps_value = rational_to_float(str(media_record["fps"]))
    return {
        "action": target.action,
        "benchmark_id": target.benchmark_id,
        "context": {
            "height": int(context.shape[0]),
            "path": context_rel.as_posix(),
            "sha256": media.sha256_file(context_path),
            "width": int(context.shape[1]),
        },
        "crop": {
            "height": y2 - y1,
            "origin_xy": [x1, y1],
            "path": crop_rel.as_posix(),
            "roi_provenance": "qa_search_only",
            "sha256": media.sha256_file(crop_path),
            "transform_to_original": {
                "x": f"original_x = crop_x + {x1}",
                "y": f"original_y = crop_y + {y1}",
            },
            "width": x2 - x1,
        },
        "dense_frame": dense_frame,
        "frame_height": height,
        "frame_id": f"{target.benchmark_id}_{parent_frame:06d}",
        "frame_width": width,
        "object_type": target.object_type,
        "parent_frame": parent_frame,
        "source_frame_sha256_bgr24": hashlib.sha256(frame.tobytes()).hexdigest(),
        "target_id": target.target_id,
        "timestamp_seconds": round(parent_frame / fps_value, 6) if fps_value else None,
    }


def context_image(frame: Any, *, max_width: int = 960) -> Any:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv.resize(frame, (max_width, int(round(height * scale))), interpolation=cv.INTER_AREA)


def rational_to_float(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    return float(value)


def build_manifest(
    *,
    package_id: str,
    targets: tuple[PlateTargetConfig, ...],
    media_records: dict[str, dict[str, Any]],
    draft_import: dict[str, Any],
    git_commit: str | None,
    repo_root: Path,
    certified_config_version: str,
) -> dict[str, Any]:
    return {
        "certified_config_version": certified_config_version,
        "created_at_utc": media.utc_timestamp(),
        "excluded_inputs": [
            {
                "path": media.portable_source_path(EXCLUDED_CLOSE_VIDEO, repo_root),
                "reason": "not_part_of_phase_a_review_targets",
            }
        ],
        "expected_mandatory_total": sum(target.mandatory_frame_count for target in targets),
        "generator": {
            "git_commit": git_commit,
            "version": GENERATOR_VERSION,
        },
        "package_id": package_id,
        "package_identity_path": "package_identity.json",
        "package_role": "local_plate_annotation_workspace",
        "review_state_path": "annotations.draft.json",
        "schema_version": SCHEMA_VERSION,
        "targets": {
            target.benchmark_id: {
                "action": target.action,
                "broad_search_roi_xyxy": list(target.broad_roi_xyxy),
                "dense_frame_interval": [target.dense_frame_start, target.dense_frame_end],
                "evaluation_video": media_records[target.benchmark_id],
                "expected_mandatory_count": target.mandatory_frame_count,
                "frozen_parent_frame_interval": [
                    target.parent_frame_start,
                    target.parent_frame_end,
                ],
                "object_type": target.object_type,
                "parent_to_dense_mapping": {
                    "formula": f"dense_frame = parent_frame {target.dense_frame_offset:+d}",
                    "offset": target.dense_frame_offset,
                },
                "roi_provenance": "qa_search_only",
                "source_interval": target.source_interval,
                "source_video_label": target.source_video_label,
                "target_id": target.target_id,
            }
            for target in targets
        },
        "drafts": {
            "default_visible": False,
            "import_audit_path": "draft_import_audit.json",
            "imported": draft_import["imported"],
            "imported_target_frame_count": draft_import["imported_target_frame_count"],
        },
    }


def build_package_identity(
    *,
    package_root: Path,
    package_id: str,
    targets: tuple[PlateTargetConfig, ...],
    media_records: dict[str, dict[str, Any]],
    certified_config_version: str,
    repo_root: Path,
) -> dict[str, Any]:
    certified_targets = {
        target.benchmark_id: certified_target_identity(
            target=target,
            media_record=media_records[target.benchmark_id],
            repo_root=repo_root,
        )
        for target in targets
    }
    return {
        "certified_config_version": certified_config_version,
        "certified_target_config_sha256": stable_json_sha256(certified_targets),
        "certified_targets": certified_targets,
        "generator_version": GENERATOR_VERSION,
        "package_id": package_id,
        "package_manifest_sha256": media.sha256_file(package_root / "package_manifest.json"),
        "review_data_sha256": media.sha256_file(package_root / "review_data.json"),
        "schema_version": PACKAGE_IDENTITY_SCHEMA_VERSION,
    }


def certified_target_identity(
    *,
    target: PlateTargetConfig,
    media_record: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "action": target.action,
        "benchmark_id": target.benchmark_id,
        "broad_search_roi_xyxy": list(target.broad_roi_xyxy),
        "dense_frame_interval": [target.dense_frame_start, target.dense_frame_end],
        "evaluation_video": {
            "fps": media_record.get("fps"),
            "frame_count": media_record.get("frame_count"),
            "height": media_record.get("height"),
            "path": media.portable_source_path(target.video_path, repo_root),
            "sha256": media_record.get("sha256"),
            "width": media_record.get("width"),
        },
        "expected_mandatory_count": target.mandatory_frame_count,
        "frozen_parent_frame_interval": [target.parent_frame_start, target.parent_frame_end],
        "object_type": target.object_type,
        "parent_to_dense_offset": target.dense_frame_offset,
        "roi_provenance": "qa_search_only",
        "source_interval": target.source_interval,
        "source_video_label": target.source_video_label,
        "target_id": target.target_id,
    }


def import_drafts(
    *,
    targets: tuple[PlateTargetConfig, ...],
    media_records: dict[str, dict[str, Any]],
    draft_path: Path,
    draft_mapping_audit_path: Path,
    include_drafts: bool,
    repo_root: Path,
) -> dict[str, Any]:
    empty = {
        "excluded_by_reason": {},
        "frame_drafts": {},
        "imported": False,
        "imported_annotation_count": 0,
        "imported_by_benchmark": {},
        "imported_target_frame_count": 0,
        "schema_version": "plate_annotation_draft_import_audit.v1",
        "source": None,
    }
    if not include_drafts:
        return empty | {"disabled_reason": "draft_import_disabled"}
    if not draft_path.exists():
        return empty | {"disabled_reason": "draft_file_missing"}
    payload = read_json(draft_path)
    mapping_payload = read_json(draft_mapping_audit_path) if draft_mapping_audit_path.exists() else {}
    targets_by_hash: dict[str, list[PlateTargetConfig]] = {}
    for target in targets:
        targets_by_hash.setdefault(media_records[target.benchmark_id]["sha256"], []).append(target)
    target_by_benchmark = {target.benchmark_id: target for target in targets}
    frame_drafts: dict[str, dict[str, Any]] = {}
    excluded = Counter()
    imported_by_benchmark: dict[str, int] = Counter()
    annotations = payload.get("annotations") or []
    for annotation in annotations:
        source_sha = str(annotation.get("source_sha256") or "")
        target_candidates = targets_by_hash.get(source_sha) or []
        if not target_candidates:
            excluded["source_hash_not_in_phase_a_targets"] += 1
            continue
        frame_index = annotation.get("frame_index")
        if not isinstance(frame_index, int):
            excluded["invalid_frame_index"] += 1
            continue
        interval_matches = [
            target
            for target in target_candidates
            if target.parent_frame_start <= frame_index <= target.parent_frame_end
        ]
        if not interval_matches:
            excluded["outside_frozen_parent_interval"] += 1
            continue
        if len(interval_matches) != 1:
            excluded["ambiguous_target_interval"] += 1
            continue
        target = interval_matches[0]
        try:
            bbox = validate_bbox_xyxy(
                annotation.get("bbox_xyxy"),
                width=target.expected_width,
                height=target.expected_height,
            )
        except ValueError:
            excluded["invalid_bbox"] += 1
            continue
        frame_id = f"{target.benchmark_id}_{frame_index:06d}"
        draft = {
            "bbox_xyxy": bbox,
            "draft_kind": "proposal_seeded_human_review_draft",
            "object_id": annotation.get("object_id"),
            "provenance": annotation.get("reviewer_provenance"),
            "source_clip_id": annotation.get("clip_id"),
            "source_path": media.portable_source_path(draft_path, repo_root),
            "source_schema_version": annotation.get("schema_version"),
            "visibility": annotation.get("visibility"),
        }
        frame_entry = frame_drafts.setdefault(
            frame_id,
            {
                "benchmark_id": target.benchmark_id,
                "candidates": [],
                "main_candidate": None,
                "parent_frame": frame_index,
                "target_id": target.target_id,
            },
        )
        frame_entry["candidates"].append(draft)
        imported_by_benchmark[target.benchmark_id] += 1

    for frame_id, frame_entry in frame_drafts.items():
        target = target_by_benchmark[frame_entry["benchmark_id"]]
        frame_entry["main_candidate"] = choose_main_draft_candidate(
            frame_entry["candidates"],
            roi_xyxy=target.broad_roi_xyxy,
        )
    imported_target_frame_count = sum(
        1 for entry in frame_drafts.values() if entry["main_candidate"] is not None
    )
    return {
        "clip_mapping_audit": mapping_summary(mapping_payload),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "frame_drafts": frame_drafts,
        "imported": True,
        "imported_annotation_count": sum(imported_by_benchmark.values()),
        "imported_by_benchmark": dict(sorted(imported_by_benchmark.items())),
        "imported_target_frame_count": imported_target_frame_count,
        "schema_version": "plate_annotation_draft_import_audit.v1",
        "source": {
            "path": media.portable_source_path(draft_path, repo_root),
            "schema_version": payload.get("schema_version"),
            "sha256": media.sha256_file(draft_path),
        },
        "source_finalists": payload.get("finalists") or [],
        "source_frame_count_by_clip": payload.get("sample_frame_count_by_clip") or {},
    }


def choose_main_draft_candidate(
    candidates: list[dict[str, Any]],
    *,
    roi_xyxy: tuple[int, int, int, int],
) -> dict[str, Any] | None:
    inside = [
        candidate
        for candidate in candidates
        if point_inside_box(center_xy(candidate["bbox_xyxy"]), roi_xyxy)
    ]
    if not inside:
        inside = list(candidates)
    if not inside:
        return None
    roi_center = center_xy(roi_xyxy)
    return sorted(
        inside,
        key=lambda candidate: (
            squared_distance(center_xy(candidate["bbox_xyxy"]), roi_center),
            -box_area(candidate["bbox_xyxy"]),
            str(candidate.get("object_id") or ""),
        ),
    )[0]


def center_xy(box: tuple[int, int, int, int] | list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def squared_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def box_area(box: list[int] | tuple[int, int, int, int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def mapping_summary(payload: dict[str, Any]) -> dict[str, Any]:
    clips = payload.get("clips") or []
    return {
        "clips": [
            {
                "clip_id": clip.get("clip_id"),
                "source_sha256": clip.get("source_sha256"),
                "video_path": clip.get("video_path"),
            }
            for clip in clips
        ],
        "schema_version": payload.get("schema_version"),
    }


def initial_state_for_frames(
    *,
    package_id: str,
    frame_records: list[dict[str, Any]],
    draft_import: dict[str, Any],
) -> dict[str, Any]:
    frame_drafts = draft_import.get("frame_drafts") or {}
    state_frames = {}
    for frame in frame_records:
        draft_entry = frame_drafts.get(frame["frame_id"], {})
        main_candidate = draft_entry.get("main_candidate")
        state_frames[frame["frame_id"]] = {
            "annotation_status": STATUS_UNREVIEWED,
            "benchmark_id": frame["benchmark_id"],
            "conflicts": [],
            "dense_frame": frame["dense_frame"],
            "draft_available": main_candidate is not None,
            "draft_bbox": main_candidate.get("bbox_xyxy") if main_candidate else None,
            "draft_candidates": draft_entry.get("candidates", []),
            "draft_model": ",".join(draft_import.get("source_finalists") or [])
            if main_candidate
            else None,
            "draft_shown": False,
            "draft_source": draft_import.get("source", {}).get("path") if main_candidate else None,
            "frame_id": frame["frame_id"],
            "human_bbox": None,
            "human_reviewed": False,
            "parent_frame": frame["parent_frame"],
            "review_note": "",
            "reviewed_at": None,
            "secondary_instances": [],
            "target_id": frame["target_id"],
        }
    return {
        "created_at_utc": media.utc_timestamp(),
        "frames": state_frames,
        "package_id": package_id,
        "schema_version": STATE_SCHEMA_VERSION,
        "state_revision": 0,
    }


def _validate_state_record(
    record: dict[str, Any],
    *,
    width: int,
    height: int,
    errors: list[str],
    context: str,
) -> None:
    try:
        status = record.get("annotation_status")
        if status not in ALLOWED_MAIN_STATUSES:
            raise ValueError(f"unsupported annotation_status: {status}")
        conflicts = normalize_conflicts(record.get("conflicts", []))
        unresolved_temporal = [
            conflict
            for conflict in conflicts
            if conflict["type"] == "temporal_gt_conflict" and not conflict["resolved"]
        ]
        if status == STATUS_REVIEWED_BOX:
            validate_bbox_xyxy(record.get("human_bbox"), width=width, height=height)
            if record.get("human_reviewed") is not True:
                raise ValueError("reviewed_box requires human_reviewed=true")
            if unresolved_temporal:
                raise ValueError(
                    "unresolved temporal_gt_conflict requires annotation_status temporal_gt_conflict"
                )
        elif status == STATUS_TEMPORAL_GT_CONFLICT:
            note = str(record.get("review_note") or "").strip()
            if not note:
                raise ValueError("temporal_gt_conflict requires review_note")
            if not unresolved_temporal:
                raise ValueError("temporal_gt_conflict status requires an unresolved conflict record")
            for conflict in unresolved_temporal:
                if not str(conflict.get("note") or "").strip():
                    raise ValueError("temporal_gt_conflict conflict record requires note")
            if record.get("human_bbox") is not None:
                validate_bbox_xyxy(record.get("human_bbox"), width=width, height=height)
            if record.get("human_reviewed"):
                raise ValueError("temporal_gt_conflict must not be human_reviewed")
        elif record.get("human_bbox") is not None:
            raise ValueError("human_bbox is only allowed for reviewed_box")
        elif unresolved_temporal:
            raise ValueError(
                "unresolved temporal_gt_conflict requires annotation_status temporal_gt_conflict"
            )
        if record.get("draft_bbox") is not None:
            validate_bbox_xyxy(record.get("draft_bbox"), width=width, height=height)
        normalize_secondary_instances(
            record.get("secondary_instances", []),
            frame_id=str(record.get("frame_id") or context),
            width=width,
            height=height,
        )
    except ValueError as exc:
        if errors is not None:
            errors.append(f"{context}: {exc}")
        else:
            raise


def _validate_gate_without_recursing(
    package_root: Path,
    manifest: dict[str, Any],
    review_data: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    frame_records = review_data.get("frames") or []
    state_frames = state.get("frames") or {}
    if not frame_records:
        errors.append("review_data has no frames")
    expected_total = manifest.get("expected_mandatory_total")
    if len(frame_records) != expected_total:
        errors.append(f"mandatory frame count mismatch: {len(frame_records)} != {expected_total}")
    if len({frame["frame_id"] for frame in frame_records}) != len(frame_records):
        errors.append("duplicate mandatory frame exists")
    if set(state_frames) != {frame["frame_id"] for frame in frame_records}:
        errors.append("state frames do not match mandatory frames")
    for frame in frame_records:
        if not (package_root / frame["crop"]["path"]).exists():
            errors.append(f"missing crop asset: {frame['frame_id']}")
        if frame["dense_frame"] != frame["parent_frame"] - manifest["targets"][frame["benchmark_id"]][
            "frozen_parent_frame_interval"
        ][0]:
            errors.append(f"mapping is not exact: {frame['frame_id']}")
        state_record = state_frames.get(frame["frame_id"])
        if isinstance(state_record, dict):
            _validate_state_record(
                state_record,
                width=int(frame["frame_width"]),
                height=int(frame["frame_height"]),
                errors=errors,
                context=str(frame["frame_id"]),
            )
    return {"errors": errors}


def instructions_markdown(targets: tuple[PlateTargetConfig, ...]) -> str:
    rows = "\n".join(
        f"- `{target.benchmark_id}`: `{target.target_id}`, parent frames "
        f"{target.parent_frame_start}-{target.parent_frame_end}, dense frames "
        f"{target.dense_frame_start}-{target.dense_frame_end}, source interval "
        f"{target.source_interval}."
        for target in targets
    )
    return f"""# Plate Annotation Package

This local package prepares tight license-plate boxes for human review. It is not yet Ground Truth.

## Frozen Target Inputs

{rows}

## Box Policy

Draw the smallest defensible rectangle around the physical plate panel. Do not draw the vehicle,
bumper, broad search ROI, text overlay, or reflection as the target box. Characters do not need to be
readable. If glare or overlay blocks part of the view, annotate the defensible panel extent and add a
note.

The broad ROI is search-only metadata. It is never a target box.

## Draft Policy

Imported proposals are hidden by default. Showing a proposal records `draft_shown=true`, and a
saved human box remains separate from any proposal box. Proposals never increase reviewed counts.

## Secondary Instances

Use secondary records for `additional_localizable_plate`, `ignore_small`, or
`ambiguous_review_required`. Additional plates require a box. Ignore-small and ambiguous records
require a note. Ambiguous records block final export until resolved.

## Temporal Conflicts

If the mandatory target is not defensibly visible/localizable, save `temporal_gt_conflict` with a
non-empty note. This is an explicit unresolved state, not a reviewed box. It blocks final export until
you resolve it by saving a valid tight `reviewed_box`. Showing a draft never resolves a conflict.

## Local Save Contract

This is a local single-reviewer workspace. Saves include a state revision; stale browser tabs are
rejected instead of silently overwriting newer work. No multi-user merge semantics are provided.

## Export Gate

The final review export is locked until every mandatory frame has a valid human box, package identity,
frame mappings, certified source hashes validate, and no unresolved temporal or ambiguous records
remain.
"""


def report_markdown(
    manifest: dict[str, Any],
    progress: dict[str, Any],
    draft_import: dict[str, Any],
) -> str:
    target_lines = "\n".join(
        f"- `{benchmark_id}`: {target['expected_mandatory_count']} mandatory frames, "
        f"{target['evaluation_video']['path']}, hash `{target['evaluation_video']['sha256']}`."
        for benchmark_id, target in sorted(manifest["targets"].items())
    )
    return f"""# Plate Annotation Package Report

Package ID: `{manifest['package_id']}`

Status: `{progress['reviewer_completion_state']['state']}`

## Targets

{target_lines}

## Progress

- Reviewed boxes: {progress['total']['reviewed_box']} / {progress['total']['mandatory']}
- Needs recheck: {progress['total']['needs_recheck']}
- Temporal conflicts: {progress['total']['unresolved_temporal_gt_conflict']}
- Ambiguous secondary records: {progress['total']['unresolved_ambiguous_secondary']}

## Drafts

- Imported: {draft_import['imported']}
- Target-frame proposals: {draft_import['imported_target_frame_count']}
- Exclusions: {json.dumps(draft_import['excluded_by_reason'], sort_keys=True)}

This package is not yet Ground Truth.
"""


def index_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Plate Annotation</title>
    <link rel="stylesheet" href="assets/style.css">
  </head>
  <body>
    <main class="shell">
      <section class="toolbar">
        <button id="prev" title="Previous frame">Prev</button>
        <button id="next" title="Next frame">Next</button>
        <label>Jump <input id="jump" type="number" min="1"></label>
        <select id="status">
          <option value="unreviewed">Unreviewed</option>
          <option value="reviewed_box">Reviewed Box</option>
          <option value="needs_recheck">Needs Recheck</option>
          <option value="temporal_gt_conflict">Temporal GT Conflict</option>
        </select>
        <button id="save" title="Save current frame">Save</button>
        <button id="saveNext" title="Save and advance">Save+Next</button>
        <button id="reset" title="Clear box">Reset</button>
        <label>Draft <input id="showDraft" type="checkbox"></label>
        <label>Zoom <input id="zoom" type="range" min="1" max="4" step="0.25" value="1"></label>
      </section>
      <section class="meta">
        <div id="frameTitle"></div>
        <div id="progress"></div>
        <div id="saveState"></div>
      </section>
      <section class="workspace">
        <figure>
          <img id="context" alt="Full-frame context">
          <figcaption>Context</figcaption>
        </figure>
        <div class="canvasWrap">
          <canvas id="cropCanvas"></canvas>
        </div>
      </section>
      <section class="details">
        <label>Notes <textarea id="note" rows="3"></textarea></label>
        <div class="secondary">
          <select id="secondaryKind">
            <option value="additional_localizable_plate">Additional Plate</option>
            <option value="ignore_small">Ignore Small</option>
            <option value="ambiguous_review_required">Ambiguous</option>
          </select>
          <input id="secondaryNote" placeholder="Secondary note">
          <button id="addSecondary">Add Secondary From Box</button>
        </div>
        <pre id="secondaryList"></pre>
      </section>
    </main>
    <script src="assets/app.js"></script>
  </body>
</html>
"""


def style_css() -> str:
    return """:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
body {
  margin: 0;
  background: #f5f7fb;
  color: #17202a;
}
.shell {
  display: grid;
  gap: 12px;
  padding: 12px;
}
.toolbar,
.meta,
.details {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
button,
select,
input,
textarea {
  border: 1px solid #b7c0ce;
  border-radius: 6px;
  font: inherit;
  padding: 7px 9px;
}
button {
  background: #fff;
  cursor: pointer;
}
button:hover {
  background: #eef4ff;
}
.workspace {
  display: grid;
  gap: 12px;
  grid-template-columns: minmax(280px, 35vw) 1fr;
}
figure {
  margin: 0;
}
#context {
  background: #202734;
  display: block;
  max-width: 100%;
}
.canvasWrap {
  background: #202734;
  overflow: auto;
  padding: 8px;
}
#cropCanvas {
  image-rendering: auto;
}
#frameTitle {
  font-weight: 700;
}
#saveState {
  color: #42526e;
}
textarea {
  min-width: min(680px, 100%);
}
.secondary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
pre {
  background: #fff;
  border: 1px solid #d5dce8;
  border-radius: 6px;
  min-height: 48px;
  overflow: auto;
  padding: 8px;
  width: 100%;
}
@media (max-width: 840px) {
  .workspace {
    grid-template-columns: 1fr;
  }
}
"""


def app_js() -> str:
    return """const state = {
  annotations: null,
  drag: null,
  frames: [],
  image: new Image(),
  index: 0,
  box: null,
  secondary: [],
};

const el = (id) => document.getElementById(id);
const canvas = el("cropCanvas");
const ctx = canvas.getContext("2d");

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function load() {
  const review = await jsonFetch("review_data.json");
  state.frames = review.frames;
  try {
    const apiState = await jsonFetch("/api/state");
    state.annotations = apiState.annotations;
  } catch {
    state.annotations = await jsonFetch("annotations.draft.json");
    el("saveState").textContent = "Serve with redactor to save changes";
  }
  bind();
  renderFrame(0);
}

function bind() {
  el("prev").onclick = () => renderFrame(Math.max(0, state.index - 1));
  el("next").onclick = () => renderFrame(Math.min(state.frames.length - 1, state.index + 1));
  el("save").onclick = () => save(false);
  el("saveNext").onclick = () => save(true);
  el("reset").onclick = () => {
    state.box = null;
    draw();
  };
  el("jump").onchange = () => {
    const nextIndex = Math.max(0, Math.min(state.frames.length - 1, Number(el("jump").value) - 1));
    renderFrame(nextIndex);
  };
  el("zoom").oninput = draw;
  el("showDraft").onchange = async () => {
    const frame = currentFrame();
    if (el("showDraft").checked && frameState().draft_available) {
      try {
        const result = await jsonFetch("/api/drafts/shown", {
          body: JSON.stringify({
            expected_state_revision: state.annotations.state_revision,
            frame_id: frame.frame_id,
          }),
          headers: { "Content-Type": "application/json" },
          method: "POST",
        });
        state.annotations.frames[frame.frame_id] = result.frame;
        state.annotations.state_revision = result.state_revision;
      } catch {}
    }
    draw();
  };
  el("addSecondary").onclick = addSecondary;
  canvas.onmousedown = pointerDown;
  canvas.onmousemove = pointerMove;
  canvas.onmouseup = pointerUp;
  window.onkeydown = (event) => {
    if (event.key === "ArrowLeft") renderFrame(Math.max(0, state.index - 1));
    if (event.key === "ArrowRight") renderFrame(Math.min(state.frames.length - 1, state.index + 1));
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      save(false);
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      state.box = null;
      draw();
    }
  };
}

function currentFrame() {
  return state.frames[state.index];
}

function frameState() {
  return state.annotations.frames[currentFrame().frame_id];
}

function renderFrame(index) {
  state.index = index;
  const frame = currentFrame();
  const annotation = frameState();
  state.box = annotation.human_bbox ? annotation.human_bbox.slice() : null;
  state.secondary = (annotation.secondary_instances || []).map((item) => ({ ...item }));
  el("context").src = frame.context.path;
  el("status").value = annotation.annotation_status;
  el("note").value = annotation.review_note || "";
  el("showDraft").checked = false;
  el("jump").value = index + 1;
  el("frameTitle").textContent = `${frame.benchmark_id} ${frame.target_id} parent ${frame.parent_frame} dense ${frame.dense_frame} time ${frame.timestamp_seconds}s`;
  el("secondaryList").textContent = JSON.stringify(state.secondary, null, 2);
  state.image.onload = draw;
  state.image.src = frame.crop.path;
  updateProgress();
}

function updateProgress() {
  const frames = Object.values(state.annotations.frames);
  const reviewed = frames.filter((item) => item.annotation_status === "reviewed_box").length;
  el("progress").textContent = `${reviewed} / ${frames.length} reviewed`;
}

function cropScale() {
  return Number(el("zoom").value);
}

function draw() {
  if (!state.image.complete) return;
  const zoom = cropScale();
  canvas.width = state.image.naturalWidth * zoom;
  canvas.height = state.image.naturalHeight * zoom;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  const frame = currentFrame();
  if (el("showDraft").checked && frameState().draft_bbox) {
    drawOriginalBox(frameState().draft_bbox, "rgba(255, 204, 0, 0.95)", [8, 6]);
  }
  for (const item of state.secondary) {
    if (item.bbox_xyxy) drawOriginalBox(item.bbox_xyxy, "rgba(0, 180, 130, 0.95)", []);
  }
  if (state.box) drawOriginalBox(state.box, "rgba(35, 120, 255, 0.98)", []);
}

function drawOriginalBox(box, color, dash) {
  const frame = currentFrame();
  const ox = frame.crop.origin_xy[0];
  const oy = frame.crop.origin_xy[1];
  const zoom = cropScale();
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dash);
  ctx.strokeRect((box[0] - ox) * zoom, (box[1] - oy) * zoom, (box[2] - box[0]) * zoom, (box[3] - box[1]) * zoom);
  ctx.restore();
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / cropScale(),
    y: (event.clientY - rect.top) / cropScale(),
  };
}

function cropToOriginal(point) {
  const crop = currentFrame().crop;
  return [Math.round(point.x + crop.origin_xy[0]), Math.round(point.y + crop.origin_xy[1])];
}

function pointerDown(event) {
  const [x, y] = cropToOriginal(pointerPosition(event));
  state.drag = {
    mode: dragModeForPoint(x, y),
    start: [x, y],
    original: state.box ? state.box.slice() : null,
  };
}

function pointerMove(event) {
  if (!state.drag) return;
  const [x, y] = cropToOriginal(pointerPosition(event));
  const start = state.drag.start;
  if (state.drag.mode === "move" && state.drag.original) {
    const dx = x - start[0];
    const dy = y - start[1];
    state.box = clampBox([
      state.drag.original[0] + dx,
      state.drag.original[1] + dy,
      state.drag.original[2] + dx,
      state.drag.original[3] + dy,
    ]);
  } else if (state.drag.mode && state.drag.mode.startsWith("resize") && state.drag.original) {
    state.box = resizeBox(state.drag.original, state.drag.mode, x, y);
  } else {
    state.box = normalizeBox([start[0], start[1], x, y]);
  }
  draw();
}

function pointerUp() {
  state.drag = null;
}

function normalizeBox(box) {
  const frame = currentFrame();
  const crop = frame.crop;
  const x1 = Math.max(crop.origin_xy[0], Math.min(box[0], box[2]));
  const y1 = Math.max(crop.origin_xy[1], Math.min(box[1], box[3]));
  const x2 = Math.min(crop.origin_xy[0] + crop.width, Math.max(box[0], box[2]));
  const y2 = Math.min(crop.origin_xy[1] + crop.height, Math.max(box[1], box[3]));
  return [x1, y1, x2, y2];
}

function clampBox(box) {
  const frame = currentFrame();
  const crop = frame.crop;
  const width = box[2] - box[0];
  const height = box[3] - box[1];
  const minX = crop.origin_xy[0];
  const minY = crop.origin_xy[1];
  const maxX = crop.origin_xy[0] + crop.width;
  const maxY = crop.origin_xy[1] + crop.height;
  const x1 = Math.min(Math.max(box[0], minX), maxX - width);
  const y1 = Math.min(Math.max(box[1], minY), maxY - height);
  return [Math.round(x1), Math.round(y1), Math.round(x1 + width), Math.round(y1 + height)];
}

function resizeBox(original, mode, x, y) {
  const next = original.slice();
  if (mode.includes("w")) next[0] = x;
  if (mode.includes("e")) next[2] = x;
  if (mode.includes("n")) next[1] = y;
  if (mode.includes("s")) next[3] = y;
  return normalizeBox(next);
}

function dragModeForPoint(x, y) {
  if (!state.box) return "draw";
  const tolerance = 10;
  const corners = [
    ["resize-nw", state.box[0], state.box[1]],
    ["resize-ne", state.box[2], state.box[1]],
    ["resize-sw", state.box[0], state.box[3]],
    ["resize-se", state.box[2], state.box[3]],
  ];
  for (const [mode, cx, cy] of corners) {
    if (Math.abs(x - cx) <= tolerance && Math.abs(y - cy) <= tolerance) return mode;
  }
  if (x >= state.box[0] && x <= state.box[2] && y >= state.box[1] && y <= state.box[3]) return "move";
  return "draw";
}

function addSecondary() {
  const kind = el("secondaryKind").value;
  const note = el("secondaryNote").value;
  const item = { bbox_xyxy: state.box ? state.box.slice() : null, kind, note };
  state.secondary.push(item);
  el("secondaryList").textContent = JSON.stringify(state.secondary, null, 2);
}

async function save(advance) {
  const frame = currentFrame();
  const payload = {
    annotation_status: el("status").value,
    draft_shown: frameState().draft_shown || el("showDraft").checked,
    human_bbox: state.box,
    review_note: el("note").value,
    secondary_instances: state.secondary,
  };
  try {
    const result = await jsonFetch("/api/annotations/save", {
      body: JSON.stringify({
        annotation: payload,
        expected_state_revision: state.annotations.state_revision,
        frame_id: frame.frame_id,
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
    state.annotations.frames[frame.frame_id] = result.frame;
    state.annotations.state_revision = result.state_revision;
    el("saveState").textContent = "Saved";
    updateProgress();
    if (advance) renderFrame(Math.min(state.frames.length - 1, state.index + 1));
  } catch (error) {
    el("saveState").textContent = `Save failed: ${error.message}`;
  }
}

load();
"""


class PlateAnnotationRequestHandler(BaseHTTPRequestHandler):
    package_root: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(
                {
                    "annotations": read_json(self.package_root / "annotations.draft.json"),
                    "progress": read_json(self.package_root / "progress.json"),
                    "schema_version": "plate_annotation_server_state.v1",
                }
            )
            return
        self._serve_file(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_payload()
            if parsed.path == "/api/annotations/save":
                frame_id = str(payload.get("frame_id") or "")
                annotation = payload.get("annotation")
                if not isinstance(annotation, dict):
                    raise ValueError("annotation must be an object")
                self._send_json(
                    save_frame_annotation(
                        self.package_root,
                        frame_id,
                        annotation,
                        expected_state_revision=payload.get("expected_state_revision"),
                    )
                )
                return
            if parsed.path == "/api/drafts/shown":
                frame_id = str(payload.get("frame_id") or "")
                self._send_json(
                    mark_draft_shown(
                        self.package_root,
                        frame_id,
                        expected_state_revision=payload.get("expected_state_revision"),
                    )
                )
                return
            if parsed.path == "/api/export":
                final = bool(payload.get("final", False))
                self._send_json(export_plate_annotation_review(self.package_root, final=final))
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
        except StaleStateRevisionError as exc:
            self._send_json(
                {
                    "current_state_revision": exc.current_revision,
                    "error": str(exc),
                    "schema_version": "plate_annotation_stale_write.v1",
                },
                status=HTTPStatus.CONFLICT,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length > MAX_SAVE_PAYLOAD_BYTES:
            raise ValueError("payload too large")
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8") or "{}")

    def _serve_file(self, path: str) -> None:
        relative = unquote(path.lstrip("/")) or "index.html"
        try:
            file_path = safe_package_join(self.package_root, relative)
        except ValueError:
            self._send_json({"error": "path traversal rejected"}, status=HTTPStatus.FORBIDDEN)
            return
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with file_path.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def handler_for_package(package_root: Path) -> type[PlateAnnotationRequestHandler]:
    package_root = package_root.resolve()
    if not (package_root / "package_manifest.json").exists():
        raise ValueError(f"not a plate annotation package: {package_root}")

    class BoundPlateAnnotationRequestHandler(PlateAnnotationRequestHandler):
        pass

    BoundPlateAnnotationRequestHandler.package_root = package_root
    return BoundPlateAnnotationRequestHandler


def serve_plate_annotation_package(
    package_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("plate annotation server binds to loopback only")
    handler = handler_for_package(package_root)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}/"
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        print(f"Serving plate annotation package: {url}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
