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
from collections.abc import Iterable
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
DEFAULT_ASSISTED_OUTPUT_ROOT = Path("artifacts/plate_ground_truth_annotation_v2_assisted")
DEFAULT_ASSISTED_SOURCE_PACKAGE = DEFAULT_OUTPUT_ROOT / "run_20260811T001554Z"
DEFAULT_DAY_VIDEO = Path("artifacts/eval_day_plate.mp4")
DEFAULT_NIGHT_VIDEO = Path("artifacts/eval_night_selective.mp4")
DEFAULT_DRAFT_PATH = Path(
    "artifacts/license_plate_detector_finalists_v2_1/annotation_review/"
    "all_visible_plate_annotations.json"
)
DEFAULT_DRAFT_MAPPING_AUDIT = Path(
    "artifacts/license_plate_detector_finalists_v2_1/clip_mapping_audit.json"
)
DEFAULT_ASSISTED_TRACK_EVIDENCE_PATHS = (
    Path("artifacts/shared_temporal_evidence_v1/night/tracking/tracks.jsonl"),
    Path("artifacts/oim_plate_full_video_v1/night/tracking/tracks.jsonl"),
    Path("artifacts/oim_plate_tracking_v1/night/tracks.jsonl"),
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
ANNOTATION_MODE_MANUAL = "manual_human_review"
ANNOTATION_MODE_ASSISTED = "proposal_assisted_human_review"
PROPOSAL_HISTORICAL_DETECTOR = "historical_detector_draft"
PROPOSAL_AUTOMATIC_TRACKER = "automatic_tracker_draft"
PROPOSAL_DIRECT_DETECTOR = "direct_detector_draft"
PROPOSAL_BOUNDED_INTERPOLATION = "bounded_interpolation_draft"
PROPOSAL_HUMAN_MOTION = "human_seeded_motion_draft"
PROPOSAL_PRIVACY_ENVELOPE = "privacy_envelope_draft"
PROPOSAL_HUMAN_SEEDED = "human_seeded_propagation_draft"
PROPOSAL_INTERPOLATED = "interpolated_draft"
PROPOSAL_QUALITY_SAFE_TO_REVIEW = "safe_to_review"
PROPOSAL_QUALITY_NEEDS_ADJUSTMENT = "needs_human_adjustment"
PROPOSAL_QUALITY_NEEDS_MANUAL_START = "needs_manual_start"
TRACK_PROVENANCE_DETECTOR = "plate_detector_observation"
TRACK_PROVENANCE_INTERPOLATION = "plate_track_interpolation"
TRACK_STATUS_ACCEPTED = "accepted_plate_track"
MAX_STATIC_HUMAN_COPY_DIAGNOSTIC_WINDOW_FRAMES = 1
OPTIONAL_PROGRESS_ROW_KEYS = (
    "human_reviewed_total",
    "no_proposal_remaining",
    "proposal_accepted_unchanged",
    "proposal_available_total",
    "proposal_human_adjusted",
    "proposal_source_breakdown",
    "temporal_gt_conflicts",
)


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


def build_assisted_plate_annotation_package(
    *,
    source_package_root: Path = DEFAULT_ASSISTED_SOURCE_PACKAGE,
    output_root: Path = DEFAULT_ASSISTED_OUTPUT_ROOT,
    overwrite: bool = False,
    run_id: str | None = None,
    verify_git_ignore: bool = True,
    repo_root: Path | None = None,
    proposal_evidence_paths: tuple[Path, ...] | None = None,
    targets: tuple[PlateTargetConfig, ...] | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or Path.cwd()).resolve()
    output_root = output_root.resolve()
    source_package_root = source_package_root.resolve()
    if verify_git_ignore:
        assert_annotation_root_is_ignored(output_root, repo_root)
    source_validation = validate_plate_annotation_package(source_package_root)
    if not source_validation["passed"]:
        raise ValueError(
            f"source package validation failed: {'; '.join(source_validation['errors'])}"
        )

    source_manifest = read_json(source_package_root / "package_manifest.json")
    target_configs = targets or target_configs_from_manifest(source_package_root, source_manifest)
    certified_config_version = str(
        source_manifest.get("certified_config_version") or CUSTOM_CERTIFIED_CONFIG_VERSION
    )
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
            annotation_mode=ANNOTATION_MODE_ASSISTED,
            include_drafts=True,
            draft_path=DEFAULT_DRAFT_PATH,
            draft_mapping_audit_path=DEFAULT_DRAFT_MAPPING_AUDIT,
            repo_root=repo_root,
        )
        migration_audit = apply_assisted_migration(
            source_package_root=source_package_root,
            package_root=temp_root,
            final_package_root=package_root,
            repo_root=repo_root,
            proposal_evidence_paths=proposal_evidence_paths,
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
    summary = summarize_plate_annotation_package(package_root)
    return summary | {
        "assisted_migration": migration_audit,
        "package_root": package_root.as_posix(),
        "validation": validation,
    }


def target_configs_from_manifest(
    package_root: Path,
    manifest: dict[str, Any],
) -> tuple[PlateTargetConfig, ...]:
    configs = []
    for benchmark_id, target in sorted(manifest.get("targets", {}).items()):
        parent_start, parent_end = target["frozen_parent_frame_interval"]
        offset = int(target["parent_to_dense_mapping"]["offset"])
        video = target["evaluation_video"]
        configs.append(
            PlateTargetConfig(
                action=str(target["action"]),
                benchmark_id=str(benchmark_id),
                broad_roi_xyxy=tuple(int(value) for value in target["broad_search_roi_xyxy"]),
                dense_frame_offset=offset,
                expected_fps=video.get("fps"),
                expected_height=int(video["height"]),
                expected_sha256=video.get("sha256"),
                expected_width=int(video["width"]),
                object_type=str(target["object_type"]),
                parent_frame_end=int(parent_end),
                parent_frame_start=int(parent_start),
                source_interval=str(target["source_interval"]),
                source_video_label=str(target["source_video_label"]),
                target_id=str(target["target_id"]),
                total_video_frames=int(video["frame_count"]),
                video_path=resolve_evaluation_video_path(package_root, str(video["path"])),
            )
        )
    return tuple(configs)


def package_core_hashes(package_root: Path) -> dict[str, str]:
    return {
        name: media.sha256_file(package_root / name)
        for name in (
            "annotations.draft.json",
            "change_log.jsonl",
            "package_identity.json",
            "package_manifest.json",
            "progress.json",
            "review_data.json",
        )
    }


def refresh_package_identity_hashes(package_root: Path) -> None:
    identity = read_json(package_root / "package_identity.json")
    identity["package_manifest_sha256"] = media.sha256_file(package_root / "package_manifest.json")
    identity["review_data_sha256"] = media.sha256_file(package_root / "review_data.json")
    write_json(package_root / "package_identity.json", identity)


def reviewed_annotation_semantic_hash(
    review_data: dict[str, Any],
    state: dict[str, Any],
) -> str:
    frames = []
    for frame in review_data.get("frames", []):
        record = state.get("frames", {}).get(frame["frame_id"], {})
        if record.get("annotation_status") != STATUS_REVIEWED_BOX:
            continue
        frames.append(migration_compare_record(frame, record))
    return stable_json_sha256(frames)


def migration_compare_record(frame: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_status": record.get("annotation_status"),
        "benchmark_id": frame.get("benchmark_id"),
        "conflicts": record.get("conflicts", []),
        "dense_frame": frame.get("dense_frame"),
        "draft_shown": record.get("draft_shown", False),
        "frame_id": frame.get("frame_id"),
        "human_bbox": record.get("human_bbox"),
        "human_reviewed": record.get("human_reviewed"),
        "parent_frame": frame.get("parent_frame"),
        "review_note": record.get("review_note", ""),
        "reviewed_at": record.get("reviewed_at"),
        "secondary_instances": record.get("secondary_instances", []),
        "target_id": frame.get("target_id"),
    }


def apply_assisted_migration(
    *,
    source_package_root: Path,
    package_root: Path,
    final_package_root: Path,
    repo_root: Path | None = None,
    proposal_evidence_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    source_manifest = read_json(source_package_root / "package_manifest.json")
    source_review = read_json(source_package_root / "review_data.json")
    source_state = read_json(source_package_root / "annotations.draft.json")
    review_data = read_json(package_root / "review_data.json")
    state = read_json(package_root / "annotations.draft.json")
    manifest = read_json(package_root / "package_manifest.json")
    manifest["annotation_mode"] = ANNOTATION_MODE_ASSISTED
    manifest["assisted_migration"] = {
        "source_package_hashes": package_core_hashes(source_package_root),
        "source_package_id": source_manifest.get("package_id"),
        "source_package_root": source_package_root.as_posix(),
    }
    source_frames = {frame["frame_id"]: frame for frame in source_review["frames"]}
    source_records = source_state["frames"]
    proposals, proposal_audit = assisted_proposals_for_frames(
        source_review,
        source_state,
        review_data,
        manifest=manifest,
        repo_root=repo_root or Path.cwd(),
        evidence_paths=proposal_evidence_paths,
    )

    migrated_count = 0
    bbox_mismatch_count = 0
    missing_record_count = 0
    changed_record_count = 0
    migrated_frames = []
    for frame in review_data["frames"]:
        frame_id = frame["frame_id"]
        record = dict(state["frames"][frame_id])
        source_frame = source_frames.get(frame_id)
        source_record = source_records.get(frame_id)
        if source_frame is None or not isinstance(source_record, dict):
            missing_record_count += 1
            continue
        if source_record.get("annotation_status") == STATUS_REVIEWED_BOX:
            migrated_count += 1
            old_bbox = source_record.get("human_bbox")
            record.update(
                {
                    "accepted_without_geometry_change": False,
                    "annotation_status": STATUS_REVIEWED_BOX,
                    "conflicts": source_record.get("conflicts", []),
                    "draft_available": False,
                    "draft_bbox": None,
                    "draft_candidates": [],
                    "draft_model": None,
                    "draft_shown": bool(source_record.get("draft_shown", False)),
                    "draft_source": None,
                    "human_adjusted_geometry": False,
                    "human_bbox": old_bbox,
                    "human_reviewed": True,
                    "legacy_draft_available": bool(source_record.get("draft_available", False)),
                    "legacy_draft_bbox": source_record.get("draft_bbox"),
                    "migration_source_package_id": source_manifest.get("package_id"),
                    "proposal_available": False,
                    "proposal_bbox": None,
                    "proposal_candidates": [],
                    "proposal_quality": None,
                    "proposal_model": None,
                    "proposal_shown": False,
                    "proposal_source": None,
                    "review_note": source_record.get("review_note", ""),
                    "reviewed_at": source_record.get("reviewed_at"),
                    "secondary_instances": source_record.get("secondary_instances", []),
                }
            )
            state["frames"][frame_id] = record
            if record["human_bbox"] != old_bbox:
                bbox_mismatch_count += 1
            if migration_compare_record(frame, record) != migration_compare_record(
                source_frame, source_record
            ):
                changed_record_count += 1
            migrated_frames.append(frame_id)
            continue

        proposal = proposals.get(frame_id)
        record.update(
            {
                "accepted_without_geometry_change": False,
                "annotation_status": STATUS_UNREVIEWED,
                "conflicts": source_record.get("conflicts", []),
                "draft_available": proposal is not None,
                "draft_bbox": proposal.get("bbox_xyxy") if proposal else None,
                "draft_candidates": proposal.get("candidates", []) if proposal else [],
                "draft_model": proposal.get("model") if proposal else None,
                "draft_shown": False,
                "draft_source": proposal.get("source") if proposal else None,
                "human_adjusted_geometry": False,
                "human_bbox": None,
                "human_reviewed": False,
                "proposal_available": proposal is not None,
                "proposal_bbox": proposal.get("bbox_xyxy") if proposal else None,
                "proposal_candidates": proposal.get("candidates", []) if proposal else [],
                "proposal_quality": proposal.get("proposal_quality") if proposal else None,
                "proposal_model": proposal.get("model") if proposal else None,
                "proposal_shown": False,
                "proposal_source": proposal.get("source") if proposal else None,
                "review_note": "",
                "reviewed_at": None,
                "secondary_instances": source_record.get("secondary_instances", []),
            }
        )
        state["frames"][frame_id] = record

    state["package_id"] = final_package_root.name
    progress = compute_progress(review_data, state)
    proposal_breakdown = dict(progress["total"]["proposal_source_breakdown"])
    no_proposal_frames = [
        frame_id
        for frame_id, record in state["frames"].items()
        if record.get("annotation_status") != STATUS_REVIEWED_BOX
        and not record.get("proposal_available")
    ]
    migration_audit = {
        "bbox_mismatch_count": bbox_mismatch_count,
        "changed_record_count": changed_record_count,
        "migrated_record_count": migrated_count,
        "migrated_reviewed_frame_ids": migrated_frames,
        "missing_record_count": missing_record_count,
        "new_package_id": final_package_root.name,
        "old_annotation_semantic_hash": reviewed_annotation_semantic_hash(
            source_review, source_state
        ),
        "old_package_hash": stable_json_sha256(package_core_hashes(source_package_root)),
        "old_package_hashes": package_core_hashes(source_package_root),
        "schema_version": "plate_annotation_assisted_migration_audit.v1",
        "source_package_id": source_manifest.get("package_id"),
        "source_package_root": source_package_root.as_posix(),
    }
    proposal_audit |= {
        "annotation_mode": ANNOTATION_MODE_ASSISTED,
        "no_proposal_frame_ids": no_proposal_frames,
        "no_proposal_remaining": len(no_proposal_frames),
        "proposal_available_total": progress["total"]["proposal_available_total"],
        "proposal_source_breakdown": proposal_breakdown,
        "remaining_unreviewed": progress["total"]["mandatory"] - progress["total"]["reviewed_box"],
    }
    write_json(package_root / "package_manifest.json", manifest)
    write_json(package_root / "annotations.draft.json", state)
    write_json(package_root / "progress.json", progress)
    write_json(package_root / "migration_audit.json", migration_audit)
    write_json(package_root / "proposal_generation_audit.json", proposal_audit)
    refresh_package_identity_hashes(package_root)
    shutil.copyfile(
        source_package_root / "change_log.jsonl", package_root / "migration_source_change_log.jsonl"
    )
    write_text(
        package_root / "instructions.md",
        instructions_markdown(target_configs_from_manifest(package_root, manifest)),
    )
    write_text(
        package_root / "report.md",
        report_markdown(manifest, progress, read_json(package_root / "draft_import_audit.json")),
    )
    return migration_audit | {"proposal_generation": proposal_audit}


def assisted_proposals_for_frames(
    source_review: dict[str, Any],
    source_state: dict[str, Any],
    review_data: dict[str, Any],
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    evidence_paths: tuple[Path, ...] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_records = source_state["frames"]
    human_anchors: dict[str, list[tuple[int, list[int], str]]] = {}
    for frame in source_review["frames"]:
        record = source_records[frame["frame_id"]]
        if record.get("annotation_status") == STATUS_REVIEWED_BOX and record.get("human_bbox"):
            human_anchors.setdefault(frame["benchmark_id"], []).append(
                (int(frame["parent_frame"]), list(record["human_bbox"]), frame["frame_id"])
            )
    for anchors in human_anchors.values():
        anchors.sort(key=lambda item: item[0])

    evidence = load_assisted_track_evidence(
        manifest=manifest,
        human_anchors=human_anchors,
        repo_root=repo_root,
        evidence_paths=evidence_paths,
    )

    proposals: dict[str, dict[str, Any]] = {}
    frame_audits: list[dict[str, Any]] = []
    rejected_candidate_count = 0
    for frame in review_data["frames"]:
        frame_id = frame["frame_id"]
        source_record = source_records.get(frame_id)
        if not isinstance(source_record, dict):
            continue
        if source_record.get("annotation_status") == STATUS_REVIEWED_BOX:
            continue
        candidates, discarded = assisted_track_candidates_for_frame(frame, evidence)
        rejected_candidate_count += len(discarded)
        primary = (
            sorted(
                candidates,
                key=lambda item: (
                    proposal_priority(str(item["source"])),
                    int(item.get("evidence_source_index", 999)),
                    -float(item.get("score") or 0.0),
                    str(item.get("source_track_uid") or ""),
                ),
            )[0]
            if candidates
            else None
        )
        nearest_human = nearest_anchor(
            human_anchors.get(frame["benchmark_id"], []), int(frame["parent_frame"])
        )
        temporal_distance = (
            abs(int(frame["parent_frame"]) - nearest_human[0])
            if nearest_human is not None
            else None
        )
        frame_audit = {
            "benchmark_id": frame["benchmark_id"],
            "dense_frame": frame["dense_frame"],
            "discarded_candidates": discarded,
            "frame_id": frame_id,
            "no_proposal_reason": None,
            "parent_frame": frame["parent_frame"],
            "proposal_available": primary is not None,
            "proposal_qa_result": primary.get("proposal_qa_result")
            if primary
            else {"status": "no_candidate"},
            "rejection_reason": None,
            "selected_proposal_bbox": primary.get("bbox_xyxy") if primary else None,
            "selected_proposal_provenance": primary.get("source") if primary else None,
            "source_detection_ids": primary.get("source_detection_ids") if primary else [],
            "source_track_id": primary.get("source_track_id") if primary else None,
            "source_track_uid": primary.get("source_track_uid") if primary else None,
            "evidence_source_path": primary.get("evidence_source_path") if primary else None,
            "evidence_source_sha256": primary.get("evidence_source_sha256") if primary else None,
            "left_anchor": primary.get("left_anchor") if primary else None,
            "right_anchor": primary.get("right_anchor") if primary else None,
            "temporal_distance_from_nearest_human": temporal_distance,
        }
        if primary is None:
            frame_audit["no_proposal_reason"] = (
                "candidate_rejected_by_qa"
                if discarded
                else "no_target_associated_evidence_on_frame"
            )
            frame_audit["rejection_reason"] = frame_audit["no_proposal_reason"]
        else:
            proposals[frame_id] = primary | {"candidates": candidates}
        frame_audits.append(frame_audit)

    source_counts = Counter(str(proposal["source"]) for proposal in proposals.values())
    proposal_sources = (
        PROPOSAL_AUTOMATIC_TRACKER,
        PROPOSAL_DIRECT_DETECTOR,
        PROPOSAL_BOUNDED_INTERPOLATION,
        PROPOSAL_HUMAN_MOTION,
        PROPOSAL_PRIVACY_ENVELOPE,
    )
    proposal_audit = {
        "evidence_sources": [
            public_evidence_source_audit(source) for source in evidence["sources"]
        ],
        "frames": frame_audits,
        "human_anchor_counts": {
            benchmark: len(anchors) for benchmark, anchors in sorted(human_anchors.items())
        },
        "proposal_policy": {
            "forbidden_static_source": PROPOSAL_HUMAN_SEEDED,
            "max_static_human_copy_diagnostic_window_frames": MAX_STATIC_HUMAN_COPY_DIAGNOSTIC_WINDOW_FRAMES,
            "priority_order": list(proposal_sources),
            "qa_only_not_metrics": True,
            "unbounded_extrapolation_allowed": False,
        },
        "proposal_source_counts": {
            source: source_counts.get(source, 0) for source in proposal_sources
        },
        "rejected_candidate_count": rejected_candidate_count,
        "schema_version": "plate_annotation_assisted_proposal_audit.v2",
        "target_association": evidence["target_association"],
        "total_unreviewed": len(frame_audits),
    }
    return proposals, proposal_audit


def load_assisted_track_evidence(
    *,
    manifest: dict[str, Any],
    human_anchors: dict[str, list[tuple[int, list[int], str]]],
    repo_root: Path,
    evidence_paths: tuple[Path, ...] | None,
) -> dict[str, Any]:
    paths = DEFAULT_ASSISTED_TRACK_EVIDENCE_PATHS if evidence_paths is None else evidence_paths
    targets_by_hash = {
        str(target.get("evaluation_video", {}).get("sha256")): benchmark_id
        for benchmark_id, target in sorted(manifest.get("targets", {}).items())
        if isinstance(target, dict) and target.get("evaluation_video", {}).get("sha256")
    }
    sources = []
    target_association: dict[str, Any] = {}
    for source_index, evidence_path in enumerate(paths):
        resolved_path = resolve_repo_relative_path(evidence_path, repo_root)
        source = {
            "exists": resolved_path.exists(),
            "index": source_index,
            "manifest": {},
            "manifest_path": None,
            "path": display_repo_path(resolved_path, repo_root),
            "reason": None,
            "sha256": media.sha256_file(resolved_path) if resolved_path.exists() else None,
            "target_benchmark": None,
            "tracks_by_benchmark": {},
            "selected_tracks_by_benchmark": {},
            "usable": False,
        }
        if not resolved_path.exists():
            source["reason"] = "evidence_path_missing"
            sources.append(source)
            continue
        manifest_path = tracking_manifest_for_tracks_path(resolved_path)
        source["manifest_path"] = (
            display_repo_path(manifest_path, repo_root) if manifest_path else None
        )
        source["manifest_sha256"] = media.sha256_file(manifest_path) if manifest_path else None
        evidence_manifest = read_json(manifest_path) if manifest_path else {}
        source["manifest"] = evidence_manifest
        source_sha256 = str(evidence_manifest.get("source_sha256") or "")
        benchmark_id = targets_by_hash.get(source_sha256)
        if benchmark_id is None:
            source["reason"] = "source_sha256_does_not_match_package_targets"
            sources.append(source)
            continue
        target = manifest["targets"][benchmark_id]
        if evidence_manifest.get("input_width") != target.get("evaluation_video", {}).get(
            "width"
        ) or evidence_manifest.get("input_height") != target.get("evaluation_video", {}).get(
            "height"
        ):
            source["reason"] = "source_dimensions_do_not_match_package_target"
            sources.append(source)
            continue
        tracks_by_uid: dict[str, dict[int, dict[str, Any]]] = {}
        for frame_index, track_record in iter_track_evidence_records(resolved_path):
            track_uid = str(
                track_record.get("track_uid")
                or track_record.get("track_id")
                or f"track_{len(tracks_by_uid):04d}"
            )
            tracks_by_uid.setdefault(track_uid, {})[frame_index] = normalized_track_record(
                track_record,
                frame_index=frame_index,
            )
        source["target_benchmark"] = benchmark_id
        source["tracks_by_benchmark"] = {benchmark_id: tracks_by_uid}
        source["usable"] = True
        source["max_safe_interpolation_gap_frames"] = max_safe_interpolation_gap_frames(
            evidence_manifest
        )
        source["max_one_frame_center_jump_norm"] = float(
            (evidence_manifest.get("tracking_configuration") or {}).get(
                "max_accepted_center_jump_norm", 4.0
            )
        )
        source["qa_bounds"] = {
            "max_area_fraction": float(
                (evidence_manifest.get("tracking_configuration") or {}).get(
                    "max_area_fraction", 0.08
                )
            ),
            "max_aspect_ratio": float(
                (evidence_manifest.get("tracking_configuration") or {}).get("max_aspect_ratio", 8.0)
            ),
            "min_area_fraction": float(
                (evidence_manifest.get("tracking_configuration") or {}).get(
                    "min_area_fraction", 4e-05
                )
            ),
            "min_aspect_ratio": float(
                (evidence_manifest.get("tracking_configuration") or {}).get("min_aspect_ratio", 0.7)
            ),
        }
        associate_source_target_tracks(source, manifest, human_anchors)
        target_association[source["path"]] = source.get("target_association", {})
        sources.append(source)
    return {"sources": sources, "target_association": target_association}


def resolve_repo_relative_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def display_repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def tracking_manifest_for_tracks_path(tracks_path: Path) -> Path | None:
    for candidate in (
        tracks_path.with_name("tracking_manifest.json"),
        tracks_path.with_name("summary.json"),
    ):
        if candidate.exists():
            return candidate
    return None


def iter_track_evidence_records(tracks_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with tracks_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload.get("tracks"), list):
                frame_index = int(payload["frame_index"])
                for track in payload["tracks"]:
                    if isinstance(track, dict):
                        yield frame_index, track
                continue
            if isinstance(payload.get("observations"), list):
                track_uid = payload.get("track_uid") or payload.get("track_id")
                for observation in payload["observations"]:
                    if not isinstance(observation, dict) or "frame_index" not in observation:
                        continue
                    frame_index = int(observation["frame_index"])
                    yield (
                        frame_index,
                        observation
                        | {
                            "status": payload.get("status"),
                            "track_uid": track_uid,
                        },
                    )
                continue
            if "frame_index" in payload and "bbox_xyxy" in payload:
                yield int(payload["frame_index"]), payload


def normalized_track_record(track_record: dict[str, Any], *, frame_index: int) -> dict[str, Any]:
    return {
        "anchor_frames": track_record.get("anchor_frames", []),
        "bbox_xyxy": list(track_record.get("bbox_xyxy") or []),
        "category": track_record.get("category"),
        "frame_index": frame_index,
        "provenance": track_record.get("provenance"),
        "score": track_record.get("score"),
        "source_candidate_ids": list(track_record.get("source_candidate_ids") or []),
        "status": track_record.get("status"),
        "track_id": track_record.get("track_id"),
        "track_uid": track_record.get("track_uid") or track_record.get("track_id"),
    }


def max_safe_interpolation_gap_frames(evidence_manifest: dict[str, Any]) -> int:
    summary = evidence_manifest.get("tracking_summary") or {}
    if isinstance(summary.get("max_safe_interpolation_gap_frames"), int):
        return int(summary["max_safe_interpolation_gap_frames"])
    configuration = evidence_manifest.get("tracking_configuration") or {}
    seconds = configuration.get("max_safe_interpolation_gap_seconds")
    fps = evidence_manifest.get("nominal_fps") or 30.0
    if isinstance(seconds, int | float) and isinstance(fps, int | float):
        return max(1, int(math.floor(float(seconds) * float(fps))))
    return 9


def associate_source_target_tracks(
    source: dict[str, Any],
    manifest: dict[str, Any],
    human_anchors: dict[str, list[tuple[int, list[int], str]]],
) -> None:
    source["target_association"] = {}
    source["selected_tracks_by_benchmark"] = {}
    benchmark_id = source.get("target_benchmark")
    if not benchmark_id:
        return
    target = manifest["targets"][benchmark_id]
    start, end = target["frozen_parent_frame_interval"]
    anchors = human_anchors.get(str(benchmark_id), [])
    tracks_by_uid = source.get("tracks_by_benchmark", {}).get(benchmark_id, {})
    required_overlap = max(1, min(5, len(anchors)))
    selected: set[str] = set()
    track_audits = []
    for track_uid, by_frame in sorted(tracks_by_uid.items()):
        interval_frames = sorted(frame for frame in by_frame if int(start) <= frame <= int(end))
        overlap_values = []
        overlap_frames = []
        for parent_frame, human_bbox, frame_id in anchors:
            record = by_frame.get(parent_frame)
            if record is None:
                continue
            candidate_bbox = record.get("bbox_xyxy")
            if not isinstance(candidate_bbox, list) or len(candidate_bbox) != 4:
                continue
            value = bbox_iou(human_bbox, candidate_bbox)
            overlap_values.append(value)
            overlap_frames.append(
                {
                    "frame_id": frame_id,
                    "human_bbox": human_bbox,
                    "iou": round(value, 6),
                    "parent_frame": parent_frame,
                    "track_bbox": candidate_bbox,
                }
            )
        if not interval_frames and not overlap_values:
            continue
        mean_iou = sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
        max_iou = max(overlap_values) if overlap_values else 0.0
        min_iou = min(overlap_values) if overlap_values else 0.0
        selected_reason = None
        rejected_reason = None
        if len(overlap_values) >= required_overlap and (mean_iou >= 0.15 or max_iou >= 0.45):
            selected.add(str(track_uid))
            selected_reason = "selected_by_human_anchor_overlap"
        elif len(overlap_values) < required_overlap:
            rejected_reason = "insufficient_human_anchor_overlap"
        else:
            rejected_reason = "insufficient_human_anchor_agreement"
        statuses = Counter(str(record.get("status")) for record in by_frame.values())
        provenances = Counter(str(record.get("provenance")) for record in by_frame.values())
        track_audits.append(
            {
                "first_target_frame": interval_frames[0] if interval_frames else None,
                "frame_count_in_target_interval": len(interval_frames),
                "last_target_frame": interval_frames[-1] if interval_frames else None,
                "max_iou_with_human_anchors": round(max_iou, 6),
                "mean_iou_with_human_anchors": round(mean_iou, 6),
                "min_iou_with_human_anchors": round(min_iou, 6),
                "overlap_frame_count": len(overlap_values),
                "overlap_frames": overlap_frames,
                "provenance_counts": dict(sorted(provenances.items())),
                "rejected_reason": rejected_reason,
                "selected": str(track_uid) in selected,
                "selected_reason": selected_reason,
                "status_counts": dict(sorted(statuses.items())),
                "track_uid": str(track_uid),
            }
        )
    source["selected_tracks_by_benchmark"][benchmark_id] = selected
    source["target_association"][benchmark_id] = {
        "human_anchor_count": len(anchors),
        "required_overlap": required_overlap,
        "selected_track_uids": sorted(selected),
        "track_audits": track_audits,
    }


def bbox_iou(
    a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = [int(value) for value in a]
    bx1, by1, bx2, by2 = [int(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    union = box_area([ax1, ay1, ax2, ay2]) + box_area([bx1, by1, bx2, by2]) - intersection
    return intersection / union if union > 0 else 0.0


def assisted_track_candidates_for_frame(
    frame: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_id = str(frame["benchmark_id"])
    parent_frame = int(frame["parent_frame"])
    candidates = []
    discarded = []
    for source in evidence["sources"]:
        if not source.get("usable"):
            continue
        selected_tracks = source.get("selected_tracks_by_benchmark", {}).get(benchmark_id, set())
        tracks_by_uid = source.get("tracks_by_benchmark", {}).get(benchmark_id, {})
        for track_uid in sorted(selected_tracks):
            by_frame = tracks_by_uid.get(track_uid, {})
            record = by_frame.get(parent_frame)
            if record is None:
                continue
            candidate, rejection = assisted_candidate_from_track_record(
                frame=frame,
                source=source,
                track_uid=track_uid,
                by_frame=by_frame,
                record=record,
            )
            if candidate is not None:
                candidates.append(candidate)
            elif rejection is not None:
                discarded.append(rejection)
    return candidates, discarded


def assisted_candidate_from_track_record(
    *,
    frame: dict[str, Any],
    source: dict[str, Any],
    track_uid: str,
    by_frame: dict[int, dict[str, Any]],
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parent_frame = int(frame["parent_frame"])
    provenance = str(record.get("provenance") or "")
    status = str(record.get("status") or "")
    bbox_value = list(record.get("bbox_xyxy") or [])
    left_anchor = None
    right_anchor = None
    if status == TRACK_STATUS_ACCEPTED:
        source_name = PROPOSAL_AUTOMATIC_TRACKER
        quality = (
            PROPOSAL_QUALITY_NEEDS_ADJUSTMENT
            if provenance == TRACK_PROVENANCE_INTERPOLATION
            else PROPOSAL_QUALITY_SAFE_TO_REVIEW
        )
        if provenance == TRACK_PROVENANCE_INTERPOLATION:
            left_anchor, right_anchor = bounded_interpolation_anchors(
                parent_frame,
                by_frame=by_frame,
                max_gap_frames=int(source.get("max_safe_interpolation_gap_frames", 9)),
            )
    elif provenance == TRACK_PROVENANCE_DETECTOR:
        source_name = PROPOSAL_DIRECT_DETECTOR
        quality = PROPOSAL_QUALITY_SAFE_TO_REVIEW
    elif provenance == TRACK_PROVENANCE_INTERPOLATION:
        left_anchor, right_anchor = bounded_interpolation_anchors(
            parent_frame,
            by_frame=by_frame,
            max_gap_frames=int(source.get("max_safe_interpolation_gap_frames", 9)),
        )
        if left_anchor is None or right_anchor is None:
            return None, discarded_candidate_audit(
                bbox_value,
                source=source,
                track_uid=track_uid,
                record=record,
                rejection_reason="interpolation_not_bounded_by_trusted_detector_anchors",
            )
        source_name = PROPOSAL_BOUNDED_INTERPOLATION
        quality = PROPOSAL_QUALITY_NEEDS_ADJUSTMENT
    else:
        return None, discarded_candidate_audit(
            bbox_value,
            source=source,
            track_uid=track_uid,
            record=record,
            rejection_reason="unsupported_track_record_provenance",
        )

    qa_result = proposal_candidate_qa(
        bbox_value,
        frame=frame,
        source=source,
        by_frame=by_frame,
        parent_frame=parent_frame,
    )
    if qa_result["status"] != "passed":
        return None, discarded_candidate_audit(
            bbox_value,
            source=source,
            track_uid=track_uid,
            record=record,
            rejection_reason=str(qa_result.get("reason") or "candidate_failed_qa"),
            qa_result=qa_result,
        )
    if qa_result.get("needs_adjustment"):
        quality = PROPOSAL_QUALITY_NEEDS_ADJUSTMENT
    candidate = {
        "anchor_frame_id": None,
        "anchor_parent_frame": None,
        "bbox_xyxy": qa_result["bbox_xyxy"],
        "evidence_source_index": source["index"],
        "evidence_source_path": source["path"],
        "evidence_source_sha256": source["sha256"],
        "left_anchor": left_anchor,
        "model": source.get("manifest", {}).get("detector_id")
        or source.get("manifest", {}).get("tracking_configuration_hash"),
        "proposal_quality": quality,
        "proposal_qa_result": qa_result | {"quality": quality},
        "right_anchor": right_anchor,
        "score": record.get("score"),
        "source": source_name,
        "source_detection_ids": list(record.get("source_candidate_ids") or []),
        "source_track_id": record.get("track_id"),
        "source_track_provenance": provenance,
        "source_track_status": status,
        "source_track_uid": track_uid,
    }
    return candidate, None


def bounded_interpolation_anchors(
    parent_frame: int,
    *,
    by_frame: dict[int, dict[str, Any]],
    max_gap_frames: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    trusted_detector_frames = sorted(
        frame_index
        for frame_index, record in by_frame.items()
        if str(record.get("provenance") or "") == TRACK_PROVENANCE_DETECTOR
    )
    left_frames = [
        frame_index for frame_index in trusted_detector_frames if frame_index < parent_frame
    ]
    right_frames = [
        frame_index for frame_index in trusted_detector_frames if frame_index > parent_frame
    ]
    if not left_frames or not right_frames:
        return None, None
    left_frame = left_frames[-1]
    right_frame = right_frames[0]
    if right_frame - left_frame - 1 > max_gap_frames:
        return None, None
    return track_anchor_audit(left_frame, by_frame[left_frame]), track_anchor_audit(
        right_frame,
        by_frame[right_frame],
    )


def track_anchor_audit(parent_frame: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_xyxy": record.get("bbox_xyxy"),
        "parent_frame": parent_frame,
        "provenance": record.get("provenance"),
        "score": record.get("score"),
        "source_detection_ids": list(record.get("source_candidate_ids") or []),
    }


def proposal_candidate_qa(
    bbox_value: list[Any],
    *,
    frame: dict[str, Any],
    source: dict[str, Any],
    by_frame: dict[int, dict[str, Any]],
    parent_frame: int,
) -> dict[str, Any]:
    try:
        bbox = validate_bbox_xyxy(
            bbox_value,
            width=int(frame["frame_width"]),
            height=int(frame["frame_height"]),
        )
    except ValueError as exc:
        return {"reason": str(exc), "status": "rejected"}
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect_ratio = width / height
    area_fraction = box_area(bbox) / (int(frame["frame_width"]) * int(frame["frame_height"]))
    bounds = source.get("qa_bounds") or {}
    min_aspect = float(bounds.get("min_aspect_ratio", 0.7))
    max_aspect = float(bounds.get("max_aspect_ratio", 8.0))
    min_area = float(bounds.get("min_area_fraction", 4e-05))
    max_area = float(bounds.get("max_area_fraction", 0.08))
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        return {
            "aspect_ratio": round(aspect_ratio, 6),
            "bbox_xyxy": bbox,
            "reason": "aspect_ratio_outside_source_bounds",
            "status": "rejected",
        }
    if area_fraction < min_area or area_fraction > max_area:
        return {
            "area_fraction": round(area_fraction, 8),
            "bbox_xyxy": bbox,
            "reason": "area_fraction_outside_source_bounds",
            "status": "rejected",
        }
    neighbor_check = nearest_neighbor_motion_check(
        bbox,
        parent_frame=parent_frame,
        by_frame=by_frame,
        max_center_jump_norm=float(source.get("max_one_frame_center_jump_norm", 4.0)),
    )
    if neighbor_check["status"] != "passed":
        return {
            "bbox_xyxy": bbox,
            "reason": neighbor_check["reason"],
            "status": "rejected",
        } | neighbor_check
    scale_check = track_scale_quality_check(bbox, by_frame=by_frame)
    return {
        "area_fraction": round(area_fraction, 8),
        "aspect_ratio": round(aspect_ratio, 6),
        "bbox_xyxy": bbox,
        "needs_adjustment": scale_check["needs_adjustment"],
        "scale_check": scale_check,
        "motion_check": neighbor_check,
        "status": "passed",
    }


def track_scale_quality_check(
    bbox: list[int],
    *,
    by_frame: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    widths = []
    heights = []
    areas = []
    for record in by_frame.values():
        candidate = record.get("bbox_xyxy")
        if not isinstance(candidate, list) or len(candidate) != 4:
            continue
        width = int(candidate[2]) - int(candidate[0])
        height = int(candidate[3]) - int(candidate[1])
        if width <= 0 or height <= 0:
            continue
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
    if len(widths) < 6:
        return {"needs_adjustment": False, "scale_check_reason": "insufficient_track_history"}
    median_width = median_number(widths)
    median_height = median_number(heights)
    median_area = median_number(areas)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = box_area(bbox)
    width_ratio = width / max(1.0, median_width)
    height_ratio = height / max(1.0, median_height)
    area_ratio = area / max(1.0, median_area)
    needs_adjustment = width_ratio > 1.8 or height_ratio > 1.8 or area_ratio > 2.75
    return {
        "area_ratio_to_track_median": round(area_ratio, 6),
        "height_ratio_to_track_median": round(height_ratio, 6),
        "needs_adjustment": needs_adjustment,
        "scale_check_reason": "track_scale_outlier"
        if needs_adjustment
        else "within_track_scale_history",
        "track_median_area": round(median_area, 6),
        "track_median_height": round(median_height, 6),
        "track_median_width": round(median_width, 6),
        "width_ratio_to_track_median": round(width_ratio, 6),
    }


def median_number(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def nearest_neighbor_motion_check(
    bbox: list[int],
    *,
    parent_frame: int,
    by_frame: dict[int, dict[str, Any]],
    max_center_jump_norm: float,
) -> dict[str, Any]:
    neighbor_frames = sorted(frame for frame in by_frame if frame != parent_frame)
    if not neighbor_frames:
        return {"status": "passed", "reason": "single_observation_track"}
    nearest_frame = sorted(neighbor_frames, key=lambda frame: (abs(frame - parent_frame), frame))[0]
    frame_gap = abs(nearest_frame - parent_frame)
    if frame_gap != 1:
        return {
            "nearest_frame_gap": frame_gap,
            "status": "passed",
            "reason": "nearest_neighbor_not_adjacent",
        }
    neighbor_bbox = by_frame[nearest_frame].get("bbox_xyxy")
    if not isinstance(neighbor_bbox, list) or len(neighbor_bbox) != 4:
        return {
            "nearest_frame_gap": frame_gap,
            "status": "passed",
            "reason": "nearest_neighbor_missing_bbox",
        }
    distance = math.sqrt(squared_distance(center_xy(bbox), center_xy(neighbor_bbox)))
    normalizer = max(
        1,
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        int(neighbor_bbox[2]) - int(neighbor_bbox[0]),
        int(neighbor_bbox[3]) - int(neighbor_bbox[1]),
    )
    center_jump_norm = distance / normalizer
    if center_jump_norm > max_center_jump_norm:
        return {
            "center_jump_norm": round(center_jump_norm, 6),
            "nearest_frame": nearest_frame,
            "nearest_frame_gap": frame_gap,
            "reason": "adjacent_frame_center_jump_outside_source_bounds",
            "status": "rejected",
        }
    return {
        "center_jump_norm": round(center_jump_norm, 6),
        "nearest_frame": nearest_frame,
        "nearest_frame_gap": frame_gap,
        "status": "passed",
    }


def discarded_candidate_audit(
    bbox: list[Any],
    *,
    source: dict[str, Any],
    track_uid: str,
    record: dict[str, Any],
    rejection_reason: str,
    qa_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "bbox_xyxy": bbox if isinstance(bbox, list) else None,
        "evidence_source_path": source.get("path"),
        "proposal_qa_result": qa_result or {"status": "rejected", "reason": rejection_reason},
        "rejection_reason": rejection_reason,
        "source_detection_ids": list(record.get("source_candidate_ids") or []),
        "source_track_provenance": record.get("provenance"),
        "source_track_status": record.get("status"),
        "source_track_uid": track_uid,
    }


def public_evidence_source_audit(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "exists": source.get("exists"),
        "manifest_path": source.get("manifest_path"),
        "manifest_sha256": source.get("manifest_sha256"),
        "path": source.get("path"),
        "reason": source.get("reason"),
        "sha256": source.get("sha256"),
        "source_sha256": source.get("manifest", {}).get("source_sha256"),
        "source_video": source.get("manifest", {}).get("source_video"),
        "target_benchmark": source.get("target_benchmark"),
        "usable": source.get("usable"),
    }


def proposal_candidate(
    *,
    bbox: list[int],
    source: str,
    anchor_frame_id: str,
    anchor_parent_frame: int | None = None,
) -> dict[str, Any]:
    return {
        "anchor_frame_id": anchor_frame_id,
        "anchor_parent_frame": anchor_parent_frame,
        "bbox_xyxy": bbox,
        "model": None if source == PROPOSAL_HUMAN_SEEDED else source,
        "source": source,
    }


def proposal_priority(source: str) -> int:
    priorities = {
        PROPOSAL_AUTOMATIC_TRACKER: 0,
        PROPOSAL_DIRECT_DETECTOR: 1,
        PROPOSAL_BOUNDED_INTERPOLATION: 2,
        PROPOSAL_HUMAN_MOTION: 3,
        PROPOSAL_PRIVACY_ENVELOPE: 4,
        PROPOSAL_HISTORICAL_DETECTOR: 50,
        PROPOSAL_INTERPOLATED: 51,
        PROPOSAL_HUMAN_SEEDED: 99,
    }
    return priorities.get(source, 99)


def nearest_anchor(
    anchors: list[tuple[int, list[int], str]],
    parent_frame: int,
) -> tuple[int, list[int], str] | None:
    if not anchors:
        return None
    return sorted(anchors, key=lambda item: (abs(item[0] - parent_frame), item[0]))[0]


def interpolated_draft(
    anchors: list[tuple[int, list[int], str]],
    parent_frame: int,
) -> tuple[list[int], str, str] | None:
    before = [anchor for anchor in anchors if anchor[0] <= parent_frame]
    after = [anchor for anchor in anchors if anchor[0] >= parent_frame]
    if not before or not after:
        return None
    left = sorted(before, key=lambda item: item[0])[-1]
    right = sorted(after, key=lambda item: item[0])[0]
    if left[0] == right[0]:
        return list(left[1]), left[2], right[2]
    weight = (parent_frame - left[0]) / (right[0] - left[0])
    bbox = [
        int(round(left[1][index] + (right[1][index] - left[1][index]) * weight))
        for index in range(4)
    ]
    return bbox, left[2], right[2]


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
        asset_keys = ["crop", "context"]
        if "full_frame" in frame:
            asset_keys.append("full_frame")
        for asset_key in asset_keys:
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
            if asset_key == "full_frame":
                if asset.get("width") != frame.get("frame_width") or asset.get(
                    "height"
                ) != frame.get("frame_height"):
                    errors.append(f"{frame_id} full_frame dimensions mismatch")
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
        errors.append(
            f"expected {expected_total} total mandatory frames, got {len(seen_frame_ids)}"
        )
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
    if normalized_progress_for_compare(progress, computed_progress) != computed_progress:
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


def normalized_progress_for_compare(
    progress: dict[str, Any],
    computed_progress: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(progress)
    normalized["total"] = normalized_progress_row_for_compare(
        progress.get("total"),
        computed_progress.get("total"),
    )
    progress_by_benchmark = progress.get("by_benchmark") or {}
    computed_by_benchmark = computed_progress.get("by_benchmark") or {}
    if isinstance(progress_by_benchmark, dict) and isinstance(computed_by_benchmark, dict):
        normalized_by_benchmark = {}
        for benchmark_id, row in progress_by_benchmark.items():
            normalized_by_benchmark[benchmark_id] = normalized_progress_row_for_compare(
                row,
                computed_by_benchmark.get(benchmark_id),
            )
        normalized["by_benchmark"] = normalized_by_benchmark
    return normalized


def normalized_progress_row_for_compare(row: Any, computed_row: Any) -> Any:
    if not isinstance(row, dict) or not isinstance(computed_row, dict):
        return row
    normalized = dict(row)
    for key in OPTIONAL_PROGRESS_ROW_KEYS:
        if key not in normalized and key in computed_row:
            normalized[key] = computed_row[key]
    return normalized


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
        raise ValueError(
            "target frames cannot be marked invisible or detector_ignore in this phase"
        )

    now_utc = now_utc or media.utc_timestamp()
    review_note = str(payload.get("review_note", "")).strip()
    conflicts = normalize_conflicts(existing.get("conflicts", []))
    proposal_bbox = existing.get("proposal_bbox") or existing.get("draft_bbox")
    proposal_available = bool(existing.get("proposal_available", existing.get("draft_available")))
    proposal_shown = bool(
        existing.get("proposal_shown")
        or existing.get("draft_shown")
        or payload.get("proposal_shown")
        or payload.get("draft_shown")
        or payload.get("accept_draft")
    )
    unresolved_temporal = [
        conflict
        for conflict in conflicts
        if conflict["type"] == "temporal_gt_conflict" and not conflict["resolved"]
    ]
    temporal_transition = "none"
    accepted_without_geometry_change = False
    human_adjusted_geometry = False
    if status == STATUS_REVIEWED_BOX:
        human_bbox = validate_bbox_xyxy(payload.get("human_bbox"), width=width, height=height)
        human_reviewed = True
        if proposal_available and proposal_bbox is not None:
            proposal_shown = True
            accepted_without_geometry_change = human_bbox == proposal_bbox
            human_adjusted_geometry = human_bbox != proposal_bbox
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
        "accepted_without_geometry_change": accepted_without_geometry_change,
        "annotation_status": status,
        "conflicts": conflicts,
        "draft_shown": proposal_shown,
        "human_bbox": human_bbox,
        "human_adjusted_geometry": human_adjusted_geometry,
        "human_reviewed": human_reviewed,
        "proposal_shown": proposal_shown,
        "review_note": review_note,
        "reviewed_at": now_utc if human_reviewed else None,
        "secondary_instances": secondary_instances,
    }
    state_errors: list[str] = []
    _validate_state_record(
        updated, width=width, height=height, errors=state_errors, context=frame_id
    )
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
            "accepted_without_geometry_change": accepted_without_geometry_change,
            "benchmark_id": frame.get("benchmark_id"),
            "dense_frame": frame.get("dense_frame"),
            "event": "accept_draft" if payload.get("accept_draft") else "save_annotation",
            "frame_id": frame_id,
            "human_bbox": human_bbox,
            "human_adjusted_geometry": human_adjusted_geometry,
            "new_state_revision": new_revision,
            "new_status": status,
            "parent_frame": frame.get("parent_frame"),
            "previous_state_revision": previous_revision,
            "previous_status": existing.get("annotation_status"),
            "proposal_available": proposal_available,
            "proposal_bbox": proposal_bbox,
            "proposal_quality": existing.get("proposal_quality"),
            "proposal_model": existing.get("proposal_model") or existing.get("draft_model"),
            "proposal_shown": proposal_shown,
            "proposal_source": existing.get("proposal_source") or existing.get("draft_source"),
            "recorded_at_utc": now_utc,
            "review_note": review_note,
            "schema_version": "plate_annotation_change.v1",
            "semantic_state_hash": stable_json_sha256(updated),
            "target_id": frame.get("target_id"),
            "temporal_gt_conflict_transition": temporal_transition,
        },
    )
    return {"frame": updated, "progress": progress, "state_revision": new_revision}


def accept_draft_annotation(
    package_root: Path,
    frame_id: str,
    *,
    expected_state_revision: int | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    state = read_json(package_root / "annotations.draft.json")
    record = state.get("frames", {}).get(frame_id)
    if not isinstance(record, dict):
        raise ValueError(f"unknown frame_id: {frame_id}")
    proposal_bbox = record.get("proposal_bbox") or record.get("draft_bbox")
    if not record.get("proposal_available", record.get("draft_available")) or proposal_bbox is None:
        raise ValueError("no proposal is available to accept")
    return save_frame_annotation(
        package_root,
        frame_id,
        {
            "accept_draft": True,
            "annotation_status": STATUS_REVIEWED_BOX,
            "draft_shown": True,
            "human_bbox": proposal_bbox,
            "proposal_shown": True,
            "review_note": record.get("review_note", ""),
        },
        expected_state_revision=expected_state_revision,
        now_utc=now_utc,
    )


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
    if record.get("proposal_available", record.get("draft_available")):
        record["draft_shown"] = True
        record["proposal_shown"] = True
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

    package_file_sha256 = {
        name: media.sha256_file(package_root / name)
        for name in (
            "annotations.draft.json",
            "change_log.jsonl",
            "package_identity.json",
            "package_manifest.json",
            "progress.json",
            "review_data.json",
        )
    }
    assisted_migration = manifest.get("assisted_migration") or {}
    source_package_lineage = {
        "source_package_hashes": assisted_migration.get("source_package_hashes", {}),
        "source_package_id": assisted_migration.get("source_package_id"),
    }

    records = [
        {
            "annotation_provenance": {
                "accepted_without_geometry_change": state["frames"][frame["frame_id"]].get(
                    "accepted_without_geometry_change", False
                ),
                "annotation_status": state["frames"][frame["frame_id"]]["annotation_status"],
                "draft_available": state["frames"][frame["frame_id"]]["draft_available"],
                "draft_model": state["frames"][frame["frame_id"]]["draft_model"],
                "draft_shown": state["frames"][frame["frame_id"]]["draft_shown"],
                "draft_source": state["frames"][frame["frame_id"]]["draft_source"],
                "human_adjusted_geometry": state["frames"][frame["frame_id"]].get(
                    "human_adjusted_geometry", False
                ),
                "human_reviewed": state["frames"][frame["frame_id"]]["human_reviewed"],
                "migration_source_package_id": state["frames"][frame["frame_id"]].get(
                    "migration_source_package_id"
                ),
                "proposal_available": state["frames"][frame["frame_id"]].get(
                    "proposal_available", state["frames"][frame["frame_id"]]["draft_available"]
                ),
                "proposal_bbox": state["frames"][frame["frame_id"]].get("proposal_bbox"),
                "proposal_model": state["frames"][frame["frame_id"]].get("proposal_model"),
                "proposal_quality": state["frames"][frame["frame_id"]].get("proposal_quality"),
                "proposal_shown": state["frames"][frame["frame_id"]].get(
                    "proposal_shown", state["frames"][frame["frame_id"]]["draft_shown"]
                ),
                "proposal_source": state["frames"][frame["frame_id"]].get("proposal_source"),
                "review_completion_bucket": review_completion_bucket(
                    state["frames"][frame["frame_id"]]
                ),
                "reviewed_at": state["frames"][frame["frame_id"]]["reviewed_at"],
            },
            "action": frame["action"],
            "benchmark_id": frame["benchmark_id"],
            "conflicts": state["frames"][frame["frame_id"]]["conflicts"],
            "dense_frame": frame["dense_frame"],
            "frame_id": frame["frame_id"],
            "human_bbox": state["frames"][frame["frame_id"]]["human_bbox"],
            "object_type": frame["object_type"],
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
                "action": target["action"],
                "evaluation_video": target["evaluation_video"],
                "frozen_parent_frame_interval": target["frozen_parent_frame_interval"],
                "object_type": target["object_type"],
                "source_interval": target["source_interval"],
                "source_video_label": target["source_video_label"],
                "target_id": target["target_id"],
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
        "annotation_package_schema_version": manifest["schema_version"],
        "package_id": manifest["package_id"],
        "package_file_sha256": package_file_sha256,
        "package_manifest_sha256": media.sha256_file(package_root / "package_manifest.json"),
        "progress": progress,
        "reviewer_completion_state": progress["reviewer_completion_state"],
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source_package_lineage": source_package_lineage,
        "state_revision": state["state_revision"],
    }
    payload["semantic_content_hash"] = semantic_content_hash(payload)
    payload["exported_at_utc"] = now_utc or media.utc_timestamp()
    output_path = output_path or (package_root / default_name)
    atomic_write_json(output_path, payload)
    return payload


def review_completion_bucket(record: dict[str, Any]) -> str:
    proposal_available = bool(
        record.get("proposal_available", record.get("draft_available", False))
    )
    if record.get("migration_source_package_id") and not proposal_available:
        return "migrated_manual_review"
    if proposal_available and record.get("accepted_without_geometry_change"):
        return "proposal_accepted_unchanged"
    if proposal_available and record.get("human_adjusted_geometry"):
        return "proposal_human_adjusted"
    if not proposal_available:
        return "manual_no_proposal"
    return "unknown_review_completion_bucket"


def compute_progress(review_data: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    frame_records = review_data.get("frames") or []
    state_frames = state.get("frames") or {}
    by_benchmark: dict[str, dict[str, Any]] = {}

    def progress_row(*, mandatory: int = 0) -> dict[str, Any]:
        return {
            "additional_localizable_plate": 0,
            "ambiguous_review_required": 0,
            "draft_shown": 0,
            "human_reviewed": 0,
            "human_reviewed_total": 0,
            "ignore_small": 0,
            "mandatory": mandatory,
            "needs_recheck": 0,
            "no_proposal_remaining": 0,
            "proposal_accepted_unchanged": 0,
            "proposal_available_total": 0,
            "proposal_human_adjusted": 0,
            "proposal_source_breakdown": {},
            "reviewed_box": 0,
            "temporal_gt_conflict": 0,
            "temporal_gt_conflicts": 0,
            "unreviewed": 0,
            "unresolved_ambiguous_secondary": 0,
            "unresolved_temporal_gt_conflict": 0,
        }

    total = progress_row(mandatory=len(frame_records))
    for frame in frame_records:
        benchmark = frame["benchmark_id"]
        row = by_benchmark.setdefault(benchmark, progress_row())
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
            row["temporal_gt_conflicts"] += 1
            row["unresolved_temporal_gt_conflict"] += 1
            total["temporal_gt_conflict"] += 1
            total["temporal_gt_conflicts"] += 1
            total["unresolved_temporal_gt_conflict"] += 1
        else:
            row["unreviewed"] += 1
            total["unreviewed"] += 1

        if record.get("human_reviewed"):
            row["human_reviewed"] += 1
            row["human_reviewed_total"] += 1
            total["human_reviewed"] += 1
            total["human_reviewed_total"] += 1

        proposal_available = bool(record.get("proposal_available", record.get("draft_available")))
        if proposal_available:
            row["proposal_available_total"] += 1
            total["proposal_available_total"] += 1
            source = str(record.get("proposal_source") or record.get("draft_source") or "unknown")
            row_sources = row["proposal_source_breakdown"]
            total_sources = total["proposal_source_breakdown"]
            row_sources[source] = row_sources.get(source, 0) + 1
            total_sources[source] = total_sources.get(source, 0) + 1
        elif status != STATUS_REVIEWED_BOX:
            row["no_proposal_remaining"] += 1
            total["no_proposal_remaining"] += 1

        if record.get("accepted_without_geometry_change"):
            row["proposal_accepted_unchanged"] += 1
            total["proposal_accepted_unchanged"] += 1
        if record.get("human_adjusted_geometry"):
            row["proposal_human_adjusted"] += 1
            total["proposal_human_adjusted"] += 1

        if record.get("draft_shown"):
            row["draft_shown"] += 1
            total["draft_shown"] += 1
        if status != STATUS_TEMPORAL_GT_CONFLICT:
            for conflict in record.get("conflicts") or []:
                if conflict.get("type") != "temporal_gt_conflict" or conflict.get("resolved"):
                    continue
                row["temporal_gt_conflict"] += 1
                row["temporal_gt_conflicts"] += 1
                row["unresolved_temporal_gt_conflict"] += 1
                total["temporal_gt_conflict"] += 1
                total["temporal_gt_conflicts"] += 1
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
        [
            original_bbox[0] - ox,
            original_bbox[1] - oy,
            original_bbox[2] - ox,
            original_bbox[3] - oy,
        ],
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
    annotation_mode: str = ANNOTATION_MODE_MANUAL,
    include_drafts: bool,
    draft_path: Path,
    draft_mapping_audit_path: Path,
    repo_root: Path,
) -> None:
    for child in ("assets", "frames/day_plate", "frames/night_selective"):
        (package_root / child).mkdir(parents=True, exist_ok=True)
    git_commit = git_commit_for(repo_root)
    media_records = {
        target.benchmark_id: probe_and_validate_target(target, repo_root) for target in targets
    }
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
        annotation_mode=annotation_mode,
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
    full_rel = (
        Path("frames")
        / target.benchmark_id
        / FRAME_ASSET_NAME_TEMPLATE.format(
            benchmark=target.benchmark_id,
            extension="jpg",
            kind="full",
            parent_frame=parent_frame,
        )
    )
    crop_path = package_root / crop_rel
    context_path = package_root / context_rel
    full_path = package_root / full_rel
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(crop_path), crop, [cv.IMWRITE_PNG_COMPRESSION, 3]):
        raise ValueError(f"could not write crop frame: {crop_path}")
    context = context_image(frame)
    if not cv.imwrite(str(context_path), context, [cv.IMWRITE_JPEG_QUALITY, 88]):
        raise ValueError(f"could not write context frame: {context_path}")
    if not cv.imwrite(str(full_path), frame, [cv.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"could not write full frame: {full_path}")
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
        "full_frame": {
            "height": height,
            "path": full_rel.as_posix(),
            "sha256": media.sha256_file(full_path),
            "width": width,
        },
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
    annotation_mode: str = ANNOTATION_MODE_MANUAL,
) -> dict[str, Any]:
    return {
        "annotation_mode": annotation_mode,
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
    mapping_payload = (
        read_json(draft_mapping_audit_path) if draft_mapping_audit_path.exists() else {}
    )
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
        proposal_source = PROPOSAL_HISTORICAL_DETECTOR if main_candidate else None
        state_frames[frame["frame_id"]] = {
            "accepted_without_geometry_change": False,
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
            "human_adjusted_geometry": False,
            "human_reviewed": False,
            "parent_frame": frame["parent_frame"],
            "proposal_available": main_candidate is not None,
            "proposal_bbox": main_candidate.get("bbox_xyxy") if main_candidate else None,
            "proposal_candidates": draft_entry.get("candidates", []),
            "proposal_model": ",".join(draft_import.get("source_finalists") or [])
            if main_candidate
            else None,
            "proposal_quality": None,
            "proposal_shown": False,
            "proposal_source": proposal_source,
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
                raise ValueError(
                    "temporal_gt_conflict status requires an unresolved conflict record"
                )
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
        if record.get("proposal_bbox") is not None:
            validate_bbox_xyxy(record.get("proposal_bbox"), width=width, height=height)
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
        if (
            frame["dense_frame"]
            != frame["parent_frame"]
            - manifest["targets"][frame["benchmark_id"]]["frozen_parent_frame_interval"][0]
        ):
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

## Full-Frame Review

Review from the full source frame. The crop and QA ROI overlay are search aids only; saved boxes are
always original-frame coordinates. It is valid to draw outside the QA ROI when the physical plate
panel is visible there.

## Draft Policy

Imported proposals remain separate from human boxes. In assisted packages, a proposal may be shown
automatically for unreviewed frames and accepted with no geometry change or adjusted before saving.
Showing a proposal records `draft_shown=true`/`proposal_shown=true`; proposals never increase
reviewed counts until the human saves or accepts a box.

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

Package ID: `{manifest["package_id"]}`

Annotation mode: `{manifest.get("annotation_mode", ANNOTATION_MODE_MANUAL)}`

Status: `{progress["reviewer_completion_state"]["state"]}`

## Targets

{target_lines}

## Progress

- Reviewed boxes: {progress["total"]["reviewed_box"]} / {progress["total"]["mandatory"]}
- Needs recheck: {progress["total"]["needs_recheck"]}
- Temporal conflicts: {progress["total"]["unresolved_temporal_gt_conflict"]}
- Ambiguous secondary records: {progress["total"]["unresolved_ambiguous_secondary"]}
- Proposal frames: {progress["total"].get("proposal_available_total", 0)}
- Accepted unchanged: {progress["total"].get("proposal_accepted_unchanged", 0)}
- Human-adjusted proposals: {progress["total"].get("proposal_human_adjusted", 0)}
- Frames without proposal: {progress["total"].get("no_proposal_remaining", 0)}

## Drafts

- Imported: {draft_import["imported"]}
- Target-frame proposals: {draft_import["imported_target_frame_count"]}
- Exclusions: {json.dumps(draft_import["excluded_by_reason"], sort_keys=True)}

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
        <button id="acceptDraft" title="Accept the visible proposal and advance">Accept Draft + Next</button>
        <button id="save" title="Save current frame">Save</button>
        <button id="saveNext" title="Save and advance">Save+Next</button>
        <button id="reset" title="Clear box">Reset</button>
        <button id="fitFrame" title="Fit the whole source frame">Fit Frame</button>
        <button id="fitTarget" title="Fit the active box or proposal">Fit Draft</button>
        <label>Proposal <input id="showDraft" type="checkbox"></label>
        <label>QA ROI <input id="showRoi" type="checkbox" checked></label>
        <label>Zoom <input id="zoom" type="range" min="0.2" max="8" step="0.05" value="1"></label>
      </section>
      <section class="meta">
        <div id="frameTitle"></div>
        <div id="progress"></div>
        <div id="saveState"></div>
        <div id="proposalBanner" class="proposalBanner"></div>
        <div class="shortcuts">A/Enter accept, S save+next, arrows move, F fit frame, D fit draft, R recheck</div>
      </section>
      <section class="workspace">
        <div class="canvasWrap">
          <canvas id="cropCanvas"></canvas>
        </div>
        <aside class="contextPane">
          <img id="context" alt="Frame context">
          <pre id="boxReadout"></pre>
        </aside>
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
  background: #eef1f5;
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
.meta {
  justify-content: space-between;
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
button:disabled {
  color: #7c8797;
  cursor: not-allowed;
}
button:hover {
  background: #eef4ff;
}
.workspace {
  display: grid;
  gap: 12px;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 18vw);
}
#context {
  background: #202734;
  display: block;
  max-width: 100%;
}
.canvasWrap {
  background: #111820;
  border: 1px solid #0e141b;
  border-radius: 8px;
  height: calc(100vh - 250px);
  min-height: 440px;
  overflow: hidden;
}
#cropCanvas {
  display: block;
  image-rendering: auto;
  touch-action: none;
}
#frameTitle {
  font-weight: 700;
}
#saveState {
  color: #42526e;
}
.shortcuts {
  color: #556274;
  font-size: 13px;
}
.proposalBanner {
  color: #8a5200;
  font-weight: 700;
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
.contextPane {
  display: grid;
  gap: 8px;
  grid-template-rows: auto 1fr;
  min-width: 0;
}
#boxReadout {
  min-height: 160px;
}
@media (max-width: 840px) {
  .workspace {
    grid-template-columns: 1fr;
  }
  .canvasWrap {
    height: 54vh;
    min-height: 320px;
  }
  .contextPane {
    grid-template-columns: minmax(120px, 36vw) 1fr;
    grid-template-rows: auto;
  }
}
"""


def app_js() -> str:
    return """const STATUS_REVIEWED = "reviewed_box";
const STATUS_RECHECK = "needs_recheck";
const STATUS_UNREVIEWED = "unreviewed";

const state = {
  annotations: null,
  box: null,
  drag: null,
  frames: [],
  image: new Image(),
  index: 0,
  secondary: [],
  showProposal: false,
  viewport: { offsetX: 0, offsetY: 0, scale: 1 },
};

const el = (id) => document.getElementById(id);
const canvas = el("cropCanvas");
const ctx = canvas.getContext("2d");

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  if (!response.ok) {
    let message = text;
    try {
      message = JSON.parse(text).error || text;
    } catch {}
    const error = new Error(message || response.statusText);
    error.status = response.status;
    throw error;
  }
  return text ? JSON.parse(text) : {};
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
  el("acceptDraft").onclick = () => acceptDraft(true);
  el("save").onclick = () => save(false);
  el("saveNext").onclick = () => save(true);
  el("reset").onclick = () => {
    state.box = null;
    el("status").value = STATUS_UNREVIEWED;
    draw();
  };
  el("fitFrame").onclick = fitFrame;
  el("fitTarget").onclick = fitTarget;
  el("showRoi").onchange = draw;
  el("jump").onchange = () => {
    const nextIndex = Math.max(0, Math.min(state.frames.length - 1, Number(el("jump").value) - 1));
    renderFrame(nextIndex);
  };
  el("zoom").oninput = () => zoomTo(Number(el("zoom").value));
  el("showDraft").onchange = async () => {
    state.showProposal = el("showDraft").checked;
    if (state.showProposal) await markProposalShown();
    updateButtons();
    draw();
  };
  el("addSecondary").onclick = addSecondary;
  canvas.addEventListener("mousedown", pointerDown);
  canvas.addEventListener("mousemove", pointerMove);
  canvas.addEventListener("mouseup", pointerUp);
  canvas.addEventListener("mouseleave", pointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  window.addEventListener("resize", () => {
    resizeCanvas();
    draw();
  });
  window.addEventListener("keydown", onKeyDown);
}

function currentFrame() {
  return state.frames[state.index];
}

function frameState() {
  return state.annotations.frames[currentFrame().frame_id];
}

function proposalBox() {
  const annotation = frameState();
  return annotation.proposal_bbox || annotation.draft_bbox || null;
}

function hasProposal() {
  const annotation = frameState();
  return Boolean(annotation.proposal_available ?? annotation.draft_available) && proposalBox();
}

function proposalNeedsAdjustment() {
  const quality = frameState().proposal_quality || "";
  return quality === "needs_human_adjustment" || quality === "needs_manual_start";
}

function renderFrame(index) {
  state.index = index;
  const frame = currentFrame();
  const annotation = frameState();
  state.box = annotation.human_bbox ? annotation.human_bbox.slice() : null;
  state.secondary = (annotation.secondary_instances || []).map((item) => ({ ...item }));
  state.showProposal = hasProposal() && annotation.annotation_status !== STATUS_REVIEWED;
  el("context").src = frame.context.path;
  el("status").value = annotation.annotation_status;
  el("note").value = annotation.review_note || "";
  el("showDraft").checked = state.showProposal;
  el("showDraft").disabled = !hasProposal();
  el("jump").value = index + 1;
  el("frameTitle").textContent = `${frame.benchmark_id} ${frame.target_id} parent ${frame.parent_frame} dense ${frame.dense_frame} time ${frame.timestamp_seconds}s`;
  el("secondaryList").textContent = JSON.stringify(state.secondary, null, 2);
  const fullFrame = frame.full_frame || frame.context;
  state.image.onload = () => {
    resizeCanvas();
    if (state.showProposal && proposalBox()) {
      fitTarget();
    } else {
      fitFrame();
    }
  };
  state.image.src = fullFrame.path;
  updateProgress();
  updateButtons();
  updateReadout();
}

function updateProgress() {
  const frames = Object.values(state.annotations.frames);
  const reviewed = frames.filter((item) => item.annotation_status === STATUS_REVIEWED).length;
  const accepted = frames.filter((item) => item.accepted_without_geometry_change).length;
  const adjusted = frames.filter((item) => item.human_adjusted_geometry).length;
  el("progress").textContent = `${reviewed} / ${frames.length} reviewed; ${accepted} accepted; ${adjusted} adjusted`;
}

function updateButtons() {
  const available = hasProposal();
  el("acceptDraft").disabled = !available || frameState().annotation_status === STATUS_REVIEWED;
  el("proposalBanner").textContent =
    available && frameState().annotation_status !== STATUS_REVIEWED
      ? (proposalNeedsAdjustment() ? "PROPOSAL NEEDS ADJUSTMENT" : "PROPOSAL - NOT YET REVIEWED")
      : "";
  el("fitTarget").textContent = state.box ? "Fit Box" : "Fit Draft";
  el("fitTarget").disabled = !state.box && !available;
}

function resizeCanvas() {
  const wrap = canvas.parentElement;
  const rect = wrap.getBoundingClientRect();
  canvas.width = Math.max(320, Math.floor(rect.width));
  canvas.height = Math.max(240, Math.floor(rect.height));
}

function fitFrame() {
  const frame = currentFrame();
  const scale = Math.min(canvas.width / frame.frame_width, canvas.height / frame.frame_height);
  state.viewport.scale = clamp(scale, 0.05, 16);
  state.viewport.offsetX = (canvas.width - frame.frame_width * state.viewport.scale) / 2;
  state.viewport.offsetY = (canvas.height - frame.frame_height * state.viewport.scale) / 2;
  syncZoom();
  draw();
}

function fitTarget() {
  const box = state.box || proposalBox();
  if (!box) {
    fitFrame();
    return;
  }
  const frame = currentFrame();
  const boxWidth = Math.max(1, box[2] - box[0]);
  const boxHeight = Math.max(1, box[3] - box[1]);
  const pad = Math.max(36, boxWidth * 2.2, boxHeight * 2.2);
  const x1 = clamp(box[0] - pad, 0, frame.frame_width);
  const y1 = clamp(box[1] - pad, 0, frame.frame_height);
  const x2 = clamp(box[2] + pad, 0, frame.frame_width);
  const y2 = clamp(box[3] + pad, 0, frame.frame_height);
  const scale = Math.min(canvas.width / Math.max(1, x2 - x1), canvas.height / Math.max(1, y2 - y1));
  state.viewport.scale = clamp(scale, 0.05, 16);
  state.viewport.offsetX = canvas.width / 2 - ((x1 + x2) / 2) * state.viewport.scale;
  state.viewport.offsetY = canvas.height / 2 - ((y1 + y2) / 2) * state.viewport.scale;
  syncZoom();
  draw();
}

function zoomTo(nextScale) {
  const center = { x: canvas.width / 2, y: canvas.height / 2 };
  const original = screenToOriginalFloat(center.x, center.y);
  state.viewport.scale = clamp(nextScale, 0.05, 16);
  state.viewport.offsetX = center.x - original.x * state.viewport.scale;
  state.viewport.offsetY = center.y - original.y * state.viewport.scale;
  draw();
}

function syncZoom() {
  el("zoom").value = String(clamp(state.viewport.scale, Number(el("zoom").min), Number(el("zoom").max)));
}

function onWheel(event) {
  event.preventDefault();
  const point = pointerScreen(event);
  const original = screenToOriginalFloat(point.x, point.y);
  const factor = event.deltaY < 0 ? 1.12 : 0.89;
  state.viewport.scale = clamp(state.viewport.scale * factor, 0.05, 16);
  state.viewport.offsetX = point.x - original.x * state.viewport.scale;
  state.viewport.offsetY = point.y - original.y * state.viewport.scale;
  syncZoom();
  draw();
}

function draw() {
  if (!state.image.complete || !state.image.naturalWidth) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111820";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const frame = currentFrame();
  ctx.drawImage(
    state.image,
    state.viewport.offsetX,
    state.viewport.offsetY,
    frame.frame_width * state.viewport.scale,
    frame.frame_height * state.viewport.scale,
  );
  if (el("showRoi").checked) drawOriginalBox(frame.crop.origin_xy.concat([frame.crop.origin_xy[0] + frame.crop.width, frame.crop.origin_xy[1] + frame.crop.height]), "rgba(255, 255, 255, 0.6)", [5, 6], 1);
  if (state.showProposal && proposalBox()) drawOriginalBox(proposalBox(), "rgba(255, 184, 64, 0.98)", [9, 6], 3);
  for (const item of state.secondary) {
    if (item.bbox_xyxy) drawOriginalBox(item.bbox_xyxy, "rgba(0, 190, 140, 0.98)", [2, 5], 2);
  }
  if (state.box) {
    drawOriginalBox(state.box, "rgba(64, 145, 255, 1)", [], 3);
    drawHandles(state.box);
  }
  updateReadout();
}

function drawOriginalBox(box, color, dash, lineWidth) {
  const left = originalToScreen(box[0], box[1]);
  const right = originalToScreen(box[2], box[3]);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash);
  ctx.strokeRect(left.x, left.y, right.x - left.x, right.y - left.y);
  ctx.restore();
}

function drawHandles(box) {
  const points = [
    originalToScreen(box[0], box[1]),
    originalToScreen(box[2], box[1]),
    originalToScreen(box[0], box[3]),
    originalToScreen(box[2], box[3]),
  ];
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#286edb";
  for (const point of points) {
    ctx.beginPath();
    ctx.rect(point.x - 4, point.y - 4, 8, 8);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function pointerScreen(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function screenToOriginalFloat(x, y) {
  return {
    x: (x - state.viewport.offsetX) / state.viewport.scale,
    y: (y - state.viewport.offsetY) / state.viewport.scale,
  };
}

function screenToOriginal(x, y) {
  const frame = currentFrame();
  const point = screenToOriginalFloat(x, y);
  return [
    Math.round(clamp(point.x, 0, frame.frame_width)),
    Math.round(clamp(point.y, 0, frame.frame_height)),
  ];
}

function originalToScreen(x, y) {
  return {
    x: x * state.viewport.scale + state.viewport.offsetX,
    y: y * state.viewport.scale + state.viewport.offsetY,
  };
}

function pointerDown(event) {
  const screen = pointerScreen(event);
  if (event.button === 1 || event.button === 2 || event.shiftKey || event.altKey) {
    state.drag = {
      mode: "pan",
      offset: { ...state.viewport },
      screenStart: screen,
    };
    return;
  }
  const [x, y] = screenToOriginal(screen.x, screen.y);
  if (!state.box && state.showProposal && proposalBox() && hitTestBox(proposalBox(), x, y)) {
    state.box = proposalBox().slice();
    el("status").value = STATUS_REVIEWED;
  }
  state.drag = {
    mode: dragModeForPoint(x, y),
    original: state.box ? state.box.slice() : null,
    start: [x, y],
  };
}

function pointerMove(event) {
  if (!state.drag) return;
  const screen = pointerScreen(event);
  if (state.drag.mode === "pan") {
    state.viewport.offsetX = state.drag.offset.offsetX + screen.x - state.drag.screenStart.x;
    state.viewport.offsetY = state.drag.offset.offsetY + screen.y - state.drag.screenStart.y;
    draw();
    return;
  }
  const [x, y] = screenToOriginal(screen.x, screen.y);
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
  if (state.box) el("status").value = STATUS_REVIEWED;
  updateButtons();
  draw();
}

function pointerUp() {
  if (state.box && (state.box[2] - state.box[0] < 2 || state.box[3] - state.box[1] < 2)) {
    state.box = null;
  }
  state.drag = null;
  draw();
}

function normalizeBox(box) {
  const frame = currentFrame();
  const x1 = clamp(Math.min(box[0], box[2]), 0, frame.frame_width);
  const y1 = clamp(Math.min(box[1], box[3]), 0, frame.frame_height);
  const x2 = clamp(Math.max(box[0], box[2]), 0, frame.frame_width);
  const y2 = clamp(Math.max(box[1], box[3]), 0, frame.frame_height);
  return [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
}

function clampBox(box) {
  const frame = currentFrame();
  const width = Math.max(1, box[2] - box[0]);
  const height = Math.max(1, box[3] - box[1]);
  const x1 = clamp(box[0], 0, frame.frame_width - width);
  const y1 = clamp(box[1], 0, frame.frame_height - height);
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
  const tolerance = Math.max(4, 9 / state.viewport.scale);
  const corners = [
    ["resize-nw", state.box[0], state.box[1]],
    ["resize-ne", state.box[2], state.box[1]],
    ["resize-sw", state.box[0], state.box[3]],
    ["resize-se", state.box[2], state.box[3]],
  ];
  for (const [mode, cx, cy] of corners) {
    if (Math.abs(x - cx) <= tolerance && Math.abs(y - cy) <= tolerance) return mode;
  }
  if (hitTestBox(state.box, x, y)) return "move";
  return "draw";
}

function hitTestBox(box, x, y) {
  return x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3];
}

function addSecondary() {
  const kind = el("secondaryKind").value;
  const note = el("secondaryNote").value;
  const item = { bbox_xyxy: state.box ? state.box.slice() : null, kind, note };
  state.secondary.push(item);
  el("secondaryList").textContent = JSON.stringify(state.secondary, null, 2);
}

async function acceptDraft(advance) {
  const frame = currentFrame();
  if (!hasProposal()) {
    el("saveState").textContent = "No proposal available";
    return;
  }
  try {
    const result = await jsonFetch("/api/annotations/accept-draft", {
      body: JSON.stringify({
        expected_state_revision: state.annotations.state_revision,
        frame_id: frame.frame_id,
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
    state.annotations.frames[frame.frame_id] = result.frame;
    state.annotations.state_revision = result.state_revision;
    el("saveState").textContent = "Accepted proposal";
    updateProgress();
    if (advance) {
      renderFrame(Math.min(state.frames.length - 1, state.index + 1));
    } else {
      renderFrame(state.index);
    }
  } catch (error) {
    showSaveError(error);
  }
}

async function markProposalShown() {
  const frame = currentFrame();
  const annotation = frameState();
  if (!hasProposal() || annotation.proposal_shown || annotation.draft_shown) return;
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

async function save(advance) {
  const frame = currentFrame();
  let status = el("status").value;
  if (status === STATUS_UNREVIEWED && state.box) status = STATUS_REVIEWED;
  const proposalShown = frameState().proposal_shown || frameState().draft_shown || state.showProposal || Boolean(hasProposal() && state.box);
  const payload = {
    annotation_status: status,
    draft_shown: proposalShown,
    human_bbox: state.box,
    proposal_shown: proposalShown,
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
    if (advance) {
      renderFrame(Math.min(state.frames.length - 1, state.index + 1));
    } else {
      renderFrame(state.index);
    }
  } catch (error) {
    showSaveError(error);
  }
}

function showSaveError(error) {
  const prefix = error.status === 409 ? "Stale save rejected" : "Save failed";
  el("saveState").textContent = `${prefix}: ${error.message}`;
}

function onKeyDown(event) {
  if (isTypingTarget(event.target)) return;
  const key = event.key.toLowerCase();
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    renderFrame(Math.max(0, state.index - 1));
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    renderFrame(Math.min(state.frames.length - 1, state.index + 1));
  } else if (key === "a" || event.key === "Enter") {
    event.preventDefault();
    acceptDraft(true);
  } else if (key === "s") {
    event.preventDefault();
    save(true);
  } else if (key === "f") {
    event.preventDefault();
    fitFrame();
  } else if (key === "d") {
    event.preventDefault();
    if (!state.showProposal && hasProposal()) {
      state.showProposal = true;
      el("showDraft").checked = true;
      markProposalShown();
    }
    fitTarget();
  } else if (key === "r") {
    event.preventDefault();
    el("status").value = STATUS_RECHECK;
    if (el("note").value.trim()) {
      save(true);
    } else {
      el("saveState").textContent = "Needs Recheck requires a note";
      el("note").focus();
    }
  } else if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    state.box = null;
    draw();
  }
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = target.tagName ? target.tagName.toLowerCase() : "";
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}

function updateReadout() {
  const annotation = frameState();
  const payload = {
    human_bbox: state.box,
    proposal_bbox: proposalBox(),
    proposal_quality: annotation.proposal_quality,
    proposal_source: annotation.proposal_source || annotation.draft_source,
    status: el("status").value,
  };
  el("boxReadout").textContent = JSON.stringify(payload, null, 2);
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
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
            if parsed.path == "/api/annotations/accept-draft":
                frame_id = str(payload.get("frame_id") or "")
                self._send_json(
                    accept_draft_annotation(
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
