from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from redactor import detection, media, plate_detection, plate_rendering, plate_tracking
from redactor.privacy_evidence import TemporalEvidenceConfiguration

SCHEMA_VERSION = "final_pipeline_package.v1"
AUDIT_SCHEMA_VERSION = "final_full_video_audit.v1"

FACE_DETECTOR_ID = detection.DETECTOR_BACKEND_ADAS
PLATE_DETECTOR_ID = plate_detection.DETECTOR_ID
PLATE_DETECTOR_CONFIGURATION_HASH = (
    "7d914438255905e2a249e5bbd30fd6e70eace1e91b57e205bd572d004ffb5a55"
)
PLATE_TRACKING_CONFIGURATION_HASH = (
    "27fc564737bf7bf3d2bc5389c4394fe991b5b9e90f9e76789ed0d8231e3dff9d"
)
PLATE_EVIDENCE_CONFIGURATION_HASH = (
    "9938297fdb54d3b3cc0817ec1112fd4ac94689ba93e3d8d55f458a7009535c2f"
)
PLATE_RENDER_CONFIGURATION_HASH = "508146780991e009b1d0683084236d5dce00e4547bd21e1ea7286c6f972ec02d"

AUDIT_ISSUE_TYPES = {
    "visible_face_leak",
    "visible_plate_leak",
    "partial_face_coverage",
    "partial_plate_coverage",
    "false_face_redaction",
    "false_plate_redaction",
    "wrong_identity_redaction",
    "tracking_jump",
    "cosmetic_only",
    "not_sensitive",
    "uncertain",
}
AUDIT_CATEGORIES = {"face", "plate"}
AUDIT_SEVERITIES = {"blocker", "high", "medium", "low", "cosmetic", "uncertain"}
AUDIT_DECISIONS = {
    "pending_review",
    "confirmed_issue",
    "not_an_issue",
    "cosmetic_only",
    "needs_correction",
    "uncertain",
}


@dataclass(frozen=True)
class SourceArtifactRecord:
    logical_clip_id: str
    category: str
    role: str
    source_path: str
    source_sha256: str
    frame_count: int | None
    fps: float | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    artifact_path: str | None = None
    artifact_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "category": self.category,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "height": self.height,
            "logical_clip_id": self.logical_clip_id,
            "role": self.role,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "width": self.width,
        }


def active_configuration_snapshot() -> dict[str, Any]:
    plate_detector_config = plate_detection.OIMPlateDetectorConfig()
    plate_tracking_config = plate_tracking.PlateTrackingConfig()
    plate_evidence_config = plate_tracking.default_evidence_config()
    plate_render_config = plate_rendering.default_config()
    return {
        "face": {
            "detector_id": FACE_DETECTOR_ID,
            "detector_model": "OpenVINO face-detection-adas-0001",
            "production_shared_evidence_enabled": False,
            "tracking_behavior": "frozen_face_pipeline_v1",
        },
        "plate": {
            "detector_configuration_hash": plate_detector_config.configuration_hash(),
            "detector_id": PLATE_DETECTOR_ID,
            "evidence_configuration_hash": plate_evidence_config.configuration_hash(),
            "inference_mode": plate_detector_config.inference_mode,
            "model_revision": plate_detection.MODEL_REVISION,
            "primary_threshold": plate_detector_config.primary_threshold,
            "render_configuration_hash": plate_render_config.configuration_hash(),
            "tracking_configuration_hash": plate_tracking_config.configuration_hash(),
        },
        "runtime": runtime_versions(),
        "schema_version": SCHEMA_VERSION,
    }


def assert_active_configuration_frozen() -> None:
    snapshot = active_configuration_snapshot()
    failures: list[str] = []
    if snapshot["face"]["detector_id"] != FACE_DETECTOR_ID:
        failures.append("face detector changed")
    plate = snapshot["plate"]
    expected = {
        "detector_configuration_hash": PLATE_DETECTOR_CONFIGURATION_HASH,
        "detector_id": PLATE_DETECTOR_ID,
        "evidence_configuration_hash": PLATE_EVIDENCE_CONFIGURATION_HASH,
        "inference_mode": plate_detection.INFERENCE_MODE,
        "primary_threshold": 0.30,
        "render_configuration_hash": PLATE_RENDER_CONFIGURATION_HASH,
        "tracking_configuration_hash": PLATE_TRACKING_CONFIGURATION_HASH,
    }
    for key, value in expected.items():
        if plate.get(key) != value:
            failures.append(f"plate {key} changed: expected {value!r}, got {plate.get(key)!r}")
    if not isinstance(plate_tracking.default_evidence_config(), TemporalEvidenceConfiguration):
        failures.append("plate evidence config no longer uses shared TemporalEvidenceConfiguration")
    if failures:
        raise AssertionError("; ".join(failures))


def runtime_versions() -> dict[str, str]:
    packages = ("numpy", "opencv-python", "openvino", "onnxruntime", "pytest", "ruff")
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    for executable in ("ffmpeg", "ffprobe"):
        try:
            versions[executable] = media.version_line(executable)
        except (FileNotFoundError, OSError):
            versions[executable] = "not_available"
    return versions


def source_artifacts_compatible(records: list[SourceArtifactRecord]) -> dict[str, Any]:
    if len(records) < 2:
        return {"compatible": True, "reasons": []}
    first = records[0]
    checks = {
        "source_sha256": all(record.source_sha256 == first.source_sha256 for record in records),
        "frame_count": all(record.frame_count == first.frame_count for record in records),
        "fps": all(record.fps == first.fps for record in records),
        "width": all(record.width == first.width for record in records),
        "height": all(record.height == first.height for record in records),
    }
    reasons = [f"mismatched_{key}" for key, passed in checks.items() if not passed]
    return {"compatible": not reasons, "reasons": reasons}


def build_source_compatibility_matrix(
    groups: dict[str, list[SourceArtifactRecord]],
) -> dict[str, Any]:
    return {
        "groups": {
            clip_id: {
                "can_combine_face_and_plate": source_artifacts_compatible(records)["compatible"],
                "compatibility": source_artifacts_compatible(records),
                "records": [record.to_dict() for record in records],
            }
            for clip_id, records in sorted(groups.items())
        },
        "schema_version": SCHEMA_VERSION,
    }


def audit_schema() -> dict[str, Any]:
    return {
        "allowed_categories": sorted(AUDIT_CATEGORIES),
        "allowed_decisions": sorted(AUDIT_DECISIONS),
        "allowed_issue_types": sorted(AUDIT_ISSUE_TYPES),
        "allowed_severities": sorted(AUDIT_SEVERITIES),
        "required_issue_fields": [
            "audit_item_id",
            "clip_id",
            "category",
            "issue_type",
            "source_frame_start",
            "source_frame_end",
            "severity",
            "reviewer_decision",
            "notes",
        ],
        "schema_version": AUDIT_SCHEMA_VERSION,
    }


def pending_human_metrics() -> dict[str, Any]:
    video_metrics = {
        "privacy_required_frames": None,
        "covered_privacy_required_frames": None,
        "leak_frames": None,
        "false_redaction_frames": None,
        "frames_with_any_issue": None,
    }
    object_metrics = {
        "total_privacy_object_frames": None,
        "covered_object_frames": None,
        "leaked_object_frames": None,
    }
    interval_metrics = {
        "leak_intervals": None,
        "longest_leak_run": None,
        "approximate_leak_duration_seconds": None,
        "false_redaction_intervals": None,
        "wrong_target_events": None,
    }
    return {
        "categories": {
            "face": {
                "interval_metrics": dict(interval_metrics),
                "object_frame_metrics": dict(object_metrics),
                "review_status": "pending_review",
                "video_frame_metrics": dict(video_metrics),
            },
            "plate": {
                "interval_metrics": dict(interval_metrics),
                "object_frame_metrics": dict(object_metrics),
                "review_status": "pending_review",
                "video_frame_metrics": dict(video_metrics),
            },
        },
        "combined": {
            "frames_with_any_face_or_plate_leak": None,
            "frames_with_both_face_and_plate_privacy_requirements": None,
            "review_status": "pending_review",
            "total_blocker_intervals": None,
        },
        "schema_version": AUDIT_SCHEMA_VERSION,
    }


def validate_audit_tasks(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload.get("tasks", [])):
        task_id = str(item.get("audit_item_id") or "")
        if not task_id:
            errors.append(f"tasks[{index}] missing audit_item_id")
        elif task_id in seen:
            errors.append(f"duplicate audit_item_id: {task_id}")
        seen.add(task_id)
        validate_audit_issue(item, errors, f"tasks[{index}]")
    return {"error_count": len(errors), "errors": errors, "passed": not errors}


def validate_audit_decisions(
    decisions_payload: dict[str, Any],
    tasks_payload: dict[str, Any],
) -> dict[str, Any]:
    task_validation = validate_audit_tasks(tasks_payload)
    errors = list(task_validation["errors"])
    task_ids = {str(item.get("audit_item_id")) for item in tasks_payload.get("tasks", [])}
    decisions = decisions_payload.get("decisions", [])
    seen_decisions: set[str] = set()
    for index, decision in enumerate(decisions):
        item_id = str(decision.get("audit_item_id") or "")
        if item_id not in task_ids:
            errors.append(f"decisions[{index}] references unknown audit_item_id: {item_id}")
        if item_id in seen_decisions:
            errors.append(f"duplicate decision for audit_item_id: {item_id}")
        seen_decisions.add(item_id)
        if decision.get("reviewer_decision") not in AUDIT_DECISIONS:
            errors.append(f"decisions[{index}] has invalid reviewer_decision")
    blocking_task_ids = {
        str(item["audit_item_id"])
        for item in tasks_payload.get("tasks", [])
        if item.get("severity") == "blocker"
    }
    resolved_blocking = {
        str(item.get("audit_item_id"))
        for item in decisions
        if item.get("reviewer_decision") in {"confirmed_issue", "not_an_issue", "cosmetic_only"}
    }
    unresolved_blocking = sorted(blocking_task_ids - resolved_blocking)
    return {
        "decision_count": len(decisions),
        "error_count": len(errors),
        "errors": errors,
        "passed": not errors,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "unresolved_blocking_items": unresolved_blocking,
    }


def summarize_audit(
    tasks_payload: dict[str, Any],
    decisions_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decisions_by_id = {
        str(item.get("audit_item_id")): item
        for item in (decisions_payload or {}).get("decisions", [])
    }
    summary = {
        "by_category": {},
        "by_severity": {},
        "completed_decisions": len(decisions_by_id),
        "human_metrics_ready": False,
        "pending_review_count": 0,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "total_tasks": len(tasks_payload.get("tasks", [])),
        "unresolved_blocking_count": 0,
    }
    for item in tasks_payload.get("tasks", []):
        category = str(item.get("category"))
        severity = str(item.get("severity"))
        summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        decision = decisions_by_id.get(str(item.get("audit_item_id")))
        if decision is None:
            summary["pending_review_count"] += 1
            if severity == "blocker":
                summary["unresolved_blocking_count"] += 1
    summary["human_metrics_ready"] = summary["pending_review_count"] == 0
    return summary


def validate_audit_issue(item: dict[str, Any], errors: list[str], prefix: str) -> None:
    for field in audit_schema()["required_issue_fields"]:
        if field not in item:
            errors.append(f"{prefix} missing {field}")
    if item.get("category") not in AUDIT_CATEGORIES:
        errors.append(f"{prefix} invalid category")
    if item.get("issue_type") not in AUDIT_ISSUE_TYPES:
        errors.append(f"{prefix} invalid issue_type")
    if item.get("severity") not in AUDIT_SEVERITIES:
        errors.append(f"{prefix} invalid severity")
    if item.get("reviewer_decision") not in AUDIT_DECISIONS:
        errors.append(f"{prefix} invalid reviewer_decision")
    start = item.get("source_frame_start")
    end = item.get("source_frame_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        errors.append(f"{prefix} invalid frame range")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
