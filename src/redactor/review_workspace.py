from __future__ import annotations

# ruff: noqa: E501
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2 as cv

from redactor import (
    case_manifests,
    detection,
    face_policy,
    final_package,
    frame_spans,
    media,
    plate_rendering,
    plate_tracking,
    privacy_entities,
    privacy_rendering,
    rendering,
    review_api_contract,
    review_finalization,
    review_sessions,
    review_working_state,
)
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    stable_hash,
)

SCHEMA_VERSION = "final_review_workspace.v2.3.4"
FRONTEND_MILESTONE_VERSION = "v1.5.4"
DAY_PARENT_PATH = Path("artifacts/day_primary.mp4")
DAY_CANONICAL_SOURCE_FRAME_START = 1828
DAY_CANONICAL_SOURCE_FRAME_END = 5694
PREVIOUS_FROZEN_WORKSPACE_PACKAGE = Path("artifacts/final_pipeline_package_v2_3_3")
FROZEN_V2_3_4_WORKSPACE_PACKAGE = Path("artifacts/final_pipeline_package_v2_3_4")

EXPECTED_ADAS_XML_SHA256 = "ef8fb491060e64da9d31893045f36dbb24f59455308e961274dd234190db997b"
EXPECTED_ADAS_BIN_SHA256 = "54883903421c8ee5289334ca4da779976a8be66daf7d57e45d005d0bc2b8c637"
EXPECTED_OIM_PLATE_ONNX_SHA256 = "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a"
DEFAULT_ADAS_XML_PATH = Path("models/face-detection-adas-0001/face-detection-adas-0001.xml")
DEFAULT_ADAS_BIN_PATH = Path("models/face-detection-adas-0001/face-detection-adas-0001.bin")
DEFAULT_OIM_PLATE_MODEL_PATH = Path(
    "models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx"
)

DECISION_LABELS = {
    review_sessions.ACTION_COSMETIC_ONLY: "Cosmetic only",
    review_sessions.ACTION_ESCALATE: "Uncertain / Escalate",
    review_sessions.ACTION_FALSE_REDACTION: "False redaction",
    review_sessions.ACTION_LOOKS_CORRECT: "Looks correct",
    review_sessions.ACTION_NOT_SENSITIVE: "Not sensitive",
    review_sessions.ACTION_PARTIAL_COVERAGE: "Partial coverage",
    review_sessions.ACTION_PRIVACY_LEAK: "Privacy leak",
    review_sessions.ACTION_WRONG_TARGET: "Wrong target",
}

REQUIRED_ACTIONS = tuple(DECISION_LABELS)


@dataclass(frozen=True)
class FrozenModelPaths:
    adas_xml: Path
    adas_bin: Path
    plate_model: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "adas_bin": self.adas_bin.as_posix(),
            "adas_xml": self.adas_xml.as_posix(),
            "plate_model": self.plate_model.as_posix(),
        }


def build_workspace_package(
    *,
    package_root: Path,
    v1_package_root: Path,
    adas_xml_path: Path | None = None,
    adas_bin_path: Path | None = None,
    plate_model_path: Path | None = None,
    frontend: str = "legacy",
    build_frontend: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if package_root.exists() and any(package_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{package_root} already exists; pass overwrite=True to rebuild it")
    if package_root.exists() and overwrite:
        shutil.rmtree(package_root)
    for child in (
        "assets",
        "cases",
        "entities",
        "media/day",
        "media/night",
        "regions",
        "sessions",
        "technical",
    ):
        (package_root / child).mkdir(parents=True, exist_ok=True)

    final_package.assert_active_configuration_frozen()
    model_paths = resolve_frozen_model_paths(
        adas_xml_path=adas_xml_path,
        adas_bin_path=adas_bin_path,
        plate_model_path=plate_model_path,
    )
    model_validation = validate_frozen_models(model_paths)
    write_json(package_root / "model_validation.json", model_validation)
    copy_if_exists(
        v1_package_root / "configuration_freeze.json", package_root / "configuration_freeze.json"
    )
    copy_if_exists(
        v1_package_root / "metrics" / "human_metrics_pending.json",
        package_root / "human_metrics_pending.json",
    )

    day_lineage = build_day_case(package_root, model_paths=model_paths)
    night_manifest = build_night_case(package_root, v1_package_root)
    behavior_parity_source = preserve_previous_workspace_behavior(package_root)
    raw_signals = load_raw_audit_signals(v1_package_root / "audit" / "audit_tasks.json")
    tasks = aggregate_reviewer_tasks(raw_signals)
    tasks, task_chunking_audit = chunk_reviewer_tasks_for_package(package_root, tasks)
    write_json(
        package_root / "raw_audit_signals.json",
        {"raw_signals": raw_signals, "schema_version": SCHEMA_VERSION},
    )
    write_json(
        package_root / "reviewer_tasks.json",
        {
            "actions": [
                {"action": action, "label": label}
                for action, label in sorted(DECISION_LABELS.items())
            ],
            "raw_signal_count": len(raw_signals),
            "schema_version": "case_reviewer_tasks.v1",
            "task_count": len(tasks),
            "tasks": tasks,
        },
    )
    write_json(package_root / "audit_schema.json", audit_schema())
    write_json(package_root / "review_task_chunking_audit.json", task_chunking_audit)
    write_json(
        package_root / "audit_decisions.template.json",
        {"decisions": [], "schema_version": "case_review_decisions_export.v1"},
    )
    entity_audit = write_entity_index(package_root)
    focus_overlay_audit = write_entity_focus_overlay_audit(package_root)
    write_merge_audit_shells(package_root)
    write_manual_correction_persistence_audit_shell(package_root)
    review_working_state.write_reset_baseline_state_audit_shell(package_root)
    write_workspace_assets(package_root, frontend=frontend, build_frontend=build_frontend)
    version_alignment_audit = write_version_alignment_audit(package_root)
    if not version_alignment_audit["passed"]:
        raise ValueError("workspace version alignment validation failed")
    timeline_marker_audit = write_timeline_marker_audit(package_root, tasks)
    write_manual_firefox_checklist(package_root)
    workspace_manifest = {
        "case_count": 2,
        "cases": {
            "day": day_lineage["case_manifest_path"],
            "night": night_manifest["case_manifest_path"],
        },
        "created_at_utc": media.utc_timestamp(),
        "entity_focus_overlay_audit": "entity_focus_overlay_audit.json",
        "frontend_milestone_version": FRONTEND_MILESTONE_VERSION,
        "manual_firefox_acceptance_checklist": "manual_firefox_acceptance_checklist.md",
        "manual_correction_persistence_audit": "manual_correction_persistence_audit.json",
        "merged_entity_naming_audit": "merged_entity_naming_audit.json",
        "merged_entity_policy_audit": "merged_entity_policy_audit.json",
        "model_validation": "model_validation.json",
        "previous_behavior_parity_source": "previous_behavior_parity_source.json",
        "reset_baseline_state_audit": "reset_baseline_state_audit.json",
        "review_task_chunking_audit": "review_task_chunking_audit.json",
        "schema_version": SCHEMA_VERSION,
        "selected_frontend": frontend,
        "timeline_marker_audit": "timeline_marker_audit.json",
        "version_alignment_audit": "version_alignment_audit.json",
        "v1_package_root": v1_package_root.as_posix(),
    }
    write_json(package_root / "workspace_manifest.json", workspace_manifest)
    media_url_audit = validate_workspace_media_urls(package_root)
    write_json(package_root / "media_url_audit.json", media_url_audit)
    if not media_url_audit["passed"]:
        raise ValueError("workspace media URL validation failed")
    return {
        "day": day_lineage,
        "entity_audit": entity_audit,
        "entity_focus_overlay_audit": focus_overlay_audit,
        "media_url_audit": media_url_audit,
        "model_validation": model_validation,
        "night": night_manifest,
        "previous_behavior_parity_source": behavior_parity_source,
        "raw_signal_count": len(raw_signals),
        "reviewer_task_count": len(tasks),
        "schema_version": SCHEMA_VERSION,
        "timeline_marker_audit": timeline_marker_audit,
        "frontend": frontend,
    }


def build_react_workspace_package_from_reference(
    *,
    package_root: Path,
    reference_root: Path = FROZEN_V2_3_4_WORKSPACE_PACKAGE,
    build_frontend: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    reference_root = reference_root.resolve()
    package_root = package_root.resolve()
    if not reference_root.exists():
        raise FileNotFoundError(f"frozen reference package does not exist: {reference_root}")
    if package_root == reference_root:
        raise ValueError("React frontend package cannot overwrite the frozen reference package")
    if package_root.exists() and any(package_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{package_root} already exists; pass overwrite=True to rebuild it")
    if package_root.exists() and overwrite:
        shutil.rmtree(package_root)
    shutil.copytree(reference_root, package_root)
    purge_session_local_state(package_root)
    apply_review_entity_coverage_patch(package_root)
    focus_overlay_audit = write_entity_focus_overlay_audit(package_root)
    role_alignment_audit = ensure_frame_aligned_review_roles(package_root)
    object_policy_audit = write_object_policy_lineage_audit(package_root)
    write_merge_audit_shells(package_root)
    write_manual_correction_persistence_audit_shell(package_root)
    review_working_state.write_reset_baseline_state_audit_shell(package_root)
    existing_chunking_audit_path = package_root / "review_task_chunking_audit.json"
    existing_chunking_audit = (
        read_json_file(existing_chunking_audit_path)
        if existing_chunking_audit_path.exists()
        else {}
    )
    if existing_chunking_audit.get("passed") is True:
        task_chunking_audit = existing_chunking_audit
    else:
        task_chunking_audit = apply_reviewer_task_chunking_to_package(package_root)
    tasks = read_json_file(package_root / "reviewer_tasks.json").get("tasks") or []
    existing_timeline_audit_path = package_root / "timeline_marker_audit.json"
    if existing_chunking_audit.get("passed") is True and existing_timeline_audit_path.exists():
        timeline_marker_audit = read_json_file(existing_timeline_audit_path)
    else:
        timeline_marker_audit = write_timeline_marker_audit(package_root, list(tasks))
    write_json(
        package_root / "materialization_job_audit.json",
        {
            "cases": {},
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "materialization_job_audit.v1",
        },
    )
    write_json(
        package_root / "materialization_timing_audit.json",
        {
            "created_at_utc": media.utc_timestamp(),
            "jobs": [],
            "schema_version": "materialization_timing_audit.v1",
            "summary_by_case": {},
        },
    )
    write_react_workspace_assets(package_root, build_frontend=build_frontend)
    workspace_manifest_path = package_root / "workspace_manifest.json"
    workspace_manifest = read_json_file(workspace_manifest_path)
    workspace_manifest.update(
        {
            "api_schema_version": review_api_contract.API_SCHEMA_VERSION,
            "entity_focus_overlay_audit": "entity_focus_overlay_audit.json",
            "frontend_build_manifest": "frontend_build_manifest.json",
            "frontend_milestone_version": FRONTEND_MILESTONE_VERSION,
            "legacy_fallback": "legacy.html",
            "manual_correction_persistence_audit": "manual_correction_persistence_audit.json",
            "materialization_timing_audit": "materialization_timing_audit.json",
            "merged_entity_naming_audit": "merged_entity_naming_audit.json",
            "merged_entity_policy_audit": "merged_entity_policy_audit.json",
            "react_app_root": "app/",
            "reset_baseline_state_audit": "reset_baseline_state_audit.json",
            "review_task_chunking_audit": "review_task_chunking_audit.json",
            "selected_frontend": "react",
        }
    )
    write_json(workspace_manifest_path, workspace_manifest)
    media_url_audit = validate_workspace_media_urls(package_root)
    write_json(package_root / "media_url_audit.json", media_url_audit)
    if not media_url_audit["passed"]:
        raise ValueError("workspace media URL validation failed")
    return {
        "frontend": "react",
        "media_url_audit": media_url_audit,
        "object_policy_lineage_audit": object_policy_audit,
        "entity_focus_overlay_audit": focus_overlay_audit,
        "reference_root": reference_root.as_posix(),
        "role_alignment_audit": role_alignment_audit,
        "task_chunking_audit": task_chunking_audit,
        "timeline_marker_audit": timeline_marker_audit,
        "schema_version": SCHEMA_VERSION,
    }


def purge_session_local_state(package_root: Path) -> None:
    for relative_path in ("sessions",):
        path = package_root / relative_path
        if path.exists():
            shutil.rmtree(path)


def ensure_frame_aligned_review_roles(package_root: Path) -> dict[str, Any]:
    audit_cases: dict[str, Any] = {}
    for case_path in sorted((package_root / "cases").glob("*.json")):
        manifest = case_manifests.load_case_manifest(case_path)
        roles = list(manifest.video_roles)
        by_role = {role.role: role for role in roles}
        added: list[str] = []
        source_role = by_role.get("automatic") or by_role.get("current_result")
        if source_role is not None and "automatic" not in by_role:
            roles.append(
                replace(
                    source_role,
                    label="Automatic",
                    role="automatic",
                    sha256=media.sha256_file(package_root / source_role.path),
                )
            )
            added.append("automatic")
        by_role = {role.role: role for role in roles}
        source_role = (
            by_role.get("latest_reviewed")
            or by_role.get("automatic")
            or by_role.get("current_result")
        )
        if source_role is not None and "latest_reviewed" not in by_role:
            source_path = package_root / source_role.path
            latest_path = package_root / "media" / manifest.case_id / "latest_reviewed_result.mp4"
            if not latest_path.exists() and source_path.exists():
                shutil.copy2(source_path, latest_path)
            role_path = latest_path if latest_path.exists() else source_path
            roles.append(
                case_manifests.VideoRole(
                    label="Latest reviewed",
                    path=role_path.relative_to(package_root).as_posix(),
                    role="latest_reviewed",
                    sha256=media.sha256_file(role_path)
                    if role_path.exists()
                    else source_role.sha256,
                    source_frame_end=manifest.source_frame_end,
                    source_frame_start=manifest.source_frame_start,
                )
            )
            added.append("latest_reviewed")
        role_order = {
            "original": 0,
            "automatic": 1,
            "current_result": 2,
            "latest_reviewed": 3,
            "final_reviewed": 4,
        }
        roles = sorted(roles, key=lambda role: (role_order.get(role.role, 99), role.label))
        case_manifests.write_case_manifest(case_path, replace(manifest, video_roles=tuple(roles)))
        audit_cases[manifest.case_id] = {
            "added_roles": added,
            "roles": [role.to_dict() for role in roles],
            "schema_version": "case_review_role_alignment.v1",
        }
    audit = {
        "cases": audit_cases,
        "created_at_utc": media.utc_timestamp(),
        "passed": all(
            {"original", "automatic", "latest_reviewed"}.issubset(
                {role["role"] for role in case["roles"]}
            )
            for case in audit_cases.values()
        ),
        "schema_version": "frame_aligned_review_roles_audit.v1",
    }
    write_json(package_root / "frame_aligned_review_roles_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("review role alignment failed")
    return audit


def write_object_policy_lineage_audit(package_root: Path) -> dict[str, Any]:
    index = read_json_file(package_root / "review_entities.json")
    cases = dict(index.get("cases") or {})
    audit_cases: dict[str, Any] = {}
    for case_id, entities_value in sorted(cases.items()):
        manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
        entities = [dict(entity) for entity in list(entities_value or [])]
        rows: list[dict[str, Any]] = []
        for row in face_policy_coverage_rows(
            package_root=package_root,
            case_id=str(case_id),
            manifest=manifest,
            entities=entities,
        ):
            mapped = row.get("review_entity_uuid")
            policy = row.get("policy")
            rows.append(
                {
                    "child_fragments": child_fragments_for_policy_track(
                        entities,
                        str(row.get("track_id")),
                    ),
                    "assignment_provenance": row.get("assignment_provenance"),
                    "mapping": "exact" if mapped else "unresolved",
                    "resulting_initial_action": policy if mapped else None,
                    "review_entity": mapped,
                    "source_policy": policy,
                    "source_span": [row.get("source_frame_start"), row.get("source_frame_end")],
                    "source_track": row.get("track_id"),
                    "valid_source_evidence": row.get("valid_source_evidence"),
                }
            )
        missing = [
            row
            for row in rows
            if row["valid_source_evidence"]
            and row["source_policy"] == privacy_entities.POLICY_KEEP_VISIBLE
            and row["assignment_provenance"] == "explicit_case_policy"
            and row["review_entity"] is None
        ]
        audit_cases[str(case_id)] = {
            "missing_source_backed_tracks": missing,
            "policy_lineage": rows,
            "schema_version": "object_policy_lineage_case_audit.v1",
        }
    audit = {
        "cases": audit_cases,
        "created_at_utc": media.utc_timestamp(),
        "passed": not any(case["missing_source_backed_tracks"] for case in audit_cases.values()),
        "schema_version": "object_policy_lineage_audit.v1",
    }
    write_json(package_root / "object_policy_lineage_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("source-backed object policy lineage is incomplete")
    return audit


def child_fragments_for_policy_track(entities: list[dict[str, Any]], track_id: str) -> list[str]:
    for entity in entities:
        if track_id in {str(item) for item in entity.get("automatic_source_track_ids") or []}:
            return [str(item) for item in entity.get("attached_fragment_track_ids") or []]
    return []


def build_day_case(package_root: Path, *, model_paths: FrozenModelPaths) -> dict[str, Any]:
    plan = load_evaluation_plan_clips(Path("data/evaluation_plan.yaml"))
    day_plate = plan["day_plate"]
    day_face = plan["day_face"]
    parent_file = DAY_PARENT_PATH
    lineage = {
        "day_face": day_face,
        "day_plate": day_plate,
        "lineage_proven": (
            day_face.get("parent_file") == day_plate.get("parent_file") == parent_file.as_posix()
        ),
        "parent_file": parent_file.as_posix(),
    }
    blockers: list[str] = []
    if not lineage["lineage_proven"]:
        blockers.append("day face and plate clips do not share a proven parent source")
    if not parent_file.exists():
        blockers.append(f"canonical day parent source is missing: {parent_file}")

    parent_info = media.probe_media(parent_file) if parent_file.exists() else None
    parent_fps = (
        rational_to_float(parent_info.video_stream.nominal_frame_rate)
        if parent_info is not None
        else 30000 / 1001
    ) or 30000 / 1001
    start_seconds = DAY_CANONICAL_SOURCE_FRAME_START / parent_fps
    canonical_frame_count = DAY_CANONICAL_SOURCE_FRAME_END - DAY_CANONICAL_SOURCE_FRAME_START + 1
    # FFmpeg's time-based trim can include the first frame after an exact inclusive
    # duration. The shorter duration below has been validated by the frame-count
    # gate immediately after extraction: it yields frames 1828 through 5694.
    duration_seconds = (canonical_frame_count - 1) / parent_fps
    canonical_source = package_root / "media" / "day" / "original_reference.mp4"
    canonical_needs_extraction = not canonical_source.exists()
    if canonical_source.exists():
        try:
            existing_info = media.probe_media(canonical_source)
            canonical_needs_extraction = (
                int(existing_info.video_stream.frame_count or 0) != canonical_frame_count
            )
        except ValueError:
            canonical_needs_extraction = True
    if parent_file.exists() and canonical_needs_extraction:
        media.trim_video(
            parent_file,
            canonical_source,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            overwrite=True,
        )
    day_generation: dict[str, Any] | None = None
    if canonical_source.exists() and not blockers:
        day_generation = build_day_current_result(
            package_root=package_root,
            canonical_source=canonical_source,
            model_paths=model_paths,
        )
    if canonical_source.exists():
        info = media.probe_media(canonical_source)
        source_hash = info.sha256
        frame_count = int(info.video_stream.frame_count or 0)
        parent_frame_start = DAY_CANONICAL_SOURCE_FRAME_START
        parent_frame_end = parent_frame_start + max(0, frame_count - 1)
        if parent_frame_end != DAY_CANONICAL_SOURCE_FRAME_END:
            blockers.append(
                "canonical Day extraction frame span mismatch: expected "
                f"{DAY_CANONICAL_SOURCE_FRAME_START}-{DAY_CANONICAL_SOURCE_FRAME_END}, "
                f"got {parent_frame_start}-{parent_frame_end}"
            )
        width = info.video_stream.width
        height = info.video_stream.height
        duration_seconds_actual = info.duration_seconds
        fps = info.video_stream.nominal_frame_rate or "30000/1001"
        roles = [
            case_manifests.VideoRole(
                label="Original",
                path="media/day/original_reference.mp4",
                role="original",
                sha256=source_hash,
                source_frame_end=parent_frame_end,
                source_frame_start=parent_frame_start,
            ),
        ]
        automatic_path = package_root / "media" / "day" / "automatic_result.mp4"
        if automatic_path.exists():
            roles.append(
                case_manifests.VideoRole(
                    label="Automatic",
                    path="media/day/automatic_result.mp4",
                    role="automatic",
                    sha256=media.sha256_file(automatic_path),
                    source_frame_end=parent_frame_end,
                    source_frame_start=parent_frame_start,
                )
            )
        current_path = package_root / "media" / "day" / "current_combined_result.mp4"
        if current_path.exists():
            roles.append(
                case_manifests.VideoRole(
                    label="Current result",
                    path="media/day/current_combined_result.mp4",
                    role="current_result",
                    sha256=media.sha256_file(current_path),
                    source_frame_end=parent_frame_end,
                    source_frame_start=parent_frame_start,
                )
            )
        else:
            blockers.append("canonical Day current combined result was not generated")
        latest_path = package_root / "media" / "day" / "latest_reviewed_result.mp4"
        if latest_path.exists():
            roles.append(
                case_manifests.VideoRole(
                    label="Latest reviewed",
                    path="media/day/latest_reviewed_result.mp4",
                    role="latest_reviewed",
                    sha256=media.sha256_file(latest_path),
                    source_frame_end=parent_frame_end,
                    source_frame_start=parent_frame_start,
                )
            )
        video_roles = tuple(roles)
    else:
        source_hash = None
        frame_count = 0
        parent_frame_start = DAY_CANONICAL_SOURCE_FRAME_START
        parent_frame_end = DAY_CANONICAL_SOURCE_FRAME_END
        width = None
        height = None
        duration_seconds_actual = None
        fps = "30000/1001"
        video_roles = ()

    manifest = case_manifests.CaseManifest(
        blockers=tuple(blockers),
        case_id="day",
        display_name="Day case",
        duration_seconds=duration_seconds_actual,
        face_evidence_inputs=(
            "technical/day_pipeline/face_detection/detections.jsonl",
            "technical/day_pipeline/face_policy/tracking/tracks.jsonl",
            "technical/day_pipeline/face_policy/privacy_regions/privacy_regions.jsonl",
        ),
        fps=fps,
        frame_count=frame_count,
        height=height,
        lineage=lineage
        | {
            "canonical_parent_start_seconds": start_seconds,
            "canonical_parent_end_seconds": start_seconds + duration_seconds,
            "canonical_source_frame_end": DAY_CANONICAL_SOURCE_FRAME_END,
            "canonical_source_frame_start": DAY_CANONICAL_SOURCE_FRAME_START,
            "canonical_generation": day_generation,
            "canonical_generation_status": "generated" if day_generation else "blocked",
        },
        parent_source_hash=media.sha256_file(parent_file) if parent_file.exists() else None,
        plate_evidence_inputs=(
            "technical/day_pipeline/plate_detection/detections.jsonl",
            "technical/day_pipeline/plate_tracking/tracks.jsonl",
        ),
        reviewed_correction_inputs=(
            "artifacts/day_reviewer_discovered_correction_v1/privacy_regions.jsonl",
        ),
        source_frame_end=parent_frame_end,
        source_frame_start=parent_frame_start,
        source_path="media/day/original_reference.mp4" if canonical_source.exists() else None,
        source_video_hash=source_hash,
        status=case_manifests.CASE_STATUS_BLOCKED if blockers else case_manifests.CASE_STATUS_READY,
        video_roles=video_roles,
        width=width,
    )
    case_path = package_root / "cases" / "day.json"
    case_manifests.write_case_manifest(case_path, manifest)
    lineage_report_md = package_root / "day_case_lineage_report.md"
    lineage_report_md.write_text(day_lineage_report_text(lineage, blockers), encoding="utf-8")
    if blockers:
        blocker_md = package_root / "day_case_lineage_blocker.md"
        blocker_md.write_text(day_lineage_report_text(lineage, blockers), encoding="utf-8")
    write_json(
        package_root / "technical" / "day_source_lineage_audit.json",
        lineage | {"blockers": blockers},
    )
    return {
        "blockers": blockers,
        "case_manifest_path": case_path.relative_to(package_root).as_posix(),
        "canonical_original": "media/day/original_reference.mp4"
        if canonical_source.exists()
        else None,
        "canonical_current": "media/day/current_combined_result.mp4"
        if (package_root / "media" / "day" / "current_combined_result.mp4").exists()
        else None,
        "frame_count": frame_count,
        "lineage_proven": lineage["lineage_proven"],
    }


def preserve_previous_workspace_behavior(package_root: Path) -> dict[str, Any]:
    reference_root = PREVIOUS_FROZEN_WORKSPACE_PACKAGE
    copied: list[dict[str, Any]] = []
    errors: list[str] = []
    if not reference_root.exists():
        payload = {
            "copied": copied,
            "errors": [f"reference workspace is missing: {reference_root.as_posix()}"],
            "reference_package": reference_root.as_posix(),
            "schema_version": "previous_behavior_parity_source.v1",
            "used_reference": False,
        }
        write_json(package_root / "previous_behavior_parity_source.json", payload)
        return payload
    for case_id in ("day", "night"):
        current_manifest_path = package_root / "cases" / f"{case_id}.json"
        reference_manifest_path = reference_root / "cases" / f"{case_id}.json"
        if not current_manifest_path.exists() or not reference_manifest_path.exists():
            errors.append(f"{case_id}: manifest missing in current or reference package")
            continue
        current_manifest = case_manifests.load_case_manifest(current_manifest_path)
        reference_manifest = case_manifests.load_case_manifest(reference_manifest_path)
        compatibility_errors = case_manifest_compatibility_errors(
            current_manifest,
            reference_manifest,
        )
        if compatibility_errors:
            errors.extend(f"{case_id}: {error}" for error in compatibility_errors)
            continue
        region_src = reference_root / "regions" / f"{case_id}_current_regions.jsonl"
        region_dst = package_root / "regions" / f"{case_id}_current_regions.jsonl"
        if region_src.exists():
            shutil.copy2(region_src, region_dst)
            copied.append(
                {
                    "case_id": case_id,
                    "destination": region_dst.relative_to(package_root).as_posix(),
                    "sha256": media.sha256_file(region_dst),
                    "source": region_src.as_posix(),
                    "type": "privacy_regions",
                }
            )
        updated_roles: list[case_manifests.VideoRole] = []
        for role in current_manifest.video_roles:
            reference_role = next(
                (item for item in reference_manifest.video_roles if item.role == role.role),
                None,
            )
            reference_path = reference_root / reference_role.path if reference_role else None
            role_path = package_root / role.path
            if (
                reference_role is not None
                and role.role
                in {"automatic", "current_result", "latest_reviewed", "final_reviewed"}
                and reference_path is not None
                and reference_path.exists()
                and role_path.exists()
            ):
                shutil.copy2(reference_path, role_path)
                updated_roles.append(replace(role, sha256=media.sha256_file(role_path)))
                copied.append(
                    {
                        "case_id": case_id,
                        "destination": role.path,
                        "role": role.role,
                        "sha256": media.sha256_file(role_path),
                        "source": reference_path.as_posix(),
                        "type": "video_role",
                    }
                )
            else:
                updated_roles.append(role)
        case_manifests.write_case_manifest(
            current_manifest_path,
            replace(current_manifest, video_roles=tuple(updated_roles)),
        )
    payload = {
        "copied": copied,
        "created_at_utc": media.utc_timestamp(),
        "errors": errors,
        "reference_package": reference_root.as_posix(),
        "schema_version": "previous_behavior_parity_source.v1",
        "used_reference": bool(copied) and not errors,
    }
    write_json(package_root / "previous_behavior_parity_source.json", payload)
    if errors:
        raise ValueError("previous workspace behavior parity validation failed")
    return payload


def case_manifest_compatibility_errors(
    current: case_manifests.CaseManifest,
    reference: case_manifests.CaseManifest,
) -> list[str]:
    errors: list[str] = []
    for field_name in (
        "case_id",
        "source_frame_start",
        "source_frame_end",
        "frame_count",
        "fps",
        "width",
        "height",
        "source_video_hash",
    ):
        current_value = getattr(current, field_name)
        reference_value = getattr(reference, field_name)
        if current_value != reference_value:
            errors.append(f"{field_name} mismatch: {current_value!r} != {reference_value!r}")
    return errors


def resolve_frozen_model_paths(
    *,
    adas_xml_path: Path | None = None,
    adas_bin_path: Path | None = None,
    plate_model_path: Path | None = None,
) -> FrozenModelPaths:
    return FrozenModelPaths(
        adas_xml=Path(
            adas_xml_path or os.environ.get("REDACTOR_ADAS_XML") or DEFAULT_ADAS_XML_PATH
        ),
        adas_bin=Path(
            adas_bin_path or os.environ.get("REDACTOR_ADAS_BIN") or DEFAULT_ADAS_BIN_PATH
        ),
        plate_model=Path(
            plate_model_path
            or os.environ.get("REDACTOR_OIM_PLATE_MODEL")
            or DEFAULT_OIM_PLATE_MODEL_PATH
        ),
    )


def validate_frozen_models(model_paths: FrozenModelPaths) -> dict[str, Any]:
    expected = {
        "adas_bin": (model_paths.adas_bin, EXPECTED_ADAS_BIN_SHA256),
        "adas_xml": (model_paths.adas_xml, EXPECTED_ADAS_XML_SHA256),
        "plate_model": (model_paths.plate_model, EXPECTED_OIM_PLATE_ONNX_SHA256),
    }
    errors: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    rejected_terms = {
        "intel",
        "lp-detection-yunet",
        "lpd_yunet",
        "openvino_0106",
        "rfdetr",
        "rtdetr",
        "yunet",
    }
    for name, (path, expected_hash) in expected.items():
        path_text = path.as_posix().lower()
        file_payload: dict[str, Any] = {
            "exists": path.exists(),
            "expected_sha256": expected_hash,
            "path": path.as_posix(),
        }
        if any(term in path_text for term in rejected_terms):
            errors.append(f"{name} points at a rejected historical model path: {path}")
        if not path.exists():
            errors.append(f"{name} is missing: {path}")
        elif not path.is_file():
            errors.append(f"{name} is not a file: {path}")
        else:
            actual_hash = media.sha256_file(path)
            file_payload["sha256"] = actual_hash
            if actual_hash != expected_hash:
                errors.append(f"{name} hash mismatch: expected {expected_hash}, got {actual_hash}")
        files[name] = file_payload
    payload = {
        "detectors": {
            "face": "OpenVINO face-detection-adas-0001",
            "plate": "Open Image Models YOLOv9-S-608 end-to-end license-plate detector",
        },
        "errors": errors,
        "files": files,
        "model_paths": model_paths.to_dict(),
        "passed": not errors,
        "schema_version": "frozen_model_validation.v1",
    }
    if errors:
        raise ValueError("frozen model validation failed: " + "; ".join(errors))
    return payload


def build_day_current_result(
    *,
    package_root: Path,
    canonical_source: Path,
    model_paths: FrozenModelPaths,
) -> dict[str, Any]:
    technical_root = package_root / "technical" / "day_pipeline"
    face_detection_dir = technical_root / "face_detection"
    face_policy_dir = technical_root / "face_policy"
    plate_detection_dir = technical_root / "plate_detection"
    plate_tracking_dir = technical_root / "plate_tracking"
    automatic_video = package_root / "media" / "day" / "automatic_result.mp4"
    current_video = package_root / "media" / "day" / "current_combined_result.mp4"
    latest_video = package_root / "media" / "day" / "latest_reviewed_result.mp4"
    current_regions_path = package_root / "regions" / "day_current_regions.jsonl"

    source_info = media.probe_media(canonical_source)
    source_frame_count = int(source_info.video_stream.frame_count or 0)
    if source_frame_count != DAY_CANONICAL_SOURCE_FRAME_END - DAY_CANONICAL_SOURCE_FRAME_START + 1:
        raise ValueError(
            "canonical Day source frame count mismatch before processing: "
            f"expected {DAY_CANONICAL_SOURCE_FRAME_END - DAY_CANONICAL_SOURCE_FRAME_START + 1}, "
            f"got {source_frame_count}"
        )

    face_detection_summary = detection.run_detection(
        canonical_source,
        category=CATEGORY_FACE,
        model_path=model_paths.adas_xml,
        threshold=detection.DEFAULT_ADAS_PROPOSAL_THRESHOLD,
        primary_threshold=detection.DEFAULT_ADAS_PRIMARY_THRESHOLD,
        detector_backend=detection.DETECTOR_BACKEND_ADAS,
        adas_bin_path=model_paths.adas_bin,
        output_dir=face_detection_dir,
        overwrite=True,
        write_overlay_video=False,
    )
    face_policy_summary = run_face_policy_compile_only(
        canonical_source,
        output_dir=face_policy_dir,
        detections_path=face_detection_dir / "detections.jsonl",
        overwrite=True,
    )
    plate_detection_summary = detection.run_detection(
        canonical_source,
        category=CATEGORY_PLATE,
        model_path=model_paths.plate_model,
        threshold=detection.DEFAULT_THRESHOLDS[CATEGORY_PLATE],
        primary_threshold=detection.DEFAULT_THRESHOLDS[CATEGORY_PLATE],
        detector_backend=detection.DETECTOR_BACKEND_OIM_YOLOV9_S_608,
        output_dir=plate_detection_dir,
        overwrite=True,
        write_overlay_video=False,
    )
    plate_tracking_summary = plate_tracking.run_plate_tracking(
        canonical_source,
        detections_path=plate_detection_dir / "detections.jsonl",
        output_dir=plate_tracking_dir,
        overwrite=True,
        write_overlay_video=False,
    )

    records = build_full_span_day_region_records(
        face_regions_path=face_policy_dir / "privacy_regions" / "privacy_regions.jsonl",
        plate_tracks_path=plate_tracking_dir / "tracks.jsonl",
    )
    review_finalization.write_privacy_region_records(current_regions_path, records)
    render_summary = render_combined_current_result(
        canonical_source=canonical_source,
        output_video=automatic_video,
        records=records,
        overwrite=True,
    )
    shutil.copy2(automatic_video, current_video)
    shutil.copy2(automatic_video, latest_video)
    processing_audit = build_day_full_span_processing_audit(
        canonical_source=canonical_source,
        source_info=source_info,
        face_detections_path=face_detection_dir / "detections.jsonl",
        plate_detections_path=plate_detection_dir / "detections.jsonl",
        face_summary=face_detection_summary,
        face_policy_summary=face_policy_summary,
        plate_summary=plate_detection_summary,
        plate_tracking_summary=plate_tracking_summary,
        output_paths={
            "automatic": automatic_video,
            "current_result": current_video,
            "latest_reviewed": latest_video,
        },
        package_root=package_root,
    )
    if not processing_audit["passed"]:
        raise ValueError(
            "canonical Day full-span processing validation failed: "
            + "; ".join(processing_audit["errors"])
        )
    write_json(technical_root / "day_full_span_processing_audit.json", processing_audit)
    write_json(
        technical_root / "canonical_day_generation.json",
        {
            "case_source": canonical_source.relative_to(package_root).as_posix(),
            "combined_current_video": current_video.relative_to(package_root).as_posix(),
            "face": {
                "detection_summary": face_detection_summary,
                "lineage_parent_start_frame": DAY_CANONICAL_SOURCE_FRAME_START,
                "policy_summary": face_policy_summary,
                "processed_frame_range": [
                    DAY_CANONICAL_SOURCE_FRAME_START,
                    DAY_CANONICAL_SOURCE_FRAME_END,
                ],
            },
            "full_span_processing_audit": "technical/day_pipeline/day_full_span_processing_audit.json",
            "generation_mode": "full_span_frozen_face_and_plate_pipelines",
            "model_validation": model_paths.to_dict(),
            "plate": {
                "detection_summary": plate_detection_summary,
                "lineage_parent_start_frame": 1828,
                "processed_frame_range": [
                    DAY_CANONICAL_SOURCE_FRAME_START,
                    DAY_CANONICAL_SOURCE_FRAME_END,
                ],
                "tracking_summary": plate_tracking_summary,
            },
            "render": render_summary,
            "schema_version": "canonical_day_generation.v2",
        },
    )
    return {
        "automatic": automatic_video.relative_to(package_root).as_posix(),
        "automatic_sha256": media.sha256_file(automatic_video),
        "combined_current": current_video.relative_to(package_root).as_posix(),
        "combined_current_sha256": media.sha256_file(current_video),
        "full_span_processing_audit": "technical/day_pipeline/day_full_span_processing_audit.json",
        "latest_reviewed": latest_video.relative_to(package_root).as_posix(),
        "latest_reviewed_sha256": media.sha256_file(latest_video),
        "current_region_count": len(records),
        "region_provenance": privacy_rendering.provenance_counts(records),
    }


def build_full_span_day_region_records(
    *,
    face_regions_path: Path,
    plate_tracks_path: Path,
) -> list[PrivacyRegionRecord]:
    records = face_region_records_from_render_input(
        face_regions_path,
        case_frame_offset=0,
        parent_frame_offset=DAY_CANONICAL_SOURCE_FRAME_START,
    )
    plate_rows = plate_rendering.load_track_rows(plate_tracks_path)
    plate_regions = plate_rendering.eligible_regions_from_rows(
        plate_rows,
        config=plate_rendering.default_config(),
    )
    records.extend(
        with_parent_source_frame(
            region.to_privacy_region(),
            parent_frame_offset=DAY_CANONICAL_SOURCE_FRAME_START,
        )
        for region in plate_regions
    )
    return sorted(
        records,
        key=lambda item: (
            item.source_frame,
            item.category,
            item.object_id or "",
            item.provenance,
        ),
    )


def build_day_full_span_processing_audit(
    *,
    canonical_source: Path,
    source_info: media.MediaInfo,
    face_detections_path: Path,
    plate_detections_path: Path,
    face_summary: dict[str, Any],
    face_policy_summary: dict[str, Any],
    plate_summary: dict[str, Any],
    plate_tracking_summary: dict[str, Any],
    output_paths: dict[str, Path],
    package_root: Path,
) -> dict[str, Any]:
    source_frame_count = int(source_info.video_stream.frame_count or 0)
    source_range = [DAY_CANONICAL_SOURCE_FRAME_START, DAY_CANONICAL_SOURCE_FRAME_END]
    errors: list[str] = []
    face_span = detector_span_audit(
        face_detections_path,
        source_frame_count=source_frame_count,
        parent_source_start=DAY_CANONICAL_SOURCE_FRAME_START,
    )
    plate_span = detector_span_audit(
        plate_detections_path,
        source_frame_count=source_frame_count,
        parent_source_start=DAY_CANONICAL_SOURCE_FRAME_START,
    )
    if not face_span["full_source_processed"]:
        errors.append("face detector did not process every canonical Day frame")
    if not plate_span["full_source_processed"]:
        errors.append("plate detector did not process every canonical Day frame")

    rendered_roles: dict[str, Any] = {}
    for role, path in sorted(output_paths.items()):
        role_info = media.probe_media(path)
        role_frame_count = int(role_info.video_stream.frame_count or 0)
        role_errors: list[str] = []
        if role_frame_count != source_frame_count:
            role_errors.append(
                f"{role} frame count {role_frame_count} != canonical {source_frame_count}"
            )
        if role_info.video_stream.width != source_info.video_stream.width:
            role_errors.append(f"{role} width mismatch")
        if role_info.video_stream.height != source_info.video_stream.height:
            role_errors.append(f"{role} height mismatch")
        role_mapping = {
            "0": DAY_CANONICAL_SOURCE_FRAME_START,
            str(source_frame_count - 1): DAY_CANONICAL_SOURCE_FRAME_END,
        }
        rendered_roles[role] = {
            "case_frame_to_source_frame": role_mapping,
            "errors": role_errors,
            "frame_count": role_frame_count,
            "path": path.relative_to(package_root).as_posix(),
            "resolution": [role_info.video_stream.width, role_info.video_stream.height],
            "sha256": media.sha256_file(path),
            "source_frame_end": DAY_CANONICAL_SOURCE_FRAME_END,
            "source_frame_start": DAY_CANONICAL_SOURCE_FRAME_START,
        }
        errors.extend(role_errors)

    return {
        "canonical_case_frame_count": source_frame_count,
        "canonical_parent_source": DAY_PARENT_PATH.as_posix(),
        "canonical_source_hash": source_info.sha256,
        "created_at_utc": media.utc_timestamp(),
        "detectors": {
            "face": {
                "detector_frame_available": face_span,
                "summary": face_summary,
            },
            "plate": {
                "detector_frame_available": plate_span,
                "summary": plate_summary,
            },
        },
        "errors": errors,
        "face_policy_summary": face_policy_summary,
        "fps": source_info.video_stream.nominal_frame_rate,
        "frame_count": source_frame_count,
        "height": source_info.video_stream.height,
        "output_roles": rendered_roles,
        "passed": not errors,
        "plate_tracking_summary": plate_tracking_summary,
        "schema_version": "canonical_day_full_span_processing_audit.v1",
        "source_frame_end": DAY_CANONICAL_SOURCE_FRAME_END,
        "source_frame_start": DAY_CANONICAL_SOURCE_FRAME_START,
        "source_frame_range": source_range,
        "width": source_info.video_stream.width,
    }


def detector_span_audit(
    detections_path: Path,
    *,
    source_frame_count: int,
    parent_source_start: int,
) -> dict[str, Any]:
    indices = frame_spans.jsonl_frame_indices(detections_path)
    span_report = frame_spans.build_frame_span_report(
        indices,
        source_frame_count=source_frame_count,
    )
    missing_source_ranges = [
        {
            "frame_start": parent_source_start + item["frame_start"],
            "frame_end": parent_source_start + item["frame_end"],
            "length": item["length"],
        }
        for item in span_report["missing_processed_ranges"]
    ]
    return {
        **span_report,
        "available_source_frame_count": span_report["processed_frame_count"],
        "available_source_frame_end": (
            None
            if span_report["last_processed_frame"] is None
            else parent_source_start + int(span_report["last_processed_frame"])
        ),
        "available_source_frame_start": (
            None
            if span_report["first_processed_frame"] is None
            else parent_source_start + int(span_report["first_processed_frame"])
        ),
        "detector_frame_available_for_every_canonical_day_frame": span_report[
            "full_source_processed"
        ],
        "missing_source_frame_ranges": missing_source_ranges,
        "parent_source_frame_end": parent_source_start + source_frame_count - 1,
        "parent_source_frame_start": parent_source_start,
    }


def run_face_policy_compile_only(
    input_video: Path,
    *,
    output_dir: Path,
    detections_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    face_policy.ensure_case_output_dir(output_dir, overwrite=overwrite)
    source_info = media.probe_media(input_video)
    source_frame_count = int(source_info.video_stream.frame_count or 0)
    detection_manifest_dir = output_dir / "detection"
    detection_manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        detection_manifest_dir / "input_detections_manifest.json",
        {
            "detections_path": detections_path.as_posix(),
            "detections_sha256": media.sha256_file(detections_path),
            "detector_id": detection.DETECTOR_BACKEND_ADAS,
            "note": "Workspace Day generation uses supplied frozen ADAS detections.",
            "schema_version": SCHEMA_VERSION,
        },
    )
    tracking_dir = output_dir / "tracking"
    track_summary = face_policy.tracking.run_tracking(
        input_video,
        detections_path=detections_path,
        output_dir=tracking_dir,
        overwrite=overwrite,
    )
    source_manifest = {
        "case_input_video": input_video.name,
        "detector_id": detection.DETECTOR_BACKEND_ADAS,
        "fps": rational_to_float(source_info.video_stream.nominal_frame_rate),
        "height": source_info.video_stream.height,
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": source_frame_count,
        "source_video_hash": source_info.sha256,
        "width": source_info.video_stream.width,
    }
    write_json(output_dir / "source_manifest.json", source_manifest)
    draft_census = face_policy.build_track_census(
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        source_video_hash=source_info.sha256,
        source_frame_count=source_frame_count,
    )
    policy = face_policy.policy_from_census(
        case_id=output_dir.name,
        source_video_hash=source_info.sha256,
        source_frame_count=source_frame_count,
        census=draft_census,
        mode=face_policy.MODE_REDACT_ALL,
    )
    write_json(output_dir / "privacy_policy.json", policy.to_dict())
    census = face_policy.build_track_census(
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        source_video_hash=source_info.sha256,
        source_frame_count=source_frame_count,
        policy=policy,
    )
    write_json(output_dir / "track_census.json", census)
    compile_result = face_policy.compile_policy(
        input_video,
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        policy=policy,
    )
    privacy_dir = output_dir / "privacy_regions"
    privacy_summary = face_policy.write_compiled_outputs(
        privacy_dir,
        result=compile_result,
        policy=policy,
        census=census,
        source_video_hash=source_info.sha256,
        tracks_hash=media.sha256_file(tracking_dir / "tracks.jsonl"),
        track_quality_hash=media.sha256_file(tracking_dir / "track_quality.json"),
        overwrite=overwrite,
    )
    audit = {
        "compiler_id": face_policy.COMPILER_ID,
        "detection_input_sha256": media.sha256_file(detections_path),
        "final_export_status": {
            "blocking_reasons": ["review_required_policy_or_track_intervals"],
            "status": "blocked_review_required",
        },
        "policy": policy.to_dict(),
        "privacy_summary": privacy_summary,
        "schema_version": SCHEMA_VERSION,
        "source_manifest": source_manifest,
        "tracking_summary": track_summary,
    }
    write_json(output_dir / "audit_manifest.json", audit)
    return audit


def face_region_records_from_render_input(
    path: Path,
    *,
    case_frame_offset: int,
    parent_frame_offset: int,
) -> list[PrivacyRegionRecord]:
    face_input = rendering.load_privacy_regions(path)
    records: list[PrivacyRegionRecord] = []
    for frame_index, regions in face_input.regions_by_frame.items():
        for index, region in enumerate(regions):
            if not region.is_renderable:
                continue
            source_record = dict(region.source_record)
            track_id = face_track_identifier(source_record, fallback=f"{frame_index}_{index}")
            region_id = source_record.get("region_id") or f"face_{frame_index}_{index}"
            records.append(
                PrivacyRegionRecord(
                    approval_state=APPROVAL_ACCEPTED,
                    bbox_xyxy=region.bbox_xyxy,
                    category=CATEGORY_FACE,
                    object_id=f"face_track_{track_id}",
                    provenance=region.provenance,
                    render_eligible=True,
                    source_frame=case_frame_offset + frame_index,
                    source_record=source_record
                    | {
                        "parent_source_frame": parent_frame_offset + frame_index,
                        "source_clip_frame": frame_index,
                        "region_id": region_id,
                        "track_id": track_id,
                    },
                )
            )
    return records


def face_track_identifier(source_record: dict[str, Any], *, fallback: str) -> str:
    if source_record.get("track_id") is not None:
        return str(source_record["track_id"])
    if source_record.get("track_uid") is not None:
        return str(source_record["track_uid"])
    target_track_ids = source_record.get("target_track_ids")
    if isinstance(target_track_ids, list) and target_track_ids:
        return str(target_track_ids[0])
    metrics = source_record.get("metrics")
    if isinstance(metrics, dict) and metrics.get("source_track_id") is not None:
        return str(metrics["source_track_id"])
    return fallback


def with_parent_source_frame(
    record: PrivacyRegionRecord,
    *,
    parent_frame_offset: int,
) -> PrivacyRegionRecord:
    return PrivacyRegionRecord(
        approval_state=record.approval_state,
        bbox_xyxy=record.bbox_xyxy,
        category=record.category,
        object_id=record.object_id,
        provenance=record.provenance,
        reasons=record.reasons,
        render_eligible=record.render_eligible,
        source_frame=record.source_frame,
        source_record=dict(record.source_record)
        | {"parent_source_frame": parent_frame_offset + record.source_frame},
    )


def render_combined_current_result(
    *,
    canonical_source: Path,
    output_video: Path,
    records: list[PrivacyRegionRecord],
    overwrite: bool,
) -> dict[str, Any]:
    source_info = media.probe_media(canonical_source)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    settings = rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_fraction=0.15,
        margin_pixels=0,
        pixel_size=16,
        blur_kernel=51,
        preview_watermark=False,
    )
    rendered_frames, rendered_regions, audio_handling = privacy_rendering.render_privacy_video(
        canonical_source,
        output_video,
        privacy_rendering.regions_by_frame(records),
        settings=settings,
        fps=fps,
        width=source_info.video_stream.width,
        height=source_info.video_stream.height,
        source_has_audio=source_info.audio_stream is not None,
        overwrite=overwrite,
    )
    media_validation = rendering.validate_output_media(
        canonical_source,
        source_info,
        output_video,
        expected_frame_count=rendered_frames,
        expected_width=source_info.video_stream.width,
        expected_height=source_info.video_stream.height,
        expected_fps=fps,
        source_has_audio=source_info.audio_stream is not None,
        settings=settings,
    )
    region_validation = privacy_rendering.verify_rendered_regions(
        canonical_source,
        output_video,
        [record.to_render_region() for record in records],
        settings=settings,
        frame_width=source_info.video_stream.width,
        frame_height=source_info.video_stream.height,
    )
    return {
        "audio_handling": audio_handling,
        "media_validation": media_validation,
        "output_sha256": media.sha256_file(output_video),
        "render_configuration": {
            "blur_kernel": 51,
            "margin_fraction": 0.15,
            "margin_pixels": 0,
            "method": "pixelate",
            "pixel_format": "yuv420p",
            "pixel_size": 16,
            "video_codec": "libx264",
        },
        "rendered_frame_count": rendered_frames,
        "rendered_region_count": rendered_regions,
        "rendered_region_verification": region_validation,
    }


def build_night_case(package_root: Path, v1_package_root: Path) -> dict[str, Any]:
    source = v1_package_root / "media" / "night" / "original_reference.mp4"
    current = v1_package_root / "media" / "night" / "combined_face_plate_demo_preview.mp4"
    dst_source = package_root / "media" / "night" / "original_reference.mp4"
    dst_current = package_root / "media" / "night" / "current_combined_result.mp4"
    copy_if_exists(source, dst_source)
    copy_if_exists(current, dst_current)
    source_info = media.probe_media(dst_source)
    frame_count = int(source_info.video_stream.frame_count or 0)
    roles = (
        case_manifests.VideoRole(
            label="Original",
            path="media/night/original_reference.mp4",
            role="original",
            sha256=source_info.sha256,
            source_frame_end=frame_count - 1,
            source_frame_start=0,
        ),
        case_manifests.VideoRole(
            label="Current result",
            path="media/night/current_combined_result.mp4",
            role="current_result",
            sha256=media.sha256_file(dst_current),
            source_frame_end=frame_count - 1,
            source_frame_start=0,
        ),
    )
    manifest = case_manifests.CaseManifest(
        case_id="night",
        display_name="Night case",
        duration_seconds=source_info.duration_seconds,
        face_evidence_inputs=(
            "artifacts/face_full_video_acceptance_v2/night/reviewed_candidate_privacy_regions.jsonl",
        ),
        fps=source_info.video_stream.nominal_frame_rate or "30000/1001",
        frame_count=frame_count,
        height=source_info.video_stream.height,
        lineage={
            "source": "compatible full-night source validated in final_pipeline_package_v1",
            "source_file": "artifacts/eval_night_selective.mp4",
        },
        plate_evidence_inputs=(
            "artifacts/shared_temporal_evidence_v1/night/tracking/tracks.jsonl",
        ),
        reviewed_correction_inputs=(
            "artifacts/face_full_video_acceptance_v2/night/reviewed_candidate_privacy_regions.jsonl",
        ),
        source_frame_end=frame_count - 1,
        source_frame_start=0,
        source_path="media/night/original_reference.mp4",
        source_video_hash=source_info.sha256,
        status=case_manifests.CASE_STATUS_READY,
        video_roles=roles,
        width=source_info.video_stream.width,
    )
    case_path = package_root / "cases" / "night.json"
    case_manifests.write_case_manifest(case_path, manifest)
    write_night_region_records(package_root / "regions" / "night_current_regions.jsonl")
    return {
        "case_manifest_path": case_path.relative_to(package_root).as_posix(),
        "source_hash": source_info.sha256,
        "source_frames": frame_count,
    }


def write_night_region_records(output_path: Path) -> None:
    face_input = rendering.load_privacy_regions(
        Path(
            "artifacts/face_full_video_acceptance_v2/night/reviewed_candidate_privacy_regions.jsonl"
        )
    )
    records: list[PrivacyRegionRecord] = []
    for frame_index, regions in face_input.regions_by_frame.items():
        for index, region in enumerate(regions):
            if not region.is_renderable:
                continue
            source_record = dict(region.source_record)
            track_id = face_track_identifier(source_record, fallback=f"{frame_index}_{index}")
            records.append(
                PrivacyRegionRecord(
                    approval_state="accepted",
                    bbox_xyxy=region.bbox_xyxy,
                    category=CATEGORY_FACE,
                    object_id=f"face_track_{track_id}",
                    provenance=region.provenance,
                    render_eligible=True,
                    source_frame=frame_index,
                    source_record=source_record
                    | {
                        "region_id": f"face_{frame_index}_{index}",
                        "track_id": track_id,
                    },
                )
            )
    plate_rows = plate_rendering.load_track_rows(
        Path("artifacts/shared_temporal_evidence_v1/night/tracking/tracks.jsonl")
    )
    plate_regions = plate_rendering.eligible_regions_from_rows(
        plate_rows,
        config=plate_rendering.default_config(),
    )
    records.extend(region.to_privacy_region() for region in plate_regions)
    review_finalization.write_privacy_region_records(output_path, records)


def render_correction_interval_preview(
    package_root: Path,
    *,
    session_dir: Path,
    correction: Any,
    include_unsaved_correction: bool = False,
) -> dict[str, Any]:
    manifest_path = package_root / "cases" / f"{correction.case_id}.json"
    if not manifest_path.exists():
        return {"ok": False, "reason": "case manifest missing"}
    manifest = case_manifests.load_case_manifest(manifest_path)
    source_path = review_finalization.resolve_package_path(package_root, manifest.source_path)
    if source_path is None or not source_path.exists():
        return {"ok": False, "reason": "case source missing"}
    local_start = max(0, correction.source_frame_start - manifest.source_frame_start)
    local_end = max(local_start, correction.source_frame_end - manifest.source_frame_start)
    fps = rational_to_float(manifest.fps) or 30.0
    preview_dir = session_dir / "previews" / correction.case_id / correction.correction_id
    preview_dir.mkdir(parents=True, exist_ok=True)
    original_preview = preview_dir / "original_interval.mp4"
    corrected_preview = preview_dir / "corrected_interval.mp4"
    media.trim_video(
        source_path,
        original_preview,
        start_seconds=local_start / fps,
        duration_seconds=(local_end - local_start + 1) / fps,
        overwrite=True,
    )
    base_regions_path = package_root / "regions" / f"{correction.case_id}_current_regions.jsonl"
    if not base_regions_path.exists():
        return {
            "ok": False,
            "reason": "current privacy-region provenance missing",
        }
    base_regions = review_finalization.load_privacy_region_records(base_regions_path)
    correction_records = review_finalization.load_corrections_jsonl(
        session_dir / "corrections.jsonl"
    )
    if include_unsaved_correction:
        correction_records = [*correction_records, correction]
    corrections = review_finalization.latest_corrections(
        correction_records, case_id=correction.case_id
    )
    final_regions, _ = review_finalization.apply_corrections(
        base_regions,
        corrections,
        case_id=correction.case_id,
        source_frame_offset=manifest.source_frame_start,
    )
    preview_regions = [
        replace(record, source_frame=record.source_frame - local_start)
        for record in final_regions
        if local_start <= record.source_frame <= local_end
    ]
    source_info = media.probe_media(original_preview)
    settings = rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_fraction=0.15,
        margin_pixels=0,
        pixel_size=16,
        blur_kernel=51,
        preview_watermark=False,
    )
    rendered_frames, rendered_regions, audio_handling = privacy_rendering.render_privacy_video(
        original_preview,
        corrected_preview,
        privacy_rendering.regions_by_frame(preview_regions),
        settings=settings,
        fps=fps,
        width=source_info.video_stream.width,
        height=source_info.video_stream.height,
        source_has_audio=source_info.audio_stream is not None,
        overwrite=True,
    )
    latest_reviewed = preview_dir.parent / "latest_reviewed.mp4"
    full_source_info = media.probe_media(source_path)
    privacy_rendering.render_privacy_video(
        source_path,
        latest_reviewed,
        privacy_rendering.regions_by_frame(final_regions),
        settings=settings,
        fps=fps,
        width=full_source_info.video_stream.width,
        height=full_source_info.video_stream.height,
        source_has_audio=full_source_info.audio_stream is not None,
        overwrite=True,
    )
    return {
        "audio_handling": audio_handling,
        "corrected_preview": corrected_preview.relative_to(package_root).as_posix(),
        "frame_end": correction.source_frame_end,
        "frame_start": correction.source_frame_start,
        "latest_reviewed": latest_reviewed.relative_to(package_root).as_posix(),
        "ok": True,
        "original_preview": original_preview.relative_to(package_root).as_posix(),
        "rendered_frames": rendered_frames,
        "rendered_regions": rendered_regions,
        "schema_version": "correction_interval_preview.v1",
    }


def write_entity_index(package_root: Path) -> dict[str, Any]:
    entities_by_case: dict[str, list[dict[str, Any]]] = {}
    unresolved_by_case: dict[str, dict[str, Any]] = {}
    thumbnail_audit_by_case: dict[str, list[dict[str, Any]]] = {}
    generation_audit_by_case: dict[str, Any] = {}
    metrics_by_case: dict[str, Any] = {}
    merge_audit_by_case: dict[str, Any] = {}
    cluster_audit_by_case: dict[str, Any] = {}
    cannot_link_audit_by_case: dict[str, Any] = {}
    pairwise_audit_by_case: dict[str, Any] = {}
    scores_by_case: dict[str, list[dict[str, Any]]] = {}
    profile = review_entity_profile()
    for case_id in ("day", "night"):
        region_path = package_root / "regions" / f"{case_id}_current_regions.jsonl"
        if not region_path.exists():
            entities_by_case[case_id] = []
            unresolved_by_case[case_id] = {
                "fragment_count": 0,
                "fragments": [],
                "schema_version": "review_unresolved_evidence.v1",
            }
            thumbnail_audit_by_case[case_id] = []
            generation_audit_by_case[case_id] = empty_entity_generation_audit(case_id)
            metrics_by_case[case_id] = empty_entity_metrics(case_id)
            merge_audit_by_case[case_id] = empty_entity_merge_audit(case_id)
            cluster_audit_by_case[case_id] = empty_entity_merge_audit(case_id)
            cannot_link_audit_by_case[case_id] = {"case_id": case_id, "pairs": []}
            pairwise_audit_by_case[case_id] = {"case_id": case_id, "pairs": []}
            scores_by_case[case_id] = []
            write_json(
                package_root / "entities" / f"{case_id}.json",
                {"case_id": case_id, "entities": [], "schema_version": "privacy_entities.v1"},
            )
            continue
        records = review_finalization.load_privacy_region_records(region_path)
        manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
        tracklets = review_entity_tracklets(case_id, records, manifest, package_root)
        previous_entities = privacy_entities.entities_from_region_records(case_id, records)
        accumulation = ReviewEntityAccumulator(profile).accumulate(
            case_id=case_id,
            tracklets=tracklets,
            previous_user_entity_count=len(previous_entities),
        )
        entities: list[privacy_entities.PrivacyEntity] = []
        for item in accumulation.user_entities:
            entities.append(privacy_entity_from_accumulated(case_id, item, entities))
        entity_payloads = [
            review_entity_payload(
                entity=entity,
                records=records,
                manifest=manifest,
                package_root=package_root,
            )
            for entity in entities
        ]
        payload_by_track = {
            track_id: payload
            for payload in entity_payloads
            for track_id in payload.get("automatic_source_track_ids", [])
        }
        for payload in entity_payloads:
            track_ids = payload.get("automatic_source_track_ids", [])
            payload["review_entity_state"] = accumulation.entity_state_for_tracks(track_ids)
            payload["evidence_summary"] = payload.get(
                "evidence_summary", {}
            ) | accumulation.summary_for_tracks(track_ids)
            payload["attached_fragment_track_ids"] = sorted(
                accumulation.attached_fragments_for_tracks(track_ids)
            )
            payload["possible_duplicate"] = bool(
                accumulation.possible_duplicates_for_tracks(track_ids)
            )
            payload["possible_duplicate_suggestions"] = accumulation.possible_duplicates_for_tracks(
                track_ids
            )
        unresolved = [
            unresolved_fragment_payload(fragment, manifest, package_root)
            for fragment in accumulation.unresolved_fragments
        ]
        thumbnail_audit_by_case[case_id] = [
            thumbnail_audit_payload(entity) for entity in entity_payloads
        ]
        generation_audit_by_case[case_id] = entity_generation_audit(
            case_id=case_id,
            tracklets=tracklets,
            previous_entities=previous_entities,
            accumulation=accumulation,
        )
        metrics_by_case[case_id] = accumulation.metrics()
        merge_audit_by_case[case_id] = accumulation.merge_audit()
        cluster_audit_by_case[case_id] = accumulation.cluster_audit()
        cannot_link_audit_by_case[case_id] = accumulation.cannot_link_audit()
        pairwise_audit_by_case[case_id] = accumulation.pairwise_audit()
        scores_by_case[case_id] = [tracklet.to_dict() for tracklet in accumulation.tracklets]
        contact_sheets = write_entity_contact_sheets(
            package_root=package_root,
            manifest=manifest,
            accumulation=accumulation,
            payload_by_track=payload_by_track,
        )
        generation_audit_by_case[case_id]["contact_sheets"] = contact_sheets
        payload = {
            "case_id": case_id,
            "entities": entity_payloads,
            "entity_count": len(entities),
            "previous_user_facing_entity_count": len(previous_entities),
            "review_entity_profile_hash": stable_hash(profile),
            "schema_version": "review_entities.v2",
            "unresolved_evidence": {
                "fragments": unresolved,
                "fragment_count": len(unresolved),
                "schema_version": "review_unresolved_evidence.v1",
            },
        }
        entities_by_case[case_id] = payload["entities"]
        unresolved_by_case[case_id] = payload["unresolved_evidence"]
        write_json(package_root / "entities" / f"{case_id}.json", payload)
        write_json(
            package_root / "entities" / f"{case_id}_unresolved_evidence.json",
            payload["unresolved_evidence"],
        )
    write_json(
        package_root / "privacy_entities.json",
        {
            "cases": entities_by_case,
            "schema_version": "review_entities_index.v2",
            "unresolved_evidence_by_case": unresolved_by_case,
        },
    )
    write_json(
        package_root / "review_objects.json",
        {
            "cases": entities_by_case,
            "schema_version": "review_object_registry.v2",
            "unresolved_evidence_by_case": unresolved_by_case,
        },
    )
    write_json(
        package_root / "review_entities.json",
        {
            "cases": entities_by_case,
            "schema_version": "review_entity_registry.v2",
            "unresolved_evidence_by_case": unresolved_by_case,
        },
    )
    write_json(
        package_root / "review_entity_evidence_profile.json",
        {
            "configuration_hash": stable_hash(profile),
            "profile": profile,
            "schema_version": "review_entity_evidence_profile.v1",
        },
    )
    shutil.copy2(
        package_root / "review_entity_evidence_profile.json",
        package_root / "review_entity_profile.json",
    )
    write_text(
        package_root / "review_entity_evidence_scores.jsonl",
        "\n".join(
            json.dumps({"case_id": case_id, **score}, sort_keys=True)
            for case_id, rows in sorted(scores_by_case.items())
            for score in rows
        )
        + "\n",
    )
    write_json(
        package_root / "entity_generation_audit.json",
        {
            "cases": generation_audit_by_case,
            "created_at_utc": media.utc_timestamp(),
            "root_cause": (
                "v2.3.2 created one user-facing review entity for each source "
                "track/region group, so short weak fragments and duplicate fragments "
                "were presented as named people or plates."
            ),
            "schema_version": "review_entity_generation_audit.v1",
        },
    )
    write_json(
        package_root / "review_entity_metrics.json",
        {
            "cases": metrics_by_case,
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_metrics.v1",
        },
    )
    write_json(
        package_root / "review_entity_before_after_metrics.json",
        {
            "cases": metrics_by_case,
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_before_after_metrics.v1",
        },
    )
    write_json(
        package_root / "review_entity_merge_audit.json",
        {
            "cases": merge_audit_by_case,
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_merge_audit.v1",
        },
    )
    write_json(
        package_root / "review_entity_clusters.json",
        {
            "cases": cluster_audit_by_case,
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_clusters.v1",
        },
    )
    write_json(
        package_root / "review_entity_pairwise_audit.json",
        {
            "cases": pairwise_audit_by_case,
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_pairwise_audit.v1",
        },
    )
    write_json(
        package_root / "review_entity_cannot_link_audit.json",
        {
            "cases": cannot_link_audit_by_case,
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_cannot_link_audit.v1",
        },
    )
    write_json(
        package_root / "review_entity_consolidation_audit.json",
        {
            "cases": {
                case_id: {
                    "clusters": cluster_audit_by_case.get(case_id, {}),
                    "generation": generation_audit_by_case.get(case_id, {}),
                    "metrics": metrics_by_case.get(case_id, {}),
                    "pairwise": pairwise_audit_by_case.get(case_id, {}),
                }
                for case_id in sorted(metrics_by_case)
            },
            "created_at_utc": media.utc_timestamp(),
            "schema_version": "review_entity_consolidation_audit.v1",
        },
    )
    audit = {
        "cases": thumbnail_audit_by_case,
        "created_at_utc": media.utc_timestamp(),
        "schema_version": "representative_thumbnail_audit.v1",
        "summary": {
            case_id: {
                "entity_count": len(items),
                "warning_count": sum(1 for item in items if item.get("warnings")),
            }
            for case_id, items in thumbnail_audit_by_case.items()
        },
    }
    write_json(package_root / "representative_thumbnail_audit.json", audit)
    write_json(package_root / "review_entity_thumbnail_audit.json", audit)
    write_text(package_root / "before_after_summary.md", before_after_summary(metrics_by_case))
    return audit


def apply_review_entity_coverage_patch(package_root: Path) -> dict[str, Any]:
    index_path = package_root / "review_entities.json"
    if not index_path.exists():
        return {"cases": {}, "schema_version": "review_entity_coverage_audit.v1"}
    index = read_json_file(index_path)
    cases = dict(index.get("cases") or {})
    audit_cases: dict[str, Any] = {}
    changed = False
    for case_id, entities_value in sorted(cases.items()):
        case_path = package_root / "cases" / f"{case_id}.json"
        if not case_path.exists():
            continue
        manifest = case_manifests.load_case_manifest(case_path)
        entities = [dict(entity) for entity in list(entities_value or [])]
        before_count = len(entities)
        case_changed = False
        representative_bbox_enriched_entity_ids: list[str] = []
        for entity in entities:
            bbox = representative_bbox(entity)
            if bbox and entity.get("representative_bbox") != bbox:
                entity["representative_bbox"] = bbox
                representative_bbox_enriched_entity_ids.append(str(entity.get("entity_id") or ""))
                case_changed = True
                changed = True
        policy_rows = face_policy_coverage_rows(
            package_root=package_root,
            case_id=case_id,
            manifest=manifest,
            entities=entities,
        )
        added_keep_track_ids: set[str] = set()
        for row in policy_rows:
            if (
                row["policy"] == privacy_entities.POLICY_KEEP_VISIBLE
                and row["assignment_provenance"] == "explicit_case_policy"
                and row["valid_source_evidence"]
                and row["review_entity_uuid"] is None
            ):
                entity = source_backed_keep_entity_payload(
                    package_root=package_root,
                    case_id=case_id,
                    manifest=manifest,
                    existing=entities,
                    row=row,
                )
                entities.append(entity)
                row["review_entity_uuid"] = entity["entity_id"]
                row["review_entity_added"] = True
                added_keep_track_ids.add(str(row["track_id"]))
                case_changed = True
                changed = True
            if (
                row["policy"] == privacy_entities.POLICY_KEEP_VISIBLE
                and row["assignment_provenance"] == "explicit_case_policy"
                and row["valid_source_evidence"]
                and row["review_entity_uuid"] is not None
            ):
                packaged_source = copy_face_policy_source_evidence(
                    package_root=package_root,
                    case_id=case_id,
                    source_dir=Path(str(row["face_policy_source"])),
                )
                for entity in entities:
                    if entity.get("entity_id") == row["review_entity_uuid"]:
                        provenance = dict(entity.get("created_provenance") or {})
                        if provenance.get("face_policy_source") != packaged_source:
                            entity["created_provenance"] = provenance | {
                                "face_policy_source": packaged_source
                            }
                            case_changed = True
                            changed = True
        refreshed_rows = face_policy_coverage_rows(
            package_root=package_root,
            case_id=case_id,
            manifest=manifest,
            entities=entities,
        )
        missing_source_backed = [
            row
            for row in refreshed_rows
            if row["valid_source_evidence"]
            and row["policy"]
            in {privacy_entities.POLICY_KEEP_VISIBLE, privacy_entities.POLICY_REDACT}
            and row["review_entity_uuid"] is None
        ]
        missing_keep = [
            row
            for row in missing_source_backed
            if row["policy"] == privacy_entities.POLICY_KEEP_VISIBLE
            and row["assignment_provenance"] == "explicit_case_policy"
        ]
        if missing_keep:
            missing = ", ".join(str(row["track_id"]) for row in missing_keep)
            raise ValueError(
                f"explicit keep-policy face tracks omitted from review entity registry: {case_id} {missing}"
            )
        cases[case_id] = entities
        audit_cases[case_id] = {
            "after_entity_count": len(entities),
            "before_entity_count": before_count,
            "explicit_keep_tracks_added": [
                row
                for row in refreshed_rows
                if row["policy"] == privacy_entities.POLICY_KEEP_VISIBLE
                and str(row["track_id"]) in added_keep_track_ids
            ],
            "explicit_keep_tracks": [
                row
                for row in refreshed_rows
                if row["policy"] == privacy_entities.POLICY_KEEP_VISIBLE
            ],
            "face_census_tracks": refreshed_rows,
            "missing_source_backed_tracks": missing_source_backed,
            "redacted_tracks": [
                row for row in refreshed_rows if row["policy"] == privacy_entities.POLICY_REDACT
            ],
            "representative_bbox_enriched_entity_ids": representative_bbox_enriched_entity_ids,
            "schema_version": "review_entity_coverage_case_audit.v1",
            "sampled_false_merge_audit": {
                "sampled_suspected_false_merge_count": 0,
                "strategy": "No automatic visual-similarity merge was introduced for this coverage patch.",
            },
        }
        if case_changed:
            write_json(
                package_root / "entities" / f"{case_id}.json",
                {
                    "case_id": case_id,
                    "entities": entities,
                    "entity_count": len(entities),
                    "schema_version": "review_entities.v2",
                    "unresolved_evidence": read_json_file(
                        package_root / "entities" / f"{case_id}_unresolved_evidence.json"
                    )
                    if (package_root / "entities" / f"{case_id}_unresolved_evidence.json").exists()
                    else {
                        "fragment_count": 0,
                        "fragments": [],
                        "schema_version": "review_unresolved_evidence.v1",
                    },
                },
            )
    if changed:
        index["cases"] = cases
        for registry_name in (
            "review_entities.json",
            "review_objects.json",
            "privacy_entities.json",
        ):
            registry_path = package_root / registry_name
            if registry_path.exists():
                registry = read_json_file(registry_path)
                registry["cases"] = cases
                write_json(registry_path, registry)
    audit = {
        "cases": audit_cases,
        "created_at_utc": media.utc_timestamp(),
        "summary": {
            "after_counts": {
                case_id: case["after_entity_count"] for case_id, case in sorted(audit_cases.items())
            },
            "before_counts": {
                case_id: case["before_entity_count"]
                for case_id, case in sorted(audit_cases.items())
            },
            "entity_count_delta": {
                case_id: case["after_entity_count"] - case["before_entity_count"]
                for case_id, case in sorted(audit_cases.items())
            },
            "explicit_keep_tracks": {
                case_id: case["explicit_keep_tracks"]
                for case_id, case in sorted(audit_cases.items())
            },
            "explicit_keep_tracks_added": {
                case_id: case["explicit_keep_tracks_added"]
                for case_id, case in sorted(audit_cases.items())
            },
            "missing_source_backed_tracks": {
                case_id: case["missing_source_backed_tracks"]
                for case_id, case in sorted(audit_cases.items())
            },
            "representative_bbox_enriched": {
                case_id: case["representative_bbox_enriched_entity_ids"]
                for case_id, case in sorted(audit_cases.items())
            },
            "sampled_false_merge_audit": {
                case_id: case["sampled_false_merge_audit"]
                for case_id, case in sorted(audit_cases.items())
            },
        },
        "passed": not any(
            case.get("missing_source_backed_tracks")
            for case in audit_cases.values()
            if any(
                row["policy"] == privacy_entities.POLICY_KEEP_VISIBLE
                and row["assignment_provenance"] == "explicit_case_policy"
                for row in case.get("missing_source_backed_tracks", [])
            )
        ),
        "root_cause": (
            "The v1 registry was derived from rendered privacy regions. Explicit keep-policy face "
            "tracks are intentionally not rendered, so source-backed keep tracks could be absent "
            "from reviewer-facing People."
        ),
        "schema_version": "review_entity_coverage_audit.v1",
    }
    write_json(package_root / "review_entity_coverage_audit.json", audit)
    return audit


def write_entity_focus_overlay_audit(package_root: Path) -> dict[str, Any]:
    index_path = package_root / "review_entities.json"
    if not index_path.exists():
        audit = {
            "cases": {},
            "created_at_utc": media.utc_timestamp(),
            "passed": True,
            "schema_version": "entity_focus_overlay_audit.v1",
        }
        write_json(package_root / "entity_focus_overlay_audit.json", audit)
        return audit
    index = read_json_file(index_path)
    cases = dict(index.get("cases") or {})
    audit_cases: dict[str, Any] = {}
    changed = False
    for case_id, entities_value in sorted(cases.items()):
        case_path = package_root / "cases" / f"{case_id}.json"
        if not case_path.exists():
            continue
        manifest = case_manifests.load_case_manifest(case_path)
        entities = [dict(entity) for entity in list(entities_value or [])]
        enriched_ids: list[str] = []
        missing: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for entity in entities:
            entity_id = str(entity.get("entity_id") or "")
            label = str(entity.get("label") or entity.get("display_name") or entity_id)
            bbox = representative_bbox(entity)
            frame = entity.get("representative_source_frame")
            source = "existing_representative_bbox" if bbox else None
            if bbox is None or not isinstance(frame, int):
                fallback = focus_representative_from_source_evidence(
                    package_root=package_root,
                    manifest=manifest,
                    entity=entity,
                )
                if fallback is not None:
                    if bbox is None:
                        entity["representative_bbox"] = fallback["bbox_xyxy"]
                        bbox = list(fallback["bbox_xyxy"])
                        enriched_ids.append(entity_id)
                        changed = True
                    if not isinstance(frame, int):
                        entity["representative_source_frame"] = fallback[
                            "representative_source_frame"
                        ]
                        frame = int(fallback["representative_source_frame"])
                        changed = True
                    source = str(fallback["source"])
            ready = bbox is not None and isinstance(frame, int)
            row = {
                "category": entity.get("category"),
                "current_policy": entity.get("current_policy") or entity.get("privacy_action"),
                "entity_id": entity_id,
                "focus_ready": ready,
                "label": label,
                "representative_bbox": bbox,
                "representative_source_frame": frame if isinstance(frame, int) else None,
                "source": source,
            }
            rows.append(row)
            if not ready:
                missing.append(
                    {
                        "category": entity.get("category"),
                        "entity_id": entity_id,
                        "label": label,
                        "reason": "no_source_backed_representative_geometry",
                    }
                )
        cases[case_id] = entities
        audit_cases[case_id] = {
            "enriched_entity_ids": enriched_ids,
            "focus_ready_count": sum(1 for row in rows if row["focus_ready"]),
            "missing_entities": missing,
            "missing_geometry_count": len(missing),
            "rows": rows,
            "schema_version": "entity_focus_overlay_case_audit.v1",
            "total_entities": len(entities),
        }
        if enriched_ids:
            entity_payload_path = package_root / "entities" / f"{case_id}.json"
            if entity_payload_path.exists():
                payload = read_json_file(entity_payload_path)
                payload["entities"] = entities
                payload["entity_count"] = len(entities)
                write_json(entity_payload_path, payload)
    if changed:
        index["cases"] = cases
        for registry_name in (
            "review_entities.json",
            "review_objects.json",
            "privacy_entities.json",
        ):
            registry_path = package_root / registry_name
            if registry_path.exists():
                registry = read_json_file(registry_path)
                registry["cases"] = cases
                write_json(registry_path, registry)
    audit = {
        "cases": audit_cases,
        "created_at_utc": media.utc_timestamp(),
        "passed": True,
        "schema_version": "entity_focus_overlay_audit.v1",
        "summary": {
            "enriched_entity_ids": {
                case_id: case["enriched_entity_ids"]
                for case_id, case in sorted(audit_cases.items())
            },
            "focus_ready_count": {
                case_id: case["focus_ready_count"] for case_id, case in sorted(audit_cases.items())
            },
            "missing_geometry_count": {
                case_id: case["missing_geometry_count"]
                for case_id, case in sorted(audit_cases.items())
            },
            "total_entities": {
                case_id: case["total_entities"] for case_id, case in sorted(audit_cases.items())
            },
        },
    }
    write_json(package_root / "entity_focus_overlay_audit.json", audit)
    return audit


def focus_representative_from_source_evidence(
    *,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    entity: dict[str, Any],
) -> dict[str, Any] | None:
    preferred_source_frame = entity.get("representative_source_frame")
    preferred_case_frame = (
        int(preferred_source_frame) - manifest.source_frame_start
        if isinstance(preferred_source_frame, int)
        else None
    )
    source_records = review_working_state.source_backed_entity_regions(
        package_root=package_root,
        manifest=manifest,
        entity=entity,
    )
    if source_records:
        selected = select_focus_record(source_records, preferred_case_frame)
        return {
            "bbox_xyxy": list(selected.bbox_xyxy),
            "representative_source_frame": manifest.source_frame_start + selected.source_frame,
            "source": "source_policy_track_geometry",
        }
    region_path = package_root / "regions" / f"{manifest.case_id}_current_regions.jsonl"
    if region_path.exists():
        baseline_records = [
            record
            for record in review_finalization.load_privacy_region_records(region_path)
            if review_working_state.region_belongs_to_entity(record, entity)
        ]
        if baseline_records:
            selected = select_focus_record(baseline_records, preferred_case_frame)
            return {
                "bbox_xyxy": list(selected.bbox_xyxy),
                "representative_source_frame": manifest.source_frame_start + selected.source_frame,
                "source": "automatic_privacy_region",
            }
    return None


def select_focus_record(
    records: list[PrivacyRegionRecord],
    preferred_case_frame: int | None,
) -> PrivacyRegionRecord:
    if preferred_case_frame is None:
        return sorted(records, key=focus_record_sort_key)[0]
    return min(
        records,
        key=lambda record: (
            abs(record.source_frame - preferred_case_frame),
            focus_record_sort_key(record),
        ),
    )


def focus_record_sort_key(record: PrivacyRegionRecord) -> tuple[int, int]:
    x1, y1, x2, y2 = record.bbox_xyxy
    area = max(1, x2 - x1) * max(1, y2 - y1)
    return (-area, record.source_frame)


def write_merge_audit_shells(package_root: Path) -> None:
    write_json(
        package_root / "merged_entity_naming_audit.json",
        {
            "created_at_utc": media.utc_timestamp(),
            "events": [],
            "schema_version": "merged_entity_naming_audit.v1",
        },
    )
    write_json(
        package_root / "merged_entity_policy_audit.json",
        {
            "created_at_utc": media.utc_timestamp(),
            "events": [],
            "schema_version": "merged_entity_policy_audit.v1",
        },
    )


def write_manual_correction_persistence_audit_shell(package_root: Path) -> None:
    write_json(
        package_root / "manual_correction_persistence_audit.json",
        {
            "created_at_utc": media.utc_timestamp(),
            "events": [],
            "schema_version": "manual_correction_persistence_audit.v1",
        },
    )


def append_merged_entity_audits(
    package_root: Path,
    *,
    case_id: str,
    canonical: dict[str, Any],
    retired_entities: list[dict[str, Any]],
) -> None:
    naming_path = package_root / "merged_entity_naming_audit.json"
    policy_path = package_root / "merged_entity_policy_audit.json"
    if not naming_path.exists() or not policy_path.exists():
        write_merge_audit_shells(package_root)
    retired_ids = [str(entity.get("entity_id") or "") for entity in retired_entities]
    source_track_ids = unique_sorted(
        item
        for entity in [canonical, *retired_entities]
        for item in list(entity.get("automatic_source_track_ids") or [])
    )
    naming = read_json_file(naming_path)
    naming.setdefault("events", []).append(
        {
            "case_id": case_id,
            "category": canonical.get("category"),
            "display_label": canonical.get("label") or canonical.get("display_name"),
            "merged_display_sequence": canonical.get("merged_display_sequence"),
            "merged_entity_id": canonical.get("entity_id"),
            "retired_entity_ids": retired_ids,
            "schema_version": "merged_entity_naming_event.v1",
            "source_track_ids": source_track_ids,
        }
    )
    write_json(naming_path, naming)
    policy = read_json_file(policy_path)
    policy.setdefault("events", []).append(
        {
            "case_id": case_id,
            "category": canonical.get("category"),
            "initialization": canonical.get("merge_policy_initialization"),
            "merged_entity_id": canonical.get("entity_id"),
            "resulting_policy": canonical.get("current_policy"),
            "retired_entity_policies": {
                str(entity.get("entity_id") or ""): review_working_state.inherited_policy(entity)
                for entity in retired_entities
            },
            "schema_version": "merged_entity_policy_event.v1",
        }
    )
    write_json(policy_path, policy)


def face_policy_coverage_rows(
    *,
    package_root: Path,
    case_id: str,
    manifest: case_manifests.CaseManifest,
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    track_to_entity = {
        str(track_id): entity.get("entity_id")
        for entity in entities
        for track_id in entity.get("automatic_source_track_ids", [])
    }
    rows: dict[str, dict[str, Any]] = {}
    for source_dir in face_policy_source_dirs(package_root, case_id, manifest):
        census = read_json_file(source_dir / "track_census.json")
        policy = (
            read_json_file(source_dir / "privacy_policy.json")
            if (source_dir / "privacy_policy.json").exists()
            else {}
        )
        assignments = {
            str(item.get("track_id")): item
            for item in policy.get("assignments", [])
            if item.get("track_id") is not None
        }
        for track in census.get("tracks", []):
            track_id = str(track.get("track_id"))
            if not track_id or track_id == "None":
                continue
            assignment = assignments.get(track_id, {})
            raw_policy = str(
                assignment.get("decision") or track.get("policy_assignment") or ""
            ).lower()
            normalized_policy = {
                "keep": privacy_entities.POLICY_KEEP_VISIBLE,
                "keep_visible": privacy_entities.POLICY_KEEP_VISIBLE,
                "redact": privacy_entities.POLICY_REDACT,
                "review": privacy_entities.POLICY_REVIEW_REQUIRED,
            }.get(raw_policy, raw_policy)
            if normalized_policy not in {
                privacy_entities.POLICY_KEEP_VISIBLE,
                privacy_entities.POLICY_REDACT,
                privacy_entities.POLICY_REVIEW_REQUIRED,
            }:
                continue
            observed = list(track.get("observed_frames") or [])
            predicted = list(track.get("predicted_frames") or [])
            interpolated = list(track.get("interpolated_frames") or [])
            valid_source_evidence = bool(observed or predicted or interpolated)
            row = {
                "assignment_provenance": str(
                    assignment.get("provenance") or track.get("assignment_provenance") or ""
                ),
                "face_policy_source": display_source_path(source_dir),
                "first_frame": track.get("first_frame"),
                "last_frame": track.get("last_frame"),
                "missing_source_backed_track": valid_source_evidence
                and normalized_policy
                in {privacy_entities.POLICY_KEEP_VISIBLE, privacy_entities.POLICY_REDACT}
                and track_to_entity.get(track_id) is None,
                "observed_frame_count": len(observed),
                "policy": normalized_policy,
                "quality_status": track.get("quality_status"),
                "representative_frame": track.get("representative_frame")
                if track.get("representative_frame") is not None
                else (observed or predicted or interpolated or [track.get("first_frame") or 0])[0],
                "review_entity_uuid": track_to_entity.get(track_id),
                "source_frame_end": manifest.source_frame_start + int(track.get("last_frame") or 0),
                "source_frame_start": manifest.source_frame_start
                + int(track.get("first_frame") or 0),
                "track_id": track_id,
                "valid_source_evidence": valid_source_evidence,
            }
            rows.setdefault(track_id, row)
    return sorted(
        rows.values(),
        key=lambda row: int(row["track_id"]) if str(row["track_id"]).isdigit() else 999999,
    )


def face_policy_source_dirs(
    package_root: Path,
    case_id: str,
    manifest: case_manifests.CaseManifest,
) -> list[Path]:
    candidates: list[Path] = []
    local = package_root / "technical" / f"{case_id}_pipeline" / "face_policy"
    if (local / "track_census.json").exists():
        candidates.append(local)
    artifacts_root = Path("artifacts")
    if artifacts_root.exists():
        candidates.extend(
            path.parent for path in artifacts_root.glob(f"*/{case_id}/**/track_census.json")
        )
    source_hash = manifest.source_video_hash
    matched: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not (candidate / "track_census.json").exists():
            continue
        seen.add(candidate)
        source_manifest_path = candidate / "source_manifest.json"
        census = read_json_file(candidate / "track_census.json")
        source_manifest = (
            read_json_file(source_manifest_path) if source_manifest_path.exists() else {}
        )
        candidate_hash = source_manifest.get("source_video_hash") or census.get("source_video_hash")
        if source_hash and candidate_hash and str(candidate_hash) != str(source_hash):
            continue
        if not (candidate / "privacy_policy.json").exists():
            continue
        matched.append(candidate)
    return sorted(matched)


def display_source_path(path: Path) -> str:
    resolved = path.resolve()
    for root in (Path.cwd().resolve(),):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def source_backed_keep_entity_payload(
    *,
    package_root: Path,
    case_id: str,
    manifest: case_manifests.CaseManifest,
    existing: list[dict[str, Any]],
    row: dict[str, Any],
) -> dict[str, Any]:
    track_id = str(row["track_id"])
    object_uuid = stable_hash(
        {
            "case_id": case_id,
            "category": CATEGORY_FACE,
            "schema_version": "review_source_backed_keep_entity_uuid.v1",
            "track_id": track_id,
        }
    )[:32]
    entity_id = f"entity_{object_uuid}"
    source_dir = Path(str(row["face_policy_source"]))
    packaged_source = copy_face_policy_source_evidence(
        package_root=package_root,
        case_id=case_id,
        source_dir=source_dir,
    )
    thumbnail_path = copy_source_backed_track_thumbnail(
        package_root=package_root,
        case_id=case_id,
        entity_id=entity_id,
        source_dir=source_dir,
        track_id=track_id,
    )
    representative_case_frame = int(row.get("representative_frame") or row.get("first_frame") or 0)
    representative_source_frame = manifest.source_frame_start + representative_case_frame
    representative_box = source_backed_track_bbox(
        source_dir=source_dir,
        track_id=track_id,
        preferred_frame=representative_case_frame,
    )
    return {
        "automatic_source_track_ids": [track_id],
        "case_id": case_id,
        "category": CATEGORY_FACE,
        "created_provenance": {
            "assignment_provenance": row.get("assignment_provenance"),
            "face_policy_source": packaged_source,
            "policy": privacy_entities.POLICY_KEEP_VISIBLE,
            "source": "review_entity_coverage_patch",
            "track_id": track_id,
        },
        "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
        "current_status": privacy_entities.STATUS_ACTIVE,
        "default_privacy_policy": privacy_entities.POLICY_KEEP_VISIBLE,
        "display_name": next_payload_entity_label(CATEGORY_FACE, existing),
        "entity_id": entity_id,
        "entity_uuid": object_uuid,
        "evidence_summary": {
            "face_census_track": True,
            "policy_assignment": privacy_entities.POLICY_KEEP_VISIBLE,
            "region_count": 0,
            "review_observation_count": int(row.get("observed_frame_count") or 0),
            "review_observed_span_frames": int(row.get("last_frame") or 0)
            - int(row.get("first_frame") or 0)
            + 1,
            "tracklet_count": 1,
        },
        "first_visible_source_frame": row.get("source_frame_start"),
        "geometry_locked": False,
        "label": next_payload_entity_label(CATEGORY_FACE, existing),
        "last_visible_source_frame": row.get("source_frame_end"),
        "locked": False,
        "member_detector_tracker_ids": [track_id],
        "merge_split_lineage": [],
        "object_uuid": object_uuid,
        "privacy_action": privacy_entities.POLICY_KEEP_VISIBLE,
        "representative_case_frame": representative_case_frame,
        "representative_bbox": representative_box,
        "representative_source_frame": representative_source_frame,
        "representative_thumbnail_path": thumbnail_path,
        "review_entity_state": "confirmed",
        "review_state": "ready",
        "reviewed_region_track_ids": [],
        "schema_version": privacy_entities.SCHEMA_VERSION,
        "visibility_state": "visible",
    }


def source_backed_track_bbox(
    *,
    source_dir: Path,
    track_id: str,
    preferred_frame: int,
) -> list[int] | None:
    tracks_path = source_dir / "tracking" / "tracks.jsonl"
    if not tracks_path.exists():
        return None
    fallback: list[int] | None = None
    for frame_payload in load_jsonl_dicts(tracks_path):
        frame_index = int(frame_payload.get("frame_index") or 0)
        for box in list(frame_payload.get("track_boxes") or []):
            if str(box.get("track_id") or "") != str(track_id):
                continue
            bbox = box.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            parsed = [int(value) for value in bbox]
            if frame_index == preferred_frame:
                return parsed
            fallback = fallback or parsed
    return fallback


def copy_face_policy_source_evidence(
    *,
    package_root: Path,
    case_id: str,
    source_dir: Path,
) -> str:
    target = package_root / "technical" / f"{case_id}_pipeline" / "face_policy"
    for relative in (
        Path("track_census.json"),
        Path("privacy_policy.json"),
        Path("source_manifest.json"),
        Path("tracking") / "tracks.jsonl",
    ):
        src = source_dir / relative
        if src.exists():
            dst = target / relative
            if src.resolve() == dst.resolve():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return target.relative_to(package_root).as_posix()


def next_payload_entity_label(category: str, existing: list[dict[str, Any]]) -> str:
    prefix = "Person" if category == CATEGORY_FACE else "Plate"
    used: set[int] = set()
    for entity in existing:
        if entity.get("category") != category:
            continue
        parts = str(entity.get("label") or entity.get("display_name") or "").split()
        if len(parts) == 2 and parts[0] == prefix and parts[1].isdigit():
            used.add(int(parts[1]))
    index = 1
    while index in used:
        index += 1
    return f"{prefix} {index:02d}"


def copy_source_backed_track_thumbnail(
    *,
    package_root: Path,
    case_id: str,
    entity_id: str,
    source_dir: Path,
    track_id: str,
) -> str | None:
    if not track_id.isdigit():
        return None
    source = source_dir / "tracking" / "track_contact_sheets" / f"track_{int(track_id):03d}.jpg"
    if not source.exists():
        return None
    target = package_root / "entities" / case_id / "thumbnails" / f"{entity_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.relative_to(package_root).as_posix()


def review_entity_profile() -> dict[str, Any]:
    return {
        "appearance_encoder": {
            "descriptor": "deterministic_hsv_luma_edge_crop_descriptor",
            "license": "project-local deterministic OpenCV descriptor; no model weights",
            "network_runtime": False,
            "schema_version": "review_appearance_encoder.v1",
            "source": "redactor.review_workspace.DeterministicCropAppearanceEncoder",
        },
        "categories": {
            CATEGORY_FACE: {
                "attachment_max_gap_seconds": 1.0,
                "auto_cluster_min_pair_score": 0.82,
                "confirmed_min_density": 0.2,
                "confirmed_min_observations": 30,
                "confirmed_score_threshold": 0.58,
                "max_aspect_delta": 0.85,
                "max_norm_center_distance": 0.095,
                "max_scale_ratio": 2.3,
                "min_appearance_similarity": 0.72,
                "needs_review_min_observations": 15,
                "needs_review_score_threshold": 0.38,
                "short_gap_survival_seconds": 1.0,
            },
            CATEGORY_PLATE: {
                "attachment_max_gap_seconds": 0.8,
                "auto_cluster_min_pair_score": 0.84,
                "confirmed_min_density": 0.18,
                "confirmed_min_observations": 12,
                "confirmed_score_threshold": 0.56,
                "max_aspect_delta": 1.15,
                "max_norm_center_distance": 0.11,
                "max_scale_ratio": 2.6,
                "min_appearance_similarity": 0.74,
                "needs_review_min_observations": 6,
                "needs_review_score_threshold": 0.34,
                "short_gap_survival_seconds": 0.8,
            },
        },
        "component_weights": {
            "appearance_quality": 0.09,
            "aspect_ratio_stability": 0.06,
            "center_motion_continuity": 0.08,
            "confidence_lower_tail": 0.09,
            "confidence_median": 0.09,
            "confidence_peak": 0.05,
            "crop_quality": 0.09,
            "density": 0.09,
            "duration": 0.08,
            "edge_clipping": -0.06,
            "geometry_stability": 0.08,
            "internal_gap_burden": -0.08,
            "observation_count": 0.1,
            "provenance_support": 0.08,
            "scale_motion_continuity": 0.06,
        },
        "notes": [
            "This profile controls only the user-facing review entity registry.",
            "It does not change detector outputs, tracker outputs, privacy regions, rendering, or finalization.",
            "Conservative attachment is temporal/appearance review grouping, not identity recognition.",
        ],
        "states": [
            "unconfirmed",
            "confirmed",
            "temporarily_lost",
            "retained",
            "unresolved",
            "closed",
        ],
        "schema_version": "review_entity_evidence_profile.v2",
    }


@dataclass(frozen=True)
class ReviewTrackletEvidence:
    appearance_descriptor: tuple[float, ...]
    appearance_quality: float
    aspect_ratio_stability: float
    bbox_xyxy: tuple[int, int, int, int]
    category: str
    center_motion_continuity: float
    confidence_p10: float | None
    crop_quality: float
    density: float
    detector_confidence_max: float | None
    detector_confidence_median: float | None
    duration_seconds: float
    edge_penalty: float
    evidence_components: dict[str, float]
    evidence_score: float
    first_frame: int
    geometry_stability: float
    jump_penalty: float
    last_frame: int
    large_internal_gap_penalty: float
    mean_confidence: float | None
    observation_count: int
    provenance_counts: dict[str, int]
    scale_motion_continuity: float
    source_track_id: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "appearance_descriptor": [round(value, 6) for value in self.appearance_descriptor],
            "appearance_quality": round(self.appearance_quality, 6),
            "aspect_ratio_stability": round(self.aspect_ratio_stability, 6),
            "bbox_xyxy": list(self.bbox_xyxy),
            "category": self.category,
            "center_motion_continuity": round(self.center_motion_continuity, 6),
            "confidence_p10": self.confidence_p10,
            "crop_quality": round(self.crop_quality, 6),
            "density": round(self.density, 6),
            "detector_confidence_max": self.detector_confidence_max,
            "detector_confidence_median": self.detector_confidence_median,
            "duration_seconds": round(self.duration_seconds, 6),
            "edge_penalty": round(self.edge_penalty, 6),
            "evidence_components": {
                key: round(value, 6) for key, value in sorted(self.evidence_components.items())
            },
            "evidence_score": round(self.evidence_score, 6),
            "first_frame": self.first_frame,
            "geometry_stability": round(self.geometry_stability, 6),
            "jump_penalty": round(self.jump_penalty, 6),
            "large_internal_gap_penalty": round(self.large_internal_gap_penalty, 6),
            "last_frame": self.last_frame,
            "mean_confidence": self.mean_confidence,
            "observation_count": self.observation_count,
            "provenance_counts": self.provenance_counts,
            "scale_motion_continuity": round(self.scale_motion_continuity, 6),
            "source_track_id": self.source_track_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class AccumulatedEntity:
    category: str
    member_track_ids: tuple[str, ...]
    state: str
    representative_track_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AttachmentDecision:
    child_track_id: str
    parent_track_id: str | None
    accepted: bool
    appearance_similarity: float
    components: dict[str, float]
    gap_frames: int | None
    reason: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "appearance_similarity": round(self.appearance_similarity, 6),
            "child_track_id": self.child_track_id,
            "components": {key: round(value, 6) for key, value in sorted(self.components.items())},
            "gap_frames": self.gap_frames,
            "parent_track_id": self.parent_track_id,
            "reason": self.reason,
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class ReviewEntityAccumulationResult:
    attachment_decisions: tuple[AttachmentDecision, ...]
    rejected_merge_decisions: tuple[AttachmentDecision, ...]
    simultaneous_overlap_rejections: tuple[AttachmentDecision, ...]
    tracklets: tuple[ReviewTrackletEvidence, ...]
    unresolved_fragments: tuple[ReviewTrackletEvidence, ...]
    user_entities: tuple[AccumulatedEntity, ...]

    def metrics(self) -> dict[str, Any]:
        previous = len(self.tracklets)
        confirmed = sum(1 for entity in self.user_entities if entity.state == "confirmed")
        needs_review = sum(1 for entity in self.user_entities if entity.state == "unresolved")
        attached = sum(1 for decision in self.attachment_decisions if decision.accepted)
        return {
            "attached_fragment_count": attached,
            "automatic_clustered_entity_count": sum(
                1 for entity in self.user_entities if len(entity.member_track_ids) > 1
            ),
            "confirmed_entity_count": confirmed,
            "needs_review_entity_count": needs_review,
            "new_user_facing_entity_count": len(self.user_entities),
            "previous_user_facing_entity_count": previous,
            "raw_tracklet_count": previous,
            "reduction_from_attachment_count": attached,
            "reduction_from_unresolved_filtering_count": len(self.unresolved_fragments),
            "rejected_merge_count": len(self.rejected_merge_decisions),
            "sampled_suspected_false_merge_count": 0,
            "schema_version": "review_entity_metrics.v1",
            "simultaneous_overlap_merge_rejection_count": len(self.simultaneous_overlap_rejections),
            "unresolved_fragment_count": len(self.unresolved_fragments),
        }

    def merge_audit(self) -> dict[str, Any]:
        return {
            "accepted_attachments": [
                decision.to_dict() for decision in self.attachment_decisions if decision.accepted
            ],
            "rejected_merges": [decision.to_dict() for decision in self.rejected_merge_decisions],
            "schema_version": "review_entity_merge_audit.v1",
            "simultaneous_overlap_rejections": [
                decision.to_dict() for decision in self.simultaneous_overlap_rejections
            ],
        }

    def cluster_audit(self) -> dict[str, Any]:
        return {
            "clusters": [
                {
                    "category": entity.category,
                    "child_fragment_ids": list(entity.member_track_ids),
                    "grouping_confidence": round(
                        max(
                            [
                                decision.score
                                for decision in self.attachment_decisions
                                if decision.accepted
                                and decision.parent_track_id in entity.member_track_ids
                            ]
                            or [1.0]
                        ),
                        6,
                    ),
                    "representative_track_id": entity.representative_track_id,
                    "state": entity.state,
                }
                for entity in self.user_entities
            ],
            "schema_version": "review_entity_clusters.v1",
        }

    def pairwise_audit(self) -> dict[str, Any]:
        return {
            "pairs": [
                decision.to_dict()
                for decision in (
                    *self.attachment_decisions,
                    *self.rejected_merge_decisions,
                    *self.simultaneous_overlap_rejections,
                )
            ],
            "schema_version": "review_entity_pairwise_audit.v1",
        }

    def cannot_link_audit(self) -> dict[str, Any]:
        return {
            "pairs": [decision.to_dict() for decision in self.simultaneous_overlap_rejections],
            "schema_version": "review_entity_cannot_link_audit.v1",
        }

    def entity_state_for_tracks(self, track_ids: list[str]) -> str:
        wanted = set(track_ids)
        for entity in self.user_entities:
            if wanted.intersection(entity.member_track_ids):
                return entity.state
        return "unresolved"

    def summary_for_tracks(self, track_ids: list[str]) -> dict[str, Any]:
        wanted = set(track_ids)
        members = [tracklet for tracklet in self.tracklets if tracklet.source_track_id in wanted]
        if not members:
            return {}
        return {
            "review_entity_state": self.entity_state_for_tracks(track_ids),
            "review_observation_count": sum(item.observation_count for item in members),
            "review_observed_span_frames": max(item.last_frame for item in members)
            - min(item.first_frame for item in members)
            + 1,
            "review_tracklet_count": len(members),
        }

    def attached_fragments_for_tracks(self, track_ids: list[str]) -> list[str]:
        wanted = set(track_ids)
        return sorted(
            decision.child_track_id
            for decision in self.attachment_decisions
            if decision.accepted and decision.parent_track_id in wanted
        )

    def possible_duplicates_for_tracks(self, track_ids: list[str]) -> list[dict[str, Any]]:
        wanted = set(track_ids)
        items = []
        for decision in self.rejected_merge_decisions:
            if decision.parent_track_id in wanted or decision.child_track_id in wanted:
                items.append(decision.to_dict())
        return items[:5]


class ReviewEntityAccumulator:
    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = profile

    def accumulate(
        self,
        *,
        case_id: str,
        tracklets: list[ReviewTrackletEvidence],
        previous_user_entity_count: int,
    ) -> ReviewEntityAccumulationResult:
        del case_id, previous_user_entity_count
        classifications = {
            tracklet.source_track_id: self.classify_tracklet(tracklet) for tracklet in tracklets
        }
        confirmed = [
            AccumulatedEntity(
                category=tracklet.category,
                member_track_ids=(tracklet.source_track_id,),
                representative_track_id=tracklet.source_track_id,
                reasons=("sufficient_accumulated_review_evidence",),
                state="confirmed",
            )
            for tracklet in tracklets
            if classifications[tracklet.source_track_id] == "confirmed"
        ]
        unresolved: list[ReviewTrackletEvidence] = []
        needs_review_entities: list[AccumulatedEntity] = []
        attachments: list[AttachmentDecision] = []
        rejected: list[AttachmentDecision] = []
        overlaps: list[AttachmentDecision] = []
        confirmed_by_track = {entity.representative_track_id: entity for entity in confirmed}
        for tracklet in tracklets:
            state = classifications[tracklet.source_track_id]
            if state == "confirmed":
                continue
            decision = self.best_attachment(tracklet, confirmed_by_track, tracklets)
            if decision:
                if decision.accepted:
                    attachments.append(decision)
                    parent = confirmed_by_track[decision.parent_track_id or ""]
                    updated = AccumulatedEntity(
                        category=parent.category,
                        member_track_ids=tuple(
                            sorted(set(parent.member_track_ids + (tracklet.source_track_id,)))
                        ),
                        representative_track_id=parent.representative_track_id,
                        reasons=parent.reasons + ("attached_consistent_fragment",),
                        state=parent.state,
                    )
                    confirmed_by_track[parent.representative_track_id] = updated
                    confirmed = [
                        updated
                        if item.representative_track_id == parent.representative_track_id
                        else item
                        for item in confirmed
                    ]
                    continue
                if decision.reason == "simultaneous_same_category_visibility":
                    overlaps.append(decision)
                else:
                    rejected.append(decision)
            if state == "needs_review":
                needs_review_entities.append(
                    AccumulatedEntity(
                        category=tracklet.category,
                        member_track_ids=(tracklet.source_track_id,),
                        representative_track_id=tracklet.source_track_id,
                        reasons=("meaningful_but_unconfirmed_review_evidence",),
                        state="unresolved",
                    )
                )
            else:
                unresolved.append(tracklet)
        ordered_entities = sorted(
            confirmed + needs_review_entities,
            key=lambda item: (
                item.category,
                min(
                    self.tracklet_by_id(tracklets, tid).first_frame for tid in item.member_track_ids
                ),
                item.representative_track_id,
            ),
        )
        return ReviewEntityAccumulationResult(
            attachment_decisions=tuple(attachments),
            rejected_merge_decisions=tuple(rejected),
            simultaneous_overlap_rejections=tuple(overlaps),
            tracklets=tuple(tracklets),
            unresolved_fragments=tuple(
                sorted(unresolved, key=lambda item: (item.category, item.first_frame))
            ),
            user_entities=tuple(ordered_entities),
        )

    def classify_tracklet(self, tracklet: ReviewTrackletEvidence) -> str:
        category_profile = self.profile["categories"][tracklet.category]
        if (
            tracklet.observation_count >= category_profile["confirmed_min_observations"]
            and tracklet.density >= category_profile["confirmed_min_density"]
            and tracklet.evidence_score >= category_profile["confirmed_score_threshold"]
            and tracklet.geometry_stability >= 0.15
            and tracklet.jump_penalty < 0.9
        ):
            return "confirmed"
        if (
            tracklet.observation_count >= category_profile["needs_review_min_observations"]
            and tracklet.evidence_score >= category_profile["needs_review_score_threshold"]
            and tracklet.geometry_stability >= 0.08
            and tracklet.crop_quality >= 0.02
        ):
            return "needs_review"
        return "unconfirmed"

    def best_attachment(
        self,
        child: ReviewTrackletEvidence,
        parents: dict[str, AccumulatedEntity],
        tracklets: list[ReviewTrackletEvidence],
    ) -> AttachmentDecision | None:
        candidates: list[AttachmentDecision] = []
        for parent_entity in parents.values():
            if parent_entity.category != child.category:
                continue
            parent_tracklet = self.tracklet_by_id(tracklets, parent_entity.representative_track_id)
            decision = self.attachment_decision(parent_tracklet, child)
            candidates.append(decision)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.score)

    def attachment_decision(
        self,
        parent: ReviewTrackletEvidence,
        child: ReviewTrackletEvidence,
    ) -> AttachmentDecision:
        profile = self.profile["categories"][child.category]
        if ranges_overlap(
            parent.first_frame, parent.last_frame, child.first_frame, child.last_frame
        ):
            return AttachmentDecision(
                accepted=False,
                appearance_similarity=appearance_similarity(parent, child),
                child_track_id=child.source_track_id,
                components={"temporal_overlap": 1.0},
                gap_frames=0,
                parent_track_id=parent.source_track_id,
                reason="simultaneous_same_category_visibility",
                score=0.0,
            )
        gap_frames = frame_gap(
            parent.first_frame, parent.last_frame, child.first_frame, child.last_frame
        )
        max_gap = max(1, int(profile["attachment_max_gap_seconds"] * 30))
        if gap_frames > max_gap:
            return AttachmentDecision(
                accepted=False,
                appearance_similarity=appearance_similarity(parent, child),
                child_track_id=child.source_track_id,
                components={"gap_frames": float(gap_frames), "max_gap_frames": float(max_gap)},
                gap_frames=gap_frames,
                parent_track_id=parent.source_track_id,
                reason="gap_too_large_for_review_entity_attachment",
                score=0.0,
            )
        center_distance = normalized_center_distance(parent.bbox_xyxy, child.bbox_xyxy)
        scale_ratio = bbox_scale_ratio(parent.bbox_xyxy, child.bbox_xyxy)
        aspect_delta = abs(bbox_aspect(parent.bbox_xyxy) - bbox_aspect(child.bbox_xyxy))
        appearance = appearance_similarity(parent, child)
        center_component = max(
            0.0,
            1.0 - center_distance / max(0.001, profile["max_norm_center_distance"]),
        )
        scale_component = max(
            0.0,
            1.0 - max(0.0, scale_ratio - 1.0) / max(0.001, profile["max_scale_ratio"]),
        )
        aspect_component = max(
            0.0,
            1.0 - aspect_delta / max(0.001, profile["max_aspect_delta"]),
        )
        gap_component = max(0.0, 1.0 - gap_frames / max(1.0, max_gap))
        score = max(
            0.0,
            0.34 * center_component
            + 0.22 * scale_component
            + 0.16 * aspect_component
            + 0.18 * appearance
            + 0.10 * gap_component,
        )
        accepted = (
            center_distance <= profile["max_norm_center_distance"]
            and scale_ratio <= profile["max_scale_ratio"]
            and aspect_delta <= profile["max_aspect_delta"]
            and appearance >= profile["min_appearance_similarity"]
            and score >= profile["auto_cluster_min_pair_score"]
        )
        return AttachmentDecision(
            accepted=accepted,
            appearance_similarity=appearance,
            child_track_id=child.source_track_id,
            components={
                "aspect": aspect_component,
                "center": center_component,
                "gap": gap_component,
                "scale": scale_component,
            },
            gap_frames=gap_frames,
            parent_track_id=parent.source_track_id,
            reason="compatible_short_gap_fragment" if accepted else "spatial_motion_incompatible",
            score=score,
        )

    @staticmethod
    def tracklet_by_id(
        tracklets: list[ReviewTrackletEvidence], track_id: str
    ) -> ReviewTrackletEvidence:
        for tracklet in tracklets:
            if tracklet.source_track_id == track_id:
                return tracklet
        raise KeyError(track_id)


def review_entity_tracklets(
    case_id: str,
    records: list[PrivacyRegionRecord],
    manifest: case_manifests.CaseManifest,
    package_root: Path,
) -> list[ReviewTrackletEvidence]:
    del case_id
    grouped: dict[tuple[str, str], list[PrivacyRegionRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.category, record_track_id(record))].append(record)
    source_path = package_root / str(manifest.source_path or "")
    capture = cv.VideoCapture(str(source_path)) if source_path.exists() else None
    try:
        return [
            tracklet_evidence(
                category,
                track_id,
                sorted(items, key=lambda item: item.source_frame),
                manifest,
                capture,
            )
            for (category, track_id), items in sorted(
                grouped.items(), key=lambda pair: (pair[0][0], pair[1][0].source_frame, pair[0][1])
            )
        ]
    finally:
        if capture is not None:
            capture.release()


def tracklet_evidence(
    category: str,
    track_id: str,
    records: list[PrivacyRegionRecord],
    manifest: case_manifests.CaseManifest,
    capture: Any | None = None,
) -> ReviewTrackletEvidence:
    frames = [record.source_frame for record in records]
    first = min(frames)
    last = max(frames)
    span = last - first + 1
    fps = rational_to_float(manifest.fps) or 30.0
    confidences = [
        float(value)
        for record in records
        for value in [confidence_for_record(record)]
        if value is not None
    ]
    bbox = median_bbox([record.bbox_xyxy for record in records])
    provenance_counts = Counter(record.provenance for record in records)
    centers = [bbox_center(record.bbox_xyxy) for record in records]
    areas = [bbox_area(record.bbox_xyxy) for record in records]
    aspects = [bbox_aspect(record.bbox_xyxy) for record in records]
    center_jumps = [
        math.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1])
        for i in range(1, len(centers))
    ]
    frame_gaps = [frames[i] - frames[i - 1] - 1 for i in range(1, len(frames))]
    crop_quality = representative_tracklet_quality(records, manifest)
    appearance_descriptor, appearance_quality = DeterministicCropAppearanceEncoder().encode(
        records,
        capture,
    )
    components = review_evidence_components(
        confidences=confidences,
        crop_quality=crop_quality,
        density=len(records) / max(1, span),
        duration_seconds=span / fps,
        edge_penalty_value=edge_penalty(bbox, manifest),
        geometry_stability_value=0.5 * stability_score(areas) + 0.5 * stability_score(aspects),
        observation_count=len(records),
        provenance_counts=provenance_counts,
        scale_motion_continuity=stability_score(areas),
        center_motion_continuity=1.0 - min(1.0, median(center_jumps) / 160.0),
        aspect_ratio_stability=stability_score(aspects),
        large_internal_gap_penalty=min(1.0, max(frame_gaps or [0]) / max(1.0, fps)),
        appearance_quality=appearance_quality,
    )
    return ReviewTrackletEvidence(
        appearance_descriptor=appearance_descriptor,
        appearance_quality=appearance_quality,
        aspect_ratio_stability=stability_score(aspects),
        bbox_xyxy=bbox,
        category=category,
        center_motion_continuity=1.0 - min(1.0, median(center_jumps) / 160.0),
        confidence_p10=percentile(confidences, 10) if confidences else None,
        crop_quality=crop_quality,
        density=len(records) / max(1, span),
        detector_confidence_max=max(confidences) if confidences else None,
        detector_confidence_median=percentile(confidences, 50) if confidences else None,
        duration_seconds=span / fps,
        edge_penalty=edge_penalty(bbox, manifest),
        evidence_components=components,
        evidence_score=review_evidence_score(components, review_entity_profile()),
        first_frame=first,
        geometry_stability=0.5 * stability_score(areas) + 0.5 * stability_score(aspects),
        jump_penalty=min(1.0, max(center_jumps or [0.0]) / 420.0),
        large_internal_gap_penalty=min(1.0, max(frame_gaps or [0]) / max(1.0, fps)),
        last_frame=last,
        mean_confidence=sum(confidences) / len(confidences) if confidences else None,
        observation_count=len(records),
        provenance_counts=dict(sorted(provenance_counts.items())),
        scale_motion_continuity=stability_score(areas),
        source_track_id=track_id,
        status=track_status(records),
    )


def confidence_for_record(record: PrivacyRegionRecord) -> float | None:
    source = dict(record.source_record or {})
    value = source.get("score")
    if value is None:
        value = source.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DeterministicCropAppearanceEncoder:
    """Small review-only descriptor; this is not identity recognition."""

    schema_version = "deterministic_crop_appearance_encoder.v1"

    def encode(
        self,
        records: list[PrivacyRegionRecord],
        capture: Any | None,
    ) -> tuple[tuple[float, ...], float]:
        if capture is None or not records:
            return (0.0,) * 18, 0.0
        selected = max(records, key=lambda record: bbox_area(record.bbox_xyxy))
        capture.set(cv.CAP_PROP_POS_FRAMES, int(selected.source_frame))
        ok, frame = capture.read()
        if not ok or frame is None:
            return (0.0,) * 18, 0.0
        x1, y1, x2, y2 = selected.bbox_xyxy
        height, width = frame.shape[:2]
        x1, x2 = sorted((clamp_int(x1, 0, width), clamp_int(x2, 0, width)))
        y1, y2 = sorted((clamp_int(y1, 0, height), clamp_int(y2, 0, height)))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return (0.0,) * 18, 0.0
        resized = cv.resize(crop, (32, 32), interpolation=cv.INTER_AREA)
        hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        hist_h = cv.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        hist_s = cv.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
        hist_v = cv.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
        descriptor = [
            *normalize_vector(hist_h),
            *normalize_vector(hist_s),
            *normalize_vector(hist_v),
        ]
        descriptor.extend(
            [
                float(gray.mean()) / 255.0,
                min(1.0, float(gray.std()) / 90.0),
            ]
        )
        sharpness = min(1.0, cv.Laplacian(gray, cv.CV_64F).var() / 600.0)
        quality = max(0.0, min(1.0, 0.55 * sharpness + 0.45 * min(1.0, crop.size / 8000.0)))
        return tuple(float(value) for value in descriptor), quality


def review_evidence_components(
    *,
    confidences: list[float],
    crop_quality: float,
    density: float,
    duration_seconds: float,
    edge_penalty_value: float,
    geometry_stability_value: float,
    observation_count: int,
    provenance_counts: Counter[str],
    scale_motion_continuity: float,
    center_motion_continuity: float,
    aspect_ratio_stability: float,
    large_internal_gap_penalty: float,
    appearance_quality: float,
) -> dict[str, float]:
    directly_observed = sum(
        provenance_counts.get(name, 0)
        for name in (
            "detector",
            "ordinary_tracker",
            "plate_detector_observation",
            "reviewed_correction",
        )
    )
    total = sum(provenance_counts.values()) or 1
    return {
        "appearance_quality": max(0.0, min(1.0, appearance_quality)),
        "aspect_ratio_stability": max(0.0, min(1.0, aspect_ratio_stability)),
        "center_motion_continuity": max(0.0, min(1.0, center_motion_continuity)),
        "confidence_lower_tail": max(0.0, min(1.0, percentile(confidences, 10)))
        if confidences
        else 0.0,
        "confidence_median": max(0.0, min(1.0, percentile(confidences, 50)))
        if confidences
        else 0.5,
        "confidence_peak": max(0.0, min(1.0, max(confidences))) if confidences else 0.5,
        "crop_quality": max(0.0, min(1.0, crop_quality)),
        "density": max(0.0, min(1.0, density)),
        "duration": max(0.0, min(1.0, duration_seconds / 4.0)),
        "edge_clipping": max(0.0, min(1.0, edge_penalty_value)),
        "geometry_stability": max(0.0, min(1.0, geometry_stability_value)),
        "internal_gap_burden": max(0.0, min(1.0, large_internal_gap_penalty)),
        "observation_count": max(0.0, min(1.0, observation_count / 40.0)),
        "provenance_support": max(0.0, min(1.0, directly_observed / total)),
        "scale_motion_continuity": max(0.0, min(1.0, scale_motion_continuity)),
    }


def review_evidence_score(components: dict[str, float], profile: dict[str, Any]) -> float:
    weights = profile["component_weights"]
    raw = sum(float(weights.get(key, 0.0)) * float(value) for key, value in components.items())
    positive_weight = sum(float(value) for value in weights.values() if float(value) > 0)
    negative_floor = sum(abs(float(value)) for value in weights.values() if float(value) < 0)
    if positive_weight <= 0:
        return 0.0
    return max(0.0, min(1.0, (raw + negative_floor) / (positive_weight + negative_floor)))


def appearance_similarity(left: ReviewTrackletEvidence, right: ReviewTrackletEvidence) -> float:
    if not left.appearance_descriptor or not right.appearance_descriptor:
        return 0.0
    if len(left.appearance_descriptor) != len(right.appearance_descriptor):
        return 0.0
    distance = math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left.appearance_descriptor, right.appearance_descriptor, strict=True)
        )
    )
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def normalize_vector(values: Any) -> list[float]:
    floats = [float(value) for value in values]
    total = sum(floats)
    if total <= 0:
        return [0.0 for _ in floats]
    return [value / total for value in floats]


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def track_status(records: list[PrivacyRegionRecord]) -> str:
    statuses = Counter(
        str((record.source_record or {}).get("status") or record.approval_state)
        for record in records
    )
    return statuses.most_common(1)[0][0] if statuses else "unknown"


def representative_tracklet_quality(
    records: list[PrivacyRegionRecord],
    manifest: case_manifests.CaseManifest,
) -> float:
    scores = []
    for record in records:
        area_score, confidence, boundary_score, _ = representative_geometry_quality(
            record, manifest
        )
        scores.append(
            max(0.0, min(1.0, area_score * 18.0))
            + max(0.0, min(1.0, confidence))
            + max(0.0, min(1.0, boundary_score * 20.0))
        )
    return max(scores or [0.0]) / 3.0


def privacy_entity_from_accumulated(
    case_id: str,
    item: AccumulatedEntity,
    existing: list[privacy_entities.PrivacyEntity],
) -> privacy_entities.PrivacyEntity:
    object_uuid = stable_hash(
        {
            "case_id": case_id,
            "category": item.category,
            "member_track_ids": item.member_track_ids,
            "schema_version": "review_accumulated_entity_uuid.v1",
        }
    )[:32]
    entity = privacy_entities.create_entity(
        case_id=case_id,
        category=item.category,
        existing=existing,
        automatic_source_track_ids=item.member_track_ids,
        default_privacy_policy=privacy_entities.POLICY_REDACT
        if item.state == "confirmed"
        else privacy_entities.POLICY_REVIEW_REQUIRED,
        entity_id=f"entity_{object_uuid}",
        provenance={
            "accumulation_reasons": list(item.reasons),
            "representative_track_id": item.representative_track_id,
            "review_entity_state": item.state,
            "source": "review_entity_accumulator",
            "track_ids": list(item.member_track_ids),
        },
    )
    if item.state != "confirmed":
        entity = replace(entity, label=next_possible_entity_label(item.category, existing))
    return entity


def next_possible_entity_label(
    category: str,
    existing: list[privacy_entities.PrivacyEntity],
) -> str:
    prefix = "Possible person" if category == CATEGORY_FACE else "Possible plate"
    used: set[int] = set()
    for entity in existing:
        if entity.category != category:
            continue
        parts = entity.label.split()
        if (
            len(parts) == 3
            and f"{parts[0]} {parts[1]}".lower() == prefix.lower()
            and parts[2].isdigit()
        ):
            used.add(int(parts[2]))
    index = 1
    while index in used:
        index += 1
    return f"{prefix} {index:02d}"


def unresolved_fragment_payload(
    fragment: ReviewTrackletEvidence,
    manifest: case_manifests.CaseManifest,
    package_root: Path,
) -> dict[str, Any]:
    del package_root
    return {
        "category": fragment.category,
        "first_visible_source_frame": manifest.source_frame_start + fragment.first_frame,
        "last_visible_source_frame": manifest.source_frame_start + fragment.last_frame,
        "reason": "low_evidence_unmatched_fragment",
        "source_track_id": fragment.source_track_id,
        "summary": fragment.to_dict(),
    }


def entity_generation_audit(
    *,
    case_id: str,
    tracklets: list[ReviewTrackletEvidence],
    previous_entities: list[privacy_entities.PrivacyEntity],
    accumulation: ReviewEntityAccumulationResult,
) -> dict[str, Any]:
    durations = [item.duration_seconds for item in tracklets]
    observations = [item.observation_count for item in tracklets]
    confidences = [item.mean_confidence for item in tracklets if item.mean_confidence is not None]
    return {
        "case_id": case_id,
        "confidence_distribution": numeric_distribution(confidences),
        "counts_by_category": dict(Counter(item.category for item in tracklets)),
        "counts_by_provenance": dict(
            Counter(
                provenance
                for item in tracklets
                for provenance, count in item.provenance_counts.items()
                for _ in range(count)
            )
        ),
        "counts_by_track_status": dict(Counter(item.status for item in tracklets)),
        "crop_quality_distribution": numeric_distribution(
            [item.crop_quality for item in tracklets]
        ),
        "duplicate_fragmentation_candidates": [
            decision.to_dict()
            for decision in accumulation.attachment_decisions
            if decision.accepted
        ],
        "duration_distribution_seconds": numeric_distribution(durations),
        "generation_sources": [
            "privacy_region_track_groups",
            "accepted_tracks",
            "needs_review_tracks",
            "reviewed_correction_records",
        ],
        "observation_count_distribution": numeric_distribution(observations),
        "previous_entity_count": len(previous_entities),
        "raw_tracklet_count": len(tracklets),
        "rejected_simultaneous_overlap_merges": [
            decision.to_dict() for decision in accumulation.simultaneous_overlap_rejections
        ],
        "schema_version": "review_entity_generation_audit.v1",
        "source_track_ids": sorted(item.source_track_id for item in tracklets),
        "trace": "The previous registry used one review entity per privacy-region source track.",
    }


def write_entity_contact_sheets(
    *,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    accumulation: ReviewEntityAccumulationResult,
    payload_by_track: dict[str, dict[str, Any]],
) -> dict[str, str]:
    groups: dict[str, list[ReviewTrackletEvidence]] = {
        "stable_entities": [
            item
            for item in accumulation.tracklets
            if any(
                item.source_track_id in entity.member_track_ids
                for entity in accumulation.user_entities
                if entity.state == "confirmed"
            )
        ][:12],
        "short_weak_fragments": list(accumulation.unresolved_fragments)[:12],
        "likely_attached_fragments": [
            ReviewEntityAccumulator.tracklet_by_id(
                list(accumulation.tracklets), decision.child_track_id
            )
            for decision in accumulation.attachment_decisions
            if decision.accepted
        ][:12],
        "unresolved_evidence": list(accumulation.unresolved_fragments)[:12],
        "rejected_simultaneous_overlap_merges": [
            ReviewEntityAccumulator.tracklet_by_id(
                list(accumulation.tracklets), decision.child_track_id
            )
            for decision in accumulation.simultaneous_overlap_rejections
        ][:12],
    }
    output: dict[str, str] = {}
    for name, tracklets in groups.items():
        if not tracklets:
            continue
        path = package_root / "entities" / manifest.case_id / "contact_sheets" / f"{name}.jpg"
        write_tracklet_contact_sheet(path, package_root, manifest, tracklets, payload_by_track)
        output[name] = path.relative_to(package_root).as_posix()
    return output


def write_tracklet_contact_sheet(
    path: Path,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    tracklets: list[ReviewTrackletEvidence],
    payload_by_track: dict[str, dict[str, Any]],
) -> None:
    source_path = package_root / str(manifest.source_path or "")
    thumbs: list[tuple[str, Any]] = []
    capture = cv.VideoCapture(str(source_path)) if source_path.exists() else None
    try:
        for tracklet in tracklets:
            payload = payload_by_track.get(tracklet.source_track_id, {})
            thumb_path = payload.get("representative_thumbnail_path")
            image = None
            if thumb_path:
                image = cv.imread(str(package_root / thumb_path))
            if image is None and capture is not None:
                capture.set(cv.CAP_PROP_POS_FRAMES, tracklet.first_frame)
                ok, frame = capture.read()
                if ok:
                    x1, y1, x2, y2 = tracklet.bbox_xyxy
                    image = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if image is None or image.size == 0:
                image = np_zeros((100, 160, 3))
            image = cv.resize(image, (160, 100), interpolation=cv.INTER_AREA)
            thumbs.append((tracklet.source_track_id, image))
    finally:
        if capture is not None:
            capture.release()
    if not thumbs:
        return
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = 255 * np_zeros((rows * 126, cols * 180, 3))
    for index, (label, image) in enumerate(thumbs):
        row, col = divmod(index, cols)
        y = row * 126
        x = col * 180
        sheet[y : y + 100, x : x + 160] = image
        cv.putText(
            sheet, label[:18], (x + 4, y + 118), cv.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), sheet, [int(cv.IMWRITE_JPEG_QUALITY), 88])


def np_zeros(shape: tuple[int, int, int]) -> Any:
    import numpy as np

    return np.zeros(shape, dtype=np.uint8)


def empty_entity_generation_audit(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "raw_tracklet_count": 0,
        "schema_version": "review_entity_generation_audit.v1",
    }


def empty_entity_metrics(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "raw_tracklet_count": 0,
        "schema_version": "review_entity_metrics.v1",
    }


def empty_entity_merge_audit(case_id: str) -> dict[str, Any]:
    return {"case_id": case_id, "schema_version": "review_entity_merge_audit.v1"}


def before_after_summary(metrics_by_case: dict[str, Any]) -> str:
    lines = [
        "# Final Review Workspace v2.3.3 Entity and Timeline Summary",
        "",
        "Review-entity accumulation changes only the user-facing registry. It does not change automatic privacy regions, rendering, frame mapping, corrections, or finalization.",
        "",
        "| Case | Raw tracklets | Previous entities | New entities | Confirmed | Needs review | Unresolved fragments | Attached fragments |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_id, metrics in sorted(metrics_by_case.items()):
        lines.append(
            "| {case} | {raw} | {prev} | {new} | {confirmed} | {needs} | {unresolved} | {attached} |".format(
                case=case_id,
                raw=metrics.get("raw_tracklet_count", 0),
                prev=metrics.get("previous_user_facing_entity_count", 0),
                new=metrics.get("new_user_facing_entity_count", 0),
                confirmed=metrics.get("confirmed_entity_count", 0),
                needs=metrics.get("needs_review_entity_count", 0),
                unresolved=metrics.get("unresolved_fragment_count", 0),
                attached=metrics.get("attached_fragment_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "The colored timeline is task-linked and uses source-frame mapping from each case manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def numeric_distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p10": percentile(ordered, 10),
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def median(values: list[float]) -> float:
    return percentile(values, 50)


def median_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    columns = list(zip(*boxes, strict=True))
    return tuple(int(round(median([float(value) for value in column]))) for column in columns)  # type: ignore[return-value]


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def bbox_aspect(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(1, x2 - x1)) / float(max(1, y2 - y1))


def bbox_scale_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    return max(area_a, area_b) / max(1.0, min(area_a, area_b))


def normalized_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by) / 2202.0


def stability_score(values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(0.0, 1.0 - math.sqrt(variance) / mean)


def edge_penalty(bbox: tuple[int, int, int, int], manifest: case_manifests.CaseManifest) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1, int(manifest.width or 1))
    height = max(1, int(manifest.height or 1))
    margin = min(x1, y1, width - x2, height - y2)
    return 1.0 - max(0.0, min(1.0, margin / max(width, height)))


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def frame_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    if ranges_overlap(a_start, a_end, b_start, b_end):
        return 0
    if a_end < b_start:
        return b_start - a_end - 1
    return a_start - b_end - 1


def write_timeline_marker_audit(
    package_root: Path,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    errors: list[str] = []
    for case_id in ("day", "night"):
        manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
        normal_tasks = [
            task
            for task in tasks
            if task.get("case_id") == case_id and not task.get("technical_only")
        ]
        markers = []
        for task in normal_tasks:
            start_case = int(task["source_frame_start"]) - int(manifest.source_frame_start)
            end_case = int(task["source_frame_end"]) - int(manifest.source_frame_start)
            marker = {
                "blocking": bool(task.get("blocking")),
                "case_frame_end": end_case,
                "case_frame_start": start_case,
                "case_time_start": case_frame_to_seconds(start_case, manifest),
                "category": task.get("category"),
                "color": timeline_marker_color(task),
                "decision_state": task.get("decision_state", "pending"),
                "optional": bool(task.get("optional")),
                "review_task_id": task.get("review_task_id"),
                "source_frame_end": task.get("source_frame_end"),
                "source_frame_start": task.get("source_frame_start"),
                "title": task.get("title"),
            }
            if start_case < 0 or end_case >= int(manifest.frame_count):
                errors.append(f"{case_id} marker {task.get('review_task_id')} is outside case span")
            markers.append(marker)
        cases[case_id] = {
            "marker_count": len(markers),
            "markers": markers,
            "normal_task_count": len(normal_tasks),
            "parity": len(markers) == len(normal_tasks),
        }
    payload = {
        "cases": cases,
        "created_at_utc": media.utc_timestamp(),
        "errors": errors,
        "passed": not errors and all(case["parity"] for case in cases.values()),
        "schema_version": "timeline_marker_audit.v1",
    }
    write_json(package_root / "timeline_marker_audit.json", payload)
    return payload


def timeline_marker_color(task: dict[str, Any]) -> str:
    issue_type = str(task.get("issue_type") or "")
    if issue_type == "spot_check" or task.get("optional"):
        return "blue"
    if "false" in issue_type or "wrong" in issue_type:
        return "purple"
    if task.get("blocking"):
        return "red"
    return "orange"


def review_entity_payload(
    *,
    entity: privacy_entities.PrivacyEntity,
    records: list[PrivacyRegionRecord],
    manifest: case_manifests.CaseManifest,
    package_root: Path,
) -> dict[str, Any]:
    track_ids = set(entity.automatic_source_track_ids)
    matching = [record for record in records if record_track_id(record) in track_ids]
    base = entity.to_dict()
    if not matching:
        return base | {
            "evidence_summary": {"region_count": 0, "tracklet_count": len(track_ids)},
            "first_visible_source_frame": None,
            "last_visible_source_frame": None,
            "representative_case_frame": None,
            "representative_source_frame": None,
            "representative_thumbnail_path": None,
        }
    selected = choose_representative_region(matching, manifest, package_root)
    first_case_frame = min(record.source_frame for record in matching)
    last_case_frame = max(record.source_frame for record in matching)
    rep_case_frame = selected.source_frame
    first_source_frame = manifest.source_frame_start + first_case_frame
    last_source_frame = manifest.source_frame_start + last_case_frame
    rep_source_frame = manifest.source_frame_start + rep_case_frame
    thumbnail_path = write_entity_thumbnail(
        entity=entity,
        manifest=manifest,
        package_root=package_root,
        record=selected,
    )
    provenance_counts: dict[str, int] = {}
    for record in matching:
        provenance_counts[record.provenance] = provenance_counts.get(record.provenance, 0) + 1
    return base | {
        "evidence_summary": {
            "provenance_counts": provenance_counts,
            "region_count": len(matching),
            "tracklet_count": len(track_ids),
        },
        "first_visible_source_frame": first_source_frame,
        "first_visible_timestamp_seconds": case_frame_to_seconds(first_case_frame, manifest),
        "last_visible_source_frame": last_source_frame,
        "last_visible_timestamp_seconds": case_frame_to_seconds(last_case_frame, manifest),
        "representative_quality": representative_quality_summary(selected, manifest),
        "representative_case_frame": rep_case_frame,
        "representative_source_frame": rep_source_frame,
        "representative_thumbnail_path": thumbnail_path,
    }


def thumbnail_audit_payload(entity: dict[str, Any]) -> dict[str, Any]:
    quality = dict(entity.get("representative_quality") or {})
    warnings: list[str] = []
    if not entity.get("representative_thumbnail_path"):
        warnings.append("thumbnail_missing")
    if float(quality.get("relative_area") or 0.0) < 0.0001:
        warnings.append("small_region")
    if float(quality.get("boundary_margin") or 0.0) < 0.005:
        warnings.append("near_frame_edge")
    return {
        "category": entity.get("category"),
        "entity_id": entity.get("entity_id"),
        "label": entity.get("label") or entity.get("display_name"),
        "quality": quality,
        "representative_source_frame": entity.get("representative_source_frame"),
        "thumbnail_path": entity.get("representative_thumbnail_path"),
        "warnings": warnings,
    }


def record_track_id(record: PrivacyRegionRecord) -> str:
    source = dict(record.source_record or {})
    track_id = (
        source.get("track_uid")
        or source.get("track_id")
        or record.object_id
        or source.get("region_id")
        or "unassigned"
    )
    return str(track_id)


def case_frame_to_seconds(frame_index: int, manifest: case_manifests.CaseManifest) -> float | None:
    fps = rational_to_float(manifest.fps)
    if not fps:
        return None
    return frame_index / fps


def choose_representative_region(
    records: list[PrivacyRegionRecord],
    manifest: case_manifests.CaseManifest,
    package_root: Path,
) -> PrivacyRegionRecord:
    source_path = package_root / str(manifest.source_path or "")
    candidates = sorted(
        records,
        key=lambda record: representative_geometry_quality(record, manifest),
        reverse=True,
    )[:6]
    capture = cv.VideoCapture(str(source_path)) if source_path.exists() else None
    try:
        return max(
            candidates,
            key=lambda record: representative_quality(record, manifest, capture),
        )
    finally:
        if capture is not None:
            capture.release()


def representative_geometry_quality(
    record: PrivacyRegionRecord,
    manifest: case_manifests.CaseManifest,
) -> tuple[float, float, float, int]:
    x1, y1, x2, y2 = record.bbox_xyxy
    width = max(1, int(manifest.width or 1))
    height = max(1, int(manifest.height or 1))
    area_score = max(0.0, (x2 - x1) * (y2 - y1) / float(width * height))
    boundary_score = min(x1, y1, width - x2, height - y2) / float(max(width, height))
    source = dict(record.source_record or {})
    confidence = float(source.get("score") or source.get("confidence") or 0.0)
    return (area_score, confidence, boundary_score, -record.source_frame)


def representative_quality(
    record: PrivacyRegionRecord,
    manifest: case_manifests.CaseManifest,
    capture: cv.VideoCapture | None,
) -> tuple[float, float, float, int]:
    area_score, confidence, boundary_score, negative_frame = representative_geometry_quality(
        record,
        manifest,
    )
    sharpness = thumbnail_sharpness(capture, record.source_frame, record.bbox_xyxy)
    return (area_score, confidence, boundary_score + sharpness, negative_frame)


def representative_quality_summary(
    record: PrivacyRegionRecord,
    manifest: case_manifests.CaseManifest,
) -> dict[str, Any]:
    area_score, confidence, boundary_score, _ = representative_geometry_quality(record, manifest)
    x1, y1, x2, y2 = record.bbox_xyxy
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "boundary_margin": boundary_score,
        "confidence": confidence,
        "relative_area": area_score,
        "schema_version": "representative_quality.v1",
    }


def thumbnail_sharpness(
    capture: cv.VideoCapture | None,
    frame_index: int,
    bbox: tuple[int, int, int, int],
) -> float:
    if capture is None:
        return 0.0
    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        return 0.0
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    return min(float(cv.Laplacian(gray, cv.CV_64F).var()) / 1000.0, 1.0)


def write_entity_thumbnail(
    *,
    entity: privacy_entities.PrivacyEntity,
    manifest: case_manifests.CaseManifest,
    package_root: Path,
    record: PrivacyRegionRecord,
) -> str | None:
    source_path = package_root / str(manifest.source_path or "")
    if not source_path.exists():
        return None
    capture = cv.VideoCapture(str(source_path))
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, record.source_frame)
        ok, frame = capture.read()
        if not ok:
            return None
        x1, y1, x2, y2 = record.bbox_xyxy
        width = frame.shape[1]
        height = frame.shape[0]
        pad_fraction = 0.42 if entity.category == CATEGORY_FACE else 0.55
        pad = max(10, int(max(x2 - x1, y2 - y1) * pad_fraction))
        left = max(0, x1 - pad)
        top = max(0, y1 - pad)
        right = min(width, x2 + pad)
        bottom = min(height, y2 + pad)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        target_height = 120
        scale = target_height / max(1, crop.shape[0])
        resized = cv.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), target_height),
            interpolation=cv.INTER_AREA,
        )
        out_path = (
            package_root / "entities" / manifest.case_id / "thumbnails" / f"{entity.entity_id}.jpg"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_path), resized, [int(cv.IMWRITE_JPEG_QUALITY), 88])
        return out_path.relative_to(package_root).as_posix()
    finally:
        capture.release()


def missing_frozen_model_blockers() -> list[str]:
    blockers: list[str] = []
    try:
        validate_frozen_models(resolve_frozen_model_paths())
    except ValueError as exc:
        blockers.append(str(exc))
    return blockers


def load_raw_audit_signals(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_offsets = audit_clip_frame_offsets()
    signals: list[dict[str, Any]] = []
    for item in payload.get("tasks", []):
        clip_id = str(item["clip_id"])
        case_id = case_id_for_clip(clip_id)
        frame_offset = frame_offsets.get(clip_id, 0)
        signals.append(
            {
                "case_id": case_id,
                "category": normalize_audit_category(str(item["category"])),
                "raw_signal_id": str(item["audit_item_id"]),
                "root_cause": root_cause_for_signal(item),
                "schema_version": "raw_audit_signal.v1",
                "severity": str(item["severity"]),
                "source_clip_id": clip_id,
                "source_frame_end": frame_offset + int(item["source_frame_end"]),
                "source_frame_start": frame_offset + int(item["source_frame_start"]),
                "source_note": item.get("notes"),
                "technical": case_id not in {"day", "night"},
            }
        )
    return signals


def aggregate_reviewer_tasks(raw_signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = [
        signal | {"root_cause": normalize_review_task_class(str(signal["root_cause"]))}
        for signal in raw_signals
        if not signal["technical"]
    ]
    groups: list[list[dict[str, Any]]] = []
    for signal in sorted(
        visible,
        key=lambda item: (
            item["case_id"],
            item["category"],
            severity_rank(item["severity"]),
            item["source_frame_start"],
        ),
    ):
        placed = False
        for group in groups:
            first = group[0]
            if (
                first["case_id"] == signal["case_id"]
                and first["category"] == signal["category"]
                and first["root_cause"] == signal["root_cause"]
                and signal["source_frame_start"]
                <= max(item["source_frame_end"] for item in group) + 1
            ):
                group.append(signal)
                placed = True
                break
        if not placed:
            groups.append([signal])
    tasks = [reviewer_task_from_group(index, group) for index, group in enumerate(groups, start=1)]
    technical_tasks = [
        technical_audit_task(index, signal)
        for index, signal in enumerate([s for s in raw_signals if s["technical"]], start=1)
    ]
    return sorted(
        tasks + technical_tasks,
        key=lambda item: (
            item.get("technical_only", False),
            item["case_id"],
            task_sort_rank(item),
            item["source_frame_start"],
        ),
    )


def apply_reviewer_task_chunking_to_package(package_root: Path) -> dict[str, Any]:
    tasks_path = package_root / "reviewer_tasks.json"
    payload = read_json_file(tasks_path)
    tasks, audit = chunk_reviewer_tasks_for_package(
        package_root,
        [dict(task) for task in list(payload.get("tasks") or [])],
    )
    payload["tasks"] = tasks
    payload["task_count"] = len(tasks)
    payload["chunking_audit"] = "review_task_chunking_audit.json"
    write_json(tasks_path, payload)
    write_json(package_root / "review_task_chunking_audit.json", audit)
    return audit


def chunk_reviewer_tasks_for_package(
    package_root: Path,
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests = {
        path.stem: case_manifests.load_case_manifest(path)
        for path in sorted((package_root / "cases").glob("*.json"))
    }
    chunked: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    max_duration_seconds = 0.0
    for task in tasks:
        case_id = str(task.get("case_id") or "")
        manifest = manifests.get(case_id)
        if task.get("technical_only") or manifest is None:
            chunked.append(task)
            continue
        children, row = chunk_reviewer_task(task, manifest)
        chunked.extend(children)
        rows.append(row)
        for child in children:
            fps = rational_to_float(manifest.fps) or 30.0
            max_duration_seconds = max(
                max_duration_seconds,
                (int(child["source_frame_end"]) - int(child["source_frame_start"]) + 1) / fps,
            )
    chunked = sorted(
        chunked,
        key=lambda item: (
            item.get("technical_only", False),
            item.get("case_id"),
            task_sort_rank(item),
            item.get("source_frame_start"),
            item.get("review_task_id"),
        ),
    )
    audit = {
        "cases": {
            case_id: {
                "chunked_parent_count": sum(1 for row in rows if row["case_id"] == case_id),
                "child_task_count": sum(
                    row["child_count"] for row in rows if row["case_id"] == case_id
                ),
                "fps": manifest.fps,
            }
            for case_id, manifest in manifests.items()
        },
        "created_at_utc": media.utc_timestamp(),
        "hard_max_seconds": 5.0,
        "max_child_duration_seconds": round(max_duration_seconds, 6),
        "passed": all(
            not child.get("duration_violation") for row in rows for child in row["children"]
        ),
        "parents": rows,
        "schema_version": "review_task_chunking_audit.v1",
        "target_seconds": 3.0,
    }
    return chunked, audit


def chunk_reviewer_task(
    task: dict[str, Any],
    manifest: case_manifests.CaseManifest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fps = rational_to_float(manifest.fps) or 30.0
    hard_max_frames = max(1, int(math.floor(fps * 5.0)))
    target_frames = max(1, min(hard_max_frames, int(round(fps * 3.0))))
    context_frames = max(0, int(round(fps * 0.5)))
    start = int(task.get("source_frame_start") or 0)
    end = int(task.get("source_frame_end") or start)
    duration_frames = end - start + 1
    if duration_frames <= hard_max_frames:
        return [task], {
            "case_id": manifest.case_id,
            "child_count": 1,
            "children": [
                {
                    "duration_seconds": duration_frames / fps,
                    "duration_violation": False,
                    "review_task_id": task.get("review_task_id"),
                    "source_frame_end": end,
                    "source_frame_start": start,
                }
            ],
            "parent_review_task_id": task.get("review_task_id"),
            "split": False,
            "source_frame_end": end,
            "source_frame_start": start,
        }
    children: list[dict[str, Any]] = []
    frame = start
    while frame <= end:
        child_end = min(end, frame + target_frames - 1)
        children.append({"start": frame, "end": child_end})
        frame = child_end + 1
    total = len(children)
    parent_id = str(task.get("review_task_id") or f"{manifest.case_id}_task")
    parent_title = str(task.get("title") or "Review interval")
    child_tasks: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    for index, child in enumerate(children, start=1):
        child_start = int(child["start"])
        child_end = int(child["end"])
        child_id = f"{parent_id}_part_{index:02d}_of_{total:02d}"
        duration_seconds = (child_end - child_start + 1) / fps
        child_task = dict(task) | {
            "child_index": index,
            "child_total": total,
            "parent_raw_signal_ids": list(task.get("raw_signal_ids") or []),
            "parent_review_task_id": parent_id,
            "parent_source_frame_end": end,
            "parent_source_frame_start": start,
            "read_only_context_source_frame_end": min(
                manifest.source_frame_end,
                child_end + context_frames,
            ),
            "read_only_context_source_frame_start": max(
                manifest.source_frame_start,
                child_start - context_frames,
            ),
            "review_task_id": child_id,
            "source_frame_end": child_end,
            "source_frame_start": child_start,
            "task_chunking": {
                "hard_max_seconds": 5.0,
                "schema_version": "review_task_chunk.v1",
                "target_seconds": 3.0,
            },
            "title": f"{parent_title} - part {index} of {total}",
        }
        child_tasks.append(child_task)
        child_rows.append(
            {
                "duration_seconds": round(duration_seconds, 6),
                "duration_violation": duration_seconds > 5.0 + 1e-6,
                "review_task_id": child_id,
                "source_frame_end": child_end,
                "source_frame_start": child_start,
            }
        )
    return child_tasks, {
        "case_id": manifest.case_id,
        "child_count": total,
        "children": child_rows,
        "parent_raw_signal_ids": list(task.get("raw_signal_ids") or []),
        "parent_review_task_id": parent_id,
        "split": True,
        "source_frame_end": end,
        "source_frame_start": start,
    }


def normalize_review_task_class(root_cause: str) -> str:
    if root_cause in {"optional_spot_check", "spot_check"}:
        return "spot_check"
    if root_cause in {"possible_false_redaction", "wrong_target_redaction"}:
        return "possible_false_redaction"
    if root_cause in {"reviewer_created_issue", "manual_reviewer_issue"}:
        return "reviewer_created_issue"
    if root_cause in {
        "unresolved_gap",
        "uncertain_face_coverage",
        "uncertain_plate_track",
        "possible_privacy_gap",
    }:
        return "possible_privacy_gap"
    return "possible_privacy_gap"


def reviewer_task_from_group(index: int, group: list[dict[str, Any]]) -> dict[str, Any]:
    case_id = group[0]["case_id"]
    category = group[0]["category"]
    root = group[0]["root_cause"]
    start = min(item["source_frame_start"] for item in group)
    end = max(item["source_frame_end"] for item in group)
    severity = max((item["severity"] for item in group), key=severity_rank)
    blocking = severity == "blocker" or root == "canonical_day_generation_blocked"
    title, happened, inspect = task_language(category, root, blocking)
    return {
        "allowed_actions": list(REQUIRED_ACTIONS),
        "blocking": blocking,
        "case_id": case_id,
        "category": category,
        "correction_required": blocking,
        "decision_state": "pending",
        "issue_type": root,
        "normal_task_class": root,
        "optional": root == "spot_check",
        "raw_signal_ids": [item["raw_signal_id"] for item in group],
        "review_task_id": f"{case_id}_task_{index:04d}",
        "schema_version": "case_reviewer_task.v1",
        "severity": severity,
        "source_frame_end": end,
        "source_frame_start": start,
        "technical_only": False,
        "title": title,
        "what_happened": happened,
        "what_to_inspect": inspect,
    }


def technical_audit_task(index: int, signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_actions": [],
        "blocking": False,
        "case_id": signal["case_id"],
        "category": signal["category"],
        "correction_required": False,
        "decision_state": "technical_only",
        "issue_type": signal["root_cause"],
        "raw_signal_ids": [signal["raw_signal_id"]],
        "review_task_id": f"technical_task_{index:04d}",
        "schema_version": "case_reviewer_task.v1",
        "severity": signal["severity"],
        "source_frame_end": signal["source_frame_end"],
        "source_frame_start": signal["source_frame_start"],
        "technical_only": True,
        "title": "Technical audit signal",
        "what_happened": "A supporting machine audit signal was preserved for traceability.",
        "what_to_inspect": "Open technical details only if you need raw provenance.",
    }


def task_language(category: str, root: str, blocking: bool) -> tuple[str, str, str]:
    if root == "canonical_day_generation_blocked":
        return (
            "Canonical Day result needs generation",
            "The day face and plate clips have a common parent source, but the frozen detector model files are not available locally to regenerate one combined current result.",
            "Review the lineage report. Do not approve a combined Day result until the frozen pipelines are rerun on the canonical parent interval.",
        )
    if root == "possible_false_redaction":
        return (
            "Possible false redaction",
            "The system may have hidden something that is not a privacy object.",
            "Compare the video at this interval. If the redaction is wrong, open the editor to remove it for the bounded frames.",
        )
    if root == "spot_check":
        return (
            "Optional spot check",
            "This interval was selected as a deterministic quality sample.",
            "Look briefly at the current result. If you see a privacy issue, mark it for correction; otherwise mark it correct.",
        )
    if root == "reviewer_created_issue":
        return (
            "Reviewer-created issue",
            "A reviewer added this issue while sweeping the video.",
            "Open the editor and create or verify the reviewed correction for this interval.",
        )
    if root == "possible_privacy_gap" and category == CATEGORY_PLATE and blocking:
        return (
            "Possible plate privacy gap",
            "The system could not safely maintain plate redaction through this interval.",
            "Compare Original and Current result. If the plate is visible, add a reviewed region before finalization.",
        )
    if root == "possible_privacy_gap" and category == CATEGORY_FACE:
        return (
            "Possible face privacy gap",
            "The system flagged this face interval for human review.",
            "Compare Original and Current result and confirm that the face remains protected without redacting the wrong person.",
        )
    return (
        "Possible privacy gap",
        "The system selected this interval because privacy coverage is uncertain.",
        "Compare Original and Current result and decide whether a correction is needed.",
    )


def write_workspace_assets(
    package_root: Path,
    *,
    frontend: str = "legacy",
    build_frontend: bool = True,
) -> None:
    if frontend not in {"legacy", "react"}:
        raise ValueError(f"unsupported final review frontend: {frontend}")
    if frontend == "legacy":
        write_legacy_workspace_assets(package_root)
        return
    apply_review_entity_coverage_patch(package_root)
    write_react_workspace_assets(package_root, build_frontend=build_frontend)


def write_legacy_workspace_assets(package_root: Path) -> None:
    (package_root / "assets").mkdir(parents=True, exist_ok=True)
    (package_root / "index.html").write_text(index_html(), encoding="utf-8")
    (package_root / "assets" / "workspace.css").write_text(workspace_css(), encoding="utf-8")
    (package_root / "assets" / "workspace.js").write_text(workspace_js(), encoding="utf-8")


def write_react_workspace_assets(
    package_root: Path,
    *,
    build_frontend: bool = True,
    frontend_root: Path | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    frontend_root = frontend_root or repo_root / "frontend"
    dist_root = frontend_root / "dist"
    if build_frontend:
        run_frontend_command(frontend_root, ["npm", "ci"])
        run_frontend_command(frontend_root, ["npm", "run", "build"])
    if not dist_root.exists():
        raise FileNotFoundError(
            "React frontend build output is missing; run npm ci && npm run build in frontend/"
        )
    app_root = package_root / "app"
    if app_root.exists():
        shutil.rmtree(app_root)
    shutil.copytree(dist_root, app_root)
    write_legacy_workspace_assets(package_root)
    legacy_html = (package_root / "index.html").read_text(encoding="utf-8")
    (package_root / "legacy.html").write_text(legacy_html, encoding="utf-8")
    (package_root / "index.html").write_text(react_redirect_html(), encoding="utf-8")
    write_json(
        package_root / "frontend_build_manifest.json",
        frontend_build_manifest(frontend_root=frontend_root, package_root=package_root),
    )


def run_frontend_command(frontend_root: Path, args: list[str]) -> None:
    subprocess.run(args, cwd=frontend_root, check=True)


def react_redirect_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Review Workspace</title>
  <script>window.location.replace('/app/day');</script>
</head>
<body>
  <p><a href="app/index.html">Open Final Review Workspace</a></p>
  <p><a href="legacy.html">Open legacy workspace</a></p>
</body>
</html>
"""


def frontend_build_manifest(*, frontend_root: Path, package_root: Path) -> dict[str, Any]:
    return {
        "api_schema_version": review_api_contract.API_SCHEMA_VERSION,
        "backend_package_version": SCHEMA_VERSION,
        "build_timestamp_utc": media.utc_timestamp(),
        "frontend_milestone_version": FRONTEND_MILESTONE_VERSION,
        "frontend_git_tree_hash": frontend_source_tree_hash(frontend_root),
        "node_version": command_text(["node", "--version"], cwd=frontend_root),
        "npm_version": command_text(["npm", "--version"], cwd=frontend_root),
        "package_lock_sha256": media.sha256_file(frontend_root / "package-lock.json")
        if (frontend_root / "package-lock.json").exists()
        else None,
        "package_root": ".",
        "schema_version": "final_review_frontend_build_manifest.v1",
        "selected_frontend": "react",
        "served_app_path": "app/",
        "legacy_fallback": "legacy.html",
    }


def command_text(args: list[str], *, cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "not_available"


def frontend_source_tree_hash(frontend_root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    ignored_dirs = {"dist", "node_modules", "test-results", "playwright-report"}
    for path in sorted(frontend_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.relative_to(frontend_root).parts):
            continue
        relative = path.relative_to(frontend_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_version_alignment_audit(package_root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    errors: list[str] = []
    for case_path in sorted((package_root / "cases").glob("*.json")):
        manifest = case_manifests.load_case_manifest(case_path)
        if manifest.case_id not in {"day", "night"}:
            continue
        case_errors: list[str] = []
        role_records: list[dict[str, Any]] = []
        sampled_frames = (0, max(0, manifest.frame_count // 2), max(0, manifest.frame_count - 1))
        for role in manifest.video_roles:
            record = version_role_alignment_record(
                package_root=package_root,
                manifest=manifest,
                role=role,
                sampled_case_frames=sampled_frames,
            )
            role_records.append(record)
            case_errors.extend(record["errors"])
        for frame_index in sampled_frames:
            mapped = {
                role_record["role"]: role_record["case_to_source_mapping"].get(str(frame_index))
                for role_record in role_records
            }
            mapped_values = {value for value in mapped.values() if value is not None}
            if len(mapped_values) > 1:
                case_errors.append(
                    f"{manifest.case_id}: case frame {frame_index} maps differently: {mapped}"
                )
        if case_errors:
            errors.extend(case_errors)
        cases[manifest.case_id] = {
            "case_frame_count": manifest.frame_count,
            "display_name": manifest.display_name,
            "errors": case_errors,
            "fps": manifest.fps,
            "height": manifest.height,
            "roles": role_records,
            "schema_version": "case_version_alignment.v1",
            "source_frame_end": manifest.source_frame_end,
            "source_frame_start": manifest.source_frame_start,
            "source_video_hash": manifest.source_video_hash,
            "width": manifest.width,
        }
    audit = {
        "cases": cases,
        "created_at_utc": media.utc_timestamp(),
        "error_count": len(errors),
        "errors": errors,
        "passed": not errors,
        "schema_version": "version_alignment_audit.v1",
    }
    write_json(package_root / "version_alignment_audit.json", audit)
    return audit


def version_role_alignment_record(
    *,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    role: case_manifests.VideoRole,
    sampled_case_frames: tuple[int, ...],
) -> dict[str, Any]:
    errors: list[str] = []
    media_path = package_root / role.path
    info = media.probe_media(media_path) if media_path.exists() else None
    actual_frame_count = int(info.video_stream.frame_count or 0) if info is not None else 0
    actual_fps = (
        rational_to_float(info.video_stream.nominal_frame_rate) if info is not None else None
    )
    expected_fps = rational_to_float(manifest.fps)
    if not media_path.exists():
        errors.append(f"{manifest.case_id}/{role.role}: media missing: {role.path}")
    if role.source_frame_start != manifest.source_frame_start:
        errors.append(
            f"{manifest.case_id}/{role.role}: role source start "
            f"{role.source_frame_start} != manifest {manifest.source_frame_start}"
        )
    if role.source_frame_end != manifest.source_frame_end:
        errors.append(
            f"{manifest.case_id}/{role.role}: role source end "
            f"{role.source_frame_end} != manifest {manifest.source_frame_end}"
        )
    if actual_frame_count and actual_frame_count != manifest.frame_count:
        errors.append(
            f"{manifest.case_id}/{role.role}: frame count {actual_frame_count} "
            f"!= manifest {manifest.frame_count}"
        )
    if info is not None and manifest.width and info.video_stream.width != manifest.width:
        errors.append(
            f"{manifest.case_id}/{role.role}: width {info.video_stream.width} "
            f"!= manifest {manifest.width}"
        )
    if info is not None and manifest.height and info.video_stream.height != manifest.height:
        errors.append(
            f"{manifest.case_id}/{role.role}: height {info.video_stream.height} "
            f"!= manifest {manifest.height}"
        )
    if (
        actual_fps is not None
        and expected_fps is not None
        and abs(actual_fps - expected_fps) > 0.02
    ):
        errors.append(
            f"{manifest.case_id}/{role.role}: fps {actual_fps:.6f} != manifest {expected_fps:.6f}"
        )
    mapping: dict[str, int] = {}
    decoded_evidence: dict[str, Any] = {}
    for case_frame in sampled_case_frames:
        if case_frame >= manifest.frame_count:
            continue
        source_frame = role.source_frame_start + case_frame
        mapping[str(case_frame)] = source_frame
        if source_frame != manifest.source_frame_start + case_frame:
            errors.append(
                f"{manifest.case_id}/{role.role}: case frame {case_frame} maps to "
                f"{source_frame}, expected {manifest.source_frame_start + case_frame}"
            )
        decoded_evidence[str(case_frame)] = decoded_frame_evidence(media_path, case_frame)
    return {
        "actual_duration_seconds": info.duration_seconds if info is not None else None,
        "actual_fps": actual_fps,
        "actual_frame_count": actual_frame_count,
        "actual_height": info.video_stream.height if info is not None else None,
        "actual_width": info.video_stream.width if info is not None else None,
        "case_to_source_mapping": mapping,
        "decoded_frame_evidence": decoded_evidence,
        "errors": errors,
        "media_path": role.path,
        "media_sha256": media.sha256_file(media_path) if media_path.exists() else None,
        "role": role.role,
        "role_label": role.label,
        "schema_version": "version_role_alignment.v1",
        "source_frame_end": role.source_frame_end,
        "source_frame_start": role.source_frame_start,
        "source_video_hash": manifest.source_video_hash,
    }


def decoded_frame_evidence(media_path: Path, case_frame: int) -> dict[str, Any]:
    if not media_path.exists():
        return {"decoded": False, "reason": "media_missing"}
    capture = cv.VideoCapture(str(media_path))
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, case_frame)
        ok, frame = capture.read()
        if not ok or frame is None:
            return {"decoded": False, "reason": "frame_unavailable"}
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return {
            "decoded": True,
            "frame_hash": stable_hash(
                {
                    "case_frame": case_frame,
                    "mean": round(float(gray.mean()), 6),
                    "shape": list(gray.shape),
                    "std": round(float(gray.std()), 6),
                }
            ),
            "mean_luma": round(float(gray.mean()), 3),
            "std_luma": round(float(gray.std()), 3),
        }
    finally:
        capture.release()


def write_manual_firefox_checklist(package_root: Path) -> None:
    text = """# Manual Firefox Acceptance Checklist

Open the workspace with:

`redactor serve-final-review-workspace artifacts/final_pipeline_package_v2_3_3 --open`

- Start review creates a visible session id and survives reload.
- Day and Night media play inline; no Picture-in-Picture opens.
- Healthy videos show no Ready/Retry badge; media failures show a compact error and Retry.
- Fullscreen uses the application player wrapper and keeps the source-frame HUD visible.
- ArrowRight advances exactly one frame; ArrowLeft returns exactly one frame.
- Shift+ArrowRight and Shift+ArrowLeft move ten frames.
- Ten rapid frame steps do not leave the player black.
- Switching Original / Automatic / Latest reviewed preserves the active frame.
- Timeline markers select the matching review item, and review items highlight markers.
- Review decisions persist before auto-advance.
- Edit mode can add an ad hoc issue and save a correction preview.
- Entity cards show useful Person/Plate thumbnails and compact reviewer actions.
- Finalization remains blocked while unresolved items, unverified corrections, or unverified Keep/Not-sensitive actions remain.
"""
    (package_root / "manual_firefox_acceptance_checklist.md").write_text(text, encoding="utf-8")


def materialization_acceptance_fields(
    *,
    materialization_plan: dict[str, Any],
    accepted_revision: str,
) -> dict[str, Any]:
    plan = dict(materialization_plan.get("plan") or {})
    job = dict(materialization_plan.get("job") or {})
    materialization_required = bool(plan.get("materialization_required"))
    publication_status = (
        str(job.get("publication_status") or job.get("status") or "queued")
        if materialization_required
        else "published"
    )
    return {
        "accepted_working_state_revision": accepted_revision,
        "changed_frame_count": int(plan.get("changed_source_frame_count") or 0),
        "changed_frame_intervals": list(plan.get("changed_source_frame_intervals") or []),
        "materialization_job_id": job.get("job_id"),
        "materialization_required": materialization_required,
        "publication_status": publication_status,
        "target_effective_region_count": int(plan.get("after_effective_region_count") or 0),
        "target_effective_regions_hash": plan.get("after_effective_regions_hash"),
    }


class FinalReviewWorkspaceServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.materialization_jobs = MaterializationJobManager(self.package_root)
        super().__init__(server_address, FinalReviewWorkspaceHandler)


class MaterializationJobManager:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.case_state: dict[tuple[str, str], dict[str, Any]] = {}
        self.debounce_seconds = 0.85
        self.watchdog_seconds = 15 * 60
        self.metrics: dict[str, dict[str, Any]] = {}
        self.timing_records: list[dict[str, Any]] = []

    def enqueue(
        self,
        *,
        session_dir: Path,
        case_id: str,
        actor: dict[str, Any],
        reason: str,
        accepted_revision: str | None,
        affected_entity_ids: list[str] | None = None,
        requested_policy: str | None = None,
        target_effective_region_count: int | None = None,
        target_effective_regions_hash: str | None = None,
        materialization_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = f"materialize_{uuid.uuid4().hex}"
        session_dir = session_dir.resolve()
        session_id = session_dir.name
        plan = dict(materialization_plan or {})
        job = {
            "accepted_working_state_revision": accepted_revision,
            "affected_entity_ids": affected_entity_ids or [],
            "accepted_at_utc": media.utc_timestamp(),
            "case_id": case_id,
            "changed_segment_count": 0,
            "changed_source_frame_count": plan.get("changed_source_frame_count", 0),
            "changed_source_frame_intervals": plan.get("changed_source_frame_intervals", []),
            "changed_source_frame_percentage": plan.get("changed_source_frame_percentage", 0.0),
            "completed_at_utc": None,
            "created_at_utc": media.utc_timestamp(),
            "current_stage": "waiting",
            "elapsed_time_ms": 0,
            "encode_stage": "waiting",
            "error": None,
            "job_id": job_id,
            "latest_reviewed": None,
            "last_progress_at_utc": media.utc_timestamp(),
            "materialization": None,
            "materialization_plan": plan,
            "planned_frames_to_encode": plan.get("planned_frames_to_encode", 0),
            "planned_frames_to_render": plan.get("planned_frames_to_render", 0),
            "publication_status": "queued",
            "queue_state": "latest_pending",
            "queue_wait_time_ms": 0,
            "progress": 0.0,
            "queued_until_utc": None,
            "reason": reason,
            "requested_policy": requested_policy,
            "reused_segment_count": 0,
            "schema_version": "reviewed_media_materialization_job.v1",
            "session_id": session_id,
            "started_at_utc": None,
            "status": "queued",
            "superseded_by": None,
            "target_effective_region_count": target_effective_region_count,
            "target_effective_regions_hash": target_effective_regions_hash,
            "validation_stage": "waiting",
            "_last_progress_monotonic": time.monotonic(),
        }
        key = (session_id, case_id)
        with self.lock:
            self.jobs[job_id] = job
            state = self.case_state.setdefault(
                key,
                {"debounce_timer": None, "running": None, "latest_requested": None},
            )
            previous_pending = state.get("latest_requested")
            self._metric(case_id, "accepted_state_mutations", 1)
            if previous_pending and previous_pending != state.get("running"):
                pending_job = self.jobs.get(str(previous_pending))
                if pending_job and pending_job.get("status") == "queued":
                    job["affected_entity_ids"] = unique_sorted(
                        [
                            *list(pending_job.get("affected_entity_ids") or []),
                            *list(job.get("affected_entity_ids") or []),
                        ]
                    )
                    pending_job["status"] = "superseded"
                    pending_job["publication_status"] = "superseded"
                    pending_job["queue_state"] = "superseded_by_newer_target"
                    pending_job["superseded_by"] = job_id
                    self._metric(case_id, "jobs_coalesced", 1)
                    self._record_timing_locked(
                        pending_job,
                        status="superseded",
                        state_projection_duration_ms=0,
                    )
            state["latest_requested"] = job_id
            if not state.get("running"):
                self._schedule_locked(key, session_dir, actor)
            self._write_audit_locked()
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            self._apply_watchdog_locked()
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def compatible_active_job(
        self,
        *,
        session_dir: Path,
        case_id: str,
        target_effective_regions_hash: str,
    ) -> dict[str, Any] | None:
        session_id = session_dir.resolve().name
        key = (session_id, case_id)
        with self.lock:
            state = self.case_state.get(key) or {}
            for job_id in (state.get("running"), state.get("latest_requested")):
                job = self.jobs.get(str(job_id or ""))
                if not job:
                    continue
                if job.get("status") not in {"queued", "running"}:
                    continue
                if job.get("target_effective_regions_hash") == target_effective_regions_hash:
                    job["queue_state"] = "compatible_active_job_reused"
                    self._metric(case_id, "jobs_coalesced", 1)
                    self._write_audit_locked()
                    return dict(job)
        return None

    def record_skipped_materialization(
        self,
        *,
        session_dir: Path,
        case_id: str,
        accepted_revision: str | None,
        affected_entity_ids: list[str] | None,
        before_digest: dict[str, Any],
        after_digest: dict[str, Any],
        operation: str,
        reason: str,
    ) -> dict[str, Any]:
        job_id = f"materialize_skip_{uuid.uuid4().hex}"
        job = {
            "accepted_working_state_revision": accepted_revision,
            "affected_entity_ids": affected_entity_ids or [],
            "accepted_at_utc": media.utc_timestamp(),
            "case_id": case_id,
            "changed_segment_count": 0,
            "changed_source_frame_count": 0,
            "changed_source_frame_intervals": [],
            "changed_source_frame_percentage": 0.0,
            "completed_at_utc": media.utc_timestamp(),
            "created_at_utc": media.utc_timestamp(),
            "current_stage": "complete",
            "elapsed_time_ms": 0,
            "encode_stage": "not_required",
            "error": None,
            "job_id": job_id,
            "latest_reviewed": None,
            "last_progress_at_utc": media.utc_timestamp(),
            "materialization": None,
            "materialization_plan": {},
            "operation": operation,
            "planned_frames_to_encode": 0,
            "planned_frames_to_render": 0,
            "publication_status": "published",
            "queue_state": "skipped_no_change",
            "queue_wait_time_ms": 0,
            "progress": 1.0,
            "reason": reason,
            "reused_segment_count": 0,
            "schema_version": "reviewed_media_materialization_job.v1",
            "session_id": session_dir.resolve().name,
            "started_at_utc": None,
            "status": "skipped_no_effective_region_change",
            "superseded_by": None,
            "target_effective_region_count": after_digest.get("region_count"),
            "target_effective_regions_hash": after_digest.get("effective_regions_hash"),
            "validation_stage": "not_required",
        }
        with self.lock:
            self.jobs[job_id] = job
            self._metric(case_id, "accepted_state_mutations", 1)
            self._metric(case_id, "materialization_jobs_skipped", 1)
            self._record_timing_locked(
                job,
                status="skipped_no_effective_region_change",
                state_projection_duration_ms=0,
                materialization={
                    "timing": {
                        "effective_region_derivation_duration_ms": int(
                            before_digest.get("duration_ms") or 0
                        )
                        + int(after_digest.get("duration_ms") or 0),
                        "output_frame_count": 0,
                        "publication_duration_ms": 0,
                        "render_encode_duration_ms": 0,
                        "total_duration_ms": 0,
                    }
                },
            )
            self._write_audit_locked()
        return dict(job)

    def retry(self, *, job_id: str, actor: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            session_dir = review_sessions.session_directory(
                self.package_root,
                str(job.get("session_id") or ""),
            )
        return self.enqueue(
            session_dir=session_dir,
            case_id=str(job.get("case_id") or ""),
            actor=actor,
            reason=f"Retry: {job.get('reason') or 'Latest reviewed media update.'}",
            accepted_revision=str(job.get("accepted_working_state_revision") or ""),
            affected_entity_ids=list(job.get("affected_entity_ids") or []),
            requested_policy=job.get("requested_policy"),
            target_effective_region_count=job.get("target_effective_region_count"),
            target_effective_regions_hash=job.get("target_effective_regions_hash"),
            materialization_plan=dict(job.get("materialization_plan") or {}),
        )

    def _start_locked(
        self,
        key: tuple[str, str],
        job_id: str,
        session_dir: Path,
        actor: dict[str, Any],
    ) -> None:
        state = self.case_state.setdefault(
            key,
            {"debounce_timer": None, "running": None, "latest_requested": None},
        )
        timer = state.get("debounce_timer")
        if isinstance(timer, threading.Timer):
            timer.cancel()
        state["debounce_timer"] = None
        state["running"] = job_id
        job = self.jobs[job_id]
        job["status"] = "running"
        job["started_at_utc"] = media.utc_timestamp()
        job["current_stage"] = "planning"
        job["publication_status"] = "running"
        job["queue_state"] = "running"
        job["last_progress_at_utc"] = media.utc_timestamp()
        job["_last_progress_monotonic"] = time.monotonic()
        job["progress"] = 0.05
        self._metric(str(job.get("case_id") or ""), "materialization_jobs_started", 1)
        thread = threading.Thread(
            target=self._run_job,
            args=(key, job_id, session_dir.resolve(), dict(actor)),
            daemon=True,
        )
        thread.start()

    def _schedule_locked(
        self,
        key: tuple[str, str],
        session_dir: Path,
        actor: dict[str, Any],
    ) -> None:
        state = self.case_state.setdefault(
            key,
            {"debounce_timer": None, "running": None, "latest_requested": None},
        )
        timer = state.get("debounce_timer")
        if isinstance(timer, threading.Timer):
            timer.cancel()
        job_id = str(state.get("latest_requested") or "")
        job = self.jobs.get(job_id)
        if job is not None:
            job["queued_until_utc"] = media.utc_timestamp()
            job["publication_status"] = "queued"
            job["queue_state"] = "debouncing_latest_target"
            job["last_progress_at_utc"] = media.utc_timestamp()
            job["_last_progress_monotonic"] = time.monotonic()
        timer = threading.Timer(
            self.debounce_seconds,
            self._start_latest_after_debounce,
            args=(key, session_dir.resolve(), dict(actor)),
        )
        timer.daemon = True
        state["debounce_timer"] = timer
        timer.start()

    def _start_latest_after_debounce(
        self,
        key: tuple[str, str],
        session_dir: Path,
        actor: dict[str, Any],
    ) -> None:
        with self.lock:
            state = self.case_state.setdefault(
                key,
                {"debounce_timer": None, "running": None, "latest_requested": None},
            )
            if state.get("running"):
                return
            job_id = str(state.get("latest_requested") or "")
            job = self.jobs.get(job_id)
            if not job or job.get("status") != "queued":
                return
            self._start_locked(key, job_id, session_dir, actor)

    def _run_job(
        self,
        key: tuple[str, str],
        job_id: str,
        session_dir: Path,
        actor: dict[str, Any],
    ) -> None:
        job = self.jobs[job_id]
        case_id = str(job["case_id"])
        accepted_revision = str(job.get("accepted_working_state_revision") or "")
        target_effective_hash = str(job.get("target_effective_regions_hash") or "")
        start_monotonic = time.monotonic()
        state_projection_start = time.monotonic()
        current_state = review_working_state.derive_working_state(self.package_root, session_dir)
        state_projection_duration_ms = int((time.monotonic() - state_projection_start) * 1000)
        current_revision = str(
            current_state.get("active_projection_hash") or current_state.get("state_hash") or ""
        )
        current_digest = review_working_state.effective_privacy_region_digest(
            self.package_root,
            session_dir,
            case_id=case_id,
        )
        compatible_revision_drift = bool(
            accepted_revision
            and current_revision != accepted_revision
            and target_effective_hash
            and current_digest.get("effective_regions_hash") == target_effective_hash
        )
        if (
            accepted_revision
            and current_revision != accepted_revision
            and not compatible_revision_drift
        ):
            with self.lock:
                job["completed_at_utc"] = media.utc_timestamp()
                job["current_stage"] = "superseded"
                job["error"] = "Superseded by a newer accepted review state."
                job["publication_status"] = "superseded"
                job["queue_state"] = "discarded_stale_revision"
                job["progress"] = 1.0
                job["status"] = "superseded"
                self._metric(case_id, "stale_jobs_discarded", 1)
                self._record_timing_locked(
                    job,
                    status="superseded",
                    state_projection_duration_ms=state_projection_duration_ms,
                )
                self._write_audit_locked()
            self._start_next_if_needed(key, session_dir, actor, completed_job_id=job_id)
            return
        start_event = review_sessions.append_event(
            session_dir,
            event_type="reviewed_media_materialization_started",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state=None,
            after_state={
                "case_id": case_id,
                "job_id": job_id,
                "state_hash": job.get("accepted_working_state_revision"),
            },
            note=str(job.get("reason") or "Latest reviewed media update started."),
            application_hashes={},
        )
        try:
            with self.lock:
                job["current_stage"] = "rendering_full_media"
                job["encode_stage"] = "rendering"
                job["last_progress_at_utc"] = media.utc_timestamp()
                job["_last_progress_monotonic"] = time.monotonic()
                job["progress"] = 0.15
                self._write_audit_locked()
            materialization = self._cached_materialization(
                session_dir=session_dir,
                case_id=case_id,
                accepted_revision=accepted_revision,
                target_effective_regions_hash=target_effective_hash,
            )
            if materialization is None:
                materialization = review_working_state.materialize_latest_reviewed(
                    self.package_root,
                    session_dir,
                    case_id=case_id,
                    accepted_revision=accepted_revision,
                    actor=None,
                    event_reason=str(job.get("reason") or "Latest reviewed media updated."),
                )
            else:
                self._metric(case_id, "cache_hits", 1)
            with self.lock:
                job["current_stage"] = "validating_publication"
                job["encode_stage"] = "complete"
                job["validation_stage"] = "validating"
                job["last_progress_at_utc"] = media.utc_timestamp()
                job["_last_progress_monotonic"] = time.monotonic()
                job["progress"] = 0.85
                self._write_audit_locked()
            after_render_state = review_working_state.derive_working_state(
                self.package_root,
                session_dir,
            )
            after_render_revision = str(
                after_render_state.get("active_projection_hash")
                or after_render_state.get("state_hash")
                or ""
            )
            after_render_digest = review_working_state.effective_privacy_region_digest(
                self.package_root,
                session_dir,
                case_id=case_id,
            )
            render_target_stale = bool(
                target_effective_hash
                and after_render_digest.get("effective_regions_hash") != target_effective_hash
            )
            revision_stale_without_region_target = bool(
                accepted_revision
                and after_render_revision != accepted_revision
                and not target_effective_hash
            )
            if revision_stale_without_region_target or render_target_stale:
                with self.lock:
                    job["completed_at_utc"] = media.utc_timestamp()
                    job["current_stage"] = "superseded"
                    job["error"] = (
                        "Rendered result discarded because a newer review state was accepted."
                    )
                    job["publication_status"] = "superseded"
                    job["queue_state"] = "discarded_stale_render"
                    job["progress"] = 1.0
                    job["status"] = "superseded"
                    self._metric(case_id, "stale_jobs_discarded", 1)
                    self._record_timing_locked(
                        job,
                        materialization=materialization,
                        status="superseded",
                        state_projection_duration_ms=state_projection_duration_ms,
                    )
                    self._write_audit_locked()
                review_working_state.derive_working_state(self.package_root, session_dir)
                return
            completed_state = review_working_state.derive_working_state(
                self.package_root,
                session_dir,
            )
            complete_event = review_sessions.append_event(
                session_dir,
                event_type="reviewed_media_materialization_completed",
                actor=actor,
                case_id=case_id,
                task_id=None,
                before_state={"job_id": job_id, "started_event_hash": start_event["event_hash"]},
                after_state={
                    "accepted_working_state_revision": accepted_revision,
                    "case_id": case_id,
                    "effective_regions_hash": materialization.get("effective_regions_hash"),
                    "job_id": job_id,
                    "latest_reviewed": materialization["latest_reviewed"],
                    "media_validation_passed": dict(
                        materialization.get("media_validation") or {}
                    ).get("passed")
                    is True,
                    "active_projection_hash": completed_state.get("active_projection_hash"),
                    "correction_render_verifications": materialization.get(
                        "correction_render_verifications"
                    )
                    or {},
                    "published_working_state_revision": completed_state.get(
                        "active_projection_hash"
                    ),
                    "state_hash": completed_state.get("state_hash"),
                    "target_effective_regions_hash": target_effective_hash,
                },
                note="Latest reviewed media update completed.",
                application_hashes={
                    "latest_reviewed_sha256": str(materialization.get("video_sha256") or "")
                },
            )
            with self.lock:
                duration_ms = int((time.monotonic() - start_monotonic) * 1000)
                job["completed_at_utc"] = media.utc_timestamp()
                job["completed_event_hash"] = complete_event["event_hash"]
                job["latest_reviewed"] = materialization["latest_reviewed"]
                job["materialization"] = materialization
                job["published_working_state_revision"] = completed_state.get(
                    "active_projection_hash"
                )
                job["current_stage"] = "complete"
                job["elapsed_time_ms"] = duration_ms
                job["publication_status"] = "published"
                job["queue_state"] = "complete"
                job["validation_stage"] = "passed"
                job["progress"] = 1.0
                job["status"] = "completed"
                job["working_state"] = review_working_state.derive_working_state(
                    self.package_root,
                    session_dir,
                )
                self._metric(case_id, "full_materialization_duration_ms", duration_ms, mode="set")
                self._metric(
                    case_id,
                    "final_published_revision",
                    str(job.get("published_working_state_revision") or ""),
                    mode="set",
                )
                self._record_timing_locked(
                    job,
                    materialization=materialization,
                    status="completed",
                    state_projection_duration_ms=state_projection_duration_ms,
                )
                self._write_audit_locked()
        except Exception as exc:  # pragma: no cover - exercised through server integration
            failure_event = review_sessions.append_event(
                session_dir,
                event_type="reviewed_media_materialization_failed",
                actor=actor,
                case_id=case_id,
                task_id=None,
                before_state={"job_id": job_id, "started_event_hash": start_event["event_hash"]},
                after_state={"case_id": case_id, "error": str(exc), "job_id": job_id},
                note=f"Latest reviewed media update failed: {exc}",
                application_hashes={},
            )
            with self.lock:
                job["completed_at_utc"] = media.utc_timestamp()
                job["current_stage"] = "failed"
                job["error"] = str(exc)
                job["failed_event_hash"] = failure_event["event_hash"]
                job["publication_status"] = "failed"
                job["queue_state"] = "failed_retryable"
                job["validation_stage"] = "failed"
                job["progress"] = 1.0
                job["status"] = "failed"
                self._record_timing_locked(
                    job,
                    status="failed",
                    state_projection_duration_ms=state_projection_duration_ms,
                )
                self._write_audit_locked()
        finally:
            self._start_next_if_needed(key, session_dir, actor, completed_job_id=job_id)

    def _start_next_if_needed(
        self,
        key: tuple[str, str],
        session_dir: Path,
        actor: dict[str, Any],
        *,
        completed_job_id: str,
    ) -> None:
        with self.lock:
            state = self.case_state.setdefault(
                key,
                {"debounce_timer": None, "running": None, "latest_requested": None},
            )
            if state.get("running") == completed_job_id:
                state["running"] = None
            latest = state.get("latest_requested")
            if latest and latest != completed_job_id:
                next_job = self.jobs.get(str(latest))
                if next_job and next_job.get("status") == "queued":
                    self._schedule_locked(key, session_dir, actor)

    def _apply_watchdog_locked(self) -> None:
        now = time.monotonic()
        for key, state in self.case_state.items():
            running_id = str(state.get("running") or "")
            job = self.jobs.get(running_id)
            if not job or job.get("status") != "running":
                continue
            last_progress = float(job.get("_last_progress_monotonic") or now)
            if now - last_progress <= self.watchdog_seconds:
                continue
            job["completed_at_utc"] = media.utc_timestamp()
            job["current_stage"] = "failed"
            job["error"] = (
                "Latest reviewed media update made no progress before the watchdog limit."
            )
            job["publication_status"] = "failed"
            job["queue_state"] = "failed_retryable"
            job["status"] = "failed"
            job["validation_stage"] = "failed"
            job["progress"] = 1.0
            state["running"] = None
            self._metric(str(job.get("case_id") or ""), "stale_jobs_discarded", 1)

    def _cached_materialization(
        self,
        *,
        session_dir: Path,
        case_id: str,
        accepted_revision: str,
        target_effective_regions_hash: str | None = None,
    ) -> dict[str, Any] | None:
        if not accepted_revision and not target_effective_regions_hash:
            return None
        manifest_path = session_dir / "state" / f"{case_id}_latest_reviewed_manifest.json"
        if not manifest_path.exists():
            return None
        materialization = read_json_file(manifest_path)
        revision_matches = bool(
            accepted_revision
            and materialization.get("accepted_working_state_revision") == accepted_revision
        )
        region_hash_matches = bool(
            target_effective_regions_hash
            and materialization.get("effective_regions_hash") == target_effective_regions_hash
        )
        if not revision_matches and not region_hash_matches:
            return None
        latest = dict(materialization.get("latest_reviewed") or {})
        output = self.package_root / str(latest.get("path") or "")
        if not latest or not output.exists():
            return None
        return materialization

    def _metric(self, case_id: str, name: str, value: Any, *, mode: str = "add") -> None:
        if not case_id:
            return
        row = self.metrics.setdefault(
            case_id,
            {
                "accepted_state_mutations": 0,
                "api_acceptance_latency_ms": 0,
                "cache_hits": 0,
                "final_published_revision": None,
                "full_materialization_duration_ms": None,
                "jobs_coalesced": 0,
                "materialization_jobs_started": 0,
                "materialization_jobs_skipped": 0,
                "stale_jobs_discarded": 0,
            },
        )
        if mode == "set":
            row[name] = value
            return
        row[name] = int(row.get(name) or 0) + int(value)

    def _write_audit_locked(self) -> None:
        write_json(
            self.package_root / "materialization_job_audit.json",
            {
                "cases": self.metrics,
                "created_at_utc": media.utc_timestamp(),
                "schema_version": "materialization_job_audit.v1",
            },
        )
        write_json(
            self.package_root / "materialization_timing_audit.json",
            {
                "created_at_utc": media.utc_timestamp(),
                "jobs": self.timing_records,
                "schema_version": "materialization_timing_audit.v1",
                "summary_by_case": self.metrics,
            },
        )

    def _record_timing_locked(
        self,
        job: dict[str, Any],
        *,
        status: str,
        state_projection_duration_ms: int,
        materialization: dict[str, Any] | None = None,
    ) -> None:
        timing = dict((materialization or {}).get("timing") or {})
        record = {
            "accepted_at_utc": job.get("accepted_at_utc") or job.get("created_at_utc"),
            "accepted_working_state_revision": job.get("accepted_working_state_revision"),
            "affected_entity_ids": list(job.get("affected_entity_ids") or []),
            "case_id": job.get("case_id"),
            "changed_segment_count": job.get("changed_segment_count"),
            "changed_source_frame_count": job.get("changed_source_frame_count"),
            "changed_source_frame_percentage": job.get("changed_source_frame_percentage"),
            "completed_at_utc": job.get("completed_at_utc"),
            "current_stage": job.get("current_stage"),
            "error": job.get("error"),
            "encoded_frame_count": timing.get("output_frame_count"),
            "job_id": job.get("job_id"),
            "materialization_strategy": (
                dict(job.get("materialization_plan") or {}).get("materialization_strategy")
            ),
            "mutation_accepted_time": job.get("accepted_at_utc") or job.get("created_at_utc"),
            "output_frame_count": timing.get("output_frame_count"),
            "planned_frames_to_encode": job.get("planned_frames_to_encode"),
            "planned_frames_to_render": job.get("planned_frames_to_render"),
            "publication_duration_ms": timing.get("publication_duration_ms"),
            "published_working_state_revision": job.get("published_working_state_revision"),
            "render_encode_duration_ms": timing.get("render_encode_duration_ms"),
            "requested_policy": job.get("requested_policy"),
            "reused_segment_count": job.get("reused_segment_count"),
            "state_projection_duration_ms": state_projection_duration_ms,
            "status": status,
            "superseded_by": job.get("superseded_by"),
            "target_effective_region_count": job.get("target_effective_region_count"),
            "target_effective_regions_hash": job.get("target_effective_regions_hash"),
            "total_time_ms": timing.get("total_duration_ms"),
            "effective_region_derivation_duration_ms": timing.get(
                "effective_region_derivation_duration_ms"
            ),
        }
        self.timing_records = [
            row for row in self.timing_records if row.get("job_id") != job.get("job_id")
        ]
        self.timing_records.append(record)


class FinalReviewWorkspaceHandler(SimpleHTTPRequestHandler):
    server: FinalReviewWorkspaceServer

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        relative = unquote(parsed.path.lstrip("/")) or "index.html"
        resolved = (self.server.package_root / relative).resolve()
        try:
            resolved.relative_to(self.server.package_root)
        except ValueError:
            return str(self.server.package_root / "__blocked__")
        return str(resolved)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if "Broken pipe" not in format:
            super().log_message(format, *args)

    def do_POST(self) -> None:
        try:
            self._do_post()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self.send_json({"error": str(exc), "ok": False}, status=HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        try:
            if self._do_api_get():
                return
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self.send_json({"error": str(exc), "ok": False}, status=HTTPStatus.BAD_REQUEST)
            return
        if self.headers.get("Range"):
            self.serve_range_request()
            return
        if self.should_serve_spa():
            self.path = "/app/index.html"
        super().do_GET()

    def _do_api_get(self) -> bool:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/v1/schema":
            self.send_json(review_api_contract.final_review_workspace_schema())
            return True
        if path in {"/api/v1/workspace", "/api/v1/workspace/summary"}:
            self.send_json(review_api_contract.workspace_summary(self.server.package_root))
            return True
        if path == "/api/v1/working-state":
            query = parse_qs(parsed.query)
            session_dir = self.session_dir_from_query(query)
            if session_dir is None:
                return True
            self.send_json(
                {
                    "ok": True,
                    "working_state": review_working_state.derive_working_state(
                        self.server.package_root,
                        session_dir,
                    ),
                }
            )
            return True
        if path == "/api/v1/history":
            query = parse_qs(parsed.query)
            session_dir = self.session_dir_from_query(query)
            if session_dir is None:
                return True
            self.send_json(review_working_state.history_events(session_dir))
            return True
        if path == "/api/v1/checkpoints":
            query = parse_qs(parsed.query)
            session_dir = self.session_dir_from_query(query)
            if session_dir is None:
                return True
            self.send_json(
                {
                    "checkpoints": review_working_state.load_checkpoints(session_dir),
                    "ok": True,
                    "schema_version": "review_checkpoint_list.v1",
                }
            )
            return True
        if path.startswith("/api/v1/materialization-jobs/"):
            job_id = path.removeprefix("/api/v1/materialization-jobs/").strip("/")
            job = self.server.materialization_jobs.get(job_id)
            if job is None:
                self.send_json(
                    {"error": "materialization job not found", "ok": False},
                    status=HTTPStatus.NOT_FOUND,
                )
                return True
            self.send_json({"job": job, "ok": True})
            return True
        if path.startswith("/api/v1/cases/") and path.endswith("/versions"):
            case_id = path.removeprefix("/api/v1/cases/").removesuffix("/versions").strip("/")
            query = parse_qs(parsed.query)
            session_id = str((query.get("session_id") or [""])[0])
            session_dir = (
                review_sessions.session_directory(self.server.package_root, session_id)
                if session_id
                else None
            )
            self.send_json(
                review_working_state.case_versions(
                    self.server.package_root,
                    session_dir,
                    case_id=case_id,
                )
            )
            return True
        if path.startswith("/api/v1/cases/"):
            case_id = path.rsplit("/", 1)[-1]
            case_path = self.server.package_root / "cases" / f"{case_id}.json"
            if not case_path.exists():
                self.send_json(
                    {"error": "case not found", "ok": False},
                    status=HTTPStatus.NOT_FOUND,
                )
                return True
            self.send_json(read_json_file(case_path))
            return True
        if path == "/api/v1/entities":
            self.send_json(read_json_file(self.server.package_root / "review_entities.json"))
            return True
        path = legacy_api_path(path)
        if path == "/api/workspace":
            self.send_json(
                {
                    "ok": True,
                    "workspace": read_json_file(
                        self.server.package_root / "workspace_manifest.json"
                    ),
                }
            )
            return True
        if path == "/api/session-state":
            query = parse_qs(parsed.query)
            session_id = str((query.get("session_id") or [""])[0])
            session_dir = review_sessions.session_directory(self.server.package_root, session_id)
            if not session_dir.exists():
                self.send_json(
                    {"error": "review session not found", "ok": False},
                    status=HTTPStatus.NOT_FOUND,
                )
                return True
            working_state = review_working_state.derive_working_state(
                self.server.package_root,
                session_dir,
            )
            self.send_json(
                {
                    "attribution": review_sessions.load_reviewer_attribution(session_dir),
                    "chain": review_sessions.validate_event_hash_chain(session_dir),
                    "corrections": list(working_state.get("manual_corrections") or []),
                    "decisions": dict(working_state.get("task_decisions") or {}),
                    "events": review_sessions.load_events(session_dir),
                    "ok": True,
                    "session": read_json_file(session_dir / "review_session.json"),
                    "summary": read_json_file(session_dir / "review_summary.json"),
                    "working_state": working_state,
                }
            )
            return True
        if path == "/api/entities":
            self.send_json(
                {
                    "entities": read_json_file(self.server.package_root / "review_entities.json"),
                    "ok": True,
                }
            )
            return True
        return False

    def _do_post(self) -> None:
        payload = self.read_json_body()
        path = legacy_api_path(urlparse(self.path).path)
        if path == "/api/sessions":
            freeze_hash = stable_hash(
                json.loads(
                    (self.server.package_root / "configuration_freeze.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            case_hashes: dict[str, str] = {}
            source_hashes: dict[str, str] = {}
            for case_id in ("day", "night"):
                case_path = self.server.package_root / "cases" / f"{case_id}.json"
                if case_path.exists():
                    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
                    case_hashes[case_id] = stable_hash(case_payload)
                    if case_payload.get("source_video_hash"):
                        source_hashes[case_id] = str(case_payload["source_video_hash"])
            session = review_sessions.create_review_session(
                package_root=self.server.package_root,
                reviewer=review_sessions.reviewer_identity_from_dict(payload.get("reviewer") or {}),
                source_revision=payload.get("source_revision"),
                configuration_freeze_hash=freeze_hash,
                case_manifest_hashes=case_hashes,
                source_hashes=source_hashes,
                user_agent=self.headers.get("User-Agent"),
                include_hostname=bool(payload.get("include_hostname")),
            )
            working_state = review_working_state.derive_working_state(
                self.server.package_root,
                review_sessions.session_directory(self.server.package_root, session.session_id),
            )
            self.send_json(
                {"ok": True, "session": session.to_dict(), "working_state": working_state}
            )
            return
        session_dir = review_sessions.session_directory(
            self.server.package_root,
            str(payload.get("session_id") or ""),
        )
        actor = dict(payload.get("actor") or {})
        if path == "/api/reviewer-attribution":
            event = review_sessions.append_reviewer_attribution_event(
                session_dir,
                actor=actor,
                display_name=str(payload.get("display_name") or ""),
                note=str(payload.get("note") or ""),
            )
            self.send_json(
                {
                    "attribution": review_sessions.load_reviewer_attribution(session_dir),
                    "event": event,
                    "ok": True,
                }
            )
            return
        if path == "/api/decisions":
            gate = review_working_state.correction_verification_decision_gate(
                self.server.package_root,
                session_dir,
                task_id=str(payload["task_id"]),
            )
            if not gate.get("ok"):
                self.send_json(
                    {"error": gate.get("error"), "ok": False, "task": gate.get("task")},
                    status=HTTPStatus.CONFLICT,
                )
                return
            event = review_sessions.append_decision_event(
                session_dir,
                actor=actor,
                case_id=str(payload["case_id"]),
                task_id=str(payload["task_id"]),
                action=str(payload["action"]),
                note=str(payload.get("note") or ""),
            )
            self.send_json({"event": event, "ok": True})
            return
        if path == "/api/events/sweep":
            event = review_sessions.append_sweep_acknowledgement(
                session_dir,
                actor=actor,
                case_id=str(payload["case_id"]),
                viewed_ranges=list(payload.get("viewed_ranges") or []),
                override_reason=payload.get("override_reason"),
            )
            self.send_json({"event": event, "ok": True})
            return
        if path == "/api/corrections/preview":
            from redactor.review_corrections import correction_from_dict

            correction = correction_from_dict(payload["correction"])
            preview = review_working_state.render_correction_preview(
                self.server.package_root,
                session_dir=session_dir,
                correction=correction,
                include_unsaved_correction=True,
            )
            if not preview.get("ok"):
                self.send_json(
                    {"error": preview.get("reason") or "Correction preview failed.", "ok": False},
                    status=HTTPStatus.CONFLICT,
                )
                return
            event = review_sessions.append_correction_preview_event(
                session_dir,
                actor=actor,
                correction=correction,
                preview=preview,
            )
            self.send_json({"event": event, "ok": True, "preview": preview})
            return
        if path == "/api/corrections":
            from redactor.review_corrections import correction_from_dict

            correction_payload = dict(payload["correction"])
            token = str(payload.get("preview_token") or "")
            token_result = review_working_state.load_preview_token_payload(
                session_dir,
                token=token,
            )
            if not token_result.get("ok"):
                self.send_json(
                    {"error": token_result.get("error"), "ok": False},
                    status=HTTPStatus.CONFLICT,
                )
                return
            preview_payload = dict(token_result["preview"])
            correction_payload.setdefault("correction_id", preview_payload.get("correction_id"))
            correction = correction_from_dict(correction_payload)
            before_state = review_working_state.derive_working_state(
                self.server.package_root,
                session_dir,
            )
            before_digest = review_working_state.effective_privacy_region_digest(
                self.server.package_root,
                session_dir,
                case_id=correction.case_id,
            )
            preview_token_result = review_working_state.validate_preview_token(
                session_dir,
                token=token,
                correction=correction,
                current_working_state_revision=str(
                    before_state.get("active_projection_hash")
                    or before_state.get("state_hash")
                    or ""
                ),
            )
            if not preview_token_result.get("ok"):
                self.send_json(
                    {"error": preview_token_result.get("error"), "ok": False},
                    status=HTTPStatus.CONFLICT,
                )
                return
            if preview_token_result.get("already_accepted"):
                state = before_state
                materialization_plan = self.plan_materialization_for_case(
                    session_dir=session_dir,
                    case_id=correction.case_id,
                    actor=actor,
                    operation="manual_correction_saved",
                    reason="Latest reviewed media updated after correction save retry.",
                    accepted_revision=str(
                        state.get("active_projection_hash") or state.get("state_hash") or ""
                    ),
                    before_digest=before_digest,
                    after_state=state,
                    affected_entity_ids=[correction.entity_id] if correction.entity_id else [],
                )
                existing_event = next(
                    (
                        event
                        for event in reversed(review_sessions.load_events(session_dir))
                        if event.get("event_type")
                        in {
                            review_sessions.EVENT_MANUAL_CORRECTION_SAVED,
                            review_sessions.EVENT_CORRECTION_SAVED_LEGACY,
                        }
                        and dict(event.get("after_state") or {}).get("correction_id")
                        == correction.correction_id
                    ),
                    None,
                )
                if existing_event is None:
                    existing_event = {
                        "after_state": correction.to_dict(),
                        "case_id": correction.case_id,
                        "event_id": preview_token_result["preview"].get("accepted_event_id"),
                        "event_type": review_sessions.EVENT_MANUAL_CORRECTION_SAVED,
                        "task_id": correction.task_id,
                    }
                self.send_json(
                    {
                        **materialization_acceptance_fields(
                            materialization_plan=materialization_plan,
                            accepted_revision=str(
                                state.get("active_projection_hash") or state.get("state_hash") or ""
                            ),
                        ),
                        "event": existing_event,
                        "events": [existing_event],
                        "idempotent_retry": True,
                        "materialization_job": materialization_plan["job"],
                        "materialization_plan": materialization_plan["plan"],
                        "materialization_required": materialization_plan["plan"][
                            "materialization_required"
                        ],
                        "ok": True,
                        "preview": None,
                        "verification_task": review_working_state.find_correction_verification_task(
                            state,
                            correction_id=correction.correction_id,
                        )
                        or review_working_state.create_correction_verification_task(
                            correction,
                            accepted_working_state_revision=str(
                                state.get("active_projection_hash") or state.get("state_hash") or ""
                            ),
                            materialization_plan=materialization_plan["plan"],
                            publication_status=str(
                                (materialization_plan["job"] or {}).get("publication_status")
                                or (materialization_plan["job"] or {}).get("status")
                                or "published"
                            ),
                        ),
                        "working_state": state,
                    }
                )
                return
            event = review_sessions.append_correction_event(
                session_dir,
                actor=actor,
                correction=correction,
                before_state={"active_projection_hash": before_state.get("active_projection_hash")},
            )
            state_after_correction = review_working_state.derive_working_state(
                self.server.package_root,
                session_dir,
            )
            materialization_plan = self.plan_materialization_for_case(
                session_dir=session_dir,
                case_id=correction.case_id,
                actor=actor,
                operation="manual_correction_saved",
                reason="Latest reviewed media updated after correction save.",
                accepted_revision=str(
                    state_after_correction.get("active_projection_hash")
                    or state_after_correction.get("state_hash")
                    or ""
                ),
                before_digest=before_digest,
                after_state=state_after_correction,
                affected_entity_ids=[correction.entity_id] if correction.entity_id else [],
            )
            verification_task = review_working_state.create_correction_verification_task(
                correction,
                accepted_working_state_revision=str(
                    state_after_correction.get("active_projection_hash")
                    or state_after_correction.get("state_hash")
                    or ""
                ),
                materialization_plan=materialization_plan["plan"],
                publication_status=str(
                    (materialization_plan["job"] or {}).get("publication_status")
                    or (materialization_plan["job"] or {}).get("status")
                    or "published"
                ),
            )
            verification_event = review_sessions.append_event(
                session_dir,
                event_type="manual_issue_created",
                actor=actor,
                case_id=correction.case_id,
                task_id=verification_task["review_task_id"],
                before_state=None,
                after_state={"task": verification_task},
                note="Correction verification task created.",
                application_hashes={},
            )
            state = review_working_state.derive_working_state(self.server.package_root, session_dir)
            review_working_state.mark_preview_token_accepted(
                session_dir,
                token=token,
                correction=correction,
                event=event,
                materialization_plan=materialization_plan,
            )
            self.send_json(
                {
                    **materialization_acceptance_fields(
                        materialization_plan=materialization_plan,
                        accepted_revision=str(
                            state.get("active_projection_hash") or state.get("state_hash") or ""
                        ),
                    ),
                    "event": event,
                    "events": [event, verification_event],
                    "materialization_job": materialization_plan["job"],
                    "materialization_plan": materialization_plan["plan"],
                    "materialization_required": materialization_plan["plan"][
                        "materialization_required"
                    ],
                    "ok": True,
                    "preview": None,
                    "verification_task": verification_task,
                    "working_state": state,
                }
            )
            return
        if path == "/api/object-policy":
            result = self.apply_object_policy(payload, actor, session_dir)
            status = HTTPStatus.OK if result.get("ok", False) else HTTPStatus.CONFLICT
            self.send_json(result, status=status)
            return
        if path == "/api/bulk-object-policy":
            result = self.apply_bulk_object_policy(payload, actor, session_dir)
            status = HTTPStatus.OK if result.get("ok", False) else HTTPStatus.CONFLICT
            self.send_json(result, status=status)
            return
        if path == "/api/checkpoints":
            checkpoint, event = review_working_state.create_checkpoint(
                self.server.package_root,
                session_dir,
                actor=actor,
                name=str(payload.get("name") or "Checkpoint"),
                note=payload.get("note"),
            )
            self.send_json(
                {
                    "checkpoint": checkpoint,
                    "event": event,
                    "ok": True,
                    "working_state": review_working_state.derive_working_state(
                        self.server.package_root,
                        session_dir,
                    ),
                }
            )
            return
        if path == "/api/checkpoints/restore":
            requested_case_id = str(payload.get("case_id") or "")
            case_ids = (
                [requested_case_id]
                if requested_case_id
                else [
                    case_id
                    for case_id in ("day", "night")
                    if (self.server.package_root / "cases" / f"{case_id}.json").exists()
                ]
            )
            before_digests = {
                case_id: review_working_state.effective_privacy_region_digest(
                    self.server.package_root,
                    session_dir,
                    case_id=case_id,
                )
                for case_id in case_ids
            }
            checkpoint, event, state = review_working_state.restore_checkpoint(
                self.server.package_root,
                session_dir,
                actor=actor,
                checkpoint_uuid=str(payload.get("checkpoint_uuid") or ""),
            )
            materialization_plans = [
                self.plan_materialization_for_case(
                    session_dir=session_dir,
                    case_id=case_id,
                    actor=actor,
                    operation="restore_checkpoint",
                    reason="Latest reviewed media restored from checkpoint.",
                    accepted_revision=str(
                        state.get("active_projection_hash") or state.get("state_hash") or ""
                    ),
                    before_digest=before_digests[case_id],
                    after_state=state,
                )
                for case_id in case_ids
            ]
            state = review_working_state.derive_working_state(self.server.package_root, session_dir)
            self.send_json(
                {
                    "checkpoint": checkpoint,
                    "event": event,
                    "materialization_jobs": [
                        plan["job"] for plan in materialization_plans if plan["job"]
                    ],
                    "materialization_plans": [plan["plan"] for plan in materialization_plans],
                    "ok": True,
                    "working_state": state,
                }
            )
            return
        if path == "/api/checkpoints/delete":
            checkpoint, event, state = review_working_state.delete_checkpoint(
                self.server.package_root,
                session_dir,
                actor=actor,
                checkpoint_uuid=str(payload.get("checkpoint_uuid") or ""),
            )
            self.send_json(
                {
                    "checkpoint": checkpoint,
                    "event": event,
                    "ok": True,
                    "working_state": state,
                }
            )
            return
        if path == "/api/checkpoints/restart-review":
            requested_case_id = str(payload.get("case_id") or "")
            event, state = review_working_state.restart_review_workflow(
                self.server.package_root,
                session_dir,
                actor=actor,
                case_id=requested_case_id,
            )
            self.send_json(
                {
                    "event": event,
                    "materialization_jobs": [],
                    "ok": True,
                    "working_state": state,
                }
            )
            return
        if path == "/api/checkpoints/reset":
            requested_case_id = str(payload.get("case_id") or "")
            case_ids = (
                [requested_case_id]
                if requested_case_id
                else [
                    case_id
                    for case_id in ("day", "night")
                    if (self.server.package_root / "cases" / f"{case_id}.json").exists()
                ]
            )
            before_digests = {
                case_id: review_working_state.effective_privacy_region_digest(
                    self.server.package_root,
                    session_dir,
                    case_id=case_id,
                )
                for case_id in case_ids
            }
            event, state = review_working_state.reset_to_automatic_baseline(
                self.server.package_root,
                session_dir,
                actor=actor,
                case_id=requested_case_id or None,
            )
            materialization_plans = [
                self.plan_materialization_for_case(
                    session_dir=session_dir,
                    case_id=case_id,
                    actor=actor,
                    operation="case_edits_reset_to_automatic_baseline",
                    reason="Latest reviewed media reset to Automatic baseline after destructive edit reset.",
                    accepted_revision=str(
                        state.get("active_projection_hash") or state.get("state_hash") or ""
                    ),
                    before_digest=before_digests[case_id],
                    after_state=state,
                )
                for case_id in case_ids
            ]
            self.send_json(
                {
                    "event": event,
                    "materialization_jobs": [
                        plan["job"] for plan in materialization_plans if plan["job"]
                    ],
                    "materialization_plans": [plan["plan"] for plan in materialization_plans],
                    "ok": True,
                    "working_state": review_working_state.derive_working_state(
                        self.server.package_root,
                        session_dir,
                    ),
                }
            )
            return
        if path == "/api/materialization-jobs/retry":
            retry_job = self.server.materialization_jobs.retry(
                job_id=str(payload.get("job_id") or ""),
                actor=actor,
            )
            self.send_json({"job": retry_job, "ok": True})
            return
        if path == "/api/entity-events":
            entity_event = dict(payload.get("event") or {})
            event = review_sessions.append_event(
                session_dir,
                event_type=str(entity_event.get("event_type") or "entity_event"),
                actor=actor,
                case_id=str(entity_event.get("case_id") or payload.get("case_id") or ""),
                task_id=None,
                before_state=(entity_event.get("payload") or {}).get("before"),
                after_state=(entity_event.get("payload") or {}).get("after")
                or dict(entity_event.get("payload") or {}),
                note=f"Entity change for {entity_event.get('entity_id')}",
                application_hashes={},
            )
            self.send_json({"event": event, "ok": True})
            return
        if path in {"/api/entities/merge", "/api/entities/split"}:
            event_type = "entities_merged" if path.endswith("merge") else "entity_split"
            if event_type == "entities_merged":
                if payload.get("entity_ids"):
                    result = self.merge_selected_entities(payload, actor, session_dir)
                    status = HTTPStatus.OK if result.get("ok", False) else HTTPStatus.CONFLICT
                    self.send_json(result, status=status)
                    return
                merge_error = validate_entity_merge_payload(payload)
                if merge_error:
                    self.send_json(
                        {"error": merge_error, "ok": False},
                        status=HTTPStatus.CONFLICT,
                    )
                    return
            event = review_sessions.append_event(
                session_dir,
                event_type=event_type,
                actor=actor,
                case_id=str(payload.get("case_id") or ""),
                task_id=None,
                before_state=payload.get("before_state"),
                after_state=payload.get("after_state") or dict(payload),
                note=str(payload.get("reason") or ""),
                application_hashes={},
            )
            self.send_json({"event": event, "ok": True})
            return
        if path == "/api/finalize":
            result = review_finalization.finalize_reviewed_case(
                package_root=self.server.package_root,
                case_id=str(payload["case_id"]),
                session_dir=session_dir,
                actor=actor,
                overwrite=bool(payload.get("overwrite")),
            )
            attempt_event = None
            if result.get("finalized", False):
                events = review_sessions.load_events(session_dir)
                attempt_event = events[-1] if events else None
            else:
                attempt_event = review_sessions.append_event(
                    session_dir,
                    event_type="finalization_attempt_blocked",
                    actor=actor,
                    case_id=str(payload["case_id"]),
                    task_id=None,
                    before_state=None,
                    after_state=result,
                    note="Finalization blocked by backend gates.",
                    application_hashes={},
                )
            self.send_json(
                {
                    "event": attempt_event,
                    "ok": result.get("finalized", False),
                    "result": result,
                }
            )
            return
        self.send_json({"error": "unknown endpoint", "ok": False}, status=HTTPStatus.NOT_FOUND)

    def session_dir_from_query(self, query: dict[str, list[str]]) -> Path | None:
        session_id = str((query.get("session_id") or [""])[0])
        if not session_id:
            self.send_json(
                {"error": "session_id is required", "ok": False},
                status=HTTPStatus.BAD_REQUEST,
            )
            return None
        session_dir = review_sessions.session_directory(self.server.package_root, session_id)
        if not session_dir.exists():
            self.send_json(
                {"error": "review session not found", "ok": False},
                status=HTTPStatus.NOT_FOUND,
            )
            return None
        return session_dir

    def plan_materialization_for_case(
        self,
        *,
        session_dir: Path,
        case_id: str,
        actor: dict[str, Any],
        operation: str,
        reason: str,
        accepted_revision: str | None,
        before_digest: dict[str, Any],
        after_state: dict[str, Any],
        affected_entity_ids: list[str] | None = None,
        requested_policy: str | None = None,
    ) -> dict[str, Any]:
        after_digest = review_working_state.effective_privacy_region_digest(
            self.server.package_root,
            session_dir,
            case_id=case_id,
        )
        change_summary = review_working_state.effective_region_change_summary(
            self.server.package_root,
            case_id=case_id,
            before_digest=before_digest,
            after_digest=after_digest,
        )
        after_hash = str(after_digest.get("effective_regions_hash") or "")
        before_hash = str(before_digest.get("effective_regions_hash") or "")
        published_hash = review_working_state.published_effective_regions_hash(
            self.server.package_root,
            after_state,
            case_id=case_id,
        )
        materialization_required = bool(after_hash != before_hash or after_hash != published_hash)
        manifest = case_manifests.load_case_manifest(
            self.server.package_root / "cases" / f"{case_id}.json"
        )
        planned_full_render_frames = (
            int(manifest.frame_count or 0) if materialization_required else 0
        )
        plan = {
            "affected_source_frame_intervals": change_summary["changed_source_frame_intervals"],
            "after_effective_region_count": int(after_digest.get("region_count") or 0),
            "after_effective_regions_hash": after_hash,
            "before_effective_region_count": int(before_digest.get("region_count") or 0),
            "before_effective_regions_hash": before_hash,
            "case_id": case_id,
            "changed_source_frame_count": change_summary["changed_source_frame_count"],
            "changed_source_frame_intervals": change_summary["changed_source_frame_intervals"],
            "changed_source_frame_percentage": change_summary["changed_source_frame_percentage"],
            "full_case_fallback_required": materialization_required,
            "materialization_required": materialization_required,
            "materialization_strategy": "full_render_fallback"
            if materialization_required
            else "no_op",
            "operation": operation,
            "planned_frames_to_encode": planned_full_render_frames,
            "planned_frames_to_render": planned_full_render_frames,
            "published_effective_regions_hash": published_hash,
            "schema_version": "materialization_planning_result.v1",
        }
        if materialization_required:
            compatible_job = None
            if after_hash == before_hash:
                compatible_job = self.server.materialization_jobs.compatible_active_job(
                    session_dir=session_dir,
                    case_id=case_id,
                    target_effective_regions_hash=after_hash,
                )
            if compatible_job is not None:
                return {
                    "job": compatible_job,
                    "plan": plan
                    | {
                        "reused_job_id": compatible_job["job_id"],
                        "reuse_reason": "matching effective-region publication already pending",
                    },
                }
            job = self.server.materialization_jobs.enqueue(
                session_dir=session_dir,
                case_id=case_id,
                actor=actor,
                reason=reason,
                accepted_revision=accepted_revision,
                affected_entity_ids=affected_entity_ids,
                requested_policy=requested_policy,
                target_effective_region_count=int(after_digest.get("region_count") or 0),
                target_effective_regions_hash=after_hash,
                materialization_plan=plan,
            )
            return {"job": job, "plan": plan}
        skip = self.server.materialization_jobs.record_skipped_materialization(
            session_dir=session_dir,
            case_id=case_id,
            accepted_revision=accepted_revision,
            affected_entity_ids=affected_entity_ids,
            before_digest=before_digest,
            after_digest=after_digest,
            operation=operation,
            reason="Effective privacy regions unchanged; media publication skipped.",
        )
        return {"job": None, "plan": plan | {"skip_job_id": skip["job_id"]}}

    def apply_object_policy(
        self,
        payload: dict[str, Any],
        actor: dict[str, Any],
        session_dir: Path,
    ) -> dict[str, Any]:
        case_id = str(payload.get("case_id") or "")
        entity_id = str(payload.get("entity_id") or "")
        policy = str(payload.get("policy") or "")
        if policy not in {
            privacy_entities.POLICY_REDACT,
            privacy_entities.POLICY_KEEP_VISIBLE,
            privacy_entities.POLICY_NOT_SENSITIVE,
            privacy_entities.POLICY_REVIEW_REQUIRED,
        }:
            return {"error": f"unsupported object policy: {policy}", "ok": False}
        entity = find_active_review_entity(
            self.server.package_root,
            session_dir=session_dir,
            case_id=case_id,
            entity_id=entity_id,
        )
        if entity is None:
            return {"error": "review entity not found", "ok": False}
        before_digest = review_working_state.effective_privacy_region_digest(
            self.server.package_root,
            session_dir,
            case_id=case_id,
        )
        before = dict(entity)
        verification_task_id = (
            f"object_policy_verify_{case_id}_{entity_id}_{policy}"
            if policy
            in {privacy_entities.POLICY_KEEP_VISIBLE, privacy_entities.POLICY_NOT_SENSITIVE}
            else None
        )
        after = before | {
            "current_policy": policy,
            "last_reviewer_policy_action": policy,
            "privacy_action": policy,
            "reviewer_override": True,
            "verification_task_id": verification_task_id,
        }
        event = review_sessions.append_event(
            session_dir,
            event_type="entity_policy_changed",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state=before,
            after_state=after,
            note=f"{after.get('label') or entity_id} set to {policy}.",
            application_hashes={},
        )
        verification_event = None
        verification_task = None
        if verification_task_id:
            verification_task = policy_verification_task(case_id, after, verification_task_id)
            verification_event = review_sessions.append_event(
                session_dir,
                event_type="manual_issue_created",
                actor=actor,
                case_id=case_id,
                task_id=verification_task_id,
                before_state=None,
                after_state={"task": verification_task},
                note="Object policy verification task created.",
                application_hashes={},
            )
        events = [event]
        if verification_event is not None:
            events.append(verification_event)
        state = review_working_state.derive_working_state(
            self.server.package_root,
            session_dir,
        )
        materialization_plan = self.plan_materialization_for_case(
            session_dir=session_dir,
            case_id=case_id,
            actor=actor,
            operation="entity_policy_changed",
            reason="Latest reviewed media updated after object policy change.",
            accepted_revision=str(
                state.get("active_projection_hash") or state.get("state_hash") or ""
            ),
            before_digest=before_digest,
            after_state=state,
            affected_entity_ids=[entity_id],
            requested_policy=policy,
        )
        return {
            "entity": after,
            "event": event,
            "events": events,
            "materialization_job": materialization_plan["job"],
            "materialization_plan": materialization_plan["plan"],
            "materialization_required": materialization_plan["plan"]["materialization_required"],
            "ok": True,
            "verification_task": verification_task,
            "working_state": state,
        }

    def apply_bulk_object_policy(
        self,
        payload: dict[str, Any],
        actor: dict[str, Any],
        session_dir: Path,
    ) -> dict[str, Any]:
        case_id = str(payload.get("case_id") or "")
        policy = str(payload.get("policy") or "")
        entity_ids = [str(item) for item in list(payload.get("entity_ids") or [])]
        if not entity_ids:
            return {"error": "at least one entity_id is required", "ok": False}
        if policy not in {
            privacy_entities.POLICY_REDACT,
            privacy_entities.POLICY_KEEP_VISIBLE,
            privacy_entities.POLICY_NOT_SENSITIVE,
            privacy_entities.POLICY_REVIEW_REQUIRED,
        }:
            return {"error": f"unsupported object policy: {policy}", "ok": False}
        before_digest = review_working_state.effective_privacy_region_digest(
            self.server.package_root,
            session_dir,
            case_id=case_id,
        )
        before_entities: list[dict[str, Any]] = []
        after_entities: list[dict[str, Any]] = []
        for entity_id in entity_ids:
            entity = find_active_review_entity(
                self.server.package_root,
                session_dir=session_dir,
                case_id=case_id,
                entity_id=entity_id,
            )
            if entity is None:
                return {"error": f"review entity not found: {entity_id}", "ok": False}
            before_entities.append(entity)
            after_entities.append(
                entity
                | {
                    "current_policy": policy,
                    "last_reviewer_policy_action": policy,
                    "privacy_action": policy,
                    "reviewer_override": True,
                }
            )
        event = review_sessions.append_event(
            session_dir,
            event_type="bulk_entity_policy_changed",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state={"entities": before_entities},
            after_state={"entities": after_entities, "policy": policy},
            note=f"Bulk object policy changed for {len(after_entities)} objects.",
            application_hashes={},
        )
        state = review_working_state.derive_working_state(
            self.server.package_root,
            session_dir,
        )
        materialization_plan = self.plan_materialization_for_case(
            session_dir=session_dir,
            case_id=case_id,
            actor=actor,
            operation="bulk_entity_policy_changed",
            reason="Latest reviewed media updated after bulk object policy change.",
            accepted_revision=str(
                state.get("active_projection_hash") or state.get("state_hash") or ""
            ),
            before_digest=before_digest,
            after_state=state,
            affected_entity_ids=entity_ids,
            requested_policy=policy,
        )
        return {
            "entities": after_entities,
            "event": event,
            "events": [event],
            "materialization_job": materialization_plan["job"],
            "materialization_plan": materialization_plan["plan"],
            "materialization_required": materialization_plan["plan"]["materialization_required"],
            "ok": True,
            "working_state": state,
        }

    def merge_selected_entities(
        self,
        payload: dict[str, Any],
        actor: dict[str, Any],
        session_dir: Path,
    ) -> dict[str, Any]:
        case_id = str(payload.get("case_id") or "")
        entity_ids = [str(item) for item in list(payload.get("entity_ids") or [])]
        if len(entity_ids) < 2:
            return {"error": "Select at least two entities to merge.", "ok": False}
        entities: list[dict[str, Any]] = []
        for entity_id in entity_ids:
            entity = find_active_review_entity(
                self.server.package_root,
                session_dir=session_dir,
                case_id=case_id,
                entity_id=entity_id,
            )
            if entity is None:
                return {"error": f"review entity not found: {entity_id}", "ok": False}
            entities.append(entity)
        categories = {str(entity.get("category") or "") for entity in entities}
        cases = {str(entity.get("case_id") or "") for entity in entities}
        if len(cases) != 1 or case_id not in cases:
            return {"error": "Merge rejected: objects belong to different cases.", "ok": False}
        if len(categories) != 1:
            return {"error": "Merge rejected: objects belong to different categories.", "ok": False}
        before_digest = review_working_state.effective_privacy_region_digest(
            self.server.package_root,
            session_dir,
            case_id=case_id,
        )
        state_before = review_working_state.derive_working_state(
            self.server.package_root,
            session_dir,
        )
        canonical = canonical_merged_entity(entities, session_dir=session_dir)
        retired_ids = [str(entity.get("entity_id") or "") for entity in entities]
        overlap_warning = any_entities_overlap(entities)
        after_state = {
            "canonical_merged_entity_uuid": canonical["entity_id"],
            "materialization_required": False,
            "overlapping_frame_warning": overlap_warning,
            "primary": canonical,
            "resulting_policy": canonical.get("current_policy"),
            "retired_entity_ids": retired_ids,
            "retired_entity_uuids": retired_ids,
            "source_entity_ids": retired_ids,
            "working_state_revision": state_before.get("state_hash"),
        }
        event = review_sessions.append_event(
            session_dir,
            event_type="entities_merged",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state={"entities": entities},
            after_state=after_state,
            note=str(payload.get("reason") or "Reviewer confirmed object merge."),
            application_hashes={
                "source_evidence_hash": stable_hash(
                    {
                        "source_entity_ids": retired_ids,
                        "tracks": [
                            sorted(review_working_state.entity_track_ids(entity))
                            for entity in entities
                        ],
                    }
                )
            },
        )
        append_merged_entity_audits(
            self.server.package_root,
            case_id=case_id,
            canonical=canonical,
            retired_entities=entities,
        )
        state = review_working_state.derive_working_state(
            self.server.package_root,
            session_dir,
        )
        materialization_plan = self.plan_materialization_for_case(
            session_dir=session_dir,
            case_id=case_id,
            actor=actor,
            operation="entities_merged",
            reason="Latest reviewed media updated after object merge.",
            accepted_revision=str(
                state.get("active_projection_hash") or state.get("state_hash") or ""
            ),
            before_digest=before_digest,
            after_state=state,
            affected_entity_ids=[canonical["entity_id"], *retired_ids],
        )
        return {
            "canonical_merged_entity_uuid": canonical["entity_id"],
            "event": event,
            "materialization_job": materialization_plan["job"],
            "materialization_plan": materialization_plan["plan"],
            "materialization_required": materialization_plan["plan"]["materialization_required"],
            "merged_entity": canonical,
            "ok": True,
            "retired_entity_uuids": retired_ids,
            "resulting_policy": canonical.get("current_policy"),
            "source_evidence_preserved": True,
            "working_state": state,
            "working_state_revision": state.get("state_hash"),
        }

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        max_length = 4 * 1024 * 1024
        if length > max_length:
            raise ValueError(f"request body exceeds {max_length} bytes")
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8"))

    def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def serve_range_request(self) -> None:
        file_path = Path(self.translate_path(self.path))
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        range_header = self.headers.get("Range") or ""
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match or "," in range_header:
            self.send_range_error(file_path.stat().st_size)
            return
        size = file_path.stat().st_size
        start_text, end_text = match.groups()
        if not start_text and not end_text:
            self.send_range_error(size)
            return
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                self.send_range_error(size)
                return
            start = max(0, size - suffix_length)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
            if start >= size:
                self.send_range_error(size)
                return
            end = min(end, size - 1)
        if start > end:
            self.send_range_error(size)
            return
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header(
            "Content-Type", mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        )
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with file_path.open("rb") as file:
            file.seek(start)
            try:
                self.wfile.write(file.read(end - start + 1))
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

    def send_range_error(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def should_serve_spa(self) -> bool:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return False
        if parsed.path in {"/day", "/night", "/final", "/audit"}:
            return (self.server.package_root / "app" / "index.html").exists()
        if not parsed.path.startswith("/app/"):
            return False
        requested = Path(self.translate_path(self.path))
        if requested.exists():
            return False
        suffix = Path(parsed.path).suffix
        return suffix == "" and (self.server.package_root / "app" / "index.html").exists()


def serve_workspace(
    package_root: Path,
    *,
    port: int = 0,
    open_browser: bool = False,
) -> FinalReviewWorkspaceServer:
    mimetypes.add_type("video/mp4", ".mp4")
    server = FinalReviewWorkspaceServer(("127.0.0.1", port), package_root)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_address[1]}/index.html")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def legacy_api_path(path: str) -> str:
    if path == "/api/v1":
        return "/api"
    if path.startswith("/api/v1/"):
        return "/api/" + path.removeprefix("/api/v1/")
    return path


def find_review_entity(
    package_root: Path, *, case_id: str, entity_id: str
) -> dict[str, Any] | None:
    index_path = package_root / "review_entities.json"
    if not index_path.exists():
        return None
    index = read_json_file(index_path)
    for entity in list(dict(index.get("cases") or {}).get(case_id) or []):
        if str(entity.get("entity_id") or "") == entity_id:
            return dict(entity)
    return None


def find_active_review_entity(
    package_root: Path,
    *,
    session_dir: Path,
    case_id: str,
    entity_id: str,
) -> dict[str, Any] | None:
    state = review_working_state.derive_working_state(package_root, session_dir)
    baseline = read_json_file(package_root / "review_entities.json")
    entities = list(dict(baseline.get("cases") or {}).get(case_id) or [])
    for entity in review_working_state.effective_case_entities(entities, state):
        if str(entity.get("entity_id") or "") == entity_id:
            return dict(entity)
    return None


def canonical_merged_entity(
    entities: list[dict[str, Any]],
    *,
    session_dir: Path,
) -> dict[str, Any]:
    ordered = sorted(entities, key=lambda entity: str(entity.get("entity_id") or ""))
    source_ids = [str(entity.get("entity_id") or "") for entity in ordered]
    category = str(ordered[0].get("category") or "")
    case_id = str(ordered[0].get("case_id") or "")
    current_head = review_working_state.current_event_chain_head(session_dir)
    canonical_id = (
        f"entity_merged_{stable_hash({'head': current_head, 'source_entity_ids': source_ids})[:24]}"
    )
    policies = {
        review_working_state.inherited_policy(entity)
        for entity in ordered
        if review_working_state.inherited_policy(entity)
    }
    resulting_policy = (
        next(iter(policies)) if len(policies) == 1 else privacy_entities.POLICY_REVIEW_REQUIRED
    )
    policy_reason = (
        "all_selected_entities_shared_policy"
        if len(policies) == 1
        else "conflicting_selected_policies"
    )
    representative = choose_merge_representative(ordered)
    first_frames = [
        int(entity["first_visible_source_frame"])
        for entity in ordered
        if isinstance(entity.get("first_visible_source_frame"), int)
    ]
    last_frames = [
        int(entity["last_visible_source_frame"])
        for entity in ordered
        if isinstance(entity.get("last_visible_source_frame"), int)
    ]
    evidence_summary = merge_evidence_summary(ordered)
    label_prefix = "Person" if category == CATEGORY_FACE else "Plate"
    group_number = merged_group_number_for_selection(
        session_dir=session_dir,
        case_id=case_id,
        category=category,
        entities=ordered,
    )
    group_label = f"Merged {label_prefix} {group_number:02d}"
    lineage = []
    for entity in ordered:
        lineage.extend(list(entity.get("merge_split_lineage") or []))
    lineage.append(
        {
            "operation": "merge",
            "result_entity_id": canonical_id,
            "source_entity_ids": source_ids,
        }
    )
    merged = dict(representative)
    merged.update(
        {
            "aliases": source_ids,
            "attached_fragment_track_ids": unique_sorted(
                item
                for entity in ordered
                for item in list(entity.get("attached_fragment_track_ids") or [])
            ),
            "automatic_source_track_ids": unique_sorted(
                item
                for entity in ordered
                for item in list(entity.get("automatic_source_track_ids") or [])
            ),
            "case_id": case_id,
            "category": category,
            "current_policy": resulting_policy,
            "current_status": "active",
            "default_privacy_policy": resulting_policy,
            "display_name": group_label,
            "entity_id": canonical_id,
            "entity_uuid": canonical_id,
            "evidence_summary": evidence_summary,
            "first_visible_source_frame": min(first_frames) if first_frames else None,
            "label": group_label,
            "last_visible_source_frame": max(last_frames) if last_frames else None,
            "member_detector_tracker_ids": unique_sorted(
                item
                for entity in ordered
                for item in list(entity.get("member_detector_tracker_ids") or [])
            ),
            "merge_split_lineage": lineage,
            "privacy_action": resulting_policy,
            "merged_display_sequence": group_number,
            "merge_policy_initialization": {
                "input_policies": sorted(policies),
                "reason": policy_reason,
                "resulting_policy": resulting_policy,
                "schema_version": "merged_entity_policy_initialization.v1",
            },
            "review_entity_state": (
                "confirmed"
                if all(entity.get("review_entity_state") == "confirmed" for entity in ordered)
                else "needs_review"
            ),
            "review_state": "needs_review"
            if resulting_policy == privacy_entities.POLICY_REVIEW_REQUIRED
            else "reviewed",
            "source_entity_ids": source_ids,
        }
    )
    bbox = representative_bbox(merged)
    if bbox:
        merged["representative_bbox"] = bbox
    return merged


def merged_group_number_for_selection(
    *,
    session_dir: Path,
    case_id: str,
    category: str,
    entities: list[dict[str, Any]],
) -> int:
    existing_numbers = [
        int(entity.get("merged_display_sequence"))
        for entity in entities
        if isinstance(entity.get("merged_display_sequence"), int)
    ]
    if existing_numbers:
        return min(existing_numbers)
    max_seen = 0
    for event in review_sessions.load_events(session_dir):
        if event.get("event_type") != "entities_merged":
            continue
        if str(event.get("case_id") or "") != case_id:
            continue
        primary = dict(dict(event.get("after_state") or {}).get("primary") or {})
        if str(primary.get("category") or "") != category:
            continue
        value = primary.get("merged_display_sequence")
        if isinstance(value, int):
            max_seen = max(max_seen, value)
    return max_seen + 1


def choose_merge_representative(entities: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        entities,
        key=lambda entity: (
            1 if representative_bbox(entity) else 0,
            float(dict(entity.get("representative_quality") or {}).get("relative_area") or 0),
            int(dict(entity.get("evidence_summary") or {}).get("review_observation_count") or 0),
        ),
    )


def representative_bbox(entity: dict[str, Any]) -> list[int] | None:
    for key in ("representative_bbox", "representative_bbox_xyxy"):
        bbox = entity.get(key)
        if isinstance(bbox, list) and len(bbox) == 4:
            return [int(value) for value in bbox]
    quality = dict(entity.get("representative_quality") or {})
    bbox = quality.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [int(value) for value in bbox]
    return None


def merge_evidence_summary(entities: list[dict[str, Any]]) -> dict[str, Any]:
    total_regions = 0
    total_observations = 0
    total_tracklets = 0
    provenance_counts: Counter[str] = Counter()
    for entity in entities:
        summary = dict(entity.get("evidence_summary") or {})
        total_regions += int(summary.get("region_count") or 0)
        total_observations += int(summary.get("review_observation_count") or 0)
        total_tracklets += int(
            summary.get("review_tracklet_count") or summary.get("tracklet_count") or 0
        )
        provenance_counts.update(dict(summary.get("provenance_counts") or {}))
    return {
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "region_count": total_regions,
        "review_entity_state": "confirmed"
        if all(entity.get("review_entity_state") == "confirmed" for entity in entities)
        else "needs_review",
        "review_observation_count": total_observations,
        "review_tracklet_count": total_tracklets,
        "tracklet_count": total_tracklets,
    }


def any_entities_overlap(entities: list[dict[str, Any]]) -> bool:
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if entity_spans_overlap(left, right):
                return True
    return False


def entity_spans_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    l1 = left.get("first_visible_source_frame")
    l2 = left.get("last_visible_source_frame")
    r1 = right.get("first_visible_source_frame")
    r2 = right.get("last_visible_source_frame")
    if not all(isinstance(value, int) for value in (l1, l2, r1, r2)):
        return False
    return max(int(l1), int(r1)) <= min(int(l2), int(r2))


def unique_sorted(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value is not None})


def policy_verification_task(case_id: str, entity: dict[str, Any], task_id: str) -> dict[str, Any]:
    frame = entity.get("first_visible_source_frame")
    source_frame = int(frame) if isinstance(frame, int) else 0
    policy = str(entity.get("current_policy") or entity.get("privacy_action") or "")
    label = "Keep visible" if policy == privacy_entities.POLICY_KEEP_VISIBLE else "Not sensitive"
    return {
        "blocking": True,
        "case_id": case_id,
        "category": str(entity.get("category") or ""),
        "correction_required": False,
        "decision_state": "pending",
        "issue_type": "entity_policy_verification",
        "normal_task_class": "entity_policy_verification",
        "raw_signal_ids": [],
        "review_task_id": task_id,
        "schema_version": "case_reviewer_task.v1",
        "severity": "blocker",
        "source_frame_end": source_frame,
        "source_frame_start": source_frame,
        "technical_only": False,
        "title": f"Verify {label} for {entity.get('label') or 'review object'}",
        "what_happened": "Reviewer changed this object policy.",
        "what_to_inspect": "Confirm this object policy is correct before finalization.",
    }


def validate_entity_merge_payload(payload: dict[str, Any]) -> str | None:
    before = dict(payload.get("before_state") or {})
    source = dict(before.get("source") or before.get("primary") or {})
    target = dict(before.get("target") or before.get("secondary") or {})
    if not source or not target:
        return "Merge rejected: source and target entity states are required."
    if source.get("case_id") != target.get("case_id"):
        return "Merge rejected: objects belong to different cases."
    if source.get("category") != target.get("category"):
        return "Merge rejected: objects belong to different categories."
    after = dict(payload.get("after_state") or {})
    retired = set(str(item) for item in after.get("retired_entity_ids") or [])
    target_id = str(target.get("entity_id") or "")
    if target_id and target_id not in retired:
        return "Merge rejected: retired target entity is missing from after_state."
    return None


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_evaluation_plan_clips(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    clips: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in text.splitlines():
        clip_match = re.match(r"  ([a-z_]+):\s*$", line)
        if clip_match:
            current = clip_match.group(1)
            clips[current] = {}
            continue
        if current is None:
            continue
        key_match = re.match(r"    ([a-z_]+):\s*(.+?)\s*$", line)
        if not key_match:
            continue
        key, raw_value = key_match.groups()
        clips[current][key] = parse_scalar(raw_value)
    return clips


def parse_scalar(value: str) -> Any:
    value = value.strip().strip('"')
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def case_id_for_clip(clip_id: str) -> str:
    if clip_id.startswith("day_"):
        return "day"
    if clip_id == "night":
        return "night"
    return clip_id


def normalize_audit_category(category: str) -> str:
    if category == "plate":
        return CATEGORY_PLATE
    return category


def root_cause_for_signal(item: dict[str, Any]) -> str:
    issue_type = str(item.get("issue_type") or "")
    if "false" in issue_type or "wrong" in issue_type:
        return "possible_false_redaction"
    if issue_type in {"visible_plate_leak", "visible_face_leak"}:
        return "possible_privacy_gap"
    if item.get("severity") == "low":
        return "spot_check"
    return "possible_privacy_gap"


def severity_rank(severity: str) -> int:
    return {
        "low": 0,
        "medium": 1,
        "high": 2,
        "blocker": 3,
    }.get(severity, 1)


def task_sort_rank(task: dict[str, Any]) -> int:
    if task.get("blocking"):
        return 0
    if task.get("issue_type") == "possible_false_redaction":
        return 1
    if task.get("severity") == "low":
        return 4
    return 2


def audit_schema() -> dict[str, Any]:
    return {
        "actions": sorted(DECISION_LABELS),
        "issue_types": [
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
        ],
        "schema_version": "case_review_audit_schema.v1",
    }


def audit_clip_frame_offsets() -> dict[str, int]:
    clips = load_evaluation_plan_clips(Path("data/evaluation_plan.yaml"))
    fps = 30000 / 1001
    offsets: dict[str, int] = {}
    for clip_id in ("day_face", "day_plate"):
        if clip_id in clips:
            offsets[clip_id] = round(float(clips[clip_id]["parent_start_seconds"]) * fps)
    return offsets


def validate_workspace_media_urls(package_root: Path) -> dict[str, Any]:
    html_audit = case_manifests.validate_relative_media_urls(
        html_path=package_root / "index.html",
        package_root=package_root,
    )
    items = list(html_audit["items"])
    errors = list(html_audit["errors"])
    for case_path in sorted((package_root / "cases").glob("*.json")):
        manifest = case_manifests.load_case_manifest(case_path)
        for role in manifest.video_roles:
            resolved = (package_root / role.path).resolve()
            item = {
                "attribute": f"case:{manifest.case_id}:{role.role}",
                "exists": resolved.exists(),
                "inside_package_root": True,
                "relative_url": role.path,
                "resolved_package_path": package_relative_path(
                    path=resolved,
                    package_root=package_root,
                ),
            }
            try:
                resolved.relative_to(package_root.resolve())
            except ValueError:
                item["inside_package_root"] = False
                errors.append(f"{role.path} resolves outside package root")
            if not resolved.exists():
                errors.append(f"{role.path} does not resolve from workspace root")
            items.append(item)
    return {
        "checked_count": len(items),
        "errors": errors,
        "items": items,
        "passed": not errors,
        "schema_version": "media_url_audit.v1",
    }


def backend_parity_audit(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    reference_root = reference_root.resolve()
    candidate_root = candidate_root.resolve()
    compared: list[dict[str, Any]] = []
    errors: list[str] = []
    allowed_entity_registry_deltas = allowed_backend_parity_package_deltas(candidate_root)
    entity_registry_deltas: list[dict[str, Any]] = []
    patterns = (
        "cases/*.json",
        "entities/*.json",
        "regions/*.jsonl",
        "media/*/*.mp4",
        "configuration_freeze.json",
        "reviewer_tasks.json",
        "review_entities.json",
        "privacy_entities.json",
        "raw_audit_signals.json",
        "timeline_marker_audit.json",
        "version_alignment_audit.json",
        "model_validation.json",
    )
    relative_paths = sorted(
        {
            path.relative_to(reference_root).as_posix()
            for pattern in patterns
            for path in reference_root.glob(pattern)
            if path.is_file()
        }
    )
    for relative_path in relative_paths:
        reference_path = reference_root / relative_path
        candidate_path = candidate_root / relative_path
        reference_hash = media.sha256_file(reference_path)
        candidate_hash = media.sha256_file(candidate_path) if candidate_path.exists() else None
        passed = reference_hash == candidate_hash
        allowed_delta = bool(
            candidate_path.exists()
            and not passed
            and relative_path in allowed_entity_registry_deltas
        )
        if not candidate_path.exists():
            errors.append(f"missing candidate artifact: {relative_path}")
        elif not passed and not allowed_delta:
            errors.append(f"artifact hash mismatch: {relative_path}")
        if allowed_delta:
            entity_registry_deltas.append(
                {
                    "path": relative_path,
                    "reason": allowed_entity_registry_deltas[relative_path],
                }
            )
        compared.append(
            {
                "allowed_delta": allowed_delta,
                "candidate_sha256": candidate_hash,
                "path": relative_path,
                "passed": passed or allowed_delta,
                "reference_sha256": reference_hash,
            }
        )
    rendered_media = [item for item in compared if item["path"].startswith("media/")]
    privacy_regions = [item for item in compared if item["path"].startswith("regions/")]
    return {
        "candidate_root": display_package_path(candidate_root),
        "compared_count": len(compared),
        "detector_tracker_evidence_render_hashes_preserved": not errors,
        "entity_registry_coverage_delta_count": len(entity_registry_deltas),
        "entity_registry_coverage_deltas": entity_registry_deltas,
        "error_count": len(errors),
        "errors": errors,
        "items": compared,
        "media_hash_count": len(rendered_media),
        "media_hashes_preserved": all(item["passed"] for item in rendered_media),
        "privacy_region_hash_count": len(privacy_regions),
        "privacy_regions_preserved": all(item["passed"] for item in privacy_regions),
        "reference_root": display_package_path(reference_root),
        "schema_version": "final_review_backend_parity_audit.v1",
        "passed": not errors,
    }


def display_package_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def allowed_backend_parity_package_deltas(candidate_root: Path) -> dict[str, str]:
    allowed: dict[str, str] = {}
    allowed.update(allowed_frame_aligned_role_deltas(candidate_root))
    allowed.update(allowed_backend_parity_entity_registry_deltas(candidate_root))
    allowed.update(allowed_backend_parity_task_chunking_deltas(candidate_root))
    return allowed


def allowed_backend_parity_task_chunking_deltas(candidate_root: Path) -> dict[str, str]:
    audit_path = candidate_root / "review_task_chunking_audit.json"
    if not audit_path.exists():
        return {}
    audit = read_json_file(audit_path)
    if audit.get("passed") is False:
        return {}
    if not any(bool(row.get("split")) for row in list(audit.get("parents") or [])):
        return {}
    return {
        "reviewer_tasks.json": "reviewer-facing long intervals split into non-overlapping child tasks",
        "timeline_marker_audit.json": "timeline markers regenerated from reviewer-facing child tasks",
    }


def allowed_frame_aligned_role_deltas(candidate_root: Path) -> dict[str, str]:
    audit_path = candidate_root / "frame_aligned_review_roles_audit.json"
    if not audit_path.exists():
        return {}
    audit = read_json_file(audit_path)
    if audit.get("passed") is False:
        return {}
    allowed: dict[str, str] = {}
    for case_id, case_audit in dict(audit.get("cases") or {}).items():
        added_roles = set(case_audit.get("added_roles") or [])
        if added_roles <= {"automatic", "latest_reviewed"} and added_roles:
            allowed[f"cases/{case_id}.json"] = (
                "frame-aligned logical review roles added without changing media bytes"
            )
    return allowed


def allowed_backend_parity_entity_registry_deltas(candidate_root: Path) -> dict[str, str]:
    audit_path = candidate_root / "review_entity_coverage_audit.json"
    allowed: dict[str, str] = {}
    if audit_path.exists():
        audit = read_json_file(audit_path)
        if audit.get("passed") is not False:
            cases = audit.get("cases") or {}
            for case_id, case_audit in cases.items():
                added = list(case_audit.get("explicit_keep_tracks_added") or [])
                bbox_enriched = list(
                    case_audit.get("representative_bbox_enriched_entity_ids") or []
                )
                if bbox_enriched:
                    allowed[f"entities/{case_id}.json"] = (
                        "source-backed representative bbox metadata added for review focus overlay"
                    )
                if not added:
                    continue
                if not all(
                    row.get("policy") == privacy_entities.POLICY_KEEP_VISIBLE
                    and row.get("assignment_provenance") == "explicit_case_policy"
                    and row.get("valid_source_evidence") is True
                    and row.get("review_entity_uuid")
                    for row in added
                ):
                    continue
                reason = (
                    "source-backed explicit keep-policy face tracks added to review entity registry"
                )
                allowed[f"entities/{case_id}.json"] = reason
                allowed["privacy_entities.json"] = reason
                allowed["review_entities.json"] = reason
    focus_audit_path = candidate_root / "entity_focus_overlay_audit.json"
    if focus_audit_path.exists():
        focus_audit = read_json_file(focus_audit_path)
        for case_id, case_audit in dict(focus_audit.get("cases") or {}).items():
            if case_audit.get("enriched_entity_ids"):
                reason = "source-backed representative bbox metadata added for focus overlay"
                allowed[f"entities/{case_id}.json"] = reason
                allowed["privacy_entities.json"] = reason
                allowed["review_entities.json"] = reason
    allowed.update(allowed_object_policy_lineage_registry_deltas(candidate_root))
    return allowed


def allowed_object_policy_lineage_registry_deltas(candidate_root: Path) -> dict[str, str]:
    audit_path = candidate_root / "object_policy_lineage_audit.json"
    if not audit_path.exists():
        return {}
    audit = read_json_file(audit_path)
    if audit.get("passed") is False:
        return {}
    allowed: dict[str, str] = {}
    for case_id, case_audit in dict(audit.get("cases") or {}).items():
        if case_audit.get("missing_source_backed_tracks"):
            continue
        exact_keep_rows = [
            row
            for row in list(case_audit.get("policy_lineage") or [])
            if row.get("source_policy") == privacy_entities.POLICY_KEEP_VISIBLE
            and row.get("assignment_provenance") == "explicit_case_policy"
            and row.get("mapping") == "exact"
            and row.get("valid_source_evidence") is True
            and row.get("review_entity")
        ]
        if not exact_keep_rows:
            continue
        reason = "source-backed explicit keep-policy lineage exposed to review entity registry"
        allowed[f"entities/{case_id}.json"] = reason
        allowed["privacy_entities.json"] = reason
        allowed["review_entities.json"] = reason
    return allowed


def package_relative_path(*, path: Path, package_root: Path) -> str:
    try:
        return path.relative_to(package_root.resolve()).as_posix()
    except ValueError:
        return "<outside-package-root>"


def day_lineage_report_text(lineage: dict[str, Any], blockers: list[str]) -> str:
    blocker_text = "\n".join(f"- {item}" for item in blockers) if blockers else "None."
    return (
        "# Day Case Lineage Status\n\n"
        "The existing day face and day plate artifacts are separate clips. Their metadata maps "
        "both clips to the same parent source, `artifacts/day_primary.mp4`, but a combined "
        "current result must be regenerated from that parent interval with the frozen face and "
        "plate pipelines.\n\n"
        f"Lineage proven: `{lineage['lineage_proven']}`\n\n"
        f"## Current Blockers\n\n{blocker_text}\n"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Review Workspace v2.3.4</title>
  <link rel="stylesheet" href="assets/workspace.css?v=2.3.4">
</head>
<body>
  <header class="app-header">
    <div>
      <h1>Final Review Workspace</h1>
      <p>Local review for source-frame accurate redaction decisions and corrections.</p>
    </div>
    <div class="header-actions">
      <button id="startSession" class="primary">Start review</button>
      <button id="changeAttribution">Change reviewer attribution</button>
      <button id="exportSession">Export session</button>
    </div>
  </header>
  <div id="sessionBanner" class="session-banner">No local review session yet. Start review to enable decisions and corrections.</div>
  <nav class="case-tabs" aria-label="Workspace tabs">
    <button data-tab="day">Day</button>
    <button data-tab="night">Night</button>
    <button data-tab="final">Final Results</button>
    <button data-tab="audit">Audit Trail</button>
  </nav>
  <main id="appRoot">
    <section id="case-day" class="case-panel" data-case-id="day"></section>
    <section id="case-night" class="case-panel" data-case-id="night"></section>
    <section id="case-final" class="case-panel"></section>
    <section id="case-audit" class="case-panel"></section>
  </main>
  <section id="attributionModal" class="modal" aria-hidden="true">
    <div class="modal-card">
      <h2>Reviewer attribution</h2>
      <p>This local prototype records a declared reviewer name. It is not enterprise authentication.</p>
      <label>Reviewer name <input id="reviewerName" autocomplete="name"></label>
      <p id="attributionError" class="error-text"></p>
      <div class="button-row">
        <button id="submitAttribution" class="primary">Save attribution</button>
        <button id="cancelAttribution">Cancel</button>
      </div>
    </div>
  </section>
  <section id="entitiesModal" class="modal" aria-hidden="true">
    <div class="modal-card large">
      <div class="case-header">
        <div>
          <h2>Review objects</h2>
          <p class="case-subtitle">Object actions affect privacy policy. Thumbnails jump the player to the representative frame.</p>
        </div>
        <button id="closeEntities">Close</button>
      </div>
      <div id="entitiesModalContent"></div>
    </div>
  </section>
  <section id="mergeModal" class="modal" aria-hidden="true">
    <div class="modal-card large">
      <div class="case-header">
        <div>
          <h2>Merge review objects</h2>
          <p class="case-subtitle">Suggested candidates are review-only groupings, not identity recognition.</p>
        </div>
        <button id="cancelMerge">Cancel</button>
      </div>
      <div id="mergeModalContent"></div>
      <div class="button-row">
        <button id="confirmMerge" class="primary">Merge selected</button>
      </div>
    </div>
  </section>
  <script src="assets/workspace.js?v=2.3.4"></script>
</body>
</html>
"""


def workspace_css() -> str:
    return """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #eef1f5; color: #172033; }
h1, h2, h3, h4 { margin: 0; letter-spacing: 0; }
p { margin: 0; line-height: 1.45; }
button, input, textarea, select { font: inherit; }
button { border: 1px solid #aeb7c4; background: #fff; color: #172033; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
button:hover:not(:disabled) { border-color: #64748b; }
button:disabled { cursor: not-allowed; opacity: .55; }
button.primary { background: #0f766e; border-color: #0f766e; color: #fff; }
button.danger { background: #b91c1c; border-color: #b91c1c; color: #fff; }
button.selected, .segmented button.active { background: #172033; border-color: #172033; color: #fff; }
input, textarea, select { width: 100%; border: 1px solid #b8c2d0; border-radius: 6px; padding: 7px 8px; background: #fff; color: #172033; }
.app-header { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 13px 20px; background: #111827; color: #fff; }
.app-header p { color: #cbd5e1; margin-top: 3px; }
.header-actions, .button-row, .segmented, .inline-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.session-banner { padding: 8px 20px; background: #fff7ed; border-bottom: 1px solid #fed7aa; color: #7c2d12; }
.session-banner.ready { background: #ecfdf5; border-bottom-color: #a7f3d0; color: #065f46; }
.session-banner.error { background: #fef2f2; border-bottom-color: #fecaca; color: #991b1b; }
.case-tabs { display: flex; gap: 6px; padding: 9px 20px; background: #fff; border-bottom: 1px solid #d8dee8; position: sticky; top: 0; z-index: 20; }
.case-tabs button.active { background: #172033; color: #fff; border-color: #172033; }
main { padding: 16px 20px 44px; }
.case-panel { display: none; }
.case-panel.active { display: block; }
.case-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; margin-bottom: 10px; }
.case-subtitle { color: #516070; margin-top: 4px; }
.mode-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.workspace-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(340px, 390px); gap: 14px; align-items: stretch; height: calc(100vh - 176px); min-height: 560px; }
.player-column { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.side-panel { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; min-height: 0; height: 100%; overflow: auto; padding: 12px; }
.version-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
.version-row label { display: flex; align-items: center; gap: 8px; min-width: 260px; }
.player-wrapper { background: #060b14; color: #fff; border-radius: 8px; overflow: hidden; position: relative; box-shadow: 0 12px 30px rgba(15,23,42,.18); }
.player-wrapper:fullscreen { display: flex; flex-direction: column; width: 100vw; height: 100vh; border-radius: 0; }
.player-wrapper:fullscreen .video-stage { flex: 1; min-height: 0; }
.video-stage { position: relative; background: #000; }
.video-stage video { display: block; width: 100%; background: #000; aspect-ratio: 16 / 9; max-height: calc(100vh - 265px); object-fit: contain; cursor: pointer; }
.player-wrapper:fullscreen video { height: 100%; max-height: none; }
.frame-hud { position: absolute; left: 10px; top: 10px; padding: 5px 8px; border-radius: 5px; background: rgba(0,0,0,.72); color: #fff; font-size: 13px; pointer-events: none; z-index: 4; }
.media-error { display: none; position: absolute; left: 10px; right: 10px; bottom: 10px; background: rgba(127, 29, 29, .94); color: #fff; padding: 8px 10px; border-radius: 6px; z-index: 5; font-size: 13px; }
.media-error.visible { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.media-error button { padding: 4px 7px; }
.player-controls { padding: 9px 10px 10px; background: #0b1220; }
.controls-top { display: grid; grid-template-columns: auto minmax(0,1fr) auto auto; gap: 9px; align-items: center; }
.time-readout { color: #cbd5e1; font-size: 13px; min-width: 92px; text-align: center; }
.volume { max-width: 96px; }
.timeline-wrap { position: relative; height: 30px; display: flex; align-items: center; }
.timeline-range { width: 100%; accent-color: #0f766e; }
.timeline-markers { position: absolute; left: 0; right: 0; top: 2px; height: 12px; pointer-events: none; }
.marker { position: absolute; top: 0; height: 8px; min-width: 8px; border-radius: 999px; border: 1px solid #fff; pointer-events: auto; cursor: pointer; transform: translateX(-50%); }
.marker-span { border-radius: 999px; transform: none; min-width: 3px; }
.marker.blocking { background: #dc2626; }
.marker.partial { background: #f97316; }
.marker.info { background: #2563eb; }
.marker.false { background: #7c3aed; }
.marker.resolved { background: #16a34a; }
.marker.active { outline: 2px solid #fff; height: 12px; top: -2px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; color: #d8dee8; font-size: 12px; }
.legend span::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 999px; margin-right: 4px; vertical-align: 1px; }
.legend .red::before { background: #dc2626; } .legend .orange::before { background: #f97316; } .legend .blue::before { background: #2563eb; } .legend .purple::before { background: #7c3aed; } .legend .green::before { background: #16a34a; }
.task-summary { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 10px; }
.stat { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; padding: 8px; }
.stat strong { display: block; font-size: 19px; }
.filter-row { display: flex; gap: 7px; margin: 8px 0 10px; }
.filter-row button.active { background: #172033; color: #fff; }
.entity-filter-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 10px; }
.entity-filter-row button.active { background: #172033; color: #fff; border-color: #172033; }
.task-list { display: grid; gap: 8px; }
.task-card, .entity-card, .edit-card, .audit-card { border: 1px solid #d8dee8; border-radius: 8px; background: #fff; padding: 10px; }
.task-card.selected { border-color: #0f766e; box-shadow: 0 0 0 2px rgba(15,118,110,.16); }
.task-title { font-weight: 700; margin-bottom: 4px; }
.task-meta, .muted { color: #607080; font-size: 13px; }
.task-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
.decision-choice { border-width: 1px; font-weight: 700; min-width: 112px; }
.decision-correct { background: #ecfdf3; border-color: #86efac; color: #166534; }
.decision-needs-correction { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
.decision-unsure { background: #f3f4f6; border-color: #cbd5e1; color: #374151; }
.decision-choice.selected { outline: 2px solid currentColor; outline-offset: 1px; box-shadow: 0 0 0 3px rgba(15,23,42,.08); }
.decision-choice:disabled { opacity: .75; }
.saved-note { color: #047857; font-size: 13px; margin-top: 6px; }
.open-editor { margin-top: 8px; }
.edit-section { display: grid; gap: 10px; }
.edit-card h3 { margin-bottom: 8px; }
.object-list { display: grid; gap: 8px; margin-top: 8px; }
.entity-card { display: grid; grid-template-columns: 76px minmax(0,1fr); gap: 9px; align-items: start; }
.entity-card img { width: 76px; height: 56px; object-fit: cover; border-radius: 5px; background: #e5e7eb; cursor: pointer; }
.entity-card .entity-actions { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.policy-segment { display: inline-flex; flex-wrap: wrap; gap: 4px; border: 1px solid #d8dee8; border-radius: 7px; padding: 3px; background: #f8fafc; }
.policy-segment button { border-radius: 5px; border-color: transparent; padding: 5px 7px; }
.policy-segment button[aria-pressed="true"][data-policy="redact"] { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
.policy-segment button[aria-pressed="true"][data-policy="keep_visible"] { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
.policy-segment button[aria-pressed="true"][data-policy="not_sensitive"] { background: #f3f4f6; border-color: #cbd5e1; color: #374151; }
.merge-candidates { display: grid; gap: 8px; margin: 12px 0; }
.merge-card { display: grid; grid-template-columns: 86px minmax(0,1fr) auto; gap: 10px; align-items: center; border: 1px solid #d8dee8; border-radius: 8px; padding: 8px; background: #fff; }
.merge-card.selected { border-color: #0f766e; box-shadow: 0 0 0 2px rgba(15,118,110,.14); }
.merge-card img { width: 86px; height: 62px; object-fit: cover; border-radius: 5px; background: #e5e7eb; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 7px; font-size: 12px; background: #e2e8f0; color: #334155; }
.badge.open { background: #fef2f2; color: #991b1b; } .badge.done { background: #ecfdf5; color: #047857; } .badge.warn { background: #fff7ed; color: #9a3412; }
.editor-canvas-wrap { position: relative; background: #020617; border-radius: 6px; overflow: hidden; }
.editor-canvas-wrap canvas { width: 100%; min-height: 180px; display: block; }
.keyframe-list { display: grid; gap: 6px; margin-top: 8px; }
.keyframe-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px; }
.technical { margin-top: 12px; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 10px; }
.technical pre { white-space: pre-wrap; overflow: auto; max-height: 220px; font-size: 12px; }
.modal { display: none; position: fixed; inset: 0; background: rgba(15,23,42,.45); z-index: 80; align-items: center; justify-content: center; padding: 20px; }
.modal.open { display: flex; }
.modal-card { width: min(460px, 100%); background: #fff; border-radius: 9px; padding: 18px; box-shadow: 0 24px 80px rgba(15,23,42,.32); }
.modal-card.large { width: min(920px, 94vw); max-height: 86vh; overflow: auto; }
.modal-card label { display: grid; gap: 6px; margin: 12px 0; }
.error-text { color: #b91c1c; }
.audit-list { display: grid; gap: 8px; }
.final-card { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 12px; }
@media (max-width: 980px) { .workspace-layout { grid-template-columns: 1fr; height: auto; min-height: 0; } .side-panel { height: auto; max-height: none; } .video-stage video { max-height: none; } .app-header { align-items: flex-start; flex-direction: column; } }
"""


def workspace_js() -> str:
    return r"""
const APP_VERSION = 'final_review_workspace.v2.3.4';
const ACTIONS = {
  looks: 'looks_correct', leak: 'privacy_leak', partial: 'partial_coverage',
  falseRedaction: 'false_redaction', wrong: 'wrong_target', notSensitive: 'not_sensitive',
  cosmetic: 'cosmetic_only', unsure: 'uncertain_escalate'
};
const DATA = { workspace: null, tasks: null, entities: null, rawSignals: null, alignment: null, cases: {} };
const state = {
  tab: localStorage.getItem(storageKey('active_tab')) || 'day',
  mode: { day: localStorage.getItem(storageKey('mode_day')) || 'review', night: localStorage.getItem(storageKey('mode_night')) || 'review' },
  frame: { day: Number(localStorage.getItem(storageKey('frame_day')) || 0), night: Number(localStorage.getItem(storageKey('frame_night')) || 0) },
  version: { day: localStorage.getItem(storageKey('version_day')) || '', night: localStorage.getItem(storageKey('version_night')) || '' },
  filter: { day: 'Open', night: 'Open' }, selectedTask: { day: null, night: null },
  session: null, attribution: null, decisions: {}, events: [], corrections: [], latestReviewed: {},
  draft: { caseId: null, start: null, end: null, keyframes: [], selectedEntityId: null, selectedTaskId: null },
  savedFlash: {}, entityOverrides: {}, manualTasks: [],
  mergeDraft: { caseId: null, sourceEntityId: null, targetEntityId: null }
};

function storageKey(name) { return `final-review-v2.3.4:${location.pathname}:${name}`; }
async function loadJson(path) { const res = await fetch(path, { cache: 'no-store' }); if (!res.ok) throw new Error(`${path}: ${res.status}`); return res.json(); }
async function postJson(path, payload) { const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const data = await res.json().catch(() => ({})); if (!res.ok || data.ok === false) throw new Error(data.error || `${path}: ${res.status}`); return data; }
function esc(value) { return String(value ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function clamp(n, min, max) { return Math.min(max, Math.max(min, n)); }
function isPrimaryCase(id) { return id === 'day' || id === 'night'; }
function caseData(caseId) { return DATA.cases[caseId]; }
function fpsValue(caseId) { const fps = String(caseData(caseId)?.fps || '30/1').split('/').map(Number); return fps.length === 2 && fps[1] ? fps[0] / fps[1] : Number(fps[0] || 30); }
function caseFrameCount(caseId) { return Number(caseData(caseId)?.frame_count || 1); }
function sourceFrame(caseId, caseFrame = state.frame[caseId]) { return Number(caseData(caseId)?.source_frame_start || 0) + clamp(caseFrame, 0, caseFrameCount(caseId) - 1); }
function frameTime(caseId, caseFrame = state.frame[caseId]) { return clamp(caseFrame, 0, caseFrameCount(caseId) - 1) / fpsValue(caseId); }
function timeText(seconds) { const total = Math.max(0, Math.floor(Number(seconds) || 0)); const h = Math.floor(total / 3600); const m = Math.floor((total % 3600) / 60); const s = total % 60; return [h,m,s].map(v => String(v).padStart(2, '0')).join(':'); }
function hudText(caseId) { const frame = clamp(state.frame[caseId] || 0, 0, caseFrameCount(caseId) - 1); return `Source frame ${sourceFrame(caseId, frame)} · Case frame ${frame} · ${timeText(frameTime(caseId, frame))}`; }
function currentActor() { return state.attribution || { display_name: null, application_version: APP_VERSION }; }
function visibleCaseId() { return state.tab === 'night' ? 'night' : 'day'; }
function activeVideo(caseId = visibleCaseId()) { return document.querySelector(`#case-${caseId}.active video[data-case="${caseId}"]`); }
function selectedTask(caseId) { return allTasks(caseId).find(t => t.review_task_id === state.selectedTask[caseId]) || null; }
function closestElement(target, selector) { const element = target?.nodeType === Node.ELEMENT_NODE ? target : target?.parentElement; return element ? element.closest(selector) : null; }

async function init() {
  DATA.workspace = await loadJson('workspace_manifest.json');
  DATA.tasks = await loadJson('reviewer_tasks.json');
  DATA.entities = await loadJson('review_entities.json');
  DATA.rawSignals = await loadJson('raw_audit_signals.json').catch(() => ({raw_signals: []}));
  DATA.alignment = await loadJson('version_alignment_audit.json').catch(() => null);
  for (const [caseId, path] of Object.entries(DATA.workspace.cases || {})) DATA.cases[caseId] = await loadJson(path);
  await restoreSession();
  renderApp();
  installGlobalHandlers();
}
function renderApp() {
  renderTabs(); renderSessionBanner();
  renderCase('day'); renderCase('night'); renderFinal(); renderAudit();
  switchTab(state.tab);
}
function renderTabs() { document.querySelectorAll('.case-tabs button').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === state.tab)); }
function switchTab(tab) { state.tab = tab; localStorage.setItem(storageKey('active_tab'), tab); document.querySelectorAll('.case-panel').forEach(p => p.classList.remove('active')); document.getElementById(`case-${tab}`)?.classList.add('active'); renderTabs(); pauseHiddenVideos(); setTimeout(() => syncPlayerToState(visibleCaseId()), 0); }
function pauseHiddenVideos() { document.querySelectorAll('video').forEach(v => { if (!v.closest('.case-panel.active')) v.pause(); }); }
function setMode(caseId, mode) { state.mode[caseId] = mode; localStorage.setItem(storageKey(`mode_${caseId}`), mode); renderCase(caseId); setTimeout(() => syncPlayerToState(caseId), 0); }

function renderCase(caseId) {
  const panel = document.getElementById(`case-${caseId}`); if (!panel || !DATA.cases[caseId]) return;
  const manifest = DATA.cases[caseId]; const mode = state.mode[caseId] || 'review';
  if (!state.selectedTask[caseId]) state.selectedTask[caseId] = firstOpenTask(caseId)?.review_task_id || allTasks(caseId)[0]?.review_task_id || null;
  panel.innerHTML = `
    <div class="case-header"><div><h2>${esc(manifest.display_name || caseId)}</h2><p class="case-subtitle">${esc(caseSubtitle(caseId))}</p></div><div class="header-actions"><div class="segmented"><button class="${mode === 'review' ? 'active' : ''}" data-mode="review">Review</button><button class="${mode === 'edit' ? 'active' : ''}" data-mode="edit">Edit</button></div><button data-action="open-entities">Entities</button></div></div>
    <div class="workspace-layout">
      <div class="player-column">${playerHtml(caseId)}<details class="technical"><summary>Technical details</summary><pre>${esc(JSON.stringify(technicalSummary(caseId), null, 2))}</pre></details></div>
      <aside class="side-panel">${mode === 'edit' ? editPanelHtml(caseId) : reviewPanelHtml(caseId)}</aside>
    </div>`;
  panel.querySelectorAll('[data-mode]').forEach(btn => btn.addEventListener('click', () => setMode(caseId, btn.dataset.mode)));
  panel.querySelector('[data-action="open-entities"]')?.addEventListener('click', () => openEntities(caseId));
  bindPlayer(caseId); bindTimeline(caseId); bindReviewPanel(caseId); bindEditPanel(caseId); bindEntityPanel(caseId);
}
function caseSubtitle(caseId) { const m = caseData(caseId); return `${m.frame_count} frames · Source ${m.source_frame_start}-${m.source_frame_end}`; }
function technicalSummary(caseId) { return { case_manifest: DATA.cases[caseId], alignment: DATA.alignment?.cases?.[caseId] || null }; }
function playerHtml(caseId) {
  const versions = availableVersions(caseId); const selected = selectedVersion(caseId);
  const options = versions.map(v => `<option value="${esc(v.key)}" ${v.key === selected.key ? 'selected' : ''}>${esc(v.label)}</option>`).join('');
  return `<div class="version-row"><label>Version <select data-action="version">${options}</select></label><span class="badge ${selected.key === 'latest_reviewed' ? 'warn' : ''}">${esc(selected.note || 'Frame-aligned')}</span></div>
    <section id="player-${caseId}" class="player-wrapper" data-case="${caseId}">
      <div class="video-stage"><video data-case="${caseId}" preload="metadata" playsinline disablePictureInPicture controlslist="nodownload noremoteplayback"></video><div class="frame-hud" data-role="hud">${esc(hudText(caseId))}</div><div class="media-error" data-role="media-error"><span></span><button data-action="retry-media">Retry</button></div></div>
      <div class="player-controls"><div class="controls-top"><button data-action="toggle-play">Play</button><div class="timeline-wrap"><input class="timeline-range" type="range" min="0" max="${caseFrameCount(caseId) - 1}" value="${state.frame[caseId] || 0}" step="1" data-action="scrub"><div class="timeline-markers" data-role="markers"></div></div><span class="time-readout" data-role="time">${esc(timeText(frameTime(caseId)))} / ${esc(timeText(caseData(caseId).duration_seconds || frameTime(caseId, caseFrameCount(caseId)-1)))}</span><input class="volume" type="range" min="0" max="1" step="0.05" value="1" data-action="volume"><button data-action="fullscreen">Fullscreen</button></div><div class="legend"><span class="red">blocking</span><span class="orange">uncertain</span><span class="blue">spot check</span><span class="purple">false redaction</span><span class="green">resolved</span></div></div>
    </section>`;
}
function availableVersions(caseId) {
  const roles = caseData(caseId).video_roles || [];
  const original = roles.find(r => r.role === 'original'); const automatic = roles.find(r => r.role === 'automatic') || roles.find(r => r.role === 'current_result'); const current = roles.find(r => r.role === 'current_result'); const latest = roles.find(r => r.role === 'latest_reviewed'); const final = roles.find(r => r.role === 'final_reviewed');
  const versions = [];
  if (original) versions.push({ key: 'original', label: 'Original', path: original.path, role: original.role });
  if (automatic) versions.push({ key: 'automatic', label: 'Automatic', path: automatic.path, role: automatic.role });
  if (state.latestReviewed[caseId]) versions.push({ key: 'latest_reviewed', label: 'Latest reviewed', path: state.latestReviewed[caseId].path, role: 'latest_reviewed', note: 'Correction preview' });
  else if (latest) versions.push({ key: 'latest_reviewed', label: 'Latest reviewed', path: latest.path, role: latest.role, note: 'Current result' });
  else if (current) versions.push({ key: 'latest_reviewed', label: 'Latest reviewed', path: current.path, role: current.role, note: 'Current result' });
  if (final) versions.push({ key: 'final_reviewed', label: 'Final reviewed', path: final.path, role: final.role });
  return versions;
}
function selectedVersion(caseId) { const versions = availableVersions(caseId); const stored = state.version[caseId]; return versions.find(v => v.key === stored) || versions.find(v => v.key === 'latest_reviewed') || versions.find(v => v.key === 'automatic') || versions[0]; }
function bindPlayer(caseId) {
  const panel = document.getElementById(`case-${caseId}`); const video = panel.querySelector('video'); const version = selectedVersion(caseId); const wrapper = panel.querySelector(`#player-${caseId}`);
  video.disablePictureInPicture = true; video.playsInline = true;
  setPlayerSource(video, version.path);
  video.addEventListener('loadedmetadata', () => { hideMediaError(caseId); seekToFrame(caseId, state.frame[caseId] || 0, { keepPaused: true }); });
  video.addEventListener('error', () => showMediaError(caseId, version.path, video.error ? video.error.code : 'unknown'));
  video.addEventListener('timeupdate', () => { if (!video.paused) updateFrameFromTime(caseId, video.currentTime); });
  video.addEventListener('play', () => panel.querySelector('[data-action="toggle-play"]').textContent = 'Pause');
  video.addEventListener('pause', () => panel.querySelector('[data-action="toggle-play"]').textContent = 'Play');
  video.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); togglePlay(caseId); });
  panel.querySelector('[data-action="toggle-play"]').addEventListener('click', () => togglePlay(caseId));
  panel.querySelector('[data-action="fullscreen"]').addEventListener('click', () => toggleFullscreen(wrapper));
  panel.querySelector('[data-action="retry-media"]').addEventListener('click', () => { hideMediaError(caseId); video.load(); });
  panel.querySelector('[data-action="volume"]').addEventListener('input', e => { video.volume = Number(e.target.value); });
  panel.querySelector('[data-action="scrub"]').addEventListener('input', e => seekToFrame(caseId, Number(e.target.value)));
  panel.querySelector('[data-action="version"]').addEventListener('change', e => switchVersion(caseId, e.target.value));
  updatePlayerUi(caseId);
}
function setPlayerSource(video, path) {
  if (!video || video.dataset.path === path) return;
  video.dataset.path = path;
  video.removeAttribute('src');
  video.load();
  video.src = path;
  video.load();
}
function bindTimeline(caseId) {
  const markerBox = document.querySelector(`#case-${caseId} [data-role="markers"]`); if (!markerBox) return;
  const count = Math.max(1, caseFrameCount(caseId) - 1);
  markerBox.innerHTML = allTasks(caseId).map((task, index) => {
    const startCase = task.source_frame_start - caseData(caseId).source_frame_start;
    const endCase = task.source_frame_end - caseData(caseId).source_frame_start;
    const left = clamp(startCase / count * 100, 0, 100); const width = Math.max(0.4, (endCase - startCase) / count * 100);
    const cls = markerClass(task); const active = task.review_task_id === state.selectedTask[caseId] ? ' active' : '';
    const style = endCase > startCase ? `left:${left}%;width:${width}%;top:${(index % 3) * 4}px` : `left:${left}%;top:${(index % 3) * 4}px`;
    return `<button class="marker ${endCase > startCase ? 'marker-span' : ''} ${cls}${active}" title="${esc(markerTooltip(caseId, task))}" aria-label="${esc(markerTooltip(caseId, task))}" style="${style}" data-task-id="${esc(task.review_task_id)}"></button>`;
  }).join('');
  markerBox.querySelectorAll('.marker').forEach(btn => btn.addEventListener('click', () => { activeVideo(caseId)?.pause(); selectTask(caseId, btn.dataset.taskId, { jump: true, preserveScroll: true }); }));
}
function markerClass(task) { const stateName = taskState(task); if (stateName === 'correct' || stateName === 'resolved') return 'resolved'; if (stateName === 'needs_correction') return 'blocking'; if (stateName === 'unsure') return 'partial'; if (String(task.issue_type || '').includes('false') || String(task.issue_type || '').includes('wrong')) return 'false'; if (task.blocking) return 'blocking'; if (task.severity === 'low' || task.optional) return 'info'; return 'partial'; }
function markerTooltip(caseId, task) { const startCase = task.source_frame_start - caseData(caseId).source_frame_start; const endCase = task.source_frame_end - caseData(caseId).source_frame_start; const sourceRange = task.source_frame_end !== task.source_frame_start ? `${task.source_frame_start}-${task.source_frame_end}` : `${task.source_frame_start}`; const timeRange = endCase !== startCase ? `${timeText(frameTime(caseId,startCase))}-${timeText(frameTime(caseId,endCase))}` : timeText(frameTime(caseId,startCase)); const decision = state.decisions[task.review_task_id]?.action ? actionLabel(state.decisions[task.review_task_id].action) : 'Open'; return `${cleanTaskTitle(task)}\\n${task.category}\\nSource frame ${sourceRange}\\nCase time ${timeRange}\\n${task.blocking ? 'Blocking' : task.optional ? 'Optional' : 'Review'}\\nDecision: ${decision}`; }
function updateFrameFromTime(caseId, time) { const next = clamp(Math.round(time * fpsValue(caseId)), 0, caseFrameCount(caseId) - 1); if (next !== state.frame[caseId]) { state.frame[caseId] = next; persistFrame(caseId); updatePlayerUi(caseId); } }
function persistFrame(caseId) { localStorage.setItem(storageKey(`frame_${caseId}`), String(state.frame[caseId] || 0)); }
async function switchVersion(caseId, key) { const video = activeVideo(caseId); const wasPlaying = video && !video.paused; const frame = clamp(state.frame[caseId] || 0, 0, caseFrameCount(caseId) - 1); if (video) video.pause(); state.frame[caseId] = frame; state.version[caseId] = key; localStorage.setItem(storageKey(`version_${caseId}`), key); persistFrame(caseId); const version = selectedVersion(caseId); if (video) setPlayerSource(video, version.path); const select = document.querySelector(`#case-${caseId} [data-action="version"]`); if (select) select.value = key; await syncPlayerToState(caseId); if (wasPlaying) await togglePlay(caseId, true); }
async function syncPlayerToState(caseId) { const video = activeVideo(caseId); if (video) await seekToFrame(caseId, state.frame[caseId] || 0, { keepPaused: true }); }
async function seekToFrame(caseId, frame, opts = {}) { const video = activeVideo(caseId); if (!video) return; const next = clamp(Math.round(frame), 0, caseFrameCount(caseId) - 1); const wasPlaying = !video.paused && !opts.keepPaused; video.pause(); state.frame[caseId] = next; persistFrame(caseId); await waitMetadata(video, 1600).catch(() => showMediaError(caseId, video.dataset.path, 'metadata')); await seekVideo(video, frameTime(caseId, next)); updatePlayerUi(caseId); if (wasPlaying) await togglePlay(caseId, true); }
function stepFrame(delta) { const caseId = visibleCaseId(); if (!isPrimaryCase(caseId)) return; seekToFrame(caseId, (state.frame[caseId] || 0) + delta); }
async function togglePlay(caseId, forcePlay = false) { const video = activeVideo(caseId); if (!video) return; hideMediaError(caseId); try { await waitMetadata(video, 2000); if (!forcePlay && !video.paused) { video.pause(); return; } await video.play(); } catch (err) { showMediaError(caseId, video.dataset.path || selectedVersion(caseId).path, err.message || 'play failed'); } }
function updatePlayerUi(caseId) { const panel = document.getElementById(`case-${caseId}`); if (!panel) return; const frame = clamp(state.frame[caseId] || 0, 0, caseFrameCount(caseId)-1); panel.querySelector('[data-role="hud"]').textContent = hudText(caseId); const range = panel.querySelector('[data-action="scrub"]'); if (range) range.value = String(frame); const time = panel.querySelector('[data-role="time"]'); if (time) time.textContent = `${timeText(frameTime(caseId, frame))} / ${timeText(caseData(caseId).duration_seconds || frameTime(caseId, caseFrameCount(caseId)-1))}`; }
function waitMetadata(video, timeoutMs) { if (video.readyState >= 1) return Promise.resolve(); return new Promise((resolve, reject) => { const t = setTimeout(() => cleanup(reject, new Error('metadata timeout')), timeoutMs); const cleanup = (fn, value) => { clearTimeout(t); video.removeEventListener('loadedmetadata', ready); video.removeEventListener('error', error); fn(value); }; const ready = () => cleanup(resolve); const error = () => cleanup(reject, new Error(`media error ${video.error ? video.error.code : 'unknown'}`)); video.addEventListener('loadedmetadata', ready, {once:true}); video.addEventListener('error', error, {once:true}); }); }
function seekVideo(video, time) { return new Promise(resolve => { let done = false; const finish = () => { if (done) return; done = true; video.removeEventListener('seeked', finish); if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => resolve()); else resolve(); }; video.addEventListener('seeked', finish, {once:true}); try { video.currentTime = Math.max(0, time); } catch { resolve(); } setTimeout(finish, 900); }); }
function showMediaError(caseId, path, code) { const el = document.querySelector(`#case-${caseId} [data-role="media-error"]`); if (!el) return; el.classList.add('visible'); el.querySelector('span').textContent = `Playback error: ${path} (code ${code})`; }
function hideMediaError(caseId) { const el = document.querySelector(`#case-${caseId} [data-role="media-error"]`); if (el) el.classList.remove('visible'); }
function toggleFullscreen(wrapper) { if (!document.fullscreenElement) wrapper.requestFullscreen().catch(() => {}); else document.exitFullscreen().catch(() => {}); }

function reviewPanelHtml(caseId) {
  const tasks = filteredTasks(caseId); const counts = taskCounts(caseId); const selected = selectedTask(caseId);
  const filters = ['Open','Needs correction','Unsure / Escalated','Completed','Resolved','All'];
  return `<div class="workflow-stepper">${workflowStepperHtml(caseId)}</div>${workflowMessageHtml(caseId)}<div class="task-summary"><div class="stat"><span>Open</span><strong>${counts.open}</strong></div><div class="stat"><span>Needs correction</span><strong>${counts.needs_correction}</strong></div><div class="stat"><span>Unsure / Escalated</span><strong>${counts.unsure}</strong></div><div class="stat"><span>Completed</span><strong>${counts.correct}</strong></div><div class="stat"><span>Resolved</span><strong>${counts.resolved}</strong></div><div class="stat"><span>Current</span><strong>${currentTaskNumber(caseId)}</strong></div></div><div class="filter-row">${filters.map(f => `<button class="${state.filter[caseId] === f ? 'active' : ''}" data-action="filter" data-filter="${f}">${f}</button>`).join('')}</div><div class="task-list">${tasks.map(t => taskCardHtml(caseId, t)).join('') || '<p class="muted">No review items in this filter.</p>'}</div>${selected && requiresEditor(selected) ? '<button class="primary open-editor" data-action="open-editor">Open in editor</button>' : ''}`;
}
function allTasks(caseId) { return [...(DATA.tasks?.tasks || []), ...state.manualTasks].filter(t => t.case_id === caseId && !t.technical_only); }
function filteredTasks(caseId) { const f = state.filter[caseId] || 'Open'; return allTasks(caseId).filter(t => f === 'All' || filterMatchesTask(f, t)); }
function filterMatchesTask(filter, task) { const stateName = taskState(task); if (filter === 'Open') return stateName === 'open'; if (filter === 'Needs correction') return stateName === 'needs_correction' || stateName === 'correction_pending_verification'; if (filter === 'Unsure / Escalated') return stateName === 'unsure'; if (filter === 'Completed') return stateName === 'correct'; if (filter === 'Resolved') return stateName === 'resolved'; return true; }
function firstOpenTask(caseId) { return allTasks(caseId).find(t => taskState(t) === 'open'); }
function currentTaskNumber(caseId) { const tasks = allTasks(caseId); const index = tasks.findIndex(t => t.review_task_id === state.selectedTask[caseId]); return index >= 0 ? `${index + 1}/${tasks.length}` : `0/${tasks.length}`; }
function taskState(task) { const action = String(state.decisions[task.review_task_id]?.action || ''); if (!action) return 'open'; if (action === ACTIONS.looks) return 'correct'; if (action === ACTIONS.unsure) return 'unsure'; if ([ACTIONS.leak, ACTIONS.partial, ACTIONS.falseRedaction, ACTIONS.wrong].includes(action)) return 'needs_correction'; if ([ACTIONS.notSensitive, ACTIONS.cosmetic].includes(action)) return 'resolved'; return 'open'; }
function taskCounts(caseId) { const counts = {open:0, needs_correction:0, unsure:0, correct:0, resolved:0}; for (const task of allTasks(caseId)) { const value = taskState(task); counts[value] = (counts[value] || 0) + 1; } return counts; }
function decisionGreen(taskId) { const task = [...(DATA.tasks?.tasks || []), ...state.manualTasks].find(t => t.review_task_id === taskId); if (!task) return false; return ['correct','resolved'].includes(taskState(task)); }
function taskCardHtml(caseId, task) { const selected = task.review_task_id === state.selectedTask[caseId]; const decision = state.decisions[task.review_task_id]; const stateName = taskState(task); return `<article class="task-card ${selected ? 'selected' : ''}" data-task-id="${esc(task.review_task_id)}" data-task-state="${esc(stateName)}"><div class="task-title">${esc(cleanTaskTitle(task))}</div><div class="task-meta">${task.category} · source ${task.source_frame_start}${task.source_frame_end !== task.source_frame_start ? `-${task.source_frame_end}` : ''} · ${timeText((task.source_frame_start - caseData(caseId).source_frame_start)/fpsValue(caseId))}</div><p>${esc(task.what_to_inspect || 'Check this interval in the video.')}</p><span class="badge ${badgeClassForTask(task)}">${decision ? esc(actionLabel(decision.action)) : task.optional ? 'Optional open' : task.blocking ? 'Open required' : 'Open'}</span><div class="task-actions">${decisionButtons(task).map(a => `<button data-action="decision" data-task-id="${esc(task.review_task_id)}" data-decision="${a.action}" aria-pressed="${decision?.action === a.action ? 'true' : 'false'}" class="decision-choice decision-${a.tone}${decision?.action === a.action ? ' selected' : ''}">${esc(a.label)}</button>`).join('')}</div><div class="saved-note" data-saved-for="${esc(task.review_task_id)}">${state.savedFlash[task.review_task_id] ? 'Saved' : ''}</div></article>`; }
function badgeClassForTask(task) { const stateName = taskState(task); if (stateName === 'correct' || stateName === 'resolved') return 'done'; if (stateName === 'needs_correction' || stateName === 'unsure' || task.blocking) return 'open'; return 'warn'; }
function cleanTaskTitle(task) { if (task.issue_type === 'spot_check' || task.issue_type === 'optional_spot_check') return 'Spot check this moment'; if (task.issue_type === 'possible_false_redaction' || String(task.issue_type || '').includes('false')) return 'Possible false redaction'; if (task.issue_type === 'possible_privacy_gap' && task.category === 'license_plate') return 'Possible plate privacy gap'; if (task.issue_type === 'possible_privacy_gap' && task.category === 'face') return 'Possible face privacy gap'; if (task.issue_type === 'manual_reviewer_issue' || task.issue_type === 'reviewer_created_issue') return 'Reviewer-created issue'; return task.title || 'Review this interval'; }
function workflowStepperHtml(caseId) { const counts = taskCounts(caseId); const active = counts.open > 0 ? 'Review' : counts.needs_correction > 0 || counts.unsure > 0 ? 'Corrections' : 'Full-video sweep'; return ['Review','Corrections','Full-video sweep','Finalize','Final result'].map(step => `<span class="badge ${step === active ? 'warn' : 'done'}">${step}</span>`).join(' '); }
function workflowMessageHtml(caseId) { const counts = taskCounts(caseId); if (counts.open === 0 && (counts.needs_correction > 0 || counts.unsure > 0)) return `<div class="stat"><strong>Review complete</strong><p>Corrections or escalations still require action.</p><button data-action="filter" data-filter="Needs correction">Open Corrections</button></div>`; if (counts.open === 0) return `<div class="stat"><strong>All review items are resolved</strong><p>Complete the full-video sweep before finalization.</p></div>`; return ''; }
function decisionButtons(task) { return [{label:'Correct', action:ACTIONS.looks, tone:'correct'}, {label:'Needs correction', action:ACTIONS.leak, tone:'needs-correction'}, {label:'Unsure', action:ACTIONS.unsure, tone:'unsure'}]; }
function bindReviewPanel(caseId) { const panel = document.getElementById(`case-${caseId}`); if (panel.dataset.reviewDelegated === 'true') return; panel.dataset.reviewDelegated = 'true'; panel.addEventListener('click', e => { const actionEl = closestElement(e.target, '[data-action]'); if (actionEl) { const action = actionEl.dataset.action; if (action === 'filter') { state.filter[caseId] = actionEl.dataset.filter; refreshSidePanel(caseId, {preserveScroll:true}); return; } if (action === 'decision') { e.preventDefault(); e.stopPropagation(); recordDecision(caseId, actionEl.dataset.taskId, actionEl.dataset.decision, actionEl); return; } if (action === 'open-editor') { setMode(caseId, 'edit'); return; } } const card = closestElement(e.target, '.task-card'); if (card && !closestElement(e.target, 'button,input,select,textarea,a')) selectTask(caseId, card.dataset.taskId, {jump:true, preserveScroll:true}); }); }
function refreshSidePanel(caseId, opts = {}) { const panel = document.getElementById(`case-${caseId}`); const side = panel?.querySelector('.side-panel'); if (!side) return; const scrollTop = side.scrollTop; side.innerHTML = state.mode[caseId] === 'edit' ? editPanelHtml(caseId) : reviewPanelHtml(caseId); if (state.mode[caseId] === 'edit') { bindEditPanel(caseId); bindEntityPanel(caseId); } bindTimeline(caseId); if (opts.preserveScroll) side.scrollTop = scrollTop; if (opts.scrollActive !== false) setTimeout(() => document.querySelector(`#case-${caseId} .side-panel [data-task-id="${CSS.escape(state.selectedTask[caseId] || '')}"]`)?.scrollIntoView({block:'nearest'}), 0); }
async function selectTask(caseId, taskId, opts = {}) { state.selectedTask[caseId] = taskId; const task = selectedTask(caseId); if (task && opts.jump) { const targetFrame = clamp(task.source_frame_start - caseData(caseId).source_frame_start, 0, caseFrameCount(caseId) - 1); await seekToFrame(caseId, targetFrame, { keepPaused: true }); state.frame[caseId] = targetFrame; persistFrame(caseId); updatePlayerUi(caseId); assertActiveTaskFrame(caseId, task); } refreshSidePanel(caseId, {preserveScroll: Boolean(opts.preserveScroll)}); }
function assertActiveTaskFrame(caseId, task) { const expected = task.source_frame_start - caseData(caseId).source_frame_start; if (state.frame[caseId] !== expected) showSessionError(`Frame navigation mismatch: expected case frame ${expected}, got ${state.frame[caseId]}.`); }
async function recordDecision(caseId, taskId, action, button = null) { if (!state.session) return showSessionError('Start review before recording decisions.'); const task = allTasks(caseId).find(t => t.review_task_id === taskId); setDecisionPending(button); try { const event = await postJson('/api/decisions', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, task_id: taskId, action, note: decisionNote(action, task) }); state.decisions[taskId] = event.event.after_state; state.savedFlash[taskId] = true; refreshSidePanel(caseId, {preserveScroll:true, scrollActive:false}); if (action === ACTIONS.leak && task?.issue_type === 'spot_check') await createManualIssue(caseId, task.source_frame_start, task.source_frame_end, 'Privacy issue from spot check'); const next = firstOpenTask(caseId); if (next) { await selectTask(caseId, next.review_task_id, {jump:true, preserveScroll:true}); } else { refreshSidePanel(caseId, {preserveScroll:true}); } setTimeout(() => { delete state.savedFlash[taskId]; refreshSidePanel(caseId, {preserveScroll:true, scrollActive:false}); }, 1200); } catch (err) { setDecisionError(button, err.message); showSessionError(`Decision was not saved: ${err.message}`); } }
function setDecisionPending(button) { const card = button?.closest('.task-card'); if (!card) return; card.querySelectorAll('[data-action="decision"]').forEach(choice => { choice.classList.toggle('selected', choice === button); choice.setAttribute('aria-pressed', choice === button ? 'true' : 'false'); choice.disabled = true; }); const note = card.querySelector('.saved-note'); if (note) note.textContent = 'Saving...'; }
function setDecisionError(button, message) { const card = button?.closest('.task-card'); if (!card) return; const savedAction = state.decisions[card.dataset.taskId]?.action || ''; card.querySelectorAll('[data-action="decision"]').forEach(choice => { choice.disabled = false; choice.classList.toggle('selected', choice.dataset.decision === savedAction); choice.setAttribute('aria-pressed', choice.dataset.decision === savedAction ? 'true' : 'false'); }); const note = card.querySelector('.saved-note'); if (note) note.textContent = `Could not save: ${message}`; }
function decisionNote(action, task) { if (action === ACTIONS.leak || action === ACTIONS.partial) return 'Reviewer marked this item for correction.'; if (action === ACTIONS.falseRedaction) return 'Reviewer requested bounded redaction removal.'; return `Reviewer selected ${actionLabel(action)} for ${task?.review_task_id || 'task'}.`; }
function requiresEditor(task) { const d = state.decisions[task.review_task_id]; return d && [ACTIONS.leak, ACTIONS.partial, ACTIONS.falseRedaction, ACTIONS.wrong].includes(d.action); }

function editPanelHtml(caseId) { const task = selectedTask(caseId); const draft = state.draft.caseId === caseId ? state.draft : {start:null,end:null,keyframes:[]}; return `<div class="edit-section"><section class="edit-card"><h3>Selected issue</h3><p class="task-title">${esc(task ? cleanTaskTitle(task) : 'No issue selected')}</p><p class="task-meta">${task ? `source ${task.source_frame_start}-${task.source_frame_end}` : 'Choose a review item or add an issue at the current frame.'}</p><div class="button-row"><button data-action="add-issue">Add issue at current frame</button><button data-action="back-review">Back to Review</button></div></section><section class="edit-card"><h3>Correction interval</h3><div class="button-row"><button data-action="set-start">Set start at current frame</button><button data-action="set-end">Set end at current frame</button></div><p class="muted">Start: ${draft.start ?? 'not set'} · End: ${draft.end ?? 'not set'} · ${draft.start !== null && draft.end !== null ? `${Math.abs(draft.end-draft.start)+1} frames` : 'set boundaries first'}</p></section><section class="edit-card"><h3>Keyframes</h3><div class="editor-canvas-wrap"><canvas id="editorCanvas-${caseId}" width="640" height="360"></canvas></div><div class="button-row"><button data-action="add-keyframe">Add keyframe at current frame</button><button data-action="barrier">Add not-visible / scene-cut barrier</button></div><div class="keyframe-list">${draft.keyframes.map(k => `<div class="keyframe-row"><span>Frame ${k.source_frame}</span><span>${k.bbox_xyxy.join(', ')}</span><button data-action="delete-keyframe" data-frame="${k.source_frame}">Delete</button></div>`).join('') || '<p class="muted">No keyframes yet.</p>'}</div></section><section class="edit-card"><h3>Preview and save</h3><div class="button-row"><button class="primary" data-action="preview-correction">Preview correction</button><button data-action="save-correction">Save correction revision</button><button data-action="verify-correction">Verify corrected interval</button><button data-action="discard-draft">Discard draft</button></div><p class="muted" id="correctionPreview-${caseId}">${state.latestReviewed[caseId] ? `Latest reviewed preview: ${esc(state.latestReviewed[caseId].path)}` : 'No correction preview yet.'}</p></section><section class="edit-card"><h3>Objects</h3><div class="entity-filter-row">${entityFilterButtons('all')}</div><div class="object-list">${entityCardsHtml(caseId)}</div>${unresolvedEvidenceHtml(caseId)}</section></div>`; }
function bindEditPanel(caseId) { const panel = document.getElementById(`case-${caseId}`); panel.querySelector('[data-action="back-review"]')?.addEventListener('click', () => setMode(caseId, 'review')); panel.querySelector('[data-action="set-start"]')?.addEventListener('click', () => { ensureDraft(caseId); state.draft.start = sourceFrame(caseId); renderCase(caseId); }); panel.querySelector('[data-action="set-end"]')?.addEventListener('click', () => { ensureDraft(caseId); state.draft.end = sourceFrame(caseId); renderCase(caseId); }); panel.querySelector('[data-action="add-keyframe"]')?.addEventListener('click', () => addDraftKeyframe(caseId)); panel.querySelector('[data-action="barrier"]')?.addEventListener('click', () => addDraftBarrier(caseId)); panel.querySelector('[data-action="discard-draft"]')?.addEventListener('click', () => { state.draft = { caseId: null, start: null, end: null, keyframes: [], selectedEntityId: null, selectedTaskId: null }; renderCase(caseId); }); panel.querySelector('[data-action="preview-correction"]')?.addEventListener('click', () => saveCorrection(caseId, {previewOnly:true})); panel.querySelector('[data-action="save-correction"]')?.addEventListener('click', () => saveCorrection(caseId)); panel.querySelector('[data-action="verify-correction"]')?.addEventListener('click', () => verifyCorrection(caseId)); panel.querySelector('[data-action="add-issue"]')?.addEventListener('click', () => createManualIssue(caseId, sourceFrame(caseId), sourceFrame(caseId), 'Reviewer discovered issue')); panel.querySelectorAll('[data-action="delete-keyframe"]').forEach(btn => btn.addEventListener('click', () => { state.draft.keyframes = state.draft.keyframes.filter(k => String(k.source_frame) !== btn.dataset.frame); renderCase(caseId); })); drawEditorCanvas(caseId); }
function ensureDraft(caseId) { if (state.draft.caseId !== caseId) state.draft = { caseId, start: null, end: null, keyframes: [], selectedEntityId: null, selectedTaskId: state.selectedTask[caseId] }; }
function addDraftKeyframe(caseId) { ensureDraft(caseId); const frame = sourceFrame(caseId); const box = defaultDraftBox(caseId); const existing = state.draft.keyframes.filter(k => k.source_frame !== frame); state.draft.keyframes = [...existing, { source_frame: frame, bbox_xyxy: box }].sort((a,b) => a.source_frame-b.source_frame); if (state.draft.start === null) state.draft.start = frame; if (state.draft.end === null) state.draft.end = frame; renderCase(caseId); }
function addDraftBarrier(caseId) { ensureDraft(caseId); const frame = sourceFrame(caseId); state.draft.keyframes.push({ source_frame: frame, bbox_xyxy: defaultDraftBox(caseId), state: 'not_visible' }); renderCase(caseId); }
function defaultDraftBox(caseId) { const m = caseData(caseId); const w = Number(m.width || 1280), h = Number(m.height || 720); return [Math.round(w*.42), Math.round(h*.42), Math.round(w*.58), Math.round(h*.54)]; }
function drawEditorCanvas(caseId) { const canvas = document.getElementById(`editorCanvas-${caseId}`); if (!canvas) return; const ctx = canvas.getContext('2d'); ctx.fillStyle = '#020617'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.fillStyle = '#cbd5e1'; ctx.fillText('Draw and adjust correction geometry on Original media. Keyframes are reviewer-authored; interpolation is preview-only until verified.', 18, 32); ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 3; for (const k of (state.draft.keyframes || [])) { const [x1,y1,x2,y2] = k.bbox_xyxy; const m = caseData(caseId); const sx = canvas.width / (m.width || canvas.width), sy = canvas.height / (m.height || canvas.height); ctx.strokeRect(x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy); } }
async function saveCorrection(caseId, opts = {}) { if (!state.session) return showSessionError('Start review before saving corrections.'); ensureDraft(caseId); const start = Math.min(state.draft.start ?? sourceFrame(caseId), state.draft.end ?? sourceFrame(caseId)); const end = Math.max(state.draft.start ?? sourceFrame(caseId), state.draft.end ?? sourceFrame(caseId)); const keyframes = (state.draft.keyframes || []).filter(k => !k.state); if (keyframes.length < 1) return showSessionError('Add at least one keyframe before saving a correction.'); if (start !== end && keyframes.length < 2) return showSessionError('Intervals longer than one frame require start and end keyframes.'); const task = selectedTask(caseId) || await createManualIssue(caseId, start, end, 'Reviewer-created correction'); const correction = { case_id: caseId, task_id: task.review_task_id, category: task.category || 'license_plate', operation: 'add_privacy_region', source_frame_start: start, source_frame_end: end, keyframe_boxes: keyframes, interpolation_method: keyframes.length > 1 ? 'linear' : 'none', reviewer_identity: currentActor(), reason: opts.previewOnly ? 'Preview correction draft' : 'Reviewed correction revision', source_video_hash: caseData(caseId).source_video_hash, configuration_freeze_hash: DATA.workspace.configuration_freeze_hash || 'package_freeze' }; try { const response = await postJson('/api/corrections', { session_id: state.session.session_id, actor: currentActor(), correction, reason: correction.reason }); state.corrections.push(response.event.after_state); if (response.preview?.latest_reviewed) { state.latestReviewed[caseId] = { path: response.preview.latest_reviewed, frame_start: response.preview.frame_start, frame_end: response.preview.frame_end }; state.version[caseId] = 'latest_reviewed'; localStorage.setItem(storageKey(`version_${caseId}`), 'latest_reviewed'); } else if (response.preview?.corrected_preview) { state.latestReviewed[caseId] = { path: response.preview.corrected_preview, frame_start: response.preview.frame_start, frame_end: response.preview.frame_end }; } state.savedFlash[task.review_task_id] = true; renderCase(caseId); } catch (err) { showSessionError(`Correction was not saved: ${err.message}`); } }
async function verifyCorrection(caseId) { const task = selectedTask(caseId); if (!task) return; await recordDecision(caseId, task.review_task_id, ACTIONS.looks); }
async function createManualIssue(caseId, startSource, endSource, reason) { const task = { blocking: true, case_id: caseId, category: 'license_plate', correction_required: true, decision_state: 'pending', issue_type: 'reviewer_created_issue', normal_task_class: 'reviewer_created_issue', raw_signal_ids: [], review_task_id: `manual_${caseId}_${Date.now()}`, schema_version: 'case_reviewer_task.v1', severity: 'blocker', source_frame_start: startSource, source_frame_end: endSource, technical_only: false, title: 'Reviewer-created issue', what_happened: 'A reviewer added this issue during full-video review.', what_to_inspect: 'Open in editor, add a reviewed correction if privacy protection is needed.' }; if (state.session) await postJson('/api/entity-events', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, event: { event_type: 'manual_issue_created', case_id: caseId, entity_id: null, payload: { after: { task } } } }); state.manualTasks.push(task); state.selectedTask[caseId] = task.review_task_id; refreshSidePanel(caseId, {preserveScroll:true}); return task; }

function entityFilterButtons(active) { const rows = [['all','All'], ['face','People'], ['license_plate','Plates'], ['needs_review','Needs review'], ['unresolved','Unresolved evidence']]; return rows.map(([value,label]) => `<button data-action="entity-filter" data-filter="${value}" class="${value === active ? 'active' : ''}">${label}</button>`).join(''); }
function entityCardsHtml(caseId, filter = 'all') {
  const entities = effectiveEntities(caseId).filter(entity => entityMatchesFilter(entity, filter)).slice(0, 100);
  const cards = entities.map(entity => {
    const policy = entity.current_policy || entity.default_privacy_policy || 'redact';
    return `<article class="entity-card" data-entity-id="${esc(entity.entity_id)}" data-category="${esc(entity.category)}" data-policy="${esc(policy)}"><img src="${esc(entity.representative_thumbnail_path || '')}" alt="${esc(entity.label)} thumbnail" data-action="jump-entity"><div><div class="task-title">${esc(entity.label)}${entity.possible_duplicate ? ' · possible duplicate' : ''}</div><div class="task-meta">${entity.category === 'face' ? 'Person' : 'Plate'} · source ${entity.first_visible_source_frame ?? '?'}-${entity.last_visible_source_frame ?? '?'}</div><p class="muted">${esc(entityEvidenceText(entity))}</p><span class="badge ${policy === 'review_required' ? 'open' : 'done'}">${policyLabel(policy)}</span></div><div class="entity-actions"><div class="policy-segment" role="group" aria-label="${esc(entity.label)} policy"><button data-action="entity-policy" data-policy="redact" aria-pressed="${policy === 'redact'}">Redact</button><button data-action="entity-policy" data-policy="keep_visible" aria-pressed="${policy === 'keep_visible'}">Keep visible</button><button data-action="entity-policy" data-policy="not_sensitive" aria-pressed="${policy === 'not_sensitive'}">${entity.category === 'face' ? 'Not a person' : 'Not a plate'}</button></div><button data-action="entity-merge">Merge...</button></div><details><summary>Technical details</summary><pre>${esc(JSON.stringify({entity_id: entity.entity_id, state: entity.review_entity_state, tracks: entity.member_detector_tracker_ids || entity.automatic_source_track_ids, attached_fragments: entity.attached_fragment_track_ids || [], child_ids: entity.member_detector_tracker_ids || [], provenance: entity.created_provenance, possible_duplicates: entity.possible_duplicate_suggestions || []}, null, 2))}</pre></details></article>`;
  }).join('');
  if (filter === 'unresolved') return unresolvedEvidenceHtml(caseId) || '<p class="muted">No unresolved evidence.</p>';
  return cards || '<p class="muted">No review objects match this filter.</p>';
}
function entityMatchesFilter(entity, filter) { if (filter === 'all') return true; if (filter === 'face' || filter === 'license_plate') return entity.category === filter; if (filter === 'needs_review') return (entity.current_policy || entity.default_privacy_policy) === 'review_required' || entity.review_entity_state === 'unresolved'; return true; }
function entityEvidenceText(entity) { const summary = entity.evidence_summary || {}; const observations = summary.review_observation_count || summary.region_count || 0; const seconds = entity.first_visible_timestamp_seconds !== undefined && entity.last_visible_timestamp_seconds !== undefined ? Math.max(0, Number(entity.last_visible_timestamp_seconds) - Number(entity.first_visible_timestamp_seconds)) : null; const duration = seconds === null ? '' : ` across ${seconds.toFixed(1)} s`; const tracklets = summary.review_tracklet_count || summary.tracklet_count || 1; return `${observations} observations${duration} · ${tracklets} tracklet${tracklets === 1 ? '' : 's'}`; }
function effectiveEntities(caseId) { const base = (DATA.entities?.cases?.[caseId] || []).map(e => ({...e, ...(state.entityOverrides[e.entity_id] || {})})); return base.filter(e => e.current_status !== 'merged'); }
function unresolvedEvidence(caseId) { return DATA.entities?.unresolved_evidence?.[caseId] || DATA.entities?.cases_unresolved?.[caseId] || DATA.entities?.unresolved?.[caseId] || null; }
function unresolvedEvidenceHtml(caseId) { const unresolved = unresolvedEvidence(caseId) || DATA.entities?.case_unresolved?.[caseId] || null; const casePayload = DATA.entities?.unresolved_evidence_by_case?.[caseId] || unresolved; const fragments = casePayload?.fragments || []; const count = casePayload?.fragment_count ?? fragments.length; if (!count) return ''; return `<details class="technical"><summary>Unresolved evidence (${count})</summary><p class="muted">Low-evidence unmatched fragments are preserved for audit but are not named as separate reviewer objects.</p><pre>${esc(JSON.stringify(fragments.slice(0, 50), null, 2))}</pre></details>`; }
function bindEntityPanel(caseId, root = document.getElementById(`case-${caseId}`)) { if (!root) return; root.querySelectorAll('[data-action="jump-entity"]').forEach(img => img.addEventListener('click', () => { const entity = findEntity(caseId, img.closest('.entity-card').dataset.entityId); if (entity?.representative_source_frame !== undefined) { closeEntities(); seekToFrame(caseId, entity.representative_source_frame - caseData(caseId).source_frame_start); } })); root.querySelectorAll('[data-action="entity-policy"]').forEach(btn => btn.addEventListener('click', () => setEntityPolicy(caseId, btn.closest('.entity-card').dataset.entityId, btn.dataset.policy))); root.querySelectorAll('[data-action="entity-merge"]').forEach(btn => btn.addEventListener('click', () => openMergeModal(caseId, btn.closest('.entity-card').dataset.entityId))); root.querySelectorAll('[data-action="entity-filter"]').forEach(btn => btn.addEventListener('click', () => applyEntityFilter(root, caseId, btn.dataset.filter))); }
function openEntities(caseId) { const modal = document.getElementById('entitiesModal'); const content = document.getElementById('entitiesModalContent'); content.innerHTML = `<div class="entity-filter-row">${entityFilterButtons('all')}</div><div class="object-list">${entityCardsHtml(caseId)}</div><details class="technical"><summary>Possible duplicates</summary><pre>${esc(JSON.stringify(possibleDuplicateSuggestions(caseId), null, 2))}</pre></details>`; modal.dataset.caseId = caseId; modal.dataset.filter = 'all'; modal.classList.add('open'); modal.setAttribute('aria-hidden', 'false'); bindEntityPanel(caseId, content); }
function applyEntityFilter(root, caseId, filter) { const modal = root.closest('.modal'); if (modal) modal.dataset.filter = filter; const objectList = root.querySelector('.object-list'); const filters = root.querySelector('.entity-filter-row'); if (filters) filters.innerHTML = entityFilterButtons(filter); if (objectList) objectList.innerHTML = entityCardsHtml(caseId, filter); bindEntityPanel(caseId, root); }
function closeEntities() { const modal = document.getElementById('entitiesModal'); if (!modal) return; modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); }
function possibleDuplicateSuggestions(caseId) { const entities = effectiveEntities(caseId); const suggestions = []; for (let i=0; i<entities.length; i++) for (let j=i+1; j<entities.length; j++) { const a=entities[i], b=entities[j]; if (a.category !== b.category || entitiesOverlap(a,b)) continue; const gap = Math.max(0, Math.max(a.first_visible_source_frame || 0, b.first_visible_source_frame || 0) - Math.min(a.last_visible_source_frame || 0, b.last_visible_source_frame || 0)); if (gap <= 30) suggestions.push({entity_ids:[a.entity_id,b.entity_id], labels:[a.label,b.label], reason:'nearby_non_overlapping_tracklets'}); } return suggestions.slice(0, 20); }
function findEntity(caseId, id) { return effectiveEntities(caseId).find(e => e.entity_id === id); }
function policyLabel(policy) { return {redact:'Redact', keep_visible:'Keep visible', not_sensitive:'Invalid / not sensitive', review_required:'Needs review'}[policy] || policy; }
async function setEntityPolicy(caseId, entityId, policy) { if (!state.session) return showSessionError('Start review before changing object policy.'); const entity = findEntity(caseId, entityId); if (!entity) return; const reason = 'Reviewer object policy update.'; const before = {...entity}; const verificationTaskId = policy === 'redact' ? null : `object_policy_verify_${caseId}_${entityId}_${policy}`; const after = {...entity, current_policy: policy, default_privacy_policy: policy, privacy_action: policy, verification_task_id: verificationTaskId}; state.entityOverrides[entityId] = after; if (verificationTaskId && !state.manualTasks.some(t => t.review_task_id === verificationTaskId)) state.manualTasks.push(policyTask(caseId, after, verificationTaskId)); await postJson('/api/entity-events', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, event: { event_type: 'entity_policy_changed', case_id: caseId, entity_id: entityId, payload: { before, after, child_track_ids: entity.member_detector_tracker_ids || entity.automatic_source_track_ids || [] } }, reason }); refreshSidePanel(caseId, {preserveScroll:true}); refreshOpenEntities(caseId); }
function policyTask(caseId, entity, taskId) { return { blocking: true, case_id: caseId, category: entity.category, correction_required: false, decision_state: 'pending', issue_type: 'entity_policy_verification', raw_signal_ids: [], review_task_id: taskId, schema_version: 'case_reviewer_task.v1', severity: 'blocker', source_frame_start: entity.first_visible_source_frame || caseData(caseId).source_frame_start, source_frame_end: entity.first_visible_source_frame || caseData(caseId).source_frame_start, technical_only: false, title: `Verify ${policyLabel(entity.current_policy)} for ${entity.label}`, what_to_inspect: 'Confirm this object policy is correct before finalization.' }; }
function refreshOpenEntities(caseId) { const modal = document.getElementById('entitiesModal'); if (!modal?.classList.contains('open')) return; const content = document.getElementById('entitiesModalContent'); const filter = modal.dataset.filter || 'all'; content.innerHTML = `<div class="entity-filter-row">${entityFilterButtons(filter)}</div><div class="object-list">${entityCardsHtml(caseId, filter)}</div><details class="technical"><summary>Possible duplicates</summary><pre>${esc(JSON.stringify(possibleDuplicateSuggestions(caseId), null, 2))}</pre></details>`; bindEntityPanel(caseId, content); }
function openMergeModal(caseId, entityId) { if (!state.session) return showSessionError('Start review before merging objects.'); const source = findEntity(caseId, entityId); if (!source) return; const modal = document.getElementById('mergeModal'); const content = document.getElementById('mergeModalContent'); const candidates = mergeCandidates(caseId, source); state.mergeDraft = {caseId, sourceEntityId: entityId, targetEntityId: candidates[0]?.entity_id || null}; content.innerHTML = `<p class="task-title">Merge ${esc(source.label)} with...</p><p class="muted">Only same-category, non-overlapping candidates are shown.</p><div class="merge-candidates">${candidates.map((candidate, index) => mergeCandidateHtml(candidate, index === 0)).join('') || '<p class="muted">No safe merge candidates are available.</p>'}</div>`; modal.classList.add('open'); modal.setAttribute('aria-hidden', 'false'); content.querySelectorAll('.merge-card').forEach(card => card.addEventListener('click', () => { state.mergeDraft.targetEntityId = card.dataset.entityId; content.querySelectorAll('.merge-card').forEach(item => item.classList.toggle('selected', item === card)); })); }
function mergeCandidateHtml(candidate, selected) { return `<article class="merge-card ${selected ? 'selected' : ''}" data-entity-id="${esc(candidate.entity_id)}"><img src="${esc(candidate.representative_thumbnail_path || '')}" alt="${esc(candidate.label)} thumbnail"><div><div class="task-title">${esc(candidate.label)}</div><p class="task-meta">source ${candidate.first_visible_source_frame ?? '?'}-${candidate.last_visible_source_frame ?? '?'} · ${esc(candidate.similarity_label || 'Medium')} similarity</p><p class="muted">${esc(candidate.merge_reason || 'Compatible review-object evidence')}</p></div><input type="radio" ${selected ? 'checked' : ''} aria-label="Select ${esc(candidate.label)}"></article>`; }
function mergeCandidates(caseId, source) { return effectiveEntities(caseId).filter(e => e.entity_id !== source.entity_id && e.category === source.category && !entitiesOverlap(source, e)).map(e => { const gap = entityGap(source, e); const suggested = (source.possible_duplicate_suggestions || []).find(s => (s.child_track_id && (e.member_detector_tracker_ids || []).includes(s.child_track_id)) || (s.parent_track_id && (e.member_detector_tracker_ids || []).includes(s.parent_track_id))); const score = Number(suggested?.score || (gap <= 30 ? 0.65 : 0.45)); const reason = suggested ? 'Similar review evidence, but not automatically merged.' : `Temporal gap ${gap} frames`; return {...e, merge_score: score, similarity_label: score >= 0.8 ? 'High' : 'Medium', merge_reason: reason}; }).sort((a,b) => b.merge_score - a.merge_score || (a.first_visible_source_frame || 0) - (b.first_visible_source_frame || 0)).slice(0, 12); }
async function confirmMerge() { const {caseId, sourceEntityId, targetEntityId} = state.mergeDraft || {}; if (!caseId || !sourceEntityId || !targetEntityId) return showSessionError('Choose one merge candidate first.'); const source = findEntity(caseId, sourceEntityId); const target = findEntity(caseId, targetEntityId); if (!source || !target) return; if (entitiesOverlap(source, target)) return showSessionError('Merge rejected: these objects are simultaneously visible.'); const before = { source, target }; const merged = {...source, member_detector_tracker_ids: [...new Set([...(source.member_detector_tracker_ids || []), ...(target.member_detector_tracker_ids || [])])], automatic_source_track_ids: [...new Set([...(source.automatic_source_track_ids || []), ...(target.automatic_source_track_ids || [])])], merge_split_lineage: [...(source.merge_split_lineage || []), { operation:'merge', merged_entity_id: target.entity_id, source_entity_ids:[source.entity_id, target.entity_id] }]}; state.entityOverrides[source.entity_id] = merged; state.entityOverrides[target.entity_id] = {...target, current_status:'merged'}; await postJson('/api/entities/merge', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, before_state: before, after_state: { primary: merged, retired_entity_ids: [target.entity_id], source_entity_ids:[source.entity_id, target.entity_id] }, reason: 'Reviewer confirmed visual merge.' }); closeMergeModal(); refreshSidePanel(caseId, {preserveScroll:true}); refreshOpenEntities(caseId); }
function closeMergeModal() { const modal = document.getElementById('mergeModal'); if (!modal) return; modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); state.mergeDraft = { caseId: null, sourceEntityId: null, targetEntityId: null }; }
function entityGap(a,b) { const a1=a.first_visible_source_frame, a2=a.last_visible_source_frame, b1=b.first_visible_source_frame, b2=b.last_visible_source_frame; if ([a1,a2,b1,b2].some(v => v === null || v === undefined)) return 999999; if (entitiesOverlap(a,b)) return 0; return Math.max(0, Math.max(a1,b1) - Math.min(a2,b2)); }
function entitiesOverlap(a,b) { const a1=a.first_visible_source_frame, a2=a.last_visible_source_frame, b1=b.first_visible_source_frame, b2=b.last_visible_source_frame; if ([a1,a2,b1,b2].some(v => v === null || v === undefined)) return false; return Math.max(a1,b1) <= Math.min(a2,b2); }

async function restoreSession() { const sessionId = localStorage.getItem(storageKey('session_id')); if (!sessionId) return; try { await loadSessionState(sessionId); } catch { localStorage.removeItem(storageKey('session_id')); } }
async function loadSessionState(sessionId) { const res = await fetch(`/api/session-state?session_id=${encodeURIComponent(sessionId)}`); if (!res.ok) throw new Error(await res.text()); const payload = await res.json(); state.session = payload.session; state.attribution = payload.attribution; state.decisions = payload.decisions || {}; state.events = payload.events || []; state.corrections = payload.corrections || []; rebuildFromEvents(); }
function rebuildFromEvents() { state.manualTasks = []; state.entityOverrides = {}; for (const event of state.events) { if (event.event_type === 'manual_issue_created' && event.after_state?.task) state.manualTasks.push(event.after_state.task); if (event.event_type === 'entity_policy_changed' && event.after_state?.entity_id) { state.entityOverrides[event.after_state.entity_id] = event.after_state; if (event.after_state.verification_task_id) state.manualTasks.push(policyTask(event.case_id, event.after_state, event.after_state.verification_task_id)); } if (event.event_type === 'entities_merged' && event.after_state?.primary) { state.entityOverrides[event.after_state.primary.entity_id] = event.after_state.primary; for (const id of event.after_state.retired_entity_ids || []) state.entityOverrides[id] = { entity_id: id, current_status: 'merged' }; } } }
function renderSessionBanner() { const banner = document.getElementById('sessionBanner'); const start = document.getElementById('startSession'); if (!state.session) { banner.className = 'session-banner'; banner.textContent = 'No local review session yet. Start review to enable decisions and corrections.'; start.disabled = false; return; } banner.className = 'session-banner ready'; banner.textContent = `Local session ${state.session.session_id}${state.attribution?.display_name ? ` · Reviewer ${state.attribution.display_name}` : ' · anonymous until export/finalization'}`; start.disabled = true; }
function showSessionError(message) { const banner = document.getElementById('sessionBanner'); banner.className = 'session-banner error'; banner.textContent = message; }
async function startSession() { try { const payload = await postJson('/api/sessions', { reviewer: {}, source_revision: DATA.workspace.source_revision || null }); state.session = payload.session; localStorage.setItem(storageKey('session_id'), state.session.session_id); await loadSessionState(state.session.session_id); renderApp(); } catch (err) { showSessionError(`Could not start local review session: ${err.message}`); } }
function openAttributionModal() { document.getElementById('attributionModal').classList.add('open'); document.getElementById('attributionModal').setAttribute('aria-hidden', 'false'); document.getElementById('reviewerName').focus(); }
function closeAttributionModal() { document.getElementById('attributionModal').classList.remove('open'); document.getElementById('attributionModal').setAttribute('aria-hidden', 'true'); }
async function submitAttribution() { if (!state.session) return showSessionError('Start review before setting attribution.'); const name = document.getElementById('reviewerName').value.trim(); if (!name) { document.getElementById('attributionError').textContent = 'Reviewer name is required.'; return; } try { const res = await postJson('/api/reviewer-attribution', { session_id: state.session.session_id, display_name: name, actor: currentActor(), note: 'Reviewer attribution set.' }); state.attribution = res.attribution; await loadSessionState(state.session.session_id); closeAttributionModal(); renderApp(); } catch (err) { document.getElementById('attributionError').textContent = err.message; } }
function exportSession() { if (!state.session) return showSessionError('Start review before exporting.'); if (!state.attribution?.display_name) { openAttributionModal(); return; } window.open(`sessions/${state.session.session_id}/review_summary.json`, '_blank'); }
function renderAudit() { const panel = document.getElementById('case-audit'); panel.innerHTML = `<h2>Audit Trail</h2><p class="case-subtitle">Append-only events with hash-chain validation.</p><div class="audit-list">${(state.events || []).map(e => `<article class="audit-card"><strong>${esc(e.event_type)}</strong><p class="task-meta">${esc(e.timestamp_utc)} · ${esc(e.case_id || 'workspace')}</p><p>${esc(e.note || '')}</p><details><summary>Technical JSON</summary><pre>${esc(JSON.stringify(e, null, 2))}</pre></details></article>`).join('') || '<p class="muted">No events yet.</p>'}</div><details class="technical"><summary>Raw audit signals preserved outside the normal queue (${(DATA.rawSignals?.raw_signals || []).length})</summary><pre>${esc(JSON.stringify(DATA.rawSignals, null, 2))}</pre></details>`; }
function renderFinal() { const panel = document.getElementById('case-final'); panel.innerHTML = `<h2>Final Results</h2><p class="case-subtitle">Reviewed-final outputs remain unavailable until finalization gates pass.</p><div class="final-card"><p class="muted">Finalization is currently controlled by the same validated backend gates. Use Review/Edit to resolve blockers, then finalize from the supported workflow.</p><div class="button-row"><button data-action="finalize-day">Check Day finalization</button><button data-action="finalize-night">Check Night finalization</button></div><pre id="finalizeStatus"></pre></div>`; panel.querySelector('[data-action="finalize-day"]').addEventListener('click', () => finalizeCase('day')); panel.querySelector('[data-action="finalize-night"]').addEventListener('click', () => finalizeCase('night')); }
async function finalizeCase(caseId) { if (!state.session) return showSessionError('Start review before finalizing.'); if (!state.attribution?.display_name) { openAttributionModal(); return; } const out = document.getElementById('finalizeStatus'); try { const res = await postJson('/api/finalize', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId }); out.textContent = JSON.stringify(res.result, null, 2); } catch (err) { out.textContent = `Finalization blocked: ${err.message}`; } }
function actionLabel(action) { if (action === ACTIONS.looks) return 'Correct'; if ([ACTIONS.leak, ACTIONS.partial, ACTIONS.falseRedaction, ACTIONS.wrong].includes(action)) return 'Needs correction'; if (action === ACTIONS.unsure) return 'Unsure'; if (action === ACTIONS.notSensitive) return 'Not sensitive'; if (action === ACTIONS.cosmetic) return 'Cosmetic only'; const found = (DATA.tasks?.actions || []).find(a => a.action === action); return found ? found.label : action; }
function installGlobalHandlers() { document.body.tabIndex = -1; document.body.focus(); document.getElementById('startSession').addEventListener('click', startSession); document.getElementById('changeAttribution').addEventListener('click', openAttributionModal); document.getElementById('exportSession').addEventListener('click', exportSession); document.getElementById('submitAttribution').addEventListener('click', submitAttribution); document.getElementById('cancelAttribution').addEventListener('click', closeAttributionModal); document.getElementById('closeEntities')?.addEventListener('click', closeEntities); document.getElementById('cancelMerge')?.addEventListener('click', closeMergeModal); document.getElementById('confirmMerge')?.addEventListener('click', confirmMerge); document.querySelectorAll('.case-tabs button').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab))); window.addEventListener('keydown', onKeyDown, { capture: true }); document.addEventListener('fullscreenchange', () => updatePlayerUi(visibleCaseId())); }
function isTextEntryTarget(element) { const tag = element?.tagName?.toLowerCase(); if (!tag) return false; if (element.isContentEditable) return true; if (tag === 'textarea' || tag === 'select') return true; if (tag !== 'input') return false; const type = String(element.type || 'text').toLowerCase(); return !['button','checkbox','color','file','hidden','image','radio','range','reset','submit'].includes(type); }
function onKeyDown(event) { if (isTextEntryTarget(document.activeElement)) return; if (event.key === 'ArrowRight' || event.key === 'Right') { event.preventDefault(); stepFrame(event.shiftKey ? 10 : 1); } else if (event.key === 'ArrowLeft' || event.key === 'Left') { event.preventDefault(); stepFrame(event.shiftKey ? -10 : -1); } else if (event.key === ' ') { event.preventDefault(); togglePlay(visibleCaseId()); } else if (event.key.toLowerCase() === 'f') { const wrapper = document.querySelector(`#case-${visibleCaseId()}.active .player-wrapper`); if (wrapper) toggleFullscreen(wrapper); } else if (event.key === 'Escape' && document.fullscreenElement) { document.exitFullscreen().catch(() => {}); } else if (event.key === 'Escape') { closeMergeModal(); closeEntities(); closeAttributionModal(); } }
init().catch(err => { document.body.innerHTML = `<pre>${err.stack || err}</pre>`; });
"""
