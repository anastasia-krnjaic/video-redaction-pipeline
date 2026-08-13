from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv

from redactor import (
    case_ids,
    case_manifests,
    media,
    review_api_contract,
    review_finalization,
    review_sessions,
)
from redactor.detection import rational_to_float
from redactor.privacy_records import CATEGORY_FACE, CATEGORY_PLATE, PrivacyRegionRecord, stable_hash

SCHEMA_VERSION = "mixed_review_workspace.v1"
WORKSPACE_DIR_NAME = "review_workspace"
TASK_COLLECTION_SCHEMA_VERSION = "mixed_case_review_tasks.v1"
TASK_SCHEMA_VERSION = "case_reviewer_task.v1"
ENTITY_REGISTRY_SCHEMA_VERSION = "review_entity_registry.v2"
SESSION_STATE_SCHEMA_VERSION = "mixed_review_session_state.v1"
CATEGORY_RUN_IDENTITY_SCHEMA_VERSION = "category_run_manifest_identity.v1"
LATEST_REVIEWED_PUBLICATION_SCHEMA_VERSION = "latest_reviewed_publication_capability.v1"
TARGET_BINDING_SCHEMA_VERSION = "review_task_target_binding.v1"
TARGET_BINDING_ENTITY = "entity"
TARGET_BINDING_SOURCE_SPAN = "source_span"

CATEGORY_ORDER = {CATEGORY_FACE: 0, CATEGORY_PLATE: 1}
CATEGORY_LABEL_PREFIX = {CATEGORY_FACE: "Person", CATEGORY_PLATE: "Plate"}
ENTITY_ID_PREFIX = {CATEGORY_FACE: "entity_face", CATEGORY_PLATE: "entity_plate"}
FACE_POLICY_TO_ENTITY_POLICY = {
    "redact": "redact",
    "keep": "keep_visible",
    "review": "review_required",
}
PLATE_ACTIVE_STATUSES = {
    "accepted_plate_track": "redact",
    "needs_review": "review_required",
}
ACTION_ALIASES = {
    "Correct": review_sessions.ACTION_LOOKS_CORRECT,
    "Cosmetic only": review_sessions.ACTION_COSMETIC_ONLY,
    "False redaction": review_sessions.ACTION_FALSE_REDACTION,
    "Looks correct": review_sessions.ACTION_LOOKS_CORRECT,
    "Needs correction": review_sessions.ACTION_PRIVACY_LEAK,
    "Not a license plate": review_sessions.ACTION_NOT_SENSITIVE,
    "Not a person": review_sessions.ACTION_NOT_SENSITIVE,
    "Not sensitive": review_sessions.ACTION_NOT_SENSITIVE,
    "Partial coverage": review_sessions.ACTION_PARTIAL_COVERAGE,
    "Privacy leak": review_sessions.ACTION_PRIVACY_LEAK,
    "Uncertain / Escalate": review_sessions.ACTION_ESCALATE,
    "Wrong target": review_sessions.ACTION_WRONG_TARGET,
    "cosmetic_only": review_sessions.ACTION_COSMETIC_ONLY,
    "escalate": review_sessions.ACTION_ESCALATE,
    "false_redaction": review_sessions.ACTION_FALSE_REDACTION,
    "looks_correct": review_sessions.ACTION_LOOKS_CORRECT,
    "needs_correction": review_sessions.ACTION_PRIVACY_LEAK,
    "needs_review": review_sessions.ACTION_PRIVACY_LEAK,
    "no_issue_found": review_sessions.ACTION_LOOKS_CORRECT,
    "not_sensitive": review_sessions.ACTION_NOT_SENSITIVE,
    "partial_coverage": review_sessions.ACTION_PARTIAL_COVERAGE,
    "privacy_leak": review_sessions.ACTION_PRIVACY_LEAK,
    "wrong_target": review_sessions.ACTION_WRONG_TARGET,
}

SOURCE_SPAN_REVIEW_REASON_CODES = {
    "gap_exceeds_safe_interpolation_limit",
    "identity_competition",
    "insufficient_observations_for_auto_accept",
    "insufficient_observed_hits",
    "insufficient_strong_hits",
    "long_gap",
    "low_confidence_observations",
    "low_median_observed_score",
    "low_observation_density",
    "needs_review_interval",
    "observed_confirmation_window",
    "policy_assignment_review",
    "quality_status_rejected",
    "rejected_interval",
    "render_validation_failure",
    "strong_confirmation_window",
    "track_end",
    "track_start",
    "unresolved_interval",
    "unresolved_internal_gap",
    "unresolved_privacy_frames",
    "weak_only_observations",
}


@dataclass
class ReviewEntitySource:
    case_id: str
    category: str
    source_key: str
    policy: str
    first_frame: int | None = None
    last_frame: int | None = None
    representative_frame: int | None = None
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    provenance_counts: Counter[str] = field(default_factory=Counter)
    render_region_count: int = 0
    technical_detector_ids: set[str] = field(default_factory=set)
    technical_track_ids: set[str] = field(default_factory=set)
    source_lineage: dict[str, Any] = field(default_factory=dict)
    evidence_disposition: str = "active"
    merge_lineage: list[dict[str, Any]] = field(default_factory=list)
    task_ids: set[str] = field(default_factory=set)

    def observe_frame(self, frame: int | None) -> None:
        if frame is None:
            return
        value = int(frame)
        self.first_frame = value if self.first_frame is None else min(self.first_frame, value)
        self.last_frame = value if self.last_frame is None else max(self.last_frame, value)
        if self.representative_frame is None:
            self.representative_frame = value


@dataclass(frozen=True)
class TaskTargetBinding:
    kind: str
    entity_keys: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

ALL_REVIEW_ACTIONS = review_sessions.CANONICAL_REVIEW_ACTIONS

ACTION_LABELS = {
    review_sessions.ACTION_COSMETIC_ONLY: "Cosmetic only",
    review_sessions.ACTION_ESCALATE: "Uncertain / Escalate",
    review_sessions.ACTION_FALSE_REDACTION: "False redaction",
    review_sessions.ACTION_LOOKS_CORRECT: "Correct",
    review_sessions.ACTION_NOT_SENSITIVE: "Not sensitive",
    review_sessions.ACTION_PARTIAL_COVERAGE: "Partial coverage",
    review_sessions.ACTION_PRIVACY_LEAK: "Needs correction",
    review_sessions.ACTION_WRONG_TARGET: "Wrong target",
}


def build_case_review_workspace(
    context: Any,
    *,
    category_results: list[Any],
    render_plan: dict[str, Any],
    combined_preview: dict[str, Any] | None,
    case_blockers: tuple[str, ...] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one canonical review workspace for a case-level automatic preview."""

    case_id = case_ids.validate_case_id(context.case_id)
    selected_categories = tuple(str(category) for category in context.selected_categories)
    if not selected_categories:
        raise ValueError("review workspace requires at least one selected category")
    unsupported = sorted(set(selected_categories) - {CATEGORY_FACE, CATEGORY_PLATE})
    if unsupported:
        raise ValueError(f"review workspace unsupported categories: {unsupported}")

    case_root = Path(context.output_dir)
    workspace_dir = case_root / WORKSPACE_DIR_NAME
    prepare_workspace_dir(workspace_dir, overwrite=overwrite)

    preview_source, preview_source_role, preview_manifest = resolve_automatic_preview_source(
        context,
        render_plan=render_plan,
        combined_preview=combined_preview,
    )
    automatic_region_count = validate_automatic_preview_source(
        context,
        preview_source,
        preview_manifest,
        preview_source_role=preview_source_role,
    )
    category_run_identities = validated_category_run_manifest_identities(
        context,
        category_results=category_results,
        selected_categories=selected_categories,
    )

    media_dir = workspace_dir / "media" / case_id
    original_path = install_media(context.source_video, media_dir / "original.mp4", overwrite=True)
    automatic_path = install_media(
        preview_source,
        media_dir / "automatic_preview.mp4",
        overwrite=True,
    )

    frame_count = int(context.source_identity.frame_count or 0)
    source_frame_start = 0
    source_frame_end = max(0, frame_count - 1)
    fps = context.source_identity.fps or case_manifests.FPS_RATIONAL_DEFAULT
    duration = context.source_identity.duration_seconds
    media_inventory = build_media_inventory(
        context,
        original_path=original_path,
        automatic_path=automatic_path,
        preview_source=preview_source,
        preview_source_role=preview_source_role,
        workspace_dir=workspace_dir,
    )
    media_validation = validate_media_pair(
        context,
        original_path=original_path,
        automatic_path=automatic_path,
        media_inventory=media_inventory,
    )

    effective_regions_path = case_root / render_plan["effective_regions_artifact"]["path"]
    effective_regions = review_finalization.load_privacy_region_records(effective_regions_path)
    workspace_regions_path = workspace_dir / "regions" / f"{case_id}_current_regions.jsonl"
    install_media(effective_regions_path, workspace_regions_path, overwrite=True)
    category_tasks = load_category_review_tasks(case_root, selected_categories)
    entity_sources = build_entity_sources(
        case_root=case_root,
        case_id=case_id,
        selected_categories=selected_categories,
        records=effective_regions,
        category_tasks=category_tasks,
        fps=fps,
        source_frame_end=source_frame_end,
    )
    entity_build = build_entities_from_sources(
        case_id=case_id,
        selected_categories=selected_categories,
        sources=entity_sources,
        category_tasks=category_tasks,
        fps=fps,
        source_frame_end=source_frame_end,
    )
    tasks = build_tasks(
        case_id=case_id,
        selected_categories=selected_categories,
        category_tasks=category_tasks,
        entities_by_key=entity_build["entities_by_key"],
        source_frame_end=source_frame_end,
        fps=fps,
    )
    entities = assign_entity_labels(entity_build["entities"])
    thumbnail_audit = write_source_backed_entity_thumbnails(
        workspace_dir=workspace_dir,
        case_id=case_id,
        entities=entities,
        source_path=original_path,
        source_sha256=context.source_identity.sha256,
        source_frame_start=source_frame_start,
        width=context.source_identity.width,
        height=context.source_identity.height,
    )
    validate_workspace_payload(
        case_id=case_id,
        selected_categories=selected_categories,
        entities=entities,
        tasks=tasks,
        media_inventory=media_inventory,
        workspace_dir=workspace_dir,
        source_frame_end=source_frame_end,
    )

    configuration_freeze = build_configuration_freeze(
        context,
        category_results=category_results,
        render_plan=render_plan,
        preview_source_role=preview_source_role,
        preview_manifest=preview_manifest,
        automatic_region_count=automatic_region_count,
        media_inventory=media_inventory,
        category_run_identities=category_run_identities,
    )
    write_json(workspace_dir / "configuration_freeze.json", configuration_freeze)
    configuration_freeze_hash = stable_hash(configuration_freeze)

    case_manifest = case_manifests.CaseManifest(
        blockers=case_blockers,
        case_id=context.case_id,
        display_name=f"{context.case_id} Review",
        duration_seconds=duration,
        expected_output_roles=("original", "automatic"),
        face_evidence_inputs=category_evidence_inputs(category_results, CATEGORY_FACE),
        fps=fps,
        frame_count=frame_count,
        height=context.source_identity.height,
        lineage={
            "automatic_preview_role": preview_source_role,
            "category_run_manifest_identities": category_run_identities,
            "category_run_manifest_identity_hash": stable_hash(category_run_identities),
            "category_review_task_sources": {
                category: payload["source_path"]
                for category, payload in sorted(category_tasks.items())
            },
            "automatic_baseline_regions": {
                "path": relative_to_workspace(workspace_regions_path, workspace_dir),
                "region_count": len(effective_regions),
                "sha256": media.sha256_file(workspace_regions_path),
            },
            "mixed_review_workspace_non_final": True,
            "render_plan_hash": render_plan.get("render_plan_hash"),
            "schema_version": "mixed_review_case_lineage.v1",
            "selected_categories": list(selected_categories),
            "source_identity": context.source_identity.to_dict(),
        },
        parent_source_hash=context.source_identity.sha256,
        plate_evidence_inputs=category_evidence_inputs(category_results, CATEGORY_PLATE),
        reviewed_correction_inputs=(),
        source_frame_end=source_frame_end,
        source_frame_start=source_frame_start,
        source_path=relative_to_workspace(original_path, workspace_dir),
        source_video_hash=context.source_identity.sha256,
        status=case_manifests.CASE_STATUS_READY,
        video_roles=(
            case_manifests.VideoRole(
                label="Original",
                path=relative_to_workspace(original_path, workspace_dir),
                role="original",
                sha256=media.sha256_file(original_path),
                source_frame_end=source_frame_end,
                source_frame_start=source_frame_start,
            ),
            case_manifests.VideoRole(
                label="Automatic preview",
                path=relative_to_workspace(automatic_path, workspace_dir),
                role="automatic",
                sha256=media.sha256_file(automatic_path),
                source_frame_end=source_frame_end,
                source_frame_start=source_frame_start,
            ),
        ),
        width=context.source_identity.width,
    )
    case_payload = case_manifest.to_dict()
    case_path = workspace_dir / "cases" / f"{case_id}.json"
    atomic_write_json(case_path, case_payload)

    write_entity_registry(workspace_dir, case_id, entities)
    atomic_write_json(workspace_dir / "representative_thumbnail_audit.json", thumbnail_audit)
    atomic_write_json(workspace_dir / "review_entity_thumbnail_audit.json", thumbnail_audit)
    task_payload = {
        "actions": [
            {"action": action, "label": ACTION_LABELS[action]} for action in ALL_REVIEW_ACTIONS
        ],
        "case_id": case_id,
        "categories": list(selected_categories),
        "schema_version": TASK_COLLECTION_SCHEMA_VERSION,
        "task_count": len(tasks),
        "task_ordering": "source_frame_start,category,source_frame_end,review_task_id",
        "tasks": tasks,
    }
    atomic_write_json(workspace_dir / "reviewer_tasks.json", task_payload)
    atomic_write_json(
        workspace_dir / "review_tasks.json",
        task_payload
        | {
            "schema_version": TASK_COLLECTION_SCHEMA_VERSION,
            "workspace_alias_for": "reviewer_tasks.json",
        },
    )
    atomic_write_json(workspace_dir / "media_inventory.json", media_inventory)
    atomic_write_json(
        workspace_dir / "raw_audit_signals.json",
        {
            "raw_signal_count": 0,
            "raw_signals": [],
            "schema_version": "mixed_review_raw_audit_signals.v1",
        },
    )
    atomic_write_json(
        workspace_dir / "version_alignment_audit.json",
        version_alignment_audit(context, case_manifest),
    )
    session_state = {
        "case_id": case_id,
        "correction_count": 0,
        "decision_count": 0,
        "event_count": 0,
        "history_namespace": "sessions/",
        "latest_reviewed": {
            "reason": "Latest reviewed publication starts after a review session changes state",
            "status": "not_run",
        },
        "review_session": {
            "history_namespace": "sessions/",
            "status": "not_started",
        },
        "schema_version": SESSION_STATE_SCHEMA_VERSION,
    }
    atomic_write_json(workspace_dir / "session_state.json", session_state)

    validation = workspace_validation_report(
        context=context,
        selected_categories=selected_categories,
        entities=entities,
        tasks=tasks,
        media_inventory=media_inventory,
        preview_manifest=preview_manifest,
        automatic_region_count=automatic_region_count,
        media_validation=media_validation,
        category_run_identities=category_run_identities,
    )
    atomic_write_json(workspace_dir / "validation.json", validation)
    summary = workspace_summary(
        context=context,
        entities=entities,
        tasks=tasks,
        case_blockers=case_blockers,
        workspace_dir=workspace_dir,
        validation=validation,
        automatic_region_count=automatic_region_count,
    )
    atomic_write_json(workspace_dir / "workspace_summary.json", summary)

    support_artifacts = {
        "case_manifest": artifact_reference(case_root, case_path, "mixed_review_case_manifest"),
        "configuration_freeze": artifact_reference(
            case_root,
            workspace_dir / "configuration_freeze.json",
            "mixed_review_configuration_freeze",
        ),
        "entity_registry": artifact_reference(
            case_root,
            workspace_dir / "review_entities.json",
            "mixed_review_entities",
        ),
        "entity_thumbnail_audit": artifact_reference(
            case_root,
            workspace_dir / "representative_thumbnail_audit.json",
            "mixed_review_entity_thumbnail_audit",
        ),
        "media_inventory": artifact_reference(
            case_root,
            workspace_dir / "media_inventory.json",
            "mixed_review_media_inventory",
        ),
        "review_tasks": artifact_reference(
            case_root,
            workspace_dir / "reviewer_tasks.json",
            "mixed_review_tasks",
        ),
        "session_state": artifact_reference(
            case_root,
            workspace_dir / "session_state.json",
            "mixed_review_session_state",
        ),
        "validation": artifact_reference(
            case_root,
            workspace_dir / "validation.json",
            "mixed_review_validation",
        ),
        "workspace_summary": artifact_reference(
            case_root,
            workspace_dir / "workspace_summary.json",
            "mixed_review_workspace_summary",
        ),
        "automatic_baseline_regions": artifact_reference(
            case_root,
            workspace_regions_path,
            "mixed_review_automatic_baseline_regions",
        ),
    }
    workspace_manifest = {
        "automatic_preview": automatic_preview_identity(
            automatic_path=automatic_path,
            media_inventory=media_inventory,
            preview_manifest=preview_manifest,
            automatic_region_count=automatic_region_count,
            preview_source_role=preview_source_role,
            workspace_dir=workspace_dir,
        ),
        "case_content_identity": {
            "case_manifest_hash": stable_hash(case_payload),
            "case_manifest_path": f"cases/{case_id}.json",
            "schema_version": "mixed_review_case_content_identity.v1",
            "source_sha256": context.source_identity.sha256,
        },
        "case_id": case_id,
        "cases": {case_id: f"cases/{case_id}.json"},
        "category_run_manifest_identities": category_run_identities,
        "category_run_manifest_identity_hash": stable_hash(category_run_identities),
        "configuration_freeze": "configuration_freeze.json",
        "configuration_freeze_hash": configuration_freeze_hash,
        "content_identities": {
            key: artifact.to_dict() for key, artifact in sorted(support_artifacts.items())
        },
        "created_at_utc": media.utc_timestamp(),
        "lifecycle": {
            "evaluation": {
                "reason": "evaluation requires explicit ground truth inputs",
                "status": "not_run",
            },
            "finalization": {
                "reason": "mixed finalization is deferred",
                "status": "not_run",
            },
            "latest_reviewed": {
                "reason": "Latest reviewed publication requires reviewed-region compilation",
                "status": "not_run",
            },
            "review_session": {
                "history_namespace": "sessions/",
                "status": "not_started",
            },
            "review_workspace": "ready",
        },
        "media_inventory": "media_inventory.json",
        "latest_reviewed_publication": {
            "automatic_baseline_regions": relative_to_workspace(
                workspace_regions_path,
                workspace_dir,
            ),
            "publication_mode": "reviewed_region_compiler.v1",
            "render_source": "original",
            "schema_version": LATEST_REVIEWED_PUBLICATION_SCHEMA_VERSION,
            "status": "supported",
        },
        "non_final": True,
        "open_blockers": list(case_blockers),
        "review_entities": "review_entities.json",
        "review_tasks": "reviewer_tasks.json",
        "session_namespace": {
            "case_id": case_id,
            "history_namespace": "sessions/",
            "schema_version": "mixed_review_session_namespace.v1",
        },
        "schema_version": "final_review_workspace.v2.3.4",
        "selected_categories": list(selected_categories),
        "source_identity": context.source_identity.to_dict(),
        "source_metadata": {
            "duration_seconds": duration,
            "final_source_frame_index": source_frame_end,
            "first_source_frame_index": source_frame_start,
            "fps": fps,
            "frame_count": frame_count,
            "height": context.source_identity.height,
            "sha256": context.source_identity.sha256,
            "width": context.source_identity.width,
        },
        "source_revision": None,
        "workspace_status": "ready",
        "workspace_kind": "unified_case_review_workspace",
    }
    atomic_write_json(
        workspace_dir / "workspace_manifest.json",
        workspace_manifest,
        validator=validate_authoritative_workspace_manifest,
    )

    from redactor import review_workspace

    try:
        review_workspace.write_react_workspace_assets(workspace_dir, build_frontend=False)
    except FileNotFoundError:
        review_workspace.write_react_workspace_assets(workspace_dir, build_frontend=True)
    workspace_manifest.update(
        {
            "api_schema_version": review_api_contract.API_SCHEMA_VERSION,
            "frontend_build_manifest": "frontend_build_manifest.json",
            "frontend_milestone_version": review_workspace.FRONTEND_MILESTONE_VERSION,
            "legacy_fallback": "legacy.html",
            "react_app_root": "app/",
            "selected_frontend": "react",
        }
    )
    atomic_write_json(
        workspace_dir / "workspace_manifest.json",
        workspace_manifest,
        validator=validate_authoritative_workspace_manifest,
    )

    artifacts = (
        artifact_reference(
            case_root,
            workspace_dir / "workspace_manifest.json",
            "review_workspace_manifest",
        ),
        support_artifacts["entity_registry"],
        support_artifacts["review_tasks"],
        support_artifacts["media_inventory"],
        support_artifacts["automatic_baseline_regions"],
        support_artifacts["workspace_summary"],
        support_artifacts["session_state"],
        support_artifacts["validation"],
        artifact_reference(case_root, workspace_dir / "index.html", "mixed_review_workspace_index"),
    )
    return {
        "artifacts": artifacts,
        "manifest": workspace_manifest,
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "validation": validation,
        "workspace_dir": workspace_dir.relative_to(case_root).as_posix(),
    }


def prepare_workspace_dir(workspace_dir: Path, *, overwrite: bool) -> None:
    if workspace_dir.exists():
        if not overwrite and any(workspace_dir.iterdir()):
            raise FileExistsError(
                f"{workspace_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(workspace_dir)
    for relative in ("cases", "entities", "media", "regions", "sessions"):
        (workspace_dir / relative).mkdir(parents=True, exist_ok=True)


def install_media(source: Path, destination: Path, *, overwrite: bool) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"review workspace media source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists")
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def resolve_automatic_preview_source(
    context: Any,
    *,
    render_plan: dict[str, Any],
    combined_preview: dict[str, Any] | None,
) -> tuple[Path, str, dict[str, Any]]:
    case_root = Path(context.output_dir)
    if combined_preview is not None:
        manifest = dict(combined_preview.get("manifest") or {})
        output = manifest.get("output_video")
        if output:
            return case_root / str(output), "combined_automatic_preview", manifest
    if len(context.selected_categories) == 1:
        category = str(context.selected_categories[0])
        if category == CATEGORY_FACE:
            manifest_path = case_root / "automatic_preview" / "render_manifest.json"
            manifest = read_json_if_exists(manifest_path)
            return (
                case_root / "automatic_preview" / "redacted_preview.mp4",
                "face_automatic_preview",
                manifest,
            )
        if category == CATEGORY_PLATE:
            manifest_path = (
                case_root / CATEGORY_PLATE / "automatic_preview" / "render_manifest.json"
            )
            manifest = read_json_if_exists(manifest_path)
            return (
                case_root / CATEGORY_PLATE / "automatic_preview" / "automatic_preview.mp4",
                "plate_automatic_preview",
                manifest,
            )
    if int(render_plan.get("effective_region_count") or 0) == 0:
        return materialize_noop_automatic_preview(context, case_root)
    raise FileNotFoundError("combined automatic preview artifact is required for mixed review")


def materialize_noop_automatic_preview(
    context: Any,
    case_root: Path,
) -> tuple[Path, str, dict[str, Any]]:
    preview_dir = case_root / "automatic_preview_noop"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output = preview_dir / "automatic_preview.mp4"
    if output.exists():
        output.unlink()
    install_media(Path(context.source_video), output, overwrite=True)
    output_hash = media.sha256_file(output)
    manifest = {
        "automatic_preview_hash": output_hash,
        "frame_count": context.source_identity.frame_count,
        "full_source_span": True,
        "height": context.source_identity.height,
        "materialization_mode": "noop_source_copy",
        "non_final": True,
        "output_sha256": output_hash,
        "output_video": output.relative_to(case_root).as_posix(),
        "region_count": 0,
        "renderer_backend": "noop_source_copy",
        "schema_version": "noop_automatic_preview_manifest.v1",
        "source_frame_end": None
        if context.source_identity.frame_count is None
        else max(0, int(context.source_identity.frame_count) - 1),
        "source_frame_start": 0,
        "source_video_hash": context.source_identity.sha256,
        "width": context.source_identity.width,
    }
    atomic_write_json(preview_dir / "render_manifest.json", manifest)
    return output, "noop_source_copy", manifest


def validate_automatic_preview_source(
    context: Any,
    preview_source: Path,
    preview_manifest: dict[str, Any],
    *,
    preview_source_role: str,
) -> int:
    if not preview_source.exists():
        raise FileNotFoundError(f"automatic preview media is missing: {preview_source}")
    region_count = automatic_preview_region_count(
        preview_manifest,
        preview_source_role=preview_source_role,
    )
    declared_source_hash = preview_manifest.get("source_video_hash")
    if not declared_source_hash:
        raise ValueError("automatic preview source_video_hash is required")
    if str(declared_source_hash) != context.source_identity.sha256:
        raise ValueError(
            "automatic preview source hash mismatch: "
            f"{declared_source_hash} != {context.source_identity.sha256}"
        )
    declared_output_hash = first_present(
        preview_manifest,
        "output_video_hash",
        "output_sha256",
        "automatic_preview_hash",
    )
    if not declared_output_hash:
        raise ValueError("automatic preview output hash is required")
    actual = media.sha256_file(preview_source)
    if str(declared_output_hash) != actual:
        raise ValueError(
            "automatic preview output hash mismatch: "
            f"{declared_output_hash} != {actual}"
        )
    return region_count


def automatic_preview_region_count(
    preview_manifest: dict[str, Any],
    *,
    preview_source_role: str,
) -> int:
    if preview_source_role == "noop_source_copy":
        raw_count = preview_manifest.get("region_count", 0)
        count = parse_non_negative_int(raw_count, field_name="region_count")
        if count != 0:
            raise ValueError("no-op automatic preview region_count must be 0")
        return 0
    for field_name in ("region_count", "rendered_region_count"):
        if field_name in preview_manifest:
            return parse_non_negative_int(
                preview_manifest[field_name],
                field_name=field_name,
            )
    raise ValueError(
        "automatic preview region count is required for non-noop preview "
        "(expected region_count or rendered_region_count)"
    )


def parse_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if isinstance(value, int):
        count = value
    elif isinstance(value, str) and value.strip().isdigit():
        count = int(value.strip())
    else:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if count < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return count


def validate_media_span(context: Any, *, original_path: Path, automatic_path: Path) -> None:
    validate_media_pair(
        context,
        original_path=original_path,
        automatic_path=automatic_path,
        media_inventory=None,
    )


def validate_media_pair(
    context: Any,
    *,
    original_path: Path,
    automatic_path: Path,
    media_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_frame_count = context.source_identity.frame_count
    original_info = media.probe_media(original_path)
    automatic_info = media.probe_media(automatic_path)
    errors: list[str] = []
    expected_fps = context.source_identity.fps or case_manifests.FPS_RATIONAL_DEFAULT
    for role, info in (("original", original_info), ("automatic", automatic_info)):
        frames = int(info.video_stream.frame_count or 0)
        if expected_frame_count is not None and frames != int(expected_frame_count):
            errors.append(
                f"{role} frame-count mismatch: {frames} != {expected_frame_count}"
            )
        if int(info.video_stream.width) != int(context.source_identity.width):
            errors.append(f"{role} width mismatch")
        if int(info.video_stream.height) != int(context.source_identity.height):
            errors.append(f"{role} height mismatch")
        if not fps_rationals_equal(
            info.video_stream.nominal_frame_rate or case_manifests.FPS_RATIONAL_DEFAULT,
            expected_fps,
        ):
            errors.append(f"{role} fps mismatch")
    if original_path.resolve() == automatic_path.resolve():
        errors.append("original and automatic preview roles must be distinct files")
    span_end = None if expected_frame_count is None else max(0, int(expected_frame_count) - 1)
    if media_inventory is not None:
        for item in media_inventory.get("items") or []:
            if not item.get("sha256"):
                errors.append(f"{item.get('role')}: missing media hash")
            if item.get("case_source_frame_start") != 0:
                errors.append(f"{item.get('role')}: first source-frame mapping mismatch")
            if span_end is not None and item.get("case_source_frame_end") != span_end:
                errors.append(f"{item.get('role')}: final source-frame mapping mismatch")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "automatic": media_summary_for_probe(automatic_info, automatic_path),
        "frame_mapping": {
            "first_source_frame_index": 0,
            "final_source_frame_index": span_end,
            "intended_processing_span": [0, span_end],
        },
        "original": media_summary_for_probe(original_info, original_path),
        "passed": True,
        "schema_version": "mixed_review_media_pair_validation.v1",
    }


def build_media_inventory(
    context: Any,
    *,
    original_path: Path,
    automatic_path: Path,
    preview_source: Path,
    preview_source_role: str,
    workspace_dir: Path,
) -> dict[str, Any]:
    items = [
        media_inventory_item(
            context,
            path=original_path,
            role="original",
            source_role="source_video",
            installed_from=Path(context.source_video),
            workspace_dir=workspace_dir,
        ),
        media_inventory_item(
            context,
            path=automatic_path,
            role="automatic",
            source_role=preview_source_role,
            installed_from=preview_source,
            workspace_dir=workspace_dir,
        ),
    ]
    return {
        "case_id": context.case_id,
        "items": items,
        "schema_version": "mixed_review_media_inventory.v1",
        "source_identity_sha256": context.source_identity.sha256,
    }


def media_inventory_item(
    context: Any,
    *,
    path: Path,
    role: str,
    source_role: str,
    installed_from: Path,
    workspace_dir: Path,
) -> dict[str, Any]:
    info = media.probe_media(path)
    frame_count = int(info.video_stream.frame_count or 0)
    fps = info.video_stream.nominal_frame_rate or case_manifests.FPS_RATIONAL_DEFAULT
    return {
        "case_frame_count": context.source_identity.frame_count,
        "case_source_frame_end": None
        if context.source_identity.frame_count is None
        else max(0, int(context.source_identity.frame_count) - 1),
        "case_source_frame_start": 0,
        "final_source_frame_index": None
        if context.source_identity.frame_count is None
        else max(0, int(context.source_identity.frame_count) - 1),
        "first_source_frame_index": 0,
        "fps": fps,
        "frame_count": frame_count,
        "height": info.video_stream.height,
        "installed_from": installed_from.as_posix(),
        "intended_processing_span": [
            0,
            None
            if context.source_identity.frame_count is None
            else max(0, int(context.source_identity.frame_count) - 1),
        ],
        "path": relative_to_workspace(path, workspace_dir),
        "renderer_revision": None,
        "role": role,
        "schema_version": "mixed_review_media_item.v1",
        "sha256": media.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "source_case_id": context.case_id,
        "source_sha256": context.source_identity.sha256,
        "source_role": source_role,
        "width": info.video_stream.width,
    }


def media_summary_for_probe(info: media.MediaInfo, path: Path) -> dict[str, Any]:
    return {
        "frame_count": int(info.video_stream.frame_count or 0),
        "fps": info.video_stream.nominal_frame_rate or case_manifests.FPS_RATIONAL_DEFAULT,
        "height": info.video_stream.height,
        "path": path.as_posix(),
        "sha256": media.sha256_file(path),
        "width": info.video_stream.width,
    }


def fps_rationals_equal(left: str, right: str) -> bool:
    return abs((rational_to_float(left) or 0.0) - (rational_to_float(right) or 0.0)) < 0.001


def load_category_review_tasks(
    case_root: Path,
    selected_categories: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for category in selected_categories:
        if category == CATEGORY_FACE:
            path = case_root / "review" / "review_tasks.json"
            key = "reviewer_tasks"
        else:
            path = case_root / CATEGORY_PLATE / "review" / "review_tasks.json"
            key = "tasks"
        payload = read_json(path)
        raw_tasks = payload.get(key)
        if raw_tasks is None:
            if "tasks" in payload:
                raw_tasks = payload["tasks"]
            elif "reviewer_tasks" in payload:
                raw_tasks = payload["reviewer_tasks"]
        if not isinstance(raw_tasks, list):
            raise ValueError(f"{category} review tasks must be a list: {path}")
        tasks[category] = {
            "payload": payload,
            "source_path": path.relative_to(case_root).as_posix(),
            "tasks": [dict(task) for task in raw_tasks],
        }
    return tasks


def build_entity_sources(
    *,
    case_root: Path,
    case_id: str,
    selected_categories: tuple[str, ...],
    records: list[PrivacyRegionRecord],
    category_tasks: dict[str, dict[str, Any]],
    fps: str,
    source_frame_end: int,
) -> dict[tuple[str, str], ReviewEntitySource]:
    del fps
    sources: dict[tuple[str, str], ReviewEntitySource] = {}
    selected = set(selected_categories)
    for record in records:
        if record.category not in selected:
            raise ValueError(f"review entity category was not selected: {record.category}")
        source = ensure_entity_source(
            sources,
            case_id=case_id,
            category=record.category,
            source_key=entity_key_for_record(record),
            policy="redact",
        )
        source.render_region_count += 1
        source.observe_frame(record.source_frame)
        if record.bbox_xyxy is not None:
            source.bboxes.append(tuple(int(value) for value in record.bbox_xyxy))
        source.provenance_counts[record.provenance] += 1
        source.technical_track_ids.add(source.source_key)
        source.source_lineage.setdefault("render_region_sources", []).append(record.to_dict())

    if CATEGORY_FACE in selected:
        add_face_entity_sources(case_root, case_id=case_id, sources=sources)
    if CATEGORY_PLATE in selected:
        add_plate_entity_sources(case_root, case_id=case_id, sources=sources)

    for category, task_payload in category_tasks.items():
        for task in task_payload["tasks"]:
            start, end = task_frame_span(task, source_frame_end=source_frame_end)
            keys = task_entity_keys(category, task)
            missing_keys = [key for key in keys if (category, key) not in sources]
            fallback_keys: tuple[str, ...] = ()
            fallback_missing_keys: list[str] = []
            if (not keys or missing_keys) and task_allows_interval_fallback(category, task):
                fallback_keys = fallback_interval_task_entity_keys(
                    sources,
                    category=category,
                    frame_start=start,
                    frame_end=end,
                )
                fallback_missing_keys = [
                    key for key in fallback_keys if (category, key) not in sources
                ]
            binding = resolve_task_target_binding(
                case_id=case_id,
                category=category,
                task=task,
                frame_start=start,
                frame_end=end,
                entity_keys=keys,
                missing_entity_keys=missing_keys,
                fallback_entity_keys=fallback_keys,
                fallback_missing_entity_keys=fallback_missing_keys,
            )
            if binding.kind == TARGET_BINDING_SOURCE_SPAN:
                continue
            if binding.kind != TARGET_BINDING_ENTITY:
                raise ValueError(
                    f"unbound_review_task: {category}:{source_task_id(category, task)} "
                    "has no source-backed target identity"
                )
            for key in binding.entity_keys:
                source = sources[(category, key)]
                source.task_ids.add(source_task_id(category, task))
                source.observe_frame(start)
                source.observe_frame(end)
    return sources


def ensure_entity_source(
    sources: dict[tuple[str, str], ReviewEntitySource],
    *,
    case_id: str,
    category: str,
    source_key: str,
    policy: str,
) -> ReviewEntitySource:
    key = (category, source_key)
    if key not in sources:
        sources[key] = ReviewEntitySource(
            case_id=case_id,
            category=category,
            source_key=source_key,
            policy=policy,
            technical_track_ids={source_key},
        )
        return sources[key]
    source = sources[key]
    source.policy = merged_policy(source.policy, policy)
    return source


def merged_policy(existing: str, incoming: str) -> str:
    if existing == incoming:
        return existing
    if "keep_visible" in {existing, incoming}:
        raise ValueError(
            f"conflicting source-backed entity policies: {existing} vs {incoming}"
        )
    if "redact" in {existing, incoming}:
        return "redact"
    return "review_required"


def add_face_entity_sources(
    case_root: Path,
    *,
    case_id: str,
    sources: dict[tuple[str, str], ReviewEntitySource],
) -> None:
    census_path = case_root / "track_census.json"
    if not census_path.exists():
        return
    census = read_json(census_path)
    for track in list(census.get("tracks") or []):
        if not isinstance(track, dict):
            continue
        policy_assignment = str(track.get("policy_assignment") or "review")
        policy = FACE_POLICY_TO_ENTITY_POLICY.get(policy_assignment)
        quality_status = str(track.get("quality_status") or "")
        if policy is None or policy_assignment == "ignore" or quality_status == "rejected":
            continue
        track_id = str(track.get("track_id") or "")
        if not track_id:
            continue
        source = ensure_entity_source(
            sources,
            case_id=case_id,
            category=CATEGORY_FACE,
            source_key=face_track_key(track_id),
            policy=policy,
        )
        first = optional_int(track.get("first_frame"))
        last = optional_int(track.get("last_frame"))
        source.observe_frame(first)
        source.observe_frame(last)
        source.representative_frame = optional_int(track.get("representative_frame")) or first
        source.technical_detector_ids.add(track_id)
        source.technical_track_ids.add(face_track_key(track_id))
        source.evidence_disposition = (
            "keep_visible"
            if policy == "keep_visible"
            else "review_required"
            if policy == "review_required"
            else "accepted"
        )
        source.source_lineage.update(
            {
                "assignment_provenance": track.get("assignment_provenance"),
                "policy_assignment": policy_assignment,
                "quality_status": quality_status,
                "source_artifact": "track_census.json",
                "technical_track_id": track_id,
            }
        )


def add_plate_entity_sources(
    case_root: Path,
    *,
    case_id: str,
    sources: dict[tuple[str, str], ReviewEntitySource],
) -> None:
    tracking_dir = case_root / CATEGORY_PLATE / "tracking"
    summary_path = tracking_dir / "track_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        for track in list(summary.get("tracks") or []):
            if not isinstance(track, dict):
                continue
            status = str(track.get("status") or "")
            policy = PLATE_ACTIVE_STATUSES.get(status)
            if policy is None:
                continue
            track_uid = str(track.get("track_uid") or "")
            if not track_uid:
                continue
            source = ensure_entity_source(
                sources,
                case_id=case_id,
                category=CATEGORY_PLATE,
                source_key=track_uid,
                policy=policy,
            )
            first = optional_int(track.get("first_frame"))
            last = optional_int(track.get("last_frame"))
            source.observe_frame(first)
            source.observe_frame(last)
            source.representative_frame = first
            source.technical_detector_ids.add(str(track.get("track_id") or track_uid))
            source.technical_track_ids.add(track_uid)
            source.evidence_disposition = (
                "accepted" if policy == "redact" else "review_required"
            )
            if track.get("stitch_parent_id") or track.get("stitch_child_ids"):
                source.merge_lineage.append(
                    {
                        "stitch_child_ids": list(track.get("stitch_child_ids") or []),
                        "stitch_parent_id": track.get("stitch_parent_id"),
                    }
                )
            source.source_lineage.update(
                {
                    "review_reasons": list(track.get("review_reasons") or []),
                    "source_artifact": "license_plate/tracking/track_summary.json",
                    "status": status,
                }
            )
    tracks_path = tracking_dir / "tracks.jsonl"
    for row in read_jsonl(tracks_path):
        frame = optional_int(row.get("frame_index"))
        for track in list(row.get("tracks") or []):
            if not isinstance(track, dict):
                continue
            track_uid = str(track.get("track_uid") or "")
            source = sources.get((CATEGORY_PLATE, track_uid))
            if source is None:
                continue
            source.observe_frame(frame)
            bbox = track.get("bbox_xyxy")
            if isinstance(bbox, list | tuple) and len(bbox) == 4:
                source.bboxes.append(tuple(int(value) for value in bbox))
            source.provenance_counts[str(track.get("provenance") or "tracking")] += 1
    intervals_path = tracking_dir / "review_intervals.json"
    if intervals_path.exists():
        for interval in list(read_json(intervals_path) or []):
            if not isinstance(interval, dict):
                continue
            for track_uid in list(interval.get("track_uids") or []):
                source = ensure_entity_source(
                    sources,
                    case_id=case_id,
                    category=CATEGORY_PLATE,
                    source_key=str(track_uid),
                    policy="review_required",
                )
                source.observe_frame(optional_int(interval.get("frame_start")))
                source.observe_frame(optional_int(interval.get("frame_end")))
                source.source_lineage.setdefault("review_intervals", []).append(interval)


def build_entities_from_sources(
    *,
    case_id: str,
    selected_categories: tuple[str, ...],
    sources: dict[tuple[str, str], ReviewEntitySource],
    category_tasks: dict[str, dict[str, Any]],
    fps: str,
    source_frame_end: int,
) -> dict[str, Any]:
    del category_tasks, source_frame_end
    selected = set(selected_categories)
    entities = [
        entity_from_source(source, fps=fps)
        for source in sources.values()
        if source.category in selected
    ]
    seen_ids: set[str] = set()
    for entity in entities:
        entity_id = str(entity["entity_id"])
        if entity_id in seen_ids:
            raise ValueError(f"duplicate active review entity id: {entity_id}")
        seen_ids.add(entity_id)
    builders = {
        (source.category, source.source_key): {
            "entity_id": deterministic_entity_id(
                case_id=case_id,
                category=source.category,
                key=source.source_key,
            ),
            "first_frame": source.first_frame,
            "last_frame": source.last_frame,
        }
        for source in sources.values()
    }
    return {"entities": entities, "entities_by_key": builders}


def build_entities(
    *,
    case_id: str,
    selected_categories: tuple[str, ...],
    records: list[PrivacyRegionRecord],
    category_tasks: dict[str, dict[str, Any]],
    fps: str,
) -> dict[str, Any]:
    sources = build_entity_sources(
        case_root=Path("."),
        case_id=case_id,
        selected_categories=selected_categories,
        records=records,
        category_tasks=category_tasks,
        fps=fps,
        source_frame_end=10**12,
    )
    return build_entities_from_sources(
        case_id=case_id,
        selected_categories=selected_categories,
        sources=sources,
        category_tasks=category_tasks,
        fps=fps,
        source_frame_end=10**12,
    )


def ensure_entity_builder(
    builders: dict[tuple[str, str], dict[str, Any]],
    case_id: str,
    category: str,
    key: str,
) -> dict[str, Any]:
    builder_key = (category, key)
    if builder_key not in builders:
        entity_hash = stable_hash(
            {"case_id": case_id, "category": category, "key": key}
        )[:16]
        entity_id = f"{ENTITY_ID_PREFIX[category]}_{entity_hash}"
        builders[builder_key] = {
            "bboxes": [],
            "case_id": case_id,
            "category": category,
            "entity_id": entity_id,
            "frames": [],
            "key": key,
            "provenance_counts": Counter(),
            "records": [],
            "task_ids": [],
        }
    return builders[builder_key]


def entity_from_builder(builder: dict[str, Any], *, fps: str) -> dict[str, Any]:
    frames = [int(frame) for frame in builder["frames"]]
    first = min(frames) if frames else None
    last = max(frames) if frames else None
    representative = frames[len(frames) // 2] if frames else None
    bbox = median_bbox(builder["bboxes"])
    region_count = len(builder["records"])
    task_count = len(set(builder["task_ids"]))
    policy = "redact" if region_count else "review_required"
    fps_value = rational_to_float(fps) or 30.0
    return {
        "attached_fragment_track_ids": [],
        "automatic_source_track_ids": [builder["key"]],
        "case_id": builder["case_id"],
        "category": builder["category"],
        "configuration_lineage": {
            "schema_version": "mixed_review_entity_configuration_lineage.v1"
        },
        "created_provenance": {
            "builder": SCHEMA_VERSION,
            "source": "mixed_review_workspace_builder",
        },
        "current_policy": policy,
        "current_status": "active",
        "default_privacy_policy": policy,
        "display_name": "",
        "entity_id": builder["entity_id"],
        "evidence_summary": {
            "provenance_counts": dict(sorted(builder["provenance_counts"].items())),
            "region_count": region_count,
            "review_observation_count": region_count,
            "review_task_count": task_count,
            "review_tracklet_count": 1,
            "tracklet_count": 1,
        },
        "first_visible_source_frame": first,
        "first_visible_timestamp_seconds": frame_seconds(first, fps_value),
        "label": "",
        "last_visible_source_frame": last,
        "last_visible_timestamp_seconds": frame_seconds(last, fps_value),
        "member_detector_tracker_ids": [builder["key"]],
        "possible_duplicate": False,
        "possible_duplicate_suggestions": [],
        "privacy_action": policy,
        "representative_bbox": list(bbox) if bbox is not None else None,
        "representative_source_frame": representative,
        "representative_thumbnail_path": None,
        "review_entity_state": "ready" if region_count else "review_required",
        "schema_version": "review_entity.v2",
        "source_lineage": {
            "schema_version": "mixed_review_entity_source_lineage.v1",
            "source_task_ids": sorted(set(builder["task_ids"])),
            "technical_id": builder["key"],
        },
    }


def entity_from_source(source: ReviewEntitySource, *, fps: str) -> dict[str, Any]:
    fps_value = rational_to_float(fps) or 30.0
    bbox = median_bbox(source.bboxes)
    observation_count = max(
        int(source.render_region_count),
        len(source.bboxes),
        sum(int(value) for value in source.provenance_counts.values()),
    )
    entity_id = deterministic_entity_id(
        case_id=source.case_id,
        category=source.category,
        key=source.source_key,
    )
    first = source.first_frame
    last = source.last_frame
    representative = source.representative_frame
    if representative is None:
        representative = first
    task_ids = sorted(source.task_ids)
    return {
        "attached_fragment_track_ids": [],
        "automatic_source_track_ids": sorted(source.technical_track_ids),
        "case_id": source.case_id,
        "category": source.category,
        "configuration_lineage": {
            "schema_version": "mixed_review_entity_configuration_lineage.v1"
        },
        "created_provenance": {
            "builder": SCHEMA_VERSION,
            "source": "source_backed_mixed_review_workspace_builder",
        },
        "current_policy": source.policy,
        "current_status": "active",
        "default_privacy_policy": source.policy,
        "display_name": "",
        "entity_id": entity_id,
        "evidence_disposition": source.evidence_disposition,
        "evidence_summary": {
            "provenance_counts": dict(sorted(source.provenance_counts.items())),
            "region_count": source.render_region_count,
            "review_observation_count": observation_count,
            "review_task_count": len(task_ids),
            "review_tracklet_count": 1,
            "tracklet_count": 1,
        },
        "first_visible_source_frame": first,
        "first_visible_timestamp_seconds": frame_seconds(first, fps_value),
        "label": "",
        "last_visible_source_frame": last,
        "last_visible_timestamp_seconds": frame_seconds(last, fps_value),
        "member_detector_tracker_ids": sorted(source.technical_track_ids),
        "merge_split_lineage": list(source.merge_lineage),
        "possible_duplicate": False,
        "possible_duplicate_suggestions": [],
        "privacy_action": source.policy,
        "representative_bbox": list(bbox) if bbox is not None else None,
        "representative_source_frame": representative,
        "representative_thumbnail_path": None,
        "review_entity_state": "ready"
        if source.policy == "redact"
        else "review_required"
        if source.policy == "review_required"
        else "kept_visible",
        "schema_version": "review_entity.v2",
        "source_lineage": {
            **source.source_lineage,
            "schema_version": "mixed_review_entity_source_lineage.v1",
            "source_task_ids": task_ids,
            "technical_detector_ids": sorted(source.technical_detector_ids),
            "technical_id": source.source_key,
            "technical_track_ids": sorted(source.technical_track_ids),
        },
    }


def deterministic_entity_id(*, case_id: str, category: str, key: str) -> str:
    entity_hash = stable_hash({"case_id": case_id, "category": category, "key": key})[:16]
    return f"{ENTITY_ID_PREFIX[category]}_{entity_hash}"


def assign_entity_labels(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = defaultdict(int)
    output: list[dict[str, Any]] = []
    for entity in sorted(
        entities,
        key=lambda item: (
            CATEGORY_ORDER[str(item["category"])],
            item.get("first_visible_source_frame")
            if item.get("first_visible_source_frame") is not None
            else 10**12,
            str((item.get("automatic_source_track_ids") or [""])[0]),
            item["entity_id"],
        ),
    ):
        category = str(entity["category"])
        counters[category] += 1
        label = f"{CATEGORY_LABEL_PREFIX[category]} {counters[category]:02d}"
        output.append(entity | {"display_name": label, "label": label})
    return output


def write_source_backed_entity_thumbnails(
    *,
    workspace_dir: Path,
    case_id: str,
    entities: list[dict[str, Any]],
    source_path: Path,
    source_sha256: str,
    source_frame_start: int,
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not source_path.exists():
        for entity in entities:
            entity["thumbnail_unavailable_reason"] = "source_media_missing"
            rows.append(thumbnail_audit_row(entity, status="missing_source_media"))
        return {
            "cases": {case_id: rows},
            "passed": False,
            "schema_version": "representative_thumbnail_audit.v1",
        }

    capture = cv.VideoCapture(str(source_path))
    try:
        for entity in entities:
            row = write_one_source_backed_entity_thumbnail(
                workspace_dir=workspace_dir,
                case_id=case_id,
                entity=entity,
                capture=capture,
                source_sha256=source_sha256,
                source_frame_start=source_frame_start,
                width=width,
                height=height,
            )
            rows.append(row)
    finally:
        capture.release()

    return {
        "cases": {case_id: rows},
        "passed": not any(row["status"] == "failed" for row in rows),
        "schema_version": "representative_thumbnail_audit.v1",
    }


def write_one_source_backed_entity_thumbnail(
    *,
    workspace_dir: Path,
    case_id: str,
    entity: dict[str, Any],
    capture: cv.VideoCapture,
    source_sha256: str,
    source_frame_start: int,
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    bbox = parse_bbox(entity.get("representative_bbox"))
    source_frame = optional_int(entity.get("representative_source_frame"))
    entity_id = str(entity.get("entity_id") or "")
    category = str(entity.get("category") or "")
    if bbox is None or source_frame is None or not entity_id:
        entity["thumbnail_unavailable_reason"] = "representative_geometry_missing"
        return thumbnail_audit_row(entity, status="no_geometry")

    frame_index = source_frame - int(source_frame_start or 0)
    if frame_index < 0:
        entity["thumbnail_unavailable_reason"] = "representative_frame_outside_source"
        return thumbnail_audit_row(entity, status="failed")
    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        entity["thumbnail_unavailable_reason"] = "source_frame_unreadable"
        return thumbnail_audit_row(entity, status="failed")

    frame_height, frame_width = frame.shape[:2]
    if width is not None and int(width) != frame_width:
        entity["thumbnail_unavailable_reason"] = "source_width_mismatch"
        return thumbnail_audit_row(entity, status="failed")
    if height is not None and int(height) != frame_height:
        entity["thumbnail_unavailable_reason"] = "source_height_mismatch"
        return thumbnail_audit_row(entity, status="failed")

    clamped = clamp_bbox(bbox, frame_width=frame_width, frame_height=frame_height)
    if clamped is None:
        entity["thumbnail_unavailable_reason"] = "representative_bbox_out_of_bounds"
        return thumbnail_audit_row(entity, status="no_geometry")

    x1, y1, x2, y2 = clamped
    pad_fraction = 0.42 if category == CATEGORY_FACE else 0.55
    pad = max(10, int(max(x2 - x1, y2 - y1) * pad_fraction))
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(frame_width, x2 + pad)
    bottom = min(frame_height, y2 + pad)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        entity["thumbnail_unavailable_reason"] = "empty_source_crop"
        return thumbnail_audit_row(entity, status="no_geometry")

    target_height = 120
    scale = target_height / max(1, crop.shape[0])
    resized = cv.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), target_height),
        interpolation=cv.INTER_AREA,
    )
    out_path = workspace_dir / "entities" / case_id / "thumbnails" / f"{entity_id}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(out_path), resized, [int(cv.IMWRITE_JPEG_QUALITY), 88]):
        entity["thumbnail_unavailable_reason"] = "thumbnail_write_failed"
        return thumbnail_audit_row(entity, status="failed")

    relative = relative_to_workspace(out_path, workspace_dir)
    identity = {
        "bbox_xyxy": list(clamped),
        "category": category,
        "entity_id": entity_id,
        "path": relative,
        "schema_version": "source_backed_entity_thumbnail_identity.v1",
        "source_frame": source_frame,
        "source_frame_index": frame_index,
        "source_sha256": source_sha256,
        "thumbnail_sha256": media.sha256_file(out_path),
    }
    entity["representative_bbox"] = list(clamped)
    entity["representative_thumbnail_identity"] = identity
    entity["representative_thumbnail_path"] = relative
    entity.pop("thumbnail_unavailable_reason", None)
    return thumbnail_audit_row(entity, status="source_backed")


def thumbnail_audit_row(entity: dict[str, Any], *, status: str) -> dict[str, Any]:
    identity = dict(entity.get("representative_thumbnail_identity") or {})
    return {
        "bbox_xyxy": identity.get("bbox_xyxy") or entity.get("representative_bbox"),
        "category": entity.get("category"),
        "entity_id": entity.get("entity_id"),
        "label": entity.get("label") or entity.get("display_name"),
        "reason": entity.get("thumbnail_unavailable_reason"),
        "representative_source_frame": entity.get("representative_source_frame"),
        "source_sha256": identity.get("source_sha256"),
        "status": status,
        "thumbnail_path": entity.get("representative_thumbnail_path"),
        "thumbnail_sha256": identity.get("thumbnail_sha256"),
    }


def parse_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    left = max(0, min(frame_width, min(x1, x2)))
    right = max(0, min(frame_width, max(x1, x2)))
    top = max(0, min(frame_height, min(y1, y2)))
    bottom = max(0, min(frame_height, max(y1, y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_tasks(
    *,
    case_id: str,
    selected_categories: tuple[str, ...],
    category_tasks: dict[str, dict[str, Any]],
    entities_by_key: dict[tuple[str, str], dict[str, Any]],
    source_frame_end: int,
    fps: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for category in selected_categories:
        for source_task in category_tasks[category]["tasks"]:
            start, end = task_frame_span(source_task, source_frame_end=source_frame_end)
            validate_task_timestamps(source_task, start=start, end=end, fps=fps)
            keys = task_entity_keys(category, source_task)
            missing_keys = [key for key in keys if (category, key) not in entities_by_key]
            fallback_keys: tuple[str, ...] = ()
            fallback_missing_keys: list[str] = []
            if (not keys or missing_keys) and task_allows_interval_fallback(
                category,
                source_task,
            ):
                fallback_keys = fallback_task_entity_keys(
                    entities_by_key,
                    category=category,
                    frame_start=start,
                    frame_end=end,
                )
                fallback_missing_keys = [
                    key for key in fallback_keys if (category, key) not in entities_by_key
                ]
            binding = resolve_task_target_binding(
                case_id=case_id,
                category=category,
                task=source_task,
                frame_start=start,
                frame_end=end,
                entity_keys=keys,
                missing_entity_keys=missing_keys,
                fallback_entity_keys=fallback_keys,
                fallback_missing_entity_keys=fallback_missing_keys,
            )
            if binding.kind not in {TARGET_BINDING_ENTITY, TARGET_BINDING_SOURCE_SPAN}:
                raise ValueError(
                    f"unbound_review_task: {category}:{source_task_id(category, source_task)} "
                    "has no source-backed target identity"
                )
            keys = binding.entity_keys
            entity_ids = [
                entities_by_key[(category, key)]["entity_id"]
                for key in keys
                if (category, key) in entities_by_key
            ]
            if binding.kind == TARGET_BINDING_ENTITY and len(entity_ids) != len(keys):
                raise ValueError(
                    f"{category} task has no review entity: {source_task_id(category, source_task)}"
                )
            task_id = f"{category}_{source_task_id(category, source_task)}"
            issue_type = task_issue_type(category, source_task)
            blocking = task_blocking(category, source_task)
            allowed_actions = task_allowed_actions(category, source_task, issue_type)
            task_payload = {
                "allowed_actions": allowed_actions,
                "blocking": blocking,
                "case_id": case_id,
                "category": category,
                "category_allowed_actions": [
                    ACTION_LABELS[action] for action in allowed_actions
                ],
                "category_review_source": category_tasks[category]["source_path"],
                "correction_required": blocking or issue_type in {"possible_privacy_gap"},
                "decision_state": "pending",
                "issue_type": issue_type,
                "media_roles": {
                    "automatic": f"media/{case_id}/automatic_preview.mp4",
                    "original": f"media/{case_id}/original.mp4",
                },
                "normal_task_class": issue_type,
                "optional": not blocking,
                "raw_signal_ids": raw_signal_ids(source_task),
                "review_task_id": task_id,
                "schema_version": TASK_SCHEMA_VERSION,
                "severity": task_severity(source_task, blocking),
                "source_frame_end": end,
                "source_frame_start": start,
                "source_task": source_task,
                "source_task_id": source_task_id(category, source_task),
                "target_binding": binding.payload,
                "technical_only": bool(source_task.get("technical_only", False)),
                "title": task_title(category, source_task, issue_type),
                "what_happened": task_what_happened(category, source_task, issue_type),
                "what_to_inspect": task_what_to_inspect(category, source_task, issue_type),
            }
            if binding.kind == TARGET_BINDING_ENTITY:
                task_payload.update(
                    {
                        "entity_id": entity_ids[0],
                        "target_entity_ids": entity_ids,
                        "technical_track_ids": list(keys),
                    }
                )
            else:
                task_payload.update(
                    {
                        "entity_id": None,
                        "target_entity_ids": [],
                        "technical_track_ids": [],
                    }
                )
            tasks.append(task_payload)
    seen: set[str] = set()
    seen_semantic: set[tuple[Any, ...]] = set()
    for task in tasks:
        task_id = str(task["review_task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate mixed review task id: {task_id}")
        seen.add(task_id)
        semantic = (
            task.get("case_id"),
            task.get("category"),
            tuple(task.get("target_entity_ids") or []),
            task.get("source_frame_start"),
            task.get("source_frame_end"),
            task.get("issue_type"),
        )
        if semantic in seen_semantic:
            raise ValueError(f"duplicate mixed review task semantic identity: {task_id}")
        seen_semantic.add(semantic)
    return sorted(
        tasks,
        key=lambda item: (
            int(item["source_frame_start"]),
            CATEGORY_ORDER[str(item["category"])],
            int(item["source_frame_end"]),
            str(item["review_task_id"]),
        ),
    )


def validate_workspace_payload(
    *,
    case_id: str,
    selected_categories: tuple[str, ...],
    entities: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    media_inventory: dict[str, Any],
    workspace_dir: Path,
    source_frame_end: int,
) -> None:
    errors: list[str] = []
    selected = set(selected_categories)
    entity_ids: dict[str, str] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "")
        category = str(entity.get("category") or "")
        if not entity_id:
            errors.append("review entity missing entity_id")
        elif entity_id in entity_ids:
            errors.append(f"duplicate review entity id: {entity_id}")
        entity_ids[entity_id] = category
        if entity.get("case_id") != case_id:
            errors.append(f"{entity_id}: wrong case_id")
        if category not in selected:
            errors.append(f"{entity_id}: unselected category {category}")
    task_ids: set[str] = set()
    for task in tasks:
        task_id = str(task.get("review_task_id") or "")
        category = str(task.get("category") or "")
        raw_entity_id = task.get("entity_id")
        entity_id = str(raw_entity_id) if raw_entity_id not in {None, ""} else ""
        target_binding = dict(task.get("target_binding") or {})
        binding_kind = str(target_binding.get("kind") or "")
        if task_id in task_ids:
            errors.append(f"duplicate review task id: {task_id}")
        task_ids.add(task_id)
        if task.get("case_id") != case_id:
            errors.append(f"{task_id}: wrong case_id")
        if category not in selected:
            errors.append(f"{task_id}: unselected category {category}")
        if binding_kind == TARGET_BINDING_ENTITY:
            if entity_id not in entity_ids:
                errors.append(f"{task_id}: unknown entity_id {entity_id}")
            elif entity_ids[entity_id] != category:
                errors.append(f"{task_id}: entity category mismatch")
            target_entity_ids = [str(item) for item in list(task.get("target_entity_ids") or [])]
            if not target_entity_ids:
                errors.append(f"{task_id}: entity binding missing target_entity_ids")
            for target_entity_id in target_entity_ids:
                if target_entity_id not in entity_ids:
                    errors.append(f"{task_id}: unknown target_entity_id {target_entity_id}")
                elif entity_ids[target_entity_id] != category:
                    errors.append(f"{task_id}: target entity category mismatch")
        elif binding_kind == TARGET_BINDING_SOURCE_SPAN:
            if entity_id:
                errors.append(f"{task_id}: source-span task must not reference entity_id")
            if task.get("target_entity_ids"):
                errors.append(f"{task_id}: source-span task must not reference target entities")
            if target_binding.get("case_id") != case_id:
                errors.append(f"{task_id}: source-span binding case mismatch")
            if target_binding.get("category") != category:
                errors.append(f"{task_id}: source-span binding category mismatch")
            if target_binding.get("source_task_id") != task.get("source_task_id"):
                errors.append(f"{task_id}: source-span binding task identity mismatch")
            if target_binding.get("source_frame_start") != task.get("source_frame_start"):
                errors.append(f"{task_id}: source-span binding start mismatch")
            if target_binding.get("source_frame_end") != task.get("source_frame_end"):
                errors.append(f"{task_id}: source-span binding end mismatch")
            if not target_binding.get("event_ids"):
                errors.append(f"{task_id}: source-span binding missing event identity")
        else:
            errors.append(f"{task_id}: missing target binding")
        if int(task.get("source_frame_end") or 0) < int(task.get("source_frame_start") or 0):
            errors.append(f"{task_id}: invalid frame span")
        if int(task.get("source_frame_start") or 0) < 0:
            errors.append(f"{task_id}: negative frame span")
        if int(task.get("source_frame_end") or 0) > source_frame_end:
            errors.append(f"{task_id}: outside source frame span")
        allowed_actions = list(task.get("allowed_actions") or [])
        if not allowed_actions:
            errors.append(f"{task_id}: missing allowed actions")
        unsupported = sorted(
            set(str(action) for action in allowed_actions) - review_sessions.DECISION_ACTIONS
        )
        if unsupported:
            errors.append(f"{task_id}: unsupported actions {unsupported}")
    for item in media_inventory.get("items") or []:
        path = workspace_dir / str(item.get("path") or "")
        if not item.get("sha256"):
            errors.append(f"media inventory missing hash: {item.get('path')}")
        if not path.exists():
            errors.append(f"media inventory missing file: {item.get('path')}")
        elif media.sha256_file(path) != item.get("sha256"):
            errors.append(f"media inventory hash mismatch: {item.get('path')}")
        if item.get("case_source_frame_start") != 0:
            errors.append(f"media inventory source start mismatch: {item.get('path')}")
        if item.get("case_source_frame_end") != source_frame_end:
            errors.append(f"media inventory source end mismatch: {item.get('path')}")
    if errors:
        raise ValueError("; ".join(errors))


def workspace_validation_report(
    *,
    context: Any,
    selected_categories: tuple[str, ...],
    entities: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    media_inventory: dict[str, Any],
    preview_manifest: dict[str, Any],
    automatic_region_count: int,
    media_validation: dict[str, Any],
    category_run_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case_id": context.case_id,
        "checks": {
            "category_run_manifest_identities_bound": True,
            "declared_preview_hash_checked": True,
            "declared_source_hash_checked": True,
            "entity_ids_unique": True,
            "media_frame_mapping_valid": media_validation.get("passed") is True,
            "media_inventory_hashes_current": True,
            "selected_categories": list(selected_categories),
            "source_preview_dimensions_fps_span_valid": media_validation.get("passed") is True,
            "task_ids_unique": True,
            "tasks_reference_valid_targets": True,
            "tasks_reference_valid_entities": True,
        },
        "entity_count": len(entities),
        "media_item_count": len(media_inventory.get("items") or []),
        "passed": True,
        "category_run_manifest_identity_hash": stable_hash(category_run_identities),
        "category_run_manifest_identity_count": len(category_run_identities),
        "automatic_preview_region_count": automatic_region_count,
        "preview_output_hash": first_present(
            preview_manifest,
            "output_video_hash",
            "output_sha256",
            "automatic_preview_hash",
        ),
        "preview_source_hash": preview_manifest.get("source_video_hash"),
        "schema_version": "mixed_review_workspace_validation.v1",
        "source_span_task_count": sum(
            1
            for task in tasks
            if dict(task.get("target_binding") or {}).get("kind") == TARGET_BINDING_SOURCE_SPAN
        ),
        "task_count": len(tasks),
    }


def write_entity_registry(
    workspace_dir: Path,
    case_id: str,
    entities: list[dict[str, Any]],
) -> None:
    unresolved = {
        "fragment_count": 0,
        "fragments": [],
        "schema_version": "review_unresolved_evidence.v1",
    }
    case_payload = {
        "case_id": case_id,
        "entities": entities,
        "entity_count": len(entities),
        "schema_version": "review_entities.v2",
        "unresolved_evidence": unresolved,
    }
    write_json(workspace_dir / "entities" / f"{case_id}.json", case_payload)
    write_json(
        workspace_dir / "entities" / f"{case_id}_unresolved_evidence.json",
        unresolved,
    )
    index = {
        "cases": {case_id: entities},
        "schema_version": ENTITY_REGISTRY_SCHEMA_VERSION,
        "unresolved_evidence_by_case": {case_id: unresolved},
    }
    write_json(workspace_dir / "review_entities.json", index)
    write_json(workspace_dir / "privacy_entities.json", index)
    write_json(workspace_dir / "review_objects.json", index)


def build_configuration_freeze(
    context: Any,
    *,
    category_results: list[Any],
    render_plan: dict[str, Any],
    preview_source_role: str,
    preview_manifest: dict[str, Any],
    automatic_region_count: int,
    media_inventory: dict[str, Any],
    category_run_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case_id": context.case_id,
        "category_configuration_identities": {
            result.category: result.configuration_identity for result in category_results
        },
        "category_run_manifest_identity_hash": stable_hash(category_run_identities),
        "category_run_manifest_identities": category_run_identities,
        "created_by": SCHEMA_VERSION,
        "media_hashes": {
            item["role"]: item["sha256"] for item in media_inventory.get("items") or []
        },
        "automatic_preview_region_count": automatic_region_count,
        "preview_manifest_hash": stable_hash(preview_manifest) if preview_manifest else None,
        "preview_source_role": preview_source_role,
        "render_plan_hash": render_plan.get("render_plan_hash"),
        "schema_version": "mixed_review_configuration_freeze.v1",
        "selected_categories": list(context.selected_categories),
        "source_identity": context.source_identity.to_dict(),
    }


def category_evidence_inputs(category_results: list[Any], category: str) -> tuple[str, ...]:
    values: list[str] = []
    for result in category_results:
        if result.category != category:
            continue
        for artifact in result.artifacts:
            if artifact.category == category:
                values.append(artifact.path)
    return tuple(sorted(set(values)))


def version_alignment_audit(context: Any, manifest: case_manifests.CaseManifest) -> dict[str, Any]:
    roles = [
        {
            "errors": [],
            "path": role.path,
            "role": role.role,
            "source_frame_end": role.source_frame_end,
            "source_frame_start": role.source_frame_start,
        }
        for role in manifest.video_roles
    ]
    return {
        "cases": {
            context.case_id: {
                "case_frame_count": manifest.frame_count,
                "display_name": manifest.display_name,
                "errors": [],
                "fps": manifest.fps,
                "height": manifest.height,
                "roles": roles,
                "schema_version": "case_version_alignment.v1",
                "source_frame_end": manifest.source_frame_end,
                "source_frame_start": manifest.source_frame_start,
                "source_video_hash": manifest.source_video_hash,
                "width": manifest.width,
            }
        },
        "created_at_utc": media.utc_timestamp(),
        "errors": [],
        "passed": True,
        "schema_version": "version_alignment_audit.v1",
    }


def workspace_summary(
    *,
    context: Any,
    entities: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    case_blockers: tuple[str, ...],
    workspace_dir: Path,
    validation: dict[str, Any],
    automatic_region_count: int,
) -> dict[str, Any]:
    category_counts = Counter(str(entity["category"]) for entity in entities)
    task_counts = Counter(str(task["category"]) for task in tasks)
    return {
        "case_id": context.case_id,
        "entity_count": len(entities),
        "entity_counts_by_category": dict(sorted(category_counts.items())),
        "lifecycle": {
            "finalization": "not_run",
            "latest_reviewed": "not_run",
            "review_session": "not_started",
            "review_workspace": "ready",
        },
        "schema_version": "mixed_review_workspace_summary.v1",
        "task_count": len(tasks),
        "task_counts_by_category": dict(sorted(task_counts.items())),
        "automatic_preview_region_count": automatic_region_count,
        "validation_passed": validation.get("passed") is True,
        "workspace": workspace_dir.as_posix(),
        "workspace_blockers": list(case_blockers),
    }


def category_run_manifest_identities(category_results: list[Any]) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for result in category_results:
        for artifact in result.artifacts:
            if artifact.role == "category_run_manifest":
                identities[result.category] = artifact.to_dict()
                break
    return identities


def validated_category_run_manifest_identities(
    context: Any,
    *,
    category_results: list[Any],
    selected_categories: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    case_root = Path(context.output_dir)
    selected = set(selected_categories)
    result_by_category: dict[str, Any] = {}
    artifact_ids: set[str] = set()
    artifact_paths: set[str] = set()
    identities: dict[str, dict[str, Any]] = {}
    for result in category_results:
        category = str(result.category)
        if category not in selected:
            raise ValueError(f"unselected category run identity found: {category}")
        if category in result_by_category:
            raise ValueError(f"duplicate category run result: {category}")
        result_by_category[category] = result
        artifact = next(
            (
                artifact
                for artifact in result.artifacts
                if artifact.role == "category_run_manifest"
            ),
            None,
        )
        if artifact is None:
            continue
        artifact_payload = artifact.to_dict()
        artifact_id = str(artifact_payload.get("artifact_id") or "")
        if artifact_id in artifact_ids:
            raise ValueError(f"duplicate category run artifact identity: {artifact_id}")
        artifact_ids.add(artifact_id)
        artifact_paths.add(str(artifact_payload.get("path") or ""))
        run_manifest_path = contained_case_artifact_path(case_root, artifact.path)
        if not run_manifest_path.exists():
            raise FileNotFoundError(f"{category}: category run manifest is missing")
        actual_hash = media.sha256_file(run_manifest_path)
        if actual_hash != artifact.sha256:
            raise ValueError(
                f"{category}: category run manifest hash mismatch: "
                f"{actual_hash} != {artifact.sha256}"
            )
        run_manifest = read_json(run_manifest_path)
        validate_category_run_manifest_payload(
            context,
            category=category,
            result=result,
            run_manifest=run_manifest,
        )
        identities[category] = {
            **artifact_payload,
            "artifact_reference_schema_version": artifact_payload.get("schema_version"),
            "case_id": context.case_id,
            "category": category,
            "category_run_content_sha256": str(
                run_manifest.get("category_run_content_sha256") or ""
            ),
            "configuration_hash": str(
                dict(run_manifest.get("configuration_identity") or {}).get(
                    "configuration_hash"
                )
                or ""
            ),
            "configuration_sha256": str(
                dict(run_manifest.get("configuration_identity") or {}).get(
                    "configuration_hash"
                )
                or ""
            ),
            "processed_span": dict(
                dict(run_manifest.get("source_lineage") or {}).get("processed_span") or {}
            ),
            "schema_version": CATEGORY_RUN_IDENTITY_SCHEMA_VERSION,
            "source_sha256": str(
                dict(run_manifest.get("source_lineage") or {}).get("source_sha256") or ""
            ),
        }
    if len(artifact_paths) != len(identities):
        raise ValueError("duplicate category run manifest path")
    missing = sorted(selected - set(identities))
    extra = sorted(set(identities) - selected)
    if missing or extra:
        raise ValueError(
            "category run manifest identities must exactly match selected categories: "
            f"missing={missing}, extra={extra}"
        )
    return {category: identities[category] for category in sorted(identities)}


def contained_case_artifact_path(case_root: Path, artifact_path: str) -> Path:
    path = Path(artifact_path)
    resolved = path.resolve() if path.is_absolute() else (case_root / path).resolve()
    try:
        resolved.relative_to(case_root.resolve())
    except ValueError as exc:
        raise ValueError(f"category run manifest path escapes case root: {artifact_path}") from exc
    return resolved


def validate_category_run_manifest_payload(
    context: Any,
    *,
    category: str,
    result: Any,
    run_manifest: dict[str, Any],
) -> None:
    if run_manifest.get("case_id") != context.case_id:
        raise ValueError(
            f"{category}: category run case_id mismatch: "
            f"{run_manifest.get('case_id')} != {context.case_id}"
        )
    if run_manifest.get("category") != category:
        raise ValueError(
            f"{category}: category run category mismatch: {run_manifest.get('category')}"
        )
    source_lineage = dict(run_manifest.get("source_lineage") or {})
    if source_lineage.get("source_sha256") != context.source_identity.sha256:
        raise ValueError(
            f"{category}: category run source hash mismatch: "
            f"{source_lineage.get('source_sha256')} != {context.source_identity.sha256}"
        )
    validate_processed_span(
        source_lineage.get("processed_span"),
        source_frame_count=context.source_identity.frame_count,
    )
    configuration_identity = dict(run_manifest.get("configuration_identity") or {})
    expected_configuration = dict(result.configuration_identity or {})
    if configuration_identity != expected_configuration:
        raise ValueError(f"{category}: category run configuration identity mismatch")
    failed_stages = [
        dict(stage).get("stage") or dict(stage).get("category") or "stage"
        for stage in list(run_manifest.get("stage_outcomes") or [])
        if str(dict(stage).get("status") or "").lower() in {"failed", "error"}
    ]
    if failed_stages:
        raise ValueError(f"{category}: category run failed stages: {failed_stages}")
    content_hash = str(run_manifest.get("category_run_content_sha256") or "")
    if not content_hash:
        raise ValueError(f"{category}: category run content hash is required")


def validate_processed_span(value: Any, *, source_frame_count: int | None) -> None:
    span = dict(value or {})
    expected_first = 0 if source_frame_count else None
    expected_last = None if source_frame_count is None else max(0, int(source_frame_count) - 1)
    expected_count = source_frame_count
    comparisons = {
        "first_processed_frame": expected_first,
        "last_processed_frame": expected_last,
        "source_frame_count": expected_count,
    }
    for key, expected in comparisons.items():
        actual = span.get(key)
        if actual != expected:
            raise ValueError(
                f"category run processed span mismatch for {key}: {actual} != {expected}"
            )
    if span.get("full_source_processed") is not True:
        raise ValueError("category run processed span must cover the full source")


def automatic_preview_identity(
    *,
    automatic_path: Path,
    media_inventory: dict[str, Any],
    preview_manifest: dict[str, Any],
    automatic_region_count: int,
    preview_source_role: str,
    workspace_dir: Path,
) -> dict[str, Any]:
    item = next(
        dict(row)
        for row in media_inventory.get("items") or []
        if dict(row).get("role") == "automatic"
    )
    return {
        "artifact_identity": {
            "path": relative_to_workspace(automatic_path, workspace_dir),
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        },
        "automatic_preview_sha256": item["sha256"],
        "frame_count": item["frame_count"],
        "height": item["height"],
        "non_final": True,
        "region_count": automatic_region_count,
        "renderer_backend": first_present(
            preview_manifest,
            "renderer_backend",
            "materialization_mode",
        )
        or preview_source_role,
        "renderer_revision": first_present(
            preview_manifest,
            "renderer_id",
            "renderer_revision",
            "schema_version",
        ),
        "source_frame_end": item["case_source_frame_end"],
        "source_frame_start": item["case_source_frame_start"],
        "source_role": preview_source_role,
        "source_sha256": item["source_sha256"],
        "width": item["width"],
    }


def validate_authoritative_workspace_manifest(payload: dict[str, Any]) -> None:
    case_ids.validate_case_id(str(payload.get("case_id") or ""))
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("workspace manifest cases must be a mapping")
    case_ids.validate_case_ids(tuple(str(case_id) for case_id in cases))
    if payload.get("workspace_status") != "ready":
        raise ValueError("workspace manifest is not ready")
    lifecycle = dict(payload.get("lifecycle") or {})
    if dict(lifecycle.get("review_session") or {}).get("status") != "not_started":
        raise ValueError("review_session lifecycle must be not_started")
    for key in ("latest_reviewed", "finalization", "evaluation"):
        if dict(lifecycle.get(key) or {}).get("status") != "not_run":
            raise ValueError(f"{key} lifecycle must be not_run")
    identities = dict(payload.get("content_identities") or {})
    required = {
        "automatic_baseline_regions",
        "entity_registry",
        "media_inventory",
        "review_tasks",
        "session_state",
        "validation",
        "workspace_summary",
    }
    missing = sorted(required - set(identities))
    if missing:
        raise ValueError(f"workspace manifest missing content identities: {missing}")
    automatic = dict(payload.get("automatic_preview") or {})
    if not automatic.get("source_sha256") or not automatic.get("automatic_preview_sha256"):
        raise ValueError("workspace manifest automatic preview hashes are required")
    if parse_non_negative_int(automatic.get("region_count"), field_name="region_count") != int(
        automatic.get("region_count")
    ):
        raise ValueError("workspace manifest automatic preview region count is invalid")
    selected = set(str(category) for category in payload.get("selected_categories") or [])
    category_identities = dict(payload.get("category_run_manifest_identities") or {})
    if set(category_identities) != selected:
        raise ValueError(
            "workspace manifest category run identities must exactly match selected categories"
        )
    artifact_ids: set[str] = set()
    for category, raw_identity in sorted(category_identities.items()):
        identity = dict(raw_identity or {})
        if identity.get("category") != category:
            raise ValueError(f"{category}: workspace category run identity category mismatch")
        for key in (
            "artifact_id",
            "category_run_content_sha256",
            "configuration_hash",
            "path",
            "sha256",
            "source_sha256",
        ):
            if not identity.get(key):
                raise ValueError(f"{category}: category run identity missing {key}")
        artifact_id = str(identity["artifact_id"])
        if artifact_id in artifact_ids:
            raise ValueError(f"duplicate category run identity: {artifact_id}")
        artifact_ids.add(artifact_id)
    expected_hash = stable_hash(category_identities)
    if payload.get("category_run_manifest_identity_hash") != expected_hash:
        raise ValueError("workspace manifest category run identity hash mismatch")
    publication = dict(payload.get("latest_reviewed_publication") or {})
    if publication.get("status") != "supported":
        raise ValueError("workspace manifest must support latest reviewed publication")
    if publication.get("publication_mode") != "reviewed_region_compiler.v1":
        raise ValueError("workspace manifest latest reviewed compiler mode is required")


def artifact_reference(case_root: Path, path: Path, role: str) -> case_manifests.ArtifactReference:
    artifact = case_manifests.build_artifact_reference(path, case_root=case_root, role=role)
    if artifact is None:
        raise case_manifests.ManifestValidationError(f"{role}: artifact missing")
    return artifact


def task_frame_span(
    task: dict[str, Any],
    *,
    source_frame_end: int | None = None,
) -> tuple[int, int]:
    start = first_present(task, "source_frame_start", "frame_start", "frame")
    end = first_present(task, "source_frame_end", "frame_end", "frame")
    start_int = int(start if start is not None else 0)
    end_int = int(end if end is not None else start_int)
    if start_int < 0 or end_int < 0:
        raise ValueError(f"invalid review task frame span: {start_int}-{end_int}")
    if start_int > end_int:
        raise ValueError(f"invalid review task frame span: {start_int}>{end_int}")
    if source_frame_end is not None and end_int > source_frame_end:
        raise ValueError(
            f"review task frame span outside source: {start_int}-{end_int} > {source_frame_end}"
        )
    return start_int, end_int


def task_entity_keys(category: str, task: dict[str, Any]) -> tuple[str, ...]:
    if category == CATEGORY_FACE:
        value = first_present(
            task,
            "object_id",
            "track_id",
            "target_track_id",
        )
        if value not in {None, ""}:
            return (face_track_key(str(value)),)
        target_track_ids = task.get("target_track_ids") or []
        if isinstance(target_track_ids, (str, int)):
            target_track_ids = [target_track_ids]
        if target_track_ids:
            return tuple(face_track_key(str(item)) for item in target_track_ids)
        raw_ids = task.get("raw_review_item_ids") or task.get("primary_raw_review_item_ids") or []
        if raw_ids:
            return tuple(face_track_key(str(item)) for item in raw_ids)
        return ()
    track_uids = task.get("track_uids") or task.get("track_ids") or []
    if isinstance(track_uids, (str, int)):
        track_uids = [track_uids]
    return tuple(str(item) for item in track_uids if str(item))


def resolve_task_target_binding(
    *,
    case_id: str,
    category: str,
    task: dict[str, Any],
    frame_start: int,
    frame_end: int,
    entity_keys: tuple[str, ...],
    missing_entity_keys: list[str],
    fallback_entity_keys: tuple[str, ...],
    fallback_missing_entity_keys: list[str],
) -> TaskTargetBinding:
    if entity_keys and not missing_entity_keys:
        return TaskTargetBinding(
            kind=TARGET_BINDING_ENTITY,
            entity_keys=entity_keys,
            payload=entity_target_binding_payload(
                case_id=case_id,
                category=category,
                task=task,
                frame_start=frame_start,
                frame_end=frame_end,
                entity_keys=entity_keys,
            ),
        )
    if fallback_entity_keys and not fallback_missing_entity_keys:
        return TaskTargetBinding(
            kind=TARGET_BINDING_ENTITY,
            entity_keys=fallback_entity_keys,
            payload=entity_target_binding_payload(
                case_id=case_id,
                category=category,
                task=task,
                frame_start=frame_start,
                frame_end=frame_end,
                entity_keys=fallback_entity_keys,
            )
            | {"binding_source": "interval_entity_overlap"},
        )
    if task_supports_source_span_binding(
        case_id=case_id,
        category=category,
        task=task,
        frame_start=frame_start,
        frame_end=frame_end,
    ):
        return TaskTargetBinding(
            kind=TARGET_BINDING_SOURCE_SPAN,
            payload=source_span_target_binding_payload(
                case_id=case_id,
                category=category,
                task=task,
                frame_start=frame_start,
                frame_end=frame_end,
                unresolved_entity_keys=entity_keys,
                missing_entity_keys=missing_entity_keys,
            ),
        )
    return TaskTargetBinding(kind="unbound")


def entity_target_binding_payload(
    *,
    case_id: str,
    category: str,
    task: dict[str, Any],
    frame_start: int,
    frame_end: int,
    entity_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "kind": TARGET_BINDING_ENTITY,
        "schema_version": TARGET_BINDING_SCHEMA_VERSION,
        "source_frame_end": frame_end,
        "source_frame_start": frame_start,
        "source_task_id": source_task_id(category, task),
        "technical_track_ids": list(entity_keys),
    }


def source_span_target_binding_payload(
    *,
    case_id: str,
    category: str,
    task: dict[str, Any],
    frame_start: int,
    frame_end: int,
    unresolved_entity_keys: tuple[str, ...],
    missing_entity_keys: list[str],
) -> dict[str, Any]:
    return {
        "binding_source": "authoritative_review_task_interval",
        "case_id": case_id,
        "category": category,
        "event_ids": task_event_ids(category, task),
        "kind": TARGET_BINDING_SOURCE_SPAN,
        "missing_entity_keys": list(missing_entity_keys),
        "reason_codes": task_reason_codes(task),
        "schema_version": TARGET_BINDING_SCHEMA_VERSION,
        "source_frame_end": frame_end,
        "source_frame_start": frame_start,
        "source_review_category": str(task.get("category") or task.get("task_type") or ""),
        "source_task_id": source_task_id(category, task),
        "unresolved_entity_keys": list(unresolved_entity_keys),
    }


def task_supports_source_span_binding(
    *,
    case_id: str,
    category: str,
    task: dict[str, Any],
    frame_start: int,
    frame_end: int,
) -> bool:
    if task_has_explicit_entity_claim(category, task):
        return False
    task_case_id = task.get("case_id")
    if task_case_id not in {None, "", case_id}:
        raise ValueError(
            f"source_span_review_task: {category}:{source_task_id(category, task)} "
            f"case mismatch {task_case_id} != {case_id}"
        )
    object_category = task.get("object_category")
    if object_category not in {None, "", category}:
        raise ValueError(
            f"source_span_review_task: {category}:{source_task_id(category, task)} "
            f"category mismatch {object_category} != {category}"
        )
    explicit_id = first_present(task, "review_task_id", "reviewer_task_id")
    if explicit_id in {None, ""}:
        return False
    event_ids = task_event_ids(category, task)
    if not event_ids:
        return False
    if not source_span_media_covers_task(task, frame_start=frame_start, frame_end=frame_end):
        return False
    reasons = set(task_reason_codes(task))
    issue_type = str(task.get("category") or task.get("task_type") or "").lower()
    if issue_type not in {"action_required", "corrected_result_check"}:
        return False
    return bool(reasons & SOURCE_SPAN_REVIEW_REASON_CODES)


def task_has_explicit_entity_claim(category: str, task: dict[str, Any]) -> bool:
    fields = ("object_id", "track_id", "target_track_id", "target_track_ids")
    if any(has_meaningful_value(task.get(field)) for field in fields):
        return True
    if category == CATEGORY_PLATE:
        return bool(task.get("track_uids") or task.get("track_ids"))
    return False


def has_meaningful_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def task_event_ids(category: str, task: dict[str, Any]) -> list[str]:
    ids = raw_signal_ids(task)
    for key in ("primary_raw_review_item_ids", "review_task_id", "reviewer_task_id"):
        value = task.get(key)
        values = value if isinstance(value, list | tuple) else [value]
        for item in values:
            if item not in {None, ""} and str(item) not in ids:
                ids.append(str(item))
    if not ids and category == CATEGORY_PLATE:
        value = task.get("review_task_id")
        if value not in {None, ""}:
            ids.append(str(value))
    return ids


def task_reason_codes(task: dict[str, Any]) -> list[str]:
    raw = task.get("raw_reason_codes") or task.get("reason_codes") or task.get("reasons") or []
    if isinstance(raw, (str, int)):
        raw = [raw]
    normalized: list[str] = []
    for item in raw:
        text = str(item).strip().lower().replace(":", "_")
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def source_span_media_covers_task(
    task: dict[str, Any],
    *,
    frame_start: int,
    frame_end: int,
) -> bool:
    ranges = source_frame_ranges_from_task_media(task)
    if not ranges:
        return False
    return any(start <= frame_start and frame_end <= end for start, end in ranges)


def source_frame_ranges_from_task_media(task: dict[str, Any]) -> list[tuple[int, int]]:
    roles = task.get("media_roles") or []
    if isinstance(roles, dict):
        roles = list(roles.values())
    ranges: list[tuple[int, int]] = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        audit = dict(role.get("media_audit") or {})
        source_range = dict(audit.get("source_frame_range") or role.get("source_frame_range") or {})
        start = optional_int(source_range.get("frame_start"))
        end = optional_int(source_range.get("frame_end"))
        if start is not None and end is not None:
            ranges.append((start, end))
    return ranges


def task_allows_interval_fallback(category: str, task: dict[str, Any]) -> bool:
    if task.get("media_roles") or task.get("corrected_context"):
        return True
    task_id = source_task_id(category, task)
    issue_type = str(task.get("issue_type") or task.get("task_type") or "").lower()
    raw_reasons = task.get("reasons") or []
    if isinstance(raw_reasons, (str, int)):
        raw_reasons = [raw_reasons]
    reasons = " ".join(str(item).lower() for item in list(raw_reasons))
    title = str(task.get("title") or "").lower()
    return any(
        "spot_check" in value or "spot check" in value
        for value in (task_id.lower(), issue_type, reasons, title)
    )


def fallback_interval_task_entity_keys(
    sources: dict[tuple[str, str], ReviewEntitySource],
    *,
    category: str,
    frame_start: int,
    frame_end: int,
) -> tuple[str, ...]:
    keys: list[str] = []
    for (source_category, key), source in sorted(sources.items()):
        if source_category != category:
            continue
        first = source.first_frame
        last = source.last_frame
        if first is None or last is None or first <= frame_end and last >= frame_start:
            keys.append(key)
    return tuple(keys)


def fallback_task_entity_keys(
    entities_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    category: str,
    frame_start: int,
    frame_end: int,
) -> tuple[str, ...]:
    keys: list[str] = []
    for (source_category, key), entity in sorted(entities_by_key.items()):
        if source_category != category:
            continue
        first = entity.get("first_frame")
        last = entity.get("last_frame")
        if first is None or last is None or int(first) <= frame_end and int(last) >= frame_start:
            keys.append(key)
    return tuple(keys)


def entity_key_for_record(record: PrivacyRegionRecord) -> str:
    source = dict(record.source_record or {})
    if record.category == CATEGORY_FACE:
        target_ids = source.get("target_track_ids") or []
        if isinstance(target_ids, (str, int)):
            target_ids = [target_ids]
        if target_ids:
            return face_track_key(str(target_ids[0]))
    for key in (
        "track_uid",
        "track_id",
        "target_track_id",
        "parent_track_id",
        "region_id",
        "detection_id",
    ):
        value = source.get(key)
        if value not in {None, ""}:
            return str(value)
    if record.object_id:
        if record.category == CATEGORY_FACE:
            return face_track_key(str(record.object_id))
        return str(record.object_id)
    return f"region:{stable_hash(record.to_dict())[:16]}"


def source_task_id(category: str, task: dict[str, Any]) -> str:
    value = (
        task.get("review_task_id")
        if category == CATEGORY_PLATE
        else task.get("reviewer_task_id")
    )
    value = value or task.get("review_task_id") or task.get("reviewer_task_id")
    if value not in {None, ""}:
        return safe_id(str(value))
    start = int(first_present(task, "source_frame_start", "frame_start", "frame") or 0)
    end = int(first_present(task, "source_frame_end", "frame_end", "frame") or start)
    return f"task_{start}_{end}_{stable_hash(task)[:8]}"


def raw_signal_ids(task: dict[str, Any]) -> list[str]:
    ids = task.get("raw_signal_ids") or task.get("raw_review_item_ids") or []
    if isinstance(ids, (str, int)):
        return [str(ids)]
    return [str(item) for item in ids]


def task_issue_type(category: str, task: dict[str, Any]) -> str:
    if category == CATEGORY_PLATE:
        stage = str(task.get("source_stage") or "")
        reasons = set(str(item) for item in task.get("reason_codes") or [])
        if task.get("technical_only"):
            return "technical_signal"
        if "spot_check" in str(task.get("review_task_id") or ""):
            return "optional_spot_check"
        if "false_positive" in stage or any("false" in reason for reason in reasons):
            return "possible_false_redaction"
        if reasons & {"anchor_scale_change_too_large", "gap_exceeds_safe_interpolation_limit"}:
            return "plate_coverage_gap"
        if stage in {"plate_tracking", "license_plate_tracking"} or reasons & {
            "insufficient_observations_for_auto_accept",
            "low_observation_density",
            "unresolved_internal_gap",
        }:
            return "plate_track_review_required"
        return "possible_privacy_gap"
    task_category = str(task.get("category") or task.get("normal_task_class") or "")
    reasons = set(str(item) for item in task.get("raw_reason_codes") or [])
    if task_category in {"automatic_continuity_check", "optional_spot_check"}:
        return "optional_spot_check"
    if task_category == "corrected_result_check":
        return "correction_verification"
    if reasons & {"false_positive", "wrong_target"}:
        return "possible_false_redaction"
    return "possible_privacy_gap"


def task_blocking(category: str, task: dict[str, Any]) -> bool:
    if category == CATEGORY_FACE:
        if task.get("final_export_blocking") is not None:
            return bool(task.get("final_export_blocking"))
        if task.get("requires_decision") is not None:
            return bool(task.get("requires_decision"))
    return bool(task.get("blocking", False))


def task_severity(task: dict[str, Any], blocking: bool) -> str:
    severity = str(task.get("severity") or "")
    if severity:
        return "blocker" if severity == "blocking" else severity
    return "blocker" if blocking else "low"


def task_allowed_actions(category: str, task: dict[str, Any], issue_type: str) -> list[str]:
    source_allowed_actions(task)
    return list(review_sessions.CANONICAL_REVIEW_ACTIONS)


def source_allowed_actions(task: dict[str, Any]) -> list[str]:
    raw = task.get("allowed_user_actions") or task.get("allowed_actions") or []
    actions: list[str] = []
    for item in raw:
        value = item.get("action") if isinstance(item, dict) else item
        action = normalize_review_action(str(value))
        if action in review_sessions.LEGACY_DECISION_ACTIONS - review_sessions.DECISION_ACTIONS:
            continue
        if action not in review_sessions.DECISION_ACTIONS:
            raise ValueError(f"unsupported review action: {value}")
        if action not in actions:
            actions.append(action)
    return actions


def normalize_review_action(value: str) -> str:
    if value in review_sessions.DECISION_ACTIONS:
        return value
    if value in ACTION_ALIASES:
        return ACTION_ALIASES[value]
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    return ACTION_ALIASES.get(value.strip(), lowered)


def validate_task_timestamps(
    task: dict[str, Any],
    *,
    start: int,
    end: int,
    fps: str,
) -> None:
    fps_value = rational_to_float(fps) or 30.0
    start_seconds = first_present(
        task,
        "source_timestamp_start_seconds",
        "timestamp_start_seconds",
        "start_seconds",
    )
    end_seconds = first_present(
        task,
        "source_timestamp_end_seconds",
        "timestamp_end_seconds",
        "end_seconds",
    )
    tolerance = max(0.05, 1.0 / fps_value)
    if start_seconds is not None and abs(float(start_seconds) - (start / fps_value)) > tolerance:
        raise ValueError("review task timestamp interval is inconsistent with frame span")
    if end_seconds is not None and abs(float(end_seconds) - (end / fps_value)) > tolerance:
        raise ValueError("review task timestamp interval is inconsistent with frame span")


def face_track_key(value: str) -> str:
    text = str(value)
    if text.startswith("face_track_"):
        return text
    if text.isdigit():
        return f"face_track_{text}"
    return text


def task_title(category: str, task: dict[str, Any], issue_type: str) -> str:
    if task.get("title"):
        return str(task["title"])
    if issue_type == "optional_spot_check":
        return "Spot check this moment"
    if issue_type == "possible_false_redaction":
        return "Possible false redaction"
    if issue_type == "plate_coverage_gap":
        return "Plate coverage needs correction"
    if issue_type == "plate_track_review_required":
        return "Plate track requires review"
    if category == CATEGORY_FACE:
        return "Possible face privacy gap"
    return "Possible plate privacy gap"


def task_what_happened(category: str, task: dict[str, Any], issue_type: str) -> str:
    if task.get("what_happened"):
        return str(task["what_happened"])
    if task.get("explanation"):
        return str(task["explanation"])
    if issue_type == "optional_spot_check":
        return "The system selected this moment for a deterministic review sample."
    if issue_type == "plate_coverage_gap":
        return "The plate tracker found a coverage gap that needs human correction."
    if issue_type == "plate_track_review_required":
        return "The plate tracker marked this interval for human review."
    if category == CATEGORY_FACE:
        return "The face pipeline marked this interval for human review."
    return "The plate pipeline marked this interval for human review."


def task_what_to_inspect(category: str, task: dict[str, Any], issue_type: str) -> str:
    if task.get("what_to_inspect"):
        return str(task["what_to_inspect"])
    checklist = task.get("what_to_check")
    if isinstance(checklist, list) and checklist:
        return " ".join(str(item) for item in checklist)
    if issue_type == "possible_false_redaction":
        return (
            "Compare Original and Automatic preview and decide whether this object "
            "is not sensitive."
        )
    if category == CATEGORY_FACE:
        return (
            "Compare Original and Automatic preview and confirm that the person remains "
            "protected."
        )
    return "Compare Original and Automatic preview and confirm that the plate remains protected."


def median_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    columns = list(zip(*boxes, strict=True))
    return tuple(sorted(column)[len(column) // 2] for column in columns)  # type: ignore[return-value]


def frame_seconds(frame: int | None, fps: float) -> float | None:
    if frame is None:
        return None
    return round(float(frame) / fps, 6)


def first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    validator: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with tmp.open("w", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        loaded = json.loads(tmp.read_text(encoding="utf-8"))
        if validator is not None:
            validator(loaded)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def relative_to_workspace(path: Path, workspace_dir: Path) -> str:
    return path.relative_to(workspace_dir).as_posix()
