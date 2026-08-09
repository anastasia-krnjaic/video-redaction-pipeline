from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from redactor.privacy_records import MANUAL_REDACTION_CATEGORIES, stable_hash

API_SCHEMA_VERSION = "final_review_workspace_api.v1"
API_PREFIX = "/api/v1"


def final_review_workspace_schema() -> dict[str, Any]:
    string_or_null = {"type": ["string", "null"]}
    integer_or_null = {"type": ["integer", "null"]}
    number_or_null = {"type": ["number", "null"]}
    dict_value = {"type": "object", "additionalProperties": True}
    array_value = {"type": "array", "items": dict_value}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://video-redaction-pipeline.local/final_review_workspace_v1.schema.json",
        "title": "Final Review Workspace API v1",
        "type": "object",
        "x-api-prefix": API_PREFIX,
        "x-api-schema-version": API_SCHEMA_VERSION,
        "x-manual-redaction-categories": list(MANUAL_REDACTION_CATEGORIES),
        "definitions": {
            "FinalReviewWorkspaceApiSchema": {
                "type": "object",
                "additionalProperties": True,
                "required": ["$schema", "$id", "definitions", "paths"],
                "properties": {
                    "$id": {"type": "string"},
                    "$schema": {"type": "string"},
                    "definitions": {"type": "object"},
                    "paths": {"type": "object"},
                    "title": {"type": "string"},
                },
            },
            "ApiError": {
                "type": "object",
                "additionalProperties": False,
                "required": ["error", "ok"],
                "properties": {
                    "error": {"type": "string"},
                    "ok": {"const": False},
                },
            },
            "WorkspaceSummary": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "alignment",
                    "api_schema_version",
                    "cases",
                    "entities",
                    "raw_signals",
                    "tasks",
                    "timeline",
                    "workspace",
                ],
                "properties": {
                    "alignment": {"$ref": "#/definitions/VersionAlignmentAudit"},
                    "api_schema_version": {"const": API_SCHEMA_VERSION},
                    "backend_workspace_schema_version": {"type": "string"},
                    "configuration_freeze_hash": string_or_null,
                    "cases": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/definitions/Case"},
                    },
                    "entities": {"$ref": "#/definitions/EntityIndex"},
                    "frontend": dict_value,
                    "raw_signals": dict_value,
                    "tasks": {"$ref": "#/definitions/TaskCollection"},
                    "timeline": dict_value,
                    "workspace": dict_value,
                },
            },
            "Case": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "case_id",
                    "display_name",
                    "fps",
                    "frame_count",
                    "source_frame_start",
                    "source_frame_end",
                    "source_path",
                    "status",
                    "video_roles",
                ],
                "properties": {
                    "blockers": {"type": "array", "items": {"type": "string"}},
                    "case_id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "duration_seconds": number_or_null,
                    "fps": {"type": "string"},
                    "frame_count": {"type": "integer"},
                    "height": integer_or_null,
                    "source_frame_start": {"type": "integer"},
                    "source_frame_end": {"type": "integer"},
                    "source_path": {"type": "string"},
                    "source_video_hash": string_or_null,
                    "status": {"type": "string"},
                    "video_roles": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/MediaVersion"},
                    },
                    "width": integer_or_null,
                },
            },
            "MediaVersion": {
                "type": "object",
                "additionalProperties": True,
                "required": ["label", "path", "role", "source_frame_start", "source_frame_end"],
                "properties": {
                    "label": {"type": "string"},
                    "path": {"type": "string"},
                    "role": {
                        "enum": [
                            "original",
                            "automatic",
                            "current_result",
                            "correction_preview",
                            "checkpoint",
                            "latest_reviewed",
                            "final_reviewed",
                        ]
                    },
                    "sha256": string_or_null,
                    "source_frame_start": {"type": "integer"},
                    "source_frame_end": {"type": "integer"},
                },
            },
            "TaskCollection": {
                "type": "object",
                "additionalProperties": True,
                "required": ["actions", "task_count", "tasks"],
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["action", "label"],
                            "properties": {
                                "action": {"type": "string"},
                                "label": {"type": "string"},
                            },
                        },
                    },
                    "raw_signal_count": {"type": "integer"},
                    "schema_version": {"type": "string"},
                    "task_count": {"type": "integer"},
                    "tasks": {"type": "array", "items": {"$ref": "#/definitions/Task"}},
                },
            },
            "Task": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "case_id",
                    "category",
                    "review_task_id",
                    "source_frame_start",
                    "source_frame_end",
                    "technical_only",
                    "title",
                    "what_to_inspect",
                ],
                "properties": {
                    "allowed_actions": {"type": "array", "items": {"type": "string"}},
                    "accepted_working_state_revision": string_or_null,
                    "actionable": {"type": "boolean"},
                    "blocking": {"type": "boolean"},
                    "case_id": {"type": "string"},
                    "category": {"type": "string"},
                    "changed_frame_count": integer_or_null,
                    "changed_frame_intervals": {"type": "array", "items": dict_value},
                    "correction_id": string_or_null,
                    "correction_required": {"type": "boolean"},
                    "decision_state": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "materialization_required": {"type": "boolean"},
                    "normal_task_class": {"type": "string"},
                    "optional": {"type": "boolean"},
                    "publication_status": string_or_null,
                    "raw_signal_ids": {"type": "array", "items": {"type": "string"}},
                    "review_task_id": {"type": "string"},
                    "severity": {"type": "string"},
                    "source_frame_start": {"type": "integer"},
                    "source_frame_end": {"type": "integer"},
                    "target_effective_region_count": integer_or_null,
                    "target_effective_regions_hash": string_or_null,
                    "technical_only": {"type": "boolean"},
                    "title": {"type": "string"},
                    "verification_status": string_or_null,
                    "what_happened": {"type": "string"},
                    "what_to_inspect": {"type": "string"},
                },
            },
            "TaskDecision": {
                "type": "object",
                "additionalProperties": True,
                "required": ["action", "case_id", "decision_status", "task_id"],
                "properties": {
                    "action": {"type": "string"},
                    "case_id": {"type": "string"},
                    "decision_status": {"enum": ["green", "blocking"]},
                    "event_hash": string_or_null,
                    "event_id": string_or_null,
                    "note": {"type": "string"},
                    "reviewed_at_utc": {"type": "string"},
                    "task_id": {"type": "string"},
                },
            },
            "Session": {
                "type": "object",
                "additionalProperties": True,
                "required": ["session_id", "reviewer", "started_at_utc", "session_status"],
                "properties": {
                    "application_version": {"type": "string"},
                    "case_manifest_hashes": dict_value,
                    "configuration_freeze_hash": {"type": "string"},
                    "reviewer": {"$ref": "#/definitions/Attribution"},
                    "session_id": {"type": "string"},
                    "session_status": {"type": "string"},
                    "source_hashes": dict_value,
                    "started_at_utc": {"type": "string"},
                },
            },
            "SessionState": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "attribution",
                    "chain",
                    "corrections",
                    "decisions",
                    "events",
                    "ok",
                    "session",
                    "summary",
                    "working_state",
                ],
                "properties": {
                    "attribution": {"$ref": "#/definitions/Attribution"},
                    "chain": dict_value,
                    "corrections": array_value,
                    "decisions": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/definitions/TaskDecision"},
                    },
                    "events": {"type": "array", "items": {"$ref": "#/definitions/AuditEvent"}},
                    "ok": {"const": True},
                    "session": {"$ref": "#/definitions/Session"},
                    "summary": dict_value,
                    "working_state": {"$ref": "#/definitions/WorkingState"},
                },
            },
            "Attribution": {
                "type": ["object", "null"],
                "additionalProperties": True,
                "properties": {
                    "attributed_at_utc": string_or_null,
                    "display_name": string_or_null,
                    "event_hash": string_or_null,
                    "event_id": string_or_null,
                    "reviewer_id": string_or_null,
                    "role": string_or_null,
                    "schema_version": string_or_null,
                    "session_id": string_or_null,
                },
            },
            "EntityIndex": {
                "type": "object",
                "additionalProperties": True,
                "required": ["cases"],
                "properties": {
                    "case_unresolved": dict_value,
                    "cases": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/Entity"},
                        },
                    },
                    "consolidation_configuration_hash": string_or_null,
                    "unresolved_evidence_by_case": dict_value,
                },
            },
            "Entity": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "case_id",
                    "category",
                    "current_policy",
                    "entity_id",
                    "label",
                    "representative_source_frame",
                ],
                "properties": {
                    "attached_fragment_track_ids": {"type": "array", "items": {"type": "string"}},
                    "automatic_source_track_ids": {"type": "array", "items": {"type": "string"}},
                    "case_id": {"type": "string"},
                    "category": {"type": "string"},
                    "created_provenance": dict_value,
                    "current_policy": {"type": "string"},
                    "current_status": {"type": "string"},
                    "default_privacy_policy": {"type": "string"},
                    "display_name": {"type": "string"},
                    "entity_id": {"type": "string"},
                    "evidence_summary": dict_value,
                    "first_visible_source_frame": integer_or_null,
                    "last_visible_source_frame": integer_or_null,
                    "label": {"type": "string"},
                    "member_detector_tracker_ids": {"type": "array", "items": {"type": "string"}},
                    "merge_split_lineage": {"type": "array", "items": dict_value},
                    "possible_duplicate": {"type": "boolean"},
                    "possible_duplicate_suggestions": {"type": "array", "items": dict_value},
                    "privacy_action": {"type": "string"},
                    "representative_case_frame": integer_or_null,
                    "representative_bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "representative_source_frame": integer_or_null,
                    "representative_thumbnail_path": string_or_null,
                    "review_entity_state": {"type": "string"},
                    "review_state": {"type": "string"},
                },
            },
            "EntityPolicyAction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["case_id", "entity_id", "policy"],
                "properties": {
                    "case_id": {"type": "string"},
                    "entity_id": {"type": "string"},
                    "policy": {
                        "enum": ["redact", "keep_visible", "not_sensitive", "review_required"]
                    },
                },
            },
            "BulkEntityPolicyAction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["case_id", "entity_ids", "policy"],
                "properties": {
                    "case_id": {"type": "string"},
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                    "policy": {
                        "enum": ["redact", "keep_visible", "not_sensitive", "review_required"]
                    },
                },
            },
            "EntityMergeCandidate": {
                "type": "object",
                "additionalProperties": True,
                "required": ["entity_id", "label", "similarity_label", "temporal_gap_frames"],
                "properties": {
                    "entity_id": {"type": "string"},
                    "label": {"type": "string"},
                    "similarity_label": {"type": "string"},
                    "temporal_gap_frames": {"type": "integer"},
                },
            },
            "EntityMergeEvent": {
                "type": "object",
                "additionalProperties": True,
                "required": ["case_id", "source_entity_id", "target_entity_id"],
                "properties": {
                    "case_id": {"type": "string"},
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                },
            },
            "ManualPrivacyCategory": {"enum": list(MANUAL_REDACTION_CATEGORIES)},
            "Correction": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "case_id",
                    "category",
                    "configuration_freeze_hash",
                    "interpolation_method",
                    "keyframe_boxes",
                    "operation",
                    "reason",
                    "source_frame_start",
                    "source_frame_end",
                    "source_video_hash",
                    "task_id",
                ],
                "properties": {
                    "case_id": {"type": "string"},
                    "category": {"$ref": "#/definitions/ManualPrivacyCategory"},
                    "checkpoint_verifications": {"type": "array", "items": dict_value},
                    "configuration_freeze_hash": {"type": "string"},
                    "entity_id": string_or_null,
                    "interpolation_method": {"enum": ["linear", "none"]},
                    "keyframe_boxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["bbox_xyxy", "source_frame"],
                            "properties": {
                                "bbox_xyxy": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                "source_frame": {"type": "integer"},
                            },
                        },
                    },
                    "operation": {"type": "string"},
                    "reason": {"type": "string"},
                    "reviewer_identity": dict_value,
                    "source_frame_end": {"type": "integer"},
                    "source_frame_start": {"type": "integer"},
                    "source_video_hash": {"type": "string"},
                    "task_id": {"type": "string"},
                },
            },
            "CorrectionPreview": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "corrected_preview": string_or_null,
                    "draft_hash": string_or_null,
                    "effective_regions_hash": string_or_null,
                    "frame_end": integer_or_null,
                    "frame_start": integer_or_null,
                    "latest_reviewed": string_or_null,
                    "preview_sha256": string_or_null,
                    "preview_token": string_or_null,
                    "source_video_hash": string_or_null,
                    "working_state_revision": string_or_null,
                },
            },
            "WorkingState": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "baseline",
                    "current_event_chain_head",
                    "manual_corrections",
                    "materialized_latest_reviewed",
                    "object_policy_overrides",
                    "schema_version",
                    "state_hash",
                    "task_decisions",
                ],
                "properties": {
                    "active_correction_ids": {"type": "array", "items": {"type": "string"}},
                    "active_manual_tasks": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/Task"},
                    },
                    "active_projection_hash": {"type": "string"},
                    "baseline": dict_value,
                    "checkpoints": {"type": "array", "items": {"$ref": "#/definitions/Checkpoint"}},
                    "current_event_chain_head": string_or_null,
                    "manual_corrections": array_value,
                    "materialized_latest_reviewed": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/definitions/MediaVersion"},
                    },
                    "object_policy_overrides": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/definitions/Entity"},
                    },
                    "schema_version": {"type": "string"},
                    "state_hash": {"type": "string"},
                    "suppressions": array_value,
                    "sweep_acknowledgements": dict_value,
                    "task_decisions": dict_value,
                },
            },
            "MaterializationStatus": {
                "type": "object",
                "additionalProperties": True,
                "required": ["case_id", "latest_reviewed", "materialized", "schema_version"],
                "properties": {
                    "case_id": {"type": "string"},
                    "effective_regions_hash": {"type": "string"},
                    "latest_reviewed": {"$ref": "#/definitions/MediaVersion"},
                    "materialized": {"type": "boolean"},
                    "schema_version": {"type": "string"},
                    "state_hash": {"type": "string"},
                    "video_sha256": {"type": "string"},
                },
            },
            "MaterializationJob": {
                "type": "object",
                "additionalProperties": True,
                "required": ["case_id", "job_id", "schema_version", "status"],
                "properties": {
                    "accepted_working_state_revision": string_or_null,
                    "affected_entity_ids": {"type": "array", "items": {"type": "string"}},
                    "case_id": {"type": "string"},
                    "changed_segment_count": {"type": "integer"},
                    "changed_source_frame_count": {"type": "integer"},
                    "changed_source_frame_intervals": {"type": "array", "items": dict_value},
                    "changed_source_frame_percentage": {"type": "number"},
                    "completed_at_utc": string_or_null,
                    "created_at_utc": string_or_null,
                    "current_stage": string_or_null,
                    "elapsed_time_ms": {"type": "integer"},
                    "encode_stage": string_or_null,
                    "error": string_or_null,
                    "job_id": {"type": "string"},
                    "latest_reviewed": {
                        "anyOf": [{"$ref": "#/definitions/MediaVersion"}, {"type": "null"}]
                    },
                    "last_progress_at_utc": string_or_null,
                    "materialization": dict_value,
                    "materialization_plan": dict_value,
                    "planned_frames_to_encode": {"type": "integer"},
                    "planned_frames_to_render": {"type": "integer"},
                    "publication_status": string_or_null,
                    "progress": {"type": "number"},
                    "queue_state": string_or_null,
                    "queue_wait_time_ms": {"type": "integer"},
                    "published_working_state_revision": string_or_null,
                    "requested_policy": string_or_null,
                    "reused_segment_count": {"type": "integer"},
                    "session_id": {"type": "string"},
                    "started_at_utc": string_or_null,
                    "status": {
                        "enum": [
                            "queued",
                            "running",
                            "completed",
                            "failed",
                            "superseded",
                            "skipped_no_effective_region_change",
                        ]
                    },
                    "superseded_by": string_or_null,
                    "target_effective_region_count": {"type": "integer"},
                    "target_effective_regions_hash": string_or_null,
                    "validation_stage": string_or_null,
                    "working_state": {"$ref": "#/definitions/WorkingState"},
                },
            },
            "Checkpoint": {
                "type": "object",
                "additionalProperties": True,
                "required": ["checkpoint_uuid", "created_at_utc", "name", "state_hash"],
                "properties": {
                    "checkpoint_uuid": {"type": "string"},
                    "created_at_utc": {"type": "string"},
                    "name": {"type": "string"},
                    "note": {"type": "string"},
                    "state_hash": {"type": "string"},
                },
            },
            "History": {
                "type": "object",
                "additionalProperties": True,
                "required": ["events", "ok", "schema_version"],
                "properties": {
                    "events": {"type": "array", "items": dict_value},
                    "ok": {"const": True},
                    "schema_version": {"type": "string"},
                },
            },
            "AuditEvent": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "event_hash",
                    "event_id",
                    "event_type",
                    "previous_event_hash",
                    "schema_version",
                    "timestamp_utc",
                ],
                "properties": {
                    "actor": dict_value,
                    "after_state": dict_value,
                    "before_state": dict_value,
                    "case_id": string_or_null,
                    "event_hash": {"type": "string"},
                    "event_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "note": string_or_null,
                    "previous_event_hash": string_or_null,
                    "schema_version": {"type": "string"},
                    "task_id": string_or_null,
                    "timestamp_utc": {"type": "string"},
                },
            },
            "FinalizationStatus": {
                "type": "object",
                "additionalProperties": True,
                "required": ["case_id", "finalized", "schema_version"],
                "properties": {
                    "blockers": {"type": "array", "items": {"type": "string"}},
                    "case_id": {"type": "string"},
                    "final_video": string_or_null,
                    "finalized": {"type": "boolean"},
                    "media_validation_passed": {"type": "boolean"},
                    "schema_version": {"type": "string"},
                },
            },
            "VersionAlignmentAudit": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "cases": dict_value,
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "passed": {"type": "boolean"},
                    "schema_version": {"type": "string"},
                },
            },
        },
        "paths": {
            "GET /api/v1/schema": {"response": "#/definitions/FinalReviewWorkspaceApiSchema"},
            "GET /api/v1/workspace": {"response": "#/definitions/WorkspaceSummary"},
            "GET /api/v1/workspace/summary": {"response": "#/definitions/WorkspaceSummary"},
            "GET /api/v1/cases/{case_id}": {"response": "#/definitions/Case"},
            "GET /api/v1/entities": {"response": "#/definitions/EntityIndex"},
            "GET /api/v1/session-state?session_id=...": {"response": "#/definitions/SessionState"},
            "GET /api/v1/working-state?session_id=...": {"response": "#/definitions/WorkingState"},
            "GET /api/v1/history?session_id=...": {"response": "#/definitions/History"},
            "GET /api/v1/checkpoints?session_id=...": {"response": "#/definitions/Checkpoint"},
            "GET /api/v1/materialization-jobs/{job_id}": {
                "response": "#/definitions/MaterializationJob"
            },
            "GET /api/v1/cases/{case_id}/versions?session_id=...": {
                "response": "#/definitions/MediaVersion"
            },
            "POST /api/v1/sessions": {"response": "#/definitions/Session"},
            "POST /api/v1/decisions": {"request": "#/definitions/TaskDecision"},
            "POST /api/v1/object-policy": {
                "request": "#/definitions/EntityPolicyAction",
                "response": "#/definitions/MaterializationJob",
            },
            "POST /api/v1/bulk-object-policy": {
                "request": "#/definitions/BulkEntityPolicyAction",
                "response": "#/definitions/MaterializationJob",
            },
            "POST /api/v1/entity-events": {"request": "#/definitions/EntityPolicyAction"},
            "POST /api/v1/entities/merge": {"request": "#/definitions/EntityMergeEvent"},
            "POST /api/v1/corrections": {
                "request": "#/definitions/Correction",
                "response": "#/definitions/CorrectionPreview",
            },
            "POST /api/v1/corrections/preview": {
                "request": "#/definitions/Correction",
                "response": "#/definitions/CorrectionPreview",
            },
            "POST /api/v1/checkpoints": {"response": "#/definitions/Checkpoint"},
            "POST /api/v1/checkpoints/restore": {"response": "#/definitions/WorkingState"},
            "POST /api/v1/checkpoints/restart-review": {"response": "#/definitions/WorkingState"},
            "POST /api/v1/checkpoints/reset": {"response": "#/definitions/WorkingState"},
            "POST /api/v1/checkpoints/delete": {"response": "#/definitions/Checkpoint"},
            "POST /api/v1/materialization-jobs/retry": {
                "response": "#/definitions/MaterializationJob"
            },
            "POST /api/v1/finalize": {"response": "#/definitions/FinalizationStatus"},
        },
    }


def workspace_summary(package_root: Path) -> dict[str, Any]:
    package_root = package_root.resolve()
    workspace = load_json(package_root / "workspace_manifest.json")
    cases: dict[str, Any] = {}
    for case_id, relative_path in sorted((workspace.get("cases") or {}).items()):
        cases[str(case_id)] = load_json(package_root / str(relative_path))
    freeze_path = package_root / "configuration_freeze.json"
    return {
        "alignment": load_optional_json(package_root / "version_alignment_audit.json", {}),
        "api_schema_version": API_SCHEMA_VERSION,
        "backend_workspace_schema_version": str(workspace.get("schema_version") or ""),
        "cases": cases,
        "configuration_freeze_hash": stable_hash(load_json(freeze_path))
        if freeze_path.exists()
        else None,
        "entities": load_optional_json(package_root / "review_entities.json", {"cases": {}}),
        "frontend": load_optional_json(package_root / "frontend_build_manifest.json", {}),
        "raw_signals": load_optional_json(package_root / "raw_audit_signals.json", {}),
        "tasks": load_optional_json(
            package_root / "reviewer_tasks.json",
            {"actions": [], "task_count": 0, "tasks": []},
        ),
        "timeline": load_optional_json(package_root / "timeline_marker_audit.json", {}),
        "workspace": workspace,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return load_json(path)


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(final_review_workspace_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the Final Review Workspace API v1 schema.")
    parser.add_argument(
        "--output",
        default="docs/api/final_review_workspace_v1.schema.json",
        type=Path,
    )
    args = parser.parse_args()
    write_schema(args.output)


if __name__ == "__main__":
    main()
