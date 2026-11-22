from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2 as cv

from redactor import case_ids, case_manifests, media, privacy_entities, privacy_rendering, rendering
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    manual_redaction_category_label,
    stable_hash,
)
from redactor.review_corrections import (
    OP_ADD,
    OP_REPLACE,
    ReviewCorrection,
    interpolate_correction_boxes,
    load_corrections_jsonl,
)
from redactor.review_sessions import (
    ACTION_NOT_SENSITIVE,
    CANONICAL_REVIEW_ACTIONS,
    EVENT_CORRECTION_SAVED_LEGACY,
    EVENT_MANUAL_CORRECTION_RETRACTED,
    EVENT_MANUAL_CORRECTION_SAVED,
    EVENT_TASK_RESOLUTION_DERIVED,
    REVIEW_DECISION_IDENTITY_V2,
    append_event,
    current_decision_for_task,
    ensure_session_layout,
    flatten_decisions_by_case,
    load_events,
    normalize_decisions_by_case,
)

WORKING_STATE_SCHEMA_VERSION = "review_working_state.v1"
ACTIVE_PROJECTION_HASH_SCHEMA_VERSION = "active_projection.v2"
LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION = "active_projection.v1"
LEGACY_ACTIVE_PROJECTION_COMPATIBILITY_HASH_KEY = "active_projection.v1.compatibility"
SUPPORTED_ACTIVE_PROJECTION_HASH_SCHEMA_VERSIONS = {
    ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
    LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
}
CHECKPOINT_SCHEMA_VERSION = "review_checkpoint.v1"
PREVIEW_TOKEN_SCHEMA_VERSION = "correction_preview_token.v2"
LATEST_REVIEWED_PUBLICATION_DEFERRED_ERROR_CODE = "latest_reviewed_publication_deferred"
LATEST_REVIEWED_PUBLICATION_DEFERRED_REASON = "mixed_reviewed_region_compilation_not_implemented"
WORKING_STATE_CACHE_DISABLE_ENV = "REDACTOR_DISABLE_WORKING_STATE_CACHE"
WORKING_STATE_CACHE_MAX_ENTRIES = 64
DELIVERY_REDACTION_RECORDS_SCHEMA_VERSION = "case_delivery_redaction_records.v1"
DELIVERY_REDACTION_RECORDS_FILENAME = "reviewed_regions.jsonl"
DELIVERY_REDACTION_RECORDS_MANIFEST_FILENAME = "redaction_manifest.json"

_WORKING_STATE_CACHE_LOCK = threading.RLock()
_WORKING_STATE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_WORKING_STATE_CACHE_STATS: dict[str, int] = {
    "hits": 0,
    "misses": 0,
}

POLICY_REDACT = privacy_entities.POLICY_REDACT
POLICY_KEEP_VISIBLE = privacy_entities.POLICY_KEEP_VISIBLE
POLICY_NOT_SENSITIVE = privacy_entities.POLICY_NOT_SENSITIVE
POLICY_REVIEW_REQUIRED = privacy_entities.POLICY_REVIEW_REQUIRED
SUPPRESSING_POLICIES = {POLICY_KEEP_VISIBLE, POLICY_NOT_SENSITIVE}


def task_decisions_by_case_from_payload(
    payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    return normalize_decisions_by_case(
        {
            "decisions": payload.get("task_decisions") or payload.get("decisions") or {},
            "decisions_by_case": payload.get("task_decisions_by_case")
            or payload.get("decisions_by_case")
            or {},
        }
    )


def task_resolutions_by_case_from_payload(
    payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    return normalize_decisions_by_case(
        {
            "decisions": payload.get("task_resolutions") or payload.get("resolutions") or {},
            "decisions_by_case": payload.get("task_resolutions_by_case")
            or payload.get("resolutions_by_case")
            or {},
        }
    )


def task_decision_for_case(
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    *,
    case_id: str,
    task_id: str,
) -> dict[str, Any] | None:
    return current_decision_for_task(decisions_by_case, case_id=case_id, task_id=task_id)


def task_resolution_for_case(
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
    *,
    case_id: str,
    task_id: str,
) -> dict[str, Any] | None:
    return current_decision_for_task(resolutions_by_case, case_id=case_id, task_id=task_id)


def set_task_decision_for_case(
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    *,
    case_id: str,
    task_id: str,
    decision: dict[str, Any],
) -> None:
    decisions_by_case.setdefault(str(case_id), {})[str(task_id)] = dict(decision) | {
        "case_id": str(case_id),
        "task_id": str(task_id),
    }


def set_task_resolution_for_case(
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
    *,
    case_id: str,
    task_id: str,
    resolution: dict[str, Any],
) -> None:
    resolutions_by_case.setdefault(str(case_id), {})[str(task_id)] = dict(resolution) | {
        "case_id": str(case_id),
        "task_id": str(task_id),
    }


def effective_task_state_for_case(
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
    *,
    case_id: str,
    task_id: str,
) -> dict[str, Any] | None:
    return task_resolution_for_case(
        resolutions_by_case,
        case_id=case_id,
        task_id=task_id,
    ) or task_decision_for_case(
        decisions_by_case,
        case_id=case_id,
        task_id=task_id,
    )


def decision_count_by_case(decisions_by_case: dict[str, dict[str, Any]]) -> int:
    return sum(len(dict(case_decisions)) for case_decisions in decisions_by_case.values())


def manual_task_key(case_id: str, task_id: str) -> tuple[str, str]:
    return (str(case_id), str(task_id))


def case_scoped_task_id(task: dict[str, Any], fallback_case_id: str = "") -> tuple[str, str] | None:
    task_id = str(task.get("review_task_id") or task.get("task_id") or "")
    case_id = str(task.get("case_id") or fallback_case_id or "")
    if not case_id or not task_id:
        return None
    return manual_task_key(case_id, task_id)


def non_durable_decisions_by_case(
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, case_decisions in sorted(decisions_by_case.items()):
        rows = {
            task_id: dict(decision)
            for task_id, decision in sorted(dict(case_decisions).items())
            if not durable_edit_verification_task_id(str(task_id))
        }
        if rows:
            filtered[str(case_id)] = rows
    return filtered


def non_durable_resolutions_by_case(
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return non_durable_decisions_by_case(resolutions_by_case)


def media_affecting_decisions_by_case(
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, case_decisions in sorted(non_durable_decisions_by_case(decisions_by_case).items()):
        rows = {
            task_id: dict(decision)
            for task_id, decision in sorted(case_decisions.items())
            if str(dict(decision).get("action") or "") == ACTION_NOT_SENSITIVE
        }
        if rows:
            filtered[case_id] = rows
    return filtered


def safe_case_manifest_path(package_root: Path, case_id: str) -> Path:
    return case_ids.case_manifest_path(package_root, case_id)


def require_safe_case_manifest_path(package_root: Path, case_id: str) -> Path:
    path = safe_case_manifest_path(package_root, case_id)
    if not path.exists():
        raise FileNotFoundError(f"case manifest not found: {case_ids.validate_case_id(case_id)}")
    return path


def safe_case_region_path(package_root: Path, case_id: str) -> Path:
    return case_ids.contained_path(
        package_root,
        "regions",
        f"{case_ids.validate_case_id(case_id)}_current_regions.jsonl",
    )


def workspace_manifest_payload(package_root: Path) -> dict[str, Any]:
    path = package_root.resolve() / "workspace_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_reviewed_publication_state(package_root: Path) -> dict[str, Any]:
    manifest = workspace_manifest_payload(package_root)
    publication = manifest.get("latest_reviewed_publication")
    if isinstance(publication, dict):
        return dict(publication)
    return {"status": "supported", "schema_version": "latest_reviewed_publication_capability.v1"}


def latest_reviewed_publication_deferred(package_root: Path) -> bool:
    return latest_reviewed_publication_state(package_root).get("status") == "deferred"


def latest_reviewed_publication_supported(package_root: Path) -> bool:
    manifest = workspace_manifest_payload(package_root)
    publication = latest_reviewed_publication_state(package_root)
    return manifest.get("workspace_kind") == "unified_case_review_workspace" and publication.get(
        "status"
    ) in {"supported", "ready"}


def latest_reviewed_publication_deferred_plan(
    *,
    case_id: str,
    before_digest: dict[str, Any],
    after_digest: dict[str, Any],
    change_summary: dict[str, Any],
    operation: str,
    published_hash: str | None,
) -> dict[str, Any]:
    after_hash = str(after_digest.get("effective_regions_hash") or "")
    before_hash = str(before_digest.get("effective_regions_hash") or "")
    return {
        "affected_source_frame_intervals": change_summary["changed_source_frame_intervals"],
        "after_effective_region_count": int(after_digest.get("region_count") or 0),
        "after_effective_regions_hash": after_hash,
        "before_effective_region_count": int(before_digest.get("region_count") or 0),
        "before_effective_regions_hash": before_hash,
        "case_id": case_id,
        "changed_source_frame_count": change_summary["changed_source_frame_count"],
        "changed_source_frame_intervals": change_summary["changed_source_frame_intervals"],
        "changed_source_frame_percentage": change_summary["changed_source_frame_percentage"],
        "error_code": LATEST_REVIEWED_PUBLICATION_DEFERRED_ERROR_CODE,
        "full_case_fallback_required": False,
        "latest_reviewed_publication_reason": LATEST_REVIEWED_PUBLICATION_DEFERRED_REASON,
        "latest_reviewed_publication_status": "deferred",
        "materialization_required": False,
        "materialization_strategy": "deferred",
        "operation": operation,
        "planned_frames_to_encode": 0,
        "planned_frames_to_render": 0,
        "publication_status": "deferred",
        "published_effective_regions_hash": published_hash,
        "schema_version": "materialization_planning_result.v1",
    }


def derive_working_state(
    package_root: Path,
    session_dir: Path,
    *,
    copy_cached: bool = True,
    write_snapshot: bool = True,
) -> dict[str, Any]:
    """Return deterministic session working state plus any valid materialized versions."""
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    ensure_session_layout(session_dir)
    cache_key = _working_state_cache_key(package_root, session_dir)
    if _working_state_cache_enabled():
        with _WORKING_STATE_CACHE_LOCK:
            cached = _WORKING_STATE_CACHE.get(cache_key)
            if cached is not None:
                _WORKING_STATE_CACHE_STATS["hits"] += 1
                state = copy.deepcopy(cached) if copy_cached else cached
                if write_snapshot:
                    write_state_snapshot(session_dir, state)
                return state
            _WORKING_STATE_CACHE_STATS["misses"] += 1
    baseline = package_baseline(package_root)
    entities = load_entity_index(package_root)
    projection = derive_active_review_projection(
        package_root=package_root,
        session_dir=session_dir,
        entities=entities,
    )
    projection_hashes = active_projection_hashes(projection)
    projection_hash = projection_hashes[ACTIVE_PROJECTION_HASH_SCHEMA_VERSION]
    if latest_reviewed_publication_supported(package_root):
        projection = projection_with_authoritative_latest_reviewed(
            package_root=package_root,
            session_dir=session_dir,
            projection=projection,
            projection_hashes=projection_hashes,
        )
    corrections = load_active_corrections(
        session_dir,
        active_correction_ids=set(projection["active_correction_ids"]),
    )
    checkpoints = load_checkpoints(session_dir)
    chain_head = current_event_chain_head(session_dir)
    audit_projection_hash = stable_hash(
        {
            "active_correction_ids": sorted(
                str(item) for item in list(projection.get("active_correction_ids") or [])
            ),
            "active_manual_tasks": sorted(
                list(projection.get("active_manual_tasks") or []),
                key=stable_hash,
            ),
            "object_policy_overrides": dict(projection.get("object_policy_overrides") or {}),
            "sweep_acknowledgements": dict(projection.get("sweep_acknowledgements") or {}),
            "task_decisions_by_case": task_decisions_by_case_from_payload(projection),
            "task_resolutions_by_case": task_resolutions_by_case_from_payload(projection),
        }
    )
    state = {
        "active_correction_ids": sorted(correction.correction_id for correction in corrections),
        "active_manual_tasks": projection["active_manual_tasks"],
        "active_projection_hash": projection_hash,
        "active_projection_hash_schema_version": ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
        "active_projection_hashes": projection_hashes,
        "audit_projection_hash": audit_projection_hash,
        "baseline": baseline,
        "checkpoints": checkpoints,
        "current_event_chain_head": chain_head,
        "deterministic_state_inputs": {
            "audit_projection_hash": audit_projection_hash,
            "correction_hashes": sorted(correction.correction_hash() for correction in corrections),
            "decision_hash": stable_hash(projection["task_decisions_by_case"]),
            "decision_identity": "case_id_task_id.v1",
            "object_policy_hash": stable_hash(projection["object_policy_overrides"]),
            "projection_hash": projection_hash,
            "projection_hash_schema_version": ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
            "task_resolution_hash": stable_hash(projection["task_resolutions_by_case"]),
        },
        "manual_corrections": [
            correction.to_dict() | {"correction_hash": correction.correction_hash()}
            for correction in corrections
        ],
        "materialized_latest_reviewed": projection["materialized_latest_reviewed"],
        "object_policy_overrides": projection["object_policy_overrides"],
        "schema_version": WORKING_STATE_SCHEMA_VERSION,
        "suppressions": policy_suppressions_from_overrides(projection["object_policy_overrides"]),
        "sweep_acknowledgements": projection["sweep_acknowledgements"],
        "task_decisions": projection["task_decisions"],
        "task_decisions_by_case": projection["task_decisions_by_case"],
        "task_resolutions": projection["task_resolutions"],
        "task_resolutions_by_case": projection["task_resolutions_by_case"],
    }
    state["state_hash"] = state_hash(state)
    if write_snapshot:
        write_state_snapshot(session_dir, state)
    if _working_state_cache_enabled():
        with _WORKING_STATE_CACHE_LOCK:
            _WORKING_STATE_CACHE[cache_key] = copy.deepcopy(state)
            while len(_WORKING_STATE_CACHE) > WORKING_STATE_CACHE_MAX_ENTRIES:
                _WORKING_STATE_CACHE.pop(next(iter(_WORKING_STATE_CACHE)))
    return state


def working_state_cache_stats() -> dict[str, int]:
    with _WORKING_STATE_CACHE_LOCK:
        return dict(_WORKING_STATE_CACHE_STATS) | {
            "entries": len(_WORKING_STATE_CACHE),
        }


def clear_working_state_cache() -> None:
    with _WORKING_STATE_CACHE_LOCK:
        _WORKING_STATE_CACHE.clear()
        for key in _WORKING_STATE_CACHE_STATS:
            _WORKING_STATE_CACHE_STATS[key] = 0


def _working_state_cache_enabled() -> bool:
    return os.environ.get(WORKING_STATE_CACHE_DISABLE_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }


def _working_state_cache_key(package_root: Path, session_dir: Path) -> tuple[Any, ...]:
    from redactor import reviewed_regions

    return (
        str(package_root),
        str(session_dir),
        stable_hash(reviewed_regions.render_configuration_payload(render_settings())),
        current_event_chain_head(session_dir),
        _file_signatures(
            [
                package_root / "workspace_manifest.json",
                package_root / "reviewer_tasks.json",
                package_root / "review_entities.json",
                package_root / "latest_reviewed" / "current.json",
                session_dir / "events.jsonl",
                session_dir / "corrections.jsonl",
                session_dir / "review_session.json",
            ]
        ),
        _glob_file_signatures(package_root / "cases", "*.json"),
        _glob_file_signatures(package_root / "regions", "*.jsonl"),
        _glob_file_signatures(session_dir / "checkpoints", "*.json"),
        _glob_file_signatures(session_dir / "latest_reviewed", "**/*"),
    )


def _glob_file_signatures(root: Path, pattern: str) -> tuple[tuple[Any, ...], ...]:
    if not root.exists():
        return ()
    return _file_signatures(path for path in sorted(root.glob(pattern)) if path.is_file())


def _file_signatures(paths: Any) -> tuple[tuple[Any, ...], ...]:
    signatures: list[tuple[Any, ...]] = []
    for path in paths:
        file_path = Path(path)
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            signatures.append((str(file_path), None, None, None))
            continue
        signatures.append(
            (
                str(file_path.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(getattr(stat, "st_ino", 0)),
            )
        )
    return tuple(signatures)


def package_baseline(package_root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case_path in sorted((package_root / "cases").glob("*.json")):
        case_ids.validate_case_id(case_path.stem)
        manifest = case_manifests.load_case_manifest(case_path)
        manifest_case_id = case_ids.validate_case_id(manifest.case_id)
        if manifest_case_id != case_path.stem:
            raise ValueError(
                f"case manifest ID {manifest_case_id!r} does not match file stem {case_path.stem!r}"
            )
        base_region_path = safe_case_region_path(package_root, manifest_case_id)
        cases[manifest_case_id] = {
            "automatic_regions": base_region_path.relative_to(package_root).as_posix()
            if base_region_path.exists()
            else None,
            "automatic_regions_sha256": media.sha256_file(base_region_path)
            if base_region_path.exists()
            else None,
            "baseline_media": [
                role.to_dict()
                for role in manifest.video_roles
                if role.role in {"automatic", "current_result", "latest_reviewed"}
            ],
            "frame_count": manifest.frame_count,
            "source_frame_end": manifest.source_frame_end,
            "source_frame_start": manifest.source_frame_start,
            "source_video_hash": manifest.source_video_hash,
        }
    return {
        "cases": cases,
        "schema_version": "review_working_state_baseline.v1",
    }


def load_valid_state_snapshot(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / "state" / "current_state.json"
    if not path.exists():
        return None
    payload = read_json(path)
    expected = payload.get("state_hash")
    if expected != state_hash(payload):
        return None
    return payload


def write_state_snapshot(session_dir: Path, state: dict[str, Any]) -> None:
    ensure_session_layout(session_dir)
    write_json(session_dir / "state" / "current_state.json", state)


def state_hash(state: dict[str, Any]) -> str:
    payload = dict(state)
    payload.pop("state_hash", None)
    return stable_hash(payload)


def current_event_chain_head(session_dir: Path) -> str | None:
    events = load_events(session_dir)
    return str(events[-1]["event_hash"]) if events else None


def load_entity_index(package_root: Path) -> dict[str, list[dict[str, Any]]]:
    index_path = package_root / "review_entities.json"
    if not index_path.exists():
        return {}
    payload = read_json(index_path)
    return {
        str(case_id): [dict(entity) for entity in entities or []]
        for case_id, entities in dict(payload.get("cases") or {}).items()
    }


def derive_active_review_overrides(
    *,
    package_root: Path,
    session_dir: Path,
    entities: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], set[str] | None]:
    projection = derive_active_review_projection(
        package_root=package_root,
        session_dir=session_dir,
        entities=entities,
    )
    return (
        dict(projection["object_policy_overrides"]),
        set(projection["active_correction_ids"]),
    )


def derive_active_review_projection(
    *,
    package_root: Path,
    session_dir: Path,
    entities: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    baseline_overrides = baseline_policy_overrides(entities)
    overrides: dict[str, dict[str, Any]] = dict(baseline_overrides)
    active_correction_ids: set[str] = set()
    correction_case_by_id = correction_case_index(session_dir)
    decisions_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    manual_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    materialized: dict[str, dict[str, Any]] = {}
    sweep_acknowledgements: dict[str, dict[str, Any]] = {}
    entity_case_by_id = entity_case_index(entities)
    checkpoint_cache: dict[str, dict[str, Any]] = {}
    for event in load_events(session_dir):
        event_type = str(event.get("event_type") or "")
        after = dict(event.get("after_state") or {})
        case_id = str(event.get("case_id") or after.get("case_id") or "")
        if event_type == "review_workflow_restarted":
            reset_case_ids = set(list(after.get("case_ids") or []))
            if not reset_case_ids and case_id:
                reset_case_ids.add(case_id)
            if not reset_case_ids:
                reset_case_ids = set(entities.keys())
            reset_review_workflow_from_active_projection(
                reset_case_ids,
                decisions_by_case=decisions_by_case,
                manual_tasks=manual_tasks,
                sweep_acknowledgements=sweep_acknowledgements,
            )
            continue
        if event_type in {
            "reset_to_automatic_baseline",
            "case_review_restarted_from_automatic_baseline",
            "case_edits_reset_to_automatic_baseline",
        }:
            reset_case_ids = set(list(after.get("case_ids") or []))
            if not reset_case_ids and case_id:
                reset_case_ids.add(case_id)
            if not reset_case_ids:
                reset_case_ids = set(entities.keys())
            reset_cases_from_active_projection(
                reset_case_ids,
                baseline_overrides=baseline_overrides,
                corrections_case_by_id=correction_case_by_id,
                decisions_by_case=decisions_by_case,
                entity_case_by_id=entity_case_by_id,
                manual_tasks=manual_tasks,
                materialized=materialized,
                overrides=overrides,
                resolutions_by_case=resolutions_by_case,
                active_correction_ids=active_correction_ids,
                sweep_acknowledgements=sweep_acknowledgements,
            )
            continue
        if event_type == "restore_checkpoint":
            checkpoint_id = str(after.get("checkpoint_uuid") or "")
            checkpoint = checkpoint_cache.get(checkpoint_id)
            if checkpoint is None and checkpoint_id:
                checkpoint_path = session_dir / "checkpoints" / f"{checkpoint_id}.json"
                checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {}
                checkpoint_cache[checkpoint_id] = checkpoint
            snapshot = dict((checkpoint or after).get("state_snapshot") or {})
            overrides = {
                str(entity_id): dict(entity)
                for entity_id, entity in dict(snapshot.get("object_policy_overrides") or {}).items()
            }
            for entity_id, entity in baseline_overrides.items():
                overrides.setdefault(entity_id, dict(entity))
            active_correction_ids = {
                str(item) for item in list(snapshot.get("active_correction_ids") or [])
            }
            decisions_by_case = task_decisions_by_case_from_payload(snapshot)
            resolutions_by_case = task_resolutions_by_case_from_payload(snapshot)
            manual_tasks = {
                key: dict(task)
                for task in list(snapshot.get("active_manual_tasks") or [])
                if isinstance(task, dict) and task.get("review_task_id")
                for key in [case_scoped_task_id(task, str(task.get("case_id") or ""))]
                if key is not None
            }
            sweep_acknowledgements = {
                str(item_case): dict(value)
                for item_case, value in dict(snapshot.get("sweep_acknowledgements") or {}).items()
            }
            materialized = {
                str(item_case): dict(value)
                for item_case, value in dict(
                    snapshot.get("materialized_latest_reviewed") or {}
                ).items()
            }
            continue
        if event_type == "decision_recorded":
            task_id = str(after.get("task_id") or event.get("task_id") or "")
            decision_case_id = str(after.get("case_id") or case_id or "")
            if task_id and decision_case_id:
                set_task_decision_for_case(
                    decisions_by_case,
                    case_id=decision_case_id,
                    task_id=task_id,
                    decision=dict(after)
                    | {
                        "event_hash": event.get("event_hash"),
                        "event_id": event.get("event_id"),
                    },
                )
                resolutions_by_case.get(decision_case_id, {}).pop(task_id, None)
                if not resolutions_by_case.get(decision_case_id):
                    resolutions_by_case.pop(decision_case_id, None)
            continue
        if event_type == EVENT_TASK_RESOLUTION_DERIVED:
            task_id = str(after.get("task_id") or event.get("task_id") or "")
            resolution_case_id = str(after.get("case_id") or case_id or "")
            if task_id and resolution_case_id:
                set_task_resolution_for_case(
                    resolutions_by_case,
                    case_id=resolution_case_id,
                    task_id=task_id,
                    resolution=dict(after)
                    | {
                        "event_hash": event.get("event_hash"),
                        "event_id": event.get("event_id"),
                    },
                )
            continue
        if event_type == "full_video_sweep_acknowledged" and case_id:
            sweep_acknowledgements[case_id] = dict(after) | {
                "event_hash": event.get("event_hash"),
                "event_id": event.get("event_id"),
            }
            continue
        if event_type == "case_reopened_for_review" and case_id:
            sweep_acknowledgements.pop(case_id, None)
            continue
        if event_type == "manual_issue_created":
            task = dict(after.get("task") or {})
            task_id = str(task.get("review_task_id") or event.get("task_id") or "")
            task_case_id = str(task.get("case_id") or case_id or "")
            if task_id and task_case_id:
                task.setdefault("case_id", task_case_id)
                manual_tasks[manual_task_key(task_case_id, task_id)] = task
            continue
        if event_type in {EVENT_MANUAL_CORRECTION_SAVED, EVENT_CORRECTION_SAVED_LEGACY}:
            correction_id = str(after.get("correction_id") or "")
            if correction_id:
                active_correction_ids.add(correction_id)
                if case_id:
                    correction_case_by_id[correction_id] = case_id
            continue
        if event_type == EVENT_MANUAL_CORRECTION_RETRACTED:
            correction_id = str(after.get("correction_id") or "")
            retraction_case_id = str(after.get("case_id") or case_id or "")
            if correction_id:
                active_correction_ids.discard(correction_id)
            if correction_id and retraction_case_id:
                verification_task_id = f"correction_verify_{retraction_case_id}_{correction_id}"
                manual_tasks.pop(manual_task_key(retraction_case_id, verification_task_id), None)
                decisions_by_case.get(retraction_case_id, {}).pop(verification_task_id, None)
                if not decisions_by_case.get(retraction_case_id):
                    decisions_by_case.pop(retraction_case_id, None)
                resolutions_by_case.get(retraction_case_id, {}).pop(verification_task_id, None)
                if not resolutions_by_case.get(retraction_case_id):
                    resolutions_by_case.pop(retraction_case_id, None)
            continue
        if event_type == "entity_policy_changed":
            entity = dict(after)
            entity_id = str(entity.get("entity_id") or entity.get("object_uuid") or "")
            if entity_id:
                if case_id:
                    entity.setdefault("case_id", case_id)
                entity.setdefault("reviewer_override", True)
                entity["policy_event_hash"] = event.get("event_hash")
                entity["policy_event_id"] = event.get("event_id")
                entity["policy_event_type"] = event_type
                entity["policy_timestamp_utc"] = event.get("timestamp_utc")
                overrides[entity_id] = entity
                if case_id:
                    entity_case_by_id[entity_id] = case_id
            continue
        if event_type == "bulk_entity_policy_changed":
            for entity in list(after.get("entities") or []):
                if isinstance(entity, dict):
                    entity_id = str(entity.get("entity_id") or "")
                    if entity_id:
                        updated = dict(entity)
                        if case_id:
                            updated.setdefault("case_id", case_id)
                        updated.setdefault("reviewer_override", True)
                        updated["policy_event_hash"] = event.get("event_hash")
                        updated["policy_event_id"] = event.get("event_id")
                        updated["policy_event_type"] = event_type
                        updated["policy_timestamp_utc"] = event.get("timestamp_utc")
                        overrides[entity_id] = updated
                        if case_id:
                            entity_case_by_id[entity_id] = case_id
            continue
        if event_type in {"entities_merged", "entity_merge_candidate_accepted"}:
            primary = dict(after.get("primary") or {})
            primary_id = str(primary.get("entity_id") or "")
            if primary_id:
                if case_id:
                    primary.setdefault("case_id", case_id)
                overrides[primary_id] = primary
                if case_id:
                    entity_case_by_id[primary_id] = case_id
            for retired_id in list(after.get("retired_entity_ids") or []):
                retired_key = str(retired_id)
                overrides[retired_key] = {
                    "case_id": case_id or entity_case_by_id.get(retired_key),
                    "current_status": "merged",
                    "entity_id": retired_key,
                }
                if case_id:
                    entity_case_by_id[retired_key] = case_id
            continue
        if event_type in {"entity_mixed_split", "entity_tracklet_split"}:
            parent = dict(after.get("parent") or {})
            parent_id = str(parent.get("entity_id") or "")
            if parent_id:
                if case_id:
                    parent.setdefault("case_id", case_id)
                overrides[parent_id] = parent
                if case_id:
                    entity_case_by_id[parent_id] = case_id
            for child in list(after.get("split_entities") or []):
                if not isinstance(child, dict):
                    continue
                child_id = str(child.get("entity_id") or "")
                if not child_id:
                    continue
                updated = dict(child)
                if case_id:
                    updated.setdefault("case_id", case_id)
                overrides[child_id] = updated
                if case_id:
                    entity_case_by_id[child_id] = case_id
            continue
        if event_type in {
            "reviewed_media_materialization_completed",
            "latest_reviewed_materialized",
        }:
            latest = dict(after.get("latest_reviewed") or {})
            accepted_projection_hash = str(
                after.get("active_projection_hash")
                or after.get("accepted_working_state_revision")
                or ""
            )
            accepted_projection_schema = active_projection_pointer_schema(after)
            hash_manual_tasks = {key: dict(value) for key, value in manual_tasks.items()}
            hash_decisions_by_case = {
                item_case: {task_id: dict(decision) for task_id, decision in case_decisions.items()}
                for item_case, case_decisions in decisions_by_case.items()
            }
            hash_resolutions_by_case = {
                item_case: {
                    task_id: dict(resolution) for task_id, resolution in case_resolutions.items()
                }
                for item_case, case_resolutions in resolutions_by_case.items()
            }
            decorate_correction_verification_tasks(
                package_root=package_root,
                session_dir=session_dir,
                manual_tasks=hash_manual_tasks,
                decisions_by_case=hash_decisions_by_case,
                materialized=materialized,
            )
            current_projection_hash = active_projection_hash(
                {
                    "active_correction_ids": sorted(active_correction_ids),
                    "active_manual_tasks": sorted(
                        hash_manual_tasks.values(),
                        key=lambda task: (
                            str(task.get("case_id") or ""),
                            int(task.get("source_frame_start") or 0),
                            str(task.get("review_task_id") or ""),
                        ),
                    ),
                    "object_policy_overrides": overrides,
                    "sweep_acknowledgements": sweep_acknowledgements,
                    "task_decisions_by_case": hash_decisions_by_case,
                    "task_resolutions_by_case": hash_resolutions_by_case,
                },
                schema_version=accepted_projection_schema,
            )
            if accepted_projection_hash and accepted_projection_hash != current_projection_hash:
                continue
            if case_id and latest:
                latest["accepted_working_state_revision"] = accepted_projection_hash
                latest["effective_regions_hash"] = after.get("effective_regions_hash")
                latest["published_working_state_revision"] = (
                    after.get("published_working_state_revision")
                    or after.get("active_projection_hash")
                    or accepted_projection_hash
                )
                latest["target_effective_regions_hash"] = after.get("target_effective_regions_hash")
                latest["media_validation_passed"] = after.get("media_validation_passed")
                latest["correction_render_verifications"] = (
                    after.get("correction_render_verifications") or {}
                )
                materialized[case_id] = latest
            continue
    decorate_correction_verification_tasks(
        package_root=package_root,
        session_dir=session_dir,
        manual_tasks=manual_tasks,
        decisions_by_case=decisions_by_case,
        materialized=materialized,
    )
    apply_entity_action_task_resolutions(
        package_root=package_root,
        decisions_by_case=decisions_by_case,
        overrides=overrides,
        resolutions_by_case=resolutions_by_case,
    )
    task_decisions = flatten_decisions_by_case(decisions_by_case)
    task_resolutions = flatten_decisions_by_case(resolutions_by_case)
    return {
        "active_correction_ids": sorted(active_correction_ids),
        "active_manual_tasks": sorted(
            (
                task
                for task in manual_tasks.values()
                if not object_policy_verification_task_id(str(task.get("review_task_id") or ""))
            ),
            key=lambda task: (
                str(task.get("case_id") or ""),
                int(task.get("source_frame_start") or 0),
                str(task.get("review_task_id") or ""),
            ),
        ),
        "materialized_latest_reviewed": materialized,
        "object_policy_overrides": overrides,
        "sweep_acknowledgements": sweep_acknowledgements,
        "task_decisions": task_decisions,
        "task_decisions_by_case": decisions_by_case,
        "task_resolutions": task_resolutions,
        "task_resolutions_by_case": resolutions_by_case,
    }


def apply_entity_action_task_resolutions(
    *,
    package_root: Path,
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    overrides: dict[str, dict[str, Any]],
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
) -> None:
    tasks_path = package_root / "reviewer_tasks.json"
    if not tasks_path.exists():
        return
    tasks_payload = read_json(tasks_path)
    for task in list(tasks_payload.get("tasks") or []):
        if not isinstance(task, dict) or task.get("technical_only"):
            continue
        task_id = str(task.get("review_task_id") or "")
        case_id = str(task.get("case_id") or "")
        if not task_id or not case_id:
            continue
        if durable_edit_verification_task_id(task_id):
            continue
        entity_ids = [str(item) for item in list(task.get("target_entity_ids") or []) if item]
        if len(set(entity_ids)) != 1:
            continue
        entity_id = entity_ids[0]
        entity = dict(overrides.get(entity_id) or {})
        if str(entity.get("case_id") or case_id) != case_id:
            continue
        if entity.get("reviewer_override") is not True:
            continue
        policy = inherited_policy(entity)
        if policy not in SUPPRESSING_POLICIES:
            continue
        action = "not_sensitive" if policy == POLICY_NOT_SENSITIVE else "looks_correct"
        if action not in set(task.get("allowed_actions") or []):
            continue
        resolution = {
            "action": action,
            "case_id": case_id,
            "decision_status": "green",
            "derived_from_entity_action": True,
            "entity_id": entity_id,
            "event_hash": entity.get("policy_event_hash"),
            "event_id": entity.get("policy_event_id"),
            "policy": policy,
            "resolved_at_utc": entity.get("policy_timestamp_utc"),
            "resolved_by": "derived_from_entity_action",
            "task_id": task_id,
        }
        set_task_resolution_for_case(
            resolutions_by_case,
            case_id=case_id,
            task_id=task_id,
            resolution=resolution,
        )


def projection_with_authoritative_latest_reviewed(
    *,
    package_root: Path,
    session_dir: Path,
    projection: dict[str, Any],
    projection_hashes: dict[str, str],
) -> dict[str, Any]:
    from redactor import reviewed_regions

    materialized: dict[str, dict[str, Any]] = {}
    projection_state = dict(projection) | {
        "active_projection_hash": projection_hashes[ACTIVE_PROJECTION_HASH_SCHEMA_VERSION],
        "active_projection_hash_schema_version": ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
        "active_projection_hashes": projection_hashes,
    }
    projection_state.setdefault("state_hash", state_hash(projection_state))
    for case_id in sorted((dict(package_baseline(package_root).get("cases") or {})).keys()):
        latest = reviewed_regions.current_latest_reviewed_role(
            package_root,
            session_dir,
            case_id=case_id,
            active_projection_hashes=projection_hashes,
            working_state=projection_state,
        )
        if latest:
            materialized[case_id] = latest
    return dict(projection) | {"materialized_latest_reviewed": materialized}


def active_projection_payload(
    projection: dict[str, Any],
    *,
    schema_version: str = ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
    normalize_legacy_non_media: bool = False,
) -> dict[str, Any]:
    if schema_version not in SUPPORTED_ACTIVE_PROJECTION_HASH_SCHEMA_VERSIONS:
        raise ValueError(f"unknown active projection hash schema: {schema_version}")
    render_tasks = [
        dict(task)
        for task in list(projection.get("active_manual_tasks") or [])
        if not durable_edit_verification_task_id(str(dict(task).get("review_task_id") or ""))
    ]
    render_decisions_by_case = media_affecting_decisions_by_case(
        task_decisions_by_case_from_payload(projection)
    )
    render_resolutions_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    payload = {
        "active_correction_ids": sorted(
            str(item) for item in list(projection.get("active_correction_ids") or [])
        ),
        "active_manual_tasks": sorted(
            render_tasks,
            key=lambda task: (
                str(task.get("case_id") or ""),
                int(task.get("source_frame_start") or 0),
                str(task.get("review_task_id") or ""),
            ),
        ),
        "object_policy_overrides": dict(projection.get("object_policy_overrides") or {}),
    }
    if schema_version == LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION:
        payload["task_decisions"] = flatten_decisions_by_case(render_decisions_by_case)
        payload["sweep_acknowledgements"] = (
            {}
            if normalize_legacy_non_media
            else dict(projection.get("sweep_acknowledgements") or {})
        )
    else:
        payload["task_decisions_by_case"] = render_decisions_by_case
        payload["task_resolutions_by_case"] = render_resolutions_by_case
    return payload


def active_projection_hash(
    projection: dict[str, Any],
    *,
    schema_version: str = ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
    normalize_legacy_non_media: bool = False,
) -> str:
    return stable_hash(
        active_projection_payload(
            projection,
            schema_version=schema_version,
            normalize_legacy_non_media=normalize_legacy_non_media,
        )
    )


def active_projection_hashes(projection: dict[str, Any]) -> dict[str, str]:
    return {
        schema_version: active_projection_hash(projection, schema_version=schema_version)
        for schema_version in sorted(SUPPORTED_ACTIVE_PROJECTION_HASH_SCHEMA_VERSIONS)
    } | {
        LEGACY_ACTIVE_PROJECTION_COMPATIBILITY_HASH_KEY: active_projection_hash(
            projection,
            schema_version=LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
            normalize_legacy_non_media=True,
        )
    }


def active_projection_pointer_schema(payload: dict[str, Any]) -> str:
    schema_version = str(
        payload.get("active_projection_hash_schema_version")
        or payload.get("working_state_projection_schema_version")
        or payload.get("projection_hash_schema_version")
        or ""
    )
    if not schema_version:
        return LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION
    if schema_version not in SUPPORTED_ACTIVE_PROJECTION_HASH_SCHEMA_VERSIONS:
        raise ValueError(f"unknown active projection hash schema: {schema_version}")
    return schema_version


def active_projection_hash_for_pointer(
    projection_hashes: dict[str, str],
    payload: dict[str, Any],
) -> str:
    schema_version = active_projection_pointer_schema(payload)
    try:
        return str(projection_hashes[schema_version])
    except KeyError as exc:
        raise ValueError(f"missing active projection hash for schema: {schema_version}") from exc


def active_projection_hash_candidates_for_pointer(
    projection_hashes: dict[str, str],
    payload: dict[str, Any],
) -> tuple[str, ...]:
    schema_version = active_projection_pointer_schema(payload)
    current_hash = active_projection_hash_for_pointer(projection_hashes, payload)
    if schema_version == LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION:
        compatibility_hash = projection_hashes.get(LEGACY_ACTIVE_PROJECTION_COMPATIBILITY_HASH_KEY)
        if compatibility_hash and compatibility_hash != current_hash:
            return (current_hash, str(compatibility_hash))
    return (current_hash,)


def baseline_policy_overrides(
    entities: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for case_entities in entities.values():
        for entity in case_entities:
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                continue
            inherited = inherited_policy(entity)
            if inherited in {POLICY_KEEP_VISIBLE, POLICY_REDACT, POLICY_REVIEW_REQUIRED}:
                overrides[entity_id] = dict(entity) | {
                    "current_policy": inherited,
                    "inherited_source_policy": inherited,
                    "reviewer_override": False,
                }
    return overrides


def entity_case_index(entities: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for case_id, case_entities in entities.items():
        for entity in case_entities:
            entity_id = str(entity.get("entity_id") or "")
            if entity_id:
                index[entity_id] = str(entity.get("case_id") or case_id)
    return index


def correction_case_index(session_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for correction in load_corrections_jsonl(session_dir / "corrections.jsonl"):
        index[correction.correction_id] = correction.case_id
    return index


def reset_cases_from_active_projection(
    case_ids: set[str],
    *,
    baseline_overrides: dict[str, dict[str, Any]],
    corrections_case_by_id: dict[str, str],
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    entity_case_by_id: dict[str, str],
    manual_tasks: dict[tuple[str, str], dict[str, Any]],
    materialized: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
    active_correction_ids: set[str],
    sweep_acknowledgements: dict[str, dict[str, Any]],
) -> None:
    for case_id in list(decisions_by_case):
        if case_id in case_ids:
            decisions_by_case.pop(case_id, None)
    for case_id in list(resolutions_by_case):
        if case_id in case_ids:
            resolutions_by_case.pop(case_id, None)
    for task_key, task in list(manual_tasks.items()):
        if str(task.get("case_id") or "") in case_ids:
            manual_tasks.pop(task_key, None)
    for correction_id in list(active_correction_ids):
        if corrections_case_by_id.get(correction_id) in case_ids:
            active_correction_ids.discard(correction_id)
    for entity_id, entity in list(overrides.items()):
        if override_belongs_to_any_case(entity_id, entity, entity_case_by_id, case_ids):
            overrides.pop(entity_id, None)
    for entity_id, entity in baseline_overrides.items():
        entity_case = str(entity.get("case_id") or entity_case_by_id.get(entity_id) or "")
        if entity_case in case_ids:
            overrides[entity_id] = dict(entity)
    for case_id in case_ids:
        materialized.pop(case_id, None)
        sweep_acknowledgements.pop(case_id, None)


def override_belongs_to_any_case(
    entity_id: str,
    entity: dict[str, Any],
    entity_case_by_id: dict[str, str],
    case_ids: set[str],
) -> bool:
    candidate_cases = {
        str(entity.get("case_id") or ""),
        str(entity_case_by_id.get(entity_id) or ""),
    }
    for source_id in list(entity.get("source_entity_ids") or []):
        candidate_cases.add(str(entity_case_by_id.get(str(source_id)) or ""))
    retired_ids = list(entity.get("retired_entity_ids") or entity.get("retired_entity_uuids") or [])
    for retired_id in retired_ids:
        candidate_cases.add(str(entity_case_by_id.get(str(retired_id)) or ""))
    return any(candidate in case_ids for candidate in candidate_cases if candidate)


def reset_review_workflow_from_active_projection(
    case_ids: set[str],
    *,
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    manual_tasks: dict[tuple[str, str], dict[str, Any]],
    sweep_acknowledgements: dict[str, dict[str, Any]],
) -> None:
    for item_case_id, case_decisions in list(decisions_by_case.items()):
        if item_case_id not in case_ids:
            continue
        for task_id in list(case_decisions):
            if durable_edit_verification_task_id(task_id):
                continue
            case_decisions.pop(task_id, None)
        if not case_decisions:
            decisions_by_case.pop(item_case_id, None)
    for task_key, task in list(manual_tasks.items()):
        if str(task.get("case_id") or "") not in case_ids:
            continue
        if durable_edit_verification_task_id(str(task.get("review_task_id") or "")):
            continue
        manual_tasks.pop(task_key, None)
    for case_id in case_ids:
        sweep_acknowledgements.pop(case_id, None)


def durable_edit_verification_task_id(task_id: str) -> bool:
    return task_id.startswith("correction_verify_") or object_policy_verification_task_id(task_id)


def object_policy_verification_task_id(task_id: str) -> bool:
    return task_id.startswith("object_policy_verify_")


def correction_id_from_verification_task(task: dict[str, Any]) -> str:
    correction_id = str(task.get("correction_id") or "")
    if correction_id:
        return correction_id
    task_id = str(task.get("review_task_id") or "")
    case_id = str(task.get("case_id") or "")
    prefix = f"correction_verify_{case_id}_"
    if case_id and task_id.startswith(prefix):
        return task_id.removeprefix(prefix)
    if task_id.startswith("correction_verify_"):
        parts = task_id.split("_", 3)
        if len(parts) == 4:
            return parts[3]
    return ""


def correction_render_verification_for_latest(
    latest: dict[str, Any],
    correction_id: str,
) -> dict[str, Any]:
    verifications = dict(latest.get("correction_render_verifications") or {})
    verification = dict(verifications.get(correction_id) or {})
    return verification


def correction_verification_status(
    *,
    package_root: Path,
    task: dict[str, Any],
    materialized: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if task.get("normal_task_class") != "correction_verification":
        return {"actionable": True, "publication_status": "not_applicable"}
    if (
        task.get("publication_status") == "deferred"
        or task.get("latest_reviewed_publication_status") == "deferred"
        or latest_reviewed_publication_deferred(package_root)
    ):
        return {
            "actionable": True,
            "decision_state": "pending",
            "publication_status": "deferred",
            "reason": (
                "Correction saved; reviewed output publication is deferred until the "
                "reviewed-region compiler is available."
            ),
        }
    case_id = str(task.get("case_id") or "")
    correction_id = correction_id_from_verification_task(task)
    latest = dict(materialized.get(case_id) or {})
    target_hash = str(task.get("target_effective_regions_hash") or "")
    published_hash = str(latest.get("effective_regions_hash") or "")
    published_path = str(latest.get("path") or "")
    published_revision = latest.get("published_working_state_revision")
    media_exists = bool(published_path and (package_root / published_path).exists())
    media_validation_passed = latest.get("media_validation_passed") is True
    verification = correction_render_verification_for_latest(latest, correction_id)
    verification_present = bool(verification)
    decoded_verification_passed = verification.get("passed") is True
    published_matches_target = bool(target_hash and published_hash == target_hash)
    if not target_hash:
        return {
            "actionable": False,
            "decision_state": "waiting_for_latest_reviewed",
            "publication_status": "waiting",
            "reason": "Latest reviewed media is not current yet.",
        }
    if not latest:
        pending_status = str(task.get("publication_status") or "waiting")
        return {
            "actionable": False,
            "decision_state": "waiting_for_latest_reviewed",
            "publication_status": pending_status,
            "reason": (
                "Correction saved; build the reviewed video before verification."
                if pending_status == "pending_explicit_build"
                else "Latest reviewed media is not current yet."
            ),
        }
    if not media_exists:
        return {
            "actionable": False,
            "decision_state": "waiting_for_latest_reviewed",
            "publication_status": "missing_media",
            "reason": "Latest reviewed media is missing.",
        }
    if not media_validation_passed:
        return {
            "actionable": False,
            "decision_state": "materialization_failed",
            "publication_status": "failed",
            "reason": "Materialization failed.",
        }
    if not published_matches_target and not verification_present:
        return {
            "actionable": False,
            "decision_state": "waiting_for_latest_reviewed",
            "publication_status": "superseded_or_rendering",
            "published_effective_regions_hash": published_hash or None,
            "reason": "Latest reviewed media is not current yet.",
        }
    if not decoded_verification_passed:
        return {
            "actionable": False,
            "decision_state": "materialization_failed",
            "publication_status": "failed",
            "reason": "Rendered correction verification failed.",
        }
    if not published_matches_target and decoded_verification_passed:
        return {
            "actionable": True,
            "decision_state": "pending",
            "publication_status": "published",
            "published_effective_regions_hash": published_hash,
            "published_superseded_target_hash": target_hash,
            "published_working_state_revision": published_revision,
            "reason": "Latest reviewed media is ready for verification.",
        }
    return {
        "actionable": True,
        "decision_state": "pending",
        "publication_status": "published",
        "published_effective_regions_hash": published_hash,
        "published_working_state_revision": published_revision,
        "reason": "Latest reviewed media is ready for verification.",
    }


def decorate_correction_verification_tasks(
    *,
    package_root: Path,
    session_dir: Path,
    manual_tasks: dict[tuple[str, str], dict[str, Any]],
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    materialized: dict[str, dict[str, Any]],
) -> None:
    for task_key, task in list(manual_tasks.items()):
        if dict(task).get("normal_task_class") != "correction_verification":
            continue
        task_id = str(task.get("review_task_id") or task_key[1])
        task_case_id = str(task.get("case_id") or task_key[0])
        status = correction_verification_status(
            package_root=package_root,
            task=task,
            materialized=materialized,
        )
        task.update(
            {
                "actionable": bool(status.get("actionable")),
                "publication_status": status.get("publication_status"),
                "published_effective_regions_hash": status.get("published_effective_regions_hash"),
                "published_working_state_revision": status.get("published_working_state_revision"),
                "verification_status": status.get("decision_state"),
            }
        )
        if status.get("actionable"):
            task["allowed_actions"] = list(CANONICAL_REVIEW_ACTIONS)
            task["decision_state"] = "pending"
            task["what_to_inspect"] = str(
                status.get("reason")
                or "Confirm the published Latest reviewed media protects the intended region."
            )
        else:
            task["allowed_actions"] = []
            task["decision_state"] = str(
                status.get("decision_state") or "waiting_for_latest_reviewed"
            )
            task["what_to_inspect"] = str(
                status.get("reason") or "Latest reviewed media is not current yet."
            )
            decisions_by_case.get(task_case_id, {}).pop(task_id, None)
            if not decisions_by_case.get(task_case_id):
                decisions_by_case.pop(task_case_id, None)
        manual_tasks[task_key] = task


def correction_verification_decision_gate(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    task_id: str,
) -> dict[str, Any]:
    state = derive_working_state(package_root, session_dir)
    task = next(
        (
            dict(item)
            for item in list(state.get("active_manual_tasks") or [])
            if dict(item).get("review_task_id") == task_id and dict(item).get("case_id") == case_id
        ),
        None,
    )
    if not task or task.get("normal_task_class") != "correction_verification":
        return {"ok": True, "task": task}
    if task.get("actionable") is True:
        return {"ok": True, "task": task}
    return {
        "error": str(task.get("what_to_inspect") or "Latest reviewed media is not current yet."),
        "ok": False,
        "task": task,
    }


def inherited_policy(entity: dict[str, Any]) -> str:
    return str(
        entity.get("current_policy")
        or entity.get("privacy_action")
        or entity.get("default_privacy_policy")
        or POLICY_REVIEW_REQUIRED
    )


def load_active_corrections(
    session_dir: Path,
    *,
    active_correction_ids: set[str] | None,
) -> list[ReviewCorrection]:
    corrections = latest_corrections(load_corrections_jsonl(session_dir / "corrections.jsonl"))
    if active_correction_ids is None:
        return corrections
    return [
        correction
        for correction in corrections
        if correction.correction_id in active_correction_ids
    ]


def latest_corrections(corrections: list[ReviewCorrection]) -> list[ReviewCorrection]:
    latest: dict[str, ReviewCorrection] = {}
    order: list[str] = []
    for correction in corrections:
        if correction.correction_id not in latest:
            order.append(correction.correction_id)
        latest[correction.correction_id] = correction
    return [latest[correction_id] for correction_id in order]


def policy_suppressions_from_overrides(
    overrides: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id, entity in sorted(overrides.items()):
        policy = inherited_policy(entity)
        if policy in SUPPRESSING_POLICIES:
            rows.append(
                {
                    "case_id": entity.get("case_id"),
                    "entity_id": entity_id,
                    "policy": policy,
                    "source_spans": [
                        entity.get("first_visible_source_frame"),
                        entity.get("last_visible_source_frame"),
                    ],
                }
            )
    return rows


def compute_effective_privacy_regions(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    include_unsaved_correction: ReviewCorrection | None = None,
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    from redactor import review_finalization

    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    if latest_reviewed_publication_supported(package_root):
        from redactor import reviewed_regions

        return reviewed_regions.compile_reviewed_regions(
            package_root,
            session_dir,
            case_id=case_id,
            include_unsaved_correction=include_unsaved_correction,
            require_blockers_resolved=False,
        )
    manifest = case_manifests.load_case_manifest(safe_case_manifest_path(package_root, case_id))
    base_path = safe_case_region_path(package_root, case_id)
    base_regions = (
        review_finalization.load_privacy_region_records(base_path) if base_path.exists() else []
    )
    state = derive_working_state(package_root, session_dir)
    entity_index = load_entity_index(package_root)
    effective_entities = effective_case_entities(entity_index.get(case_id, []), state)
    policy_regions, policy_provenance = apply_object_policies_to_regions(
        package_root=package_root,
        manifest=manifest,
        base_regions=base_regions,
        entities=effective_entities,
    )
    correction_records = [
        correction
        for correction in load_active_corrections(
            session_dir,
            active_correction_ids=set(state.get("active_correction_ids") or []),
        )
        if correction.case_id == case_id
    ]
    if include_unsaved_correction is not None:
        correction_records = [*correction_records, include_unsaved_correction]
    final_regions, correction_provenance = review_finalization.apply_corrections(
        policy_regions,
        correction_records,
        case_id=case_id,
        source_frame_offset=manifest.source_frame_start,
    )
    provenance = {
        "automatic_baseline_region_count": len(base_regions),
        "case_id": case_id,
        "correction_provenance": correction_provenance,
        "effective_region_count": len(final_regions),
        "manual_correction_count": len(correction_records),
        "object_policy_provenance": policy_provenance,
        "privacy_region_changes_by_cause": {
            "automatic_baseline": len(base_regions),
            "manual_correction_added_or_replaced": correction_provenance.get(
                "applied_corrections", []
            ),
            "object_override_added": policy_provenance.get("added_region_count", 0),
            "suppression_removed": policy_provenance.get("suppressed_region_count", 0),
        },
        "schema_version": "effective_privacy_regions.v1",
        "state_hash": state["state_hash"],
    }
    return final_regions, provenance


def effective_privacy_region_digest(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    working_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if latest_reviewed_publication_supported(package_root):
        from redactor import reviewed_regions

        return reviewed_regions.reviewed_region_digest(
            package_root,
            session_dir,
            case_id=case_id,
            require_blockers_resolved=False,
            working_state=working_state,
        )
    started = time.monotonic()
    regions, provenance = compute_effective_privacy_regions(
        package_root,
        session_dir,
        case_id=case_id,
    )
    digest = stable_hash([record.to_dict() for record in regions])
    frame_hashes = privacy_region_frame_hashes(regions)
    intervals = privacy_region_intervals(regions)
    return {
        "affected_source_frame_intervals": intervals,
        "case_id": case_id,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "effective_regions_hash": digest,
        "frame_region_hashes": frame_hashes,
        "region_count": len(regions),
        "schema_version": "effective_privacy_region_digest.v1",
        "state_hash": provenance.get("state_hash"),
    }


def baseline_effective_privacy_region_digest(package_root: Path, *, case_id: str) -> dict[str, Any]:
    from redactor import review_finalization

    package_root = package_root.resolve()
    case_id = case_ids.validate_case_id(case_id)
    base_path = safe_case_region_path(package_root, case_id)
    regions = (
        review_finalization.load_privacy_region_records(base_path) if base_path.exists() else []
    )
    return {
        "affected_source_frame_intervals": privacy_region_intervals(regions),
        "case_id": case_id,
        "duration_ms": 0,
        "effective_regions_hash": stable_hash([record.to_dict() for record in regions]),
        "frame_region_hashes": privacy_region_frame_hashes(regions),
        "region_count": len(regions),
        "schema_version": "effective_privacy_region_digest.v1",
        "state_hash": None,
    }


def published_effective_regions_hash(
    package_root: Path,
    state: dict[str, Any],
    *,
    case_id: str,
) -> str | None:
    latest = dict(state.get("materialized_latest_reviewed") or {}).get(case_id)
    if isinstance(latest, dict) and latest.get("effective_regions_hash"):
        return str(latest["effective_regions_hash"])
    if latest_reviewed_publication_supported(package_root):
        publication = latest_reviewed_publication_state(package_root)
        if publication.get("status") == "ready" and publication.get("reviewed_region_set_sha256"):
            return str(publication["reviewed_region_set_sha256"])
        return None
    return str(
        baseline_effective_privacy_region_digest(package_root, case_id=case_id)[
            "effective_regions_hash"
        ]
    )


def privacy_region_intervals(records: list[PrivacyRegionRecord]) -> list[dict[str, int]]:
    frames = sorted({privacy_region_digest_frame(record) for record in records})
    return source_frame_intervals(frames)


def source_frame_intervals(frames: list[int]) -> list[dict[str, int]]:
    frames = sorted({int(frame) for frame in frames})
    if not frames:
        return []
    intervals: list[dict[str, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        intervals.append({"source_frame_end": previous, "source_frame_start": start})
        start = previous = frame
    intervals.append({"source_frame_end": previous, "source_frame_start": start})
    return intervals


def privacy_region_frame_hashes(records: list[PrivacyRegionRecord]) -> dict[str, str]:
    by_frame: dict[int, list[PrivacyRegionRecord]] = {}
    for record in records:
        by_frame.setdefault(privacy_region_digest_frame(record), []).append(record)
    return {
        str(frame): stable_hash(
            [
                record.to_dict()
                for record in sorted(
                    frame_records,
                    key=lambda item: (
                        item.source_frame,
                        item.category,
                        item.object_id or "",
                        item.provenance,
                        item.approval_state,
                        item.render_eligible,
                        item.bbox_xyxy or (-1, -1, -1, -1),
                        stable_hash(dict(item.source_record)),
                    ),
                )
            ]
        )
        for frame, frame_records in sorted(by_frame.items())
    }


def privacy_region_digest_frame(record: PrivacyRegionRecord) -> int:
    source = dict(record.source_record)
    if record.provenance == "reviewed_correction" and source.get("source_frame") is not None:
        return int(source["source_frame"])
    return int(record.source_frame)


def effective_region_change_summary(
    package_root: Path,
    *,
    case_id: str,
    before_digest: dict[str, Any],
    after_digest: dict[str, Any],
) -> dict[str, Any]:
    case_id = case_ids.validate_case_id(case_id)
    manifest = case_manifests.load_case_manifest(safe_case_manifest_path(package_root, case_id))
    before_frame_hashes = {
        str(frame): str(value)
        for frame, value in dict(before_digest.get("frame_region_hashes") or {}).items()
    }
    after_frame_hashes = {
        str(frame): str(value)
        for frame, value in dict(after_digest.get("frame_region_hashes") or {}).items()
    }
    changed_frames = sorted(
        int(frame)
        for frame in set(before_frame_hashes) | set(after_frame_hashes)
        if before_frame_hashes.get(frame) != after_frame_hashes.get(frame)
    )
    frame_count = int(manifest.frame_count or 0)
    changed_count = len(changed_frames)
    changed_percentage = (changed_count / frame_count) if frame_count > 0 else 0.0
    return {
        "changed_source_frame_count": changed_count,
        "changed_source_frame_intervals": source_frame_intervals(changed_frames),
        "changed_source_frame_percentage": round(changed_percentage, 6),
        "schema_version": "effective_region_change_summary.v1",
    }


def effective_case_entities(
    entities: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    overrides = dict(state.get("object_policy_overrides") or {})
    case_ids = {str(entity.get("case_id") or "") for entity in entities if entity.get("case_id")}
    effective: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "")
        merged = entity | dict(overrides.get(entity_id) or {})
        seen.add(entity_id)
        if merged.get("current_status") not in {"merged", "split"}:
            effective.append(merged)
    for entity_id, override in sorted(overrides.items()):
        if entity_id in seen:
            continue
        entity = dict(override)
        if entity.get("current_status") in {"merged", "split"}:
            continue
        if not entity.get("case_id"):
            continue
        if case_ids and str(entity.get("case_id") or "") not in case_ids:
            continue
        effective.append(entity)
    return effective


def apply_object_policies_to_regions(
    *,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    base_regions: list[PrivacyRegionRecord],
    entities: list[dict[str, Any]],
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    suppressed_ids: set[str] = set()
    added_regions: list[PrivacyRegionRecord] = []
    entity_rows: list[dict[str, Any]] = []
    for entity in entities:
        policy = inherited_policy(entity)
        matched = [record for record in base_regions if region_belongs_to_entity(record, entity)]
        if policy in SUPPRESSING_POLICIES:
            for record in matched:
                suppressed_ids.update(region_identity_values(record))
        elif policy == POLICY_REDACT and entity.get("reviewer_override") is True:
            existing_keys = {region_unique_key(record) for record in base_regions}
            for record in source_backed_entity_regions(
                package_root=package_root,
                manifest=manifest,
                entity=entity,
            ):
                if region_unique_key(record) not in existing_keys:
                    added_regions.append(record)
                    existing_keys.add(region_unique_key(record))
        entity_rows.append(
            {
                "category": entity.get("category"),
                "entity_id": entity.get("entity_id"),
                "matched_baseline_region_count": len(matched),
                "policy": policy,
                "source_track_ids": sorted(entity_track_ids(entity)),
            }
        )
    kept = [
        record for record in base_regions if not (region_identity_values(record) & suppressed_ids)
    ]
    final_regions = sorted(
        kept + added_regions,
        key=lambda record: (
            record.source_frame,
            record.category,
            record.object_id or "",
            record.bbox_xyxy or (-1, -1, -1, -1),
        ),
    )
    return final_regions, {
        "added_region_count": len(added_regions),
        "entity_policies": entity_rows,
        "input_region_count": len(base_regions),
        "output_region_count": len(final_regions),
        "schema_version": "object_policy_region_application.v1",
        "suppressed_region_count": len(base_regions) - len(kept),
    }


def region_belongs_to_entity(record: PrivacyRegionRecord, entity: dict[str, Any]) -> bool:
    if entity.get("category") and str(entity.get("category")) != record.category:
        return False
    return bool(region_identity_values(record) & entity_track_variants(entity))


def entity_track_ids(entity: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "automatic_source_track_ids",
        "member_detector_tracker_ids",
        "reviewed_region_track_ids",
        "attached_fragment_track_ids",
    ):
        values.update(str(item) for item in list(entity.get(key) or []) if item is not None)
    return values


def entity_track_variants(entity: dict[str, Any]) -> set[str]:
    variants: set[str] = set()
    category = str(entity.get("category") or "")
    for track_id in entity_track_ids(entity):
        variants.update(track_variants(track_id, category=category))
    entity_id = str(entity.get("entity_id") or "")
    object_uuid = str(entity.get("object_uuid") or "")
    if entity_id:
        variants.add(entity_id)
    if object_uuid:
        variants.add(object_uuid)
    return {value for value in variants if value}


def track_variants(track_id: str, *, category: str) -> set[str]:
    raw = str(track_id)
    variants = {raw}
    stripped = raw
    for prefix in ("face_track_", "plate_track_"):
        if stripped.startswith(prefix):
            stripped = stripped.removeprefix(prefix)
    variants.add(stripped)
    if stripped.isdigit():
        variants.add(f"face_track_{int(stripped)}")
        variants.add(f"plate_track_{int(stripped):04d}")
    if category == CATEGORY_FACE:
        variants.add(f"face_track_{stripped}")
    if category == CATEGORY_PLATE and stripped.isdigit():
        variants.add(f"plate_track_{int(stripped):04d}")
    return variants


def canonical_face_track_uid(track_id: str) -> str:
    stripped = str(track_id)
    if stripped.startswith("face_track_"):
        stripped = stripped.removeprefix("face_track_")
    return f"face_track_{stripped}"


def region_identity_values(record: PrivacyRegionRecord) -> set[str]:
    values = {record.object_id or ""}
    source = dict(record.source_record)
    for key in ("track_id", "track_uid", "region_id", "object_id", "candidate_id"):
        value = source.get(key)
        if value is not None:
            values.add(str(value))
    for item in list(source.get("target_track_ids") or []):
        values.add(str(item))
        values.add(f"face_track_{item}")
    category = record.category
    expanded: set[str] = set()
    for value in values:
        expanded.update(track_variants(value, category=category))
    return {value for value in values | expanded if value}


def region_unique_key(record: PrivacyRegionRecord) -> str:
    return stable_hash(
        {
            "bbox_xyxy": record.bbox_xyxy,
            "category": record.category,
            "object_id": record.object_id,
            "source_frame": record.source_frame,
            "source_record": {
                key: value
                for key, value in dict(record.source_record).items()
                if key in {"region_id", "track_id", "track_uid", "candidate_id"}
            },
        }
    )


def source_backed_entity_regions(
    *,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    entity: dict[str, Any],
) -> list[PrivacyRegionRecord]:
    if entity.get("category") != CATEGORY_FACE:
        return []
    entity_variants = entity_track_variants(entity)
    if not entity_variants:
        return []
    regions: list[PrivacyRegionRecord] = []
    seen_source_boxes: set[tuple[int, str, tuple[int, int, int, int]]] = set()
    for source_dir in face_policy_source_dirs(package_root, manifest.case_id, manifest):
        tracks_path = source_dir / "tracking" / "tracks.jsonl"
        if not tracks_path.exists():
            continue
        for frame_payload in load_jsonl(tracks_path):
            frame_index = int(frame_payload.get("frame_index") or 0)
            for box_index, box in enumerate(list(frame_payload.get("track_boxes") or [])):
                track_id = str(box.get("track_id") or "")
                source_track_variants = track_variants(track_id, category=CATEGORY_FACE)
                if not (source_track_variants & entity_variants):
                    continue
                if not bool(box.get("render_eligible", True)):
                    continue
                bbox = box.get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    continue
                bbox_xyxy = tuple(int(value) for value in bbox)
                track_uid = canonical_face_track_uid(track_id)
                source_key = (frame_index, track_uid, bbox_xyxy)
                if source_key in seen_source_boxes:
                    continue
                seen_source_boxes.add(source_key)
                regions.append(
                    PrivacyRegionRecord(
                        approval_state=APPROVAL_ACCEPTED,
                        bbox_xyxy=bbox_xyxy,
                        category=CATEGORY_FACE,
                        object_id=track_uid,
                        provenance="review_object_policy_source_track",
                        render_eligible=True,
                        source_frame=frame_index,
                        source_record={
                            "candidate_id": box.get("candidate_id"),
                            "entity_id": entity.get("entity_id"),
                            "policy_source": display_source_path(source_dir),
                            "region_id": f"source_policy_{track_uid}_{frame_index}_{box_index}",
                            "track_id": track_id,
                            "track_uid": track_uid,
                        },
                    )
                )
    return regions


def materialize_latest_reviewed(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    accepted_revision: str | None = None,
    actor: dict[str, Any] | None = None,
    event_reason: str | None = None,
    publication_currentness_assertion: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from redactor import review_finalization

    total_start = time.monotonic()
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    ensure_session_layout(session_dir)
    case_id = case_ids.validate_case_id(case_id)
    if latest_reviewed_publication_deferred(package_root):
        raise ValueError(
            f"{LATEST_REVIEWED_PUBLICATION_DEFERRED_ERROR_CODE}: "
            "decisions and corrections are persisted, but reviewed output publication "
            "is deferred until the reviewed-region compiler is available"
        )
    if latest_reviewed_publication_supported(package_root):
        from redactor import reviewed_regions

        return reviewed_regions.materialize_latest_reviewed_revision(
            package_root,
            session_dir,
            case_id=case_id,
            accepted_revision=accepted_revision,
            actor=actor,
            event_reason=event_reason,
            publication_currentness_assertion=publication_currentness_assertion,
        )
    manifest = case_manifests.load_case_manifest(safe_case_manifest_path(package_root, case_id))
    source_path = review_finalization.resolve_package_path(package_root, manifest.source_path)
    if source_path is None or not source_path.exists():
        raise ValueError(f"{case_id}: source video is missing")
    effective_start = time.monotonic()
    final_regions, provenance = compute_effective_privacy_regions(
        package_root,
        session_dir,
        case_id=case_id,
    )
    effective_duration_ms = int((time.monotonic() - effective_start) * 1000)
    region_hash = stable_hash([record.to_dict() for record in final_regions])
    base_region_path = safe_case_region_path(package_root, case_id)
    base_region_hash = (
        stable_hash(
            [
                record.to_dict()
                for record in review_finalization.load_privacy_region_records(base_region_path)
            ]
        )
        if base_region_path.exists()
        else None
    )
    case_media_dir = session_dir / "media" / case_id
    case_media_dir.mkdir(parents=True, exist_ok=True)
    render_start = time.monotonic()
    if region_hash == base_region_hash:
        source_role = automatic_or_current_role(manifest)
        if source_role is None:
            raise ValueError(f"{case_id}: no automatic/current media role is available")
        role_source_path = package_root / source_role.path
        video_hash = media.sha256_file(role_source_path)
        output_path = case_media_dir / f"latest_reviewed_{video_hash[:16]}.mp4"
        if not output_path.exists():
            shutil.copy2(role_source_path, output_path)
        media_validation = {"copied_from_baseline": True, "passed": True}
        rendered_frames = manifest.frame_count
        rendered_regions = len(final_regions)
        audio_handling = "copied"
    else:
        source_info = media.probe_media(source_path)
        fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
        settings = render_settings()
        tmp_output = case_media_dir / "latest_reviewed_rendering.mp4"
        rendered_frames, rendered_regions, audio_handling = privacy_rendering.render_privacy_video(
            source_path,
            tmp_output,
            privacy_rendering.regions_by_frame(final_regions),
            settings=settings,
            fps=fps,
            width=source_info.video_stream.width,
            height=source_info.video_stream.height,
            source_has_audio=source_info.audio_stream is not None,
            overwrite=True,
        )
        media_validation = rendering.validate_output_media(
            source_path,
            source_info,
            tmp_output,
            expected_frame_count=rendered_frames,
            expected_width=source_info.video_stream.width,
            expected_height=source_info.video_stream.height,
            expected_fps=fps,
            source_has_audio=source_info.audio_stream is not None,
            settings=settings,
        )
        if not media_validation.get("passed"):
            tmp_output.unlink(missing_ok=True)
            raise ValueError(f"{case_id}: rendered Latest reviewed media failed validation")
        video_hash = media.sha256_file(tmp_output)
        output_path = case_media_dir / f"latest_reviewed_{video_hash[:16]}.mp4"
        if output_path.exists():
            tmp_output.unlink(missing_ok=True)
        else:
            tmp_output.replace(output_path)
    render_duration_ms = int((time.monotonic() - render_start) * 1000)
    state_for_verification = derive_working_state(package_root, session_dir)
    active_correction_ids = set(
        str(item) for item in state_for_verification.get("active_correction_ids") or []
    )
    correction_records = [
        correction
        for correction in load_active_corrections(
            session_dir,
            active_correction_ids=active_correction_ids,
        )
        if correction.case_id == case_id
    ]
    correction_render_verifications = verify_rendered_manual_corrections(
        source_path=source_path,
        output_path=output_path,
        manifest=manifest,
        corrections=correction_records,
    )
    publication_start = time.monotonic()
    regions_path = session_dir / "state" / f"{case_id}_effective_regions_{region_hash[:16]}.jsonl"
    review_finalization.write_privacy_region_records(regions_path, final_regions)
    role = {
        "accepted_working_state_revision": accepted_revision,
        "correction_render_verifications": correction_render_verifications,
        "effective_regions_hash": region_hash,
        "label": "Latest reviewed",
        "media_validation_passed": media_validation.get("passed") is True,
        "path": output_path.relative_to(package_root).as_posix(),
        "published_working_state_revision": accepted_revision,
        "role": "latest_reviewed",
        "sha256": video_hash,
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
    }
    materialization = {
        "accepted_working_state_revision": accepted_revision,
        "audio_handling": audio_handling,
        "case_id": case_id,
        "created_at_utc": media.utc_timestamp(),
        "effective_region_count": len(final_regions),
        "effective_regions": regions_path.relative_to(package_root).as_posix(),
        "effective_regions_hash": region_hash,
        "latest_reviewed": role,
        "materialized": True,
        "media_validation": media_validation,
        "correction_render_verifications": correction_render_verifications,
        "output_video": role["path"],
        "rendered_frame_count": rendered_frames,
        "rendered_region_count": rendered_regions,
        "schema_version": "latest_reviewed_materialization.v1",
        "state_hash": derive_working_state(package_root, session_dir)["state_hash"],
        "timing": {
            "accepted_working_state_revision": accepted_revision,
            "case_id": case_id,
            "effective_region_derivation_duration_ms": effective_duration_ms,
            "output_frame_count": rendered_frames,
            "publication_duration_ms": 0,
            "render_encode_duration_ms": render_duration_ms,
            "schema_version": "materialization_timing.v1",
            "total_duration_ms": 0,
        },
        "video_sha256": video_hash,
    } | provenance
    materialization["timing"]["publication_duration_ms"] = int(
        (time.monotonic() - publication_start) * 1000
    )
    materialization["timing"]["total_duration_ms"] = int((time.monotonic() - total_start) * 1000)
    write_json(session_dir / "state" / f"{case_id}_latest_reviewed_manifest.json", materialization)
    append_manual_correction_persistence_audit(
        package_root,
        case_id=case_id,
        materialization=materialization,
        provenance=provenance,
    )
    state = derive_working_state(package_root, session_dir)
    state.setdefault("materialized_latest_reviewed", {})[case_id] = role | {
        "created_at_utc": materialization["created_at_utc"],
        "effective_regions_hash": region_hash,
    }
    state["state_hash"] = state_hash(state)
    write_state_snapshot(session_dir, state)
    if actor is not None:
        append_event(
            session_dir,
            event_type="latest_reviewed_materialized",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state=None,
            after_state={
                "accepted_working_state_revision": accepted_revision,
                "active_projection_hash": accepted_revision,
                "active_projection_hash_schema_version": ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
                "case_id": case_id,
                "effective_regions_hash": region_hash,
                "latest_reviewed": role,
                "state_hash": state["state_hash"],
            },
            note=event_reason or "Latest reviewed media materialized.",
            application_hashes={"latest_reviewed_sha256": video_hash},
        )
    return materialization


def verify_rendered_manual_corrections(
    *,
    source_path: Path,
    output_path: Path,
    manifest: case_manifests.CaseManifest,
    corrections: list[ReviewCorrection],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for correction in corrections:
        if correction.operation not in {OP_ADD, OP_REPLACE}:
            continue
        boxes = interpolate_correction_boxes(correction)
        frames = sorted(boxes)
        if not frames:
            results[correction.correction_id] = {
                "passed": False,
                "reason": "correction has no renderable frames",
                "schema_version": "correction_render_verification.v1",
            }
            continue
        sample_frames = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
        samples = [
            decoded_region_difference(
                source_path=source_path,
                output_path=output_path,
                case_frame=source_frame - manifest.source_frame_start,
                source_frame=source_frame,
                bbox_xyxy=boxes[source_frame],
            )
            for source_frame in sample_frames
        ]
        results[correction.correction_id] = {
            "passed": all(sample.get("passed") is True for sample in samples),
            "sampled_source_frames": sample_frames,
            "samples": samples,
            "schema_version": "correction_render_verification.v1",
        }
    return results


def decoded_region_difference(
    *,
    source_path: Path,
    output_path: Path,
    case_frame: int,
    source_frame: int,
    bbox_xyxy: tuple[int, int, int, int],
) -> dict[str, Any]:
    source = decode_video_frame(source_path, case_frame)
    rendered = decode_video_frame(output_path, case_frame)
    if source is None or rendered is None:
        return {
            "case_frame": case_frame,
            "passed": False,
            "reason": "frame decode failed",
            "source_frame": source_frame,
        }
    height, width = rendered.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    source_roi = source[y1:y2, x1:x2].astype("float32")
    rendered_roi = rendered[y1:y2, x1:x2].astype("float32")
    diff = abs(source_roi - rendered_roi)
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "case_frame": case_frame,
        "mean_abs_diff_from_source": round(mean_abs_diff, 4),
        "passed": mean_abs_diff >= 0.5,
        "source_frame": source_frame,
    }


def decode_video_frame(path: Path, frame_index: int) -> Any:
    capture = cv.VideoCapture(str(path))
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def render_correction_preview(
    package_root: Path,
    session_dir: Path,
    *,
    correction: ReviewCorrection,
    include_unsaved_correction: bool,
) -> dict[str, Any]:
    from redactor import review_finalization

    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    correction_case_id = case_ids.validate_case_id(correction.case_id)
    manifest = case_manifests.load_case_manifest(
        safe_case_manifest_path(package_root, correction_case_id)
    )
    source_path = review_finalization.resolve_package_path(package_root, manifest.source_path)
    if source_path is None or not source_path.exists():
        return {"ok": False, "reason": "case source missing"}
    working_state = derive_working_state(package_root, session_dir)
    working_state_revision = str(
        working_state.get("active_projection_hash") or working_state.get("state_hash") or ""
    )
    source_video_hash = (
        correction.source_video_hash or manifest.source_video_hash or media.sha256_file(source_path)
    )
    local_start = max(0, correction.source_frame_start - manifest.source_frame_start)
    local_end = max(local_start, correction.source_frame_end - manifest.source_frame_start)
    fps = rational_to_float(manifest.fps) or 30.0
    preview_dir = session_dir / "previews" / correction_case_id / correction.correction_id
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
    final_regions, _provenance = compute_effective_privacy_regions(
        package_root,
        session_dir,
        case_id=correction.case_id,
        include_unsaved_correction=correction if include_unsaved_correction else None,
    )
    effective_regions_hash = stable_hash([record.to_dict() for record in final_regions])
    preview_regions = [
        replace(record, source_frame=record.source_frame - local_start)
        for record in final_regions
        if local_start <= record.source_frame <= local_end
    ]
    source_info = media.probe_media(original_preview)
    settings = render_settings()
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
    preview_hash = media.sha256_file(corrected_preview)
    token_payload = preview_token_payload(
        session_dir,
        correction,
        effective_regions_hash=effective_regions_hash,
        preview_sha256=preview_hash,
        source_video_hash=source_video_hash,
        working_state_revision=working_state_revision,
    )
    token_path = preview_token_path(session_dir, token_payload["preview_token"])
    write_json(token_path, token_payload)
    publication_deferred = latest_reviewed_publication_deferred(package_root)
    return {
        "audio_handling": audio_handling,
        "corrected_preview": corrected_preview.relative_to(package_root).as_posix(),
        "diagnostic_preview": True,
        "draft_hash": token_payload["draft_hash"],
        "effective_regions_hash": effective_regions_hash,
        "frame_end": correction.source_frame_end,
        "frame_start": correction.source_frame_start,
        "latest_reviewed": None,
        "latest_reviewed_publication_status": "deferred" if publication_deferred else "supported",
        "ok": True,
        "original_preview": original_preview.relative_to(package_root).as_posix(),
        "preview_sha256": preview_hash,
        "preview_token": token_payload["preview_token"],
        "rendered_frames": rendered_frames,
        "rendered_regions": rendered_regions,
        "role": "correction_preview",
        "schema_version": "correction_interval_preview.v2",
        "source_video_hash": source_video_hash,
        "working_state_revision": working_state_revision,
    }


def preview_token_payload(
    session_dir: Path,
    correction: ReviewCorrection,
    *,
    effective_regions_hash: str,
    preview_sha256: str,
    source_video_hash: str,
    working_state_revision: str,
) -> dict[str, Any]:
    draft_hash = correction_draft_hash(correction)
    return {
        "case_id": correction.case_id,
        "correction_id": correction.correction_id,
        "created_at_utc": media.utc_timestamp(),
        "draft_hash": draft_hash,
        "effective_regions_hash": effective_regions_hash,
        "keyframe_source_frames": [keyframe.source_frame for keyframe in correction.keyframes],
        "preview_sha256": preview_sha256,
        "preview_token": f"preview_{uuid.uuid4().hex}",
        "schema_version": PREVIEW_TOKEN_SCHEMA_VERSION,
        "session_id": session_dir.name,
        "source_frame_end": correction.source_frame_end,
        "source_frame_start": correction.source_frame_start,
        "source_video_hash": source_video_hash,
        "working_state_revision": working_state_revision,
    }


def preview_token_path(session_dir: Path, token: str) -> Path:
    return session_dir / "state" / "preview_tokens" / f"{token}.json"


def load_preview_token_payload(session_dir: Path, *, token: str) -> dict[str, Any]:
    if not token:
        return {
            "error": "Preview the correction before saving it.",
            "ok": False,
        }
    safe = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-"})
    if safe != token:
        return {"error": "Preview token is invalid.", "ok": False}
    path = preview_token_path(session_dir, safe)
    if not path.exists():
        return {"error": "Correction preview has expired or was not found.", "ok": False}
    return {"ok": True, "preview": read_json(path)}


def validate_preview_token(
    session_dir: Path,
    *,
    token: str,
    correction: ReviewCorrection,
    current_working_state_revision: str | None = None,
) -> dict[str, Any]:
    token_result = load_preview_token_payload(session_dir, token=token)
    if not token_result.get("ok"):
        return token_result
    payload = dict(token_result["preview"])
    if payload.get("session_id") != session_dir.name:
        return {"error": "Preview token belongs to a different review session.", "ok": False}
    if payload.get("case_id") != correction.case_id:
        return {"error": "Preview token belongs to a different case.", "ok": False}
    if payload.get("correction_id") != correction.correction_id:
        return {"error": "Preview token belongs to a different correction draft.", "ok": False}
    if payload.get("source_video_hash") != correction.source_video_hash:
        return {"error": "Preview token belongs to a different source media revision.", "ok": False}
    if payload.get("draft_hash") != correction_draft_hash(correction):
        return {
            "error": "Correction draft changed after Preview; preview it again before saving.",
            "ok": False,
        }
    accepted_correction_id = str(payload.get("accepted_correction_id") or "")
    if accepted_correction_id:
        if accepted_correction_id == correction.correction_id:
            return {"already_accepted": True, "ok": True, "preview": payload}
        return {
            "error": "Preview token has already been used for a different correction.",
            "ok": False,
        }
    token_revision = str(payload.get("working_state_revision") or "")
    if (
        current_working_state_revision
        and token_revision
        and token_revision != current_working_state_revision
    ):
        return {
            "error": (
                "Review state changed after Preview; preview the correction again before saving."
            ),
            "ok": False,
        }
    return {"ok": True, "preview": payload}


def mark_preview_token_accepted(
    session_dir: Path,
    *,
    token: str,
    correction: ReviewCorrection,
    event: dict[str, Any],
    materialization_plan: dict[str, Any],
) -> dict[str, Any]:
    token_result = load_preview_token_payload(session_dir, token=token)
    if not token_result.get("ok"):
        return token_result
    payload = dict(token_result["preview"])
    payload.update(
        {
            "accepted_at_utc": media.utc_timestamp(),
            "accepted_correction_id": correction.correction_id,
            "accepted_event_hash": event.get("event_hash"),
            "accepted_event_id": event.get("event_id"),
            "accepted_revision_id": correction.revision_id,
            "materialization_job_id": (materialization_plan.get("job") or {}).get("job_id"),
            "materialization_required": (materialization_plan.get("plan") or {}).get(
                "materialization_required"
            ),
        }
    )
    write_json(preview_token_path(session_dir, token), payload)
    return {"ok": True, "preview": payload}


def correction_draft_hash(correction: ReviewCorrection) -> str:
    payload = correction.to_dict()
    for key in (
        "application_version",
        "correction_id",
        "created_at_utc",
        "previous_revision_id",
        "reason",
        "reviewer_identity",
        "revision_id",
        "schema_version",
    ):
        payload.pop(key, None)
    return stable_hash(payload)


def create_correction_verification_task(
    correction: ReviewCorrection,
    *,
    accepted_working_state_revision: str | None = None,
    materialization_plan: dict[str, Any] | None = None,
    publication_status: str = "waiting",
) -> dict[str, Any]:
    plan = dict(materialization_plan or {})
    category_label = manual_redaction_category_label(correction.category)
    deferred = (
        publication_status == "deferred"
        or plan.get("latest_reviewed_publication_status") == "deferred"
    )
    return {
        "accepted_working_state_revision": accepted_working_state_revision,
        "actionable": deferred,
        "allowed_actions": list(CANONICAL_REVIEW_ACTIONS) if deferred else [],
        "blocking": True,
        "case_id": correction.case_id,
        "category": correction.category,
        "changed_frame_count": plan.get("changed_source_frame_count"),
        "changed_frame_intervals": plan.get("changed_source_frame_intervals") or [],
        "correction_id": correction.correction_id,
        "correction_required": False,
        "decision_state": "pending" if deferred else "waiting_for_latest_reviewed",
        "issue_type": "correction_verification",
        "latest_reviewed_publication_status": "deferred" if deferred else None,
        "materialization_required": plan.get("materialization_required"),
        "normal_task_class": "correction_verification",
        "publication_status": publication_status,
        "raw_signal_ids": [],
        "review_task_id": f"correction_verify_{correction.case_id}_{correction.correction_id}",
        "schema_version": "case_reviewer_task.v1",
        "severity": "blocker",
        "source_frame_end": correction.source_frame_end,
        "source_frame_start": correction.source_frame_start,
        "target_effective_region_count": plan.get("after_effective_region_count"),
        "target_effective_regions_hash": plan.get("after_effective_regions_hash"),
        "technical_only": False,
        "title": f"Verify saved {category_label} correction",
        "what_happened": f"A reviewer supplied manual {category_label} redaction geometry.",
        "what_to_inspect": (
            "Correction saved; reviewed output publication is deferred."
            if deferred
            else "Latest reviewed media is not current yet."
        ),
    }


def find_correction_verification_task(
    state: dict[str, Any],
    *,
    correction_id: str,
) -> dict[str, Any] | None:
    for task in list(state.get("active_manual_tasks") or []):
        row = dict(task)
        if row.get("normal_task_class") != "correction_verification":
            continue
        if correction_id_from_verification_task(row) == correction_id:
            return row
    return None


def create_checkpoint(
    package_root: Path,
    session_dir: Path,
    *,
    actor: dict[str, Any],
    case_context_id: str | None = None,
    name: str,
    note: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    ensure_session_layout(session_dir)
    created_from_case_id = case_ids.validate_case_id(case_context_id) if case_context_id else None
    state = derive_working_state(package_root, session_dir)
    checkpoint_id = f"checkpoint_{uuid.uuid4().hex}"
    checkpoint = {
        "associated_latest_reviewed": dict(state.get("materialized_latest_reviewed") or {}),
        "checkpoint_uuid": checkpoint_id,
        "created_at_utc": media.utc_timestamp(),
        "created_from_case_id": created_from_case_id,
        "current_event_chain_head": state.get("current_event_chain_head"),
        "name": name.strip() or "Checkpoint",
        "note": note or "",
        "reviewer_attribution": actor,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "active_projection_hash": state.get("active_projection_hash"),
        "state_hash": state["state_hash"],
        "state_snapshot": {
            "active_correction_ids": list(state.get("active_correction_ids") or []),
            "active_manual_tasks": list(state.get("active_manual_tasks") or []),
            "active_projection_hash": state.get("active_projection_hash"),
            "materialized_latest_reviewed": dict(state.get("materialized_latest_reviewed") or {}),
            "object_policy_overrides": dict(state.get("object_policy_overrides") or {}),
            "sweep_acknowledgements": dict(state.get("sweep_acknowledgements") or {}),
            "task_decisions": dict(state.get("task_decisions") or {}),
            "task_decisions_by_case": dict(state.get("task_decisions_by_case") or {}),
            "task_resolutions": dict(state.get("task_resolutions") or {}),
            "task_resolutions_by_case": dict(state.get("task_resolutions_by_case") or {}),
        },
    }
    write_json(session_dir / "checkpoints" / f"{checkpoint_id}.json", checkpoint)
    event = append_event(
        session_dir,
        event_type="checkpoint_created",
        actor=actor,
        case_id=None,
        task_id=None,
        before_state=None,
        after_state={
            "checkpoint_uuid": checkpoint_id,
            "created_from_case_id": created_from_case_id,
            "name": checkpoint["name"],
            "state_hash": state["state_hash"],
        },
        note=note or f'Checkpoint "{checkpoint["name"]}" created.',
        application_hashes={"state_hash": state["state_hash"]},
    )
    return checkpoint, event


def restore_checkpoint(
    package_root: Path,
    session_dir: Path,
    *,
    actor: dict[str, Any],
    checkpoint_uuid: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    checkpoint = read_json(session_dir / "checkpoints" / f"{checkpoint_uuid}.json")
    event = append_event(
        session_dir,
        event_type="restore_checkpoint",
        actor=actor,
        case_id=None,
        task_id=None,
        before_state=derive_working_state(package_root, session_dir),
        after_state={
            "checkpoint_uuid": checkpoint_uuid,
            "name": checkpoint.get("name"),
            "state_hash": checkpoint.get("state_hash"),
        },
        note=f'Restored checkpoint "{checkpoint.get("name")}".',
        application_hashes={"state_hash": str(checkpoint.get("state_hash") or "")},
    )
    state = derive_working_state(package_root, session_dir)
    return checkpoint, event, state


def delete_checkpoint(
    package_root: Path,
    session_dir: Path,
    *,
    actor: dict[str, Any],
    checkpoint_uuid: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    checkpoint_path = session_dir / "checkpoints" / f"{checkpoint_uuid}.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_uuid}")
    checkpoint = read_json(checkpoint_path)
    tombstone = {
        "checkpoint_uuid": checkpoint_uuid,
        "deleted_at_utc": media.utc_timestamp(),
        "deleted_by": actor,
        "name": checkpoint.get("name"),
        "schema_version": "review_checkpoint_tombstone.v1",
    }
    write_json(session_dir / "checkpoints" / f"{checkpoint_uuid}.deleted.json", tombstone)
    event = append_event(
        session_dir,
        event_type="checkpoint_deleted",
        actor=actor,
        case_id=None,
        task_id=None,
        before_state={
            "checkpoint_uuid": checkpoint_uuid,
            "name": checkpoint.get("name"),
            "state_hash": checkpoint.get("state_hash"),
        },
        after_state=tombstone,
        note=f'Deleted checkpoint "{checkpoint.get("name") or "Checkpoint"}".',
        application_hashes={"state_hash": str(checkpoint.get("state_hash") or "")},
    )
    state = derive_working_state(package_root, session_dir)
    return checkpoint, event, state


def restart_review_workflow(
    package_root: Path,
    session_dir: Path,
    *,
    actor: dict[str, Any],
    case_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    selected_case = str(case_id or "")
    if not selected_case:
        raise ValueError("case_id is required to restart review workflow")
    selected_case = case_ids.validate_case_id(selected_case)
    require_safe_case_manifest_path(package_root, selected_case)
    before = derive_working_state(package_root, session_dir)
    after_state: dict[str, Any] = {
        "active_projection_reset": True,
        "case_ids": [selected_case],
        "clears": [
            "normal_review_task_decisions",
            "transient_manual_review_tasks",
            "task_completion_buckets",
            "full_video_sweep_acknowledgement",
            "review_progress_counters",
        ],
        "preserves": [
            "entity_policy_overrides",
            "entity_merges",
            "manual_corrections",
            "correction_verification_state",
            "checkpoints",
            "latest_reviewed_materialization",
        ],
    }
    event = append_event(
        session_dir,
        event_type="review_workflow_restarted",
        actor=actor,
        case_id=selected_case,
        task_id=None,
        before_state=before,
        after_state=after_state,
        note="Review workflow restarted.",
        application_hashes={},
    )
    state = derive_working_state(package_root, session_dir)
    return event, state


def reset_to_automatic_baseline(
    package_root: Path,
    session_dir: Path,
    *,
    actor: dict[str, Any],
    case_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    selected_case = case_ids.validate_case_id(case_id) if case_id else ""
    event_type = (
        "case_edits_reset_to_automatic_baseline" if selected_case else "reset_to_automatic_baseline"
    )
    before = derive_working_state(package_root, session_dir)
    reset_case_ids = (
        [selected_case]
        if selected_case
        else sorted(
            case_ids.validate_case_id(path.stem) for path in (package_root / "cases").glob("*.json")
        )
    )
    for item_case in reset_case_ids:
        require_safe_case_manifest_path(package_root, item_case)
    baseline_summaries = {
        item_case: baseline_case_active_state_summary(package_root, item_case)
        for item_case in reset_case_ids
    }
    mutated_summaries = {
        item_case: normalized_case_active_state_summary(package_root, before, item_case)
        for item_case in reset_case_ids
    }
    other_case_summaries_before = {
        case_ids.validate_case_id(path.stem): normalized_case_active_state_summary(
            package_root,
            before,
            case_ids.validate_case_id(path.stem),
        )
        for path in sorted((package_root / "cases").glob("*.json"))
        if case_ids.validate_case_id(path.stem) not in set(reset_case_ids)
    }
    after_state: dict[str, Any] = {
        "active_projection_reset": True,
        "baseline_state_hashes": {
            item_case: summary["normalized_state_hash"]
            for item_case, summary in baseline_summaries.items()
        },
        "before_active_projection_hash": before.get("active_projection_hash"),
        "case_ids": reset_case_ids,
        "clears": [
            "review_task_decisions",
            "object_policy_overrides",
            "entity_merges",
            "manual_corrections",
            "session_manual_tasks",
            "full_video_sweep_acknowledgement",
            "latest_reviewed_materialization",
            "finalization_result",
        ],
        "mutated_state_hashes": {
            item_case: summary["normalized_state_hash"]
            for item_case, summary in mutated_summaries.items()
        },
        "post_reset_expected_state_hashes": {
            item_case: summary["normalized_state_hash"]
            for item_case, summary in baseline_summaries.items()
        },
        "restores": ["automatic_baseline", "source_backed_initial_object_policies"],
    }
    if latest_reviewed_publication_supported(package_root):
        from redactor import reviewed_regions

        after_state["latest_reviewed_publication_clear"] = (
            reviewed_regions.clear_current_publication(
                package_root,
                case_ids_to_clear=reset_case_ids,
                reason="reset_to_automatic_baseline",
            )
        )
    event = append_event(
        session_dir,
        event_type=event_type,
        actor=actor,
        case_id=selected_case or None,
        task_id=None,
        before_state=before,
        after_state=after_state,
        note="Case edits reset to Automatic baseline."
        if selected_case
        else "Reset active review state to Automatic baseline.",
        application_hashes={},
    )
    state = derive_working_state(package_root, session_dir)
    reset_summaries = {
        item_case: normalized_case_active_state_summary(package_root, state, item_case)
        for item_case in reset_case_ids
    }
    other_case_isolation = {
        item_case: (
            summary["normalized_state_hash"]
            == normalized_case_active_state_summary(package_root, state, item_case)[
                "normalized_state_hash"
            ]
        )
        for item_case, summary in other_case_summaries_before.items()
    }
    append_reset_baseline_state_audit(
        package_root,
        event=event,
        case_ids=reset_case_ids,
        baseline_summaries=baseline_summaries,
        mutated_summaries=mutated_summaries,
        reset_summaries=reset_summaries,
        other_case_isolation=other_case_isolation,
    )
    return event, state


def baseline_case_active_state_summary(package_root: Path, case_id: str) -> dict[str, Any]:
    case_id = case_ids.validate_case_id(case_id)
    entities = load_entity_index(package_root)
    baseline_state = {
        "active_correction_ids": [],
        "active_manual_tasks": [],
        "manual_corrections": [],
        "materialized_latest_reviewed": {},
        "object_policy_overrides": baseline_policy_overrides(entities),
        "sweep_acknowledgements": {},
        "task_decisions": {},
        "task_decisions_by_case": {},
        "task_resolutions": {},
        "task_resolutions_by_case": {},
    }
    return normalized_case_active_state_summary(package_root, baseline_state, case_id)


def normalized_case_active_state_summary(
    package_root: Path,
    state: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    case_id = case_ids.validate_case_id(case_id)
    entities_by_case = load_entity_index(package_root)
    case_entities = entities_by_case.get(case_id, [])
    entity_case_by_id = entity_case_index(entities_by_case)
    effective = effective_case_entities(case_entities, state)
    entity_rows = [
        {
            "category": entity.get("category"),
            "current_policy": inherited_policy(entity),
            "current_status": entity.get("current_status") or "active",
            "entity_id": entity.get("entity_id"),
            "first_visible_source_frame": entity.get("first_visible_source_frame"),
            "label": entity.get("label"),
            "last_visible_source_frame": entity.get("last_visible_source_frame"),
            "review_entity_state": entity.get("review_entity_state"),
        }
        for entity in effective
    ]
    entity_rows.sort(key=lambda row: str(row.get("entity_id") or ""))
    policy_counts: dict[str, int] = {}
    for row in entity_rows:
        policy = str(row.get("current_policy") or POLICY_REVIEW_REQUIRED)
        policy_counts[policy] = policy_counts.get(policy, 0) + 1
    overrides = dict(state.get("object_policy_overrides") or {})
    retired_entity_ids = sorted(
        entity_id
        for entity_id, entity in overrides.items()
        if dict(entity).get("current_status") == "merged"
        and override_belongs_to_any_case(entity_id, dict(entity), entity_case_by_id, {case_id})
    )
    merged_entities = sorted(
        str(row.get("entity_id") or "")
        for row in entity_rows
        if str(row.get("label") or "").startswith("Merged ")
    )
    task_decisions = dict(task_decisions_by_case_from_payload(state).get(case_id) or {})
    manual_tasks = [
        dict(task)
        for task in list(state.get("active_manual_tasks") or [])
        if str(dict(task).get("case_id") or "") == case_id
    ]
    corrections = [
        dict(correction)
        for correction in list(state.get("manual_corrections") or [])
        if str(dict(correction).get("case_id") or "") == case_id
    ]
    latest = dict(state.get("materialized_latest_reviewed") or {}).get(case_id)
    summary = {
        "active_correction_ids": sorted(
            str(correction.get("correction_id") or "") for correction in corrections
        ),
        "case_id": case_id,
        "correction_count": len(corrections),
        "decision_count": len(task_decisions),
        "entity_count": len(entity_rows),
        "entity_rows": entity_rows,
        "latest_reviewed": latest
        if latest
        else {"role": "latest_reviewed", "semantic_source": "automatic_baseline"},
        "manual_task_count": len(manual_tasks),
        "merged_entity_ids": merged_entities,
        "policy_counts": dict(sorted(policy_counts.items())),
        "retired_entity_ids": retired_entity_ids,
        "sweep_acknowledged": case_id in dict(state.get("sweep_acknowledgements") or {}),
    }
    summary["normalized_state_hash"] = stable_hash(summary)
    return summary


def write_reset_baseline_state_audit_shell(package_root: Path) -> None:
    write_json(
        package_root / "reset_baseline_state_audit.json",
        {
            "created_at_utc": media.utc_timestamp(),
            "events": [],
            "schema_version": "reset_baseline_state_audit.v1",
        },
    )


def append_reset_baseline_state_audit(
    package_root: Path,
    *,
    event: dict[str, Any],
    case_ids: list[str],
    baseline_summaries: dict[str, dict[str, Any]],
    mutated_summaries: dict[str, dict[str, Any]],
    reset_summaries: dict[str, dict[str, Any]],
    other_case_isolation: dict[str, bool],
) -> None:
    path = package_root / "reset_baseline_state_audit.json"
    if not path.exists():
        write_reset_baseline_state_audit_shell(package_root)
    audit = read_json(path)
    rows = []
    for case_id in case_ids:
        baseline = baseline_summaries[case_id]
        mutated = mutated_summaries[case_id]
        reset = reset_summaries[case_id]
        rows.append(
            {
                "baseline_state_hash": baseline["normalized_state_hash"],
                "case_id": case_id,
                "entity_counts": {
                    "after_mutation": mutated["entity_count"],
                    "after_reset": reset["entity_count"],
                    "before_mutation": baseline["entity_count"],
                },
                "merged_entities": {
                    "after_mutation": mutated["merged_entity_ids"],
                    "after_reset": reset["merged_entity_ids"],
                    "before_mutation": baseline["merged_entity_ids"],
                },
                "mutated_state_hash": mutated["normalized_state_hash"],
                "policy_counts": {
                    "after_mutation": mutated["policy_counts"],
                    "after_reset": reset["policy_counts"],
                    "before_mutation": baseline["policy_counts"],
                },
                "post_reset_equals_baseline": reset["normalized_state_hash"]
                == baseline["normalized_state_hash"],
                "post_reset_state_hash": reset["normalized_state_hash"],
                "schema_version": "reset_baseline_state_case_audit.v1",
            }
        )
    events = list(audit.get("events") or [])
    events.append(
        {
            "case_isolation": other_case_isolation,
            "case_results": rows,
            "created_at_utc": media.utc_timestamp(),
            "event_hash": event.get("event_hash"),
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "schema_version": "reset_baseline_state_event_audit.v1",
        }
    )
    audit["events"] = events
    audit["created_at_utc"] = media.utc_timestamp()
    write_json(path, audit)


def load_checkpoints(session_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dir = session_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    deleted = {
        path.name.removesuffix(".deleted.json")
        for path in checkpoint_dir.glob("checkpoint_*.deleted.json")
    }
    return sorted(
        [
            read_json(path)
            for path in checkpoint_dir.glob("checkpoint_*.json")
            if path.stem not in deleted and not path.name.endswith(".deleted.json")
        ],
        key=lambda item: str(item.get("created_at_utc") or ""),
    )


def case_versions(package_root: Path, session_dir: Path | None, *, case_id: str) -> dict[str, Any]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve() if session_dir is not None else None
    case_id = case_ids.validate_case_id(case_id)
    manifest = case_manifests.load_case_manifest(
        require_safe_case_manifest_path(package_root, case_id)
    )
    by_role = {role.role: role.to_dict() for role in manifest.video_roles}
    roles: list[dict[str, Any]] = []
    if "original" in by_role:
        roles.append(by_role["original"])
    automatic = dict(by_role.get("automatic") or by_role.get("current_result") or {})
    if automatic:
        automatic["label"] = "Automatic"
        automatic["role"] = "automatic"
        roles.append(automatic)
    if latest_reviewed_publication_deferred(package_root):
        return {
            "case_id": case_id,
            "ok": True,
            "schema_version": "case_versions.v1",
            "versions": roles,
        }
    if session_dir is not None and session_dir.exists():
        latest_role = {}
        state = derive_working_state(package_root, session_dir)
        latest = dict(state.get("materialized_latest_reviewed") or {}).get(case_id)
        if latest:
            latest_role = latest
    else:
        latest_role = (
            {}
            if latest_reviewed_publication_supported(package_root)
            else dict(by_role.get("latest_reviewed") or automatic)
        )
    if latest_role:
        latest_role["label"] = "Latest reviewed"
        latest_role["role"] = "latest_reviewed"
    if latest_role:
        roles = [role for role in roles if role.get("role") != "latest_reviewed"]
        roles.append(latest_role)
    final_role = {}
    if latest_reviewed_publication_supported(package_root) and session_dir is not None:
        from redactor import review_finalization

        final_role = review_finalization.current_final_role(
            package_root,
            session_dir,
            case_id=case_id,
        )
    elif not latest_reviewed_publication_supported(package_root):
        final_role = dict(by_role.get("final_reviewed") or {})
    if final_role:
        final_role["label"] = "Final"
        final_role["role"] = "final_reviewed"
        roles.append(final_role)
    return {
        "case_id": case_id,
        "ok": True,
        "schema_version": "case_versions.v1",
        "versions": roles,
    }


DOWNLOADABLE_CASE_VERSION_ROLES = {
    "automatic",
    "final_reviewed",
    "latest_reviewed",
    "original",
}


def downloadable_case_version(
    package_root: Path,
    session_dir: Path | None,
    *,
    case_id: str,
    role: str,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    requested_role = "final_reviewed" if role == "final" else str(role)
    if requested_role not in DOWNLOADABLE_CASE_VERSION_ROLES:
        raise ValueError(f"unsupported downloadable case media role: {role}")
    versions = case_versions(package_root, session_dir, case_id=case_id)["versions"]
    candidates = [dict(item) for item in versions if str(item.get("role") or "") == requested_role]
    if not candidates:
        raise FileNotFoundError(f"{case_id}: no current {requested_role} version is available")
    selected = candidates[-1]
    raw_path = str(selected.get("path") or "")
    if not raw_path:
        raise FileNotFoundError(f"{case_id}: {requested_role} version has no media path")
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"{case_id}: {requested_role} media path is not package-relative")
    resolved = (package_root / path).resolve()
    try:
        resolved.relative_to(package_root)
    except ValueError as exc:
        raise ValueError(f"{case_id}: {requested_role} media path escapes package root") from exc
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{case_id}: {requested_role} media file is missing")
    expected_sha = str(selected.get("sha256") or "")
    actual_sha = media.sha256_file(resolved)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"{case_id}: {requested_role} media sha256 mismatch")
    revision = (
        str(selected.get("revision_id") or selected.get("latest_reviewed_revision_id") or "")
        or actual_sha[:12]
    )
    filename = f"{case_id}_{requested_role}_{revision}.mp4"
    safe_filename = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in filename)
    return {
        "case_id": case_id,
        "content_type": "video/mp4",
        "filename": safe_filename,
        "ok": True,
        "path": resolved,
        "role": selected,
        "schema_version": "case_media_download.v1",
        "sha256": actual_sha,
    }


def delivery_package_contract(
    package_root: Path,
    session_dir: Path | None,
    *,
    case_id: str,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    manifest = case_manifests.load_case_manifest(
        require_safe_case_manifest_path(package_root, case_id)
    )
    roles: dict[str, dict[str, Any]] = {}
    for role in DOWNLOADABLE_CASE_VERSION_ROLES:
        try:
            download = downloadable_case_version(
                package_root,
                session_dir,
                case_id=case_id,
                role=role,
            )
            media_path = Path(download["path"])
            roles[role] = {
                "available": True,
                "filename": download["filename"],
                "path": media_path.relative_to(package_root.resolve()).as_posix(),
                "revision_id": dict(download.get("role") or {}).get("revision_id")
                or dict(download.get("role") or {}).get("latest_reviewed_revision_id"),
                "sha256": download["sha256"],
            }
        except (FileNotFoundError, ValueError) as exc:
            roles[role] = {"available": False, "reason": str(exc)}
    required = ("original", "automatic", "latest_reviewed", "final_reviewed")
    ready = all(roles.get(role, {}).get("available") is True for role in required)
    state = derive_working_state(package_root, session_dir) if session_dir is not None else {}
    try:
        region_digest = (
            effective_privacy_region_digest(package_root, session_dir, case_id=case_id)
            if session_dir is not None
            else baseline_effective_privacy_region_digest(package_root, case_id=case_id)
        )
    except (FileNotFoundError, ValueError):
        region_digest = {}
    source_sha256 = manifest.source_video_hash
    if not source_sha256:
        source_path = package_root / manifest.source_path
        if source_path.exists() and source_path.is_file():
            source_sha256 = media.sha256_file(source_path)
    media_sha256 = {
        role: str(row.get("sha256") or "")
        for role, row in sorted(roles.items())
        if row.get("available") is True
    }
    latest_role = roles.get("latest_reviewed") or {}
    final_role = roles.get("final_reviewed") or {}
    manifest_payload = {
        "available_media_roles": [
            role for role in required if roles.get(role, {}).get("available") is True
        ],
        "case": case_id,
        "created_at": media.utc_timestamp(),
        "final_revision": str(final_role.get("revision_id") or "")
        if final_role.get("available")
        else None,
        "latest_reviewed_revision": str(latest_role.get("revision_id") or "")
        if latest_role.get("available")
        else None,
        "media_sha256": media_sha256,
        "review_state_identity": {
            "active_projection_hash": state.get("active_projection_hash"),
            "decision_identity": REVIEW_DECISION_IDENTITY_V2,
            "session_id": session_dir.name if session_dir is not None else None,
            "state_hash": state.get("state_hash"),
        },
        "reviewed_region_set_sha256": region_digest.get("effective_regions_hash"),
        "schema_version": "case_delivery_manifest.v1",
        "source_sha256": source_sha256,
        "status": "final" if final_role.get("available") else "non-final",
        "validation_state": {
            "final_current": final_role.get("available") is True,
            "latest_reviewed_current": latest_role.get("available") is True,
            "ready_for_delivery": ready,
        },
    }
    manifest_payload["redaction_records"] = delivery_redaction_records_contract(
        package_root,
        session_dir,
        case_id=case_id,
        expected_source_sha256=source_sha256,
        require_current_final=final_role.get("available") is True,
    )
    return {
        "case_id": case_id,
        "manifest": manifest_payload,
        "ok": True,
        "ready_for_delivery": ready,
        "required_roles": list(required),
        "roles": roles,
        "schema_version": "delivery_package_contract.v1",
    }


def delivery_redaction_records_contract(
    package_root: Path,
    session_dir: Path | None,
    *,
    case_id: str,
    expected_source_sha256: str | None,
    require_current_final: bool,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    case_id = case_ids.validate_case_id(case_id)
    unavailable = {
        "available": False,
        "filename": DELIVERY_REDACTION_RECORDS_FILENAME,
        "manifest_filename": DELIVERY_REDACTION_RECORDS_MANIFEST_FILENAME,
        "schema_version": DELIVERY_REDACTION_RECORDS_SCHEMA_VERSION,
    }
    if session_dir is None:
        return unavailable | {"reason": "review session is required"}
    from redactor import review_finalization, reviewed_regions

    try:
        latest_pointer = reviewed_regions.validate_current_pointer(
            package_root,
            reviewed_regions.load_current_pointer(package_root, case_id=case_id),
            case_id=case_id,
        )
        final_pointer = None
        if require_current_final:
            final_pointer = review_finalization.validate_current_final_pointer(
                package_root,
                review_finalization.load_current_final_pointer(
                    package_root,
                    case_id=case_id,
                ),
                case_id=case_id,
            )
        revision_manifest = dict(latest_pointer.get("revision_manifest_payload") or {})
        revision_manifest_path = case_ids.contained_path(
            package_root,
            *Path(str(latest_pointer.get("revision_manifest") or "")).parts,
        )
        records_path = case_ids.contained_path(
            package_root,
            *(revision_manifest_path.parent / str(revision_manifest.get("reviewed_regions") or ""))
            .relative_to(package_root)
            .parts,
        )
        binding = reviewed_regions.validate_reviewed_regions_artifact_binding(
            package_root,
            records_path,
            latest_pointer=latest_pointer,
            final_pointer=final_pointer,
            expected_source_sha256=expected_source_sha256,
            expected_case_id=case_id,
        )
    except (FileNotFoundError, ValueError, reviewed_regions.ReviewedRegionCompilationError) as exc:
        return unavailable | {"reason": str(exc)}

    content_identities = dict(latest_pointer.get("content_identities") or {})
    return {
        "authoritative_path": records_path.relative_to(package_root).as_posix(),
        "available": True,
        "case": case_id,
        "content_identities": {
            key: dict(value)
            for key, value in sorted(content_identities.items())
            if key
            in {
                "latest_reviewed_media_validation",
                "latest_reviewed_region_validation",
                "latest_reviewed_regions",
                "latest_reviewed_revision_manifest",
                "latest_reviewed_video",
            }
        },
        "filename": DELIVERY_REDACTION_RECORDS_FILENAME,
        "finalization_id": (final_pointer or {}).get("finalization_id"),
        "latest_reviewed_revision": binding["revision_id"],
        "manifest_filename": DELIVERY_REDACTION_RECORDS_MANIFEST_FILENAME,
        "record_count": binding["record_count"],
        "render_configuration": dict(revision_manifest.get("render_configuration") or {}),
        "renderer_configuration_sha256": revision_manifest.get("renderer_configuration_sha256"),
        "references": {
            "final_current_pointer": "final/current.json" if final_pointer is not None else None,
            "finalization_manifest": (final_pointer or {}).get("finalization_manifest"),
            "latest_reviewed_current_pointer": "latest_reviewed/current.json",
            "latest_reviewed_revision_manifest": latest_pointer.get("revision_manifest"),
        },
        "reviewed_region_identity": dict(revision_manifest.get("reviewed_region_identity") or {}),
        "review_session_id": session_dir.name,
        "reviewed_region_records_sha256": binding["reviewed_region_records_sha256"],
        "reviewed_region_set_sha256": binding["reviewed_region_set_sha256"],
        "reviewed_regions_sha256": binding["reviewed_regions_sha256"],
        "revision_manifest_artifact_sha256": latest_pointer.get(
            "revision_manifest_artifact_sha256"
        ),
        "revision_manifest_sha256": revision_manifest.get("revision_manifest_sha256"),
        "schema_version": DELIVERY_REDACTION_RECORDS_SCHEMA_VERSION,
        "source_sha256": binding["source_sha256"],
    }


def validate_packaged_redaction_records(
    redaction_manifest: dict[str, Any],
    records_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = dict(redaction_manifest)
    if manifest.get("schema_version") != DELIVERY_REDACTION_RECORDS_SCHEMA_VERSION:
        errors.append("redaction records manifest schema mismatch")
    if manifest.get("available") is not True:
        errors.append("redaction records are not marked available")
    if str(manifest.get("filename") or "") != records_path.name:
        errors.append("redaction records filename mismatch")
    for key in (
        "case",
        "latest_reviewed_revision",
        "record_count",
        "reviewed_region_records_sha256",
        "reviewed_region_set_sha256",
        "reviewed_regions_sha256",
        "source_sha256",
    ):
        if manifest.get(key) in (None, ""):
            errors.append(f"redaction records manifest missing {key}")
    if _contains_absolute_path_value(manifest):
        errors.append("redaction records manifest leaks an absolute local path")

    actual_count: int | None = None
    actual_sha = ""
    semantic_sha = ""
    try:
        actual_sha = media.sha256_file(records_path)
        raw_lines = [line for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        actual_count = len(raw_lines)
        if any(_contains_absolute_path_value(json.loads(line)) for line in raw_lines):
            errors.append("redaction records leak an absolute local path")
        from redactor import review_finalization, reviewed_regions

        records = review_finalization.load_privacy_region_records(records_path)
        semantic_sha = reviewed_regions.semantic_region_set_hash(records)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"redaction records are unreadable: {exc}")

    if actual_sha and str(manifest.get("reviewed_regions_sha256") or "") != actual_sha:
        errors.append("redaction records SHA mismatch")
    if actual_count is not None and int(manifest.get("record_count") or -1) != actual_count:
        errors.append("redaction records count mismatch")
    if semantic_sha and str(manifest.get("reviewed_region_records_sha256") or "") != semantic_sha:
        errors.append("redaction records semantic SHA mismatch")
    return {
        "errors": errors,
        "passed": not errors,
        "record_count": actual_count or 0,
        "reviewed_region_records_sha256": semantic_sha,
        "reviewed_regions_sha256": actual_sha,
        "schema_version": "packaged_redaction_records_validation.v1",
    }


def _contains_absolute_path_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path_value(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_absolute_path_value(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("/"):
            return True
        return len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in {"\\", "/"}
    return False


def delivery_package_zip(
    package_root: Path,
    session_dir: Path | None,
    *,
    case_id: str,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    contract = delivery_package_contract(package_root, session_dir, case_id=case_id)
    output_root = session_dir / "packages" if session_dir is not None else package_root / "packages"
    output_root.mkdir(parents=True, exist_ok=True)
    status = str(dict(contract.get("manifest") or {}).get("status") or "non-final")
    name_stem = f"{case_id}_{status}_review_package"
    output_path = output_root / f"{name_stem}.zip"
    case_folder = case_label_for_delivery(case_id)
    media_names = {
        "automatic": "Automatic.mp4",
        "final_reviewed": "Final.mp4",
        "latest_reviewed": "Latest_reviewed.mp4",
        "original": "Original.mp4",
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{case_folder}/manifest.json",
            json.dumps(contract["manifest"], indent=2, sort_keys=True) + "\n",
        )
        archive.writestr(
            f"{case_folder}/review_audit.json",
            json.dumps(
                {
                    "case": case_id,
                    "review_state_identity": contract["manifest"]["review_state_identity"],
                    "schema_version": "case_delivery_review_audit.v1",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        archive.writestr(
            f"{case_folder}/validation.json",
            json.dumps(contract["manifest"]["validation_state"], indent=2, sort_keys=True) + "\n",
        )
        redaction_records = dict(contract["manifest"].get("redaction_records") or {})
        if redaction_records.get("available") is True:
            records_path = case_ids.contained_path(
                package_root,
                *Path(str(redaction_records["authoritative_path"])).parts,
            )
            validation = validate_packaged_redaction_records(redaction_records, records_path)
            if validation["passed"] is not True:
                raise ValueError(
                    f"{case_id}: redaction records failed validation: {validation['errors']}"
                )
            redaction_manifest = redaction_records | {
                "archive_path": f"{case_folder}/{DELIVERY_REDACTION_RECORDS_FILENAME}",
            }
            archive.writestr(
                f"{case_folder}/{DELIVERY_REDACTION_RECORDS_MANIFEST_FILENAME}",
                json.dumps(redaction_manifest, indent=2, sort_keys=True) + "\n",
            )
            archive.write(
                records_path,
                arcname=f"{case_folder}/{DELIVERY_REDACTION_RECORDS_FILENAME}",
            )
        for role_name in ("original", "automatic", "latest_reviewed", "final_reviewed"):
            row = dict(dict(contract.get("roles") or {}).get(role_name) or {})
            if row.get("available") is not True:
                continue
            media_path = case_ids.contained_path(package_root, *Path(str(row["path"])).parts)
            archive.write(media_path, arcname=f"{case_folder}/{media_names[role_name]}")
    return {
        "case_id": case_id,
        "content_type": "application/zip",
        "filename": output_path.name,
        "ok": True,
        "path": output_path,
        "schema_version": "case_delivery_package_zip.v1",
    }


def case_label_for_delivery(case_id: str) -> str:
    if case_id == "day":
        return "Day"
    if case_id == "night":
        return "Night"
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in case_id) or "Case"


def history_events(session_dir: Path) -> dict[str, Any]:
    return {
        "events": [human_history_event(event) for event in load_events(session_dir)],
        "ok": True,
        "schema_version": "review_history.v1",
    }


def human_history_event(event: dict[str, Any]) -> dict[str, Any]:
    after = dict(event.get("after_state") or {})
    event_type = str(event.get("event_type") or "")
    label = human_event_label(event_type, after)
    return {
        "action": label,
        "case_id": event.get("case_id"),
        "event_id": event.get("event_id"),
        "event_type": event_type,
        "label": after.get("label") or after.get("title") or after.get("name"),
        "reviewer": (dict(event.get("actor") or {}).get("display_name") or None),
        "technical_details": event,
        "timestamp_utc": event.get("timestamp_utc"),
    }


def human_event_label(event_type: str, after: dict[str, Any]) -> str:
    if event_type == "session_started":
        return "Review session started"
    if event_type == "decision_recorded":
        return "Task decision recorded"
    if event_type == "entity_policy_changed":
        label = after.get("label") or after.get("display_name") or "Review object"
        return f"{label} set to {policy_label(inherited_policy(after), after.get('category'))}"
    if event_type == "bulk_entity_policy_changed":
        return f"Bulk object policy set to {policy_label(str(after.get('policy') or ''), None)}"
    if event_type in {EVENT_MANUAL_CORRECTION_SAVED, EVENT_CORRECTION_SAVED_LEGACY}:
        category = str(after.get("category") or "privacy")
        try:
            category = manual_redaction_category_label(category)
        except ValueError:
            pass
        return (
            f"Manual {category} correction saved, "
            f"frames {after.get('source_frame_start')}-{after.get('source_frame_end')}"
        )
    if event_type == "checkpoint_created":
        return f'Checkpoint "{after.get("name") or "Checkpoint"}" created'
    if event_type == "restore_checkpoint":
        return f'Restored checkpoint "{after.get("name") or "Checkpoint"}"'
    if event_type == "checkpoint_deleted":
        return f'Checkpoint "{after.get("name") or "Checkpoint"}" deleted'
    if event_type == "review_workflow_restarted":
        return "Review workflow restarted"
    if event_type == "case_edits_reset_to_automatic_baseline":
        return "Case edits reset to Automatic baseline"
    if event_type == "case_review_restarted_from_automatic_baseline":
        return "Case review restarted from Automatic baseline"
    if event_type == "reviewed_media_materialization_started":
        return "Latest reviewed video update started"
    if event_type == "reviewed_media_materialization_completed":
        return "Latest reviewed video update completed"
    if event_type == "reviewed_media_materialization_failed":
        return "Latest reviewed video update failed"
    if event_type == "entities_merged":
        count = len(
            list(after.get("retired_entity_uuids") or after.get("retired_entity_ids") or [])
        )
        return f"Merged {max(1, count)} review objects"
    if event_type == "correction_previewed":
        return "Manual correction previewed"
    if event_type == "latest_reviewed_materialized":
        return "Latest reviewed video materialized"
    if event_type == "case_finalized":
        return "Final video materialized"
    return event_type.replace("_", " ").title()


def policy_label(policy: str, category: Any) -> str:
    if policy == POLICY_REDACT:
        return "Redact"
    if policy == POLICY_KEEP_VISIBLE:
        return "Keep visible"
    if policy == POLICY_NOT_SENSITIVE:
        return "Not a person" if category == CATEGORY_FACE else "Not a plate"
    return "Needs review"


def automatic_or_current_role(
    manifest: case_manifests.CaseManifest,
) -> case_manifests.VideoRole | None:
    for role_name in ("automatic", "latest_reviewed", "current_result"):
        for role in manifest.video_roles:
            if role.role == role_name:
                return role
    return None


def render_settings() -> rendering.RenderSettings:
    return rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_fraction=0.15,
        margin_pixels=0,
        pixel_size=16,
        blur_kernel=51,
        preview_watermark=False,
    )


def face_policy_source_dirs(
    package_root: Path,
    case_id: str,
    manifest: case_manifests.CaseManifest,
) -> list[Path]:
    case_id = case_ids.validate_case_id(case_id)
    candidates: list[Path] = []
    local = package_root / "technical" / f"{case_id}_pipeline" / "face_policy"
    if (local / "track_census.json").exists():
        candidates.append(local)
    artifacts_root = Path("artifacts")
    if artifacts_root.exists():
        candidates.extend(
            path.parent for path in artifacts_root.glob(f"*/{case_id}/**/track_census.json")
        )
    matched: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not (candidate / "track_census.json").exists():
            continue
        seen.add(resolved)
        census = read_json(candidate / "track_census.json")
        source_manifest_path = candidate / "source_manifest.json"
        source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
        candidate_hash = source_manifest.get("source_video_hash") or census.get("source_video_hash")
        if (
            manifest.source_video_hash
            and candidate_hash
            and str(candidate_hash) != str(manifest.source_video_hash)
        ):
            continue
        if not (candidate / "privacy_policy.json").exists():
            continue
        matched.append(candidate)
    return sorted(matched)


def display_source_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return path.as_posix()


def append_manual_correction_persistence_audit(
    package_root: Path,
    *,
    case_id: str,
    materialization: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    applied = list(
        dict(provenance.get("correction_provenance") or {}).get("applied_corrections") or []
    )
    if not applied:
        return
    path = package_root / "manual_correction_persistence_audit.json"
    if path.exists():
        audit = read_json(path)
    else:
        audit = {
            "created_at_utc": media.utc_timestamp(),
            "events": [],
            "schema_version": "manual_correction_persistence_audit.v1",
        }
    audit.setdefault("events", []).append(
        {
            "case_id": case_id,
            "correction_ids": [str(item.get("correction_id") or "") for item in applied],
            "effective_regions_hash": materialization.get("effective_regions_hash"),
            "manual_correction_count": materialization.get("manual_correction_count"),
            "output_video": materialization.get("output_video"),
            "privacy_region_changes_by_cause": materialization.get(
                "privacy_region_changes_by_cause"
            ),
            "rendered_frame_count": materialization.get("rendered_frame_count"),
            "schema_version": "manual_correction_persistence_event.v1",
            "state_hash": materialization.get("state_hash"),
            "video_sha256": materialization.get("video_sha256"),
        }
    )
    write_json(path, audit)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
