from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from redactor import case_manifests, media, privacy_entities, privacy_rendering, rendering
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    stable_hash,
)
from redactor.review_corrections import ReviewCorrection, load_corrections_jsonl
from redactor.review_sessions import (
    append_event,
    ensure_session_layout,
    load_events,
)

WORKING_STATE_SCHEMA_VERSION = "review_working_state.v1"
CHECKPOINT_SCHEMA_VERSION = "review_checkpoint.v1"
PREVIEW_TOKEN_SCHEMA_VERSION = "correction_preview_token.v1"

POLICY_REDACT = privacy_entities.POLICY_REDACT
POLICY_KEEP_VISIBLE = privacy_entities.POLICY_KEEP_VISIBLE
POLICY_NOT_SENSITIVE = privacy_entities.POLICY_NOT_SENSITIVE
POLICY_REVIEW_REQUIRED = privacy_entities.POLICY_REVIEW_REQUIRED
SUPPRESSING_POLICIES = {POLICY_KEEP_VISIBLE, POLICY_NOT_SENSITIVE}


def derive_working_state(package_root: Path, session_dir: Path) -> dict[str, Any]:
    """Return deterministic session working state plus any valid materialized versions."""
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    ensure_session_layout(session_dir)
    baseline = package_baseline(package_root)
    entities = load_entity_index(package_root)
    projection = derive_active_review_projection(
        package_root=package_root,
        session_dir=session_dir,
        entities=entities,
    )
    corrections = load_active_corrections(
        session_dir,
        active_correction_ids=set(projection["active_correction_ids"]),
    )
    checkpoints = load_checkpoints(session_dir)
    chain_head = current_event_chain_head(session_dir)
    projection_hash = active_projection_hash(projection)
    state = {
        "active_correction_ids": sorted(correction.correction_id for correction in corrections),
        "active_manual_tasks": projection["active_manual_tasks"],
        "active_projection_hash": projection_hash,
        "baseline": baseline,
        "checkpoints": checkpoints,
        "current_event_chain_head": chain_head,
        "deterministic_state_inputs": {
            "correction_hashes": sorted(correction.correction_hash() for correction in corrections),
            "decision_hash": stable_hash(projection["task_decisions"]),
            "object_policy_hash": stable_hash(projection["object_policy_overrides"]),
            "projection_hash": projection_hash,
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
    }
    state["state_hash"] = state_hash(state)
    write_state_snapshot(session_dir, state)
    return state


def package_baseline(package_root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case_path in sorted((package_root / "cases").glob("*.json")):
        manifest = case_manifests.load_case_manifest(case_path)
        base_region_path = package_root / "regions" / f"{manifest.case_id}_current_regions.jsonl"
        cases[manifest.case_id] = {
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
    decisions: dict[str, dict[str, Any]] = {}
    manual_tasks: dict[str, dict[str, Any]] = {}
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
                decisions=decisions,
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
                decisions=decisions,
                entity_case_by_id=entity_case_by_id,
                manual_tasks=manual_tasks,
                materialized=materialized,
                overrides=overrides,
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
            decisions = {
                str(task_id): dict(decision)
                for task_id, decision in dict(snapshot.get("task_decisions") or {}).items()
            }
            manual_tasks = {
                str(task.get("review_task_id") or ""): dict(task)
                for task in list(snapshot.get("active_manual_tasks") or [])
                if isinstance(task, dict) and task.get("review_task_id")
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
            if task_id:
                decisions[task_id] = dict(after) | {
                    "event_hash": event.get("event_hash"),
                    "event_id": event.get("event_id"),
                }
            continue
        if event_type == "full_video_sweep_acknowledged" and case_id:
            sweep_acknowledgements[case_id] = dict(after) | {
                "event_hash": event.get("event_hash"),
                "event_id": event.get("event_id"),
            }
            continue
        if event_type == "manual_issue_created":
            task = dict(after.get("task") or {})
            task_id = str(task.get("review_task_id") or event.get("task_id") or "")
            if task_id:
                manual_tasks[task_id] = task
            continue
        if event_type == "correction_saved":
            correction_id = str(after.get("correction_id") or "")
            if correction_id:
                active_correction_ids.add(correction_id)
                if case_id:
                    correction_case_by_id[correction_id] = case_id
            continue
        if event_type == "entity_policy_changed":
            entity = dict(after)
            entity_id = str(entity.get("entity_id") or entity.get("object_uuid") or "")
            if entity_id:
                if case_id:
                    entity.setdefault("case_id", case_id)
                entity.setdefault("reviewer_override", True)
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
                        overrides[entity_id] = updated
                        if case_id:
                            entity_case_by_id[entity_id] = case_id
            continue
        if event_type == "entities_merged":
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
            current_projection_hash = active_projection_hash(
                {
                    "active_correction_ids": sorted(active_correction_ids),
                    "active_manual_tasks": sorted(
                        manual_tasks.values(),
                        key=lambda task: (
                            str(task.get("case_id") or ""),
                            int(task.get("source_frame_start") or 0),
                            str(task.get("review_task_id") or ""),
                        ),
                    ),
                    "object_policy_overrides": overrides,
                    "sweep_acknowledgements": sweep_acknowledgements,
                    "task_decisions": decisions,
                }
            )
            if accepted_projection_hash and accepted_projection_hash != current_projection_hash:
                continue
            if case_id and latest:
                materialized[case_id] = latest
            continue
    return {
        "active_correction_ids": sorted(active_correction_ids),
        "active_manual_tasks": sorted(
            manual_tasks.values(),
            key=lambda task: (
                str(task.get("case_id") or ""),
                int(task.get("source_frame_start") or 0),
                str(task.get("review_task_id") or ""),
            ),
        ),
        "materialized_latest_reviewed": materialized,
        "object_policy_overrides": overrides,
        "sweep_acknowledgements": sweep_acknowledgements,
        "task_decisions": decisions,
    }


def active_projection_hash(projection: dict[str, Any]) -> str:
    return stable_hash(
        {
            "active_correction_ids": sorted(
                str(item) for item in list(projection.get("active_correction_ids") or [])
            ),
            "active_manual_tasks": sorted(
                list(projection.get("active_manual_tasks") or []),
                key=lambda task: (
                    str(task.get("case_id") or ""),
                    int(task.get("source_frame_start") or 0),
                    str(task.get("review_task_id") or ""),
                ),
            ),
            "object_policy_overrides": dict(projection.get("object_policy_overrides") or {}),
            "sweep_acknowledgements": dict(projection.get("sweep_acknowledgements") or {}),
            "task_decisions": dict(projection.get("task_decisions") or {}),
        }
    )


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
    decisions: dict[str, dict[str, Any]],
    entity_case_by_id: dict[str, str],
    manual_tasks: dict[str, dict[str, Any]],
    materialized: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
    active_correction_ids: set[str],
    sweep_acknowledgements: dict[str, dict[str, Any]],
) -> None:
    for task_id, decision in list(decisions.items()):
        if str(decision.get("case_id") or "") in case_ids:
            decisions.pop(task_id, None)
    for task_id, task in list(manual_tasks.items()):
        if str(task.get("case_id") or "") in case_ids:
            manual_tasks.pop(task_id, None)
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
    decisions: dict[str, dict[str, Any]],
    manual_tasks: dict[str, dict[str, Any]],
    sweep_acknowledgements: dict[str, dict[str, Any]],
) -> None:
    for task_id, decision in list(decisions.items()):
        if str(decision.get("case_id") or "") not in case_ids:
            continue
        if durable_edit_verification_task_id(task_id):
            continue
        decisions.pop(task_id, None)
    for task_id, task in list(manual_tasks.items()):
        if str(task.get("case_id") or "") not in case_ids:
            continue
        if durable_edit_verification_task_id(task_id):
            continue
        manual_tasks.pop(task_id, None)
    for case_id in case_ids:
        sweep_acknowledgements.pop(case_id, None)


def durable_edit_verification_task_id(task_id: str) -> bool:
    return task_id.startswith("correction_verify_") or task_id.startswith("object_policy_verify_")


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
    manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
    base_path = package_root / "regions" / f"{case_id}_current_regions.jsonl"
    base_regions = review_finalization.load_privacy_region_records(base_path)
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
        if merged.get("current_status") != "merged":
            effective.append(merged)
    for entity_id, override in sorted(overrides.items()):
        if entity_id in seen:
            continue
        entity = dict(override)
        if entity.get("current_status") == "merged":
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
    track_ids = entity_track_ids(entity)
    if not track_ids:
        return []
    regions: list[PrivacyRegionRecord] = []
    for source_dir in face_policy_source_dirs(package_root, manifest.case_id, manifest):
        tracks_path = source_dir / "tracking" / "tracks.jsonl"
        if not tracks_path.exists():
            continue
        for frame_payload in load_jsonl(tracks_path):
            frame_index = int(frame_payload.get("frame_index") or 0)
            for box_index, box in enumerate(list(frame_payload.get("track_boxes") or [])):
                track_id = str(box.get("track_id") or "")
                if track_id not in track_ids:
                    continue
                if not bool(box.get("render_eligible", True)):
                    continue
                bbox = box.get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    continue
                regions.append(
                    PrivacyRegionRecord(
                        approval_state=APPROVAL_ACCEPTED,
                        bbox_xyxy=tuple(int(value) for value in bbox),
                        category=CATEGORY_FACE,
                        object_id=f"face_track_{track_id}",
                        provenance="review_object_policy_source_track",
                        render_eligible=True,
                        source_frame=frame_index,
                        source_record={
                            "candidate_id": box.get("candidate_id"),
                            "entity_id": entity.get("entity_id"),
                            "policy_source": display_source_path(source_dir),
                            "region_id": f"source_policy_face_{track_id}_{frame_index}_{box_index}",
                            "track_id": track_id,
                            "track_uid": f"face_track_{track_id}",
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
) -> dict[str, Any]:
    from redactor import review_finalization

    total_start = time.monotonic()
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    ensure_session_layout(session_dir)
    manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
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
    base_region_path = package_root / "regions" / f"{case_id}_current_regions.jsonl"
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
    publication_start = time.monotonic()
    regions_path = session_dir / "state" / f"{case_id}_effective_regions_{region_hash[:16]}.jsonl"
    review_finalization.write_privacy_region_records(regions_path, final_regions)
    role = {
        "label": "Latest reviewed",
        "path": output_path.relative_to(package_root).as_posix(),
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
                "case_id": case_id,
                "effective_regions_hash": region_hash,
                "latest_reviewed": role,
                "state_hash": state["state_hash"],
            },
            note=event_reason or "Latest reviewed media materialized.",
            application_hashes={"latest_reviewed_sha256": video_hash},
        )
    return materialization


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
    manifest = case_manifests.load_case_manifest(
        package_root / "cases" / f"{correction.case_id}.json"
    )
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
    final_regions, provenance = compute_effective_privacy_regions(
        package_root,
        session_dir,
        case_id=correction.case_id,
        include_unsaved_correction=correction if include_unsaved_correction else None,
    )
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
    token_payload = preview_token_payload(session_dir, correction)
    token_path = session_dir / "state" / "preview_tokens" / f"{token_payload['preview_token']}.json"
    write_json(token_path, token_payload)
    preview_hash = media.sha256_file(corrected_preview)
    return {
        "audio_handling": audio_handling,
        "corrected_preview": corrected_preview.relative_to(package_root).as_posix(),
        "draft_hash": token_payload["draft_hash"],
        "effective_regions_hash": provenance.get("state_hash"),
        "frame_end": correction.source_frame_end,
        "frame_start": correction.source_frame_start,
        "latest_reviewed": corrected_preview.relative_to(package_root).as_posix(),
        "ok": True,
        "original_preview": original_preview.relative_to(package_root).as_posix(),
        "preview_sha256": preview_hash,
        "preview_token": token_payload["preview_token"],
        "rendered_frames": rendered_frames,
        "rendered_regions": rendered_regions,
        "schema_version": "correction_interval_preview.v2",
    }


def preview_token_payload(session_dir: Path, correction: ReviewCorrection) -> dict[str, Any]:
    draft_hash = correction_draft_hash(correction)
    return {
        "case_id": correction.case_id,
        "correction_id": correction.correction_id,
        "created_at_utc": media.utc_timestamp(),
        "draft_hash": draft_hash,
        "preview_token": f"preview_{uuid.uuid4().hex}",
        "schema_version": PREVIEW_TOKEN_SCHEMA_VERSION,
        "session_id": session_dir.name,
    }


def validate_preview_token(
    session_dir: Path, *, token: str, correction: ReviewCorrection
) -> dict[str, Any]:
    if not token:
        return {"legacy_without_token": True, "ok": True}
    safe = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-"})
    if safe != token:
        return {"error": "Preview token is invalid.", "ok": False}
    path = session_dir / "state" / "preview_tokens" / f"{safe}.json"
    if not path.exists():
        return {"error": "Correction preview has expired or was not found.", "ok": False}
    payload = read_json(path)
    if payload.get("draft_hash") != correction_draft_hash(correction):
        return {
            "error": "Correction draft changed after Preview; preview it again before saving.",
            "ok": False,
        }
    return {"ok": True, "preview": payload}


def correction_draft_hash(correction: ReviewCorrection) -> str:
    payload = correction.to_dict()
    for key in (
        "application_version",
        "correction_id",
        "created_at_utc",
        "previous_revision_id",
        "reason",
        "revision_id",
        "schema_version",
    ):
        payload.pop(key, None)
    return stable_hash(payload)


def create_correction_verification_task(correction: ReviewCorrection) -> dict[str, Any]:
    return {
        "blocking": True,
        "case_id": correction.case_id,
        "category": correction.category,
        "correction_required": False,
        "decision_state": "pending",
        "issue_type": "correction_verification",
        "normal_task_class": "correction_verification",
        "raw_signal_ids": [],
        "review_task_id": f"correction_verify_{correction.case_id}_{correction.correction_id}",
        "schema_version": "case_reviewer_task.v1",
        "severity": "blocker",
        "source_frame_end": correction.source_frame_end,
        "source_frame_start": correction.source_frame_start,
        "technical_only": False,
        "title": "Verify saved correction",
        "what_happened": "A reviewer supplied manual correction geometry.",
        "what_to_inspect": (
            "Confirm the previewed correction protects the intended region before finalization."
        ),
    }


def create_checkpoint(
    package_root: Path,
    session_dir: Path,
    *,
    actor: dict[str, Any],
    name: str,
    note: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    ensure_session_layout(session_dir)
    state = derive_working_state(package_root, session_dir)
    checkpoint_id = f"checkpoint_{uuid.uuid4().hex}"
    checkpoint = {
        "associated_latest_reviewed": dict(state.get("materialized_latest_reviewed") or {}),
        "checkpoint_uuid": checkpoint_id,
        "created_at_utc": media.utc_timestamp(),
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
    selected_case = str(case_id or "")
    event_type = (
        "case_edits_reset_to_automatic_baseline" if selected_case else "reset_to_automatic_baseline"
    )
    before = derive_working_state(package_root, session_dir)
    reset_case_ids = (
        [selected_case]
        if selected_case
        else sorted(path.stem for path in (package_root / "cases").glob("*.json"))
    )
    baseline_summaries = {
        item_case: baseline_case_active_state_summary(package_root, item_case)
        for item_case in reset_case_ids
    }
    mutated_summaries = {
        item_case: normalized_case_active_state_summary(package_root, before, item_case)
        for item_case in reset_case_ids
    }
    other_case_summaries_before = {
        path.stem: normalized_case_active_state_summary(package_root, before, path.stem)
        for path in sorted((package_root / "cases").glob("*.json"))
        if path.stem not in set(reset_case_ids)
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
    entities = load_entity_index(package_root)
    baseline_state = {
        "active_correction_ids": [],
        "active_manual_tasks": [],
        "manual_corrections": [],
        "materialized_latest_reviewed": {},
        "object_policy_overrides": baseline_policy_overrides(entities),
        "sweep_acknowledgements": {},
        "task_decisions": {},
    }
    return normalized_case_active_state_summary(package_root, baseline_state, case_id)


def normalized_case_active_state_summary(
    package_root: Path,
    state: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
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
    task_decisions = {
        task_id: dict(decision)
        for task_id, decision in dict(state.get("task_decisions") or {}).items()
        if str(dict(decision).get("case_id") or "") == case_id
    }
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
    manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
    by_role = {role.role: role.to_dict() for role in manifest.video_roles}
    roles: list[dict[str, Any]] = []
    if "original" in by_role:
        roles.append(by_role["original"])
    automatic = dict(by_role.get("automatic") or by_role.get("current_result") or {})
    if automatic:
        automatic["label"] = "Automatic"
        automatic["role"] = "automatic"
        roles.append(automatic)
    latest_role = dict(by_role.get("latest_reviewed") or automatic)
    if latest_role:
        latest_role["label"] = "Latest reviewed"
        latest_role["role"] = "latest_reviewed"
    if session_dir is not None and session_dir.exists():
        state = derive_working_state(package_root, session_dir)
        latest = dict(state.get("materialized_latest_reviewed") or {}).get(case_id)
        if latest:
            latest_role = latest
    if latest_role:
        roles = [role for role in roles if role.get("role") != "latest_reviewed"]
        roles.append(latest_role)
    return {
        "case_id": case_id,
        "ok": True,
        "schema_version": "case_versions.v1",
        "versions": roles,
    }


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
    if event_type == "correction_saved":
        return (
            f"Manual {after.get('category') or 'privacy'} correction saved, "
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
