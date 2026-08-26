from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from redactor import case_ids, case_manifests, media, privacy_rendering
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    MANUAL_REDACTION_CATEGORIES,
    PrivacyRegionRecord,
    stable_hash,
)
from redactor.review_corrections import (
    OP_ADD,
    OP_NOT_SENSITIVE,
    OP_NOT_VISIBLE,
    OP_REPLACE,
    OP_SUPPRESS,
    OP_SUPPRESS_TRACK_SEGMENT,
    ReviewCorrection,
    interpolate_correction_boxes,
)
from redactor.review_sessions import (
    ACTION_NOT_SENSITIVE,
    BLOCKING_ACTIONS,
    GREEN_ACTIONS,
    append_event,
    invalid_task_decision_reason,
    session_write_lock,
)

COMPILER_SCHEMA_VERSION = "reviewed_region_compiler.v1"
REVISION_SCHEMA_VERSION = "latest_reviewed_revision.v1"
PUBLICATION_SCHEMA_VERSION = "latest_reviewed_publication.v1"
CURRENT_POINTER_SCHEMA_VERSION = "latest_reviewed_current_pointer.v1"
CURRENT_POINTER_INDEX_SCHEMA_VERSION = "latest_reviewed_current_pointer_index.v1"
LATEST_REVIEWED_MATERIALIZATION_IMPLEMENTATION_ID = (
    "latest_reviewed_materialization.v2.frame_centered_seek_cache_identity.v1"
)
SUPPORTED_PUBLICATION_STATUSES = {"supported", "ready"}
WORKSPACE_PUBLICATION_IDENTITY_KEYS = {
    "publication_mode",
    "schema_version",
    "status",
}
LATEST_REVIEWED_CONTENT_IDENTITY_PREFIX = "latest_reviewed_"
CURRENT_POINTER_RELATIVE_PATH = Path("latest_reviewed") / "current.json"
REVIEWED_REGION_DIGEST_CACHE_DISABLE_ENV = "REDACTOR_DISABLE_REVIEWED_REGION_DIGEST_CACHE"
REVIEWED_REGION_DIGEST_CACHE_MAX_ENTRIES = 128
FAILURE_INJECTION_ENV = "REDACTOR_REVIEWED_REGIONS_FAIL_AT"
REVIEWED_REGION_SET_IDENTITY_RUNTIME_KEYS = {
    "active_correction_audit_hash",
    "active_corrections_hash",
    "active_projection_hash_schema_version",
    "case_manifest_input_sha256",
    "reviewed_region_set_sha256",
    "session_head_hash",
    "session_id",
    "workspace_manifest_input_sha256",
    "working_state_projection_hash",
}
REVIEWED_REGION_MEDIA_IDENTITY_KEYS = (
    "automatic_baseline_region_set_sha256",
    "case_id",
    "category_run_manifest_identity_hash",
    "compiler_schema_version",
    "configuration_freeze_hash",
    "renderer_configuration_sha256",
    "reviewed_region_records_sha256",
    "selected_categories",
    "source_frame_end",
    "source_frame_start",
    "source_video_hash",
)
RUNTIME_REVIEW_STATE_KEYS = {
    "event_hash",
    "event_id",
    "reviewed_at_utc",
    "timestamp_utc",
}
SEMANTIC_DECISION_KEYS = {
    "action",
    "case_id",
    "correction_id",
    "correction_revision_id",
    "decision_status",
    "resolved_by",
    "task_id",
}
SUPPRESSION_OPERATIONS = {
    OP_NOT_SENSITIVE,
    OP_NOT_VISIBLE,
    OP_REPLACE,
    OP_SUPPRESS,
    OP_SUPPRESS_TRACK_SEGMENT,
}
REVISION_ARTIFACT_KEYS = (
    "reviewed_regions",
    "output_video",
    "media_validation",
    "region_validation",
)

_DIGEST_CACHE_LOCK = threading.RLock()
_DIGEST_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_DIGEST_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}


class ReviewedRegionCompilationError(ValueError):
    """Raised when an authoritative reviewed region set cannot be compiled safely."""


class ReviewedRegionPublicationStaleError(ReviewedRegionCompilationError):
    """Raised when reviewed media is no longer current enough to publish."""


def compile_reviewed_regions(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    include_unsaved_correction: ReviewCorrection | None = None,
    require_blockers_resolved: bool = True,
    working_state: dict[str, Any] | None = None,
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    from redactor import review_working_state

    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    workspace = require_supported_workspace(package_root)
    manifest = case_manifests.load_case_manifest(
        review_working_state.safe_case_manifest_path(package_root, case_id)
    )
    validate_source_identity(package_root, workspace, manifest)
    baseline_path = review_working_state.safe_case_region_path(package_root, case_id)
    baseline_regions = load_authoritative_baseline(
        package_root=package_root,
        workspace=workspace,
        manifest=manifest,
        baseline_path=baseline_path,
    )
    baseline_binding = validate_baseline_category_run_binding(
        package_root=package_root,
        workspace=workspace,
        manifest=manifest,
        baseline_path=baseline_path,
        baseline_regions=baseline_regions,
    )
    state = (
        dict(working_state)
        if working_state is not None
        else review_working_state.derive_working_state(
            package_root,
            session_dir,
            copy_cached=False,
            write_snapshot=False,
        )
    )
    state.setdefault("state_hash", review_working_state.state_hash(state))
    entities_by_case = review_working_state.load_entity_index(package_root)
    source_entities = list(entities_by_case.get(case_id, []))
    effective_entities = review_working_state.effective_case_entities(source_entities, state)
    blockers = publication_blockers(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        state=state,
        effective_entities=effective_entities,
        baseline_regions=baseline_regions,
    )
    if blockers and require_blockers_resolved:
        raise ReviewedRegionCompilationError(
            f"{case_id}: unresolved review blockers prevent Latest reviewed publication: "
            + "; ".join(blockers)
        )

    policy_regions, policy_provenance = compile_object_policy_regions(
        package_root=package_root,
        manifest=manifest,
        baseline_regions=baseline_regions,
        source_entities=source_entities,
        effective_entities=effective_entities,
    )
    policy_regions, task_suppression_provenance = apply_not_sensitive_task_suppressions(
        policy_regions,
        package_root=package_root,
        case_id=case_id,
        decisions_by_case=review_working_state.task_decisions_by_case_from_payload(state),
    )
    active_correction_ids = {
        str(item) for item in list(state.get("active_correction_ids") or [])
    }
    correction_records = [
        correction
        for correction in review_working_state.load_active_corrections(
            session_dir,
            active_correction_ids=active_correction_ids,
        )
        if correction.case_id == case_id
    ]
    if include_unsaved_correction is not None:
        correction_records = [*correction_records, include_unsaved_correction]
    reviewed_regions, correction_provenance = apply_review_corrections(
        policy_regions,
        correction_records,
        manifest=manifest,
        selected_categories=selected_categories(workspace),
        effective_entities=effective_entities,
    )
    duplicate_keys = duplicated_region_keys(reviewed_regions)
    if duplicate_keys:
        raise ReviewedRegionCompilationError(
            f"{case_id}: duplicate reviewed regions detected: {duplicate_keys[:5]}"
        )

    identity = reviewed_region_identity(
        package_root=package_root,
        session_dir=session_dir,
        workspace=workspace,
        manifest=manifest,
        baseline_regions=baseline_regions,
        reviewed_regions=reviewed_regions,
        state=state,
        effective_entities=effective_entities,
        correction_records=correction_records,
    )
    reviewed_hash = str(identity["reviewed_region_set_sha256"])
    provenance = {
        "automatic_baseline_region_count": len(baseline_regions),
        "case_id": case_id,
        "compiler": {
            "baseline_binding": baseline_binding,
            "schema_version": COMPILER_SCHEMA_VERSION,
            "selected_categories": selected_categories(workspace),
        },
        "correction_provenance": correction_provenance,
        "effective_region_count": len(reviewed_regions),
        "manual_correction_count": len(correction_records),
        "object_policy_provenance": policy_provenance,
        "privacy_region_changes_by_cause": {
            "automatic_baseline": len(baseline_regions),
            "manual_correction_added": correction_provenance["added_region_count"],
            "manual_correction_replaced": correction_provenance["replaced_region_count"],
            "object_override_added": policy_provenance["added_region_count"],
            "suppression_removed": policy_provenance["suppressed_region_count"],
            "task_not_sensitive_removed": task_suppression_provenance[
                "suppressed_region_count"
            ],
        },
        "publication_blockers": blockers,
        "reviewed_region_identity": identity,
        "reviewed_region_set_sha256": reviewed_hash,
        "schema_version": "effective_privacy_regions.v2",
        "state_hash": state["state_hash"],
        "task_not_sensitive_provenance": task_suppression_provenance,
    }
    return reviewed_regions, provenance


def reviewed_region_digest(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    require_blockers_resolved: bool = False,
    working_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from redactor import review_working_state

    started = time.monotonic()
    state = (
        dict(working_state)
        if working_state is not None
        else review_working_state.derive_working_state(
            package_root,
            session_dir,
            write_snapshot=False,
        )
    )
    cache_key = _reviewed_region_digest_cache_key(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        require_blockers_resolved=require_blockers_resolved,
        working_state=state,
    )
    if _reviewed_region_digest_cache_enabled():
        with _DIGEST_CACHE_LOCK:
            cached = _DIGEST_CACHE.get(cache_key)
            if cached is not None:
                _DIGEST_CACHE_STATS["hits"] += 1
                return dict(cached) | {
                    "cache_hit": True,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            _DIGEST_CACHE_STATS["misses"] += 1
    regions, provenance = compile_reviewed_regions(
        package_root,
        session_dir,
        case_id=case_id,
        require_blockers_resolved=require_blockers_resolved,
        working_state=state,
    )
    digest = str(provenance["reviewed_region_set_sha256"])
    result = {
        "affected_source_frame_intervals": review_working_state.privacy_region_intervals(regions),
        "cache_hit": False,
        "case_id": case_id,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "effective_regions_hash": digest,
        "frame_region_hashes": review_working_state.privacy_region_frame_hashes(regions),
        "region_count": len(regions),
        "reviewed_region_identity": provenance["reviewed_region_identity"],
        "schema_version": "effective_privacy_region_digest.v1",
        "state_hash": provenance.get("state_hash"),
    }
    if _reviewed_region_digest_cache_enabled():
        with _DIGEST_CACHE_LOCK:
            _DIGEST_CACHE[cache_key] = copy.deepcopy(result)
            while len(_DIGEST_CACHE) > REVIEWED_REGION_DIGEST_CACHE_MAX_ENTRIES:
                _DIGEST_CACHE.pop(next(iter(_DIGEST_CACHE)))
    return result


def reviewed_region_digest_cache_stats() -> dict[str, int]:
    with _DIGEST_CACHE_LOCK:
        return dict(_DIGEST_CACHE_STATS) | {"entries": len(_DIGEST_CACHE)}


def clear_reviewed_region_digest_cache() -> None:
    with _DIGEST_CACHE_LOCK:
        _DIGEST_CACHE.clear()
        for key in _DIGEST_CACHE_STATS:
            _DIGEST_CACHE_STATS[key] = 0


def _reviewed_region_digest_cache_enabled() -> bool:
    return os.environ.get(REVIEWED_REGION_DIGEST_CACHE_DISABLE_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }


def _reviewed_region_digest_cache_key(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    require_blockers_resolved: bool,
    working_state: dict[str, Any],
) -> tuple[Any, ...]:
    from redactor import review_working_state

    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    renderer_settings = review_working_state.render_settings()
    return (
        str(package_root),
        str(session_dir),
        case_id,
        bool(require_blockers_resolved),
        stable_hash(render_configuration_payload(renderer_settings)),
        working_state.get("current_event_chain_head"),
        working_state.get("active_projection_hash"),
        working_state.get("state_hash"),
        _file_signatures(
            [
                package_root / "workspace_manifest.json",
                package_root / "review_entities.json",
                package_root / "reviewer_tasks.json",
                package_root / "latest_reviewed" / "current.json",
                package_root / "cases" / f"{case_id}.json",
                package_root / "regions" / f"{case_id}_current_regions.jsonl",
                session_dir / "events.jsonl",
                session_dir / "corrections.jsonl",
            ]
        ),
        _glob_file_signatures(session_dir / "latest_reviewed" / case_id, "**/*"),
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


def materialize_latest_reviewed_revision(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    accepted_revision: str | None = None,
    actor: dict[str, Any] | None = None,
    event_reason: str | None = None,
    publication_currentness_assertion: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from redactor import review_finalization, review_working_state

    total_start = time.monotonic()
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    effective_start = time.monotonic()
    reviewed_regions, provenance = compile_reviewed_regions(
        package_root,
        session_dir,
        case_id=case_id,
        require_blockers_resolved=True,
    )
    effective_duration_ms = int((time.monotonic() - effective_start) * 1000)
    manifest = case_manifests.load_case_manifest(
        review_working_state.safe_case_manifest_path(package_root, case_id)
    )
    source_path = review_finalization.resolve_package_path(package_root, manifest.source_path)
    if source_path is None or not source_path.exists():
        raise ValueError(f"{case_id}: source video is missing")
    source_info = media.probe_media(source_path)
    source_hash = media.sha256_file(source_path)
    if manifest.source_video_hash and source_hash != manifest.source_video_hash:
        raise ValueError(f"{case_id}: source hash mismatch")
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or rational_to_float(
        manifest.fps
    )
    if fps is None or fps <= 0:
        fps = 30.0
    expected_frame_count = int(source_info.video_stream.frame_count or manifest.frame_count or 0)
    if expected_frame_count <= 0:
        raise ValueError(f"{case_id}: source frame count is unknown")

    reviewed_hash = str(provenance["reviewed_region_set_sha256"])
    revision_id = f"latest_reviewed_{reviewed_hash[:16]}"
    revision_root = session_dir / "latest_reviewed" / case_id / revision_id
    temp_root = revision_root.with_name(f".{revision_id}.{uuid.uuid4().hex}.tmp")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=False)
    settings = review_working_state.render_settings()
    render_configuration = render_configuration_payload(settings)
    render_configuration_hash = stable_hash(render_configuration)
    publication_guard = publication_guard_payload(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        accepted_revision=accepted_revision,
        provenance=provenance,
    )
    regions_path = temp_root / "reviewed_regions.jsonl"
    review_finalization.write_privacy_region_records(regions_path, reviewed_regions)
    maybe_inject_failure("after_reviewed_regions_write")
    output_path = temp_root / "latest_reviewed.mp4"
    render_start = time.monotonic()
    rendered_frames, rendered_regions, audio_handling = privacy_rendering.render_privacy_video(
        source_path,
        output_path,
        privacy_rendering.regions_by_frame(reviewed_regions),
        settings=settings,
        fps=fps,
        width=source_info.video_stream.width,
        height=source_info.video_stream.height,
        source_has_audio=source_info.audio_stream is not None,
        overwrite=True,
        log_path=temp_root / "processing_log.jsonl",
    )
    maybe_inject_failure("after_video_render")
    render_duration_ms = int((time.monotonic() - render_start) * 1000)
    media_validation = privacy_rendering.validate_full_span_output(
        source_path,
        output_path,
        expected_frame_count=expected_frame_count,
        expected_width=source_info.video_stream.width,
        expected_height=source_info.video_stream.height,
        expected_fps=fps,
        source_has_audio=source_info.audio_stream is not None,
    )
    if media_validation.get("passed") is not True:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise ValueError(f"{case_id}: rendered Latest reviewed media failed full-span validation")
    maybe_inject_failure("after_media_validation")
    region_validation = privacy_rendering.verify_rendered_regions(
        source_path,
        output_path,
        [record.to_render_region() for record in reviewed_regions],
        settings=settings,
        frame_width=source_info.video_stream.width,
        frame_height=source_info.video_stream.height,
    )
    if region_validation.get("passed") is not True:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise ValueError(f"{case_id}: rendered Latest reviewed regions failed validation")
    maybe_inject_failure("after_region_validation")

    video_hash = media.sha256_file(output_path)
    regions_hash = media.sha256_file(regions_path)
    write_json(temp_root / "media_validation.json", media_validation)
    write_json(temp_root / "region_validation.json", region_validation)
    validation_hash = stable_hash(
        {
            "media_validation": media_validation,
            "region_validation": region_validation,
            "reviewed_regions_sha256": regions_hash,
        }
    )
    revision_manifest = {
        "active_corrections_hash": provenance["reviewed_region_identity"][
            "active_corrections_hash"
        ],
        "active_decisions_hash": provenance["reviewed_region_identity"][
            "active_decisions_hash"
        ],
        "active_projection_hash_schema_version": provenance["reviewed_region_identity"].get(
            "active_projection_hash_schema_version"
        ),
        "accepted_working_state_revision": accepted_revision,
        "automatic_baseline_region_set_sha256": provenance["reviewed_region_identity"][
            "automatic_baseline_region_set_sha256"
        ],
        "audio_handling": audio_handling,
        "case_id": case_id,
        "created_at_utc": media.utc_timestamp(),
        "media_validation": "media_validation.json",
        "media_validation_passed": True,
        "output_video": "latest_reviewed.mp4",
        "region_validation": "region_validation.json",
        "region_validation_passed": True,
        "publication_guard": publication_guard,
        "render_configuration": render_configuration,
        "renderer_configuration_sha256": render_configuration_hash,
        "rendered_frame_count": rendered_frames,
        "rendered_region_count": rendered_regions,
        "review_session_id": session_dir.name,
        "reviewed_region_identity": provenance["reviewed_region_identity"],
        "reviewed_region_set_sha256": reviewed_hash,
        "reviewed_regions": "reviewed_regions.jsonl",
        "reviewed_regions_sha256": regions_hash,
        "revision_id": revision_id,
        "schema_version": REVISION_SCHEMA_VERSION,
        "session_head_hash": publication_guard["session_head_hash"],
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "source_sha256": source_hash,
        "source_video_hash": source_hash,
        "validation_artifact_sha256": validation_hash,
        "video_artifact_sha256": video_hash,
        "workspace_manifest_input_sha256": publication_guard[
            "workspace_manifest_input_sha256"
        ],
        "workspace_manifest_sha256": publication_guard["workspace_manifest_sha256"],
    }
    revision_manifest["revision_manifest_sha256"] = stable_hash(revision_manifest)
    write_json(temp_root / "revision_manifest.json", revision_manifest)
    maybe_inject_failure("after_revision_manifest_write")
    try:
        assert_publication_guard_current(
            package_root=package_root,
            session_dir=session_dir,
            case_id=case_id,
            expected_guard=publication_guard,
        )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    if revision_root.exists():
        existing_manifest_path = revision_root / "revision_manifest.json"
        if not existing_manifest_path.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ReviewedRegionPublicationStaleError(
                f"{case_id}: existing Latest reviewed revision is incomplete"
            )
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("reviewed_region_set_sha256") != reviewed_hash:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ReviewedRegionPublicationStaleError(
                f"{case_id}: existing Latest reviewed revision identity mismatch"
            )
        shutil.rmtree(temp_root, ignore_errors=True)
        revision_manifest = existing_manifest
        video_hash = str(revision_manifest["video_artifact_sha256"])
        validation_hash = str(revision_manifest["validation_artifact_sha256"])
        rendered_frames = int(revision_manifest.get("rendered_frame_count") or 0)
        rendered_regions = int(revision_manifest.get("rendered_region_count") or 0)
        media_validation = json.loads(
            (revision_root / str(revision_manifest["media_validation"])).read_text(
                encoding="utf-8"
            )
        )
        region_validation = json.loads(
            (revision_root / str(revision_manifest["region_validation"])).read_text(
                encoding="utf-8"
            )
        )
    else:
        revision_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_root, revision_root)
        maybe_inject_failure("after_immutable_revision_move")
    validate_revision_artifacts(revision_root, revision_manifest)

    published_paths = {
        "media_validation": revision_root / "media_validation.json",
        "output_video": revision_root / "latest_reviewed.mp4",
        "region_validation": revision_root / "region_validation.json",
        "revision_manifest": revision_root / "revision_manifest.json",
        "reviewed_regions": revision_root / "reviewed_regions.jsonl",
    }
    role = {
        "accepted_working_state_revision": accepted_revision,
        "effective_regions_hash": reviewed_hash,
        "label": "Latest reviewed",
        "media_validation_passed": True,
        "path": published_paths["output_video"].relative_to(package_root).as_posix(),
        "published_working_state_revision": accepted_revision,
        "rendered_region_validation_passed": True,
        "reviewed_region_set_sha256": reviewed_hash,
        "revision_id": revision_id,
        "revision_manifest": published_paths["revision_manifest"]
        .relative_to(package_root)
        .as_posix(),
        "role": "latest_reviewed",
        "sha256": video_hash,
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "validation_artifact_sha256": validation_hash,
        "video_artifact_sha256": video_hash,
    }
    correction_render_verifications = review_working_state.verify_rendered_manual_corrections(
        source_path=source_path,
        output_path=published_paths["output_video"],
        manifest=manifest,
        corrections=[
            correction
            for correction in review_working_state.load_active_corrections(
                session_dir,
                active_correction_ids=set(
                    review_working_state.derive_working_state(package_root, session_dir).get(
                        "active_correction_ids"
                    )
                    or []
                ),
            )
            if correction.case_id == case_id
        ],
    )
    publication_start = time.monotonic()
    publish_updates = publish_current_revision(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        role=role,
        revision_manifest=revision_manifest,
        paths=published_paths,
        publication_guard=publication_guard,
        accepted_revision=accepted_revision,
        publication_currentness_assertion=publication_currentness_assertion,
    )
    publication_duration_ms = int((time.monotonic() - publication_start) * 1000)
    state = review_working_state.derive_working_state(package_root, session_dir)
    materialization = {
        "accepted_working_state_revision": accepted_revision,
        "audio_handling": audio_handling,
        "case_id": case_id,
        "created_at_utc": revision_manifest["created_at_utc"],
        "effective_region_count": len(reviewed_regions),
        "effective_regions": published_paths["reviewed_regions"]
        .relative_to(package_root)
        .as_posix(),
        "effective_regions_hash": reviewed_hash,
        "latest_reviewed": role,
        "materialized": True,
        "media_validation": media_validation,
        "correction_render_verifications": correction_render_verifications,
        "output_video": role["path"],
        "publication_updates": publish_updates,
        "publication_guard": publication_guard,
        "rendered_frame_count": rendered_frames,
        "rendered_region_count": rendered_regions,
        "rendered_region_validation": region_validation,
        "revision_id": revision_id,
        "revision_manifest": role["revision_manifest"],
        "schema_version": "latest_reviewed_materialization.v2",
        "state_hash": state["state_hash"],
        "timing": {
            "accepted_working_state_revision": accepted_revision,
            "case_id": case_id,
            "effective_region_derivation_duration_ms": effective_duration_ms,
            "output_frame_count": rendered_frames,
            "publication_duration_ms": publication_duration_ms,
            "render_encode_duration_ms": render_duration_ms,
            "schema_version": "materialization_timing.v1",
            "total_duration_ms": int((time.monotonic() - total_start) * 1000),
        },
        "validation_artifact_sha256": validation_hash,
        "video_sha256": video_hash,
    } | provenance
    state.setdefault("materialized_latest_reviewed", {})[case_id] = role | {
        "created_at_utc": materialization["created_at_utc"],
        "effective_regions_hash": reviewed_hash,
    }
    state["state_hash"] = review_working_state.state_hash(state)
    review_working_state.write_state_snapshot(session_dir, state)
    review_working_state.write_json(
        session_dir / "state" / f"{case_id}_latest_reviewed_manifest.json",
        materialization,
    )
    review_working_state.append_manual_correction_persistence_audit(
        package_root,
        case_id=case_id,
        materialization=materialization,
        provenance=provenance,
    )
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
                "active_projection_hash_schema_version": (
                    review_working_state.ACTIVE_PROJECTION_HASH_SCHEMA_VERSION
                ),
                "case_id": case_id,
                "effective_regions_hash": reviewed_hash,
                "latest_reviewed": role,
                "media_validation_passed": True,
                "published_working_state_revision": accepted_revision,
                "rendered_region_validation_passed": True,
                "state_hash": state["state_hash"],
                "target_effective_regions_hash": reviewed_hash,
            },
            note=event_reason or "Latest reviewed media materialized.",
            application_hashes={"latest_reviewed_sha256": video_hash},
        )
    return materialization


def publish_automatic_reuse_latest_reviewed_revision(
    package_root: Path,
    session_dir: Path,
    *,
    case_id: str,
    accepted_revision: str | None = None,
    actor: dict[str, Any] | None = None,
    event_reason: str | None = None,
    publication_currentness_assertion: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from redactor import review_finalization, review_working_state

    total_start = time.monotonic()
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    effective_start = time.monotonic()
    reviewed_regions, provenance = compile_reviewed_regions(
        package_root,
        session_dir,
        case_id=case_id,
        require_blockers_resolved=True,
    )
    effective_duration_ms = int((time.monotonic() - effective_start) * 1000)
    state = review_working_state.derive_working_state(
        package_root,
        session_dir,
        write_snapshot=False,
    )
    active_corrections = [
        correction
        for correction in review_working_state.load_active_corrections(
            session_dir,
            active_correction_ids=set(state.get("active_correction_ids") or []),
        )
        if correction.case_id == case_id
    ]
    object_overrides = [
        override
        for override in dict(state.get("object_policy_overrides") or {}).values()
        if str(dict(override).get("case_id") or "") == case_id
    ]
    media_decisions = [
        decision
        for decision in dict(
            dict(state.get("task_decisions_by_case") or {}).get(case_id) or {}
        ).values()
        if str(dict(decision).get("action") or "") == ACTION_NOT_SENSITIVE
    ]
    if active_corrections or object_overrides or media_decisions:
        raise ReviewedRegionCompilationError(
            f"{case_id}: Automatic reuse publication requires no human media edits"
        )
    manifest = case_manifests.load_case_manifest(
        review_working_state.safe_case_manifest_path(package_root, case_id)
    )
    automatic_role = latest_reviewed_automatic_reuse_role(manifest)
    if automatic_role is None:
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic media role is missing")
    automatic_path = case_ids.contained_path(package_root, *Path(automatic_role.path).parts)
    if not automatic_path.exists():
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic media is missing")
    automatic_hash = media.sha256_file(automatic_path)
    if automatic_role.sha256 and automatic_role.sha256 != automatic_hash:
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic media hash mismatch")
    automatic_info = media.probe_media(automatic_path)
    observed_frames = int(automatic_info.video_stream.frame_count or 0)
    expected_frame_count = int(manifest.frame_count or observed_frames or 0)
    if expected_frame_count <= 0:
        raise ReviewedRegionCompilationError(f"{case_id}: media frame count is unknown")
    if observed_frames and observed_frames != expected_frame_count:
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic media frame count mismatch")
    if manifest.width and automatic_info.video_stream.width != manifest.width:
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic media width mismatch")
    if manifest.height and automatic_info.video_stream.height != manifest.height:
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic media height mismatch")
    source_path = case_ids.contained_path(
        package_root,
        *Path(str(manifest.source_path or "")).parts,
    )
    source_hash = manifest.source_video_hash or media.sha256_file(source_path)

    reviewed_hash = str(provenance["reviewed_region_set_sha256"])
    revision_id = f"latest_reviewed_{reviewed_hash[:16]}"
    revision_root = session_dir / "latest_reviewed" / case_id / revision_id
    temp_root = revision_root.with_name(f".{revision_id}.{uuid.uuid4().hex}.tmp")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=False)
    settings = review_working_state.render_settings()
    render_configuration = render_configuration_payload(settings)
    render_configuration_hash = stable_hash(render_configuration)
    publication_guard = publication_guard_payload(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        accepted_revision=accepted_revision,
        provenance=provenance,
    )
    regions_path = temp_root / "reviewed_regions.jsonl"
    review_finalization.write_privacy_region_records(regions_path, reviewed_regions)
    output_path = temp_root / "latest_reviewed.mp4"
    try:
        os.link(automatic_path, output_path)
        reuse_mode = "hardlink"
    except OSError:
        shutil.copy2(automatic_path, output_path)
        reuse_mode = "verified_copy"
    video_hash = media.sha256_file(output_path)
    if video_hash != automatic_hash:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise ReviewedRegionCompilationError(f"{case_id}: Automatic reuse media hash mismatch")
    regions_hash = media.sha256_file(regions_path)
    media_validation = {
        "automatic_media_sha256": automatic_hash,
        "automatic_reuse_mode": reuse_mode,
        "expected_frame_count": expected_frame_count,
        "expected_height": manifest.height,
        "expected_width": manifest.width,
        "observed_frame_count": observed_frames or expected_frame_count,
        "observed_height": automatic_info.video_stream.height,
        "observed_width": automatic_info.video_stream.width,
        "passed": True,
        "schema_version": "latest_reviewed_automatic_reuse_media_validation.v1",
        "source_has_audio": automatic_info.audio_stream is not None,
        "validation_mode": "automatic_reuse_publication",
    }
    region_validation = {
        "automatic_reuse": True,
        "passed": True,
        "rendered_region_count": len(reviewed_regions),
        "schema_version": "latest_reviewed_automatic_reuse_region_validation.v1",
        "validation_mode": "automatic_reuse_publication",
    }
    write_json(temp_root / "media_validation.json", media_validation)
    write_json(temp_root / "region_validation.json", region_validation)
    validation_hash = stable_hash(
        {
            "media_validation": media_validation,
            "region_validation": region_validation,
            "reviewed_regions_sha256": regions_hash,
        }
    )
    revision_manifest = {
        "accepted_working_state_revision": accepted_revision,
        "active_corrections_hash": provenance["reviewed_region_identity"][
            "active_corrections_hash"
        ],
        "active_decisions_hash": provenance["reviewed_region_identity"][
            "active_decisions_hash"
        ],
        "active_projection_hash_schema_version": provenance["reviewed_region_identity"].get(
            "active_projection_hash_schema_version"
        ),
        "automatic_baseline_region_set_sha256": provenance["reviewed_region_identity"][
            "automatic_baseline_region_set_sha256"
        ],
        "automatic_reuse_mode": reuse_mode,
        "audio_handling": f"automatic_reuse_{reuse_mode}",
        "case_id": case_id,
        "created_at_utc": media.utc_timestamp(),
        "media_validation": "media_validation.json",
        "media_validation_passed": True,
        "output_video": "latest_reviewed.mp4",
        "publication_guard": publication_guard,
        "region_validation": "region_validation.json",
        "region_validation_passed": True,
        "render_configuration": render_configuration,
        "renderer_configuration_sha256": render_configuration_hash,
        "rendered_frame_count": observed_frames or expected_frame_count,
        "rendered_region_count": len(reviewed_regions),
        "review_session_id": session_dir.name,
        "reviewed_region_identity": provenance["reviewed_region_identity"],
        "reviewed_region_set_sha256": reviewed_hash,
        "reviewed_regions": "reviewed_regions.jsonl",
        "reviewed_regions_sha256": regions_hash,
        "revision_id": revision_id,
        "schema_version": REVISION_SCHEMA_VERSION,
        "session_head_hash": publication_guard["session_head_hash"],
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "source_sha256": source_hash,
        "source_video_hash": source_hash,
        "validation_artifact_sha256": validation_hash,
        "video_artifact_sha256": video_hash,
        "workspace_manifest_input_sha256": publication_guard[
            "workspace_manifest_input_sha256"
        ],
        "workspace_manifest_sha256": publication_guard["workspace_manifest_sha256"],
    }
    revision_manifest["revision_manifest_sha256"] = stable_hash(revision_manifest)
    write_json(temp_root / "revision_manifest.json", revision_manifest)
    try:
        assert_publication_guard_current(
            package_root=package_root,
            session_dir=session_dir,
            case_id=case_id,
            expected_guard=publication_guard,
        )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    if revision_root.exists():
        existing_manifest_path = revision_root / "revision_manifest.json"
        if not existing_manifest_path.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ReviewedRegionPublicationStaleError(
                f"{case_id}: existing Latest reviewed revision is incomplete"
            )
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("reviewed_region_set_sha256") != reviewed_hash:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ReviewedRegionPublicationStaleError(
                f"{case_id}: existing Latest reviewed revision identity mismatch"
            )
        shutil.rmtree(temp_root, ignore_errors=True)
        revision_manifest = existing_manifest
        video_hash = str(revision_manifest["video_artifact_sha256"])
        validation_hash = str(revision_manifest["validation_artifact_sha256"])
        media_validation = json.loads(
            (revision_root / str(revision_manifest["media_validation"])).read_text(
                encoding="utf-8"
            )
        )
        region_validation = json.loads(
            (revision_root / str(revision_manifest["region_validation"])).read_text(
                encoding="utf-8"
            )
        )
    else:
        revision_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_root, revision_root)
    validate_revision_artifacts(revision_root, revision_manifest)

    published_paths = {
        "media_validation": revision_root / "media_validation.json",
        "output_video": revision_root / "latest_reviewed.mp4",
        "region_validation": revision_root / "region_validation.json",
        "revision_manifest": revision_root / "revision_manifest.json",
        "reviewed_regions": revision_root / "reviewed_regions.jsonl",
    }
    role = {
        "accepted_working_state_revision": accepted_revision,
        "effective_regions_hash": reviewed_hash,
        "label": "Latest reviewed",
        "media_validation_passed": True,
        "path": published_paths["output_video"].relative_to(package_root).as_posix(),
        "published_working_state_revision": accepted_revision,
        "rendered_region_validation_passed": True,
        "reviewed_region_set_sha256": reviewed_hash,
        "revision_id": revision_id,
        "revision_manifest": published_paths["revision_manifest"]
        .relative_to(package_root)
        .as_posix(),
        "role": "latest_reviewed",
        "sha256": video_hash,
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "validation_artifact_sha256": validation_hash,
        "video_artifact_sha256": video_hash,
    }
    publication_start = time.monotonic()
    publish_updates = publish_current_revision(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        role=role,
        revision_manifest=revision_manifest,
        paths=published_paths,
        publication_guard=publication_guard,
        accepted_revision=accepted_revision,
        publication_currentness_assertion=publication_currentness_assertion,
    )
    publication_duration_ms = int((time.monotonic() - publication_start) * 1000)
    state = review_working_state.derive_working_state(package_root, session_dir)
    materialization = {
        "accepted_working_state_revision": accepted_revision,
        "audio_handling": revision_manifest["audio_handling"],
        "case_id": case_id,
        "created_at_utc": revision_manifest["created_at_utc"],
        "effective_region_count": len(reviewed_regions),
        "effective_regions": published_paths["reviewed_regions"]
        .relative_to(package_root)
        .as_posix(),
        "effective_regions_hash": reviewed_hash,
        "latest_reviewed": role,
        "materialized": True,
        "media_validation": media_validation,
        "output_video": role["path"],
        "publication_guard": publication_guard,
        "publication_updates": publish_updates,
        "rendered_frame_count": int(revision_manifest.get("rendered_frame_count") or 0),
        "rendered_region_count": len(reviewed_regions),
        "rendered_region_validation": region_validation,
        "revision_id": revision_id,
        "revision_manifest": role["revision_manifest"],
        "schema_version": "latest_reviewed_automatic_reuse_materialization.v1",
        "state_hash": state["state_hash"],
        "timing": {
            "accepted_working_state_revision": accepted_revision,
            "case_id": case_id,
            "effective_region_derivation_duration_ms": effective_duration_ms,
            "output_frame_count": int(revision_manifest.get("rendered_frame_count") or 0),
            "publication_duration_ms": publication_duration_ms,
            "render_encode_duration_ms": 0,
            "schema_version": "materialization_timing.v1",
            "total_duration_ms": int((time.monotonic() - total_start) * 1000),
        },
        "validation_artifact_sha256": validation_hash,
        "video_sha256": video_hash,
    } | provenance
    state.setdefault("materialized_latest_reviewed", {})[case_id] = role | {
        "created_at_utc": materialization["created_at_utc"],
        "effective_regions_hash": reviewed_hash,
    }
    state["state_hash"] = review_working_state.state_hash(state)
    review_working_state.write_state_snapshot(session_dir, state)
    review_working_state.write_json(
        session_dir / "state" / f"{case_id}_latest_reviewed_manifest.json",
        materialization,
    )
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
                "active_projection_hash_schema_version": (
                    review_working_state.ACTIVE_PROJECTION_HASH_SCHEMA_VERSION
                ),
                "case_id": case_id,
                "effective_regions_hash": reviewed_hash,
                "latest_reviewed": role,
                "media_validation_passed": True,
                "published_working_state_revision": accepted_revision,
                "rendered_region_validation_passed": True,
                "state_hash": state["state_hash"],
                "target_effective_regions_hash": reviewed_hash,
            },
            note=event_reason or "Latest reviewed media published from Automatic media.",
            application_hashes={"latest_reviewed_sha256": video_hash},
        )
    return materialization


def latest_reviewed_automatic_reuse_role(
    manifest: case_manifests.CaseManifest,
) -> case_manifests.VideoRole | None:
    return next(
        (
            role
            for role in manifest.video_roles
            if role.role in {"automatic", "current_result"}
        ),
        None,
    )


def require_supported_workspace(package_root: Path) -> dict[str, Any]:
    from redactor import review_working_state

    workspace = review_working_state.workspace_manifest_payload(package_root)
    if workspace.get("workspace_kind") != "unified_case_review_workspace":
        raise ReviewedRegionCompilationError("not a canonical unified review workspace")
    publication = dict(workspace.get("latest_reviewed_publication") or {})
    if publication.get("status") not in SUPPORTED_PUBLICATION_STATUSES:
        raise ReviewedRegionCompilationError("Latest reviewed publication is not supported")
    return workspace


def validate_source_identity(
    package_root: Path,
    workspace: dict[str, Any],
    manifest: case_manifests.CaseManifest,
) -> None:
    source_path = package_root / str(manifest.source_path or "")
    if not source_path.exists():
        raise ReviewedRegionCompilationError(f"{manifest.case_id}: source video is missing")
    source_hash = media.sha256_file(source_path)
    expected_hashes = {
        "case_manifest": manifest.source_video_hash,
        "source_identity": dict(
            case_scoped_workspace_value(workspace, "source_identity", manifest.case_id) or {}
        ).get("sha256"),
        "source_metadata": dict(
            case_scoped_workspace_value(workspace, "source_metadata", manifest.case_id) or {}
        ).get("sha256"),
        "automatic_preview": dict(
            case_scoped_workspace_value(workspace, "automatic_preview", manifest.case_id) or {}
        ).get("source_sha256"),
    }
    for name, expected in expected_hashes.items():
        if expected and str(expected) != source_hash:
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: {name} source hash mismatch"
            )


def load_authoritative_baseline(
    *,
    package_root: Path,
    workspace: dict[str, Any],
    manifest: case_manifests.CaseManifest,
    baseline_path: Path,
) -> list[PrivacyRegionRecord]:
    from redactor import review_finalization

    if not baseline_path.exists():
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: authoritative automatic baseline regions are missing"
        )
    regions = review_finalization.load_privacy_region_records(baseline_path)
    selected = set(selected_categories(workspace))
    category_errors = sorted({record.category for record in regions} - selected)
    if category_errors:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: baseline contains categories outside selection: {category_errors}"
        )
    content_identities = dict(
        case_scoped_workspace_value(workspace, "content_identities", manifest.case_id) or {}
    )
    baseline_identity = dict(content_identities.get("automatic_baseline_regions") or {})
    if not baseline_identity:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: baseline region content identity is missing"
        )
    relative_path = baseline_path.relative_to(package_root).as_posix()
    acceptable_paths = {
        relative_path,
        f"{package_root.name}/{relative_path}",
    }
    if baseline_identity.get("path") not in acceptable_paths:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: baseline region path identity mismatch"
        )
    if baseline_identity.get("sha256") != media.sha256_file(baseline_path):
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: baseline region hash identity mismatch"
        )
    automatic = dict(
        case_scoped_workspace_value(workspace, "automatic_preview", manifest.case_id) or {}
    )
    if int(automatic.get("region_count") or 0) != len(regions):
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: baseline region count does not match automatic preview identity"
        )
    return regions


def validate_baseline_category_run_binding(
    *,
    package_root: Path,
    workspace: dict[str, Any],
    manifest: case_manifests.CaseManifest,
    baseline_path: Path,
    baseline_regions: list[PrivacyRegionRecord],
) -> dict[str, Any]:
    selected = selected_categories(workspace)
    category_identities = {
        str(category): dict(identity or {})
        for category, identity in dict(
            case_scoped_workspace_value(
                workspace,
                "category_run_manifest_identities",
                manifest.case_id,
            )
            or {}
        ).items()
    }
    missing = sorted(set(selected) - set(category_identities))
    extra = sorted(set(category_identities) - set(selected))
    if missing or extra:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: category run identities must match selected categories "
            f"(missing={missing}, extra={extra})"
        )
    expected_identity_hash = stable_hash(category_identities)
    declared_identity_hash = case_scoped_workspace_value(
        workspace,
        "category_run_manifest_identity_hash",
        manifest.case_id,
    )
    if declared_identity_hash != expected_identity_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: category run identity hash mismatch"
        )

    freeze_path = package_root / str(
        workspace.get("configuration_freeze") or "configuration_freeze.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    freeze_hash = stable_hash(freeze) if freeze else None
    declared_freeze_hash = workspace.get("configuration_freeze_hash")
    if declared_freeze_hash and declared_freeze_hash != freeze_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: configuration freeze hash mismatch"
        )
    freeze_category_configs = dict(
        case_scoped_workspace_value(
            workspace,
            "category_configuration_identities",
            manifest.case_id,
        )
        or freeze.get("category_configuration_identities")
        or {}
    )

    category_rows: dict[str, dict[str, Any]] = {}
    artifact_ids: set[str] = set()
    artifact_paths: set[str] = set()
    for category, identity in sorted(category_identities.items()):
        if identity.get("category") != category:
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: {category} category run identity category mismatch"
            )
        artifact_id = str(identity.get("artifact_id") or "")
        if not artifact_id:
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: {category} category run artifact ID is missing"
            )
        if artifact_id in artifact_ids:
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: duplicate category run artifact ID {artifact_id}"
            )
        artifact_ids.add(artifact_id)
        run_manifest_path = resolve_category_run_manifest_path(package_root, identity)
        relative_artifact_path = run_manifest_path.resolve().as_posix()
        if relative_artifact_path in artifact_paths:
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: duplicate category run manifest path"
            )
        artifact_paths.add(relative_artifact_path)
        if not run_manifest_path.exists():
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: {category} category run manifest is missing"
            )
        if identity.get("sha256") != media.sha256_file(run_manifest_path):
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: {category} category run artifact hash mismatch"
            )
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        validate_category_run_manifest_binding(
            category=category,
            identity=identity,
            manifest=manifest,
            run_manifest=run_manifest,
            freeze_category_configs=freeze_category_configs,
        )
        category_rows[category] = {
            "artifact_sha256": identity["sha256"],
            "category_run_content_sha256": identity["category_run_content_sha256"],
            "configuration_hash": identity["configuration_hash"],
            "path": identity["path"],
            "processed_span": identity.get("processed_span"),
            "source_sha256": identity["source_sha256"],
        }

    baseline_semantic_hash = semantic_region_set_hash(baseline_regions)
    return {
        "artifact_sha256": media.sha256_file(baseline_path),
        "automatic_baseline_region_count": len(baseline_regions),
        "automatic_baseline_region_set_sha256": baseline_semantic_hash,
        "case_id": manifest.case_id,
        "category_run_manifest_identity_hash": expected_identity_hash,
        "category_run_manifest_identity_count": len(category_rows),
        "category_run_manifests": category_rows,
        "configuration_freeze_hash": freeze_hash,
        "schema_version": "automatic_baseline_category_run_binding.v1",
        "selected_categories": selected,
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "source_sha256": manifest.source_video_hash,
    }


def resolve_category_run_manifest_path(
    package_root: Path,
    identity: dict[str, Any],
) -> Path:
    raw_path = Path(str(identity.get("path") or ""))
    candidates = [raw_path] if raw_path.is_absolute() else [
        package_root / raw_path,
        package_root.parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def validate_category_run_manifest_binding(
    *,
    category: str,
    identity: dict[str, Any],
    manifest: case_manifests.CaseManifest,
    run_manifest: dict[str, Any],
    freeze_category_configs: dict[str, Any],
) -> None:
    if run_manifest.get("case_id") != manifest.case_id:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category run case_id mismatch"
        )
    if run_manifest.get("category") != category:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category run category mismatch"
        )
    source_lineage = dict(run_manifest.get("source_lineage") or {})
    if source_lineage.get("source_sha256") != manifest.source_video_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category run source hash mismatch"
        )
    if identity.get("source_sha256") != manifest.source_video_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category identity source hash mismatch"
        )
    processed_span = dict(source_lineage.get("processed_span") or {})
    expected_span = {
        "first_processed_frame": manifest.source_frame_start,
        "last_processed_frame": manifest.source_frame_end,
        "source_frame_count": manifest.frame_count,
    }
    for key, expected in expected_span.items():
        if processed_span.get(key) != expected:
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: {category} processed span {key} mismatch"
            )
    configuration_identity = dict(run_manifest.get("configuration_identity") or {})
    configuration_hash = str(configuration_identity.get("configuration_hash") or "")
    if not configuration_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} configuration hash is missing"
        )
    if identity.get("configuration_hash") != configuration_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category identity configuration mismatch"
        )
    freeze_config = dict(freeze_category_configs.get(category) or {})
    if freeze_config and freeze_config.get("configuration_hash") != configuration_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} configuration freeze identity mismatch"
        )
    failed_stages = [
        str(dict(stage).get("stage") or dict(stage).get("category") or "stage")
        for stage in list(run_manifest.get("stage_outcomes") or [])
        if str(dict(stage).get("status") or "").lower() in {"failed", "error"}
    ]
    if failed_stages:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category run failed stages: {failed_stages}"
        )
    content_hash = str(run_manifest.get("category_run_content_sha256") or "")
    if not content_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category run content hash is missing"
        )
    if identity.get("category_run_content_sha256") != content_hash:
        raise ReviewedRegionCompilationError(
            f"{manifest.case_id}: {category} category run content hash mismatch"
        )


def selected_categories(workspace: dict[str, Any]) -> list[str]:
    return sorted(str(item) for item in list(workspace.get("selected_categories") or []))


def case_scoped_workspace_value(
    workspace: dict[str, Any],
    key: str,
    case_id: str,
) -> Any:
    by_case = workspace.get(f"{key}_by_case")
    if isinstance(by_case, dict) and case_id in by_case:
        return by_case[case_id]
    return workspace.get(key)


def compile_object_policy_regions(
    *,
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    baseline_regions: list[PrivacyRegionRecord],
    source_entities: list[dict[str, Any]],
    effective_entities: list[dict[str, Any]],
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    from redactor import review_working_state

    kept: list[PrivacyRegionRecord] = []
    suppressed_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    entity_infos: list[dict[str, Any]] = []
    for index, entity in enumerate(effective_entities):
        entity_infos.append(
            {
                "category": str(entity.get("category") or ""),
                "entity": entity,
                "index": index,
                "policy": review_working_state.inherited_policy(entity),
                "source_track_ids": sorted(review_working_state.entity_track_ids(entity)),
                "track_variants": review_working_state.entity_track_variants(entity),
            }
        )
    entity_match_counts = [0 for _item in entity_infos]
    for record in baseline_regions:
        record_values = review_working_state.region_identity_values(record)
        matched_infos = [
            info
            for info in entity_infos
            if (not info["category"] or info["category"] == record.category)
            and bool(record_values & info["track_variants"])
        ]
        for info in matched_infos:
            entity_match_counts[int(info["index"])] += 1
        matched_entities = [dict(info["entity"]) for info in matched_infos]
        suppressing = [
            dict(info["entity"])
            for info in matched_infos
            if info["policy"] in review_working_state.SUPPRESSING_POLICIES
        ]
        if suppressing:
            suppressed_rows.append(
                {
                    "baseline_region_id": baseline_region_id(record),
                    "category": record.category,
                    "entity_ids": [str(entity.get("entity_id") or "") for entity in suppressing],
                    "reason": "object_policy_suppressed",
                    "source_frame": record.source_frame,
                }
            )
            continue
        kept.append(annotate_region(record, matched_entities=matched_entities, manifest=manifest))

    added_regions: list[PrivacyRegionRecord] = []
    existing_keys = {reviewed_region_unique_key(record) for record in kept}
    for info in entity_infos:
        entity = dict(info["entity"])
        policy = str(info["policy"])
        entity_rows.append(
            {
                "category": entity.get("category"),
                "entity_id": entity.get("entity_id"),
                "matched_baseline_region_count": entity_match_counts[int(info["index"])],
                "policy": policy,
                "source_track_ids": list(info["source_track_ids"]),
            }
        )
        if (
            policy != review_working_state.POLICY_REDACT
            or entity.get("reviewer_override") is not True
        ):
            continue
        for source_record in review_working_state.source_backed_entity_regions(
            package_root=package_root,
            manifest=manifest,
            entity=entity,
        ):
            annotated = annotate_region(
                source_record,
                matched_entities=[entity],
                manifest=manifest,
                provenance_override="review_object_policy_source_track",
            )
            key = reviewed_region_unique_key(annotated)
            if key not in existing_keys:
                added_regions.append(annotated)
                existing_keys.add(key)

    final_regions = sorted_regions([*kept, *added_regions])
    return final_regions, {
        "added_region_count": len(added_regions),
        "entity_policies": entity_rows,
        "input_region_count": len(baseline_regions),
        "output_region_count": len(final_regions),
        "schema_version": "reviewed_object_policy_region_application.v1",
        "source_entity_count": len(source_entities),
        "suppressed_region_count": len(suppressed_rows),
        "suppressed_regions": suppressed_rows,
    }


def apply_not_sensitive_task_suppressions(
    regions: list[PrivacyRegionRecord],
    *,
    package_root: Path,
    case_id: str,
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    suppressions = not_sensitive_task_suppressions(
        package_root=package_root,
        case_id=case_id,
        decisions_by_case=decisions_by_case,
    )
    if not suppressions:
        return regions, {
            "schema_version": "task_not_sensitive_region_suppression.v1",
            "suppressed_region_count": 0,
            "suppressed_regions": [],
            "suppression_count": 0,
        }

    kept: list[PrivacyRegionRecord] = []
    suppressed_rows: list[dict[str, Any]] = []
    for record in regions:
        matched = next(
            (
                suppression
                for suppression in suppressions
                if record_matches_task_suppression(record, suppression)
            ),
            None,
        )
        if matched is None:
            kept.append(record)
            continue
        suppressed_rows.append(
            {
                "baseline_region_id": baseline_region_id(record),
                "category": record.category,
                "reason": "review_task_not_sensitive",
                "source_frame": record.source_frame,
                "task_id": matched["task_id"],
            }
        )
    return sorted_regions(kept), {
        "schema_version": "task_not_sensitive_region_suppression.v1",
        "suppressed_region_count": len(suppressed_rows),
        "suppressed_regions": suppressed_rows,
        "suppression_count": len(suppressions),
    }


def not_sensitive_task_suppressions(
    *,
    package_root: Path,
    case_id: str,
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    from redactor import review_working_state

    tasks = {
        str(task.get("review_task_id") or ""): dict(task)
        for task in load_review_tasks(package_root)
        if isinstance(task, dict) and str(task.get("case_id") or "") == case_id
    }
    suppressions: list[dict[str, Any]] = []
    for task_id, decision in sorted(dict(decisions_by_case.get(case_id) or {}).items()):
        if str(decision.get("action") or "") != ACTION_NOT_SENSITIVE:
            continue
        task = tasks.get(str(task_id))
        if not task or str(task.get("case_id") or "") != case_id or task.get("technical_only"):
            continue
        category = str(task.get("category") or "")
        entity_ids = {str(item) for item in list(task.get("target_entity_ids") or []) if item}
        binding = dict(task.get("target_binding") or {})
        for key in ("target_entity_ids", "entity_ids", "source_entity_ids"):
            entity_ids.update(str(item) for item in list(binding.get(key) or []) if item)
        track_values: set[str] = set()
        for key in (
            "technical_track_ids",
            "reviewed_region_track_ids",
            "target_track_ids",
            "automatic_source_track_ids",
            "member_detector_tracker_ids",
        ):
            track_values.update(str(item) for item in list(task.get(key) or []) if item)
            track_values.update(str(item) for item in list(binding.get(key) or []) if item)
        track_variants: set[str] = set()
        for track_id in track_values:
            track_variants.update(review_working_state.track_variants(track_id, category=category))
        start = int(task.get("source_frame_start") or binding.get("source_frame_start") or 0)
        end = int(task.get("source_frame_end") or binding.get("source_frame_end") or start)
        if end < start:
            start, end = end, start
        suppressions.append(
            {
                "broad_category_span": not entity_ids and not track_variants,
                "category": category,
                "entity_ids": entity_ids,
                "source_frame_end": end,
                "source_frame_start": start,
                "task_id": str(task_id),
                "track_variants": track_variants,
            }
        )
    return suppressions


def record_matches_task_suppression(
    record: PrivacyRegionRecord,
    suppression: dict[str, Any],
) -> bool:
    from redactor import review_working_state

    if record.category != suppression["category"]:
        return False
    if not (
        int(suppression["source_frame_start"])
        <= record.source_frame
        <= int(suppression["source_frame_end"])
    ):
        return False
    if suppression.get("broad_category_span"):
        return True

    source = dict(record.source_record)
    record_entity_values = {
        str(source.get("reviewed_entity_id") or ""),
        str(source.get("entity_id") or ""),
        str(source.get("reviewed_entity_uuid") or ""),
    }
    entity_ids = {str(item) for item in suppression.get("entity_ids") or []}
    if entity_ids and entity_ids & {value for value in record_entity_values if value}:
        return True

    track_variants = {str(item) for item in suppression.get("track_variants") or []}
    if not track_variants:
        return False
    record_variants: set[str] = set()
    for value in review_working_state.region_identity_values(record):
        record_variants.update(review_working_state.track_variants(value, category=record.category))
    return bool(record_variants & track_variants)


def apply_review_corrections(
    base_regions: list[PrivacyRegionRecord],
    corrections: list[ReviewCorrection],
    *,
    manifest: case_manifests.CaseManifest,
    selected_categories: list[str],
    effective_entities: list[dict[str, Any]],
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    active = list(base_regions)
    applied: list[dict[str, Any]] = []
    added: list[PrivacyRegionRecord] = []
    replaced_count = 0
    selected = set(selected_categories)
    allowed_manual_categories = set(MANUAL_REDACTION_CATEGORIES)
    fps = rational_to_float(manifest.fps) or 30.0
    for correction in corrections:
        if correction.case_id != manifest.case_id:
            continue
        if (
            correction.category not in selected
            and correction.category not in allowed_manual_categories
        ):
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: correction category is outside selected workspace categories"
            )
        if (
            manifest.source_video_hash
            and correction.source_video_hash != manifest.source_video_hash
        ):
            raise ReviewedRegionCompilationError(
                f"{manifest.case_id}: correction source hash mismatch"
            )
        suppressed, mode = correction_suppression_targets(
            active,
            correction,
            effective_entities=effective_entities,
            source_frame_offset=manifest.source_frame_start,
        )
        if correction.operation in SUPPRESSION_OPERATIONS:
            active = [
                record
                for record in active
                if reviewed_region_identifier(record) not in suppressed
                and stable_hash(record.to_dict()) not in suppressed
            ]
            replaced_count += len(suppressed)
        boxes = interpolate_correction_boxes(correction)
        correction_added = 0
        if correction.operation in {OP_ADD, OP_REPLACE}:
            for source_frame, box in sorted(boxes.items()):
                render_frame = source_frame - manifest.source_frame_start
                if render_frame < 0 or render_frame >= manifest.frame_count:
                    raise ReviewedRegionCompilationError(
                        f"{manifest.case_id}: correction frame is outside source span"
                    )
                frame_suppressed = [
                    record
                    for record in base_regions
                    if reviewed_region_identifier(record) in suppressed
                    and record.source_frame == render_frame
                ]
                added.append(
                    correction_region_record(
                        correction,
                        bbox_xyxy=box,
                        render_frame=render_frame,
                        source_frame=source_frame,
                        fps=fps,
                        replaced_records=frame_suppressed,
                    )
                )
                correction_added += 1
        applied.append(
            {
                "added_region_count": correction_added,
                "correction_hash": correction.correction_hash(),
                "correction_id": correction.correction_id,
                "operation": correction.operation,
                "replacement_mode": mode,
                "replaced_region_count": len(suppressed),
                "revision_id": correction.revision_id,
                "source_frame_end": correction.source_frame_end,
                "source_frame_start": correction.source_frame_start,
            }
        )
    final_regions = sorted_regions([*active, *added])
    return final_regions, {
        "added_region_count": len(added),
        "applied_corrections": applied,
        "base_region_count": len(base_regions),
        "final_region_count": len(final_regions),
        "replaced_region_count": replaced_count,
        "schema_version": "reviewed_correction_application.v1",
    }


def correction_suppression_targets(
    records: list[PrivacyRegionRecord],
    correction: ReviewCorrection,
    *,
    effective_entities: list[dict[str, Any]],
    source_frame_offset: int,
) -> tuple[set[str], str]:
    from redactor import review_working_state

    if correction.operation not in SUPPRESSION_OPERATIONS:
        return set(), "none"
    explicit_ids = {str(item) for item in list(correction.input_region_ids or []) if item}
    if explicit_ids:
        return explicit_ids, "input_region_ids"
    entity_id = str(correction.entity_id or "")
    if entity_id:
        entity = next(
            (item for item in effective_entities if str(item.get("entity_id") or "") == entity_id),
            None,
        )
        if entity is None:
            raise ReviewedRegionCompilationError(
                f"{correction.case_id}: correction references unknown entity {entity_id}"
            )
        return {
            reviewed_region_identifier(record)
            for record in records
            if correction_matches_frame(record, correction, source_frame_offset=source_frame_offset)
            and review_working_state.region_belongs_to_entity(record, entity)
        }, "entity_id"
    if correction.operation == OP_REPLACE:
        return {
            reviewed_region_identifier(record)
            for record in records
            if correction_matches_frame(record, correction, source_frame_offset=source_frame_offset)
            and record.category == correction.category
        }, "category_interval"
    if correction.operation == OP_NOT_VISIBLE:
        return {
            reviewed_region_identifier(record)
            for record in records
            if correction_matches_frame(record, correction, source_frame_offset=source_frame_offset)
            and record.category == correction.category
        }, "not_visible_category_interval"
    raise ReviewedRegionCompilationError(
        f"{correction.case_id}: suppression/replacement corrections require a target"
    )


def correction_matches_frame(
    record: PrivacyRegionRecord,
    correction: ReviewCorrection,
    *,
    source_frame_offset: int,
) -> bool:
    source_frame = record.source_frame + source_frame_offset
    return correction.source_frame_start <= source_frame <= correction.source_frame_end


def correction_region_record(
    correction: ReviewCorrection,
    *,
    bbox_xyxy: tuple[int, int, int, int],
    render_frame: int,
    source_frame: int,
    fps: float,
    replaced_records: list[PrivacyRegionRecord],
) -> PrivacyRegionRecord:
    object_id = correction.entity_id or correction.correction_id
    source_record = {
        "baseline_replaced_region_hashes": [
            stable_hash(record.to_dict()) for record in sorted_regions(replaced_records)
        ],
        "baseline_replaced_region_ids": [
            reviewed_region_identifier(record) for record in sorted_regions(replaced_records)
        ],
        "category": correction.category,
        "correction_hash": correction.correction_hash(),
        "correction_id": correction.correction_id,
        "correction_revision_id": correction.revision_id,
        "correction_semantic_sha256": correction_semantic_sha256(correction),
        "input_region_ids": list(correction.input_region_ids),
        "interpolation_method": correction.interpolation_method,
        "object_id": object_id,
        "operation": correction.operation,
        "reviewer_identity": correction.reviewer_identity,
        "reviewed_entity_id": correction.entity_id,
        "reviewed_region_kind": "manual_correction",
        "source_frame": source_frame,
        "source_timestamp_seconds": round(source_frame / fps, 6) if fps > 0 else None,
        "task_id": correction.task_id,
    }
    source_record["reviewed_region_identity_hash"] = stable_hash(
        {
            "bbox_xyxy": bbox_xyxy,
            "category": correction.category,
            "correction_id": correction.correction_id,
            "correction_revision_id": correction.revision_id,
            "source_frame": source_frame,
        }
    )
    source_record["reviewed_region_semantic_hash"] = stable_hash(
        {
            "bbox_xyxy": tuple(int(value) for value in bbox_xyxy),
            "category": correction.category,
            "correction_semantic_sha256": source_record["correction_semantic_sha256"],
            "operation": correction.operation,
            "reviewed_entity_id": correction.entity_id,
            "source_frame": source_frame,
        }
    )
    return PrivacyRegionRecord(
        approval_state=APPROVAL_ACCEPTED,
        bbox_xyxy=tuple(int(value) for value in bbox_xyxy),
        category=correction.category,
        object_id=object_id,
        provenance="reviewed_correction",
        render_eligible=True,
        source_frame=render_frame,
        source_record=source_record,
    )


def annotate_region(
    record: PrivacyRegionRecord,
    *,
    matched_entities: list[dict[str, Any]],
    manifest: case_manifests.CaseManifest,
    provenance_override: str | None = None,
) -> PrivacyRegionRecord:
    from redactor import review_working_state

    source_record = dict(record.source_record)
    entity = sorted(
        matched_entities,
        key=lambda item: (str(item.get("category") or ""), str(item.get("entity_id") or "")),
    )[0] if matched_entities else {}
    baseline_id = baseline_region_id(record)
    source_record.update(
        {
            "baseline_region_hash": stable_hash(record.to_dict()),
            "baseline_region_id": baseline_id,
            "reviewed_entity_id": entity.get("entity_id"),
            "reviewed_entity_label": entity.get("label") or entity.get("display_name"),
            "reviewed_policy": review_working_state.inherited_policy(entity)
            if entity
            else None,
            "reviewed_region_kind": "automatic_baseline",
            "source_timestamp_seconds": source_timestamp_seconds(record, manifest),
        }
    )
    source_record["reviewed_region_identity_hash"] = stable_hash(
        {
            "baseline_region_id": baseline_id,
            "bbox_xyxy": record.bbox_xyxy,
            "category": record.category,
            "entity_id": source_record.get("reviewed_entity_id"),
            "source_frame": record.source_frame,
        }
    )
    return replace(
        record,
        provenance=provenance_override or record.provenance,
        source_record=source_record,
    )


def publication_blockers(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    state: dict[str, Any] | None = None,
    effective_entities: list[dict[str, Any]] | None = None,
    baseline_regions: list[PrivacyRegionRecord] | None = None,
) -> list[str]:
    from redactor import review_working_state

    state = state or review_working_state.derive_working_state(package_root, session_dir)
    decisions_by_case = review_working_state.task_decisions_by_case_from_payload(state)
    resolutions_by_case = review_working_state.task_resolutions_by_case_from_payload(state)
    blockers: list[str] = []
    for task in load_review_tasks(package_root):
        if str(task.get("case_id") or "") != case_id or task.get("technical_only"):
            continue
        task_id = str(task.get("review_task_id") or "")
        if not task_id or review_working_state.durable_edit_verification_task_id(task_id):
            continue
        decision = review_working_state.effective_task_state_for_case(
            decisions_by_case,
            resolutions_by_case,
            case_id=case_id,
            task_id=task_id,
        )
        if decision is None:
            continue
        invalid_decision = invalid_task_decision_reason(task, decision)
        if invalid_decision is not None:
            blockers.append(f"{task_id}: invalid task decision: {invalid_decision}")
            continue
        action = str(decision.get("action") or "")
        if action in BLOCKING_ACTIONS:
            blockers.append(f"{task_id}: decision requires correction or escalation")
        if task.get("correction_required") and action not in GREEN_ACTIONS:
            blockers.append(f"{task_id}: correction has not been verified")

    for task in list(state.get("active_manual_tasks") or []):
        task = dict(task)
        if str(task.get("case_id") or "") != case_id:
            continue
        task_id = str(task.get("review_task_id") or "")
        if not task_id or review_working_state.durable_edit_verification_task_id(task_id):
            continue
        decision = review_working_state.effective_task_state_for_case(
            decisions_by_case,
            resolutions_by_case,
            case_id=case_id,
            task_id=task_id,
        )
        if decision is None:
            blockers.append(f"{task_id}: reviewer-added issue has no decision")
            continue
        invalid_decision = invalid_task_decision_reason(task, decision)
        if invalid_decision is not None:
            blockers.append(f"{task_id}: invalid task decision: {invalid_decision}")
            continue
        if str(decision.get("action") or "") not in GREEN_ACTIONS:
            blockers.append(f"{task_id}: reviewer-added issue remains unresolved")

    for entity in effective_entities or []:
        if str(entity.get("case_id") or "") != case_id:
            continue
        policy = review_working_state.inherited_policy(entity)
        if policy != review_working_state.POLICY_REVIEW_REQUIRED:
            continue
        if baseline_regions and not any(
            review_working_state.region_belongs_to_entity(record, entity)
            for record in baseline_regions
        ):
            continue
        label = entity.get("label") or entity.get("entity_id")
        blockers.append(f"{label}: object policy remains Needs review")
    return sorted(set(blockers))


def load_review_tasks(package_root: Path) -> list[dict[str, Any]]:
    path = package_root / "reviewer_tasks.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = list(payload.get("tasks") or payload.get("reviewer_tasks") or [])
    return [dict(task) for task in raw_tasks]


def reviewed_region_identity(
    *,
    package_root: Path,
    session_dir: Path,
    workspace: dict[str, Any],
    manifest: case_manifests.CaseManifest,
    baseline_regions: list[PrivacyRegionRecord],
    reviewed_regions: list[PrivacyRegionRecord],
    state: dict[str, Any],
    effective_entities: list[dict[str, Any]],
    correction_records: list[ReviewCorrection],
) -> dict[str, Any]:
    from redactor import review_working_state

    object_policy_payload = semantic_runtime_elided_payload(
        dict(state.get("object_policy_overrides") or {})
    )
    merge_payload = {
        entity_id: entity
        for entity_id, entity in sorted(object_policy_payload.items())
        if dict(entity).get("current_status") == "merged"
        or dict(entity).get("merge_split_lineage")
    }
    active_correction_audit_hash = stable_hash(
        sorted(correction.correction_hash() for correction in correction_records)
    )
    active_correction_semantic_hash = stable_hash(
        sorted(correction_semantic_sha256(correction) for correction in correction_records)
    )
    active_decisions_hash = stable_hash(
        semantic_decision_payload(
            review_working_state.media_affecting_decisions_by_case(
                review_working_state.task_decisions_by_case_from_payload(state)
            )
        )
    )
    active_task_resolutions_hash = stable_hash({})
    baseline_hash = semantic_region_set_hash(baseline_regions)
    object_policy_hash = stable_hash(object_policy_payload)
    merge_rejection_hash = stable_hash(merge_payload)
    reviewed_records_hash = semantic_region_set_hash(reviewed_regions)
    renderer_hash = stable_hash(
        render_configuration_payload(review_working_state.render_settings())
    )
    workspace_manifest_path = package_root / "workspace_manifest.json"
    case_manifest_path = case_ids.case_manifest_path(package_root, manifest.case_id)
    identity = {
        "active_correction_audit_hash": active_correction_audit_hash,
        "active_correction_semantic_sha256": active_correction_semantic_hash,
        "active_corrections_hash": active_correction_audit_hash,
        "active_decisions_hash": active_decisions_hash,
        "active_task_resolutions_hash": active_task_resolutions_hash,
        "active_projection_hash_schema_version": state.get(
            "active_projection_hash_schema_version"
        ),
        "automatic_baseline_region_count": len(baseline_regions),
        "automatic_baseline_region_set_sha256": baseline_hash,
        "case_manifest_input_sha256": case_manifest_review_input_sha256_file(
            case_manifest_path
        ),
        "case_id": manifest.case_id,
        "category_run_manifest_identity_hash": case_scoped_workspace_value(
            workspace,
            "category_run_manifest_identity_hash",
            manifest.case_id,
        ),
        "compiler_schema_version": COMPILER_SCHEMA_VERSION,
        "configuration_freeze_hash": workspace.get("configuration_freeze_hash"),
        "merge_rejection_state_hash": merge_rejection_hash,
        "object_policy_hash": object_policy_hash,
        "renderer_configuration_sha256": renderer_hash,
        "reviewed_entity_count": len(effective_entities),
        "reviewed_region_records_sha256": reviewed_records_hash,
        "reviewed_region_count": len(reviewed_regions),
        "selected_categories": selected_categories(workspace),
        "session_head_hash": state.get("active_projection_hash"),
        "session_id": session_dir.name,
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "source_video_hash": manifest.source_video_hash,
        "workspace_manifest_input_sha256": workspace_review_input_sha256_file(
            workspace_manifest_path
        ),
        "working_state_projection_hash": state.get("active_projection_hash"),
    }
    identity["reviewed_region_set_sha256"] = reviewed_region_set_identity_hash(identity)
    return identity


def reviewed_region_set_identity_hash(identity: dict[str, Any]) -> str:
    return stable_hash(
        {
            key: value
            for key, value in identity.items()
            if key not in REVIEWED_REGION_SET_IDENTITY_RUNTIME_KEYS
        }
    )


def reviewed_region_media_identity_payload(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: dict(identity).get(key)
        for key in REVIEWED_REGION_MEDIA_IDENTITY_KEYS
    }


def reviewed_region_media_identity_hash(identity: dict[str, Any]) -> str:
    return stable_hash(reviewed_region_media_identity_payload(identity))


def reviewed_region_media_identity_matches(
    current_identity: dict[str, Any],
    published_identity: dict[str, Any],
) -> bool:
    return reviewed_region_media_identity_payload(
        current_identity
    ) == reviewed_region_media_identity_payload(published_identity)


def latest_pointer_media_matches_current(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    validated_pointer: dict[str, Any],
    working_state: dict[str, Any] | None,
    require_blockers_resolved: bool = False,
) -> bool:
    if working_state is None:
        return False
    revision_manifest = dict(validated_pointer.get("revision_manifest_payload") or {})
    published_identity = dict(revision_manifest.get("reviewed_region_identity") or {})
    if not published_identity:
        return False
    current_digest = reviewed_region_digest(
        package_root,
        session_dir,
        case_id=case_id,
        require_blockers_resolved=require_blockers_resolved,
        working_state=working_state,
    )
    return reviewed_region_media_identity_matches(
        dict(current_digest.get("reviewed_region_identity") or {}),
        published_identity,
    )


def semantic_decision_payload(decisions: dict[str, Any]) -> dict[str, Any]:
    by_case = (
        decisions
        if all(isinstance(value, dict) for value in decisions.values())
        and any(
            isinstance(value, dict)
            and all(isinstance(item, dict) for item in dict(value).values())
            for value in decisions.values()
        )
        else {}
    )
    if by_case:
        return {
            str(case_id): {
                str(task_id): {
                    key: dict(decision).get(key)
                    for key in sorted(SEMANTIC_DECISION_KEYS)
                    if key in dict(decision)
                }
                for task_id, decision in sorted(dict(case_decisions).items())
            }
            for case_id, case_decisions in sorted(by_case.items())
        }
    return {
        str(task_id): {
            key: dict(decision).get(key)
            for key in sorted(SEMANTIC_DECISION_KEYS)
            if key in dict(decision)
        }
        for task_id, decision in sorted(decisions.items())
    }


def correction_semantic_payload(correction: ReviewCorrection) -> dict[str, Any]:
    return {
        "case_id": correction.case_id,
        "category": correction.category,
        "configuration_freeze_hash": correction.configuration_freeze_hash,
        "entity_id": correction.entity_id,
        "input_region_ids": sorted(str(item) for item in correction.input_region_ids),
        "interpolation_method": correction.interpolation_method,
        "keyframe_boxes": [
            {
                "bbox_xyxy": [int(value) for value in keyframe.bbox_xyxy],
                "source_frame": int(keyframe.source_frame),
            }
            for keyframe in sorted(
                correction.keyframes,
                key=lambda item: (int(item.source_frame), tuple(int(v) for v in item.bbox_xyxy)),
            )
        ],
        "operation": correction.operation,
        "schema_version": "review_correction_semantic_projection.v1",
        "source_frame_end": int(correction.source_frame_end),
        "source_frame_start": int(correction.source_frame_start),
        "source_video_hash": correction.source_video_hash,
        "task_id": correction.task_id,
    }


def correction_semantic_sha256(correction: ReviewCorrection) -> str:
    return stable_hash(correction_semantic_payload(correction))


def semantic_runtime_elided_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: semantic_runtime_elided_payload(item)
            for key, item in sorted(value.items())
            if key not in RUNTIME_REVIEW_STATE_KEYS
        }
    if isinstance(value, list):
        return [semantic_runtime_elided_payload(item) for item in value]
    return value


def workspace_review_input_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the workspace manifest fields that define reviewed compilation inputs."""
    cleaned = dict(payload)
    cleaned.pop("latest_reviewed_current", None)
    cleaned.pop("latest_reviewed_superseded", None)
    publication = dict(cleaned.get("latest_reviewed_publication") or {})
    if publication:
        cleaned["latest_reviewed_publication"] = {
            key: publication.get(key)
            for key in sorted(WORKSPACE_PUBLICATION_IDENTITY_KEYS)
            if key in publication
        }
        if cleaned["latest_reviewed_publication"].get("status") == "ready":
            cleaned["latest_reviewed_publication"]["status"] = "supported"
    lifecycle = dict(cleaned.get("lifecycle") or {})
    lifecycle.pop("latest_reviewed", None)
    if lifecycle:
        cleaned["lifecycle"] = lifecycle
    else:
        cleaned.pop("lifecycle", None)
    identities = dict(cleaned.get("content_identities") or {})
    cleaned["content_identities"] = {
        key: value
        for key, value in sorted(identities.items())
        if key != "case_manifest" and not key.startswith(LATEST_REVIEWED_CONTENT_IDENTITY_PREFIX)
    }
    return cleaned


def workspace_review_input_sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return stable_hash(workspace_review_input_payload(json.loads(path.read_text(encoding="utf-8"))))


def case_manifest_review_input_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    cleaned["expected_output_roles"] = [
        role
        for role in list(cleaned.get("expected_output_roles") or [])
        if role != "latest_reviewed"
    ]
    cleaned["video_roles"] = [
        role
        for role in list(cleaned.get("video_roles") or [])
        if dict(role).get("role") != "latest_reviewed"
    ]
    lineage = dict(cleaned.get("lineage") or {})
    lineage.pop("latest_reviewed_publication", None)
    if lineage:
        cleaned["lineage"] = lineage
    else:
        cleaned.pop("lineage", None)
    return cleaned


def case_manifest_review_input_sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return stable_hash(
        case_manifest_review_input_payload(json.loads(path.read_text(encoding="utf-8")))
    )


def publication_guard_payload(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    accepted_revision: str | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from redactor import review_sessions

    identity = dict(provenance.get("reviewed_region_identity") or {})
    workspace_manifest_path = package_root / "workspace_manifest.json"
    case_manifest_path = case_ids.case_manifest_path(package_root, case_id)
    return {
        "accepted_working_state_revision": accepted_revision,
        "active_corrections_hash": identity.get("active_corrections_hash"),
        "active_decisions_hash": identity.get("active_decisions_hash"),
        "active_projection_hash_schema_version": identity.get(
            "active_projection_hash_schema_version"
        ),
        "automatic_baseline_region_set_sha256": identity.get(
            "automatic_baseline_region_set_sha256"
        ),
        "case_id": case_id,
        "case_manifest_input_sha256": case_manifest_review_input_sha256_file(
            case_manifest_path
        ),
        "case_manifest_sha256": media.sha256_file(case_manifest_path),
        "merge_rejection_state_hash": identity.get("merge_rejection_state_hash"),
        "object_policy_hash": identity.get("object_policy_hash"),
        "renderer_configuration_sha256": identity.get("renderer_configuration_sha256"),
        "reviewed_region_media_identity_sha256": reviewed_region_media_identity_hash(
            identity
        ),
        "reviewed_region_set_sha256": provenance.get("reviewed_region_set_sha256"),
        "schema_version": "latest_reviewed_publication_guard.v1",
        "session_head_hash": review_sessions.last_event_hash(session_dir),
        "session_id": session_dir.name,
        "working_state_projection_hash": identity.get("working_state_projection_hash"),
        "workspace_manifest_input_sha256": workspace_review_input_sha256_file(
            workspace_manifest_path
        ),
        "workspace_manifest_sha256": media.sha256_file(workspace_manifest_path),
    }


def assert_publication_guard_current(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    expected_guard: dict[str, Any],
) -> None:
    _regions, provenance = compile_reviewed_regions(
        package_root,
        session_dir,
        case_id=case_id,
        require_blockers_resolved=True,
    )
    current = publication_guard_payload(
        package_root=package_root,
        session_dir=session_dir,
        case_id=case_id,
        accepted_revision=expected_guard.get("accepted_working_state_revision"),
        provenance=provenance,
    )
    compared_keys = (
        "active_projection_hash_schema_version",
        "case_id",
        "case_manifest_input_sha256",
        "reviewed_region_media_identity_sha256",
        "reviewed_region_set_sha256",
        "session_id",
        "workspace_manifest_input_sha256",
    )
    mismatches = {
        key: {"current": current.get(key), "expected": expected_guard.get(key)}
        for key in compared_keys
        if current.get(key) != expected_guard.get(key)
    }
    if mismatches:
        raise ReviewedRegionPublicationStaleError(
            f"{case_id}: Latest reviewed publication guard is stale: {mismatches}"
        )


def semantic_region_set_hash(records: list[PrivacyRegionRecord]) -> str:
    payloads = [semantic_region_payload(record) for record in records]
    return stable_hash(sorted(payloads, key=stable_hash))


def semantic_region_payload(record: PrivacyRegionRecord) -> dict[str, Any]:
    source = dict(record.source_record)
    return {
        "approval_state": record.approval_state,
        "baseline_region_id": source.get("baseline_region_id"),
        "bbox_xyxy": list(record.bbox_xyxy) if record.bbox_xyxy is not None else None,
        "category": record.category,
        "correction_semantic_sha256": source.get("correction_semantic_sha256"),
        "object_id": record.object_id,
        "operation": source.get("operation"),
        "provenance": record.provenance,
        "render_eligible": record.render_eligible,
        "reviewed_entity_id": source.get("reviewed_entity_id"),
        "reviewed_policy": source.get("reviewed_policy"),
        "reviewed_region_semantic_hash": source.get("reviewed_region_semantic_hash"),
        "source_frame": record.source_frame,
        "source_timestamp_seconds": source.get("source_timestamp_seconds"),
    }


def reviewed_region_unique_key(record: PrivacyRegionRecord) -> str:
    source = dict(record.source_record)
    return stable_hash(
        {
            "bbox_xyxy": record.bbox_xyxy,
            "category": record.category,
            "object_id": record.object_id,
            "reviewed_entity_id": source.get("reviewed_entity_id"),
            "source_frame": record.source_frame,
        }
    )


def duplicated_region_keys(records: list[PrivacyRegionRecord]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in sorted_regions(records):
        key = reviewed_region_unique_key(record)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def reviewed_region_identifier(record: PrivacyRegionRecord) -> str:
    source = dict(record.source_record)
    for key in (
        "baseline_region_id",
        "region_id",
        "track_uid",
        "correction_id",
        "reviewed_region_identity_hash",
    ):
        if source.get(key):
            return str(source[key])
    return stable_hash(record.to_dict())


def baseline_region_id(record: PrivacyRegionRecord) -> str:
    source = dict(record.source_record)
    for key in ("region_id", "track_uid", "track_id", "candidate_id"):
        if source.get(key):
            return str(source[key])
    return stable_hash(record.to_dict())


def sorted_regions(records: list[PrivacyRegionRecord]) -> list[PrivacyRegionRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.source_frame,
            record.category,
            record.object_id or "",
            record.provenance,
            record.bbox_xyxy or (-1, -1, -1, -1),
            stable_hash(dict(record.source_record)),
        ),
    )


def source_timestamp_seconds(
    record: PrivacyRegionRecord,
    manifest: case_manifests.CaseManifest,
) -> float | None:
    fps = rational_to_float(manifest.fps) or 0.0
    if fps <= 0:
        return None
    return round((manifest.source_frame_start + record.source_frame) / fps, 6)


def render_configuration_payload(settings: Any) -> dict[str, Any]:
    return {
        "blur_kernel": settings.normalized_blur_kernel,
        "implementation_id": LATEST_REVIEWED_MATERIALIZATION_IMPLEMENTATION_ID,
        "margin_fraction": settings.margin_fraction,
        "margin_pixels": settings.margin_pixels,
        "method": settings.method,
        "mode": settings.mode,
        "pixel_size": settings.pixel_size,
        "renderer_schema_version": privacy_rendering.SCHEMA_VERSION,
    }


def publish_current_revision(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    role: dict[str, Any],
    revision_manifest: dict[str, Any],
    paths: dict[str, Path],
    publication_guard: dict[str, Any],
    accepted_revision: str | None,
    publication_currentness_assertion: Callable[[], None] | None = None,
) -> dict[str, Any]:
    with session_write_lock():
        maybe_inject_failure("before_current_pointer_update")
        assert_publication_guard_current(
            package_root=package_root,
            session_dir=session_dir,
            case_id=case_id,
            expected_guard=publication_guard,
        )
        if publication_currentness_assertion is not None:
            publication_currentness_assertion()
        pointer_update = write_current_pointer(
            package_root=package_root,
            session_dir=session_dir,
            case_id=case_id,
            role=role,
            revision_manifest=revision_manifest,
            paths=paths,
            publication_guard=publication_guard,
            accepted_revision=accepted_revision,
        )
    return {
        "current_pointer": pointer_update,
        "schema_version": "latest_reviewed_publication_updates.v2",
    }


def current_pointer_path(package_root: Path) -> Path:
    return package_root.resolve() / CURRENT_POINTER_RELATIVE_PATH


def write_current_pointer(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    role: dict[str, Any],
    revision_manifest: dict[str, Any],
    paths: dict[str, Path],
    publication_guard: dict[str, Any],
    accepted_revision: str | None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    pointer_path = current_pointer_path(package_root)
    publication_guard = dict(publication_guard)
    revision_manifest_path = paths["revision_manifest"]
    revision_manifest_artifact_sha256 = media.sha256_file(revision_manifest_path)
    content_identities = {
        key: artifact_payload(package_root, path, role=key)
        for key, path in {
            "latest_reviewed_media_validation": paths["media_validation"],
            "latest_reviewed_region_validation": paths["region_validation"],
            "latest_reviewed_revision_manifest": revision_manifest_path,
            "latest_reviewed_regions": paths["reviewed_regions"],
            "latest_reviewed_video": paths["output_video"],
        }.items()
    }
    pointer = {
        "accepted_working_state_revision": accepted_revision,
        "active_projection_hash_schema_version": publication_guard.get(
            "active_projection_hash_schema_version"
        ),
        "case_id": case_id,
        "content_identities": content_identities,
        "current": True,
        "current_revision_id": role["revision_id"],
        "latest_reviewed": role,
        "publication_guard": publication_guard,
        "publication_status": "ready",
        "published_at_utc": media.utc_timestamp(),
        "renderer_configuration_sha256": revision_manifest.get(
            "renderer_configuration_sha256"
        ),
        "revision_id": role["revision_id"],
        "revision_manifest": role["revision_manifest"],
        "revision_manifest_artifact_sha256": revision_manifest_artifact_sha256,
        "revision_manifest_sha256": revision_manifest.get("revision_manifest_sha256"),
        "review_session_id": session_dir.name,
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "schema_version": CURRENT_POINTER_SCHEMA_VERSION,
        "session_head_hash": publication_guard.get("session_head_hash"),
        "source_sha256": revision_manifest["source_sha256"],
        "status": "ready",
        "video_artifact_sha256": role["sha256"],
        "working_state_generation": accepted_revision,
        "working_state_projection_hash": publication_guard.get(
            "working_state_projection_hash"
        ),
        "working_state_projection_schema_version": publication_guard.get(
            "active_projection_hash_schema_version"
        ),
        "workspace_manifest_input_sha256": publication_guard.get(
            "workspace_manifest_input_sha256"
        ),
        "workspace_manifest_sha256": publication_guard.get("workspace_manifest_sha256"),
    }
    pointer_index = current_pointer_index_payload(package_root)
    pointer_index["cases"][case_id] = pointer
    pointer_index["updated_at_utc"] = media.utc_timestamp()
    maybe_inject_failure("during_current_pointer_update")
    atomic_write_json(pointer_path, pointer_index)
    reopened = load_current_pointer(package_root, case_id=case_id)
    validated = validate_current_pointer(package_root, reopened, case_id=case_id)
    pointer_sha256 = media.sha256_file(pointer_path)
    return {
        "path": pointer_path.relative_to(package_root).as_posix(),
        "current_pointer_sha256": pointer_sha256,
        "revision_id": validated["revision_id"],
        "reviewed_region_set_sha256": validated["reviewed_region_set_sha256"],
        "schema_version": "latest_reviewed_current_pointer_update.v1",
        "status": "updated",
    }


def load_current_pointer(package_root: Path, *, case_id: str | None = None) -> dict[str, Any]:
    path = current_pointer_path(package_root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CURRENT_POINTER_INDEX_SCHEMA_VERSION:
        if case_id is None or str(payload.get("case_id") or "") == (
            case_ids.validate_case_id(case_id)
        ):
            return payload
        return {}
    cases = dict(payload.get("cases") or {})
    if case_id is not None:
        return dict(cases.get(case_ids.validate_case_id(case_id)) or {})
    current_cases = [
        dict(pointer)
        for pointer in cases.values()
        if isinstance(pointer, dict) and pointer.get("current") is True
    ]
    if len(current_cases) == 1:
        return current_cases[0]
    return payload


def current_pointer_index_payload(package_root: Path) -> dict[str, Any]:
    path = current_pointer_path(package_root)
    if not path.exists():
        return {
            "cases": {},
            "schema_version": CURRENT_POINTER_INDEX_SCHEMA_VERSION,
            "updated_at_utc": media.utc_timestamp(),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == CURRENT_POINTER_INDEX_SCHEMA_VERSION:
        return {
            "cases": dict(payload.get("cases") or {}),
            "schema_version": CURRENT_POINTER_INDEX_SCHEMA_VERSION,
            "updated_at_utc": payload.get("updated_at_utc") or media.utc_timestamp(),
        }
    case_id = case_ids.validate_case_id(str(payload.get("case_id") or ""))
    return {
        "cases": {case_id: payload},
        "schema_version": CURRENT_POINTER_INDEX_SCHEMA_VERSION,
        "updated_at_utc": media.utc_timestamp(),
    }


def current_latest_reviewed_role(
    package_root: Path,
    session_dir: Path | None,
    *,
    case_id: str,
    active_projection_hashes: dict[str, str],
    working_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from redactor import review_working_state

    pointer = load_current_pointer(package_root, case_id=case_id)
    if not pointer:
        return {}
    try:
        validated = validate_current_pointer(package_root, pointer, case_id=case_id)
    except ReviewedRegionCompilationError:
        return {}
    if validated.get("case_id") != case_ids.validate_case_id(case_id):
        return {}
    if session_dir is not None and validated.get("review_session_id") != session_dir.name:
        return {}
    expected_revision = str(
        validated.get("working_state_projection_hash")
        or validated.get("accepted_working_state_revision")
        or ""
    )
    try:
        current_revision_candidates = (
            review_working_state.active_projection_hash_candidates_for_pointer(
                active_projection_hashes,
                validated,
            )
        )
    except ValueError:
        return {}
    if expected_revision and expected_revision not in current_revision_candidates:
        if session_dir is None:
            return {}
    if session_dir is not None:
        try:
            media_current = latest_pointer_media_matches_current(
                package_root=package_root,
                session_dir=session_dir,
                case_id=case_id,
                validated_pointer=validated,
                working_state=working_state,
                require_blockers_resolved=False,
            )
        except ReviewedRegionCompilationError:
            media_current = False
        if not media_current:
            return {}
    latest = dict(validated.get("latest_reviewed") or {})
    if latest:
        latest["current_pointer"] = CURRENT_POINTER_RELATIVE_PATH.as_posix()
        latest["current_pointer_sha256"] = media.sha256_file(current_pointer_path(package_root))
    return latest


def validate_current_pointer(
    package_root: Path,
    pointer: dict[str, Any],
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    from redactor import review_working_state

    if pointer.get("schema_version") == CURRENT_POINTER_INDEX_SCHEMA_VERSION:
        cases = dict(pointer.get("cases") or {})
        if case_id is not None:
            selected = dict(cases.get(case_ids.validate_case_id(case_id)) or {})
            if not selected:
                raise ReviewedRegionCompilationError(
                    "Latest reviewed current pointer is missing for case"
                )
            return validate_current_pointer(package_root, selected, case_id=case_id)
        current_cases = [
            dict(case_pointer)
            for case_pointer in cases.values()
            if isinstance(case_pointer, dict) and case_pointer.get("current") is True
        ]
        if len(current_cases) != 1:
            raise ReviewedRegionCompilationError(
                "Latest reviewed current pointer index requires a case id"
            )
        return validate_current_pointer(package_root, current_cases[0])
    if pointer.get("schema_version") != CURRENT_POINTER_SCHEMA_VERSION:
        raise ReviewedRegionCompilationError("Latest reviewed current pointer schema mismatch")
    if pointer.get("status") != "ready" or pointer.get("current") is not True:
        raise ReviewedRegionCompilationError("Latest reviewed current pointer is not ready")
    try:
        pointer_projection_schema = review_working_state.active_projection_pointer_schema(pointer)
    except ValueError as exc:
        raise ReviewedRegionCompilationError(str(exc)) from exc
    pointer_case_id = case_ids.validate_case_id(str(pointer.get("case_id") or ""))
    if case_id is not None and pointer_case_id != case_ids.validate_case_id(case_id):
        raise ReviewedRegionCompilationError("Latest reviewed pointer case mismatch")
    revision_manifest_path = case_ids.contained_path(
        package_root,
        *Path(str(pointer.get("revision_manifest") or "")).parts,
    )
    if not revision_manifest_path.exists():
        raise ReviewedRegionCompilationError("Latest reviewed revision manifest is missing")
    if pointer.get("revision_manifest_artifact_sha256") != media.sha256_file(
        revision_manifest_path
    ):
        raise ReviewedRegionCompilationError("Latest reviewed revision manifest hash mismatch")
    revision_manifest = json.loads(revision_manifest_path.read_text(encoding="utf-8"))
    if revision_manifest.get("schema_version") != REVISION_SCHEMA_VERSION:
        raise ReviewedRegionCompilationError("Latest reviewed revision manifest schema mismatch")
    revision_projection_schema = (
        revision_manifest.get("active_projection_hash_schema_version")
        or (revision_manifest.get("publication_guard") or {}).get(
            "active_projection_hash_schema_version"
        )
        or pointer_projection_schema
    )
    if revision_projection_schema != pointer_projection_schema:
        raise ReviewedRegionCompilationError(
            "Latest reviewed projection hash schema mismatch"
        )
    for key in (
        "revision_id",
        "reviewed_region_set_sha256",
        "source_sha256",
        "video_artifact_sha256",
    ):
        pointer_value = pointer.get(key if key != "video_artifact_sha256" else key)
        revision_value = revision_manifest.get(key)
        if str(pointer_value or "") != str(revision_value or ""):
            raise ReviewedRegionCompilationError(f"Latest reviewed pointer {key} mismatch")
    if revision_manifest.get("case_id") != pointer_case_id:
        raise ReviewedRegionCompilationError("Latest reviewed pointer case mismatch")
    validate_revision_artifacts(revision_manifest_path.parent, revision_manifest)
    workspace_manifest_path = package_root / "workspace_manifest.json"
    if pointer.get("workspace_manifest_sha256") != media.sha256_file(workspace_manifest_path):
        raise ReviewedRegionCompilationError("Latest reviewed workspace manifest hash mismatch")
    latest = dict(pointer.get("latest_reviewed") or {})
    if latest.get("role") != "latest_reviewed":
        raise ReviewedRegionCompilationError("Latest reviewed pointer role mismatch")
    latest_path = case_ids.contained_path(
        package_root,
        *Path(str(latest.get("path") or "")).parts,
    )
    if not latest_path.exists():
        raise ReviewedRegionCompilationError("Latest reviewed media is missing")
    if latest.get("sha256") != media.sha256_file(latest_path):
        raise ReviewedRegionCompilationError("Latest reviewed media hash mismatch")
    return dict(pointer) | {
        "case_id": pointer_case_id,
        "revision_manifest_payload": revision_manifest,
    }


def validate_reviewed_regions_artifact_binding(
    package_root: Path,
    reviewed_regions_path: Path,
    *,
    latest_pointer: dict[str, Any],
    final_pointer: dict[str, Any] | None = None,
    expected_source_sha256: str | None = None,
    expected_case_id: str | None = None,
    expected_category: str | None = None,
) -> dict[str, Any]:
    from redactor import review_finalization

    package_root = package_root.resolve()
    supplied_path = reviewed_regions_path.resolve()
    if not supplied_path.exists():
        raise ReviewedRegionCompilationError("reviewed regions artifact is missing")
    try:
        supplied_path.relative_to(package_root)
    except ValueError as exc:
        raise ReviewedRegionCompilationError(
            "reviewed regions artifact is outside package root"
        ) from exc

    revision_manifest = dict(latest_pointer.get("revision_manifest_payload") or {})
    if not revision_manifest:
        raise ReviewedRegionCompilationError(
            "Latest reviewed revision manifest payload is missing"
        )
    revision_manifest_path = case_ids.contained_path(
        package_root,
        *Path(str(latest_pointer.get("revision_manifest") or "")).parts,
    )
    revision_root = revision_manifest_path.parent
    expected_artifact = str(revision_manifest.get("reviewed_regions") or "")
    if not expected_artifact:
        raise ReviewedRegionCompilationError(
            "Latest reviewed revision missing reviewed_regions artifact"
        )
    authoritative_path = (revision_root / expected_artifact).resolve()
    try:
        authoritative_path.relative_to(package_root)
    except ValueError as exc:
        raise ReviewedRegionCompilationError(
            "Latest reviewed reviewed-regions artifact is outside package root"
        ) from exc
    if supplied_path != authoritative_path:
        raise ReviewedRegionCompilationError(
            "reviewed regions artifact path does not match Latest reviewed revision"
        )

    validate_revision_artifacts(revision_root, revision_manifest)
    raw_sha = media.sha256_file(supplied_path)
    expected_raw_sha = str(revision_manifest.get("reviewed_regions_sha256") or "")
    if not expected_raw_sha:
        raise ReviewedRegionCompilationError("reviewed regions artifact SHA is missing")
    if raw_sha != expected_raw_sha:
        raise ReviewedRegionCompilationError("reviewed regions artifact SHA mismatch")

    try:
        records = review_finalization.load_privacy_region_records(supplied_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ReviewedRegionCompilationError(
            f"invalid reviewed regions artifact: {exc}"
        ) from exc
    semantic_records_sha = semantic_region_set_hash(records)
    identity = dict(revision_manifest.get("reviewed_region_identity") or {})
    expected_semantic_records_sha = str(
        identity.get("reviewed_region_records_sha256") or ""
    )
    if not expected_semantic_records_sha:
        raise ReviewedRegionCompilationError(
            "reviewed region semantic records SHA is missing"
        )
    if semantic_records_sha != expected_semantic_records_sha:
        raise ReviewedRegionCompilationError(
            "reviewed region semantic records SHA mismatch"
        )

    reviewed_region_set_sha = str(revision_manifest.get("reviewed_region_set_sha256") or "")
    if not reviewed_region_set_sha:
        raise ReviewedRegionCompilationError("reviewed_region_set_sha256 is missing")
    if str(identity.get("reviewed_region_set_sha256") or "") != reviewed_region_set_sha:
        raise ReviewedRegionCompilationError(
            "reviewed_region_set_sha256 identity mismatch"
        )
    if reviewed_region_set_identity_hash(identity) != reviewed_region_set_sha:
        raise ReviewedRegionCompilationError(
            "reviewed_region_set_sha256 canonical identity mismatch"
        )
    if str(latest_pointer.get("reviewed_region_set_sha256") or "") != reviewed_region_set_sha:
        raise ReviewedRegionCompilationError(
            "Latest reviewed pointer reviewed_region_set_sha256 mismatch"
        )
    if str(latest_pointer.get("revision_id") or "") != str(
        revision_manifest.get("revision_id") or ""
    ):
        raise ReviewedRegionCompilationError("Latest reviewed revision id mismatch")

    if expected_source_sha256:
        for label, payload in (
            ("Latest reviewed pointer", latest_pointer),
            ("Latest reviewed revision", revision_manifest),
            ("reviewed region identity", identity),
        ):
            actual = payload.get("source_sha256") or payload.get("source_video_hash")
            if str(actual or "") != expected_source_sha256:
                raise ReviewedRegionCompilationError(f"{label} source SHA mismatch")
        for record in records:
            source = dict(record.source_record)
            record_source = source.get("source_sha256") or source.get("source_video_hash")
            if record_source and str(record_source) != expected_source_sha256:
                raise ReviewedRegionCompilationError("reviewed region record source SHA mismatch")

    if expected_case_id:
        for label, payload in (
            ("Latest reviewed pointer", latest_pointer),
            ("Latest reviewed revision", revision_manifest),
            ("reviewed region identity", identity),
        ):
            if str(payload.get("case_id") or "") != expected_case_id:
                raise ReviewedRegionCompilationError(f"{label} case_id mismatch")
        for record in records:
            record_case_id = dict(record.source_record).get("case_id")
            if record_case_id and str(record_case_id) != expected_case_id:
                raise ReviewedRegionCompilationError("reviewed region record case_id mismatch")

    if expected_category:
        selected = {str(item) for item in list(identity.get("selected_categories") or [])}
        if selected and expected_category not in selected:
            raise ReviewedRegionCompilationError(
                "reviewed region category is outside selected categories"
            )
        if not any(record.category == expected_category for record in records):
            raise ReviewedRegionCompilationError(
                "reviewed regions do not include expected category"
            )

    final_manifest = dict((final_pointer or {}).get("finalization_manifest_payload") or {})
    if final_pointer is not None:
        for label, payload in (
            ("Final pointer", final_pointer),
            ("Finalization manifest", final_manifest),
        ):
            if str(payload.get("reviewed_region_set_sha256") or "") != reviewed_region_set_sha:
                raise ReviewedRegionCompilationError(
                    f"{label} reviewed_region_set_sha256 mismatch"
                )
            if expected_source_sha256 and str(payload.get("source_sha256") or "") != (
                expected_source_sha256
            ):
                raise ReviewedRegionCompilationError(f"{label} source SHA mismatch")
        expected_revision = str(revision_manifest.get("revision_id") or "")
        for label, payload in (
            ("Final pointer", final_pointer),
            ("Finalization manifest", final_manifest),
        ):
            actual_revision = (
                payload.get("approved_latest_reviewed_revision_id")
                or payload.get("latest_reviewed_revision_id")
            )
            if str(actual_revision or "") != expected_revision:
                raise ReviewedRegionCompilationError(
                    f"{label} Latest reviewed revision mismatch"
                )

    return {
        "case_id": revision_manifest.get("case_id"),
        "record_count": len(records),
        "reviewed_region_records_sha256": semantic_records_sha,
        "reviewed_region_set_sha256": reviewed_region_set_sha,
        "reviewed_regions_path": supplied_path.as_posix(),
        "reviewed_regions_sha256": raw_sha,
        "revision_id": revision_manifest.get("revision_id"),
        "source_sha256": revision_manifest.get("source_sha256"),
    }


def validate_revision_artifacts(
    revision_root: Path,
    revision_manifest: dict[str, Any],
) -> None:
    for key in REVISION_ARTIFACT_KEYS:
        relative = revision_manifest.get(key)
        if not relative:
            raise ReviewedRegionCompilationError(f"Latest reviewed revision missing {key}")
        path = revision_root / str(relative)
        if not path.exists():
            raise ReviewedRegionCompilationError(f"Latest reviewed artifact missing: {key}")
    checks = {
        "media_validation": bool(revision_manifest.get("media_validation_passed") is True),
        "region_validation": bool(revision_manifest.get("region_validation_passed") is True),
        "reviewed_regions_sha256": media.sha256_file(
            revision_root / str(revision_manifest["reviewed_regions"])
        )
        == revision_manifest.get("reviewed_regions_sha256"),
        "video_artifact_sha256": media.sha256_file(
            revision_root / str(revision_manifest["output_video"])
        )
        == revision_manifest.get("video_artifact_sha256"),
    }
    failures = sorted(key for key, passed in checks.items() if not passed)
    if failures:
        raise ReviewedRegionCompilationError(
            f"Latest reviewed revision artifact validation failed: {failures}"
        )


def maybe_inject_failure(stage: str) -> None:
    if os.environ.get(FAILURE_INJECTION_ENV) == stage:
        raise ReviewedRegionPublicationStaleError(f"injected Latest reviewed failure: {stage}")

def update_workspace_case_manifest(
    *,
    package_root: Path,
    case_id: str,
    role: dict[str, Any],
    revision_manifest: dict[str, Any],
) -> dict[str, Any]:
    case_path = case_ids.case_manifest_path(package_root, case_id)
    manifest = case_manifests.load_case_manifest(case_path)
    latest_role = case_manifests.VideoRole(
        label="Latest reviewed",
        path=str(role["path"]),
        role="latest_reviewed",
        sha256=str(role["sha256"]),
        source_frame_end=manifest.source_frame_end,
        source_frame_start=manifest.source_frame_start,
    )
    roles = tuple(
        [item for item in manifest.video_roles if item.role != "latest_reviewed"]
        + [latest_role]
    )
    expected = tuple(
        dict.fromkeys([*list(manifest.expected_output_roles), "latest_reviewed"])
    )
    lineage = dict(manifest.lineage)
    lineage["latest_reviewed_publication"] = {
        "revision_id": role["revision_id"],
        "revision_manifest": role["revision_manifest"],
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "video_artifact_sha256": role["sha256"],
    }
    updated = replace(
        manifest,
        expected_output_roles=expected,
        lineage=lineage,
        video_roles=roles,
    )
    case_manifests.write_case_manifest(case_path, updated)
    return {
        "case_manifest_hash": updated.manifest_hash(),
        "case_manifest_path": case_path.relative_to(package_root).as_posix(),
        "latest_reviewed_role": latest_role.to_dict(),
        "schema_version": "latest_reviewed_case_manifest_update.v1",
    }


def update_workspace_manifest(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
    role: dict[str, Any],
    paths: dict[str, Path],
    revision_manifest: dict[str, Any],
    case_manifest_path: Path,
) -> dict[str, Any]:
    manifest_path = package_root / "workspace_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_identities = dict(payload.get("content_identities") or {})
    for key, path in {
        "case_manifest": case_manifest_path,
        "latest_reviewed_media_validation": paths["media_validation"],
        "latest_reviewed_region_validation": paths["region_validation"],
        "latest_reviewed_revision_manifest": paths["revision_manifest"],
        "latest_reviewed_regions": paths["reviewed_regions"],
        "latest_reviewed_video": paths["output_video"],
    }.items():
        content_identities[key] = artifact_payload(
            package_root,
            path,
            role=key,
        )
    payload["content_identities"] = content_identities
    payload["latest_reviewed_current"] = {
        "case_id": case_id,
        "current": True,
        "published_at_utc": media.utc_timestamp(),
        "review_session_id": session_dir.name,
        "revision_id": role["revision_id"],
        "revision_manifest": role["revision_manifest"],
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "video_artifact_sha256": role["sha256"],
    }
    payload["latest_reviewed_publication"] = {
        "current_revision_id": role["revision_id"],
        "current_revision_manifest": role["revision_manifest"],
        "published_at_utc": payload["latest_reviewed_current"]["published_at_utc"],
        "publication_mode": COMPILER_SCHEMA_VERSION,
        "review_session_id": session_dir.name,
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "schema_version": "latest_reviewed_publication_capability.v1",
        "status": "ready",
        "validation_artifact_sha256": role["validation_artifact_sha256"],
        "video_artifact_sha256": role["sha256"],
    }
    lifecycle = dict(payload.get("lifecycle") or {})
    lifecycle["latest_reviewed"] = {
        "current_revision_id": role["revision_id"],
        "published_at_utc": payload["latest_reviewed_current"]["published_at_utc"],
        "revision_manifest": role["revision_manifest"],
        "status": "ready",
    }
    lifecycle.setdefault("finalization", {"status": "not_run"})
    lifecycle.setdefault("evaluation", {"status": "not_run"})
    payload["lifecycle"] = lifecycle
    atomic_write_json(manifest_path, payload)
    return {
        "path": manifest_path.relative_to(package_root).as_posix(),
        "schema_version": "latest_reviewed_workspace_manifest_update.v1",
        "workspace_manifest_sha256": media.sha256_file(manifest_path),
    }


def update_top_level_case_manifest(
    *,
    package_root: Path,
    case_id: str,
    role: dict[str, Any],
    paths: dict[str, Path],
    revision_manifest: dict[str, Any],
) -> dict[str, Any]:
    top_manifest = package_root.parent / "case_manifest.json"
    if not top_manifest.exists():
        return {"reason": "top_level_case_manifest_not_present", "status": "skipped"}
    payload = json.loads(top_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != case_manifests.UNIFIED_CASE_MANIFEST_SCHEMA_VERSION:
        return {"reason": "unsupported_top_level_manifest_schema", "status": "skipped"}
    case_root = top_manifest.parent
    artifacts = list(payload.get("artifacts") or [])
    replacements = [
        case_manifests.build_artifact_reference(
            package_root / "workspace_manifest.json",
            case_root=case_root,
            role="review_workspace_manifest",
        ),
        case_manifests.build_artifact_reference(
            package_root / "cases" / f"{case_id}.json",
            case_root=case_root,
            role="mixed_review_case_manifest",
        ),
        case_manifests.build_artifact_reference(
            paths["output_video"],
            case_root=case_root,
            role="latest_reviewed_video",
        ),
        case_manifests.build_artifact_reference(
            paths["revision_manifest"],
            case_root=case_root,
            role="latest_reviewed_revision_manifest",
        ),
        case_manifests.build_artifact_reference(
            paths["reviewed_regions"],
            case_root=case_root,
            role="latest_reviewed_regions",
        ),
    ]
    by_id = {}
    for item in artifacts:
        artifact_id = str(
            item.get("artifact_id")
            or case_manifests.artifact_id_for(role=str(item.get("role") or ""))
        )
        by_id[artifact_id] = item
    for artifact in replacements:
        if artifact is None:
            continue
        by_id[str(artifact.artifact_id)] = artifact.to_dict()
    payload["artifacts"] = [by_id[key] for key in sorted(by_id)]
    payload["latest_reviewed_publication"] = {
        "case_id": case_id,
        "current_revision_id": role["revision_id"],
        "current_revision_manifest": paths["revision_manifest"].relative_to(case_root).as_posix(),
        "published_at_utc": media.utc_timestamp(),
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "ready",
        "video_artifact_sha256": role["sha256"],
    }
    case_manifests.publish_unified_case_manifest(top_manifest, payload)
    return {
        "path": top_manifest.as_posix(),
        "schema_version": "latest_reviewed_top_level_case_manifest_update.v1",
        "status": "updated",
        "top_level_case_manifest_sha256": media.sha256_file(top_manifest),
    }


def clear_current_publication(
    package_root: Path,
    *,
    case_ids_to_clear: list[str],
    reason: str,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    target_case_ids = [case_ids.validate_case_id(case_id) for case_id in case_ids_to_clear]
    pointer_path = current_pointer_path(package_root)
    pointer_index = current_pointer_index_payload(package_root)
    cleared: list[dict[str, Any]] = []
    for case_id in target_case_ids:
        previous = dict(pointer_index["cases"].get(case_id) or {})
        if not previous:
            continue
        cleared.append(
            {
                "case_id": previous.get("case_id") or case_id,
                "previous_revision_id": previous.get("revision_id"),
                "previous_reviewed_region_set_sha256": previous.get(
                    "reviewed_region_set_sha256"
                ),
            }
        )
        pointer_index["cases"][case_id] = {
            **previous,
            "current": False,
            "reason": reason,
            "schema_version": CURRENT_POINTER_SCHEMA_VERSION,
            "status": "not_current",
            "superseded_at_utc": media.utc_timestamp(),
        }
    if cleared:
        pointer_index["updated_at_utc"] = media.utc_timestamp()
        atomic_write_json(pointer_path, pointer_index)
    return {
        "cleared": cleared,
        "reason": reason,
        "schema_version": "latest_reviewed_publication_clear.v1",
        "current_pointer": {
            "path": pointer_path.relative_to(package_root).as_posix(),
            "status": "cleared" if cleared else "unchanged",
        },
    }


def clear_workspace_manifest_current(
    package_root: Path,
    *,
    cleared: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    manifest_path = package_root / "workspace_manifest.json"
    if not manifest_path.exists() or not cleared:
        return {"status": "skipped"}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = dict(payload.get("latest_reviewed_current") or {})
    superseded = list(payload.get("latest_reviewed_superseded") or [])
    if current:
        superseded.append(
            current
            | {
                "reason": reason,
                "superseded_at_utc": media.utc_timestamp(),
                "superseded_by": "reset_to_automatic_baseline",
            }
        )
    payload["latest_reviewed_superseded"] = superseded
    payload.pop("latest_reviewed_current", None)
    publication = dict(payload.get("latest_reviewed_publication") or {})
    payload["latest_reviewed_publication"] = {
        "publication_mode": publication.get("publication_mode") or COMPILER_SCHEMA_VERSION,
        "reason": reason,
        "schema_version": "latest_reviewed_publication_capability.v1",
        "status": "supported",
        "superseded_revision_id": current.get("revision_id"),
    }
    lifecycle = dict(payload.get("lifecycle") or {})
    lifecycle["latest_reviewed"] = {
        "reason": reason,
        "status": "not_run",
    }
    payload["lifecycle"] = lifecycle
    for case in cleared:
        case_id = str(case.get("case_id") or "")
        case_path = package_root / "cases" / f"{case_id}.json"
        if case_path.exists():
            payload.setdefault("content_identities", {})["case_manifest"] = artifact_payload(
                package_root,
                case_path,
                role="case_manifest",
            )
    atomic_write_json(manifest_path, payload)
    return {
        "path": manifest_path.relative_to(package_root).as_posix(),
        "status": "cleared",
        "workspace_manifest_sha256": media.sha256_file(manifest_path),
    }


def refresh_top_level_after_clear(package_root: Path) -> dict[str, Any]:
    top_manifest = package_root.parent / "case_manifest.json"
    if not top_manifest.exists():
        return {"reason": "top_level_case_manifest_not_present", "status": "skipped"}
    payload = json.loads(top_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != case_manifests.UNIFIED_CASE_MANIFEST_SCHEMA_VERSION:
        return {"reason": "unsupported_top_level_manifest_schema", "status": "skipped"}
    case_root = top_manifest.parent
    artifacts = list(payload.get("artifacts") or [])
    by_id = {}
    for item in artifacts:
        artifact_id = str(
            item.get("artifact_id")
            or case_manifests.artifact_id_for(role=str(item.get("role") or ""))
        )
        by_id[artifact_id] = item
    for artifact in (
        case_manifests.build_artifact_reference(
            package_root / "workspace_manifest.json",
            case_root=case_root,
            role="review_workspace_manifest",
        ),
        *[
            case_manifests.build_artifact_reference(
                path,
                case_root=case_root,
                role="mixed_review_case_manifest",
            )
            for path in sorted((package_root / "cases").glob("*.json"))
        ],
    ):
        if artifact is not None:
            by_id[str(artifact.artifact_id)] = artifact.to_dict()
    payload["artifacts"] = [by_id[key] for key in sorted(by_id)]
    if isinstance(payload.get("latest_reviewed_publication"), dict):
        payload["latest_reviewed_publication"] = (
            dict(payload["latest_reviewed_publication"])
            | {
                "reason": "reset_to_automatic_baseline",
                "status": "supported",
                "superseded_at_utc": media.utc_timestamp(),
            }
        )
    case_manifests.publish_unified_case_manifest(top_manifest, payload)
    return {
        "path": top_manifest.as_posix(),
        "status": "refreshed",
        "top_level_case_manifest_sha256": media.sha256_file(top_manifest),
    }

def artifact_payload(package_root: Path, path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(package_root).as_posix(),
        "role": role,
        "schema_version": "artifact_reference.v1",
        "sha256": media.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)
