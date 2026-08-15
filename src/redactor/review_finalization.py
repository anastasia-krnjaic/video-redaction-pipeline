from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from redactor import (
    case_ids,
    case_manifests,
    media,
    privacy_rendering,
    rendering,
    review_guidance_policy,
)
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    MANUAL_REDACTION_CATEGORIES,
    PrivacyRegionRecord,
    normalize_manual_redaction_category,
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
    correction_verification_blockers,
    interpolate_correction_boxes,
    load_corrections_jsonl,
)
from redactor.review_sessions import (
    ACTION_ESCALATE,
    BLOCKING_ACTIONS,
    GREEN_ACTIONS,
    append_event,
    invalid_task_decision_reason,
    last_event_hash,
    load_events,
    validate_event_hash_chain,
)

SCHEMA_VERSION = "review_finalization.v1"
FINAL_REGION_SCHEMA_VERSION = "final_region_provenance.v1"
CANONICAL_FINALIZATION_SCHEMA_VERSION = "canonical_review_finalization.v1"
FINALIZATION_REVISION_SCHEMA_VERSION = "finalization_revision.v1"
FINAL_CURRENT_POINTER_SCHEMA_VERSION = "final_current_pointer.v1"
FULL_VIDEO_SWEEP_ACK_SCHEMA_VERSION = "full_video_sweep_acknowledgement.v3"
FINAL_CURRENT_POINTER_RELATIVE_PATH = Path("final") / "current.json"
FINALIZATION_FAILURE_INJECTION_ENV = "REDACTOR_FINALIZATION_FAIL_AT"


def finalization_blockers(
    *,
    package_root: Path,
    case_id: str,
    session_dir: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    from redactor import review_working_state

    working_state = review_working_state.derive_working_state(package_root, session_dir)
    tasks_payload = load_json(package_root / "reviewer_tasks.json")
    tasks = [
        task
        for task in tasks_payload.get("tasks", [])
        if task.get("case_id") == case_id and not task.get("technical_only")
    ]
    decisions_by_case = review_working_state.task_decisions_by_case_from_payload(working_state)
    resolutions_by_case = review_working_state.task_resolutions_by_case_from_payload(working_state)
    for task in tasks:
        task_id = str(task["review_task_id"])
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
        action = str(decision.get("action"))
        if action in BLOCKING_ACTIONS:
            blockers.append(f"{task_id}: decision requires correction or escalation")
        if action == ACTION_ESCALATE:
            blockers.append(f"{task_id}: escalation remains unresolved")
        if task.get("correction_required") and action not in GREEN_ACTIONS:
            blockers.append(f"{task_id}: correction has not been verified")

    for task in list(working_state.get("active_manual_tasks") or []):
        if dict(task).get("case_id") != case_id:
            continue
        task = dict(task)
        task_id = str(task.get("review_task_id") or "")
        if not task_id:
            blockers.append(f"{case_id}: manual issue is missing a task id")
            continue
        if review_working_state.durable_edit_verification_task_id(task_id):
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
        if str(decision.get("action")) not in GREEN_ACTIONS:
            blockers.append(f"{task_id}: reviewer-added issue remains unresolved")

    if case_id not in dict(working_state.get("sweep_acknowledgements") or {}):
        blockers.append(f"{case_id}: full-video sweep has not been acknowledged")

    blockers.extend(
        object_policy_blockers(
            working_state,
            decisions_by_case,
            resolutions_by_case,
            case_id=case_id,
        )
    )
    if review_working_state.latest_reviewed_publication_deferred(package_root):
        blockers.append(f"{case_id}: latest_reviewed_publication_deferred")
    elif review_working_state.latest_reviewed_publication_supported(package_root):
        latest = dict(working_state.get("materialized_latest_reviewed") or {}).get(case_id)
        if not latest:
            blockers.append(f"{case_id}: Latest reviewed has not been published")
        else:
            latest_path = package_root / str(latest.get("path") or "")
            if not latest_path.exists():
                blockers.append(f"{case_id}: Latest reviewed media is missing")
            if latest.get("media_validation_passed") is not True:
                blockers.append(f"{case_id}: Latest reviewed media validation has not passed")
            try:
                target_digest = review_working_state.effective_privacy_region_digest(
                    package_root,
                    session_dir,
                    case_id=case_id,
                )
            except Exception as exc:
                blockers.append(f"{case_id}: Latest reviewed digest failed: {exc}")
            else:
                published_hash = str(latest.get("effective_regions_hash") or "")
                target_hash = str(target_digest.get("effective_regions_hash") or "")
                if not published_hash or published_hash != target_hash:
                    blockers.append(f"{case_id}: Latest reviewed is not current")

    chain = validate_event_hash_chain(session_dir)
    if not chain["passed"]:
        blockers.extend(f"event log: {error}" for error in chain["errors"])

    case_manifest_path = case_ids.case_manifest_path(package_root, case_id)
    if not case_manifest_path.exists():
        blockers.append(f"{case_id}: missing case manifest")
    else:
        manifest = case_manifests.load_case_manifest(case_manifest_path)
        if manifest.status != case_manifests.CASE_STATUS_READY:
            blockers.extend(f"{case_id}: {reason}" for reason in manifest.blockers)
        source_path = resolve_package_path(package_root, manifest.source_path)
        if source_path is None or not source_path.exists():
            blockers.append(f"{case_id}: source video is missing")
        elif (
            manifest.source_video_hash
            and media.sha256_file(source_path) != manifest.source_video_hash
        ):
            blockers.append(f"{case_id}: source hash mismatch")

    session_payload = load_json(session_dir / "review_session.json")
    freeze_path = package_root / "configuration_freeze.json"
    if freeze_path.exists():
        freeze_hash = stable_hash(load_json(freeze_path))
        if session_payload.get("configuration_freeze_hash") != freeze_hash:
            blockers.append("configuration freeze hash mismatch")
    else:
        blockers.append("missing configuration_freeze.json")

    region_path = case_ids.contained_path(
        package_root,
        "regions",
        f"{case_id}_current_regions.jsonl",
    )
    if not region_path.exists():
        blockers.append(f"{case_id}: current privacy-region provenance is missing")

    active_correction_ids = set(
        str(item) for item in list(working_state.get("active_correction_ids") or [])
    )
    corrections = [
        correction
        for correction in latest_corrections(
            load_corrections_jsonl(session_dir / "corrections.jsonl"),
            case_id=case_id,
        )
        if correction.correction_id in active_correction_ids
    ]
    for correction in corrections:
        blockers.extend(correction_verification_blockers(correction))

    if review_working_state.latest_reviewed_publication_supported(package_root):
        blockers.extend(
            canonical_finalization_gate_blockers(
                package_root=package_root,
                case_id=case_id,
                session_dir=session_dir,
                working_state=working_state,
            )
        )

    return {
        "blocker_count": len(blockers),
        "blockers": blockers,
        "case_id": case_id,
        "ready": not blockers,
        "review_guidance_policy_version": review_guidance_policy.POLICY_VERSION,
        "schema_version": "review_finalization_blockers.v1",
        "unreviewed_guidance": review_guidance_policy.unreviewed_guidance_snapshot(
            package_root=package_root,
            working_state=working_state,
            case_id=case_id,
        ),
    }


def object_policy_blockers(
    working_state: dict[str, Any],
    decisions_by_case: dict[str, dict[str, dict[str, Any]]],
    resolutions_by_case: dict[str, dict[str, dict[str, Any]]],
    *,
    case_id: str,
) -> list[str]:
    blockers: list[str] = []
    for entity_id, raw_entity in sorted(
        dict(working_state.get("object_policy_overrides") or {}).items()
    ):
        after = dict(raw_entity or {})
        if after.get("current_status") == "merged":
            continue
        if str(after.get("case_id") or "") != case_id:
            continue
        if after.get("reviewer_override") is not True:
            continue
        policy = str(
            after.get("current_policy")
            or after.get("default_privacy_policy")
            or after.get("privacy_action")
            or after.get("policy")
            or ""
        )
        label = str(after.get("label") or entity_id)
        if policy == "review_required":
            blockers.append(f"{label}: object policy remains Needs review")
            continue
    return blockers


def finalize_reviewed_case(
    *,
    package_root: Path,
    case_id: str,
    session_dir: Path,
    actor: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    case_id = case_ids.validate_case_id(case_id)
    from redactor import review_working_state

    if review_working_state.latest_reviewed_publication_supported(package_root):
        return finalize_canonical_reviewed_case(
            package_root=package_root,
            case_id=case_id,
            session_dir=session_dir,
            actor=actor,
        )

    blockers = finalization_blockers(
        package_root=package_root,
        case_id=case_id,
        session_dir=session_dir,
    )
    if not blockers["ready"]:
        return {
            "blockers": blockers["blockers"],
            "case_id": case_id,
            "finalized": False,
            "schema_version": SCHEMA_VERSION,
        }

    manifest = case_manifests.load_case_manifest(case_ids.case_manifest_path(package_root, case_id))
    source_path = resolve_package_path(package_root, manifest.source_path)
    if source_path is None:
        raise ValueError("case source path is required for finalization")

    final_regions, provenance = review_working_state.compute_effective_privacy_regions(
        package_root,
        session_dir,
        case_id=case_id,
    )
    working_state = review_working_state.derive_working_state(package_root, session_dir)
    active_correction_ids = set(
        str(item) for item in list(working_state.get("active_correction_ids") or [])
    )
    corrections = [
        correction
        for correction in latest_corrections(
            load_corrections_jsonl(session_dir / "corrections.jsonl"),
            case_id=case_id,
        )
        if correction.correction_id in active_correction_ids
    ]
    regions_by_frame = privacy_rendering.regions_by_frame(final_regions)

    case_media_dir = package_root / "media" / case_id
    case_media_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_media_dir / "final_reviewed.mp4"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    source_info = media.probe_media(source_path)
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
        source_path,
        output_path,
        regions_by_frame,
        settings=settings,
        fps=fps,
        width=source_info.video_stream.width,
        height=source_info.video_stream.height,
        source_has_audio=source_info.audio_stream is not None,
        overwrite=overwrite,
    )
    media_validation = rendering.validate_output_media(
        source_path,
        source_info,
        output_path,
        expected_frame_count=rendered_frames,
        expected_width=source_info.video_stream.width,
        expected_height=source_info.video_stream.height,
        expected_fps=fps,
        source_has_audio=source_info.audio_stream is not None,
        settings=settings,
    )
    region_validation = privacy_rendering.verify_rendered_regions(
        source_path,
        output_path,
        [record.to_render_region() for record in final_regions],
        settings=settings,
        frame_width=source_info.video_stream.width,
        frame_height=source_info.video_stream.height,
    )
    final_video_hash = media.sha256_file(output_path)
    final_manifest = {
        "audio_handling": audio_handling,
        "case_id": case_id,
        "created_at_utc": media.utc_timestamp(),
        "final_video_sha256": final_video_hash,
        "media_validation": media_validation,
        "output_video": output_path.relative_to(package_root).as_posix(),
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
        "review_session_id": session_dir.name,
        "schema_version": SCHEMA_VERSION,
        "source_video_hash": source_info.sha256,
    }
    write_json(case_media_dir / "final_reviewed_manifest.json", final_manifest)
    write_json(case_media_dir / "final_region_provenance.json", provenance)
    write_json(
        case_media_dir / "final_metrics.json",
        human_reviewed_metrics(final_regions, corrections, case_id=case_id),
    )
    if actor is not None:
        append_event(
            session_dir,
            event_type="case_finalized",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state=None,
            after_state={
                "final_video_sha256": final_video_hash,
                "media_validation_passed": media_validation["passed"],
                "rendered_region_validation_passed": region_validation["passed"],
            },
            note="Reviewed final for this source case.",
            application_hashes={
                "final_video_sha256": final_video_hash,
            },
        )
    return {
        "case_id": case_id,
        "final_video": output_path.relative_to(package_root).as_posix(),
        "final_video_sha256": final_video_hash,
        "finalized": True,
        "media_validation_passed": media_validation["passed"],
        "rendered_region_validation_passed": region_validation["passed"],
        "schema_version": SCHEMA_VERSION,
    }


def canonical_finalization_gate_blockers(
    *,
    package_root: Path,
    case_id: str,
    session_dir: Path,
    working_state: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    try:
        candidate = resolve_canonical_latest_reviewed_candidate(
            package_root=package_root,
            case_id=case_id,
            session_dir=session_dir,
            require_blockers_resolved=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        return [f"{case_id}: stale_latest_reviewed: {exc}"]
    state = working_state or candidate["working_state"]
    blockers.extend(validate_canonical_sweep_acknowledgement(candidate, state))
    blockers.extend(validate_canonical_latest_validation_artifacts(candidate))
    blockers.extend(pending_latest_reviewed_job_blockers(package_root, case_id=case_id))
    return blockers


def finalize_canonical_reviewed_case(
    *,
    package_root: Path,
    case_id: str,
    session_dir: Path,
    actor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    blockers = finalization_blockers(
        package_root=package_root,
        case_id=case_id,
        session_dir=session_dir,
    )
    if not blockers["ready"]:
        return {
            "blockers": blockers["blockers"],
            "case_id": case_id,
            "finalized": False,
            "schema_version": CANONICAL_FINALIZATION_SCHEMA_VERSION,
        }

    try:
        candidate = resolve_canonical_latest_reviewed_candidate(
            package_root=package_root,
            case_id=case_id,
            session_dir=session_dir,
            require_blockers_resolved=True,
        )
        latest_media_validation = validate_canonical_latest_media(candidate)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "blockers": [f"{case_id}: stale_latest_reviewed: {exc}"],
            "case_id": case_id,
            "finalized": False,
            "schema_version": CANONICAL_FINALIZATION_SCHEMA_VERSION,
        }
    maybe_inject_finalization_failure("after_candidate_validation")

    sweep = require_canonical_sweep_acknowledgement(candidate, candidate["working_state"])
    task_identity = required_task_resolution_identity(
        package_root,
        candidate["working_state"],
        case_id=case_id,
    )
    correction_identity = correction_verification_identity(
        session_dir,
        candidate["working_state"],
        case_id=case_id,
    )
    guidance_snapshot = dict(candidate["unreviewed_guidance"])
    finalization_input = finalization_input_payload(
        candidate=candidate,
        sweep_acknowledgement=sweep,
        task_identity=task_identity,
        correction_identity=correction_identity,
    )
    finalization_input_sha256 = stable_hash(finalization_input)
    finalization_id = f"finalization_{finalization_input_sha256[:16]}"
    revision_root = package_root / "final" / "revisions" / finalization_id
    current_pointer = load_current_final_pointer(package_root)
    if current_pointer:
        try:
            current = validate_current_final_pointer(package_root, current_pointer)
            if (
                current.get("case_id") == case_id
                and current.get("finalization_input_sha256")
                == finalization_input_sha256
            ):
                final_role = current_final_role(package_root, session_dir, case_id=case_id)
                if final_role:
                    return canonical_finalization_result(
                        current["finalization_manifest_payload"],
                        final_role=final_role,
                        reused=True,
                    )
        except ValueError:
            pass

    publication_guard = finalization_publication_guard(
        candidate=candidate,
        sweep_acknowledgement=sweep,
    )
    if revision_root.exists():
        finalization_manifest = validate_finalization_revision(revision_root)
    else:
        temp_root = revision_root.with_name(f".{finalization_id}.{uuid.uuid4().hex}.tmp")
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=False)
        final_video_path = temp_root / "final_reviewed.mp4"
        materialization = materialize_final_video_from_latest(
            candidate["latest_video_path"],
            final_video_path,
        )
        maybe_inject_finalization_failure("after_final_video_link_copy")
        final_video_sha256 = media.sha256_file(final_video_path)
        if final_video_sha256 != candidate["latest_video_sha256"]:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ValueError("Final video hash does not match Latest reviewed video hash")
        materialization["byte_equal_to_latest_reviewed"] = True
        maybe_inject_finalization_failure("after_final_hash_validation")
        final_media_validation = privacy_rendering.validate_full_span_output(
            candidate["source_path"],
            final_video_path,
            expected_frame_count=candidate["expected_frame_count"],
            expected_width=candidate["expected_width"],
            expected_height=candidate["expected_height"],
            expected_fps=candidate["expected_fps"],
            source_has_audio=candidate["source_has_audio"],
        )
        final_media_validation = final_media_validation | {
            "byte_equal_to_latest_reviewed": True,
            "final_video_sha256": final_video_sha256,
            "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
            "schema_version": "final_media_validation.v1",
        }
        if final_media_validation.get("passed") is not True:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ValueError(f"{case_id}: Final media validation failed")
        write_json(temp_root / "final_media_validation.json", final_media_validation)
        maybe_inject_finalization_failure("after_final_media_validation_artifact")
        final_region_validation = {
            "case_id": case_id,
            "final_video_sha256": final_video_sha256,
            "latest_reviewed_region_validation": candidate["region_validation"],
            "latest_reviewed_region_validation_sha256": candidate[
                "region_validation_sha256"
            ],
            "latest_reviewed_revision_id": candidate["revision_id"],
            "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
            "passed": candidate["region_validation"].get("passed") is True,
            "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
            "schema_version": "final_region_validation_reference.v1",
            "source_frame_end": candidate["manifest"].source_frame_end,
            "source_frame_start": candidate["manifest"].source_frame_start,
        }
        if final_region_validation["passed"] is not True:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise ValueError(f"{case_id}: Latest reviewed region validation failed")
        write_json(temp_root / "final_region_validation.json", final_region_validation)
        maybe_inject_finalization_failure("after_final_region_validation_artifact")
        finalization_manifest = {
            "active_corrections_hash": candidate["reviewed_region_identity"].get(
                "active_corrections_hash"
            ),
            "active_decisions_hash": candidate["reviewed_region_identity"].get(
                "active_decisions_hash"
            ),
            "case_id": case_id,
            "category_run_identity_hash": candidate["category_run_identity_hash"],
            "correction_verification_set_hash": correction_identity[
                "correction_verification_set_hash"
            ],
            "final_media_validation": "final_media_validation.json",
            "final_media_validation_sha256": media.sha256_file(
                temp_root / "final_media_validation.json"
            ),
            "final_region_validation": "final_region_validation.json",
            "final_region_validation_sha256": media.sha256_file(
                temp_root / "final_region_validation.json"
            ),
            "final_video": "final_reviewed.mp4",
            "final_video_materialization": materialization,
            "final_video_materialization_mode": materialization["mode"],
            "final_video_sha256": final_video_sha256,
            "finalization_id": finalization_id,
            "finalization_input": finalization_input,
            "finalization_input_sha256": finalization_input_sha256,
            "finalization_status": "approved",
            "finalized_at": media.utc_timestamp(),
            "guidance_identity_hash": guidance_snapshot["guidance_identity_hash"],
            "guidance_unreviewed_count": guidance_snapshot["guidance_unreviewed_count"],
            "guidance_unreviewed_task_ids": guidance_snapshot[
                "guidance_unreviewed_task_ids"
            ],
            "latest_reviewed_media_validation_sha256": candidate[
                "media_validation_sha256"
            ],
            "latest_reviewed_region_validation_sha256": candidate[
                "region_validation_sha256"
            ],
            "latest_reviewed_revision_id": candidate["revision_id"],
            "latest_reviewed_revision_manifest_sha256": candidate[
                "revision_manifest_sha256"
            ],
            "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
            "media_validation_identity": final_media_validation_identity(
                candidate,
                latest_media_validation,
            ),
            "publication_guard": publication_guard,
            "region_validation_identity": {
                "latest_reviewed_region_validation_sha256": candidate[
                    "region_validation_sha256"
                ],
                "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
                "schema_version": "region_validation_identity.v1",
            },
            "resolved_task_set_hash": task_identity["resolved_task_set_hash"],
            "review_guidance_policy_version": review_guidance_policy.POLICY_VERSION,
            "review_session_id": session_dir.name,
            "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
            "schema_version": FINALIZATION_REVISION_SCHEMA_VERSION,
            "selected_categories": candidate["selected_categories"],
            "session_head_hash": candidate["session_head_hash"],
            "source_config_validation": candidate["source_config_validation"],
            "source_sha256": candidate["source_sha256"],
            "sweep_acknowledgement": sweep,
            "sweep_acknowledgement_sha256": sweep["sweep_acknowledgement_sha256"],
            "unreviewed_guidance_snapshot": guidance_snapshot,
            "workspace_manifest_sha256": candidate["workspace_manifest_sha256"],
        }
        finalization_manifest["finalization_manifest_sha256"] = stable_hash(
            finalization_manifest
        )
        write_json(temp_root / "finalization_manifest.json", finalization_manifest)
        maybe_inject_finalization_failure("after_finalization_manifest_write")
        validate_finalization_revision(temp_root)
        revision_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_root, revision_root)
        maybe_inject_finalization_failure("after_immutable_revision_move")

    latest_candidate = resolve_canonical_latest_reviewed_candidate(
        package_root=package_root,
        case_id=case_id,
        session_dir=session_dir,
        require_blockers_resolved=True,
    )
    latest_sweep = require_canonical_sweep_acknowledgement(
        latest_candidate,
        latest_candidate["working_state"],
    )
    latest_guard = finalization_publication_guard(
        candidate=latest_candidate,
        sweep_acknowledgement=latest_sweep,
    )
    if latest_guard != publication_guard:
        return {
            "blockers": [f"{case_id}: stale_latest_reviewed: finalization guard changed"],
            "case_id": case_id,
            "finalized": False,
            "schema_version": CANONICAL_FINALIZATION_SCHEMA_VERSION,
        }
    maybe_inject_finalization_failure("before_final_current_pointer_update")
    pointer_update = publish_final_current_pointer(
        package_root=package_root,
        revision_root=revision_root,
        finalization_manifest=finalization_manifest,
    )
    maybe_inject_finalization_failure("after_final_current_pointer_update")
    final_role = current_final_role(package_root, session_dir, case_id=case_id)
    if not final_role:
        raise ValueError(f"{case_id}: Final current pointer did not validate after publication")
    if actor is not None:
        append_event(
            session_dir,
            event_type="case_finalized",
            actor=actor,
            case_id=case_id,
            task_id=None,
            before_state=None,
            after_state={
                "final_video_sha256": finalization_manifest["final_video_sha256"],
                "finalization_id": finalization_id,
                "finalization_input_sha256": finalization_input_sha256,
                "latest_reviewed_revision_id": candidate["revision_id"],
                "media_validation_passed": True,
                "rendered_region_validation_passed": True,
            },
            note="Reviewed final approved from the current Latest reviewed revision.",
            application_hashes={
                "final_video_sha256": finalization_manifest["final_video_sha256"],
                "finalization_input_sha256": finalization_input_sha256,
            },
        )
    result = canonical_finalization_result(
        finalization_manifest,
        final_role=final_role,
        reused=False,
    )
    result["current_pointer"] = pointer_update
    return result


def resolve_canonical_latest_reviewed_candidate(
    *,
    package_root: Path,
    case_id: str,
    session_dir: Path,
    require_blockers_resolved: bool,
) -> dict[str, Any]:
    from redactor import review_working_state, reviewed_regions

    package_root = package_root.resolve()
    session_dir = session_dir.resolve()
    case_id = case_ids.validate_case_id(case_id)
    workspace = reviewed_regions.require_supported_workspace(package_root)
    pointer = reviewed_regions.load_current_pointer(package_root)
    if not pointer:
        raise ValueError("Latest reviewed current pointer is missing")
    validated = reviewed_regions.validate_current_pointer(package_root, pointer)
    if validated.get("case_id") != case_id:
        raise ValueError("Latest reviewed pointer belongs to a different case")
    if validated.get("review_session_id") != session_dir.name:
        raise ValueError("Latest reviewed pointer belongs to a different review session")
    revision_manifest = dict(validated["revision_manifest_payload"])
    revision_manifest_path = contained_package_path(
        package_root,
        str(validated.get("revision_manifest") or ""),
        label="Latest reviewed revision manifest",
    )
    latest_role = dict(validated.get("latest_reviewed") or {})
    latest_video_path = contained_package_path(
        package_root,
        str(latest_role.get("path") or ""),
        label="Latest reviewed video",
    )
    manifest = case_manifests.load_case_manifest(
        case_ids.case_manifest_path(package_root, case_id)
    )
    source_path = contained_package_path(
        package_root,
        str(manifest.source_path or ""),
        label="source video",
    )
    source_sha256 = media.sha256_file(source_path)
    if manifest.source_video_hash and source_sha256 != manifest.source_video_hash:
        raise ValueError("source hash mismatch")
    working_state = review_working_state.derive_working_state(package_root, session_dir)
    current_latest = dict(working_state.get("materialized_latest_reviewed") or {}).get(case_id)
    if not current_latest:
        raise ValueError("Latest reviewed is not current for the review state")
    if current_latest.get("revision_id") != validated.get("revision_id"):
        raise ValueError("Latest reviewed current revision mismatch")
    current_digest = reviewed_regions.reviewed_region_digest(
        package_root,
        session_dir,
        case_id=case_id,
        require_blockers_resolved=require_blockers_resolved,
    )
    reviewed_region_set_sha256 = str(current_digest.get("effective_regions_hash") or "")
    if reviewed_region_set_sha256 != str(revision_manifest.get("reviewed_region_set_sha256") or ""):
        raise ValueError("reviewed region hash mismatch")
    pointer_projection_hash = str(
        validated.get("working_state_projection_hash")
        or validated.get("accepted_working_state_revision")
        or ""
    )
    try:
        current_pointer_projection_hashes = (
            review_working_state.active_projection_hash_candidates_for_pointer(
                dict(working_state.get("active_projection_hashes") or {}),
                validated,
            )
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if pointer_projection_hash and pointer_projection_hash not in current_pointer_projection_hashes:
        raise ValueError("working-state generation mismatch")
    latest_video_sha256 = media.sha256_file(latest_video_path)
    if latest_video_sha256 != str(revision_manifest.get("video_artifact_sha256") or ""):
        raise ValueError("Latest reviewed video hash mismatch")
    revision_root = revision_manifest_path.parent
    media_validation_path = contained_path_under(
        revision_root,
        str(revision_manifest.get("media_validation") or ""),
        label="Latest reviewed media validation",
    )
    region_validation_path = contained_path_under(
        revision_root,
        str(revision_manifest.get("region_validation") or ""),
        label="Latest reviewed region validation",
    )
    media_validation = load_json(media_validation_path)
    region_validation = load_json(region_validation_path)
    content_identities = dict(validated.get("content_identities") or {})
    source_info = media.probe_media(source_path)
    expected_fps = rational_to_float(
        source_info.video_stream.nominal_frame_rate
    ) or rational_to_float(manifest.fps)
    if expected_fps is None or expected_fps <= 0:
        expected_fps = 30.0
    expected_frame_count = int(manifest.frame_count or source_info.video_stream.frame_count or 0)
    if expected_frame_count <= 0:
        raise ValueError("source frame count is unknown")
    source_config_validation = {
        "case_id": case_id,
        "category_run_identity_hash": workspace.get("category_run_manifest_identity_hash"),
        "configuration_freeze_hash": workspace.get("configuration_freeze_hash"),
        "latest_reviewed_revision_id": revision_manifest.get("revision_id"),
        "reviewed_region_set_sha256": reviewed_region_set_sha256,
        "schema_version": "finalization_source_config_validation.v1",
        "selected_categories": reviewed_regions.selected_categories(workspace),
        "source_frame_end": manifest.source_frame_end,
        "source_frame_start": manifest.source_frame_start,
        "source_sha256": source_sha256,
        "workspace_manifest_input_sha256": reviewed_regions.workspace_review_input_sha256_file(
            package_root / "workspace_manifest.json"
        ),
        "workspace_manifest_sha256": media.sha256_file(package_root / "workspace_manifest.json"),
    }
    return {
        "case_id": case_id,
        "category_run_identity_hash": workspace.get("category_run_manifest_identity_hash"),
        "content_identities": content_identities,
        "current_pointer_sha256": media.sha256_file(
            reviewed_regions.current_pointer_path(package_root)
        ),
        "expected_fps": float(expected_fps),
        "expected_frame_count": expected_frame_count,
        "expected_height": int(source_info.video_stream.height),
        "expected_width": int(source_info.video_stream.width),
        "latest_role": latest_role,
        "latest_video_path": latest_video_path,
        "latest_video_sha256": latest_video_sha256,
        "manifest": manifest,
        "media_validation": media_validation,
        "media_validation_path": media_validation_path,
        "media_validation_sha256": media.sha256_file(media_validation_path),
        "pointer": validated,
        "pointer_active_projection_hash_schema_version": (
            review_working_state.active_projection_pointer_schema(validated)
        ),
        "pointer_working_state_projection_hash": pointer_projection_hash,
        "region_validation": region_validation,
        "region_validation_path": region_validation_path,
        "region_validation_sha256": media.sha256_file(region_validation_path),
        "revision_id": str(revision_manifest["revision_id"]),
        "revision_manifest": revision_manifest,
        "revision_manifest_path": revision_manifest_path,
        "revision_manifest_sha256": str(revision_manifest.get("revision_manifest_sha256") or ""),
        "reviewed_region_identity": dict(current_digest.get("reviewed_region_identity") or {}),
        "reviewed_region_set_sha256": reviewed_region_set_sha256,
        "unreviewed_guidance": review_guidance_policy.unreviewed_guidance_snapshot(
            package_root=package_root,
            working_state=working_state,
            case_id=case_id,
        ),
        "selected_categories": reviewed_regions.selected_categories(workspace),
        "session_head_hash": validated.get("session_head_hash"),
        "source_config_validation": source_config_validation,
        "source_has_audio": source_info.audio_stream is not None,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "working_state": working_state,
        "workspace": workspace,
        "workspace_manifest_sha256": media.sha256_file(package_root / "workspace_manifest.json"),
    }


def validate_canonical_latest_validation_artifacts(candidate: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    content_identities = dict(candidate.get("content_identities") or {})
    expected_media = dict(content_identities.get("latest_reviewed_media_validation") or {})
    expected_region = dict(content_identities.get("latest_reviewed_region_validation") or {})
    if expected_media.get("sha256") and expected_media.get("sha256") != candidate[
        "media_validation_sha256"
    ]:
        blockers.append(f"{candidate['case_id']}: Latest reviewed media-validation hash mismatch")
    if expected_region.get("sha256") and expected_region.get("sha256") != candidate[
        "region_validation_sha256"
    ]:
        blockers.append(f"{candidate['case_id']}: Latest reviewed region-validation hash mismatch")
    if candidate["media_validation"].get("passed") is not True:
        blockers.append(f"{candidate['case_id']}: Latest reviewed media validation has not passed")
    if candidate["region_validation"].get("passed") is not True:
        blockers.append(f"{candidate['case_id']}: Latest reviewed region validation has not passed")
    manifest = candidate["manifest"]
    revision_manifest = candidate["revision_manifest"]
    if revision_manifest.get("source_frame_start") != manifest.source_frame_start:
        blockers.append(f"{candidate['case_id']}: Latest reviewed source-frame start mismatch")
    if revision_manifest.get("source_frame_end") != manifest.source_frame_end:
        blockers.append(f"{candidate['case_id']}: Latest reviewed source-frame end mismatch")
    return blockers


def validate_canonical_latest_media(candidate: dict[str, Any]) -> dict[str, Any]:
    blockers = validate_canonical_latest_validation_artifacts(candidate)
    if blockers:
        raise ValueError("; ".join(blockers))
    validation = privacy_rendering.validate_full_span_output(
        candidate["source_path"],
        candidate["latest_video_path"],
        expected_frame_count=candidate["expected_frame_count"],
        expected_width=candidate["expected_width"],
        expected_height=candidate["expected_height"],
        expected_fps=candidate["expected_fps"],
        source_has_audio=candidate["source_has_audio"],
    )
    if validation.get("passed") is not True:
        raise ValueError("Latest reviewed media failed full-span validation")
    return validation


def sweep_acknowledgement_binding(
    *,
    package_root: Path,
    session_dir: Path,
    case_id: str,
) -> dict[str, Any]:
    candidate = resolve_canonical_latest_reviewed_candidate(
        package_root=package_root,
        case_id=case_id,
        session_dir=session_dir,
        require_blockers_resolved=False,
    )
    return {
        "acknowledgement_timestamp": media.utc_timestamp(),
        "case_id": case_id,
        "current_session_head_before_ack": last_event_hash(session_dir),
        "explicit_reviewer_action": "full_video_sweep_acknowledged",
        "guidance_identity_hash": candidate["unreviewed_guidance"][
            "guidance_identity_hash"
        ],
        "guidance_unreviewed_count": candidate["unreviewed_guidance"][
            "guidance_unreviewed_count"
        ],
        "guidance_unreviewed_task_ids": candidate["unreviewed_guidance"][
            "guidance_unreviewed_task_ids"
        ],
        "latest_reviewed_revision_id": candidate["revision_id"],
        "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
        "review_guidance_policy_version": review_guidance_policy.POLICY_VERSION,
        "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
        "schema_version": FULL_VIDEO_SWEEP_ACK_SCHEMA_VERSION,
        "session_head_hash": candidate["session_head_hash"],
        "session_id": session_dir.resolve().name,
        "source_sha256": candidate["source_sha256"],
        "sweep_coverage_status": review_guidance_policy.SWEEP_COVERAGE_STATUS,
        "unadjudicated_guidance_status": review_guidance_policy.UNADJUDICATED_STATUS,
        "workspace_manifest_sha256": candidate["workspace_manifest_sha256"],
    }


def validate_canonical_sweep_acknowledgement(
    candidate: dict[str, Any],
    working_state: dict[str, Any],
) -> list[str]:
    case_id = str(candidate["case_id"])
    acknowledgement = dict(working_state.get("sweep_acknowledgements") or {}).get(case_id)
    if not acknowledgement:
        return [f"{case_id}: full-video sweep has not been acknowledged for Latest reviewed"]
    if acknowledgement.get("schema_version") != FULL_VIDEO_SWEEP_ACK_SCHEMA_VERSION:
        return [f"{case_id}: full-video sweep acknowledgement is not revision-bound"]
    expected = {
        "case_id": case_id,
        "latest_reviewed_revision_id": candidate["revision_id"],
        "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
        "review_guidance_policy_version": review_guidance_policy.POLICY_VERSION,
        "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
        "session_id": candidate["pointer"].get("review_session_id"),
        "source_sha256": candidate["source_sha256"],
        "workspace_manifest_sha256": candidate["workspace_manifest_sha256"],
    }
    guidance = dict(candidate.get("unreviewed_guidance") or {})
    expected["guidance_identity_hash"] = guidance.get("guidance_identity_hash")
    expected["guidance_unreviewed_count"] = guidance.get("guidance_unreviewed_count")
    blockers = [
        f"{case_id}: sweep_acknowledgement:{key} mismatch"
        for key, value in expected.items()
        if str(acknowledgement.get(key) or "") != str(value or "")
    ]
    if acknowledgement.get("explicit_reviewer_action") != "full_video_sweep_acknowledged":
        blockers.append(f"{case_id}: sweep_acknowledgement explicit reviewer action missing")
    return blockers


def require_canonical_sweep_acknowledgement(
    candidate: dict[str, Any],
    working_state: dict[str, Any],
) -> dict[str, Any]:
    blockers = validate_canonical_sweep_acknowledgement(candidate, working_state)
    if blockers:
        raise ValueError("; ".join(blockers))
    acknowledgement = dict(working_state.get("sweep_acknowledgements") or {})[
        candidate["case_id"]
    ]
    acknowledgement["sweep_acknowledgement_sha256"] = sweep_acknowledgement_sha256(
        acknowledgement
    )
    return acknowledgement


def sweep_acknowledgement_sha256(acknowledgement: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in dict(acknowledgement).items()
        if key not in {"event_hash", "event_id", "sweep_acknowledgement_sha256"}
    }
    return stable_hash(payload)


def required_task_resolution_identity(
    package_root: Path,
    working_state: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    from redactor import review_working_state

    decisions_by_case = review_working_state.task_decisions_by_case_from_payload(working_state)
    resolutions_by_case = review_working_state.task_resolutions_by_case_from_payload(working_state)
    tasks_payload = load_json(package_root / "reviewer_tasks.json") if (
        package_root / "reviewer_tasks.json"
    ).exists() else {"tasks": []}
    task_sources = [
        *list(tasks_payload.get("tasks") or []),
        *list(working_state.get("active_manual_tasks") or []),
    ]
    for task in task_sources:
        task = dict(task)
        if task.get("technical_only") or str(task.get("case_id") or "") != case_id:
            continue
        task_id = str(task.get("review_task_id") or "")
        if not task_id:
            continue
        if review_working_state.durable_edit_verification_task_id(task_id):
            continue
        decision = dict(
            review_working_state.effective_task_state_for_case(
                decisions_by_case,
                resolutions_by_case,
                case_id=case_id,
                task_id=task_id,
            )
            or {}
        )
        rows[task_id] = {
            "action": decision.get("action"),
            "blocking": bool(task.get("blocking")),
            "case_id": case_id,
            "category": task.get("category"),
            "correction_required": bool(task.get("correction_required")),
            "decision_status": decision.get("decision_status"),
            "issue_type": task.get("issue_type"),
            "normal_task_class": task.get("normal_task_class"),
            "resolved_by": decision.get("resolved_by") or "explicit_decision"
            if decision
            else None,
            "task_id": task_id,
        }
    ordered = [rows[key] for key in sorted(rows)]
    return {
        "resolved_required_tasks": ordered,
        "resolved_task_count": len(ordered),
        "resolved_task_set_hash": stable_hash(ordered),
        "schema_version": "resolved_task_set_identity.v1",
    }


def correction_verification_identity(
    session_dir: Path,
    working_state: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    active_ids = set(str(item) for item in list(working_state.get("active_correction_ids") or []))
    corrections = [
        correction
        for correction in latest_corrections(
            load_corrections_jsonl(session_dir / "corrections.jsonl"),
            case_id=case_id,
        )
        if correction.correction_id in active_ids
    ]
    from redactor import review_working_state

    decisions_by_case = review_working_state.task_decisions_by_case_from_payload(working_state)
    rows = []
    for correction in corrections:
        verification_task_id = f"correction_verify_{case_id}_{correction.correction_id}"
        decision = dict(
            review_working_state.task_decision_for_case(
                decisions_by_case,
                case_id=case_id,
                task_id=verification_task_id,
            )
            or {}
        )
        rows.append(
            {
                "correction_hash": correction.correction_hash(),
                "correction_id": correction.correction_id,
                "decision_action": decision.get("action"),
                "decision_status": decision.get("decision_status"),
                "revision_id": correction.revision_id,
                "verification_task_id": verification_task_id,
            }
        )
    rows.sort(key=lambda row: str(row["correction_id"]))
    return {
        "active_correction_count": len(rows),
        "correction_verification_set_hash": stable_hash(rows),
        "correction_verifications": rows,
        "schema_version": "correction_verification_set_identity.v1",
    }


def finalization_input_payload(
    *,
    candidate: dict[str, Any],
    sweep_acknowledgement: dict[str, Any],
    task_identity: dict[str, Any],
    correction_identity: dict[str, Any],
) -> dict[str, Any]:
    identity = dict(candidate["reviewed_region_identity"])
    return {
        "active_corrections_hash": identity.get("active_corrections_hash"),
        "active_decisions_hash": identity.get("active_decisions_hash"),
        "case_id": candidate["case_id"],
        "category_run_identity_hash": candidate["category_run_identity_hash"],
        "correction_verification_set_hash": correction_identity[
            "correction_verification_set_hash"
        ],
        "latest_reviewed_media_validation_sha256": candidate["media_validation_sha256"],
        "latest_reviewed_region_validation_sha256": candidate["region_validation_sha256"],
        "latest_reviewed_revision_id": candidate["revision_id"],
        "latest_reviewed_revision_manifest_sha256": candidate["revision_manifest_sha256"],
        "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
        "renderer_configuration_sha256": identity.get("renderer_configuration_sha256"),
        "resolved_task_set_hash": task_identity["resolved_task_set_hash"],
        "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
        "schema_version": "finalization_input_identity.v1",
        "selected_categories": candidate["selected_categories"],
        "session_id": candidate["pointer"].get("review_session_id"),
        "source_sha256": candidate["source_sha256"],
        "sweep_acknowledgement_sha256": sweep_acknowledgement[
            "sweep_acknowledgement_sha256"
        ],
        "working_state_generation": candidate["working_state"].get("active_projection_hash"),
        "workspace_manifest_sha256": candidate["workspace_manifest_sha256"],
    }


def finalization_publication_guard(
    *,
    candidate: dict[str, Any],
    sweep_acknowledgement: dict[str, Any],
) -> dict[str, Any]:
    identity = dict(candidate["reviewed_region_identity"])
    return {
        "active_corrections_hash": identity.get("active_corrections_hash"),
        "active_decisions_hash": identity.get("active_decisions_hash"),
        "case_id": candidate["case_id"],
        "current_latest_reviewed_pointer_sha256": candidate["current_pointer_sha256"],
        "latest_reviewed_revision_id": candidate["revision_id"],
        "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
        "reviewed_region_set_sha256": candidate["reviewed_region_set_sha256"],
        "schema_version": "finalization_publication_guard.v1",
        "session_id": candidate["pointer"].get("review_session_id"),
        "sweep_acknowledgement_sha256": sweep_acknowledgement[
            "sweep_acknowledgement_sha256"
        ],
        "working_state_generation": candidate["working_state"].get("active_projection_hash"),
        "workspace_manifest_sha256": candidate["workspace_manifest_sha256"],
    }


def media_validation_identity(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "latest_reviewed_media_validation_sha256": candidate["media_validation_sha256"],
        "latest_reviewed_video_sha256": candidate["latest_video_sha256"],
        "schema_version": "latest_reviewed_media_validation_identity.v1",
    }


def final_media_validation_identity(
    candidate: dict[str, Any],
    latest_media_validation: dict[str, Any],
) -> dict[str, Any]:
    return media_validation_identity(candidate) | {
        "latest_reviewed_full_span_validation": latest_media_validation,
    }


def materialize_final_video_from_latest(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        mode = "verified_copy"
    return {
        "file_size_bytes": dst.stat().st_size,
        "latest_reviewed_file_size_bytes": src.stat().st_size,
        "mode": mode,
        "schema_version": "final_video_materialization.v1",
    }


def final_current_pointer_path(package_root: Path) -> Path:
    return package_root.resolve() / FINAL_CURRENT_POINTER_RELATIVE_PATH


def load_current_final_pointer(package_root: Path) -> dict[str, Any]:
    path = final_current_pointer_path(package_root)
    if not path.exists():
        return {}
    return load_json(path)


def publish_final_current_pointer(
    *,
    package_root: Path,
    revision_root: Path,
    finalization_manifest: dict[str, Any],
) -> dict[str, Any]:
    pointer_path = final_current_pointer_path(package_root)
    manifest_path = revision_root / "finalization_manifest.json"
    final_video_path = revision_root / str(finalization_manifest["final_video"])
    pointer = {
        "approved_latest_reviewed_revision_id": finalization_manifest[
            "latest_reviewed_revision_id"
        ],
        "case_id": finalization_manifest["case_id"],
        "current": True,
        "final_video": final_video_path.relative_to(package_root).as_posix(),
        "final_video_sha256": finalization_manifest["final_video_sha256"],
        "finalization_id": finalization_manifest["finalization_id"],
        "finalization_input_sha256": finalization_manifest["finalization_input_sha256"],
        "finalization_manifest": manifest_path.relative_to(package_root).as_posix(),
        "finalization_manifest_artifact_sha256": media.sha256_file(manifest_path),
        "finalization_manifest_sha256": finalization_manifest[
            "finalization_manifest_sha256"
        ],
        "latest_reviewed_video_sha256": finalization_manifest[
            "latest_reviewed_video_sha256"
        ],
        "published_at_utc": media.utc_timestamp(),
        "review_session_id": finalization_manifest["review_session_id"],
        "reviewed_region_set_sha256": finalization_manifest["reviewed_region_set_sha256"],
        "schema_version": FINAL_CURRENT_POINTER_SCHEMA_VERSION,
        "session_head_hash": finalization_manifest["session_head_hash"],
        "source_sha256": finalization_manifest["source_sha256"],
        "status": "ready",
        "sweep_acknowledgement_sha256": finalization_manifest[
            "sweep_acknowledgement_sha256"
        ],
        "workspace_manifest_sha256": finalization_manifest["workspace_manifest_sha256"],
    }
    atomic_write_final_pointer(pointer_path, pointer)
    reopened = load_current_final_pointer(package_root)
    validated = validate_current_final_pointer(package_root, reopened)
    return {
        "current_pointer_sha256": media.sha256_file(pointer_path),
        "finalization_id": validated["finalization_id"],
        "path": pointer_path.relative_to(package_root).as_posix(),
        "schema_version": "final_current_pointer_update.v1",
        "status": "updated",
    }


def atomic_write_final_pointer(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    maybe_inject_finalization_failure("during_final_current_pointer_update")
    os.replace(temp_path, path)


def validate_current_final_pointer(package_root: Path, pointer: dict[str, Any]) -> dict[str, Any]:
    if pointer.get("schema_version") != FINAL_CURRENT_POINTER_SCHEMA_VERSION:
        raise ValueError("Final current pointer schema mismatch")
    if pointer.get("status") != "ready" or pointer.get("current") is not True:
        raise ValueError("Final current pointer is not ready")
    case_id = case_ids.validate_case_id(str(pointer.get("case_id") or ""))
    manifest_path = contained_package_path(
        package_root,
        str(pointer.get("finalization_manifest") or ""),
        label="finalization manifest",
    )
    if pointer.get("finalization_manifest_artifact_sha256") != media.sha256_file(
        manifest_path
    ):
        raise ValueError("Finalization manifest artifact hash mismatch")
    finalization_manifest = load_json(manifest_path)
    validate_finalization_manifest_payload(finalization_manifest)
    final_video_path = contained_package_path(
        package_root,
        str(pointer.get("final_video") or ""),
        label="Final video",
    )
    if pointer.get("final_video_sha256") != media.sha256_file(final_video_path):
        raise ValueError("Final video hash mismatch")
    for key in (
        "case_id",
        "finalization_id",
        "finalization_input_sha256",
        "reviewed_region_set_sha256",
        "source_sha256",
        "sweep_acknowledgement_sha256",
    ):
        if str(pointer.get(key) or "") != str(finalization_manifest.get(key) or ""):
            raise ValueError(f"Final pointer {key} mismatch")
    if (
        pointer.get("approved_latest_reviewed_revision_id")
        != finalization_manifest.get("latest_reviewed_revision_id")
    ):
        raise ValueError("Final pointer Latest reviewed revision mismatch")
    if pointer.get("latest_reviewed_video_sha256") != finalization_manifest.get(
        "latest_reviewed_video_sha256"
    ):
        raise ValueError("Final pointer Latest reviewed video mismatch")
    return dict(pointer) | {
        "case_id": case_id,
        "finalization_manifest_payload": finalization_manifest,
    }


def validate_finalization_revision(revision_root: Path) -> dict[str, Any]:
    manifest_path = revision_root / "finalization_manifest.json"
    finalization_manifest = load_json(manifest_path)
    validate_finalization_manifest_payload(finalization_manifest)
    final_video_path = revision_root / str(finalization_manifest["final_video"])
    final_media_path = revision_root / str(finalization_manifest["final_media_validation"])
    final_region_path = revision_root / str(finalization_manifest["final_region_validation"])
    for path in (final_video_path, final_media_path, final_region_path):
        if not path.exists():
            raise ValueError(f"Finalization revision artifact missing: {path.name}")
    if media.sha256_file(final_video_path) != finalization_manifest["final_video_sha256"]:
        raise ValueError("Final video artifact hash mismatch")
    if finalization_manifest["final_video_sha256"] != finalization_manifest[
        "latest_reviewed_video_sha256"
    ]:
        raise ValueError("Final video is not byte-identical to Latest reviewed")
    if media.sha256_file(final_media_path) != finalization_manifest[
        "final_media_validation_sha256"
    ]:
        raise ValueError("Final media validation hash mismatch")
    if media.sha256_file(final_region_path) != finalization_manifest[
        "final_region_validation_sha256"
    ]:
        raise ValueError("Final region validation hash mismatch")
    return finalization_manifest


def validate_finalization_manifest_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != FINALIZATION_REVISION_SCHEMA_VERSION:
        raise ValueError("Finalization manifest schema mismatch")
    for key in (
        "case_id",
        "final_video",
        "final_video_sha256",
        "finalization_id",
        "finalization_input_sha256",
        "latest_reviewed_revision_id",
        "latest_reviewed_video_sha256",
        "reviewed_region_set_sha256",
        "source_sha256",
        "sweep_acknowledgement_sha256",
    ):
        if not payload.get(key):
            raise ValueError(f"Finalization manifest missing {key}")


def current_final_role(
    package_root: Path,
    session_dir: Path | None,
    *,
    case_id: str,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    case_id = case_ids.validate_case_id(case_id)
    pointer = load_current_final_pointer(package_root)
    if not pointer:
        return {}
    try:
        validated = validate_current_final_pointer(package_root, pointer)
    except ValueError:
        return {}
    if validated.get("case_id") != case_id:
        return {}
    finalization_manifest = dict(validated["finalization_manifest_payload"])
    if session_dir is not None:
        session_dir = session_dir.resolve()
        if finalization_manifest.get("review_session_id") != session_dir.name:
            return {}
        try:
            candidate = resolve_canonical_latest_reviewed_candidate(
                package_root=package_root,
                case_id=case_id,
                session_dir=session_dir,
                require_blockers_resolved=False,
            )
            sweep = require_canonical_sweep_acknowledgement(
                candidate,
                candidate["working_state"],
            )
        except (FileNotFoundError, ValueError):
            return {}
        if candidate["revision_id"] != finalization_manifest["latest_reviewed_revision_id"]:
            return {}
        if candidate["latest_video_sha256"] != finalization_manifest[
            "latest_reviewed_video_sha256"
        ]:
            return {}
        if candidate["reviewed_region_set_sha256"] != finalization_manifest[
            "reviewed_region_set_sha256"
        ]:
            return {}
        if sweep["sweep_acknowledgement_sha256"] != finalization_manifest[
            "sweep_acknowledgement_sha256"
        ]:
            return {}
    return {
        "current_status": "current",
        "final_video_sha256": finalization_manifest["final_video_sha256"],
        "finalization_id": finalization_manifest["finalization_id"],
        "finalization_input_sha256": finalization_manifest["finalization_input_sha256"],
        "label": "Final",
        "latest_reviewed_revision_id": finalization_manifest["latest_reviewed_revision_id"],
        "path": str(validated["final_video"]),
        "role": "final_reviewed",
        "schema_version": "case_version_role.v1",
        "sha256": finalization_manifest["final_video_sha256"],
        "source_frame_end": finalization_manifest.get("source_config_validation", {}).get(
            "source_frame_end"
        )
        or finalization_manifest.get("finalization_input", {}).get("source_frame_end"),
        "source_frame_start": finalization_manifest.get("source_config_validation", {}).get(
            "source_frame_start"
        )
        or finalization_manifest.get("finalization_input", {}).get("source_frame_start")
        or 0,
        "sweep_acknowledgement_sha256": finalization_manifest[
            "sweep_acknowledgement_sha256"
        ],
        "validation_status": "passed",
    }


def canonical_finalization_result(
    finalization_manifest: dict[str, Any],
    *,
    final_role: dict[str, Any],
    reused: bool,
) -> dict[str, Any]:
    return {
        "byte_equal_to_latest_reviewed": True,
        "case_id": finalization_manifest["case_id"],
        "final_video": final_role["path"],
        "final_video_materialization_mode": finalization_manifest[
            "final_video_materialization_mode"
        ],
        "final_video_sha256": finalization_manifest["final_video_sha256"],
        "finalization_id": finalization_manifest["finalization_id"],
        "finalization_input_sha256": finalization_manifest["finalization_input_sha256"],
        "finalized": True,
        "latest_reviewed_revision_id": finalization_manifest["latest_reviewed_revision_id"],
        "latest_reviewed_video_sha256": finalization_manifest[
            "latest_reviewed_video_sha256"
        ],
        "media_validation_passed": True,
        "region_validation_passed": True,
        "rendered_region_validation_passed": True,
        "reused_existing_finalization": reused,
        "reviewed_region_set_sha256": finalization_manifest["reviewed_region_set_sha256"],
        "schema_version": CANONICAL_FINALIZATION_SCHEMA_VERSION,
    }


def pending_latest_reviewed_job_blockers(package_root: Path, *, case_id: str) -> list[str]:
    timing_path = package_root / "materialization_timing_audit.json"
    if not timing_path.exists():
        return []
    try:
        payload = load_json(timing_path)
    except (json.JSONDecodeError, OSError):
        return [f"{case_id}: materialization timing audit is unreadable"]
    blockers = []
    for job in list(payload.get("jobs") or []):
        if str(dict(job).get("case_id") or "") != case_id:
            continue
        status = str(dict(job).get("status") or "")
        if status in {"queued", "running"}:
            blockers.append(f"{case_id}: latest_reviewed_materialization_{status}")
    return blockers


def contained_package_path(package_root: Path, raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    if not raw_path or path.is_absolute():
        raise ValueError(f"{label} path must be relative to the workspace")
    return case_ids.contained_path(package_root, *path.parts)


def contained_path_under(root: Path, raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    if not raw_path or path.is_absolute():
        raise ValueError(f"{label} path must be relative")
    return case_ids.contained_path(root, *path.parts)


def maybe_inject_finalization_failure(stage: str) -> None:
    if os.environ.get(FINALIZATION_FAILURE_INJECTION_ENV) == stage:
        raise ValueError(f"injected finalization failure: {stage}")


def latest_corrections(
    corrections: list[ReviewCorrection],
    *,
    case_id: str | None = None,
) -> list[ReviewCorrection]:
    latest: dict[str, ReviewCorrection] = {}
    order: list[str] = []
    for correction in corrections:
        if case_id is not None and correction.case_id != case_id:
            continue
        if correction.correction_id not in latest:
            order.append(correction.correction_id)
        latest[correction.correction_id] = correction
    return [latest[correction_id] for correction_id in order]


def apply_corrections(
    base_regions: list[PrivacyRegionRecord],
    corrections: list[ReviewCorrection],
    *,
    case_id: str,
    source_frame_offset: int = 0,
) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    active = list(base_regions)
    suppressed_region_hashes: set[str] = set()
    added: list[PrivacyRegionRecord] = []
    applied: list[dict[str, Any]] = []
    for correction in corrections:
        if correction.case_id != case_id:
            continue
        if correction.operation in {
            OP_SUPPRESS,
            OP_SUPPRESS_TRACK_SEGMENT,
            OP_NOT_SENSITIVE,
            OP_NOT_VISIBLE,
            OP_REPLACE,
        }:
            if not correction.input_region_ids and correction.operation != OP_NOT_VISIBLE:
                raise ValueError("suppression/replacement corrections require input_region_ids")
            suppressed_region_hashes.update(correction.input_region_ids)
        if correction.operation in {OP_ADD, OP_REPLACE}:
            for frame, box in interpolate_correction_boxes(correction).items():
                render_frame = frame - source_frame_offset
                if render_frame < 0:
                    raise ValueError("correction frame is before the case source interval")
                added.append(
                    PrivacyRegionRecord(
                        approval_state=APPROVAL_ACCEPTED,
                        bbox_xyxy=box,
                        category=normalize_category(correction.category),
                        object_id=correction.correction_id,
                        provenance="reviewed_correction",
                        render_eligible=True,
                        source_frame=render_frame,
                        source_record={
                            "correction_hash": correction.correction_hash(),
                            "correction_id": correction.correction_id,
                            "operation": correction.operation,
                            "source_frame": frame,
                            "task_id": correction.task_id,
                        },
                    )
                )
        applied.append(
            {
                "correction_hash": correction.correction_hash(),
                "correction_id": correction.correction_id,
                "operation": correction.operation,
                "source_frame_end": correction.source_frame_end,
                "source_frame_start": correction.source_frame_start,
            }
        )
    kept = [
        record
        for record in active
        if region_identifier(record) not in suppressed_region_hashes
        and stable_hash(record.to_dict()) not in suppressed_region_hashes
    ]
    final_regions = sorted(
        kept + added,
        key=lambda item: (
            item.source_frame,
            item.category,
            item.object_id or "",
            item.provenance,
            item.bbox_xyxy or (-1, -1, -1, -1),
        ),
    )
    provenance = {
        "applied_corrections": applied,
        "base_region_count": len(base_regions),
        "final_region_count": len(final_regions),
        "schema_version": FINAL_REGION_SCHEMA_VERSION,
        "suppressed_region_count": len(base_regions) - len(kept),
    }
    return final_regions, provenance


def human_reviewed_metrics(
    final_regions: list[PrivacyRegionRecord],
    corrections: list[ReviewCorrection],
    *,
    case_id: str,
) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    frame_by_category: dict[str, set[int]] = {}
    for region in final_regions:
        category_counts[region.category] = category_counts.get(region.category, 0) + 1
        frame_by_category.setdefault(region.category, set()).add(region.source_frame)
    case_corrections = [correction for correction in corrections if correction.case_id == case_id]
    return {
        "categories": {
            category: {
                "accepted_final_privacy_regions": category_counts.get(category, 0),
                "unique_source_frames_containing_accepted_final_regions": len(
                    frame_by_category.get(category, set())
                ),
            }
            for category in sorted(set(MANUAL_REDACTION_CATEGORIES) | set(category_counts))
        },
        "combined": {
            "frames_containing_both": len(
                frame_by_category.get(CATEGORY_FACE, set())
                & frame_by_category.get(CATEGORY_PLATE, set())
            ),
            "frames_containing_either": len(
                frame_by_category.get(CATEGORY_FACE, set())
                | frame_by_category.get(CATEGORY_PLATE, set())
            ),
        },
        "correction_count": len(case_corrections),
        "correction_keyframe_count": sum(
            len(correction.keyframes) for correction in case_corrections
        ),
        "metric_scope": "human_reviewed_final_region_statistics",
        "schema_version": "human_reviewed_final_metrics.v1",
    }


def load_privacy_region_records(path: Path) -> list[PrivacyRegionRecord]:
    records: list[PrivacyRegionRecord] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(privacy_region_from_dict(payload))
    return records


def privacy_region_from_dict(payload: dict[str, Any]) -> PrivacyRegionRecord:
    bbox = payload.get("bbox_xyxy")
    return PrivacyRegionRecord(
        approval_state=str(payload.get("approval_state") or APPROVAL_ACCEPTED),
        bbox_xyxy=tuple(int(value) for value in bbox) if bbox is not None else None,
        category=normalize_category(str(payload["category"])),
        object_id=str(payload["object_id"]) if payload.get("object_id") is not None else None,
        provenance=str(payload.get("provenance") or "unknown"),
        reasons=tuple(str(item) for item in payload.get("reasons") or []),
        render_eligible=bool(payload.get("render_eligible", True)),
        source_frame=int(payload["source_frame"]),
        source_record=dict(payload.get("source_record") or {}),
    )


def write_privacy_region_records(path: Path, records: list[PrivacyRegionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in sorted(records, key=lambda item: (item.source_frame, item.category)):
            file.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def region_identifier(record: PrivacyRegionRecord) -> str:
    source = dict(record.source_record)
    for key in ("region_id", "track_uid", "correction_id"):
        if source.get(key):
            return str(source[key])
    return stable_hash(record.to_dict())


def has_sweep_acknowledgement(session_dir: Path, case_id: str) -> bool:
    return any(
        event.get("event_type") == "full_video_sweep_acknowledged"
        and event.get("case_id") == case_id
        for event in load_events(session_dir)
    )


def resolve_package_path(package_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (package_root / path).resolve()


def copy_reference_media(src: Path, dst: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": dst.as_posix(),
        "sha256": media.sha256_file(dst),
        "source_path": src.as_posix(),
    }


def normalize_category(category: str) -> str:
    return normalize_manual_redaction_category(category)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ffprobe_validate(path: Path) -> dict[str, Any]:
    info = media.probe_media(path)
    return {
        "codec": info.video_stream.codec,
        "frame_count": int(info.video_stream.frame_count or 0),
        "height": info.video_stream.height,
        "pixel_format": info.video_stream.pixel_format,
        "sha256": info.sha256,
        "width": info.video_stream.width,
    }
