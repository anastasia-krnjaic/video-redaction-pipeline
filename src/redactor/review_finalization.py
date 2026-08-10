from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from redactor import case_ids, case_manifests, media, privacy_rendering, rendering
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
    load_events,
    validate_event_hash_chain,
)

SCHEMA_VERSION = "review_finalization.v1"
FINAL_REGION_SCHEMA_VERSION = "final_region_provenance.v1"


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
    decisions = dict(working_state.get("task_decisions") or {})
    for task in tasks:
        task_id = str(task["review_task_id"])
        decision = decisions.get(task_id)
        if task.get("blocking") and decision is None:
            blockers.append(f"{task_id}: required task has no decision")
            continue
        if decision is None:
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
        decision = decisions.get(task_id)
        if decision is None:
            blockers.append(f"{task_id}: reviewer-added issue has no decision")
            continue
        if str(decision.get("action")) not in GREEN_ACTIONS:
            blockers.append(f"{task_id}: reviewer-added issue remains unresolved")

    if case_id not in dict(working_state.get("sweep_acknowledgements") or {}):
        blockers.append(f"{case_id}: full-video sweep has not been acknowledged")

    blockers.extend(object_policy_blockers(working_state, decisions, case_id=case_id))
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

    return {
        "blocker_count": len(blockers),
        "blockers": blockers,
        "case_id": case_id,
        "ready": not blockers,
        "schema_version": "review_finalization_blockers.v1",
    }


def object_policy_blockers(
    working_state: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
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
        if policy in {"keep_visible", "not_sensitive"}:
            task_id = str(
                after.get("verification_task_id") or f"object_policy_verify_{case_id}_{entity_id}"
            )
            decision = decisions.get(task_id)
            if decision is None:
                action_label = "Keep" if policy == "keep_visible" else "Not sensitive"
                blockers.append(f"{label}: {action_label} policy has not been verified")
                continue
            if str(decision.get("action")) not in GREEN_ACTIONS:
                action_label = "Keep" if policy == "keep_visible" else "Not sensitive"
                blockers.append(f"{label}: {action_label} policy verification is not green")
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

    from redactor import review_working_state

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
