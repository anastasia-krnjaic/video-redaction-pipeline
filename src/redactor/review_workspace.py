from __future__ import annotations

# ruff: noqa: E501
import json
import mimetypes
import os
import re
import shutil
import threading
import webbrowser
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
    media,
    plate_rendering,
    privacy_entities,
    privacy_rendering,
    rendering,
    review_finalization,
    review_sessions,
)
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    stable_hash,
)

SCHEMA_VERSION = "final_review_workspace.v2.2"
DAY_PARENT_PATH = Path("artifacts/day_primary.mp4")
DAY_CANONICAL_SOURCE_FRAME_START = 1828
DAY_CANONICAL_SOURCE_FRAME_END = 5694

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
    raw_signals = load_raw_audit_signals(v1_package_root / "audit" / "audit_tasks.json")
    tasks = aggregate_reviewer_tasks(raw_signals)
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
    write_json(
        package_root / "audit_decisions.template.json",
        {"decisions": [], "schema_version": "case_review_decisions_export.v1"},
    )
    write_entity_index(package_root)
    write_workspace_assets(package_root)
    workspace_manifest = {
        "case_count": 2,
        "cases": {
            "day": day_lineage["case_manifest_path"],
            "night": night_manifest["case_manifest_path"],
        },
        "created_at_utc": media.utc_timestamp(),
        "model_validation": "model_validation.json",
        "schema_version": SCHEMA_VERSION,
        "v1_package_root": v1_package_root.as_posix(),
    }
    write_json(package_root / "workspace_manifest.json", workspace_manifest)
    media_url_audit = validate_workspace_media_urls(package_root)
    write_json(package_root / "media_url_audit.json", media_url_audit)
    if not media_url_audit["passed"]:
        raise ValueError("workspace media URL validation failed")
    return {
        "day": day_lineage,
        "media_url_audit": media_url_audit,
        "model_validation": model_validation,
        "night": night_manifest,
        "raw_signal_count": len(raw_signals),
        "reviewer_task_count": len(tasks),
        "schema_version": SCHEMA_VERSION,
    }


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
    duration_seconds = (
        DAY_CANONICAL_SOURCE_FRAME_END - DAY_CANONICAL_SOURCE_FRAME_START
    ) / parent_fps
    canonical_source = package_root / "media" / "day" / "original_reference.mp4"
    if parent_file.exists() and not canonical_source.exists():
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
            "artifacts/face_full_video_acceptance_v1/day/reviewed_candidate_privacy_regions.jsonl",
            "artifacts/day_reviewer_discovered_correction_v1/privacy_regions.jsonl",
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
        plate_evidence_inputs=("artifacts/shared_temporal_evidence_v1/day/tracking/tracks.jsonl",),
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
    current_video = package_root / "media" / "day" / "current_combined_result.mp4"
    current_regions_path = package_root / "regions" / "day_current_regions.jsonl"

    records = build_mapped_day_region_records(
        face_regions_path=Path(
            "artifacts/face_full_video_acceptance_v1/day/reviewed_candidate_privacy_regions.jsonl"
        ),
        face_reviewer_correction_path=Path(
            "artifacts/day_reviewer_discovered_correction_v1/privacy_regions.jsonl"
        ),
        plate_tracks_path=Path("artifacts/shared_temporal_evidence_v1/day/tracking/tracks.jsonl"),
    )
    review_finalization.write_privacy_region_records(current_regions_path, records)
    render_summary = render_combined_current_result(
        canonical_source=canonical_source,
        output_video=current_video,
        records=records,
        overwrite=True,
    )
    write_json(
        technical_root / "canonical_day_generation.json",
        {
            "case_source": canonical_source.relative_to(package_root).as_posix(),
            "combined_current_video": current_video.relative_to(package_root).as_posix(),
            "face": {
                "lineage_parent_start_frame": 4615,
                "mapped_inputs": [
                    "artifacts/face_full_video_acceptance_v1/day/reviewed_candidate_privacy_regions.jsonl",
                    "artifacts/day_reviewer_discovered_correction_v1/privacy_regions.jsonl",
                ],
            },
            "generation_mode": "lineage_mapped_frozen_evidence",
            "model_validation": model_paths.to_dict(),
            "plate": {
                "lineage_parent_start_frame": 1828,
                "mapped_inputs": [
                    "artifacts/shared_temporal_evidence_v1/day/tracking/tracks.jsonl"
                ],
            },
            "render": render_summary,
            "schema_version": "canonical_day_generation.v1",
        },
    )
    return {
        "combined_current": current_video.relative_to(package_root).as_posix(),
        "combined_current_sha256": media.sha256_file(current_video),
        "current_region_count": len(records),
        "region_provenance": privacy_rendering.provenance_counts(records),
    }


def build_mapped_day_region_records(
    *,
    face_regions_path: Path,
    face_reviewer_correction_path: Path,
    plate_tracks_path: Path,
) -> list[PrivacyRegionRecord]:
    face_local_offset = 4615 - DAY_CANONICAL_SOURCE_FRAME_START
    records = face_region_records_from_render_input(
        face_regions_path,
        case_frame_offset=face_local_offset,
        parent_frame_offset=4615,
    )
    records.extend(
        face_region_records_from_render_input(
            face_reviewer_correction_path,
            case_frame_offset=face_local_offset,
            parent_frame_offset=4615,
        )
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
    corrections = review_finalization.latest_corrections(
        review_finalization.load_corrections_jsonl(session_dir / "corrections.jsonl"),
        case_id=correction.case_id,
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
    return {
        "audio_handling": audio_handling,
        "corrected_preview": corrected_preview.relative_to(package_root).as_posix(),
        "frame_end": correction.source_frame_end,
        "frame_start": correction.source_frame_start,
        "ok": True,
        "original_preview": original_preview.relative_to(package_root).as_posix(),
        "rendered_frames": rendered_frames,
        "rendered_regions": rendered_regions,
        "schema_version": "correction_interval_preview.v1",
    }


def write_entity_index(package_root: Path) -> None:
    entities_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id in ("day", "night"):
        region_path = package_root / "regions" / f"{case_id}_current_regions.jsonl"
        if not region_path.exists():
            entities_by_case[case_id] = []
            write_json(
                package_root / "entities" / f"{case_id}.json",
                {"case_id": case_id, "entities": [], "schema_version": "privacy_entities.v1"},
            )
            continue
        records = review_finalization.load_privacy_region_records(region_path)
        manifest = case_manifests.load_case_manifest(package_root / "cases" / f"{case_id}.json")
        entities = privacy_entities.entities_from_region_records(case_id, records)
        entity_payloads = [
            review_entity_payload(
                entity=entity,
                records=records,
                manifest=manifest,
                package_root=package_root,
            )
            for entity in entities
        ]
        payload = {
            "case_id": case_id,
            "entities": entity_payloads,
            "entity_count": len(entities),
            "schema_version": "review_entities.v2",
        }
        entities_by_case[case_id] = payload["entities"]
        write_json(package_root / "entities" / f"{case_id}.json", payload)
    write_json(
        package_root / "privacy_entities.json",
        {
            "cases": entities_by_case,
            "schema_version": "review_entities_index.v2",
        },
    )
    write_json(
        package_root / "review_objects.json",
        {
            "cases": entities_by_case,
            "schema_version": "review_object_registry.v2",
        },
    )
    write_json(
        package_root / "review_entities.json",
        {
            "cases": entities_by_case,
            "schema_version": "review_entity_registry.v2",
        },
    )


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
        "representative_case_frame": rep_case_frame,
        "representative_source_frame": rep_source_frame,
        "representative_thumbnail_path": thumbnail_path,
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
        pad = max(12, int(max(x2 - x1, y2 - y1) * 0.65))
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
    visible = [signal for signal in raw_signals if not signal["technical"]]
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
                <= max(item["source_frame_end"] for item in group) + 30
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
    if category == CATEGORY_PLATE and blocking:
        return (
            "Plate coverage needs correction",
            "The system could not safely maintain plate redaction through this interval.",
            "Compare Original and Current result. If the plate is visible, add a reviewed region before finalization.",
        )
    if category == CATEGORY_FACE:
        return (
            "Check face coverage",
            "The system flagged this face interval for human confirmation.",
            "Compare Original and Current result and confirm that the face remains protected without redacting the wrong person.",
        )
    return (
        "Check the redaction result",
        "The system selected this interval because coverage or identity is uncertain.",
        "Compare Original and Current result and record the clearest decision.",
    )


def write_workspace_assets(package_root: Path) -> None:
    (package_root / "index.html").write_text(index_html(), encoding="utf-8")
    (package_root / "assets" / "workspace.css").write_text(workspace_css(), encoding="utf-8")
    (package_root / "assets" / "workspace.js").write_text(workspace_js(), encoding="utf-8")


class FinalReviewWorkspaceServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], package_root: Path) -> None:
        self.package_root = package_root.resolve()
        super().__init__(server_address, FinalReviewWorkspaceHandler)


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
        super().do_GET()

    def _do_api_get(self) -> bool:
        parsed = urlparse(self.path)
        path = parsed.path
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
            self.send_json(
                {
                    "attribution": review_sessions.load_reviewer_attribution(session_dir),
                    "chain": review_sessions.validate_event_hash_chain(session_dir),
                    "corrections": load_jsonl_dicts(session_dir / "corrections.jsonl"),
                    "decisions": review_sessions.load_current_decisions(session_dir),
                    "events": review_sessions.load_events(session_dir),
                    "ok": True,
                    "session": read_json_file(session_dir / "review_session.json"),
                    "summary": read_json_file(session_dir / "review_summary.json"),
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
        path = urlparse(self.path).path
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
            self.send_json({"ok": True, "session": session.to_dict()})
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
        if path == "/api/corrections":
            from redactor.review_corrections import correction_from_dict

            correction = correction_from_dict(payload["correction"])
            event = review_sessions.append_correction_event(
                session_dir,
                actor=actor,
                correction=correction,
            )
            preview = render_correction_interval_preview(
                self.server.package_root,
                session_dir=session_dir,
                correction=correction,
            )
            self.send_json({"event": event, "ok": True, "preview": preview})
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
            self.send_json({"ok": result.get("finalized", False), "result": result})
            return
        self.send_json({"error": "unknown endpoint", "ok": False}, status=HTTPStatus.NOT_FOUND)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
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
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        size = file_path.stat().st_size
        start = int(match.group(1) or 0)
        end = int(match.group(2) or size - 1)
        end = min(end, size - 1)
        if start > end:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
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
    if item.get("issue_type") == "visible_plate_leak":
        return "unresolved_gap"
    if item.get("severity") == "low":
        return "optional_spot_check"
    if item.get("category") == "face":
        return "uncertain_face_coverage"
    return "uncertain_plate_track"


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
    if task.get("issue_type") == "corrected_result_check":
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
  <title>Final Review Workspace v2.2</title>
  <link rel="stylesheet" href="assets/workspace.css?v=2.2">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Final Review Workspace</h1>
      <p>Local case review for Original and Current result media. Reviewer attribution is required only when exporting or finalizing.</p>
    </div>
    <div class="header-actions">
      <button id="startSession" class="primary">Start review</button>
      <button id="changeAttribution">Change reviewer attribution</button>
      <button id="exportSession">Export session</button>
    </div>
  </header>
  <div id="sessionBanner" class="session-banner">No local review session yet. Start review to enable decisions and corrections.</div>
  <nav class="tabs" aria-label="Workspace tabs">
    <button data-tab="day">Day case</button>
    <button data-tab="night">Night case</button>
    <button data-tab="final">Final results</button>
    <button data-tab="audit">Audit trail</button>
  </nav>
  <main>
    <section id="case-day" class="case-panel" data-case-id="day"></section>
    <section id="case-night" class="case-panel" data-case-id="night"></section>
    <section id="case-final" class="case-panel">
      <h2>Final results</h2>
      <p id="finalStatus">Reviewed-final outputs appear here after a case passes all gates.</p>
      <div id="finalResults"></div>
    </section>
    <section id="case-audit" class="case-panel">
      <h2>Audit trail</h2>
      <p>Events are append-only and hash chained. Technical JSON is available on each entry.</p>
      <ol id="auditTrail"></ol>
    </section>
  </main>
  <section id="drawer" class="drawer" aria-hidden="true">
    <div class="drawer-card">
      <header>
        <h2 id="drawerTitle">Drawer</h2>
        <button id="closeDrawer">Close</button>
      </header>
      <div id="drawerBody"></div>
    </div>
  </section>
  <section id="attributionModal" class="modal" aria-hidden="true">
    <div class="modal-card">
      <h2>Reviewer attribution</h2>
      <p>This local prototype records a declared reviewer name. It is not enterprise authentication.</p>
      <label>Reviewer name <input id="reviewerName" autocomplete="name"></label>
      <p id="attributionError" class="status-red"></p>
      <div class="controls">
        <button id="submitAttribution" class="primary">Save attribution</button>
        <button id="cancelAttribution">Cancel</button>
      </div>
    </div>
  </section>
  <script src="assets/workspace.js?v=2.2"></script>
</body>
</html>
"""


def workspace_css() -> str:
    return """
:root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f3f5f7; color: #18202b; }
h1, h2, h3 { margin: 0 0 8px; letter-spacing: 0; }
p { line-height: 1.45; margin: 0 0 10px; }
button { border: 1px solid #9aa4b2; background: #fff; border-radius: 6px; padding: 8px 11px; cursor: pointer; font: inherit; }
button.primary { background: #0f766e; border-color: #0f766e; color: #fff; }
button.danger { background: #b91c1c; border-color: #b91c1c; color: #fff; }
button.selected { outline: 3px solid #2dd4bf; }
button:disabled { opacity: .55; cursor: not-allowed; }
input, textarea, select { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #aab3c0; border-radius: 6px; font: inherit; }
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 20px; padding: 14px 22px; background: #111827; color: #fff; }
.topbar p { color: #d1d5db; max-width: 80ch; }
.header-actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.session-banner { padding: 9px 22px; background: #fff7ed; border-bottom: 1px solid #fed7aa; color: #7c2d12; }
.session-banner.ready { background: #ecfdf5; border-bottom-color: #a7f3d0; color: #065f46; }
.tabs { display: flex; gap: 8px; padding: 10px 22px; background: #fff; border-bottom: 1px solid #d8dee7; position: sticky; top: 0; z-index: 20; }
.tabs button.active { background: #111827; color: #fff; }
main { padding: 18px 22px 52px; }
.case-panel { display: none; }
.case-panel.active { display: block; }
.case-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 12px; }
.case-summary { color: #4b5563; }
.case-tools { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.video-shell { background: #0b1220; color: #fff; padding: 12px; border-radius: 8px; }
.video-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.video-card { position: relative; min-width: 0; border: 1px solid #1f2937; border-radius: 8px; overflow: hidden; background: #050812; }
.video-card h3 { padding: 8px 10px; margin: 0; font-size: 15px; background: #111827; }
.video-card video { display: block; width: 100%; background: #000; aspect-ratio: 16 / 9; }
.frame-overlay { position: absolute; left: 8px; bottom: 8px; padding: 5px 7px; background: rgba(0,0,0,.76); color: #fff; border-radius: 4px; font-size: 13px; pointer-events: none; }
.media-state { position: absolute; right: 8px; bottom: 8px; padding: 5px 7px; background: rgba(17,24,39,.82); color: #fff; border-radius: 4px; font-size: 13px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; align-items: center; }
.status-green { color: #047857; font-weight: 700; }
.status-amber { color: #b45309; font-weight: 700; }
.status-red { color: #b91c1c; font-weight: 700; }
.panel, .sweep-panel, .technical, .blocker { background: #fff; border: 1px solid #d8dee7; border-radius: 8px; padding: 14px; margin-top: 14px; }
.timeline { height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 10px 0; }
.timeline div { height: 100%; background: #0f766e; width: 0%; }
.drawer { display: none; position: fixed; inset: 0; background: rgba(15, 23, 42, .34); z-index: 60; align-items: flex-end; }
.drawer.open { display: flex; }
.drawer-card { background: #fff; border-radius: 10px 10px 0 0; width: 100%; max-height: 76vh; overflow: hidden; box-shadow: 0 -20px 55px rgba(15,23,42,.25); }
.drawer-card > header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 18px; border-bottom: 1px solid #e5e7eb; }
#drawerBody { padding: 14px 18px 24px; overflow: auto; max-height: calc(76vh - 64px); }
.entity-toolbar, .task-toolbar { display: grid; grid-template-columns: minmax(220px, 2fr) minmax(130px, 1fr) minmax(130px, 1fr) auto; gap: 8px; align-items: end; margin-bottom: 12px; }
.entity-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.entity-card { border: 1px solid #d8dee7; border-radius: 8px; padding: 10px; display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 10px; align-items: start; background: #fff; }
.entity-card img { width: 96px; height: 72px; object-fit: cover; background: #111827; border-radius: 6px; cursor: pointer; }
.entity-card .placeholder-thumb { width: 96px; height: 72px; display: grid; place-items: center; background: #e5e7eb; border-radius: 6px; color: #4b5563; }
.entity-card.selected { outline: 3px solid #2dd4bf; }
.entity-card header { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.badge { border-radius: 999px; padding: 3px 8px; font-size: 12px; background: #e5e7eb; color: #374151; display: inline-block; }
.badge.redact { background: #dbeafe; color: #1e40af; }
.badge.keep_visible { background: #fef3c7; color: #92400e; }
.badge.review_required { background: #fee2e2; color: #991b1b; }
.entity-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.task-list { display: grid; gap: 10px; }
.task-card { border-left: 6px solid #f59e0b; background: #fff; border-top: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
.task-card.blocking { border-left-color: #b91c1c; }
.decision-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
.correction-canvas-wrap { position: relative; background: #111827; aspect-ratio: 16 / 9; margin-top: 10px; border-radius: 8px; overflow: hidden; }
.correction-canvas-wrap canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
.correction-tools { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
.correction-tools textarea, .correction-tools .wide { grid-column: 1 / -1; }
.keyframe-list { font-size: 13px; color: #3f4b5d; grid-column: 1 / -1; }
.modal { display: none; position: fixed; inset: 0; z-index: 80; background: rgba(15,23,42,.48); place-items: center; }
.modal.open { display: grid; }
.modal-card { background: #fff; border-radius: 8px; width: min(440px, calc(100vw - 32px)); padding: 18px; box-shadow: 0 25px 80px rgba(15,23,42,.35); }
details { margin-top: 12px; }
:fullscreen .frame-overlay, :fullscreen .media-state { font-size: 16px; }
@media (max-width: 900px) {
  .topbar, .case-header { flex-direction: column; align-items: stretch; }
  .video-grid, .correction-tools, .decision-grid, .entity-toolbar, .task-toolbar { grid-template-columns: 1fr; }
  .drawer-card { max-height: 86vh; }
  #drawerBody { max-height: calc(86vh - 64px); }
}
"""


def workspace_js() -> str:
    return r"""
const state = {
  activeTab: 'night',
  activeTask: {},
  attribution: null,
  correctionDrafts: {},
  decisions: {},
  drawer: null,
  drawing: null,
  entityFilters: {},
  entityHistory: [],
  entityPage: {},
  entitySelection: {},
  entityVisibility: {},
  frame: {},
  mediaReady: {},
  session: null,
  tasks: [],
  viewed: {}
};
const DATA = {};
const ENTITY_PAGE_SIZE = 24;
let pendingAttributionAction = null;

async function loadJson(path) { const res = await fetch(path); if (!res.ok) throw new Error(path); return await res.json(); }

async function init() {
  DATA.workspace = await loadJson('workspace_manifest.json');
  DATA.tasks = await loadJson('reviewer_tasks.json');
  DATA.freeze = await loadJson('configuration_freeze.json');
  DATA.entities = await loadJson('review_entities.json').catch(() => loadJson('privacy_entities.json'));
  for (const id of ['day','night']) DATA[id] = await loadJson(`cases/${id}.json`);
  state.tasks = DATA.tasks.tasks.filter(t => !t.technical_only);
  state.frame.day = 0; state.frame.night = 0;
  renderCase('day'); renderCase('night'); renderFinal(); bindGlobalControls();
  await restoreSession();
  setTab(location.hash.replace('#','') || 'night');
  window.addEventListener('hashchange', () => setTab(location.hash.replace('#','') || 'night'));
  document.addEventListener('keydown', handleKeys, true);
  updateSessionBanner();
}

function setTab(tab) {
  if (!['day','night','final','audit'].includes(tab)) tab = 'night';
  pauseInactiveCases(tab);
  state.activeTab = tab;
  document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.case-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById(`case-${tab}`); if (panel) panel.classList.add('active');
  if (location.hash.replace('#','') !== tab) location.hash = tab;
  updateControls();
  updateFrameOverlay(tab);
}

function renderCase(caseId) {
  const c = DATA[caseId]; const panel = document.getElementById(`case-${caseId}`);
  const tasks = state.tasks.filter(t => t.case_id === caseId);
  const roles = c.video_roles || [];
  panel.innerHTML = `<div class="case-header"><div><h2>${escapeHtml(c.display_name)}</h2>${caseStatus(c)}
    <p class="case-summary">${c.frame_count} case frames · source frames ${c.source_frame_start}-${c.source_frame_end}</p></div>
    <div class="case-tools">
      <button data-action="open-drawer" data-drawer="entities" data-case="${caseId}">Entities</button>
      <button data-action="open-drawer" data-drawer="tasks" data-case="${caseId}">Review queue</button>
      <button data-action="open-drawer" data-drawer="corrections" data-case="${caseId}">Corrections</button>
    </div></div>
    <div class="video-shell" id="player-${caseId}" data-player-wrapper="${caseId}">${videoGrid(c, roles)}${playerControls(caseId)}</div>
    <section class="sweep-panel"><h3>Full-video sweep</h3><p>Use the player to review the complete case. Coverage records where playback has been visited; it does not prove the absence of misses.</p>
      <div class="timeline"><div id="sweep-${caseId}"></div></div>
      <button data-action="add-issue" data-case="${caseId}">Add issue at current frame</button>
      <button data-action="ack-sweep" data-case="${caseId}">Acknowledge sweep complete</button>
      <textarea id="sweep-note-${caseId}" placeholder="Override reason or sweep notes"></textarea>
    </section>
    <details class="technical"><summary>Technical details</summary><pre>${escapeHtml(JSON.stringify(c.lineage || {}, null, 2))}</pre></details>`;
  state.activeTask[caseId] = state.activeTask[caseId] || 0;
  bindCaseMedia(panel);
  updateTask(caseId);
}

function caseStatus(c) {
  if (c.status === 'ready') return `<p class="status-amber">Current result is a review preview until all blockers are resolved.</p>`;
  return `<div class="blocker"><h3>Case blocked</h3><p>${(c.blockers || []).map(escapeHtml).join('<br>')}</p></div>`;
}

function videoGrid(c, roles) {
  if (!roles.length) return `<p>No compatible current-result media is available for this case yet.</p>`;
  return `<div class="video-grid">${roles.slice(0,2).map(role => `<div class="video-card" data-video-card="${c.case_id}:${role.role}"><h3>${escapeHtml(role.label)}</h3><video preload="metadata" data-role="${role.role}" data-case="${c.case_id}" data-media-path="${role.path}" src="${role.path}" playsinline disablepictureinpicture controls></video><div class="frame-overlay" data-frame-overlay="${c.case_id}">${frameHudText(c.case_id)}</div><div class="media-state"><span data-media-state="${c.case_id}:${role.role}">Loading</span> <button data-action="retry-media" data-case="${c.case_id}" data-role="${role.role}">Retry</button></div></div>`).join('')}</div>`;
}

function playerControls(caseId) {
  return `<div class="controls">
    <button data-action="play" data-case="${caseId}">Play together</button>
    <button data-action="restart" data-case="${caseId}">Restart</button>
    <button data-action="back" data-case="${caseId}">One frame back</button>
    <button data-action="forward" data-case="${caseId}">One frame forward</button>
    <label>Go to frame <input data-go-frame="${caseId}" type="number" min="0"></label>
    <button data-action="fullscreen-comparison" data-case="${caseId}">Fullscreen comparison</button>
    <button data-action="fullscreen-original" data-case="${caseId}">Fullscreen Original</button>
    <button data-action="fullscreen-current" data-case="${caseId}">Fullscreen Current/Final</button>
    <span id="mediaError-${caseId}" class="status-red"></span>
  </div>`;
}

function taskPanel(caseId, tasks) {
  if (!tasks.length) return `<section class="task-panel"><h3>No tasks</h3></section>`;
  return `<section class="panel"><h3>Guided risk review</h3><p id="progress-${caseId}"></p><div id="task-${caseId}"></div>
    <div class="controls"><button data-action="prev-task" data-case="${caseId}">Previous</button><button data-action="next-task" data-case="${caseId}">Next</button><button data-action="loop-task" data-case="${caseId}">Loop interval</button></div></section>`;
}

function correctionPanel(caseId) {
  return `<section class="panel"><h3>Temporal correction editor</h3><p>Draw and edit geometry on Original. The second pane remains the current or corrected preview; corrections are a reviewed override layer and never change automatic evidence.</p>
    <div class="correction-canvas-wrap"><canvas data-correction-canvas="${caseId}"></canvas></div>
    <div class="correction-tools">
      <label>Entity <select data-correction-entity="${caseId}">${entityOptions(caseId)}</select></label>
      <label>Operation <select data-correction-operation="${caseId}">
        <option value="add_privacy_region">Add privacy region</option>
        <option value="replace_privacy_region">Replace privacy region</option>
        <option value="suppress_region">Suppress existing region</option>
        <option value="suppress_track_segment">Suppress track segment</option>
        <option value="mark_not_sensitive">Mark region not sensitive</option>
        <option value="mark_not_visible">Mark interval not visible or occluded</option>
      </select></label>
      <label class="wide">Reason <textarea data-correction-reason="${caseId}"></textarea></label>
      <div class="keyframe-list" id="keyframes-${caseId}">No reviewed keyframes yet.</div>
      <div class="controls wide">
        <button data-action="set-interval-start" data-case="${caseId}">Set interval start</button>
        <button data-action="set-interval-end" data-case="${caseId}">Set interval end</button>
        <button data-action="add-keyframe" data-case="${caseId}">Create/update keyframe</button>
        <button data-action="prev-keyframe" data-case="${caseId}">Previous keyframe</button>
        <button data-action="next-keyframe" data-case="${caseId}">Next keyframe</button>
        <button data-action="preview-interpolation" data-case="${caseId}">Preview correction</button>
        <button data-action="delete-correction" data-case="${caseId}">Discard draft</button>
        <button data-action="save-correction" data-case="${caseId}" ${state.session ? '' : 'disabled'}>Save correction</button>
        <button data-action="verify-correction" data-case="${caseId}" ${state.session ? '' : 'disabled'}>Verify corrected interval</button>
      </div>
    </div></section>`;
}

function entityPanel(caseId) {
  const filters = state.entityFilters[caseId] || { query: '', category: 'all', policy: 'all' };
  const all = caseEntities(caseId).filter(e => e.current_status !== 'merged');
  const filtered = all.filter(e => {
    const q = filters.query.toLowerCase();
    const matchesQuery = !q || `${e.label} ${e.category}`.toLowerCase().includes(q);
    const matchesCategory = filters.category === 'all' || e.category === filters.category;
    const policy = e.default_privacy_policy || e.current_policy || 'redact';
    const matchesPolicy = filters.policy === 'all' || policy === filters.policy;
    return matchesQuery && matchesCategory && matchesPolicy;
  });
  const page = Math.max(0, state.entityPage[caseId] || 0);
  const start = page * ENTITY_PAGE_SIZE;
  const visible = filtered.slice(start, start + ENTITY_PAGE_SIZE);
  if (!all.length) return `<section class="panel"><h3>Privacy entities</h3><p>No current automatic entities are indexed for this case.</p></section>`;
  return `<section class="panel"><h3>Privacy entities</h3><p>Overlay visibility is editor-only. Redact, Keep, and Needs review are privacy policy choices and are audit events.</p>
    <div class="entity-toolbar">
      <label>Search <input data-entity-filter="query" data-case="${caseId}" value="${escapeHtml(filters.query)}"></label>
      <label>Category <select data-entity-filter="category" data-case="${caseId}">
        ${filterOption('all', 'All categories', filters.category)}
        ${filterOption('face', 'People', filters.category)}
        ${filterOption('license_plate', 'Plates', filters.category)}
      </select></label>
      <label>Policy <select data-entity-filter="policy" data-case="${caseId}">
        ${filterOption('all', 'All policies', filters.policy)}
        ${filterOption('redact', 'Redact', filters.policy)}
        ${filterOption('keep_visible', 'Keep', filters.policy)}
        ${filterOption('review_required', 'Needs review', filters.policy)}
      </select></label>
      <div class="controls"><button data-action="merge-selected-entities" data-case="${caseId}" ${state.session ? '' : 'disabled'}>Merge selected</button><button data-action="split-selected-entity" data-case="${caseId}" ${state.session ? '' : 'disabled'}>Split selected</button></div>
    </div>
    <p>${filtered.length} of ${all.length} entities shown. Page ${page + 1} of ${Math.max(1, Math.ceil(filtered.length / ENTITY_PAGE_SIZE))}.</p>
    <div class="entity-grid">${visible.map(entityCard).join('')}</div>
    <div class="controls"><button data-action="entity-page" data-page-delta="-1" data-case="${caseId}">Previous page</button><button data-action="entity-page" data-page-delta="1" data-case="${caseId}">Next page</button></div></section>`;
}

function entityCard(entity) {
  const selected = !!state.entitySelection[entity.entity_id];
  const policy = entity.default_privacy_policy || entity.current_policy || 'redact';
  const thumb = entity.representative_thumbnail_path ? `<img alt="${escapeHtml(entity.label)} representative thumbnail" src="${entity.representative_thumbnail_path}" data-action="jump-entity" data-case="${entity.case_id}" data-entity="${entity.entity_id}">` : `<div class="placeholder-thumb" data-action="jump-entity" data-case="${entity.case_id}" data-entity="${entity.entity_id}">No image</div>`;
  return `<article class="entity-card ${selected ? 'selected' : ''}" data-entity-card="${entity.entity_id}">
    ${thumb}<div><header><strong>${escapeHtml(entity.label)}</strong><span class="badge ${policy}">${policyLabel(policy)}</span></header>
    <p><small>${categoryLabel(entity.category)} · first seen ${formatEntityFrame(entity)}</small></p>
    <label><input type="checkbox" data-action="select-entity" data-case="${entity.case_id}" data-entity="${entity.entity_id}" ${selected ? 'checked' : ''}> Select</label>
    <div class="entity-actions">
      <button data-action="toggle-entity-overlay" data-case="${entity.case_id}" data-entity="${entity.entity_id}">${entityVisible(entity.entity_id) ? 'Hide overlay' : 'Show overlay'}</button>
      <button data-action="toggle-entity-lock" data-case="${entity.case_id}" data-entity="${entity.entity_id}">${entity.locked ? 'Unlock geometry' : 'Lock geometry'}</button>
    </div>
    <label>Privacy action <select data-action="entity-policy" data-case="${entity.case_id}" data-entity="${entity.entity_id}" ${state.session ? '' : 'disabled'}>
      ${policyOption('redact', 'Redact', policy)}
      ${policyOption('keep_visible', 'Keep', policy)}
      ${policyOption('review_required', 'Needs review', policy)}
    </select></label>
    <details><summary>More details</summary><pre>${escapeHtml(JSON.stringify(entity, null, 2))}</pre></details></div>
  </article>`;
}

function policyOption(value, label, selected) { return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`; }
function filterOption(value, label, selected) { return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`; }
function caseEntities(caseId) { return ((DATA.entities && DATA.entities.cases && DATA.entities.cases[caseId]) || []); }
function entityOptions(caseId) { return caseEntities(caseId).map(e => `<option value="${e.entity_id}">${e.label}</option>`).join('') || '<option value="">New entity from correction</option>'; }
function categoryLabel(category) { return category === 'face' ? 'Person' : 'Plate'; }
function entityVisible(entityId) { return state.entityVisibility[entityId] !== false; }
function policyLabel(policy) { return policy === 'keep_visible' ? 'Keep' : policy === 'review_required' ? 'Needs review' : 'Redact'; }
function formatEntityFrame(entity) { return entity.representative_source_frame === null || entity.representative_source_frame === undefined ? 'unknown' : `frame ${entity.representative_source_frame}`; }

function updateTask(caseId) {
  const tasks = state.tasks.filter(t => t.case_id === caseId && !t.technical_only);
  const idx = Math.min(state.activeTask[caseId] || 0, Math.max(0, tasks.length - 1));
  state.activeTask[caseId] = idx;
  const task = tasks[idx]; const box = document.getElementById(`task-${caseId}`); if (!box) return;
  document.getElementById(`progress-${caseId}`).textContent = progressText(caseId, tasks);
  if (!task) { box.innerHTML = '<p>No current task.</p>'; return; }
  box.innerHTML = `<article class="task-card ${task.blocking ? 'blocking' : ''}"><h3>Task ${idx + 1} of ${tasks.length}: ${task.title}</h3>
    <p><strong>Frames:</strong> ${task.source_frame_start}-${task.source_frame_end}</p>
    <p><strong>What happened:</strong> ${task.what_happened}</p><p><strong>What to inspect:</strong> ${task.what_to_inspect}</p>
    <div class="decision-grid">${Object.entries(DATA.tasks.actions.reduce((m,a)=>{m[a.action]=a.label;return m;},{})).map(([action,label]) => `<button data-action="decision" data-decision="${action}" data-case="${caseId}" data-task="${task.review_task_id}" ${state.session ? '' : 'disabled'}>${label}</button>`).join('')}</div>
    <p id="taskError-${caseId}" class="status-red"></p></article>`;
  markSelectedDecision(caseId, task.review_task_id);
  seekToFrame(caseId, task.source_frame_start, true);
}

function progressText(caseId, tasks) {
  const required = tasks.filter(t => t.blocking);
  const doneRequired = required.filter(t => state.decisions[t.review_task_id]).length;
  const doneAll = tasks.filter(t => state.decisions[t.review_task_id]).length;
  return `Required: ${doneRequired}/${required.length} · All visible tasks: ${doneAll}/${tasks.length}`;
}

function bindGlobalControls() {
  document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
  document.getElementById('startSession').addEventListener('click', startSession);
  document.getElementById('exportSession').addEventListener('click', exportSession);
  document.getElementById('changeAttribution').addEventListener('click', () => openAttributionModal('change'));
  document.getElementById('closeDrawer').addEventListener('click', closeDrawer);
  document.getElementById('cancelAttribution').addEventListener('click', closeAttributionModal);
  document.getElementById('submitAttribution').addEventListener('click', submitAttribution);
  document.addEventListener('click', handleClick);
  document.addEventListener('change', handleChange);
}

function handleClick(event) {
  const el = event.target.closest('[data-action]');
  if (!el) return;
  const caseId = el.dataset.case || state.activeTab;
  const action = el.dataset.action;
  if (action === 'open-drawer') return openDrawer(el.dataset.drawer, caseId);
  if (action === 'play') return playPauseTogether(caseId);
  if (action === 'restart') return restartMedia(caseId);
  if (action === 'back') return stepFrame(caseId, -1);
  if (action === 'forward') return stepFrame(caseId, 1);
  if (action === 'fullscreen-comparison') return fullscreen(document.getElementById(`player-${caseId}`));
  if (action === 'fullscreen-original') return fullscreen(activeVideos(caseId)[0]?.parentElement);
  if (action === 'fullscreen-current') return fullscreen(activeVideos(caseId)[1]?.parentElement);
  if (action === 'retry-media') return retryMedia(caseId, el.dataset.role);
  if (action === 'prev-task') return moveTask(caseId, -1);
  if (action === 'next-task') return moveTask(caseId, 1);
  if (action === 'loop-task') return loopTask(caseId);
  if (action === 'decision') return recordDecision(caseId, el.dataset.task, el.dataset.decision, el);
  if (action === 'ack-sweep') return acknowledgeSweep(caseId);
  if (action === 'add-issue') return addIssueAtFrame(caseId);
  if (action === 'toggle-entity-overlay') return toggleEntityOverlay(caseId, el.dataset.entity);
  if (action === 'toggle-entity-lock') return toggleEntityLock(caseId, el.dataset.entity);
  if (action === 'jump-entity') return jumpToEntity(caseId, el.dataset.entity);
  if (action === 'select-entity') return toggleEntitySelection(el.dataset.entity, el.checked);
  if (action === 'entity-page') return moveEntityPage(caseId, Number(el.dataset.pageDelta || 0));
  if (action === 'merge-selected-entities') return mergeSelectedEntities(caseId);
  if (action === 'split-selected-entity') return splitSelectedEntity(caseId);
  if (action === 'set-interval-start') return setDraftIntervalBoundary(caseId, 'start');
  if (action === 'set-interval-end') return setDraftIntervalBoundary(caseId, 'end');
  if (action === 'add-keyframe') return addKeyframe(caseId);
  if (action === 'prev-keyframe') return jumpKeyframe(caseId, -1);
  if (action === 'next-keyframe') return jumpKeyframe(caseId, 1);
  if (action === 'preview-interpolation') return previewInterpolation(caseId);
  if (action === 'delete-correction') return deleteCorrectionDraft(caseId);
  if (action === 'save-correction') return saveCorrection(caseId);
  if (action === 'verify-correction') return verifyCorrection(caseId);
}

function handleChange(event) {
  const el = event.target;
  if (el.matches('[data-go-frame]')) return seekToFrame(el.dataset.goFrame, Number(el.value), true);
  if (el.matches('[data-action="entity-policy"]')) return changeEntityPolicy(el.dataset.case, el.dataset.entity, el.value);
  if (el.matches('[data-entity-filter]')) {
    const caseId = el.dataset.case;
    state.entityFilters[caseId] = state.entityFilters[caseId] || { query: '', category: 'all', policy: 'all' };
    state.entityFilters[caseId][el.dataset.entityFilter] = el.value;
    state.entityPage[caseId] = 0;
    renderDrawer();
  }
}

function bindCaseMedia(panel) {
  panel.querySelectorAll('video').forEach(v => {
    v.disablePictureInPicture = true;
    v.setAttribute('disablePictureInPicture', '');
    v.addEventListener('loadedmetadata', () => setMediaState(v, 'Ready'));
    v.addEventListener('canplay', () => setMediaState(v, 'Ready'));
    v.addEventListener('error', () => setMediaState(v, `Playback error: ${v.dataset.mediaPath} code ${v.error ? v.error.code : 'unknown'}`));
    v.addEventListener('play', () => updateControls());
    v.addEventListener('pause', () => updateControls());
    v.addEventListener('timeupdate', () => { syncStateFromVideo(v); updateFrameOverlay(v.dataset.case); refreshCorrectionCanvas(v.dataset.case); });
    scheduleFrameCallback(v);
  });
}

function initCorrectionCanvas(caseId) {
  const canvas = document.querySelector(`[data-correction-canvas="${caseId}"]`);
  if (!canvas || canvas.dataset.bound === '1') return;
  canvas.dataset.bound = '1';
  const c = DATA[caseId];
  canvas.width = c.width || 1280; canvas.height = c.height || 720;
  canvas.addEventListener('pointerdown', e => {
    const p = canvasPoint(caseId, e);
    state.drawing = { caseId, start: p, current: p };
    canvas.setPointerCapture(e.pointerId);
    refreshCorrectionCanvas(caseId);
  });
  canvas.addEventListener('pointermove', e => {
    if (!state.drawing || state.drawing.caseId !== caseId) return;
    state.drawing.current = canvasPoint(caseId, e);
    refreshCorrectionCanvas(caseId);
  });
  canvas.addEventListener('pointerup', e => {
    if (!state.drawing || state.drawing.caseId !== caseId) return;
    state.drawing.current = canvasPoint(caseId, e);
    const box = normalizedDraftBox(caseId, state.drawing.start, state.drawing.current);
    if (box && box[2] - box[0] >= 4 && box[3] - box[1] >= 4) {
      const draft = ensureDraft(caseId);
      draft.bbox_xyxy = box;
      draft.source_frame = sourceFrame(caseId);
      state.correctionDrafts[caseId] = draft;
    }
    state.drawing = null;
    refreshCorrectionCanvas(caseId);
  });
  refreshCorrectionCanvas(caseId);
}

function canvasPoint(caseId, event) {
  const canvas = document.querySelector(`[data-correction-canvas="${caseId}"]`);
  const rect = canvas.getBoundingClientRect();
  const c = DATA[caseId];
  return {
    x: Math.round(((event.clientX - rect.left) / rect.width) * (c.width || canvas.width)),
    y: Math.round(((event.clientY - rect.top) / rect.height) * (c.height || canvas.height))
  };
}

function normalizedDraftBox(caseId, a, b) {
  const c = DATA[caseId]; const w = c.width || 1280; const h = c.height || 720;
  const x1 = Math.max(0, Math.min(w - 1, Math.min(a.x, b.x)));
  const y1 = Math.max(0, Math.min(h - 1, Math.min(a.y, b.y)));
  const x2 = Math.max(0, Math.min(w - 1, Math.max(a.x, b.x)));
  const y2 = Math.max(0, Math.min(h - 1, Math.max(a.y, b.y)));
  return [x1, y1, x2, y2];
}

function refreshCorrectionCanvas(caseId) {
  const canvas = document.querySelector(`[data-correction-canvas="${caseId}"]`);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const c = DATA[caseId];
  canvas.width = c.width || canvas.width || 1280; canvas.height = c.height || canvas.height || 720;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const original = document.querySelector(`#player-${caseId} video[data-role="original"]`) || activeVideos(caseId)[0];
  if (original && original.readyState >= 2) {
    try { ctx.drawImage(original, 0, 0, canvas.width, canvas.height); }
    catch (e) { ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, canvas.width, canvas.height); }
  } else {
    ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 640));
  ctx.strokeStyle = '#facc15'; ctx.fillStyle = 'rgba(250, 204, 21, 0.16)';
  const draft = state.correctionDrafts[caseId];
  if (draft) drawBox(ctx, draft.bbox_xyxy);
  if (draft && draft.keyframes) {
    ctx.strokeStyle = '#22c55e';
    ctx.fillStyle = 'rgba(34, 197, 94, 0.12)';
    draft.keyframes.forEach(k => drawBox(ctx, k.bbox_xyxy));
  }
  if (state.drawing && state.drawing.caseId === caseId) {
    const box = normalizedDraftBox(caseId, state.drawing.start, state.drawing.current);
    if (box) drawBox(ctx, box);
  }
  ctx.fillStyle = 'rgba(0, 0, 0, .70)'; ctx.fillRect(8, 8, 260, 30);
  ctx.fillStyle = '#fff'; ctx.font = '18px system-ui';
  ctx.fillText(`Source frame ${sourceFrame(caseId)}`, 18, 30);
  refreshKeyframeList(caseId);
}

function drawBox(ctx, box) {
  const [x1, y1, x2, y2] = box;
  ctx.fillRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
  ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
}

function activeVideos(caseId) { return [...document.querySelectorAll(`#player-${caseId} video`)].filter(v => v.offsetParent !== null); }
function fpsValue(c) { const [a,b] = String(c.fps || '30000/1001').split('/').map(Number); return b ? a / b : Number(c.fps || 29.97); }
function currentFrame(caseId) {
  const v = activeVideos(caseId)[0];
  if (v && !Number.isNaN(v.currentTime)) return clampCaseFrame(caseId, Math.round(v.currentTime * fpsValue(DATA[caseId])));
  return clampCaseFrame(caseId, state.frame[caseId] || 0);
}
function sourceFrame(caseId) { return (DATA[caseId].source_frame_start || 0) + currentFrame(caseId); }
function clampCaseFrame(caseId, frame) { return Math.max(0, Math.min(Math.max(0, (DATA[caseId].frame_count || 1) - 1), Number(frame) || 0)); }
function frameHudText(caseId) {
  const c = DATA[caseId]; const local = clampCaseFrame(caseId, state.frame[caseId] || 0); const src = (c.source_frame_start || 0) + local;
  const t = local / fpsValue(c);
  return `Source frame ${src} · Case frame ${local} / ${Math.max(0, c.frame_count - 1)} · ${formatTime(t)}`;
}

async function playPauseTogether(caseId) {
  const videos = activeVideos(caseId); const err = document.getElementById(`mediaError-${caseId}`); err.textContent = '';
  if (!videos.length) { err.textContent = 'No playable media for this task.'; return; }
  const anyPlaying = videos.some(v => !v.paused && !v.ended);
  if (anyPlaying) { videos.forEach(v => v.pause()); updateControls(); return; }
  const ready = await Promise.allSettled(videos.map(v => waitMetadata(v, 2500)));
  const failedIndex = ready.findIndex(r => r.status === 'rejected');
  if (failedIndex >= 0) {
    err.textContent = `Playback is not ready for ${videos[failedIndex].dataset.mediaPath}.`;
    return;
  }
  const t = currentFrame(caseId) / fpsValue(DATA[caseId]);
  videos.forEach(v => { v.muted = true; v.currentTime = t; });
  const results = await Promise.allSettled(videos.map(v => v.play()));
  const failedIndexPlay = results.findIndex(r => r.status === 'rejected');
  if (failedIndexPlay >= 0) err.textContent = `Playback failed for ${videos[failedIndexPlay].dataset.mediaPath}.`;
  updateControls();
}

async function restartMedia(caseId) {
  await seekToCaseFrame(caseId, 0, true);
  updateControls();
  clearMediaError(caseId);
}
async function stepFrame(caseId, delta) {
  if (!DATA[caseId]) return;
  const videos = activeVideos(caseId);
  videos.forEach(v => v.pause());
  await Promise.allSettled(videos.map(v => waitMetadata(v, 500)));
  const base = clampCaseFrame(caseId, state.frame[caseId] ?? currentFrame(caseId));
  return seekToCaseFrame(caseId, base + delta, true);
}
async function seekToFrame(caseId, sourceFrameValue, pause) {
  const c = DATA[caseId]; const local = Number(sourceFrameValue) - (c.source_frame_start || 0);
  await seekToCaseFrame(caseId, local, pause);
}
async function seekToCaseFrame(caseId, frameValue, pause) {
  const c = DATA[caseId]; const local = clampCaseFrame(caseId, frameValue);
  state.frame[caseId] = local;
  const videos = activeVideos(caseId);
  if (pause) videos.forEach(v => v.pause());
  const t = local / fpsValue(c);
  const ready = await Promise.allSettled(videos.map(v => waitMetadata(v, 1200)));
  const err = document.getElementById(`mediaError-${caseId}`);
  const failedIndex = ready.findIndex(r => r.status === 'rejected');
  if (failedIndex >= 0 && err) err.textContent = `Seek failed because ${videos[failedIndex].dataset.mediaPath} is not ready.`;
  await Promise.allSettled(videos.map(v => seekVideo(v, t)));
  state.frame[caseId] = local;
  updateFrameOverlay(caseId, local);
  setTimeout(() => refreshCorrectionCanvas(caseId), 80);
}
function updateFrameOverlay(caseId, exactFrame) {
  if (!DATA[caseId]) return;
  const c = DATA[caseId]; const local = exactFrame === undefined ? currentFrame(caseId) : clampCaseFrame(caseId, exactFrame); const src = (c.source_frame_start || 0) + local;
  state.frame[caseId] = local;
  const t = local / fpsValue(c); const text = `Source frame ${src} · Case frame ${local} / ${Math.max(0, c.frame_count - 1)} · ${formatTime(t)}`;
  document.querySelectorAll(`[data-frame-overlay="${caseId}"]`).forEach(o => o.textContent = text);
  state.viewed[caseId] = Math.max(state.viewed[caseId] || 0, local); const bar = document.getElementById(`sweep-${caseId}`);
  if (bar && c.frame_count) bar.style.width = `${Math.min(100, 100 * (state.viewed[caseId] || 0) / c.frame_count)}%`;
}
function formatTime(t) { const m = Math.floor(t / 60); const s = (t % 60).toFixed(3).padStart(6,'0'); return `${String(m).padStart(2,'0')}:${s}`; }
function updateControls() { document.querySelectorAll('[data-action="play"]').forEach(b => { const vids = activeVideos(b.dataset.case); b.textContent = vids.some(v => !v.paused && !v.ended) ? 'Pause together' : 'Play together'; }); }
function fullscreen(el) { if (el && el.requestFullscreen) el.requestFullscreen(); }
function moveTask(caseId, delta) { state.activeTask[caseId] = Math.max(0, (state.activeTask[caseId] || 0) + delta); updateTask(caseId); }
function loopTask(caseId) { const t = visibleCaseTasks(caseId)[state.activeTask[caseId] || 0]; if (t) seekToFrame(caseId, t.source_frame_start, true); }
function visibleCaseTasks(caseId) { return state.tasks.filter(t => t.case_id === caseId && !t.technical_only); }

async function startSession() {
  const banner = document.getElementById('sessionBanner');
  const freezeHash = await digestJson(DATA.freeze);
  const manifests = {}; const hashes = {};
  for (const id of ['day','night']) { manifests[id] = await digestJson(DATA[id]); if (DATA[id].source_video_hash) hashes[id] = DATA[id].source_video_hash; }
  try {
    const res = await postJson('/api/sessions', { reviewer: {}, configuration_freeze_hash: freezeHash, case_manifest_hashes: manifests, source_hashes: hashes });
    state.session = res.session;
    state.decisions = {};
    localStorage.setItem(storageKey('session_id'), state.session.session_id);
    await loadSessionState(state.session.session_id);
    addAudit(`Session started: ${state.session.session_id}.`);
  } catch (e) {
    banner.textContent = `Could not start local review session: ${e.message}`;
    banner.classList.remove('ready');
  }
  updateSessionBanner();
  rerenderActiveDrawer();
}

async function recordDecision(caseId, taskId, action, el) {
  if (!requireSession()) return;
  const error = document.getElementById(`taskError-${caseId}`);
  if (error) error.textContent = '';
  try {
    const response = await postJson('/api/decisions', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, task_id: taskId, action, note: '' });
    state.decisions[taskId] = { action, case_id: caseId, task_id: taskId, event_hash: response.event.event_hash, reviewed_at_utc: response.event.timestamp_utc };
    el.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('selected'));
    el.classList.add('selected');
    addAudit(`Decision saved for ${taskId}: ${actionLabel(action)}.`);
    updateTask(caseId);
    setTimeout(() => moveTask(caseId, 1), 350);
  } catch (e) {
    if (error) error.textContent = `Decision was not saved: ${e.message}`;
  }
}

async function acknowledgeSweep(caseId) {
  if (!requireSession()) return;
  const note = value(`sweep-note-${caseId}`);
  await postJson('/api/events/sweep', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, viewed_ranges: [{ start: 0, end: state.viewed[caseId] || 0 }], override_reason: note });
  addAudit(`Sweep acknowledged for ${caseId}.`);
}
function addIssueAtFrame(caseId) {
  if (!requireSession()) return;
  const frame = sourceFrame(caseId);
  const categoryChoice = (window.prompt('Category for this issue: Face or Plate?', 'Plate') || '').trim().toLowerCase();
  const category = categoryChoice.startsWith('f') ? 'face' : (categoryChoice.startsWith('p') ? 'license_plate' : 'both');
  const issueChoice = (window.prompt('Issue type: Privacy leak, Partial coverage, False redaction, or Wrong target?', 'Privacy leak') || 'Privacy leak').trim().toLowerCase();
  const issueType = issueChoice.includes('partial') ? 'partial_coverage' : issueChoice.includes('false') ? 'false_redaction' : issueChoice.includes('wrong') ? 'wrong_target' : 'privacy_leak';
  const task = {
    allowed_actions: DATA.tasks.actions.map(a => a.action),
    blocking: true,
    case_id: caseId,
    category,
    correction_required: true,
    issue_type: `reviewer_added_${issueType}`,
    raw_signal_ids: [],
    review_task_id: `${caseId}_manual_issue_${Date.now()}`,
    severity: 'blocker',
    source_frame_end: frame,
    source_frame_start: frame,
    technical_only: false,
    title: 'Reviewer-added privacy issue',
    what_happened: 'A reviewer found a possible privacy issue during full-video sweep.',
    what_to_inspect: 'Define the affected interval, add reviewed regions or suppressions, and verify the corrected preview.'
  };
  state.tasks.push(task);
  state.activeTask[caseId] = visibleCaseTasks(caseId).length - 1;
  if (category === 'face' || category === 'license_plate') ensureManualEntity(caseId, category);
  renderCase(caseId);
  addEntityEvent(caseId, '', 'reviewer_discovered_issue_created', { task });
  addAudit(`New blocking issue started at source frame ${frame} for ${caseId}.`);
}
function ensureManualEntity(caseId, category) {
  const entities = caseEntities(caseId);
  const prefix = category === 'face' ? 'Person' : 'Plate';
  const next = 1 + Math.max(0, ...entities.filter(e => e.category === category).map(e => {
    const parts = String(e.label || '').split(' ');
    return parts[0] === prefix && Number.isFinite(Number(parts[1])) ? Number(parts[1]) : 0;
  }));
  const entity = {
    automatic_source_track_ids: [],
    case_id: caseId,
    category,
    created_provenance: { source: 'reviewer_created', source_frame: sourceFrame(caseId) },
    current_status: 'active',
    default_privacy_policy: 'review_required',
    entity_id: `entity_${crypto.randomUUID()}`,
    interval_policy_overrides: [],
    label: `${prefix} ${String(next).padStart(2, '0')}`,
    locked: false,
    object_uuid: '',
    reviewed_region_track_ids: [],
    schema_version: 'privacy_entity.v1'
  };
  entity.object_uuid = entity.entity_id.replace(/^entity_/, '');
  DATA.entities.cases[caseId].push(entity);
  addEntityEvent(caseId, entity.entity_id, 'entity_created_by_reviewer', { after: entity });
  return entity;
}
function ensureObjectPolicyVerificationTask(caseId, entity, title, note) {
  const taskId = `object_policy_verify_${caseId}_${entity.entity_id}`;
  let task = state.tasks.find(t => t.review_task_id === taskId);
  if (!task) {
    task = {
      allowed_actions: ['looks_correct', 'privacy_leak', 'false_redaction', 'wrong_target', 'uncertain_escalate'],
      blocking: true,
      case_id: caseId,
      category: entity.category,
      correction_required: false,
      issue_type: 'object_policy_verification',
      raw_signal_ids: [],
      review_task_id: taskId,
      severity: 'blocker',
      source_frame_end: DATA[caseId].source_frame_end,
      source_frame_start: DATA[caseId].source_frame_start,
      technical_only: false,
      title,
      what_happened: `${entity.label} privacy action changed. ${note || ''}`.trim(),
      what_to_inspect: 'Verify that this object policy is intentional before finalization.'
    };
    state.tasks.push(task);
  }
  return task;
}
function toggleEntityOverlay(caseId, entityId) {
  state.entityVisibility[entityId] = !entityVisible(entityId);
  addEntityEvent(caseId, entityId, 'entity_overlay_visibility_changed', { visible: state.entityVisibility[entityId] });
  renderDrawer();
}
function changeEntityPolicy(caseId, entityId, policy) {
  if (!requireSession()) return;
  const entity = caseEntities(caseId).find(e => e.entity_id === entityId);
  if (!entity) return;
  const previousPolicy = entity.default_privacy_policy;
  let note = '';
  if (policy === 'keep_visible') {
    note = (window.prompt('A note is required before keeping this object visible.', '') || '').trim();
    if (!note) { renderCase(caseId); addAudit('Keep action cancelled because no note was provided.'); return; }
  }
  const before = {...entity};
  entity.default_privacy_policy = policy;
  entity.privacy_action = policy;
  let verificationTask = null;
  if (policy === 'keep_visible') verificationTask = ensureObjectPolicyVerificationTask(caseId, entity, 'Keep policy verification', note);
  if (policy === 'review_required') verificationTask = ensureObjectPolicyVerificationTask(caseId, entity, 'Object needs review', note || 'Object policy set to Needs review.');
  if (verificationTask) entity.verification_task_id = verificationTask.review_task_id;
  addEntityEvent(caseId, entityId, 'entity_policy_changed', { before, after: entity });
  if (policy === 'keep_visible') addAudit(`${entity.label} policy set to Keep. Verification is required before finalization.`);
  else if (policy === 'review_required') addAudit(`${entity.label} policy set to Needs review. Finalization remains blocked.`);
  else addAudit(`${entity.label} policy changed from ${previousPolicy.replace('_', ' ')} to Redact.`);
  renderDrawer();
}
function toggleEntityLock(caseId, entityId) {
  const entity = caseEntities(caseId).find(e => e.entity_id === entityId);
  if (!entity) return;
  const before = {...entity};
  entity.locked = !entity.locked;
  addEntityEvent(caseId, entityId, entity.locked ? 'entity_locked' : 'entity_unlocked', { before, after: entity });
  renderDrawer();
}
function addEntityEvent(caseId, entityId, eventType, payload) {
  const event = {
    actor: currentActor(),
    case_id: caseId,
    entity_id: entityId,
    event_id: `browser_entity_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    event_type: eventType,
    payload,
    timestamp_utc: new Date().toISOString()
  };
  state.entityHistory.push(event);
  if (state.session) {
    postJson('/api/entity-events', { session_id: state.session.session_id, actor: currentActor(), event }).catch(e => addAudit(`Entity persistence warning: ${e.message}`));
  }
}
function ensureDraft(caseId) { return state.correctionDrafts[caseId] || { keyframes: [], checkpoints: [] }; }
function setDraftIntervalBoundary(caseId, which) {
  const draft = ensureDraft(caseId);
  if (which === 'start') draft.interval_start = sourceFrame(caseId);
  else draft.interval_end = sourceFrame(caseId);
  if (draft.interval_start !== undefined && draft.interval_end !== undefined && draft.interval_end < draft.interval_start) {
    const tmp = draft.interval_start; draft.interval_start = draft.interval_end; draft.interval_end = tmp;
  }
  state.correctionDrafts[caseId] = draft;
  refreshKeyframeList(caseId);
  addAudit(`Correction interval ${which} set to source frame ${sourceFrame(caseId)}.`);
}
function addKeyframe(caseId) {
  const draft = ensureDraft(caseId);
  if (!draft.bbox_xyxy) { addAudit('Draw a rectangle before creating a keyframe.'); return; }
  const frame = draft.source_frame || sourceFrame(caseId);
  draft.keyframes = (draft.keyframes || []).filter(k => k.source_frame !== frame);
  draft.keyframes.push({ source_frame: frame, bbox_xyxy: draft.bbox_xyxy });
  draft.keyframes.sort((a, b) => a.source_frame - b.source_frame);
  draft.checkpoints = requiredCheckpointsForDraft(caseId, draft).map(f => ({ source_frame: f, state: 'unreviewed', suggested: true }));
  state.correctionDrafts[caseId] = draft;
  refreshCorrectionCanvas(caseId);
  addAudit(`Keyframe saved at source frame ${frame}.`);
}
function jumpKeyframe(caseId, direction) {
  const draft = state.correctionDrafts[caseId];
  if (!draft || !draft.keyframes || !draft.keyframes.length) return;
  const current = sourceFrame(caseId);
  const sorted = draft.keyframes.map(k => k.source_frame).sort((a, b) => a - b);
  const target = direction < 0 ? [...sorted].reverse().find(f => f < current) : sorted.find(f => f > current);
  if (target !== undefined) seekToFrame(caseId, target, true);
}
function previewInterpolation(caseId) {
  const draft = state.correctionDrafts[caseId];
  if (!draft || !draft.keyframes || draft.keyframes.length < 2) {
    addAudit('Add at least two keyframes to preview interpolation.');
    return;
  }
  addAudit(`Interpolation preview ready across ${draft.keyframes.length} keyframes and ${requiredCheckpointsForDraft(caseId, draft).length} checkpoints.`);
}
function requiredCheckpointsForDraft(caseId, draft) {
  const keyframes = (draft.keyframes || []).map(k => k.source_frame).sort((a, b) => a - b);
  const start = draft.interval_start ?? keyframes[0];
  const end = draft.interval_end ?? keyframes[keyframes.length - 1];
  if (start === undefined || end === undefined) return [];
  const fps = fpsValue(DATA[caseId]);
  const seconds = Math.max(0, (end - start) / fps);
  const frames = new Set([start, end]);
  if (seconds > 0.5 && seconds <= 2 && end > start) frames.add(Math.floor((start + end) / 2));
  if (seconds > 2) {
    const step = Math.max(1, Math.round(fps));
    for (let f = start + step; f < end; f += step) frames.add(f);
  }
  return [...frames].sort((a, b) => a - b);
}
function refreshKeyframeList(caseId) {
  const el = document.getElementById(`keyframes-${caseId}`);
  if (!el) return;
  const draft = state.correctionDrafts[caseId];
  if (!draft || !draft.keyframes || !draft.keyframes.length) {
    const interval = draft && (draft.interval_start !== undefined || draft.interval_end !== undefined) ? `Interval: ${draft.interval_start ?? '?'}-${draft.interval_end ?? '?'}. ` : '';
    el.textContent = `${interval}No reviewed keyframes yet.`;
    return;
  }
  const pages = reviewPagesForDraft(caseId, draft);
  const pageText = pages.length > 1 ? ` · review pages: ${pages.map(p => `${p[0]}-${p[1]}`).join('; ')}` : '';
  el.textContent = `Interval: ${draft.interval_start ?? draft.keyframes[0].source_frame}-${draft.interval_end ?? draft.keyframes[draft.keyframes.length - 1].source_frame} · Keyframes: ${draft.keyframes.map(k => k.source_frame).join(', ')} · suggested checkpoints: ${requiredCheckpointsForDraft(caseId, draft).join(', ')}${pageText}`;
}
function reviewPagesForDraft(caseId, draft) {
  const keyframes = (draft.keyframes || []).map(k => k.source_frame).sort((a,b)=>a-b);
  const start = draft.interval_start ?? keyframes[0];
  const end = draft.interval_end ?? keyframes[keyframes.length - 1];
  if (start === undefined || end === undefined) return [];
  const maxFrames = Math.max(1, Math.round(5 * fpsValue(DATA[caseId])));
  const pages = [];
  for (let frame = start; frame <= end; frame += maxFrames) pages.push([frame, Math.min(end, frame + maxFrames - 1)]);
  return pages;
}
function deleteCorrectionDraft(caseId) {
  if (!window.confirm('Discard the current correction draft?')) return;
  delete state.correctionDrafts[caseId]; refreshCorrectionCanvas(caseId); addAudit(`Correction draft cleared for ${caseId}.`);
}
function ensureCorrectionVerificationTask(caseId, correction) {
  const taskId = `verify_${correction.correction_id}`;
  if (!state.tasks.find(t => t.review_task_id === taskId)) {
    state.tasks.push({
      allowed_actions: ['looks_correct', 'partial_coverage', 'privacy_leak', 'wrong_target', 'uncertain_escalate'],
      blocking: true,
      case_id: caseId,
      category: correction.category,
      correction_required: false,
      issue_type: 'correction_verification',
      raw_signal_ids: [],
      review_task_id: taskId,
      severity: 'blocker',
      source_frame_end: correction.source_frame_end,
      source_frame_start: correction.source_frame_start,
      technical_only: false,
      title: 'Verify corrected interval',
      what_happened: 'A reviewed correction was saved for this interval.',
      what_to_inspect: 'Compare Original and Corrected preview. Choose Looks correct only if the corrected interval is acceptable.'
    });
  }
  return taskId;
}
async function verifyCorrection(caseId) {
  if (!requireSession()) return;
  const stored = JSON.parse(localStorage.getItem(storageKey('corrections')) || '[]').filter(c => c.case_id === caseId);
  if (!stored.length) { addAudit('No saved correction is available to verify.'); return; }
  const correction = stored[stored.length - 1];
  const actor = currentActor();
  const revised = {
    ...correction,
    checkpoint_verifications: (correction.checkpoint_verifications || []).map(checkpoint => ({ ...checkpoint, state: 'verified' })),
    previous_revision_id: correction.revision_id,
    revision_id: `corr_rev_${crypto.randomUUID()}`
  };
  const allCorrections = JSON.parse(localStorage.getItem(storageKey('corrections')) || '[]');
  allCorrections.push(revised);
  localStorage.setItem(storageKey('corrections'), JSON.stringify(allCorrections));
  const response = await postJson('/api/corrections', { session_id: state.session.session_id, actor, correction: revised, reason: 'Corrected interval verified' }).catch(e => { addAudit(`Correction verification persistence warning: ${e.message}`); return null; });
  if (response && response.preview && response.preview.ok) addAudit(`Corrected preview refreshed: ${response.preview.corrected_preview}`);
  const taskId = ensureCorrectionVerificationTask(caseId, correction);
  const tasks = visibleCaseTasks(caseId);
  state.activeTask[caseId] = Math.max(0, tasks.findIndex(t => t.review_task_id === taskId));
  updateTask(caseId);
  addAudit('Corrected interval checkpoints marked verified. Select Looks correct on the verification task after review.');
}
async function saveCorrection(caseId) {
  if (!requireSession()) return;
  const tasks = visibleCaseTasks(caseId);
  const task = tasks[state.activeTask[caseId] || 0];
  const operation = document.querySelector(`[data-correction-operation="${caseId}"]`)?.value || 'add_privacy_region';
  const entityId = document.querySelector(`[data-correction-entity="${caseId}"]`)?.value || null;
  const reason = document.querySelector(`[data-correction-reason="${caseId}"]`)?.value.trim() || 'Reviewed correction';
  const draft = state.correctionDrafts[caseId];
  if (!task) { addAudit('No active task is selected for the correction.'); return; }
  if (['add_privacy_region','replace_privacy_region'].includes(operation) && !draft) {
    addAudit('Draw a reviewed region on the Original frame before saving this correction.');
    return;
  }
  const entity = entityId ? caseEntities(caseId).find(e => e.entity_id === entityId) : null;
  if (task.category === 'both' && !entity) {
    addAudit('Choose a Human or Plate entity before saving a new correction for a combined task.');
    return;
  }
  const correctionCategory = task.category === 'both' && entity ? entity.category : task.category;
  const actor = currentActor();
  const keyframes = draft && draft.keyframes && draft.keyframes.length ? draft.keyframes : (draft ? [{ source_frame: draft.source_frame, bbox_xyxy: draft.bbox_xyxy }] : []);
  const frameStart = draft && draft.interval_start !== undefined ? draft.interval_start : (keyframes.length ? Math.min(...keyframes.map(k => k.source_frame)) : task.source_frame_start);
  const frameEnd = draft && draft.interval_end !== undefined ? draft.interval_end : (keyframes.length ? Math.max(...keyframes.map(k => k.source_frame)) : task.source_frame_end);
  const checkpointFrames = keyframes.length ? requiredCheckpointsForDraft(caseId, { ...draft, keyframes, interval_start: frameStart, interval_end: frameEnd }) : [];
  const checkpoints = checkpointFrames.map(frame => ({ source_frame: frame, state: keyframes.some(k => k.source_frame === frame) ? 'verified' : 'unreviewed' }));
  const correction = {
    application_version: 'final_review_workspace.v2.2',
    case_id: caseId,
    category: correctionCategory,
    checkpoint_verifications: checkpoints,
    configuration_freeze_hash: (state.session && state.session.configuration_freeze_hash) || await digestJson(DATA.freeze),
    correction_id: `corr_${crypto.randomUUID()}`,
    revision_id: `corr_rev_${crypto.randomUUID()}`,
    entity_id: entityId,
    input_region_ids: [],
    interpolation_method: 'linear',
    keyframe_boxes: keyframes,
    operation,
    reason,
    reviewer_identity: actor,
    schema_version: 'review_correction.v1',
    source_frame_end: frameEnd,
    source_frame_start: frameStart,
    source_video_hash: DATA[caseId].source_video_hash || '',
    task_id: task.review_task_id
  };
  const stored = JSON.parse(localStorage.getItem(storageKey('corrections')) || '[]');
  stored.push(correction); localStorage.setItem(storageKey('corrections'), JSON.stringify(stored));
  const response = await postJson('/api/corrections', { session_id: state.session.session_id, actor, correction, reason }).catch(e => { addAudit(`Correction persistence warning: ${e.message}`); return null; });
  if (response && response.preview && response.preview.ok) addAudit(`Corrected preview rendered: ${response.preview.corrected_preview}`);
  ensureCorrectionVerificationTask(caseId, correction);
  addAudit(`Reviewed correction saved for ${caseId}, task ${task.review_task_id}. A correction verification task remains required before finalization.`);
  delete state.correctionDrafts[caseId]; refreshCorrectionCanvas(caseId);
}
function renderFinal() { document.getElementById('finalResults').innerHTML = '<p>No reviewed-final case has passed all gates yet.</p>'; }
function addAudit(text) { const li = document.createElement('li'); li.textContent = `${new Date().toISOString()} · ${text}`; document.getElementById('auditTrail').prepend(li); }
function value(id) { return document.getElementById(id).value.trim(); }
function storageKey(name) { return `final-review-v2.2:${location.pathname}:${name}`; }
async function exportSession() {
  if (!requireSession()) return;
  if (!state.attribution || !state.attribution.display_name) {
    pendingAttributionAction = exportSession;
    openAttributionModal('export');
    return;
  }
  await loadSessionState(state.session.session_id);
  const payload = { session: state.session, attribution: state.attribution, decisions: state.decisions, corrections: JSON.parse(localStorage.getItem(storageKey('corrections')) || '[]'), entity_events: state.entityHistory };
  const blob = new Blob([JSON.stringify(payload,null,2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${state.session.session_id}_review_export.json`; a.click(); URL.revokeObjectURL(a.href);
}
async function postJson(path, payload) { const res = await fetch(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) }); if (!res.ok) throw new Error(await res.text()); return await res.json(); }
async function digestJson(obj) { const text = JSON.stringify(obj, Object.keys(obj).sort()); const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)); return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2,'0')).join(''); }
function escapeHtml(text) { return String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function handleKeys(e) {
  if (['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName) || document.activeElement.isContentEditable || document.activeElement.dataset.correctionCanvas) return;
  const caseId = ['day','night'].includes(state.activeTab) ? state.activeTab : null;
  if (e.key === 'Escape') {
    if (document.getElementById('drawer').classList.contains('open')) { closeDrawer(); return; }
    if (document.getElementById('attributionModal').classList.contains('open')) { closeAttributionModal(); return; }
    if (document.fullscreenElement) document.exitFullscreen();
  }
  if (e.code === 'Space' && caseId) { e.preventDefault(); e.stopPropagation(); playPauseTogether(caseId); return; }
  if (e.key === 'ArrowRight' && caseId) { e.preventDefault(); e.stopPropagation(); stepFrame(caseId, e.shiftKey ? 10 : 1); return; }
  if (e.key === 'ArrowLeft' && caseId) { e.preventDefault(); e.stopPropagation(); stepFrame(caseId, e.shiftKey ? -10 : -1); return; }
  if (e.key.toLowerCase() === 'f') {
    e.preventDefault();
    if (document.fullscreenElement) document.exitFullscreen();
    else if (caseId) fullscreen(document.getElementById(`player-${caseId}`));
    return;
  }
  if (e.key === ']' && caseId) { e.preventDefault(); e.stopPropagation(); moveTask(caseId, 1); return; }
  if (e.key === '[' && caseId) { e.preventDefault(); e.stopPropagation(); moveTask(caseId, -1); return; }
  if (e.key.toLowerCase() === 'k' && caseId) { e.preventDefault(); e.stopPropagation(); addKeyframe(caseId); return; }
  if (e.key.toLowerCase() === 'i' && caseId) { e.preventDefault(); e.stopPropagation(); setDraftIntervalBoundary(caseId, 'start'); return; }
  if (e.key.toLowerCase() === 'o' && caseId) { e.preventDefault(); e.stopPropagation(); setDraftIntervalBoundary(caseId, 'end'); return; }
  if (e.key === 'Delete' && caseId) { e.preventDefault(); e.stopPropagation(); deleteCorrectionDraft(caseId); }
}

function setMediaState(video, text) {
  const key = `${video.dataset.case}:${video.dataset.role}`;
  state.mediaReady[key] = text === 'Ready';
  const el = document.querySelector(`[data-media-state="${key}"]`);
  if (el) {
    el.textContent = text;
    el.className = text.startsWith('Playback error') ? 'status-red' : '';
  }
  if (text.startsWith('Playback error')) {
    const err = document.getElementById(`mediaError-${video.dataset.case}`);
    if (err) err.textContent = text;
  }
}
function retryMedia(caseId, role) {
  const video = document.querySelector(`#player-${caseId} video[data-role="${role}"]`);
  if (!video) return;
  setMediaState(video, 'Loading');
  video.load();
}
function clearMediaError(caseId) { const err = document.getElementById(`mediaError-${caseId}`); if (err) err.textContent = ''; }
function waitMetadata(video, timeoutMs) {
  if (video.readyState >= 1) return Promise.resolve();
  video.load();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => cleanup(reject, new Error('metadata timeout')), timeoutMs);
    const cleanup = (done, value) => {
      clearTimeout(timeout);
      video.removeEventListener('loadedmetadata', onReady);
      video.removeEventListener('error', onError);
      done(value);
    };
    const onReady = () => cleanup(resolve);
    const onError = () => cleanup(reject, new Error(`media error ${video.error ? video.error.code : 'unknown'}`));
    video.addEventListener('loadedmetadata', onReady, { once: true });
    video.addEventListener('error', onError, { once: true });
  });
}
function seekVideo(video, time) {
  return new Promise(resolve => {
    const done = () => { video.removeEventListener('seeked', done); resolve(); };
    video.addEventListener('seeked', done, { once: true });
    try { video.currentTime = Math.max(0, time); } catch (e) { resolve(); }
    setTimeout(done, 900);
  });
}
function syncStateFromVideo(video) {
  const caseId = video.dataset.case;
  state.frame[caseId] = clampCaseFrame(caseId, Math.round(video.currentTime * fpsValue(DATA[caseId])));
}
function scheduleFrameCallback(video) {
  if (!video.requestVideoFrameCallback) return;
  const loop = (_now, meta) => {
    const caseId = video.dataset.case;
    if (video.offsetParent !== null && !video.paused) {
      state.frame[caseId] = clampCaseFrame(caseId, Math.round(meta.mediaTime * fpsValue(DATA[caseId])));
      updateFrameOverlay(caseId);
    }
    video.requestVideoFrameCallback(loop);
  };
  video.requestVideoFrameCallback(loop);
}
function pauseInactiveCases(activeTab) {
  for (const caseId of ['day','night']) {
    if (caseId !== activeTab) activeVideos(caseId).forEach(v => v.pause());
  }
}

async function restoreSession() {
  const sessionId = localStorage.getItem(storageKey('session_id'));
  if (!sessionId) return;
  try {
    await loadSessionState(sessionId);
    addAudit(`Restored local session ${sessionId}.`);
  } catch (e) {
    localStorage.removeItem(storageKey('session_id'));
    addAudit(`Stored session could not be restored: ${e.message}.`);
  }
}
async function loadSessionState(sessionId) {
  const res = await fetch(`/api/session-state?session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(await res.text());
  const payload = await res.json();
  state.session = payload.session;
  state.attribution = payload.attribution;
  state.decisions = payload.decisions || {};
  renderAuditEvents(payload.events || []);
  updateSessionBanner();
}
function updateSessionBanner() {
  const banner = document.getElementById('sessionBanner');
  const startButton = document.getElementById('startSession');
  const attributionButton = document.getElementById('changeAttribution');
  const exportButton = document.getElementById('exportSession');
  if (!state.session) {
    banner.classList.remove('ready');
    banner.textContent = 'No local review session yet. Start review to enable decisions and corrections.';
    if (startButton) { startButton.disabled = false; startButton.textContent = 'Start review'; }
    if (attributionButton) attributionButton.disabled = true;
    if (exportButton) exportButton.disabled = true;
    return;
  }
  banner.classList.add('ready');
  const name = state.attribution && state.attribution.display_name ? ` · reviewer ${state.attribution.display_name}` : ' · anonymous until export/finalization';
  banner.textContent = `Session ${state.session.session_id} is active${name}.`;
  if (startButton) { startButton.disabled = true; startButton.textContent = 'Review active'; }
  if (attributionButton) attributionButton.disabled = false;
  if (exportButton) exportButton.disabled = false;
}
function requireSession() {
  if (state.session) return true;
  const banner = document.getElementById('sessionBanner');
  banner.classList.remove('ready');
  banner.textContent = 'Start review before saving decisions, entity changes, or corrections.';
  return false;
}
function currentActor() {
  if (state.attribution && state.attribution.display_name) return { display_name: state.attribution.display_name };
  return { display_name: null, reviewer_id: null };
}
function openAttributionModal(action) {
  if (!requireSession()) return;
  pendingAttributionAction = action === 'change' ? null : pendingAttributionAction;
  document.getElementById('attributionError').textContent = '';
  document.getElementById('attributionModal').classList.add('open');
  document.getElementById('attributionModal').setAttribute('aria-hidden', 'false');
  document.getElementById('reviewerName').focus();
}
function closeAttributionModal() {
  document.getElementById('attributionModal').classList.remove('open');
  document.getElementById('attributionModal').setAttribute('aria-hidden', 'true');
}
async function submitAttribution() {
  const name = value('reviewerName');
  const error = document.getElementById('attributionError');
  if (!name) { error.textContent = 'Reviewer name is required.'; return; }
  try {
    const response = await postJson('/api/reviewer-attribution', { session_id: state.session.session_id, actor: currentActor(), display_name: name });
    state.attribution = response.attribution;
    closeAttributionModal();
    updateSessionBanner();
    addAudit(`Reviewer attribution set to ${name}.`);
    if (pendingAttributionAction) { const next = pendingAttributionAction; pendingAttributionAction = null; await next(); }
  } catch (e) {
    error.textContent = `Attribution was not saved: ${e.message}`;
  }
}
function renderAuditEvents(events) {
  const list = document.getElementById('auditTrail');
  list.innerHTML = '';
  for (const event of [...events].reverse()) {
    const li = document.createElement('li');
    const name = event.actor && event.actor.display_name ? event.actor.display_name : 'Anonymous reviewer';
    li.innerHTML = `${escapeHtml(event.timestamp_utc)} · ${escapeHtml(name)} · ${escapeHtml(event.event_type)}${event.case_id ? ` · ${escapeHtml(event.case_id)}` : ''}<details><summary>Technical JSON</summary><pre>${escapeHtml(JSON.stringify(event, null, 2))}</pre></details>`;
    list.appendChild(li);
  }
}

function openDrawer(type, caseId) {
  state.drawer = { type, caseId };
  const drawer = document.getElementById('drawer');
  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  renderDrawer();
}
function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer').setAttribute('aria-hidden', 'true');
}
function rerenderActiveDrawer() { if (state.drawer) renderDrawer(); }
function renderDrawer() {
  if (!state.drawer) return;
  const { type, caseId } = state.drawer;
  const title = type === 'entities' ? 'Entities' : type === 'tasks' ? 'Review queue' : type === 'corrections' ? 'Corrections' : 'Technical details';
  document.getElementById('drawerTitle').textContent = `${DATA[caseId].display_name}: ${title}`;
  const body = document.getElementById('drawerBody');
  if (type === 'entities') body.innerHTML = entityPanel(caseId);
  else if (type === 'tasks') body.innerHTML = taskPanel(caseId, visibleCaseTasks(caseId));
  else if (type === 'corrections') body.innerHTML = correctionPanel(caseId);
  else body.innerHTML = `<pre>${escapeHtml(JSON.stringify(DATA[caseId], null, 2))}</pre>`;
  updateTask(caseId);
  initCorrectionCanvas(caseId);
  refreshCorrectionCanvas(caseId);
}
function moveEntityPage(caseId, delta) {
  state.entityPage[caseId] = Math.max(0, (state.entityPage[caseId] || 0) + delta);
  renderDrawer();
}
function toggleEntitySelection(entityId, checked) {
  state.entitySelection[entityId] = checked;
}
function jumpToEntity(caseId, entityId) {
  const entity = caseEntities(caseId).find(e => e.entity_id === entityId);
  if (!entity) return;
  if (entity.representative_source_frame !== null && entity.representative_source_frame !== undefined) seekToFrame(caseId, entity.representative_source_frame, true);
  document.querySelectorAll('.entity-card').forEach(card => card.classList.remove('selected'));
  const card = document.querySelector(`[data-entity-card="${entityId}"]`);
  if (card) card.classList.add('selected');
}
async function mergeSelectedEntities(caseId) {
  if (!requireSession()) return;
  const entities = selectedEntities(caseId);
  if (entities.length < 2) { addAudit('Select at least two same-category entities to merge.'); return; }
  if (!entities.every(e => e.category === entities[0].category)) { addAudit('Only same-category entities can be merged.'); return; }
  if (entities.some((a, i) => entities.slice(i + 1).some(b => entitiesOverlap(a, b)))) {
    addAudit('Merge rejected because selected entities are simultaneously visible or ambiguous.');
    return;
  }
  const reason = (window.prompt('Reason for merging these entities?', '') || '').trim();
  if (!reason) return;
  const primary = entities[0];
  const before = entities.map(e => ({ ...e }));
  const mergedTrackIds = [...new Set(entities.flatMap(e => e.member_detector_tracker_ids || e.automatic_source_track_ids || []))].sort();
  primary.member_detector_tracker_ids = mergedTrackIds;
  primary.automatic_source_track_ids = mergedTrackIds;
  primary.merge_split_lineage = [...(primary.merge_split_lineage || []), { operation: 'merge', merged_entity_ids: entities.slice(1).map(e => e.entity_id) }];
  for (const entity of entities.slice(1)) entity.current_status = 'merged';
  const event = await postJson('/api/entities/merge', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, before_state: before, after_state: { primary, retired_entity_ids: entities.slice(1).map(e => e.entity_id) }, reason });
  addAudit(`Merged entities into ${primary.label}. Event ${event.event.event_id}.`);
  state.entitySelection = {};
  renderDrawer();
}
async function splitSelectedEntity(caseId) {
  if (!requireSession()) return;
  const entities = selectedEntities(caseId);
  if (entities.length !== 1) { addAudit('Select exactly one entity to split.'); return; }
  const source = entities[0];
  const reason = (window.prompt('Reason for splitting this entity?', '') || '').trim();
  if (!reason) return;
  const trackText = (window.prompt('Track IDs to split into a new entity, comma separated. Leave blank for a bounded interval split marker.', '') || '').trim();
  const moved = trackText ? trackText.split(',').map(x => x.trim()).filter(Boolean) : [];
  const before = { ...source };
  const categoryEntities = caseEntities(caseId).filter(e => e.category === source.category);
  const nextNumber = 1 + Math.max(0, ...categoryEntities.map(e => Number(String(e.label).split(' ')[1]) || 0));
  const prefix = source.category === 'face' ? 'Person' : 'Plate';
  const newEntity = { ...source, entity_id: `entity_${crypto.randomUUID()}`, object_uuid: '', entity_uuid: '', label: `${prefix} ${String(nextNumber).padStart(2, '0')}`, member_detector_tracker_ids: moved, automatic_source_track_ids: moved, merge_split_lineage: [{ operation: 'split', parent_entity_id: source.entity_id }] };
  newEntity.object_uuid = newEntity.entity_id.replace(/^entity_/, '');
  newEntity.entity_uuid = newEntity.object_uuid;
  DATA.entities.cases[caseId].push(newEntity);
  source.merge_split_lineage = [...(source.merge_split_lineage || []), { operation: 'split', child_entity_id: newEntity.entity_id, moved_track_ids: moved }];
  await postJson('/api/entities/split', { session_id: state.session.session_id, actor: currentActor(), case_id: caseId, before_state: before, after_state: { source, new_entity: newEntity }, reason });
  addAudit(`Split ${source.label}; created ${newEntity.label}.`);
  state.entitySelection = {};
  renderDrawer();
}
function selectedEntities(caseId) { return caseEntities(caseId).filter(e => state.entitySelection[e.entity_id]); }
function entitiesOverlap(a, b) {
  const a1 = a.first_visible_source_frame, a2 = a.last_visible_source_frame;
  const b1 = b.first_visible_source_frame, b2 = b.last_visible_source_frame;
  if ([a1,a2,b1,b2].some(v => v === null || v === undefined)) return false;
  return Math.max(a1, b1) <= Math.min(a2, b2);
}

function markSelectedDecision(caseId, taskId) {
  const decision = state.decisions[taskId];
  document.querySelectorAll(`#task-${caseId} [data-action="decision"]`).forEach(button => {
    button.classList.toggle('selected', !!decision && button.dataset.decision === decision.action);
  });
}
function actionLabel(action) {
  const row = (DATA.tasks.actions || []).find(item => item.action === action);
  return row ? row.label : action;
}
init().catch(err => { document.body.innerHTML = `<pre>${err.stack || err}</pre>`; });
"""
