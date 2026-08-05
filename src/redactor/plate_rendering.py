from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import (
    frame_spans,
    media,
    plate_detection,
    plate_tracking,
    privacy_rendering,
    rendering,
)
from redactor.detection import rational_to_float
from redactor.privacy_records import APPROVAL_ACCEPTED, CATEGORY_PLATE, PrivacyRegionRecord

SCHEMA_VERSION = "plate_rendering.v1"
REVIEW_SCHEMA_VERSION = "plate_review_bundle.v1"
RENDERER_ID = "plate_redaction_renderer_v1"
PIPELINE_ID = "plate_pipeline_v1"

MODE_PREVIEW = rendering.MODE_PREVIEW
MODE_REVIEWED_FINAL = "reviewed_final"
METHOD_PIXELATE = rendering.METHOD_PIXELATE

ELIGIBLE_STATUSES = (plate_tracking.ACCEPTED_TRACK,)
ELIGIBLE_PROVENANCES = (
    plate_tracking.PROVENANCE_DETECTOR,
    plate_tracking.PROVENANCE_INTERPOLATION,
)
BLOCKING_REVIEW_REASONS = {
    "gap_exceeds_safe_interpolation_limit",
    "anchor_center_motion_too_large",
    "anchor_scale_change_too_large",
    "scene_cut_inside_gap",
}

DEFAULT_MARGIN_FRACTION = 0.15
DEFAULT_MARGIN_PIXELS = 0
DEFAULT_PIXEL_SIZE = 16
DEFAULT_BLUR_KERNEL = 51
DEFAULT_FRAME_CONTEXT = 15
DEFAULT_SPOT_CHECK_COUNT = 3
SHORT_ACCEPTED_TRACK_REVIEW_MAX_FRAMES = 12

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PlateRenderConfig:
    method: str = METHOD_PIXELATE
    margin_fraction: float = DEFAULT_MARGIN_FRACTION
    margin_pixels: int = DEFAULT_MARGIN_PIXELS
    pixel_size: int = DEFAULT_PIXEL_SIZE
    blur_kernel: int = DEFAULT_BLUR_KERNEL
    video_codec: str = "libx264"
    pixel_format: str = "yuv420p"
    movflags: str = "+faststart"
    detector_id: str = plate_detection.DETECTOR_ID
    detector_configuration_hash: str = plate_detection.OIMPlateDetectorConfig().configuration_hash()
    tracking_configuration_hash: str = plate_tracking.PlateTrackingConfig().configuration_hash()
    eligible_statuses: tuple[str, ...] = ELIGIBLE_STATUSES
    eligible_provenances: tuple[str, ...] = ELIGIBLE_PROVENANCES

    def __post_init__(self) -> None:
        if self.method != METHOD_PIXELATE:
            raise ValueError("plate renderer v1 supports only pixelate")
        if self.margin_fraction < 0:
            raise ValueError("margin_fraction cannot be negative")
        if self.margin_pixels < 0:
            raise ValueError("margin_pixels cannot be negative")
        if self.pixel_size < 2:
            raise ValueError("pixel_size must be at least 2")
        if self.detector_id != plate_detection.DETECTOR_ID:
            raise ValueError(f"plate renderer supports only {plate_detection.DETECTOR_ID}")
        if (
            self.detector_configuration_hash
            != plate_detection.OIMPlateDetectorConfig().configuration_hash()
        ):
            raise ValueError("plate renderer must use the selected OIM detector hash")
        if (
            self.tracking_configuration_hash
            != plate_tracking.PlateTrackingConfig().configuration_hash()
        ):
            raise ValueError("plate renderer must use the selected plate tracking hash")
        if self.eligible_statuses != ELIGIBLE_STATUSES:
            raise ValueError("automatic plate rendering may use only accepted tracks")
        if self.eligible_provenances != ELIGIBLE_PROVENANCES:
            raise ValueError(
                "automatic plate rendering may use only observations and interpolation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_kernel": self.blur_kernel,
            "detector_configuration_hash": self.detector_configuration_hash,
            "detector_id": self.detector_id,
            "eligible_provenances": list(self.eligible_provenances),
            "eligible_statuses": list(self.eligible_statuses),
            "margin_fraction": self.margin_fraction,
            "margin_pixels": self.margin_pixels,
            "method": self.method,
            "movflags": self.movflags,
            "pixel_format": self.pixel_format,
            "pixel_size": self.pixel_size,
            "renderer_id": RENDERER_ID,
            "tracking_configuration_hash": self.tracking_configuration_hash,
            "video_codec": self.video_codec,
        }

    def configuration_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render_settings(self) -> rendering.RenderSettings:
        return rendering.RenderSettings(
            mode=rendering.MODE_PREVIEW,
            method=self.method,
            margin_pixels=self.margin_pixels,
            margin_fraction=self.margin_fraction,
            blur_kernel=self.blur_kernel,
            pixel_size=self.pixel_size,
            preview_watermark=False,
        )


@dataclass(frozen=True)
class PlateRenderRegion:
    frame_index: int
    bbox_xyxy: Box
    provenance: str
    status: str
    track_uid: str
    track_id: int
    score: float | None
    source_record: dict[str, Any]

    def to_render_region(self) -> rendering.RenderRegion:
        return rendering.RenderRegion(
            frame_index=self.frame_index,
            bbox_xyxy=self.bbox_xyxy,
            provenance=self.provenance,
            state="covered",
            source_record=self.source_record,
        )

    def to_privacy_region(self) -> PrivacyRegionRecord:
        return PrivacyRegionRecord(
            category=CATEGORY_PLATE,
            source_frame=self.frame_index,
            bbox_xyxy=self.bbox_xyxy,
            provenance=self.provenance,
            approval_state=APPROVAL_ACCEPTED,
            render_eligible=True,
            object_id=self.track_uid,
            source_record={
                **self.source_record,
                "category": CATEGORY_PLATE,
                "track_id": self.track_id,
                "track_uid": self.track_uid,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "frame_index": self.frame_index,
            "provenance": self.provenance,
            "score": None if self.score is None else round(self.score, 6),
            "status": self.status,
            "track_id": self.track_id,
            "track_uid": self.track_uid,
        }


@dataclass(frozen=True)
class PlateReviewTask:
    review_task_id: str
    object_category: str
    frame_start: int
    frame_end: int
    severity: str
    blocking: bool
    source_stage: str
    reason_codes: tuple[str, ...]
    title: str
    explanation: str
    allowed_actions: tuple[str, ...]
    default_action_statuses: dict[str, str]
    track_uids: tuple[str, ...]
    media_roles: dict[str, str]
    contact_sheet: str | None = None
    technical_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_actions": list(self.allowed_actions),
            "blocking": self.blocking,
            "contact_sheet": self.contact_sheet,
            "default_action_statuses": dict(self.default_action_statuses),
            "explanation": self.explanation,
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "media_roles": self.media_roles,
            "object_category": self.object_category,
            "reason_codes": list(self.reason_codes),
            "review_status": "pending",
            "review_task_id": self.review_task_id,
            "schema_version": REVIEW_SCHEMA_VERSION,
            "severity": self.severity,
            "source_stage": self.source_stage,
            "technical_only": self.technical_only,
            "title": self.title,
            "track_uids": list(self.track_uids),
        }


def default_config() -> PlateRenderConfig:
    return PlateRenderConfig()


def run_plate_render(
    source_video: Path,
    *,
    tracks_path: Path,
    review_intervals_path: Path,
    output_dir: Path,
    mode: str = MODE_PREVIEW,
    config: PlateRenderConfig | None = None,
    output_video_path: Path | None = None,
    allow_partial_preview: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if mode not in {MODE_PREVIEW, MODE_REVIEWED_FINAL}:
        raise ValueError("plate render mode must be preview or reviewed_final")
    config = config or default_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")

    source_info = media.probe_media(source_video)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    source_frame_count = _optional_int(source_info.video_stream.frame_count)
    track_rows = load_track_rows(tracks_path)
    track_frame_span_report = frame_spans.build_frame_span_report(
        track_available_source_frame_indices(track_rows),
        source_frame_count=source_frame_count,
    )
    review_intervals = load_review_intervals(review_intervals_path)
    eligible_regions = eligible_regions_from_rows(track_rows, config=config)
    regions_by_frame = regions_by_frame_for_rendering(eligible_regions)
    plan_validation = validate_plate_render_plan(
        eligible_regions,
        review_intervals,
        mode=mode,
        frame_width=width,
        frame_height=height,
        frame_span_report=track_frame_span_report,
        allow_partial_preview=allow_partial_preview,
    )
    output_video = output_video_path or output_dir / "automatic_preview.mp4"
    if not plan_validation["render_allowed"]:
        payload = render_payloads(
            source_video=source_video,
            tracks_path=tracks_path,
            review_intervals_path=review_intervals_path,
            source_info=source_info,
            output_video=output_video,
            config=config,
            mode=mode,
            validation=plan_validation,
            frame_span_report=track_frame_span_report,
            media_validation=None,
            region_validation=None,
            rendered_frame_count=0,
            rendered_region_count=0,
            audio_handling="not_rendered_validation_failed",
            elapsed_seconds=0.0,
        )
        write_render_payloads(output_dir, payload)
        raise ValueError("plate render blocked: " + ", ".join(plan_validation["reasons"]))

    log_path = output_dir / "processing_log.txt"
    started = time.perf_counter()
    rendered_frame_count, rendered_region_count, audio_handling = render_plate_video(
        source_video,
        output_video,
        regions_by_frame,
        config=config,
        fps=fps,
        width=width,
        height=height,
        source_has_audio=source_info.audio_stream is not None,
        overwrite=overwrite,
        log_path=log_path,
        partial_processing_label=plan_validation["partial_processing"],
    )
    elapsed = time.perf_counter() - started
    media_validation = rendering.validate_output_media(
        source_video,
        source_info,
        output_video,
        expected_frame_count=rendered_frame_count,
        expected_width=width,
        expected_height=height,
        expected_fps=fps,
        source_has_audio=source_info.audio_stream is not None,
        settings=config.render_settings(),
    )
    region_validation = verify_rendered_regions(
        source_video,
        output_video,
        eligible_regions,
        config=config,
        frame_width=width,
        frame_height=height,
    )
    payload = render_payloads(
        source_video=source_video,
        tracks_path=tracks_path,
        review_intervals_path=review_intervals_path,
        source_info=source_info,
        output_video=output_video,
        config=config,
        mode=mode,
        validation=plan_validation,
        frame_span_report=track_frame_span_report,
        media_validation=media_validation,
        region_validation=region_validation,
        rendered_frame_count=rendered_frame_count,
        rendered_region_count=rendered_region_count,
        audio_handling=audio_handling,
        elapsed_seconds=elapsed,
    )
    write_render_payloads(output_dir, payload)
    return payload["summary"]


def load_track_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def track_available_source_frame_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [
        int(row["frame_index"]) for row in rows if bool(row.get("detector_frame_available", True))
    ]


def load_review_intervals(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return [dict(item) for item in payload.get("review_intervals", [])]


def eligible_regions_from_rows(
    rows: list[dict[str, Any]],
    *,
    config: PlateRenderConfig,
) -> list[PlateRenderRegion]:
    regions: list[PlateRenderRegion] = []
    for row in rows:
        frame_index = int(row["frame_index"])
        for track in row.get("tracks") or []:
            status = str(track.get("status"))
            provenance = str(track.get("provenance"))
            if status not in config.eligible_statuses:
                continue
            if provenance not in config.eligible_provenances:
                continue
            bbox = track.get("bbox_xyxy")
            if bbox is None:
                continue
            regions.append(
                PlateRenderRegion(
                    frame_index=frame_index,
                    bbox_xyxy=tuple(int(value) for value in bbox),
                    provenance=provenance,
                    status=status,
                    track_uid=str(track.get("track_uid") or ""),
                    track_id=int(track.get("track_id") or 0),
                    score=_optional_float(track.get("score")),
                    source_record=dict(track),
                )
            )
    return sorted(regions, key=lambda item: (item.frame_index, item.track_uid, item.bbox_xyxy))


def regions_by_frame_for_rendering(
    regions: list[PlateRenderRegion],
) -> dict[int, list[rendering.RenderRegion]]:
    by_frame: dict[int, list[rendering.RenderRegion]] = {}
    for region in regions:
        by_frame.setdefault(region.frame_index, []).append(region.to_render_region())
    return by_frame


def validate_plate_render_plan(
    eligible_regions: list[PlateRenderRegion],
    review_intervals: list[dict[str, Any]],
    *,
    mode: str,
    frame_width: int,
    frame_height: int,
    frame_span_report: dict[str, Any] | None = None,
    processed_frame_count: int | None = None,
    source_frame_count: int | None = None,
    allow_partial_preview: bool = False,
) -> dict[str, Any]:
    if frame_span_report is None:
        if processed_frame_count is None:
            raise ValueError("processed_frame_count is required without frame_span_report")
        frame_span_report = frame_spans.build_frame_span_report(
            range(int(processed_frame_count)),
            source_frame_count=source_frame_count,
        )
    invalid_regions: list[dict[str, Any]] = []
    for region in eligible_regions:
        if rendering.clipped_box(region.bbox_xyxy, frame_width, frame_height) is None:
            invalid_regions.append(region.to_dict())

    blocking_gap_intervals = [
        interval
        for interval in review_intervals
        if bool(interval.get("blocking")) or interval.get("severity") == "blocking"
    ]
    review_required_intervals = list(review_intervals)
    partial_processing = not bool(frame_span_report["full_source_processed"])
    unprocessed_source_frames = int(frame_span_report["missing_source_frame_count"])
    reasons: list[str] = []
    if review_required_intervals:
        reasons.append("plate_review_intervals")
    if invalid_regions:
        reasons.append("invalid_render_regions")
    if partial_processing:
        if mode == MODE_REVIEWED_FINAL:
            reasons.append("partial_plate_processing")
        elif not allow_partial_preview:
            reasons.append("partial_plate_processing_requires_explicit_preview")

    render_allowed = mode == MODE_PREVIEW and (not partial_processing or allow_partial_preview)
    if mode == MODE_REVIEWED_FINAL:
        render_allowed = not reasons

    return {
        "allow_partial_preview": allow_partial_preview,
        "approved_for_final": mode == MODE_REVIEWED_FINAL and not reasons,
        "blocking_gap_interval_count": len(blocking_gap_intervals),
        "blocking_gap_intervals": interval_summaries(blocking_gap_intervals),
        "eligible_region_count": len(eligible_regions),
        "frame_span_report": frame_span_report,
        "full_source_processed": frame_span_report["full_source_processed"],
        "invalid_region_count": len(invalid_regions),
        "invalid_regions": invalid_regions,
        "mode": mode,
        "partial_processing": partial_processing,
        "preview_non_final": mode == MODE_PREVIEW,
        "processed_frame_count": frame_span_report["processed_frame_count"],
        "reasons": reasons,
        "render_allowed": render_allowed,
        "review_required_interval_count": len(review_required_intervals),
        "review_required_intervals": interval_summaries(review_required_intervals),
        "source_frame_count": frame_span_report["source_frame_count"],
        "unprocessed_source_frames": unprocessed_source_frames,
    }


def render_plate_video(
    source_video: Path,
    output_video: Path,
    regions_by_frame: dict[int, list[rendering.RenderRegion]],
    *,
    config: PlateRenderConfig,
    fps: float,
    width: int,
    height: int,
    source_has_audio: bool,
    overwrite: bool,
    log_path: Path,
    partial_processing_label: bool = False,
) -> tuple[int, int, str]:
    return privacy_rendering.render_privacy_video(
        source_video,
        output_video,
        regions_by_frame,
        settings=config.render_settings(),
        fps=fps,
        width=width,
        height=height,
        source_has_audio=source_has_audio,
        overwrite=overwrite,
        log_path=log_path,
        video_codec=config.video_codec,
        pixel_format=config.pixel_format,
        movflags=config.movflags,
        partial_processing_label=partial_processing_label,
    )


def encode_browser_mp4(
    source_video: Path,
    video_only_path: Path,
    output_path: Path,
    *,
    source_has_audio: bool,
    config: PlateRenderConfig,
) -> str:
    command = [
        media.executable_path("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
    ]
    if source_has_audio:
        command.extend(["-i", str(source_video)])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-c:v",
            config.video_codec,
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            config.pixel_format,
            "-movflags",
            config.movflags,
        ]
    )
    if source_has_audio:
        command.extend(["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k"])
    command.append(str(output_path))
    subprocess.run(command, capture_output=True, text=True, check=True)
    return (
        "source_audio_muxed_from_input_video_duration_authoritative"
        if source_has_audio
        else "source_had_no_audio"
    )


def verify_rendered_regions(
    source_video: Path,
    output_video: Path,
    regions: list[PlateRenderRegion],
    *,
    config: PlateRenderConfig,
    frame_width: int,
    frame_height: int,
    max_failures_reported: int = 50,
) -> dict[str, Any]:
    return privacy_rendering.verify_rendered_regions(
        source_video,
        output_video,
        [region.to_render_region() for region in regions],
        settings=config.render_settings(),
        frame_width=frame_width,
        frame_height=frame_height,
        max_failures_reported=max_failures_reported,
    )


def render_payloads(
    *,
    source_video: Path,
    tracks_path: Path,
    review_intervals_path: Path,
    source_info: Any,
    output_video: Path,
    config: PlateRenderConfig,
    mode: str,
    validation: dict[str, Any],
    frame_span_report: dict[str, Any],
    media_validation: dict[str, Any] | None,
    region_validation: dict[str, Any] | None,
    rendered_frame_count: int,
    rendered_region_count: int,
    audio_handling: str,
    elapsed_seconds: float,
) -> dict[str, dict[str, Any]]:
    output_hash = media.sha256_file(output_video) if output_video.exists() else None
    provenance_counts = provenance_counts_for_tracks(tracks_path, config=config)
    review_status = {
        "blocking_gap_interval_count": validation["blocking_gap_interval_count"],
        "blocking_reasons": validation["reasons"],
        "final_approval": mode == MODE_REVIEWED_FINAL and validation["approved_for_final"],
        "full_source_processed": validation["full_source_processed"],
        "partial_processing": validation["partial_processing"],
        "preview_non_final": mode == MODE_PREVIEW,
        "review_required_interval_count": validation["review_required_interval_count"],
    }
    manifest = {
        "audio_handling": audio_handling,
        "configuration": config.to_dict(),
        "configuration_hash": config.configuration_hash(),
        "created_at": media.utc_timestamp(),
        "detector_id": plate_detection.DETECTOR_ID,
        "final_approval_status": review_status["final_approval"],
        "frame_span_report": frame_span_report,
        "full_source_processed": validation["full_source_processed"],
        "input_height": source_info.video_stream.height,
        "input_width": source_info.video_stream.width,
        "model_revision": plate_detection.MODEL_REVISION,
        "output_video": output_video.name,
        "output_video_hash": output_hash,
        "pipeline_id": PIPELINE_ID,
        "preview_non_final": mode == MODE_PREVIEW,
        "partial_processing": validation["partial_processing"],
        "privacy_region_input_hash": media.sha256_file(tracks_path),
        "provenance_counts": provenance_counts,
        "rendered_frame_count": rendered_frame_count,
        "rendered_region_count": rendered_region_count,
        "renderer_id": RENDERER_ID,
        "rendering_mode": mode,
        "review_intervals_hash": media.sha256_file(review_intervals_path),
        "review_status": review_status,
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": _optional_int(source_info.video_stream.frame_count),
        "source_video": media.portable_source_path(source_video),
        "source_video_hash": source_info.sha256,
        "tracking_configuration_hash": config.tracking_configuration_hash,
    }
    summary = {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "final_approval_status": review_status["final_approval"],
        "full_source_processed": validation["full_source_processed"],
        "media_validation_passed": None if media_validation is None else media_validation["passed"],
        "mode": mode,
        "preview_non_final": mode == MODE_PREVIEW,
        "partial_processing": validation["partial_processing"],
        "processing_fps": round(rendered_frame_count / elapsed_seconds, 6)
        if elapsed_seconds > 0
        else 0.0,
        "provenance_counts": provenance_counts,
        "render_configuration_hash": config.configuration_hash(),
        "rendered_frame_count": rendered_frame_count,
        "rendered_region_count": rendered_region_count,
        "rendered_region_validation_passed": None
        if region_validation is None
        else region_validation["passed"],
        "review_status": review_status,
        "schema_version": SCHEMA_VERSION,
    }
    validation_payload = {
        "media_validation": media_validation,
        "render_plan_validation": validation,
        "frame_span_report": frame_span_report,
        "rendered_region_validation": region_validation,
        "schema_version": SCHEMA_VERSION,
    }
    return {
        "manifest": manifest,
        "summary": summary,
        "validation": validation_payload,
    }


def write_render_payloads(output_dir: Path, payload: dict[str, dict[str, Any]]) -> None:
    write_json(output_dir / "render_manifest.json", payload["manifest"])
    write_json(output_dir / "render_summary.json", payload["summary"])
    write_json(output_dir / "validation.json", payload["validation"])


def build_plate_review_bundle(
    source_video: Path,
    *,
    tracks_path: Path,
    review_intervals_path: Path,
    automatic_preview_path: Path,
    diagnostic_overlay_path: Path | None,
    output_dir: Path,
    config: PlateRenderConfig | None = None,
    spot_check_count: int = DEFAULT_SPOT_CHECK_COUNT,
    context_frames: int = DEFAULT_FRAME_CONTEXT,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = config or default_config()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("assets", "clips", "contact_sheets", "decisions"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    source_info = media.probe_media(source_video)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    source_frame_count = _optional_int(source_info.video_stream.frame_count) or 0
    rows = load_track_rows(tracks_path)
    intervals = load_review_intervals(review_intervals_path)
    tasks = build_plate_review_tasks(
        rows,
        intervals,
        source_frame_count=source_frame_count,
        spot_check_count=spot_check_count,
    )
    media_roles = create_task_media(
        tasks,
        source_video=source_video,
        automatic_preview_path=automatic_preview_path,
        diagnostic_overlay_path=diagnostic_overlay_path,
        output_dir=output_dir,
        fps=fps,
        source_frame_count=source_frame_count,
        context_frames=context_frames,
        overwrite=overwrite,
    )
    tasks = [
        PlateReviewTask(
            **{
                **task.__dict__,
                "media_roles": media_roles.get(task.review_task_id, {}),
                "contact_sheet": contact_sheet_path_for(task.review_task_id, media_roles),
            }
        )
        for task in tasks
    ]
    task_payloads = [task.to_dict() for task in tasks]
    summary = build_review_summary(task_payloads, config=config)
    write_json(
        output_dir / "review_tasks.json",
        {"schema_version": REVIEW_SCHEMA_VERSION, "tasks": task_payloads},
    )
    write_json(output_dir / "review_summary.json", summary)
    write_json(output_dir / "review_config.json", config.to_dict())
    write_json(
        output_dir / "decisions" / "analyst_decisions.template.json",
        {"decisions": [], "schema_version": "plate_review_decisions.v1"},
    )
    write_review_assets(output_dir)
    write_review_index(output_dir, task_payloads, summary)
    write_review_markdown(output_dir, task_payloads, summary)
    return summary


def build_plate_review_tasks(
    rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    *,
    source_frame_count: int,
    spot_check_count: int,
) -> list[PlateReviewTask]:
    tasks: list[PlateReviewTask] = []
    for index, interval in enumerate(intervals, start=1):
        reason_codes = tuple(str(code) for code in interval.get("reason_codes") or [])
        blocking = True
        has_region = not bool(interval.get("blocking")) and not any(
            code in BLOCKING_REVIEW_REASONS for code in reason_codes
        )
        allowed = (
            ("Needs correction", "Not a license plate")
            if not has_region
            else (
                "Looks correct",
                "Needs correction",
                "Not a license plate",
            )
        )
        title = (
            "Plate coverage needs correction"
            if not has_region
            else "Check an uncertain plate track"
        )
        explanation = (
            "The tracker found a gap or uncertain plate track. Final plate export remains "
            "blocked until reviewed regions are supplied."
            if not has_region
            else "This plate-like track is plausible but not strong enough for automatic approval."
        )
        tasks.append(
            PlateReviewTask(
                review_task_id=f"plate_review_{index:04d}",
                object_category=plate_detection.CATEGORY,
                frame_start=int(interval["frame_start"]),
                frame_end=int(interval["frame_end"]),
                severity="blocking" if blocking else "medium",
                blocking=blocking,
                source_stage="plate_tracking",
                reason_codes=reason_codes,
                title=title,
                explanation=explanation,
                allowed_actions=allowed,
                default_action_statuses=action_status_map(allowed),
                track_uids=tuple(str(uid) for uid in interval.get("track_uids") or []),
                media_roles={},
            )
        )

    accepted_frames = sorted(
        {
            int(row["frame_index"])
            for row in rows
            for track in row.get("tracks") or []
            if track.get("status") == plate_tracking.ACCEPTED_TRACK
        }
    )
    for index, frame_index in enumerate(sample_frames(accepted_frames, spot_check_count), start=1):
        tasks.append(
            PlateReviewTask(
                review_task_id=f"plate_spot_check_{index:04d}",
                object_category=plate_detection.CATEGORY,
                frame_start=max(0, frame_index - 10),
                frame_end=min(source_frame_count - 1, frame_index + 10)
                if source_frame_count
                else frame_index + 10,
                severity="random_audit",
                blocking=False,
                source_stage="random_audit",
                reason_codes=("deterministic_plate_spot_check",),
                title="Optional plate spot check",
                explanation=(
                    "This sample looks normal to the automatic system. Spot checks reduce the "
                    "risk of silent mistakes but do not prove the full video is perfect."
                ),
                allowed_actions=("Looks correct", "Needs correction", "Not a license plate"),
                default_action_statuses=action_status_map(
                    ("Looks correct", "Needs correction", "Not a license plate")
                ),
                track_uids=(),
                media_roles={},
            )
        )

    tasks.extend(short_accepted_track_review_tasks(rows, start_index=len(tasks) + 1))
    tasks.extend(technical_track_boundary_tasks(rows, start_index=len(tasks) + 1))
    return sorted(
        tasks,
        key=lambda item: (not item.blocking, item.technical_only, item.frame_start),
    )


def technical_track_boundary_tasks(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
) -> list[PlateReviewTask]:
    by_uid: dict[str, list[int]] = {}
    for row in rows:
        frame_index = int(row["frame_index"])
        for track in row.get("tracks") or []:
            if track.get("status") != plate_tracking.ACCEPTED_TRACK:
                continue
            by_uid.setdefault(str(track.get("track_uid")), []).append(frame_index)
    tasks: list[PlateReviewTask] = []
    for offset, (uid, frames) in enumerate(sorted(by_uid.items()), start=start_index):
        tasks.append(
            PlateReviewTask(
                review_task_id=f"plate_technical_{offset:04d}",
                object_category=plate_detection.CATEGORY,
                frame_start=min(frames),
                frame_end=max(frames),
                severity="informational",
                blocking=False,
                source_stage="plate_track_boundaries",
                reason_codes=("track_start", "track_end"),
                title="Accepted plate track boundary",
                explanation=(
                    "Technical audit record for where an accepted plate track starts and ends."
                ),
                allowed_actions=(),
                default_action_statuses={},
                track_uids=(uid,),
                media_roles={},
                technical_only=True,
            )
        )
    return tasks


def short_accepted_track_review_tasks(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
) -> list[PlateReviewTask]:
    by_uid: dict[str, list[int]] = {}
    for row in rows:
        frame_index = int(row["frame_index"])
        for track in row.get("tracks") or []:
            if track.get("status") != plate_tracking.ACCEPTED_TRACK:
                continue
            by_uid.setdefault(str(track.get("track_uid")), []).append(frame_index)

    tasks: list[PlateReviewTask] = []
    for offset, (uid, frames) in enumerate(sorted(by_uid.items()), start=start_index):
        if len(set(frames)) > SHORT_ACCEPTED_TRACK_REVIEW_MAX_FRAMES:
            continue
        tasks.append(
            PlateReviewTask(
                review_task_id=f"plate_false_positive_check_{offset:04d}",
                object_category=plate_detection.CATEGORY,
                frame_start=min(frames),
                frame_end=max(frames),
                severity="medium",
                blocking=False,
                source_stage="plate_rendering_review",
                reason_codes=("possible_false_accepted_track", "short_accepted_track"),
                title="Check a short accepted plate track",
                explanation=(
                    "A short accepted track will be redacted in the preview. Confirm it is a "
                    "real license plate, not a transient text or graphic detection."
                ),
                allowed_actions=("Looks correct", "Needs correction", "Not a license plate"),
                default_action_statuses=action_status_map(
                    ("Looks correct", "Needs correction", "Not a license plate")
                ),
                track_uids=(uid,),
                media_roles={},
            )
        )
    return tasks


def action_status_map(actions: tuple[str, ...]) -> dict[str, str]:
    mapping = {
        "Looks correct": "accepted_current",
        "Needs correction": "correction_required",
        "Not a license plate": "false_positive",
    }
    return {action: mapping[action] for action in actions}


def sample_frames(frames: list[int], count: int) -> list[int]:
    if count <= 0 or not frames:
        return []
    if len(frames) <= count:
        return frames
    if count == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (count - 1)
    return [frames[round(index * step)] for index in range(count)]


def create_task_media(
    tasks: list[PlateReviewTask],
    *,
    source_video: Path,
    automatic_preview_path: Path,
    diagnostic_overlay_path: Path | None,
    output_dir: Path,
    fps: float,
    source_frame_count: int,
    context_frames: int,
    overwrite: bool,
) -> dict[str, dict[str, str]]:
    media_roles: dict[str, dict[str, str]] = {}
    for task in tasks:
        if task.technical_only:
            media_roles[task.review_task_id] = {}
            continue
        start_frame = max(0, task.frame_start - context_frames)
        end_frame = min(source_frame_count - 1, task.frame_end + context_frames)
        roles = {
            "original": output_dir / "clips" / f"{task.review_task_id}_original.mp4",
            "automatic_preview": output_dir
            / "clips"
            / f"{task.review_task_id}_automatic_preview.mp4",
        }
        write_frame_interval_clip(
            source_video,
            roles["original"],
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            overwrite=overwrite,
        )
        write_frame_interval_clip(
            automatic_preview_path,
            roles["automatic_preview"],
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            overwrite=overwrite,
        )
        if diagnostic_overlay_path is not None and diagnostic_overlay_path.exists():
            diagnostic_path = output_dir / "clips" / f"{task.review_task_id}_diagnostic.mp4"
            write_frame_interval_clip(
                diagnostic_overlay_path,
                diagnostic_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                overwrite=overwrite,
            )
            roles["diagnostic"] = diagnostic_path
        contact_sheet = output_dir / "contact_sheets" / f"{task.review_task_id}.jpg"
        write_task_contact_sheet(
            source_video,
            automatic_preview_path,
            contact_sheet,
            frames=sample_frames(list(range(start_frame, end_frame + 1)), 6),
            overwrite=overwrite,
        )
        media_roles[task.review_task_id] = {
            key: path.relative_to(output_dir).as_posix() for key, path in roles.items()
        }
        media_roles[task.review_task_id]["contact_sheet"] = contact_sheet.relative_to(
            output_dir
        ).as_posix()
    return media_roles


def contact_sheet_path_for(
    review_task_id: str,
    media_roles: dict[str, dict[str, str]],
) -> str | None:
    return media_roles.get(review_task_id, {}).get("contact_sheet")


def write_frame_interval_clip(
    input_video: Path,
    output_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_video}")
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    temp_path = output_path.with_name(f".{output_path.stem}.mp4v.mp4")
    h264_path = output_path.with_name(f".{output_path.stem}.h264.mp4")
    for path in (temp_path, h264_path):
        if path.exists():
            path.unlink()
    writer = cv.VideoWriter(
        str(temp_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create video: {temp_path}")
    capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    try:
        while frame_index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    encode_browser_mp4(
        input_video,
        temp_path,
        h264_path,
        source_has_audio=False,
        config=default_config(),
    )
    h264_path.replace(output_path)
    if temp_path.exists():
        temp_path.unlink()


def write_task_contact_sheet(
    source_video: Path,
    automatic_preview_path: Path,
    output_path: Path,
    *,
    frames: list[int],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []
    for frame_index in frames:
        source = read_frame(source_video, frame_index)
        preview = read_frame(automatic_preview_path, frame_index)
        if source is None or preview is None:
            continue
        source = resize_for_sheet(source)
        preview = resize_for_sheet(preview)
        cv.putText(
            source,
            f"original f{frame_index}",
            (8, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv.putText(
            preview,
            f"automatic f{frame_index}",
            (8, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        tiles.append(np.hstack([source, preview]))
    if not tiles:
        raise ValueError("no frames available for contact sheet")
    width = max(tile.shape[1] for tile in tiles)
    padded: list[np.ndarray] = []
    for tile in tiles:
        if tile.shape[1] < width:
            pad = np.full((tile.shape[0], width - tile.shape[1], 3), 255, dtype=np.uint8)
            tile = np.hstack([tile, pad])
        padded.append(tile)
    sheet = np.vstack(padded)
    cv.imwrite(str(output_path), sheet)


def read_frame(path: Path, frame_index: int) -> np.ndarray | None:
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def resize_for_sheet(frame: np.ndarray, width: int = 360) -> np.ndarray:
    height = max(1, round(frame.shape[0] * (width / frame.shape[1])))
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def build_review_summary(
    tasks: list[dict[str, Any]],
    *,
    config: PlateRenderConfig,
) -> dict[str, Any]:
    visible_tasks = [task for task in tasks if not task.get("technical_only")]
    blocking = [task for task in visible_tasks if task["blocking"]]
    return {
        "blocking_items": len(blocking),
        "final_export_blocked": bool(blocking),
        "object_category": plate_detection.CATEGORY,
        "optional_spot_checks": sum(
            1 for task in visible_tasks if task["severity"] == "random_audit"
        ),
        "render_configuration_hash": config.configuration_hash(),
        "required_review_items": len(blocking),
        "schema_version": REVIEW_SCHEMA_VERSION,
        "technical_items": sum(1 for task in tasks if task.get("technical_only")),
        "total_reviewer_tasks": len(visible_tasks),
    }


def validate_plate_review_decisions(
    tasks_path: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = {task["review_task_id"]: task for task in tasks_payload.get("tasks", [])}
    decisions_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions = decisions_payload.get("decisions") or []
    seen: set[str] = set()
    errors: list[str] = []
    unresolved_blocking = {
        task_id
        for task_id, task in tasks.items()
        if task.get("blocking") and not task.get("technical_only")
    }
    for decision in decisions:
        task_id = str(decision.get("review_task_id"))
        if task_id not in tasks:
            errors.append(f"unknown review_task_id: {task_id}")
            continue
        if task_id in seen:
            errors.append(f"duplicate decision for review_task_id: {task_id}")
            continue
        seen.add(task_id)
        action = str(decision.get("action"))
        allowed = set(tasks[task_id].get("allowed_actions") or [])
        if action not in allowed:
            errors.append(f"invalid action for {task_id}: {action}")
        status = str(decision.get("status"))
        expected_status = tasks[task_id].get("default_action_statuses", {}).get(action)
        if expected_status and status != expected_status:
            errors.append(f"invalid status for {task_id}: {status}")
        if status in {"accepted_current", "false_positive"}:
            unresolved_blocking.discard(task_id)
    return {
        "correction_files_required": any(
            str(decision.get("status")) == "correction_required" for decision in decisions
        ),
        "duplicate_decision_count": len(decisions) - len(seen),
        "errors": errors,
        "passed": not errors,
        "schema_version": "plate_review_decision_validation.v1",
        "unresolved_blocking_items": sorted(unresolved_blocking),
        "unresolved_blocking_count": len(unresolved_blocking),
    }


def write_review_assets(output_dir: Path) -> None:
    css = "\n".join(
        [
            "body { margin: 0; font: 16px/1.45 -apple-system, BlinkMacSystemFont,",
            '  "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }',
            "header { padding: 28px 32px; background: #13202f; color: #fff; }",
            "main { max-width: 1120px; margin: 0 auto; padding: 24px; }",
            ".summary { display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }",
            ".pill { background: #fff; border: 1px solid #d8dee8; border-radius: 6px;",
            "  padding: 8px 12px; }",
            ".card { background: #fff; border: 1px solid #d8dee8; border-radius: 8px;",
            "  padding: 18px; margin: 18px 0; }",
            ".blocking { border-left: 5px solid #bd2727; }",
            ".random_audit { border-left: 5px solid #4f759b; }",
            ".media-grid { display: grid;",
            "  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));",
            "  gap: 12px; margin: 14px 0; }",
            "video, img { width: 100%; background: #111; border-radius: 6px; }",
            ".actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }",
            "button { border: 1px solid #8290a0; background: #fff;",
            "  padding: 8px 12px; border-radius: 6px; cursor: pointer; }",
            "button.selected { background: #174ea6; color: #fff; border-color: #174ea6; }",
            "details { margin-top: 10px; }",
            "code { background: #eef1f5; padding: 1px 4px; border-radius: 4px; }",
        ]
    )
    js = "\n".join(
        [
            "const storageKey = `plate-review-decisions:${location.pathname}`;",
            'const decisions = JSON.parse(localStorage.getItem(storageKey) || "{}");',
            "",
            "function saveDecision(taskId, action, status) {",
            "  decisions[taskId] = {",
            "    review_task_id: taskId,",
            "    action,",
            "    status,",
            "    decided_at: new Date().toISOString(),",
            "  };",
            "  localStorage.setItem(storageKey, JSON.stringify(decisions));",
            "  renderDecisionState();",
            "}",
            "",
            "function renderDecisionState() {",
            '  document.querySelectorAll("[data-task-id]").forEach((button) => {',
            "    const saved = decisions[button.dataset.taskId];",
            "    button.classList.toggle(",
            '      "selected",',
            "      Boolean(saved && saved.action === button.dataset.action),",
            "    );",
            "  });",
            "  const count = Object.keys(decisions).length;",
            '  const counter = document.getElementById("decision-count");',
            "  if (counter) {",
            "    counter.textContent =",
            '      `${count} decision${count === 1 ? "" : "s"} saved locally`;',
            "  }",
            "}",
            "",
            "function exportDecisions() {",
            "  const payload = {",
            '    schema_version: "plate_review_decisions.v1",',
            "    decisions: Object.values(decisions),",
            "  };",
            "  const blob = new Blob(",
            '    [JSON.stringify(payload, null, 2) + "\\n"],',
            '    { type: "application/json" },',
            "  );",
            "  const url = URL.createObjectURL(blob);",
            '  const link = document.createElement("a");',
            "  link.href = url;",
            '  link.download = "analyst_decisions.json";',
            "  link.click();",
            "  URL.revokeObjectURL(url);",
            "}",
            "",
            'document.addEventListener("click", (event) => {',
            '  const button = event.target.closest("[data-task-id]");',
            "  if (button) {",
            "    saveDecision(",
            "      button.dataset.taskId,",
            "      button.dataset.action,",
            "      button.dataset.status,",
            "    );",
            "  }",
            '  if (event.target.id === "export-decisions") exportDecisions();',
            "});",
            'document.addEventListener("DOMContentLoaded", renderDecisionState);',
        ]
    )
    (output_dir / "assets" / "review.css").write_text(css + "\n", encoding="utf-8")
    (output_dir / "assets" / "review.js").write_text(js + "\n", encoding="utf-8")


def write_review_index(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    cards = "\n".join(task_card(task) for task in tasks if not task.get("technical_only"))
    technical_cards = "\n".join(task_card(task) for task in tasks if task.get("technical_only"))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plate Review Bundle</title>
  <link rel="stylesheet" href="assets/review.css">
</head>
<body>
<header>
  <h1>Plate Review Bundle</h1>
  <p>
    Automatic preview uses accepted plate tracks only. Final plate export remains
    blocked while required issues are unresolved.
  </p>
</header>
<main>
  <section class="summary">
    <div class="pill">Required review: {summary["required_review_items"]}</div>
    <div class="pill">Optional spot checks: {summary["optional_spot_checks"]}</div>
    <div class="pill" id="decision-count">0 decisions saved locally</div>
  </section>
  <p>Your choices stay only in this browser until you export them.</p>
  <button id="export-decisions">Export Review Decisions</button>
  <h2>Needs Review</h2>
  {cards}
  <details>
    <summary>Technical audit</summary>
    {technical_cards or "<p>No technical-only items.</p>"}
  </details>
</main>
<script src="assets/review.js"></script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def task_card(task: dict[str, Any]) -> str:
    media_html = "".join(
        media_role_html(role, path)
        for role, path in task.get("media_roles", {}).items()
        if role != "contact_sheet"
    )
    contact_sheet = task.get("contact_sheet")
    contact_html = (
        f'<details><summary>Contact sheet</summary><img src="{contact_sheet}" '
        'alt="Contact sheet"></details>'
        if contact_sheet
        else ""
    )
    actions = "".join(
        f'<button data-task-id="{task["review_task_id"]}" data-action="{action}" '
        f'data-status="{task["default_action_statuses"].get(action, "")}">{action}</button>'
        for action in task.get("allowed_actions", [])
    )
    return f"""<article class="card {task["severity"]}">
  <h3>{task["title"]}</h3>
  <p><strong>Frames:</strong> {task["frame_start"]}–{task["frame_end"]}</p>
  <p>{task["explanation"]}</p>
  <div class="media-grid">{media_html}</div>
  {contact_html}
  <div class="actions">{actions}</div>
  <details>
    <summary>Technical details</summary>
    <p>Reasons: <code>{", ".join(task["reason_codes"])}</code></p>
  </details>
</article>"""


def media_role_html(role: str, path: str) -> str:
    label = {
        "automatic_preview": "Automatic plate preview",
        "diagnostic": "Diagnostic overlay",
        "original": "Original",
    }.get(role, role.replace("_", " ").title())
    return f"""<section>
  <h4>{label}</h4>
  <video controls preload="metadata" src="{path}"></video>
  <p><a href="{path}">Open {label}</a></p>
</section>"""


def write_review_markdown(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Plate Review Bundle",
        "",
        f"- Required review items: {summary['required_review_items']}",
        f"- Optional spot checks: {summary['optional_spot_checks']}",
        f"- Final export blocked: {summary['final_export_blocked']}",
        "",
    ]
    for task in tasks:
        if task.get("technical_only"):
            continue
        lines.extend(
            [
                f"## {task['title']}",
                "",
                f"- Frames: {task['frame_start']}-{task['frame_end']}",
                f"- Reasons: {', '.join(task['reason_codes'])}",
                f"- Actions: {', '.join(task['allowed_actions'])}",
                "",
            ]
        )
    content = "\n".join(lines) + "\n"
    (output_dir / "README.md").write_text(content, encoding="utf-8")
    (output_dir / "INDEX.md").write_text(content, encoding="utf-8")
    checklist = [
        "# Plate Review Checklist",
        "",
        "- Confirm unresolved gaps are not approved without correction regions.",
        "- Confirm accepted automatic plate regions cover visible plate characters.",
        "- Mark text, graphics, or other non-plate redactions as Not a license plate.",
        "- Export decisions before closing the browser.",
        "",
    ]
    (output_dir / "REVIEW_CHECKLIST.md").write_text(
        "\n".join(checklist),
        encoding="utf-8",
    )


def provenance_counts_for_tracks(
    tracks_path: Path,
    *,
    config: PlateRenderConfig,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in eligible_regions_from_rows(load_track_rows(tracks_path), config=config):
        counts[region.provenance] = counts.get(region.provenance, 0) + 1
    return dict(sorted(counts.items()))


def summarize_track_render_inputs(
    tracks_path: Path,
    *,
    config: PlateRenderConfig,
) -> dict[str, Any]:
    rows = load_track_rows(tracks_path)
    eligible = eligible_regions_from_rows(rows, config=config)
    review_track_frames = {
        int(row["frame_index"])
        for row in rows
        for track in row.get("tracks") or []
        if track.get("status") == plate_tracking.NEEDS_REVIEW
    }
    rejected_noise_frames = {
        int(row["frame_index"])
        for row in rows
        for track in row.get("tracks") or []
        if track.get("status") == plate_tracking.REJECTED_NOISE
    }
    return {
        "detector_observation_frames_rendered": len(
            {
                region.frame_index
                for region in eligible
                if region.provenance == plate_tracking.PROVENANCE_DETECTOR
            }
        ),
        "interpolated_frames_rendered": len(
            {
                region.frame_index
                for region in eligible
                if region.provenance == plate_tracking.PROVENANCE_INTERPOLATION
            }
        ),
        "rendered_region_count": len(eligible),
        "rejected_noise_frames": len(rejected_noise_frames),
        "review_track_frames": len(review_track_frames),
        "tracked_observation_frames_rendered": len({region.frame_index for region in eligible}),
    }


def interval_summaries(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "blocking": bool(interval.get("blocking")),
            "frame_end": interval.get("frame_end"),
            "frame_start": interval.get("frame_start"),
            "reason_codes": interval.get("reason_codes", []),
            "severity": interval.get("severity"),
            "track_uids": interval.get("track_uids", []),
        }
        for interval in intervals
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def copy_diagnostic_media(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
