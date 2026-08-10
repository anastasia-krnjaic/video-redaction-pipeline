from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2 as cv

from redactor import media, privacy_rendering, rendering
from redactor.detection import rational_to_float
from redactor.privacy_records import APPROVAL_ACCEPTED, CATEGORY_FACE, PrivacyRegionRecord

SCHEMA_VERSION = "face_automatic_preview_shared_renderer.v1"
RENDERER_ID = "shared_face_automatic_preview_renderer_v1"
RENDERER_BACKEND = "shared_privacy_renderer"


def run_shared_face_preview(
    input_video: Path,
    *,
    privacy_regions_path: Path,
    output_dir: Path,
    settings: rendering.RenderSettings,
    interval_decisions_path: Path | None = None,
    output_video_path: Path | None = None,
    comparison_frames: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    if settings.mode != rendering.MODE_PREVIEW:
        raise ValueError("shared face preview renderer only supports preview mode")
    settings.validate()
    rendering.ensure_output_dir(output_dir, overwrite=overwrite)
    log_path = output_dir / "processing_log.jsonl"
    if log_path.exists() and overwrite:
        log_path.unlink()
    rendering.log_event(
        log_path,
        "started",
        {"backend": RENDERER_BACKEND, "mode": settings.mode, "method": settings.method},
    )

    output_path = output_video_path or output_dir / "redacted_preview.mp4"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    privacy_input = rendering.load_privacy_regions(privacy_regions_path)
    intervals = rendering.load_interval_decisions(interval_decisions_path)
    source_info = media.probe_media(input_video)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    plan_validation = rendering.validate_render_plan(
        privacy_input,
        intervals,
        mode=settings.mode,
        frame_width=width,
        frame_height=height,
    )
    records = face_privacy_region_records_from_input(privacy_input)
    started_at = time.perf_counter()
    rendered_frame_count, rendered_region_count, audio_handling = (
        privacy_rendering.render_privacy_video(
            input_video,
            output_path,
            privacy_rendering.regions_by_frame(records),
            settings=settings,
            fps=fps,
            width=width,
            height=height,
            source_has_audio=source_info.audio_stream is not None,
            overwrite=overwrite,
            log_path=log_path,
        )
    )
    elapsed_seconds = time.perf_counter() - started_at
    if comparison_frames > 0:
        write_diagnostic_comparison(
            input_video,
            output_path,
            output_dir / "diagnostic_comparison.mp4",
            fps=fps,
            width=width,
            height=height,
            start_frame=min(privacy_input.required_frames)
            if privacy_input.required_frames
            else 0,
            frame_count=comparison_frames,
            overwrite=overwrite,
        )

    media_validation = rendering.validate_output_media(
        input_video,
        source_info,
        output_path,
        expected_frame_count=rendered_frame_count,
        expected_width=width,
        expected_height=height,
        expected_fps=fps,
        source_has_audio=source_info.audio_stream is not None,
        settings=settings,
    )
    full_span_validation = privacy_rendering.validate_full_span_output(
        input_video,
        output_path,
        expected_frame_count=rendered_frame_count,
        expected_width=width,
        expected_height=height,
        expected_fps=fps,
        source_has_audio=source_info.audio_stream is not None,
    )
    region_validation = privacy_rendering.verify_rendered_regions(
        input_video,
        output_path,
        [record.to_render_region() for record in records],
        settings=settings,
        frame_width=width,
        frame_height=height,
    )
    payload = rendering.output_payloads(
        input_video=input_video,
        privacy_regions_path=privacy_regions_path,
        interval_decisions_path=interval_decisions_path,
        source_info=source_info,
        output_path=output_path,
        settings=settings,
        privacy_input=privacy_input,
        intervals=intervals,
        validation=plan_validation,
        media_validation=media_validation,
        rendered_frame_count=rendered_frame_count,
        rendered_region_count=rendered_region_count,
        output_sha256=media.sha256_file(output_path),
        audio_handling=audio_handling,
        elapsed_seconds=elapsed_seconds,
    )
    payload = add_shared_preview_identity(
        payload,
        full_span_validation=full_span_validation,
        region_validation=region_validation,
    )
    rendering.write_payloads(output_dir, payload)
    rendering.log_event(
        log_path,
        "finished",
        {
            "backend": RENDERER_BACKEND,
            "output_video": output_path.name,
            "rendered_frame_count": rendered_frame_count,
            "rendered_region_count": rendered_region_count,
        },
    )
    if media_validation["passed"] is not True:
        raise RuntimeError("shared face automatic preview media validation failed")
    if full_span_validation["passed"] is not True:
        raise RuntimeError("shared face automatic preview full-span validation failed")
    if region_validation["passed"] is not True:
        raise RuntimeError("shared face automatic preview rendered-region validation failed")
    return payload["summary"]


def face_privacy_region_records(path: Path) -> list[PrivacyRegionRecord]:
    if not path.exists():
        return []
    return face_privacy_region_records_from_input(rendering.load_privacy_regions(path))


def face_privacy_region_records_from_input(
    privacy_input: rendering.PrivacyRegionInput,
) -> list[PrivacyRegionRecord]:
    records: list[PrivacyRegionRecord] = []
    for frame_index in sorted(privacy_input.regions_by_frame):
        for index, region in enumerate(privacy_input.regions_by_frame[frame_index]):
            if not region.is_renderable:
                continue
            source = dict(region.source_record)
            track_ids = source.get("target_track_ids") or []
            object_id = (
                f"face_track_{track_ids[0]}"
                if track_ids
                else source.get("track_uid") or source.get("track_id")
            )
            records.append(
                PrivacyRegionRecord(
                    approval_state=APPROVAL_ACCEPTED,
                    bbox_xyxy=region.bbox_xyxy,
                    category=CATEGORY_FACE,
                    object_id=None if object_id is None else str(object_id),
                    provenance=region.provenance,
                    render_eligible=True,
                    source_frame=frame_index,
                    source_record={
                        **source,
                        "category": CATEGORY_FACE,
                        "render_order": index,
                        "region_id": source.get("region_id", f"face_region_{frame_index}_{index}"),
                    },
                )
            )
    return records


def add_shared_preview_identity(
    payload: dict[str, dict[str, object]],
    *,
    full_span_validation: dict[str, Any],
    region_validation: dict[str, Any],
) -> dict[str, dict[str, object]]:
    manifest = payload["manifest"]
    summary = payload["summary"]
    validation = payload["validation"]
    manifest.update(
        {
            "effective_regions_hash": manifest["privacy_region_input_hash"],
            "legacy_renderer_id": rendering.RENDERER_ID,
            "renderer_backend": RENDERER_BACKEND,
            "renderer_id": RENDERER_ID,
            "rendered_region_validation_passed": region_validation["passed"],
            "schema_version": SCHEMA_VERSION,
            "strict_full_span_validation_passed": full_span_validation["passed"],
        }
    )
    summary.update(
        {
            "effective_regions_hash": manifest["privacy_region_input_hash"],
            "renderer_backend": RENDERER_BACKEND,
            "renderer_id": RENDERER_ID,
            "rendered_region_validation_passed": region_validation["passed"],
            "schema_version": SCHEMA_VERSION,
            "strict_full_span_validation_passed": full_span_validation["passed"],
        }
    )
    validation.update(
        {
            "automatic_preview_full_span_validation": full_span_validation,
            "rendered_region_validation": region_validation,
            "schema_version": SCHEMA_VERSION,
        }
    )
    return payload


def write_diagnostic_comparison(
    source_video: Path,
    rendered_video: Path,
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    start_frame: int,
    frame_count: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    source_capture = cv.VideoCapture(str(source_video))
    rendered_capture = cv.VideoCapture(str(rendered_video))
    if not source_capture.isOpened() or not rendered_capture.isOpened():
        source_capture.release()
        rendered_capture.release()
        raise ValueError("could not open source or rendered video for comparison")
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        source_capture.release()
        rendered_capture.release()
        raise ValueError(f"could not create comparison video: {output_path}")
    try:
        source_capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        rendered_capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, start_frame + frame_count):
            source_ok, source_frame = source_capture.read()
            rendered_ok, rendered_frame = rendered_capture.read()
            if not source_ok or not rendered_ok:
                break
            writer.write(rendering.comparison_frame(source_frame, rendered_frame, frame_index))
    finally:
        source_capture.release()
        rendered_capture.release()
        writer.release()
