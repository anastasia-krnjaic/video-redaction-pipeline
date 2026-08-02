from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from redactor import media
from redactor.detection import DETECTOR_BACKEND_ADAS, rational_to_float
from redactor.models import MediaInfo

SCHEMA_VERSION = "1.0"
PIPELINE_ID = "face_pipeline_v1"
RENDERER_ID = "face_redaction_renderer_v1"

MODE_PREVIEW = "preview"
MODE_FINAL = "final"
METHOD_GAUSSIAN_BLUR = "gaussian_blur"
METHOD_PIXELATE = "pixelate"

FINAL_BLOCKING_STATUSES = {"needs_review", "rejected"}

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class RenderSettings:
    mode: str = MODE_PREVIEW
    method: str = METHOD_GAUSSIAN_BLUR
    margin_pixels: int = 0
    margin_fraction: float = 0.0
    blur_kernel: int = 51
    pixel_size: int = 12
    preview_watermark: bool = True
    frame_count_tolerance: int = 2
    duration_tolerance_seconds: float = 0.25

    def validate(self) -> None:
        if self.mode not in {MODE_PREVIEW, MODE_FINAL}:
            raise ValueError("render mode must be preview or final")
        if self.method not in {METHOD_GAUSSIAN_BLUR, METHOD_PIXELATE}:
            raise ValueError("render method must be gaussian_blur or pixelate")
        if self.margin_pixels < 0:
            raise ValueError("margin-pixels cannot be negative")
        if self.margin_fraction < 0:
            raise ValueError("margin-fraction cannot be negative")
        if self.blur_kernel < 3:
            raise ValueError("blur-kernel must be at least 3")
        if self.pixel_size < 2:
            raise ValueError("pixel-size must be at least 2")
        if self.frame_count_tolerance < 0:
            raise ValueError("frame-count-tolerance cannot be negative")
        if self.duration_tolerance_seconds < 0:
            raise ValueError("duration-tolerance-seconds cannot be negative")

    @property
    def normalized_blur_kernel(self) -> int:
        return self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1


@dataclass(frozen=True)
class RenderRegion:
    frame_index: int
    bbox_xyxy: Box | None
    provenance: str
    state: str
    source_record: dict[str, object]
    reasons: tuple[str, ...] = ()

    @property
    def is_renderable(self) -> bool:
        return self.state == "covered" and self.bbox_xyxy is not None


@dataclass(frozen=True)
class PrivacyRegionInput:
    regions_by_frame: dict[int, list[RenderRegion]]
    required_frames: tuple[int, ...]
    schema_hint: str


def run_face_render(
    input_video: Path,
    *,
    privacy_regions_path: Path,
    output_dir: Path,
    settings: RenderSettings,
    interval_decisions_path: Path | None = None,
    output_video_path: Path | None = None,
    comparison_frames: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    settings.validate()
    ensure_output_dir(output_dir, overwrite=overwrite)
    log_path = output_dir / "processing_log.jsonl"
    if log_path.exists() and overwrite:
        log_path.unlink()
    log_event(log_path, "started", {"mode": settings.mode, "method": settings.method})

    output_path = output_video_path or output_dir / f"redacted_{settings.mode}.mp4"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    privacy_input = load_privacy_regions(privacy_regions_path)
    intervals = load_interval_decisions(interval_decisions_path)
    source_info = media.probe_media(input_video)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    validation = validate_render_plan(
        privacy_input,
        intervals,
        mode=settings.mode,
        frame_width=width,
        frame_height=height,
    )

    if settings.mode == MODE_FINAL and not validation["approved_for_final"]:
        payload = output_payloads(
            input_video=input_video,
            privacy_regions_path=privacy_regions_path,
            interval_decisions_path=interval_decisions_path,
            source_info=source_info,
            output_path=output_path,
            settings=settings,
            privacy_input=privacy_input,
            intervals=intervals,
            validation=validation,
            media_validation=None,
            rendered_frame_count=0,
            rendered_region_count=0,
            output_sha256=None,
            audio_handling="not_rendered_final_validation_failed",
            elapsed_seconds=0.0,
        )
        write_payloads(output_dir, payload)
        log_event(log_path, "blocked", {"reasons": validation["blocking_reasons"]})
        reason_text = ", ".join(str(item) for item in validation["blocking_reasons"])
        raise ValueError("final render blocked: " + reason_text)

    started_at = time.perf_counter()
    rendered_frame_count, rendered_region_count, audio_handling = render_video(
        input_video,
        output_path,
        privacy_input.regions_by_frame,
        settings=settings,
        fps=fps,
        width=width,
        height=height,
        source_has_audio=source_info.audio_stream is not None,
        comparison_path=output_dir / "diagnostic_comparison.mp4" if comparison_frames > 0 else None,
        comparison_frames=comparison_frames,
        comparison_start_frame=min(privacy_input.required_frames)
        if privacy_input.required_frames
        else 0,
        overwrite=overwrite,
    )
    elapsed_seconds = time.perf_counter() - started_at
    output_sha = media.sha256_file(output_path)
    media_validation = validate_output_media(
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
    payload = output_payloads(
        input_video=input_video,
        privacy_regions_path=privacy_regions_path,
        interval_decisions_path=interval_decisions_path,
        source_info=source_info,
        output_path=output_path,
        settings=settings,
        privacy_input=privacy_input,
        intervals=intervals,
        validation=validation,
        media_validation=media_validation,
        rendered_frame_count=rendered_frame_count,
        rendered_region_count=rendered_region_count,
        output_sha256=output_sha,
        audio_handling=audio_handling,
        elapsed_seconds=elapsed_seconds,
    )
    write_payloads(output_dir, payload)
    log_event(
        log_path,
        "finished",
        {
            "output_video": output_path.name,
            "rendered_frame_count": rendered_frame_count,
            "rendered_region_count": rendered_region_count,
        },
    )
    return payload["summary"]


def load_privacy_regions(path: Path) -> PrivacyRegionInput:
    regions_by_frame: dict[int, list[RenderRegion]] = {}
    required_frames: set[int] = set()
    schema_hint = "unknown"
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            required_frames.add(frame_index)
            if "privacy_regions" in record:
                schema_hint = "privacy_gap_recovery"
                raw_regions = record.get("privacy_regions") or []
                regions_by_frame[frame_index] = [
                    region_from_record(frame_index, region) for region in raw_regions
                ]
                if not raw_regions:
                    regions_by_frame[frame_index] = [
                        RenderRegion(
                            frame_index=frame_index,
                            bbox_xyxy=None,
                            provenance="missing_privacy_region",
                            state="unresolved",
                            source_record=record,
                            reasons=("missing_privacy_region",),
                        )
                    ]
            else:
                schema_hint = "flat_privacy_track"
                regions_by_frame.setdefault(frame_index, []).append(
                    region_from_record(frame_index, record)
                )
    return PrivacyRegionInput(
        regions_by_frame=regions_by_frame,
        required_frames=tuple(sorted(required_frames)),
        schema_hint=schema_hint,
    )


def region_from_record(frame_index: int, record: dict[str, object]) -> RenderRegion:
    bbox = record.get("bbox_xyxy")
    reasons = record.get("reasons") or []
    return RenderRegion(
        frame_index=frame_index,
        bbox_xyxy=tuple(int(value) for value in bbox) if bbox is not None else None,
        provenance=str(record.get("provenance") or record.get("method") or "unknown"),
        state=str(record.get("state") or ("covered" if bbox is not None else "unresolved")),
        source_record=record,
        reasons=tuple(str(reason) for reason in reasons),
    )


def load_interval_decisions(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload.get("interval_decisions", [])]


def validate_render_plan(
    privacy_input: PrivacyRegionInput,
    intervals: list[dict[str, object]],
    *,
    mode: str,
    frame_width: int,
    frame_height: int,
) -> dict[str, object]:
    needs_review_intervals = [
        interval for interval in intervals if interval.get("status") == "needs_review"
    ]
    rejected_intervals = [
        interval for interval in intervals if interval.get("status") == "rejected"
    ]
    unresolved_frames: list[int] = []
    invalid_boxes: list[dict[str, object]] = []
    out_of_frame_boxes: list[dict[str, object]] = []

    for frame_index in privacy_input.required_frames:
        regions = privacy_input.regions_by_frame.get(frame_index, [])
        if not regions:
            unresolved_frames.append(frame_index)
            continue
        for region in regions:
            if region.state != "covered" or region.bbox_xyxy is None:
                unresolved_frames.append(frame_index)
                continue
            if not box_has_area(region.bbox_xyxy):
                invalid_boxes.append(
                    {"frame_index": frame_index, "bbox_xyxy": list(region.bbox_xyxy)}
                )
                continue
            if box_outside_frame(region.bbox_xyxy, frame_width, frame_height):
                out_of_frame_boxes.append(
                    {"frame_index": frame_index, "bbox_xyxy": list(region.bbox_xyxy)}
                )
            if clipped_box(region.bbox_xyxy, frame_width, frame_height) is None:
                invalid_boxes.append(
                    {"frame_index": frame_index, "bbox_xyxy": list(region.bbox_xyxy)}
                )

    blocking_reasons: list[str] = []
    if needs_review_intervals:
        blocking_reasons.append("needs_review_intervals")
    if rejected_intervals:
        blocking_reasons.append("rejected_intervals")
    if unresolved_frames:
        blocking_reasons.append("unresolved_privacy_frames")
    if invalid_boxes:
        blocking_reasons.append("invalid_boxes")

    return {
        "approved_for_final": mode == MODE_FINAL and not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "final_export_blocked": mode == MODE_FINAL and bool(blocking_reasons),
        "interval_count": len(intervals),
        "invalid_box_count": len(invalid_boxes),
        "invalid_boxes": invalid_boxes,
        "mode": mode,
        "needs_review_interval_count": len(needs_review_intervals),
        "needs_review_intervals": interval_summaries(needs_review_intervals),
        "out_of_frame_box_count": len(out_of_frame_boxes),
        "out_of_frame_boxes": out_of_frame_boxes,
        "preview_non_final": mode == MODE_PREVIEW,
        "rejected_interval_count": len(rejected_intervals),
        "rejected_intervals": interval_summaries(rejected_intervals),
        "render_allowed": mode == MODE_PREVIEW or not blocking_reasons,
        "required_frame_count": len(privacy_input.required_frames),
        "unresolved_frame_count": len(set(unresolved_frames)),
        "unresolved_frames": sorted(set(unresolved_frames)),
    }


def interval_summaries(intervals: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "frame_end": interval.get("frame_end"),
            "frame_start": interval.get("frame_start"),
            "reasons": interval.get("reasons", []),
            "status": interval.get("status"),
            "strategy": interval.get("strategy"),
            "unresolved_frames": interval.get("unresolved_frames"),
        }
        for interval in intervals
    ]


def render_video(
    input_video: Path,
    output_path: Path,
    regions_by_frame: dict[int, list[RenderRegion]],
    *,
    settings: RenderSettings,
    fps: float,
    width: int,
    height: int,
    source_has_audio: bool,
    comparison_path: Path | None,
    comparison_frames: int,
    comparison_start_frame: int,
    overwrite: bool,
) -> tuple[int, int, str]:
    video_only_path = output_path.with_name(f".{output_path.stem}.video_only.mp4")
    if video_only_path.exists():
        video_only_path.unlink()
    if output_path.exists() and overwrite:
        output_path.unlink()

    capture = cv.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_video}")
    writer = cv.VideoWriter(
        str(video_only_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create video: {video_only_path}")

    comparison_writer = None
    if comparison_path is not None:
        if comparison_path.exists() and overwrite:
            comparison_path.unlink()
        comparison_writer = cv.VideoWriter(
            str(comparison_path),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2, height),
        )
        if not comparison_writer.isOpened():
            capture.release()
            writer.release()
            raise ValueError(f"could not create comparison video: {comparison_path}")

    rendered_frames = 0
    rendered_regions = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            redacted = render_frame(frame, rendered_frames, regions_by_frame, settings)
            rendered_regions += count_renderable_regions(
                regions_by_frame.get(rendered_frames, []),
                frame_width=width,
                frame_height=height,
                settings=settings,
            )
            writer.write(redacted)
            if (
                comparison_writer is not None
                and comparison_start_frame
                <= rendered_frames
                < comparison_start_frame + comparison_frames
            ):
                comparison_writer.write(comparison_frame(frame, redacted, rendered_frames))
            rendered_frames += 1
    finally:
        capture.release()
        writer.release()
        if comparison_writer is not None:
            comparison_writer.release()

    audio_handling = mux_audio_if_needed(
        input_video,
        video_only_path,
        output_path,
        source_has_audio=source_has_audio,
    )
    if video_only_path.exists():
        video_only_path.unlink()
    return rendered_frames, rendered_regions, audio_handling


def render_frame(
    frame: np.ndarray,
    frame_index: int,
    regions_by_frame: dict[int, list[RenderRegion]],
    settings: RenderSettings,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for region in regions_by_frame.get(frame_index, []):
        if not region.is_renderable:
            continue
        box = expanded_clipped_box(
            region.bbox_xyxy,
            frame_width=width,
            frame_height=height,
            margin_pixels=settings.margin_pixels,
            margin_fraction=settings.margin_fraction,
        )
        if box is None:
            continue
        apply_redaction(output, box, settings)
    if settings.mode == MODE_PREVIEW and settings.preview_watermark:
        draw_preview_watermark(output)
    return output


def apply_redaction(frame: np.ndarray, box: Box, settings: RenderSettings) -> None:
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    if settings.method == METHOD_GAUSSIAN_BLUR:
        frame[y1:y2, x1:x2] = cv.GaussianBlur(
            roi,
            (settings.normalized_blur_kernel, settings.normalized_blur_kernel),
            0,
        )
    elif settings.method == METHOD_PIXELATE:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        small = cv.resize(
            roi,
            (max(1, width // settings.pixel_size), max(1, height // settings.pixel_size)),
            interpolation=cv.INTER_LINEAR,
        )
        frame[y1:y2, x1:x2] = cv.resize(small, (width, height), interpolation=cv.INTER_NEAREST)
    else:
        raise ValueError(f"unsupported render method: {settings.method}")


def expanded_clipped_box(
    box: Box | None,
    *,
    frame_width: int,
    frame_height: int,
    margin_pixels: int,
    margin_fraction: float,
) -> Box | None:
    if box is None or not box_has_area(box):
        return None
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    margin = margin_pixels + round(max(width, height) * margin_fraction)
    return clipped_box(
        (x1 - margin, y1 - margin, x2 + margin, y2 + margin),
        frame_width,
        frame_height,
    )


def clipped_box(box: Box, frame_width: int, frame_height: int) -> Box | None:
    x1, y1, x2, y2 = box
    clipped = (
        max(0, min(frame_width, x1)),
        max(0, min(frame_height, y1)),
        max(0, min(frame_width, x2)),
        max(0, min(frame_height, y2)),
    )
    return clipped if box_has_area(clipped) else None


def box_has_area(box: Box) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def box_outside_frame(box: Box, frame_width: int, frame_height: int) -> bool:
    return box[0] < 0 or box[1] < 0 or box[2] > frame_width or box[3] > frame_height


def count_renderable_regions(
    regions: list[RenderRegion],
    *,
    frame_width: int,
    frame_height: int,
    settings: RenderSettings,
) -> int:
    return sum(
        1
        for region in regions
        if region.is_renderable
        and expanded_clipped_box(
            region.bbox_xyxy,
            frame_width=frame_width,
            frame_height=frame_height,
            margin_pixels=settings.margin_pixels,
            margin_fraction=settings.margin_fraction,
        )
        is not None
    )


def draw_preview_watermark(frame: np.ndarray) -> None:
    cv.rectangle(frame, (8, 8), (315, 44), (0, 0, 0), -1)
    cv.putText(
        frame,
        "PREVIEW - NOT FINAL",
        (16, 33),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )


def comparison_frame(original: np.ndarray, redacted: np.ndarray, frame_index: int) -> np.ndarray:
    left = original.copy()
    right = redacted.copy()
    cv.putText(left, "original", (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv.putText(right, "redacted", (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv.putText(
        right,
        f"frame {frame_index}",
        (18, 72),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    return np.hstack([left, right])


def mux_audio_if_needed(
    input_video: Path,
    video_only_path: Path,
    output_path: Path,
    *,
    source_has_audio: bool,
) -> str:
    if not source_has_audio:
        video_only_path.replace(output_path)
        return "source_had_no_audio_video_only_output"

    command = [
        media.executable_path("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return "source_audio_muxed_from_input"


def validate_output_media(
    input_video: Path,
    source_info: MediaInfo,
    output_path: Path,
    *,
    expected_frame_count: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    source_has_audio: bool,
    settings: RenderSettings,
) -> dict[str, object]:
    output_info = media.probe_media(output_path)
    output_frames = _optional_int(output_info.video_stream.frame_count)
    output_fps = rational_to_float(output_info.video_stream.nominal_frame_rate) or 0.0
    source_duration = source_info.duration_seconds
    output_duration = output_info.duration_seconds
    frame_count_ok = (
        output_frames is None
        or abs(output_frames - expected_frame_count) <= settings.frame_count_tolerance
    )
    duration_delta = (
        None
        if source_duration is None or output_duration is None
        else abs(output_duration - source_duration)
    )
    duration_ok = duration_delta is None or duration_delta <= max(
        settings.duration_tolerance_seconds, 2.0 / max(expected_fps, 1.0)
    )
    fps_ok = abs(output_fps - expected_fps) <= 0.01
    resolution_ok = (
        output_info.video_stream.width == expected_width
        and output_info.video_stream.height == expected_height
    )
    audio_ok = (output_info.audio_stream is not None) if source_has_audio else True
    source_audio_duration = audio_duration_seconds(input_video) if source_has_audio else None
    output_audio_duration = audio_duration_seconds(output_path) if source_has_audio else None
    audio_duration_delta = (
        None
        if source_audio_duration is None or output_audio_duration is None
        else abs(output_audio_duration - source_audio_duration)
    )
    audio_duration_ok = (
        True
        if not source_has_audio or audio_duration_delta is None
        else audio_duration_delta
        <= max(settings.duration_tolerance_seconds, 2.0 / max(expected_fps, 1.0))
    )
    return {
        "audio_ok": audio_ok,
        "audio_duration_delta_seconds": None
        if audio_duration_delta is None
        else round(audio_duration_delta, 6),
        "audio_duration_ok": audio_duration_ok,
        "duration_delta_seconds": None if duration_delta is None else round(duration_delta, 6),
        "duration_ok": duration_ok,
        "expected_audio": source_has_audio,
        "expected_frame_count": expected_frame_count,
        "expected_fps": round(expected_fps, 6),
        "expected_height": expected_height,
        "expected_width": expected_width,
        "frame_count_ok": frame_count_ok,
        "fps_ok": fps_ok,
        "output_audio_present": output_info.audio_stream is not None,
        "output_audio_duration_seconds": output_audio_duration,
        "output_duration_seconds": output_duration,
        "output_frame_count": output_frames,
        "output_fps": round(output_fps, 6),
        "output_height": output_info.video_stream.height,
        "output_width": output_info.video_stream.width,
        "passed": all(
            [frame_count_ok, fps_ok, resolution_ok, duration_ok, audio_ok, audio_duration_ok]
        ),
        "resolution_ok": resolution_ok,
        "source_audio_duration_seconds": source_audio_duration,
    }


def audio_duration_seconds(path: Path) -> float | None:
    command = [
        media.executable_path("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        return None
    return _optional_float(streams[0].get("duration"))


def output_payloads(
    *,
    input_video: Path,
    privacy_regions_path: Path,
    interval_decisions_path: Path | None,
    source_info: MediaInfo,
    output_path: Path,
    settings: RenderSettings,
    privacy_input: PrivacyRegionInput,
    intervals: list[dict[str, object]],
    validation: dict[str, object],
    media_validation: dict[str, object] | None,
    rendered_frame_count: int,
    rendered_region_count: int,
    output_sha256: str | None,
    audio_handling: str,
    elapsed_seconds: float,
) -> dict[str, dict[str, object]]:
    provenance_counts = provenance_counts_for(privacy_input)
    final_approved = settings.mode == MODE_FINAL and bool(validation["approved_for_final"])
    manifest = {
        "audio_handling": audio_handling,
        "blur_method": settings.method,
        "blur_parameters": {
            "blur_kernel": settings.normalized_blur_kernel,
            "margin_fraction": settings.margin_fraction,
            "margin_pixels": settings.margin_pixels,
            "pixel_size": settings.pixel_size,
        },
        "detector_id": DETECTOR_BACKEND_ADAS,
        "final_approval_status": final_approved,
        "interval_decisions_hash": media.sha256_file(interval_decisions_path)
        if interval_decisions_path is not None
        else None,
        "model_id": "OpenVINO face-detection-adas-0001",
        "output_video_hash": output_sha256,
        "output_video": output_path.name,
        "pipeline_id": PIPELINE_ID,
        "preview_non_final": settings.mode == MODE_PREVIEW,
        "privacy_region_input_hash": media.sha256_file(privacy_regions_path),
        "provenance_counts": provenance_counts,
        "rendered_frame_count": rendered_frame_count,
        "rendered_region_count": rendered_region_count,
        "renderer_id": RENDERER_ID,
        "rendering_mode": settings.mode,
        "review_status": {
            "blocking_reasons": validation["blocking_reasons"],
            "needs_review_interval_count": validation["needs_review_interval_count"],
            "rejected_interval_count": validation["rejected_interval_count"],
            "unresolved_frame_count": validation["unresolved_frame_count"],
        },
        "schema_version": SCHEMA_VERSION,
        "source_video_hash": source_info.sha256,
        "source_video": input_video.name,
        "unresolved_intervals": validation["needs_review_intervals"]
        + validation["rejected_intervals"],
    }
    summary = {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "final_approval_status": final_approved,
        "media_validation_passed": None if media_validation is None else media_validation["passed"],
        "output_video": output_path.name if output_sha256 is not None else None,
        "preview_non_final": settings.mode == MODE_PREVIEW,
        "processed_frame_count": rendered_frame_count,
        "provenance_counts": provenance_counts,
        "rendered_region_count": rendered_region_count,
        "rendering_mode": settings.mode,
        "schema_version": SCHEMA_VERSION,
        "source_video_frame_count": _optional_int(source_info.video_stream.frame_count),
        "validation_blocking_reasons": validation["blocking_reasons"],
    }
    validation_payload = {
        "media_validation": media_validation,
        "render_plan_validation": validation,
        "schema_version": SCHEMA_VERSION,
    }
    return {
        "manifest": manifest,
        "summary": summary,
        "validation": validation_payload,
    }


def provenance_counts_for(privacy_input: PrivacyRegionInput) -> dict[str, int]:
    counts: dict[str, int] = {}
    for regions in privacy_input.regions_by_frame.values():
        for region in regions:
            if not region.is_renderable:
                continue
            counts[region.provenance] = counts.get(region.provenance, 0) + 1
    return dict(sorted(counts.items()))


def write_payloads(output_dir: Path, payload: dict[str, dict[str, object]]) -> None:
    write_json(output_dir / "render_manifest.json", payload["manifest"])
    write_json(output_dir / "render_summary.json", payload["summary"])
    write_json(output_dir / "validation.json", payload["validation"])


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_event(path: Path, event: str, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {"event": event, "payload": payload, "timestamp": media.utc_timestamp()},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
