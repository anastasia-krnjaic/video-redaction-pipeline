from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import media, rendering
from redactor.detection import rational_to_float
from redactor.privacy_records import PrivacyRegionRecord

SCHEMA_VERSION = "privacy_rendering_core.v1"
AUDIO_POLICY_VIDEO_DURATION_AUTHORITATIVE = "video_duration_authoritative"
FULL_SPAN_TIMESTAMP_TOLERANCE_SECONDS = 0.05

Box = tuple[int, int, int, int]


def regions_by_frame(
    records: list[PrivacyRegionRecord],
) -> dict[int, list[rendering.RenderRegion]]:
    grouped: dict[int, list[rendering.RenderRegion]] = {}
    for record in sorted(
        records,
        key=lambda item: (
            item.source_frame,
            item.category,
            render_order(item),
            item.object_id or "",
            item.provenance,
            item.bbox_xyxy or (-1, -1, -1, -1),
        ),
    ):
        grouped.setdefault(record.source_frame, []).append(record.to_render_region())
    return grouped


def render_order(record: PrivacyRegionRecord) -> int:
    value = record.source_record.get("render_order", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def provenance_counts(records: list[PrivacyRegionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.provenance] = counts.get(record.provenance, 0) + 1
    return dict(sorted(counts.items()))


def sequential_video_summary(path: Path) -> dict[str, Any]:
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")

    frame_count = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    width: int | None = None
    height: int | None = None
    max_pixel_value = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv.CAP_PROP_POS_MSEC)
            timestamp_seconds = round(timestamp / 1000.0, 6) if np.isfinite(timestamp) else None
            if frame_count == 0:
                first_timestamp = timestamp_seconds
                height, width = frame.shape[:2]
            last_timestamp = timestamp_seconds
            max_pixel_value = max(max_pixel_value, int(frame.max()))
            frame_count += 1
    finally:
        capture.release()

    return {
        "decoded_frame_count": frame_count,
        "first_timestamp": first_timestamp,
        "has_non_black_frame": max_pixel_value > 0,
        "height": height,
        "last_timestamp": last_timestamp,
        "max_pixel_value": max_pixel_value,
        "path": path.as_posix(),
        "width": width,
    }


def validate_full_span_output(
    source_video: Path,
    output_video: Path,
    *,
    expected_frame_count: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    source_has_audio: bool,
) -> dict[str, Any]:
    source_summary = sequential_video_summary(source_video)
    output_summary = sequential_video_summary(output_video)
    output_info = media.probe_media(output_video)
    output_fps = rational_to_float(output_info.video_stream.nominal_frame_rate) or 0.0
    source_count_ok = source_summary["decoded_frame_count"] == expected_frame_count
    output_count_ok = output_summary["decoded_frame_count"] == expected_frame_count
    dimensions_ok = (
        output_summary["width"] == expected_width
        and output_summary["height"] == expected_height
        and output_info.video_stream.width == expected_width
        and output_info.video_stream.height == expected_height
    )
    fps_ok = abs(output_fps - expected_fps) <= 0.01
    first_timestamp_ok = _timestamps_close(
        source_summary["first_timestamp"],
        output_summary["first_timestamp"],
    )
    last_timestamp_ok = _timestamps_close(
        source_summary["last_timestamp"],
        output_summary["last_timestamp"],
    )
    audio_ok = (output_info.audio_stream is not None) if source_has_audio else True
    non_black_ok = bool(output_summary["has_non_black_frame"]) or not bool(
        source_summary["has_non_black_frame"]
    )
    passed = all(
        [
            source_count_ok,
            output_count_ok,
            dimensions_ok,
            fps_ok,
            first_timestamp_ok,
            last_timestamp_ok,
            audio_ok,
            non_black_ok,
        ]
    )
    return {
        "audio_ok": audio_ok,
        "dimensions_ok": dimensions_ok,
        "expected_frame_count": expected_frame_count,
        "expected_fps": round(expected_fps, 6),
        "expected_height": expected_height,
        "expected_width": expected_width,
        "first_timestamp_ok": first_timestamp_ok,
        "fps_ok": fps_ok,
        "last_timestamp_ok": last_timestamp_ok,
        "non_black_ok": non_black_ok,
        "output_audio_present": output_info.audio_stream is not None,
        "output_summary": output_summary,
        "passed": passed,
        "source_count_ok": source_count_ok,
        "source_summary": source_summary,
        "timestamp_tolerance_seconds": FULL_SPAN_TIMESTAMP_TOLERANCE_SECONDS,
        "video_duration_authoritative": True,
    }


def render_privacy_video(
    source_video: Path,
    output_video: Path,
    regions: dict[int, list[rendering.RenderRegion]],
    *,
    settings: rendering.RenderSettings,
    fps: float,
    width: int,
    height: int,
    source_has_audio: bool,
    overwrite: bool,
    log_path: Path | None = None,
    video_codec: str = "libx264",
    pixel_format: str = "yuv420p",
    movflags: str = "+faststart",
    partial_processing_label: bool = False,
) -> tuple[int, int, str]:
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"{output_video} already exists; pass --overwrite to replace it")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    video_only_path = output_video.with_name(f".{output_video.stem}.video_only.mp4")
    h264_path = output_video.with_name(f".{output_video.stem}.h264.mp4")
    for path in (video_only_path, h264_path):
        if path.exists():
            path.unlink()

    capture = cv.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source_video}")
    writer = cv.VideoWriter(
        str(video_only_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create video: {video_only_path}")

    rendered_frames = 0
    rendered_regions = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            redacted = rendering.render_frame(frame, rendered_frames, regions, settings)
            if partial_processing_label:
                draw_partial_processing_label(redacted)
            rendered_regions += rendering.count_renderable_regions(
                regions.get(rendered_frames, []),
                frame_width=width,
                frame_height=height,
                settings=settings,
            )
            writer.write(redacted)
            rendered_frames += 1
    finally:
        capture.release()
        writer.release()

    audio_handling = encode_browser_mp4(
        source_video,
        video_only_path,
        h264_path,
        source_has_audio=source_has_audio,
        video_codec=video_codec,
        pixel_format=pixel_format,
        movflags=movflags,
    )
    h264_path.replace(output_video)
    if video_only_path.exists():
        video_only_path.unlink()
    if log_path is not None:
        append_jsonl(
            log_path,
            {
                "audio_handling": audio_handling,
                "event": "render_finished",
                "output_video": output_video.name,
                "rendered_frames": rendered_frames,
                "rendered_regions": rendered_regions,
                "schema_version": SCHEMA_VERSION,
            },
        )
    return rendered_frames, rendered_regions, audio_handling


def _timestamps_close(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= FULL_SPAN_TIMESTAMP_TOLERANCE_SECONDS


def encode_browser_mp4(
    source_video: Path,
    video_only_path: Path,
    output_path: Path,
    *,
    source_has_audio: bool,
    video_codec: str,
    pixel_format: str,
    movflags: str,
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
            video_codec,
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            pixel_format,
            "-movflags",
            movflags,
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
    regions: list[rendering.RenderRegion],
    *,
    settings: rendering.RenderSettings,
    frame_width: int,
    frame_height: int,
    max_failures_reported: int = 50,
) -> dict[str, Any]:
    expected_by_frame: dict[int, list[rendering.RenderRegion]] = {}
    for region in regions:
        if region.is_renderable:
            expected_by_frame.setdefault(region.frame_index, []).append(region)
    expected_frames = set(expected_by_frame)
    if not expected_frames:
        return {
            "expected_region_count": 0,
            "expected_region_frames": 0,
            "missing_rendered_region_count": 0,
            "missing_rendered_regions": [],
            "passed": True,
            "verified_region_count": 0,
            "verified_region_frames": 0,
        }

    source_capture = cv.VideoCapture(str(source_video))
    output_capture = cv.VideoCapture(str(output_video))
    if not source_capture.isOpened() or not output_capture.isOpened():
        source_capture.release()
        output_capture.release()
        raise ValueError("could not open source or rendered video for region verification")

    missing: list[dict[str, Any]] = []
    verified_regions = 0
    verified_frames: set[int] = set()
    frame_index = 0
    try:
        while True:
            source_ok, source_frame = source_capture.read()
            output_ok, output_frame = output_capture.read()
            if not source_ok or not output_ok:
                break
            if frame_index in expected_frames:
                frame_has_verified_region = False
                for region in expected_by_frame[frame_index]:
                    box = rendering.expanded_clipped_box(
                        region.bbox_xyxy,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        margin_pixels=settings.margin_pixels,
                        margin_fraction=settings.margin_fraction,
                    )
                    if box is None:
                        append_missing(missing, region, "invalid_expanded_box", None)
                        continue
                    metrics = localized_change_metrics(source_frame, output_frame, box)
                    if metrics["region_changed"]:
                        verified_regions += 1
                        frame_has_verified_region = True
                    else:
                        append_missing(
                            missing,
                            region,
                            "expected_region_not_visibly_redacted",
                            metrics | {"expanded_bbox_xyxy": list(box)},
                        )
                if frame_has_verified_region:
                    verified_frames.add(frame_index)
            frame_index += 1
    finally:
        source_capture.release()
        output_capture.release()

    for expected_frame in sorted(frame for frame in expected_frames if frame >= frame_index):
        for region in expected_by_frame[expected_frame]:
            append_missing(missing, region, "expected_frame_not_decoded", None)

    expected_region_count = sum(len(items) for items in expected_by_frame.values())
    missing_count = max(0, expected_region_count - verified_regions)
    return {
        "expected_region_count": expected_region_count,
        "expected_region_frames": len(expected_frames),
        "missing_rendered_region_count": missing_count,
        "missing_rendered_regions": missing[:max_failures_reported],
        "passed": missing_count == 0,
        "verified_region_count": verified_regions,
        "verified_region_frames": len(verified_frames),
    }


def localized_change_metrics(
    original: np.ndarray,
    rendered: np.ndarray,
    box: Box,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    original_roi = original[y1:y2, x1:x2]
    rendered_roi = rendered[y1:y2, x1:x2]
    if original_roi.size == 0 or rendered_roi.size == 0:
        return {
            "changed_pixel_fraction": 0.0,
            "mean_absdiff": 0.0,
            "region_changed": False,
        }
    diff = cv.absdiff(original_roi, rendered_roi)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    changed_fraction = float(
        np.count_nonzero(gray > rendering.REDACTION_DIFF_VISIBILITY_THRESHOLD) / gray.size
    )
    mean_absdiff = float(gray.mean())
    return {
        "changed_pixel_fraction": round(changed_fraction, 6),
        "mean_absdiff": round(mean_absdiff, 6),
        "region_changed": rendering.region_changed_enough(original_roi, rendered_roi),
    }


def append_missing(
    missing: list[dict[str, Any]],
    region: rendering.RenderRegion,
    reason: str,
    metrics: dict[str, Any] | None,
) -> None:
    source = region.source_record
    item = {
        "bbox_xyxy": list(region.bbox_xyxy) if region.bbox_xyxy is not None else None,
        "category": source.get("category"),
        "frame_index": region.frame_index,
        "object_id": source.get("object_id") or source.get("track_uid"),
        "reason": reason,
        "track_uid": source.get("track_uid"),
    }
    if metrics:
        item.update(metrics)
    missing.append(item)


def draw_partial_processing_label(frame: np.ndarray) -> None:
    cv.rectangle(frame, (16, 16), (470, 58), (0, 0, 0), -1)
    cv.putText(
        frame,
        "PARTIAL PROCESSING - NOT FINAL",
        (28, 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
