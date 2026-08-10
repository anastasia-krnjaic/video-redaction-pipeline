from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np

from redactor import face_rendering, media, privacy_rendering, rendering


def write_synthetic_video(path: Path, *, frame_count: int) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 80))
    if not writer.isOpened():
        raise RuntimeError("could not create synthetic video")
    try:
        for frame_index in range(frame_count):
            frame = np.full((80, 120, 3), 20 + frame_index * 12, dtype=np.uint8)
            cv.rectangle(frame, (10, 10), (36, 36), (230, 230, 230), -1)
            writer.write(frame)
    finally:
        writer.release()


def write_face_regions(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "bbox_xyxy": [10, 10, 36, 36],
                "frame_index": 1,
                "provenance": "detector",
                "state": "covered",
                "track_id": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def preview_settings() -> rendering.RenderSettings:
    return rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_fraction=0.15,
        margin_pixels=0,
        pixel_size=16,
    )


def test_shared_face_preview_writes_legacy_paths_with_strict_span_validation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    regions = tmp_path / "privacy_regions.jsonl"
    output_dir = tmp_path / "automatic_preview"
    write_synthetic_video(source, frame_count=3)
    write_face_regions(regions)

    summary = face_rendering.run_shared_face_preview(
        source,
        privacy_regions_path=regions,
        output_dir=output_dir,
        settings=preview_settings(),
    )

    output = output_dir / "redacted_preview.mp4"
    manifest = json.loads((output_dir / "render_manifest.json").read_text())
    validation = json.loads((output_dir / "validation.json").read_text())
    span = validation["automatic_preview_full_span_validation"]
    info = media.probe_media(output)

    assert output.exists()
    assert summary["renderer_backend"] == face_rendering.RENDERER_BACKEND
    assert summary["strict_full_span_validation_passed"] is True
    assert manifest["output_video"] == "redacted_preview.mp4"
    assert manifest["effective_regions_hash"] == manifest["privacy_region_input_hash"]
    assert manifest["renderer_backend"] == face_rendering.RENDERER_BACKEND
    assert manifest["rendered_frame_count"] == 3
    assert manifest["rendered_region_count"] == 1
    assert validation["rendered_region_validation"]["passed"] is True
    assert span["passed"] is True
    assert span["source_summary"]["decoded_frame_count"] == 3
    assert span["output_summary"]["decoded_frame_count"] == 3
    assert info.video_stream.codec == "h264"
    assert info.video_stream.pixel_format == "yuv420p"


def test_full_span_validation_rejects_missing_terminal_frame(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    write_synthetic_video(source, frame_count=3)
    write_synthetic_video(output, frame_count=2)

    validation = privacy_rendering.validate_full_span_output(
        source,
        output,
        expected_frame_count=3,
        expected_width=120,
        expected_height=80,
        expected_fps=10.0,
        source_has_audio=False,
    )

    assert validation["passed"] is False
    assert validation["output_summary"]["decoded_frame_count"] == 2


def test_full_span_validation_rejects_extra_terminal_frame(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    write_synthetic_video(source, frame_count=3)
    write_synthetic_video(output, frame_count=4)

    validation = privacy_rendering.validate_full_span_output(
        source,
        output,
        expected_frame_count=3,
        expected_width=120,
        expected_height=80,
        expected_fps=10.0,
        source_has_audio=False,
    )

    assert validation["passed"] is False
    assert validation["output_summary"]["decoded_frame_count"] == 4
