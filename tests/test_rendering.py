from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from redactor import privacy_rendering, rendering
from redactor.models import MediaInfo, VideoStreamInfo


def checker_frame(height: int = 48, width: int = 64) -> np.ndarray:
    y, x = np.indices((height, width))
    base = ((x * 17 + y * 29) % 256).astype(np.uint8)
    return np.dstack([base, np.roll(base, 3, axis=1), np.roll(base, 5, axis=0)])


def region(
    frame_index: int,
    bbox: tuple[int, int, int, int] | None = (10, 10, 30, 30),
    *,
    provenance: str = "synthetic_privacy_bridge",
    state: str = "covered",
    reasons: tuple[str, ...] = (),
) -> rendering.RenderRegion:
    return rendering.RenderRegion(
        frame_index=frame_index,
        bbox_xyxy=bbox,
        provenance=provenance,
        state=state,
        source_record={"frame_index": frame_index},
        reasons=reasons,
    )


def privacy_input(*regions: rendering.RenderRegion) -> rendering.PrivacyRegionInput:
    by_frame: dict[int, list[rendering.RenderRegion]] = {}
    for item in regions:
        by_frame.setdefault(item.frame_index, []).append(item)
    return rendering.PrivacyRegionInput(
        regions_by_frame=by_frame,
        required_frames=tuple(sorted(by_frame)),
        schema_hint="test",
    )


def media_info_for(path: Path) -> MediaInfo:
    return MediaInfo(
        schema_version="1.0",
        created_at="2026-01-01T00:00:00Z",
        source_filename=path.name,
        source_path=path.name,
        file_size_bytes=123,
        sha256="1" * 64,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=1.0,
        video_stream=VideoStreamInfo(
            index=0,
            codec="h264",
            width=64,
            height=48,
            pixel_format="yuv420p",
            nominal_frame_rate="30/1",
            average_frame_rate="30/1",
            time_base="1/30",
            frame_count="30",
        ),
        audio_stream=None,
        ffprobe_version="ffprobe version test",
    )


def test_gaussian_blur_changes_only_rendered_region() -> None:
    frame = checker_frame()
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_GAUSSIAN_BLUR,
        blur_kernel=7,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(frame, 0, {0: [region(0)]}, settings)

    assert not np.array_equal(rendered[10:30, 10:30], frame[10:30, 10:30])
    assert np.array_equal(rendered[:8, :8], frame[:8, :8])


def test_pixelation_changes_region_into_blocks() -> None:
    frame = checker_frame()
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=8,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(frame, 0, {0: [region(0, (8, 8, 40, 40))]}, settings)

    assert not np.array_equal(rendered[8:40, 8:40], frame[8:40, 8:40])
    assert rendered[8, 8].tolist() == rendered[9, 9].tolist()


def test_pixelation_enforces_visible_redaction_on_low_texture_roi() -> None:
    frame = np.full((80, 120, 3), 24, dtype=np.uint8)
    gradient = np.linspace(0, 4, 40, dtype=np.uint8)[:, None]
    roi = np.repeat(24 + gradient, 60, axis=1)
    frame[20:60, 30:90] = np.dstack([roi, roi, roi])
    box = (30, 20, 90, 60)
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=16,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(frame, 0, {0: [region(0, box)]}, settings)

    metrics = privacy_rendering.localized_change_metrics(frame, rendered, box)
    assert metrics["changed_pixel_fraction"] == 1.0
    assert metrics["mean_absdiff"] >= rendering.MIN_REDACTION_MEAN_ABSDIFF
    assert metrics["region_changed"] is True
    assert np.array_equal(rendered[:20, :], frame[:20, :])
    assert np.array_equal(rendered[60:, :], frame[60:, :])
    assert np.array_equal(rendered[:, :30], frame[:, :30])
    assert np.array_equal(rendered[:, 90:], frame[:, 90:])


def test_pixelation_fallback_adds_codec_headroom_for_barely_changed_roi() -> None:
    frame = np.full((80, 120, 3), 56, dtype=np.uint8)
    frame[38:40, 30:90] = 92
    box = (30, 20, 90, 60)
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=16,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(frame, 0, {0: [region(0, box)]}, settings)
    metrics = privacy_rendering.localized_change_metrics(frame, rendered, box)

    assert metrics["changed_pixel_fraction"] >= (
        rendering.PIXELATE_MIN_DIRECT_CHANGED_PIXEL_FRACTION
    )
    assert metrics["mean_absdiff"] >= rendering.PIXELATE_MIN_DIRECT_MEAN_ABSDIFF
    assert metrics["region_changed"] is True


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 17, 19),
        (-4, -3, 12, 14),
        (48, 30, 64, 48),
        (7, 9, 23, 28),
    ],
)
def test_pixelation_privacy_signal_handles_neighboring_geometry_edges(
    box: tuple[int, int, int, int],
) -> None:
    frame = np.full((48, 64, 3), 130, dtype=np.uint8)
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=16,
        preview_watermark=False,
    )
    expected = rendering.expanded_clipped_box(
        box,
        frame_width=64,
        frame_height=48,
        margin_pixels=0,
        margin_fraction=0.0,
    )

    rendered = rendering.render_frame(frame, 0, {0: [region(0, box)]}, settings)

    assert expected is not None
    assert privacy_rendering.localized_change_metrics(frame, rendered, expected)[
        "region_changed"
    ] is True


def test_pixelation_privacy_signal_handles_consecutive_and_overlapping_regions() -> None:
    frame0 = np.full((48, 64, 3), 32, dtype=np.uint8)
    frame1 = np.full((48, 64, 3), 36, dtype=np.uint8)
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=16,
        preview_watermark=False,
    )
    regions = {
        0: [region(0, (8, 8, 30, 30)), region(0, (20, 20, 42, 42))],
        1: [region(1, (10, 9, 32, 31))],
    }

    rendered0 = rendering.render_frame(frame0, 0, regions, settings)
    rendered1 = rendering.render_frame(frame1, 1, regions, settings)

    assert privacy_rendering.localized_change_metrics(frame0, rendered0, (8, 8, 30, 30))[
        "region_changed"
    ] is True
    assert privacy_rendering.localized_change_metrics(frame0, rendered0, (20, 20, 42, 42))[
        "region_changed"
    ] is True
    assert privacy_rendering.localized_change_metrics(frame1, rendered1, (10, 9, 32, 31))[
        "region_changed"
    ] is True


def test_overlapping_pixelation_renders_broad_regions_before_specific_regions() -> None:
    frame = np.full((80, 120, 3), 24, dtype=np.uint8)
    small = (62, 34, 92, 58)
    broad = (0, 0, 120, 80)
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=16,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(
        frame,
        0,
        {
            0: [
                region(0, small, provenance="specific_privacy_region"),
                region(0, broad, provenance="broad_overlapping_privacy_region"),
            ]
        },
        settings,
    )

    assert privacy_rendering.localized_change_metrics(frame, rendered, small)[
        "region_changed"
    ] is True
    assert privacy_rendering.localized_change_metrics(frame, rendered, broad)[
        "region_changed"
    ] is True


def test_box_clipping_and_privacy_margin() -> None:
    frame = checker_frame()
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_GAUSSIAN_BLUR,
        margin_pixels=3,
        blur_kernel=5,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(frame, 0, {0: [region(0, (-3, -3, 8, 8))]}, settings)

    assert not np.array_equal(rendered[0:8, 0:8], frame[0:8, 0:8])
    assert not np.array_equal(rendered[9:11, 9:11], frame[9:11, 9:11])


def test_multiple_regions_on_one_frame() -> None:
    frame = checker_frame()
    settings = rendering.RenderSettings(
        mode=rendering.MODE_FINAL,
        method=rendering.METHOD_PIXELATE,
        pixel_size=5,
        preview_watermark=False,
    )

    rendered = rendering.render_frame(
        frame,
        0,
        {0: [region(0, (4, 4, 18, 18)), region(0, (34, 24, 50, 42))]},
        settings,
    )

    assert not np.array_equal(rendered[4:18, 4:18], frame[4:18, 4:18])
    assert not np.array_equal(rendered[24:42, 34:50], frame[24:42, 34:50])
    assert np.array_equal(rendered[20:24, 20:24], frame[20:24, 20:24])


def test_empty_frame_is_unchanged_when_watermark_disabled() -> None:
    frame = checker_frame()
    settings = rendering.RenderSettings(mode=rendering.MODE_PREVIEW, preview_watermark=False)

    rendered = rendering.render_frame(frame, 0, {}, settings)

    assert np.array_equal(rendered, frame)


def test_preview_allows_needs_review_but_is_non_final() -> None:
    validation = rendering.validate_render_plan(
        privacy_input(region(0)),
        [{"frame_start": 0, "frame_end": 3, "status": "needs_review"}],
        mode=rendering.MODE_PREVIEW,
        frame_width=64,
        frame_height=48,
    )

    assert validation["render_allowed"] is True
    assert validation["approved_for_final"] is False
    assert validation["preview_non_final"] is True
    assert validation["blocking_reasons"] == ["needs_review_intervals"]


def test_final_export_blocked_by_needs_review_interval() -> None:
    validation = rendering.validate_render_plan(
        privacy_input(region(0)),
        [{"frame_start": 0, "frame_end": 3, "status": "needs_review"}],
        mode=rendering.MODE_FINAL,
        frame_width=64,
        frame_height=48,
    )

    assert validation["render_allowed"] is False
    assert validation["final_export_blocked"] is True
    assert "needs_review_intervals" in validation["blocking_reasons"]


def test_final_export_blocked_by_unresolved_frame() -> None:
    unresolved_input = privacy_input(region(0, None, state="unresolved"))

    validation = rendering.validate_render_plan(
        unresolved_input,
        [],
        mode=rendering.MODE_FINAL,
        frame_width=64,
        frame_height=48,
    )

    assert validation["render_allowed"] is False
    assert validation["unresolved_frame_count"] == 1
    assert "unresolved_privacy_frames" in validation["blocking_reasons"]


def test_final_export_allowed_with_preserved_reviewed_corrections() -> None:
    validation = rendering.validate_render_plan(
        privacy_input(region(0, provenance="preserved_reviewed_correction")),
        [{"frame_start": 0, "frame_end": 0, "status": "accepted_automatically"}],
        mode=rendering.MODE_FINAL,
        frame_width=64,
        frame_height=48,
    )

    assert validation["approved_for_final"] is True
    assert validation["blocking_reasons"] == []


def test_provenance_counts_only_renderable_regions() -> None:
    counts = rendering.provenance_counts_for(
        privacy_input(
            region(0, provenance="ordinary_tracker"),
            region(1, provenance="synthetic_privacy_bridge"),
            region(2, None, provenance="synthetic_privacy_bridge", state="unresolved"),
        )
    )

    assert counts == {"ordinary_tracker": 1, "synthetic_privacy_bridge": 1}


def test_manifest_generation_is_deterministic(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    privacy_path = tmp_path / "privacy_regions.jsonl"
    intervals_path = tmp_path / "intervals.json"
    output_path = tmp_path / "redacted.mp4"
    source_path.write_bytes(b"source")
    privacy_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "privacy_regions": [
                    {
                        "bbox_xyxy": [10, 10, 30, 30],
                        "provenance": "ordinary_tracker",
                        "state": "covered",
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    intervals_path.write_text(
        json.dumps({"interval_decisions": []}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_info = media_info_for(source_path)
    loaded_input = rendering.load_privacy_regions(privacy_path)
    settings = rendering.RenderSettings(mode=rendering.MODE_FINAL, preview_watermark=False)
    validation = rendering.validate_render_plan(
        loaded_input,
        [],
        mode=rendering.MODE_FINAL,
        frame_width=64,
        frame_height=48,
    )

    kwargs = {
        "input_video": source_path,
        "privacy_regions_path": privacy_path,
        "interval_decisions_path": intervals_path,
        "source_info": source_info,
        "output_path": output_path,
        "settings": settings,
        "privacy_input": loaded_input,
        "intervals": [],
        "validation": validation,
        "media_validation": {"passed": True},
        "rendered_frame_count": 30,
        "rendered_region_count": 1,
        "output_sha256": "2" * 64,
        "audio_handling": "source_had_no_audio_video_only_output",
        "elapsed_seconds": 1.234567,
    }

    first = rendering.output_payloads(**kwargs)
    second = rendering.output_payloads(**kwargs)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["manifest"]["source_video"] == "source.mp4"
    assert first["manifest"]["privacy_region_input_hash"]
