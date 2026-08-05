from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np
from click.testing import CliRunner

from redactor import cli as redactor_cli
from redactor import frame_spans, media, plate_detection, plate_rendering, plate_tracking, rendering


def track_row(frame_index: int, tracks: list[dict[str, object]]) -> dict[str, object]:
    return {"frame_index": frame_index, "tracks": tracks}


def track(
    *,
    status: str = plate_tracking.ACCEPTED_TRACK,
    provenance: str = plate_tracking.PROVENANCE_DETECTOR,
    bbox: tuple[int, int, int, int] = (20, 20, 70, 40),
    uid: str = "plate_track_0001",
) -> dict[str, object]:
    return {
        "bbox_xyxy": list(bbox),
        "category": plate_detection.CATEGORY,
        "frame_index": 0,
        "provenance": provenance,
        "score": 0.8,
        "status": status,
        "track_id": 1,
        "track_uid": uid,
    }


def write_tracks(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_review_intervals(path: Path, intervals: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(intervals) + "\n", encoding="utf-8")


def write_synthetic_video(path: Path, *, frame_count: int = 5) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 80))
    try:
        for frame_index in range(frame_count):
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            frame[:, :] = (25 + frame_index * 18, 60, 105)
            cv.rectangle(frame, (18 + frame_index, 20), (74 + frame_index, 42), (250, 250, 250), -1)
            cv.putText(
                frame,
                "ABC",
                (24 + frame_index, 36),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()


def test_render_eligibility_uses_only_accepted_observations_and_interpolation() -> None:
    rows = [
        track_row(
            0,
            [
                track(provenance=plate_tracking.PROVENANCE_DETECTOR),
                track(provenance=plate_tracking.PROVENANCE_INTERPOLATION, uid="plate_track_0002"),
                track(provenance=plate_tracking.PROVENANCE_PREDICTION, uid="plate_track_0003"),
                track(status=plate_tracking.NEEDS_REVIEW, uid="plate_track_0004"),
                track(status=plate_tracking.REJECTED_NOISE, uid="plate_track_0005"),
            ],
        )
    ]

    eligible = plate_rendering.eligible_regions_from_rows(
        rows,
        config=plate_rendering.default_config(),
    )

    assert [region.track_uid for region in eligible] == [
        "plate_track_0001",
        "plate_track_0002",
    ]
    assert {region.provenance for region in eligible} == {
        plate_tracking.PROVENANCE_DETECTOR,
        plate_tracking.PROVENANCE_INTERPOLATION,
    }


def test_unresolved_gap_blocks_reviewed_final_but_allows_preview() -> None:
    config = plate_rendering.default_config()
    eligible = [
        plate_rendering.PlateRenderRegion(
            frame_index=0,
            bbox_xyxy=(20, 20, 70, 40),
            provenance=plate_tracking.PROVENANCE_DETECTOR,
            status=plate_tracking.ACCEPTED_TRACK,
            track_uid="plate_track_0001",
            track_id=1,
            score=0.8,
            source_record={},
        )
    ]
    intervals = [
        {
            "blocking": True,
            "frame_start": 2,
            "frame_end": 5,
            "reason_codes": ["gap_exceeds_safe_interpolation_limit"],
        }
    ]

    preview = plate_rendering.validate_plate_render_plan(
        eligible,
        intervals,
        mode=plate_rendering.MODE_PREVIEW,
        frame_width=120,
        frame_height=80,
        processed_frame_count=10,
        source_frame_count=10,
    )
    final = plate_rendering.validate_plate_render_plan(
        eligible,
        intervals,
        mode=plate_rendering.MODE_REVIEWED_FINAL,
        frame_width=120,
        frame_height=80,
        processed_frame_count=10,
        source_frame_count=10,
    )

    assert preview["render_allowed"] is True
    assert preview["full_source_processed"] is True
    assert preview["preview_non_final"] is True
    assert final["approved_for_final"] is False
    assert "plate_review_intervals" in final["reasons"]
    assert config.configuration_hash() == plate_rendering.default_config().configuration_hash()


def test_frame_span_reports_missing_source_ranges() -> None:
    report = frame_spans.build_frame_span_report([0, 1, 4], source_frame_count=6)

    assert report["full_source_processed"] is False
    assert report["processed_frame_count"] == 3
    assert report["processed_frame_span_count"] == 5
    assert report["missing_processed_ranges"] == [
        {"frame_start": 2, "frame_end": 3, "length": 2},
        {"frame_start": 5, "frame_end": 5, "length": 1},
    ]


def test_partial_tracking_blocks_default_preview(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, frame_count=5)
    tracks_path = tmp_path / "tracks.jsonl"
    write_tracks(
        tracks_path,
        [track_row(index, [track(bbox=(18 + index, 20, 74 + index, 42))]) for index in range(3)],
    )
    intervals_path = tmp_path / "review_intervals.json"
    write_review_intervals(intervals_path, [])

    try:
        plate_rendering.run_plate_render(
            source,
            tracks_path=tracks_path,
            review_intervals_path=intervals_path,
            output_dir=tmp_path / "render",
        )
    except ValueError as exc:
        assert "partial_plate_processing_requires_explicit_preview" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("partial preview was rendered without explicit opt-in")

    validation = json.loads((tmp_path / "render" / "validation.json").read_text())
    assert validation["render_plan_validation"]["full_source_processed"] is False
    assert validation["render_plan_validation"]["render_allowed"] is False
    assert not (tmp_path / "render" / "automatic_preview.mp4").exists()


def test_silent_missing_detector_rows_block_preview(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, frame_count=5)
    tracks_path = tmp_path / "tracks.jsonl"
    rows = []
    for index in range(5):
        row = track_row(index, [track(bbox=(18 + index, 20, 74 + index, 42))])
        row["detector_frame_available"] = index not in {1, 3}
        rows.append(row)
    write_tracks(tracks_path, rows)
    intervals_path = tmp_path / "review_intervals.json"
    write_review_intervals(intervals_path, [])

    try:
        plate_rendering.run_plate_render(
            source,
            tracks_path=tracks_path,
            review_intervals_path=intervals_path,
            output_dir=tmp_path / "render",
        )
    except ValueError as exc:
        assert "partial_plate_processing_requires_explicit_preview" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("render accepted track rows with missing detector frames")

    validation = json.loads((tmp_path / "render" / "validation.json").read_text())
    missing = validation["frame_span_report"]["missing_processed_ranges"]
    assert missing == [
        {"frame_start": 1, "frame_end": 1, "length": 1},
        {"frame_start": 3, "frame_end": 3, "length": 1},
    ]


def test_explicit_partial_preview_is_watermarked_and_non_final(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, frame_count=5)
    tracks_path = tmp_path / "tracks.jsonl"
    write_tracks(
        tracks_path,
        [track_row(index, [track(bbox=(18 + index, 20, 74 + index, 42))]) for index in range(3)],
    )
    intervals_path = tmp_path / "review_intervals.json"
    write_review_intervals(intervals_path, [])

    summary = plate_rendering.run_plate_render(
        source,
        tracks_path=tracks_path,
        review_intervals_path=intervals_path,
        output_dir=tmp_path / "render",
        allow_partial_preview=True,
    )

    assert summary["full_source_processed"] is False
    assert summary["partial_processing"] is True
    assert summary["final_approval_status"] is False
    assert (tmp_path / "render" / "automatic_preview.mp4").exists()


def test_reviewed_final_blocks_partial_tracking_even_when_preview_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, frame_count=5)
    tracks_path = tmp_path / "tracks.jsonl"
    write_tracks(
        tracks_path,
        [track_row(index, [track(bbox=(18 + index, 20, 74 + index, 42))]) for index in range(3)],
    )
    intervals_path = tmp_path / "review_intervals.json"
    write_review_intervals(intervals_path, [])

    try:
        plate_rendering.run_plate_render(
            source,
            tracks_path=tracks_path,
            review_intervals_path=intervals_path,
            output_dir=tmp_path / "render",
            mode=plate_rendering.MODE_REVIEWED_FINAL,
            allow_partial_preview=True,
        )
    except ValueError as exc:
        assert "partial_plate_processing" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("reviewed-final render accepted partial tracking")


def test_render_preview_outputs_h264_yuv420p_and_verifies_region(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source)
    tracks_path = tmp_path / "tracks.jsonl"
    write_tracks(
        tracks_path,
        [track_row(index, [track(bbox=(18 + index, 20, 74 + index, 42))]) for index in range(5)],
    )
    intervals_path = tmp_path / "review_intervals.json"
    write_review_intervals(intervals_path, [])

    summary = plate_rendering.run_plate_render(
        source,
        tracks_path=tracks_path,
        review_intervals_path=intervals_path,
        output_dir=tmp_path / "render",
    )
    info = media.probe_media(tmp_path / "render" / "automatic_preview.mp4")
    validation = json.loads((tmp_path / "render" / "validation.json").read_text())

    assert info.video_stream.codec == "h264"
    assert info.video_stream.pixel_format == "yuv420p"
    assert summary["rendered_region_validation_passed"] is True
    assert validation["rendered_region_validation"]["missing_rendered_region_count"] == 0


def test_region_verification_fails_when_output_is_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source)
    copied = tmp_path / "copied.mp4"
    copied.write_bytes(source.read_bytes())
    region = plate_rendering.PlateRenderRegion(
        frame_index=0,
        bbox_xyxy=(18, 20, 74, 42),
        provenance=plate_tracking.PROVENANCE_DETECTOR,
        status=plate_tracking.ACCEPTED_TRACK,
        track_uid="plate_track_0001",
        track_id=1,
        score=0.8,
        source_record={},
    )

    result = plate_rendering.verify_rendered_regions(
        source,
        copied,
        [region],
        config=plate_rendering.default_config(),
        frame_width=120,
        frame_height=80,
    )

    assert result["passed"] is False
    assert result["missing_rendered_region_count"] == 1


def test_plate_review_task_blocks_looks_correct_for_unresolved_gap() -> None:
    tasks = plate_rendering.build_plate_review_tasks(
        rows=[track_row(0, [track()])],
        intervals=[
            {
                "blocking": True,
                "frame_start": 10,
                "frame_end": 30,
                "reason_codes": ["gap_exceeds_safe_interpolation_limit"],
                "track_uids": ["plate_track_0001"],
            }
        ],
        source_frame_count=100,
        spot_check_count=0,
    )
    blocking = [task for task in tasks if task.blocking][0]

    assert "Looks correct" not in blocking.allowed_actions
    assert blocking.allowed_actions == ("Needs correction", "Not a license plate")


def test_short_accepted_track_becomes_false_positive_review_task() -> None:
    rows = [
        track_row(
            index,
            [track(bbox=(20 + index, 20, 70 + index, 40), uid="plate_track_short")],
        )
        for index in range(4)
    ]

    tasks = plate_rendering.short_accepted_track_review_tasks(rows, start_index=1)

    assert len(tasks) == 1
    assert tasks[0].reason_codes == (
        "possible_false_accepted_track",
        "short_accepted_track",
    )
    assert "Not a license plate" in tasks[0].allowed_actions


def test_review_decision_validation_rejects_duplicate_decisions(tmp_path: Path) -> None:
    tasks_path = tmp_path / "review_tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    plate_rendering.PlateReviewTask(
                        review_task_id="plate_review_0001",
                        object_category=plate_detection.CATEGORY,
                        frame_start=0,
                        frame_end=5,
                        severity="blocking",
                        blocking=True,
                        source_stage="plate_tracking",
                        reason_codes=("gap_exceeds_safe_interpolation_limit",),
                        title="Plate coverage needs correction",
                        explanation="Needs correction.",
                        allowed_actions=("Needs correction", "Not a license plate"),
                        default_action_statuses={
                            "Needs correction": "correction_required",
                            "Not a license plate": "false_positive",
                        },
                        track_uids=("plate_track_0001",),
                        media_roles={},
                    ).to_dict()
                ]
            }
        ),
        encoding="utf-8",
    )
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "review_task_id": "plate_review_0001",
                        "action": "Needs correction",
                        "status": "correction_required",
                    },
                    {
                        "review_task_id": "plate_review_0001",
                        "action": "Needs correction",
                        "status": "correction_required",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = plate_rendering.validate_plate_review_decisions(tasks_path, decisions_path)

    assert result["passed"] is False
    assert result["duplicate_decision_count"] == 1


def test_margin_expands_tight_plate_box() -> None:
    expanded = rendering.expanded_clipped_box(
        (50, 20, 70, 30),
        frame_width=120,
        frame_height=80,
        margin_pixels=0,
        margin_fraction=plate_rendering.DEFAULT_MARGIN_FRACTION,
    )

    assert expanded == (47, 17, 73, 33)


def test_render_plates_help_mentions_accepted_oim_tracks_only() -> None:
    result = CliRunner().invoke(redactor_cli.main, ["render-plates", "--help"])

    assert result.exit_code == 0
    assert "accepted OIM tracks" in result.output
    assert "--allow-partial-preview" in result.output
    assert "rtdetr" not in result.output.lower()
    assert "rfdetr" not in result.output.lower()


def test_plate_rendering_has_no_benchmark_or_watermark_specific_behavior() -> None:
    text = Path("src/redactor/plate_rendering.py").read_text(encoding="utf-8")
    forbidden = [
        "Law & Crime",
        "day_plate_primary",
        "night_plate_primary",
        "day_plate_close_stress",
        "eval_day_plate",
        "eval_night_selective",
        "frame_start=420",
        "frame_end=779",
    ]
    for token in forbidden:
        assert token not in text
