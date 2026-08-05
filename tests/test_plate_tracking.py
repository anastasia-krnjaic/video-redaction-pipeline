from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from redactor import cli as redactor_cli
from redactor import plate_detection, plate_tracking


def obs(
    frame_index: int,
    bbox: tuple[int, int, int, int],
    score: float = 0.80,
    *,
    candidate: str | None = None,
) -> plate_tracking.PlateObservation:
    return plate_tracking.PlateObservation(
        frame_index=frame_index,
        bbox_xyxy=bbox,
        score=score,
        detector_id=plate_detection.DETECTOR_ID,
        candidate_id=candidate or f"c{frame_index}",
        source_detection_index=0,
        metadata={},
    )


def run_tracks(
    by_frame: dict[int, list[plate_tracking.PlateObservation]],
    *,
    fps: float = 30.0,
    frame_count: int = 30,
    config: plate_tracking.PlateTrackingConfig | None = None,
    scene_cut_frames: set[tuple[int, int]] | None = None,
) -> plate_tracking.PlateTrackingResult:
    return plate_tracking.track_plate_observations(
        by_frame,
        frame_count=frame_count,
        fps=fps,
        width=1920,
        height=1080,
        config=config or plate_tracking.default_config(),
        scene_cut_frames=scene_cut_frames,
    )


def test_seconds_to_frames_is_fps_independent() -> None:
    assert plate_tracking.seconds_to_frames(0.30, 30.0) == 9
    assert plate_tracking.seconds_to_frames(0.30, 60.0) == 18


def test_iou_and_center_association_creates_one_track() -> None:
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140))],
            1: [obs(1, (108, 101, 188, 141))],
        }
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].first_frame == 0
    assert result.tracks[0].last_frame == 1


def test_scale_gating_prevents_bad_association() -> None:
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140))],
            1: [obs(1, (100, 100, 900, 500))],
        }
    )

    assert len(result.tracks) == 2


def test_track_ids_are_deterministic() -> None:
    payload = {
        0: [obs(0, (100, 100, 180, 140)), obs(0, (500, 100, 580, 140))],
        1: [obs(1, (110, 100, 190, 140)), obs(1, (510, 100, 590, 140))],
    }

    first = run_tracks(payload)
    second = run_tracks(payload)

    assert [track.to_summary_dict()["track_uid"] for track in first.tracks] == [
        track.to_summary_dict()["track_uid"] for track in second.tracks
    ]


def test_track_terminates_after_bounded_gap() -> None:
    config = plate_tracking.PlateTrackingConfig(max_association_gap_seconds=0.30)
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140))],
            20: [obs(20, (120, 100, 200, 140))],
        },
        config=config,
        frame_count=25,
    )

    assert len(result.tracks) == 2


def test_no_interpolation_without_two_anchors() -> None:
    result = run_tracks({0: [obs(0, (100, 100, 180, 140))]}, frame_count=10)

    assert all(
        frame.provenance != plate_tracking.PROVENANCE_INTERPOLATION
        for frames in result.frame_tracks.values()
        for frame in frames
    )


def test_bounded_interpolation_between_reliable_anchors() -> None:
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140))],
            3: [obs(3, (130, 100, 210, 140))],
        },
        frame_count=5,
    )

    interpolated = [
        frame
        for frames in result.frame_tracks.values()
        for frame in frames
        if frame.provenance == plate_tracking.PROVENANCE_INTERPOLATION
    ]
    assert [frame.frame_index for frame in interpolated] == [1, 2]
    assert all(frame.anchor_frames == (0, 3) for frame in interpolated)


def test_long_gap_becomes_review_interval_not_interpolation() -> None:
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140))],
            20: [obs(20, (140, 100, 220, 140))],
        },
        frame_count=25,
    )

    assert any(decision.decision == "needs_review" for decision in result.gap_decisions)
    assert any(interval.blocking for interval in result.review_intervals)
    assert not any(
        frame.provenance == plate_tracking.PROVENANCE_INTERPOLATION and 1 <= frame.frame_index <= 19
        for frames in result.frame_tracks.values()
        for frame in frames
    )


def test_scene_cut_inside_gap_blocks_interpolation() -> None:
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140))],
            3: [obs(3, (130, 100, 210, 140))],
        },
        frame_count=5,
        scene_cut_frames={(0, 3)},
    )

    assert result.gap_decisions[0].decision == "needs_review"
    assert "scene_cut_inside_gap" in result.gap_decisions[0].reasons


def test_isolated_false_positive_is_rejected() -> None:
    result = run_tracks({0: [obs(0, (100, 100, 180, 140))]})

    assert result.tracks[0].status == plate_tracking.REJECTED_NOISE
    assert "isolated_plate_proposal" in result.tracks[0].review_reasons


def test_persistent_moving_plate_track_is_accepted() -> None:
    result = run_tracks(
        {frame: [obs(frame, (100 + frame * 2, 100, 180 + frame * 2, 140))] for frame in range(10)},
        frame_count=10,
    )

    assert result.tracks[0].status == plate_tracking.ACCEPTED_TRACK


def test_duplicate_proposal_merging_records_rejection() -> None:
    kept, rejected = plate_tracking.deduplicate_same_frame_observations(
        [
            obs(0, (100, 100, 180, 140), 0.90, candidate="strong"),
            obs(0, (102, 101, 182, 141), 0.70, candidate="weak"),
        ],
        config=plate_tracking.default_config(),
    )

    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0].reasons[0] == "duplicate_same_frame_proposal"


def test_provenance_separates_detector_and_interpolation() -> None:
    result = run_tracks(
        {
            0: [obs(0, (100, 100, 180, 140), candidate="left")],
            2: [obs(2, (120, 100, 200, 140), candidate="right")],
        },
        frame_count=3,
    )

    provenances = {frame.provenance for frames in result.frame_tracks.values() for frame in frames}
    assert provenances == {
        plate_tracking.PROVENANCE_DETECTOR,
        plate_tracking.PROVENANCE_INTERPOLATION,
    }


def test_configuration_hash_is_shared_for_all_clips() -> None:
    config = plate_tracking.default_config()

    assert config.configuration_hash() == plate_tracking.PlateTrackingConfig().configuration_hash()
    assert config.detector_configuration_hash == (
        "7d914438255905e2a249e5bbd30fd6e70eace1e91b57e205bd572d004ffb5a55"
    )


def test_plate_tracking_rejects_detector_threshold_changes() -> None:
    try:
        plate_tracking.PlateTrackingConfig(primary_threshold=0.20)
    except ValueError as exc:
        assert "must not change" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("plate tracking accepted a detector threshold change")


def test_proposal_geometry_validation() -> None:
    config = plate_tracking.default_config()

    reasons = plate_tracking.proposal_rejection_reasons(
        (10, 10, 12, 80),
        score=0.80,
        width=1920,
        height=1080,
        config=config,
        detector_id=plate_detection.DETECTOR_ID,
    )

    assert "implausible_plate_aspect_ratio" in reasons


def test_track_plates_help_mentions_oim_without_old_selectors() -> None:
    result = CliRunner().invoke(redactor_cli.main, ["track-plates", "--help"])

    assert result.exit_code == 0
    assert "OIM YOLOv9-S-608" in result.output
    assert "rtdetr" not in result.output.lower()
    assert "rfdetr" not in result.output.lower()


def test_production_plate_tracking_has_no_benchmark_specific_behavior() -> None:
    text = Path("src/redactor/plate_tracking.py").read_text(encoding="utf-8")

    forbidden = [
        "Law & Crime",
        "watermark",
        "day_plate_primary",
        "night_plate_primary",
        "day_plate_close_stress",
        "eval_day_plate",
        "eval_night_selective",
        "frame_start=420",
        "frame_end=779",
        "[420, 779]",
        "(420, 779)",
    ]
    for token in forbidden:
        assert token not in text


def test_synthetic_video_smoke_without_model(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"
    writer = None
    try:
        writer = cv2_writer(video_path, fps=10.0, size=(120, 80))
        for frame_index in range(5):
            frame = np.full((80, 120, 3), 40 + frame_index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    detections = tmp_path / "detections.jsonl"
    with detections.open("w", encoding="utf-8") as file:
        for frame_index in range(5):
            record = {
                "frame_index": frame_index,
                "detections": [
                    {
                        "bbox_xyxy": [20 + frame_index, 20, 60 + frame_index, 35],
                        "candidate_id": f"plate:f{frame_index}",
                        "category": "license_plate",
                        "detector_id": plate_detection.DETECTOR_ID,
                        "score": 0.8,
                    }
                ],
            }
            file.write(__import__("json").dumps(record) + "\n")

    summary = plate_tracking.run_plate_tracking(
        video_path,
        detections_path=detections,
        output_dir=tmp_path / "out",
    )

    assert summary["frames_processed"] == 5
    assert (tmp_path / "out" / "tracks.jsonl").exists()


def cv2_writer(path: Path, *, fps: float, size: tuple[int, int]):
    import cv2 as cv

    return cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, size)
