from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from redactor import tracking
from redactor.models import MediaInfo, TrackBox, VideoStreamInfo


class FakeCapture:
    instances: list[FakeCapture] = []

    def __init__(self, path: str, frame_count: int = 3):
        self.path = path
        self.frame_count = frame_count
        self.read_calls = 0
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_calls >= self.frame_count:
            return False, None
        frame = np.full((80, 100, 3), self.read_calls, dtype=np.uint8)
        self.read_calls += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeWriter:
    instances: list[FakeWriter] = []

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.write_count = 0
        self.released = False
        FakeWriter.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape[:2] == (80, 100)
        self.write_count += 1

    def release(self) -> None:
        self.released = True


def obs(
    frame_index: int,
    *,
    score: float = 0.9,
    bbox: tuple[int, int, int, int] | None = None,
) -> tracking.TrackObservation:
    if bbox is None:
        bbox = (10 + frame_index, 10, 30 + frame_index, 30)
    return tracking.TrackObservation(
        frame_index=frame_index,
        bbox_xyxy=bbox,
        score=score,
        provenance="observed_weak",
    )


def manual(
    frame_index: int,
    *,
    bbox: tuple[int, int, int, int] = (10, 10, 30, 30),
) -> tracking.TrackObservation:
    return tracking.TrackObservation(
        frame_index=frame_index,
        bbox_xyxy=bbox,
        score=None,
        provenance="manual",
        action="redact",
    )


def quality_settings(**overrides) -> tracking.QualitySettings:
    values = {
        "min_observed_hits": 3,
        "min_strong_hits": 2,
        "confirmation_window": 5,
        "min_auto_median_score": 0.70,
        "max_auto_synthetic_fraction": 0.40,
        "min_auto_area_fraction": 0.0005,
        "max_auto_area_fraction": 0.08,
    }
    values.update(overrides)
    return tracking.QualitySettings(**values)


def association_settings(max_frame_distance: int = 5) -> tracking.AssociationSettings:
    return tracking.AssociationSettings(
        iou_threshold=0.15,
        center_distance_threshold=0.75,
        max_frame_distance=max_frame_distance,
    )


def build(
    detections: dict[int, list[tracking.TrackObservation]],
    *,
    manual_seeds: dict[int, list[tracking.TrackObservation]] | None = None,
    frame_count: int = 8,
    max_gap_frames: int = 0,
    max_tail_prediction_frames: int = 0,
    settings: tracking.QualitySettings | None = None,
    association: tracking.AssociationSettings | None = None,
) -> tracking.TrackingResult:
    return tracking.build_tracks(
        detections,
        manual_seeds=manual_seeds or {},
        frame_count=frame_count,
        strong_threshold=0.7,
        max_gap_frames=max_gap_frames,
        max_tail_prediction_frames=max_tail_prediction_frames,
        fps=10.0,
        width=100,
        height=100,
        quality_settings=settings or quality_settings(),
        association_settings=association or association_settings(),
    )


def boxes_at(result: tracking.TrackingResult, frame_index: int) -> list[TrackBox]:
    return result.frame_boxes[frame_index]


def quality(result: tracking.TrackingResult, track_id: int = 1) -> tracking.TrackQuality:
    return next(item for item in result.track_qualities if item.track_id == track_id)


def flag_types(result: tracking.TrackingResult) -> set[str]:
    return {flag.type for flag in result.review_flags}


def media_info_for(path: Path, frame_count: str = "3") -> MediaInfo:
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
            width=100,
            height=80,
            pixel_format="yuv420p",
            nominal_frame_rate="10/1",
            average_frame_rate="10/1",
            time_base="1/10",
            frame_count=frame_count,
        ),
        audio_stream=None,
        ffprobe_version="ffprobe version test",
    )


def patch_video_io(monkeypatch: pytest.MonkeyPatch, frame_count: int = 3) -> None:
    FakeCapture.instances.clear()
    FakeWriter.instances.clear()

    def fake_capture(path: str) -> FakeCapture:
        return FakeCapture(path, frame_count=frame_count)

    monkeypatch.setattr(tracking.cv, "VideoCapture", fake_capture)
    monkeypatch.setattr(tracking.cv, "VideoWriter", FakeWriter)


def write_detections(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_one_strong_observation_does_not_auto_confirm() -> None:
    result = build({0: [obs(0, score=0.9)]})

    assert quality(result).quality_status == "rejected"
    assert quality(result).render_eligible is False
    assert boxes_at(result, 0)[0].provenance == "observed_strong"
    assert boxes_at(result, 0)[0].quality_status == "rejected"


def test_one_strong_observation_plus_predictions_does_not_auto_confirm() -> None:
    result = build({0: [obs(0, score=0.9)]}, frame_count=5, max_tail_prediction_frames=2)

    assert quality(result).quality_status == "rejected"
    assert result.predicted_frames == 0
    assert all(
        box.provenance != "predicted" for boxes in result.frame_boxes.values() for box in boxes
    )


def test_predictions_never_count_as_observed_hits() -> None:
    result = build(
        {0: [obs(0)], 1: [obs(1)], 2: [obs(2)]},
        frame_count=8,
        max_tail_prediction_frames=2,
    )

    assert quality(result).quality_status == "auto_confirmed"
    assert quality(result).observed_hits == 3
    assert quality(result).predicted_boxes == 2
    assert result.predicted_frames == 2


def test_three_consistent_observed_detections_create_one_track() -> None:
    result = build({0: [obs(0)], 1: [obs(1)], 2: [obs(2)]})

    assert result.total_track_proposals == 1
    assert quality(result).quality_status == "auto_confirmed"
    assert [boxes_at(result, frame)[0].track_id for frame in (0, 1, 2)] == [1, 1, 1]


def test_weak_only_repeated_evidence_becomes_review_candidate() -> None:
    result = build({0: [obs(0, score=0.5)], 1: [obs(1, score=0.51)], 2: [obs(2, score=0.52)]})

    assert quality(result).quality_status == "review_candidate"
    assert quality(result).render_eligible is False
    assert "insufficient_strong_hits" in quality(result).quality_gate_failures


def test_high_quality_repeated_track_becomes_auto_confirmed() -> None:
    result = build({0: [obs(0, score=0.88)], 1: [obs(1, score=0.9)], 2: [obs(2, score=0.92)]})

    assert result.auto_confirmed_track_ids == {1}
    assert boxes_at(result, 1)[0].render_eligible is True


def test_tiny_repeated_track_is_review_candidate_not_auto_confirmed() -> None:
    tiny = (10, 10, 11, 11)
    result = build({0: [obs(0, bbox=tiny)], 1: [obs(1, bbox=tiny)], 2: [obs(2, bbox=tiny)]})

    assert quality(result).quality_status == "review_candidate"
    assert "tiny_median_area" in quality(result).quality_gate_failures


def test_huge_repeated_track_is_review_candidate_not_auto_confirmed() -> None:
    huge = (0, 0, 95, 95)
    result = build({0: [obs(0, bbox=huge)], 1: [obs(1, bbox=huge)], 2: [obs(2, bbox=huge)]})

    assert quality(result).quality_status == "review_candidate"
    assert "large_median_area" in quality(result).quality_gate_failures


def test_track_with_too_much_synthetic_content_is_not_auto_confirmed() -> None:
    settings = quality_settings(
        min_observed_hits=2,
        min_strong_hits=2,
        max_auto_synthetic_fraction=0.25,
    )
    result = build(
        {0: [obs(0, bbox=(10, 10, 30, 30))], 4: [obs(4, bbox=(14, 10, 34, 30))]},
        max_gap_frames=3,
        settings=settings,
    )

    assert quality(result).quality_status == "review_candidate"
    assert quality(result).interpolated_boxes == 3
    assert "synthetic_fraction" in quality(result).quality_gate_failures


def test_review_candidate_tracks_receive_no_tail_prediction() -> None:
    result = build(
        {0: [obs(0, score=0.5)], 1: [obs(1, score=0.51)], 2: [obs(2, score=0.52)]},
        max_tail_prediction_frames=2,
    )

    assert quality(result).quality_status == "review_candidate"
    assert result.predicted_frames == 0


def test_auto_confirmed_tracks_receive_only_configured_short_prediction() -> None:
    result = build(
        {0: [obs(0)], 1: [obs(1)], 2: [obs(2)]},
        frame_count=8,
        max_tail_prediction_frames=1,
    )

    assert [box.provenance for box in boxes_at(result, 3)] == ["predicted"]
    assert boxes_at(result, 4) == []
    assert result.predicted_frames == 1


def test_internal_interpolation_requires_observed_endpoints() -> None:
    result = build({0: [obs(0)]}, frame_count=5, max_gap_frames=3)

    assert result.interpolated_frames == 0
    assert boxes_at(result, 1) == []


def test_short_internal_gap_is_interpolated() -> None:
    result = build(
        {0: [obs(0)], 2: [obs(2)], 3: [obs(3)]},
        frame_count=6,
        max_gap_frames=1,
    )

    assert boxes_at(result, 1)[0].provenance == "interpolated"
    assert result.interpolated_frames == 1
    assert result.maximum_filled_gap_frames == 1
    assert "interpolated_span" in flag_types(result)


def test_gap_longer_than_limit_is_not_filled() -> None:
    result = build({0: [obs(0)], 3: [obs(3)], 4: [obs(4)]}, max_gap_frames=1)

    assert boxes_at(result, 1) == []
    assert boxes_at(result, 2) == []
    assert "long_gap" in flag_types(result)
    assert "unresolved_interval" in flag_types(result)


def test_weak_proposals_may_be_associated_backward() -> None:
    result = build(
        {
            0: [obs(0, score=0.5, bbox=(10, 10, 30, 30))],
            3: [obs(3, score=0.9, bbox=(13, 10, 33, 30))],
            4: [obs(4, score=0.9, bbox=(14, 10, 34, 30))],
        },
        association=association_settings(max_frame_distance=10),
    )

    assert result.total_track_proposals == 1
    assert boxes_at(result, 0)[0].provenance == "observed_weak"
    assert boxes_at(result, 0)[0].track_id == boxes_at(result, 4)[0].track_id


def test_boxes_are_not_invented_before_first_observation() -> None:
    result = build({3: [obs(3)], 4: [obs(4)]}, frame_count=5, max_gap_frames=0)

    assert boxes_at(result, 0) == []
    assert boxes_at(result, 1) == []
    assert boxes_at(result, 2) == []
    assert boxes_at(result, 3)[0].provenance == "observed_strong"


def test_manual_seed_can_precede_and_connect_to_automatic_observation() -> None:
    result = build(
        {2: [obs(2, bbox=(12, 10, 32, 30))], 3: [obs(3, bbox=(13, 10, 33, 30))]},
        manual_seeds={0: [manual(0)]},
        max_gap_frames=1,
    )

    assert boxes_at(result, 0)[0].provenance == "manual"
    assert boxes_at(result, 1)[0].provenance == "interpolated"
    assert boxes_at(result, 2)[0].provenance == "observed_strong"
    assert "manual_seed" in flag_types(result)


def test_unsafe_scale_jump_is_not_associated() -> None:
    result = build(
        {
            0: [obs(0, bbox=(10, 10, 30, 30))],
            1: [obs(1, bbox=(11, 10, 31, 30))],
            2: [obs(2, bbox=(10, 10, 60, 60))],
            3: [obs(3, bbox=(13, 10, 33, 30))],
        },
        association=association_settings(max_frame_distance=5),
    )

    assert boxes_at(result, 3)[0].track_id == 1
    assert boxes_at(result, 2)[0].track_id != 1
    assert result.total_track_proposals == 2


def test_one_to_one_matching_is_preserved() -> None:
    result = build(
        {
            0: [
                obs(0, bbox=(10, 10, 30, 30)),
                obs(0, bbox=(50, 10, 70, 30)),
            ],
            1: [
                obs(1, bbox=(11, 10, 31, 30)),
                obs(1, bbox=(51, 10, 71, 30)),
            ],
            2: [
                obs(2, bbox=(12, 10, 32, 30)),
                obs(2, bbox=(52, 10, 72, 30)),
            ],
        }
    )

    frame_one_track_ids = [box.track_id for box in boxes_at(result, 1)]
    assert sorted(frame_one_track_ids) == [1, 2]
    assert len(set(frame_one_track_ids)) == 2


def test_review_flags_aggregate_into_fewer_review_windows() -> None:
    result = build(
        {0: [obs(0, score=0.5)], 10: [obs(10, bbox=(11, 10, 31, 30), score=0.5)]},
        frame_count=20,
    )

    assert len(result.review_windows) < len(result.review_flags)


def test_quality_summaries_are_deterministic() -> None:
    detections = {0: [obs(0)], 1: [obs(1)], 2: [obs(2)]}

    first = [item.to_dict() for item in build(detections).track_qualities]
    second = [item.to_dict() for item in build(detections).track_qualities]

    assert first == second


def test_output_contains_one_record_per_input_frame_and_safe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "video with spaces.mp4"
    detections_path = tmp_path / "detections with spaces.jsonl"
    output_dir = tmp_path / "tracking output"
    write_detections(
        detections_path,
        [
            {
                "frame_index": 0,
                "detections": [{"category": "face", "score": 0.9, "bbox_xyxy": [10, 10, 30, 30]}],
            },
            {"frame_index": 1, "detections": []},
            {"frame_index": 2, "detections": []},
        ],
    )
    patch_video_io(monkeypatch, frame_count=3)
    monkeypatch.setattr(tracking.media, "probe_media", lambda path: media_info_for(path))

    tracking.run_tracking(
        input_path,
        detections_path=detections_path,
        output_dir=output_dir,
        max_gap_seconds=0,
        overwrite=True,
    )

    lines = (output_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[1]) == {"frame_index": 1, "track_boxes": []}
    assert (output_dir / "review_windows.json").exists()
    assert (output_dir / "track_quality.json").exists()
    assert FakeCapture.instances[0].path == str(input_path)
    assert FakeWriter.instances[0].path == str(output_dir / "overlay.mp4")


def test_overwrite_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        tracking.ensure_output_dir(output_dir, overwrite=False)

    tracking.ensure_output_dir(output_dir, overwrite=True)
