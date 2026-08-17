from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from redactor import privacy_gap_recovery as recovery


class DictFrameProvider:
    def __init__(self, frames: dict[int, np.ndarray] | None = None):
        self.frames = frames or {}

    def frame(self, frame_index: int) -> np.ndarray:
        if frame_index in self.frames:
            return self.frames[frame_index].copy()
        return np.full((100, 120, 3), frame_index, dtype=np.uint8)


@dataclass
class SequenceTracker:
    trajectory: dict[int, recovery.Box | None]

    def init(self, frame: np.ndarray, bbox_xyxy: recovery.Box) -> None:
        del frame, bbox_xyxy

    def update(self, frame: np.ndarray) -> tuple[bool, recovery.Box | None]:
        frame_index = int(frame[0, 0, 0])
        box = self.trajectory.get(frame_index)
        return box is not None, box


class SequenceTrackerFactory:
    def __init__(self, *trajectories: dict[int, recovery.Box | None]):
        self.trajectories = list(trajectories)

    def __call__(self) -> SequenceTracker:
        return SequenceTracker(self.trajectories.pop(0))


class FixedRefinement:
    name = "fixed"

    def __init__(self, mapping: dict[recovery.Box, recovery.Box]):
        self.mapping = mapping

    def refine(
        self,
        frame: np.ndarray,
        prior_box: recovery.Box,
        *,
        previous_box: recovery.Box | None,
        frame_size: tuple[int, int],
    ) -> tuple[recovery.Box | None, dict[str, object]]:
        del frame, previous_box, frame_size
        return self.mapping.get(prior_box, prior_box), {
            "candidate_score": 1.0,
            "strategy": self.name,
        }


def source(frame: int, box: recovery.Box, track_id: int = 1) -> recovery.TrackSourceBox:
    return recovery.TrackSourceBox(
        frame_index=frame,
        track_id=track_id,
        bbox_xyxy=box,
        source_provenance="observed_strong",
        score=0.9,
        quality_status="auto_confirmed",
    )


def run(
    target_boxes: dict[int, recovery.TrackSourceBox],
    *,
    frame_start: int = 0,
    frame_end: int = 4,
    settings: recovery.PrivacyGapRecoverySettings | None = None,
    tracker_factory: recovery.TrackerFactory | None = None,
    keep_boxes: dict[int, list[recovery.TrackSourceBox]] | None = None,
    competing_boxes: dict[int, list[recovery.TrackSourceBox]] | None = None,
    frames: dict[int, np.ndarray] | None = None,
    preserved: dict[int, recovery.PreservedRegion] | None = None,
) -> recovery.RecoveryResult:
    return recovery.recover_privacy_gaps(
        frame_provider=DictFrameProvider(frames),
        target_boxes=target_boxes,
        target_track_ids=(1,),
        keep_boxes_by_frame=keep_boxes or {},
        competing_boxes_by_frame=competing_boxes or {},
        preserved_regions=preserved or {},
        frame_start=frame_start,
        frame_end=frame_end,
        frame_size=(120, 100),
        settings=settings or recovery.PrivacyGapRecoverySettings(render_overlay=False),
        tracker_factory=tracker_factory or SequenceTrackerFactory({}),
    )


def recovered_box(result: recovery.RecoveryResult, frame_index: int) -> recovery.Box | None:
    return result.regions_by_frame[frame_index][0].bbox_xyxy


def decision(result: recovery.RecoveryResult, index: int = 0) -> recovery.IntervalDecision:
    return result.interval_decisions[index]


def semantic_hash(result: recovery.RecoveryResult) -> str:
    records = []
    for frame_index in sorted(result.regions_by_frame):
        records.extend(region.to_dict() for region in result.regions_by_frame[frame_index])
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_short_gap_is_interpolated() -> None:
    result = run({0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))}, frame_end=2)

    assert decision(result).strategy == recovery.PROVENANCE_INTERPOLATION
    assert decision(result).status == recovery.DECISION_ACCEPTED_AUTOMATICALLY
    assert recovered_box(result, 1) == (11, 10, 31, 30)


def test_valid_bidirectional_gap_is_covered_but_review_required() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (10, 10, 30, 30)), 3: source(3, (13, 10, 33, 30))},
        frame_end=3,
        settings=settings,
        tracker_factory=SequenceTrackerFactory(
            {1: (11, 10, 31, 30), 2: (12, 10, 32, 30)},
            {1: (11, 10, 31, 30), 2: (12, 10, 32, 30)},
        ),
    )

    assert decision(result).strategy == recovery.PROVENANCE_BIDIRECTIONAL_BRIDGE
    assert decision(result).status == recovery.DECISION_NEEDS_REVIEW
    assert recovery.REASON_DETERMINISTIC_BRIDGE_REQUIRES_REVIEW in decision(result).reasons
    assert recovered_box(result, 1) == (11, 10, 31, 30)
    assert result.regions_by_frame[1][0].tracking_direction == "deterministic_source_bound"
    assert result.regions_by_frame[1][0].metrics["authoritative_algorithm"] == (
        recovery.DETERMINISTIC_BRIDGE_ALGORITHM
    )
    assert result.regions_by_frame[1][0].metrics["visual_tracker_authoritative"] is False


def test_bidirectional_gap_ignores_tracker_failures_for_authoritative_geometry() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (10, 10, 30, 30)), 3: source(3, (13, 10, 33, 30))},
        frame_end=3,
        settings=settings,
        tracker_factory=SequenceTrackerFactory(
            {1: (90, 10, 110, 30), 2: None},
            {1: None, 2: (90, 10, 110, 30)},
        ),
    )

    assert result.regions_by_frame[2][0].provenance == recovery.PROVENANCE_BIDIRECTIONAL_BRIDGE
    assert result.regions_by_frame[2][0].tracking_direction == "deterministic_source_bound"
    assert recovered_box(result, 2) == (12, 10, 32, 30)


def test_visual_refinement_does_not_change_authoritative_bidirectional_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        enable_visual_refinement=True,
        visual_refinement="bottom_component",
        render_overlay=False,
    )
    monkeypatch.setattr(
        recovery,
        "refinement_strategy",
        lambda settings: pytest.fail("bidirectional bridge must not refine MIL boxes"),
    )

    result = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory(
            {1: (60, 10, 80, 30)},
            {1: (10, 10, 30, 30)},
        ),
    )

    assert decision(result).status == recovery.DECISION_NEEDS_REVIEW
    assert recovered_box(result, 1) == (11, 10, 31, 30)
    assert "normalized_raw_disagreement" not in result.regions_by_frame[1][0].metrics


def test_tracker_disagreement_cannot_change_authoritative_bidirectional_hash() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        render_overlay=False,
    )
    first = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory(
            {1: (10, 10, 30, 30)},
            {1: (70, 10, 90, 30)},
        ),
    )
    second = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory(
            {1: (70, 10, 90, 30)},
            {1: (10, 10, 30, 30)},
        ),
    )

    assert semantic_hash(first) == semantic_hash(second)
    assert recovered_box(first, 1) == (11, 10, 31, 30)


def test_one_sided_gap_is_review_required_without_authoritative_visual_tracking() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        max_one_sided_gap_frames=3,
        render_overlay=False,
    )
    prefix = run(
        {2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({0: (10, 10, 30, 30), 1: (11, 10, 31, 30)}),
    )
    tail = run(
        {0: source(0, (10, 10, 30, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({1: (11, 10, 31, 30), 2: (12, 10, 32, 30)}),
    )

    assert decision(prefix).status == recovery.DECISION_NEEDS_REVIEW
    assert recovered_box(prefix, 0) is None
    assert prefix.regions_by_frame[0][0].provenance == recovery.PROVENANCE_BACKWARD_BRIDGE
    assert prefix.regions_by_frame[0][0].metrics["visual_tracker_authoritative"] is False
    assert (
        recovery.REASON_ONE_SIDED_VISUAL_TRACKING_NOT_AUTHORITATIVE
        in prefix.regions_by_frame[0][0].reasons
    )
    assert decision(tail).status == recovery.DECISION_NEEDS_REVIEW
    assert recovered_box(tail, 2) is None
    assert tail.regions_by_frame[2][0].provenance == recovery.PROVENANCE_FORWARD_BRIDGE
    assert tail.regions_by_frame[2][0].metrics["visual_tracker_authoritative"] is False
    assert (
        recovery.REASON_ONE_SIDED_VISUAL_TRACKING_NOT_AUTHORITATIVE
        in tail.regions_by_frame[2][0].reasons
    )


def test_identity_competition_flags_frame() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        competing_identity_distance_norm=0.8,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({1: (11, 10, 31, 30)}, {1: (11, 10, 31, 30)}),
        competing_boxes={1: [source(1, (12, 11, 32, 31), track_id=2)]},
    )

    assert decision(result).status == recovery.DECISION_NEEDS_REVIEW
    assert "identity_competition" in result.regions_by_frame[1][0].reasons


def test_keep_person_overlap_flags_frame() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        max_keep_overlap_fraction=0.0,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({1: (11, 10, 31, 30)}, {1: (11, 10, 31, 30)}),
        keep_boxes={1: [source(1, (12, 10, 32, 30), track_id=9)]},
    )

    assert decision(result).status == recovery.DECISION_NEEDS_REVIEW
    assert "keep_person_overlap" in result.regions_by_frame[1][0].reasons


def test_deterministic_bridge_clips_conservative_envelope_to_frame() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.5,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (0, 0, 20, 20)), 2: source(2, (2, 0, 22, 20))},
        frame_end=2,
        settings=settings,
    )

    assert recovered_box(result, 1) == (0, 0, 31, 30)
    assert result.regions_by_frame[1][0].metrics["rounding"] == (
        recovery.DETERMINISTIC_ROUNDING_RULE
    )


def test_scene_cut_inside_gap_needs_review() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        scene_cut_mean_delta=20,
        render_overlay=False,
    )
    frames = {}
    for index, value in ((0, 0), (1, 255), (2, 255)):
        frame = np.full((100, 120, 3), value, dtype=np.uint8)
        frame[0, 0, 0] = index
        frames[index] = frame
    result = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (12, 10, 32, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({1: (11, 10, 31, 30)}, {1: (11, 10, 31, 30)}),
        frames=frames,
    )

    assert decision(result).status == recovery.DECISION_NEEDS_REVIEW
    assert "scene_cut_inside_gap" in result.regions_by_frame[1][0].reasons


def test_excessive_center_jump_needs_review() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        max_center_jump_norm=0.2,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (80, 10, 100, 30))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({1: (80, 10, 100, 30)}, {1: (80, 10, 100, 30)}),
    )

    assert "excessive_center_jump" in result.regions_by_frame[1][0].reasons


def test_excessive_scale_change_needs_review() -> None:
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        max_scale_change=0.2,
        render_overlay=False,
    )
    result = run(
        {0: source(0, (10, 10, 30, 30)), 2: source(2, (10, 10, 70, 70))},
        frame_end=2,
        settings=settings,
        tracker_factory=SequenceTrackerFactory({1: (10, 10, 70, 70)}, {1: (10, 10, 70, 70)}),
    )

    assert "excessive_scale_change" in result.regions_by_frame[1][0].reasons


def test_preserved_regions_are_kept_for_accepted_gaps() -> None:
    preserved = {
        frame: recovery.PreservedRegion(
            frame_index=frame,
            bbox_xyxy=(10 + frame, 20, 30 + frame, 40),
            source_record={
                "bbox_xyxy": [10 + frame, 20, 30 + frame, 40],
                "frame_index": frame,
                "provenance": "synthetic_privacy_bridge",
                "source_anchor_frames": [0, 5],
                "state": "covered",
            },
        )
        for frame in range(1, 5)
    }
    result = run(
        {0: source(0, (10, 20, 30, 40)), 5: source(5, (15, 20, 35, 40))},
        frame_end=5,
        preserved=preserved,
        tracker_factory=SequenceTrackerFactory(
            {1: (50, 50, 70, 70), 2: (50, 50, 70, 70), 3: (50, 50, 70, 70), 4: (50, 50, 70, 70)}
        ),
    )

    assert decision(result).strategy == recovery.PROVENANCE_PRESERVED_REGION
    assert [recovered_box(result, frame) for frame in range(1, 5)] == [
        (11, 20, 31, 40),
        (12, 20, 32, 40),
        (13, 20, 33, 40),
        (14, 20, 34, 40),
    ]


def test_rightward_drift_fixture_ignores_optional_refinement_for_authority() -> None:
    frame = np.full((100, 120, 3), 170, dtype=np.uint8)
    frame[78:100, 35:66] = 10
    frames = {}
    for index in range(5):
        indexed = frame.copy()
        indexed[0, 0, 0] = index
        frames[index] = indexed
    settings = recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=0,
        deterministic_bridge_margin_fraction=0.0,
        enable_visual_refinement=True,
        visual_refinement="bottom_component",
        max_raw_disagreement_norm=0.3,
        render_overlay=False,
    )

    result = run(
        {0: source(0, (30, 70, 70, 98)), 4: source(4, (32, 70, 72, 98))},
        frame_end=4,
        settings=settings,
        tracker_factory=SequenceTrackerFactory(
            {
                1: (54, 70, 94, 98),
                2: (56, 70, 96, 98),
                3: (58, 70, 98, 98),
            },
            {
                1: (30, 70, 70, 98),
                2: (31, 70, 71, 98),
                3: (32, 70, 72, 98),
            },
        ),
        frames=frames,
    )

    assert decision(result).status == recovery.DECISION_NEEDS_REVIEW
    assert [recovered_box(result, frame_index) for frame_index in (1, 2, 3)] == [
        (31, 70, 71, 98),
        (31, 70, 71, 98),
        (32, 70, 72, 98),
    ]
    assert all(
        region.metrics["visual_tracker_authoritative"] is False
        for frame_index in (1, 2, 3)
        for region in result.regions_by_frame[frame_index]
    )


def test_load_preserved_regions_reads_v2_shape(tmp_path: Path) -> None:
    path = tmp_path / "privacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "bbox_xyxy": [1, 2, 3, 4],
                "frame_index": 7,
                "state": "covered",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = recovery.load_preserved_regions(path)

    assert loaded[7].bbox_xyxy == (1, 2, 3, 4)
