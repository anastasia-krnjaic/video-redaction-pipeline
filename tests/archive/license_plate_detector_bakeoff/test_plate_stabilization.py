from __future__ import annotations

import inspect

from experiments.archive.license_plate_detector_bakeoff import plate_stabilization


def proposal(
    frame_index: int,
    bbox: tuple[int, int, int, int],
    *,
    score: float = 0.8,
) -> plate_stabilization.PlateProposal:
    return plate_stabilization.PlateProposal(
        detector_id="intel_yolov8n_lpr_openvino",
        frame_index=frame_index,
        score=score,
        bbox_xyxy=bbox,
        source_detection_index=frame_index,
    )


def test_proposal_geometry_validation_rejects_implausible_records() -> None:
    config = plate_stabilization.PlateStabilizationConfig(
        min_area_fraction=0.01,
        max_area_fraction=0.20,
        min_aspect_ratio=1.0,
        max_aspect_ratio=5.0,
        proposal_threshold=0.5,
    )

    assert plate_stabilization.validate_proposal_geometry(
        proposal(0, (10, 10, 20, 20), score=0.1),
        frame_width=100,
        frame_height=100,
        config=config,
    ) == ["below_proposal_threshold"]
    assert "area_too_small" in plate_stabilization.validate_proposal_geometry(
        proposal(0, (10, 10, 12, 12)),
        frame_width=100,
        frame_height=100,
        config=config,
    )
    assert "area_too_large" in plate_stabilization.validate_proposal_geometry(
        proposal(0, (0, 0, 100, 100)),
        frame_width=100,
        frame_height=100,
        config=config,
    )
    assert "aspect_ratio_too_small" in plate_stabilization.validate_proposal_geometry(
        proposal(0, (10, 10, 20, 40)),
        frame_width=100,
        frame_height=100,
        config=config,
    )
    assert "aspect_ratio_too_large" in plate_stabilization.validate_proposal_geometry(
        proposal(0, (10, 10, 90, 20)),
        frame_width=100,
        frame_height=100,
        config=config,
    )
    assert "box_outside_frame" in plate_stabilization.validate_proposal_geometry(
        proposal(0, (-1, 10, 20, 20)),
        frame_width=100,
        frame_height=100,
        config=config,
    )


def test_duplicate_proposal_merging_preserves_highest_score() -> None:
    kept, duplicates = plate_stabilization.suppress_duplicate_proposals(
        [
            proposal(0, (10, 10, 40, 22), score=0.7),
            proposal(0, (11, 10, 41, 22), score=0.9),
            proposal(0, (80, 80, 120, 96), score=0.6),
        ],
        iou_threshold=0.5,
    )

    assert [item.score for item in kept] == [0.9, 0.6]
    assert [item.score for item in duplicates] == [0.7]


def test_iou_association_builds_one_track() -> None:
    tracks = plate_stabilization.associate_plate_tracks(
        {
            0: [proposal(0, (10, 10, 40, 22))],
            1: [proposal(1, (12, 11, 42, 23))],
            2: [proposal(2, (14, 12, 44, 24))],
        },
        config=plate_stabilization.PlateStabilizationConfig(
            accepted_min_observations=3,
            accepted_min_span_frames=3,
            accepted_min_observation_density=0.8,
            review_min_observations=2,
            review_min_span_frames=2,
        ),
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == "plate_track_0001"
    assert [item.frame_index for item in tracks[0].observed] == [0, 1, 2]


def test_center_motion_gate_prevents_far_jump() -> None:
    tracks = plate_stabilization.associate_plate_tracks(
        {
            0: [proposal(0, (10, 10, 40, 22))],
            1: [proposal(1, (500, 500, 530, 512))],
        },
        config=plate_stabilization.PlateStabilizationConfig(max_frame_gap=2),
    )

    assert [track.track_id for track in tracks] == ["plate_track_0001", "plate_track_0002"]


def test_short_gap_association_and_interpolation() -> None:
    tracks = plate_stabilization.associate_plate_tracks(
        {
            0: [proposal(0, (10, 10, 40, 22))],
            2: [proposal(2, (14, 12, 44, 24))],
        },
        config=plate_stabilization.PlateStabilizationConfig(
            accepted_min_observations=2,
            accepted_min_span_frames=2,
            accepted_min_observation_density=0.6,
            interpolation_max_gap=1,
            review_min_observations=2,
            review_min_span_frames=2,
        ),
    )

    assert len(tracks) == 1
    assert tracks[0].observation_at(1) is not None
    assert tracks[0].observation_at(1).provenance == "short_gap_interpolation"


def test_track_termination_after_bounded_gap() -> None:
    tracks = plate_stabilization.associate_plate_tracks(
        {
            0: [proposal(0, (10, 10, 40, 22))],
            10: [proposal(10, (12, 11, 42, 23))],
        },
        config=plate_stabilization.PlateStabilizationConfig(max_frame_gap=2),
    )

    assert [track.track_id for track in tracks] == ["plate_track_0001", "plate_track_0002"]


def test_persistent_moving_plate_track_is_accepted() -> None:
    proposals = {
        frame: [proposal(frame, (10 + frame, 10, 40 + frame, 22), score=0.7)] for frame in range(25)
    }
    tracks = plate_stabilization.associate_plate_tracks(
        proposals,
        config=plate_stabilization.PlateStabilizationConfig(),
    )

    assert len(tracks) == 1
    assert tracks[0].status == "accepted_plate_track"
    summary = plate_stabilization.summarize_track(tracks[0])
    assert summary["observation_density"] == 1.0
    assert summary["max_internal_gap"] == 0


def test_false_single_frame_proposal_is_rejected() -> None:
    tracks = plate_stabilization.associate_plate_tracks(
        {0: [proposal(0, (10, 10, 40, 22))]},
        config=plate_stabilization.PlateStabilizationConfig(),
    )

    assert tracks[0].status == "rejected_noise_track"
    assert "single_frame_or_ephemeral_track" in tracks[0].rejection_reasons


def test_short_unstable_track_goes_to_review() -> None:
    tracks = plate_stabilization.associate_plate_tracks(
        {
            0: [proposal(0, (10, 10, 40, 22))],
            1: [proposal(1, (12, 10, 42, 22))],
            2: [proposal(2, (14, 10, 44, 22))],
            3: [proposal(3, (16, 10, 46, 22))],
            4: [proposal(4, (18, 10, 48, 22))],
        },
        config=plate_stabilization.PlateStabilizationConfig(),
    )

    assert tracks[0].status == "review_candidate"
    assert "accepted_min_observations_not_met" in tracks[0].review_reasons


def test_same_configuration_can_process_multiple_synthetic_clips() -> None:
    config = plate_stabilization.PlateStabilizationConfig()
    first = plate_stabilization.associate_plate_tracks(
        {0: [proposal(0, (10, 10, 40, 22))]},
        config=config,
    )
    second = plate_stabilization.associate_plate_tracks(
        {5: [proposal(5, (100, 100, 140, 116))]},
        config=config,
    )

    assert config.to_dict()["proposal_threshold"] == 0.2
    assert first[0].track_id == "plate_track_0001"
    assert second[0].track_id == "plate_track_0001"


def test_generic_module_has_no_benchmark_specific_behavior() -> None:
    source = inspect.getsource(plate_stabilization)

    forbidden_terms = [
        "day_plate_far_gold",
        "night_selective_plate_gold",
        "plate_day_01",
        "plate_night_01",
        "eval_day_plate",
        "eval_night_selective",
        "frame 441",
        "frame 483",
    ]
    assert not any(term in source for term in forbidden_terms)
