from __future__ import annotations

from redactor import plate_tracking, privacy_evidence
from redactor.privacy_records import (
    CATEGORY_PLATE,
    GAP_ACCEPTED,
    LIFECYCLE_CONFIRMED,
    LIFECYCLE_NEEDS_REVIEW,
    LIFECYCLE_TEMPORARILY_LOST,
    LIFECYCLE_TERMINATED,
    GapDecisionRecord,
    TemporalRegion,
)


def evidence_config(**overrides) -> privacy_evidence.TemporalEvidenceConfiguration:
    payload = {
        "category": CATEGORY_PLATE,
        "component_weights": {
            "abrupt_jump_penalty": -0.10,
            "aspect_ratio_consistency": 0.08,
            "center_motion_consistency": 0.10,
            "competing_track_penalty": -0.12,
            "internal_gap_penalty": -0.05,
            "lower_tail_confidence": 0.10,
            "mean_confidence": 0.14,
            "observation_density": 0.14,
            "persistence": 0.16,
            "scale_consistency": 0.10,
            "scene_cut_penalty": -0.15,
            "screen_lock_risk": -0.20,
            "source_provenance_support": 0.10,
        },
        "confirmation_threshold": 0.45,
        "continuation_threshold": 0.35,
        "temporary_loss_threshold": 0.25,
        "rejection_threshold": 0.15,
        "min_persistence_seconds": 0.10,
        "min_observation_count": 3,
        "min_review_observation_count": 2,
        "min_observation_density": 0.20,
        "min_mean_confidence": 0.40,
        "min_lower_tail_confidence": 0.35,
        "max_safe_gap_seconds": 0.10,
        "max_stitch_gap_seconds": 0.50,
        "max_temporary_loss_seconds": 0.50,
        "termination_gap_seconds": 0.90,
        "max_center_jump_norm": 4.00,
        "max_scale_jump": 3.00,
        "max_aspect_ratio_std": 1.00,
        "stitch_max_center_distance_norm": 2.00,
        "stitch_max_scale_ratio": 2.00,
        "stitch_max_aspect_ratio_delta": 1.00,
        "screen_lock_min_seconds": 0.40,
        "screen_lock_center_std_norm": 0.03,
        "screen_lock_size_std_norm": 0.02,
        "screen_lock_review_threshold": 0.75,
        "max_competing_frame_fraction": 0.10,
        "profile_id": "test_plate_evidence",
    }
    payload.update(overrides)
    return privacy_evidence.TemporalEvidenceConfiguration(**payload)


def region(
    frame_index: int,
    bbox: tuple[int, int, int, int],
    *,
    category: str = CATEGORY_PLATE,
    track_id: str = "track_a",
    confidence: float = 0.80,
    render_eligible: bool = True,
) -> TemporalRegion:
    return TemporalRegion(
        category=category,
        object_id=track_id,
        source_frame=frame_index,
        bbox_xyxy=bbox,
        source_provenance="plate_detector_observation",
        track_state="tentative",
        render_eligible=render_eligible,
        review_status="needs_review",
        confidence=confidence,
    )


def tracklet(
    track_id: str,
    frames: list[int],
    *,
    category: str = CATEGORY_PLATE,
    start_x: int = 100,
    step_x: int = 3,
    confidence: float = 0.80,
    render_eligible: bool = True,
    competing_frames: tuple[int, ...] = (),
) -> privacy_evidence.EvidenceTracklet:
    return privacy_evidence.EvidenceTracklet(
        category=category,
        track_id=track_id,
        regions=tuple(
            region(
                frame,
                (start_x + index * step_x, 100, start_x + index * step_x + 80, 140),
                category=category,
                track_id=track_id,
                confidence=confidence,
                render_eligible=render_eligible,
            )
            for index, frame in enumerate(frames)
        ),
        competing_frames=competing_frames,
    )


def test_component_normalization_and_deterministic_score() -> None:
    config = evidence_config()
    engine = privacy_evidence.TemporalEvidenceEngine(config, fps=30.0)
    item = tracklet("track_a", [0, 1, 2, 3])

    first = engine.snapshot(item)
    second = engine.snapshot(item)

    assert first.accumulated_score == second.accumulated_score
    assert first.components.persistence == 1.0
    assert first.components.observation_density == 1.0
    assert first.components.mean_confidence == 0.8
    assert config.configuration_hash() == evidence_config().configuration_hash()


def test_confirmation_transition_and_retrospective_promotion_preserve_provenance() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(evidence_config(), fps=30.0)
    result = engine.evaluate_tracklet(tracklet("track_a", [0, 1, 2, 3]))

    assert result.final_state == LIFECYCLE_CONFIRMED
    assert [transition.to_state for transition in result.transitions] == [
        "tentative",
        LIFECYCLE_CONFIRMED,
    ]
    assert result.promotion.first_confirmation_frame == 2
    assert result.promotion.promoted_frames == (0, 1, 2, 3)
    assert all(
        item.source_provenance == "plate_detector_observation"
        for item in tracklet("track_a", [0, 1, 2, 3]).regions
    )


def test_no_retrospective_promotion_for_nonrenderable_regions() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(evidence_config(), fps=30.0)
    result = engine.evaluate_tracklet(tracklet("track_a", [0, 1, 2, 3], render_eligible=False))

    assert result.final_state == LIFECYCLE_CONFIRMED
    assert result.promotion.promoted_frames == ()
    assert {item["reason"] for item in result.promotion.rejected_frames} == {"not_render_eligible"}


def test_interpolation_cannot_satisfy_detector_observation_confirmation_count() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(
        evidence_config(min_observation_count=4),
        fps=30.0,
    )
    regions = (
        region(0, (100, 100, 180, 140)),
        region(1, (104, 100, 184, 140)),
        *(
            TemporalRegion(
                category=CATEGORY_PLATE,
                object_id="track_a",
                source_frame=frame,
                bbox_xyxy=(100 + frame * 4, 100, 180 + frame * 4, 140),
                source_provenance="plate_track_interpolation",
                track_state="tentative",
                render_eligible=True,
                review_status="needs_review",
                confidence=0.80,
            )
            for frame in range(2, 8)
        ),
    )

    result = engine.evaluate_tracklet(
        privacy_evidence.EvidenceTracklet(
            category=CATEGORY_PLATE,
            track_id="track_a",
            regions=regions,
        )
    )

    assert result.final_state == LIFECYCLE_NEEDS_REVIEW
    assert result.snapshot.region_count == 8
    assert result.snapshot.detector_observation_count == 2
    assert "insufficient_detector_observations" in result.snapshot.reasons


def test_hysteresis_records_temporary_loss_and_termination() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(evidence_config(), fps=30.0)
    temporary = engine.evaluate_tracklet(tracklet("track_a", [0, 1, 2, 14, 15, 16]))
    terminated = engine.evaluate_tracklet(tracklet("track_b", [0, 1, 2, 40, 41, 42]))

    assert LIFECYCLE_TEMPORARILY_LOST in {
        transition.to_state for transition in temporary.transitions
    }
    assert LIFECYCLE_TERMINATED in {transition.to_state for transition in terminated.transitions}


def test_compatible_tracklet_stitching_promotes_children() -> None:
    config = evidence_config(min_observation_count=4)
    engine = privacy_evidence.TemporalEvidenceEngine(config, fps=30.0)
    run = engine.evaluate_tracklets(
        (
            tracklet("left", [0, 1], start_x=100),
            tracklet("right", [5, 6], start_x=112),
        )
    )

    assert run.stitch_decisions[0].state == GAP_ACCEPTED
    assert run.results["left"].final_state == LIFECYCLE_CONFIRMED
    assert run.results["right"].stitch_parent_id is not None
    assert run.results["left"].stitch_child_ids == ("left", "right")


def test_incompatible_or_competing_tracklets_remain_reviewable() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(evidence_config(), fps=30.0)
    incompatible = engine.evaluate_tracklets(
        (
            tracklet("left", [0, 1, 2], start_x=100),
            tracklet("right", [5, 6, 7], start_x=900),
        )
    )
    competing = engine.evaluate_tracklet(
        tracklet("track_c", [0, 1, 2, 3], competing_frames=(0, 1, 2, 3))
    )

    assert incompatible.stitch_decisions[0].state != GAP_ACCEPTED
    assert competing.final_state == LIFECYCLE_NEEDS_REVIEW
    assert "competing_track_penalty" in competing.snapshot.reasons


def test_persistent_screen_fixed_overlay_is_not_confirmed() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(evidence_config(), fps=30.0)
    fixed = privacy_evidence.EvidenceTracklet(
        category=CATEGORY_PLATE,
        track_id="fixed_overlay",
        regions=tuple(region(frame, (100, 100, 180, 140)) for frame in range(20)),
    )

    result = engine.evaluate_tracklet(fixed)

    assert result.final_state == LIFECYCLE_NEEDS_REVIEW
    assert result.snapshot.components.screen_lock_risk >= 0.75


def test_stationary_real_plate_with_detector_jitter_is_not_automatically_rejected() -> None:
    engine = privacy_evidence.TemporalEvidenceEngine(evidence_config(), fps=30.0)
    jittered = privacy_evidence.EvidenceTracklet(
        category=CATEGORY_PLATE,
        track_id="stationary_real",
        regions=tuple(
            region(frame, (100 + (frame % 3) * 3, 100, 180 + (frame % 3) * 3, 140))
            for frame in range(20)
        ),
    )

    result = engine.evaluate_tracklet(jittered)

    assert result.final_state == LIFECYCLE_CONFIRMED
    assert result.snapshot.components.screen_lock_risk < 0.75


def test_long_unresolved_gap_stays_blocking_review_evidence() -> None:
    config = evidence_config(max_safe_gap_seconds=0.10, max_temporary_loss_seconds=0.30)
    engine = privacy_evidence.TemporalEvidenceEngine(config, fps=30.0)
    gap_record = GapDecisionRecord(
        category=CATEGORY_PLATE,
        track_id="track_a",
        frame_start=3,
        frame_end=20,
        strategy="review_required_gap",
        state="needs_review",
        reasons=("long_gap_not_defensible",),
        unresolved_frames=18,
    )
    result = engine.evaluate_tracklet(
        privacy_evidence.EvidenceTracklet(
            category=CATEGORY_PLATE,
            track_id="track_a",
            regions=tracklet("track_a", [0, 1, 2, 21, 22, 23]).regions,
            gap_decisions=(gap_record,),
        )
    )

    assert result.final_state == LIFECYCLE_CONFIRMED
    assert "internal_gap_penalty" in result.snapshot.reasons
    assert result.promotion.promoted_frames == (0, 1, 2, 21, 22, 23)


def test_future_mock_category_uses_engine_without_core_changes() -> None:
    config = evidence_config(category="badge", profile_id="badge_temporal_evidence_test")
    engine = privacy_evidence.TemporalEvidenceEngine(config, fps=30.0)
    item = tracklet("badge_track", [0, 1, 2, 3], category="badge")

    result = engine.evaluate_tracklet(item)

    assert result.final_state == LIFECYCLE_CONFIRMED
    assert result.snapshot.category == "badge"


def test_plate_tracking_exposes_evidence_hash_and_promoted_frames() -> None:
    result = plate_tracking.track_plate_observations(
        {
            frame: [
                plate_tracking.PlateObservation(
                    frame_index=frame,
                    bbox_xyxy=(100 + frame * 2, 100, 180 + frame * 2, 140),
                    score=0.8,
                    detector_id="oim_yolov9_s_608",
                    candidate_id=f"c{frame}",
                    source_detection_index=0,
                    metadata={},
                )
            ]
            for frame in range(10)
        },
        frame_count=10,
        fps=30.0,
        width=1920,
        height=1080,
        config=plate_tracking.default_config(),
    )

    assert result.evidence_configuration_hash == (
        plate_tracking.default_evidence_config().configuration_hash()
    )
    assert result.tracks[0].promoted_frames == list(range(10))
    assert all(frame.status == plate_tracking.ACCEPTED_TRACK for frame in result.frame_tracks[0])
