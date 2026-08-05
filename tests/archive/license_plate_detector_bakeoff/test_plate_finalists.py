from __future__ import annotations

import pytest

from experiments.archive.license_plate_detector_bakeoff import (
    license_plate_detector_finalists_v2_1 as finalists_v2_1,
)
from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff, plate_finalists


def detection(
    frame_index: int,
    bbox: tuple[int, int, int, int],
    *,
    score: float = 0.9,
    detector_id: str = "candidate",
) -> plate_bakeoff.PlateDetection:
    return plate_bakeoff.PlateDetection(
        detector_id=detector_id,
        frame_index=frame_index,
        score=score,
        bbox_xyxy=bbox,
        inference_time_ms=1.0,
    )


def annotation(
    frame_index: int,
    object_id: str,
    bbox: tuple[int, int, int, int],
    *,
    visibility: str = plate_finalists.VISIBILITY_CLEAR,
    scoring_policy: str = plate_finalists.POLICY_SCORED,
) -> plate_finalists.PlateAnnotation:
    return plate_finalists.PlateAnnotation(
        clip_id="clip",
        source_sha256="abc",
        frame_index=frame_index,
        object_id=object_id,
        bbox_xyxy=bbox,
        visibility=visibility,
        scoring_policy=scoring_policy,
        reviewer_provenance="test",
    )


def test_annotation_schema_rejects_invalid_visibility() -> None:
    item = annotation(1, "plate", (10, 10, 30, 20), visibility="opaque")

    with pytest.raises(ValueError, match="invalid plate visibility"):
        item.validate()


def test_multiple_real_plates_in_one_frame_are_scored() -> None:
    annotations = [
        annotation(1, "primary", (10, 10, 50, 25)),
        annotation(1, "passing_vehicle", (100, 30, 150, 50)),
    ]
    records = {1: [detection(1, (10, 10, 50, 25)), detection(1, (100, 30, 150, 50))]}

    metrics = plate_finalists.evaluate_all_visible_plate_sample(
        annotations,
        records,
        threshold=0.5,
    )

    assert metrics["scored_plate_instances"] == 2
    assert metrics["plate_instance_recall"] == 1.0
    assert metrics["false_positive_detections"] == 0
    assert metrics["frame_complete_recall"] == 1.0


def test_secondary_real_plate_is_not_false_positive() -> None:
    annotations = [
        annotation(1, "benchmark_target", (10, 10, 50, 25)),
        annotation(1, "secondary_real_plate", (80, 12, 130, 28)),
    ]
    records = {1: [detection(1, (80, 12, 130, 28))]}

    metrics = plate_finalists.evaluate_all_visible_plate_sample(
        annotations,
        records,
        threshold=0.5,
    )

    assert metrics["detected_plate_instances"] == 1
    assert metrics["missed_plate_instances"] == 1
    assert metrics["false_positive_detections"] == 0


def test_true_non_plate_detection_counts_as_false_positive() -> None:
    annotations = [annotation(1, "target", (10, 10, 50, 25))]
    records = {1: [detection(1, (10, 10, 50, 25)), detection(1, (200, 200, 260, 230))]}

    metrics = plate_finalists.evaluate_all_visible_plate_sample(
        annotations,
        records,
        threshold=0.5,
    )

    assert metrics["false_positive_detections"] == 1
    assert metrics["false_positive_frames"] == 1
    assert metrics["plate_instance_precision"] == 0.5


def test_ignored_ambiguous_plate_does_not_count_as_false_positive() -> None:
    annotations = [
        annotation(
            1,
            "tiny_possible_plate",
            (200, 200, 212, 206),
            visibility=plate_finalists.VISIBILITY_AMBIGUOUS,
            scoring_policy=plate_finalists.POLICY_IGNORE,
        )
    ]
    records = {1: [detection(1, (200, 200, 212, 206))]}

    metrics = plate_finalists.evaluate_all_visible_plate_sample(
        annotations,
        records,
        threshold=0.5,
    )

    assert metrics["scored_plate_instances"] == 0
    assert metrics["false_positive_detections"] == 0
    assert metrics["frame_details"][0]["detections"][0]["match_kind"] == "ignored_plate"


def test_duplicate_detection_on_same_plate_is_classified() -> None:
    annotations = [annotation(1, "target", (10, 10, 50, 25))]
    records = {
        1: [detection(1, (10, 10, 50, 25), score=0.9), detection(1, (11, 10, 51, 25), score=0.8)]
    }

    metrics = plate_finalists.evaluate_all_visible_plate_sample(
        annotations,
        records,
        threshold=0.5,
    )

    assert metrics["detected_plate_instances"] == 1
    assert metrics["duplicate_detections"] == 1
    assert metrics["duplicate_frames"] == 1


def test_candidate_disagreement_sampling_detects_strong_difference() -> None:
    first = {1: [detection(1, (10, 10, 50, 25), score=0.7)], 2: []}
    second = {1: [], 2: [detection(2, (10, 10, 50, 25), score=0.4)]}

    frames = plate_finalists.candidate_disagreement_frames(
        first,
        second,
        [1, 2],
        first_threshold=0.5,
        second_threshold=0.3,
    )

    assert frames == [1]


def test_temporal_gap_distribution_buckets_runs() -> None:
    gaps = plate_finalists.gap_distribution([0, 4, 15], list(range(0, 20)))

    assert gaps["gaps_1_to_3"] == 1
    assert gaps["gaps_4_to_10"] == 2
    assert gaps["gaps_over_10"] == 0
    assert gaps["longest_gap"] == 10


def test_finalists_exclude_rejected_rfdetr() -> None:
    candidate_ids = {config.candidate_id for config in finalists_v2_1.finalist_configs()}

    assert candidate_ids == {
        finalists_v2_1.RT_DETR_ID,
        finalists_v2_1.OIM_ID,
    }
    assert finalists_v2_1.RF_DETR_REJECTED_ID not in candidate_ids


def test_finalist_configuration_is_identical_across_clips() -> None:
    configs = finalists_v2_1.finalist_configs()

    assert {config.output_id: config.threshold for config in configs} == {
        "oim_yolov9_s_608": 0.30,
        "rtdetr_v2": 0.05,
    }
    assert {config.output_id: config.mode_id for config in configs} == {
        "oim_yolov9_s_608": "uniform_small_object_tiles",
        "rtdetr_v2": "native_whole_frame",
    }


def test_day_sample_adjudication_rejects_road_surface_proposal() -> None:
    road_marking = detection(100, (88, 895, 568, 1016), score=0.55)

    assert (
        finalists_v2_1.adjudicate_detection_semantics(
            "day_plate_primary",
            road_marking,
        )
        == "false"
    )


def test_day_sample_adjudication_counts_secondary_real_plate() -> None:
    secondary_vehicle_plate = detection(0, (562, 581, 600, 598), score=0.52)

    assert (
        finalists_v2_1.adjudicate_detection_semantics(
            "day_plate_primary",
            secondary_vehicle_plate,
        )
        == "plate"
    )


def test_close_sample_adjudication_keeps_stress_plate_and_rejects_overlay() -> None:
    stress_plate = detection(300, (801, 453, 846, 476), score=0.6)
    overlay_false_positive = detection(300, (201, 896, 466, 947), score=0.8)
    sign_false_positive = detection(330, (972, 313, 1110, 386), score=0.4)

    assert (
        finalists_v2_1.adjudicate_detection_semantics(
            "day_plate_close_stress",
            stress_plate,
        )
        == "plate"
    )
    assert (
        finalists_v2_1.adjudicate_detection_semantics(
            "day_plate_close_stress",
            overlay_false_positive,
        )
        == "false"
    )
    assert (
        finalists_v2_1.adjudicate_detection_semantics(
            "day_plate_close_stress",
            sign_false_positive,
        )
        == "false"
    )


def test_night_sample_adjudication_rejects_lower_false_band() -> None:
    lower_false_band = detection(441, (99, 897, 565, 1017), score=0.06)

    assert (
        finalists_v2_1.adjudicate_detection_semantics(
            "night_plate_primary",
            lower_false_band,
        )
        == "false"
    )
