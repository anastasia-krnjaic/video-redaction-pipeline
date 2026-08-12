from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from redactor import (
    evaluation,
    media,
    plate_detection,
    plate_evaluation,
    review_finalization,
    reviewed_regions,
)
from redactor.cli import main
from redactor.privacy_records import APPROVAL_ACCEPTED, PrivacyRegionRecord

POLICY_PATH = Path("data/evaluations/plate_evaluation_policy_v1.0.json")
PLATE_GT_PATH = Path("data/ground_truth/plate_ground_truth_v1.0.json")


def policy() -> dict[str, Any]:
    return copy.deepcopy(plate_evaluation.load_policy(POLICY_PATH))


def restamp_policy(payload: dict[str, Any]) -> dict[str, Any]:
    restamped = copy.deepcopy(payload)
    restamped["content_sha256"] = plate_evaluation.policy_content_sha256(restamped)
    return restamped


def gt_record(
    frame: int = 0,
    bbox: plate_evaluation.Box = (100, 100, 110, 110),
    *,
    target_id: str = "plate_target",
) -> plate_evaluation.GroundTruthRecord:
    return plate_evaluation.GroundTruthRecord(
        benchmark_id="synthetic",
        bbox_xyxy=bbox,
        dense_frame=frame,
        parent_frame=frame,
        target_id=target_id,
    )


def prediction(
    bbox: plate_evaluation.Box | None = (100, 100, 110, 110),
    *,
    frame: int = 0,
    score: float = 0.30,
    category: str = "license_plate",
    detector_id: str = plate_detection.DETECTOR_ID,
    configuration_hash: str = plate_evaluation.EXPECTED_DETECTOR_CONFIGURATION_HASH,
    provenance: str = "",
    source_order: int = 0,
) -> plate_evaluation.DetectorPrediction:
    return plate_evaluation.DetectorPrediction(
        bbox_xyxy=bbox,
        category=category,
        configuration_hash=configuration_hash,
        detector_id=detector_id,
        frame_index=frame,
        provenance=provenance,
        score=score,
        source_order=source_order,
    )


def region(
    bbox: plate_evaluation.Box | None = (0, 0, 100, 100),
    *,
    frame: int = 0,
    category: str = "license_plate",
    object_id: str | None = "plate_target",
    provenance: str = "detector",
    status: str = "covered",
    source_order: int = 0,
    track_uid: str | None = None,
) -> plate_evaluation.AutomaticRegion:
    return plate_evaluation.AutomaticRegion(
        bbox_xyxy=bbox,
        category=category,
        frame_index=frame,
        object_id=object_id,
        provenance=provenance,
        source_order=source_order,
        status=status,
        track_uid=track_uid,
    )


def covered_metric(
    records: list[plate_evaluation.GroundTruthRecord],
    regions: list[plate_evaluation.AutomaticRegion],
) -> dict[str, Any]:
    return plate_evaluation.automatic_metrics_for_target(
        records,
        regions,
        policy(),
        margin_fraction=0.0,
        margin_pixels=0,
    )


@pytest.mark.parametrize(
    ("name", "record_box", "pred_box", "score", "expected_hit"),
    [
        ("exact_match", (100, 100, 110, 110), (100, 100, 110, 110), 0.90, True),
        ("threshold_exactly_030", (100, 100, 110, 110), (100, 100, 110, 110), 0.30, True),
        ("below_confidence", (100, 100, 110, 110), (100, 100, 110, 110), 0.299, False),
        ("shifted_valid_tiny_plate", (100, 100, 132, 116), (102, 101, 134, 117), 0.80, True),
        ("iou_exactly_020", (100, 100, 110, 110), (93, 95, 118, 115), 0.80, True),
        ("iou_below_020", (100, 100, 110, 110), (92, 95, 118, 116), 0.80, False),
        ("gt_coverage_exactly_060", (100, 100, 110, 110), (100, 100, 106, 110), 0.80, True),
        ("below_gt_coverage", (100, 100, 110, 110), (100, 100, 105, 110), 0.80, False),
        ("area_ratio_upper_boundary", (100, 100, 110, 110), (93, 95, 118, 115), 0.80, True),
        ("width_ratio_upper_boundary", (100, 100, 110, 110), (90, 98, 120, 113), 0.80, True),
        ("height_ratio_upper_boundary", (100, 100, 110, 110), (98, 90, 113, 120), 0.80, True),
        ("center_expansion_boundary", (100, 100, 110, 110), (100, 100, 130, 110), 0.80, True),
        ("center_expansion_outside", (100, 100, 110, 110), (101, 100, 131, 110), 0.80, False),
        ("oversized_bumper_like_box", (100, 100, 110, 110), (80, 90, 140, 120), 0.80, False),
        ("tiny_inner_box", (100, 100, 110, 110), (102, 102, 104, 104), 0.80, False),
    ],
)
def test_detector_matching_policy_cases(
    name: str,
    record_box: plate_evaluation.Box,
    pred_box: plate_evaluation.Box,
    score: float,
    expected_hit: bool,
) -> None:
    metrics = plate_evaluation.detector_metrics_for_target(
        [gt_record(bbox=record_box)],
        [prediction(pred_box, score=score)],
        policy(),
    )

    assert metrics["hit_frame_count"] == (1 if expected_hit else 0), name
    assert metrics["formal_precision"] == {
        "reason": "insufficient_spatial_negative_ignore_map",
        "value": None,
    }


def test_detector_pair_diagnostics_cover_redundant_lower_boundaries() -> None:
    loaded_policy = policy()
    gt_box = (100, 100, 110, 110)
    exact_pred_cov = plate_evaluation.pair_diagnostics(
        (104, 95, 129, 115),
        gt_box,
        loaded_policy,
    )
    below_pred_cov = plate_evaluation.pair_diagnostics(
        (105, 95, 130, 115),
        gt_box,
        loaded_policy,
    )
    exact_area_ratio = plate_evaluation.pair_diagnostics(
        (100, 100, 102, 110),
        gt_box,
        loaded_policy,
    )
    exact_width_ratio = plate_evaluation.pair_diagnostics(
        (100, 100, 133, 110),
        (100, 100, 200, 110),
        loaded_policy,
    )

    assert exact_pred_cov.prediction_coverage == pytest.approx(0.12)
    assert below_pred_cov.prediction_coverage < 0.12
    assert exact_area_ratio.area_ratio == pytest.approx(0.20)
    assert exact_width_ratio.width_ratio == pytest.approx(0.33)
    assert not exact_pred_cov.matches
    assert not below_pred_cov.matches


def test_detector_duplicates_tie_breaking_and_miss_intervals() -> None:
    records = [gt_record(frame=frame) for frame in range(4)]
    predictions = [
        prediction((100, 100, 110, 110), score=0.60, frame=0, source_order=0),
        prediction((100, 100, 110, 110), score=0.90, frame=0, source_order=1),
        prediction((100, 100, 110, 110), score=0.75, frame=3, source_order=2),
    ]

    metrics = plate_evaluation.detector_metrics_for_target(records, predictions, policy())

    assert metrics["hit_frames"] == [0, 3]
    assert metrics["duplicate_matched_target_proposal_frames"] == [0]
    assert metrics["confidence_summary"]["max"] == 0.90
    assert metrics["miss_intervals"] == [[1, 2]]
    assert metrics["maximum_consecutive_miss_run"] == 2


def test_detector_global_assignment_two_targets_one_prediction() -> None:
    records = [
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_a"),
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_b"),
    ]
    metrics = plate_evaluation.detector_metrics_for_targets(
        records,
        [prediction((100, 100, 120, 120), frame=0, score=0.90, source_order=0)],
        policy(),
    )

    assert metrics["target_a"]["hit_frames"] == [0]
    assert metrics["target_b"]["miss_frames"] == [0]
    assert sum(item["hit_frame_count"] for item in metrics.values()) == 1


def test_detector_global_assignment_uses_pair_quality_before_source_order() -> None:
    records = [
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_a"),
        gt_record(frame=0, bbox=(104, 100, 124, 120), target_id="target_b"),
    ]
    predictions = [
        prediction((104, 100, 124, 120), frame=0, score=0.60, source_order=0),
        prediction((100, 100, 120, 120), frame=0, score=0.90, source_order=1),
    ]

    metrics = plate_evaluation.detector_metrics_for_targets(records, predictions, policy())

    assert metrics["target_a"]["hit_frames"] == [0]
    assert metrics["target_b"]["hit_frames"] == [0]
    assert sum(item["hit_frame_count"] for item in metrics.values()) == 2


def test_detector_global_assignment_identical_pair_quality_is_deterministic() -> None:
    records = [
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_b"),
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_a"),
    ]
    predictions = [prediction((100, 100, 120, 120), frame=0, source_order=0)]

    first = plate_evaluation.detector_metrics_for_targets(records, predictions, policy())
    second = plate_evaluation.detector_metrics_for_targets(records, predictions, policy())

    assert first == second
    assert first["target_a"]["hit_frames"] == [0]
    assert first["target_b"]["miss_frames"] == [0]


def test_detector_duplicate_predictions_do_not_duplicate_target_hit() -> None:
    records = [gt_record(frame=0, bbox=(100, 100, 120, 120))]
    predictions = [
        prediction((100, 100, 120, 120), frame=0, score=0.80, source_order=0),
        prediction((100, 100, 120, 120), frame=0, score=0.90, source_order=1),
    ]

    metrics = plate_evaluation.detector_metrics_for_targets(records, predictions, policy())
    target = metrics["plate_target"]

    assert target["hit_frame_count"] == 1
    assert target["duplicate_matched_target_proposal_frames"] == [0]
    assert target["unmatched_proposal_count"] == 1


def test_detector_global_assignment_three_targets_mixed_matchability_deterministic() -> None:
    records = [
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_a"),
        gt_record(frame=0, bbox=(100, 100, 120, 120), target_id="target_b"),
        gt_record(frame=1, bbox=(200, 200, 230, 220), target_id="target_c"),
    ]
    predictions = [
        prediction((100, 100, 120, 120), frame=0, score=0.90, source_order=0),
        prediction((200, 200, 230, 220), frame=1, score=0.70, source_order=1),
    ]

    first = plate_evaluation.detector_metrics_for_targets(records, predictions, policy())
    second = plate_evaluation.detector_metrics_for_targets(records, predictions, policy())

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["target_a"]["hit_frame_count"] == 1
    assert first["target_b"]["miss_frame_count"] == 1
    assert first["target_c"]["hit_frame_count"] == 1
    assert sum(item["hit_frame_count"] for item in first.values()) == 2


@pytest.mark.parametrize(
    ("bad_prediction", "reason"),
    [
        (
            prediction(category="face"),
            "wrong category in detector artifact",
        ),
        (
            prediction(configuration_hash="bad"),
            "wrong detector configuration in detector artifact",
        ),
        (
            prediction(provenance="plate_track_interpolation"),
            "detector artifact contains non-direct provenance",
        ),
    ],
)
def test_detector_artifact_fail_closed_blockers(
    bad_prediction: plate_evaluation.DetectorPrediction,
    reason: str,
) -> None:
    assert reason in plate_evaluation.detector_artifact_blockers([bad_prediction])


def test_detector_source_and_config_mismatch_blockers(tmp_path: Path) -> None:
    loaded_policy = policy()
    ground_truth = plate_evaluation.load_ground_truth(PLATE_GT_PATH, loaded_policy)
    benchmark = plate_evaluation.benchmark_by_id(ground_truth, "day_plate")
    source = {"frame_count": 570, "height": 1080, "sha256": "bad", "width": 1920}
    config = json.loads(plate_evaluation.DEFAULT_DETECTOR_CONFIG_PATH.read_text())
    config["configuration_hash"] = "bad"
    config_path = tmp_path / "selected_configuration.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert "source video hash mismatch" in plate_evaluation.validate_source_for_benchmark(
        source,
        benchmark,
    )
    assert "selected detector configuration file hash mismatch" in (
        plate_evaluation.validate_detector_config_file(config_path, loaded_policy)
    )


@pytest.mark.parametrize(
    ("regions", "expected_frame_coverage", "expected_rate"),
    [
        ([region((0, 0, 100, 100))], 1.0, 1.0),
        ([region((0, 0, 95, 100))], 0.95, 1.0),
        ([region((0, 0, 94, 100))], 0.94, 0.0),
        (
            [region((0, 0, 50, 100)), region((50, 0, 95, 100), source_order=1)],
            0.95,
            1.0,
        ),
        (
            [region((0, 0, 60, 100)), region((40, 0, 95, 100), source_order=1)],
            0.95,
            1.0,
        ),
    ],
)
def test_automatic_coverage_union_cases(
    regions: list[plate_evaluation.AutomaticRegion],
    expected_frame_coverage: float,
    expected_rate: float,
) -> None:
    metrics = covered_metric([gt_record(bbox=(0, 0, 100, 100))], regions)

    assert metrics["coverage_rate"] == expected_rate
    assert metrics["per_frame_gt_coverage"]["0"] == pytest.approx(expected_frame_coverage)


def test_automatic_coverage_intervals_prefix_flicker_and_tail() -> None:
    records = [gt_record(frame=frame, bbox=(0, 0, 100, 100)) for frame in range(5)]
    metrics = covered_metric(
        records,
        [region((0, 0, 100, 100), frame=2), region((0, 0, 100, 100), frame=4)],
    )
    trailing = covered_metric(records[:4], [region((0, 0, 100, 100), frame=0)])

    assert metrics["unresolved_prefix"] == [0, 1]
    assert metrics["uncovered_intervals"] == [[0, 1], [3, 3]]
    assert metrics["flicker_events"] == 1
    assert trailing["uncovered_intervals"] == [[1, 3]]
    assert trailing["flicker_events"] == 0


def test_automatic_provenance_and_over_redaction_diagnostics() -> None:
    records = [gt_record(frame=frame, bbox=(0, 0, 100, 100)) for frame in range(3)]
    metrics = covered_metric(
        records,
        [
            region((0, 0, 200, 200), frame=0, provenance="direct_detector"),
            region((0, 0, 100, 100), frame=1, provenance="ordinary_tracker"),
            region((0, 0, 100, 100), frame=2, provenance="plate_track_interpolation"),
            region((200, 200, 240, 240), frame=2, provenance="detector", source_order=3),
        ],
    )

    assert metrics["coverage_rate"] == 1.0
    assert metrics["provenance_breakdown"] == {
        "direct_detector": 1,
        "ordinary_tracker": 1,
        "plate_track_interpolation": 1,
        "detector": 1,
    }
    assert metrics["over_redaction"]["unmatched_automatic_region_count"] == 1
    assert metrics["over_redaction"]["redacted_area_ratio_by_frame"]["0"] > 0


def test_automatic_excludes_needs_review_and_rejects_manual_reviewed() -> None:
    loaded_policy = policy()
    needs_review = region(status="needs_review", provenance="detector")
    manual = region(provenance="manual_correction")
    reviewed = region(provenance="reviewed_replacement")

    metrics = covered_metric([gt_record(bbox=(0, 0, 100, 100))], [needs_review])

    assert metrics["coverage_rate"] == 0.0
    assert "automatic input contains manual/reviewed provenance" in (
        plate_evaluation.automatic_contamination_blockers([manual, reviewed], loaded_policy)
    )


def test_automatic_target_association_is_separate_from_privacy_coverage() -> None:
    metrics = covered_metric(
        [gt_record(bbox=(0, 0, 100, 100))],
        [region((0, 0, 100, 100), object_id="other_plate")],
    )

    assert metrics["coverage_rate"] == 1.0
    assert metrics["associated_covered_frames"] == []
    assert metrics["association_unverified_coverage_frames"] == [0]


def reviewed_record(
    *,
    bbox: plate_evaluation.Box = (0, 0, 100, 100),
    case_id: str = "case_a",
    category: str = "license_plate",
    object_id: str = "plate_target",
    source_frame: int = 0,
    source_sha: str = "source_sha",
) -> PrivacyRegionRecord:
    return PrivacyRegionRecord(
        approval_state=APPROVAL_ACCEPTED,
        bbox_xyxy=bbox,
        category=category,
        object_id=object_id,
        provenance="manual_redraw",
        render_eligible=True,
        source_frame=source_frame,
        source_record={
            "case_id": case_id,
            "source_sha256": source_sha,
            "source_video_hash": source_sha,
        },
    )


def reviewed_binding_fixture(
    tmp_path: Path,
    *,
    case_id: str = "case_a",
    records: list[PrivacyRegionRecord] | None = None,
    source_sha: str = "source_sha",
) -> dict[str, Any]:
    records = records or [reviewed_record(case_id=case_id, source_sha=source_sha)]
    package_root = tmp_path / f"package_{case_id}"
    revision_id = f"latest_reviewed_{case_id}"
    revision_root = (
        package_root
        / "sessions"
        / "session_a"
        / "latest_reviewed"
        / case_id
        / revision_id
    )
    revision_root.mkdir(parents=True)
    regions_path = revision_root / "reviewed_regions.jsonl"
    review_finalization.write_privacy_region_records(regions_path, records)
    video_path = revision_root / "latest_reviewed.mp4"
    video_path.write_bytes(b"latest reviewed video")
    media_validation_path = revision_root / "media_validation.json"
    media_validation_path.write_text('{"passed":true}\n', encoding="utf-8")
    region_validation_path = revision_root / "region_validation.json"
    region_validation_path.write_text('{"passed":true}\n', encoding="utf-8")
    records_sha = reviewed_regions.semantic_region_set_hash(records)
    identity = {
        "active_decisions_hash": "decisions",
        "automatic_baseline_region_set_sha256": "baseline",
        "case_id": case_id,
        "category_run_manifest_identity_hash": "category_runs",
        "compiler_schema_version": reviewed_regions.COMPILER_SCHEMA_VERSION,
        "configuration_freeze_hash": "freeze",
        "merge_rejection_state_hash": "merge",
        "object_policy_hash": "policy",
        "renderer_configuration_sha256": "renderer",
        "reviewed_entity_count": 1,
        "reviewed_region_count": len(records),
        "reviewed_region_records_sha256": records_sha,
        "selected_categories": sorted({record.category for record in records}),
        "source_frame_end": 10,
        "source_frame_start": 0,
        "source_video_hash": source_sha,
    }
    identity["reviewed_region_set_sha256"] = reviewed_regions.reviewed_region_set_identity_hash(
        identity
    )
    revision_manifest = {
        "case_id": case_id,
        "media_validation": "media_validation.json",
        "media_validation_passed": True,
        "output_video": "latest_reviewed.mp4",
        "region_validation": "region_validation.json",
        "region_validation_passed": True,
        "reviewed_region_identity": identity,
        "reviewed_region_set_sha256": identity["reviewed_region_set_sha256"],
        "reviewed_regions": "reviewed_regions.jsonl",
        "reviewed_regions_sha256": media.sha256_file(regions_path),
        "revision_id": revision_id,
        "schema_version": reviewed_regions.REVISION_SCHEMA_VERSION,
        "source_sha256": source_sha,
        "source_video_hash": source_sha,
        "video_artifact_sha256": media.sha256_file(video_path),
    }
    revision_manifest_path = revision_root / "revision_manifest.json"
    revision_manifest_path.write_text(
        json.dumps(revision_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_pointer = {
        "case_id": case_id,
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "revision_id": revision_id,
        "revision_manifest": revision_manifest_path.relative_to(package_root).as_posix(),
        "revision_manifest_payload": revision_manifest,
        "source_sha256": source_sha,
    }
    final_manifest = {
        "latest_reviewed_revision_id": revision_id,
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "source_sha256": source_sha,
    }
    final_pointer = {
        "approved_latest_reviewed_revision_id": revision_id,
        "finalization_manifest_payload": final_manifest,
        "reviewed_region_set_sha256": revision_manifest["reviewed_region_set_sha256"],
        "source_sha256": source_sha,
    }
    return {
        "final_pointer": final_pointer,
        "latest_pointer": latest_pointer,
        "package_root": package_root,
        "records": records,
        "records_sha": records_sha,
        "regions_path": regions_path,
        "revision_manifest": revision_manifest,
        "revision_root": revision_root,
    }


def validate_reviewed_fixture_binding(
    fixture: dict[str, Any],
    *,
    expected_case_id: str = "case_a",
    expected_category: str = "license_plate",
    expected_source_sha256: str = "source_sha",
) -> dict[str, Any]:
    return reviewed_regions.validate_reviewed_regions_artifact_binding(
        fixture["package_root"],
        fixture["regions_path"],
        latest_pointer=fixture["latest_pointer"],
        final_pointer=fixture["final_pointer"],
        expected_case_id=expected_case_id,
        expected_category=expected_category,
        expected_source_sha256=expected_source_sha256,
    )


def test_reviewed_regions_artifact_binding_accepts_exact_set(tmp_path: Path) -> None:
    fixture = reviewed_binding_fixture(tmp_path)

    binding = validate_reviewed_fixture_binding(fixture)

    assert binding["reviewed_region_records_sha256"] == fixture["records_sha"]
    assert binding["reviewed_region_set_sha256"] == (
        fixture["revision_manifest"]["reviewed_region_set_sha256"]
    )


def test_reviewed_regions_artifact_binding_rejects_bbox_tamper(tmp_path: Path) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    review_finalization.write_privacy_region_records(
        fixture["regions_path"],
        [reviewed_record(bbox=(1, 0, 100, 100))],
    )

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="reviewed_regions_sha256",
    ):
        validate_reviewed_fixture_binding(fixture)


@pytest.mark.parametrize(
    ("field", "expected_value", "match"),
    [
        ("source", "other_source", "source SHA mismatch"),
        ("case", "other_case", "case_id mismatch"),
        ("category", "face", "category"),
    ],
)
def test_reviewed_regions_artifact_binding_rejects_identity_mismatch(
    tmp_path: Path,
    field: str,
    expected_value: str,
    match: str,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    kwargs = {
        "expected_case_id": "case_a",
        "expected_category": "license_plate",
        "expected_source_sha256": "source_sha",
    }
    if field == "source":
        kwargs["expected_source_sha256"] = expected_value
    elif field == "case":
        kwargs["expected_case_id"] = expected_value
    else:
        kwargs["expected_category"] = expected_value

    with pytest.raises(reviewed_regions.ReviewedRegionCompilationError, match=match):
        validate_reviewed_fixture_binding(fixture, **kwargs)


def test_reviewed_regions_artifact_binding_rejects_semantic_hash_tamper(
    tmp_path: Path,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    fixture["revision_manifest"]["reviewed_region_identity"][
        "reviewed_region_records_sha256"
    ] = "bad"

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="semantic records SHA mismatch",
    ):
        validate_reviewed_fixture_binding(fixture)


def test_reviewed_regions_artifact_binding_rejects_set_identity_tamper(
    tmp_path: Path,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    fixture["revision_manifest"]["reviewed_region_identity"]["object_policy_hash"] = "changed"

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="canonical identity mismatch",
    ):
        validate_reviewed_fixture_binding(fixture)


def test_reviewed_regions_artifact_binding_rejects_latest_pointer_mismatch(
    tmp_path: Path,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    fixture["latest_pointer"]["reviewed_region_set_sha256"] = "bad"

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="Latest reviewed pointer reviewed_region_set_sha256 mismatch",
    ):
        validate_reviewed_fixture_binding(fixture)


def test_reviewed_regions_artifact_binding_rejects_final_latest_mismatch(
    tmp_path: Path,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    fixture["final_pointer"]["approved_latest_reviewed_revision_id"] = "other_revision"

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="Latest reviewed revision mismatch",
    ):
        validate_reviewed_fixture_binding(fixture)


def test_reviewed_regions_artifact_binding_rejects_wrong_supplied_path(
    tmp_path: Path,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    wrong_path = fixture["package_root"] / "wrong_reviewed_regions.jsonl"
    wrong_path.write_text(fixture["regions_path"].read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="path does not match",
    ):
        reviewed_regions.validate_reviewed_regions_artifact_binding(
            fixture["package_root"],
            wrong_path,
            latest_pointer=fixture["latest_pointer"],
            final_pointer=fixture["final_pointer"],
            expected_case_id="case_a",
            expected_category="license_plate",
            expected_source_sha256="source_sha",
        )


def test_reviewed_regions_artifact_binding_accepts_semantic_reorder_when_manifest_bound(
    tmp_path: Path,
) -> None:
    records = [
        reviewed_record(object_id="plate_a", source_frame=0),
        reviewed_record(category="face", object_id="face_a", source_frame=1),
    ]
    fixture = reviewed_binding_fixture(tmp_path, records=records)
    lines = [
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        for record in reversed(records)
    ]
    fixture["regions_path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    fixture["revision_manifest"]["reviewed_regions_sha256"] = media.sha256_file(
        fixture["regions_path"]
    )

    binding = validate_reviewed_fixture_binding(fixture)

    assert binding["reviewed_region_records_sha256"] == fixture["records_sha"]


def test_reviewed_regions_artifact_binding_rejects_missing_set_sha(tmp_path: Path) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    fixture["revision_manifest"].pop("reviewed_region_set_sha256")

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="reviewed_region_set_sha256 is missing",
    ):
        validate_reviewed_fixture_binding(fixture)


def test_reviewed_regions_artifact_binding_rejects_invalid_region_file(
    tmp_path: Path,
) -> None:
    fixture = reviewed_binding_fixture(tmp_path)
    fixture["regions_path"].write_text("{not json}\n", encoding="utf-8")
    fixture["revision_manifest"]["reviewed_regions_sha256"] = media.sha256_file(
        fixture["regions_path"]
    )

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="invalid reviewed regions artifact",
    ):
        validate_reviewed_fixture_binding(fixture)


def test_reviewed_final_validation_failure_does_not_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"score": 0}

    def fake_reviewed_final_validation(**_kwargs) -> dict[str, Any]:
        return {
            "blockers": ["reviewed regions validation failed: synthetic tamper"],
            "configuration_identity": {},
            "ground_truth_content_sha256": plate_evaluation.EXPECTED_GT_CONTENT_SHA256,
            "input_artifact_hashes": {},
            "mode": plate_evaluation.MODE_REVIEWED_FINAL,
            "policy_sha256": plate_evaluation.EXPECTED_POLICY_SHA256,
            "source_identity": {},
            "status": "blocked",
            "validation_only": True,
        }

    def fake_score_mode(**_kwargs) -> dict[str, Any]:
        calls["score"] += 1
        raise AssertionError("scoring must not run after validation failure")

    monkeypatch.setattr(
        plate_evaluation,
        "reviewed_final_validation",
        fake_reviewed_final_validation,
    )
    monkeypatch.setattr(plate_evaluation, "score_mode", fake_score_mode)

    metrics = plate_evaluation.run_plate_evaluation(
        benchmark_id="day_plate",
        ground_truth_path=PLATE_GT_PATH,
        mode=plate_evaluation.MODE_REVIEWED_FINAL,
        output_dir=tmp_path / "out",
        source_video_path=tmp_path / "source.mp4",
    )

    assert calls["score"] == 0
    assert metrics["publishable_metrics"] is False
    assert metrics["validation"]["blockers"] == [
        "reviewed regions validation failed: synthetic tamper"
    ]


def valid_reviewed_evidence() -> dict[str, Any]:
    return {
        "category_run_lineage_valid": True,
        "expected_reviewed_region_set_sha256": "reviewed_sha",
        "final_current": True,
        "final_pointer_matches_latest": True,
        "final_revision_identity": "final_rev",
        "finalization_valid": True,
        "human_sweep_acknowledged": True,
        "latest_reviewed_current": True,
        "latest_reviewed_revision_identity": "latest_rev",
        "media_validation_valid": True,
        "region_validation_valid": True,
        "reviewed_regions_artifact_valid": True,
        "reviewed_region_set_sha256": "reviewed_sha",
        "source_lineage_valid": True,
    }


def test_reviewed_final_complete_evidence_and_manual_geometry() -> None:
    metrics = plate_evaluation.reviewed_final_metrics_for_target(
        [gt_record(bbox=(0, 0, 100, 100))],
        [region((0, 0, 95, 100), provenance="manual_redraw")],
        policy(),
        evidence=valid_reviewed_evidence(),
    )

    assert metrics["coverage_rate"] == 1.0
    assert metrics["publishable_metrics"] is True
    assert metrics["reviewed_region_set_sha256"] == "reviewed_sha"
    assert metrics["trust_gate_blockers"] == []


def test_reviewed_final_one_uncovered_frame_blocks_publishability() -> None:
    metrics = plate_evaluation.reviewed_final_metrics_for_target(
        [gt_record(frame=0, bbox=(0, 0, 100, 100)), gt_record(frame=1, bbox=(0, 0, 100, 100))],
        [region((0, 0, 100, 100), frame=0, provenance="reviewed_replacement")],
        policy(),
        evidence=valid_reviewed_evidence(),
    )

    assert metrics["uncovered_frames"] == [1]
    assert metrics["publishable_metrics"] is False


@pytest.mark.parametrize(
    ("field", "blocker"),
    [
        ("latest_reviewed_current", "stale_latest_reviewed"),
        ("final_current", "stale_final"),
        ("final_pointer_matches_latest", "final_pointer_mismatch"),
        ("source_lineage_valid", "source_lineage_invalid"),
        ("category_run_lineage_valid", "category_run_lineage_invalid"),
        ("media_validation_valid", "media_validation_failed"),
        ("region_validation_valid", "region_validation_failed"),
        ("reviewed_regions_artifact_valid", "reviewed_regions_artifact_invalid"),
        ("human_sweep_acknowledged", "missing_sweep"),
        ("finalization_valid", "finalization_invalid"),
    ],
)
def test_reviewed_final_trust_gate_blockers(field: str, blocker: str) -> None:
    evidence = valid_reviewed_evidence()
    evidence[field] = False

    assert blocker in plate_evaluation.reviewed_final_trust_gate_blockers(evidence)


def test_reviewed_final_pointer_and_region_set_tamper_blockers() -> None:
    missing = valid_reviewed_evidence()
    missing["reviewed_region_set_sha256"] = None
    mismatch = valid_reviewed_evidence()
    mismatch["reviewed_region_set_sha256"] = "wrong"

    assert "missing_reviewed_region_set_sha256" in (
        plate_evaluation.reviewed_final_trust_gate_blockers(missing)
    )
    assert "reviewed_region_set_sha256_mismatch" in (
        plate_evaluation.reviewed_final_trust_gate_blockers(mismatch)
    )


def test_reviewed_final_report_limitations_are_window_limited_not_zero_leak() -> None:
    metrics = plate_evaluation.report_payload(
        aggregate_metrics={},
        benchmark_id="day_plate",
        failure_intervals=[],
        ground_truth=plate_evaluation.load_ground_truth(PLATE_GT_PATH, policy()),
        mode=plate_evaluation.MODE_REVIEWED_FINAL,
        per_target_metrics={},
        policy=policy(),
        publishable_metrics=False,
        validation={"configuration_identity": {}, "source_identity": None},
    )

    assert metrics["limitations"] == [
        "geometric GT coverage is a pre-check, not automated pixel-level zero-leak"
    ]


def test_policy_hash_determinism_formatting_and_tamper() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    reformatted = json.loads(json.dumps(payload, indent=4))
    tampered = copy.deepcopy(payload)
    tampered["detector_matching"]["iou_min"] = 0.21
    unknown = copy.deepcopy(payload)
    unknown["schema_version"] = "plate_evaluation_policy.v9"

    assert plate_evaluation.policy_content_sha256(payload) == payload["content_sha256"]
    assert plate_evaluation.policy_content_sha256(reformatted) == payload["content_sha256"]
    assert plate_evaluation.policy_content_sha256(tampered) != payload["content_sha256"]
    assert "policy content_sha256 mismatch" in plate_evaluation.validate_policy_payload(tampered)
    assert "unknown or unsupported policy schema_version" in (
        plate_evaluation.validate_policy_payload(unknown)
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda item: item["detector_matching"].__setitem__("iou_min", 0.21),
        lambda item: item["detector_matching"].__setitem__("gt_coverage_min", 0.61),
        lambda item: item["detector_matching"].__setitem__(
            "prediction_coverage_min",
            0.13,
        ),
        lambda item: item["detector_matching"].__setitem__("area_ratio_range", [0.21, 5.0]),
        lambda item: item["detector_matching"].__setitem__("width_ratio_range", [0.34, 3.0]),
        lambda item: item["detector_matching"].__setitem__("height_ratio_range", [0.34, 3.0]),
        lambda item: item["detector_matching"][
            "center_expansion_gt_fraction_each_side"
        ].__setitem__("x", 0.51),
        lambda item: item["detector_matching"].__setitem__("min_confidence", 0.31),
        lambda item: item["automatic_coverage"].__setitem__("coverage_threshold", 0.96),
        lambda item: item["automatic_provenance"]["eligible_provenances"].append(
            "human_seeded_propagation_draft"
        ),
        lambda item: item["automatic_provenance"]["ineligible_statuses"].remove(
            "needs_review"
        ),
        lambda item: item["canonical_frame_namespace"]["benchmarks"]["day_plate"].__setitem__(
            "parent_equals_dense_plus",
            241,
        ),
        lambda item: item["canonical_frame_namespace"]["benchmarks"][
            "night_selective"
        ].__setitem__("canonical_parent_frame_interval", [421, 780]),
        lambda item: item["detector_assignment"].__setitem__(
            "pair_order",
            list(reversed(item["detector_assignment"]["pair_order"])),
        ),
        lambda item: item["formal_precision"].__setitem__("publishable", True),
    ],
)
def test_restamped_policy_semantic_changes_fail_pinned_identity(mutator) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    mutator(payload)
    restamped = restamp_policy(payload)

    errors = plate_evaluation.validate_policy_payload(restamped)

    assert restamped["content_sha256"] != plate_evaluation.EXPECTED_POLICY_SHA256
    assert "evaluation policy semantic SHA is not supported" in errors


def test_exact_policy_and_unknown_restamped_policy_id() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    unknown = copy.deepcopy(payload)
    unknown["evaluation_policy_id"] = "plate_evaluation_policy_v9.0"
    unknown = restamp_policy(unknown)

    assert plate_evaluation.validate_policy_payload(payload) == []
    assert payload["content_sha256"] == plate_evaluation.EXPECTED_POLICY_SHA256
    assert "unknown or unsupported evaluation policy id" in (
        plate_evaluation.validate_policy_payload(unknown)
    )


def report(
    *,
    mode: str,
    target_id: str,
    numerator: int,
    denominator: int,
    rate: float,
) -> dict[str, Any]:
    metric = (
        {
            "detector_scored_frames": denominator,
            "hit_frame_count": numerator,
            "maximum_consecutive_miss_run": 3,
            "miss_frame_count": denominator - numerator,
            "target_frame_recall": rate,
            "target_id": target_id,
        }
        if mode == plate_evaluation.MODE_DETECTOR
        else {
            "covered_frame_count": numerator,
            "coverage_rate": rate,
            "longest_uncovered_run": 3,
            "required_visible_frames": denominator,
            "target_id": target_id,
            "uncovered_frame_count": denominator - numerator,
        }
    )
    return {
        "configuration_identity": {"config": "same"},
        "evaluation_policy_sha256": plate_evaluation.EXPECTED_POLICY_SHA256,
        "ground_truth_content_sha256": plate_evaluation.EXPECTED_GT_CONTENT_SHA256,
        "mode": mode,
        "per_target_metrics": {target_id: metric},
        "schema_version": plate_evaluation.REPORT_SCHEMA_VERSION,
    }


def test_aggregation_day_night_micro_denominator_and_macro_mean() -> None:
    day = report(
        mode=plate_evaluation.MODE_DETECTOR,
        target_id="day",
        numerator=120,
        denominator=240,
        rate=0.5,
    )
    night = report(
        mode=plate_evaluation.MODE_DETECTOR,
        target_id="night",
        numerator=360,
        denominator=360,
        rate=1.0,
    )

    aggregate = plate_evaluation.aggregate_reports([day, night])

    assert aggregate["micro_denominator"] == 600
    assert aggregate["micro_rate"] == 0.8
    assert aggregate["macro_mean"] == 0.75


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_policy_sha256", "other_policy"),
        ("ground_truth_content_sha256", "other_gt"),
        ("mode", plate_evaluation.MODE_AUTOMATIC),
    ],
)
def test_aggregation_rejects_incompatible_reports(field: str, value: str) -> None:
    left = report(
        mode=plate_evaluation.MODE_DETECTOR,
        target_id="day",
        numerator=1,
        denominator=1,
        rate=1.0,
    )
    right = copy.deepcopy(left)
    right[field] = value

    with pytest.raises(ValueError, match=f"incompatible {field}"):
        plate_evaluation.aggregate_reports([left, right])


def test_legacy_capability_path_recognizes_plate_gt_without_publishable_metrics(
    tmp_path: Path,
) -> None:
    capability = evaluation.plate_evaluation_capability(
        ground_truth_path=PLATE_GT_PATH,
        clip_id="day_plate",
    )
    metrics = evaluation.run_plate_evaluation(
        ground_truth_path=PLATE_GT_PATH,
        clip_id="day_plate",
        output_dir=tmp_path / "legacy_plate_eval",
    )

    assert capability["gt_capability_available"] is True
    assert capability["publishable_metrics"] is False
    assert capability["blockers"] == []
    assert metrics["publishable_metrics"] is False
    assert metrics["detector"]["object_frame_recall"] is None
    assert metrics["privacy"]["privacy_coverage"] is None


def test_evaluate_plates_help_exposes_modes_and_validation_only() -> None:
    result = CliRunner().invoke(main, ["evaluate-plates", "--help"])

    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "--benchmark-id" in result.output
    assert "--validate-only" in result.output
