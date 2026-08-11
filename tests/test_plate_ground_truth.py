from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from redactor import evaluation, media, plate_ground_truth
from redactor import plate_annotation_package as annotation_package
from redactor.cli import main

GROUND_TRUTH_PATH = Path("data/ground_truth/plate_ground_truth_v1.0.json")
SCHEMA_PATH = Path("data/ground_truth/plate_ground_truth_schema_v1.0.json")
ANNOTATION_PACKAGE_ROOT = Path(
    "artifacts/plate_ground_truth_annotation_v2_assisted/run_20260811T185326Z"
)
EXPORT_TIMESTAMP = "2026-08-11T20:00:00Z"


@pytest.fixture(scope="module")
def completed_annotation_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    export_path = tmp_path_factory.mktemp("plate_gt_export") / "plate_annotation_review.export.json"
    annotation_package.export_plate_annotation_review(
        ANNOTATION_PACKAGE_ROOT,
        final=True,
        output_path=export_path,
        now_utc=EXPORT_TIMESTAMP,
    )
    return export_path


def load_ground_truth() -> dict[str, Any]:
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def restamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["content_sha256"] = plate_ground_truth.content_sha256(payload)
    return payload


def validate_payload(
    tmp_path: Path,
    payload: dict[str, Any],
    completed_annotation_export: Path,
    *,
    require_local_media: bool = False,
) -> dict[str, Any]:
    path = write_json(tmp_path / "plate_gt.json", payload)
    return plate_ground_truth.validate_plate_ground_truth_file(
        path,
        schema_path=SCHEMA_PATH,
        annotation_export_path=completed_annotation_export,
        require_local_media=require_local_media,
    )


def first_day_record(payload: dict[str, Any]) -> dict[str, Any]:
    day = next(item for item in payload["benchmarks"] if item["benchmark_id"] == "day_plate")
    return day["spatial_records"][0]


def test_published_plate_ground_truth_validates(
    completed_annotation_export: Path,
) -> None:
    result = plate_ground_truth.validate_plate_ground_truth_file(
        GROUND_TRUTH_PATH,
        schema_path=SCHEMA_PATH,
        annotation_export_path=completed_annotation_export,
        require_local_media=True,
    )

    assert result["status"] == "valid"
    assert result["record_counts"] == {
        "day_plate": 240,
        "night_selective": 360,
        "total": 600,
    }
    assert result["annotation_export_match_count"] == 600
    assert result["content_sha256"] == (
        "5cf36ff48fa93743caf460ff6ee0470799e165dba2b9b42b9ad58168e5233314"
    )


def test_plate_ground_truth_cli_help_and_validation(completed_annotation_export: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["validate-plate-ground-truth", "--help"])
    assert help_result.exit_code == 0
    assert "--ground-truth" in help_result.output

    result = runner.invoke(
        main,
        [
            "validate-plate-ground-truth",
            "--ground-truth",
            GROUND_TRUTH_PATH.as_posix(),
            "--schema",
            SCHEMA_PATH.as_posix(),
            "--annotation-export",
            completed_annotation_export.as_posix(),
        ],
    )

    assert result.exit_code == 0
    assert "valid (0 errors)" in result.output
    assert "Annotation export bbox matches: 600 / 600" in result.output


@pytest.mark.parametrize(
    ("case_name", "mutator", "expected_error"),
    [
        (
            "unknown_schema",
            lambda payload: payload.__setitem__("schema_version", "unknown"),
            "unsupported plate ground truth schema_version",
        ),
        (
            "missing_required_field",
            lambda payload: payload.pop("ground_truth_id"),
            "missing top-level field: ground_truth_id",
        ),
        (
            "missing_frame",
            lambda payload: payload["benchmarks"][0]["spatial_records"].pop(0),
            "day_plate: expected 240 records, got 239",
        ),
        (
            "duplicate_frame",
            lambda payload: payload["benchmarks"][0]["spatial_records"].append(
                copy.deepcopy(payload["benchmarks"][0]["spatial_records"][0])
            ),
            "day_plate: duplicate dense frame 0",
        ),
        (
            "extra_frame",
            lambda payload: _append_extra_day_record(payload),
            "day_plate: unexpected dense frames: [999]",
        ),
        (
            "wrong_target",
            lambda payload: first_day_record(payload).__setitem__("target_id", "wrong_target"),
            "target_id mismatch",
        ),
        (
            "wrong_benchmark",
            lambda payload: first_day_record(payload).__setitem__(
                "benchmark_id", "wrong_benchmark"
            ),
            "benchmark_id mismatch",
        ),
        (
            "wrong_action",
            lambda payload: first_day_record(payload).__setitem__("action", "keep"),
            "action mismatch",
        ),
        (
            "wrong_class",
            lambda payload: first_day_record(payload).__setitem__("object_type", "face"),
            "object_type mismatch",
        ),
        (
            "parent_dense_mismatch",
            lambda payload: first_day_record(payload).__setitem__("parent_evaluation_frame", 241),
            "parent/dense mapping mismatch",
        ),
        (
            "range_mismatch",
            lambda payload: payload["benchmarks"][0]["frame_mapping"].__setitem__(
                "dense_frame_interval", [0, 238]
            ),
            "day_plate: frame mapping mismatch",
        ),
        (
            "invalid_bbox",
            lambda payload: first_day_record(payload).__setitem__("bbox_xyxy", [1, 1, 1, 4]),
            "bbox out of bounds or non-positive",
        ),
        (
            "out_of_bounds_bbox",
            lambda payload: first_day_record(payload).__setitem__(
                "bbox_xyxy", [0, 0, 1921, 4]
            ),
            "bbox out of bounds or non-positive",
        ),
        (
            "source_sha_tamper",
            lambda payload: payload["benchmarks"][0]["source_video"].__setitem__(
                "sha256", "bad"
            ),
            "day_plate: source video identity mismatch",
        ),
        (
            "evaluation_sha_tamper",
            lambda payload: payload["benchmarks"][0]["evaluation_video"].__setitem__(
                "sha256", "bad"
            ),
            "day_plate: evaluation video identity mismatch",
        ),
        (
            "annotation_export_hash_tamper",
            lambda payload: payload["created_from"]["annotation_review_export"].__setitem__(
                "sha256", "bad"
            ),
            "annotation export sha256 mismatch",
        ),
    ],
)
def test_plate_ground_truth_validator_rejects_tampering(
    tmp_path: Path,
    completed_annotation_export: Path,
    case_name: str,
    mutator: Any,
    expected_error: str,
) -> None:
    payload = load_ground_truth()
    mutator(payload)
    restamp(payload)

    result = validate_payload(tmp_path / case_name, payload, completed_annotation_export)

    assert result["status"] == "invalid"
    assert any(expected_error in error for error in result["errors"])


def _append_extra_day_record(payload: dict[str, Any]) -> None:
    extra = copy.deepcopy(payload["benchmarks"][0]["spatial_records"][0])
    extra["frame_id"] = "day_plate_000999"
    extra["dense_frame"] = 999
    extra["parent_evaluation_frame"] = 1239
    payload["benchmarks"][0]["spatial_records"].append(extra)


def test_plate_ground_truth_content_hash_tamper_fails(
    tmp_path: Path, completed_annotation_export: Path
) -> None:
    payload = load_ground_truth()
    payload["content_sha256"] = "bad"

    result = validate_payload(tmp_path, payload, completed_annotation_export)

    assert result["status"] == "invalid"
    assert "content_sha256 mismatch" in result["errors"]


def test_plate_ground_truth_semantic_hash_behavior() -> None:
    payload = load_ground_truth()
    original = plate_ground_truth.content_sha256(payload)

    reloaded = json.loads(json.dumps(payload, sort_keys=False))
    assert plate_ground_truth.content_sha256(reloaded) == original

    reordered = copy.deepcopy(payload)
    reordered["benchmarks"].reverse()
    for benchmark in reordered["benchmarks"]:
        benchmark["spatial_records"].reverse()
    assert plate_ground_truth.content_sha256(reordered) == original

    changed_bbox = copy.deepcopy(payload)
    first_day_record(changed_bbox)["bbox_xyxy"][0] += 1
    assert plate_ground_truth.content_sha256(changed_bbox) != original

    changed_mapping = copy.deepcopy(payload)
    first_day_record(changed_mapping)["parent_evaluation_frame"] += 1
    assert plate_ground_truth.content_sha256(changed_mapping) != original

    audit_timestamp_only = copy.deepcopy(payload)
    audit_timestamp_only["audit_metadata"] = {"generated_at_utc": "2030-01-01T00:00:00Z"}
    assert plate_ground_truth.content_sha256(audit_timestamp_only) == original


def test_annotation_export_one_coordinate_mismatch_fails_lineage(
    tmp_path: Path, completed_annotation_export: Path
) -> None:
    export = json.loads(completed_annotation_export.read_text(encoding="utf-8"))
    export["mandatory_frames"][0]["human_bbox"][0] += 1
    export["semantic_content_hash"] = annotation_package.semantic_content_hash(export)
    export_path = write_json(tmp_path / "tampered_export.json", export)

    payload = load_ground_truth()
    payload["created_from"]["annotation_review_export"]["sha256"] = media.sha256_file(export_path)
    payload["created_from"]["annotation_review_export"]["semantic_content_hash"] = export[
        "semantic_content_hash"
    ]
    restamp(payload)

    result = validate_payload(tmp_path, payload, export_path)

    assert result["status"] == "invalid"
    assert any("bbox differs from annotation export" in error for error in result["errors"])


def test_plate_evaluation_remains_capability_only_for_plate_gt(tmp_path: Path) -> None:
    metrics = evaluation.run_plate_evaluation(
        ground_truth_path=GROUND_TRUTH_PATH,
        clip_id="day_plate",
        output_dir=tmp_path / "plate_eval",
    )

    assert metrics["publishable_metrics"] is False
    assert metrics["detector"]["object_frame_recall"] is None
    assert metrics["privacy"]["privacy_coverage"] is None
    assert metrics["acceptance"]["final_status"] == evaluation.STATUS_NOT_SCORABLE


def test_build_plate_ground_truth_from_completed_export(
    tmp_path: Path, completed_annotation_export: Path
) -> None:
    output_path = tmp_path / "plate_gt.json"

    payload = plate_ground_truth.build_plate_ground_truth_from_export(
        completed_annotation_export,
        output_path=output_path,
        schema_path=SCHEMA_PATH,
    )
    result = plate_ground_truth.validate_plate_ground_truth_file(
        output_path,
        schema_path=SCHEMA_PATH,
        annotation_export_path=completed_annotation_export,
        require_local_media=True,
    )

    assert output_path.exists()
    assert payload["annotation_provenance"]["completion_bucket_counts"] == {
        "manual_no_proposal": 2,
        "migrated_manual_review": 268,
        "proposal_accepted_unchanged": 284,
        "proposal_human_adjusted": 46,
    }
    assert result["status"] == "valid"
