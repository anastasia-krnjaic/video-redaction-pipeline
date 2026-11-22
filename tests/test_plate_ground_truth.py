from __future__ import annotations

import copy
import json
import os
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
SYNTHETIC_RECIPE_PATH = (
    Path(__file__).parent / "fixtures" / "plate_annotation_synthetic_recipe.json"
)


@pytest.fixture(scope="module")
def completed_annotation_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Expand fake geometry; never present these records as historical human evidence."""
    recipe = json.loads(SYNTHETIC_RECIPE_PATH.read_text(encoding="utf-8"))
    frozen_package = load_ground_truth()["created_from"]["annotation_package"]
    buckets = iter(
        bucket["name"] for bucket in recipe["review_buckets"] for _ in range(bucket["count"])
    )
    records = []
    for benchmark in recipe["benchmarks"]:
        for dense_frame in range(benchmark["frame_count"]):
            bucket = next(buckets)
            proposal_available = bucket.startswith("proposal_")
            adjusted = bucket == "proposal_human_adjusted"
            proposal_bbox = (
                recipe["adjusted_proposal_bbox_xyxy"] if adjusted else recipe["bbox_xyxy"]
            )
            parent_frame = benchmark["parent_start"] + dense_frame
            records.append(
                {
                    "action": "redact",
                    "annotation_provenance": {
                        "annotation_status": "reviewed_box",
                        "human_adjusted_geometry": adjusted,
                        "human_reviewed": True,
                        "migration_source_package_id": (
                            "synthetic_prior_review" if bucket == "migrated_manual_review" else None
                        ),
                        "proposal_available": proposal_available,
                        "proposal_bbox": proposal_bbox if proposal_available else None,
                        "proposal_shown": proposal_available,
                        "proposal_source": "synthetic_fixture" if proposal_available else None,
                        "review_completion_bucket": bucket,
                    },
                    "benchmark_id": benchmark["benchmark_id"],
                    "dense_frame": dense_frame,
                    "frame_id": f"{benchmark['benchmark_id']}_{parent_frame:06d}",
                    "human_bbox": recipe["bbox_xyxy"],
                    "object_type": "license_plate",
                    "parent_frame": parent_frame,
                    "target_id": benchmark["target_id"],
                }
            )
    export = {
        "annotation_package_schema_version": frozen_package["schema_version"],
        "export_kind": "final_review",
        "fixture_note": recipe["fixture_note"],
        "mandatory_frames": records,
        # The v1 validator requires these frozen identities even for synthetic contract tests.
        "package_file_sha256": frozen_package["frozen_file_sha256"],
        "package_id": frozen_package["package_id"],
        "reviewer_completion_state": {
            "all_mandatory_boxes_reviewed": True,
            "final_export_ready": True,
        },
        "schema_version": "plate_annotation_review_export.v1",
        "state_revision": 1,
    }
    export["semantic_content_hash"] = annotation_package.semantic_content_hash(export)
    export_path = tmp_path_factory.mktemp("plate_gt_export") / "synthetic_annotation_export.json"
    return write_json(export_path, export)


def load_ground_truth(annotation_export: Path | None = None) -> dict[str, Any]:
    payload = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    if annotation_export is None:
        return payload
    export = json.loads(annotation_export.read_text(encoding="utf-8"))
    export_sha256 = media.sha256_file(annotation_export)
    export_hash = export["semantic_content_hash"]
    payload["audit_metadata"] = {"fixture_note": export["fixture_note"]}
    payload["created_from"]["annotation_review_export"].update(
        path=annotation_export.as_posix(), sha256=export_sha256, semantic_content_hash=export_hash
    )
    by_frame = {record["frame_id"]: record for record in export["mandatory_frames"]}
    for benchmark in payload["benchmarks"]:
        for record in benchmark["spatial_records"]:
            synthetic = by_frame[record["frame_id"]]
            provenance = synthetic["annotation_provenance"]
            record["bbox_xyxy"] = synthetic["human_bbox"]
            record["annotation_review_provenance"].update(
                {key: value for key, value in provenance.items() if key != "annotation_status"}
            )
            record["annotation_review_provenance"].update(
                annotation_review_export_sha256=export_sha256,
                annotation_review_export_semantic_content_hash=export_hash,
                proposal_reviewed_by_human=provenance["proposal_available"],
            )
            record["geometry_provenance"]["review_completion_bucket"] = provenance[
                "review_completion_bucket"
            ]
    return restamp(payload)


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


def test_published_plate_ground_truth_payload_validates() -> None:
    payload = load_ground_truth()
    assert plate_ground_truth.validate_plate_ground_truth_payload(payload) == []
    assert plate_ground_truth.record_counts(payload) == {
        "day_plate": 240,
        "night_selective": 360,
        "total": 600,
    }
    assert payload["content_sha256"] == (
        "5cf36ff48fa93743caf460ff6ee0470799e165dba2b9b42b9ad58168e5233314"
    )


def test_synthetic_plate_ground_truth_validates(
    tmp_path: Path, completed_annotation_export: Path
) -> None:
    result = validate_payload(
        tmp_path, load_ground_truth(completed_annotation_export), completed_annotation_export
    )
    assert result["status"] == "valid", result["errors"]
    assert result["annotation_export_match_count"] == 600


@pytest.mark.local_assets
@pytest.mark.skipif(
    os.environ.get("REDACTOR_RUN_LOCAL_ASSET_TESTS") != "1",
    reason="set REDACTOR_RUN_LOCAL_ASSET_TESTS=1 to validate private annotation/media lineage",
)
def test_published_plate_ground_truth_local_lineage(tmp_path: Path) -> None:
    export_path = tmp_path / "plate_annotation_review.export.json"
    annotation_package.export_plate_annotation_review(
        ANNOTATION_PACKAGE_ROOT,
        final=True,
        output_path=export_path,
        now_utc=EXPORT_TIMESTAMP,
    )
    result = plate_ground_truth.validate_plate_ground_truth_file(
        GROUND_TRUTH_PATH,
        schema_path=SCHEMA_PATH,
        annotation_export_path=export_path,
        require_local_media=True,
    )
    assert result["status"] == "valid", result["errors"]
    assert result["annotation_export_match_count"] == 600


def test_plate_ground_truth_cli_help_and_validation(
    tmp_path: Path, completed_annotation_export: Path
) -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["validate-plate-ground-truth", "--help"])
    assert help_result.exit_code == 0
    assert "--ground-truth" in help_result.output

    ground_truth_path = write_json(
        tmp_path / "synthetic_gt.json", load_ground_truth(completed_annotation_export)
    )
    result = runner.invoke(
        main,
        [
            "validate-plate-ground-truth",
            "--ground-truth",
            ground_truth_path.as_posix(),
            "--schema",
            SCHEMA_PATH.as_posix(),
            "--annotation-export",
            completed_annotation_export.as_posix(),
            "--no-local-media",
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
            lambda payload: first_day_record(payload).__setitem__("bbox_xyxy", [0, 0, 1921, 4]),
            "bbox out of bounds or non-positive",
        ),
        (
            "source_sha_tamper",
            lambda payload: payload["benchmarks"][0]["source_video"].__setitem__("sha256", "bad"),
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
    payload = load_ground_truth(completed_annotation_export)
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
    payload = load_ground_truth(completed_annotation_export)
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


def test_local_media_hash_validation_rejects_missing_and_changed_files(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_source.bin"
    source.write_bytes(b"synthetic source bytes, not video")
    missing = tmp_path / "synthetic_evaluation.bin"
    payload = {
        "benchmarks": [
            {
                "benchmark_id": "synthetic",
                "source_video": {"path": str(source), "sha256": media.sha256_file(source)},
                "evaluation_video": {"path": str(missing), "sha256": "not-present"},
            }
        ]
    }
    errors: list[str] = []
    plate_ground_truth.validate_local_media_hashes(payload, errors)
    assert errors == [f"synthetic: evaluation_video file missing: {missing}"]

    source.write_bytes(b"changed synthetic source bytes")
    errors = []
    plate_ground_truth.validate_local_media_hashes(payload, errors)
    assert "synthetic: source_video local sha256 mismatch" in errors


def test_annotation_export_one_coordinate_mismatch_fails_lineage(
    tmp_path: Path, completed_annotation_export: Path
) -> None:
    export = json.loads(completed_annotation_export.read_text(encoding="utf-8"))
    export["mandatory_frames"][0]["human_bbox"][0] += 1
    export["semantic_content_hash"] = annotation_package.semantic_content_hash(export)
    export_path = write_json(tmp_path / "tampered_export.json", export)

    payload = load_ground_truth(completed_annotation_export)
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
        require_local_media=False,
    )

    assert output_path.exists()
    assert payload["annotation_provenance"]["completion_bucket_counts"] == {
        "manual_no_proposal": 2,
        "migrated_manual_review": 268,
        "proposal_accepted_unchanged": 284,
        "proposal_human_adjusted": 46,
    }
    assert result["status"] == "valid"
