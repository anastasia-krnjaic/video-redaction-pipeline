from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from redactor import media
from redactor.plate_annotation_package import semantic_content_hash, stable_json_sha256

PLATE_GT_SCHEMA_VERSION = "plate_ground_truth.v1.0"
PLATE_GT_ID = "plate_ground_truth_v1.0"
PLATE_GT_VERSION = "1.0"
PLATE_GT_STATUS = "published_machine_readable_plate_spatial_ground_truth"
PLATE_GT_SCHEMA_PATH = Path("data/ground_truth/plate_ground_truth_schema_v1.0.json")
TEMPORAL_POLICY_DOCUMENT = Path("data/ground_truth/GroundTruth_v1.1.md")
DEFAULT_OUTPUT_PATH = Path("data/ground_truth/plate_ground_truth_v1.0.json")

EXPECTED_ANNOTATION_PACKAGE = Path(
    "artifacts/plate_ground_truth_annotation_v2_assisted/run_20260811T185326Z"
)
EXPECTED_PACKAGE_ID = "run_20260811T185326Z"
EXPECTED_PACKAGE_FILE_SHA256 = {
    "annotations.draft.json": "7f1946dd25be9584af60dff2486767a9149bc6b3d50516d622ee35a00a6b3877",
    "change_log.jsonl": "037bfea8bd0737ca189f466d1a5c848e03f475eb54b87f4be9978846cbd6645c",
    "package_identity.json": "7455fb6202bcfa9da518fd140a339ca262840be7000cf13687be02d801667b6e",
    "package_manifest.json": "f412a8df42225dd54c512a845827914537f16a8f370c8a66bf50fb8d24b2bde5",
    "progress.json": "f79b364cd43e0a96a67e1043ebffe6d17d5bc62845b1710f76a92680d779a534",
}

REVIEW_BUCKETS = {
    "manual_no_proposal",
    "migrated_manual_review",
    "proposal_accepted_unchanged",
    "proposal_human_adjusted",
}
EXPECTED_REVIEW_COUNTS = {
    "manual_no_proposal": 2,
    "migrated_manual_review": 268,
    "proposal_accepted_unchanged": 284,
    "proposal_human_adjusted": 46,
}

EXPECTED_BENCHMARKS: dict[str, dict[str, Any]] = {
    "day_plate": {
        "action": "redact",
        "dense_frame_interval": [0, 239],
        "evaluation_video": {
            "fps": "30000/1001",
            "frame_count": 570,
            "height": 1080,
            "path": "artifacts/eval_day_plate.mp4",
            "sha256": "63d41905e2b6c65a3cdc72aa73ee31668fe1c1b4fdfefa2c5688cd38c90d1097",
            "width": 1920,
        },
        "expected_record_count": 240,
        "frozen_parent_frame_interval": [240, 479],
        "object_type": "license_plate",
        "parent_to_dense_offset": -240,
        "source_interval": "00:01:41.000-00:01:49.000",
        "source_video": {
            "path": "data/raw/bodycam_sample.mp4",
            "sha256": "ab9b5f007c1ab1cb4acff90366ce4ae1c4b447ba036c00c4daab57e7fd801a57",
            "source_video_label": "bodycam_sample.mp4",
        },
        "target_id": "plate_day_01",
    },
    "night_selective": {
        "action": "redact",
        "dense_frame_interval": [0, 359],
        "evaluation_video": {
            "fps": "30000/1001",
            "frame_count": 1589,
            "height": 1080,
            "path": "artifacts/eval_night_selective.mp4",
            "sha256": "0b55aa3a497941711ac8634a049c557637e20928e8d3bd8b63e5b030b6b272e0",
            "width": 1920,
        },
        "expected_record_count": 360,
        "frozen_parent_frame_interval": [420, 779],
        "object_type": "license_plate",
        "parent_to_dense_offset": -420,
        "source_interval": "00:04:56.000-00:05:08.000",
        "source_video": {
            "path": "data/raw/night_candidate.mp4",
            "sha256": "6748c67baa06ee97858af3b7bb44780bfce142fdc5b26e90c086192d4dba2f48",
            "source_video_label": "night_candidate.mp4",
        },
        "target_id": "plate_night_01",
    },
}

TOP_LEVEL_REQUIRED = {
    "annotation_provenance",
    "benchmarks",
    "certification_scope",
    "content_sha256",
    "coordinate_system",
    "created_from",
    "ground_truth_id",
    "ground_truth_version",
    "matching_policy",
    "review_status",
    "schema_version",
    "status",
}
TOP_LEVEL_ALLOWED = TOP_LEVEL_REQUIRED | {"audit_metadata", "schema"}

BENCHMARK_REQUIRED = {
    "benchmark_id",
    "evaluation_video",
    "frame_mapping",
    "record_count",
    "source_video",
    "spatial_records",
    "target",
    "temporal_policy",
}
BENCHMARK_ALLOWED = BENCHMARK_REQUIRED

RECORD_REQUIRED = {
    "action",
    "annotation_review_provenance",
    "bbox_xyxy",
    "benchmark_id",
    "dense_frame",
    "detector_ignore",
    "detector_scored",
    "final_redaction_required",
    "frame_id",
    "geometry_provenance",
    "object_type",
    "parent_evaluation_frame",
    "target_id",
    "visibility",
}
RECORD_ALLOWED = RECORD_REQUIRED


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_content_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(payload)
    canonical.pop("content_sha256", None)
    audit_metadata = canonical.get("audit_metadata")
    if isinstance(audit_metadata, dict):
        audit_metadata.pop("generated_at_utc", None)
        if not audit_metadata:
            canonical.pop("audit_metadata", None)
    benchmarks = canonical.get("benchmarks")
    if isinstance(benchmarks, list):
        for benchmark in benchmarks:
            records = benchmark.get("spatial_records") if isinstance(benchmark, dict) else None
            if isinstance(records, list):
                benchmark["spatial_records"] = sorted(
                    records,
                    key=lambda record: (
                        str(record.get("benchmark_id", "")),
                        str(record.get("target_id", "")),
                        int(record.get("dense_frame", -1)),
                    ),
                )
        canonical["benchmarks"] = sorted(
            benchmarks, key=lambda benchmark: str(benchmark.get("benchmark_id", ""))
        )
    return canonical


def content_sha256(payload: dict[str, Any]) -> str:
    return stable_json_sha256(canonical_content_payload(payload))


def with_content_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    stamped = copy.deepcopy(payload)
    stamped["content_sha256"] = content_sha256(stamped)
    return stamped


def build_plate_ground_truth_from_export(
    annotation_export_path: Path,
    *,
    output_path: Path | None = None,
    schema_path: Path = PLATE_GT_SCHEMA_PATH,
) -> dict[str, Any]:
    annotation_export_path = annotation_export_path.resolve()
    export = read_json(annotation_export_path)
    export_errors = validate_completed_annotation_export(export)
    if export_errors:
        raise ValueError("; ".join(export_errors))

    export_sha256 = media.sha256_file(annotation_export_path)
    benchmarks = []
    bucket_counts: Counter[str] = Counter()
    for benchmark_id in sorted(EXPECTED_BENCHMARKS):
        expected = EXPECTED_BENCHMARKS[benchmark_id]
        records = []
        for record in sorted(
            (item for item in export["mandatory_frames"] if item["benchmark_id"] == benchmark_id),
            key=lambda item: item["dense_frame"],
        ):
            provenance = record["annotation_provenance"]
            bucket = provenance["review_completion_bucket"]
            bucket_counts[bucket] += 1
            proposal_available = bool(provenance.get("proposal_available"))
            records.append(
                {
                    "action": record["action"],
                    "annotation_review_provenance": {
                        "annotation_package_id": export["package_id"],
                        "annotation_review_export_semantic_content_hash": export[
                            "semantic_content_hash"
                        ],
                        "annotation_review_export_sha256": export_sha256,
                        "human_adjusted_geometry": bool(provenance.get("human_adjusted_geometry")),
                        "human_reviewed": bool(provenance.get("human_reviewed")),
                        "migration_source_package_id": provenance.get(
                            "migration_source_package_id"
                        ),
                        "proposal_available": proposal_available,
                        "proposal_bbox": provenance.get("proposal_bbox"),
                        "proposal_reviewed_by_human": proposal_available,
                        "proposal_shown": bool(provenance.get("proposal_shown")),
                        "proposal_source": provenance.get("proposal_source"),
                        "review_completion_bucket": bucket,
                        "review_status": provenance.get("annotation_status"),
                    },
                    "bbox_xyxy": record["human_bbox"],
                    "benchmark_id": benchmark_id,
                    "dense_frame": record["dense_frame"],
                    "detector_ignore": False,
                    "detector_scored": True,
                    "final_redaction_required": True,
                    "frame_id": record["frame_id"],
                    "geometry_provenance": {
                        "bbox_source_field": "mandatory_frames[].human_bbox",
                        "review_completion_bucket": bucket,
                        "source": "completed_annotation_review_export",
                    },
                    "object_type": record["object_type"],
                    "parent_evaluation_frame": record["parent_frame"],
                    "target_id": record["target_id"],
                    "visibility": "visible_localizable",
                }
            )
        benchmarks.append(
            {
                "benchmark_id": benchmark_id,
                "evaluation_video": expected["evaluation_video"],
                "frame_mapping": {
                    "dense_frame_interval": expected["dense_frame_interval"],
                    "parent_evaluation_frame_interval": expected["frozen_parent_frame_interval"],
                    "parent_to_dense_offset": expected["parent_to_dense_offset"],
                },
                "record_count": expected["expected_record_count"],
                "source_video": {
                    **expected["source_video"],
                    "certified_source_interval": expected["source_interval"],
                },
                "spatial_records": records,
                "target": {
                    "action": expected["action"],
                    "object_type": expected["object_type"],
                    "target_id": expected["target_id"],
                },
                "temporal_policy": {
                    "detector_ignore_dense_ranges": [],
                    "detector_scored_dense_ranges": [expected["dense_frame_interval"]],
                    "final_redaction_required_dense_ranges": [expected["dense_frame_interval"]],
                },
            }
        )

    payload = {
        "annotation_provenance": {
            "annotation_mode": "proposal_assisted_human_review",
            "completion_bucket_counts": dict(sorted(bucket_counts.items())),
            "human_reviewed_records": sum(bucket_counts.values()),
            "independent_annotation_audit": {
                "safe_to_publish_gt_decision": "YES",
                "verdict": (
                    "PASS -- annotations are safe to freeze for machine-readable plate Ground Truth"
                ),
            },
            "proposal_assistance_disclosure": (
                "Proposal-assisted human-reviewed plate spatial Ground Truth; "
                "proposal geometry was explicitly reviewed before acceptance."
            ),
        },
        "benchmarks": benchmarks,
        "certification_scope": {
            "certified_benchmark_ids": sorted(EXPECTED_BENCHMARKS),
            "certified_frame_count": 600,
            "certified_object_type": "license_plate",
            "close_plate_clip_exclusion": {
                "path": "artifacts/eval_day_plate_close.mp4",
                "reason": "not_part_of_certified_plate_ground_truth_v1.0",
            },
            "scope": "certified_plate_windows_only",
            "secondary_spatial_review": {
                "additional_localizable_plate": 0,
                "ambiguous_review_required": 0,
                "ignore_small": 0,
            },
        },
        "coordinate_system": {
            "bbox_format": "xyxy",
            "height": 1080,
            "origin": "top_left",
            "units": "pixels",
            "width": 1920,
        },
        "created_from": {
            "annotation_package": {
                "frozen_file_sha256": EXPECTED_PACKAGE_FILE_SHA256,
                "package_id": export["package_id"],
                "path": EXPECTED_ANNOTATION_PACKAGE.as_posix(),
                "schema_version": export["annotation_package_schema_version"],
                "state_revision": export["state_revision"],
            },
            "annotation_review_export": {
                "path": annotation_export_path.relative_to(Path.cwd()).as_posix()
                if annotation_export_path.is_relative_to(Path.cwd())
                else annotation_export_path.as_posix(),
                "schema_version": export["schema_version"],
                "semantic_content_hash": export["semantic_content_hash"],
                "sha256": export_sha256,
            },
            "source_package_lineage": export.get("source_package_lineage", {}),
            "temporal_policy_source": {
                "path": TEMPORAL_POLICY_DOCUMENT.as_posix(),
                "sha256": media.sha256_file(TEMPORAL_POLICY_DOCUMENT),
                "source": "GroundTruth_v1.1 frozen plate temporal policy",
            },
        },
        "ground_truth_id": PLATE_GT_ID,
        "ground_truth_version": PLATE_GT_VERSION,
        "matching_policy": {
            "automatic_coverage": {
                "automatic_regions_may_include": [
                    "direct_detector",
                    "ordinary_tracker",
                    "interpolation",
                ],
                "manual_corrections_receive_automatic_credit": False,
            },
            "detector_matching": {
                "direct_plate_detections_only": True,
                "gt_coverage_min": 0.60,
                "iou_min": 0.20,
                "one_prediction_to_one_gt_instance": True,
                "prediction_center": "inside_gt_box_or_half_gt_margin_expansion",
                "prediction_to_gt_area_ratio": [0.20, 5.0],
                "tracker_or_interpolation_detector_credit": False,
            },
            "policy_version": "plate_matching_policy_proposed_v1",
            "reviewed_final": {
                "automated_geometry_overlap_is_precheck_only": True,
                "human_visual_zero_leak_review_is_distinct": True,
            },
            "status": "proposed_not_scored",
        },
        "review_status": {
            "final_export_ready": True,
            "human_reviewed_records": 600,
            "needs_recheck": 0,
            "status": "complete_human_reviewed_audited",
            "temporal_gt_conflicts": 0,
            "unresolved_ambiguous_secondary": 0,
        },
        "schema": schema_path.as_posix(),
        "schema_version": PLATE_GT_SCHEMA_VERSION,
        "status": PLATE_GT_STATUS,
    }
    payload = with_content_sha256(payload)
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def validate_completed_annotation_export(export: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if export.get("schema_version") != "plate_annotation_review_export.v1":
        errors.append("annotation export schema_version mismatch")
    if export.get("export_kind") != "final_review":
        errors.append("annotation export is not final_review")
    if export.get("package_id") != EXPECTED_PACKAGE_ID:
        errors.append("annotation export package_id mismatch")
    if (
        export.get("package_file_sha256", {}).get("package_manifest.json")
        != (EXPECTED_PACKAGE_FILE_SHA256["package_manifest.json"])
    ):
        errors.append("annotation export package manifest hash mismatch")
    if export.get("semantic_content_hash") != semantic_content_hash(export):
        errors.append("annotation export semantic_content_hash mismatch")

    completion = export.get("reviewer_completion_state") or {}
    if completion.get("final_export_ready") is not True:
        errors.append("annotation export final_export_ready is not true")
    if completion.get("all_mandatory_boxes_reviewed") is not True:
        errors.append("annotation export mandatory boxes are not complete")

    records = export.get("mandatory_frames")
    if not isinstance(records, list):
        errors.append("annotation export mandatory_frames must be a list")
        return errors
    if len(records) != 600:
        errors.append(f"annotation export expected 600 records, got {len(records)}")
    counts = Counter()
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        benchmark_id = record.get("benchmark_id")
        expected = EXPECTED_BENCHMARKS.get(str(benchmark_id))
        if expected is None:
            errors.append(f"annotation export unexpected benchmark_id: {benchmark_id}")
            continue
        key = (str(benchmark_id), int(record.get("dense_frame", -1)), str(record.get("target_id")))
        if key in seen:
            errors.append(f"annotation export duplicate frame: {key}")
        seen.add(key)
        counts[str(benchmark_id)] += 1
        provenance = record.get("annotation_provenance") or {}
        bucket = provenance.get("review_completion_bucket")
        if bucket not in REVIEW_BUCKETS:
            errors.append(f"{record.get('frame_id')}: unsupported review completion bucket")
        bbox = record.get("human_bbox")
        validate_bbox(errors, str(record.get("frame_id")), bbox)
        if record.get("action") != expected["action"]:
            errors.append(f"{record.get('frame_id')}: action mismatch")
        if record.get("object_type") != expected["object_type"]:
            errors.append(f"{record.get('frame_id')}: object_type mismatch")
        if record.get("target_id") != expected["target_id"]:
            errors.append(f"{record.get('frame_id')}: target_id mismatch")
        dense = record.get("dense_frame")
        parent = record.get("parent_frame")
        if not isinstance(dense, int) or not isinstance(parent, int):
            errors.append(f"{record.get('frame_id')}: frame mapping must be integer")
        elif parent + expected["parent_to_dense_offset"] != dense:
            errors.append(f"{record.get('frame_id')}: parent/dense frame mismatch")
    for benchmark_id, expected in EXPECTED_BENCHMARKS.items():
        if counts[benchmark_id] != expected["expected_record_count"]:
            errors.append(
                f"annotation export {benchmark_id} expected "
                f"{expected['expected_record_count']} records, got {counts[benchmark_id]}"
            )
    return errors


def validate_plate_ground_truth_file(
    ground_truth_path: Path,
    *,
    schema_path: Path | None = None,
    annotation_export_path: Path | None = None,
    require_local_media: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    ground_truth = read_json(ground_truth_path)
    schema_path = schema_path or Path(str(ground_truth.get("schema") or PLATE_GT_SCHEMA_PATH))
    if schema_path.exists():
        schema = read_json(schema_path)
        if schema.get("schema_id") != "plate_ground_truth_schema_v1.0":
            errors.append("plate GT schema_id mismatch")
        if schema.get("schema_version") != "1.0":
            errors.append("plate GT schema_version mismatch")
    else:
        errors.append(f"plate GT schema file missing: {schema_path}")

    errors.extend(validate_plate_ground_truth_payload(ground_truth))
    export_match_count = compare_with_annotation_export(
        ground_truth,
        annotation_export_path=annotation_export_path,
        errors=errors,
    )
    if require_local_media:
        validate_local_media_hashes(ground_truth, errors)
    counts = record_counts(ground_truth)
    return {
        "annotation_export_match_count": export_match_count,
        "content_sha256": ground_truth.get("content_sha256"),
        "error_count": len(errors),
        "errors": errors,
        "ground_truth_id": ground_truth.get("ground_truth_id"),
        "record_counts": counts,
        "schema_version": ground_truth.get("schema_version"),
        "status": "valid" if not errors else "invalid",
    }


def validate_plate_ground_truth_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(TOP_LEVEL_REQUIRED - set(payload))
    unknown = sorted(set(payload) - TOP_LEVEL_ALLOWED)
    errors.extend(f"missing top-level field: {field}" for field in missing)
    errors.extend(f"unknown top-level field: {field}" for field in unknown)
    if payload.get("schema_version") != PLATE_GT_SCHEMA_VERSION:
        errors.append("unsupported plate ground truth schema_version")
    if payload.get("ground_truth_id") != PLATE_GT_ID:
        errors.append("ground_truth_id mismatch")
    if payload.get("ground_truth_version") != PLATE_GT_VERSION:
        errors.append("ground_truth_version mismatch")
    if payload.get("status") != PLATE_GT_STATUS:
        errors.append("status mismatch")
    if payload.get("content_sha256") != content_sha256(payload):
        errors.append("content_sha256 mismatch")
    if payload.get("coordinate_system") != {
        "bbox_format": "xyxy",
        "height": 1080,
        "origin": "top_left",
        "units": "pixels",
        "width": 1920,
    }:
        errors.append("coordinate_system mismatch")
    errors.extend(validate_created_from(payload.get("created_from")))
    errors.extend(validate_certification_scope(payload.get("certification_scope")))
    errors.extend(validate_annotation_provenance(payload.get("annotation_provenance")))
    errors.extend(validate_matching_policy(payload.get("matching_policy")))
    errors.extend(validate_benchmarks(payload.get("benchmarks")))
    return errors


def validate_created_from(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["created_from must be an object"]
    package = value.get("annotation_package") or {}
    if package.get("package_id") != EXPECTED_PACKAGE_ID:
        errors.append("created_from annotation package id mismatch")
    frozen_hashes = package.get("frozen_file_sha256") or {}
    for name, expected in EXPECTED_PACKAGE_FILE_SHA256.items():
        if frozen_hashes.get(name) != expected:
            errors.append(f"created_from frozen package hash mismatch: {name}")
    temporal = value.get("temporal_policy_source") or {}
    if temporal.get("path") != TEMPORAL_POLICY_DOCUMENT.as_posix():
        errors.append("temporal policy source path mismatch")
    elif Path(temporal["path"]).exists() and media.sha256_file(
        Path(temporal["path"])
    ) != temporal.get("sha256"):
        errors.append("temporal policy source hash mismatch")
    export = value.get("annotation_review_export") or {}
    if not export.get("sha256"):
        errors.append("annotation review export sha256 missing")
    if not export.get("semantic_content_hash"):
        errors.append("annotation review export semantic content hash missing")
    return errors


def validate_certification_scope(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["certification_scope must be an object"]
    errors: list[str] = []
    if value.get("certified_benchmark_ids") != sorted(EXPECTED_BENCHMARKS):
        errors.append("certified benchmark ids mismatch")
    if value.get("certified_frame_count") != 600:
        errors.append("certified frame count mismatch")
    close = value.get("close_plate_clip_exclusion") or {}
    if close.get("path") != "artifacts/eval_day_plate_close.mp4":
        errors.append("close-plate exclusion path mismatch")
    secondary = value.get("secondary_spatial_review") or {}
    if secondary != {
        "additional_localizable_plate": 0,
        "ambiguous_review_required": 0,
        "ignore_small": 0,
    }:
        errors.append("secondary spatial review counts mismatch")
    return errors


def validate_annotation_provenance(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["annotation_provenance must be an object"]
    errors: list[str] = []
    if value.get("annotation_mode") != "proposal_assisted_human_review":
        errors.append("annotation mode mismatch")
    if value.get("completion_bucket_counts") != EXPECTED_REVIEW_COUNTS:
        errors.append("completion bucket counts mismatch")
    if value.get("human_reviewed_records") != 600:
        errors.append("human reviewed record count mismatch")
    audit = value.get("independent_annotation_audit") or {}
    if audit.get("safe_to_publish_gt_decision") != "YES":
        errors.append("independent annotation audit decision mismatch")
    return errors


def validate_matching_policy(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["matching_policy must be an object"]
    errors: list[str] = []
    if value.get("status") != "proposed_not_scored":
        errors.append("matching policy status mismatch")
    detector = value.get("detector_matching") or {}
    if detector.get("direct_plate_detections_only") is not True:
        errors.append("detector matching direct-only flag mismatch")
    if detector.get("tracker_or_interpolation_detector_credit") is not False:
        errors.append("detector matching tracker credit flag mismatch")
    return errors


def validate_benchmarks(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list):
        return ["benchmarks must be a list"]
    if {item.get("benchmark_id") for item in value if isinstance(item, dict)} != set(
        EXPECTED_BENCHMARKS
    ):
        errors.append("benchmark id set mismatch")
    total_records = 0
    bucket_counts: Counter[str] = Counter()
    for benchmark in value:
        if not isinstance(benchmark, dict):
            errors.append("benchmark must be an object")
            continue
        benchmark_id = str(benchmark.get("benchmark_id") or "")
        expected = EXPECTED_BENCHMARKS.get(benchmark_id)
        if expected is None:
            errors.append(f"unexpected benchmark_id: {benchmark_id}")
            continue
        missing = sorted(BENCHMARK_REQUIRED - set(benchmark))
        unknown = sorted(set(benchmark) - BENCHMARK_ALLOWED)
        errors.extend(f"{benchmark_id}: missing benchmark field: {field}" for field in missing)
        errors.extend(f"{benchmark_id}: unknown benchmark field: {field}" for field in unknown)
        errors.extend(validate_benchmark_identity(benchmark, expected))
        records = benchmark.get("spatial_records")
        if not isinstance(records, list):
            errors.append(f"{benchmark_id}: spatial_records must be a list")
            continue
        total_records += len(records)
        errors.extend(validate_spatial_records(benchmark_id, expected, records, bucket_counts))
    if total_records != 600:
        errors.append(f"expected 600 total spatial records, got {total_records}")
    if dict(sorted(bucket_counts.items())) != EXPECTED_REVIEW_COUNTS:
        errors.append("spatial record provenance bucket counts mismatch")
    return errors


def validate_benchmark_identity(benchmark: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    benchmark_id = str(benchmark.get("benchmark_id") or "")
    if benchmark.get("record_count") != expected["expected_record_count"]:
        errors.append(f"{benchmark_id}: record_count mismatch")
    if benchmark.get("evaluation_video") != expected["evaluation_video"]:
        errors.append(f"{benchmark_id}: evaluation video identity mismatch")
    expected_source = {
        **expected["source_video"],
        "certified_source_interval": expected["source_interval"],
    }
    if benchmark.get("source_video") != expected_source:
        errors.append(f"{benchmark_id}: source video identity mismatch")
    if benchmark.get("frame_mapping") != {
        "dense_frame_interval": expected["dense_frame_interval"],
        "parent_evaluation_frame_interval": expected["frozen_parent_frame_interval"],
        "parent_to_dense_offset": expected["parent_to_dense_offset"],
    }:
        errors.append(f"{benchmark_id}: frame mapping mismatch")
    if benchmark.get("target") != {
        "action": expected["action"],
        "object_type": expected["object_type"],
        "target_id": expected["target_id"],
    }:
        errors.append(f"{benchmark_id}: target identity mismatch")
    if benchmark.get("temporal_policy") != {
        "detector_ignore_dense_ranges": [],
        "detector_scored_dense_ranges": [expected["dense_frame_interval"]],
        "final_redaction_required_dense_ranges": [expected["dense_frame_interval"]],
    }:
        errors.append(f"{benchmark_id}: temporal policy mismatch")
    return errors


def validate_spatial_records(
    benchmark_id: str,
    expected: dict[str, Any],
    records: list[dict[str, Any]],
    bucket_counts: Counter[str],
) -> list[str]:
    errors: list[str] = []
    if len(records) != expected["expected_record_count"]:
        errors.append(
            f"{benchmark_id}: expected {expected['expected_record_count']} records, "
            f"got {len(records)}"
        )
    seen_dense: set[int] = set()
    seen_parent: set[int] = set()
    expected_dense = list(
        range(
            expected["dense_frame_interval"][0],
            expected["dense_frame_interval"][1] + 1,
        )
    )
    expected_parent = list(
        range(
            expected["frozen_parent_frame_interval"][0],
            expected["frozen_parent_frame_interval"][1] + 1,
        )
    )
    for record in records:
        missing = sorted(RECORD_REQUIRED - set(record))
        unknown = sorted(set(record) - RECORD_ALLOWED)
        context = str(record.get("frame_id") or f"{benchmark_id}:unknown")
        errors.extend(f"{context}: missing record field: {field}" for field in missing)
        errors.extend(f"{context}: unknown record field: {field}" for field in unknown)
        if record.get("benchmark_id") != benchmark_id:
            errors.append(f"{context}: benchmark_id mismatch")
        if record.get("target_id") != expected["target_id"]:
            errors.append(f"{context}: target_id mismatch")
        if record.get("action") != expected["action"]:
            errors.append(f"{context}: action mismatch")
        if record.get("object_type") != expected["object_type"]:
            errors.append(f"{context}: object_type mismatch")
        if record.get("visibility") != "visible_localizable":
            errors.append(f"{context}: visibility mismatch")
        if record.get("detector_scored") is not True:
            errors.append(f"{context}: detector_scored must be true")
        if record.get("detector_ignore") is not False:
            errors.append(f"{context}: detector_ignore must be false")
        if record.get("final_redaction_required") is not True:
            errors.append(f"{context}: final_redaction_required must be true")
        dense = record.get("dense_frame")
        parent = record.get("parent_evaluation_frame")
        if not isinstance(dense, int) or not isinstance(parent, int):
            errors.append(f"{context}: dense/parent frames must be integers")
            continue
        if dense in seen_dense:
            errors.append(f"{benchmark_id}: duplicate dense frame {dense}")
        if parent in seen_parent:
            errors.append(f"{benchmark_id}: duplicate parent frame {parent}")
        seen_dense.add(dense)
        seen_parent.add(parent)
        if parent + expected["parent_to_dense_offset"] != dense:
            errors.append(f"{context}: parent/dense mapping mismatch")
        validate_bbox(errors, context, record.get("bbox_xyxy"))
        provenance = record.get("annotation_review_provenance") or {}
        bucket = provenance.get("review_completion_bucket")
        if bucket not in REVIEW_BUCKETS:
            errors.append(f"{context}: review completion bucket mismatch")
        else:
            bucket_counts[str(bucket)] += 1
        errors.extend(validate_record_provenance(context, provenance, record))
    missing_dense = sorted(set(expected_dense) - seen_dense)
    extra_dense = sorted(seen_dense - set(expected_dense))
    missing_parent = sorted(set(expected_parent) - seen_parent)
    extra_parent = sorted(seen_parent - set(expected_parent))
    if missing_dense:
        errors.append(f"{benchmark_id}: missing dense frames: {missing_dense[:5]}")
    if extra_dense:
        errors.append(f"{benchmark_id}: unexpected dense frames: {extra_dense[:5]}")
    if missing_parent:
        errors.append(f"{benchmark_id}: missing parent frames: {missing_parent[:5]}")
    if extra_parent:
        errors.append(f"{benchmark_id}: unexpected parent frames: {extra_parent[:5]}")
    return errors


def validate_record_provenance(
    context: str, provenance: dict[str, Any], record: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if provenance.get("annotation_package_id") != EXPECTED_PACKAGE_ID:
        errors.append(f"{context}: annotation package provenance mismatch")
    if provenance.get("human_reviewed") is not True:
        errors.append(f"{context}: human_reviewed provenance must be true")
    if provenance.get("review_status") != "reviewed_box":
        errors.append(f"{context}: review_status provenance mismatch")
    bucket = provenance.get("review_completion_bucket")
    proposal_available = provenance.get("proposal_available")
    proposal_bbox = provenance.get("proposal_bbox")
    if bucket == "migrated_manual_review":
        if not provenance.get("migration_source_package_id"):
            errors.append(f"{context}: migrated record lacks migration source")
        if proposal_available:
            errors.append(f"{context}: migrated record cannot have proposal_available")
    elif bucket == "manual_no_proposal":
        if proposal_available or proposal_bbox is not None:
            errors.append(f"{context}: manual/no-proposal record has proposal geometry")
    elif bucket == "proposal_accepted_unchanged":
        if proposal_available is not True or proposal_bbox != record.get("bbox_xyxy"):
            errors.append(f"{context}: accepted proposal provenance mismatch")
    elif bucket == "proposal_human_adjusted":
        if proposal_available is not True:
            errors.append(f"{context}: adjusted proposal lacks proposal availability")
        if provenance.get("human_adjusted_geometry") is not True:
            errors.append(f"{context}: adjusted proposal lacks human_adjusted_geometry")
        if proposal_bbox == record.get("bbox_xyxy"):
            errors.append(f"{context}: adjusted proposal bbox did not change")
    return errors


def validate_bbox(errors: list[str], context: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        errors.append(f"{context}: bbox must be [x1, y1, x2, y2]")
        return
    if any(not isinstance(coord, int) or isinstance(coord, bool) for coord in value):
        errors.append(f"{context}: bbox coordinates must be integers")
        return
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
        errors.append(f"{context}: bbox out of bounds or non-positive")


def compare_with_annotation_export(
    ground_truth: dict[str, Any],
    *,
    annotation_export_path: Path | None,
    errors: list[str],
) -> int:
    created_from = ground_truth.get("created_from") or {}
    export_ref = created_from.get("annotation_review_export") or {}
    path = annotation_export_path or Path(str(export_ref.get("path") or ""))
    if not path:
        errors.append("annotation export path missing")
        return 0
    if not path.exists():
        errors.append(f"annotation export missing: {path}")
        return 0
    actual_export_sha = media.sha256_file(path)
    if actual_export_sha != export_ref.get("sha256"):
        errors.append("annotation export sha256 mismatch")
    export = read_json(path)
    if export.get("semantic_content_hash") != export_ref.get("semantic_content_hash"):
        errors.append("annotation export semantic hash reference mismatch")
    export_errors = validate_completed_annotation_export(export)
    errors.extend(f"annotation export: {error}" for error in export_errors)
    export_index = {
        (
            record.get("benchmark_id"),
            record.get("target_id"),
            record.get("dense_frame"),
        ): record
        for record in export.get("mandatory_frames", [])
    }
    matches = 0
    for record in iter_spatial_records(ground_truth):
        key = (record.get("benchmark_id"), record.get("target_id"), record.get("dense_frame"))
        exported = export_index.get(key)
        context = str(record.get("frame_id") or key)
        if exported is None:
            errors.append(f"{context}: missing matching annotation export record")
            continue
        if exported.get("human_bbox") != record.get("bbox_xyxy"):
            errors.append(f"{context}: bbox differs from annotation export")
            continue
        if exported.get("parent_frame") != record.get("parent_evaluation_frame"):
            errors.append(f"{context}: parent frame differs from annotation export")
            continue
        if exported.get("frame_id") != record.get("frame_id"):
            errors.append(f"{context}: frame_id differs from annotation export")
            continue
        matches += 1
    if len(export_index) != 600:
        errors.append(f"annotation export expected 600 records, got {len(export_index)}")
    if matches != 600:
        errors.append(f"annotation export exact bbox matches {matches} / 600")
    return matches


def validate_local_media_hashes(ground_truth: dict[str, Any], errors: list[str]) -> None:
    for benchmark in ground_truth.get("benchmarks", []):
        if not isinstance(benchmark, dict):
            continue
        benchmark_id = str(benchmark.get("benchmark_id") or "")
        for field in ("evaluation_video", "source_video"):
            identity = benchmark.get(field) or {}
            path = Path(str(identity.get("path") or ""))
            if not path.exists():
                errors.append(f"{benchmark_id}: {field} file missing: {path}")
                continue
            if media.sha256_file(path) != identity.get("sha256"):
                errors.append(f"{benchmark_id}: {field} local sha256 mismatch")


def iter_spatial_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        return records
    for benchmark in benchmarks:
        if isinstance(benchmark, dict) and isinstance(benchmark.get("spatial_records"), list):
            records.extend(benchmark["spatial_records"])
    return records


def record_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in iter_spatial_records(payload):
        counts[str(record.get("benchmark_id"))] += 1
    counts["total"] = sum(counts.values())
    return dict(counts)
