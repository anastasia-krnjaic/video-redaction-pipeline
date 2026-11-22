from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redactor import (
    media,
    plate_detection,
    plate_ground_truth,
    plate_rendering,
    plate_tracking,
    rendering,
    review_finalization,
    reviewed_regions,
)

POLICY_SCHEMA_VERSION = "plate_evaluation_policy.v1.0"
POLICY_ID = "plate_evaluation_policy_v1.0"
REPORT_SCHEMA_VERSION = "plate_evaluation_report.v1.0"
DEFAULT_POLICY_PATH = Path("data/evaluations/plate_evaluation_policy_v1.0.json")
DEFAULT_DETECTOR_CONFIG_PATH = Path(
    "data/evaluations/license_plate_detector_oim_yolov9_s_608_v1/selected_configuration.json"
)
EXPECTED_GT_CONTENT_SHA256 = "5cf36ff48fa93743caf460ff6ee0470799e165dba2b9b42b9ad58168e5233314"
SUPPORTED_PLATE_EVALUATION_POLICIES = {
    POLICY_ID: "34515ecb8d32bffea5bb69093ef0c653ee12228abb3864182943eb4e394645ac",
}
EXPECTED_POLICY_SHA256 = SUPPORTED_PLATE_EVALUATION_POLICIES[POLICY_ID]
EXPECTED_DETECTOR_CONFIGURATION_HASH = (
    "7d914438255905e2a249e5bbd30fd6e70eace1e91b57e205bd572d004ffb5a55"
)

MODE_CAPABILITY = "capability"
MODE_DETECTOR = "detector"
MODE_AUTOMATIC = "automatic"
MODE_REVIEWED_FINAL = "reviewed_final"
SCORING_MODES = {MODE_DETECTOR, MODE_AUTOMATIC, MODE_REVIEWED_FINAL}

CATEGORY = "license_plate"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT
DIRECT_DETECTOR_PROVENANCES = {"", "direct_detector", "detector"}
REVIEWED_PROVENANCE_MARKERS = ("manual", "reviewed", "preserved", "correction")
DUPLICATE_REGION_IOU_THRESHOLD = 0.95

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class GroundTruthRecord:
    benchmark_id: str
    target_id: str
    dense_frame: int
    parent_frame: int
    bbox_xyxy: Box


@dataclass(frozen=True)
class DetectorPrediction:
    frame_index: int
    bbox_xyxy: Box | None
    score: float
    category: str
    detector_id: str
    configuration_hash: str
    source_order: int
    candidate_id: str | None = None
    provenance: str = ""
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class AutomaticRegion:
    frame_index: int
    bbox_xyxy: Box | None
    category: str
    provenance: str
    status: str
    source_order: int
    track_uid: str | None = None
    object_id: str | None = None
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class PairDiagnostics:
    intersection_area: float
    iou: float
    gt_coverage: float
    prediction_coverage: float
    area_ratio: float
    width_ratio: float
    height_ratio: float
    normalized_center_distance: float
    scale_error: float
    prediction_center_inside_expanded_gt: bool
    matches: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_ratio": round(self.area_ratio, 6),
            "gt_coverage": round(self.gt_coverage, 6),
            "height_ratio": round(self.height_ratio, 6),
            "intersection_area": round(self.intersection_area, 6),
            "iou": round(self.iou, 6),
            "matches": self.matches,
            "normalized_center_distance": round(self.normalized_center_distance, 6),
            "prediction_center_inside_expanded_gt": self.prediction_center_inside_expanded_gt,
            "prediction_coverage": round(self.prediction_coverage, 6),
            "scale_error": round(self.scale_error, 6),
            "width_ratio": round(self.width_ratio, 6),
        }


def stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def policy_content_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "content_sha256"}


def policy_content_sha256(payload: dict[str, Any]) -> str:
    return stable_json_sha256(policy_content_payload(payload))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_policy_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policy_id = str(payload.get("evaluation_policy_id") or "")
    semantic_sha = policy_content_sha256(payload)
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
        errors.append("unknown or unsupported policy schema_version")
    pinned_sha = SUPPORTED_PLATE_EVALUATION_POLICIES.get(policy_id)
    if pinned_sha is None:
        errors.append("unknown or unsupported evaluation policy id")
    elif semantic_sha != pinned_sha:
        errors.append("evaluation policy semantic SHA is not supported")
    if payload.get("category") != CATEGORY:
        errors.append("policy category mismatch")
    if payload.get("content_sha256") != semantic_sha:
        errors.append("policy content_sha256 mismatch")
    compatible_gt = dict(payload.get("compatible_ground_truth") or {})
    if compatible_gt.get("ground_truth_id") != plate_ground_truth.PLATE_GT_ID:
        errors.append("unsupported GT id")
    if compatible_gt.get("schema_version") != plate_ground_truth.PLATE_GT_SCHEMA_VERSION:
        errors.append("unsupported GT version")
    if compatible_gt.get("content_sha256") != EXPECTED_GT_CONTENT_SHA256:
        errors.append("unsupported GT content SHA")
    detector = dict(payload.get("detector_matching") or {})
    required_thresholds = (
        "iou_min",
        "gt_coverage_min",
        "prediction_coverage_min",
        "area_ratio_range",
        "width_ratio_range",
        "height_ratio_range",
        "min_confidence",
    )
    for field in required_thresholds:
        if field not in detector:
            errors.append(f"missing detector threshold: {field}")
    for field in ("area_ratio_range", "width_ratio_range", "height_ratio_range"):
        if field in detector and not valid_numeric_range(detector[field]):
            errors.append(f"malformed detector range: {field}")
    automatic = dict(payload.get("automatic_coverage") or {})
    if automatic.get("coverage_threshold") != 0.95:
        errors.append("automatic coverage threshold mismatch")
    reviewed = dict(payload.get("reviewed_final") or {})
    if reviewed.get("coverage_threshold") != 0.95:
        errors.append("reviewed-final coverage threshold mismatch")
    binding = dict(payload.get("detector_configuration_binding") or {})
    if binding.get("configuration_hash") != EXPECTED_DETECTOR_CONFIGURATION_HASH:
        errors.append("detector configuration binding hash mismatch")
    if binding.get("detector_id") != plate_detection.DETECTOR_ID:
        errors.append("detector configuration binding detector id mismatch")
    return errors


def valid_numeric_range(value: Any) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    try:
        left = float(value[0])
        right = float(value[1])
    except (TypeError, ValueError):
        return False
    return left <= right


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    payload = read_json(path)
    errors = validate_policy_payload(payload)
    if errors:
        raise ValueError("invalid plate evaluation policy: " + "; ".join(errors))
    return payload


def validate_policy_file(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    payload = read_json(path)
    errors = validate_policy_payload(payload)
    return {
        "content_sha256": payload.get("content_sha256"),
        "error_count": len(errors),
        "errors": errors,
        "evaluation_policy_id": payload.get("evaluation_policy_id"),
        "schema_version": payload.get("schema_version"),
        "status": "valid" if not errors else "invalid",
    }


def validate_detector_config_file(path: Path, policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    payload = read_json(path)
    binding = dict(policy.get("detector_configuration_binding") or {})
    expected_hash = binding.get("configuration_hash")
    active_hash = plate_detection.OIMPlateDetectorConfig().configuration_hash()
    if active_hash != expected_hash:
        errors.append("active selected detector configuration hash mismatch")
    if payload.get("configuration_hash") != expected_hash:
        errors.append("selected detector configuration file hash mismatch")
    checks = {
        "detector_id": plate_detection.DETECTOR_ID,
        "inference_mode": plate_detection.INFERENCE_MODE,
        "input_size": plate_detection.MODEL_INPUT_SIZE,
        "tile_size": plate_detection.TILE_SIZE,
        "tile_overlap": plate_detection.TILE_OVERLAP,
        "nms_iou_threshold": plate_detection.NMS_IOU_THRESHOLD,
        "primary_threshold": plate_detection.DEFAULT_PRIMARY_THRESHOLD,
        "proposal_threshold": None,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            errors.append(f"selected detector configuration {key} mismatch")
    return errors


def load_ground_truth(path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    errors = validate_ground_truth_payload(payload, policy)
    if errors:
        raise ValueError("invalid plate ground truth: " + "; ".join(errors))
    return payload


def validate_ground_truth_payload(payload: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    errors = plate_ground_truth.validate_plate_ground_truth_payload(payload)
    compatible = dict(policy.get("compatible_ground_truth") or {})
    if payload.get("ground_truth_id") != compatible.get("ground_truth_id"):
        errors.append("GT id incompatible with policy")
    if payload.get("schema_version") != compatible.get("schema_version"):
        errors.append("GT schema incompatible with policy")
    if payload.get("content_sha256") != compatible.get("content_sha256"):
        errors.append("GT content SHA incompatible with policy")
    return errors


def benchmark_by_id(ground_truth: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    for benchmark in ground_truth.get("benchmarks", []):
        if benchmark.get("benchmark_id") == benchmark_id:
            return benchmark
    raise ValueError(f"benchmark not found in plate ground truth: {benchmark_id}")


def records_for_benchmark(
    ground_truth: dict[str, Any],
    benchmark_id: str,
) -> list[GroundTruthRecord]:
    benchmark = benchmark_by_id(ground_truth, benchmark_id)
    records = []
    for item in benchmark.get("spatial_records", []):
        records.append(
            GroundTruthRecord(
                benchmark_id=str(item["benchmark_id"]),
                bbox_xyxy=as_box_strict(item["bbox_xyxy"]),
                dense_frame=int(item["dense_frame"]),
                parent_frame=int(item["parent_evaluation_frame"]),
                target_id=str(item["target_id"]),
            )
        )
    return sorted(records, key=lambda item: (item.target_id, item.parent_frame))


def certified_scope(ground_truth: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    benchmark = benchmark_by_id(ground_truth, benchmark_id)
    target = dict(benchmark.get("target") or {})
    mapping = dict(benchmark.get("frame_mapping") or {})
    return {
        "benchmark_id": benchmark_id,
        "canonical_frame_namespace": "evaluation_video_parent_frame",
        "dense_frame_interval": mapping.get("dense_frame_interval"),
        "detector_scored_frames": benchmark.get("record_count"),
        "evaluation_video": benchmark.get("evaluation_video"),
        "parent_evaluation_frame_interval": mapping.get("parent_evaluation_frame_interval"),
        "target_id": target.get("target_id"),
    }


def validate_frame_namespace(
    benchmark: dict[str, Any],
    policy: dict[str, Any],
    benchmark_id: str,
) -> list[str]:
    errors: list[str] = []
    policy_mapping = (
        dict(policy.get("canonical_frame_namespace") or {}).get("benchmarks", {}).get(benchmark_id)
    )
    if not isinstance(policy_mapping, dict):
        return ["wrong benchmark or missing canonical frame mapping"]
    mapping = dict(benchmark.get("frame_mapping") or {})
    dense_interval = list(mapping.get("dense_frame_interval") or [])
    parent_interval = list(mapping.get("parent_evaluation_frame_interval") or [])
    if dense_interval != list(policy_mapping.get("dense_frame_interval") or []):
        errors.append("wrong dense frame mapping")
    if parent_interval != list(policy_mapping.get("canonical_parent_frame_interval") or []):
        errors.append("wrong canonical parent frame mapping")
    offset = int(policy_mapping.get("parent_equals_dense_plus"))
    if int(mapping.get("parent_to_dense_offset", 0)) != -offset:
        errors.append("wrong parent/dense frame offset")
    records = benchmark.get("spatial_records") or []
    if records:
        dense_frames = sorted(int(record["dense_frame"]) for record in records)
        parent_frames = sorted(int(record["parent_evaluation_frame"]) for record in records)
        if [dense_frames[0], dense_frames[-1]] != dense_interval:
            errors.append("GT dense frame span differs from canonical interval")
        if [parent_frames[0], parent_frames[-1]] != parent_interval:
            errors.append("GT parent frame span differs from canonical interval")
        if any(parent != dense + offset for dense, parent in zip(dense_frames, parent_frames)):
            errors.append("GT record frame mapping is not parent=dense+offset")
    return errors


def plate_ground_truth_capability(
    ground_truth_path: Path,
    *,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    payload = read_json(ground_truth_path)
    if payload.get("schema_version") != plate_ground_truth.PLATE_GT_SCHEMA_VERSION:
        return {
            "blockers": ["machine_readable_ground_truth_has_no_license_plate_objects"],
            "category": CATEGORY,
            "clip_id": benchmark_id,
            "gt_capability_available": False,
            "ground_truth_id": payload.get("ground_truth_id"),
            "license_plate_object_count": 0,
            "object_ids": [],
            "objects_with_tight_boxes": 0,
            "publishable_metrics": False,
            "schema_version": "plate_evaluation_capability.v2",
        }
    policy = load_policy(DEFAULT_POLICY_PATH)
    validation_errors = validate_ground_truth_payload(payload, policy)
    benchmarks = payload.get("benchmarks", [])
    selected = (
        [benchmark_by_id(payload, benchmark_id)]
        if benchmark_id
        else [benchmark for benchmark in benchmarks if isinstance(benchmark, dict)]
    )
    target_ids = [str(item.get("target", {}).get("target_id")) for item in selected]
    record_count = sum(int(item.get("record_count") or 0) for item in selected)
    blockers = list(validation_errors)
    return {
        "blockers": blockers,
        "can_score_invocation_when_lineage_validates": not blockers and record_count > 0,
        "category": CATEGORY,
        "clip_id": benchmark_id,
        "gt_capability_available": not blockers and record_count > 0,
        "ground_truth_id": payload.get("ground_truth_id"),
        "license_plate_object_count": len(target_ids),
        "object_ids": target_ids,
        "objects_with_tight_boxes": len(target_ids),
        "publishable_metrics": False,
        "record_count": record_count,
        "schema_version": "plate_evaluation_capability.v2",
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }


def as_box(values: Any) -> Box | None:
    if not isinstance(values, list | tuple) or len(values) != 4:
        return None
    try:
        box = tuple(int(round(float(value))) for value in values)
    except (TypeError, ValueError):
        return None
    return box if valid_box(box) else None


def as_box_strict(values: Any) -> Box:
    box = as_box(values)
    if box is None:
        raise ValueError(f"invalid xyxy box: {values!r}")
    return box


def valid_box(box: Box) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def box_inside_frame(box: Box, *, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> bool:
    return 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height


def area(box: Box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_box(left: Box, right: Box) -> Box | None:
    box = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return box if valid_box(box) else None


def intersection_area(left: Box, right: Box) -> float:
    box = intersection_box(left, right)
    return 0.0 if box is None else area(box)


def iou(left: Box, right: Box) -> float:
    inter = intersection_area(left, right)
    union = area(left) + area(right) - inter
    return 0.0 if union <= 0 else inter / union


def center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def pair_diagnostics(
    prediction_box: Box,
    gt_box: Box,
    policy: dict[str, Any],
) -> PairDiagnostics:
    detector = dict(policy["detector_matching"])
    inter = intersection_area(prediction_box, gt_box)
    pred_area = area(prediction_box)
    gt_area = area(gt_box)
    pred_width = prediction_box[2] - prediction_box[0]
    pred_height = prediction_box[3] - prediction_box[1]
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]
    gt_coverage = inter / gt_area if gt_area else 0.0
    prediction_coverage = inter / pred_area if pred_area else 0.0
    area_ratio = pred_area / gt_area if gt_area else math.inf
    width_ratio = pred_width / gt_width if gt_width else math.inf
    height_ratio = pred_height / gt_height if gt_height else math.inf
    px, py = center(prediction_box)
    gx, gy = center(gt_box)
    normalized_center = math.hypot((px - gx) / gt_width, (py - gy) / gt_height)
    center_expansion = dict(detector["center_expansion_gt_fraction_each_side"])
    expanded = (
        gt_box[0] - gt_width * float(center_expansion["x"]),
        gt_box[1] - gt_height * float(center_expansion["y"]),
        gt_box[2] + gt_width * float(center_expansion["x"]),
        gt_box[3] + gt_height * float(center_expansion["y"]),
    )
    center_inside = expanded[0] <= px <= expanded[2] and expanded[1] <= py <= expanded[3]
    scale_error = abs(math.log(width_ratio)) + abs(math.log(height_ratio))
    pair_iou = iou(prediction_box, gt_box)
    matches = (
        inter > 0
        and pair_iou >= float(detector["iou_min"])
        and gt_coverage >= float(detector["gt_coverage_min"])
        and prediction_coverage >= float(detector["prediction_coverage_min"])
        and inside_range(area_ratio, detector["area_ratio_range"])
        and inside_range(width_ratio, detector["width_ratio_range"])
        and inside_range(height_ratio, detector["height_ratio_range"])
        and center_inside
    )
    return PairDiagnostics(
        area_ratio=area_ratio,
        gt_coverage=gt_coverage,
        height_ratio=height_ratio,
        intersection_area=inter,
        iou=pair_iou,
        matches=matches,
        normalized_center_distance=normalized_center,
        prediction_center_inside_expanded_gt=center_inside,
        prediction_coverage=prediction_coverage,
        scale_error=scale_error,
        width_ratio=width_ratio,
    )


def inside_range(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])


def eligible_detector_prediction(
    prediction: DetectorPrediction,
    policy: dict[str, Any],
) -> tuple[bool, list[str]]:
    detector = dict(policy["detector_matching"])
    reasons: list[str] = []
    if prediction.category != CATEGORY:
        reasons.append("wrong_category")
    if prediction.detector_id != plate_detection.DETECTOR_ID:
        reasons.append("wrong_detector_id")
    if prediction.configuration_hash != EXPECTED_DETECTOR_CONFIGURATION_HASH:
        reasons.append("wrong_detector_configuration")
    if prediction.score < float(detector["min_confidence"]):
        reasons.append("below_confidence_threshold")
    if prediction.bbox_xyxy is None or not box_inside_frame(prediction.bbox_xyxy):
        reasons.append("invalid_bbox")
    if any(marker in prediction.provenance.lower() for marker in REVIEWED_PROVENANCE_MARKERS):
        reasons.append("manual_or_reviewed_detector_contamination")
    if prediction.provenance and prediction.provenance not in DIRECT_DETECTOR_PROVENANCES:
        reasons.append("non_direct_detector_provenance")
    return (not reasons, reasons)


def detector_metrics_for_targets(
    records: list[GroundTruthRecord],
    predictions: list[DetectorPrediction],
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not records:
        raise ValueError("detector metrics require at least one GT record")
    ordered_records = sorted(
        records,
        key=lambda item: (item.parent_frame, item.target_id, item.dense_frame, item.bbox_xyxy),
    )
    record_order = {id(record): index for index, record in enumerate(ordered_records)}
    records_by_frame: dict[int, list[GroundTruthRecord]] = {}
    records_by_target: dict[str, list[GroundTruthRecord]] = {}
    for record in ordered_records:
        records_by_frame.setdefault(record.parent_frame, []).append(record)
        records_by_target.setdefault(record.target_id, []).append(record)
    predictions_by_frame: dict[int, list[DetectorPrediction]] = {}
    for prediction in predictions:
        ok, _ = eligible_detector_prediction(prediction, policy)
        if ok:
            predictions_by_frame.setdefault(prediction.frame_index, []).append(prediction)

    assignments: dict[tuple[str, int], DetectorPrediction] = {}
    pair_rows_by_target: dict[str, list[dict[str, Any]]] = {
        target_id: [] for target_id in records_by_target
    }
    matching_pair_count_by_gt: Counter[tuple[str, int]] = Counter()
    for frame, frame_records in sorted(records_by_frame.items()):
        pairs: list[
            tuple[
                tuple[float | int | str, ...],
                GroundTruthRecord,
                DetectorPrediction,
                PairDiagnostics,
            ]
        ] = []
        used_predictions_on_frame: set[int] = set()
        used_gt_on_frame: set[tuple[str, int]] = set()
        for record in frame_records:
            gt_key = detector_gt_key(record)
            for prediction in predictions_by_frame.get(frame, []):
                if prediction.bbox_xyxy is None:
                    continue
                diagnostics = pair_diagnostics(prediction.bbox_xyxy, record.bbox_xyxy, policy)
                pair_rows_by_target[record.target_id].append(
                    {
                        "candidate_id": prediction.candidate_id,
                        "frame": frame,
                        "matched_candidate": diagnostics.matches,
                        "pair": diagnostics.to_dict(),
                        "source_order": prediction.source_order,
                        "target_id": record.target_id,
                    }
                )
                if not diagnostics.matches:
                    continue
                matching_pair_count_by_gt[gt_key] += 1
                key = (
                    -diagnostics.gt_coverage,
                    -diagnostics.iou,
                    -diagnostics.prediction_coverage,
                    diagnostics.normalized_center_distance,
                    diagnostics.scale_error,
                    -prediction.score,
                    prediction.source_order,
                    record.target_id,
                    record_order[id(record)],
                )
                pairs.append((key, record, prediction, diagnostics))
        for _key, record, prediction, _diagnostics in sorted(pairs, key=lambda item: item[0]):
            prediction_key = detector_prediction_key(prediction)
            gt_key = detector_gt_key(record)
            if prediction_key in used_predictions_on_frame or gt_key in used_gt_on_frame:
                continue
            assignments[gt_key] = prediction
            used_predictions_on_frame.add(prediction_key)
            used_gt_on_frame.add(gt_key)

    metrics_by_target: dict[str, dict[str, Any]] = {}
    for target_id, target_records in sorted(records_by_target.items()):
        hit_frames = sorted(
            record.parent_frame
            for record in target_records
            if detector_gt_key(record) in assignments
        )
        miss_frames = sorted(
            record.parent_frame
            for record in target_records
            if detector_gt_key(record) not in assignments
        )
        scores = [
            assignments[detector_gt_key(record)].score
            for record in target_records
            if detector_gt_key(record) in assignments
        ]
        target_frame_set = {record.parent_frame for record in target_records}
        eligible_prediction_ids = {
            detector_prediction_key(prediction)
            for frame in target_frame_set
            for prediction in predictions_by_frame.get(frame, [])
        }
        target_assignment_ids = {
            detector_prediction_key(assignments[detector_gt_key(record)])
            for record in target_records
            if detector_gt_key(record) in assignments
        }
        duplicate_frames = sorted(
            {
                record.parent_frame
                for record in target_records
                if matching_pair_count_by_gt[detector_gt_key(record)] > 1
            }
        )
        metrics_by_target[target_id] = {
            "confidence_summary": numeric_summary(scores),
            "detector_scored_frames": len(target_records),
            "duplicate_matched_target_proposal_frames": duplicate_frames,
            "formal_precision": {
                "reason": "insufficient_spatial_negative_ignore_map",
                "value": None,
            },
            "hit_frames": hit_frames,
            "hit_frame_count": len(hit_frames),
            "maximum_consecutive_miss_run": longest_run(miss_frames),
            "miss_frames": miss_frames,
            "miss_frame_count": len(miss_frames),
            "miss_intervals": intervals_from_frames(miss_frames),
            "pair_diagnostics": pair_rows_by_target.get(target_id, []),
            "target_frame_recall": ratio(len(hit_frames), len(target_records)),
            "target_id": target_id,
            "unmatched_proposal_count": max(
                0,
                len(eligible_prediction_ids - target_assignment_ids),
            ),
        }
    return metrics_by_target


def detector_metrics_for_target(
    records: list[GroundTruthRecord],
    predictions: list[DetectorPrediction],
    policy: dict[str, Any],
) -> dict[str, Any]:
    metrics_by_target = detector_metrics_for_targets(records, predictions, policy)
    if len(metrics_by_target) != 1:
        raise ValueError("detector_metrics_for_target received multiple GT targets")
    return next(iter(metrics_by_target.values()))


def detector_gt_key(record: GroundTruthRecord) -> tuple[str, int]:
    return (record.target_id, record.parent_frame)


def detector_prediction_key(prediction: DetectorPrediction) -> int:
    return prediction.source_order


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "max": None, "mean": None, "min": None}
    return {
        "count": len(values),
        "max": round(max(values), 6),
        "mean": round(statistics.mean(values), 6),
        "min": round(min(values), 6),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def intervals_from_frames(frames: list[int] | set[int]) -> list[list[int]]:
    ordered = sorted(frames)
    if not ordered:
        return []
    intervals: list[list[int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        intervals.append([start, previous])
        start = previous = frame
    intervals.append([start, previous])
    return intervals


def longest_run(frames: list[int] | set[int]) -> int:
    intervals = intervals_from_frames(frames)
    return max((end - start + 1 for start, end in intervals), default=0)


def load_detector_predictions(path: Path) -> tuple[list[DetectorPrediction], list[int]]:
    predictions: list[DetectorPrediction] = []
    frame_records: list[int] = []
    source_order = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if "tracks" in record or "privacy_regions" in record:
                raise ValueError("detector input contains non-detector records")
            frame = int(record["frame_index"])
            frame_records.append(frame)
            for item in record.get("detections", []):
                metadata = dict(item.get("metadata") or {})
                predictions.append(
                    DetectorPrediction(
                        bbox_xyxy=as_box(item.get("bbox_xyxy")),
                        candidate_id=item.get("candidate_id"),
                        category=str(item.get("category") or ""),
                        configuration_hash=str(metadata.get("configuration_hash") or ""),
                        detector_id=str(item.get("detector_id") or ""),
                        frame_index=frame,
                        provenance=str(item.get("provenance") or ""),
                        score=float(item.get("score", 0.0)),
                        source=dict(item),
                        source_order=source_order,
                    )
                )
                source_order += 1
    return predictions, frame_records


def load_automatic_regions(path: Path) -> tuple[list[AutomaticRegion], list[int]]:
    regions: list[AutomaticRegion] = []
    frame_records: list[int] = []
    source_order = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record.get("frame_index", record.get("source_frame", -1)))
            frame_records.append(frame)
            if "tracks" in record:
                items = record.get("tracks", [])
                for item in items:
                    regions.append(
                        AutomaticRegion(
                            bbox_xyxy=as_box(item.get("bbox_xyxy")),
                            category=str(item.get("category") or ""),
                            frame_index=frame,
                            object_id=item.get("object_id") or item.get("target_id"),
                            provenance=str(item.get("provenance") or ""),
                            source=dict(item),
                            source_order=source_order,
                            status=str(item.get("status") or ""),
                            track_uid=item.get("track_uid"),
                        )
                    )
                    source_order += 1
            elif "privacy_regions" in record:
                for item in record.get("privacy_regions", []):
                    regions.append(region_from_privacy_payload(item, frame, source_order))
                    source_order += 1
            else:
                regions.append(region_from_privacy_payload(record, frame, source_order))
                source_order += 1
    return regions, frame_records


def region_from_privacy_payload(
    payload: dict[str, Any],
    frame: int,
    source_order: int,
) -> AutomaticRegion:
    return AutomaticRegion(
        bbox_xyxy=as_box(payload.get("bbox_xyxy")),
        category=str(payload.get("category") or CATEGORY),
        frame_index=frame,
        object_id=payload.get("object_id") or payload.get("target_id"),
        provenance=str(payload.get("provenance") or payload.get("method") or ""),
        source=dict(payload),
        source_order=source_order,
        status=str(payload.get("status") or payload.get("approval_state") or "covered"),
        track_uid=payload.get("track_uid"),
    )


def automatic_contamination_blockers(
    regions: list[AutomaticRegion],
    policy: dict[str, Any],
) -> list[str]:
    auto = dict(policy.get("automatic_provenance") or {})
    blacklist = tuple(str(item).lower() for item in auto.get("blacklist_markers", []))
    blockers: set[str] = set()
    for region in regions:
        text = f"{region.status} {region.provenance}".lower()
        if any(marker in text for marker in blacklist):
            blockers.add("automatic input contains manual/reviewed provenance")
        if region.category != CATEGORY:
            blockers.add("automatic input contains wrong category")
    return sorted(blockers)


def is_eligible_automatic_region(region: AutomaticRegion, policy: dict[str, Any]) -> bool:
    auto = dict(policy.get("automatic_provenance") or {})
    eligible_statuses = set(str(item) for item in auto.get("eligible_statuses", []))
    eligible_provenances = set(str(item) for item in auto.get("eligible_provenances", []))
    ineligible_statuses = set(str(item) for item in auto.get("ineligible_statuses", []))
    if region.category != CATEGORY or region.bbox_xyxy is None:
        return False
    if any(marker in region.provenance.lower() for marker in REVIEWED_PROVENANCE_MARKERS):
        return False
    if region.status in ineligible_statuses:
        return False
    return region.status in eligible_statuses and region.provenance in eligible_provenances


def expanded_region_box(
    region: AutomaticRegion,
    *,
    margin_fraction: float,
    margin_pixels: int,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
) -> Box | None:
    return rendering.expanded_clipped_box(
        region.bbox_xyxy,
        frame_width=frame_width,
        frame_height=frame_height,
        margin_fraction=margin_fraction,
        margin_pixels=margin_pixels,
    )


def automatic_metrics_for_target(
    records: list[GroundTruthRecord],
    regions: list[AutomaticRegion],
    policy: dict[str, Any],
    *,
    margin_fraction: float = 0.0,
    margin_pixels: int = 0,
    associated_track_uids: set[str] | None = None,
    eligible_region_predicate=None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("automatic metrics require at least one GT record")
    target_id = records[0].target_id
    associated_track_uids = associated_track_uids or set()
    is_eligible = eligible_region_predicate or is_eligible_automatic_region
    regions_by_frame: dict[int, list[AutomaticRegion]] = {}
    for region in regions:
        if is_eligible(region, policy):
            regions_by_frame.setdefault(region.frame_index, []).append(region)

    threshold = float(policy["automatic_coverage"]["coverage_threshold"])
    covered_frames: list[int] = []
    uncovered_frames: list[int] = []
    per_frame_coverage: dict[str, float] = {}
    provenance_counts: Counter[str] = Counter()
    association_unverified_frames: set[int] = set()
    associated_covered_frames: set[int] = set()
    total_redacted_area_by_frame: dict[str, int] = {}
    redacted_area_ratio_by_frame: dict[str, float] = {}
    unmatched_region_count = 0
    duplicate_region_count = 0

    for record in records:
        expanded_boxes: list[tuple[Box, AutomaticRegion]] = []
        associated_boxes: list[Box] = []
        for region in regions_by_frame.get(record.parent_frame, []):
            box = expanded_region_box(
                region,
                margin_fraction=margin_fraction,
                margin_pixels=margin_pixels,
            )
            if box is None:
                continue
            expanded_boxes.append((box, region))
            provenance_counts[region.provenance] += 1
            if is_region_associated(region, target_id, associated_track_uids):
                associated_boxes.append(box)
            if intersection_box(box, record.bbox_xyxy) is None:
                unmatched_region_count += 1
        clipped = [
            intersection_box(box, record.bbox_xyxy)
            for box, _region in expanded_boxes
            if intersection_box(box, record.bbox_xyxy) is not None
        ]
        clipped_boxes = [box for box in clipped if box is not None]
        gt_coverage = rectangle_union_area(clipped_boxes) / area(record.bbox_xyxy)
        per_frame_coverage[str(record.parent_frame)] = round(gt_coverage, 6)
        full_union = rectangle_union_area([box for box, _region in expanded_boxes])
        total_redacted_area_by_frame[str(record.parent_frame)] = int(full_union)
        redacted_area_ratio_by_frame[str(record.parent_frame)] = round(full_union / FRAME_AREA, 6)
        if gt_coverage >= threshold:
            covered_frames.append(record.parent_frame)
            associated_coverage = rectangle_union_area(
                [
                    intersection
                    for box in associated_boxes
                    if (intersection := intersection_box(box, record.bbox_xyxy)) is not None
                ]
            ) / area(record.bbox_xyxy)
            if associated_coverage >= threshold:
                associated_covered_frames.add(record.parent_frame)
            else:
                association_unverified_frames.add(record.parent_frame)
        else:
            uncovered_frames.append(record.parent_frame)
        duplicate_region_count += duplicate_region_pairs([box for box, _region in expanded_boxes])

    uncovered_intervals = intervals_from_frames(uncovered_frames)
    return {
        "associated_covered_frames": sorted(associated_covered_frames),
        "association_unverified_coverage_frames": sorted(association_unverified_frames),
        "coverage_rate": ratio(len(covered_frames), len(records)),
        "covered_frames": covered_frames,
        "covered_frame_count": len(covered_frames),
        "duplicate_region_count": duplicate_region_count,
        "flicker_events": flicker_events(records, uncovered_intervals),
        "longest_uncovered_run": longest_run(uncovered_frames),
        "over_redaction": {
            "redacted_area_ratio_by_frame": redacted_area_ratio_by_frame,
            "total_redacted_area_by_frame": total_redacted_area_by_frame,
            "unmatched_automatic_region_count": unmatched_region_count,
        },
        "per_frame_gt_coverage": per_frame_coverage,
        "provenance_breakdown": dict(sorted(provenance_counts.items())),
        "required_visible_frames": len(records),
        "target_id": target_id,
        "uncovered_frames": uncovered_frames,
        "uncovered_frame_count": len(uncovered_frames),
        "uncovered_intervals": uncovered_intervals,
        "unresolved_prefix": unresolved_prefix(records, uncovered_frames),
    }


def is_region_associated(
    region: AutomaticRegion,
    target_id: str,
    associated_track_uids: set[str],
) -> bool:
    return (
        region.object_id == target_id
        or (region.track_uid is not None and region.track_uid in associated_track_uids)
        or (region.source or {}).get("target_id") == target_id
    )


def is_eligible_reviewed_final_region(region: AutomaticRegion, policy: dict[str, Any]) -> bool:
    _ = policy
    return (
        region.category == CATEGORY and region.bbox_xyxy is not None and region.status == "covered"
    )


def rectangle_union_area(boxes: list[Box]) -> float:
    boxes = [box for box in boxes if valid_box(box)]
    if not boxes:
        return 0.0
    xs = sorted({coord for box in boxes for coord in (box[0], box[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((box[1], box[3]) for box in boxes if box[0] < right and box[2] > left)
        covered_y = 0
        current_start = current_end = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_y += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            covered_y += current_end - current_start
        total += (right - left) * covered_y
    return total


def duplicate_region_pairs(boxes: list[Box]) -> int:
    count = 0
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if iou(left, right) >= DUPLICATE_REGION_IOU_THRESHOLD:
                count += 1
    return count


def unresolved_prefix(
    records: list[GroundTruthRecord],
    uncovered_frames: list[int],
) -> list[int] | None:
    required = [record.parent_frame for record in records]
    uncovered = set(uncovered_frames)
    if not required or required[0] not in uncovered:
        return None
    end = required[0]
    for frame in required[1:]:
        if frame not in uncovered:
            break
        end = frame
    return [required[0], end]


def flicker_events(records: list[GroundTruthRecord], uncovered_intervals: list[list[int]]) -> int:
    required = [record.parent_frame for record in records]
    required_set = set(required)
    return sum(
        1
        for start, end in uncovered_intervals
        if start - 1 in required_set and end + 1 in required_set
    )


def reviewed_final_trust_gate_blockers(evidence: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not evidence.get("latest_reviewed_current"):
        blockers.append("stale_latest_reviewed")
    if not evidence.get("final_current"):
        blockers.append("stale_final")
    if not evidence.get("final_pointer_matches_latest"):
        blockers.append("final_pointer_mismatch")
    if not evidence.get("reviewed_region_set_sha256"):
        blockers.append("missing_reviewed_region_set_sha256")
    elif evidence.get("expected_reviewed_region_set_sha256") and evidence.get(
        "reviewed_region_set_sha256"
    ) != evidence.get("expected_reviewed_region_set_sha256"):
        blockers.append("reviewed_region_set_sha256_mismatch")
    if not evidence.get("reviewed_regions_artifact_valid"):
        blockers.append("reviewed_regions_artifact_invalid")
    if not evidence.get("source_lineage_valid"):
        blockers.append("source_lineage_invalid")
    if not evidence.get("category_run_lineage_valid"):
        blockers.append("category_run_lineage_invalid")
    if not evidence.get("media_validation_valid"):
        blockers.append("media_validation_failed")
    if not evidence.get("region_validation_valid"):
        blockers.append("region_validation_failed")
    if not evidence.get("human_sweep_acknowledged"):
        blockers.append("missing_sweep")
    if not evidence.get("finalization_valid"):
        blockers.append("finalization_invalid")
    return blockers


def reviewed_final_metrics_for_target(
    records: list[GroundTruthRecord],
    regions: list[AutomaticRegion],
    policy: dict[str, Any],
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    metrics = automatic_metrics_for_target(
        records,
        regions,
        policy,
        eligible_region_predicate=is_eligible_reviewed_final_region,
    )
    blockers = reviewed_final_trust_gate_blockers(evidence)
    return {
        "category_run_lineage_valid": bool(evidence.get("category_run_lineage_valid")),
        "coverage_rate": metrics["coverage_rate"],
        "final_current": bool(evidence.get("final_current")),
        "final_revision_identity": evidence.get("final_revision_identity"),
        "finalization_current_valid": bool(evidence.get("finalization_valid")),
        "geometrically_covered_frames": metrics["covered_frames"],
        "human_sweep_acknowledged": bool(evidence.get("human_sweep_acknowledged")),
        "latest_reviewed_current": bool(evidence.get("latest_reviewed_current")),
        "latest_reviewed_revision_identity": evidence.get("latest_reviewed_revision_identity"),
        "longest_uncovered_run": metrics["longest_uncovered_run"],
        "media_validation_valid": bool(evidence.get("media_validation_valid")),
        "publishable_metrics": not blockers and not metrics["uncovered_frames"],
        "region_validation_valid": bool(evidence.get("region_validation_valid")),
        "required_gt_frames": metrics["required_visible_frames"],
        "reviewed_regions_artifact_valid": bool(evidence.get("reviewed_regions_artifact_valid")),
        "reviewed_region_set_sha256": evidence.get("reviewed_region_set_sha256"),
        "source_lineage_valid": bool(evidence.get("source_lineage_valid")),
        "target_id": metrics["target_id"],
        "trust_gate_blockers": blockers,
        "uncovered_frames": metrics["uncovered_frames"],
        "uncovered_intervals": metrics["uncovered_intervals"],
    }


def load_reviewed_regions(path: Path) -> list[AutomaticRegion]:
    records = review_finalization.load_privacy_region_records(path)
    regions: list[AutomaticRegion] = []
    for index, record in enumerate(records):
        regions.append(
            AutomaticRegion(
                bbox_xyxy=record.bbox_xyxy,
                category=record.category,
                frame_index=record.source_frame,
                object_id=record.object_id,
                provenance=record.provenance,
                source=record.to_dict(),
                source_order=index,
                status="covered" if record.render_eligible else "unresolved",
            )
        )
    return regions


def source_identity(path: Path) -> dict[str, Any]:
    info = media.probe_media(path)
    return {
        "frame_count": int(info.video_stream.frame_count or 0),
        "height": int(info.video_stream.height),
        "path": path.as_posix(),
        "sha256": media.sha256_file(path),
        "width": int(info.video_stream.width),
    }


def validate_source_for_benchmark(
    source: dict[str, Any],
    benchmark: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected = dict(benchmark.get("evaluation_video") or {})
    if source.get("sha256") != expected.get("sha256"):
        errors.append("source video hash mismatch")
    for key in ("frame_count", "height", "width"):
        if int(source.get(key) or -1) != int(expected.get(key) or -2):
            errors.append(f"source video {key} mismatch")
    return errors


def validate_frame_records(frame_records: list[int], source_frame_count: int) -> list[str]:
    if source_frame_count <= 0:
        return ["source frame count unknown"]
    expected = set(range(source_frame_count))
    actual = set(frame_records)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing frame records starting {missing[:5]}")
        if extra:
            parts.append(f"extra frame records starting {extra[:5]}")
        return ["incomplete source span: " + "; ".join(parts)]
    return []


def detector_validation(
    *,
    ground_truth_path: Path,
    benchmark_id: str,
    source_video_path: Path,
    detections_path: Path | None,
    detector_config_path: Path | None,
    policy_path: Path,
    detection_summary_path: Path | None = None,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    ground_truth = load_ground_truth(ground_truth_path, policy)
    benchmark = benchmark_by_id(ground_truth, benchmark_id)
    source = source_identity(source_video_path)
    blockers = validate_source_for_benchmark(source, benchmark)
    blockers.extend(validate_frame_namespace(benchmark, policy, benchmark_id))
    if detector_config_path is None:
        blockers.append("missing detector config")
    else:
        blockers.extend(validate_detector_config_file(detector_config_path, policy))
    predictions: list[DetectorPrediction] = []
    frame_records: list[int] = []
    resolved_summary_path = detection_summary_path
    if detections_path is None:
        blockers.append("missing detections")
    else:
        if resolved_summary_path is None:
            candidate_summary = detections_path.parent / "summary.json"
            if candidate_summary.exists():
                resolved_summary_path = candidate_summary
        if resolved_summary_path is None:
            blockers.append("missing detector source/config summary")
        else:
            blockers.extend(validate_detection_summary(resolved_summary_path, source))
        try:
            predictions, frame_records = load_detector_predictions(detections_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"invalid detections artifact: {exc}")
        else:
            blockers.extend(validate_frame_records(frame_records, int(source["frame_count"])))
            blockers.extend(detector_artifact_blockers(predictions))
    return validation_payload(
        mode=MODE_DETECTOR,
        blockers=blockers,
        ground_truth=ground_truth,
        policy=policy,
        source=source,
        artifact_paths={
            "detections": detections_path,
            "detection_summary": resolved_summary_path,
            "detector_config": detector_config_path,
        },
        extra={
            "prediction_count": len(predictions),
            "record_frame_count": len(frame_records),
        },
    )


def validate_detection_summary(path: Path, source: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid detector summary: {exc}"]
    if payload.get("source_sha256") != source.get("sha256"):
        blockers.append("detector summary source hash mismatch")
    if int(payload.get("source_frame_count") or -1) != int(source.get("frame_count") or -2):
        blockers.append("detector summary source frame count mismatch")
    if int(payload.get("input_width") or -1) != FRAME_WIDTH:
        blockers.append("detector summary input width mismatch")
    if int(payload.get("input_height") or -1) != FRAME_HEIGHT:
        blockers.append("detector summary input height mismatch")
    if payload.get("detector_id") != plate_detection.DETECTOR_ID:
        blockers.append("detector summary detector id mismatch")
    if payload.get("configuration_hash") != EXPECTED_DETECTOR_CONFIGURATION_HASH:
        blockers.append("detector summary configuration hash mismatch")
    if payload.get("inference_mode") != plate_detection.INFERENCE_MODE:
        blockers.append("detector summary inference mode mismatch")
    if int(payload.get("tile_size") or 0) != plate_detection.TILE_SIZE:
        blockers.append("detector summary tile size mismatch")
    if float(payload.get("tile_overlap", -1.0)) != plate_detection.TILE_OVERLAP:
        blockers.append("detector summary tile overlap mismatch")
    if float(payload.get("nms_iou_threshold", -1.0)) != plate_detection.NMS_IOU_THRESHOLD:
        blockers.append("detector summary NMS IoU mismatch")
    if float(payload.get("primary_threshold", -1.0)) != plate_detection.DEFAULT_PRIMARY_THRESHOLD:
        blockers.append("detector summary primary threshold mismatch")
    proposal_threshold = payload.get("proposal_threshold")
    if proposal_threshold not in (None, plate_detection.DEFAULT_PRIMARY_THRESHOLD):
        blockers.append("detector summary proposal threshold mismatch")
    return blockers


def detector_artifact_blockers(predictions: list[DetectorPrediction]) -> list[str]:
    blockers: set[str] = set()
    for prediction in predictions:
        if prediction.category != CATEGORY:
            blockers.add("wrong category in detector artifact")
        if prediction.detector_id != plate_detection.DETECTOR_ID:
            blockers.add("wrong detector id in detector artifact")
        if prediction.configuration_hash != EXPECTED_DETECTOR_CONFIGURATION_HASH:
            blockers.add("wrong detector configuration in detector artifact")
        if prediction.provenance and prediction.provenance not in DIRECT_DETECTOR_PROVENANCES:
            blockers.add("detector artifact contains non-direct provenance")
        if prediction.bbox_xyxy is None or not box_inside_frame(prediction.bbox_xyxy):
            blockers.add("invalid detector bbox")
    return sorted(blockers)


def automatic_validation(
    *,
    ground_truth_path: Path,
    benchmark_id: str,
    source_video_path: Path,
    tracks_path: Path | None,
    review_intervals_path: Path | None,
    render_manifest_path: Path | None,
    policy_path: Path,
    tracking_manifest_path: Path | None = None,
    gap_decisions_path: Path | None = None,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    ground_truth = load_ground_truth(ground_truth_path, policy)
    benchmark = benchmark_by_id(ground_truth, benchmark_id)
    source = source_identity(source_video_path)
    blockers = validate_source_for_benchmark(source, benchmark)
    blockers.extend(validate_frame_namespace(benchmark, policy, benchmark_id))
    regions: list[AutomaticRegion] = []
    frame_records: list[int] = []
    resolved_tracking_manifest_path = tracking_manifest_path
    resolved_gap_decisions_path = gap_decisions_path
    if tracks_path is None:
        blockers.append("missing tracks")
    else:
        if resolved_tracking_manifest_path is None:
            candidate_manifest = tracks_path.parent / "tracking_manifest.json"
            if candidate_manifest.exists():
                resolved_tracking_manifest_path = candidate_manifest
        if resolved_gap_decisions_path is None:
            candidate_gaps = tracks_path.parent / "gap_decisions.json"
            if candidate_gaps.exists():
                resolved_gap_decisions_path = candidate_gaps
        try:
            regions, frame_records = load_automatic_regions(tracks_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"invalid automatic artifact: {exc}")
        else:
            blockers.extend(validate_frame_records(frame_records, int(source["frame_count"])))
            blockers.extend(automatic_contamination_blockers(regions, policy))
    if resolved_tracking_manifest_path is not None:
        blockers.extend(validate_tracking_manifest(resolved_tracking_manifest_path, source))
    if resolved_gap_decisions_path is not None:
        blockers.extend(validate_gap_decisions(resolved_gap_decisions_path))
    review_payload = load_json_optional(review_intervals_path, "review intervals", blockers)
    render_manifest = load_json_optional(render_manifest_path, "render manifest", blockers)
    if render_manifest:
        blockers.extend(
            validate_render_manifest_for_automatic(
                render_manifest,
                source=source,
                tracks_path=tracks_path,
                review_intervals_path=review_intervals_path,
            )
        )
    if isinstance(review_payload, dict) and review_payload.get("interval_decisions"):
        blockers.append("unexpected face interval decision artifact for plate automatic input")
    return validation_payload(
        mode=MODE_AUTOMATIC,
        blockers=blockers,
        ground_truth=ground_truth,
        policy=policy,
        source=source,
        artifact_paths={
            "gap_decisions": resolved_gap_decisions_path,
            "render_manifest": render_manifest_path,
            "review_intervals": review_intervals_path,
            "tracking_manifest": resolved_tracking_manifest_path,
            "tracks": tracks_path,
        },
        extra={
            "automatic_region_count": len(regions),
            "manual_or_reviewed_contamination_absent": not automatic_contamination_blockers(
                regions, policy
            ),
            "needs_review_excluded_from_automatic_credit": True,
            "record_frame_count": len(frame_records),
        },
    )


def load_json_optional(
    path: Path | None,
    label: str,
    blockers: list[str],
) -> dict[str, Any] | list[Any] | None:
    if path is None:
        blockers.append(f"missing {label}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"invalid {label}: {exc}")
        return None


def validate_tracking_manifest(path: Path, source: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid tracking manifest: {exc}"]
    if manifest.get("schema_version") != "plate_tracking.v1":
        blockers.append("tracking manifest schema mismatch")
    if manifest.get("source_sha256") != source.get("sha256"):
        blockers.append("tracking manifest source hash mismatch")
    if int(manifest.get("source_frame_count") or -1) != int(source.get("frame_count") or -2):
        blockers.append("tracking manifest source frame count mismatch")
    if int(manifest.get("input_width") or -1) != FRAME_WIDTH:
        blockers.append("tracking manifest input width mismatch")
    if int(manifest.get("input_height") or -1) != FRAME_HEIGHT:
        blockers.append("tracking manifest input height mismatch")
    if manifest.get("detector_id") != plate_detection.DETECTOR_ID:
        blockers.append("tracking manifest detector id mismatch")
    if manifest.get("detector_configuration_hash") != EXPECTED_DETECTOR_CONFIGURATION_HASH:
        blockers.append("tracking manifest detector configuration mismatch")
    expected_tracking_hash = plate_tracking.PlateTrackingConfig().configuration_hash()
    if manifest.get("tracking_configuration_hash") != expected_tracking_hash:
        blockers.append("tracking manifest tracking configuration mismatch")
    if manifest.get("full_source_processed") is not True:
        blockers.append("tracking manifest incomplete source span")
    return blockers


def validate_gap_decisions(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid gap decisions: {exc}"]
    if not isinstance(payload, list):
        return ["gap decisions must be a list"]
    valid_decisions = {"accepted_interpolation", "needs_review"}
    blockers: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            blockers.append("gap decision row is not an object")
            continue
        if item.get("decision") not in valid_decisions:
            blockers.append("gap decisions contain unsupported decision")
        for key in ("frame_start", "frame_end", "track_uid"):
            if key not in item:
                blockers.append(f"gap decision missing {key}")
    return sorted(set(blockers))


def validate_render_manifest_for_automatic(
    manifest: dict[str, Any],
    *,
    source: dict[str, Any],
    tracks_path: Path | None,
    review_intervals_path: Path | None,
) -> list[str]:
    blockers: list[str] = []
    if manifest.get("source_video_hash") != source.get("sha256"):
        blockers.append("render manifest source hash mismatch")
    if int(manifest.get("source_frame_count") or -1) != int(source.get("frame_count") or -2):
        blockers.append("render manifest source frame count mismatch")
    if int(manifest.get("input_width") or -1) != FRAME_WIDTH:
        blockers.append("render manifest input width mismatch")
    if int(manifest.get("input_height") or -1) != FRAME_HEIGHT:
        blockers.append("render manifest input height mismatch")
    if manifest.get("detector_id") != plate_detection.DETECTOR_ID:
        blockers.append("render manifest detector id mismatch")
    expected_tracking_hash = plate_tracking.PlateTrackingConfig().configuration_hash()
    if manifest.get("tracking_configuration_hash") != expected_tracking_hash:
        blockers.append("render manifest tracking configuration mismatch")
    expected_render_hash = plate_rendering.default_config().configuration_hash()
    if manifest.get("configuration_hash") != expected_render_hash:
        blockers.append("render manifest render configuration mismatch")
    if manifest.get("final_approval_status") is True:
        blockers.append("automatic input points at final-approved render")
    if manifest.get("preview_non_final") is not True:
        blockers.append("automatic render manifest is not preview/non-final")
    if manifest.get("full_source_processed") is not True:
        blockers.append("render manifest incomplete source span")
    if tracks_path is not None and manifest.get("privacy_region_input_hash") != media.sha256_file(
        tracks_path
    ):
        blockers.append("render manifest tracks/privacy-region input hash mismatch")
    if review_intervals_path is not None and manifest.get(
        "review_intervals_hash"
    ) != media.sha256_file(review_intervals_path):
        blockers.append("render manifest review intervals hash mismatch")
    config = dict(manifest.get("configuration") or {})
    if config.get("detector_configuration_hash") != EXPECTED_DETECTOR_CONFIGURATION_HASH:
        blockers.append("render manifest detector configuration mismatch")
    if tuple(config.get("eligible_statuses") or []) != plate_rendering.ELIGIBLE_STATUSES:
        blockers.append("render manifest eligible statuses mismatch")
    if tuple(config.get("eligible_provenances") or []) != plate_rendering.ELIGIBLE_PROVENANCES:
        blockers.append("render manifest eligible provenances mismatch")
    return blockers


def reviewed_final_validation(
    *,
    ground_truth_path: Path,
    benchmark_id: str,
    source_video_path: Path,
    reviewed_regions_path: Path | None,
    latest_reviewed_pointer_path: Path | None,
    final_pointer_path: Path | None,
    rendered_video_path: Path | None,
    media_validation_path: Path | None,
    region_validation_path: Path | None,
    policy_path: Path,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    ground_truth = load_ground_truth(ground_truth_path, policy)
    benchmark = benchmark_by_id(ground_truth, benchmark_id)
    source = source_identity(source_video_path)
    blockers = validate_source_for_benchmark(source, benchmark)
    blockers.extend(validate_frame_namespace(benchmark, policy, benchmark_id))
    evidence = {
        "category_run_lineage_valid": False,
        "final_current": False,
        "final_pointer_matches_latest": False,
        "finalization_valid": False,
        "human_sweep_acknowledged": False,
        "latest_reviewed_current": False,
        "media_validation_valid": False,
        "region_validation_valid": False,
        "reviewed_regions_artifact_valid": False,
        "reviewed_region_set_sha256": None,
        "source_lineage_valid": not blockers,
    }
    latest_payload: dict[str, Any] | None = None
    final_payload: dict[str, Any] | None = None
    if reviewed_regions_path is None:
        blockers.append("missing reviewed regions")
    if latest_reviewed_pointer_path is None:
        blockers.append("missing Latest reviewed pointer")
    else:
        latest_payload = validate_latest_pointer_path(latest_reviewed_pointer_path, blockers)
        if latest_payload:
            evidence["latest_reviewed_current"] = True
            evidence["reviewed_region_set_sha256"] = latest_payload.get(
                "reviewed_region_set_sha256"
            )
    if final_pointer_path is None:
        blockers.append("missing Final pointer")
    else:
        final_payload = validate_final_pointer_path(final_pointer_path, blockers)
        if final_payload:
            evidence["final_current"] = True
            evidence["finalization_valid"] = True
            evidence["human_sweep_acknowledged"] = bool(
                final_payload.get("sweep_acknowledgement_sha256")
            )
            evidence["category_run_lineage_valid"] = True
            if evidence["reviewed_region_set_sha256"] is None:
                evidence["reviewed_region_set_sha256"] = final_payload.get(
                    "reviewed_region_set_sha256"
                )
    if latest_payload and final_payload:
        evidence["final_pointer_matches_latest"] = final_payload.get(
            "approved_latest_reviewed_revision_id"
        ) == latest_payload.get("revision_id") and final_payload.get(
            "reviewed_region_set_sha256"
        ) == latest_payload.get("reviewed_region_set_sha256")
    if reviewed_regions_path is not None and latest_payload:
        latest_package_root = (
            latest_reviewed_pointer_path.parent.parent
            if latest_reviewed_pointer_path is not None
            else reviewed_regions_path.parent
        )
        try:
            binding = reviewed_regions.validate_reviewed_regions_artifact_binding(
                latest_package_root,
                reviewed_regions_path,
                latest_pointer=latest_payload,
                final_pointer=final_payload,
                expected_case_id=str(benchmark.get("case_id") or ""),
                expected_category=CATEGORY,
                expected_source_sha256=str(source["sha256"]),
            )
        except (
            OSError,
            ValueError,
            reviewed_regions.ReviewedRegionCompilationError,
        ) as exc:
            blockers.append(f"reviewed regions validation failed: {exc}")
        else:
            evidence["reviewed_regions_artifact_valid"] = True
            evidence["reviewed_region_set_sha256"] = binding["reviewed_region_set_sha256"]
            evidence["reviewed_regions_binding"] = binding
    if media_validation_path is None:
        blockers.append("missing media validation")
    else:
        media_validation = read_json(media_validation_path)
        evidence["media_validation_valid"] = validation_passed(media_validation)
        if not evidence["media_validation_valid"]:
            blockers.append("media validation failed")
    if region_validation_path is None:
        blockers.append("missing region validation")
    else:
        region_validation = read_json(region_validation_path)
        evidence["region_validation_valid"] = validation_passed(region_validation)
        if not evidence["region_validation_valid"]:
            blockers.append("region validation failed")
    if rendered_video_path is None:
        blockers.append("missing rendered video")
    elif final_payload and final_payload.get("final_video_sha256") != media.sha256_file(
        rendered_video_path
    ):
        blockers.append("rendered final video hash mismatch")
    blockers.extend(reviewed_final_trust_gate_blockers(evidence))
    return validation_payload(
        mode=MODE_REVIEWED_FINAL,
        blockers=sorted(set(blockers)),
        ground_truth=ground_truth,
        policy=policy,
        source=source,
        artifact_paths={
            "final_pointer": final_pointer_path,
            "latest_reviewed_pointer": latest_reviewed_pointer_path,
            "media_validation": media_validation_path,
            "region_validation": region_validation_path,
            "rendered_video": rendered_video_path,
            "reviewed_regions": reviewed_regions_path,
        },
        extra={"reviewed_final_evidence": evidence},
    )


def validation_passed(payload: dict[str, Any]) -> bool:
    nested_media = dict(payload.get("media_validation") or {})
    return payload.get("passed") is True or nested_media.get("passed") is True


def validate_latest_pointer_path(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    try:
        if path.name != "current.json" or path.parent.name != "latest_reviewed":
            raise ValueError("Latest reviewed pointer path is not latest_reviewed/current.json")
        package_root = path.parent.parent
        pointer = read_json(path)
        return reviewed_regions.validate_current_pointer(package_root, pointer)
    except (OSError, ValueError, reviewed_regions.ReviewedRegionCompilationError) as exc:
        blockers.append(f"Latest reviewed validation failed: {exc}")
        return None


def validate_final_pointer_path(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    try:
        if path.name != "current.json" or path.parent.name != "final":
            raise ValueError("Final pointer path is not final/current.json")
        package_root = path.parent.parent
        pointer = read_json(path)
        return review_finalization.validate_current_final_pointer(package_root, pointer)
    except (OSError, ValueError) as exc:
        blockers.append(f"Final validation failed: {exc}")
        return None


def validation_payload(
    *,
    mode: str,
    blockers: list[str],
    ground_truth: dict[str, Any],
    policy: dict[str, Any],
    source: dict[str, Any],
    artifact_paths: dict[str, Path | None],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_hashes = {
        key: None if path is None else media.sha256_file(path)
        for key, path in artifact_paths.items()
    }
    return {
        "blockers": sorted(set(blockers)),
        "configuration_identity": configuration_identity_for_mode(mode, policy),
        "ground_truth_content_sha256": ground_truth.get("content_sha256"),
        "input_artifact_hashes": artifact_hashes,
        "mode": mode,
        "policy_sha256": policy.get("content_sha256"),
        "schema_version": "plate_evaluation_validation.v1",
        "source_identity": source,
        "status": "ready" if not blockers else "blocked",
        "validation_only": True,
    } | (extra or {})


def configuration_identity_for_mode(mode: str, policy: dict[str, Any]) -> dict[str, Any]:
    if mode == MODE_DETECTOR:
        return dict(policy.get("detector_configuration_binding") or {})
    if mode == MODE_AUTOMATIC:
        return {
            "detector_configuration_hash": EXPECTED_DETECTOR_CONFIGURATION_HASH,
            "render_configuration_hash": plate_rendering.default_config().configuration_hash(),
            "tracking_configuration_hash": (
                plate_tracking.PlateTrackingConfig().configuration_hash()
            ),
        }
    if mode == MODE_REVIEWED_FINAL:
        return {"reviewed_final_evidence": "lineage_bound"}
    return {}


def run_plate_evaluation(
    *,
    ground_truth_path: Path,
    benchmark_id: str,
    output_dir: Path,
    mode: str = MODE_CAPABILITY,
    source_video_path: Path | None = None,
    detections_path: Path | None = None,
    detection_summary_path: Path | None = None,
    detector_config_path: Path | None = None,
    tracks_path: Path | None = None,
    tracking_manifest_path: Path | None = None,
    gap_decisions_path: Path | None = None,
    review_intervals_path: Path | None = None,
    render_manifest_path: Path | None = None,
    reviewed_regions_path: Path | None = None,
    latest_reviewed_pointer_path: Path | None = None,
    final_pointer_path: Path | None = None,
    rendered_video_path: Path | None = None,
    media_validation_path: Path | None = None,
    region_validation_path: Path | None = None,
    evaluation_policy_path: Path = DEFAULT_POLICY_PATH,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    ensure_output_dir(output_dir, overwrite=overwrite)
    if mode == MODE_CAPABILITY:
        capability = plate_ground_truth_capability(ground_truth_path, benchmark_id=benchmark_id)
        metrics = capability_metrics(ground_truth_path, benchmark_id, capability)
        write_outputs(output_dir, metrics, validation={"status": "not_applicable"})
        return metrics
    if source_video_path is None:
        raise ValueError("plate scoring modes require source_video_path")
    if mode == MODE_DETECTOR:
        validation = detector_validation(
            benchmark_id=benchmark_id,
            detection_summary_path=detection_summary_path,
            detections_path=detections_path,
            detector_config_path=detector_config_path,
            ground_truth_path=ground_truth_path,
            policy_path=evaluation_policy_path,
            source_video_path=source_video_path,
        )
    elif mode == MODE_AUTOMATIC:
        validation = automatic_validation(
            benchmark_id=benchmark_id,
            ground_truth_path=ground_truth_path,
            policy_path=evaluation_policy_path,
            gap_decisions_path=gap_decisions_path,
            render_manifest_path=render_manifest_path,
            review_intervals_path=review_intervals_path,
            source_video_path=source_video_path,
            tracking_manifest_path=tracking_manifest_path,
            tracks_path=tracks_path,
        )
    elif mode == MODE_REVIEWED_FINAL:
        validation = reviewed_final_validation(
            benchmark_id=benchmark_id,
            final_pointer_path=final_pointer_path,
            ground_truth_path=ground_truth_path,
            latest_reviewed_pointer_path=latest_reviewed_pointer_path,
            media_validation_path=media_validation_path,
            policy_path=evaluation_policy_path,
            region_validation_path=region_validation_path,
            rendered_video_path=rendered_video_path,
            reviewed_regions_path=reviewed_regions_path,
            source_video_path=source_video_path,
        )
    else:
        raise ValueError(f"unsupported plate evaluation mode: {mode}")
    if validate_only:
        metrics = validation_only_metrics(ground_truth_path, benchmark_id, validation)
        write_outputs(output_dir, metrics, validation=validation)
        return metrics
    if validation["status"] != "ready":
        metrics = validation_only_metrics(ground_truth_path, benchmark_id, validation)
        write_outputs(output_dir, metrics, validation=validation)
        return metrics
    metrics = score_mode(
        benchmark_id=benchmark_id,
        detections_path=detections_path,
        evaluation_policy_path=evaluation_policy_path,
        ground_truth_path=ground_truth_path,
        mode=mode,
        render_manifest_path=render_manifest_path,
        reviewed_regions_path=reviewed_regions_path,
        tracks_path=tracks_path,
        validation=validation,
    )
    write_outputs(output_dir, metrics, validation=validation)
    return metrics


def score_mode(
    *,
    benchmark_id: str,
    detections_path: Path | None,
    evaluation_policy_path: Path,
    ground_truth_path: Path,
    mode: str,
    render_manifest_path: Path | None,
    reviewed_regions_path: Path | None,
    tracks_path: Path | None,
    validation: dict[str, Any],
) -> dict[str, Any]:
    policy = load_policy(evaluation_policy_path)
    ground_truth = load_ground_truth(ground_truth_path, policy)
    records = records_for_benchmark(ground_truth, benchmark_id)
    if mode == MODE_DETECTOR:
        assert detections_path is not None
        predictions, _ = load_detector_predictions(detections_path)
        target_metrics_by_id = detector_metrics_for_targets(records, predictions, policy)
    elif mode == MODE_AUTOMATIC:
        assert tracks_path is not None and render_manifest_path is not None
        regions, _ = load_automatic_regions(tracks_path)
        manifest = read_json(render_manifest_path)
        config = dict(manifest.get("configuration") or {})
        target_metrics = automatic_metrics_for_target(
            records,
            regions,
            policy,
            margin_fraction=float(config.get("margin_fraction", 0.0)),
            margin_pixels=int(config.get("margin_pixels", 0)),
        )
        target_metrics_by_id = {target_metrics["target_id"]: target_metrics}
    elif mode == MODE_REVIEWED_FINAL:
        assert reviewed_regions_path is not None
        regions = load_reviewed_regions(reviewed_regions_path)
        target_metrics = reviewed_final_metrics_for_target(
            records,
            regions,
            policy,
            evidence=dict(validation.get("reviewed_final_evidence") or {}),
        )
        target_metrics_by_id = {target_metrics["target_id"]: target_metrics}
    else:
        raise ValueError(f"unsupported score mode: {mode}")
    return report_payload(
        aggregate_metrics=aggregate_target_metrics(
            list(target_metrics_by_id.values()),
            mode=mode,
        ),
        benchmark_id=benchmark_id,
        failure_intervals=failure_intervals_from_target_metrics(target_metrics_by_id),
        ground_truth=ground_truth,
        mode=mode,
        per_target_metrics=target_metrics_by_id,
        policy=policy,
        publishable_metrics=validation["status"] == "ready",
        validation=validation,
    )


def failure_intervals_from_target_metrics(
    target_metrics_by_id: dict[str, dict[str, Any]],
) -> list[Any]:
    if len(target_metrics_by_id) == 1:
        metrics = next(iter(target_metrics_by_id.values()))
        return metrics.get("miss_intervals") or metrics.get("uncovered_intervals") or []
    intervals = []
    for target_id, metrics in sorted(target_metrics_by_id.items()):
        target_intervals = metrics.get("miss_intervals") or metrics.get("uncovered_intervals") or []
        if target_intervals:
            intervals.append({"intervals": target_intervals, "target_id": target_id})
    return intervals


def capability_metrics(
    ground_truth_path: Path,
    benchmark_id: str,
    capability: dict[str, Any],
) -> dict[str, Any]:
    return {
        "aggregate_metrics": {},
        "capability": capability,
        "category": CATEGORY,
        "certified_scope": {"benchmark_id": benchmark_id},
        "configuration_identity": {},
        "evaluation_policy_id": None,
        "evaluation_policy_sha256": None,
        "failure_intervals": [],
        "ground_truth": ground_truth_path.as_posix(),
        "ground_truth_content_sha256": capability.get("content_sha256"),
        "ground_truth_id": capability.get("ground_truth_id"),
        "input_artifact_hashes": {},
        "limitations": ["capability report only; no invocation metrics"],
        "matching_policy_version": None,
        "mode": MODE_CAPABILITY,
        "per_target_metrics": {},
        "provenance_summary": {},
        "publishable_metrics": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": None,
        "validation": {"status": "not_applicable"},
    }


def validation_only_metrics(
    ground_truth_path: Path,
    benchmark_id: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "aggregate_metrics": {},
        "category": CATEGORY,
        "certified_scope": {"benchmark_id": benchmark_id},
        "configuration_identity": validation.get("configuration_identity", {}),
        "evaluation_policy_id": POLICY_ID,
        "evaluation_policy_sha256": validation.get("policy_sha256"),
        "failure_intervals": [],
        "ground_truth": ground_truth_path.as_posix(),
        "ground_truth_content_sha256": validation.get("ground_truth_content_sha256"),
        "ground_truth_id": plate_ground_truth.PLATE_GT_ID,
        "input_artifact_hashes": validation.get("input_artifact_hashes", {}),
        "limitations": [
            "validation-only invocation",
            "certified metrics were not calculated",
        ],
        "matching_policy_version": validation.get("mode"),
        "mode": validation["mode"],
        "per_target_metrics": {},
        "provenance_summary": {},
        "publishable_metrics": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": validation.get("source_identity"),
        "validation": validation,
    }


def report_payload(
    *,
    aggregate_metrics: dict[str, Any],
    benchmark_id: str,
    failure_intervals: list[Any],
    ground_truth: dict[str, Any],
    mode: str,
    per_target_metrics: dict[str, Any],
    policy: dict[str, Any],
    publishable_metrics: bool,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "aggregate_metrics": aggregate_metrics,
        "category": CATEGORY,
        "certified_scope": certified_scope(ground_truth, benchmark_id),
        "configuration_identity": validation.get("configuration_identity", {}),
        "evaluation_policy_id": policy.get("evaluation_policy_id"),
        "evaluation_policy_sha256": policy.get("content_sha256"),
        "failure_intervals": failure_intervals,
        "ground_truth_content_sha256": ground_truth.get("content_sha256"),
        "ground_truth_id": ground_truth.get("ground_truth_id"),
        "input_artifact_hashes": validation.get("input_artifact_hashes", {}),
        "limitations": limitations_for_mode(mode),
        "matching_policy_version": matching_policy_version(policy, mode),
        "mode": mode,
        "per_target_metrics": per_target_metrics,
        "provenance_summary": provenance_summary(per_target_metrics),
        "publishable_metrics": publishable_metrics,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": validation.get("source_identity"),
        "validation": validation,
    }


def limitations_for_mode(mode: str) -> list[str]:
    if mode == MODE_DETECTOR:
        return ["formal precision unavailable: insufficient_spatial_negative_ignore_map"]
    if mode == MODE_REVIEWED_FINAL:
        return ["geometric GT coverage is a pre-check, not automated pixel-level zero-leak"]
    return ["coverage is geometric GT-box coverage, not a universal scene precision score"]


def matching_policy_version(policy: dict[str, Any], mode: str) -> str:
    if mode == MODE_DETECTOR:
        return str(policy["detector_matching"]["policy_version"])
    if mode == MODE_AUTOMATIC:
        return str(policy["automatic_coverage"]["policy_version"])
    if mode == MODE_REVIEWED_FINAL:
        return str(policy["reviewed_final"]["coverage_policy_version"])
    return "not_applicable"


def provenance_summary(per_target_metrics: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for metrics in per_target_metrics.values():
        counts.update(metrics.get("provenance_breakdown") or {})
    return dict(sorted(counts.items()))


def aggregate_target_metrics(target_metrics: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    if not target_metrics:
        return {}
    if mode == MODE_DETECTOR:
        numerator = sum(int(item.get("hit_frame_count", 0)) for item in target_metrics)
        denominator = sum(int(item.get("detector_scored_frames", 0)) for item in target_metrics)
        rates = [item.get("target_frame_recall") for item in target_metrics]
        return aggregate_common(numerator, denominator, rates) | {
            "metric": "target_frame_recall",
            "total_misses": sum(int(item.get("miss_frame_count", 0)) for item in target_metrics),
            "max_consecutive_run_by_target": {
                item["target_id"]: item.get("maximum_consecutive_miss_run", 0)
                for item in target_metrics
            },
        }
    numerator = sum(int(item.get("covered_frame_count", 0)) for item in target_metrics)
    denominator = sum(int(item.get("required_visible_frames", 0)) for item in target_metrics)
    rates = [item.get("coverage_rate") for item in target_metrics]
    return aggregate_common(numerator, denominator, rates) | {
        "metric": "coverage_rate",
        "total_uncovered": sum(
            int(item.get("uncovered_frame_count", 0)) for item in target_metrics
        ),
        "max_consecutive_run_by_target": {
            item["target_id"]: item.get("longest_uncovered_run", 0) for item in target_metrics
        },
    }


def aggregate_common(
    numerator: int,
    denominator: int,
    rates: list[float | None],
) -> dict[str, Any]:
    valid_rates = [float(rate) for rate in rates if rate is not None]
    return {
        "macro_mean": round(statistics.mean(valid_rates), 6) if valid_rates else None,
        "micro_denominator": denominator,
        "micro_numerator": numerator,
        "micro_rate": ratio(numerator, denominator),
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    first = reports[0]
    for report in reports:
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported report schema")
        for key in ("ground_truth_content_sha256", "evaluation_policy_sha256", "mode"):
            if report.get(key) != first.get(key):
                raise ValueError(f"incompatible {key}")
        if report.get("configuration_identity") != first.get("configuration_identity"):
            raise ValueError("incompatible configuration identity")
    target_metrics = [
        metric
        for report in reports
        for metric in dict(report.get("per_target_metrics") or {}).values()
    ]
    return aggregate_target_metrics(target_metrics, mode=str(first["mode"])) | {
        "report_count": len(reports)
    }


def write_outputs(output_dir: Path, metrics: dict[str, Any], *, validation: dict[str, Any]) -> None:
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "validation.json", validation)
    (output_dir / "report.md").write_text(report_markdown(metrics), encoding="utf-8")


def report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Plate Evaluation Report",
        "",
        f"- Mode: `{metrics.get('mode')}`",
        f"- Publishable metrics: `{metrics.get('publishable_metrics')}`",
        f"- Validation status: `{dict(metrics.get('validation') or {}).get('status')}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in metrics.get("limitations", []))
    blockers = dict(metrics.get("validation") or {}).get("blockers") or []
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    lines.append("")
    return "\n".join(lines)


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def timed_validation_call(fn, **kwargs) -> dict[str, Any]:
    start = time.perf_counter()
    result = fn(**kwargs)
    result["elapsed_seconds"] = round(time.perf_counter() - start, 6)
    return result
