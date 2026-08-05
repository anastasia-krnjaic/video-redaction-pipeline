from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff

SCHEMA_VERSION = "license_plate_finalists_v2_1"

VISIBILITY_CLEAR = "clear"
VISIBILITY_PARTIAL = "partial"
VISIBILITY_SMALL = "small"
VISIBILITY_AMBIGUOUS = "ambiguous"
VALID_VISIBILITIES = {
    VISIBILITY_CLEAR,
    VISIBILITY_PARTIAL,
    VISIBILITY_SMALL,
    VISIBILITY_AMBIGUOUS,
}

POLICY_SCORED = "scored_plate"
POLICY_IGNORE = "ignore"
VALID_POLICIES = {POLICY_SCORED, POLICY_IGNORE}


@dataclass(frozen=True)
class PlateAnnotation:
    clip_id: str
    source_sha256: str
    frame_index: int
    object_id: str
    bbox_xyxy: tuple[int, int, int, int]
    visibility: str
    scoring_policy: str
    reviewer_provenance: str
    notes: str = ""

    def validate(self) -> None:
        if not self.clip_id:
            raise ValueError("annotation clip_id is required")
        if not self.source_sha256:
            raise ValueError("annotation source_sha256 is required")
        if self.frame_index < 0:
            raise ValueError("annotation frame_index cannot be negative")
        if not self.object_id:
            raise ValueError("annotation object_id is required")
        if self.visibility not in VALID_VISIBILITIES:
            raise ValueError(f"invalid plate visibility: {self.visibility}")
        if self.scoring_policy not in VALID_POLICIES:
            raise ValueError(f"invalid plate scoring policy: {self.scoring_policy}")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("annotation bbox must have positive area")
        if x1 < 0 or y1 < 0:
            raise ValueError("annotation bbox cannot start outside the frame")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "clip_id": self.clip_id,
            "frame_index": self.frame_index,
            "notes": self.notes,
            "object_id": self.object_id,
            "reviewer_provenance": self.reviewer_provenance,
            "schema_version": SCHEMA_VERSION,
            "scoring_policy": self.scoring_policy,
            "source_sha256": self.source_sha256,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlateAnnotation:
        annotation = cls(
            clip_id=str(payload["clip_id"]),
            source_sha256=str(payload["source_sha256"]),
            frame_index=int(payload["frame_index"]),
            object_id=str(payload["object_id"]),
            bbox_xyxy=tuple(int(value) for value in payload["bbox_xyxy"]),  # type: ignore[arg-type]
            visibility=str(payload["visibility"]),
            scoring_policy=str(payload["scoring_policy"]),
            reviewer_provenance=str(payload["reviewer_provenance"]),
            notes=str(payload.get("notes") or ""),
        )
        annotation.validate()
        return annotation


@dataclass(frozen=True)
class DetectionMatch:
    detection: plate_bakeoff.PlateDetection
    annotation: PlateAnnotation | None
    match_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_object_id": self.annotation.object_id if self.annotation else None,
            "bbox_xyxy": list(self.detection.bbox_xyxy),
            "detection_score": self.detection.score,
            "frame_index": self.detection.frame_index,
            "match_kind": self.match_kind,
        }


def validate_annotation_set(annotations: list[PlateAnnotation]) -> None:
    seen: set[tuple[str, int, str]] = set()
    for annotation in annotations:
        annotation.validate()
        key = (annotation.clip_id, annotation.frame_index, annotation.object_id)
        if key in seen:
            raise ValueError(f"duplicate annotation object in frame: {key}")
        seen.add(key)


def load_annotation_set(payload: dict[str, Any]) -> list[PlateAnnotation]:
    if payload.get("schema_version") != f"{SCHEMA_VERSION}.annotations.v1":
        raise ValueError("unsupported plate-finalist annotation schema")
    annotations = [PlateAnnotation.from_dict(item) for item in payload.get("annotations", [])]
    validate_annotation_set(annotations)
    return annotations


def evaluate_all_visible_plate_sample(
    annotations: list[PlateAnnotation],
    records_by_frame: dict[int, list[plate_bakeoff.PlateDetection]],
    *,
    threshold: float,
    iou_threshold: float = 0.10,
) -> dict[str, Any]:
    validate_annotation_set(annotations)
    by_frame: dict[int, list[PlateAnnotation]] = defaultdict(list)
    for annotation in annotations:
        by_frame[annotation.frame_index].append(annotation)

    scored_total = 0
    detected_total = 0
    frames_with_scored = 0
    complete_frames = 0
    false_positive_detections = 0
    false_positive_frames: set[int] = set()
    duplicate_detections = 0
    duplicate_frames: set[int] = set()
    visibility_totals: Counter[str] = Counter()
    visibility_hits: Counter[str] = Counter()
    frame_details: list[dict[str, Any]] = []

    for frame_index in sorted(by_frame):
        frame_annotations = by_frame[frame_index]
        scored = [
            annotation
            for annotation in frame_annotations
            if annotation.scoring_policy == POLICY_SCORED
        ]
        ignored = [
            annotation
            for annotation in frame_annotations
            if annotation.scoring_policy == POLICY_IGNORE
        ]
        detections = [
            detection
            for detection in records_by_frame.get(frame_index, [])
            if detection.score >= threshold
        ]
        matches = classify_frame_detections(
            detections,
            scored,
            ignored,
            iou_threshold=iou_threshold,
        )
        matched_ids = {
            match.annotation.object_id
            for match in matches
            if match.annotation is not None and match.match_kind in {"target", "duplicate"}
        }
        first_hits = {
            match.annotation.object_id
            for match in matches
            if match.annotation is not None and match.match_kind == "target"
        }
        scored_total += len(scored)
        detected_total += len(first_hits)
        frames_with_scored += int(bool(scored))
        complete_frames += int(bool(scored) and len(first_hits) == len(scored))
        for annotation in scored:
            visibility_totals[annotation.visibility] += 1
            if annotation.object_id in first_hits:
                visibility_hits[annotation.visibility] += 1
        frame_false = [match for match in matches if match.match_kind == "false_positive"]
        frame_duplicates = [match for match in matches if match.match_kind == "duplicate"]
        false_positive_detections += len(frame_false)
        duplicate_detections += len(frame_duplicates)
        if frame_false:
            false_positive_frames.add(frame_index)
        if frame_duplicates:
            duplicate_frames.add(frame_index)
        frame_details.append(
            {
                "clip_id": frame_annotations[0].clip_id,
                "detected_scored_plate_instances": len(first_hits),
                "detections": [match.to_dict() for match in matches],
                "duplicate_detections": len(frame_duplicates),
                "false_positive_detections": len(frame_false),
                "frame_complete": bool(scored) and len(first_hits) == len(scored),
                "frame_index": frame_index,
                "ignored_plate_instances": len(ignored),
                "matched_or_duplicate_annotation_ids": sorted(matched_ids),
                "scored_plate_instances": len(scored),
            }
        )

    return {
        "detected_plate_instances": detected_total,
        "detections_per_visible_plate": round(
            (detected_total + duplicate_detections) / scored_total, 6
        )
        if scored_total
        else None,
        "duplicate_detections": duplicate_detections,
        "duplicate_frames": len(duplicate_frames),
        "false_positive_detections": false_positive_detections,
        "false_positive_frames": len(false_positive_frames),
        "frame_complete_recall": round(complete_frames / frames_with_scored, 6)
        if frames_with_scored
        else None,
        "frames_with_every_visible_plate_detected": complete_frames,
        "frames_with_scored_plates": frames_with_scored,
        "missed_plate_instances": scored_total - detected_total,
        "plate_instance_precision": round(
            detected_total / (detected_total + duplicate_detections + false_positive_detections),
            6,
        )
        if detected_total + duplicate_detections + false_positive_detections
        else None,
        "plate_instance_recall": round(detected_total / scored_total, 6) if scored_total else None,
        "scored_plate_instances": scored_total,
        "threshold": threshold,
        "visibility_recall": {
            visibility: {
                "detected": visibility_hits[visibility],
                "recall": round(visibility_hits[visibility] / total, 6) if total else None,
                "total": total,
            }
            for visibility, total in sorted(visibility_totals.items())
        },
        "frame_details": frame_details,
    }


def classify_frame_detections(
    detections: list[plate_bakeoff.PlateDetection],
    scored_annotations: list[PlateAnnotation],
    ignored_annotations: list[PlateAnnotation],
    *,
    iou_threshold: float,
) -> list[DetectionMatch]:
    annotation_hits: Counter[str] = Counter()
    matches: list[DetectionMatch] = []
    for detection in plate_bakeoff.sort_plate_detections(detections):
        annotation = best_matching_annotation(detection, scored_annotations, iou_threshold)
        if annotation is not None:
            annotation_hits[annotation.object_id] += 1
            matches.append(
                DetectionMatch(
                    detection=detection,
                    annotation=annotation,
                    match_kind="target"
                    if annotation_hits[annotation.object_id] == 1
                    else "duplicate",
                )
            )
            continue
        ignored = best_matching_annotation(detection, ignored_annotations, iou_threshold)
        matches.append(
            DetectionMatch(
                detection=detection,
                annotation=ignored,
                match_kind="ignored_plate" if ignored is not None else "false_positive",
            )
        )
    return matches


def best_matching_annotation(
    detection: plate_bakeoff.PlateDetection,
    annotations: list[PlateAnnotation],
    iou_threshold: float,
) -> PlateAnnotation | None:
    best: tuple[float, PlateAnnotation] | None = None
    for annotation in annotations:
        if not detection_overlaps_annotation(detection, annotation, iou_threshold):
            continue
        score = bbox_iou(detection.bbox_xyxy, annotation.bbox_xyxy)
        if best is None or score > best[0]:
            best = (score, annotation)
    return best[1] if best else None


def detection_overlaps_annotation(
    detection: plate_bakeoff.PlateDetection,
    annotation: PlateAnnotation,
    iou_threshold: float,
) -> bool:
    return (
        bbox_iou(detection.bbox_xyxy, annotation.bbox_xyxy) >= iou_threshold
        or center_inside_box(detection.center, annotation.bbox_xyxy)
        or center_inside_box(box_center(annotation.bbox_xyxy), detection.bbox_xyxy)
    )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def center_inside_box(center: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y = center
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    if not boxes:
        raise ValueError("cannot union an empty box list")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def candidate_disagreement_frames(
    first_records: dict[int, list[plate_bakeoff.PlateDetection]],
    second_records: dict[int, list[plate_bakeoff.PlateDetection]],
    frames: list[int],
    *,
    first_threshold: float,
    second_threshold: float,
    high_confidence_threshold: float = 0.50,
) -> list[int]:
    disagreement: list[int] = []
    for frame_index in frames:
        first = [
            item for item in first_records.get(frame_index, []) if item.score >= first_threshold
        ]
        second = [
            item for item in second_records.get(frame_index, []) if item.score >= second_threshold
        ]
        first_high = any(item.score >= high_confidence_threshold for item in first)
        second_high = any(item.score >= high_confidence_threshold for item in second)
        if (
            (first_high and not second)
            or (second_high and not first)
            or abs(len(first) - len(second)) >= 2
        ):
            disagreement.append(frame_index)
    return disagreement


def gap_distribution(hit_frames: list[int], evaluated_frames: list[int]) -> dict[str, Any]:
    hit_set = set(hit_frames)
    missed = [frame for frame in evaluated_frames if frame not in hit_set]
    runs = frame_runs(missed)
    return {
        "gaps_1_to_3": sum(1 for run in runs if 1 <= run["length"] <= 3),
        "gaps_4_to_10": sum(1 for run in runs if 4 <= run["length"] <= 10),
        "gaps_over_10": sum(1 for run in runs if run["length"] > 10),
        "longest_gap": max((run["length"] for run in runs), default=0),
        "missed_frame_count": len(missed),
        "missed_runs": runs,
    }


def frame_runs(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    ordered = sorted(set(frames))
    start = previous = ordered[0]
    runs: list[dict[str, int]] = []
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        runs.append({"end": previous, "length": previous - start + 1, "start": start})
        start = previous = frame
    runs.append({"end": previous, "length": previous - start + 1, "start": start})
    return runs


def summarize_box_stability(detections: list[plate_bakeoff.PlateDetection]) -> dict[str, Any]:
    ordered = sorted(detections, key=lambda item: item.frame_index)
    if len(ordered) < 2:
        return {
            "max_center_jump": 0.0,
            "max_size_ratio": 1.0,
            "observed_frame_count": len(ordered),
        }
    max_center_jump = 0.0
    max_size_ratio = 1.0
    for previous, current in zip(ordered, ordered[1:], strict=False):
        px, py = previous.center
        cx, cy = current.center
        max_center_jump = max(max_center_jump, ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
        previous_area = bbox_area(previous.bbox_xyxy)
        current_area = bbox_area(current.bbox_xyxy)
        if previous_area > 0 and current_area > 0:
            ratio = max(previous_area / current_area, current_area / previous_area)
            max_size_ratio = max(max_size_ratio, ratio)
    return {
        "max_center_jump": round(max_center_jump, 6),
        "max_size_ratio": round(max_size_ratio, 6),
        "observed_frame_count": len(ordered),
    }


def bbox_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
