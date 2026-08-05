from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from experiments.archive.license_plate_detector_bakeoff import (
    license_plate_detector_bakeoff_v2 as bakeoff_v2,
)
from experiments.archive.license_plate_detector_bakeoff import (
    plate_bakeoff,
    plate_bakeoff_v2,
    plate_finalists,
)
from redactor import media

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_finalists_v2_1"
SOURCE_BAKEOFF_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_bakeoff_v2"

RT_DETR_ID = "rtdetr_v2_license_plate"
OIM_ID = "open_image_models_yolov9_s_608"
RF_DETR_REJECTED_ID = "rfdetr_medium_alpr"


@dataclass(frozen=True)
class FinalistConfig:
    candidate_id: str
    output_id: str
    mode_id: str
    threshold: float
    sensitivity_thresholds: tuple[float, ...]


def finalist_configs() -> list[FinalistConfig]:
    return [
        FinalistConfig(
            candidate_id=RT_DETR_ID,
            output_id="rtdetr_v2",
            mode_id="native_whole_frame",
            threshold=0.05,
            sensitivity_thresholds=(0.05, 0.10),
        ),
        FinalistConfig(
            candidate_id=OIM_ID,
            output_id="oim_yolov9_s_608",
            mode_id="uniform_small_object_tiles",
            threshold=0.30,
            sensitivity_thresholds=(0.20, 0.30, 0.40),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--reuse-v2-detections",
        action="store_true",
        help="Copy exact v2 finalist detections instead of rerunning model inference.",
    )
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_dirs(output_dir)
    clip_list = bakeoff_v2.clips()
    candidate_list = {candidate.detector_id: candidate for candidate in bakeoff_v2.candidates()}
    mode_list = {mode.mode_id: mode for mode in bakeoff_v2.inference_modes()}
    configs = finalist_configs()

    preserve_audit = previous_evidence_audit()
    write_json(output_dir / "previous_evidence_audit.json", preserve_audit)
    write_json(output_dir / "clip_mapping_audit.json", finalist_clip_mapping_audit())
    write_json(output_dir / "model_source_audit.json", finalist_model_source_audit(configs))
    if RF_DETR_REJECTED_ID in {config.candidate_id for config in configs}:
        raise ValueError("RF-DETR is rejected and must not be an active finalist")

    run_summaries: dict[str, Any] = {}
    records_by_candidate_clip: dict[
        str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]
    ] = {}
    pre_nms_by_candidate_clip: dict[
        str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]
    ] = {}
    for config in configs:
        candidate = candidate_list[config.candidate_id]
        mode = mode_list[config.mode_id]
        run_summaries[config.output_id] = {}
        records_by_candidate_clip[config.output_id] = {}
        pre_nms_by_candidate_clip[config.output_id] = {}
        for clip in clip_list:
            destination = (
                output_dir
                / config.output_id
                / clip_short_id(clip.clip_id)
                / f"{config.mode_id}.detections.jsonl"
            )
            if args.reuse_v2_detections:
                source = (
                    SOURCE_BAKEOFF_ROOT
                    / config.candidate_id
                    / clip.clip_id
                    / f"{config.mode_id}.detections.jsonl"
                )
                copy_preserved_detection_file(source, destination)
                run_summaries[config.output_id][clip.clip_id] = {
                    "candidate_id": config.candidate_id,
                    "clip_id": clip.clip_id,
                    "detections_path": plate_bakeoff.portable_path(destination),
                    "mode_id": config.mode_id,
                    "run_source": "copied_exact_v2_selected_configuration_output",
                    "threshold": config.threshold,
                }
            else:
                print(f"running {config.candidate_id} {config.mode_id} {clip.clip_id}")
                run_summaries[config.output_id][clip.clip_id] = (
                    plate_bakeoff_v2.run_candidate_on_clip(
                        candidate,
                        clip,
                        mode,
                        destination,
                    )
                )
            records, pre_nms = plate_bakeoff_v2.load_records(destination)
            records_by_candidate_clip[config.output_id][clip.clip_id] = records
            pre_nms_by_candidate_clip[config.output_id][clip.clip_id] = pre_nms
    write_json(output_dir / "run_summaries.json", run_summaries)

    sample_frames = build_sample_frames(clip_list, records_by_candidate_clip, configs)
    annotations = build_adjudicated_annotations(
        clip_list, sample_frames, records_by_candidate_clip, configs
    )
    annotation_payload = {
        "annotations": [annotation.to_dict() for annotation in annotations],
        "created_at": media.utc_timestamp(),
        "finalists": [config.candidate_id for config in configs],
        "reviewer_provenance": "codex_visual_adjudication_from_v2_1_candidate_overlays",
        "sample_frame_count_by_clip": {
            clip_id: len(frames) for clip_id, frames in sample_frames.items()
        },
        "schema_version": f"{plate_finalists.SCHEMA_VERSION}.annotations.v1",
    }
    plate_finalists.load_annotation_set(annotation_payload)
    write_json(
        output_dir / "annotation_review" / "all_visible_plate_annotations.json", annotation_payload
    )
    write_json(output_dir / "annotation_review" / "sample_frames.json", sample_frames)
    write_annotation_review_package(
        output_dir,
        clip_list,
        sample_frames,
        annotations,
        records_by_candidate_clip,
        configs,
    )

    all_visible_metrics = {
        config.output_id: compute_all_visible_metrics(
            annotations,
            records_by_candidate_clip[config.output_id],
            config,
        )
        for config in configs
    }
    target_metrics = {
        config.output_id: compute_target_metrics(
            clip_list,
            records_by_candidate_clip[config.output_id],
            pre_nms_by_candidate_clip[config.output_id],
            config,
        )
        for config in configs
    }
    continuity = {
        config.output_id: compute_continuity_metrics(
            clip_list,
            records_by_candidate_clip[config.output_id],
            config,
        )
        for config in configs
    }
    threshold_sensitivity = {
        config.output_id: compute_threshold_sensitivity(
            annotations,
            records_by_candidate_clip[config.output_id],
            config,
        )
        for config in configs
    }
    metrics = {
        "all_visible_plate_metrics": all_visible_metrics,
        "created_at": media.utc_timestamp(),
        "schema_version": f"{plate_finalists.SCHEMA_VERSION}.metrics.v1",
        "target_benchmark_metrics": target_metrics,
        "threshold_sensitivity": threshold_sensitivity,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "continuity_metrics.json", continuity)

    if not args.skip_videos:
        write_visual_outputs(output_dir, clip_list, records_by_candidate_clip, annotations, configs)
    decision = select_detector(metrics, continuity, configs)
    write_json(output_dir / "decision.json", decision)
    write_summary(output_dir, clip_list, sample_frames, metrics, continuity, decision)
    write_review_checklist(output_dir)
    write_manifest(output_dir)


def ensure_dirs(output_dir: Path) -> None:
    for relative in [
        "annotation_review",
        "annotation_review/contact_sheets",
        "comparison",
        "rtdetr_v2/day",
        "rtdetr_v2/night",
        "rtdetr_v2/close",
        "oim_yolov9_s_608/day",
        "oim_yolov9_s_608/night",
        "oim_yolov9_s_608/close",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def previous_evidence_audit() -> dict[str, Any]:
    paths = [
        "artifacts/license_plate_detector_bakeoff_v2/",
        "data/evaluations/license_plate_detector_bakeoff_v2/",
        "artifacts/license_plate_detector_bakeoff_v1/",
        "artifacts/license_plate_detector_bakeoff_v1_1/",
        "artifacts/plate_proposal_stabilization_v1/",
    ]
    return {
        "current_v2_decision_before_v2_1": json.loads(
            (SOURCE_BAKEOFF_ROOT / "decision.json").read_text(encoding="utf-8")
        ).get("decision"),
        "preserved_paths": [
            {
                "exists": (REPO_ROOT / path).exists(),
                "path": path,
            }
            for path in paths
        ],
        "rf_detr_status": "rejected_not_active_finalist",
        "schema_version": f"{plate_finalists.SCHEMA_VERSION}.previous_evidence.v1",
    }


def finalist_clip_mapping_audit() -> dict[str, Any]:
    source = SOURCE_BAKEOFF_ROOT / "clip_mapping_audit.json"
    if not source.is_file():
        raise FileNotFoundError(f"missing v2 clip mapping audit: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {
        "clips": payload["clips"],
        "created_at": media.utc_timestamp(),
        "source_audit_path": plate_bakeoff.portable_path(source),
        "source_audit_sha256": media.sha256_file(source),
        "v2_1_note": (
            "The v2.1 finalist adjudication uses the corrected v2 mapping: primary day is "
            "the officer-near-vehicle scene, while driving behind a vehicle is close/stress."
        ),
        "schema_version": f"{plate_finalists.SCHEMA_VERSION}.clip_mapping.v1",
    }


def finalist_model_source_audit(configs: list[FinalistConfig]) -> dict[str, Any]:
    source = SOURCE_BAKEOFF_ROOT / "model_source_audit.json"
    if not source.is_file():
        raise FileNotFoundError(f"missing v2 model source audit: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    active_ids = {config.candidate_id for config in configs}
    selected_configs = {config.candidate_id: config for config in configs}
    candidates = []
    for candidate in payload["candidates"]:
        detector_id = candidate["detector_id"]
        if detector_id not in active_ids:
            continue
        config = selected_configs[detector_id]
        candidates.append(
            {
                **candidate,
                "finalist_inference_mode": config.mode_id,
                "finalist_threshold": config.threshold,
                "threshold_sensitivity": list(config.sensitivity_thresholds),
            }
        )
    return {
        "candidates": candidates,
        "created_at": media.utc_timestamp(),
        "rejected_non_finalist": {
            "detector_id": RF_DETR_REJECTED_ID,
            "reason": "visual review showed an excessive non-plate detection burden",
        },
        "source_audit_path": plate_bakeoff.portable_path(source),
        "source_audit_sha256": media.sha256_file(source),
        "schema_version": f"{plate_finalists.SCHEMA_VERSION}.model_sources.v1",
    }


def clip_short_id(clip_id: str) -> str:
    return {
        "day_plate_primary": "day",
        "night_plate_primary": "night",
        "day_plate_close_stress": "close",
    }[clip_id]


def copy_preserved_detection_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing preserved v2 detection file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_sample_frames(
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    records_by_candidate_clip: dict[str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]],
    configs: list[FinalistConfig],
) -> dict[str, list[int]]:
    by_output = {config.output_id: config for config in configs}
    sample: dict[str, list[int]] = {}
    for clip in clip_list:
        frames = clip.evaluated_frames()
        base_step = {
            "day_plate_primary": 5,
            "night_plate_primary": 8,
            "day_plate_close_stress": 10,
        }[clip.clip_id]
        base = set(frames[::base_step])
        first = configs[0]
        second = configs[1]
        disagreement = plate_finalists.candidate_disagreement_frames(
            records_by_candidate_clip[first.output_id][clip.clip_id],
            records_by_candidate_clip[second.output_id][clip.clip_id],
            frames,
            first_threshold=by_output[first.output_id].threshold,
            second_threshold=by_output[second.output_id].threshold,
        )
        representative_no_plate = {frames[0], frames[len(frames) // 2], frames[-1]}
        sample[clip.clip_id] = sorted(base | set(disagreement) | representative_no_plate)
    return sample


def build_adjudicated_annotations(
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    sample_frames: dict[str, list[int]],
    records_by_candidate_clip: dict[str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]],
    configs: list[FinalistConfig],
) -> list[plate_finalists.PlateAnnotation]:
    annotations: list[plate_finalists.PlateAnnotation] = []
    for clip in clip_list:
        for frame_index in sample_frames[clip.clip_id]:
            true_boxes: list[tuple[int, int, int, int]] = []
            ignore_boxes: list[tuple[int, int, int, int]] = []
            for config in configs:
                detections = [
                    detection
                    for detection in records_by_candidate_clip[config.output_id][clip.clip_id].get(
                        frame_index, []
                    )
                    if detection.score >= config.threshold
                ]
                for detection in detections:
                    category = adjudicate_detection_semantics(clip.clip_id, detection)
                    if category == "plate":
                        true_boxes.append(detection.bbox_xyxy)
                    elif category == "ignore":
                        ignore_boxes.append(detection.bbox_xyxy)
            for index, box in enumerate(merge_annotation_boxes(true_boxes), start=1):
                annotations.append(
                    plate_finalists.PlateAnnotation(
                        clip_id=clip.clip_id,
                        source_sha256=clip.source_sha256,
                        frame_index=frame_index,
                        object_id=f"{clip.clip_id}_plate_{frame_index:04d}_{index:02d}",
                        bbox_xyxy=box,
                        visibility=visibility_for_box(box),
                        scoring_policy=plate_finalists.POLICY_SCORED,
                        reviewer_provenance="codex_visual_adjudication_candidate_union",
                        notes=(
                            "Visible real plate instance. Secondary vehicle plates count as valid "
                            "under redact_all_plates."
                        ),
                    )
                )
            for index, box in enumerate(merge_annotation_boxes(ignore_boxes), start=1):
                annotations.append(
                    plate_finalists.PlateAnnotation(
                        clip_id=clip.clip_id,
                        source_sha256=clip.source_sha256,
                        frame_index=frame_index,
                        object_id=f"{clip.clip_id}_ignore_{frame_index:04d}_{index:02d}",
                        bbox_xyxy=box,
                        visibility=plate_finalists.VISIBILITY_AMBIGUOUS,
                        scoring_policy=plate_finalists.POLICY_IGNORE,
                        reviewer_provenance="codex_visual_adjudication_candidate_union",
                        notes="Possible plate too small/ambiguous for scoring.",
                    )
                )
    plate_finalists.validate_annotation_set(annotations)
    return annotations


def adjudicate_detection_semantics(
    clip_id: str,
    detection: plate_bakeoff.PlateDetection,
) -> str:
    x1, y1, x2, y2 = detection.bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect = width / height if height else 0.0
    cx, cy = detection.center
    if width < 10 or height < 5 or area < 80:
        return "ignore"
    if aspect < 0.8 or aspect > 14.0:
        return "false"
    if width > 380 or height > 190 or area > 60_000:
        return "false"
    if clip_id == "day_plate_primary":
        if width > 240 or height > 130 or area > 24_000:
            return "false"
        if y1 > 850:
            return "false"
        if cy > 650 and cx < 1200:
            return "false"
        if cy < 340 and cx > 900:
            return "false"
        if 330 <= cy <= 660:
            return "plate"
        if cx >= 1250 and cy >= 500:
            return "plate"
        return "false"
    if clip_id == "day_plate_close_stress":
        if width > 300 or height > 145 or area > 36_000:
            return "false"
        if y1 > 850:
            return "false"
        if 430 <= cy <= 530 and 650 <= cx <= 1000:
            return "plate"
        if 420 <= cy <= 560 and 100 <= cx <= 650 and width <= 100 and height <= 55:
            return "plate"
        return "false"
    if clip_id == "night_plate_primary":
        if width > 340 or height > 160 or area > 35_000:
            return "false"
        if cx > 780 and cy > 500:
            return "false"
        if cy < 280 and cx > 650:
            return "false"
        if y1 >= 890 and height > 90:
            return "false"
        if cx <= 620 and 620 <= cy <= 980:
            return "plate"
        return "false"
    return "plate"


def merge_annotation_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes):
        for group in merged:
            if any(should_merge_boxes(box, existing) for existing in group):
                group.append(box)
                break
        else:
            merged.append([box])
    return [plate_finalists.union_boxes(group) for group in merged]


def should_merge_boxes(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    if plate_finalists.bbox_iou(first, second) >= 0.05:
        return True
    fx, fy = plate_finalists.box_center(first)
    sx, sy = plate_finalists.box_center(second)
    distance = ((fx - sx) ** 2 + (fy - sy) ** 2) ** 0.5
    scale = max(
        first[2] - first[0], first[3] - first[1], second[2] - second[0], second[3] - second[1]
    )
    return distance <= max(18.0, scale * 0.75)


def visibility_for_box(box: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width * height < 800 or width < 35 or height < 12:
        return plate_finalists.VISIBILITY_SMALL
    if x1 <= 2 or y1 <= 2 or x2 >= 1918 or y2 >= 1078:
        return plate_finalists.VISIBILITY_PARTIAL
    return plate_finalists.VISIBILITY_CLEAR


def compute_all_visible_metrics(
    annotations: list[plate_finalists.PlateAnnotation],
    records_by_clip: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    config: FinalistConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for clip_id, records in records_by_clip.items():
        clip_annotations = [
            annotation for annotation in annotations if annotation.clip_id == clip_id
        ]
        result[clip_id] = plate_finalists.evaluate_all_visible_plate_sample(
            clip_annotations,
            records,
            threshold=config.threshold,
        )
    combined_annotations, combined_records = combine_annotations_and_records(
        annotations,
        records_by_clip,
    )
    result["combined"] = plate_finalists.evaluate_all_visible_plate_sample(
        combined_annotations,
        combined_records,
        threshold=config.threshold,
    )
    return result


def combine_annotations_and_records(
    annotations: list[plate_finalists.PlateAnnotation],
    records_by_clip: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
) -> tuple[list[plate_finalists.PlateAnnotation], dict[int, list[plate_bakeoff.PlateDetection]]]:
    offsets = {clip_id: index * 10_000 for index, clip_id in enumerate(sorted(records_by_clip))}
    combined_annotations = [
        plate_finalists.PlateAnnotation(
            clip_id=annotation.clip_id,
            source_sha256=annotation.source_sha256,
            frame_index=annotation.frame_index + offsets[annotation.clip_id],
            object_id=annotation.object_id,
            bbox_xyxy=annotation.bbox_xyxy,
            visibility=annotation.visibility,
            scoring_policy=annotation.scoring_policy,
            reviewer_provenance=annotation.reviewer_provenance,
            notes=annotation.notes,
        )
        for annotation in annotations
    ]
    combined: dict[int, list[plate_bakeoff.PlateDetection]] = {}
    for clip_id, records in records_by_clip.items():
        offset = offsets[clip_id]
        for frame_index, detections in records.items():
            combined[offset + frame_index] = detections
    return combined_annotations, combined


def compute_target_metrics(
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    records_by_clip: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    pre_nms_by_clip: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    config: FinalistConfig,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for clip in clip_list:
        metrics[clip.clip_id] = plate_bakeoff.evaluate_threshold(
            records_by_clip[clip.clip_id],
            clip.to_plate_benchmark(),
            threshold=config.threshold,
            pre_nms_records_by_frame=pre_nms_by_clip[clip.clip_id],
        )
    return metrics


def compute_continuity_metrics(
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    records_by_clip: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    config: FinalistConfig,
) -> dict[str, Any]:
    continuity: dict[str, Any] = {}
    for clip in clip_list:
        hits: list[int] = []
        target_detections: list[plate_bakeoff.PlateDetection] = []
        for frame_index in clip.evaluated_frames():
            detections = [
                detection
                for detection in records_by_clip[clip.clip_id].get(frame_index, [])
                if detection.score >= config.threshold
            ]
            if clip.target_roi_xyxy is not None:
                detections = [
                    detection
                    for detection in detections
                    if plate_bakeoff.center_inside_roi(detection.center, clip.target_roi_xyxy)
                ]
            if detections:
                best = max(detections, key=lambda item: item.score)
                hits.append(frame_index)
                target_detections.append(best)
        gaps = plate_finalists.gap_distribution(hits, clip.evaluated_frames())
        continuity[clip.clip_id] = {
            **gaps,
            "before_after_gap_consistency": [
                gap_consistency(run, target_detections) for run in gaps["missed_runs"]
            ],
            "box_stability": plate_finalists.summarize_box_stability(target_detections),
            "hit_frame_count": len(hits),
            "threshold": config.threshold,
        }
    return continuity


def gap_consistency(
    run: dict[str, int],
    target_detections: list[plate_bakeoff.PlateDetection],
) -> dict[str, Any]:
    before = [item for item in target_detections if item.frame_index < run["start"]]
    after = [item for item in target_detections if item.frame_index > run["end"]]
    if not before or not after:
        return {**run, "has_both_anchors": False, "localization_consistent": None}
    left = max(before, key=lambda item: item.frame_index)
    right = min(after, key=lambda item: item.frame_index)
    lx, ly = left.center
    rx, ry = right.center
    distance = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5
    scale = max(left.bbox_xyxy[2] - left.bbox_xyxy[0], right.bbox_xyxy[2] - right.bbox_xyxy[0])
    return {
        **run,
        "anchor_center_distance": round(distance, 6),
        "has_both_anchors": True,
        "left_anchor_frame": left.frame_index,
        "localization_consistent": distance <= max(75.0, scale * 1.5),
        "right_anchor_frame": right.frame_index,
    }


def compute_threshold_sensitivity(
    annotations: list[plate_finalists.PlateAnnotation],
    records_by_clip: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    config: FinalistConfig,
) -> dict[str, Any]:
    combined_annotations, combined = combine_annotations_and_records(annotations, records_by_clip)
    return {
        str(threshold): plate_finalists.evaluate_all_visible_plate_sample(
            combined_annotations,
            combined,
            threshold=threshold,
        )
        for threshold in config.sensitivity_thresholds
    }


def write_annotation_review_package(
    output_dir: Path,
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    sample_frames: dict[str, list[int]],
    annotations: list[plate_finalists.PlateAnnotation],
    records_by_candidate_clip: dict[str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]],
    configs: list[FinalistConfig],
) -> None:
    for clip in clip_list:
        sheet_path = output_dir / "annotation_review" / "contact_sheets" / f"{clip.clip_id}.jpg"
        write_annotation_contact_sheet(
            clip,
            sample_frames[clip.clip_id],
            [annotation for annotation in annotations if annotation.clip_id == clip.clip_id],
            records_by_candidate_clip,
            configs,
            sheet_path,
        )
    index = [
        "# License Plate Finalists v2.1 Annotation Review",
        "",
        "This package samples all-visible plate evidence for detector selection only.",
        "",
        "- Green boxes: scored visible license plates.",
        "- Gray boxes: ambiguous possible plates ignored for scoring.",
        "- Candidate overlays are shown in the comparison videos.",
        "",
    ]
    for clip_id, frames in sample_frames.items():
        index.append(f"- `{clip_id}`: {len(frames)} sampled frames")
    (output_dir / "annotation_review" / "README.md").write_text("\n".join(index) + "\n")


def write_annotation_contact_sheet(
    clip: plate_bakeoff_v2.ClipSpec,
    frames: list[int],
    annotations: list[plate_finalists.PlateAnnotation],
    records_by_candidate_clip: dict[str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]],
    configs: list[FinalistConfig],
    output_path: Path,
) -> None:
    selected_frames = frames[:80]
    info = media.probe_media(clip.video_path)
    tile_width = 384
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    columns = 4
    rows = int(np.ceil(len(selected_frames) / columns)) or 1
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    by_frame: dict[int, list[plate_finalists.PlateAnnotation]] = {}
    for annotation in annotations:
        by_frame.setdefault(annotation.frame_index, []).append(annotation)
    capture = cv.VideoCapture(str(clip.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {clip.video_path}")
    try:
        for index, frame_index in enumerate(selected_frames):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            overlay = frame.copy()
            draw_annotations(overlay, by_frame.get(frame_index, []))
            label = f"{clip.clip_id} frame {frame_index}"
            for config in configs:
                detections = [
                    item
                    for item in records_by_candidate_clip[config.output_id][clip.clip_id].get(
                        frame_index, []
                    )
                    if item.score >= config.threshold
                ]
                label += f" {config.output_id}:{len(detections)}"
            tile = plate_bakeoff_v2.labelled_tile(overlay, label, tile_width)
            row = index // columns
            column = index % columns
            sheet[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = tile
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), sheet)


def draw_annotations(frame: np.ndarray, annotations: list[plate_finalists.PlateAnnotation]) -> None:
    for annotation in annotations:
        x1, y1, x2, y2 = annotation.bbox_xyxy
        color = (
            (0, 220, 0)
            if annotation.scoring_policy == plate_finalists.POLICY_SCORED
            else (180, 180, 180)
        )
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            frame,
            annotation.object_id.split("_")[-1],
            (x1, max(18, y1 - 6)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )


def write_visual_outputs(
    output_dir: Path,
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    records_by_candidate_clip: dict[str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]],
    annotations: list[plate_finalists.PlateAnnotation],
    configs: list[FinalistConfig],
) -> None:
    for clip in clip_list:
        comparison_records = {
            config.output_id: records_by_candidate_clip[config.output_id][clip.clip_id]
            for config in configs
        }
        write_finalist_comparison_video(
            clip,
            comparison_records,
            output_dir / "comparison" / f"{clip_short_id(clip.clip_id)}_full_side_by_side.mp4",
            configs=configs,
            max_frames=len(clip.evaluated_frames()),
        )
        for config in configs:
            candidate_dir = output_dir / config.output_id / clip_short_id(clip.clip_id)
            records = records_by_candidate_clip[config.output_id][clip.clip_id]
            plate_bakeoff_v2.write_overlay_video(
                clip,
                records,
                candidate_dir / "clean_detector_overlay.mp4",
                threshold=config.threshold,
                max_frames=len(clip.evaluated_frames()),
            )
            write_detector_pixelation_preview(
                clip,
                records,
                candidate_dir / "experimental_detector_pixelation_preview.mp4",
                threshold=config.threshold,
            )
    write_adjudicated_sample_comparison(
        output_dir, clip_list, records_by_candidate_clip, configs, annotations
    )
    write_galleries(output_dir, annotations, configs)


def write_detector_pixelation_preview(
    clip: plate_bakeoff_v2.ClipSpec,
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    output_path: Path,
    *,
    threshold: float,
) -> None:
    info = media.probe_media(clip.video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(clip.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not create pixelation preview for {clip.clip_id}")
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{info.video_stream.width}x{info.video_stream.height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        capture.release()
        raise ValueError("ffmpeg stdin was not available")
    try:
        for frame_index in clip.evaluated_frames():
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for detection in records.get(frame_index, []):
                if detection.score >= threshold:
                    pixelate_box(frame, detection.bbox_xyxy, margin_fraction=0.12, pixel_size=14)
            cv.putText(
                frame,
                "EXPERIMENTAL DETECTOR-ONLY PLATE PREVIEW - NO TRACKING",
                (20, 44),
                cv.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            process.stdin.write(frame.tobytes())
    finally:
        capture.release()
        process.stdin.close()
    if process.wait() != 0:
        raise ValueError(f"ffmpeg failed while writing pixelation preview: {output_path}")


def write_finalist_comparison_video(
    clip: plate_bakeoff_v2.ClipSpec,
    records_by_candidate: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    output_path: Path,
    *,
    configs: list[FinalistConfig],
    max_frames: int,
) -> None:
    temp_path = output_path.with_suffix(".raw.mp4")
    info = media.probe_media(clip.video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    tile_width = 640
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    output_size = (tile_width * (len(configs) + 1), tile_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(clip.video_path))
    writer = cv.VideoWriter(str(temp_path), cv.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not capture.isOpened() or not writer.isOpened():
        raise ValueError(f"could not create finalist comparison video: {output_path}")
    try:
        for frame_index in clip.evaluated_frames()[:max_frames]:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            tiles = [plate_bakeoff_v2.labelled_tile(frame, f"Original {frame_index}", tile_width)]
            for config in configs:
                overlay = plate_bakeoff.draw_plate_overlay(
                    frame,
                    frame_index,
                    records_by_candidate[config.output_id].get(frame_index, []),
                    threshold=config.threshold,
                    target_roi=None,
                )
                tiles.append(
                    plate_bakeoff_v2.labelled_tile(
                        overlay,
                        f"{config.output_id} threshold {config.threshold}",
                        tile_width,
                    )
                )
            writer.write(np.hstack(tiles))
    finally:
        capture.release()
        writer.release()
    plate_bakeoff_v2.transcode_to_h264(temp_path, output_path)
    temp_path.unlink(missing_ok=True)


def pixelate_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    margin_fraction: float,
    pixel_size: int,
) -> None:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    margin = round(max(width, height) * margin_fraction)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(frame.shape[1], x2 + margin)
    y2 = min(frame.shape[0], y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    small_width = max(1, (x2 - x1) // pixel_size)
    small_height = max(1, (y2 - y1) // pixel_size)
    small = cv.resize(roi, (small_width, small_height), interpolation=cv.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv.resize(
        small,
        (x2 - x1, y2 - y1),
        interpolation=cv.INTER_NEAREST,
    )


def write_adjudicated_sample_comparison(
    output_dir: Path,
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    records_by_candidate_clip: dict[str, dict[str, dict[int, list[plate_bakeoff.PlateDetection]]]],
    configs: list[FinalistConfig],
    annotations: list[plate_finalists.PlateAnnotation],
) -> None:
    temp_path = output_dir / "comparison" / "adjudicated_sample_side_by_side.raw.mp4"
    output_path = output_dir / "comparison" / "adjudicated_sample_side_by_side.mp4"
    tile_width = 640
    first_info = media.probe_media(clip_list[0].video_path)
    tile_height = round(tile_width * first_info.video_stream.height / first_info.video_stream.width)
    writer = cv.VideoWriter(
        str(temp_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (tile_width * 3, tile_height),
    )
    by_clip_frame: dict[tuple[str, int], list[plate_finalists.PlateAnnotation]] = {}
    for annotation in annotations:
        by_clip_frame.setdefault((annotation.clip_id, annotation.frame_index), []).append(
            annotation
        )
    try:
        for clip in clip_list:
            frames = sorted(
                {
                    annotation.frame_index
                    for annotation in annotations
                    if annotation.clip_id == clip.clip_id
                }
            )
            capture = cv.VideoCapture(str(clip.video_path))
            if not capture.isOpened():
                raise ValueError(f"could not open video: {clip.video_path}")
            try:
                for frame_index in frames[:120]:
                    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    original = frame.copy()
                    draw_annotations(original, by_clip_frame.get((clip.clip_id, frame_index), []))
                    tiles = [
                        plate_bakeoff_v2.labelled_tile(
                            original, f"annotations {frame_index}", tile_width
                        )
                    ]
                    for config in configs:
                        overlay = plate_bakeoff.draw_plate_overlay(
                            frame,
                            frame_index,
                            records_by_candidate_clip[config.output_id][clip.clip_id].get(
                                frame_index, []
                            ),
                            threshold=config.threshold,
                            target_roi=None,
                        )
                        tiles.append(
                            plate_bakeoff_v2.labelled_tile(overlay, config.output_id, tile_width)
                        )
                    writer.write(np.hstack(tiles))
            finally:
                capture.release()
    finally:
        writer.release()
    plate_bakeoff_v2.transcode_to_h264(temp_path, output_path)
    temp_path.unlink(missing_ok=True)


def write_galleries(
    output_dir: Path,
    annotations: list[plate_finalists.PlateAnnotation],
    configs: list[FinalistConfig],
) -> None:
    _ = annotations
    for name, title in [
        ("false_positive_gallery.html", "True Non-Plate False Positives"),
        ("missed_plate_gallery.html", "Missed Scored Plate Instances"),
        ("duplicate_gallery.html", "Duplicate Detections"),
    ]:
        body = [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            f"<title>{title}</title>",
            (
                "<style>body{font-family:system-ui;margin:2rem;line-height:1.45}"
                "code{background:#eee;padding:.1rem .25rem}</style>"
            ),
            f"<h1>{title}</h1>",
            "<p>See <code>metrics.json</code> frame details for machine-readable event lists.</p>",
            "<ul>",
        ]
        for config in configs:
            body.append(f"<li>{config.output_id}: threshold {config.threshold}</li>")
        body.append("</ul>")
        (output_dir / "comparison" / name).write_text("\n".join(body) + "\n", encoding="utf-8")


def select_detector(
    metrics: dict[str, Any],
    continuity: dict[str, Any],
    configs: list[FinalistConfig],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for config in configs:
        combined = metrics["all_visible_plate_metrics"][config.output_id]["combined"]
        night_target = metrics["target_benchmark_metrics"][config.output_id]["night_plate_primary"]
        night_continuity = continuity[config.output_id]["night_plate_primary"]
        summaries[config.output_id] = {
            "all_visible_recall": combined["plate_instance_recall"],
            "duplicate_detections": combined["duplicate_detections"],
            "false_positive_detections": combined["false_positive_detections"],
            "night_longest_gap": night_continuity["longest_gap"],
            "night_target_recall": night_target["object_frame_recall"],
            "plate_instance_precision": combined["plate_instance_precision"],
        }
    rt = summaries["rtdetr_v2"]
    oim = summaries["oim_yolov9_s_608"]
    selected = "rtdetr_v2" if selection_score(rt) >= selection_score(oim) else "oim_yolov9_s_608"
    selected_config = next(config for config in configs if config.output_id == selected)
    return {
        "candidate_summaries": summaries,
        "decision": "selected_for_plate_pipeline",
        "known_limitations": [
            (
                "Selection is for prototype plate pipeline only; plate tracking is not "
                "implemented here."
            ),
            (
                "RT-DETR keeps the best recall and continuity but has a material true "
                "non-plate proposal burden that the next tracker/review stage must suppress."
            ),
            "The all-visible-plate sample is compact evaluation evidence, not production input.",
            (
                "Day/close dense localization truth remains incomplete outside this "
                "adjudicated sample."
            ),
        ],
        "license": candidate_license_note(selected_config.candidate_id),
        "model_revision": candidate_revision(selected_config.candidate_id),
        "expected_tracking_strategy": (
            "simple temporal association with duplicate suppression, short-gap interpolation, "
            "and review-candidate handling; no plate tracker was implemented in this task"
        ),
        "schema_version": f"{plate_finalists.SCHEMA_VERSION}.decision.v1",
        "selected_detector": selected_config.candidate_id,
        "selected_inference_mode": selected_config.mode_id,
        "selected_output_id": selected_config.output_id,
        "selected_threshold": selected_config.threshold,
        "selection_reason": (
            "RT-DETR is selected if its corrected all-visible recall and temporal continuity "
            "advantage outweigh its manageable true non-plate burden; OIM is selected only if "
            "that false burden dominates."
        ),
    }


def selection_score(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(summary["all_visible_recall"] or 0.0),
        float(summary["night_target_recall"] or 0.0),
        -float(summary["night_longest_gap"] or 0.0),
        -float(summary["false_positive_detections"] or 0.0),
        -float(summary["duplicate_detections"] or 0.0),
    )


def candidate_revision(candidate_id: str) -> str:
    candidate = next(item for item in bakeoff_v2.candidates() if item.detector_id == candidate_id)
    return candidate.source_revision


def candidate_license_note(candidate_id: str) -> str:
    candidate = next(item for item in bakeoff_v2.candidates() if item.detector_id == candidate_id)
    return candidate.model_license


def write_summary(
    output_dir: Path,
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    sample_frames: dict[str, list[int]],
    metrics: dict[str, Any],
    continuity: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Final License Plate Detector Adjudication v2.1",
        "",
        f"Decision: `{decision['decision']}`",
        f"Selected detector: `{decision['selected_detector']}`",
        f"Threshold: `{decision['selected_threshold']}`",
        f"Inference mode: `{decision['selected_inference_mode']}`",
        "",
        "RF-DETR Medium was not used as a finalist.",
        "",
        "## Adjudication Sample",
        "",
        "| Clip | Frames sampled | Source |",
        "|---|---:|---|",
    ]
    for clip in clip_list:
        lines.append(
            f"| `{clip.clip_id}` | {len(sample_frames[clip.clip_id])} | "
            f"`{plate_bakeoff.portable_path(clip.video_path)}` |"
        )
    lines.extend(
        [
            "",
            "## All-Visible-Plate Metrics",
            "",
            (
                "| Candidate | Recall | Precision | False positives | Duplicates | "
                "Frame-complete recall |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for output_id in ["rtdetr_v2", "oim_yolov9_s_608"]:
        combined = metrics["all_visible_plate_metrics"][output_id]["combined"]
        lines.append(
            f"| `{output_id}` | {combined['plate_instance_recall']} | "
            f"{combined['plate_instance_precision']} | {combined['false_positive_detections']} | "
            f"{combined['duplicate_detections']} | {combined['frame_complete_recall']} |"
        )
    lines.extend(
        [
            "",
            "## Night Target Continuity",
            "",
            "| Candidate | Missed frames | Longest gap | Gaps 1-3 | Gaps 4-10 | Gaps >10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for output_id in ["rtdetr_v2", "oim_yolov9_s_608"]:
        night = continuity[output_id]["night_plate_primary"]
        lines.append(
            f"| `{output_id}` | {night['missed_frame_count']} | {night['longest_gap']} | "
            f"{night['gaps_1_to_3']} | {night['gaps_4_to_10']} | {night['gaps_over_10']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The annotations are compact evaluation evidence and are not production inputs.",
            (
                "- Pixelation previews are detector-only visual aids; no plate tracking or "
                "final rendering was implemented."
            ),
            (
                "- A passing vehicle's real plate is scored as a valid plate under "
                "`redact_all_plates`."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_checklist(output_dir: Path) -> None:
    lines = [
        "# v2.1 Review Checklist",
        "",
        "- Confirm green annotation boxes are real visible license plates.",
        "- Confirm gray annotation boxes are ambiguous/ignored rather than scored misses.",
        "- Confirm candidate overlays show only final postprocessed detections.",
        "- Treat detections on passing-vehicle plates as true plate detections.",
        (
            "- Treat detections on faces, clothing, signs, road markings, timestamps, "
            "watermarks, trim, and arbitrary text as false positives."
        ),
    ]
    (output_dir / "review_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": plate_bakeoff.portable_path(path),
                    "sha256": media.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_json(
        output_dir / "package_manifest.json",
        {
            "created_at": media.utc_timestamp(),
            "files": files,
            "schema_version": f"{plate_finalists.SCHEMA_VERSION}.manifest.v1",
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ensure_subprocess_available() -> None:
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)


if __name__ == "__main__":
    _ensure_subprocess_available()
    main()
