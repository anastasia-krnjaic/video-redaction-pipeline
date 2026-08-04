from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2 as cv
import license_plate_detector_bakeoff_v1 as v1
import numpy as np

from redactor import media, plate_bakeoff

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_bakeoff_v1_1"
V1_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_bakeoff_v1"
SCORE_FLOOR = 0.05
REVIEW_THRESHOLD = 0.20
THRESHOLD_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def candidates() -> list[plate_bakeoff.PlateCandidate]:
    openvino_0106 = V1_ARTIFACT_ROOT / "models" / "openvino_0106"
    intel_yolo = (
        V1_ARTIFACT_ROOT
        / "models"
        / "intel_yolov8_lpr"
        / "extracted"
        / "license-plate-reader"
        / "models"
        / "yolov8n"
    )
    return [
        plate_bakeoff.PlateCandidate(
            detector_id="intel_yolov8n_lpr_openvino",
            display_name="Intel YOLOv8n license-plate detector OpenVINO IR",
            adapter_type="openvino_yolov8",
            model_path=intel_yolo / "yolov8n_retrained.xml",
            model_bin_path=intel_yolo / "yolov8n_retrained.bin",
            threshold_grid=THRESHOLD_GRID,
            source_url="https://huggingface.co/Intel/license-plate-recognition",
            license_name=(
                "Hugging Face card has MIT tag; bundled detector metadata and "
                "third-party notices state AGPL-3.0 for YOLOv8n"
            ),
            notes=(
                "Postprocessing-audited detector-only run. The local IR output is "
                "single-class YOLOv8 detect [1, 5, 8400], not an OBB tensor."
            ),
            adapter_options={"nms_iou_threshold": 0.45},
        ),
        plate_bakeoff.PlateCandidate(
            detector_id="openvino_vehicle_license_plate_barrier_0106_fp32_tiled",
            display_name="OpenVINO Barrier 0106 FP32 whole-frame plus uniform tiles",
            adapter_type="openvino_ssd_tiled",
            model_path=openvino_0106 / "FP32" / "vehicle-license-plate-detection-barrier-0106.xml",
            model_bin_path=openvino_0106
            / "FP32"
            / "vehicle-license-plate-detection-barrier-0106.bin",
            threshold_grid=THRESHOLD_GRID,
            source_url=(
                "https://docs.openvino.ai/2023.3/"
                "omz_models_model_vehicle_license_plate_detection_barrier_0106.html"
            ),
            license_name="Open Model Zoo / Apache-2.0 project license",
            notes=(
                "Single general small-object experiment: whole frame plus 960 px "
                "uniform tiles with 25% overlap, merged by cross-tile NMS."
            ),
            adapter_options={
                "cross_tile_nms_iou_threshold": 0.45,
                "include_full_frame": True,
                "tile_overlap": 0.25,
                "tile_size": 960,
            },
        ),
    ]


def benchmarks() -> list[plate_bakeoff.PlateBenchmark]:
    official_video = ARTIFACT_ROOT / "official_reference" / "ParkingVideo.mp4"
    benchmarks = v1.benchmarks()
    if official_video.exists():
        info = media.probe_media(official_video)
        frame_count = int(info.video_stream.frame_count or 0)
        benchmarks.append(
            plate_bakeoff.PlateBenchmark(
                clip_id="official_intel_parking_sample",
                video_path=official_video,
                source_sha256=info.sha256,
                evaluated_ranges=((0, max(0, frame_count - 1)),),
                certified_quantitative=False,
                notes=(
                    "Official Intel sample video downloaded by the published "
                    "export_and_quantize.sh flow. Used for visual/reference sanity "
                    "because DLStreamer gvadetect is not installed locally."
                ),
            )
        )
    return benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    v1.ensure_package_dirs(output_dir)
    (output_dir / "official_reference").mkdir(parents=True, exist_ok=True)
    candidate_list = candidates()
    benchmark_list = benchmarks()
    v1.write_json(
        output_dir / "candidate_source_audit.json", candidate_source_audit(candidate_list)
    )
    v1.write_json(
        output_dir / "ground_truth_inventory.json", v1.ground_truth_inventory(v1.benchmarks())
    )
    v1.write_json(output_dir / "intel_adapter_audit.json", intel_adapter_audit(candidate_list[0]))
    v1.write_json(output_dir / "historical_baselines.json", v1.historical_baselines())
    v1.write_json(output_dir / "unsupported_candidates.json", v1.unsupported_candidates())

    all_run_summaries: dict[str, dict[str, Any]] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    visual_artifacts: list[dict[str, Any]] = []
    for candidate in candidate_list:
        all_run_summaries[candidate.detector_id] = {}
        all_metrics[candidate.detector_id] = {}
        candidate_benchmarks = benchmark_list
        if candidate.adapter_type == "openvino_ssd_tiled":
            candidate_benchmarks = [
                benchmark
                for benchmark in benchmark_list
                if benchmark.clip_id != "official_intel_parking_sample"
            ]
        for benchmark in candidate_benchmarks:
            detections_path = (
                output_dir
                / "detections"
                / candidate.detector_id
                / f"{benchmark.clip_id}.detections.jsonl"
            )
            summary = plate_bakeoff.run_candidate_on_clip(
                candidate,
                benchmark,
                detections_path,
                score_floor=SCORE_FLOOR,
                max_frames=args.max_frames,
            )
            all_run_summaries[candidate.detector_id][benchmark.clip_id] = summary
            records, pre_nms_records = plate_bakeoff.load_detection_jsonl_with_pre_nms(
                detections_path
            )
            metrics = plate_bakeoff.evaluate_threshold_grid(
                records,
                benchmark,
                candidate.threshold_grid,
                pre_nms_records_by_frame=pre_nms_records,
            )
            all_metrics[candidate.detector_id][benchmark.clip_id] = metrics
            v1.write_json(
                output_dir / "metrics" / candidate.detector_id / f"{benchmark.clip_id}.json",
                metrics,
            )
            if not args.skip_overlays:
                visual_artifacts.extend(
                    v1.write_visual_artifacts(
                        output_dir,
                        candidate,
                        benchmark,
                        records,
                        REVIEW_THRESHOLD,
                    )
                )

    summary = build_summary(candidate_list, benchmark_list, all_run_summaries, all_metrics)
    v1.write_json(output_dir / "run_summaries.json", all_run_summaries)
    v1.write_json(output_dir / "metrics_summary.json", summary)
    v1.write_json(output_dir / "visual_artifacts.json", visual_artifacts)
    write_csv(output_dir / "metrics_summary.csv", summary)
    write_threshold_comparison(output_dir / "threshold_comparison.csv", all_metrics)
    if not args.skip_overlays:
        write_comparison_sheets(output_dir, candidate_list, benchmark_list)
        write_official_reference_sheet(output_dir)
    v1.write_json(output_dir / "official_sample_parity.json", official_sample_parity(output_dir))
    v1.write_json(output_dir / "selection_decision.json", selection_decision(summary))
    write_report(output_dir, summary)
    v1.write_manifest(output_dir)


def candidate_source_audit(candidate_list: list[plate_bakeoff.PlateCandidate]) -> dict[str, Any]:
    audit = v1.candidate_source_audit(candidate_list)
    audit["schema_version"] = plate_bakeoff.SCHEMA_VERSION
    audit["audit_notes"] = [
        "No large new model search was performed for v1.1.",
        "Intel YOLO postprocessing was audited against local IR metadata and the "
        "published Intel/DLStreamer model card.",
        "OpenVINO Barrier 0106 was evaluated once with a general tiling mode, not "
        "with clip-specific crops or ROIs.",
    ]
    return audit


def intel_adapter_audit(candidate: plate_bakeoff.PlateCandidate) -> dict[str, Any]:
    metadata_path = candidate.model_path.parent / "metadata.yaml"
    readme_path = candidate.model_path.parents[2] / "README.md"
    licenses_path = candidate.model_path.parents[2] / "LICENSES.md"
    xml_text = candidate.model_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "adapter_root_cause": (
            "The v1 adapter did not misdecode an OBB tensor. The downloaded IR exposes "
            "a single [1, 5, 8400] YOLOv8 detect output with no angle row. The v1 "
            "implementation was incomplete because it did not preserve pre-NMS "
            "diagnostics and did not document the mismatch with the public OBB wording."
        ),
        "bundled_license_files": {
            "LICENSES.md": media.sha256_file(licenses_path),
            "metadata.yaml": media.sha256_file(metadata_path),
            "README.md": media.sha256_file(readme_path),
        },
        "class_filter": 0,
        "confidence_extraction": "single-class YOLOv8 class score row; no objectness row",
        "coordinate_order": "cx, cy, width, height in 640x640 letterboxed input coordinates",
        "input_shape": [1, 3, 640, 640],
        "letterbox_reversal": "subtract pad, divide by resize scale, clip to source frame",
        "license_status": (
            "Ambiguous for production selection: Hugging Face model-card tag is MIT, "
            "but bundled metadata/third-party notices identify YOLOv8n as AGPL-3.0."
        ),
        "local_ir_rt_info": {
            "contains_nms": "NonMaxSuppression" in xml_text,
            "contains_scale_values_255": 'scale_values value="255"' in xml_text,
            "contains_reverse_input_channels": 'reverse_input_channels value="YES"' in xml_text,
            "contains_yolo_v8_model_type": 'model_type value="yolo_v8"' in xml_text,
        },
        "nms": "axis-aligned OpenCV NMS after score filtering",
        "official_reference_limit": (
            "DLStreamer gvadetect was not installed in this environment, so exact "
            "GstAnalytics metadata parity could not be run locally."
        ),
        "orientation": {
            "public_model_card": "describes oriented bounding boxes",
            "downloaded_ir": "no angle channel present; axis-aligned redaction bboxes recorded",
            "output_shape": [1, 5, 8400],
        },
        "postprocessing_status": "partially incomplete audit trail, not a fundamental decode error",
        "score_floor": SCORE_FLOOR,
    }


def build_summary(
    candidate_list: list[plate_bakeoff.PlateCandidate],
    benchmark_list: list[plate_bakeoff.PlateBenchmark],
    all_run_summaries: dict[str, dict[str, Any]],
    all_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_summaries: dict[str, Any] = {}
    for candidate in candidate_list:
        candidate_summaries[candidate.detector_id] = {
            "clips": {
                benchmark.clip_id: selected_metric_row(
                    all_metrics[candidate.detector_id][benchmark.clip_id],
                    REVIEW_THRESHOLD,
                )
                for benchmark in benchmark_list
                if benchmark.clip_id in all_metrics[candidate.detector_id]
            },
            "run": all_run_summaries[candidate.detector_id],
            "threshold": REVIEW_THRESHOLD,
        }
    return {
        "candidate_summaries_at_review_threshold": candidate_summaries,
        "created_at": media.utc_timestamp(),
        "evaluation_notes": [
            "Range-level plate truth provides target-reference ROIs, not tight boxes.",
            "False/off-target burden is measured by detections whose centers are outside "
            "the target-reference ROI on certified intervals.",
            "The close plate and official Intel sample are visual/reference stress clips, "
            "not certified quantitative Ground Truth intervals.",
        ],
        "previous_v1_summary_path": (
            "artifacts/license_plate_detector_bakeoff_v1/metrics_summary.json"
        ),
        "review_threshold": REVIEW_THRESHOLD,
        "schema_version": plate_bakeoff.SCHEMA_VERSION,
        "threshold_grid": list(THRESHOLD_GRID),
    }


def selected_metric_row(metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    for row in metrics["thresholds"]:
        if float(row["threshold"]) == threshold:
            return row
    raise ValueError(f"threshold {threshold} not found")


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate_summary in summary[
        "candidate_summaries_at_review_threshold"
    ].items():
        for clip_id, metrics in candidate_summary["clips"].items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "certified_quantitative": metrics["certified_quantitative"],
                    "clip_id": clip_id,
                    "detected_target_frames": metrics["detected_target_frames"],
                    "duplicate_frames_after_nms": metrics["duplicate_frames_after_nms"],
                    "duplicate_frames_before_nms": metrics["duplicate_frames_before_nms"],
                    "evaluated_frame_count": metrics["evaluated_frame_count"],
                    "false_positive_frames": metrics["false_positive_frames"],
                    "longest_consecutive_miss_run": metrics["longest_consecutive_miss_run"],
                    "mean_proposals_after_nms": metrics["detections_per_frame_mean_after_nms"],
                    "mean_proposals_before_nms": metrics["detections_per_frame_mean_before_nms"],
                    "missed_frames": metrics["missed_frames"],
                    "object_frame_recall": metrics["object_frame_recall"],
                    "threshold": metrics["threshold"],
                }
            )
    v1.write_dict_csv(path, rows)


def write_threshold_comparison(path: Path, all_metrics: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for candidate_id, by_clip in all_metrics.items():
        for clip_id, metrics in by_clip.items():
            for row in metrics["thresholds"]:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "certified_quantitative": row["certified_quantitative"],
                        "clip_id": clip_id,
                        "detected_target_frames": row["detected_target_frames"],
                        "duplicate_frames_after_nms": row["duplicate_frames_after_nms"],
                        "duplicate_frames_before_nms": row["duplicate_frames_before_nms"],
                        "false_positive_frames": row["false_positive_frames"],
                        "longest_consecutive_miss_run": row["longest_consecutive_miss_run"],
                        "mean_proposals_after_nms": row["detections_per_frame_mean_after_nms"],
                        "mean_proposals_before_nms": row["detections_per_frame_mean_before_nms"],
                        "missed_frames": row["missed_frames"],
                        "object_frame_recall": row["object_frame_recall"],
                        "threshold": row["threshold"],
                    }
                )
    v1.write_dict_csv(path, rows)


def write_comparison_sheets(
    output_dir: Path,
    candidate_list: list[plate_bakeoff.PlateCandidate],
    benchmark_list: list[plate_bakeoff.PlateBenchmark],
) -> None:
    for benchmark in benchmark_list:
        if benchmark.clip_id == "official_intel_parking_sample":
            continue
        candidates_for_benchmark = [
            candidate
            for candidate in candidate_list
            if (
                output_dir
                / "detections"
                / candidate.detector_id
                / f"{benchmark.clip_id}.detections.jsonl"
            ).exists()
        ]
        records_by_candidate = {
            candidate.detector_id: plate_bakeoff.load_detection_jsonl(
                output_dir
                / "detections"
                / candidate.detector_id
                / f"{benchmark.clip_id}.detections.jsonl"
            )
            for candidate in candidates_for_benchmark
        }
        evaluated = benchmark.evaluated_frames()
        step = max(1, len(evaluated) // 6)
        selected_frames = evaluated[::step][:6]
        v1.write_candidate_comparison_sheet(
            benchmark,
            records_by_candidate,
            output_dir / "comparison_contact_sheets" / f"{benchmark.clip_id}.jpg",
            frames=selected_frames,
            threshold=REVIEW_THRESHOLD,
        )


def write_official_reference_sheet(output_dir: Path) -> None:
    gif_path = output_dir / "official_reference" / "expected_output_dlstreamer.gif"
    video_path = output_dir / "official_reference" / "ParkingVideo.mp4"
    detections_path = (
        output_dir
        / "detections"
        / "intel_yolov8n_lpr_openvino"
        / "official_intel_parking_sample.detections.jsonl"
    )
    if not gif_path.exists() or not video_path.exists() or not detections_path.exists():
        return
    records = plate_bakeoff.load_detection_jsonl(detections_path)
    video_cap = cv.VideoCapture(str(video_path))
    gif_cap = cv.VideoCapture(str(gif_path))
    if not video_cap.isOpened() or not gif_cap.isOpened():
        video_cap.release()
        gif_cap.release()
        return
    selected_frames = [0, 90, 180, 270, 360, 450]
    tile_width = 360
    tile_height = 203
    sheet = np.full((len(selected_frames) * tile_height, tile_width * 3, 3), 255, dtype=np.uint8)
    try:
        for row, frame_index in enumerate(selected_frames):
            video_cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = video_cap.read()
            if not ok:
                continue
            overlay = plate_bakeoff.draw_plate_overlay(
                frame,
                frame_index,
                records.get(frame_index, []),
                threshold=REVIEW_THRESHOLD,
            )
            gif_frame_index = min(47, round(frame_index * 8 / 30))
            gif_cap.set(cv.CAP_PROP_POS_FRAMES, gif_frame_index)
            ok_gif, gif_frame = gif_cap.read()
            if not ok_gif:
                gif_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            tiles = [
                v1.resize_labelled_tile(frame, f"official source {frame_index}", tile_width),
                v1.resize_labelled_tile(overlay, "audited adapter overlay", tile_width),
                v1.resize_labelled_tile(
                    gif_frame,
                    f"published expected gif frame {gif_frame_index}",
                    tile_width,
                ),
            ]
            for column, tile in enumerate(tiles):
                sheet[
                    row * tile_height : (row + 1) * tile_height,
                    column * tile_width : (column + 1) * tile_width,
                ] = cv.resize(tile, (tile_width, tile_height), interpolation=cv.INTER_AREA)
    finally:
        video_cap.release()
        gif_cap.release()
    output_path = output_dir / "comparison_contact_sheets" / "official_sample_reference.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), sheet)


def official_sample_parity(output_dir: Path) -> dict[str, Any]:
    sample = output_dir / "official_reference" / "ParkingVideo.mp4"
    expected = output_dir / "official_reference" / "expected_output_dlstreamer.gif"
    detections_path = (
        output_dir
        / "detections"
        / "intel_yolov8n_lpr_openvino"
        / "official_intel_parking_sample.detections.jsonl"
    )
    records = (
        plate_bakeoff.load_detection_jsonl(detections_path) if detections_path.exists() else {}
    )
    counts = [len(records.get(frame_index, [])) for frame_index in sorted(records)]
    return {
        "adapter_output_available": bool(records),
        "comparison_sheet": (
            "artifacts/license_plate_detector_bakeoff_v1_1/"
            "comparison_contact_sheets/official_sample_reference.jpg"
        ),
        "dlstreamer_gvadetect_available": False,
        "dlstreamer_parity_status": "not_run_gvadetect_plugin_unavailable",
        "expected_gif_sha256": media.sha256_file(expected) if expected.exists() else None,
        "local_adapter_result": {
            "frames_processed": len(counts),
            "frames_with_detections": sum(1 for count in counts if count),
            "mean_detections_per_frame_after_nms": round(sum(counts) / len(counts), 6)
            if counts
            else 0.0,
        },
        "official_sample_sha256": media.sha256_file(sample) if sample.exists() else None,
        "reference_statement": (
            "The adapter was run on the official Intel sample video and a contact sheet "
            "was generated beside the published expected GIF. Exact GstAnalytics "
            "metadata parity could not be executed because gvadetect is not installed."
        ),
    }


def selection_decision(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "bounded_fallback_plan": {
            "direction": (
                "Fine-tune one permissively licensed detector architecture such as "
                "RT-DETR or YOLOX on clearly licensed plate data plus project-specific "
                "annotations."
            ),
            "expected_deliverables": [
                "license-cleared training-data manifest",
                "project-specific annotation split",
                "trained detector weights kept outside Git",
                "frozen validation detections and threshold grid",
                "visual missed/false-positive review package",
            ],
            "licensing_checks": [
                "dataset redistribution and derivative-model rights",
                "model architecture and training-code license",
                "pretrained checkpoint license",
            ],
            "project_annotations": (
                "At least 300-500 tight plate boxes across day, night, small, oblique, "
                "motion-blurred, and occluded cases before trusting a first local model."
            ),
            "validation_split": "hold out at least 20% by source clip/scene, not by random frame",
        },
        "final_decision": "no_production_detector_selected",
        "reason": (
            "Intel YOLO remains too noisy after audited postprocessing, and OpenVINO "
            "Barrier 0106 remains insufficient on the day benchmark even with one "
            "general tiled small-object mode."
        ),
        "selected_detector": None,
        "selected_threshold_policy": None,
    }


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# License Plate Detector Bake-off v1.1",
        "",
        "This is a bounded correction to the v1 bake-off. It audits Intel YOLO "
        "postprocessing and runs one general OpenVINO Barrier 0106 tiled experiment.",
        "",
        "## Final Decision",
        "",
        "No production license-plate detector is selected.",
        "",
        "## Review-threshold summary",
        "",
        "| Candidate | Clip | Recall | Missed | Longest miss | FP frames | "
        "Dups pre/post NMS | Mean proposals pre/post NMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, candidate_summary in summary[
        "candidate_summaries_at_review_threshold"
    ].items():
        for clip_id, row in candidate_summary["clips"].items():
            recall = row["object_frame_recall"]
            recall_text = "n/a" if recall is None else f"{recall:.6f}"
            lines.append(
                "| "
                f"{candidate_id} | {clip_id} | {recall_text} | "
                f"{row['missed_frames']} | {row['longest_consecutive_miss_run']} | "
                f"{row['false_positive_frames']} | "
                f"{row['duplicate_frames_before_nms']}/{row['duplicate_frames_after_nms']} | "
                f"{row['detections_per_frame_mean_before_nms']:.3f}/"
                f"{row['detections_per_frame_mean_after_nms']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Adapter audit result",
            "",
            "The downloaded Intel IR has one OpenVINO output shaped `[1, 5, 8400]`, "
            "metadata `task: detect`, and no angle channel. The earlier adapter did "
            "not misdecode an OBB tensor; v1.1 records that the output is axis-aligned "
            "and preserves pre-NMS diagnostics.",
            "",
            "## 0106 tiled result",
            "",
            "The tiled mode uses whole-frame inference plus uniform 960 px tiles with "
            "25% overlap on every clip. It does not use target-specific crops.",
            "",
            "## Fallback plan",
            "",
            "The next bounded direction is to fine-tune one permissively licensed "
            "detector architecture such as RT-DETR or YOLOX on clearly licensed plate "
            "data plus project-specific annotations. Training was not started here.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
