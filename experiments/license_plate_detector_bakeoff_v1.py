from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import media, plate_bakeoff

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_bakeoff_v1"
SCORE_FLOOR = 0.05
REVIEW_THRESHOLD = 0.20
THRESHOLD_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def candidates() -> list[plate_bakeoff.PlateCandidate]:
    openvino_0106 = ARTIFACT_ROOT / "models" / "openvino_0106"
    intel_yolo = (
        ARTIFACT_ROOT
        / "models"
        / "intel_yolov8_lpr"
        / "extracted"
        / "license-plate-reader"
        / "models"
        / "yolov8n"
    )
    return [
        plate_bakeoff.PlateCandidate(
            detector_id="openvino_vehicle_license_plate_barrier_0106_fp32",
            display_name="OpenVINO vehicle-license-plate-detection-barrier-0106 FP32",
            adapter_type="openvino_ssd",
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
                "SSD MobileNetV2 Barrier vehicle/license-plate detector; label 2 is "
                "license plate; minimum documented plate width is 96 px."
            ),
        ),
        plate_bakeoff.PlateCandidate(
            detector_id="openvino_vehicle_license_plate_barrier_0106_fp16_int8",
            display_name="OpenVINO vehicle-license-plate-detection-barrier-0106 FP16-INT8",
            adapter_type="openvino_ssd",
            model_path=openvino_0106
            / "FP16-INT8"
            / "vehicle-license-plate-detection-barrier-0106.xml",
            model_bin_path=openvino_0106
            / "FP16-INT8"
            / "vehicle-license-plate-detection-barrier-0106.bin",
            threshold_grid=THRESHOLD_GRID,
            source_url=(
                "https://docs.openvino.ai/2023.3/"
                "omz_models_model_vehicle_license_plate_detection_barrier_0106.html"
            ),
            license_name="Open Model Zoo / Apache-2.0 project license",
            notes="Quantized runtime candidate for the same Barrier 0106 detector family.",
        ),
        plate_bakeoff.PlateCandidate(
            detector_id="intel_yolov8n_lpr_openvino",
            display_name="Intel YOLOv8n license-plate detector OpenVINO IR",
            adapter_type="openvino_yolov8",
            model_path=intel_yolo / "yolov8n_retrained.xml",
            model_bin_path=intel_yolo / "yolov8n_retrained.bin",
            threshold_grid=THRESHOLD_GRID,
            source_url="https://huggingface.co/Intel/license-plate-recognition",
            license_name=(
                "Model-card MIT tag; bundled detector metadata and third-party "
                "notice list AGPL-3.0 for YOLOv8n"
            ),
            notes=(
                "Detector-only use of the package; OCR model and plate text are not used or stored."
            ),
        ),
    ]


def historical_baselines() -> list[dict[str, Any]]:
    return [
        {
            "detector_id": "opencv_lpd_yunet_2023mar",
            "display_name": "OpenCV LPD-YuNet 2023mar historical baseline",
            "existing_artifacts": [
                "artifacts/baseline/day_plate/summary.json",
                "artifacts/baseline/night_plate/summary.json",
                "artifacts/reference_check/lpd_reference_results.json",
            ],
            "license": "Apache-2.0 per OpenCV Zoo model directory",
            "model_path": "models/license_plate_detection_lpd_yunet_2023mar.onnx",
            "notes": (
                "Historical evidence shows zero detections on the important daytime "
                "close plate and sparse/false nighttime detections."
            ),
            "source_url": (
                "https://github.com/opencv/opencv_zoo/tree/main/models/"
                "license_plate_detection_yunet"
            ),
        }
    ]


def unsupported_candidates() -> list[dict[str, Any]]:
    return [
        {
            "detector_id": "openvino_vehicle_license_plate_barrier_0123",
            "display_name": "OpenVINO vehicle-license-plate-detection-barrier-0123",
            "framework": "TensorFlow public model requiring Open Model Zoo conversion",
            "license": "Apache-2.0 for original model",
            "reason_not_run": (
                "Downloaded source graph could not be converted by installed OpenVINO "
                "2026 OVC; the older omz_converter/Model Optimizer transformation path "
                "is not available in this environment."
            ),
            "source_url": (
                "https://docs.openvino.ai/2023.3/"
                "omz_models_model_vehicle_license_plate_detection_barrier_0123.html"
            ),
        }
    ]


def benchmarks() -> list[plate_bakeoff.PlateBenchmark]:
    return [
        plate_bakeoff.PlateBenchmark(
            clip_id="day_plate_far_gold",
            video_path=REPO_ROOT / "artifacts" / "eval_day_plate.mp4",
            source_sha256="63d41905e2b6c65a3cdc72aa73ee31668fe1c1b4fdfefa2c5688cd38c90d1097",
            evaluated_ranges=((240, 479),),
            target_id="plate_day_01",
            target_roi_xyxy=(650, 380, 1060, 570),
            certified_quantitative=True,
            notes=(
                "Frozen GroundTruth v1.1 day-plate dense interval. Reference ROI is a "
                "generous search aid, not a tight plate box."
            ),
        ),
        plate_bakeoff.PlateBenchmark(
            clip_id="night_selective_plate_gold",
            video_path=REPO_ROOT / "artifacts" / "eval_night_selective.mp4",
            source_sha256="0b55aa3a497941711ac8634a049c557637e20928e8d3bd8b63e5b030b6b272e0",
            evaluated_ranges=((420, 779),),
            target_id="plate_night_01",
            target_roi_xyxy=(60, 560, 500, 900),
            certified_quantitative=True,
            notes=(
                "Frozen GroundTruth v1.1 night plate dense interval. The television "
                "graphic is an occluder, not an ignore state."
            ),
        ),
        plate_bakeoff.PlateBenchmark(
            clip_id="day_plate_close_stress",
            video_path=REPO_ROOT / "artifacts" / "eval_day_plate_close.mp4",
            source_sha256="4f19396fa1393c2861bfd26302b3147fc8ae29acfd55bf10f6e7be46a30231c9",
            evaluated_ranges=((60, 359),),
            target_id="plate_day_01",
            target_roi_xyxy=None,
            certified_quantitative=False,
            notes=(
                "Current evaluation_plan.yaml close-plate stress clip. GroundTruth README "
                "warns it is not the frozen day-plate gold mapping; no certified ROI or "
                "per-frame boxes are available."
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_package_dirs(output_dir)
    candidate_list = candidates()
    benchmark_list = benchmarks()
    write_json(output_dir / "candidate_source_audit.json", candidate_source_audit(candidate_list))
    write_json(output_dir / "ground_truth_inventory.json", ground_truth_inventory(benchmark_list))
    write_json(output_dir / "unsupported_candidates.json", unsupported_candidates())
    write_json(output_dir / "historical_baselines.json", historical_baselines())

    all_run_summaries: dict[str, dict[str, Any]] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    visual_artifacts: list[dict[str, Any]] = []
    for candidate in candidate_list:
        all_run_summaries[candidate.detector_id] = {}
        all_metrics[candidate.detector_id] = {}
        for benchmark in benchmark_list:
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
            records = plate_bakeoff.load_detection_jsonl(detections_path)
            metrics = plate_bakeoff.evaluate_threshold_grid(
                records,
                benchmark,
                candidate.threshold_grid,
            )
            all_metrics[candidate.detector_id][benchmark.clip_id] = metrics
            metrics_path = (
                output_dir / "metrics" / candidate.detector_id / f"{benchmark.clip_id}.json"
            )
            write_json(metrics_path, metrics)
            if not args.skip_overlays:
                visual_artifacts.extend(
                    write_visual_artifacts(
                        output_dir,
                        candidate,
                        benchmark,
                        records,
                        REVIEW_THRESHOLD,
                    )
                )

    summary = build_summary(candidate_list, benchmark_list, all_run_summaries, all_metrics)
    write_json(output_dir / "run_summaries.json", all_run_summaries)
    write_json(output_dir / "metrics_summary.json", summary)
    write_json(output_dir / "visual_artifacts.json", visual_artifacts)
    write_csv(output_dir / "metrics_summary.csv", summary)
    write_threshold_comparison(output_dir / "threshold_comparison.csv", all_metrics)
    if not args.skip_overlays:
        write_comparison_sheets(output_dir, candidate_list, benchmark_list)
    write_report(output_dir, summary)
    write_manifest(output_dir)


def ensure_package_dirs(output_dir: Path) -> None:
    for relative in [
        "contact_sheets",
        "detections",
        "metrics",
        "overlays",
        "comparison_contact_sheets",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def candidate_source_audit(candidate_list: list[plate_bakeoff.PlateCandidate]) -> dict[str, Any]:
    return {
        "candidates": [candidate.metadata() for candidate in candidate_list],
        "created_at": media.utc_timestamp(),
        "historical_baselines": historical_baselines(),
        "schema_version": plate_bakeoff.SCHEMA_VERSION,
        "unsupported_candidates": unsupported_candidates(),
    }


def ground_truth_inventory(benchmark_list: list[plate_bakeoff.PlateBenchmark]) -> dict[str, Any]:
    return {
        "benchmarks": [benchmark.metadata() for benchmark in benchmark_list],
        "created_at": media.utc_timestamp(),
        "ground_truth_sources": [
            "data/ground_truth/GroundTruth_v1.1.md",
            "data/evaluation_plan.yaml",
            "data/ground_truth/README.md",
        ],
        "localization_truth": {
            "close_plate": "No certified per-frame boxes or target ROI.",
            "day_plate_far_gold": "Range-level truth plus generous reference ROI only.",
            "night_selective_plate_gold": "Range-level truth plus generous reference ROI only.",
        },
        "schema_version": plate_bakeoff.SCHEMA_VERSION,
    }


def write_visual_artifacts(
    output_dir: Path,
    candidate: plate_bakeoff.PlateCandidate,
    benchmark: plate_bakeoff.PlateBenchmark,
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    threshold: float,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    overlay_path = output_dir / "overlays" / f"{candidate.detector_id}_{benchmark.clip_id}.mp4"
    artifacts.append(
        {
            "candidate": candidate.detector_id,
            "clip_id": benchmark.clip_id,
            "kind": "overlay",
            **plate_bakeoff.write_overlay_video(
                benchmark,
                records,
                overlay_path,
                threshold=threshold,
            ),
        }
    )
    missed = plate_bakeoff.frames_missing_at_threshold(records, benchmark, threshold=threshold)
    false_positive = plate_bakeoff.frames_with_offtarget_detections(
        records,
        benchmark,
        threshold=threshold,
    )
    missed_path = (
        output_dir / "contact_sheets" / f"{candidate.detector_id}_{benchmark.clip_id}_missed.jpg"
    )
    artifacts.append(
        {
            "candidate": candidate.detector_id,
            "clip_id": benchmark.clip_id,
            "kind": "missed_contact_sheet",
            **plate_bakeoff.write_contact_sheet_for_frames(
                benchmark,
                records,
                missed_path,
                frames=missed,
                threshold=threshold,
            ),
        }
    )
    false_positive_path = (
        output_dir
        / "contact_sheets"
        / f"{candidate.detector_id}_{benchmark.clip_id}_false_positive.jpg"
    )
    artifacts.append(
        {
            "candidate": candidate.detector_id,
            "clip_id": benchmark.clip_id,
            "kind": "false_positive_contact_sheet",
            **plate_bakeoff.write_contact_sheet_for_frames(
                benchmark,
                records,
                false_positive_path,
                frames=false_positive,
                threshold=threshold,
            ),
        }
    )
    return artifacts


def build_summary(
    candidate_list: list[plate_bakeoff.PlateCandidate],
    benchmark_list: list[plate_bakeoff.PlateBenchmark],
    all_run_summaries: dict[str, dict[str, Any]],
    all_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_threshold = 0.20
    candidate_summaries: dict[str, Any] = {}
    for candidate in candidate_list:
        candidate_summaries[candidate.detector_id] = {
            "clips": {
                benchmark.clip_id: selected_metric_row(
                    all_metrics[candidate.detector_id][benchmark.clip_id],
                    selected_threshold,
                )
                for benchmark in benchmark_list
            },
            "run": all_run_summaries[candidate.detector_id],
            "threshold": selected_threshold,
        }
    return {
        "candidate_summaries_at_review_threshold": candidate_summaries,
        "created_at": media.utc_timestamp(),
        "evaluation_notes": [
            "Object-frame recall is computed only where frozen range truth and a target "
            "reference ROI exist.",
            "The close-plate stress clip is reported qualitatively because no certified "
            "close-clip ROI or per-frame localization boxes exist.",
            "IoU is not reported because the plate benchmarks do not contain tight "
            "per-frame boxes.",
        ],
        "historical_baselines": historical_baselines(),
        "review_threshold": selected_threshold,
        "schema_version": plate_bakeoff.SCHEMA_VERSION,
        "unsupported_candidates": unsupported_candidates(),
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
                    "duplicate_frames": metrics["duplicate_frames"],
                    "evaluated_frame_count": metrics["evaluated_frame_count"],
                    "false_positive_frames": metrics["false_positive_frames"],
                    "longest_consecutive_miss_run": metrics["longest_consecutive_miss_run"],
                    "missed_frames": metrics["missed_frames"],
                    "object_frame_recall": metrics["object_frame_recall"],
                    "threshold": metrics["threshold"],
                }
            )
    write_dict_csv(path, rows)


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
                        "duplicate_frames": row["duplicate_frames"],
                        "false_positive_frames": row["false_positive_frames"],
                        "longest_consecutive_miss_run": row["longest_consecutive_miss_run"],
                        "missed_frames": row["missed_frames"],
                        "object_frame_recall": row["object_frame_recall"],
                        "threshold": row["threshold"],
                    }
                )
    write_dict_csv(path, rows)


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_sheets(
    output_dir: Path,
    candidate_list: list[plate_bakeoff.PlateCandidate],
    benchmark_list: list[plate_bakeoff.PlateBenchmark],
) -> None:
    for benchmark in benchmark_list:
        if benchmark.clip_id == "day_plate_close_stress":
            selected_frames = [60, 120, 180, 240, 300, 359]
        else:
            evaluated = benchmark.evaluated_frames()
            step = max(1, len(evaluated) // 6)
            selected_frames = evaluated[::step][:6]
        records_by_candidate = {
            candidate.detector_id: plate_bakeoff.load_detection_jsonl(
                output_dir
                / "detections"
                / candidate.detector_id
                / f"{benchmark.clip_id}.detections.jsonl"
            )
            for candidate in candidate_list
        }
        output_path = output_dir / "comparison_contact_sheets" / f"{benchmark.clip_id}.jpg"
        write_candidate_comparison_sheet(
            benchmark,
            records_by_candidate,
            output_path,
            frames=selected_frames,
            threshold=REVIEW_THRESHOLD,
        )


def write_candidate_comparison_sheet(
    benchmark: plate_bakeoff.PlateBenchmark,
    records_by_candidate: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    output_path: Path,
    *,
    frames: list[int],
    threshold: float,
) -> None:
    info = media.probe_media(benchmark.video_path)
    tile_width = 320
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    candidates = list(records_by_candidate)
    columns = len(candidates) + 1
    rows = len(frames)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    capture = cv.VideoCapture(str(benchmark.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {benchmark.video_path}")
    try:
        for row, frame_index in enumerate(frames):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            original = resize_labelled_tile(frame, f"original frame {frame_index}", tile_width)
            sheet[row * tile_height : (row + 1) * tile_height, 0:tile_width] = original
            for column, candidate_id in enumerate(candidates, start=1):
                overlay = plate_bakeoff.draw_plate_overlay(
                    frame,
                    frame_index,
                    records_by_candidate[candidate_id].get(frame_index, []),
                    threshold=threshold,
                    target_roi=benchmark.target_roi_xyxy,
                )
                tile = resize_labelled_tile(overlay, candidate_id, tile_width)
                sheet[
                    row * tile_height : (row + 1) * tile_height,
                    column * tile_width : (column + 1) * tile_width,
                ] = tile
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), sheet)


def resize_labelled_tile(frame: np.ndarray, label: str, tile_width: int) -> np.ndarray:
    tile_height = round(tile_width * frame.shape[0] / frame.shape[1])
    resized = cv.resize(frame, (tile_width, tile_height), interpolation=cv.INTER_AREA)
    cv.rectangle(resized, (0, 0), (tile_width, 24), (0, 0, 0), -1)
    cv.putText(
        resized,
        label[:38],
        (6, 17),
        cv.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return resized


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# License Plate Detector Bake-off v1",
        "",
        "This is an experimental detector-selection package. It does not integrate "
        "license plates into the active face pipeline.",
        "",
        "## Scope",
        "",
        "- Frozen day and night plate intervals are scored with range-level truth and "
        "generous target ROIs only.",
        "- The close-plate clip is visual/qualitative because the frozen GroundTruth "
        "README says it does not have the selected day-plate gold mapping.",
        "- No plate OCR is run and no plate text is stored.",
        "",
        "## Review-threshold summary",
        "",
        "| Candidate | Clip | Recall | Missed | Longest miss | FP frames | Duplicate frames |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
                f"{row['false_positive_frames']} | {row['duplicate_frames']} |"
            )
    lines.extend(
        [
            "",
            "## Selection note",
            "",
            "The final selection is recorded in the tracked evaluation decision file. "
            "This ignored report preserves the full run outputs and visual evidence.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(output_dir).parts
        if relative_parts[0] in {"models", "smoke_run"}:
            continue
        files.append(
            {
                "path": media.portable_source_path(path, REPO_ROOT),
                "sha256": media.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(
        output_dir / "package_manifest.json",
        {
            "created_at": media.utc_timestamp(),
            "files": files,
            "schema_version": plate_bakeoff.SCHEMA_VERSION,
        },
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
