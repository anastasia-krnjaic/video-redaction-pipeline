from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff, plate_bakeoff_v2
from redactor import media

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_bakeoff_v2"
REVIEW_THRESHOLD = 0.20


def clips() -> list[plate_bakeoff_v2.ClipSpec]:
    return [
        plate_bakeoff_v2.ClipSpec(
            clip_id="day_plate_primary",
            role="primary daytime plate benchmark",
            video_path=REPO_ROOT / "artifacts" / "eval_day_plate_close.mp4",
            source_sha256="4f19396fa1393c2861bfd26302b3147fc8ae29acfd55bf10f6e7be46a30231c9",
            evaluated_ranges=((0, 239),),
            target_id="plate_day_01",
            target_roi_xyxy=None,
            certified_quantitative=False,
            scene_description=(
                "Officer approaches/stands near vehicle; close plate with changing scale "
                "and perspective."
            ),
            notes=(
                "Primary day mapping required by bake-off v2. No validated tight boxes "
                "or reference ROI exist for this derivative clip, so target recall is "
                "reported as visual/proxy evidence rather than certified ROI recall."
            ),
        ),
        plate_bakeoff_v2.ClipSpec(
            clip_id="night_plate_primary",
            role="nighttime plate benchmark",
            video_path=REPO_ROOT / "artifacts" / "eval_night_selective.mp4",
            source_sha256="0b55aa3a497941711ac8634a049c557637e20928e8d3bd8b63e5b030b6b272e0",
            evaluated_ranges=((420, 779),),
            target_id="plate_night_01",
            target_roi_xyxy=(60, 560, 500, 900),
            certified_quantitative=True,
            scene_description="White pickup rear plate in low light with TV overlay/glare.",
            notes=(
                "GroundTruth v1.1 night plate dense interval. The ROI is a generous "
                "QA/search aid, not a tight localization box."
            ),
        ),
        plate_bakeoff_v2.ClipSpec(
            clip_id="day_plate_close_stress",
            role="driving-behind-vehicle stress clip",
            video_path=REPO_ROOT / "artifacts" / "eval_day_plate.mp4",
            source_sha256="63d41905e2b6c65a3cdc72aa73ee31668fe1c1b4fdfefa2c5688cd38c90d1097",
            evaluated_ranges=((240, 479),),
            target_id="plate_day_01",
            target_roi_xyxy=None,
            certified_quantitative=False,
            scene_description=(
                "Driving behind a vehicle; small/distant/ambiguous plate stress clip, "
                "not the primary daytime benchmark."
            ),
            notes="Supplementary qualitative stress run only.",
        ),
    ]


def candidates() -> list[plate_bakeoff_v2.CandidateSpec]:
    sources = ARTIFACT_ROOT / "sources"
    models = ARTIFACT_ROOT / "models"
    oim_source = sources / "open-image-models"
    rtdetr_source = sources / "rtdetr-v2-license-plate-detection"
    rfdetr_source = sources / "rfdetr-alpr"
    return [
        plate_bakeoff_v2.CandidateSpec(
            detector_id="open_image_models_yolov9_s_608",
            display_name="Open Image Models YOLOv9-s 608 license plate end-to-end",
            adapter_type="open_image_models_yolov9_openvino",
            model_path=models
            / "open_image_models_yolov9_s_608"
            / "yolo-v9-s-608-license-plates-end2end.onnx",
            model_files=(
                models
                / "open_image_models_yolov9_s_608"
                / "yolo-v9-s-608-license-plates-end2end.onnx",
            ),
            source_repo="https://github.com/ankandrew/open-image-models",
            source_revision=git_revision(oim_source),
            source_license="MIT",
            model_license="MIT per repository license; no separate weight notice found.",
            architecture="YOLOv9-s license-plate detector with end-to-end NMS output",
            native_input=(608, 608),
            preprocessing="Letterbox BGR input to square, convert to RGB, NCHW float32 0..1.",
            output_format="End-to-end detections [batch,x1,y1,x2,y2,class,score].",
            postprocessing="Official letterbox reversal and class filtering; no extra NMS.",
            runtime="OpenVINO CPU on the ONNX model.",
            local_source_path=oim_source,
            notes=(
                "The package README publishes license-plate model table metrics; not used as proof."
            ),
        ),
        plate_bakeoff_v2.CandidateSpec(
            detector_id="rtdetr_v2_license_plate",
            display_name="RT-DETR v2 license-plate detection",
            adapter_type="rtdetr_v2_transformers",
            model_path=rtdetr_source,
            model_files=(
                rtdetr_source / "config.json",
                rtdetr_source / "preprocessor_config.json",
                rtdetr_source / "model.safetensors",
            ),
            source_repo="https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection",
            source_revision=git_revision(rtdetr_source),
            source_license="Apache-2.0 model card",
            model_license="Apache-2.0 model card; no separate local LICENSE file in clone.",
            architecture="RT-DETR v2 R50 object detector fine-tuned for one license-plate class",
            native_input=(640, 640),
            preprocessing="Official Hugging Face image processor, resize/rescale to 640.",
            output_format="Transformers object-detection boxes, scores, labels.",
            postprocessing="Official post_process_object_detection plus axis-aligned NMS.",
            runtime="Hugging Face Transformers / PyTorch, MPS when available, CPU fallback.",
            local_source_path=rtdetr_source,
            notes="Tiled mode is run only when practical; CPU-only tiled mode is very slow.",
        ),
        plate_bakeoff_v2.CandidateSpec(
            detector_id="rfdetr_medium_alpr",
            display_name="RF-DETR Medium ALPR ONNX",
            adapter_type="rfdetr_alpr_openvino",
            model_path=rfdetr_source / "rfdetr_alpr.onnx",
            model_files=(rfdetr_source / "rfdetr_alpr.onnx",),
            source_repo="https://huggingface.co/autolane/rfdetr-alpr",
            source_revision=git_revision(rfdetr_source),
            source_license="Apache-2.0 model card",
            model_license="Apache-2.0 model card; README says same as RF-DETR.",
            architecture="RF-DETR Medium fine-tuned for license-plate detection",
            native_input=(576, 576),
            preprocessing="Resize to 576x576 RGB, ImageNet normalization, NCHW float32.",
            output_format=(
                "Normalized cxcywh boxes and raw logits; local output names are misleading."
            ),
            postprocessing="Sigmoid class score, normalized box restoration, axis-aligned NMS.",
            runtime="OpenVINO CPU on rfdetr_alpr.onnx.",
            local_source_path=rfdetr_source,
            notes="Preferred ONNX artifact from the model card.",
        ),
    ]


def inference_modes() -> list[plate_bakeoff_v2.InferenceMode]:
    return [
        plate_bakeoff_v2.InferenceMode(
            mode_id="native_whole_frame",
            display_name="Native whole-frame inference",
            tiled=False,
        ),
        plate_bakeoff_v2.InferenceMode(
            mode_id="uniform_small_object_tiles",
            display_name="Whole frame plus uniform 1280 px overlapping tiles",
            tiled=True,
            tile_size=plate_bakeoff_v2.TILE_SIZE,
            tile_overlap=plate_bakeoff_v2.TILE_OVERLAP,
            include_full_frame=True,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--skip-rtdetr-tiled-full",
        action="store_true",
        help="Record RT-DETR tiled mode as blocked instead of running the very slow full grid.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_dirs(output_dir)
    candidate_list = candidates()
    clip_list = clips()
    mode_list = inference_modes()
    write_json(output_dir / "clip_mapping_audit.json", clip_mapping_audit(clip_list))
    write_json(output_dir / "model_source_audit.json", model_source_audit(candidate_list))
    write_json(output_dir / "adapter_sanity_report.json", adapter_sanity_report(candidate_list))
    write_json(
        output_dir / "official_sample_inference.json",
        write_repository_sample_inference(output_dir, candidate_list, mode_list[0]),
    )
    smoke_results = write_smoke_tests(output_dir, candidate_list, clip_list, mode_list[0])
    write_json(output_dir / "smoke_tests.json", smoke_results)

    all_runs: dict[str, Any] = {}
    all_metrics: dict[str, Any] = {}
    visual_artifacts: list[dict[str, Any]] = []
    blocked_runs: list[dict[str, Any]] = []
    for candidate in candidate_list:
        all_runs[candidate.detector_id] = {}
        all_metrics[candidate.detector_id] = {}
        for mode in mode_list:
            if (
                args.skip_rtdetr_tiled_full
                and candidate.detector_id == "rtdetr_v2_license_plate"
                and mode.tiled
            ):
                blocked = {
                    "candidate_id": candidate.detector_id,
                    "inference_mode": mode.mode_id,
                    "reason": "Skipped by explicit operator flag due very slow tiled PyTorch run.",
                }
                blocked_runs.append(blocked)
                continue
            all_runs[candidate.detector_id][mode.mode_id] = {}
            all_metrics[candidate.detector_id][mode.mode_id] = {}
            for clip in clip_list:
                print(f"running {candidate.detector_id} {mode.mode_id} {clip.clip_id}")
                detections_path = (
                    output_dir
                    / candidate.detector_id
                    / clip.clip_id
                    / f"{mode.mode_id}.detections.jsonl"
                )
                run_summary = plate_bakeoff_v2.run_candidate_on_clip(
                    candidate,
                    clip,
                    mode,
                    detections_path,
                    max_frames=args.max_frames,
                )
                all_runs[candidate.detector_id][mode.mode_id][clip.clip_id] = run_summary
                records, pre_nms = plate_bakeoff_v2.load_records(detections_path)
                metrics = plate_bakeoff_v2.evaluate_threshold_grid(records, pre_nms, clip)
                metrics["unlocalized_frame_presence"] = frame_presence_grid(records, pre_nms, clip)
                all_metrics[candidate.detector_id][mode.mode_id][clip.clip_id] = metrics
                write_json(
                    output_dir
                    / candidate.detector_id
                    / clip.clip_id
                    / f"{mode.mode_id}.metrics.json",
                    metrics,
                )
                if not args.skip_overlays:
                    visual_artifacts.extend(
                        write_visual_artifacts(
                            output_dir,
                            candidate,
                            clip,
                            mode,
                            records,
                        )
                    )

    summary = build_summary(
        candidate_list, clip_list, mode_list, all_runs, all_metrics, blocked_runs
    )
    write_json(output_dir / "run_summaries.json", all_runs)
    write_json(output_dir / "metrics.json", all_metrics)
    write_json(output_dir / "summary_data.json", summary)
    write_json(output_dir / "visual_artifacts.json", visual_artifacts)
    write_metrics_csv(output_dir / "metrics_summary.csv", summary)
    write_threshold_csv(output_dir / "threshold_comparison.csv", all_metrics)
    if not args.skip_overlays:
        write_comparison_artifacts(output_dir, candidate_list, clip_list, summary)
    decision = selection_decision(summary)
    write_json(output_dir / "decision.json", decision)
    write_report(output_dir, summary, decision)
    write_manifest(output_dir)
    write_tracked_decision(decision, summary)


def ensure_dirs(output_dir: Path) -> None:
    for relative in [
        "comparison",
        "official_samples",
        "smoke_tests",
        "smoke_tests/contact_sheets",
        "logs",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def git_revision(path: Path) -> str:
    if not (path / ".git").exists():
        return "unknown"
    result = subprocess_run(["git", "-C", str(path), "rev-parse", "HEAD"])
    return result.strip()


def subprocess_run(command: list[str]) -> str:
    import subprocess

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def clip_mapping_audit(clip_list: list[plate_bakeoff_v2.ClipSpec]) -> dict[str, Any]:
    return {
        "clips": [clip.to_dict() for clip in clip_list],
        "created_at": media.utc_timestamp(),
        "ground_truth_conflict_note": (
            "data/evaluation_plan.yaml maps day_plate to eval_day_plate_close.mp4. "
            "data/ground_truth/README.md still contains a stale warning from the "
            "previous mapping. Bake-off v2 follows the explicit user instruction and "
            "evaluation_plan.yaml: the officer-near-vehicle clip is primary, and "
            "eval_day_plate.mp4 is qualitative driving-behind stress."
        ),
        "previous_visualization_issue": (
            "Earlier bake-off code labeled eval_day_plate.mp4/day_plate_far_gold as "
            "the quantitative day plate benchmark. That produced visuals from the "
            "driving-behind scene instead of the intended officer-near-vehicle scene."
        ),
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.clip_mapping.v1",
        "validated_before_full_candidate_evaluation": True,
        "validation": {
            "driving_behind_vehicle_clip_role": "stress/qualitative only",
            "proper_daytime_clip_used": "artifacts/eval_day_plate_close.mp4",
        },
    }


def model_source_audit(candidate_list: list[plate_bakeoff_v2.CandidateSpec]) -> dict[str, Any]:
    return {
        "candidates": [candidate.to_dict() for candidate in candidate_list],
        "created_at": media.utc_timestamp(),
        "historical_negative_evidence_preserved": [
            "data/evaluations/license_plate_detector_bakeoff_v1/",
            "artifacts/license_plate_detector_bakeoff_v1/",
            "artifacts/license_plate_detector_bakeoff_v1_1/",
            "artifacts/plate_proposal_stabilization_v1/",
        ],
        "license_notes": [
            "No legal conclusion is made; this audit records repository/model-card notices.",
            "All model weights are stored under ignored artifacts and are not tracked.",
        ],
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.source_audit.v1",
    }


def adapter_sanity_report(candidate_list: list[plate_bakeoff_v2.CandidateSpec]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for candidate in candidate_list:
        try:
            adapter = plate_bakeoff_v2.create_adapter(candidate)
            reports[candidate.detector_id] = {
                "adapter_load_status": "loaded",
                "metadata": adapter.metadata(),
            }
        except Exception as error:  # noqa: BLE001 - experimental audit must preserve blocker.
            reports[candidate.detector_id] = {
                "adapter_load_status": "failed",
                "error": str(error),
            }
    return {
        "created_at": media.utc_timestamp(),
        "reports": reports,
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.adapter_sanity.v1",
    }


def write_smoke_tests(
    output_dir: Path,
    candidate_list: list[plate_bakeoff_v2.CandidateSpec],
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    mode: plate_bakeoff_v2.InferenceMode,
) -> dict[str, Any]:
    sample_frames = {
        "day_plate_primary": [0, 80, 160, 239],
        "night_plate_primary": [420, 540, 660, 779],
        "day_plate_close_stress": [240, 320, 400, 479],
    }
    reports: dict[str, Any] = {}
    for candidate in candidate_list:
        reports[candidate.detector_id] = {}
        try:
            adapter = plate_bakeoff_v2.create_adapter(candidate, mode=mode)
        except Exception as error:  # noqa: BLE001
            reports[candidate.detector_id]["load_error"] = str(error)
            continue
        for clip in clip_list:
            frames = sample_frames[clip.clip_id]
            records: dict[int, list[plate_bakeoff.PlateDetection]] = {}
            capture = cv.VideoCapture(str(clip.video_path))
            if not capture.isOpened():
                raise ValueError(f"could not open video: {clip.video_path}")
            try:
                for frame_index in frames:
                    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    run = adapter.detect_with_diagnostics(frame)
                    records[frame_index] = [
                        item.with_frame_and_index(frame_index, index)
                        for index, item in enumerate(run.final_detections)
                    ]
            finally:
                capture.release()
            sheet_path = (
                output_dir
                / "smoke_tests"
                / "contact_sheets"
                / f"{candidate.detector_id}_{clip.clip_id}.jpg"
            )
            sheet = plate_bakeoff_v2.write_contact_sheet(
                clip,
                records,
                sheet_path,
                frames=frames,
                threshold=REVIEW_THRESHOLD,
            )
            reports[candidate.detector_id][clip.clip_id] = {
                "contact_sheet": sheet,
                "detections_by_frame": {
                    str(frame): [item.to_dict() for item in records.get(frame, [])]
                    for frame in frames
                },
                "official_or_repository_sample_status": official_sample_status(candidate),
            }
    return {
        "created_at": media.utc_timestamp(),
        "mode": mode.to_dict(),
        "reports": reports,
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.smoke.v1",
    }


def official_sample_status(candidate: plate_bakeoff_v2.CandidateSpec) -> dict[str, Any]:
    sample = official_sample_path(candidate)
    if sample is not None:
        return {
            "sample_available": sample.is_file(),
            "sample_path": plate_bakeoff.portable_path(sample) if sample.exists() else None,
            "note": (
                "Repository test asset exists; it is a generic car image, not a "
                "certified bodycam sample."
            ),
        }
    return {
        "sample_available": False,
        "sample_path": None,
        "note": "No official sample image is present in the local model repository clone.",
    }


def official_sample_path(candidate: plate_bakeoff_v2.CandidateSpec) -> Path | None:
    if candidate.detector_id == "open_image_models_yolov9_s_608":
        return (
            ARTIFACT_ROOT / "sources" / "open-image-models" / "test" / "assets" / "car_image.webp"
        )
    return None


def write_repository_sample_inference(
    output_dir: Path,
    candidate_list: list[plate_bakeoff_v2.CandidateSpec],
    mode: plate_bakeoff_v2.InferenceMode,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for candidate in candidate_list:
        status = official_sample_status(candidate)
        reports[candidate.detector_id] = {"sample": status}
        sample_path = official_sample_path(candidate)
        if sample_path is None or not sample_path.is_file():
            reports[candidate.detector_id]["inference_status"] = "not_available"
            continue
        frame = cv.imread(str(sample_path))
        if frame is None:
            reports[candidate.detector_id]["inference_status"] = "sample_decode_failed"
            continue
        try:
            adapter = plate_bakeoff_v2.create_adapter(candidate, mode=mode)
            run = adapter.detect_with_diagnostics(frame)
        except Exception as error:  # noqa: BLE001 - experimental audit must record blocker.
            reports[candidate.detector_id]["inference_status"] = "failed"
            reports[candidate.detector_id]["error"] = str(error)
            continue

        detections = [
            item.with_frame_and_index(0, index) for index, item in enumerate(run.final_detections)
        ]
        overlay = plate_bakeoff.draw_plate_overlay(
            frame,
            0,
            detections,
            threshold=REVIEW_THRESHOLD,
        )
        overlay_path = output_dir / "official_samples" / f"{candidate.detector_id}_overlay.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(overlay_path), overlay)
        reports[candidate.detector_id].update(
            {
                "detections": [item.to_dict() for item in detections],
                "diagnostics": run.diagnostics,
                "image_shape": list(frame.shape),
                "inference_mode": mode.to_dict(),
                "inference_status": "completed",
                "overlay_path": plate_bakeoff.portable_path(overlay_path),
                "overlay_sha256": media.sha256_file(overlay_path),
            }
        )
    return {
        "created_at": media.utc_timestamp(),
        "reports": reports,
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.official_sample.v1",
    }


def frame_presence_grid(
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    pre_nms: dict[int, list[plate_bakeoff.PlateDetection]],
    clip: plate_bakeoff_v2.ClipSpec,
) -> list[dict[str, Any]]:
    return [
        frame_presence_metrics(records, pre_nms, clip, threshold=threshold)
        for threshold in plate_bakeoff_v2.THRESHOLD_GRID
    ]


def frame_presence_metrics(
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    pre_nms: dict[int, list[plate_bakeoff.PlateDetection]],
    clip: plate_bakeoff_v2.ClipSpec,
    *,
    threshold: float,
) -> dict[str, Any]:
    frames = clip.evaluated_frames()
    frames_with_any_detection = []
    frames_without_any_detection = []
    total_detections = 0
    total_pre_nms = 0
    duplicate_frames = 0
    for frame_index in frames:
        detections = [item for item in records.get(frame_index, []) if item.score >= threshold]
        pre = [item for item in pre_nms.get(frame_index, []) if item.score >= threshold]
        total_detections += len(detections)
        total_pre_nms += len(pre)
        if detections:
            frames_with_any_detection.append(frame_index)
            if len(detections) > 1:
                duplicate_frames += 1
        else:
            frames_without_any_detection.append(frame_index)
    return {
        "duplicate_frames_after_nms": duplicate_frames,
        "evaluated_frame_count": len(frames),
        "frames_with_any_detection": len(frames_with_any_detection),
        "frames_without_any_detection": len(frames_without_any_detection),
        "longest_no_detection_run": plate_bakeoff.longest_run(frames_without_any_detection),
        "mean_detections_per_frame_after_nms": round(total_detections / len(frames), 6)
        if frames
        else 0.0,
        "mean_detections_per_frame_before_nms": round(total_pre_nms / len(frames), 6)
        if frames
        else 0.0,
        "presence_proxy_recall": round(len(frames_with_any_detection) / len(frames), 6)
        if frames
        else None,
        "threshold": threshold,
        "warning": (
            "This is not target recall when no target ROI/per-frame boxes are available; "
            "it only reports whether at least one plate proposal existed on a visible range."
        ),
    }


def write_visual_artifacts(
    output_dir: Path,
    candidate: plate_bakeoff_v2.CandidateSpec,
    clip: plate_bakeoff_v2.ClipSpec,
    mode: plate_bakeoff_v2.InferenceMode,
    records: dict[int, list[plate_bakeoff.PlateDetection]],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    candidate_clip_dir = output_dir / candidate.detector_id / clip.clip_id
    overlay_path = candidate_clip_dir / f"{mode.mode_id}.overlay.mp4"
    artifacts.append(
        {
            "candidate": candidate.detector_id,
            "clip_id": clip.clip_id,
            "inference_mode": mode.mode_id,
            "kind": "clean_detector_overlay",
            **plate_bakeoff_v2.write_overlay_video(
                clip,
                records,
                overlay_path,
                threshold=REVIEW_THRESHOLD,
                max_frames=180,
            ),
        }
    )
    missed = plate_bakeoff_v2.frames_missing_at_threshold(
        records,
        clip,
        threshold=REVIEW_THRESHOLD,
    )
    false_positive = plate_bakeoff_v2.frames_with_offtarget_detections(
        records,
        clip,
        threshold=REVIEW_THRESHOLD,
    )
    artifacts.append(
        {
            "candidate": candidate.detector_id,
            "clip_id": clip.clip_id,
            "inference_mode": mode.mode_id,
            "kind": "missed_frame_contact_sheet",
            **plate_bakeoff_v2.write_contact_sheet(
                clip,
                records,
                candidate_clip_dir / f"{mode.mode_id}.missed_frames.jpg",
                frames=missed,
                threshold=REVIEW_THRESHOLD,
            ),
        }
    )
    artifacts.append(
        {
            "candidate": candidate.detector_id,
            "clip_id": clip.clip_id,
            "inference_mode": mode.mode_id,
            "kind": "false_offtarget_contact_sheet",
            **plate_bakeoff_v2.write_contact_sheet(
                clip,
                records,
                candidate_clip_dir / f"{mode.mode_id}.false_offtarget.jpg",
                frames=false_positive,
                threshold=REVIEW_THRESHOLD,
            ),
        }
    )
    return artifacts


def build_summary(
    candidate_list: list[plate_bakeoff_v2.CandidateSpec],
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    mode_list: list[plate_bakeoff_v2.InferenceMode],
    all_runs: dict[str, Any],
    all_metrics: dict[str, Any],
    blocked_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_summaries: dict[str, Any] = {}
    for candidate in candidate_list:
        mode_summaries: dict[str, Any] = {}
        for mode in mode_list:
            if mode.mode_id not in all_metrics.get(candidate.detector_id, {}):
                continue
            threshold, rows = plate_bakeoff_v2.choose_candidate_threshold(
                all_metrics[candidate.detector_id][mode.mode_id]
            )
            mode_summaries[mode.mode_id] = {
                "clips": {
                    clip.clip_id: {
                        "certified_threshold_metrics": rows[clip.clip_id],
                        "unlocalized_frame_presence": selected_presence_row(
                            all_metrics[candidate.detector_id][mode.mode_id][clip.clip_id],
                            threshold,
                        ),
                    }
                    for clip in clip_list
                },
                "mode": mode.to_dict(),
                "run": all_runs[candidate.detector_id][mode.mode_id],
                "selected_threshold_for_candidate_mode": threshold,
            }
        candidate_summaries[candidate.detector_id] = {
            "candidate": candidate.to_dict(),
            "modes": mode_summaries,
            "selected_mode": select_candidate_mode(mode_summaries),
        }
    return {
        "blocked_runs": blocked_runs,
        "candidate_summaries": candidate_summaries,
        "clips": [clip.to_dict() for clip in clip_list],
        "created_at": media.utc_timestamp(),
        "metric_limitations": [
            "The corrected primary day clip lacks validated per-frame boxes or a valid ROI.",
            (
                "Day and close clip presence metrics are proposal-presence proxies, "
                "not certified target recall."
            ),
            "Night target hits use a generous search ROI; IoU is not reported.",
        ],
        "review_threshold": REVIEW_THRESHOLD,
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.summary.v1",
    }


def selected_presence_row(metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    for row in metrics["unlocalized_frame_presence"]:
        if float(row["threshold"]) == threshold:
            return row
    raise ValueError(f"missing presence threshold {threshold}")


def select_candidate_mode(mode_summaries: dict[str, Any]) -> dict[str, Any] | None:
    if not mode_summaries:
        return None
    ranked: list[tuple[tuple[float, float, float, float], str, dict[str, Any]]] = []
    for mode_id, summary in mode_summaries.items():
        clips = summary["clips"]
        night = clips.get("night_plate_primary", {})
        day = clips.get("day_plate_primary", {})
        night_row = night.get("certified_threshold_metrics", {})
        day_presence = day.get("unlocalized_frame_presence", {})
        recall = float(night_row.get("object_frame_recall") or 0.0)
        day_proxy = float(day_presence.get("presence_proxy_recall") or 0.0)
        false_burden = float(night_row.get("false_positive_frames") or 0.0)
        fps_values = [
            float(run.get("processing_fps") or 0.0) for run in summary.get("run", {}).values()
        ]
        min_fps = min(fps_values, default=0.0)
        ranked.append(((recall, day_proxy, -false_burden, min_fps), mode_id, summary))
    ranked.sort(reverse=True, key=lambda item: item[0])
    _score, mode_id, summary = ranked[0]
    return {
        "mode_id": mode_id,
        "selected_threshold": summary["selected_threshold_for_candidate_mode"],
        "selection_basis": (
            "ranked by night ROI recall, day proposal-presence proxy, burden, runtime"
        ),
    }


def selection_decision(summary: dict[str, Any]) -> dict[str, Any]:
    eligible: list[tuple[tuple[float, float, float, float], str, dict[str, Any]]] = []
    blockers: dict[str, list[str]] = {}
    for candidate_id, candidate_summary in summary["candidate_summaries"].items():
        selected = candidate_summary.get("selected_mode")
        if not selected:
            blockers[candidate_id] = ["no completed inference mode"]
            continue
        mode = candidate_summary["modes"][selected["mode_id"]]
        clips = mode["clips"]
        threshold = float(selected["selected_threshold"])
        night = clips["night_plate_primary"]["certified_threshold_metrics"]
        day_presence = clips["day_plate_primary"]["unlocalized_frame_presence"]
        night_recall = float(night.get("object_frame_recall") or 0.0)
        day_proxy = float(day_presence.get("presence_proxy_recall") or 0.0)
        night_false = int(night.get("false_positive_frames") or 0)
        night_dup = int(night.get("duplicate_frames_after_nms") or 0)
        reasons: list[str] = []
        if day_proxy < 0.90:
            reasons.append("primary day proposal-presence proxy below 0.90")
        if night_recall < 0.95:
            reasons.append("night ROI target recall below 0.95")
        if night_false > 90:
            reasons.append("night off-target false-frame burden above 90 frames")
        if night_dup > 60:
            reasons.append("night duplicate burden above 60 frames")
        if clips["day_plate_primary"]["certified_threshold_metrics"]["object_frame_recall"] is None:
            reasons.append(
                "primary day target recall is not certifiable without validated boxes/ROI"
            )
        blockers[candidate_id] = reasons
        if not reasons:
            runtime_values = [
                float(run.get("processing_fps") or 0.0) for run in mode.get("run", {}).values()
            ]
            score = (night_recall, day_proxy, -night_false, min(runtime_values, default=0.0))
            eligible.append((score, candidate_id, {"mode": mode, "threshold": threshold}))
    if not eligible:
        return {
            "blockers_by_candidate": blockers,
            "decision": "no_detector_selected",
            "reason": (
                "No candidate met all v2 acceptance criteria. In particular, the corrected "
                "primary day clip lacks certified localization truth, so a production "
                "selection would require separate reviewed plate-localization evidence."
            ),
            "selected_detector": None,
            "selected_inference_mode": None,
            "selected_threshold": None,
            "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.decision.v1",
        }
    eligible.sort(reverse=True, key=lambda item: item[0])
    _score, candidate_id, payload = eligible[0]
    return {
        "blockers_by_candidate": blockers,
        "decision": "selected_for_plate_pipeline",
        "selected_detector": candidate_id,
        "selected_inference_mode": payload["mode"]["mode"]["mode_id"],
        "selected_threshold": payload["threshold"],
        "selection_note": (
            "Selection is based on local bodycam runs, not model-card metrics. Plate tracking "
            "and rendering remain out of scope."
        ),
        "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.decision.v1",
    }


def write_comparison_artifacts(
    output_dir: Path,
    candidate_list: list[plate_bakeoff_v2.CandidateSpec],
    clip_list: list[plate_bakeoff_v2.ClipSpec],
    summary: dict[str, Any],
) -> None:
    for clip in clip_list:
        records_by_candidate: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]] = {}
        for candidate in candidate_list:
            selected = summary["candidate_summaries"][candidate.detector_id].get("selected_mode")
            if not selected:
                continue
            detections = (
                output_dir
                / candidate.detector_id
                / clip.clip_id
                / f"{selected['mode_id']}.detections.jsonl"
            )
            if detections.exists():
                records_by_candidate[candidate.detector_id] = plate_bakeoff.load_detection_jsonl(
                    detections
                )
        if not records_by_candidate:
            continue
        plate_bakeoff_v2.write_comparison_video(
            clip,
            records_by_candidate,
            output_dir / "comparison" / f"{clip.clip_id}_side_by_side.mp4",
            threshold=REVIEW_THRESHOLD,
        )
        write_candidate_comparison_sheet(
            clip,
            records_by_candidate,
            output_dir / "comparison" / f"{clip.clip_id}_contact_sheet.jpg",
            threshold=REVIEW_THRESHOLD,
        )


def write_candidate_comparison_sheet(
    clip: plate_bakeoff_v2.ClipSpec,
    records_by_candidate: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    output_path: Path,
    *,
    threshold: float,
) -> None:
    frames = clip.evaluated_frames()
    step = max(1, len(frames) // 8)
    selected_frames = frames[::step][:8]
    info = media.probe_media(clip.video_path)
    tile_width = 320
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    columns = len(records_by_candidate) + 1
    sheet = np.full(
        (len(selected_frames) * tile_height, columns * tile_width, 3), 255, dtype=np.uint8
    )
    capture = cv.VideoCapture(str(clip.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {clip.video_path}")
    try:
        for row, frame_index in enumerate(selected_frames):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            sheet[row * tile_height : (row + 1) * tile_height, 0:tile_width] = (
                plate_bakeoff_v2.labelled_tile(frame, f"original {frame_index}", tile_width)
            )
            for column, (candidate_id, records) in enumerate(records_by_candidate.items(), start=1):
                overlay = plate_bakeoff.draw_plate_overlay(
                    frame,
                    frame_index,
                    records.get(frame_index, []),
                    threshold=threshold,
                    target_roi=clip.target_roi_xyxy,
                )
                sheet[
                    row * tile_height : (row + 1) * tile_height,
                    column * tile_width : (column + 1) * tile_width,
                ] = plate_bakeoff_v2.labelled_tile(overlay, candidate_id, tile_width)
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), sheet)


def write_metrics_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate_summary in summary["candidate_summaries"].items():
        for mode_id, mode_summary in candidate_summary["modes"].items():
            threshold = mode_summary["selected_threshold_for_candidate_mode"]
            for clip_id, clip_summary in mode_summary["clips"].items():
                metrics = clip_summary["certified_threshold_metrics"]
                presence = clip_summary["unlocalized_frame_presence"]
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "certified_quantitative": metrics["certified_quantitative"],
                        "clip_id": clip_id,
                        "duplicate_frames_after_nms": metrics["duplicate_frames_after_nms"],
                        "false_positive_frames": metrics["false_positive_frames"],
                        "inference_mode": mode_id,
                        "longest_consecutive_miss_run": metrics["longest_consecutive_miss_run"],
                        "object_frame_recall": metrics["object_frame_recall"],
                        "presence_proxy_recall": presence["presence_proxy_recall"],
                        "selected_threshold": threshold,
                    }
                )
    write_dict_csv(path, rows)


def write_threshold_csv(path: Path, all_metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for candidate_id, by_mode in all_metrics.items():
        for mode_id, by_clip in by_mode.items():
            for clip_id, metrics in by_clip.items():
                for threshold_row, presence_row in zip(
                    metrics["thresholds"],
                    metrics["unlocalized_frame_presence"],
                    strict=True,
                ):
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "certified_quantitative": threshold_row["certified_quantitative"],
                            "clip_id": clip_id,
                            "detected_target_frames": threshold_row["detected_target_frames"],
                            "duplicate_frames_after_nms": threshold_row[
                                "duplicate_frames_after_nms"
                            ],
                            "false_positive_frames": threshold_row["false_positive_frames"],
                            "inference_mode": mode_id,
                            "longest_consecutive_miss_run": threshold_row[
                                "longest_consecutive_miss_run"
                            ],
                            "object_frame_recall": threshold_row["object_frame_recall"],
                            "presence_proxy_recall": presence_row["presence_proxy_recall"],
                            "threshold": threshold_row["threshold"],
                        }
                    )
    write_dict_csv(path, rows)


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, summary: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# License Plate Detector Bake-off v2",
        "",
        "This ignored package evaluates exactly three new detector candidates. It does "
        "not change the active Face Pipeline and does not implement plate tracking or rendering.",
        "",
        "## Clip Mapping",
        "",
        "| Clip | Role | Frames | Quantitative? | Notes |",
        "|---|---|---:|---|---|",
    ]
    for clip in summary["clips"]:
        lines.append(
            f"| `{clip['clip_id']}` | {clip['role']} | {clip['evaluated_frame_count']} | "
            f"{clip['certified_quantitative']} | {clip['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            (
                "| Candidate | Selected mode | Threshold | Day proposal proxy | Night recall | "
                "Night longest miss | Night false frames |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, candidate_summary in summary["candidate_summaries"].items():
        selected = candidate_summary.get("selected_mode") or {}
        if not selected:
            lines.append(f"| `{candidate_id}` | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        mode = candidate_summary["modes"][selected["mode_id"]]
        day = mode["clips"]["day_plate_primary"]["unlocalized_frame_presence"]
        night = mode["clips"]["night_plate_primary"]["certified_threshold_metrics"]
        lines.append(
            f"| `{candidate_id}` | `{selected['mode_id']}` | {selected['selected_threshold']} | "
            f"{day['presence_proxy_recall']} | {night['object_frame_recall']} | "
            f"{night['longest_consecutive_miss_run']} | {night['false_positive_frames']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision: `{decision['decision']}`",
            "",
            decision.get("reason") or decision.get("selection_note", ""),
            "",
            "## Limitations",
            "",
            (
                "- The corrected day-primary clip has no validated per-frame plate boxes "
                "or valid reference ROI."
            ),
            "- IoU and center-distance error are not certified for these plate runs.",
            "- Model-card metrics are source context only and are not selection evidence.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tracked_decision(decision: dict[str, Any], summary: dict[str, Any]) -> None:
    output_dir = REPO_ROOT / "data" / "evaluations" / "license_plate_detector_bakeoff_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# License Plate Detector Bake-off v2",
        "",
        f"Status: `{decision['decision']}`.",
        "",
        "This tracked record summarizes the bounded v2 detector-selection attempt. The "
        "active Face Pipeline was not changed, and plate tracking/rendering/review work "
        "was not started.",
        "",
        "## Candidates",
        "",
        "| Candidate | Source | License finding | Runtime |",
        "|---|---|---|---|",
    ]
    for candidate in summary["candidate_summaries"].values():
        data = candidate["candidate"]
        lines.append(
            f"| `{data['detector_id']}` | {data['source_repository']} @ "
            f"`{data['source_revision']}` | "
            f"{data['model_license']} | {data['runtime']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision.get("reason") or decision.get("selection_note", ""),
            "",
            "Ignored artifact package: `artifacts/license_plate_detector_bakeoff_v2/`.",
            "",
            "The previous v1/v1.1 and stabilization results remain historical negative evidence.",
        ]
    )
    (output_dir / "selected_detector_decision.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if relative.parts[0] in {"models", "sources"}:
            continue
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
            "schema_version": f"{plate_bakeoff_v2.SCHEMA_VERSION}.manifest.v1",
        },
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
