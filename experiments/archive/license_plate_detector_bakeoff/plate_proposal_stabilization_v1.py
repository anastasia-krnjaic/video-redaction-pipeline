from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import license_plate_detector_bakeoff_v1 as bakeoff_v1

from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff, plate_stabilization
from redactor import media

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BAKEOFF_ROOT = REPO_ROOT / "artifacts" / "license_plate_detector_bakeoff_v1_1"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "plate_proposal_stabilization_v1"
DETECTOR_ID = "intel_yolov8n_lpr_openvino"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = plate_stabilization.PlateStabilizationConfig()
    write_json(output_dir / "configurations.json", configuration_report(config))

    metrics: dict[str, Any] = {
        "clips": {},
        "created_at": media.utc_timestamp(),
        "detector_id": DETECTOR_ID,
        "schema_version": plate_stabilization.SCHEMA_VERSION,
        "stabilization_config": config.to_dict(),
    }
    visual_artifacts: dict[str, Any] = {}
    for benchmark in bakeoff_v1.benchmarks():
        result = run_clip(
            benchmark,
            output_dir=output_dir,
            config=config,
            skip_videos=args.skip_videos,
        )
        metrics["clips"][benchmark.clip_id] = result["metrics"]
        visual_artifacts[benchmark.clip_id] = result["visual_artifacts"]

    decision = selection_decision(metrics)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "visual_artifacts.json", visual_artifacts)
    write_json(output_dir / "decision.json", decision)
    write_summary(output_dir, metrics, decision)
    write_manifest(output_dir)


def run_clip(
    benchmark: plate_bakeoff.PlateBenchmark,
    *,
    output_dir: Path,
    config: plate_stabilization.PlateStabilizationConfig,
    skip_videos: bool,
) -> dict[str, Any]:
    clip_dir = output_dir / clip_output_name(benchmark.clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    detections_path = (
        SOURCE_BAKEOFF_ROOT / "detections" / DETECTOR_ID / f"{benchmark.clip_id}.detections.jsonl"
    )
    if not detections_path.exists():
        raise FileNotFoundError(f"missing corrected Intel YOLO proposals: {detections_path}")
    media_info = media.probe_media(benchmark.video_path)
    plate_bakeoff.validate_clip_hash(benchmark, media_info.sha256)

    post_nms_records, pre_nms_records = plate_bakeoff.load_detection_jsonl_with_pre_nms(
        detections_path
    )
    raw_metrics = plate_bakeoff.evaluate_threshold(
        post_nms_records,
        benchmark,
        threshold=config.proposal_threshold,
        pre_nms_records_by_frame=pre_nms_records,
    )
    post_nms_proposals, pre_nms_proposals = plate_stabilization.detection_records_to_proposals(
        post_nms_records,
        pre_nms_records,
    )
    stabilization = plate_stabilization.stabilize_proposals(
        post_nms_proposals,
        pre_nms_proposals_by_frame=pre_nms_proposals,
        frame_width=media_info.video_stream.width,
        frame_height=media_info.video_stream.height,
        config=config,
    )
    tracks: list[plate_stabilization.PlateTrack] = stabilization["tracks"]
    track_metrics = evaluate_tracks(
        tracks,
        benchmark=benchmark,
        processed_frames=stabilization["processed_frames"],
        elapsed_seconds=stabilization["elapsed_seconds"],
    )
    clip_metrics = {
        "benchmark": benchmark.metadata(),
        "clip_id": benchmark.clip_id,
        "configuration_hash": stable_json_hash(config.to_dict()),
        "geometry_rejection_reasons": stabilization["rejection_reason_counts"],
        "post_geometry_proposal_count": sum(
            len(items) for items in stabilization["filtered_proposals_by_frame"].values()
        ),
        "pre_nms_proposal_count": sum(len(items) for items in pre_nms_records.values()),
        "raw_detector_metrics": raw_metrics,
        "schema_version": plate_stabilization.SCHEMA_VERSION,
        "source_detection_path": media.portable_source_path(detections_path, REPO_ROOT),
        "source_detection_sha256": media.sha256_file(detections_path),
        "stabilized_track_metrics": track_metrics,
    }
    plate_stabilization.write_track_summary_json(
        tracks,
        clip_dir / "track_summary.json",
        config=config,
        extra={
            "benchmark": benchmark.metadata(),
            "metrics": track_metrics,
            "source_detection_path": media.portable_source_path(detections_path, REPO_ROOT),
            "source_detection_sha256": media.sha256_file(detections_path),
        },
    )
    plate_stabilization.write_tracks_jsonl(tracks, clip_dir / "tracks.jsonl")
    plate_stabilization.write_rejected_proposals_jsonl(
        stabilization["rejected_proposals_by_frame"],
        clip_dir / "rejected_proposals.jsonl",
    )

    visual_artifacts: dict[str, Any] = {}
    if not skip_videos:
        visual_artifacts.update(
            write_visual_outputs(
                benchmark,
                clip_dir=clip_dir,
                post_nms_records=post_nms_records,
                tracks=tracks,
                config=config,
            )
        )
    else:
        gallery_dir = clip_dir / "track_gallery"
        visual_artifacts["track_gallery"] = plate_stabilization.write_track_gallery(
            benchmark.video_path,
            tracks,
            gallery_dir,
            title=f"{benchmark.clip_id} plate tracks",
        )
        maybe_copy_gallery_index(gallery_dir, clip_dir / "track_gallery.html")

    return {"metrics": clip_metrics, "visual_artifacts": visual_artifacts}


def write_visual_outputs(
    benchmark: plate_bakeoff.PlateBenchmark,
    *,
    clip_dir: Path,
    post_nms_records: dict[int, list[plate_bakeoff.PlateDetection]],
    tracks: list[plate_stabilization.PlateTrack],
    config: plate_stabilization.PlateStabilizationConfig,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if benchmark.clip_id != "day_plate_close_stress":
        artifacts["raw_proposals_overlay"] = write_raw_proposals_overlay_clip(
            benchmark,
            post_nms_records,
            clip_dir / "raw_proposals_overlay.mp4",
            threshold=config.proposal_threshold,
        )
    artifacts["stabilized_tracks_overlay"] = write_stabilized_tracks_overlay_clip(
        benchmark,
        tracks,
        clip_dir / "stabilized_tracks_overlay.mp4",
        max_tracks_drawn=config.max_tracks_drawn,
    )
    gallery_dir = clip_dir / "track_gallery"
    artifacts["track_gallery"] = plate_stabilization.write_track_gallery(
        benchmark.video_path,
        tracks,
        gallery_dir,
        title=f"{benchmark.clip_id} plate tracks",
    )
    maybe_copy_gallery_index(gallery_dir, clip_dir / "track_gallery.html")
    if benchmark.clip_id != "day_plate_close_stress":
        missed_frames = missed_target_frames(tracks, benchmark)
        review_frames = review_candidate_frames(tracks, benchmark)
        artifacts["missed_frames"] = plate_stabilization.write_track_contact_sheet(
            benchmark.video_path,
            tracks,
            clip_dir / "missed_frames.jpg",
            frames=missed_frames,
            target_roi=benchmark.target_roi_xyxy,
        )
        artifacts["review_candidates"] = plate_stabilization.write_track_contact_sheet(
            benchmark.video_path,
            tracks,
            clip_dir / "review_candidates.jpg",
            frames=review_frames,
            target_roi=benchmark.target_roi_xyxy,
        )
    return artifacts


def write_raw_proposals_overlay_clip(
    benchmark: plate_bakeoff.PlateBenchmark,
    records_by_frame: dict[int, list[plate_bakeoff.PlateDetection]],
    output_path: Path,
    *,
    threshold: float,
) -> dict[str, Any]:
    info = media.probe_media(benchmark.video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    writer, capture = open_clip_writer(benchmark.video_path, output_path, info)
    frames = benchmark.evaluated_frames()
    try:
        for frame_index, frame in iter_video_range_frames(
            capture,
            benchmark.evaluated_ranges,
            video_path=benchmark.video_path,
        ):
            writer.write(
                draw_raw_proposal_overlay_fast(
                    frame,
                    frame_index,
                    records_by_frame.get(frame_index, []),
                    threshold=threshold,
                    target_roi=benchmark.target_roi_xyxy,
                )
            )
    finally:
        capture.release()
        writer.release()
    return {
        "clip_frame_count": len(frames),
        "fps": fps,
        "output_path": media.portable_source_path(output_path, REPO_ROOT),
        "output_sha256": media.sha256_file(output_path),
        "source_frame_end": frames[-1] if frames else None,
        "source_frame_start": frames[0] if frames else None,
        "threshold": threshold,
    }


def draw_raw_proposal_overlay_fast(
    frame: Any,
    frame_index: int,
    detections: list[plate_bakeoff.PlateDetection],
    *,
    threshold: float,
    target_roi: tuple[int, int, int, int] | None,
    max_boxes: int = 60,
) -> Any:
    output = frame.copy()
    if target_roi is not None:
        x1, y1, x2, y2 = target_roi
        cv.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
    selected = sorted(
        [item for item in detections if item.score >= threshold],
        key=lambda item: -item.score,
    )
    for detection in selected[:max_boxes]:
        color = (
            (0, 255, 0)
            if target_roi is not None
            and plate_bakeoff.center_inside_roi(detection.center, target_roi)
            else (0, 180, 255)
        )
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv.rectangle(output, (x1, y1), (x2, y2), color, 1)
    cv.rectangle(output, (0, 0), (760, 42), (0, 0, 0), -1)
    cv.putText(
        output,
        f"frame {frame_index} raw proposals >= {threshold:.2f}: "
        f"{len(selected)} (showing {min(max_boxes, len(selected))})",
        (12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return output


def write_stabilized_tracks_overlay_clip(
    benchmark: plate_bakeoff.PlateBenchmark,
    tracks: list[plate_stabilization.PlateTrack],
    output_path: Path,
    *,
    max_tracks_drawn: int,
) -> dict[str, Any]:
    info = media.probe_media(benchmark.video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    writer, capture = open_clip_writer(benchmark.video_path, output_path, info)
    frames = benchmark.evaluated_frames()
    frame_entries = plate_stabilization.track_frame_index(tracks)
    try:
        for frame_index, frame in iter_video_range_frames(
            capture,
            benchmark.evaluated_ranges,
            video_path=benchmark.video_path,
        ):
            writer.write(
                plate_stabilization.draw_track_overlay(
                    frame,
                    frame_index,
                    tracks,
                    track_entries=frame_entries.get(frame_index, []),
                    target_roi=benchmark.target_roi_xyxy,
                    max_tracks_drawn=max_tracks_drawn,
                )
            )
    finally:
        capture.release()
        writer.release()
    return {
        "clip_frame_count": len(frames),
        "fps": fps,
        "output_path": media.portable_source_path(output_path, REPO_ROOT),
        "output_sha256": media.sha256_file(output_path),
        "source_frame_end": frames[-1] if frames else None,
        "source_frame_start": frames[0] if frames else None,
    }


def open_clip_writer(
    video_path: Path,
    output_path: Path,
    info: Any,
) -> tuple[cv.VideoWriter, cv.VideoCapture]:
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info.video_stream.width, info.video_stream.height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create overlay video: {output_path}")
    return writer, capture


def iter_video_range_frames(
    capture: cv.VideoCapture,
    ranges: tuple[tuple[int, int], ...],
    *,
    video_path: Path,
) -> Any:
    for start, end in ranges:
        capture.set(cv.CAP_PROP_POS_FRAMES, start)
        for frame_index in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not read frame {frame_index} from {video_path}")
            yield frame_index, frame


def evaluate_tracks(
    tracks: list[plate_stabilization.PlateTrack],
    *,
    benchmark: plate_bakeoff.PlateBenchmark,
    processed_frames: list[int],
    elapsed_seconds: float,
) -> dict[str, Any]:
    status_counts = plate_stabilization.status_counts(tracks)
    active_counts = active_track_counts_by_frame(
        tracks,
        statuses={"accepted_plate_track", "review_candidate"},
    )
    quantitative = benchmark.certified_quantitative and benchmark.target_roi_xyxy is not None
    metrics: dict[str, Any] = {
        "accepted_tracks": status_counts.get("accepted_plate_track", 0),
        "average_tracks_active_per_frame": average_active_tracks(
            active_counts,
            frames=processed_frames,
        ),
        "certified_quantitative": quantitative,
        "processing_fps": (
            round(len(processed_frames) / elapsed_seconds, 6) if elapsed_seconds > 0 else None
        ),
        "processed_frame_count": len(processed_frames),
        "rejected_tracks": status_counts.get("rejected_noise_track", 0),
        "review_candidate_tracks": status_counts.get("review_candidate", 0),
        "track_count": len(tracks),
        "track_status_counts": status_counts,
        "unscored_reason": None
        if quantitative
        else "no certified plate ROI or per-frame localization truth",
    }
    if not quantitative:
        metrics["visual_stability_summary"] = visual_stability_summary(tracks)
        return metrics

    evaluated_frames = benchmark.evaluated_frames()
    target_covered = covered_target_frames(
        tracks,
        benchmark,
        statuses={"accepted_plate_track", "review_candidate"},
    )
    accepted_target = covered_target_frames(
        tracks,
        benchmark,
        statuses={"accepted_plate_track"},
    )
    review_target = covered_target_frames(
        tracks,
        benchmark,
        statuses={"review_candidate"},
    )
    uncovered = sorted(set(evaluated_frames) - target_covered)
    metrics.update(
        {
            "accepted_or_review_target_frames_covered": len(target_covered),
            "accepted_target_frames_covered": len(accepted_target),
            "false_accepted_track_frame_count": false_track_frame_count(
                tracks,
                benchmark,
                statuses={"accepted_plate_track"},
            ),
            "false_review_track_frame_count": false_track_frame_count(
                tracks,
                benchmark,
                statuses={"review_candidate"},
            ),
            "longest_uncovered_visible_run": plate_bakeoff.longest_run(uncovered),
            "review_target_frames_covered": len(review_target),
            "stabilized_target_frame_recall": round(
                len(target_covered) / len(evaluated_frames),
                6,
            ),
            "target_evaluated_frame_count": len(evaluated_frames),
            "uncovered_visible_frames": len(uncovered),
        }
    )
    return metrics


def covered_target_frames(
    tracks: list[plate_stabilization.PlateTrack],
    benchmark: plate_bakeoff.PlateBenchmark,
    *,
    statuses: set[str],
) -> set[int]:
    if benchmark.target_roi_xyxy is None:
        return set()
    evaluated = set(benchmark.evaluated_frames())
    covered: set[int] = set()
    for track in tracks:
        if track.status not in statuses:
            continue
        for observation in track.sorted_observations():
            if observation.frame_index in evaluated and plate_bakeoff.center_inside_roi(
                observation.center, benchmark.target_roi_xyxy
            ):
                covered.add(observation.frame_index)
    return covered


def false_track_frame_count(
    tracks: list[plate_stabilization.PlateTrack],
    benchmark: plate_bakeoff.PlateBenchmark,
    *,
    statuses: set[str],
) -> int:
    if benchmark.target_roi_xyxy is None:
        return 0
    evaluated = set(benchmark.evaluated_frames())
    count = 0
    for track in tracks:
        if track.status not in statuses:
            continue
        for observation in track.sorted_observations():
            if observation.frame_index not in evaluated:
                continue
            if not plate_bakeoff.center_inside_roi(observation.center, benchmark.target_roi_xyxy):
                count += 1
    return count


def missed_target_frames(
    tracks: list[plate_stabilization.PlateTrack],
    benchmark: plate_bakeoff.PlateBenchmark,
) -> list[int]:
    if benchmark.target_roi_xyxy is None:
        return []
    covered = covered_target_frames(
        tracks,
        benchmark,
        statuses={"accepted_plate_track", "review_candidate"},
    )
    return [frame for frame in benchmark.evaluated_frames() if frame not in covered]


def review_candidate_frames(
    tracks: list[plate_stabilization.PlateTrack],
    benchmark: plate_bakeoff.PlateBenchmark,
) -> list[int]:
    frames = []
    evaluated = set(benchmark.evaluated_frames())
    for track in tracks:
        if track.status != "review_candidate":
            continue
        frame = plate_stabilization.representative_frame_for_track(track)
        if not evaluated or frame in evaluated:
            frames.append(frame)
    return sorted(dict.fromkeys(frames))


def active_track_counts_by_frame(
    tracks: list[plate_stabilization.PlateTrack],
    *,
    statuses: set[str],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for track in tracks:
        if track.status not in statuses:
            continue
        for observation in track.sorted_observations():
            counts[observation.frame_index] = counts.get(observation.frame_index, 0) + 1
    return counts


def average_active_tracks(active_counts: dict[int, int], *, frames: list[int]) -> float:
    if not frames:
        return 0.0
    return round(sum(active_counts.get(frame, 0) for frame in frames) / len(frames), 6)


def visual_stability_summary(tracks: list[plate_stabilization.PlateTrack]) -> dict[str, Any]:
    accepted_or_review = [
        track for track in tracks if track.status in {"accepted_plate_track", "review_candidate"}
    ]
    if not accepted_or_review:
        return {
            "max_center_jump_normalized": None,
            "max_scale_ratio": None,
            "mean_observation_density": None,
        }
    summaries = [plate_stabilization.summarize_track(track) for track in accepted_or_review]
    return {
        "max_center_jump_normalized": max(item["max_center_jump_normalized"] for item in summaries),
        "max_scale_ratio": max(item["max_scale_ratio"] for item in summaries),
        "mean_observation_density": round(
            sum(item["observation_density"] for item in summaries) / len(summaries),
            6,
        ),
    }


def selection_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    day = metrics["clips"]["day_plate_far_gold"]["stabilized_track_metrics"]
    night = metrics["clips"]["night_selective_plate_gold"]["stabilized_track_metrics"]
    recall_floor_met = (
        day["stabilized_target_frame_recall"] >= 0.90
        and night["stabilized_target_frame_recall"] >= 0.95
    )
    miss_runs_manageable = (
        day["longest_uncovered_visible_run"] <= 10 and night["longest_uncovered_visible_run"] <= 10
    )
    false_accepted_low = (
        day["false_accepted_track_frame_count"] <= 24
        and night["false_accepted_track_frame_count"] <= 36
    )
    review_burden = day["review_candidate_tracks"] + night["review_candidate_tracks"]
    false_review_frames = (
        day["false_review_track_frame_count"] + night["false_review_track_frame_count"]
    )
    if recall_floor_met and miss_runs_manageable and false_accepted_low and review_burden <= 25:
        decision = "suitable_for_prototype_plate_pipeline"
        reason = "Recall and false accepted tracks are both within the prototype gates."
    elif recall_floor_met and miss_runs_manageable and false_accepted_low and review_burden <= 200:
        decision = "usable_only_with_heavy_review"
        reason = (
            "Recall remained high, but the number of review-candidate tracks is too high "
            "for a low-friction prototype."
        )
    else:
        decision = "temporal_filtering_insufficient"
        reason = (
            "Generic stabilization either lost target coverage or left an impractical "
            "false/review-track burden."
        )
    return {
        "bounded_decision": decision,
        "decision_reason": reason,
        "fine_tuning_fallback_plan": {
            "next_direction": (
                "Fine-tune one permissively licensed detector architecture such as "
                "RT-DETR or YOLOX on a clearly licensed plate dataset plus "
                "project-specific annotations."
            ),
            "project_annotation_minimum": (
                "At least 400-800 project frames with tight plate boxes spanning day, "
                "night, close, oblique, motion-blurred, and occluded examples."
            ),
            "validation_split": (
                "Hold out complete clips or contiguous temporal blocks, not random "
                "near-duplicate frames."
            ),
            "licensing_checks": [
                "Verify dataset redistribution and commercial-use terms.",
                "Verify detector architecture and pretrained-weight license.",
                "Record hashes for training data manifests and exported model files.",
            ],
            "expected_deliverables": [
                "training manifest",
                "validation metrics",
                "calibrated threshold recommendation",
                "qualitative overlays",
                "model-card/license record",
            ],
        },
        "gates": {
            "false_accepted_low": false_accepted_low,
            "false_review_frames": false_review_frames,
            "miss_runs_manageable": miss_runs_manageable,
            "recall_floor_met": recall_floor_met,
            "review_candidate_tracks_day_plus_night": review_burden,
        },
        "schema_version": plate_stabilization.SCHEMA_VERSION,
    }


def configuration_report(config: plate_stabilization.PlateStabilizationConfig) -> dict[str, Any]:
    return {
        "created_at": media.utc_timestamp(),
        "detector_input": {
            "detector_id": DETECTOR_ID,
            "source": (
                "Corrected Intel YOLOv8n OpenVINO proposal records from "
                "license-plate detector bake-off v1.1."
            ),
            "threshold": config.proposal_threshold,
        },
        "notes": [
            "One fixed proposal and tracking configuration is used across day, night, "
            "close-plate, and synthetic tests.",
            "This is an experimental stabilization layer, not an active production "
            "license-plate pipeline.",
            "Ground-truth ROIs are used only by this experiment's scoring code.",
        ],
        "schema_version": plate_stabilization.SCHEMA_VERSION,
        "stabilization_config": config.to_dict(),
    }


def write_summary(output_dir: Path, metrics: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# Plate Proposal Stabilization v1",
        "",
        "This bounded prototype reuses the corrected Intel YOLOv8n plate output as "
        "an experimental high-recall proposal source. It does not approve a production "
        "plate detector and does not integrate plates into the active face pipeline.",
        "",
        "## Fixed configuration",
        "",
        f"- Proposal threshold: {metrics['stabilization_config']['proposal_threshold']}",
        "- Same geometry and tracking policy used for every clip.",
        "- No frame-specific ROIs, coordinates, or per-video thresholds are used by "
        "the stabilizer.",
        "",
        "## Metrics",
        "",
        "| Clip | Raw recall | Stabilized accepted/review recall | Longest uncovered run | "
        "Accepted tracks | Review tracks | Rejected tracks | False accepted track frames | "
        "False review track frames |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip_id, clip in metrics["clips"].items():
        raw = clip["raw_detector_metrics"]
        track = clip["stabilized_track_metrics"]
        raw_recall = raw["object_frame_recall"]
        stable_recall = track.get("stabilized_target_frame_recall")
        lines.append(
            "| "
            f"{clip_id} | {format_metric(raw_recall)} | {format_metric(stable_recall)} | "
            f"{track.get('longest_uncovered_visible_run', 'n/a')} | "
            f"{track['accepted_tracks']} | {track['review_candidate_tracks']} | "
            f"{track['rejected_tracks']} | "
            f"{track.get('false_accepted_track_frame_count', 'n/a')} | "
            f"{track.get('false_review_track_frame_count', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['bounded_decision']}**",
            "",
            decision["decision_reason"],
            "",
            "If this remains insufficient, the bounded next step is the fine-tuning "
            "plan recorded in `decision.json`; no training was started here.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
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
            "schema_version": plate_stabilization.SCHEMA_VERSION,
        },
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_json_hash(data: Any) -> str:
    import hashlib

    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def clip_output_name(clip_id: str) -> str:
    if clip_id == "day_plate_far_gold":
        return "day"
    if clip_id == "night_selective_plate_gold":
        return "night"
    if clip_id == "day_plate_close_stress":
        return "close"
    return clip_id


def maybe_copy_gallery_index(gallery_dir: Path, destination: Path) -> None:
    source = gallery_dir / "track_gallery.html"
    if source.exists() and source != destination:
        shutil.copyfile(source, destination)


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"plate proposal stabilization failed: {exc}", file=sys.stderr)
        raise
