from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import cv2 as cv
import numpy as np

from redactor import frame_spans, media, plate_detection
from redactor.detection import clip_xyxy_to_frame, rational_to_float

SCHEMA_VERSION = "plate_tracking.v1"

CATEGORY = "license_plate"
PROVENANCE_DETECTOR = "plate_detector_observation"
PROVENANCE_INTERPOLATION = "plate_track_interpolation"
PROVENANCE_PREDICTION = "plate_track_prediction"
PROVENANCE_REVIEWED_CORRECTION = "reviewed_plate_correction"

ACCEPTED_TRACK = "accepted_plate_track"
NEEDS_REVIEW = "needs_review"
REJECTED_NOISE = "rejected_noise_track"

DEFAULT_MAX_ASSOCIATION_GAP_SECONDS = 1.50
DEFAULT_MAX_SAFE_INTERPOLATION_GAP_SECONDS = 0.30
DEFAULT_MIN_ASSOCIATION_IOU = 0.02
DEFAULT_MAX_CENTER_DISTANCE_NORM = 2.75
DEFAULT_MAX_MOTION_DISTANCE_NORM = 2.75
DEFAULT_MAX_ASSOCIATION_SCALE_RATIO = 3.00
DEFAULT_MAX_INTERPOLATION_CENTER_DISTANCE_NORM = 3.00
DEFAULT_MAX_INTERPOLATION_SCALE_RATIO = 2.25
DEFAULT_MIN_ASPECT_RATIO = 0.70
DEFAULT_MAX_ASPECT_RATIO = 8.00
DEFAULT_MIN_AREA_FRACTION = 0.00004
DEFAULT_MAX_AREA_FRACTION = 0.08
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.50
DEFAULT_DUPLICATE_CENTER_DISTANCE_NORM = 0.35
DEFAULT_MIN_ACCEPTED_OBSERVATIONS = 6
DEFAULT_MIN_REVIEW_OBSERVATIONS = 2
DEFAULT_MIN_ACCEPTED_OBSERVATION_DENSITY = 0.20
DEFAULT_MAX_ACCEPTED_CENTER_JUMP_NORM = 4.00
DEFAULT_MAX_ACCEPTED_SCALE_JUMP = 4.00
DEFAULT_STATIC_OVERLAY_MIN_OBSERVATIONS = 30
DEFAULT_STATIC_OVERLAY_MAX_CENTER_STD_NORM = 0.05
DEFAULT_SCENE_CUT_LUMA_DELTA = 55.0
TRACK_ID_PREFIX = "plate_track"


@dataclass(frozen=True)
class PlateTrackingConfig:
    detector_id: str = plate_detection.DETECTOR_ID
    detector_configuration_hash: str = plate_detection.OIMPlateDetectorConfig().configuration_hash()
    primary_threshold: float = plate_detection.DEFAULT_PRIMARY_THRESHOLD
    max_association_gap_seconds: float = DEFAULT_MAX_ASSOCIATION_GAP_SECONDS
    max_safe_interpolation_gap_seconds: float = DEFAULT_MAX_SAFE_INTERPOLATION_GAP_SECONDS
    min_association_iou: float = DEFAULT_MIN_ASSOCIATION_IOU
    max_center_distance_norm: float = DEFAULT_MAX_CENTER_DISTANCE_NORM
    max_motion_distance_norm: float = DEFAULT_MAX_MOTION_DISTANCE_NORM
    max_association_scale_ratio: float = DEFAULT_MAX_ASSOCIATION_SCALE_RATIO
    max_interpolation_center_distance_norm: float = DEFAULT_MAX_INTERPOLATION_CENTER_DISTANCE_NORM
    max_interpolation_scale_ratio: float = DEFAULT_MAX_INTERPOLATION_SCALE_RATIO
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION
    max_area_fraction: float = DEFAULT_MAX_AREA_FRACTION
    same_frame_duplicate_iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD
    same_frame_duplicate_center_distance_norm: float = DEFAULT_DUPLICATE_CENTER_DISTANCE_NORM
    min_accepted_observations: int = DEFAULT_MIN_ACCEPTED_OBSERVATIONS
    min_review_observations: int = DEFAULT_MIN_REVIEW_OBSERVATIONS
    min_accepted_observation_density: float = DEFAULT_MIN_ACCEPTED_OBSERVATION_DENSITY
    max_accepted_center_jump_norm: float = DEFAULT_MAX_ACCEPTED_CENTER_JUMP_NORM
    max_accepted_scale_jump: float = DEFAULT_MAX_ACCEPTED_SCALE_JUMP
    static_overlay_min_observations: int = DEFAULT_STATIC_OVERLAY_MIN_OBSERVATIONS
    static_overlay_max_center_std_norm: float = DEFAULT_STATIC_OVERLAY_MAX_CENTER_STD_NORM
    scene_cut_luma_delta: float = DEFAULT_SCENE_CUT_LUMA_DELTA

    def __post_init__(self) -> None:
        if self.detector_id != plate_detection.DETECTOR_ID:
            raise ValueError(f"plate tracking supports only {plate_detection.DETECTOR_ID}")
        if (
            self.detector_configuration_hash
            != plate_detection.OIMPlateDetectorConfig().configuration_hash()
        ):
            raise ValueError("plate tracking configuration must use the selected OIM detector hash")
        if not 0 <= self.primary_threshold <= 1:
            raise ValueError("primary_threshold must be between 0 and 1")
        if self.primary_threshold != plate_detection.DEFAULT_PRIMARY_THRESHOLD:
            raise ValueError(
                "plate tracking v1 must not change the selected OIM detector threshold"
            )
        if self.max_association_gap_seconds < 0:
            raise ValueError("max_association_gap_seconds cannot be negative")
        if self.max_safe_interpolation_gap_seconds < 0:
            raise ValueError("max_safe_interpolation_gap_seconds cannot be negative")
        if self.max_safe_interpolation_gap_seconds > self.max_association_gap_seconds:
            raise ValueError("safe interpolation cannot exceed the association gap")
        if not 0 <= self.min_association_iou <= 1:
            raise ValueError("min_association_iou must be between 0 and 1")
        if self.max_center_distance_norm <= 0 or self.max_motion_distance_norm <= 0:
            raise ValueError("association distance gates must be positive")
        if self.max_association_scale_ratio < 1:
            raise ValueError("scale ratio gates must be at least 1")
        if self.min_aspect_ratio <= 0 or self.max_aspect_ratio < self.min_aspect_ratio:
            raise ValueError("invalid aspect-ratio bounds")
        if self.min_area_fraction < 0 or self.max_area_fraction <= self.min_area_fraction:
            raise ValueError("invalid area-fraction bounds")
        if self.min_accepted_observations < 1 or self.min_review_observations < 1:
            raise ValueError("observation-count gates must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_configuration_hash": self.detector_configuration_hash,
            "detector_id": self.detector_id,
            "max_accepted_center_jump_norm": self.max_accepted_center_jump_norm,
            "max_accepted_scale_jump": self.max_accepted_scale_jump,
            "max_area_fraction": self.max_area_fraction,
            "max_aspect_ratio": self.max_aspect_ratio,
            "max_association_gap_seconds": self.max_association_gap_seconds,
            "max_association_scale_ratio": self.max_association_scale_ratio,
            "max_center_distance_norm": self.max_center_distance_norm,
            "max_interpolation_center_distance_norm": (self.max_interpolation_center_distance_norm),
            "max_interpolation_scale_ratio": self.max_interpolation_scale_ratio,
            "max_motion_distance_norm": self.max_motion_distance_norm,
            "max_safe_interpolation_gap_seconds": self.max_safe_interpolation_gap_seconds,
            "min_accepted_observation_density": self.min_accepted_observation_density,
            "min_accepted_observations": self.min_accepted_observations,
            "min_area_fraction": self.min_area_fraction,
            "min_aspect_ratio": self.min_aspect_ratio,
            "min_association_iou": self.min_association_iou,
            "min_review_observations": self.min_review_observations,
            "primary_threshold": self.primary_threshold,
            "same_frame_duplicate_center_distance_norm": (
                self.same_frame_duplicate_center_distance_norm
            ),
            "same_frame_duplicate_iou_threshold": self.same_frame_duplicate_iou_threshold,
            "scene_cut_luma_delta": self.scene_cut_luma_delta,
            "static_overlay_max_center_std_norm": self.static_overlay_max_center_std_norm,
            "static_overlay_min_observations": self.static_overlay_min_observations,
        }

    def configuration_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlateObservation:
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    detector_id: str
    candidate_id: str | None
    source_detection_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "candidate_id": self.candidate_id,
            "detector_id": self.detector_id,
            "frame_index": self.frame_index,
            "metadata": self.metadata,
            "score": round(self.score, 6),
            "source_detection_index": self.source_detection_index,
        }


@dataclass(frozen=True)
class RejectedProposal:
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int] | None
    score: float | None
    candidate_id: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": None if self.bbox_xyxy is None else list(self.bbox_xyxy),
            "candidate_id": self.candidate_id,
            "frame_index": self.frame_index,
            "reasons": list(self.reasons),
            "score": None if self.score is None else round(self.score, 6),
        }


@dataclass(frozen=True)
class PlateTrackFrame:
    frame_index: int
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    provenance: str
    score: float | None
    status: str
    source_candidate_ids: tuple[str, ...] = ()
    anchor_frames: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_frames": list(self.anchor_frames),
            "bbox_xyxy": list(self.bbox_xyxy),
            "category": CATEGORY,
            "frame_index": self.frame_index,
            "provenance": self.provenance,
            "score": None if self.score is None else round(self.score, 6),
            "source_candidate_ids": list(self.source_candidate_ids),
            "status": self.status,
            "track_id": self.track_id,
            "track_uid": format_track_uid(self.track_id),
        }


@dataclass(frozen=True)
class GapDecision:
    track_id: int
    frame_start: int
    frame_end: int
    length: int
    left_anchor_frame: int
    right_anchor_frame: int
    decision: str
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | bool | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "left_anchor_frame": self.left_anchor_frame,
            "length": self.length,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
            "right_anchor_frame": self.right_anchor_frame,
            "track_id": self.track_id,
            "track_uid": format_track_uid(self.track_id),
        }


@dataclass(frozen=True)
class ReviewInterval:
    frame_start: int
    frame_end: int
    track_ids: tuple[int, ...]
    severity: str
    blocking: bool
    reason_codes: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocking": self.blocking,
            "explanation": self.explanation,
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "reason_codes": list(self.reason_codes),
            "severity": self.severity,
            "track_ids": list(self.track_ids),
            "track_uids": [format_track_uid(track_id) for track_id in self.track_ids],
        }


@dataclass
class PlateTrack:
    track_id: int
    observations: list[PlateObservation] = field(default_factory=list)
    frames: list[PlateTrackFrame] = field(default_factory=list)
    gap_decisions: list[GapDecision] = field(default_factory=list)
    status: str = NEEDS_REVIEW
    review_reasons: list[str] = field(default_factory=list)

    @property
    def sorted_observations(self) -> list[PlateObservation]:
        return sorted(self.observations, key=lambda item: item.frame_index)

    @property
    def last_observation(self) -> PlateObservation:
        return self.sorted_observations[-1]

    @property
    def first_frame(self) -> int:
        return self.sorted_observations[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.sorted_observations[-1].frame_index

    def add_observation(self, observation: PlateObservation) -> None:
        self.observations.append(observation)

    def to_summary_dict(self) -> dict[str, Any]:
        observed = self.sorted_observations
        frames = sorted(self.frames, key=lambda item: (item.frame_index, item.provenance))
        observed_frames = [item.frame_index for item in observed]
        interpolated_frames = [
            item.frame_index for item in frames if item.provenance == PROVENANCE_INTERPOLATION
        ]
        scores = [item.score for item in observed]
        span = max(1, self.last_frame - self.first_frame + 1) if observed else 1
        centers = [bbox_center(item.bbox_xyxy) for item in observed]
        return {
            "first_frame": self.first_frame,
            "frame_count": len(frames),
            "interpolated_frame_count": len(interpolated_frames),
            "interpolated_frames": interpolated_frames,
            "last_frame": self.last_frame,
            "max_center_jump_norm": round(max_center_jump_norm(observed), 6),
            "max_internal_observed_gap": max_internal_gap(observed_frames),
            "max_scale_jump": round(max_scale_jump(observed), 6),
            "mean_score": round(mean(scores), 6) if scores else None,
            "min_score": round(min(scores), 6) if scores else None,
            "observation_count": len(observed),
            "observation_density": round(len(observed) / span, 6),
            "observed_frames": observed_frames,
            "possible_static_overlay": possible_static_overlay(
                centers,
                observed,
                PlateTrackingConfig(),
            ),
            "provenance_counts": provenance_counts(frames),
            "review_reasons": sorted(set(self.review_reasons)),
            "span_frames": span,
            "status": self.status,
            "track_id": self.track_id,
            "track_uid": format_track_uid(self.track_id),
            "unresolved_gap_count": sum(
                1 for decision in self.gap_decisions if decision.decision == "needs_review"
            ),
        }


@dataclass
class PlateTrackingResult:
    tracks: list[PlateTrack]
    rejected_proposals: list[RejectedProposal]
    gap_decisions: list[GapDecision]
    review_intervals: list[ReviewInterval]
    frame_tracks: dict[int, list[PlateTrackFrame]]
    frames_processed: int
    max_association_gap_frames: int
    max_safe_interpolation_gap_frames: int


def default_config() -> PlateTrackingConfig:
    return PlateTrackingConfig()


def run_plate_tracking(
    source_video: Path,
    *,
    detections_path: Path,
    output_dir: Path,
    config: PlateTrackingConfig | None = None,
    overwrite: bool = False,
    write_overlay_video: bool = True,
    show_rejected_overlay: bool = False,
) -> dict[str, Any]:
    resolved_config = config or default_config()
    ensure_output_dir(output_dir, overwrite=overwrite)

    source_info = media.probe_media(source_video)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    source_frame_count = optional_int(source_info.video_stream.frame_count)
    detections_sha = media.sha256_file(detections_path)
    source_sha = source_info.sha256

    started_at = time.perf_counter()
    observations_by_frame, rejected = load_plate_detection_records(
        detections_path,
        width=width,
        height=height,
        config=resolved_config,
    )
    detection_indices = frame_spans.jsonl_frame_indices(detections_path)
    detection_span_report = frame_spans.build_frame_span_report(
        detection_indices,
        source_frame_count=source_frame_count,
    )
    frames_processed = int(detection_span_report["processed_frame_span_count"])
    result = track_plate_observations(
        observations_by_frame,
        frame_count=frames_processed,
        fps=fps,
        width=width,
        height=height,
        config=resolved_config,
        rejected_proposals=rejected,
        source_video=source_video,
    )
    elapsed_seconds = time.perf_counter() - started_at

    write_tracks_jsonl(
        output_dir / "tracks.jsonl",
        result.frame_tracks,
        frames_processed,
        detector_frame_indices=detection_indices,
    )
    write_json(
        output_dir / "track_summary.json",
        build_track_summary(result, config=resolved_config, elapsed_seconds=elapsed_seconds),
    )
    write_json(output_dir / "gap_decisions.json", [item.to_dict() for item in result.gap_decisions])
    write_json(
        output_dir / "review_intervals.json",
        [item.to_dict() for item in result.review_intervals],
    )
    manifest = build_tracking_manifest(
        source_video=source_video,
        detections_path=detections_path,
        source_sha256=source_sha,
        detections_sha256=detections_sha,
        source_frame_count=source_frame_count,
        frames_processed=frames_processed,
        width=width,
        height=height,
        fps=fps,
        config=resolved_config,
        result=result,
        elapsed_seconds=elapsed_seconds,
        detection_span_report=detection_span_report,
    )
    write_json(output_dir / "tracking_manifest.json", manifest)
    write_json(output_dir / "summary.json", manifest)
    write_json(output_dir / "frame_span_report.json", detection_span_report)
    write_processing_log(output_dir / "processing_log.txt", manifest, result)
    write_track_gallery(source_video, output_dir / "track_gallery.html", result, width, height)
    write_gap_contact_sheet(
        source_video,
        output_dir / "gap_contact_sheet.jpg",
        result,
        width=width,
        height=height,
    )
    write_false_track_contact_sheet(
        source_video,
        output_dir / "false_track_contact_sheet.jpg",
        result,
        width=width,
        height=height,
    )
    if write_overlay_video:
        write_tracking_overlay(
            source_video,
            output_dir / "tracking_overlay.mp4",
            result,
            width=width,
            height=height,
            fps=fps,
            frames_processed=frames_processed,
            show_rejected_overlay=show_rejected_overlay,
        )

    return manifest


def track_plate_observations(
    observations_by_frame: dict[int, list[PlateObservation]],
    *,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    config: PlateTrackingConfig | None = None,
    rejected_proposals: list[RejectedProposal] | None = None,
    source_video: Path | None = None,
    scene_cut_frames: set[tuple[int, int]] | None = None,
) -> PlateTrackingResult:
    resolved_config = config or default_config()
    max_association_gap_frames = seconds_to_frames(
        resolved_config.max_association_gap_seconds,
        fps,
    )
    max_safe_interpolation_gap_frames = seconds_to_frames(
        resolved_config.max_safe_interpolation_gap_seconds,
        fps,
    )

    tracks = build_plate_track_proposals(
        observations_by_frame,
        max_association_gap_frames=max_association_gap_frames,
        config=resolved_config,
    )
    scene_cuts = scene_cut_frames or set()
    if source_video is not None:
        scene_cuts = scene_cuts | detect_scene_cuts_for_track_gaps(
            source_video,
            tracks,
            config=resolved_config,
        )
    finalize_plate_tracks(
        tracks,
        max_safe_interpolation_gap_frames=max_safe_interpolation_gap_frames,
        width=width,
        height=height,
        config=resolved_config,
        scene_cut_frames=scene_cuts,
    )
    frame_tracks = build_frame_track_index(tracks, frame_count=frame_count)
    gap_decisions = sorted(
        [decision for track in tracks for decision in track.gap_decisions],
        key=lambda item: (item.frame_start, item.frame_end, item.track_id),
    )
    review_intervals = build_review_intervals(tracks)
    return PlateTrackingResult(
        tracks=sorted(tracks, key=lambda item: item.track_id),
        rejected_proposals=sorted(
            rejected_proposals or [],
            key=lambda item: (item.frame_index, item.candidate_id or "", item.reasons),
        ),
        gap_decisions=gap_decisions,
        review_intervals=review_intervals,
        frame_tracks=frame_tracks,
        frames_processed=frame_count,
        max_association_gap_frames=max_association_gap_frames,
        max_safe_interpolation_gap_frames=max_safe_interpolation_gap_frames,
    )


def load_plate_detection_records(
    path: Path,
    *,
    width: int,
    height: int,
    config: PlateTrackingConfig | None = None,
) -> tuple[dict[int, list[PlateObservation]], list[RejectedProposal]]:
    resolved_config = config or default_config()
    observations_by_frame: dict[int, list[PlateObservation]] = {}
    rejected: list[RejectedProposal] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            frame_observations: list[PlateObservation] = []
            for detection_index, detection in enumerate(record.get("detections", [])):
                if detection.get("category") != CATEGORY:
                    continue
                candidate_id = detection.get("candidate_id")
                score = float(detection["score"])
                bbox = tuple(int(value) for value in detection.get("bbox_xyxy", []))
                if len(bbox) != 4:
                    rejected.append(
                        RejectedProposal(
                            frame_index,
                            None,
                            score,
                            candidate_id,
                            ("invalid_bbox",),
                        )
                    )
                    continue
                clipped = clip_xyxy_to_frame(bbox, width, height)
                reasons = proposal_rejection_reasons(
                    clipped,
                    score=score,
                    width=width,
                    height=height,
                    config=resolved_config,
                    detector_id=detection.get("detector_id"),
                )
                if reasons:
                    rejected.append(
                        RejectedProposal(
                            frame_index,
                            clipped,
                            score,
                            candidate_id,
                            tuple(reasons),
                        )
                    )
                    continue
                frame_observations.append(
                    PlateObservation(
                        frame_index=frame_index,
                        bbox_xyxy=clipped,
                        score=score,
                        detector_id=str(detection.get("detector_id") or ""),
                        candidate_id=candidate_id,
                        source_detection_index=detection.get(
                            "source_detection_index",
                            detection_index,
                        ),
                        metadata=detection.get("metadata") or {},
                    )
                )
            kept, duplicates = deduplicate_same_frame_observations(
                frame_observations,
                config=resolved_config,
            )
            observations_by_frame[frame_index] = kept
            rejected.extend(duplicates)
    return observations_by_frame, rejected


def proposal_rejection_reasons(
    bbox: tuple[int, int, int, int] | None,
    *,
    score: float,
    width: int,
    height: int,
    config: PlateTrackingConfig,
    detector_id: str | None,
) -> list[str]:
    reasons: list[str] = []
    if detector_id != config.detector_id:
        reasons.append("unsupported_detector_id")
    if score < config.primary_threshold:
        reasons.append("below_primary_threshold")
    if bbox is None:
        reasons.append("invalid_or_out_of_frame_bbox")
        return reasons
    box_width, box_height = bbox_size(bbox)
    if box_width <= 0 or box_height <= 0:
        reasons.append("degenerate_bbox")
        return reasons
    aspect = box_width / box_height
    if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
        reasons.append("implausible_plate_aspect_ratio")
    area_fraction = bbox_area(bbox) / float(width * height)
    if area_fraction < config.min_area_fraction:
        reasons.append("area_too_small")
    if area_fraction > config.max_area_fraction:
        reasons.append("area_too_large")
    return reasons


def deduplicate_same_frame_observations(
    observations: list[PlateObservation],
    *,
    config: PlateTrackingConfig,
) -> tuple[list[PlateObservation], list[RejectedProposal]]:
    kept: list[PlateObservation] = []
    rejected: list[RejectedProposal] = []
    for observation in sorted(
        observations,
        key=lambda item: (-item.score, item.bbox_xyxy, item.candidate_id or ""),
    ):
        duplicate_of = None
        for kept_observation in kept:
            iou = plate_detection.bbox_iou(observation.bbox_xyxy, kept_observation.bbox_xyxy)
            center_norm = center_distance_norm(
                observation.bbox_xyxy,
                kept_observation.bbox_xyxy,
            )
            if (
                iou >= config.same_frame_duplicate_iou_threshold
                or center_norm <= config.same_frame_duplicate_center_distance_norm
            ):
                duplicate_of = kept_observation
                break
        if duplicate_of is None:
            kept.append(observation)
        else:
            rejected.append(
                RejectedProposal(
                    observation.frame_index,
                    observation.bbox_xyxy,
                    observation.score,
                    observation.candidate_id,
                    (
                        "duplicate_same_frame_proposal",
                        f"duplicate_of={duplicate_of.candidate_id or 'unknown'}",
                    ),
                )
            )
    return kept, rejected


def build_plate_track_proposals(
    observations_by_frame: dict[int, list[PlateObservation]],
    *,
    max_association_gap_frames: int,
    config: PlateTrackingConfig,
) -> list[PlateTrack]:
    tracks: list[PlateTrack] = []
    next_track_id = 1
    for frame_index in sorted(observations_by_frame):
        observations = sorted(
            observations_by_frame[frame_index],
            key=lambda item: (-item.score, item.bbox_xyxy, item.candidate_id or ""),
        )
        if not observations:
            continue
        assignments = match_observations_to_plate_tracks(
            tracks,
            observations,
            max_association_gap_frames=max_association_gap_frames,
            config=config,
        )
        assigned_observations = set(assignments)
        for observation_index, track in assignments.items():
            track.add_observation(observations[observation_index])
        for observation_index, observation in enumerate(observations):
            if observation_index in assigned_observations:
                continue
            track = PlateTrack(next_track_id)
            track.add_observation(observation)
            tracks.append(track)
            next_track_id += 1
    return tracks


def match_observations_to_plate_tracks(
    tracks: list[PlateTrack],
    observations: list[PlateObservation],
    *,
    max_association_gap_frames: int,
    config: PlateTrackingConfig,
) -> dict[int, PlateTrack]:
    candidates: list[tuple[float, int, int, int]] = []
    for observation_index, observation in enumerate(observations):
        for track_index, track in enumerate(tracks):
            last_observation = track.last_observation
            frame_gap = observation.frame_index - last_observation.frame_index
            if frame_gap <= 0 or frame_gap > max_association_gap_frames:
                continue
            candidate = association_metrics(track, observation)
            if not association_allowed(candidate, config):
                continue
            candidates.append(
                (association_cost(candidate), track.track_id, observation_index, track_index)
            )

    assignments: dict[int, PlateTrack] = {}
    used_track_indexes: set[int] = set()
    for _cost, _track_id, observation_index, track_index in sorted(candidates):
        if observation_index in assignments or track_index in used_track_indexes:
            continue
        assignments[observation_index] = tracks[track_index]
        used_track_indexes.add(track_index)
    return assignments


def association_metrics(track: PlateTrack, observation: PlateObservation) -> dict[str, float | int]:
    previous = track.sorted_observations
    last = previous[-1]
    frame_gap = observation.frame_index - last.frame_index
    iou = plate_detection.bbox_iou(last.bbox_xyxy, observation.bbox_xyxy)
    center_norm = center_distance_norm(last.bbox_xyxy, observation.bbox_xyxy)
    scale_ratio = bbox_scale_ratio(last.bbox_xyxy, observation.bbox_xyxy)
    motion_norm = center_norm
    if len(previous) >= 2:
        earlier = previous[-2]
        prior_gap = max(1, last.frame_index - earlier.frame_index)
        vx = (bbox_center(last.bbox_xyxy)[0] - bbox_center(earlier.bbox_xyxy)[0]) / prior_gap
        vy = (bbox_center(last.bbox_xyxy)[1] - bbox_center(earlier.bbox_xyxy)[1]) / prior_gap
        predicted = (
            bbox_center(last.bbox_xyxy)[0] + vx * frame_gap,
            bbox_center(last.bbox_xyxy)[1] + vy * frame_gap,
        )
        observed_center = bbox_center(observation.bbox_xyxy)
        norm = mean_bbox_diagonal(last.bbox_xyxy, observation.bbox_xyxy)
        motion_norm = euclidean(predicted, observed_center) / norm if norm > 0 else float("inf")
    return {
        "center_distance_norm": center_norm,
        "frame_gap": frame_gap,
        "iou": iou,
        "motion_distance_norm": motion_norm,
        "scale_ratio": scale_ratio,
        "score": observation.score,
    }


def association_allowed(metrics: dict[str, float | int], config: PlateTrackingConfig) -> bool:
    if metrics["scale_ratio"] > config.max_association_scale_ratio:
        return False
    if metrics["center_distance_norm"] <= config.max_center_distance_norm:
        return True
    if metrics["motion_distance_norm"] <= config.max_motion_distance_norm:
        return True
    return metrics["iou"] >= config.min_association_iou


def association_cost(metrics: dict[str, float | int]) -> float:
    return (
        float(metrics["center_distance_norm"])
        + 0.50 * float(metrics["motion_distance_norm"])
        + 0.35 * abs(math.log(max(float(metrics["scale_ratio"]), 1e-6)))
        + 0.03 * float(metrics["frame_gap"])
        - 0.80 * float(metrics["iou"])
        - 0.10 * float(metrics["score"])
    )


def finalize_plate_tracks(
    tracks: list[PlateTrack],
    *,
    max_safe_interpolation_gap_frames: int,
    width: int,
    height: int,
    config: PlateTrackingConfig,
    scene_cut_frames: set[tuple[int, int]] | None = None,
) -> None:
    scene_cuts = scene_cut_frames or set()
    for track in tracks:
        observed_frames = [
            PlateTrackFrame(
                frame_index=observation.frame_index,
                track_id=track.track_id,
                bbox_xyxy=observation.bbox_xyxy,
                provenance=PROVENANCE_DETECTOR,
                score=observation.score,
                status=NEEDS_REVIEW,
                source_candidate_ids=(observation.candidate_id,)
                if observation.candidate_id is not None
                else (),
            )
            for observation in track.sorted_observations
        ]
        interpolated: list[PlateTrackFrame] = []
        gap_decisions: list[GapDecision] = []
        observations = track.sorted_observations
        for left, right in zip(observations, observations[1:], strict=False):
            missing_length = right.frame_index - left.frame_index - 1
            if missing_length <= 0:
                continue
            decision = decide_gap(
                track.track_id,
                left,
                right,
                missing_length=missing_length,
                max_safe_interpolation_gap_frames=max_safe_interpolation_gap_frames,
                config=config,
                scene_cut=(left.frame_index, right.frame_index) in scene_cuts,
            )
            gap_decisions.append(decision)
            if decision.decision == "accepted_interpolation":
                interpolated.extend(
                    interpolate_plate_gap(
                        track.track_id,
                        left,
                        right,
                        width=width,
                        height=height,
                    )
                )
        track.gap_decisions = gap_decisions
        track.status, track.review_reasons = classify_track(
            track,
            observed_frames=observed_frames,
            interpolated_frames=interpolated,
            config=config,
        )
        all_frames = []
        for frame in [*observed_frames, *interpolated]:
            all_frames.append(
                PlateTrackFrame(
                    frame_index=frame.frame_index,
                    track_id=frame.track_id,
                    bbox_xyxy=frame.bbox_xyxy,
                    provenance=frame.provenance,
                    score=frame.score,
                    status=track.status,
                    source_candidate_ids=frame.source_candidate_ids,
                    anchor_frames=frame.anchor_frames,
                )
            )
        track.frames = sorted(all_frames, key=lambda item: (item.frame_index, item.provenance))


def decide_gap(
    track_id: int,
    left: PlateObservation,
    right: PlateObservation,
    *,
    missing_length: int,
    max_safe_interpolation_gap_frames: int,
    config: PlateTrackingConfig,
    scene_cut: bool,
) -> GapDecision:
    reasons: list[str] = []
    center_norm = center_distance_norm(left.bbox_xyxy, right.bbox_xyxy)
    scale_ratio = bbox_scale_ratio(left.bbox_xyxy, right.bbox_xyxy)
    if missing_length > max_safe_interpolation_gap_frames:
        reasons.append("gap_exceeds_safe_interpolation_limit")
    if center_norm > config.max_interpolation_center_distance_norm:
        reasons.append("anchor_center_motion_too_large")
    if scale_ratio > config.max_interpolation_scale_ratio:
        reasons.append("anchor_scale_change_too_large")
    if scene_cut:
        reasons.append("scene_cut_inside_gap")
    decision = "accepted_interpolation" if not reasons else "needs_review"
    return GapDecision(
        track_id=track_id,
        frame_start=left.frame_index + 1,
        frame_end=right.frame_index - 1,
        length=missing_length,
        left_anchor_frame=left.frame_index,
        right_anchor_frame=right.frame_index,
        decision=decision,
        reasons=tuple(reasons or ["bounded_two_anchor_interpolation"]),
        metrics={
            "anchor_center_distance_norm": round(center_norm, 6),
            "anchor_scale_ratio": round(scale_ratio, 6),
            "max_safe_interpolation_gap_frames": max_safe_interpolation_gap_frames,
            "scene_cut": scene_cut,
        },
    )


def interpolate_plate_gap(
    track_id: int,
    left: PlateObservation,
    right: PlateObservation,
    *,
    width: int,
    height: int,
) -> list[PlateTrackFrame]:
    start = left.frame_index + 1
    end = right.frame_index - 1
    total_steps = right.frame_index - left.frame_index
    left_center = bbox_center(left.bbox_xyxy)
    right_center = bbox_center(right.bbox_xyxy)
    left_width, left_height = bbox_size(left.bbox_xyxy)
    right_width, right_height = bbox_size(right.bbox_xyxy)
    frames: list[PlateTrackFrame] = []
    for frame_index in range(start, end + 1):
        alpha = (frame_index - left.frame_index) / total_steps
        center_x = linear(left_center[0], right_center[0], alpha)
        center_y = linear(left_center[1], right_center[1], alpha)
        box_width = math.exp(linear(math.log(left_width), math.log(right_width), alpha))
        box_height = math.exp(linear(math.log(left_height), math.log(right_height), alpha))
        bbox = center_size_to_bbox(center_x, center_y, box_width, box_height, width, height)
        if bbox is None:
            continue
        frames.append(
            PlateTrackFrame(
                frame_index=frame_index,
                track_id=track_id,
                bbox_xyxy=bbox,
                provenance=PROVENANCE_INTERPOLATION,
                score=None,
                status=NEEDS_REVIEW,
                source_candidate_ids=tuple(
                    item for item in (left.candidate_id, right.candidate_id) if item is not None
                ),
                anchor_frames=(left.frame_index, right.frame_index),
            )
        )
    return frames


def classify_track(
    track: PlateTrack,
    *,
    observed_frames: list[PlateTrackFrame],
    interpolated_frames: list[PlateTrackFrame],
    config: PlateTrackingConfig,
) -> tuple[str, list[str]]:
    observations = track.sorted_observations
    observed_count = len(observations)
    if observed_count < config.min_review_observations:
        return REJECTED_NOISE, ["isolated_plate_proposal"]

    reasons: list[str] = []
    span = max(1, track.last_frame - track.first_frame + 1)
    density = observed_count / span
    unresolved_gaps = [
        decision for decision in track.gap_decisions if decision.decision == "needs_review"
    ]
    center_jump = max_center_jump_norm(observations)
    scale_jump = max_scale_jump(observations)
    if observed_count < config.min_accepted_observations:
        reasons.append("insufficient_observations_for_auto_accept")
    if density < config.min_accepted_observation_density:
        reasons.append("low_observation_density")
    if unresolved_gaps:
        reasons.append("unresolved_internal_gap")
    if center_jump > config.max_accepted_center_jump_norm:
        reasons.append("excessive_center_jump")
    if scale_jump > config.max_accepted_scale_jump:
        reasons.append("excessive_scale_change")
    if possible_static_overlay(
        [bbox_center(item.bbox_xyxy) for item in observations],
        observations,
        config,
    ):
        reasons.append("possible_static_overlay_false_positive")
    if reasons:
        return NEEDS_REVIEW, reasons
    if not observed_frames and not interpolated_frames:
        return REJECTED_NOISE, ["empty_track"]
    return ACCEPTED_TRACK, ["stable_temporal_plate_track"]


def build_frame_track_index(
    tracks: list[PlateTrack],
    *,
    frame_count: int,
) -> dict[int, list[PlateTrackFrame]]:
    frame_tracks: dict[int, list[PlateTrackFrame]] = {
        frame_index: [] for frame_index in range(frame_count)
    }
    for track in tracks:
        for frame in track.frames:
            if 0 <= frame.frame_index < frame_count:
                frame_tracks[frame.frame_index].append(frame)
    for frames in frame_tracks.values():
        frames.sort(key=lambda item: (item.track_id, item.provenance, item.bbox_xyxy))
    return frame_tracks


def build_review_intervals(tracks: list[PlateTrack]) -> list[ReviewInterval]:
    intervals: list[ReviewInterval] = []
    for track in tracks:
        for decision in track.gap_decisions:
            if decision.decision != "needs_review":
                continue
            intervals.append(
                ReviewInterval(
                    frame_start=decision.frame_start,
                    frame_end=decision.frame_end,
                    track_ids=(track.track_id,),
                    severity="blocking",
                    blocking=True,
                    reason_codes=tuple(decision.reasons),
                    explanation="Plate tracking could not safely bridge this detection gap.",
                )
            )
        if track.status == NEEDS_REVIEW:
            intervals.append(
                ReviewInterval(
                    frame_start=track.first_frame,
                    frame_end=track.last_frame,
                    track_ids=(track.track_id,),
                    severity="medium",
                    blocking=False,
                    reason_codes=tuple(sorted(set(track.review_reasons))),
                    explanation="This plate-like track needs review before it can be trusted.",
                )
            )
    return merge_review_intervals(intervals)


def merge_review_intervals(intervals: list[ReviewInterval]) -> list[ReviewInterval]:
    if not intervals:
        return []
    sorted_intervals = sorted(
        intervals,
        key=lambda item: (
            not item.blocking,
            item.frame_start,
            item.frame_end,
            item.track_ids,
        ),
    )
    merged: list[ReviewInterval] = []
    for interval in sorted_intervals:
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        if (
            previous.blocking == interval.blocking
            and previous.severity == interval.severity
            and previous.frame_end + 10 >= interval.frame_start
            and set(previous.track_ids) & set(interval.track_ids)
        ):
            merged[-1] = ReviewInterval(
                frame_start=min(previous.frame_start, interval.frame_start),
                frame_end=max(previous.frame_end, interval.frame_end),
                track_ids=tuple(sorted(set(previous.track_ids) | set(interval.track_ids))),
                severity=previous.severity,
                blocking=previous.blocking,
                reason_codes=tuple(sorted(set(previous.reason_codes) | set(interval.reason_codes))),
                explanation=previous.explanation,
            )
        else:
            merged.append(interval)
    return sorted(merged, key=lambda item: (not item.blocking, item.frame_start, item.frame_end))


def detect_scene_cuts_for_track_gaps(
    source_video: Path,
    tracks: list[PlateTrack],
    *,
    config: PlateTrackingConfig,
) -> set[tuple[int, int]]:
    gap_pairs = sorted(
        {
            (left.frame_index, right.frame_index)
            for track in tracks
            for left, right in zip(
                track.sorted_observations, track.sorted_observations[1:], strict=False
            )
            if right.frame_index - left.frame_index > 1
        }
    )
    if not gap_pairs:
        return set()
    capture = cv.VideoCapture(str(source_video))
    if not capture.isOpened():
        return set()
    scene_cuts: set[tuple[int, int]] = set()
    try:
        for left_frame, right_frame in gap_pairs:
            left_image = read_frame(capture, left_frame)
            right_image = read_frame(capture, right_frame)
            if left_image is None or right_image is None:
                continue
            delta = mean_luma_delta(left_image, right_image)
            if delta > config.scene_cut_luma_delta:
                scene_cuts.add((left_frame, right_frame))
    finally:
        capture.release()
    return scene_cuts


def write_tracks_jsonl(
    path: Path,
    frame_tracks: dict[int, list[PlateTrackFrame]],
    frame_count: int,
    *,
    detector_frame_indices: list[int] | None = None,
) -> None:
    available_detector_frames = None
    if detector_frame_indices is not None:
        available_detector_frames = {int(index) for index in detector_frame_indices}
    with path.open("w", encoding="utf-8") as file:
        for frame_index in range(frame_count):
            record = {
                "detector_frame_available": (
                    True
                    if available_detector_frames is None
                    else frame_index in available_detector_frames
                ),
                "frame_index": frame_index,
                "tracks": [track.to_dict() for track in frame_tracks.get(frame_index, [])],
            }
            file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def build_track_summary(
    result: PlateTrackingResult,
    *,
    config: PlateTrackingConfig,
    elapsed_seconds: float,
) -> dict[str, Any]:
    frames = [frame for frame_tracks in result.frame_tracks.values() for frame in frame_tracks]
    status_counts = {
        ACCEPTED_TRACK: sum(1 for track in result.tracks if track.status == ACCEPTED_TRACK),
        NEEDS_REVIEW: sum(1 for track in result.tracks if track.status == NEEDS_REVIEW),
        REJECTED_NOISE: sum(1 for track in result.tracks if track.status == REJECTED_NOISE),
    }
    gap_lengths = [
        decision.length for decision in result.gap_decisions if decision.decision == "needs_review"
    ]
    return {
        "configuration": config.to_dict(),
        "configuration_hash": config.configuration_hash(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "gap_distribution": gap_distribution(
            [decision.length for decision in result.gap_decisions]
        ),
        "max_association_gap_frames": result.max_association_gap_frames,
        "max_safe_interpolation_gap_frames": result.max_safe_interpolation_gap_frames,
        "processing_fps": round(result.frames_processed / elapsed_seconds, 6)
        if elapsed_seconds > 0
        else 0.0,
        "provenance_counts": provenance_counts(frames),
        "rejected_proposal_count": len(result.rejected_proposals),
        "review_interval_count": len(result.review_intervals),
        "schema_version": SCHEMA_VERSION,
        "status_counts": status_counts,
        "track_count": len(result.tracks),
        "tracks": [track.to_summary_dict() for track in result.tracks],
        "unresolved_gap_count": len(gap_lengths),
        "unresolved_gap_distribution": gap_distribution(gap_lengths),
    }


def build_tracking_manifest(
    *,
    source_video: Path,
    detections_path: Path,
    source_sha256: str,
    detections_sha256: str,
    source_frame_count: int | None,
    frames_processed: int,
    width: int,
    height: int,
    fps: float,
    config: PlateTrackingConfig,
    result: PlateTrackingResult,
    elapsed_seconds: float,
    detection_span_report: dict[str, Any],
) -> dict[str, Any]:
    track_summary = build_track_summary(result, config=config, elapsed_seconds=elapsed_seconds)
    return {
        "created_at": media.utc_timestamp(),
        "detections_sha256": detections_sha256,
        "detections_path": media.portable_source_path(detections_path),
        "detection_frame_span": detection_span_report,
        "detector_configuration_hash": config.detector_configuration_hash,
        "detector_id": config.detector_id,
        "first_processed_frame": detection_span_report["first_processed_frame"],
        "frames_processed": frames_processed,
        "full_source_processed": detection_span_report["full_source_processed"],
        "input_height": height,
        "input_width": width,
        "last_processed_frame": detection_span_report["last_processed_frame"],
        "missing_processed_ranges": detection_span_report["missing_processed_ranges"],
        "nominal_fps": round(fps, 6),
        "processing_fps": track_summary["processing_fps"],
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": source_frame_count,
        "source_sha256": source_sha256,
        "source_video": media.portable_source_path(source_video),
        "tracking_configuration": config.to_dict(),
        "tracking_configuration_hash": config.configuration_hash(),
        "tracking_summary": {
            key: value
            for key, value in track_summary.items()
            if key not in {"configuration", "tracks"}
        },
    }


def write_processing_log(
    path: Path,
    manifest: dict[str, Any],
    result: PlateTrackingResult,
) -> None:
    lines = [
        "Plate Tracking and Gap Recovery v1",
        f"created_at: {manifest['created_at']}",
        f"source_video: {manifest['source_video']}",
        f"detections: {manifest['detections_path']}",
        f"tracking_configuration_hash: {manifest['tracking_configuration_hash']}",
        f"tracks: {len(result.tracks)}",
        f"review_intervals: {len(result.review_intervals)}",
        f"rejected_proposals: {len(result.rejected_proposals)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tracking_overlay(
    source_video: Path,
    output_path: Path,
    result: PlateTrackingResult,
    *,
    width: int,
    height: int,
    fps: float,
    frames_processed: int,
    show_rejected_overlay: bool,
) -> None:
    temp_path = output_path.with_name(output_path.stem + ".raw.mp4")
    capture = cv.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source_video}")
    writer = cv.VideoWriter(
        str(temp_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create overlay video: {temp_path}")
    rejected_by_frame: dict[int, list[RejectedProposal]] = {}
    for proposal in result.rejected_proposals:
        rejected_by_frame.setdefault(proposal.frame_index, []).append(proposal)
    try:
        frame_index = 0
        while frame_index < frames_processed:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(
                draw_tracking_overlay(
                    frame,
                    frame_index,
                    result.frame_tracks.get(frame_index, []),
                    rejected_by_frame.get(frame_index, []) if show_rejected_overlay else [],
                )
            )
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    transcode_h264(temp_path, output_path)
    temp_path.unlink(missing_ok=True)


def draw_tracking_overlay(
    frame: np.ndarray,
    frame_index: int,
    track_frames: list[PlateTrackFrame],
    rejected: list[RejectedProposal] | None = None,
) -> np.ndarray:
    output = frame.copy()
    cv.putText(
        output,
        f"frame {frame_index}",
        (12, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    for track_frame in track_frames:
        color = overlay_color(track_frame)
        x1, y1, x2, y2 = track_frame.bbox_xyxy
        cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{format_track_uid(track_frame.track_id)} "
            f"{short_status(track_frame.status)} {short_provenance(track_frame.provenance)}"
        )
        cv.putText(
            output,
            label,
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv.LINE_AA,
        )
    for proposal in rejected or []:
        if proposal.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = proposal.bbox_xyxy
        cv.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return output


def write_track_gallery(
    source_video: Path,
    output_path: Path,
    result: PlateTrackingResult,
    width: int,
    height: int,
) -> None:
    asset_dir = output_path.parent / "track_gallery_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(source_video))
    thumbnails: dict[int, list[str]] = {}
    if capture.isOpened():
        try:
            for track in result.tracks:
                frame_indexes = representative_track_frames(track)
                thumbnails[track.track_id] = []
                for frame_index in frame_indexes:
                    image = read_frame(capture, frame_index)
                    if image is None:
                        continue
                    frames = [item for item in track.frames if item.frame_index == frame_index]
                    rendered = draw_tracking_overlay(image, frame_index, frames)
                    thumb = resize_for_gallery(rendered, max_width=360)
                    filename = f"{format_track_uid(track.track_id)}_f{frame_index:04d}.jpg"
                    cv.imwrite(str(asset_dir / filename), thumb)
                    thumbnails[track.track_id].append(f"track_gallery_assets/{filename}")
        finally:
            capture.release()
    rows = []
    for track in result.tracks:
        summary = track.to_summary_dict()
        images = "".join(
            f'<img src="{path}" alt="{format_track_uid(track.track_id)} sample">'
            for path in thumbnails.get(track.track_id, [])
        )
        rows.append(
            "<article>"
            f"<h2>{format_track_uid(track.track_id)} - {track.status}</h2>"
            f"<p>Frames {track.first_frame}-{track.last_frame}; "
            f"{summary['observation_count']} observations; "
            f"{summary['interpolated_frame_count']} interpolated; "
            f"reasons: {', '.join(summary['review_reasons']) or 'none'}.</p>"
            f'<div class="thumbs">{images}</div>'
            "</article>"
        )
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Plate Track Gallery</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;line-height:1.4}"
        "article{border:1px solid #ddd;padding:12px;margin:0 0 16px}"
        ".thumbs{display:flex;gap:8px;flex-wrap:wrap}img{max-width:360px;height:auto}</style>"
        "</head><body><h1>Plate Track Gallery</h1>"
        f"<p>Source: {source_video.name}; frame size {width}x{height}.</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")


def write_gap_contact_sheet(
    source_video: Path,
    output_path: Path,
    result: PlateTrackingResult,
    *,
    width: int,
    height: int,
) -> None:
    samples: list[int] = []
    for interval in result.review_intervals:
        if not interval.blocking:
            continue
        samples.extend(interval_samples(interval.frame_start, interval.frame_end))
    write_contact_sheet_for_frames(
        source_video,
        output_path,
        sorted(set(samples))[:18],
        result,
        width=width,
        height=height,
        empty_label="No unresolved gap frames",
    )


def write_false_track_contact_sheet(
    source_video: Path,
    output_path: Path,
    result: PlateTrackingResult,
    *,
    width: int,
    height: int,
) -> None:
    samples: list[int] = []
    for track in result.tracks:
        if (
            track.status == REJECTED_NOISE
            or "possible_static_overlay_false_positive" in track.review_reasons
        ):
            samples.extend(representative_track_frames(track))
    write_contact_sheet_for_frames(
        source_video,
        output_path,
        sorted(set(samples))[:18],
        result,
        width=width,
        height=height,
        empty_label="No rejected or static-overlay candidate tracks",
    )


def write_contact_sheet_for_frames(
    source_video: Path,
    output_path: Path,
    frame_indexes: list[int],
    result: PlateTrackingResult,
    *,
    width: int,
    height: int,
    empty_label: str,
) -> None:
    if not frame_indexes:
        blank = np.full((240, 420, 3), 245, dtype=np.uint8)
        cv.putText(blank, empty_label, (16, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        cv.imwrite(str(output_path), blank)
        return
    capture = cv.VideoCapture(str(source_video))
    tiles: list[np.ndarray] = []
    if capture.isOpened():
        try:
            for frame_index in frame_indexes:
                image = read_frame(capture, frame_index)
                if image is None:
                    continue
                rendered = draw_tracking_overlay(
                    image,
                    frame_index,
                    result.frame_tracks.get(frame_index, []),
                )
                tiles.append(label_tile(resize_for_gallery(rendered, max_width=320), frame_index))
        finally:
            capture.release()
    if not tiles:
        blank = np.full((240, 420, 3), 245, dtype=np.uint8)
        cv.putText(
            blank, "No frames decoded", (16, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2
        )
        cv.imwrite(str(output_path), blank)
        return
    sheet = tile_images(tiles, columns=3, fill_size=(320, max(tile.shape[0] for tile in tiles)))
    if sheet.shape[1] == 0 or sheet.shape[0] == 0:
        sheet = np.full((height // 4, width // 4, 3), 245, dtype=np.uint8)
    cv.imwrite(str(output_path), sheet)


def build_summary_counts(result: PlateTrackingResult) -> dict[str, int]:
    return {
        "accepted_tracks": sum(1 for track in result.tracks if track.status == ACCEPTED_TRACK),
        "interpolated_frames": sum(
            1
            for frame_tracks in result.frame_tracks.values()
            for frame in frame_tracks
            if frame.provenance == PROVENANCE_INTERPOLATION
        ),
        "needs_review_tracks": sum(1 for track in result.tracks if track.status == NEEDS_REVIEW),
        "rejected_noise_tracks": sum(
            1 for track in result.tracks if track.status == REJECTED_NOISE
        ),
        "review_intervals": len(result.review_intervals),
        "tracks": len(result.tracks),
    }


def gap_distribution(lengths: list[int]) -> dict[str, int]:
    return {
        "1_to_3": sum(1 for length in lengths if 1 <= length <= 3),
        "4_to_10": sum(1 for length in lengths if 4 <= length <= 10),
        "11_to_20": sum(1 for length in lengths if 11 <= length <= 20),
        "over_20": sum(1 for length in lengths if length > 20),
    }


def representative_track_frames(track: PlateTrack) -> list[int]:
    frames = sorted({frame.frame_index for frame in track.frames})
    if not frames:
        return []
    return sorted({frames[0], frames[len(frames) // 2], frames[-1]})


def interval_samples(start: int, end: int) -> list[int]:
    if end < start:
        return []
    mid = (start + end) // 2
    return sorted({start, mid, end})


def resize_for_gallery(frame: np.ndarray, *, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv.resize(frame, (max_width, max(1, round(height * scale))), interpolation=cv.INTER_AREA)


def label_tile(tile: np.ndarray, frame_index: int) -> np.ndarray:
    output = tile.copy()
    cv.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), -1)
    cv.putText(
        output,
        f"frame {frame_index}",
        (8, 18),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return output


def tile_images(tiles: list[np.ndarray], *, columns: int, fill_size: tuple[int, int]) -> np.ndarray:
    tile_width, tile_height = fill_size
    rows = math.ceil(len(tiles) / columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        resized = cv.resize(tile, (tile_width, tile_height), interpolation=cv.INTER_AREA)
        row = index // columns
        col = index % columns
        y = row * tile_height
        x = col * tile_width
        canvas[y : y + tile_height, x : x + tile_width] = resized
    return canvas


def transcode_h264(input_path: Path, output_path: Path) -> None:
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def ffmpeg_path() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    return executable


def read_frame(capture: cv.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    return frame if ok else None


def mean_luma_delta(left: np.ndarray, right: np.ndarray) -> float:
    small_left = cv.resize(left, (64, 36), interpolation=cv.INTER_AREA)
    small_right = cv.resize(right, (64, 36), interpolation=cv.INTER_AREA)
    left_gray = cv.cvtColor(small_left, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(small_right, cv.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(left_gray.astype(np.float32) - right_gray.astype(np.float32))))


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def detection_frame_count(path: Path) -> int:
    max_frame = -1
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            max_frame = max(max_frame, int(json.loads(line)["frame_index"]))
    return max_frame + 1


def seconds_to_frames(seconds: float, fps: float) -> int:
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    return max(0, int(round(seconds * fps)))


def optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def format_track_uid(track_id: int) -> str:
    return f"{TRACK_ID_PREFIX}_{track_id:04d}"


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    width, height = bbox_size(bbox)
    return width * height


def bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    width, height = bbox_size(bbox)
    return math.hypot(width, height)


def mean_bbox_diagonal(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    return max(1.0, (bbox_diagonal(left) + bbox_diagonal(right)) / 2.0)


def center_distance_norm(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    return euclidean(bbox_center(left), bbox_center(right)) / mean_bbox_diagonal(left, right)


def euclidean(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def bbox_scale_ratio(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_width, left_height = bbox_size(left)
    right_width, right_height = bbox_size(right)
    width_ratio = max(left_width, right_width) / max(1, min(left_width, right_width))
    height_ratio = max(left_height, right_height) / max(1, min(left_height, right_height))
    return max(width_ratio, height_ratio)


def linear(left: float, right: float, alpha: float) -> float:
    return left + (right - left) * alpha


def center_size_to_bbox(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    bbox = (
        int(round(center_x - width / 2.0)),
        int(round(center_y - height / 2.0)),
        int(round(center_x + width / 2.0)),
        int(round(center_y + height / 2.0)),
    )
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def max_internal_gap(frames: list[int]) -> int:
    if len(frames) < 2:
        return 0
    return max(right - left - 1 for left, right in zip(frames, frames[1:], strict=False))


def max_center_jump_norm(observations: list[PlateObservation]) -> float:
    if len(observations) < 2:
        return 0.0
    return max(
        center_distance_norm(left.bbox_xyxy, right.bbox_xyxy)
        for left, right in zip(observations, observations[1:], strict=False)
    )


def max_scale_jump(observations: list[PlateObservation]) -> float:
    if len(observations) < 2:
        return 1.0
    return max(
        bbox_scale_ratio(left.bbox_xyxy, right.bbox_xyxy)
        for left, right in zip(observations, observations[1:], strict=False)
    )


def possible_static_overlay(
    centers: list[tuple[float, float]],
    observations: list[PlateObservation],
    config: PlateTrackingConfig,
) -> bool:
    if len(observations) < config.static_overlay_min_observations:
        return False
    diagonals = [bbox_diagonal(item.bbox_xyxy) for item in observations]
    if not diagonals:
        return False
    mean_diag = max(1.0, mean(diagonals))
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    std_norm = math.hypot(population_std(xs), population_std(ys)) / mean_diag
    return std_norm <= config.static_overlay_max_center_std_norm


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def provenance_counts(frames: list[PlateTrackFrame]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        counts[frame.provenance] = counts.get(frame.provenance, 0) + 1
    return dict(sorted(counts.items()))


def overlay_color(track_frame: PlateTrackFrame) -> tuple[int, int, int]:
    if track_frame.status == REJECTED_NOISE:
        return (0, 0, 255)
    if track_frame.provenance == PROVENANCE_INTERPOLATION:
        return (255, 255, 0)
    if track_frame.status == NEEDS_REVIEW:
        return (0, 165, 255)
    return (0, 255, 0)


def short_status(status: str) -> str:
    return {
        ACCEPTED_TRACK: "accepted",
        NEEDS_REVIEW: "review",
        REJECTED_NOISE: "rejected",
    }.get(status, status)


def short_provenance(provenance: str) -> str:
    return {
        PROVENANCE_DETECTOR: "obs",
        PROVENANCE_INTERPOLATION: "interp",
        PROVENANCE_PREDICTION: "pred",
        PROVENANCE_REVIEWED_CORRECTION: "reviewed",
    }.get(provenance, provenance)
