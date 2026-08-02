from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2 as cv
import numpy as np

from redactor import media
from redactor.detection import rational_to_float
from redactor.models import TrackBox

SCHEMA_VERSION = "1.0"

DEFAULT_STRONG_THRESHOLD = 0.30
DEFAULT_WEAK_THRESHOLD = 0.20
DEFAULT_MIN_OBSERVED_HITS = 3
DEFAULT_MIN_STRONG_HITS = 2
DEFAULT_CONFIRMATION_WINDOW = 5
DEFAULT_MAX_GAP_SECONDS = 0.30
DEFAULT_BACKWARD_SECONDS = 1.00
DEFAULT_MIN_AUTO_MEDIAN_SCORE = 0.30
DEFAULT_MAX_AUTO_SYNTHETIC_FRACTION = 0.40
DEFAULT_MIN_AUTO_AREA_FRACTION = 0.0005
DEFAULT_MAX_AUTO_AREA_FRACTION = 0.08
DEFAULT_MAX_TAIL_PREDICTION_SECONDS = 0.10
DEFAULT_ASSOCIATION_IOU_THRESHOLD = 0.15
DEFAULT_ASSOCIATION_CENTER_DISTANCE = 0.75
MIN_ASSOCIATION_AREA_RATIO = 0.50
MAX_ASSOCIATION_AREA_RATIO = 2.00
REVIEW_WINDOW_MERGE_GAP_FRAMES = 15

OBSERVED_PROVENANCES = {"observed_strong", "observed_weak", "manual"}
AUTO_CONFIRMED = "auto_confirmed"
REVIEW_CANDIDATE = "review_candidate"
REJECTED = "rejected"


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float | None
    provenance: str
    action: str | None = None
    detector_id: str | None = None
    candidate_id: str | None = None


@dataclass
class Track:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    action: str | None = None
    lifecycle_state: str = "tentative"
    quality_status: str = REJECTED
    render_eligible: bool = False
    quality_gate_failures: list[str] = field(default_factory=list)

    @property
    def sorted_observations(self) -> list[TrackObservation]:
        return sorted(self.observations, key=lambda item: item.frame_index)

    @property
    def first_frame(self) -> int:
        return self.sorted_observations[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.sorted_observations[-1].frame_index

    @property
    def last_bbox(self) -> tuple[int, int, int, int]:
        return self.sorted_observations[-1].bbox_xyxy

    def add_observation(self, observation: TrackObservation) -> None:
        self.observations.append(observation)
        if observation.action is not None:
            self.action = observation.action
        if len(self.observations) > 1:
            self.lifecycle_state = "active"


@dataclass(frozen=True)
class ReviewFlag:
    type: str
    frame_start: int
    frame_end: int
    track_id: int | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "reason": self.reason,
            "track_id": self.track_id,
            "type": self.type,
        }


@dataclass(frozen=True)
class ReviewWindow:
    frame_start: int
    frame_end: int
    track_ids: tuple[int, ...]
    flag_types: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "flag_types": list(self.flag_types),
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "reasons": list(self.reasons),
            "track_ids": list(self.track_ids),
        }


@dataclass(frozen=True)
class TrackQuality:
    track_id: int
    lifecycle_state: str
    quality_status: str
    render_eligible: bool
    first_frame: int
    last_frame: int
    observed_hits: int
    strong_hits: int
    weak_hits: int
    interpolated_boxes: int
    predicted_boxes: int
    median_observed_score: float | None
    median_area_fraction: float
    synthetic_fraction: float
    quality_gate_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "first_frame": self.first_frame,
            "interpolated_boxes": self.interpolated_boxes,
            "last_frame": self.last_frame,
            "lifecycle_state": self.lifecycle_state,
            "median_area_fraction": round(self.median_area_fraction, 8),
            "median_observed_score": (
                None if self.median_observed_score is None else round(self.median_observed_score, 6)
            ),
            "observed_hits": self.observed_hits,
            "predicted_boxes": self.predicted_boxes,
            "quality_gate_failures": list(self.quality_gate_failures),
            "quality_status": self.quality_status,
            "render_eligible": self.render_eligible,
            "strong_hits": self.strong_hits,
            "synthetic_fraction": round(self.synthetic_fraction, 6),
            "track_id": self.track_id,
            "weak_hits": self.weak_hits,
        }


@dataclass
class TrackingResult:
    frame_boxes: dict[int, list[TrackBox]]
    review_flags: list[ReviewFlag]
    review_windows: list[ReviewWindow]
    track_qualities: list[TrackQuality]
    total_track_proposals: int
    auto_confirmed_track_ids: set[int]
    review_candidate_track_ids: set[int]
    rejected_track_ids: set[int]
    strong_observations: int
    weak_observations: int
    interpolated_frames: int
    predicted_frames: int
    manual_seed_count: int
    maximum_filled_gap_frames: int


@dataclass(frozen=True)
class QualitySettings:
    min_observed_hits: int
    min_strong_hits: int
    confirmation_window: int
    min_auto_median_score: float
    max_auto_synthetic_fraction: float
    min_auto_area_fraction: float
    max_auto_area_fraction: float


@dataclass(frozen=True)
class AssociationSettings:
    iou_threshold: float
    center_distance_threshold: float
    max_frame_distance: int


def run_tracking(
    input_video: Path,
    *,
    detections_path: Path,
    output_dir: Path,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    min_observed_hits: int = DEFAULT_MIN_OBSERVED_HITS,
    min_strong_hits: int = DEFAULT_MIN_STRONG_HITS,
    confirmation_window: int = DEFAULT_CONFIRMATION_WINDOW,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    backward_seconds: float = DEFAULT_BACKWARD_SECONDS,
    min_auto_median_score: float = DEFAULT_MIN_AUTO_MEDIAN_SCORE,
    max_auto_synthetic_fraction: float = DEFAULT_MAX_AUTO_SYNTHETIC_FRACTION,
    min_auto_area_fraction: float = DEFAULT_MIN_AUTO_AREA_FRACTION,
    max_auto_area_fraction: float = DEFAULT_MAX_AUTO_AREA_FRACTION,
    max_tail_prediction_seconds: float = DEFAULT_MAX_TAIL_PREDICTION_SECONDS,
    association_iou_threshold: float = DEFAULT_ASSOCIATION_IOU_THRESHOLD,
    association_center_distance: float = DEFAULT_ASSOCIATION_CENTER_DISTANCE,
    manual_overrides: Path | None = None,
    overwrite: bool = False,
    show_rejected_overlay: bool = False,
) -> dict[str, object]:
    validate_tracking_options(
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        min_observed_hits=min_observed_hits,
        min_strong_hits=min_strong_hits,
        confirmation_window=confirmation_window,
        max_gap_seconds=max_gap_seconds,
        backward_seconds=backward_seconds,
        min_auto_median_score=min_auto_median_score,
        max_auto_synthetic_fraction=max_auto_synthetic_fraction,
        min_auto_area_fraction=min_auto_area_fraction,
        max_auto_area_fraction=max_auto_area_fraction,
        max_tail_prediction_seconds=max_tail_prediction_seconds,
        association_iou_threshold=association_iou_threshold,
        association_center_distance=association_center_distance,
    )
    ensure_output_dir(output_dir, overwrite=overwrite)

    source_info = media.probe_media(input_video)
    frame_rate = source_info.video_stream.nominal_frame_rate
    fps = rational_to_float(frame_rate) or 30.0
    source_video_frame_count = _optional_int(source_info.video_stream.frame_count)
    detection_frame_count = detection_record_frame_count(detections_path)
    frame_count = detection_frame_count
    max_gap_frames = seconds_to_frames(max_gap_seconds, fps)
    backward_frames = seconds_to_frames(backward_seconds, fps)
    max_tail_prediction_frames = seconds_to_frames(max_tail_prediction_seconds, fps)

    started_at = time.perf_counter()
    detections = load_detection_records(detections_path, weak_threshold=weak_threshold)
    manual_seeds = load_manual_seeds(manual_overrides)
    quality_settings = QualitySettings(
        min_observed_hits=min_observed_hits,
        min_strong_hits=min_strong_hits,
        confirmation_window=confirmation_window,
        min_auto_median_score=min_auto_median_score,
        max_auto_synthetic_fraction=max_auto_synthetic_fraction,
        min_auto_area_fraction=min_auto_area_fraction,
        max_auto_area_fraction=max_auto_area_fraction,
    )
    association_settings = AssociationSettings(
        iou_threshold=association_iou_threshold,
        center_distance_threshold=association_center_distance,
        max_frame_distance=max(confirmation_window, max_gap_frames + 1, backward_frames, 1),
    )
    result = build_tracks(
        detections,
        manual_seeds=manual_seeds,
        frame_count=frame_count,
        strong_threshold=strong_threshold,
        max_gap_frames=max_gap_frames,
        max_tail_prediction_frames=max_tail_prediction_frames,
        fps=fps,
        width=source_info.video_stream.width,
        height=source_info.video_stream.height,
        quality_settings=quality_settings,
        association_settings=association_settings,
    )
    frames_processed = write_tracking_outputs(
        input_video,
        output_dir=output_dir,
        frame_boxes=result.frame_boxes,
        track_qualities=result.track_qualities,
        frame_count=frame_count,
        fps=fps,
        width=source_info.video_stream.width,
        height=source_info.video_stream.height,
        show_rejected_overlay=show_rejected_overlay,
    )
    elapsed_seconds = time.perf_counter() - started_at
    summary = build_summary(
        input_video=input_video,
        detections_path=detections_path,
        source_sha256=source_info.sha256,
        detections_sha256=media.sha256_file(detections_path),
        source_video_frame_count=source_video_frame_count,
        detection_frame_count=detection_frame_count,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        min_observed_hits=min_observed_hits,
        min_strong_hits=min_strong_hits,
        confirmation_window=confirmation_window,
        max_gap_seconds=max_gap_seconds,
        max_tail_prediction_seconds=max_tail_prediction_seconds,
        min_auto_median_score=min_auto_median_score,
        max_auto_synthetic_fraction=max_auto_synthetic_fraction,
        min_auto_area_fraction=min_auto_area_fraction,
        max_auto_area_fraction=max_auto_area_fraction,
        association_iou_threshold=association_iou_threshold,
        association_center_distance=association_center_distance,
        frames_processed=frames_processed,
        result=result,
        elapsed_seconds=elapsed_seconds,
    )
    write_json(output_dir / "summary.json", summary)
    write_review_flags(output_dir / "review_flags.json", result.review_flags)
    write_review_windows(output_dir / "review_windows.json", result.review_windows)
    write_track_quality(output_dir / "track_quality.json", result.track_qualities)
    return summary


def validate_tracking_options(
    *,
    strong_threshold: float,
    weak_threshold: float,
    min_observed_hits: int,
    min_strong_hits: int,
    confirmation_window: int,
    max_gap_seconds: float,
    backward_seconds: float,
    min_auto_median_score: float,
    max_auto_synthetic_fraction: float,
    min_auto_area_fraction: float,
    max_auto_area_fraction: float,
    max_tail_prediction_seconds: float,
    association_iou_threshold: float,
    association_center_distance: float,
) -> None:
    if not 0 <= weak_threshold <= 1:
        raise ValueError("weak-threshold must be between 0 and 1")
    if not 0 <= strong_threshold <= 1:
        raise ValueError("strong-threshold must be between 0 and 1")
    if weak_threshold > strong_threshold:
        raise ValueError("weak-threshold must be less than or equal to strong-threshold")
    if min_observed_hits <= 0:
        raise ValueError("min-observed-hits must be greater than 0")
    if min_strong_hits < 0:
        raise ValueError("min-strong-hits cannot be negative")
    if confirmation_window <= 0:
        raise ValueError("confirmation-window must be greater than 0")
    if min_observed_hits > confirmation_window:
        raise ValueError("min-observed-hits cannot exceed confirmation-window")
    if min_strong_hits > confirmation_window:
        raise ValueError("min-strong-hits cannot exceed confirmation-window")
    if max_gap_seconds < 0:
        raise ValueError("max-gap-seconds cannot be negative")
    if backward_seconds < 0:
        raise ValueError("backward-seconds cannot be negative")
    if not 0 <= min_auto_median_score <= 1:
        raise ValueError("min-auto-median-score must be between 0 and 1")
    if not 0 <= max_auto_synthetic_fraction <= 1:
        raise ValueError("max-auto-synthetic-fraction must be between 0 and 1")
    if min_auto_area_fraction < 0:
        raise ValueError("min-auto-area-fraction cannot be negative")
    if max_auto_area_fraction <= 0:
        raise ValueError("max-auto-area-fraction must be greater than 0")
    if min_auto_area_fraction > max_auto_area_fraction:
        raise ValueError("min-auto-area-fraction cannot exceed max-auto-area-fraction")
    if max_tail_prediction_seconds < 0:
        raise ValueError("max-tail-prediction-seconds cannot be negative")
    if not 0 <= association_iou_threshold <= 1:
        raise ValueError("association-iou-threshold must be between 0 and 1")
    if association_center_distance <= 0:
        raise ValueError("association-center-distance must be greater than 0")


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def load_detection_records(
    path: Path,
    *,
    weak_threshold: float,
) -> dict[int, list[TrackObservation]]:
    records: dict[int, list[TrackObservation]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            observations = []
            for detection in record.get("detections", []):
                if detection.get("category") != "face":
                    continue
                score = float(detection["score"])
                if score < weak_threshold:
                    continue
                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        bbox_xyxy=_bbox_tuple(detection["bbox_xyxy"]),
                        score=score,
                        provenance="observed_weak",
                        detector_id=detection.get("detector_id"),
                        candidate_id=detection.get("candidate_id"),
                    )
                )
            records[frame_index] = observations
    return records


def load_manual_seeds(path: Path | None) -> dict[int, list[TrackObservation]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds: dict[int, list[TrackObservation]] = {}
    for seed in payload.get("seeds", []):
        frame_index = int(seed["frame_index"])
        observation = TrackObservation(
            frame_index=frame_index,
            bbox_xyxy=_bbox_tuple(seed["bbox_xyxy"]),
            score=None,
            provenance="manual",
            action=seed.get("action"),
        )
        seeds.setdefault(frame_index, []).append(observation)
    return seeds


def build_tracks(
    detections_by_frame: dict[int, list[TrackObservation]],
    *,
    manual_seeds: dict[int, list[TrackObservation]],
    frame_count: int,
    strong_threshold: float,
    max_gap_frames: int,
    max_tail_prediction_frames: int,
    fps: float,
    width: int,
    height: int,
    quality_settings: QualitySettings | None = None,
    association_settings: AssociationSettings | None = None,
) -> TrackingResult:
    if quality_settings is None:
        quality_settings = QualitySettings(
            DEFAULT_MIN_OBSERVED_HITS,
            DEFAULT_MIN_STRONG_HITS,
            DEFAULT_CONFIRMATION_WINDOW,
            DEFAULT_MIN_AUTO_MEDIAN_SCORE,
            DEFAULT_MAX_AUTO_SYNTHETIC_FRACTION,
            DEFAULT_MIN_AUTO_AREA_FRACTION,
            DEFAULT_MAX_AUTO_AREA_FRACTION,
        )
    if association_settings is None:
        association_settings = AssociationSettings(
            DEFAULT_ASSOCIATION_IOU_THRESHOLD,
            DEFAULT_ASSOCIATION_CENTER_DISTANCE,
            max(quality_settings.confirmation_window, max_gap_frames + 1, 1),
        )

    tracks = build_track_proposals(
        detections_by_frame,
        manual_seeds=manual_seeds,
        strong_threshold=strong_threshold,
        association_settings=association_settings,
    )
    manual_seed_count = sum(len(seeds) for seeds in manual_seeds.values())
    return finalize_tracks(
        tracks,
        frame_count=frame_count,
        max_gap_frames=max_gap_frames,
        max_tail_prediction_frames=max_tail_prediction_frames,
        half_second_frames=max(1, seconds_to_frames(0.5, fps)),
        scene_review_frames=max(1, seconds_to_frames(2.0, fps)),
        manual_seed_count=manual_seed_count,
        width=width,
        height=height,
        quality_settings=quality_settings,
        association_settings=association_settings,
    )


def build_track_proposals(
    detections_by_frame: dict[int, list[TrackObservation]],
    *,
    manual_seeds: dict[int, list[TrackObservation]],
    strong_threshold: float,
    association_settings: AssociationSettings,
) -> list[Track]:
    tracks: list[Track] = []
    next_track_id = 1
    frames = sorted(set(detections_by_frame) | set(manual_seeds))

    for frame_index in frames:
        observations = observations_for_frame(
            frame_index,
            detections_by_frame=detections_by_frame,
            manual_seeds=manual_seeds,
            strong_threshold=strong_threshold,
        )
        if not observations:
            continue

        matches = match_observations_to_tracks(tracks, observations, association_settings)
        matched_observation_indexes = set(matches)
        for observation_index, track in matches.items():
            track.add_observation(observations[observation_index])

        for observation_index, observation in enumerate(observations):
            if observation_index in matched_observation_indexes:
                continue
            track = Track(next_track_id)
            track.add_observation(observation)
            tracks.append(track)
            next_track_id += 1

    return tracks


def observations_for_frame(
    frame_index: int,
    *,
    detections_by_frame: dict[int, list[TrackObservation]],
    manual_seeds: dict[int, list[TrackObservation]],
    strong_threshold: float,
) -> list[TrackObservation]:
    observations = list(manual_seeds.get(frame_index, []))
    for observation in detections_by_frame.get(frame_index, []):
        provenance = "observed_strong" if observation.score >= strong_threshold else "observed_weak"
        observations.append(
            TrackObservation(
                frame_index=frame_index,
                bbox_xyxy=observation.bbox_xyxy,
                score=observation.score,
                provenance=provenance,
                action=observation.action,
                detector_id=observation.detector_id,
                candidate_id=observation.candidate_id,
            )
        )
    return sorted(observations, key=observation_sort_key)


def observation_sort_key(
    observation: TrackObservation,
) -> tuple[int, float, tuple[int, int, int, int]]:
    provenance_rank = {
        "manual": 0,
        "observed_strong": 1,
        "observed_weak": 2,
    }.get(observation.provenance, 3)
    return (provenance_rank, -(observation.score or 1.0), observation.bbox_xyxy)


def match_observations_to_tracks(
    tracks: list[Track],
    observations: list[TrackObservation],
    association_settings: AssociationSettings,
) -> dict[int, Track]:
    candidates: list[tuple[float, int, int, int, int]] = []
    for observation_index, observation in enumerate(observations):
        for track_index, track in enumerate(tracks):
            score = association_score(track.last_bbox, observation.bbox_xyxy, association_settings)
            if score is None:
                continue
            frame_gap = observation.frame_index - track.last_frame
            if frame_gap <= 0 or frame_gap > association_settings.max_frame_distance:
                continue
            candidates.append((score, -frame_gap, track.track_id, observation_index, track_index))

    assignments: dict[int, Track] = {}
    used_tracks: set[int] = set()
    for _, _, _, observation_index, track_index in sorted(candidates, reverse=True):
        if observation_index in assignments or track_index in used_tracks:
            continue
        assignments[observation_index] = tracks[track_index]
        used_tracks.add(track_index)
    return assignments


def finalize_tracks(
    tracks: list[Track],
    *,
    frame_count: int,
    max_gap_frames: int,
    max_tail_prediction_frames: int,
    half_second_frames: int,
    scene_review_frames: int,
    manual_seed_count: int,
    width: int,
    height: int,
    quality_settings: QualitySettings,
    association_settings: AssociationSettings,
) -> TrackingResult:
    frame_boxes: dict[int, list[TrackBox]] = {frame_index: [] for frame_index in range(frame_count)}
    review_flags = [
        ReviewFlag(
            "mandatory_scene_start_review",
            0,
            min(frame_count - 1, scene_review_frames - 1),
            None,
            "first two seconds require human review",
        ),
        ReviewFlag(
            "mandatory_scene_end_review",
            max(0, frame_count - scene_review_frames),
            max(0, frame_count - 1),
            None,
            "last two seconds require human review",
        ),
    ]
    track_qualities: list[TrackQuality] = []
    maximum_filled_gap_frames = 0

    for track in tracks:
        observed = track.sorted_observations
        track.lifecycle_state = "ended"
        failures = quality_failures_for_observations(
            observed,
            width=width,
            height=height,
            settings=quality_settings,
            association_settings=association_settings,
        )
        track.quality_gate_failures = failures
        track.quality_status = classify_quality_status(observed, failures)
        track.render_eligible = track.quality_status == AUTO_CONFIRMED

        interpolated_boxes, interpolation_flags, longest_gap = plan_internal_interpolation(
            track,
            max_gap_frames=max_gap_frames,
            association_settings=association_settings,
        )
        maximum_filled_gap_frames = max(maximum_filled_gap_frames, longest_gap)

        if (
            track.quality_status == AUTO_CONFIRMED
            and synthetic_fraction(len(observed), len(interpolated_boxes))
            > quality_settings.max_auto_synthetic_fraction
        ):
            track.quality_gate_failures.append("synthetic_fraction")
            track.quality_status = REVIEW_CANDIDATE
            track.render_eligible = False
            interpolated_boxes, interpolation_flags, longest_gap = plan_internal_interpolation(
                track,
                max_gap_frames=max_gap_frames,
                association_settings=association_settings,
            )
            maximum_filled_gap_frames = max(maximum_filled_gap_frames, longest_gap)

        predicted_boxes = plan_tail_prediction(
            track,
            frame_count=frame_count,
            max_tail_prediction_frames=max_tail_prediction_frames,
            max_synthetic_fraction=quality_settings.max_auto_synthetic_fraction,
            current_synthetic_count=len(interpolated_boxes),
        )

        for box in observed_boxes_for_track(track):
            append_box(frame_boxes, box)
            if box.provenance == "manual":
                review_flags.append(
                    ReviewFlag(
                        "manual_seed",
                        box.frame_index,
                        box.frame_index,
                        track.track_id,
                        "manual seed requires human verification",
                    )
                )

        for box in interpolated_boxes:
            append_box(frame_boxes, box)
        for box in predicted_boxes:
            append_box(frame_boxes, box)

        review_flags.extend(interpolation_flags)
        review_flags.extend(long_gap_flags(track, max_gap_frames=max_gap_frames))
        review_flags.extend(track_status_flags(track))
        review_flags.extend(edge_review_flags(track, frame_count, half_second_frames))
        if predicted_boxes:
            review_flags.append(
                ReviewFlag(
                    "predicted_span",
                    predicted_boxes[0].frame_index,
                    predicted_boxes[-1].frame_index,
                    track.track_id,
                    "short unmatched tail predicted for review",
                )
            )

        track_qualities.append(
            quality_summary_for_track(
                track,
                width=width,
                height=height,
                interpolated_count=len(interpolated_boxes),
                predicted_count=len(predicted_boxes),
            )
        )

    for boxes in frame_boxes.values():
        boxes.sort(key=lambda item: (item.track_id, item.frame_index, item.provenance))

    sorted_review_flags = sorted(
        review_flags,
        key=lambda item: (item.frame_start, item.frame_end, item.type, item.track_id or 0),
    )
    review_windows = aggregate_review_windows(sorted_review_flags)
    auto_confirmed = {
        quality.track_id for quality in track_qualities if quality.quality_status == AUTO_CONFIRMED
    }
    review_candidates = {
        quality.track_id
        for quality in track_qualities
        if quality.quality_status == REVIEW_CANDIDATE
    }
    rejected = {
        quality.track_id for quality in track_qualities if quality.quality_status == REJECTED
    }

    return TrackingResult(
        frame_boxes=frame_boxes,
        review_flags=sorted_review_flags,
        review_windows=review_windows,
        track_qualities=sorted(track_qualities, key=lambda item: item.track_id),
        total_track_proposals=len(tracks),
        auto_confirmed_track_ids=auto_confirmed,
        review_candidate_track_ids=review_candidates,
        rejected_track_ids=rejected,
        strong_observations=sum(
            1
            for boxes in frame_boxes.values()
            for box in boxes
            if box.provenance == "observed_strong"
        ),
        weak_observations=sum(
            1
            for boxes in frame_boxes.values()
            for box in boxes
            if box.provenance == "observed_weak"
        ),
        interpolated_frames=sum(
            1 for boxes in frame_boxes.values() for box in boxes if box.provenance == "interpolated"
        ),
        predicted_frames=sum(
            1 for boxes in frame_boxes.values() for box in boxes if box.provenance == "predicted"
        ),
        manual_seed_count=manual_seed_count,
        maximum_filled_gap_frames=maximum_filled_gap_frames,
    )


def quality_failures_for_observations(
    observations: list[TrackObservation],
    *,
    width: int,
    height: int,
    settings: QualitySettings,
    association_settings: AssociationSettings,
) -> list[str]:
    failures: list[str] = []
    observed_hits = len(observations)
    strong_hits = count_provenance(observations, "observed_strong")
    observed_scores = [item.score for item in observations if item.score is not None]
    median_score = median(observed_scores) if observed_scores else None
    median_area = median_area_fraction(observations, width=width, height=height)

    if observed_hits < settings.min_observed_hits:
        failures.append("insufficient_observed_hits")
    if strong_hits < settings.min_strong_hits:
        failures.append("insufficient_strong_hits")
    if not has_windowed_hits(
        observations, settings.min_observed_hits, settings.confirmation_window
    ):
        failures.append("observed_confirmation_window")
    strong_observations = [item for item in observations if item.provenance == "observed_strong"]
    if not has_windowed_hits(
        strong_observations, settings.min_strong_hits, settings.confirmation_window
    ):
        failures.append("strong_confirmation_window")
    if median_score is None or median_score < settings.min_auto_median_score:
        failures.append("low_median_observed_score")
    if median_area < settings.min_auto_area_fraction:
        failures.append("tiny_median_area")
    if median_area > settings.max_auto_area_fraction:
        failures.append("large_median_area")
    if not spatial_continuity_passes(observations, association_settings):
        failures.append("spatial_continuity")
    return failures


def classify_quality_status(observations: list[TrackObservation], failures: list[str]) -> str:
    if not failures:
        return AUTO_CONFIRMED
    has_manual = any(item.provenance == "manual" for item in observations)
    if "spatial_continuity" in failures:
        return REJECTED
    if len(observations) == 1 and not has_manual:
        return REJECTED
    return REVIEW_CANDIDATE


def observed_boxes_for_track(track: Track) -> list[TrackBox]:
    return [
        make_track_box(
            observation,
            track,
            provenance=observation.provenance,
            score=observation.score,
            bbox=observation.bbox_xyxy,
        )
        for observation in track.sorted_observations
    ]


def plan_internal_interpolation(
    track: Track,
    *,
    max_gap_frames: int,
    association_settings: AssociationSettings,
) -> tuple[list[TrackBox], list[ReviewFlag], int]:
    if track.quality_status == REJECTED:
        return [], [], 0
    interpolated: list[TrackBox] = []
    flags: list[ReviewFlag] = []
    longest_gap = 0
    observations = track.sorted_observations
    for left, right in zip(observations, observations[1:]):
        gap = right.frame_index - left.frame_index - 1
        if gap <= 0:
            continue
        if gap <= max_gap_frames and safe_spatial_transition(
            left.bbox_xyxy,
            right.bbox_xyxy,
            association_settings,
        ):
            longest_gap = max(longest_gap, gap)
            for frame_index in range(left.frame_index + 1, right.frame_index):
                fraction = (frame_index - left.frame_index) / (right.frame_index - left.frame_index)
                interpolated.append(
                    make_track_box(
                        left,
                        track,
                        provenance="interpolated",
                        score=None,
                        bbox=interpolate_bbox(left.bbox_xyxy, right.bbox_xyxy, fraction),
                        frame_index=frame_index,
                    )
                )
            flags.append(
                ReviewFlag(
                    "interpolated_span",
                    left.frame_index + 1,
                    right.frame_index - 1,
                    track.track_id,
                    "short internal gap filled by interpolation",
                )
            )
    return interpolated, flags, longest_gap


def plan_tail_prediction(
    track: Track,
    *,
    frame_count: int,
    max_tail_prediction_frames: int,
    max_synthetic_fraction: float,
    current_synthetic_count: int,
) -> list[TrackBox]:
    if track.quality_status != AUTO_CONFIRMED:
        return []
    observations = track.sorted_observations
    if len(observations) < 3 or max_tail_prediction_frames <= 0:
        return []
    remaining_synthetic_allowed = (
        max_allowed_synthetic_count(
            len(observations),
            max_synthetic_fraction,
        )
        - current_synthetic_count
    )
    if remaining_synthetic_allowed <= 0:
        return []

    last_observation = observations[-1]
    prediction_count = min(
        max_tail_prediction_frames,
        remaining_synthetic_allowed,
        max(0, frame_count - 1 - last_observation.frame_index),
    )
    return [
        make_track_box(
            last_observation,
            track,
            provenance="predicted",
            score=None,
            bbox=last_observation.bbox_xyxy,
            frame_index=frame_index,
        )
        for frame_index in range(
            last_observation.frame_index + 1,
            last_observation.frame_index + 1 + prediction_count,
        )
    ]


def long_gap_flags(track: Track, *, max_gap_frames: int) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []
    if track.quality_status == REJECTED:
        return flags
    for left, right in zip(track.sorted_observations, track.sorted_observations[1:]):
        gap = right.frame_index - left.frame_index - 1
        if gap <= max_gap_frames:
            continue
        flags.append(
            ReviewFlag(
                "long_gap",
                left.frame_index + 1,
                right.frame_index - 1,
                track.track_id,
                "gap exceeds configured maximum and was not filled",
            )
        )
        flags.append(
            ReviewFlag(
                "unresolved_interval",
                left.frame_index + 1,
                right.frame_index - 1,
                track.track_id,
                "interval could not be safely connected automatically",
            )
        )
    return flags


def track_status_flags(track: Track) -> list[ReviewFlag]:
    if track.quality_status == AUTO_CONFIRMED:
        return []
    flag_type = (
        "review_candidate_track" if track.quality_status == REVIEW_CANDIDATE else "rejected_track"
    )
    reason = (
        "tracklet requires analyst review before rendering"
        if track.quality_status == REVIEW_CANDIDATE
        else "tracklet failed conservative automatic quality gates"
    )
    return [
        ReviewFlag(
            flag_type,
            track.first_frame,
            track.last_frame,
            track.track_id,
            reason,
        )
    ]


def edge_review_flags(track: Track, frame_count: int, half_second_frames: int) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []
    if track.first_frame > half_second_frames:
        flags.append(
            ReviewFlag(
                "late_track_start",
                max(0, track.first_frame - half_second_frames),
                track.first_frame,
                track.track_id,
                "track begins more than half a second after scene start",
            )
        )
    if track.last_frame < frame_count - 1 - half_second_frames:
        flags.append(
            ReviewFlag(
                "early_track_end",
                track.last_frame,
                min(frame_count - 1, track.last_frame + half_second_frames),
                track.track_id,
                "track ends more than half a second before scene end",
            )
        )
    return flags


def make_track_box(
    observation: TrackObservation,
    track: Track,
    *,
    provenance: str,
    score: float | None,
    bbox: tuple[int, int, int, int],
    frame_index: int | None = None,
) -> TrackBox:
    return TrackBox(
        frame_index=observation.frame_index if frame_index is None else frame_index,
        track_id=track.track_id,
        bbox_xyxy=bbox,
        score=score,
        provenance=provenance,
        track_state=track.lifecycle_state,
        quality_status=track.quality_status,
        render_eligible=track.render_eligible,
        action=observation.action or track.action,
        detector_id=observation.detector_id,
        candidate_id=observation.candidate_id,
    )


def append_box(frame_boxes: dict[int, list[TrackBox]], box: TrackBox) -> None:
    if box.frame_index in frame_boxes:
        frame_boxes[box.frame_index].append(box)


def quality_summary_for_track(
    track: Track,
    *,
    width: int,
    height: int,
    interpolated_count: int,
    predicted_count: int,
) -> TrackQuality:
    observations = track.sorted_observations
    scores = [item.score for item in observations if item.score is not None]
    synthetic_count = interpolated_count + predicted_count
    return TrackQuality(
        track_id=track.track_id,
        lifecycle_state=track.lifecycle_state,
        quality_status=track.quality_status,
        render_eligible=track.render_eligible,
        first_frame=track.first_frame,
        last_frame=max(track.last_frame, track.last_frame + predicted_count),
        observed_hits=len(observations),
        strong_hits=count_provenance(observations, "observed_strong"),
        weak_hits=count_provenance(observations, "observed_weak"),
        interpolated_boxes=interpolated_count,
        predicted_boxes=predicted_count,
        median_observed_score=median(scores) if scores else None,
        median_area_fraction=median_area_fraction(observations, width=width, height=height),
        synthetic_fraction=synthetic_fraction(len(observations), synthetic_count),
        quality_gate_failures=tuple(sorted(track.quality_gate_failures)),
    )


def aggregate_review_windows(review_flags: list[ReviewFlag]) -> list[ReviewWindow]:
    windows: list[dict[str, Any]] = []
    for flag in review_flags:
        if (
            not windows
            or flag.frame_start > windows[-1]["frame_end"] + REVIEW_WINDOW_MERGE_GAP_FRAMES
        ):
            windows.append(
                {
                    "frame_start": flag.frame_start,
                    "frame_end": flag.frame_end,
                    "track_ids": set(),
                    "flag_types": set(),
                    "reasons": set(),
                }
            )
        window = windows[-1]
        window["frame_end"] = max(window["frame_end"], flag.frame_end)
        if flag.track_id is not None:
            window["track_ids"].add(flag.track_id)
        window["flag_types"].add(flag.type)
        window["reasons"].add(flag.reason)

    return [
        ReviewWindow(
            frame_start=int(window["frame_start"]),
            frame_end=int(window["frame_end"]),
            track_ids=tuple(sorted(window["track_ids"])),
            flag_types=tuple(sorted(window["flag_types"])),
            reasons=tuple(sorted(window["reasons"])),
        )
        for window in windows
    ]


def write_tracking_outputs(
    input_video: Path,
    *,
    output_dir: Path,
    frame_boxes: dict[int, list[TrackBox]],
    track_qualities: list[TrackQuality],
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    show_rejected_overlay: bool,
) -> int:
    capture = cv.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_video}")
    overlay_path = output_dir / "overlay.mp4"
    writer = cv.VideoWriter(
        str(overlay_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create overlay video: {overlay_path}")

    tracks_path = output_dir / "tracks.jsonl"
    sample_plan = build_contact_sheet_sample_plan(frame_boxes, track_qualities)
    sampled_crops: dict[int, list[np.ndarray]] = {
        track_id: [] for track_id in sample_plan.track_ids
    }
    frames_processed = 0
    try:
        with tracks_path.open("w", encoding="utf-8") as tracks_file:
            while frames_processed < frame_count:
                ok, frame = capture.read()
                if not ok:
                    break
                boxes = frame_boxes.get(frames_processed, [])
                write_frame_record(tracks_file, frames_processed, boxes)
                collect_contact_sheet_crops(frame, frames_processed, sample_plan, sampled_crops)
                writer.write(
                    draw_overlay(
                        frame,
                        frames_processed,
                        boxes,
                        show_rejected=show_rejected_overlay,
                    )
                )
                frames_processed += 1
    finally:
        capture.release()
        writer.release()

    write_track_contact_sheets(output_dir, sampled_crops)
    return frames_processed


@dataclass(frozen=True)
class ContactSheetSamplePlan:
    by_frame: dict[int, list[TrackBox]]
    track_ids: tuple[int, ...]


def build_contact_sheet_sample_plan(
    frame_boxes: dict[int, list[TrackBox]],
    track_qualities: list[TrackQuality],
) -> ContactSheetSamplePlan:
    eligible_track_ids = {
        quality.track_id
        for quality in track_qualities
        if quality.quality_status in {AUTO_CONFIRMED, REVIEW_CANDIDATE}
    }
    observed_by_track: dict[int, list[TrackBox]] = {track_id: [] for track_id in eligible_track_ids}
    for frame_index in sorted(frame_boxes):
        for box in frame_boxes[frame_index]:
            if box.track_id in eligible_track_ids and box.provenance in OBSERVED_PROVENANCES:
                observed_by_track[box.track_id].append(box)

    by_frame: dict[int, list[TrackBox]] = {}
    for track_id, boxes in observed_by_track.items():
        for box in sample_track_boxes(boxes):
            by_frame.setdefault(box.frame_index, []).append(box)
    return ContactSheetSamplePlan(by_frame=by_frame, track_ids=tuple(sorted(eligible_track_ids)))


def sample_track_boxes(boxes: list[TrackBox], max_samples: int = 5) -> list[TrackBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda item: item.frame_index)
    if len(boxes) <= max_samples:
        return boxes
    positions = [0, 0.25, 0.5, 0.75, 1.0]
    indexes = sorted({round((len(boxes) - 1) * position) for position in positions})
    return [boxes[index] for index in indexes]


def collect_contact_sheet_crops(
    frame: np.ndarray,
    frame_index: int,
    sample_plan: ContactSheetSamplePlan,
    sampled_crops: dict[int, list[np.ndarray]],
) -> None:
    for box in sample_plan.by_frame.get(frame_index, []):
        sampled_crops.setdefault(box.track_id, []).append(crop_for_contact_sheet(frame, box))


def crop_for_contact_sheet(frame: np.ndarray, box: TrackBox) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box.bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    pad = round(max(width, height) * 0.75)
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(frame_width, x2 + pad)
    crop_y2 = min(frame_height, y2 + pad)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)

    relative_x1 = x1 - crop_x1
    relative_y1 = y1 - crop_y1
    relative_x2 = x2 - crop_x1
    relative_y2 = y2 - crop_y1
    color = quality_color(box.quality_status)
    cv.rectangle(crop, (relative_x1, relative_y1), (relative_x2, relative_y2), color, 2)
    cv.putText(
        crop,
        f"T{box.track_id} f{box.frame_index}",
        (6, 18),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv.LINE_AA,
    )
    return letterbox(crop, tile_width=220, tile_height=170)


def letterbox(image: np.ndarray, *, tile_width: int, tile_height: int) -> np.ndarray:
    canvas = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(tile_width / max(width, 1), tile_height / max(height, 1))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv.resize(image, (resized_width, resized_height), interpolation=cv.INTER_AREA)
    x = (tile_width - resized_width) // 2
    y = (tile_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def write_track_contact_sheets(
    output_dir: Path, sampled_crops: dict[int, list[np.ndarray]]
) -> None:
    contact_dir = output_dir / "track_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    for track_id, crops in sampled_crops.items():
        if not crops:
            continue
        sheet = np.hstack(crops)
        cv.imwrite(str(contact_dir / f"track_{track_id:03d}.jpg"), sheet)


def write_frame_record(tracks_file, frame_index: int, boxes: list[TrackBox]) -> None:
    record = {
        "frame_index": frame_index,
        "track_boxes": [box.to_dict() for box in boxes],
    }
    tracks_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def write_review_flags(path: Path, review_flags: list[ReviewFlag]) -> None:
    write_json(
        path,
        {
            "review_flags": [flag.to_dict() for flag in review_flags],
            "schema_version": SCHEMA_VERSION,
        },
    )


def write_review_windows(path: Path, review_windows: list[ReviewWindow]) -> None:
    write_json(
        path,
        {
            "review_windows": [window.to_dict() for window in review_windows],
            "schema_version": SCHEMA_VERSION,
        },
    )


def write_track_quality(path: Path, track_qualities: list[TrackQuality]) -> None:
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "tracks": [
                quality.to_dict()
                for quality in sorted(track_qualities, key=lambda item: item.track_id)
            ],
        },
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(
    *,
    input_video: Path,
    detections_path: Path,
    source_sha256: str,
    detections_sha256: str,
    source_video_frame_count: int | None,
    detection_frame_count: int,
    strong_threshold: float,
    weak_threshold: float,
    min_observed_hits: int,
    min_strong_hits: int,
    confirmation_window: int,
    max_gap_seconds: float,
    max_tail_prediction_seconds: float,
    min_auto_median_score: float,
    max_auto_synthetic_fraction: float,
    min_auto_area_fraction: float,
    max_auto_area_fraction: float,
    association_iou_threshold: float,
    association_center_distance: float,
    frames_processed: int,
    result: TrackingResult,
    elapsed_seconds: float,
) -> dict[str, object]:
    processing_fps = frames_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "association_center_distance": association_center_distance,
        "association_iou_threshold": association_iou_threshold,
        "auto_confirmed_tracks": len(result.auto_confirmed_track_ids),
        "confirmed_tracks": len(result.auto_confirmed_track_ids),
        "created_at": media.utc_timestamp(),
        "detections_filename": detections_path.name,
        "detections_sha256": detections_sha256,
        "diagnostic_overlay_has_audio": False,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "ended_tracks": result.total_track_proposals,
        "frames_processed": frames_processed,
        "input_detection_frame_count": detection_frame_count,
        "interpolated_frames": result.interpolated_frames,
        "manual_seed_count": result.manual_seed_count,
        "max_auto_area_fraction": max_auto_area_fraction,
        "max_auto_synthetic_fraction": max_auto_synthetic_fraction,
        "max_gap_seconds": max_gap_seconds,
        "max_tail_prediction_seconds": max_tail_prediction_seconds,
        "maximum_filled_gap_frames": result.maximum_filled_gap_frames,
        "min_auto_area_fraction": min_auto_area_fraction,
        "min_auto_median_score": min_auto_median_score,
        "min_observed_hits": min_observed_hits,
        "min_strong_hits": min_strong_hits,
        "opencv_version": cv.__version__,
        "predicted_frames": result.predicted_frames,
        "processing_fps": round(processing_fps, 6),
        "processed_frame_count": frames_processed,
        "quality_gate_failure_counts": quality_gate_failure_counts(result.track_qualities),
        "rejected_tracks": len(result.rejected_track_ids),
        "review_candidate_tracks": len(result.review_candidate_track_ids),
        "review_flags": len(result.review_flags),
        "review_windows": len(result.review_windows),
        "schema_version": SCHEMA_VERSION,
        "source_filename": input_video.name,
        "source_video_frame_count": source_video_frame_count,
        "source_sha256": source_sha256,
        "strong_observations": result.strong_observations,
        "strong_threshold": strong_threshold,
        "total_track_proposals": result.total_track_proposals,
        "weak_observations": result.weak_observations,
        "weak_threshold": weak_threshold,
        "confirmation_window": confirmation_window,
    }


def quality_gate_failure_counts(track_qualities: list[TrackQuality]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for quality in track_qualities:
        for failure in quality.quality_gate_failures:
            counts[failure] = counts.get(failure, 0) + 1
    return dict(sorted(counts.items()))


def draw_overlay(
    frame: np.ndarray,
    frame_index: int,
    boxes: list[TrackBox],
    *,
    show_rejected: bool = False,
) -> np.ndarray:
    output = frame.copy()
    cv.putText(
        output,
        f"frame {frame_index}",
        (12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    for box in boxes:
        if box.quality_status == REJECTED and not show_rejected:
            continue
        x1, y1, x2, y2 = box.bbox_xyxy
        color = overlay_color(box)
        thickness = 3 if box.render_eligible else 2
        cv.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        label = f"T{box.track_id} {box.quality_status} {box.provenance}"
        cv.putText(
            output,
            label,
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv.LINE_AA,
        )
    return output


def overlay_color(box: TrackBox) -> tuple[int, int, int]:
    if box.provenance == "interpolated":
        return (255, 180, 0)
    if box.provenance == "predicted":
        return (255, 0, 255)
    if box.provenance == "manual":
        return (255, 255, 255)
    return quality_color(box.quality_status)


def quality_color(quality_status: str) -> tuple[int, int, int]:
    return {
        AUTO_CONFIRMED: (0, 220, 0),
        REVIEW_CANDIDATE: (0, 180, 255),
        REJECTED: (120, 120, 120),
    }.get(quality_status, (200, 200, 200))


def association_score(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    settings: AssociationSettings | None = None,
) -> float | None:
    if settings is None:
        settings = AssociationSettings(
            DEFAULT_ASSOCIATION_IOU_THRESHOLD,
            DEFAULT_ASSOCIATION_CENTER_DISTANCE,
            DEFAULT_CONFIRMATION_WINDOW,
        )
    if not area_ratio_is_safe(left, right):
        return None
    iou_value = bbox_iou(left, right)
    distance = normalized_center_distance(left, right)
    if iou_value < settings.iou_threshold and distance > settings.center_distance_threshold:
        return None
    return iou_value + max(0.0, settings.center_distance_threshold - distance)


def safe_spatial_transition(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    settings: AssociationSettings,
) -> bool:
    return association_score(left, right, settings) is not None


def spatial_continuity_passes(
    observations: list[TrackObservation],
    settings: AssociationSettings,
) -> bool:
    if len(observations) < 2:
        return True
    return all(
        safe_spatial_transition(left.bbox_xyxy, right.bbox_xyxy, settings)
        for left, right in zip(observations, observations[1:])
    )


def area_ratio_is_safe(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    left_area = bbox_area(left)
    right_area = bbox_area(right)
    ratio = right_area / max(left_area, 1)
    return MIN_ASSOCIATION_AREA_RATIO <= ratio <= MAX_ASSOCIATION_AREA_RATIO


def bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = bbox_area(left)
    right_area = bbox_area(right)
    return intersection / (left_area + right_area - intersection)


def normalized_center_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    left_diag = math.hypot(left[2] - left[0], left[3] - left[1])
    right_diag = math.hypot(right[2] - right[0], right[3] - right[1])
    return distance / max((left_diag + right_diag) / 2, 1.0)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def median_area_fraction(
    observations: list[TrackObservation],
    *,
    width: int,
    height: int,
) -> float:
    if not observations:
        return 0.0
    image_area = max(width * height, 1)
    return float(median(bbox_area(item.bbox_xyxy) / image_area for item in observations))


def has_windowed_hits(
    observations: list[TrackObservation],
    required_hits: int,
    window_frames: int,
) -> bool:
    if required_hits <= 0:
        return True
    if len(observations) < required_hits:
        return False
    frames = sorted(item.frame_index for item in observations)
    for index, frame_start in enumerate(frames):
        frame_end = frame_start + window_frames - 1
        if sum(1 for frame in frames[index:] if frame <= frame_end) >= required_hits:
            return True
    return False


def count_provenance(observations: list[TrackObservation], provenance: str) -> int:
    return sum(1 for item in observations if item.provenance == provenance)


def synthetic_fraction(observed_count: int, synthetic_count: int) -> float:
    total = observed_count + synthetic_count
    if total == 0:
        return 0.0
    return synthetic_count / total


def max_allowed_synthetic_count(observed_count: int, max_synthetic_fraction: float) -> int:
    if max_synthetic_fraction >= 1:
        return observed_count
    return math.floor(
        (max_synthetic_fraction * observed_count) / max(1 - max_synthetic_fraction, 1e-9)
    )


def interpolate_bbox(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    fraction: float,
) -> tuple[int, int, int, int]:
    return tuple(round(left[index] + (right[index] - left[index]) * fraction) for index in range(4))


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(0, math.ceil(seconds * fps))


def detection_record_frame_count(detections_path: Path) -> int:
    max_frame = -1
    with detections_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                max_frame = max(max_frame, int(json.loads(line)["frame_index"]))
    return max_frame + 1


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _bbox_tuple(values: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)
