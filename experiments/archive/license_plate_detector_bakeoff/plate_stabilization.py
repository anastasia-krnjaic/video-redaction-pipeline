from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff
from redactor import media

SCHEMA_VERSION = "plate_proposal_stabilization_v1"
TRACK_CATEGORY = "license_plate"


@dataclass(frozen=True)
class PlateStabilizationConfig:
    proposal_threshold: float = 0.20
    min_area_fraction: float = 0.00005
    max_area_fraction: float = 0.02
    min_aspect_ratio: float = 0.40
    max_aspect_ratio: float = 6.00
    duplicate_nms_iou_threshold: float = 0.45
    association_iou_threshold: float = 0.10
    association_center_distance_factor: float = 1.50
    association_scale_ratio_limit: float = 3.00
    max_frame_gap: int = 5
    interpolation_max_gap: int = 3
    accepted_min_observations: int = 20
    accepted_min_span_frames: int = 20
    accepted_min_observation_density: float = 0.55
    accepted_max_internal_gap: int = 5
    accepted_min_mean_score: float = 0.30
    review_min_observations: int = 4
    review_min_span_frames: int = 4
    review_min_observation_density: float = 0.20
    max_tracks_drawn: int = 80

    def __post_init__(self) -> None:
        if self.proposal_threshold < 0:
            raise ValueError("proposal_threshold cannot be negative")
        if self.min_area_fraction <= 0:
            raise ValueError("min_area_fraction must be positive")
        if self.max_area_fraction <= self.min_area_fraction:
            raise ValueError("max_area_fraction must be greater than min_area_fraction")
        if self.min_aspect_ratio <= 0:
            raise ValueError("min_aspect_ratio must be positive")
        if self.max_aspect_ratio < self.min_aspect_ratio:
            raise ValueError("max_aspect_ratio must be greater than or equal to min_aspect_ratio")
        if not 0 <= self.duplicate_nms_iou_threshold <= 1:
            raise ValueError("duplicate_nms_iou_threshold must be in [0, 1]")
        if not 0 <= self.association_iou_threshold <= 1:
            raise ValueError("association_iou_threshold must be in [0, 1]")
        if self.association_center_distance_factor <= 0:
            raise ValueError("association_center_distance_factor must be positive")
        if self.association_scale_ratio_limit < 1:
            raise ValueError("association_scale_ratio_limit must be at least 1")
        if self.max_frame_gap < 0:
            raise ValueError("max_frame_gap cannot be negative")
        if self.interpolation_max_gap < 0:
            raise ValueError("interpolation_max_gap cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_max_internal_gap": self.accepted_max_internal_gap,
            "accepted_min_mean_score": self.accepted_min_mean_score,
            "accepted_min_observation_density": self.accepted_min_observation_density,
            "accepted_min_observations": self.accepted_min_observations,
            "accepted_min_span_frames": self.accepted_min_span_frames,
            "association_center_distance_factor": self.association_center_distance_factor,
            "association_iou_threshold": self.association_iou_threshold,
            "association_scale_ratio_limit": self.association_scale_ratio_limit,
            "duplicate_nms_iou_threshold": self.duplicate_nms_iou_threshold,
            "interpolation_max_gap": self.interpolation_max_gap,
            "max_area_fraction": self.max_area_fraction,
            "max_aspect_ratio": self.max_aspect_ratio,
            "max_frame_gap": self.max_frame_gap,
            "max_tracks_drawn": self.max_tracks_drawn,
            "min_area_fraction": self.min_area_fraction,
            "min_aspect_ratio": self.min_aspect_ratio,
            "proposal_threshold": self.proposal_threshold,
            "review_min_observation_density": self.review_min_observation_density,
            "review_min_observations": self.review_min_observations,
            "review_min_span_frames": self.review_min_span_frames,
        }


@dataclass(frozen=True)
class PlateProposal:
    detector_id: str
    frame_index: int
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    source_detection_index: int | None = None
    proposal_stage: str = "post_nms"
    inference_time_ms: float = 0.0
    geometry: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox_xyxy)

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / float(self.height)

    @classmethod
    def from_detection(
        cls,
        detection: plate_bakeoff.PlateDetection,
        *,
        proposal_stage: str = "post_nms",
        rejection_reasons: tuple[str, ...] = (),
    ) -> PlateProposal:
        return cls(
            detector_id=detection.detector_id,
            frame_index=detection.frame_index,
            score=detection.score,
            bbox_xyxy=detection.bbox_xyxy,
            source_detection_index=detection.source_detection_index,
            proposal_stage=proposal_stage,
            inference_time_ms=detection.inference_time_ms,
            geometry=detection.geometry,
            preprocessing=detection.preprocessing,
            postprocessing=detection.postprocessing,
            rejection_reasons=rejection_reasons,
        )

    def with_rejection_reasons(self, reasons: list[str]) -> PlateProposal:
        return PlateProposal(
            detector_id=self.detector_id,
            frame_index=self.frame_index,
            score=self.score,
            bbox_xyxy=self.bbox_xyxy,
            source_detection_index=self.source_detection_index,
            proposal_stage=self.proposal_stage,
            inference_time_ms=self.inference_time_ms,
            geometry=self.geometry,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing,
            rejection_reasons=tuple(sorted(set(reasons))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect_ratio": round(self.aspect_ratio, 6),
            "bbox_xyxy": list(self.bbox_xyxy),
            "category": TRACK_CATEGORY,
            "detector_id": self.detector_id,
            "frame_index": self.frame_index,
            "geometry": self.geometry,
            "inference_time_ms": round(self.inference_time_ms, 6),
            "postprocessing": self.postprocessing,
            "preprocessing": self.preprocessing,
            "proposal_stage": self.proposal_stage,
            "rejection_reasons": list(self.rejection_reasons),
            "score": self.score,
            "source_detection_index": self.source_detection_index,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "aspect_ratio": round(self.aspect_ratio, 6),
            "bbox_xyxy": list(self.bbox_xyxy),
            "detector_id": self.detector_id,
            "frame_index": self.frame_index,
            "proposal_stage": self.proposal_stage,
            "rejection_reasons": list(self.rejection_reasons),
            "score": self.score,
            "source_detection_index": self.source_detection_index,
        }


@dataclass(frozen=True)
class PlateTrackObservation:
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float | None
    provenance: str
    source_proposal: PlateProposal | None = None

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox_xyxy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "frame_index": self.frame_index,
            "provenance": self.provenance,
            "score": self.score,
            "source_proposal": (
                self.source_proposal.to_compact_dict() if self.source_proposal else None
            ),
        }


@dataclass
class PlateTrack:
    track_id: str
    observations: list[PlateTrackObservation] = field(default_factory=list)
    duplicate_proposals: int = 0
    source_proposal_count: int = 0
    status: str = "unclassified"
    review_reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def add_observation(self, proposal: PlateProposal) -> None:
        self.observations.append(
            PlateTrackObservation(
                frame_index=proposal.frame_index,
                bbox_xyxy=proposal.bbox_xyxy,
                score=proposal.score,
                provenance="observed_plate_proposal",
                source_proposal=proposal,
            )
        )
        self.source_proposal_count += 1

    @property
    def observed(self) -> list[PlateTrackObservation]:
        return [item for item in self.observations if item.provenance == "observed_plate_proposal"]

    @property
    def first_frame(self) -> int:
        return min(item.frame_index for item in self.observations)

    @property
    def last_frame(self) -> int:
        return max(item.frame_index for item in self.observations)

    @property
    def span_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    def observation_at(self, frame_index: int) -> PlateTrackObservation | None:
        return next((item for item in self.observations if item.frame_index == frame_index), None)

    def sorted_observations(self) -> list[PlateTrackObservation]:
        return sorted(self.observations, key=lambda item: (item.frame_index, item.provenance))

    def to_dict(self) -> dict[str, Any]:
        summary = summarize_track(self)
        return {
            **summary,
            "observations": [item.to_dict() for item in self.sorted_observations()],
        }


def load_proposals_from_detection_jsonl(
    path: Path,
) -> tuple[dict[int, list[PlateProposal]], dict[int, list[PlateProposal]]]:
    post_nms, pre_nms = plate_bakeoff.load_detection_jsonl_with_pre_nms(path)
    return detection_records_to_proposals(post_nms, pre_nms)


def detection_records_to_proposals(
    post_nms: dict[int, list[plate_bakeoff.PlateDetection]],
    pre_nms: dict[int, list[plate_bakeoff.PlateDetection]],
) -> tuple[dict[int, list[PlateProposal]], dict[int, list[PlateProposal]]]:
    return (
        {
            frame_index: [
                PlateProposal.from_detection(item, proposal_stage="post_nms") for item in detections
            ]
            for frame_index, detections in post_nms.items()
        },
        {
            frame_index: [
                PlateProposal.from_detection(item, proposal_stage="pre_nms") for item in detections
            ]
            for frame_index, detections in pre_nms.items()
        },
    )


def filter_and_suppress_proposals(
    proposals_by_frame: dict[int, list[PlateProposal]],
    *,
    frame_width: int,
    frame_height: int,
    config: PlateStabilizationConfig,
) -> tuple[dict[int, list[PlateProposal]], dict[int, list[PlateProposal]], dict[str, int]]:
    accepted: dict[int, list[PlateProposal]] = {}
    rejected: dict[int, list[PlateProposal]] = {}
    reason_counts: dict[str, int] = {}
    for frame_index in sorted(proposals_by_frame):
        valid: list[PlateProposal] = []
        rejected[frame_index] = []
        for proposal in proposals_by_frame[frame_index]:
            reasons = validate_proposal_geometry(
                proposal,
                frame_width=frame_width,
                frame_height=frame_height,
                config=config,
            )
            if reasons:
                rejected_item = proposal.with_rejection_reasons(reasons)
                rejected[frame_index].append(rejected_item)
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            else:
                valid.append(proposal)
        deduped, duplicates = suppress_duplicate_proposals(
            valid,
            iou_threshold=config.duplicate_nms_iou_threshold,
        )
        accepted[frame_index] = deduped
        for duplicate in duplicates:
            rejected[frame_index].append(duplicate.with_rejection_reasons(["duplicate_suppressed"]))
        if duplicates:
            reason_counts["duplicate_suppressed"] = reason_counts.get(
                "duplicate_suppressed", 0
            ) + len(duplicates)
    return accepted, rejected, reason_counts


def validate_proposal_geometry(
    proposal: PlateProposal,
    *,
    frame_width: int,
    frame_height: int,
    config: PlateStabilizationConfig,
) -> list[str]:
    reasons: list[str] = []
    x1, y1, x2, y2 = proposal.bbox_xyxy
    if proposal.score < config.proposal_threshold:
        reasons.append("below_proposal_threshold")
    if x2 <= x1 or y2 <= y1:
        return [*reasons, "invalid_or_zero_area_box"]
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        reasons.append("box_outside_frame")
    area_fraction = ((x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)
    if area_fraction < config.min_area_fraction:
        reasons.append("area_too_small")
    if area_fraction > config.max_area_fraction:
        reasons.append("area_too_large")
    aspect_ratio = (x2 - x1) / float(y2 - y1)
    if aspect_ratio < config.min_aspect_ratio:
        reasons.append("aspect_ratio_too_small")
    if aspect_ratio > config.max_aspect_ratio:
        reasons.append("aspect_ratio_too_large")
    return reasons


def suppress_duplicate_proposals(
    proposals: list[PlateProposal],
    *,
    iou_threshold: float,
) -> tuple[list[PlateProposal], list[PlateProposal]]:
    kept: list[PlateProposal] = []
    duplicates: list[PlateProposal] = []
    for proposal in sorted(proposals, key=proposal_sort_key):
        if any(bbox_iou(proposal.bbox_xyxy, item.bbox_xyxy) >= iou_threshold for item in kept):
            duplicates.append(proposal)
        else:
            kept.append(proposal)
    return kept, duplicates


def associate_plate_tracks(
    proposals_by_frame: dict[int, list[PlateProposal]],
    *,
    config: PlateStabilizationConfig,
) -> list[PlateTrack]:
    tracks: list[PlateTrack] = []
    active: list[PlateTrack] = []
    next_id = 1
    for frame_index in sorted(proposals_by_frame):
        proposals = sorted(proposals_by_frame[frame_index], key=proposal_sort_key)
        active = [
            track
            for track in active
            if frame_index - last_observed(track).frame_index <= config.max_frame_gap + 1
        ]
        used_proposals: set[int] = set()
        used_tracks: set[int] = set()
        for proposal_index, proposal in enumerate(proposals):
            best_match: tuple[float, int] | None = None
            for track_index, track in enumerate(active):
                if track_index in used_tracks:
                    continue
                score = association_score(track, proposal, config=config)
                if score is None:
                    continue
                if best_match is None or score < best_match[0]:
                    best_match = (score, track_index)
            if best_match is None:
                continue
            _, track_index = best_match
            active[track_index].add_observation(proposals[proposal_index])
            used_proposals.add(proposal_index)
            used_tracks.add(track_index)
        for proposal_index, proposal in enumerate(proposals):
            if proposal_index in used_proposals:
                continue
            track = PlateTrack(track_id=f"plate_track_{next_id:04d}")
            next_id += 1
            track.add_observation(proposal)
            tracks.append(track)
            active.append(track)
    for track in tracks:
        interpolate_short_gaps(track, config=config)
        classify_track(track, config=config)
    return sorted(tracks, key=lambda item: (item.first_frame, item.track_id))


def association_score(
    track: PlateTrack,
    proposal: PlateProposal,
    *,
    config: PlateStabilizationConfig,
) -> float | None:
    previous = last_observed(track)
    gap = proposal.frame_index - previous.frame_index
    if gap <= 0 or gap > config.max_frame_gap + 1:
        return None
    iou = bbox_iou(previous.bbox_xyxy, proposal.bbox_xyxy)
    center_distance = euclidean_distance(previous.center, proposal.center)
    diagonal = max(1.0, bbox_diagonal(previous.bbox_xyxy))
    normalized_center_distance = center_distance / diagonal
    scale_ratio = bbox_scale_ratio(previous.bbox_xyxy, proposal.bbox_xyxy)
    center_gate = normalized_center_distance <= config.association_center_distance_factor * gap
    iou_gate = iou >= config.association_iou_threshold
    scale_gate = scale_ratio <= config.association_scale_ratio_limit
    if scale_gate and (iou_gate or center_gate):
        # Lower is better. Prefer overlap, then small normalized movement, then high score.
        return (
            (1.0 - iou)
            + normalized_center_distance
            + (gap - 1) * 0.25
            + (1.0 - min(1.0, proposal.score)) * 0.01
        )
    return None


def interpolate_short_gaps(
    track: PlateTrack,
    *,
    config: PlateStabilizationConfig,
) -> None:
    observed = sorted(track.observed, key=lambda item: item.frame_index)
    additions: list[PlateTrackObservation] = []
    for left, right in zip(observed, observed[1:]):
        gap = right.frame_index - left.frame_index - 1
        if gap <= 0 or gap > config.interpolation_max_gap:
            continue
        for offset in range(1, gap + 1):
            alpha = offset / float(gap + 1)
            additions.append(
                PlateTrackObservation(
                    frame_index=left.frame_index + offset,
                    bbox_xyxy=interpolate_bbox(left.bbox_xyxy, right.bbox_xyxy, alpha),
                    score=None,
                    provenance="short_gap_interpolation",
                    source_proposal=None,
                )
            )
    if additions:
        track.observations.extend(additions)
        track.observations.sort(key=lambda item: (item.frame_index, item.provenance))


def classify_track(track: PlateTrack, *, config: PlateStabilizationConfig) -> None:
    summary = summarize_track(track)
    rejection_reasons: list[str] = []
    review_reasons: list[str] = []
    if summary["observed_frame_count"] <= 1:
        rejection_reasons.append("single_frame_or_ephemeral_track")
    if summary["observed_frame_count"] < config.review_min_observations:
        rejection_reasons.append("too_few_observations")
    if summary["span_frames"] < config.review_min_span_frames:
        rejection_reasons.append("too_short_for_review")
    if summary["observation_density"] < config.review_min_observation_density:
        rejection_reasons.append("low_observation_density")

    accepted_failures = []
    if summary["observed_frame_count"] < config.accepted_min_observations:
        accepted_failures.append("accepted_min_observations_not_met")
    if summary["span_frames"] < config.accepted_min_span_frames:
        accepted_failures.append("accepted_min_span_not_met")
    if summary["observation_density"] < config.accepted_min_observation_density:
        accepted_failures.append("accepted_density_not_met")
    if summary["max_internal_gap"] > config.accepted_max_internal_gap:
        accepted_failures.append("accepted_gap_limit_exceeded")
    if summary["mean_score"] is None or summary["mean_score"] < config.accepted_min_mean_score:
        accepted_failures.append("accepted_mean_score_not_met")
    if summary["max_center_jump_normalized"] > config.association_center_distance_factor * 2:
        accepted_failures.append("unstable_center_motion")
    if summary["max_scale_ratio"] > config.association_scale_ratio_limit:
        accepted_failures.append("unstable_box_scale")

    if not rejection_reasons and not accepted_failures:
        track.status = "accepted_plate_track"
        track.review_reasons = []
        track.rejection_reasons = []
        return
    if rejection_reasons:
        track.status = "rejected_noise_track"
        track.rejection_reasons = rejection_reasons
        track.review_reasons = []
        return
    review_reasons.extend(accepted_failures)
    track.status = "review_candidate"
    track.review_reasons = review_reasons
    track.rejection_reasons = []


def summarize_track(track: PlateTrack) -> dict[str, Any]:
    observed = sorted(track.observed, key=lambda item: item.frame_index)
    if not observed:
        raise ValueError("track must contain at least one observed proposal")
    frame_numbers = [item.frame_index for item in observed]
    gaps = [right - left - 1 for left, right in zip(frame_numbers, frame_numbers[1:])]
    scores = [item.score for item in observed if item.score is not None]
    center_jumps = [
        euclidean_distance(left.center, right.center) for left, right in zip(observed, observed[1:])
    ]
    normalized_jumps = [
        euclidean_distance(left.center, right.center) / max(1.0, bbox_diagonal(left.bbox_xyxy))
        for left, right in zip(observed, observed[1:])
    ]
    scale_ratios = [
        bbox_scale_ratio(left.bbox_xyxy, right.bbox_xyxy)
        for left, right in zip(observed, observed[1:])
    ]
    areas = [bbox_area(item.bbox_xyxy) for item in observed]
    span = max(frame_numbers) - min(frame_numbers) + 1
    return {
        "box_area_mean": round(sum(areas) / len(areas), 6),
        "box_size_stability": round(max(scale_ratios), 6) if scale_ratios else 1.0,
        "center_motion_stability": (round(max(normalized_jumps), 6) if normalized_jumps else 0.0),
        "duplicate_proposals": track.duplicate_proposals,
        "first_frame": min(frame_numbers),
        "last_frame": max(frame_numbers),
        "max_center_jump": round(max(center_jumps), 6) if center_jumps else 0.0,
        "max_center_jump_normalized": (
            round(max(normalized_jumps), 6) if normalized_jumps else 0.0
        ),
        "max_internal_gap": max(gaps) if gaps else 0,
        "max_scale_ratio": round(max(scale_ratios), 6) if scale_ratios else 1.0,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
        "min_score": round(min(scores), 6) if scores else None,
        "observation_density": round(len(observed) / span, 6),
        "observed_frame_count": len(observed),
        "provenance_counts": provenance_counts(track.observations),
        "rejection_reasons": list(track.rejection_reasons),
        "review_reasons": list(track.review_reasons),
        "source_proposal_count": track.source_proposal_count,
        "span_frames": span,
        "status": track.status,
        "synthetic_interpolation_count": len(track.observations) - len(observed),
        "track_id": track.track_id,
    }


def run_stabilization(
    records_path: Path,
    *,
    frame_width: int,
    frame_height: int,
    config: PlateStabilizationConfig,
) -> dict[str, Any]:
    post_nms, pre_nms = load_proposals_from_detection_jsonl(records_path)
    return stabilize_proposals(
        post_nms,
        pre_nms_proposals_by_frame=pre_nms,
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
    )


def stabilize_proposals(
    post_nms_proposals_by_frame: dict[int, list[PlateProposal]],
    *,
    pre_nms_proposals_by_frame: dict[int, list[PlateProposal]] | None = None,
    frame_width: int,
    frame_height: int,
    config: PlateStabilizationConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    filtered, rejected, rejection_counts = filter_and_suppress_proposals(
        post_nms_proposals_by_frame,
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
    )
    tracks = associate_plate_tracks(filtered, config=config)
    elapsed = time.perf_counter() - started
    processed_frames = sorted(post_nms_proposals_by_frame)
    return {
        "config": config.to_dict(),
        "elapsed_seconds": elapsed,
        "filtered_proposals_by_frame": filtered,
        "pre_nms_proposals_by_frame": pre_nms_proposals_by_frame or {},
        "processed_frame_count": len(processed_frames),
        "processed_frames": processed_frames,
        "proposals_by_frame": post_nms_proposals_by_frame,
        "rejected_proposals_by_frame": rejected,
        "rejection_reason_counts": rejection_counts,
        "schema_version": SCHEMA_VERSION,
        "tracks": tracks,
    }


def track_frames_by_status(
    tracks: list[PlateTrack],
    statuses: set[str],
) -> dict[int, list[PlateTrackObservation]]:
    frames: dict[int, list[PlateTrackObservation]] = {}
    for track in tracks:
        if track.status not in statuses:
            continue
        for observation in track.sorted_observations():
            frames.setdefault(observation.frame_index, []).append(observation)
    return frames


def write_tracks_jsonl(tracks: list[PlateTrack], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for track in sorted(tracks, key=lambda item: item.track_id):
            file.write(json.dumps(track.to_dict(), sort_keys=True) + "\n")


def write_rejected_proposals_jsonl(
    rejected_by_frame: dict[int, list[PlateProposal]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for frame_index in sorted(rejected_by_frame):
            file.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "rejected_proposals": [
                            proposal.to_compact_dict()
                            for proposal in rejected_by_frame[frame_index]
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def draw_track_overlay(
    frame: np.ndarray,
    frame_index: int,
    tracks: list[PlateTrack],
    *,
    track_entries: list[tuple[PlateTrack, PlateTrackObservation]] | None = None,
    include_rejected: bool = False,
    target_roi: tuple[int, int, int, int] | None = None,
    max_tracks_drawn: int = 80,
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
    if target_roi is not None:
        x1, y1, x2, y2 = target_roi
        cv.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv.putText(
            output,
            "scored target ROI",
            (x1, max(24, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv.LINE_AA,
        )
    entries = (
        track_entries
        if track_entries is not None
        else [
            (track, observation)
            for track in tracks
            if include_rejected or track.status != "rejected_noise_track"
            for observation in [track.observation_at(frame_index)]
            if observation is not None
        ]
    )
    drawn = 0
    for track, observation in entries:
        color = status_color(track.status, observation.provenance)
        x1, y1, x2, y2 = observation.bbox_xyxy
        cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{track.track_id} {track.status.replace('_', ' ')}"
        if observation.score is not None:
            label += f" {observation.score:.2f}"
        if observation.provenance != "observed_plate_proposal":
            label += " interp"
        cv.putText(
            output,
            label[:54],
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )
        drawn += 1
        if drawn >= max_tracks_drawn:
            break
    return output


def write_tracks_overlay_video(
    video_path: Path,
    tracks: list[PlateTrack],
    output_path: Path,
    *,
    target_roi: tuple[int, int, int, int] | None = None,
    include_rejected: bool = False,
    max_tracks_drawn: int = 80,
) -> dict[str, Any]:
    info = media.probe_media(video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    width = info.video_stream.width
    height = info.video_stream.height
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create track overlay video: {output_path}")
    frame_entries = track_frame_index(tracks, include_rejected=include_rejected)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(
                draw_track_overlay(
                    frame,
                    frame_index,
                    tracks,
                    track_entries=frame_entries.get(frame_index, []),
                    include_rejected=include_rejected,
                    target_roi=target_roi,
                    max_tracks_drawn=max_tracks_drawn,
                )
            )
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return {
        "frames": frame_index,
        "output_path": media.portable_source_path(output_path, Path.cwd()),
        "output_sha256": media.sha256_file(output_path),
    }


def write_track_contact_sheet(
    video_path: Path,
    tracks: list[PlateTrack],
    output_path: Path,
    *,
    frames: list[int],
    target_roi: tuple[int, int, int, int] | None = None,
    columns: int = 4,
    max_tracks_drawn: int = 80,
) -> dict[str, Any]:
    info = media.probe_media(video_path)
    tile_width = 360
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    selected_frames = frames[:16]
    if not selected_frames:
        selected_frames = representative_track_frames(tracks)[:8]
    selected_frames = selected_frames or [0]
    rows = math.ceil(len(selected_frames) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    frame_entries = track_frame_index(tracks)
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        for index, frame_index in enumerate(selected_frames):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            overlay = draw_track_overlay(
                frame,
                frame_index,
                tracks,
                track_entries=frame_entries.get(frame_index, []),
                target_roi=target_roi,
                max_tracks_drawn=max_tracks_drawn,
            )
            resized = cv.resize(overlay, (tile_width, tile_height), interpolation=cv.INTER_AREA)
            row = index // columns
            column = index % columns
            y1 = row * tile_height
            x1 = column * tile_width
            sheet[y1 : y1 + tile_height, x1 : x1 + tile_width] = resized
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), sheet)
    return {
        "frames": selected_frames,
        "output_path": media.portable_source_path(output_path, Path.cwd()),
        "output_sha256": media.sha256_file(output_path),
    }


def write_track_gallery(
    video_path: Path,
    tracks: list[PlateTrack],
    output_dir: Path,
    *,
    title: str,
    max_thumbnails: int = 120,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    thumbnails: list[dict[str, Any]] = []
    try:
        for track in gallery_thumbnail_tracks(tracks, limit=max_thumbnails):
            frame_index = representative_frame_for_track(track)
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            overlay = draw_track_overlay(frame, frame_index, [track], include_rejected=True)
            path = assets_dir / f"{track.track_id}.jpg"
            cv.imwrite(str(path), cv.resize(overlay, (480, 270), interpolation=cv.INTER_AREA))
            thumbnails.append(
                {
                    "frame_index": frame_index,
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": media.sha256_file(path),
                    "track_id": track.track_id,
                }
            )
    finally:
        capture.release()
    html = build_track_gallery_html(title=title, tracks=tracks, thumbnails=thumbnails)
    index_path = output_dir / "track_gallery.html"
    index_path.write_text(html, encoding="utf-8")
    return {
        "max_thumbnails": max_thumbnails,
        "output_path": media.portable_source_path(index_path, Path.cwd()),
        "thumbnail_count": len(thumbnails),
        "total_tracks": len(tracks),
    }


def build_track_gallery_html(
    *,
    title: str,
    tracks: list[PlateTrack],
    thumbnails: list[dict[str, Any]],
) -> str:
    thumb_by_track = {item["track_id"]: item for item in thumbnails}
    cards = []
    for track in sorted(tracks, key=lambda item: (item.status, item.first_frame, item.track_id)):
        summary = summarize_track(track)
        thumb = thumb_by_track.get(track.track_id)
        image = (
            f'<img src="{thumb["path"]}" alt="{track.track_id} representative frame">'
            if thumb
            else '<div class="missing">No thumbnail</div>'
        )
        reasons = track.review_reasons or track.rejection_reasons or ["accepted"]
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    image,
                    f"<h2>{track.track_id}</h2>",
                    f'<p class="status">{summary["status"]}</p>',
                    (
                        f"<p>Frames {summary['first_frame']}-{summary['last_frame']} "
                        f"({summary['observed_frame_count']} observations, "
                        f"density {summary['observation_density']:.3f})</p>"
                    ),
                    f"<p>Mean score: {summary['mean_score']}</p>",
                    f"<p>Reasons: {', '.join(reasons)}</p>",
                    "</article>",
                ]
            )
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{escape_html(title)}</title>",
            "<style>",
            (
                "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
                "margin:24px;background:#f7f7f5;color:#181818}"
            ),
            "h1{font-size:28px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}",
            ".card{background:white;border:1px solid #ddd;border-radius:8px;padding:12px}",
            ".card img{width:100%;height:auto;border-radius:4px}",
            ".status{font-weight:700;text-transform:capitalize}",
            ".missing{height:160px;background:#eee;display:grid;place-items:center}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{escape_html(title)}</h1>",
            '<main class="grid">',
            *cards,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def track_frame_index(
    tracks: list[PlateTrack],
    *,
    include_rejected: bool = False,
) -> dict[int, list[tuple[PlateTrack, PlateTrackObservation]]]:
    frame_entries: dict[int, list[tuple[PlateTrack, PlateTrackObservation]]] = {}
    for track in tracks:
        if track.status == "rejected_noise_track" and not include_rejected:
            continue
        for observation in track.sorted_observations():
            frame_entries.setdefault(observation.frame_index, []).append((track, observation))
    for entries in frame_entries.values():
        entries.sort(key=lambda item: (status_rank(item[0].status), item[0].track_id))
    return frame_entries


def gallery_thumbnail_tracks(tracks: list[PlateTrack], *, limit: int) -> list[PlateTrack]:
    ordered = sorted(
        tracks,
        key=lambda item: (status_rank(item.status), item.first_frame, item.track_id),
    )
    return ordered[:limit]


def status_rank(status: str) -> int:
    return {
        "accepted_plate_track": 0,
        "review_candidate": 1,
        "rejected_noise_track": 2,
    }.get(status, 99)


def write_track_summary_json(
    tracks: list[PlateTrack],
    output_path: Path,
    *,
    config: PlateStabilizationConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "schema_version": SCHEMA_VERSION,
        "status_counts": status_counts(tracks),
        "tracks": [
            summarize_track(track) for track in sorted(tracks, key=lambda item: item.track_id)
        ],
        **(extra or {}),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_counts(tracks: list[PlateTrack]) -> dict[str, int]:
    counts = {
        "accepted_plate_track": 0,
        "rejected_noise_track": 0,
        "review_candidate": 0,
    }
    for track in tracks:
        counts[track.status] = counts.get(track.status, 0) + 1
    return counts


def representative_track_frames(tracks: list[PlateTrack]) -> list[int]:
    frames = [representative_frame_for_track(track) for track in tracks]
    return sorted(dict.fromkeys(frames))


def representative_frame_for_track(track: PlateTrack) -> int:
    observed = sorted(track.observed, key=lambda item: item.frame_index)
    return observed[len(observed) // 2].frame_index


def last_observed(track: PlateTrack) -> PlateTrackObservation:
    observed = track.observed
    if not observed:
        raise ValueError("track has no observed proposals")
    return max(observed, key=lambda item: item.frame_index)


def proposal_sort_key(proposal: PlateProposal) -> tuple[float, int, int, int, int, int]:
    x1, y1, x2, y2 = proposal.bbox_xyxy
    return (-proposal.score, x1, y1, x2, y2, proposal.source_detection_index or -1)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


def bbox_scale_ratio(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_area = max(1.0, float(bbox_area(left)))
    right_area = max(1.0, float(bbox_area(right)))
    ratio = right_area / left_area
    return max(ratio, 1.0 / ratio)


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    return plate_bakeoff.bbox_iou(left, right)


def euclidean_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def interpolate_bbox(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    values = [round(a + (b - a) * alpha) for a, b in zip(left, right)]
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def provenance_counts(observations: list[PlateTrackObservation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for observation in observations:
        counts[observation.provenance] = counts.get(observation.provenance, 0) + 1
    return counts


def status_color(status: str, provenance: str) -> tuple[int, int, int]:
    if provenance == "short_gap_interpolation":
        return (0, 255, 255)
    if status == "accepted_plate_track":
        return (0, 220, 0)
    if status == "review_candidate":
        return (0, 180, 255)
    return (160, 160, 160)


def escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
