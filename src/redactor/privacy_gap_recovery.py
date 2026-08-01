from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np

from redactor import media
from redactor.detection import rational_to_float
from redactor.tracking import normalized_center_distance

SCHEMA_VERSION = "1.0"

PROVENANCE_DETECTOR = "detector"
PROVENANCE_ORDINARY_TRACKER = "ordinary_tracker"
PROVENANCE_INTERPOLATION = "interpolation"
PROVENANCE_BIDIRECTIONAL_BRIDGE = "bidirectional_privacy_bridge"
PROVENANCE_FORWARD_BRIDGE = "forward_only_bridge"
PROVENANCE_BACKWARD_BRIDGE = "backward_only_bridge"
PROVENANCE_CONSERVATIVE_FALLBACK = "conservative_fallback"
PROVENANCE_MANUAL_CORRECTION = "manual_correction"
PROVENANCE_PRESERVED_REGION = "preserved_privacy_region"

DECISION_ACCEPTED_AUTOMATICALLY = "accepted_automatically"
DECISION_ACCEPTED_AFTER_REFINEMENT = "accepted_after_refinement"
DECISION_NEEDS_REVIEW = "needs_review"
DECISION_REJECTED = "rejected"

OBSERVED_TRACK_PROVENANCES = {"observed_strong", "observed_weak"}
MANUAL_TRACK_PROVENANCES = {"manual"}
ORDINARY_TRACKER_PROVENANCES = {"interpolated", "predicted"}


Box = tuple[int, int, int, int]


class FrameProvider(Protocol):
    def frame(self, frame_index: int) -> np.ndarray: ...


class VisualTracker(Protocol):
    def init(self, frame: np.ndarray, bbox_xyxy: Box) -> None: ...

    def update(self, frame: np.ndarray) -> tuple[bool, Box | None]: ...


class TrackerFactory(Protocol):
    def __call__(self) -> VisualTracker: ...


class VisualRefinementStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def refine(
        self,
        frame: np.ndarray,
        prior_box: Box,
        *,
        previous_box: Box | None,
        frame_size: tuple[int, int],
    ) -> tuple[Box | None, dict[str, object]]: ...


@dataclass(frozen=True)
class TrackSourceBox:
    frame_index: int
    track_id: int
    bbox_xyxy: Box
    source_provenance: str
    score: float | None = None
    action: str | None = None
    detector_id: str | None = None
    candidate_id: str | None = None
    quality_status: str | None = None

    @property
    def privacy_provenance(self) -> str:
        if self.source_provenance in OBSERVED_TRACK_PROVENANCES:
            return PROVENANCE_DETECTOR
        if self.source_provenance in MANUAL_TRACK_PROVENANCES:
            return PROVENANCE_MANUAL_CORRECTION
        return PROVENANCE_ORDINARY_TRACKER


@dataclass(frozen=True)
class PreservedRegion:
    frame_index: int
    bbox_xyxy: Box
    source_record: dict[str, object]


@dataclass
class PrivacyRegion:
    frame_index: int
    bbox_xyxy: Box | None
    provenance: str
    state: str
    target_track_ids: tuple[int, ...]
    source_anchor_frames: tuple[int, ...] = ()
    source_track_provenance: str | None = None
    tracking_direction: str | None = None
    raw_forward_box_xyxy: Box | None = None
    raw_backward_box_xyxy: Box | None = None
    refined_forward_box_xyxy: Box | None = None
    refined_backward_box_xyxy: Box | None = None
    metrics: dict[str, object] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "frame_index": self.frame_index,
            "metrics": self.metrics,
            "provenance": self.provenance,
            "raw_backward_box_xyxy": _optional_box(self.raw_backward_box_xyxy),
            "raw_forward_box_xyxy": _optional_box(self.raw_forward_box_xyxy),
            "reasons": list(self.reasons),
            "refined_backward_box_xyxy": _optional_box(self.refined_backward_box_xyxy),
            "refined_forward_box_xyxy": _optional_box(self.refined_forward_box_xyxy),
            "source_anchor_frames": list(self.source_anchor_frames),
            "source_track_provenance": self.source_track_provenance,
            "state": self.state,
            "target_track_ids": list(self.target_track_ids),
            "tracking_direction": self.tracking_direction,
        }


@dataclass(frozen=True)
class GapInterval:
    frame_start: int
    frame_end: int
    target_track_ids: tuple[int, ...]
    left_anchor: TrackSourceBox | None
    right_anchor: TrackSourceBox | None

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start + 1


@dataclass
class IntervalDecision:
    frame_start: int
    frame_end: int
    target_track_ids: tuple[int, ...]
    strategy: str
    status: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    regions: list[PrivacyRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "metrics": self.metrics,
            "reasons": self.reasons,
            "recovered_frames": sum(1 for region in self.regions if region.state == "covered"),
            "status": self.status,
            "strategy": self.strategy,
            "target_track_ids": list(self.target_track_ids),
            "unresolved_frames": sum(1 for region in self.regions if region.state != "covered"),
        }


@dataclass(frozen=True)
class PrivacyGapRecoverySettings:
    short_interpolation_gap_frames: int = 9
    max_bridge_gap_frames: int = 180
    max_one_sided_gap_frames: int = 45
    max_raw_disagreement_norm: float = 1.50
    max_refined_disagreement_norm: float = 0.75
    max_center_jump_norm: float = 0.85
    max_scale_change: float = 0.65
    max_keep_overlap_fraction: float = 0.01
    competing_identity_distance_norm: float = 0.35
    scene_cut_mean_delta: float = 45.0
    enable_visual_refinement: bool = False
    visual_refinement: str = "none"
    conservative_fallback_enabled: bool = False
    render_overlay: bool = True
    preserve_existing_regions: bool = True

    def validate(self) -> None:
        if self.short_interpolation_gap_frames < 0:
            raise ValueError("short-interpolation-gap-frames cannot be negative")
        if self.max_bridge_gap_frames < 0:
            raise ValueError("max-bridge-gap-frames cannot be negative")
        if self.max_one_sided_gap_frames < 0:
            raise ValueError("max-one-sided-gap-frames cannot be negative")
        if self.max_raw_disagreement_norm <= 0:
            raise ValueError("max-raw-disagreement-norm must be positive")
        if self.max_refined_disagreement_norm <= 0:
            raise ValueError("max-refined-disagreement-norm must be positive")
        if self.max_center_jump_norm <= 0:
            raise ValueError("max-center-jump-norm must be positive")
        if self.max_scale_change < 0:
            raise ValueError("max-scale-change cannot be negative")
        if not 0 <= self.max_keep_overlap_fraction <= 1:
            raise ValueError("max-keep-overlap-fraction must be between 0 and 1")
        if self.competing_identity_distance_norm <= 0:
            raise ValueError("competing-identity-distance-norm must be positive")
        if self.scene_cut_mean_delta <= 0:
            raise ValueError("scene-cut-mean-delta must be positive")
        if self.visual_refinement not in {"none", "bottom_component"}:
            raise ValueError("visual-refinement must be one of: none, bottom_component")


@dataclass(frozen=True)
class RecoveryMetrics:
    gaps_considered: int
    accepted_intervals: int
    needs_review_intervals: int
    rejected_intervals: int
    recovered_frames: int
    unresolved_frames: int
    maximum_raw_disagreement: float
    maximum_normalized_raw_disagreement: float
    maximum_refined_disagreement: float
    maximum_normalized_refined_disagreement: float
    maximum_center_jump: float
    maximum_normalized_center_jump: float
    maximum_scale_jump: float
    identity_switch_events: int
    keep_overlap_events: int
    processing_fps: float

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_intervals": self.accepted_intervals,
            "gaps_considered": self.gaps_considered,
            "identity_switch_events": self.identity_switch_events,
            "keep_overlap_events": self.keep_overlap_events,
            "maximum_center_jump": round(self.maximum_center_jump, 6),
            "maximum_normalized_center_jump": round(self.maximum_normalized_center_jump, 6),
            "maximum_normalized_raw_disagreement": round(
                self.maximum_normalized_raw_disagreement, 6
            ),
            "maximum_normalized_refined_disagreement": round(
                self.maximum_normalized_refined_disagreement, 6
            ),
            "maximum_raw_disagreement": round(self.maximum_raw_disagreement, 6),
            "maximum_refined_disagreement": round(self.maximum_refined_disagreement, 6),
            "maximum_scale_jump": round(self.maximum_scale_jump, 6),
            "needs_review_intervals": self.needs_review_intervals,
            "processing_fps": round(self.processing_fps, 6),
            "recovered_frames": self.recovered_frames,
            "rejected_intervals": self.rejected_intervals,
            "unresolved_frames": self.unresolved_frames,
        }


@dataclass(frozen=True)
class RecoveryResult:
    regions_by_frame: dict[int, list[PrivacyRegion]]
    interval_decisions: list[IntervalDecision]
    metrics: RecoveryMetrics


class OpenCVMILTracker:
    def __init__(self) -> None:
        self._tracker = cv.TrackerMIL_create()

    def init(self, frame: np.ndarray, bbox_xyxy: Box) -> None:
        self._tracker.init(frame, xyxy_to_xywh(bbox_xyxy))

    def update(self, frame: np.ndarray) -> tuple[bool, Box | None]:
        ok, rect = self._tracker.update(frame)
        return bool(ok), xywh_to_xyxy(rect) if ok else None


class VideoFrameProvider:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._capture = cv.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        self._cache: dict[int, np.ndarray] = {}

    def frame(self, frame_index: int) -> np.ndarray:
        if frame_index not in self._cache:
            self._capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self._capture.read()
            if not ok:
                raise ValueError(f"could not read frame {frame_index} from {self.video_path}")
            self._cache[frame_index] = frame
        return self._cache[frame_index].copy()

    def release(self) -> None:
        self._capture.release()


class NoRefinementStrategy:
    name = "none"

    def refine(
        self,
        frame: np.ndarray,
        prior_box: Box,
        *,
        previous_box: Box | None,
        frame_size: tuple[int, int],
    ) -> tuple[Box | None, dict[str, object]]:
        del frame, previous_box, frame_size
        return prior_box, {"strategy": self.name}


@dataclass(frozen=True)
class BottomConnectedComponentRefinement:
    """Optional refinement for targets with a dark component touching a configured edge.

    This is intentionally opt-in and parameterized. The default gap-recovery path does not assume
    a pose, a cap, or any specific edge contact.
    """

    name: str = "bottom_component"
    dark_percentile: float = 30.0
    min_component_area_fraction_of_box: float = 0.08
    edge: str = "bottom"
    edge_tolerance_fraction: float = 0.25
    search_padding_fraction: float = 0.45
    output_margin_fraction: float = 0.18

    def refine(
        self,
        frame: np.ndarray,
        prior_box: Box,
        *,
        previous_box: Box | None,
        frame_size: tuple[int, int],
    ) -> tuple[Box | None, dict[str, object]]:
        del previous_box
        width, height = frame_size
        prior_width = prior_box[2] - prior_box[0]
        prior_height = prior_box[3] - prior_box[1]
        pad_x = round(prior_width * self.search_padding_fraction)
        pad_y = round(prior_height * self.search_padding_fraction)
        search = clip_box(
            (
                prior_box[0] - pad_x,
                prior_box[1] - pad_y,
                prior_box[2] + pad_x,
                prior_box[3] + pad_y,
            ),
            width,
            height,
        )
        crop = frame[search[1] : search[3], search[0] : search[2]]
        if crop.size == 0:
            return None, {"reason": "empty_search_crop", "search_box_xyxy": list(search)}
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        threshold = min(100, max(1, int(np.percentile(gray, self.dark_percentile))))
        mask = (gray <= threshold).astype(np.uint8) * 255
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        count, _labels, stats, cents = cv.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, Box, int, tuple[float, float]]] = []
        minimum_area = max(16, round(box_area(prior_box) * self.min_component_area_fraction_of_box))
        prior_center = box_center(prior_box)
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            if area < minimum_area or component_width <= 1 or component_height <= 1:
                continue
            component = clip_box(
                (
                    search[0] + x,
                    search[1] + y,
                    search[0] + x + component_width,
                    search[1] + y + component_height,
                ),
                width,
                height,
            )
            if not self._component_touches_required_edge(component, search):
                continue
            component_center = (
                search[0] + float(cents[label][0]),
                search[1] + float(cents[label][1]),
            )
            distance_penalty = math.hypot(
                component_center[0] - prior_center[0],
                component_center[1] - prior_center[1],
            ) / max(box_diagonal(prior_box), 1.0)
            score = area / max(box_area(search), 1) - distance_penalty * 0.15
            candidates.append((score, component, int(area), component_center))
        if not candidates:
            return None, {
                "reason": "no_edge_connected_dark_component",
                "search_box_xyxy": list(search),
                "threshold": threshold,
            }

        candidates.sort(reverse=True, key=lambda item: item[0])
        _score, component, component_area, component_center = candidates[0]
        margin_x = round((component[2] - component[0]) * self.output_margin_fraction)
        margin_y = round((component[3] - component[1]) * self.output_margin_fraction)
        refined = clip_box(
            (
                component[0] - margin_x,
                component[1] - margin_y,
                component[2] + margin_x,
                component[3] + margin_y,
            ),
            width,
            height,
        )
        return refined, {
            "component_area": component_area,
            "component_box_xyxy": list(component),
            "component_center_xy": [round(component_center[0], 3), round(component_center[1], 3)],
            "edge": self.edge,
            "search_box_xyxy": list(search),
            "strategy": self.name,
            "threshold": threshold,
        }

    def _component_touches_required_edge(self, component: Box, search: Box) -> bool:
        tolerance = max(
            1,
            round(self.edge_tolerance_fraction * max(search[2] - search[0], search[3] - search[1])),
        )
        if self.edge == "bottom":
            return component[3] >= search[3] - tolerance
        if self.edge == "top":
            return component[1] <= search[1] + tolerance
        if self.edge == "left":
            return component[0] <= search[0] + tolerance
        if self.edge == "right":
            return component[2] >= search[2] - tolerance
        raise ValueError("unsupported component edge")


def run_privacy_gap_recovery(
    input_video: Path,
    *,
    tracks_path: Path,
    output_dir: Path,
    target_track_ids: tuple[int, ...],
    keep_track_ids: tuple[int, ...] = (),
    competing_track_ids: tuple[int, ...] = (),
    frame_start: int | None = None,
    frame_end: int | None = None,
    preserved_regions_path: Path | None = None,
    settings: PrivacyGapRecoverySettings | None = None,
    tracker_factory: TrackerFactory | None = None,
    frame_provider: FrameProvider | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if not target_track_ids:
        raise ValueError("at least one target track id is required")
    settings = settings or PrivacyGapRecoverySettings()
    settings.validate()
    ensure_output_dir(output_dir, overwrite=overwrite)
    source_info = media.probe_media(input_video)
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    source_frame_count = _optional_int(source_info.video_stream.frame_count)
    if frame_provider is None:
        frame_provider = VideoFrameProvider(input_video)
    close_provider = getattr(frame_provider, "release", None)

    try:
        track_records = load_track_records(tracks_path)
        target_boxes = select_track_boxes_by_frame(track_records, target_track_ids)
        keep_boxes = boxes_for_track_ids(track_records, keep_track_ids)
        competing_boxes = boxes_for_track_ids(track_records, competing_track_ids)
        preserved_regions = (
            load_preserved_regions(preserved_regions_path)
            if preserved_regions_path is not None and settings.preserve_existing_regions
            else {}
        )
        resolved_frame_start, resolved_frame_end = resolve_recovery_span(
            target_boxes,
            preserved_regions,
            frame_start=frame_start,
            frame_end=frame_end,
            source_frame_count=source_frame_count,
        )
        started_at = time.perf_counter()
        result = recover_privacy_gaps(
            frame_provider=frame_provider,
            target_boxes=target_boxes,
            target_track_ids=target_track_ids,
            keep_boxes_by_frame=keep_boxes,
            competing_boxes_by_frame=competing_boxes,
            preserved_regions=preserved_regions,
            frame_start=resolved_frame_start,
            frame_end=resolved_frame_end,
            frame_size=(width, height),
            settings=settings,
            tracker_factory=tracker_factory or OpenCVMILTracker,
        )
        elapsed_seconds = time.perf_counter() - started_at
        metrics = metrics_with_elapsed(result, elapsed_seconds)
        write_outputs(
            input_video,
            output_dir=output_dir,
            result=RecoveryResult(
                regions_by_frame=result.regions_by_frame,
                interval_decisions=result.interval_decisions,
                metrics=metrics,
            ),
            settings=settings,
            source_sha256=source_info.sha256,
            tracks_sha256=media.sha256_file(tracks_path),
            preserved_regions_sha256=(
                media.sha256_file(preserved_regions_path)
                if preserved_regions_path is not None and preserved_regions_path.exists()
                else None
            ),
            frame_start=resolved_frame_start,
            frame_end=resolved_frame_end,
            fps=fps,
            width=width,
            height=height,
        )
        if settings.render_overlay:
            render_overlay_video(
                input_video,
                output_dir / "overlay.mp4",
                regions_by_frame=result.regions_by_frame,
                frame_start=resolved_frame_start,
                frame_end=resolved_frame_end,
                fps=fps,
                width=width,
                height=height,
                keep_boxes_by_frame=keep_boxes,
            )
        return summary_payload(
            settings=settings,
            result=RecoveryResult(result.regions_by_frame, result.interval_decisions, metrics),
            source_sha256=source_info.sha256,
            tracks_sha256=media.sha256_file(tracks_path),
            preserved_regions_sha256=(
                media.sha256_file(preserved_regions_path)
                if preserved_regions_path is not None and preserved_regions_path.exists()
                else None
            ),
            frame_start=resolved_frame_start,
            frame_end=resolved_frame_end,
        )
    finally:
        if close_provider is not None:
            close_provider()


def recover_privacy_gaps(
    *,
    frame_provider: FrameProvider,
    target_boxes: dict[int, TrackSourceBox],
    target_track_ids: tuple[int, ...],
    keep_boxes_by_frame: dict[int, list[TrackSourceBox]],
    competing_boxes_by_frame: dict[int, list[TrackSourceBox]],
    preserved_regions: dict[int, PreservedRegion],
    frame_start: int,
    frame_end: int,
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
    tracker_factory: TrackerFactory,
) -> RecoveryResult:
    settings.validate()
    regions_by_frame: dict[int, list[PrivacyRegion]] = {
        frame_index: [] for frame_index in range(frame_start, frame_end + 1)
    }
    for frame_index in range(frame_start, frame_end + 1):
        if frame_index in target_boxes:
            regions_by_frame[frame_index].append(region_from_source_box(target_boxes[frame_index]))

    decisions: list[IntervalDecision] = []
    for interval in find_gap_intervals(
        target_boxes,
        target_track_ids=target_track_ids,
        frame_start=frame_start,
        frame_end=frame_end,
    ):
        if settings.preserve_existing_regions and gap_has_preserved_regions(
            preserved_regions,
            interval.frame_start,
            interval.frame_end,
        ):
            decision = preserved_interval_decision(interval, preserved_regions)
        else:
            decision = recover_interval(
                interval,
                frame_provider=frame_provider,
                keep_boxes_by_frame=keep_boxes_by_frame,
                competing_boxes_by_frame=competing_boxes_by_frame,
                frame_size=frame_size,
                settings=settings,
                tracker_factory=tracker_factory,
            )
        decisions.append(decision)
        for region in decision.regions:
            regions_by_frame[region.frame_index] = [region]

    return RecoveryResult(
        regions_by_frame=regions_by_frame,
        interval_decisions=decisions,
        metrics=calculate_metrics(decisions, processing_fps=0.0),
    )


def recover_interval(
    interval: GapInterval,
    *,
    frame_provider: FrameProvider,
    keep_boxes_by_frame: dict[int, list[TrackSourceBox]],
    competing_boxes_by_frame: dict[int, list[TrackSourceBox]],
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
    tracker_factory: TrackerFactory,
) -> IntervalDecision:
    reasons: list[str] = []
    if interval.frame_count > settings.max_bridge_gap_frames:
        return unresolved_decision(
            interval,
            strategy="excessive_gap",
            status=DECISION_REJECTED,
            reason="gap_duration_exceeds_max_bridge_gap_frames",
        )
    if interval.left_anchor is not None and interval.right_anchor is not None:
        if (
            interval.frame_count <= settings.short_interpolation_gap_frames
            and normalized_center_distance(
                interval.left_anchor.bbox_xyxy, interval.right_anchor.bbox_xyxy
            )
            <= settings.max_center_jump_norm
        ):
            return interpolate_interval(
                interval,
                keep_boxes_by_frame=keep_boxes_by_frame,
                competing_boxes_by_frame=competing_boxes_by_frame,
                frame_size=frame_size,
                settings=settings,
            )
        return bidirectional_interval(
            interval,
            frame_provider=frame_provider,
            keep_boxes_by_frame=keep_boxes_by_frame,
            competing_boxes_by_frame=competing_boxes_by_frame,
            frame_size=frame_size,
            settings=settings,
            tracker_factory=tracker_factory,
        )
    if interval.left_anchor is not None or interval.right_anchor is not None:
        if interval.frame_count > settings.max_one_sided_gap_frames:
            return unresolved_decision(
                interval,
                strategy="one_sided_tracking",
                status=DECISION_NEEDS_REVIEW,
                reason="gap_duration_exceeds_max_one_sided_gap_frames",
            )
        return one_sided_interval(
            interval,
            frame_provider=frame_provider,
            keep_boxes_by_frame=keep_boxes_by_frame,
            competing_boxes_by_frame=competing_boxes_by_frame,
            frame_size=frame_size,
            settings=settings,
            tracker_factory=tracker_factory,
        )
    reasons.append("missing_both_anchors")
    if settings.conservative_fallback_enabled:
        return unresolved_decision(
            interval,
            strategy=PROVENANCE_CONSERVATIVE_FALLBACK,
            status=DECISION_NEEDS_REVIEW,
            reason="conservative_fallback_requires_review",
        )
    return unresolved_decision(
        interval,
        strategy="missing_anchors",
        status=DECISION_REJECTED,
        reason="missing_both_anchors",
    )


def interpolate_interval(
    interval: GapInterval,
    *,
    keep_boxes_by_frame: dict[int, list[TrackSourceBox]],
    competing_boxes_by_frame: dict[int, list[TrackSourceBox]],
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
) -> IntervalDecision:
    if interval.left_anchor is None or interval.right_anchor is None:
        return unresolved_decision(
            interval,
            strategy=PROVENANCE_INTERPOLATION,
            status=DECISION_REJECTED,
            reason="interpolation_requires_two_anchors",
        )
    regions = []
    previous_box = interval.left_anchor.bbox_xyxy
    for frame_index in range(interval.frame_start, interval.frame_end + 1):
        fraction = (frame_index - interval.left_anchor.frame_index) / (
            interval.right_anchor.frame_index - interval.left_anchor.frame_index
        )
        box = interpolate_box(
            interval.left_anchor.bbox_xyxy, interval.right_anchor.bbox_xyxy, fraction
        )
        reasons, metrics = safety_reasons(
            box,
            frame_index=frame_index,
            previous_box=previous_box,
            keep_boxes=keep_boxes_by_frame.get(frame_index, []),
            competing_boxes=competing_boxes_by_frame.get(frame_index, []),
            frame_size=frame_size,
            settings=settings,
        )
        regions.append(
            PrivacyRegion(
                frame_index=frame_index,
                bbox_xyxy=None if reasons else box,
                provenance=PROVENANCE_INTERPOLATION,
                state="unresolved" if reasons else "covered",
                target_track_ids=interval.target_track_ids,
                source_anchor_frames=(
                    interval.left_anchor.frame_index,
                    interval.right_anchor.frame_index,
                ),
                tracking_direction="interpolated",
                metrics=metrics,
                reasons=tuple(reasons),
            )
        )
        if not reasons:
            previous_box = box
    return interval_decision_for_regions(
        interval,
        strategy=PROVENANCE_INTERPOLATION,
        regions=regions,
        used_refinement=False,
    )


def bidirectional_interval(
    interval: GapInterval,
    *,
    frame_provider: FrameProvider,
    keep_boxes_by_frame: dict[int, list[TrackSourceBox]],
    competing_boxes_by_frame: dict[int, list[TrackSourceBox]],
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
    tracker_factory: TrackerFactory,
) -> IntervalDecision:
    if interval.left_anchor is None or interval.right_anchor is None:
        return unresolved_decision(
            interval,
            strategy=PROVENANCE_BIDIRECTIONAL_BRIDGE,
            status=DECISION_REJECTED,
            reason="bidirectional_tracking_requires_two_anchors",
        )
    refinement = refinement_strategy(settings)
    forward = track_forward(
        frame_provider,
        tracker_factory,
        anchor_frame=interval.left_anchor.frame_index,
        anchor_box=interval.left_anchor.bbox_xyxy,
        frame_start=interval.frame_start,
        frame_end=interval.frame_end,
    )
    backward = track_backward(
        frame_provider,
        tracker_factory,
        anchor_frame=interval.right_anchor.frame_index,
        anchor_box=interval.right_anchor.bbox_xyxy,
        frame_start=interval.frame_start,
        frame_end=interval.frame_end,
    )
    regions: list[PrivacyRegion] = []
    previous_box: Box | None = interval.left_anchor.bbox_xyxy
    used_refinement = False
    scene_cut_frames = detect_scene_cuts(
        frame_provider,
        frame_start=interval.left_anchor.frame_index,
        frame_end=interval.right_anchor.frame_index,
        threshold=settings.scene_cut_mean_delta,
    )
    for frame_index in range(interval.frame_start, interval.frame_end + 1):
        raw_forward = forward.get(frame_index)
        raw_backward = backward.get(frame_index)
        refined_forward, forward_meta = refine_box(
            refinement,
            frame_provider,
            frame_index,
            raw_forward,
            previous_box=previous_box,
            frame_size=frame_size,
        )
        refined_backward, backward_meta = refine_box(
            refinement,
            frame_provider,
            frame_index,
            raw_backward,
            previous_box=previous_box,
            frame_size=frame_size,
        )
        used_refinement = used_refinement or box_changed(raw_forward, refined_forward)
        used_refinement = used_refinement or box_changed(raw_backward, refined_backward)
        region = select_bidirectional_region(
            interval,
            frame_index=frame_index,
            raw_forward=raw_forward,
            raw_backward=raw_backward,
            refined_forward=refined_forward,
            refined_backward=refined_backward,
            forward_meta=forward_meta,
            backward_meta=backward_meta,
            previous_box=previous_box,
            keep_boxes=keep_boxes_by_frame.get(frame_index, []),
            competing_boxes=competing_boxes_by_frame.get(frame_index, []),
            frame_size=frame_size,
            settings=settings,
            scene_cut=frame_index in scene_cut_frames,
        )
        if region.bbox_xyxy is not None:
            previous_box = region.bbox_xyxy
        regions.append(region)
    return interval_decision_for_regions(
        interval,
        strategy=PROVENANCE_BIDIRECTIONAL_BRIDGE,
        regions=regions,
        used_refinement=used_refinement,
    )


def one_sided_interval(
    interval: GapInterval,
    *,
    frame_provider: FrameProvider,
    keep_boxes_by_frame: dict[int, list[TrackSourceBox]],
    competing_boxes_by_frame: dict[int, list[TrackSourceBox]],
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
    tracker_factory: TrackerFactory,
) -> IntervalDecision:
    if interval.left_anchor is not None:
        raw = track_forward(
            frame_provider,
            tracker_factory,
            anchor_frame=interval.left_anchor.frame_index,
            anchor_box=interval.left_anchor.bbox_xyxy,
            frame_start=interval.frame_start,
            frame_end=interval.frame_end,
        )
        provenance = PROVENANCE_FORWARD_BRIDGE
        direction = "forward"
        anchor_frame = interval.left_anchor.frame_index
        previous_box: Box | None = interval.left_anchor.bbox_xyxy
    elif interval.right_anchor is not None:
        raw = track_backward(
            frame_provider,
            tracker_factory,
            anchor_frame=interval.right_anchor.frame_index,
            anchor_box=interval.right_anchor.bbox_xyxy,
            frame_start=interval.frame_start,
            frame_end=interval.frame_end,
        )
        provenance = PROVENANCE_BACKWARD_BRIDGE
        direction = "backward"
        anchor_frame = interval.right_anchor.frame_index
        previous_box = interval.right_anchor.bbox_xyxy
    else:
        return unresolved_decision(
            interval,
            strategy="one_sided_tracking",
            status=DECISION_REJECTED,
            reason="one_sided_tracking_requires_an_anchor",
        )

    refinement = refinement_strategy(settings)
    regions = []
    scene_cut_frames = detect_scene_cuts(
        frame_provider,
        frame_start=(
            interval.left_anchor.frame_index if interval.left_anchor else interval.frame_start
        ),
        frame_end=(
            interval.right_anchor.frame_index if interval.right_anchor else interval.frame_end
        ),
        threshold=settings.scene_cut_mean_delta,
    )
    used_refinement = False
    frame_range = range(interval.frame_start, interval.frame_end + 1)
    if direction == "backward":
        frame_range = reversed(list(frame_range))
    temporary: dict[int, PrivacyRegion] = {}
    for frame_index in frame_range:
        refined, meta = refine_box(
            refinement,
            frame_provider,
            frame_index,
            raw.get(frame_index),
            previous_box=previous_box,
            frame_size=frame_size,
        )
        used_refinement = used_refinement or box_changed(raw.get(frame_index), refined)
        reasons, metrics = safety_reasons(
            refined,
            frame_index=frame_index,
            previous_box=previous_box,
            keep_boxes=keep_boxes_by_frame.get(frame_index, []),
            competing_boxes=competing_boxes_by_frame.get(frame_index, []),
            frame_size=frame_size,
            settings=settings,
            scene_cut=frame_index in scene_cut_frames,
        )
        metrics["refinement"] = meta
        region = PrivacyRegion(
            frame_index=frame_index,
            bbox_xyxy=None if reasons else refined,
            provenance=provenance,
            state="unresolved" if reasons else "covered",
            target_track_ids=interval.target_track_ids,
            source_anchor_frames=(anchor_frame,),
            tracking_direction=direction,
            raw_forward_box_xyxy=raw.get(frame_index) if direction == "forward" else None,
            raw_backward_box_xyxy=raw.get(frame_index) if direction == "backward" else None,
            refined_forward_box_xyxy=refined if direction == "forward" else None,
            refined_backward_box_xyxy=refined if direction == "backward" else None,
            metrics=metrics,
            reasons=tuple(reasons),
        )
        if region.bbox_xyxy is not None:
            previous_box = region.bbox_xyxy
        temporary[frame_index] = region
    regions = [
        temporary[frame_index]
        for frame_index in range(interval.frame_start, interval.frame_end + 1)
    ]
    return interval_decision_for_regions(
        interval,
        strategy=provenance,
        regions=regions,
        used_refinement=used_refinement,
    )


def select_bidirectional_region(
    interval: GapInterval,
    *,
    frame_index: int,
    raw_forward: Box | None,
    raw_backward: Box | None,
    refined_forward: Box | None,
    refined_backward: Box | None,
    forward_meta: dict[str, object],
    backward_meta: dict[str, object],
    previous_box: Box | None,
    keep_boxes: list[TrackSourceBox],
    competing_boxes: list[TrackSourceBox],
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
    scene_cut: bool,
) -> PrivacyRegion:
    metrics = disagreement_metrics(
        raw_forward=raw_forward,
        raw_backward=raw_backward,
        refined_forward=refined_forward,
        refined_backward=refined_backward,
    )
    metrics["forward_refinement"] = forward_meta
    metrics["backward_refinement"] = backward_meta

    selected: Box | None = None
    tracking_direction = "unresolved"
    provenance = PROVENANCE_BIDIRECTIONAL_BRIDGE
    reasons: list[str] = []
    if refined_forward is not None and refined_backward is not None:
        refined_norm = metrics["normalized_refined_disagreement"]
        if refined_norm <= settings.max_refined_disagreement_norm:
            selected = fuse_boxes(refined_forward, refined_backward)
            tracking_direction = "fused"
        else:
            forward_score = visual_quality_score(forward_meta)
            backward_score = visual_quality_score(backward_meta)
            if forward_score > backward_score + 0.15:
                selected = refined_forward
                tracking_direction = "forward"
                provenance = PROVENANCE_FORWARD_BRIDGE
                reasons.append("refined_disagreement_forward_selected_by_visual_quality")
            elif backward_score > forward_score + 0.15:
                selected = refined_backward
                tracking_direction = "backward"
                provenance = PROVENANCE_BACKWARD_BRIDGE
                reasons.append("refined_disagreement_backward_selected_by_visual_quality")
            else:
                reasons.append("forward_backward_refined_disagreement")
    elif refined_forward is not None:
        selected = refined_forward
        tracking_direction = "forward"
        provenance = PROVENANCE_FORWARD_BRIDGE
        reasons.append("backward_tracker_unavailable")
    elif refined_backward is not None:
        selected = refined_backward
        tracking_direction = "backward"
        provenance = PROVENANCE_BACKWARD_BRIDGE
        reasons.append("forward_tracker_unavailable")
    else:
        reasons.append("no_tracker_box")

    safety, safety_metrics = safety_reasons(
        selected,
        frame_index=frame_index,
        previous_box=previous_box,
        keep_boxes=keep_boxes,
        competing_boxes=competing_boxes,
        frame_size=frame_size,
        settings=settings,
        raw_disagreement_norm=metrics.get("normalized_raw_disagreement"),
        refined_disagreement_norm=metrics.get("normalized_refined_disagreement"),
        scene_cut=scene_cut,
    )
    reasons.extend(safety)
    metrics.update(safety_metrics)
    if reasons and not only_selection_reason(reasons):
        selected = None
    return PrivacyRegion(
        frame_index=frame_index,
        bbox_xyxy=selected,
        provenance=provenance,
        state="covered" if selected is not None else "unresolved",
        target_track_ids=interval.target_track_ids,
        source_anchor_frames=(
            interval.left_anchor.frame_index if interval.left_anchor else -1,
            interval.right_anchor.frame_index if interval.right_anchor else -1,
        ),
        tracking_direction=tracking_direction,
        raw_forward_box_xyxy=raw_forward,
        raw_backward_box_xyxy=raw_backward,
        refined_forward_box_xyxy=refined_forward,
        refined_backward_box_xyxy=refined_backward,
        metrics=metrics,
        reasons=tuple(reasons),
    )


def safety_reasons(
    box: Box | None,
    *,
    frame_index: int,
    previous_box: Box | None,
    keep_boxes: list[TrackSourceBox],
    competing_boxes: list[TrackSourceBox],
    frame_size: tuple[int, int],
    settings: PrivacyGapRecoverySettings,
    raw_disagreement_norm: float | None = None,
    refined_disagreement_norm: float | None = None,
    scene_cut: bool = False,
) -> tuple[list[str], dict[str, object]]:
    width, height = frame_size
    reasons: list[str] = []
    metrics: dict[str, object] = {
        "frame_index": frame_index,
        "scene_cut": scene_cut,
    }
    if box is None:
        return ["missing_box"], metrics
    metrics["bbox_area_fraction"] = round(box_area(box) / max(width * height, 1), 8)
    if not box_is_valid(box):
        reasons.append("invalid_box")
        return reasons, metrics
    if not box_inside_frame(box, width, height):
        reasons.append("out_of_frame_box")
    if scene_cut:
        reasons.append("scene_cut_inside_gap")
    if (
        raw_disagreement_norm is not None
        and raw_disagreement_norm > settings.max_raw_disagreement_norm
    ):
        metrics["raw_disagreement_exceeds_threshold"] = True
    if (
        refined_disagreement_norm is not None
        and refined_disagreement_norm > settings.max_refined_disagreement_norm
    ):
        reasons.append("forward_backward_refined_disagreement")
    if previous_box is not None:
        center_jump = box_center_distance(previous_box, box)
        normalized_jump = center_jump / max(average_diagonal(previous_box, box), 1.0)
        scale_jump = scale_change(previous_box, box)
        metrics["center_jump"] = round(center_jump, 6)
        metrics["normalized_center_jump"] = round(normalized_jump, 6)
        metrics["scale_jump"] = round(scale_jump, 6)
        if normalized_jump > settings.max_center_jump_norm:
            reasons.append("excessive_center_jump")
        if scale_jump > settings.max_scale_change:
            reasons.append("excessive_scale_change")
    keep_overlap = max(
        (intersection_fraction(box, item.bbox_xyxy) for item in keep_boxes), default=0.0
    )
    metrics["keep_overlap_fraction"] = round(keep_overlap, 8)
    if keep_overlap > settings.max_keep_overlap_fraction:
        reasons.append("keep_person_overlap")
    competing_distance = min(
        (
            normalized_center_distance(box, item.bbox_xyxy)
            for item in competing_boxes
            if box_is_valid(item.bbox_xyxy)
        ),
        default=None,
    )
    metrics["nearest_competing_identity_distance_norm"] = (
        None if competing_distance is None else round(competing_distance, 6)
    )
    if (
        competing_distance is not None
        and competing_distance < settings.competing_identity_distance_norm
    ):
        reasons.append("identity_competition")
    return reasons, metrics


def interval_decision_for_regions(
    interval: GapInterval,
    *,
    strategy: str,
    regions: list[PrivacyRegion],
    used_refinement: bool,
) -> IntervalDecision:
    unresolved = [region for region in regions if region.state != "covered"]
    reasons = sorted({reason for region in regions for reason in region.reasons})
    if not regions or len(unresolved) == len(regions):
        status = DECISION_REJECTED if reasons == ["missing_box"] else DECISION_NEEDS_REVIEW
    elif unresolved:
        status = DECISION_NEEDS_REVIEW
    elif used_refinement:
        status = DECISION_ACCEPTED_AFTER_REFINEMENT
    else:
        status = DECISION_ACCEPTED_AUTOMATICALLY
    return IntervalDecision(
        frame_start=interval.frame_start,
        frame_end=interval.frame_end,
        target_track_ids=interval.target_track_ids,
        strategy=strategy,
        status=status,
        reasons=reasons,
        metrics=interval_metrics(regions),
        regions=regions,
    )


def unresolved_decision(
    interval: GapInterval,
    *,
    strategy: str,
    status: str,
    reason: str,
) -> IntervalDecision:
    regions = [
        PrivacyRegion(
            frame_index=frame_index,
            bbox_xyxy=None,
            provenance=PROVENANCE_CONSERVATIVE_FALLBACK
            if strategy == PROVENANCE_CONSERVATIVE_FALLBACK
            else strategy,
            state="unresolved",
            target_track_ids=interval.target_track_ids,
            reasons=(reason,),
        )
        for frame_index in range(interval.frame_start, interval.frame_end + 1)
    ]
    return IntervalDecision(
        frame_start=interval.frame_start,
        frame_end=interval.frame_end,
        target_track_ids=interval.target_track_ids,
        strategy=strategy,
        status=status,
        reasons=[reason],
        metrics=interval_metrics(regions),
        regions=regions,
    )


def preserved_interval_decision(
    interval: GapInterval,
    preserved_regions: dict[int, PreservedRegion],
) -> IntervalDecision:
    regions = [
        PrivacyRegion(
            frame_index=frame_index,
            bbox_xyxy=preserved_regions[frame_index].bbox_xyxy,
            provenance=PROVENANCE_PRESERVED_REGION,
            state="covered",
            target_track_ids=interval.target_track_ids,
            source_anchor_frames=tuple(
                int(value)
                for value in preserved_regions[frame_index].source_record.get(
                    "source_anchor_frames", []
                )
            ),
            tracking_direction=str(
                preserved_regions[frame_index].source_record.get("tracking_direction")
                or preserved_regions[frame_index].source_record.get("provenance")
                or "preserved"
            ),
            metrics={
                "preserved_source_provenance": preserved_regions[frame_index].source_record.get(
                    "provenance"
                )
            },
        )
        for frame_index in range(interval.frame_start, interval.frame_end + 1)
    ]
    return IntervalDecision(
        frame_start=interval.frame_start,
        frame_end=interval.frame_end,
        target_track_ids=interval.target_track_ids,
        strategy=PROVENANCE_PRESERVED_REGION,
        status=DECISION_ACCEPTED_AUTOMATICALLY,
        reasons=["preserved_existing_accepted_privacy_region"],
        metrics=interval_metrics(regions),
        regions=regions,
    )


def track_forward(
    frame_provider: FrameProvider,
    tracker_factory: TrackerFactory,
    *,
    anchor_frame: int,
    anchor_box: Box,
    frame_start: int,
    frame_end: int,
) -> dict[int, Box | None]:
    tracker = tracker_factory()
    tracker.init(frame_provider.frame(anchor_frame), anchor_box)
    boxes: dict[int, Box | None] = {}
    for frame_index in range(anchor_frame + 1, frame_end + 1):
        ok, box = tracker.update(frame_provider.frame(frame_index))
        if frame_index >= frame_start:
            boxes[frame_index] = box if ok else None
    return boxes


def track_backward(
    frame_provider: FrameProvider,
    tracker_factory: TrackerFactory,
    *,
    anchor_frame: int,
    anchor_box: Box,
    frame_start: int,
    frame_end: int,
) -> dict[int, Box | None]:
    tracker = tracker_factory()
    tracker.init(frame_provider.frame(anchor_frame), anchor_box)
    boxes: dict[int, Box | None] = {}
    for frame_index in range(anchor_frame - 1, frame_start - 1, -1):
        ok, box = tracker.update(frame_provider.frame(frame_index))
        if frame_index <= frame_end:
            boxes[frame_index] = box if ok else None
    return boxes


def refine_box(
    refinement: VisualRefinementStrategy,
    frame_provider: FrameProvider,
    frame_index: int,
    box: Box | None,
    *,
    previous_box: Box | None,
    frame_size: tuple[int, int],
) -> tuple[Box | None, dict[str, object]]:
    if box is None:
        return None, {"reason": "missing_prior", "strategy": refinement.name}
    return refinement.refine(
        frame_provider.frame(frame_index),
        box,
        previous_box=previous_box,
        frame_size=frame_size,
    )


def refinement_strategy(settings: PrivacyGapRecoverySettings) -> VisualRefinementStrategy:
    if not settings.enable_visual_refinement or settings.visual_refinement == "none":
        return NoRefinementStrategy()
    if settings.visual_refinement == "bottom_component":
        return BottomConnectedComponentRefinement()
    raise ValueError(f"unsupported visual refinement: {settings.visual_refinement}")


def find_gap_intervals(
    target_boxes: dict[int, TrackSourceBox],
    *,
    target_track_ids: tuple[int, ...],
    frame_start: int,
    frame_end: int,
) -> list[GapInterval]:
    intervals = []
    cursor = frame_start
    while cursor <= frame_end:
        if cursor in target_boxes:
            cursor += 1
            continue
        start = cursor
        while cursor <= frame_end and cursor not in target_boxes:
            cursor += 1
        end = cursor - 1
        left = nearest_anchor_before(target_boxes, start)
        right = nearest_anchor_after(target_boxes, end)
        intervals.append(GapInterval(start, end, target_track_ids, left, right))
    return intervals


def nearest_anchor_before(
    target_boxes: dict[int, TrackSourceBox], frame_index: int
) -> TrackSourceBox | None:
    frames = [frame for frame in target_boxes if frame < frame_index]
    return target_boxes[max(frames)] if frames else None


def nearest_anchor_after(
    target_boxes: dict[int, TrackSourceBox], frame_index: int
) -> TrackSourceBox | None:
    frames = [frame for frame in target_boxes if frame > frame_index]
    return target_boxes[min(frames)] if frames else None


def region_from_source_box(source: TrackSourceBox) -> PrivacyRegion:
    return PrivacyRegion(
        frame_index=source.frame_index,
        bbox_xyxy=source.bbox_xyxy,
        provenance=source.privacy_provenance,
        state="covered",
        target_track_ids=(source.track_id,),
        source_track_provenance=source.source_provenance,
        metrics={
            "candidate_id": source.candidate_id,
            "detector_id": source.detector_id,
            "quality_status": source.quality_status,
            "score": source.score,
        },
    )


def calculate_metrics(
    decisions: list[IntervalDecision],
    *,
    processing_fps: float,
) -> RecoveryMetrics:
    regions = [region for decision in decisions for region in decision.regions]
    raw_values = [float(region.metrics.get("raw_disagreement", 0.0) or 0.0) for region in regions]
    raw_norm_values = [
        float(region.metrics.get("normalized_raw_disagreement", 0.0) or 0.0) for region in regions
    ]
    refined_values = [
        float(region.metrics.get("refined_disagreement", 0.0) or 0.0) for region in regions
    ]
    refined_norm_values = [
        float(region.metrics.get("normalized_refined_disagreement", 0.0) or 0.0)
        for region in regions
    ]
    center_values = [float(region.metrics.get("center_jump", 0.0) or 0.0) for region in regions]
    center_norm_values = [
        float(region.metrics.get("normalized_center_jump", 0.0) or 0.0) for region in regions
    ]
    scale_values = [float(region.metrics.get("scale_jump", 0.0) or 0.0) for region in regions]
    return RecoveryMetrics(
        gaps_considered=len(decisions),
        accepted_intervals=sum(
            1
            for decision in decisions
            if decision.status
            in {DECISION_ACCEPTED_AUTOMATICALLY, DECISION_ACCEPTED_AFTER_REFINEMENT}
        ),
        needs_review_intervals=sum(
            1 for decision in decisions if decision.status == DECISION_NEEDS_REVIEW
        ),
        rejected_intervals=sum(1 for decision in decisions if decision.status == DECISION_REJECTED),
        recovered_frames=sum(1 for region in regions if region.state == "covered"),
        unresolved_frames=sum(1 for region in regions if region.state != "covered"),
        maximum_raw_disagreement=max(raw_values, default=0.0),
        maximum_normalized_raw_disagreement=max(raw_norm_values, default=0.0),
        maximum_refined_disagreement=max(refined_values, default=0.0),
        maximum_normalized_refined_disagreement=max(refined_norm_values, default=0.0),
        maximum_center_jump=max(center_values, default=0.0),
        maximum_normalized_center_jump=max(center_norm_values, default=0.0),
        maximum_scale_jump=max(scale_values, default=0.0),
        identity_switch_events=sum(
            1 for region in regions if "identity_competition" in region.reasons
        ),
        keep_overlap_events=sum(1 for region in regions if "keep_person_overlap" in region.reasons),
        processing_fps=processing_fps,
    )


def metrics_with_elapsed(result: RecoveryResult, elapsed_seconds: float) -> RecoveryMetrics:
    recovered_or_unresolved = sum(
        1 for decision in result.interval_decisions for _region in decision.regions
    )
    fps = recovered_or_unresolved / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return calculate_metrics(result.interval_decisions, processing_fps=fps)


def interval_metrics(regions: list[PrivacyRegion]) -> dict[str, object]:
    return {
        "identity_switch_checks": sum(
            1 for region in regions if "identity_competition" in region.reasons
        ),
        "keep_overlap_checks": sum(
            1 for region in regions if "keep_person_overlap" in region.reasons
        ),
        "maximum_center_jump": round(
            max(
                (float(region.metrics.get("center_jump", 0.0) or 0.0) for region in regions),
                default=0.0,
            ),
            6,
        ),
        "maximum_normalized_center_jump": round(
            max(
                (
                    float(region.metrics.get("normalized_center_jump", 0.0) or 0.0)
                    for region in regions
                ),
                default=0.0,
            ),
            6,
        ),
        "maximum_normalized_raw_disagreement": round(
            max(
                (
                    float(region.metrics.get("normalized_raw_disagreement", 0.0) or 0.0)
                    for region in regions
                ),
                default=0.0,
            ),
            6,
        ),
        "maximum_normalized_refined_disagreement": round(
            max(
                (
                    float(region.metrics.get("normalized_refined_disagreement", 0.0) or 0.0)
                    for region in regions
                ),
                default=0.0,
            ),
            6,
        ),
        "maximum_raw_disagreement": round(
            max(
                (float(region.metrics.get("raw_disagreement", 0.0) or 0.0) for region in regions),
                default=0.0,
            ),
            6,
        ),
        "maximum_refined_disagreement": round(
            max(
                (
                    float(region.metrics.get("refined_disagreement", 0.0) or 0.0)
                    for region in regions
                ),
                default=0.0,
            ),
            6,
        ),
        "maximum_scale_jump": round(
            max(
                (float(region.metrics.get("scale_jump", 0.0) or 0.0) for region in regions),
                default=0.0,
            ),
            6,
        ),
    }


def disagreement_metrics(
    *,
    raw_forward: Box | None,
    raw_backward: Box | None,
    refined_forward: Box | None,
    refined_backward: Box | None,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if raw_forward is not None and raw_backward is not None:
        raw = box_center_distance(raw_forward, raw_backward)
        metrics["raw_disagreement"] = round(raw, 6)
        metrics["normalized_raw_disagreement"] = round(
            raw / max(average_diagonal(raw_forward, raw_backward), 1.0), 6
        )
    if refined_forward is not None and refined_backward is not None:
        refined = box_center_distance(refined_forward, refined_backward)
        metrics["refined_disagreement"] = round(refined, 6)
        metrics["normalized_refined_disagreement"] = round(
            refined / max(average_diagonal(refined_forward, refined_backward), 1.0), 6
        )
    return metrics


def detect_scene_cuts(
    frame_provider: FrameProvider,
    *,
    frame_start: int,
    frame_end: int,
    threshold: float,
) -> set[int]:
    scene_cuts = set()
    previous = frame_provider.frame(frame_start)
    for frame_index in range(frame_start + 1, frame_end + 1):
        current = frame_provider.frame(frame_index)
        delta = float(np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16))))
        if delta > threshold:
            scene_cuts.add(frame_index)
        previous = current
    return scene_cuts


def load_track_records(path: Path) -> dict[int, list[TrackSourceBox]]:
    records: dict[int, list[TrackSourceBox]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            frame_index = int(payload["frame_index"])
            boxes = []
            for box in payload.get("track_boxes", []):
                boxes.append(
                    TrackSourceBox(
                        frame_index=frame_index,
                        track_id=int(box["track_id"]),
                        bbox_xyxy=tuple(int(value) for value in box["bbox_xyxy"]),
                        source_provenance=str(box["provenance"]),
                        score=box.get("score"),
                        action=box.get("action"),
                        detector_id=box.get("detector_id"),
                        candidate_id=box.get("candidate_id"),
                        quality_status=box.get("quality_status"),
                    )
                )
            records[frame_index] = boxes
    return records


def select_track_boxes_by_frame(
    track_records: dict[int, list[TrackSourceBox]],
    track_ids: tuple[int, ...],
) -> dict[int, TrackSourceBox]:
    selected = {}
    track_id_set = set(track_ids)
    for frame_index, boxes in track_records.items():
        candidates = [box for box in boxes if box.track_id in track_id_set]
        if candidates:
            selected[frame_index] = sorted(candidates, key=source_box_priority)[0]
    return selected


def boxes_for_track_ids(
    track_records: dict[int, list[TrackSourceBox]],
    track_ids: tuple[int, ...],
) -> dict[int, list[TrackSourceBox]]:
    if not track_ids:
        return {}
    track_id_set = set(track_ids)
    return {
        frame_index: [box for box in boxes if box.track_id in track_id_set]
        for frame_index, boxes in track_records.items()
        if any(box.track_id in track_id_set for box in boxes)
    }


def source_box_priority(box: TrackSourceBox) -> tuple[int, int, float, int]:
    provenance_rank = {
        "manual": 0,
        "observed_strong": 1,
        "observed_weak": 2,
        "interpolated": 3,
        "predicted": 4,
    }.get(box.source_provenance, 5)
    quality_rank = {"auto_confirmed": 0, "review_candidate": 1, "rejected": 2}.get(
        box.quality_status or "", 3
    )
    return (provenance_rank, quality_rank, -(box.score or 0.0), box.track_id)


def load_preserved_regions(path: Path | None) -> dict[int, PreservedRegion]:
    if path is None:
        return {}
    regions = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            bbox = record.get("bbox_xyxy")
            if bbox is None or record.get("state") != "covered":
                continue
            regions[int(record["frame_index"])] = PreservedRegion(
                frame_index=int(record["frame_index"]),
                bbox_xyxy=tuple(int(value) for value in bbox),
                source_record=record,
            )
    return regions


def resolve_recovery_span(
    target_boxes: dict[int, TrackSourceBox],
    preserved_regions: dict[int, PreservedRegion],
    *,
    frame_start: int | None,
    frame_end: int | None,
    source_frame_count: int | None,
) -> tuple[int, int]:
    if frame_start is not None and frame_end is not None:
        if frame_end < frame_start:
            raise ValueError("frame-end cannot be before frame-start")
        return frame_start, frame_end
    frames = set(target_boxes) | set(preserved_regions)
    if not frames:
        if source_frame_count is None:
            raise ValueError("cannot infer recovery span without target boxes or frame count")
        return 0, source_frame_count - 1
    start = min(frames) if frame_start is None else frame_start
    end = max(frames) if frame_end is None else frame_end
    if end < start:
        raise ValueError("frame-end cannot be before frame-start")
    return start, end


def gap_has_preserved_regions(
    preserved_regions: dict[int, PreservedRegion],
    frame_start: int,
    frame_end: int,
) -> bool:
    return all(
        frame_index in preserved_regions for frame_index in range(frame_start, frame_end + 1)
    )


def write_outputs(
    input_video: Path,
    *,
    output_dir: Path,
    result: RecoveryResult,
    settings: PrivacyGapRecoverySettings,
    source_sha256: str,
    tracks_sha256: str,
    preserved_regions_sha256: str | None,
    frame_start: int,
    frame_end: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    write_privacy_regions(output_dir / "privacy_regions.jsonl", result.regions_by_frame)
    write_json(
        output_dir / "interval_decisions.json",
        {
            "interval_decisions": [decision.to_dict() for decision in result.interval_decisions],
            "schema_version": SCHEMA_VERSION,
        },
    )
    write_json(
        output_dir / "metrics.json", result.metrics.to_dict() | {"schema_version": SCHEMA_VERSION}
    )
    summary = summary_payload(
        settings=settings,
        result=result,
        source_sha256=source_sha256,
        tracks_sha256=tracks_sha256,
        preserved_regions_sha256=preserved_regions_sha256,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    summary.update(
        {
            "diagnostic_overlay": "overlay.mp4" if settings.render_overlay else None,
            "fps": round(fps, 6),
            "input_height": height,
            "input_video": input_video.name,
            "input_width": width,
        }
    )
    write_json(output_dir / "summary.json", summary)


def summary_payload(
    *,
    settings: PrivacyGapRecoverySettings,
    result: RecoveryResult,
    source_sha256: str,
    tracks_sha256: str,
    preserved_regions_sha256: str | None,
    frame_start: int,
    frame_end: int,
) -> dict[str, object]:
    return {
        "frame_end": frame_end,
        "frame_start": frame_start,
        "metrics": result.metrics.to_dict(),
        "preserved_regions_sha256": preserved_regions_sha256,
        "schema_version": SCHEMA_VERSION,
        "settings": settings_to_dict(settings),
        "source_sha256": source_sha256,
        "tracks_sha256": tracks_sha256,
    }


def write_privacy_regions(path: Path, regions_by_frame: dict[int, list[PrivacyRegion]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for frame_index in sorted(regions_by_frame):
            file.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "privacy_regions": [
                            region.to_dict() for region in regions_by_frame[frame_index]
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def render_overlay_video(
    input_video: Path,
    output_path: Path,
    *,
    regions_by_frame: dict[int, list[PrivacyRegion]],
    frame_start: int,
    frame_end: int,
    fps: float,
    width: int,
    height: int,
    keep_boxes_by_frame: dict[int, list[TrackSourceBox]],
) -> None:
    capture = cv.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_video}")
    writer = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create overlay video: {output_path}")
    try:
        for frame_index in range(frame_start, frame_end + 1):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(
                draw_overlay(
                    frame,
                    frame_index,
                    regions_by_frame.get(frame_index, []),
                    keep_boxes_by_frame.get(frame_index, []),
                )
            )
    finally:
        capture.release()
        writer.release()


def draw_overlay(
    frame: np.ndarray,
    frame_index: int,
    regions: list[PrivacyRegion],
    keep_boxes: list[TrackSourceBox],
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
    for keep_box in keep_boxes:
        draw_labeled_box(output, keep_box.bbox_xyxy, (255, 0, 255), f"keep T{keep_box.track_id}")
    for region in regions:
        if region.bbox_xyxy is None:
            cv.putText(
                output,
                "privacy gap unresolved",
                (12, 62),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            continue
        draw_labeled_box(output, region.bbox_xyxy, color_for_region(region), region.provenance)
    return output


def draw_labeled_box(
    frame: np.ndarray,
    box: Box,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = box
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv.putText(
        frame,
        label,
        (x1, max(18, y1 - 8)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv.LINE_AA,
    )


def color_for_region(region: PrivacyRegion) -> tuple[int, int, int]:
    return {
        PROVENANCE_DETECTOR: (0, 220, 0),
        PROVENANCE_ORDINARY_TRACKER: (0, 200, 220),
        PROVENANCE_INTERPOLATION: (0, 200, 220),
        PROVENANCE_BIDIRECTIONAL_BRIDGE: (255, 255, 0),
        PROVENANCE_FORWARD_BRIDGE: (255, 180, 0),
        PROVENANCE_BACKWARD_BRIDGE: (255, 120, 0),
        PROVENANCE_CONSERVATIVE_FALLBACK: (0, 0, 255),
        PROVENANCE_MANUAL_CORRECTION: (255, 255, 255),
        PROVENANCE_PRESERVED_REGION: (255, 255, 0),
    }.get(region.provenance, (200, 200, 200))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def settings_to_dict(settings: PrivacyGapRecoverySettings) -> dict[str, object]:
    return {
        "competing_identity_distance_norm": settings.competing_identity_distance_norm,
        "conservative_fallback_enabled": settings.conservative_fallback_enabled,
        "enable_visual_refinement": settings.enable_visual_refinement,
        "max_bridge_gap_frames": settings.max_bridge_gap_frames,
        "max_center_jump_norm": settings.max_center_jump_norm,
        "max_keep_overlap_fraction": settings.max_keep_overlap_fraction,
        "max_one_sided_gap_frames": settings.max_one_sided_gap_frames,
        "max_raw_disagreement_norm": settings.max_raw_disagreement_norm,
        "max_refined_disagreement_norm": settings.max_refined_disagreement_norm,
        "max_scale_change": settings.max_scale_change,
        "preserve_existing_regions": settings.preserve_existing_regions,
        "render_overlay": settings.render_overlay,
        "scene_cut_mean_delta": settings.scene_cut_mean_delta,
        "short_interpolation_gap_frames": settings.short_interpolation_gap_frames,
        "visual_refinement": settings.visual_refinement,
    }


def xyxy_to_xywh(box: Box) -> tuple[int, int, int, int]:
    return (box[0], box[1], box[2] - box[0], box[3] - box[1])


def xywh_to_xyxy(rect: tuple[float, float, float, float]) -> Box:
    x, y, width, height = rect
    return (round(x), round(y), round(x + width), round(y + height))


def clip_box(box: Box | tuple[float, float, float, float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = (round(value) for value in box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return (x1, y1, x2, y2)


def box_is_valid(box: Box) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def box_inside_frame(box: Box, width: int, height: int) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_center_distance(left: Box, right: Box) -> float:
    left_center = box_center(left)
    right_center = box_center(right)
    return math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])


def box_diagonal(box: Box) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def average_diagonal(left: Box, right: Box) -> float:
    return (box_diagonal(left) + box_diagonal(right)) / 2.0


def scale_change(left: Box, right: Box) -> float:
    left_width = max(left[2] - left[0], 1)
    left_height = max(left[3] - left[1], 1)
    right_width = max(right[2] - right[0], 1)
    right_height = max(right[3] - right[1], 1)
    return max(
        abs(right_width - left_width) / left_width,
        abs(right_height - left_height) / left_height,
    )


def intersection_fraction(left: Box, right: Box) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return intersection / max(min(box_area(left), box_area(right)), 1)


def interpolate_box(left: Box, right: Box, fraction: float) -> Box:
    return tuple(round(left[index] + (right[index] - left[index]) * fraction) for index in range(4))


def fuse_boxes(left: Box, right: Box) -> Box:
    left_center = box_center(left)
    right_center = box_center(right)
    left_width, left_height = left[2] - left[0], left[3] - left[1]
    right_width, right_height = right[2] - right[0], right[3] - right[1]
    center_x = (left_center[0] + right_center[0]) / 2.0
    center_y = (left_center[1] + right_center[1]) / 2.0
    width = math.exp((math.log(max(left_width, 1)) + math.log(max(right_width, 1))) / 2.0)
    height = math.exp((math.log(max(left_height, 1)) + math.log(max(right_height, 1))) / 2.0)
    return (
        round(center_x - width / 2),
        round(center_y - height / 2),
        round(center_x + width / 2),
        round(center_y + height / 2),
    )


def visual_quality_score(metadata: dict[str, object]) -> float:
    if "candidate_score" in metadata:
        return float(metadata["candidate_score"])
    if "evidence" in metadata and isinstance(metadata["evidence"], dict):
        evidence = metadata["evidence"]
        if "score" in evidence:
            return float(evidence["score"])
    if "strategy" in metadata and metadata["strategy"] == "none":
        return 0.0
    return -1.0


def box_changed(before: Box | None, after: Box | None) -> bool:
    return before is not None and after is not None and before != after


def only_selection_reason(reasons: list[str]) -> bool:
    allowed = {
        "backward_tracker_unavailable",
        "forward_tracker_unavailable",
    }
    return all(
        reason.endswith("_selected_by_visual_quality") or reason in allowed for reason in reasons
    )


def _optional_box(box: Box | None) -> list[int] | None:
    return None if box is None else list(box)


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
