from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

from redactor.privacy_lifecycle import HysteresisDecision, TrackletStitchDecision
from redactor.privacy_records import (
    GAP_ACCEPTED,
    GAP_REJECTED,
    LIFECYCLE_CONFIRMED,
    LIFECYCLE_NEEDS_REVIEW,
    LIFECYCLE_REJECTED,
    LIFECYCLE_TEMPORARILY_LOST,
    LIFECYCLE_TENTATIVE,
    LIFECYCLE_TERMINATED,
    GapDecisionRecord,
    TemporalRegion,
    stable_hash,
)

SCHEMA_VERSION = "privacy_temporal_evidence.v1"

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TemporalEvidenceConfiguration:
    profile_id: str
    category: str
    component_weights: dict[str, float]
    confirmation_threshold: float
    continuation_threshold: float
    temporary_loss_threshold: float
    rejection_threshold: float
    min_persistence_seconds: float
    min_observation_count: int
    min_review_observation_count: int
    min_observation_density: float
    min_mean_confidence: float
    min_lower_tail_confidence: float
    max_safe_gap_seconds: float
    max_stitch_gap_seconds: float
    max_temporary_loss_seconds: float
    termination_gap_seconds: float
    max_center_jump_norm: float
    max_scale_jump: float
    max_aspect_ratio_std: float
    stitch_max_center_distance_norm: float
    stitch_max_scale_ratio: float
    stitch_max_aspect_ratio_delta: float
    screen_lock_min_seconds: float
    screen_lock_center_std_norm: float
    screen_lock_size_std_norm: float
    screen_lock_review_threshold: float
    max_competing_frame_fraction: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.component_weights:
            raise ValueError("component_weights are required")
        if not 0 <= self.rejection_threshold <= self.temporary_loss_threshold:
            raise ValueError("invalid lower evidence thresholds")
        if not self.temporary_loss_threshold <= self.continuation_threshold:
            raise ValueError("invalid continuation threshold ordering")
        if not self.continuation_threshold <= self.confirmation_threshold <= 1:
            raise ValueError("invalid confirmation threshold ordering")
        if self.min_observation_count < 1:
            raise ValueError("min_observation_count must be positive")
        if self.min_review_observation_count < 1:
            raise ValueError("min_review_observation_count must be positive")
        if self.min_review_observation_count > self.min_observation_count:
            raise ValueError("min_review_observation_count cannot exceed min_observation_count")
        if self.max_safe_gap_seconds < 0 or self.max_stitch_gap_seconds < 0:
            raise ValueError("gap limits cannot be negative")
        if self.max_temporary_loss_seconds < self.max_safe_gap_seconds:
            raise ValueError("temporary-loss span must cover safe-gap span")
        if self.termination_gap_seconds < self.max_temporary_loss_seconds:
            raise ValueError("termination span must cover temporary-loss span")

    def seconds_to_frames(self, seconds: float, fps: float) -> int:
        if seconds < 0:
            raise ValueError("seconds cannot be negative")
        return max(0, int(round(seconds * fps)))

    def min_persistence_frames(self, fps: float) -> int:
        return max(
            self.min_observation_count,
            self.seconds_to_frames(self.min_persistence_seconds, fps),
        )

    def max_safe_gap_frames(self, fps: float) -> int:
        return self.seconds_to_frames(self.max_safe_gap_seconds, fps)

    def max_stitch_gap_frames(self, fps: float) -> int:
        return self.seconds_to_frames(self.max_stitch_gap_seconds, fps)

    def max_temporary_loss_frames(self, fps: float) -> int:
        return self.seconds_to_frames(self.max_temporary_loss_seconds, fps)

    def termination_gap_frames(self, fps: float) -> int:
        return self.seconds_to_frames(self.termination_gap_seconds, fps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "component_weights": dict(sorted(self.component_weights.items())),
            "confirmation_threshold": self.confirmation_threshold,
            "continuation_threshold": self.continuation_threshold,
            "max_aspect_ratio_std": self.max_aspect_ratio_std,
            "max_center_jump_norm": self.max_center_jump_norm,
            "max_competing_frame_fraction": self.max_competing_frame_fraction,
            "max_safe_gap_seconds": self.max_safe_gap_seconds,
            "max_scale_jump": self.max_scale_jump,
            "max_stitch_gap_seconds": self.max_stitch_gap_seconds,
            "max_temporary_loss_seconds": self.max_temporary_loss_seconds,
            "min_lower_tail_confidence": self.min_lower_tail_confidence,
            "min_mean_confidence": self.min_mean_confidence,
            "min_observation_count": self.min_observation_count,
            "min_observation_density": self.min_observation_density,
            "min_persistence_seconds": self.min_persistence_seconds,
            "min_review_observation_count": self.min_review_observation_count,
            "profile_id": self.profile_id,
            "rejection_threshold": self.rejection_threshold,
            "schema_version": self.schema_version,
            "screen_lock_center_std_norm": self.screen_lock_center_std_norm,
            "screen_lock_min_seconds": self.screen_lock_min_seconds,
            "screen_lock_review_threshold": self.screen_lock_review_threshold,
            "screen_lock_size_std_norm": self.screen_lock_size_std_norm,
            "stitch_max_aspect_ratio_delta": self.stitch_max_aspect_ratio_delta,
            "stitch_max_center_distance_norm": self.stitch_max_center_distance_norm,
            "stitch_max_scale_ratio": self.stitch_max_scale_ratio,
            "temporary_loss_threshold": self.temporary_loss_threshold,
            "termination_gap_seconds": self.termination_gap_seconds,
        }

    def configuration_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class TrackEvidenceComponents:
    persistence: float
    observation_density: float
    mean_confidence: float
    lower_tail_confidence: float
    aspect_ratio_consistency: float
    center_motion_consistency: float
    scale_consistency: float
    source_provenance_support: float
    internal_gap_penalty: float
    abrupt_jump_penalty: float
    competing_track_penalty: float
    scene_cut_penalty: float
    screen_lock_risk: float

    def to_dict(self) -> dict[str, float]:
        return {
            "abrupt_jump_penalty": round(self.abrupt_jump_penalty, 6),
            "aspect_ratio_consistency": round(self.aspect_ratio_consistency, 6),
            "center_motion_consistency": round(self.center_motion_consistency, 6),
            "competing_track_penalty": round(self.competing_track_penalty, 6),
            "internal_gap_penalty": round(self.internal_gap_penalty, 6),
            "lower_tail_confidence": round(self.lower_tail_confidence, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "observation_density": round(self.observation_density, 6),
            "persistence": round(self.persistence, 6),
            "scale_consistency": round(self.scale_consistency, 6),
            "scene_cut_penalty": round(self.scene_cut_penalty, 6),
            "screen_lock_risk": round(self.screen_lock_risk, 6),
            "source_provenance_support": round(self.source_provenance_support, 6),
        }


@dataclass(frozen=True)
class TrackEvidenceSnapshot:
    category: str
    track_id: str
    first_frame: int | None
    last_frame: int | None
    region_count: int
    detector_observation_count: int
    span_frames: int
    accumulated_score: float
    components: TrackEvidenceComponents
    reasons: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "accumulated_score": round(self.accumulated_score, 6),
            "category": self.category,
            "components": self.components.to_dict(),
            "detector_observation_count": self.detector_observation_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "reasons": list(self.reasons),
            "region_count": self.region_count,
            "schema_version": self.schema_version,
            "span_frames": self.span_frames,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class EvidenceTracklet:
    category: str
    track_id: str
    regions: tuple[TemporalRegion, ...]
    gap_decisions: tuple[GapDecisionRecord, ...] = ()
    competing_frames: tuple[int, ...] = ()
    scene_cut_count: int = 0
    child_track_ids: tuple[str, ...] = ()

    @property
    def sorted_regions(self) -> tuple[TemporalRegion, ...]:
        return tuple(
            sorted(self.regions, key=lambda item: (item.source_frame, item.bbox_xyxy or ()))
        )

    @property
    def first_frame(self) -> int | None:
        regions = self.sorted_regions
        return regions[0].source_frame if regions else None

    @property
    def last_frame(self) -> int | None:
        regions = self.sorted_regions
        return regions[-1].source_frame if regions else None


@dataclass(frozen=True)
class RetrospectivePromotionDecision:
    category: str
    track_id: str
    first_confirmation_frame: int | None
    promoted_frames: tuple[int, ...]
    rejected_frames: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "first_confirmation_frame": self.first_confirmation_frame,
            "promoted_frames": list(self.promoted_frames),
            "reason_codes": list(self.reasons),
            "rejected_frames": list(self.rejected_frames),
            "schema_version": self.schema_version,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class TemporalEvidenceResult:
    track_id: str
    final_state: str
    snapshot: TrackEvidenceSnapshot
    transitions: tuple[HysteresisDecision, ...]
    promotion: RetrospectivePromotionDecision
    stitch_parent_id: str | None = None
    stitch_child_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_state": self.final_state,
            "promotion": self.promotion.to_dict(),
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.to_dict(),
            "stitch_child_ids": list(self.stitch_child_ids),
            "stitch_parent_id": self.stitch_parent_id,
            "track_id": self.track_id,
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


@dataclass(frozen=True)
class TemporalEvidenceRun:
    configuration_hash: str
    results: dict[str, TemporalEvidenceResult]
    stitch_decisions: tuple[TrackletStitchDecision, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_hash": self.configuration_hash,
            "results": {
                track_id: result.to_dict() for track_id, result in sorted(self.results.items())
            },
            "schema_version": self.schema_version,
            "stitch_decisions": [decision.to_dict() for decision in self.stitch_decisions],
        }


class TemporalEvidenceEngine:
    def __init__(self, configuration: TemporalEvidenceConfiguration, *, fps: float) -> None:
        self.configuration = configuration
        self.fps = fps

    def evaluate_tracklets(self, tracklets: tuple[EvidenceTracklet, ...]) -> TemporalEvidenceRun:
        stitch_decisions = self.stitch_tracklets(tracklets)
        stitched_children = stitch_children_by_track(stitch_decisions)
        base_results = {
            tracklet.track_id: self.evaluate_tracklet(tracklet) for tracklet in tracklets
        }
        promoted_by_stitch = self.promoted_track_ids_from_stitches(
            tracklets,
            stitch_decisions,
        )
        results: dict[str, TemporalEvidenceResult] = {}
        for track_id, result in base_results.items():
            if track_id in promoted_by_stitch and result.final_state != LIFECYCLE_REJECTED:
                result = TemporalEvidenceResult(
                    track_id=result.track_id,
                    final_state=LIFECYCLE_CONFIRMED,
                    snapshot=result.snapshot,
                    transitions=result.transitions,
                    promotion=result.promotion,
                    stitch_parent_id=promoted_by_stitch[track_id],
                    stitch_child_ids=stitched_children.get(promoted_by_stitch[track_id], ()),
                )
            results[track_id] = result
        return TemporalEvidenceRun(
            configuration_hash=self.configuration.configuration_hash(),
            results=results,
            stitch_decisions=stitch_decisions,
        )

    def evaluate_tracklet(self, tracklet: EvidenceTracklet) -> TemporalEvidenceResult:
        regions = tracklet.sorted_regions
        snapshot = self.snapshot(tracklet)
        if not regions:
            promotion = RetrospectivePromotionDecision(
                category=tracklet.category,
                track_id=tracklet.track_id,
                first_confirmation_frame=None,
                promoted_frames=(),
                rejected_frames=(),
                reasons=("empty_tracklet",),
            )
            return TemporalEvidenceResult(
                track_id=tracklet.track_id,
                final_state=LIFECYCLE_REJECTED,
                snapshot=snapshot,
                transitions=(
                    HysteresisDecision(
                        category=tracklet.category,
                        track_id=tracklet.track_id,
                        source_frame=0,
                        from_state=LIFECYCLE_TENTATIVE,
                        to_state=LIFECYCLE_REJECTED,
                        reason="empty_tracklet",
                        metrics=snapshot.to_dict(),
                    ),
                ),
                promotion=promotion,
            )

        transitions: list[HysteresisDecision] = [
            HysteresisDecision(
                category=tracklet.category,
                track_id=tracklet.track_id,
                source_frame=regions[0].source_frame,
                from_state=LIFECYCLE_TENTATIVE,
                to_state=LIFECYCLE_TENTATIVE,
                reason="first_observation",
                metrics={},
            )
        ]
        first_confirmation_frame = self.first_confirmation_frame(tracklet)
        if first_confirmation_frame is not None:
            transitions.append(
                HysteresisDecision(
                    category=tracklet.category,
                    track_id=tracklet.track_id,
                    source_frame=first_confirmation_frame,
                    from_state=LIFECYCLE_TENTATIVE,
                    to_state=LIFECYCLE_CONFIRMED,
                    reason="accumulated_evidence_confirmed",
                    metrics=snapshot.to_dict(),
                )
            )
            transitions.extend(self.gap_transitions(tracklet))
            final_state = LIFECYCLE_CONFIRMED
            final_reason = "confirmed_temporal_evidence"
            if (
                snapshot.components.screen_lock_risk
                >= self.configuration.screen_lock_review_threshold
            ):
                final_state = LIFECYCLE_NEEDS_REVIEW
                final_reason = "screen_lock_risk_requires_review"
            elif snapshot.accumulated_score < self.configuration.continuation_threshold:
                final_state = LIFECYCLE_TEMPORARILY_LOST
                final_reason = "below_continuation_threshold"
        elif snapshot.region_count < self.configuration.min_review_observation_count:
            final_state = LIFECYCLE_REJECTED
            final_reason = "insufficient_observations_for_review"
        elif snapshot.accumulated_score <= self.configuration.rejection_threshold:
            final_state = LIFECYCLE_REJECTED
            final_reason = "below_rejection_threshold"
        else:
            final_state = LIFECYCLE_NEEDS_REVIEW
            final_reason = "insufficient_accumulated_evidence"

        if transitions[-1].to_state != final_state:
            transitions.append(
                HysteresisDecision(
                    category=tracklet.category,
                    track_id=tracklet.track_id,
                    source_frame=regions[-1].source_frame,
                    from_state=transitions[-1].to_state,
                    to_state=final_state,
                    reason=final_reason,
                    metrics=snapshot.to_dict(),
                )
            )

        promotion = self.retrospective_promotion(
            tracklet,
            first_confirmation_frame=first_confirmation_frame
            if final_state == LIFECYCLE_CONFIRMED
            else None,
        )
        return TemporalEvidenceResult(
            track_id=tracklet.track_id,
            final_state=final_state,
            snapshot=snapshot,
            transitions=tuple(transitions),
            promotion=promotion,
        )

    def first_confirmation_frame(self, tracklet: EvidenceTracklet) -> int | None:
        regions = tracklet.sorted_regions
        for index in range(len(regions)):
            prefix = EvidenceTracklet(
                category=tracklet.category,
                track_id=tracklet.track_id,
                regions=tuple(regions[: index + 1]),
                gap_decisions=tuple(
                    decision
                    for decision in tracklet.gap_decisions
                    if decision.frame_end <= regions[index].source_frame
                ),
                competing_frames=tuple(
                    frame
                    for frame in tracklet.competing_frames
                    if frame <= regions[index].source_frame
                ),
                scene_cut_count=tracklet.scene_cut_count,
            )
            snapshot = self.snapshot(prefix)
            if self.confirmation_allowed(snapshot):
                return regions[index].source_frame
        return None

    def confirmation_allowed(self, snapshot: TrackEvidenceSnapshot) -> bool:
        components = snapshot.components
        reasons = set(snapshot.reasons)
        return (
            snapshot.accumulated_score >= self.configuration.confirmation_threshold
            and snapshot.detector_observation_count >= self.configuration.min_observation_count
            and "below_minimum_persistence" not in reasons
            and components.observation_density >= 1.0
            and components.mean_confidence >= self.configuration.min_mean_confidence
            and components.lower_tail_confidence >= self.configuration.min_lower_tail_confidence
            and components.screen_lock_risk < self.configuration.screen_lock_review_threshold
            and (
                components.competing_track_penalty
                <= self.configuration.max_competing_frame_fraction
            )
        )

    def snapshot(self, tracklet: EvidenceTracklet) -> TrackEvidenceSnapshot:
        regions = tracklet.sorted_regions
        components = self.components(tracklet)
        score = accumulated_score(components, self.configuration.component_weights)
        reasons = evidence_reasons(tracklet, components, self.configuration, self.fps)
        first = regions[0].source_frame if regions else None
        last = regions[-1].source_frame if regions else None
        span = max(0, (last - first + 1) if first is not None and last is not None else 0)
        return TrackEvidenceSnapshot(
            category=tracklet.category,
            track_id=tracklet.track_id,
            first_frame=first,
            last_frame=last,
            region_count=len(regions),
            detector_observation_count=detector_observation_count(regions),
            span_frames=span,
            accumulated_score=score,
            components=components,
            reasons=reasons,
        )

    def components(self, tracklet: EvidenceTracklet) -> TrackEvidenceComponents:
        regions = tracklet.sorted_regions
        if not regions:
            return TrackEvidenceComponents(*(0.0 for _ in range(13)))
        boxes = [region.bbox_xyxy for region in regions if region.bbox_xyxy is not None]
        scores = [score_from_region(region) for region in regions]
        scores = [score for score in scores if score is not None]
        first = regions[0].source_frame
        last = regions[-1].source_frame
        span = max(1, last - first + 1)
        min_persistence = self.configuration.min_persistence_frames(self.fps)
        persistence = clamp(len(regions) / max(1, min_persistence))
        density = clamp(
            (len(regions) / span) / max(self.configuration.min_observation_density, 1e-6)
        )
        mean_conf = mean(scores) if scores else 0.0
        lower_tail_conf = min(scores) if scores else 0.0
        aspect_consistency = consistency_from_std(
            aspect_ratios(boxes),
            self.configuration.max_aspect_ratio_std,
        )
        max_center_jump = max_pairwise_metric(boxes, center_distance_norm)
        center_consistency = clamp(
            1.0 - max_center_jump / max(self.configuration.max_center_jump_norm, 1e-6)
        )
        max_scale = max_pairwise_metric(boxes, bbox_scale_ratio, default=1.0)
        scale_consistency = clamp(
            1.0 - max(0.0, max_scale - 1.0) / max(self.configuration.max_scale_jump - 1.0, 1e-6)
        )
        provenance_support = provenance_support_score(regions)
        max_gap = max_internal_gap([region.source_frame for region in regions])
        internal_gap_penalty = clamp(
            max_gap / max(1, self.configuration.max_safe_gap_frames(self.fps) + 1)
        )
        abrupt_jump_penalty = max(
            clamp(max_center_jump / max(self.configuration.max_center_jump_norm, 1e-6) - 1.0),
            clamp(
                (max_scale - self.configuration.max_scale_jump)
                / max(self.configuration.max_scale_jump, 1e-6)
            ),
        )
        competing_penalty = clamp(len(set(tracklet.competing_frames)) / span)
        scene_cut_penalty = clamp(tracklet.scene_cut_count / max(1, len(regions)))
        screen_lock = screen_lock_risk(
            boxes,
            span_frames=span,
            fps=self.fps,
            configuration=self.configuration,
        )
        return TrackEvidenceComponents(
            persistence=round(persistence, 6),
            observation_density=round(density, 6),
            mean_confidence=round(mean_conf, 6),
            lower_tail_confidence=round(lower_tail_conf, 6),
            aspect_ratio_consistency=round(aspect_consistency, 6),
            center_motion_consistency=round(center_consistency, 6),
            scale_consistency=round(scale_consistency, 6),
            source_provenance_support=round(provenance_support, 6),
            internal_gap_penalty=round(internal_gap_penalty, 6),
            abrupt_jump_penalty=round(abrupt_jump_penalty, 6),
            competing_track_penalty=round(competing_penalty, 6),
            scene_cut_penalty=round(scene_cut_penalty, 6),
            screen_lock_risk=round(screen_lock, 6),
        )

    def gap_transitions(self, tracklet: EvidenceTracklet) -> tuple[HysteresisDecision, ...]:
        regions = tracklet.sorted_regions
        transitions: list[HysteresisDecision] = []
        temporary_loss_frames = self.configuration.max_temporary_loss_frames(self.fps)
        termination_frames = self.configuration.termination_gap_frames(self.fps)
        for left, right in zip(regions, regions[1:], strict=False):
            missing = right.source_frame - left.source_frame - 1
            if missing <= self.configuration.max_safe_gap_frames(self.fps):
                continue
            if missing <= temporary_loss_frames:
                transitions.append(
                    HysteresisDecision(
                        category=tracklet.category,
                        track_id=tracklet.track_id,
                        source_frame=left.source_frame + 1,
                        from_state=LIFECYCLE_CONFIRMED,
                        to_state=LIFECYCLE_TEMPORARILY_LOST,
                        reason="bounded_temporary_loss_gap",
                        metrics={"missing_frames": missing},
                    )
                )
                transitions.append(
                    HysteresisDecision(
                        category=tracklet.category,
                        track_id=tracklet.track_id,
                        source_frame=right.source_frame,
                        from_state=LIFECYCLE_TEMPORARILY_LOST,
                        to_state=LIFECYCLE_CONFIRMED,
                        reason="observation_resumed_after_temporary_loss",
                        metrics={"missing_frames": missing},
                    )
                )
            elif missing > termination_frames:
                transitions.append(
                    HysteresisDecision(
                        category=tracklet.category,
                        track_id=tracklet.track_id,
                        source_frame=left.source_frame + 1,
                        from_state=LIFECYCLE_CONFIRMED,
                        to_state=LIFECYCLE_TERMINATED,
                        reason="gap_exceeds_termination_limit",
                        metrics={"missing_frames": missing},
                    )
                )
        return tuple(transitions)

    def retrospective_promotion(
        self,
        tracklet: EvidenceTracklet,
        *,
        first_confirmation_frame: int | None,
    ) -> RetrospectivePromotionDecision:
        promoted: list[int] = []
        rejected: list[dict[str, Any]] = []
        if first_confirmation_frame is None:
            for region in tracklet.sorted_regions:
                rejected.append(
                    {
                        "frame": region.source_frame,
                        "reason": "tracklet_not_confirmed",
                    }
                )
            return RetrospectivePromotionDecision(
                category=tracklet.category,
                track_id=tracklet.track_id,
                first_confirmation_frame=None,
                promoted_frames=(),
                rejected_frames=tuple(rejected),
                reasons=("tracklet_not_confirmed",),
            )
        for region in tracklet.sorted_regions:
            if region.bbox_xyxy is None:
                rejected.append({"frame": region.source_frame, "reason": "missing_bbox"})
                continue
            if not region.render_eligible:
                rejected.append({"frame": region.source_frame, "reason": "not_render_eligible"})
                continue
            promoted.append(region.source_frame)
        return RetrospectivePromotionDecision(
            category=tracklet.category,
            track_id=tracklet.track_id,
            first_confirmation_frame=first_confirmation_frame,
            promoted_frames=tuple(sorted(set(promoted))),
            rejected_frames=tuple(rejected),
            reasons=("retrospective_promotion_after_confirmation",),
        )

    def stitch_tracklets(
        self,
        tracklets: tuple[EvidenceTracklet, ...],
    ) -> tuple[TrackletStitchDecision, ...]:
        decisions: list[TrackletStitchDecision] = []
        max_gap = self.configuration.max_stitch_gap_frames(self.fps)
        ordered = sorted(
            [tracklet for tracklet in tracklets if tracklet.sorted_regions],
            key=lambda item: (item.first_frame or 0, item.track_id),
        )
        for left_index, left in enumerate(ordered):
            left_regions = left.sorted_regions
            for right in ordered[left_index + 1 :]:
                if left.category != right.category:
                    continue
                if left.last_frame is None or right.first_frame is None:
                    continue
                gap = right.first_frame - left.last_frame - 1
                if gap < 0:
                    continue
                if gap > max_gap:
                    continue
                decision = self.stitch_decision(left, right, gap)
                decisions.append(decision)
                if decision.state == GAP_ACCEPTED:
                    break
            if not left_regions:
                continue
        return tuple(sorted(decisions, key=lambda item: (item.left_track_id, item.right_track_id)))

    def stitch_decision(
        self,
        left: EvidenceTracklet,
        right: EvidenceTracklet,
        gap: int,
    ) -> TrackletStitchDecision:
        left_box = last_box(left.sorted_regions)
        right_box = first_box(right.sorted_regions)
        reasons: list[str] = []
        metrics: dict[str, Any] = {"gap_frames": gap}
        if left_box is None or right_box is None:
            reasons.append("missing_endpoint_box")
        else:
            center_norm = center_distance_norm(left_box, right_box)
            scale_ratio = bbox_scale_ratio(left_box, right_box)
            aspect_delta = abs(aspect_ratio(left_box) - aspect_ratio(right_box))
            metrics.update(
                {
                    "center_distance_norm": round(center_norm, 6),
                    "scale_ratio": round(scale_ratio, 6),
                    "aspect_ratio_delta": round(aspect_delta, 6),
                }
            )
            if center_norm > self.configuration.stitch_max_center_distance_norm:
                reasons.append("stitch_center_motion_too_large")
            if scale_ratio > self.configuration.stitch_max_scale_ratio:
                reasons.append("stitch_scale_change_too_large")
            if aspect_delta > self.configuration.stitch_max_aspect_ratio_delta:
                reasons.append("stitch_aspect_change_too_large")
        if left.scene_cut_count or right.scene_cut_count:
            reasons.append("scene_cut_near_stitch")
        state = GAP_REJECTED if reasons else GAP_ACCEPTED
        return TrackletStitchDecision(
            category=left.category,
            left_track_id=left.track_id,
            right_track_id=right.track_id,
            state=state,
            reason="bounded_compatible_tracklet_stitch" if not reasons else ",".join(reasons),
            metrics=metrics,
        )

    def promoted_track_ids_from_stitches(
        self,
        tracklets: tuple[EvidenceTracklet, ...],
        decisions: tuple[TrackletStitchDecision, ...],
    ) -> dict[str, str]:
        by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
        promoted: dict[str, str] = {}
        for decision in decisions:
            if decision.state != GAP_ACCEPTED:
                continue
            left = by_id.get(decision.left_track_id)
            right = by_id.get(decision.right_track_id)
            if left is None or right is None:
                continue
            parent_id = stable_stitch_parent_id(left.track_id, right.track_id)
            combined = EvidenceTracklet(
                category=left.category,
                track_id=parent_id,
                regions=tuple([*left.sorted_regions, *right.sorted_regions]),
                gap_decisions=tuple([*left.gap_decisions, *right.gap_decisions]),
                competing_frames=tuple([*left.competing_frames, *right.competing_frames]),
                scene_cut_count=left.scene_cut_count + right.scene_cut_count,
                child_track_ids=(left.track_id, right.track_id),
            )
            snapshot = self.snapshot(combined)
            if self.confirmation_allowed(snapshot):
                promoted[left.track_id] = parent_id
                promoted[right.track_id] = parent_id
        return promoted


def stitch_children_by_track(
    decisions: tuple[TrackletStitchDecision, ...],
) -> dict[str, tuple[str, ...]]:
    children: dict[str, tuple[str, ...]] = {}
    for decision in decisions:
        if decision.state != GAP_ACCEPTED:
            continue
        parent_id = stable_stitch_parent_id(decision.left_track_id, decision.right_track_id)
        children[parent_id] = (decision.left_track_id, decision.right_track_id)
    return children


def stable_stitch_parent_id(left_track_id: str, right_track_id: str) -> str:
    digest = hashlib.sha256(f"{left_track_id}|{right_track_id}".encode()).hexdigest()[:10]
    return f"stitched_{digest}"


def accumulated_score(
    components: TrackEvidenceComponents,
    weights: dict[str, float],
) -> float:
    payload = components.to_dict()
    support_total = 0.0
    weight_total = 0.0
    for name, weight in weights.items():
        value = float(payload.get(name, 0.0))
        if weight < 0:
            support_total += weight * value
            weight_total += abs(weight)
        else:
            support_total += weight * value
            weight_total += weight
    if weight_total <= 0:
        return 0.0
    return round(clamp(support_total / weight_total), 6)


def evidence_reasons(
    tracklet: EvidenceTracklet,
    components: TrackEvidenceComponents,
    config: TemporalEvidenceConfiguration,
    fps: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(tracklet.sorted_regions) < config.min_observation_count:
        reasons.append("insufficient_observations")
    if detector_observation_count(tracklet.sorted_regions) < config.min_observation_count:
        reasons.append("insufficient_detector_observations")
    if len(tracklet.sorted_regions) < config.min_persistence_frames(fps):
        reasons.append("below_minimum_persistence")
    if components.observation_density < 1.0:
        reasons.append("low_observation_density")
    if components.mean_confidence < config.min_mean_confidence:
        reasons.append("low_mean_confidence")
    if components.lower_tail_confidence < config.min_lower_tail_confidence:
        reasons.append("weak_lower_tail_confidence")
    if components.internal_gap_penalty > 0:
        reasons.append("internal_gap_penalty")
    if components.abrupt_jump_penalty > 0:
        reasons.append("abrupt_jump_penalty")
    if components.screen_lock_risk >= config.screen_lock_review_threshold:
        reasons.append("screen_lock_risk")
    if components.competing_track_penalty > config.max_competing_frame_fraction:
        reasons.append("competing_track_penalty")
    if components.scene_cut_penalty > 0:
        reasons.append("scene_cut_penalty")
    return tuple(sorted(set(reasons)))


def score_from_region(region: TemporalRegion) -> float | None:
    if region.confidence is None:
        return None
    return float(region.confidence)


def detector_observation_count(regions: tuple[TemporalRegion, ...]) -> int:
    return sum(1 for region in regions if "detector" in region.source_provenance)


def provenance_support_score(regions: tuple[TemporalRegion, ...]) -> float:
    if not regions:
        return 0.0
    total = 0.0
    for region in regions:
        provenance = region.source_provenance
        if "detector" in provenance:
            total += 1.0
        elif "interpolation" in provenance:
            total += 0.65
        elif "review" in provenance:
            total += 0.80
        else:
            total += 0.35
    return total / len(regions)


def screen_lock_risk(
    boxes: list[Box],
    *,
    span_frames: int,
    fps: float,
    configuration: TemporalEvidenceConfiguration,
) -> float:
    if not boxes:
        return 0.0
    if span_frames < configuration.seconds_to_frames(configuration.screen_lock_min_seconds, fps):
        return 0.0
    diagonals = [bbox_diagonal(box) for box in boxes]
    mean_diag = max(1.0, mean(diagonals))
    center_std = (
        math.hypot(
            population_std([bbox_center(box)[0] for box in boxes]),
            population_std([bbox_center(box)[1] for box in boxes]),
        )
        / mean_diag
    )
    widths = [bbox_size(box)[0] for box in boxes]
    heights = [bbox_size(box)[1] for box in boxes]
    size_std = math.hypot(population_std(widths), population_std(heights)) / mean_diag
    center_risk = clamp(1.0 - center_std / max(configuration.screen_lock_center_std_norm, 1e-6))
    size_risk = clamp(1.0 - size_std / max(configuration.screen_lock_size_std_norm, 1e-6))
    return round(center_risk * size_risk, 6)


def consistency_from_std(values: list[float], max_std: float) -> float:
    if len(values) < 2:
        return 1.0
    std = population_std(values)
    return clamp(1.0 - std / max(max_std, 1e-6))


def max_pairwise_metric(
    boxes: list[Box],
    metric: Any,
    *,
    default: float = 0.0,
) -> float:
    if len(boxes) < 2:
        return default
    return max(metric(left, right) for left, right in zip(boxes, boxes[1:], strict=False))


def aspect_ratios(boxes: list[Box]) -> list[float]:
    return [aspect_ratio(box) for box in boxes if bbox_size(box)[1] > 0]


def aspect_ratio(box: Box) -> float:
    width, height = bbox_size(box)
    return width / max(1.0, height)


def max_internal_gap(frames: list[int]) -> int:
    if len(frames) < 2:
        return 0
    sorted_frames = sorted(frames)
    return max(
        right - left - 1 for left, right in zip(sorted_frames, sorted_frames[1:], strict=False)
    )


def first_box(regions: tuple[TemporalRegion, ...]) -> Box | None:
    for region in regions:
        if region.bbox_xyxy is not None:
            return region.bbox_xyxy
    return None


def last_box(regions: tuple[TemporalRegion, ...]) -> Box | None:
    for region in reversed(regions):
        if region.bbox_xyxy is not None:
            return region.bbox_xyxy
    return None


def bbox_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return float(x2 - x1), float(y2 - y1)


def bbox_diagonal(box: Box) -> float:
    width, height = bbox_size(box)
    return math.hypot(width, height)


def mean_bbox_diagonal(left: Box, right: Box) -> float:
    return max(1.0, (bbox_diagonal(left) + bbox_diagonal(right)) / 2.0)


def center_distance_norm(left: Box, right: Box) -> float:
    return euclidean(bbox_center(left), bbox_center(right)) / mean_bbox_diagonal(left, right)


def bbox_scale_ratio(left: Box, right: Box) -> float:
    left_width, left_height = bbox_size(left)
    right_width, right_height = bbox_size(right)
    width_ratio = max(left_width, right_width) / max(1.0, min(left_width, right_width))
    height_ratio = max(left_height, right_height) / max(1.0, min(left_height, right_height))
    return max(width_ratio, height_ratio)


def euclidean(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def deterministic_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
