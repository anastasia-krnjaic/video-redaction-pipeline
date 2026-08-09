from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from redactor import frame_spans, rendering

SCHEMA_VERSION = "privacy_pipeline_records.v1"

CATEGORY_FACE = "face"
CATEGORY_PLATE = "license_plate"
CATEGORY_OTHER_SENSITIVE = "other_sensitive"
MANUAL_REDACTION_CATEGORIES = (CATEGORY_FACE, CATEGORY_PLATE, CATEGORY_OTHER_SENSITIVE)
MANUAL_REDACTION_CATEGORY_LABELS = {
    CATEGORY_FACE: "Face / person",
    CATEGORY_PLATE: "Plate",
    CATEGORY_OTHER_SENSITIVE: "Other sensitive region",
}
MANUAL_REDACTION_CATEGORY_ALIASES = {
    "face": CATEGORY_FACE,
    "person": CATEGORY_FACE,
    "plate": CATEGORY_PLATE,
    "license_plate": CATEGORY_PLATE,
    "other_sensitive": CATEGORY_OTHER_SENSITIVE,
}

LIFECYCLE_TENTATIVE = "tentative"
LIFECYCLE_CONFIRMED = "confirmed"
LIFECYCLE_TEMPORARILY_LOST = "temporarily_lost"
LIFECYCLE_NEEDS_REVIEW = "needs_review"
LIFECYCLE_REJECTED = "rejected"
LIFECYCLE_TERMINATED = "terminated"
LIFECYCLE_STATES = {
    LIFECYCLE_TENTATIVE,
    LIFECYCLE_CONFIRMED,
    LIFECYCLE_TEMPORARILY_LOST,
    LIFECYCLE_NEEDS_REVIEW,
    LIFECYCLE_REJECTED,
    LIFECYCLE_TERMINATED,
}

APPROVAL_ACCEPTED = "accepted"
APPROVAL_NEEDS_REVIEW = "needs_review"
APPROVAL_REJECTED = "rejected"
APPROVAL_MANUAL = "manual"

GAP_ACCEPTED = "accepted"
GAP_NEEDS_REVIEW = "needs_review"
GAP_REJECTED = "rejected"

ISSUE_UNRESOLVED_GAP = "unresolved_gap"
ISSUE_UNCERTAIN_TRACK = "uncertain_track"
ISSUE_POSSIBLE_FALSE_POSITIVE = "possible_false_positive"
ISSUE_CONTINUITY_CHECK = "continuity_check"
ISSUE_CORRECTED_RESULT_CHECK = "corrected_result_check"
ISSUE_OPTIONAL_SPOT_CHECK = "optional_spot_check"

Box = tuple[int, int, int, int]


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_manual_redaction_category(category: str) -> str:
    normalized = MANUAL_REDACTION_CATEGORY_ALIASES.get(category)
    if normalized is None:
        raise ValueError(f"unsupported privacy category: {category}")
    return normalized


def manual_redaction_category_label(category: str) -> str:
    normalized = normalize_manual_redaction_category(category)
    return MANUAL_REDACTION_CATEGORY_LABELS[normalized]


@dataclass(frozen=True)
class PrivacyObservation:
    category: str
    detector_id: str
    source_frame: int
    score: float
    bbox_xyxy: Box
    detector_provenance: str
    configuration_hash: str
    timing_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "category": self.category,
            "configuration_hash": self.configuration_hash,
            "detector_id": self.detector_id,
            "detector_provenance": self.detector_provenance,
            "schema_version": self.schema_version,
            "score": round(float(self.score), 6),
            "source_frame": int(self.source_frame),
            "timing_metadata": self.timing_metadata,
        }

    def record_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class TemporalRegion:
    category: str
    object_id: str
    source_frame: int
    bbox_xyxy: Box | None
    source_provenance: str
    track_state: str
    render_eligible: bool
    review_status: str
    confidence: float | None = None
    reasons: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.track_state not in LIFECYCLE_STATES:
            raise ValueError(f"unsupported lifecycle state: {self.track_state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "category": self.category,
            "object_id": self.object_id,
            "reasons": list(self.reasons),
            "render_eligible": self.render_eligible,
            "review_status": self.review_status,
            "schema_version": self.schema_version,
            "confidence": round(float(self.confidence), 6) if self.confidence is not None else None,
            "source_frame": self.source_frame,
            "source_provenance": self.source_provenance,
            "track_state": self.track_state,
        }


@dataclass(frozen=True)
class GapDecisionRecord:
    category: str
    track_id: str
    frame_start: int
    frame_end: int
    strategy: str
    state: str
    reasons: tuple[str, ...] = ()
    recovered_frames: int = 0
    unresolved_frames: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.frame_end < self.frame_start:
            raise ValueError("gap frame_end cannot be before frame_start")
        if self.state not in {GAP_ACCEPTED, GAP_NEEDS_REVIEW, GAP_REJECTED}:
            raise ValueError(f"unsupported gap decision state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
            "recovered_frames": self.recovered_frames,
            "schema_version": self.schema_version,
            "state": self.state,
            "strategy": self.strategy,
            "track_id": self.track_id,
            "unresolved_frames": self.unresolved_frames,
        }


@dataclass(frozen=True)
class PrivacyRegionRecord:
    category: str
    source_frame: int
    bbox_xyxy: Box | None
    provenance: str
    approval_state: str
    render_eligible: bool
    object_id: str | None = None
    reasons: tuple[str, ...] = ()
    source_record: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_state": self.approval_state,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "category": self.category,
            "object_id": self.object_id,
            "provenance": self.provenance,
            "reasons": list(self.reasons),
            "render_eligible": self.render_eligible,
            "schema_version": self.schema_version,
            "source_frame": self.source_frame,
            "source_record": self.source_record,
        }

    def to_render_region(self) -> rendering.RenderRegion:
        return rendering.RenderRegion(
            frame_index=self.source_frame,
            bbox_xyxy=self.bbox_xyxy,
            provenance=self.provenance,
            state="covered"
            if self.render_eligible and self.bbox_xyxy is not None
            else "unresolved",
            source_record=self.to_dict(),
            reasons=self.reasons,
        )


def privacy_region_from_render_region(
    category: str,
    region: rendering.RenderRegion,
    *,
    object_id: str | None = None,
    approval_state: str | None = None,
) -> PrivacyRegionRecord:
    render_eligible = region.is_renderable
    return PrivacyRegionRecord(
        category=category,
        source_frame=region.frame_index,
        bbox_xyxy=region.bbox_xyxy,
        provenance=region.provenance,
        approval_state=approval_state
        or (APPROVAL_ACCEPTED if render_eligible else APPROVAL_NEEDS_REVIEW),
        render_eligible=render_eligible,
        object_id=object_id,
        reasons=region.reasons,
        source_record=dict(region.source_record),
    )


def privacy_regions_from_render_regions(
    category: str,
    regions_by_frame: dict[int, list[rendering.RenderRegion]],
    *,
    object_id: str | None = None,
) -> list[PrivacyRegionRecord]:
    records: list[PrivacyRegionRecord] = []
    for frame_index in sorted(regions_by_frame):
        for region in regions_by_frame[frame_index]:
            records.append(
                privacy_region_from_render_region(
                    category,
                    region,
                    object_id=object_id,
                )
            )
    return records


@dataclass(frozen=True)
class ProcessingSpan:
    source_frame_count: int | None
    processed_frame_count: int
    first_processed_frame: int | None
    last_processed_frame: int | None
    missing_processed_ranges: tuple[tuple[int, int], ...]
    full_source_processed: bool
    schema_version: str = "processing_span.v1"

    @classmethod
    def from_frame_indices(
        cls,
        frame_indices: list[int],
        *,
        source_frame_count: int | None,
    ) -> ProcessingSpan:
        report = frame_spans.build_frame_span_report(
            frame_indices,
            source_frame_count=source_frame_count,
        )
        return cls.from_report(report)

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> ProcessingSpan:
        return cls(
            source_frame_count=report["source_frame_count"],
            processed_frame_count=int(report["processed_frame_count"]),
            first_processed_frame=report["first_processed_frame"],
            last_processed_frame=report["last_processed_frame"],
            missing_processed_ranges=tuple(
                (int(item["frame_start"]), int(item["frame_end"]))
                for item in report["missing_processed_ranges"]
            ),
            full_source_processed=bool(report["full_source_processed"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_processed_frame": self.first_processed_frame,
            "full_source_processed": self.full_source_processed,
            "last_processed_frame": self.last_processed_frame,
            "missing_processed_ranges": [
                {"frame_start": start, "frame_end": end, "length": end - start + 1}
                for start, end in self.missing_processed_ranges
            ],
            "processed_frame_count": self.processed_frame_count,
            "schema_version": self.schema_version,
            "source_frame_count": self.source_frame_count,
        }


@dataclass(frozen=True)
class LifecycleTransition:
    category: str
    object_id: str
    source_frame: int
    from_state: str
    to_state: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.from_state not in LIFECYCLE_STATES or self.to_state not in LIFECYCLE_STATES:
            raise ValueError("unsupported lifecycle transition state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "from_state": self.from_state,
            "metrics": self.metrics,
            "object_id": self.object_id,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "source_frame": self.source_frame,
            "to_state": self.to_state,
        }


@dataclass(frozen=True)
class ReviewTaskRecord:
    category: str
    task_id: str
    object_id: str | None
    frame_start: int
    frame_end: int
    issue_type: str
    blocking: bool
    allowed_actions: tuple[str, ...]
    correction_required: bool
    media_roles: dict[str, str] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    schema_version: str = "privacy_review_task.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_actions": list(self.allowed_actions),
            "blocking": self.blocking,
            "category": self.category,
            "correction_required": self.correction_required,
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "issue_type": self.issue_type,
            "media_roles": dict(self.media_roles),
            "object_id": self.object_id,
            "reasons": list(self.reasons),
            "schema_version": self.schema_version,
            "task_id": self.task_id,
        }
