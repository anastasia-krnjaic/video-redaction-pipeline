from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from redactor.privacy_records import (
    GAP_ACCEPTED,
    GAP_NEEDS_REVIEW,
    GAP_REJECTED,
    LIFECYCLE_STATES,
    GapDecisionRecord,
    PrivacyObservation,
    TemporalRegion,
)

SCHEMA_VERSION = "privacy_lifecycle_contracts.v1"


@dataclass(frozen=True)
class AssociationDecision:
    category: str
    track_id: str
    source_frame: int
    observation_hashes: tuple[str, ...]
    accepted: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "category": self.category,
            "metrics": self.metrics,
            "observation_hashes": list(self.observation_hashes),
            "reason": self.reason,
            "schema_version": self.schema_version,
            "source_frame": self.source_frame,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class EvidenceUpdate:
    category: str
    track_id: str
    source_frame: int
    observation_hashes: tuple[str, ...]
    reason: str
    score_delta: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "metrics": self.metrics,
            "observation_hashes": list(self.observation_hashes),
            "reason": self.reason,
            "schema_version": self.schema_version,
            "score_delta": round(float(self.score_delta), 6),
            "source_frame": self.source_frame,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class HysteresisDecision:
    category: str
    track_id: str
    source_frame: int
    from_state: str
    to_state: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.from_state not in LIFECYCLE_STATES or self.to_state not in LIFECYCLE_STATES:
            raise ValueError("unsupported hysteresis state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "from_state": self.from_state,
            "metrics": self.metrics,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "source_frame": self.source_frame,
            "to_state": self.to_state,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class TrackletStitchDecision:
    category: str
    left_track_id: str
    right_track_id: str
    state: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state not in {GAP_ACCEPTED, GAP_NEEDS_REVIEW, GAP_REJECTED}:
            raise ValueError(f"unsupported stitch decision state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "left_track_id": self.left_track_id,
            "metrics": self.metrics,
            "reason": self.reason,
            "right_track_id": self.right_track_id,
            "schema_version": self.schema_version,
            "state": self.state,
        }


@runtime_checkable
class AssociationPolicy(Protocol):
    def associate(
        self,
        *,
        track_id: str,
        observations: tuple[PrivacyObservation, ...],
        source_frame: int,
    ) -> AssociationDecision: ...


@runtime_checkable
class EvidenceAccumulator(Protocol):
    def update(
        self,
        *,
        track_id: str,
        observations: tuple[PrivacyObservation, ...],
        source_frame: int,
    ) -> EvidenceUpdate: ...


@runtime_checkable
class HysteresisPolicy(Protocol):
    def transition(
        self,
        *,
        track_id: str,
        current_state: str,
        source_frame: int,
        evidence: EvidenceUpdate,
    ) -> HysteresisDecision: ...


@runtime_checkable
class GapDecisionPolicy(Protocol):
    def decide_gap(
        self,
        *,
        track_id: str,
        frame_start: int,
        frame_end: int,
        anchors: tuple[TemporalRegion, ...],
    ) -> GapDecisionRecord: ...


@runtime_checkable
class RetrospectivePromotionPolicy(Protocol):
    def promote(
        self,
        *,
        track_id: str,
        regions: tuple[TemporalRegion, ...],
    ) -> HysteresisDecision: ...


@runtime_checkable
class TrackletStitchPolicy(Protocol):
    def stitch(
        self,
        *,
        left_track_id: str,
        right_track_id: str,
        gap: GapDecisionRecord,
    ) -> TrackletStitchDecision: ...
