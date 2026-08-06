from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from redactor import media
from redactor.privacy_records import CATEGORY_FACE, CATEGORY_PLATE, stable_hash

SCHEMA_VERSION = "privacy_entity.v1"
OBJECT_UUID_NAMESPACE = uuid.UUID("7a93792c-1ad6-4d45-9b45-9a92bb7814df")

POLICY_REDACT = "redact"
POLICY_KEEP_VISIBLE = "keep_visible"
POLICY_NOT_SENSITIVE = "not_sensitive"
POLICY_REVIEW_REQUIRED = "review_required"
POLICIES = {
    POLICY_REDACT,
    POLICY_KEEP_VISIBLE,
    POLICY_NOT_SENSITIVE,
    POLICY_REVIEW_REQUIRED,
}

STATUS_ACTIVE = "active"
STATUS_MERGED = "merged"
STATUS_SPLIT = "split"
STATUSES = {STATUS_ACTIVE, STATUS_MERGED, STATUS_SPLIT}


@dataclass(frozen=True)
class PolicyOverride:
    frame_start: int
    frame_end: int
    policy: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.frame_end < self.frame_start:
            raise ValueError("policy override frame_end cannot be before frame_start")
        if self.policy not in POLICIES:
            raise ValueError(f"unsupported entity policy: {self.policy}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_end": self.frame_end,
            "frame_start": self.frame_start,
            "policy": self.policy,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PrivacyEntity:
    entity_id: str
    case_id: str
    label: str
    category: str
    automatic_source_track_ids: tuple[str, ...] = ()
    reviewed_region_track_ids: tuple[str, ...] = ()
    default_privacy_policy: str = POLICY_REDACT
    interval_policy_overrides: tuple[PolicyOverride, ...] = ()
    current_status: str = STATUS_ACTIVE
    locked: bool = False
    created_provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def object_uuid(self) -> str:
        return self.entity_id.removeprefix("entity_")

    def __post_init__(self) -> None:
        if self.category not in {CATEGORY_FACE, CATEGORY_PLATE}:
            raise ValueError(f"unsupported privacy entity category: {self.category}")
        if self.default_privacy_policy not in POLICIES:
            raise ValueError(f"unsupported entity policy: {self.default_privacy_policy}")
        if self.current_status not in STATUSES:
            raise ValueError(f"unsupported entity status: {self.current_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "automatic_source_track_ids": list(self.automatic_source_track_ids),
            "case_id": self.case_id,
            "category": self.category,
            "created_provenance": self.created_provenance,
            "current_policy": self.default_privacy_policy,
            "current_status": self.current_status,
            "default_privacy_policy": self.default_privacy_policy,
            "display_name": self.label,
            "entity_id": self.entity_id,
            "entity_uuid": self.object_uuid,
            "geometry_locked": self.locked,
            "interval_policy_overrides": [
                override.to_dict() for override in self.interval_policy_overrides
            ],
            "label": self.label,
            "locked": self.locked,
            "member_detector_tracker_ids": list(self.automatic_source_track_ids),
            "merge_split_lineage": self.created_provenance.get("merge_split_lineage", []),
            "object_uuid": self.object_uuid,
            "privacy_action": self.default_privacy_policy,
            "review_state": "needs_review"
            if self.default_privacy_policy == POLICY_REVIEW_REQUIRED
            else "ready",
            "reviewed_region_track_ids": list(self.reviewed_region_track_ids),
            "schema_version": self.schema_version,
            "visibility_state": "visible",
        }

    def entity_hash(self) -> str:
        return stable_hash(self.to_dict())


def next_entity_label(category: str, existing: list[PrivacyEntity]) -> str:
    prefix = "Person" if category == CATEGORY_FACE else "Plate"
    legacy_prefix = "Human" if category == CATEGORY_FACE else "Plate"
    used: set[int] = set()
    for entity in existing:
        if entity.category != category:
            continue
        parts = entity.label.split()
        if len(parts) == 2 and parts[0] in {prefix, legacy_prefix} and parts[1].isdigit():
            used.add(int(parts[1]))
    index = 1
    while index in used:
        index += 1
    return f"{prefix} {index:02d}"


def create_entity(
    *,
    case_id: str,
    category: str,
    existing: list[PrivacyEntity] | None = None,
    automatic_source_track_ids: tuple[str, ...] = (),
    reviewed_region_track_ids: tuple[str, ...] = (),
    default_privacy_policy: str = POLICY_REDACT,
    entity_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> PrivacyEntity:
    existing = existing or []
    return PrivacyEntity(
        automatic_source_track_ids=tuple(sorted(set(automatic_source_track_ids))),
        case_id=case_id,
        category=category,
        created_provenance=provenance or {"source": "review_workspace"},
        default_privacy_policy=default_privacy_policy,
        entity_id=entity_id or f"entity_{uuid.uuid4()}",
        label=next_entity_label(category, existing),
        reviewed_region_track_ids=tuple(sorted(set(reviewed_region_track_ids))),
    )


def entities_from_region_records(case_id: str, records: list[Any]) -> list[PrivacyEntity]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        category = str(getattr(record, "category", ""))
        if category not in {CATEGORY_FACE, CATEGORY_PLATE}:
            continue
        source = dict(getattr(record, "source_record", {}) or {})
        track_id = (
            source.get("track_uid")
            or source.get("track_id")
            or getattr(record, "object_id", None)
            or source.get("region_id")
            or "unassigned"
        )
        key = (category, str(track_id))
        item = grouped.setdefault(
            key,
            {
                "first_frame": int(getattr(record, "source_frame", 0)),
                "track_ids": set(),
            },
        )
        item["first_frame"] = min(item["first_frame"], int(getattr(record, "source_frame", 0)))
        item["track_ids"].add(str(track_id))
    entities: list[PrivacyEntity] = []
    for (category, track_id), item in sorted(
        grouped.items(), key=lambda pair: (pair[0][0], pair[1]["first_frame"], pair[0][1])
    ):
        object_uuid = uuid.uuid5(OBJECT_UUID_NAMESPACE, f"{case_id}:{category}:{track_id}")
        entities.append(
            create_entity(
                case_id=case_id,
                category=category,
                existing=entities,
                automatic_source_track_ids=tuple(sorted(item["track_ids"])),
                entity_id=f"entity_{object_uuid}",
                provenance={
                    "first_source_frame": item["first_frame"],
                    "source": "automatic_region_index",
                    "track_id": track_id,
                },
            )
        )
    return entities


def rename_entity(entity: PrivacyEntity, label: str) -> PrivacyEntity:
    cleaned = label.strip()
    if not cleaned:
        raise ValueError("entity label cannot be empty")
    return replace(entity, label=cleaned)


def set_entity_lock(entity: PrivacyEntity, locked: bool) -> PrivacyEntity:
    return replace(entity, locked=locked)


def set_entity_policy(
    entity: PrivacyEntity,
    policy: str,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    reason: str = "",
) -> PrivacyEntity:
    if policy not in POLICIES:
        raise ValueError(f"unsupported entity policy: {policy}")
    if frame_start is None and frame_end is None:
        return replace(entity, default_privacy_policy=policy)
    if frame_start is None or frame_end is None:
        raise ValueError("interval policy changes require both frame_start and frame_end")
    override = PolicyOverride(
        frame_start=frame_start,
        frame_end=frame_end,
        policy=policy,
        reason=reason,
    )
    return replace(entity, interval_policy_overrides=(*entity.interval_policy_overrides, override))


def merge_entities(
    *,
    primary: PrivacyEntity,
    secondary: PrivacyEntity,
    label: str | None = None,
) -> tuple[PrivacyEntity, PrivacyEntity]:
    if primary.case_id != secondary.case_id or primary.category != secondary.category:
        raise ValueError("only entities from the same case and category can be merged")
    merged = replace(
        primary,
        automatic_source_track_ids=tuple(
            sorted(set(primary.automatic_source_track_ids + secondary.automatic_source_track_ids))
        ),
        interval_policy_overrides=(
            primary.interval_policy_overrides + secondary.interval_policy_overrides
        ),
        label=label or primary.label,
        reviewed_region_track_ids=tuple(
            sorted(set(primary.reviewed_region_track_ids + secondary.reviewed_region_track_ids))
        ),
    )
    retired = replace(secondary, current_status=STATUS_MERGED, locked=True)
    return merged, retired


def entity_frame_range(entity: PrivacyEntity) -> tuple[int | None, int | None]:
    provenance = entity.created_provenance or {}
    first = provenance.get("first_source_frame")
    last = provenance.get("last_source_frame", first)
    return (
        int(first) if first is not None else None,
        int(last) if last is not None else None,
    )


def can_conservatively_merge(
    primary: PrivacyEntity,
    secondary: PrivacyEntity,
) -> dict[str, Any]:
    reasons: list[str] = []
    if primary.case_id != secondary.case_id:
        reasons.append("different_case")
    if primary.category != secondary.category:
        reasons.append("different_category")
    first_a, last_a = entity_frame_range(primary)
    first_b, last_b = entity_frame_range(secondary)
    if None not in {first_a, last_a, first_b, last_b}:
        assert first_a is not None
        assert last_a is not None
        assert first_b is not None
        assert last_b is not None
        if max(first_a, first_b) <= min(last_a, last_b):
            reasons.append("simultaneously_visible")
    return {
        "allowed": not reasons,
        "primary_entity_id": primary.entity_id,
        "reasons": reasons,
        "schema_version": "privacy_entity_consolidation_check.v1",
        "secondary_entity_id": secondary.entity_id,
    }


def split_entity(
    *,
    entity: PrivacyEntity,
    moved_track_ids: tuple[str, ...],
    existing: list[PrivacyEntity] | None = None,
) -> tuple[PrivacyEntity, PrivacyEntity]:
    moved = set(moved_track_ids)
    if not moved:
        raise ValueError("split requires at least one moved track id")
    if not moved.issubset(set(entity.automatic_source_track_ids)):
        raise ValueError("cannot split track ids that are not assigned to the entity")
    remaining = tuple(item for item in entity.automatic_source_track_ids if item not in moved)
    original = replace(entity, automatic_source_track_ids=remaining, current_status=STATUS_SPLIT)
    new_entity = create_entity(
        case_id=entity.case_id,
        category=entity.category,
        existing=existing or [entity],
        automatic_source_track_ids=tuple(sorted(moved)),
        default_privacy_policy=entity.default_privacy_policy,
        provenance={"source": "entity_split", "parent_entity_id": entity.entity_id},
    )
    return original, new_entity


def reassign_region_track(
    *,
    source: PrivacyEntity,
    target: PrivacyEntity,
    track_id: str,
) -> tuple[PrivacyEntity, PrivacyEntity]:
    if source.case_id != target.case_id or source.category != target.category:
        raise ValueError("track reassignment requires same case and category")
    if track_id not in source.automatic_source_track_ids:
        raise ValueError("track id is not assigned to the source entity")
    updated_source = replace(
        source,
        automatic_source_track_ids=tuple(
            item for item in source.automatic_source_track_ids if item != track_id
        ),
    )
    updated_target = replace(
        target,
        automatic_source_track_ids=tuple(
            sorted(set(target.automatic_source_track_ids + (track_id,)))
        ),
    )
    return updated_source, updated_target


def entity_event(
    *,
    event_type: str,
    actor: dict[str, Any],
    case_id: str,
    entity_id: str,
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any] | None,
    reason: str = "",
) -> dict[str, Any]:
    payload = {
        "actor": actor,
        "after_state": after_state,
        "before_state": before_state,
        "case_id": case_id,
        "entity_id": entity_id,
        "event_id": f"entity_event_{uuid.uuid4().hex}",
        "event_type": event_type,
        "reason": reason,
        "schema_version": "privacy_entity_event.v1",
        "timestamp_utc": media.utc_timestamp(),
    }
    return payload | {"event_hash": stable_hash(payload)}
