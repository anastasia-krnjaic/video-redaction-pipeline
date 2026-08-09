from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redactor import media
from redactor.privacy_records import normalize_manual_redaction_category, stable_hash

SCHEMA_VERSION = "review_correction.v1"

OP_ADD = "add_privacy_region"
OP_REPLACE = "replace_privacy_region"
OP_SUPPRESS = "suppress_region"
OP_SUPPRESS_TRACK_SEGMENT = "suppress_track_segment"
OP_NOT_SENSITIVE = "mark_not_sensitive"
OP_NOT_VISIBLE = "mark_not_visible"
OPERATIONS = {
    OP_ADD,
    OP_REPLACE,
    OP_SUPPRESS,
    OP_SUPPRESS_TRACK_SEGMENT,
    OP_NOT_SENSITIVE,
    OP_NOT_VISIBLE,
}

INTERPOLATION_LINEAR = "linear"
INTERPOLATION_NONE = "none"
INTERPOLATION_METHODS = {INTERPOLATION_LINEAR, INTERPOLATION_NONE}

CHECKPOINT_UNREVIEWED = "unreviewed"
CHECKPOINT_VERIFIED = "verified"
CHECKPOINT_NEEDS_ADJUSTMENT = "needs_adjustment"
CHECKPOINT_NOT_VISIBLE = "not_visible"
CHECKPOINT_BLOCKED_BY_SCENE_CUT = "blocked_by_scene_cut"
CHECKPOINT_STATES = {
    CHECKPOINT_UNREVIEWED,
    CHECKPOINT_VERIFIED,
    CHECKPOINT_NEEDS_ADJUSTMENT,
    CHECKPOINT_NOT_VISIBLE,
    CHECKPOINT_BLOCKED_BY_SCENE_CUT,
}

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class CorrectionKeyframe:
    source_frame: int
    bbox_xyxy: Box

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "source_frame": self.source_frame,
        }


@dataclass(frozen=True)
class CorrectionCheckpoint:
    source_frame: int
    state: str = CHECKPOINT_UNREVIEWED
    bbox_xyxy: Box | None = None
    reviewer_note: str = ""

    def __post_init__(self) -> None:
        if self.state not in CHECKPOINT_STATES:
            raise ValueError(f"unsupported checkpoint state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "reviewer_note": self.reviewer_note,
            "source_frame": self.source_frame,
            "state": self.state,
        }


@dataclass(frozen=True)
class ReviewCorrection:
    case_id: str
    task_id: str
    category: str
    operation: str
    source_frame_start: int
    source_frame_end: int
    keyframes: tuple[CorrectionKeyframe, ...]
    interpolation_method: str
    reviewer_identity: dict[str, Any]
    reason: str
    source_video_hash: str
    configuration_freeze_hash: str
    checkpoints: tuple[CorrectionCheckpoint, ...] = ()
    entity_id: str | None = None
    input_region_ids: tuple[str, ...] = ()
    correction_id: str = field(default_factory=lambda: f"corr_{uuid.uuid4().hex}")
    revision_id: str = field(default_factory=lambda: f"corr_rev_{uuid.uuid4().hex}")
    previous_revision_id: str | None = None
    created_at_utc: str = field(default_factory=media.utc_timestamp)
    application_version: str = "final_review_workspace.v2.3.1"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_manual_redaction_category(self.category))
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported correction operation: {self.operation}")
        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(f"unsupported interpolation method: {self.interpolation_method}")
        if self.source_frame_end < self.source_frame_start:
            raise ValueError("correction source_frame_end cannot be before source_frame_start")
        if self.operation in {OP_ADD, OP_REPLACE} and not self.keyframes:
            raise ValueError("add/replace corrections require keyframes")
        for keyframe in self.keyframes:
            if keyframe.source_frame < self.source_frame_start:
                raise ValueError("keyframe cannot be before correction interval")
            if keyframe.source_frame > self.source_frame_end:
                raise ValueError("keyframe cannot be after correction interval")
        for checkpoint in self.checkpoints:
            if checkpoint.source_frame < self.source_frame_start:
                raise ValueError("checkpoint cannot be before correction interval")
            if checkpoint.source_frame > self.source_frame_end:
                raise ValueError("checkpoint cannot be after correction interval")
        self._validate_mode_specific_keyframes()

    def _validate_mode_specific_keyframes(self) -> None:
        if self.operation not in {OP_ADD, OP_REPLACE}:
            return
        source_frames = {keyframe.source_frame for keyframe in self.keyframes}
        if self.interpolation_method == INTERPOLATION_NONE:
            return
        if self.source_frame_start == self.source_frame_end:
            if self.source_frame_start not in source_frames:
                raise ValueError(
                    "single-frame correction requires a keyframe at source frame "
                    f"{self.source_frame_start}"
                )
            return
        if self.source_frame_start not in source_frames:
            raise ValueError(
                "linear interpolation requires a start keyframe at source frame "
                f"{self.source_frame_start}"
            )
        if self.source_frame_end not in source_frames:
            raise ValueError(
                "linear interpolation requires an end keyframe at source frame "
                f"{self.source_frame_end}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_version": self.application_version,
            "case_id": self.case_id,
            "category": self.category,
            "checkpoint_verifications": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "configuration_freeze_hash": self.configuration_freeze_hash,
            "correction_id": self.correction_id,
            "created_at_utc": self.created_at_utc,
            "entity_id": self.entity_id,
            "input_region_ids": list(self.input_region_ids),
            "interpolation_method": self.interpolation_method,
            "keyframe_boxes": [keyframe.to_dict() for keyframe in self.keyframes],
            "operation": self.operation,
            "previous_revision_id": self.previous_revision_id,
            "reason": self.reason,
            "reviewer_identity": self.reviewer_identity,
            "revision_id": self.revision_id,
            "schema_version": self.schema_version,
            "source_frame_end": self.source_frame_end,
            "source_frame_start": self.source_frame_start,
            "source_video_hash": self.source_video_hash,
            "task_id": self.task_id,
        }

    def correction_hash(self) -> str:
        return stable_hash(self.to_dict())


def correction_from_dict(payload: dict[str, Any]) -> ReviewCorrection:
    return ReviewCorrection(
        application_version=str(
            payload.get("application_version") or "final_review_workspace.v2.3.1"
        ),
        case_id=str(payload["case_id"]),
        category=str(payload["category"]),
        checkpoints=tuple(
            CorrectionCheckpoint(
                bbox_xyxy=tuple(int(value) for value in checkpoint["bbox_xyxy"])
                if checkpoint.get("bbox_xyxy") is not None
                else None,
                reviewer_note=str(checkpoint.get("reviewer_note") or ""),
                source_frame=int(checkpoint["source_frame"]),
                state=str(checkpoint.get("state") or CHECKPOINT_UNREVIEWED),
            )
            for checkpoint in payload.get("checkpoint_verifications")
            or payload.get("checkpoints")
            or []
        ),
        configuration_freeze_hash=str(payload["configuration_freeze_hash"]),
        correction_id=str(payload.get("correction_id") or f"corr_{uuid.uuid4().hex}"),
        created_at_utc=str(payload.get("created_at_utc") or media.utc_timestamp()),
        entity_id=str(payload["entity_id"]) if payload.get("entity_id") is not None else None,
        input_region_ids=tuple(str(item) for item in payload.get("input_region_ids") or []),
        interpolation_method=str(payload.get("interpolation_method") or INTERPOLATION_LINEAR),
        keyframes=tuple(
            CorrectionKeyframe(
                bbox_xyxy=tuple(int(value) for value in keyframe["bbox_xyxy"]),
                source_frame=int(keyframe["source_frame"]),
            )
            for keyframe in payload.get("keyframe_boxes") or payload.get("keyframes") or []
        ),
        operation=str(payload["operation"]),
        previous_revision_id=str(payload["previous_revision_id"])
        if payload.get("previous_revision_id") is not None
        else None,
        reason=str(payload.get("reason") or ""),
        reviewer_identity=dict(payload.get("reviewer_identity") or {}),
        revision_id=str(payload.get("revision_id") or f"corr_rev_{uuid.uuid4().hex}"),
        schema_version=str(payload.get("schema_version") or SCHEMA_VERSION),
        source_frame_end=int(payload["source_frame_end"]),
        source_frame_start=int(payload["source_frame_start"]),
        source_video_hash=str(payload["source_video_hash"]),
        task_id=str(payload["task_id"]),
    )


def interpolate_correction_boxes(correction: ReviewCorrection) -> dict[int, Box]:
    if correction.operation not in {OP_ADD, OP_REPLACE}:
        return {}
    if correction.interpolation_method == INTERPOLATION_NONE:
        if len(correction.keyframes) == 1:
            keyframe = correction.keyframes[0]
            return {
                frame: keyframe.bbox_xyxy
                for frame in range(correction.source_frame_start, correction.source_frame_end + 1)
            }
        return {keyframe.source_frame: keyframe.bbox_xyxy for keyframe in correction.keyframes}
    keyframes = sorted(correction.keyframes, key=lambda item: item.source_frame)
    if not keyframes:
        return {}
    boxes: dict[int, Box] = {}
    if len(keyframes) == 1:
        keyframe = keyframes[0]
        for frame in range(correction.source_frame_start, correction.source_frame_end + 1):
            boxes[frame] = keyframe.bbox_xyxy
        return boxes
    for left, right in zip(keyframes, keyframes[1:], strict=False):
        span = right.source_frame - left.source_frame
        if span <= 0:
            raise ValueError("correction keyframes must use increasing source frames")
        blocked = [
            checkpoint
            for checkpoint in correction.checkpoints
            if left.source_frame < checkpoint.source_frame < right.source_frame
            and checkpoint.state in {CHECKPOINT_NOT_VISIBLE, CHECKPOINT_BLOCKED_BY_SCENE_CUT}
        ]
        if blocked:
            blocked_frames = ", ".join(str(checkpoint.source_frame) for checkpoint in blocked)
            raise ValueError(
                "linear interpolation cannot cross not-visible spans or scene cuts "
                f"at frames {blocked_frames}"
            )
        for frame in range(left.source_frame, right.source_frame + 1):
            t = (frame - left.source_frame) / span
            boxes[frame] = tuple(
                int(round((1.0 - t) * left_value + t * right_value))
                for left_value, right_value in zip(left.bbox_xyxy, right.bbox_xyxy, strict=True)
            )
    missing = [
        frame
        for frame in range(correction.source_frame_start, correction.source_frame_end + 1)
        if frame not in boxes
    ]
    if missing:
        raise ValueError("linear interpolation cannot extrapolate outside reviewed keyframes")
    return boxes


def checkpoint_spacing_seconds(
    *,
    interval_seconds: float,
    high_motion: bool = False,
    very_high_motion: bool = False,
) -> float:
    if very_high_motion:
        return 0.25
    if high_motion:
        return 0.5
    if interval_seconds <= 0.5:
        return max(interval_seconds, 1 / 30)
    if interval_seconds <= 2.0:
        return max(interval_seconds / 2.0, 1 / 30)
    return 1.0


def suggested_keyframe_frames(
    *,
    source_frame_start: int,
    source_frame_end: int,
    fps: float,
    evidence_event_frames: tuple[int, ...] = (),
    high_motion: bool = False,
    very_high_motion: bool = False,
) -> tuple[int, ...]:
    if source_frame_end < source_frame_start:
        raise ValueError("suggested keyframe interval end cannot be before start")
    if fps <= 0:
        raise ValueError("fps must be positive")
    duration_seconds = (source_frame_end - source_frame_start) / fps
    frames = {source_frame_start, source_frame_end}
    if duration_seconds > 0.5 and duration_seconds <= 2.0 and source_frame_end > source_frame_start:
        frames.add((source_frame_start + source_frame_end) // 2)
    elif duration_seconds > 2.0:
        step = max(
            1, int(round(checkpoint_spacing_seconds(interval_seconds=duration_seconds) * fps))
        )
        frame = source_frame_start + step
        while frame < source_frame_end:
            frames.add(frame)
            frame += step
    if high_motion or very_high_motion:
        spacing = checkpoint_spacing_seconds(
            interval_seconds=duration_seconds,
            high_motion=high_motion,
            very_high_motion=very_high_motion,
        )
        step = max(1, int(round(spacing * fps)))
        frame = source_frame_start + step
        while frame < source_frame_end:
            frames.add(frame)
            frame += step
    for frame in evidence_event_frames:
        if source_frame_start <= frame <= source_frame_end:
            frames.add(frame)
    return tuple(sorted(frames))


def required_checkpoint_frames(
    *,
    source_frame_start: int,
    source_frame_end: int,
    fps: float,
    high_motion: bool = False,
    very_high_motion: bool = False,
) -> tuple[int, ...]:
    return suggested_keyframe_frames(
        source_frame_start=source_frame_start,
        source_frame_end=source_frame_end,
        fps=fps,
        high_motion=high_motion,
        very_high_motion=very_high_motion,
    )


def review_pages_for_interval(
    *,
    source_frame_start: int,
    source_frame_end: int,
    fps: float,
    page_seconds: float = 5.0,
) -> tuple[tuple[int, int], ...]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    page_frames = max(1, int(round(page_seconds * fps)))
    pages: list[tuple[int, int]] = []
    start = source_frame_start
    while start <= source_frame_end:
        end = min(source_frame_end, start + page_frames - 1)
        pages.append((start, end))
        start = end + 1
    return tuple(pages)


def default_checkpoints_for_interval(
    *,
    source_frame_start: int,
    source_frame_end: int,
    fps: float,
    high_motion: bool = False,
    very_high_motion: bool = False,
) -> tuple[CorrectionCheckpoint, ...]:
    return tuple(
        CorrectionCheckpoint(source_frame=frame)
        for frame in required_checkpoint_frames(
            source_frame_start=source_frame_start,
            source_frame_end=source_frame_end,
            fps=fps,
            high_motion=high_motion,
            very_high_motion=very_high_motion,
        )
    )


def correction_verification_blockers(correction: ReviewCorrection) -> list[str]:
    if correction.operation not in {OP_ADD, OP_REPLACE}:
        return []
    blockers: list[str] = []
    for checkpoint in correction.checkpoints:
        if checkpoint.state in {
            CHECKPOINT_UNREVIEWED,
            CHECKPOINT_NEEDS_ADJUSTMENT,
            CHECKPOINT_BLOCKED_BY_SCENE_CUT,
        }:
            blockers.append(
                f"{correction.correction_id}:{checkpoint.source_frame} checkpoint "
                f"is {checkpoint.state}"
            )
    return blockers


def add_or_update_keyframe(
    correction: ReviewCorrection,
    *,
    source_frame: int,
    bbox_xyxy: Box,
) -> ReviewCorrection:
    if source_frame < correction.source_frame_start or source_frame > correction.source_frame_end:
        raise ValueError("keyframe cannot be outside the correction interval")
    keyframes = [
        keyframe for keyframe in correction.keyframes if keyframe.source_frame != source_frame
    ]
    keyframes.append(CorrectionKeyframe(source_frame=source_frame, bbox_xyxy=bbox_xyxy))
    return _revision(
        correction,
        keyframes=tuple(sorted(keyframes, key=lambda item: item.source_frame)),
    )


def delete_keyframe(correction: ReviewCorrection, *, source_frame: int) -> ReviewCorrection:
    keyframes = tuple(
        keyframe for keyframe in correction.keyframes if keyframe.source_frame != source_frame
    )
    if correction.operation in {OP_ADD, OP_REPLACE} and not keyframes:
        raise ValueError("add/replace corrections require at least one keyframe")
    return _revision(correction, keyframes=keyframes)


def mark_checkpoint(
    correction: ReviewCorrection,
    *,
    source_frame: int,
    state: str,
    bbox_xyxy: Box | None = None,
    reviewer_note: str = "",
) -> ReviewCorrection:
    checkpoint = CorrectionCheckpoint(
        bbox_xyxy=bbox_xyxy,
        reviewer_note=reviewer_note,
        source_frame=source_frame,
        state=state,
    )
    checkpoints = [item for item in correction.checkpoints if item.source_frame != source_frame]
    checkpoints.append(checkpoint)
    keyframes = correction.keyframes
    if bbox_xyxy is not None and state == CHECKPOINT_VERIFIED:
        keyframes = add_or_update_keyframe(
            correction,
            source_frame=source_frame,
            bbox_xyxy=bbox_xyxy,
        ).keyframes
    return _revision(
        correction,
        checkpoints=tuple(sorted(checkpoints, key=lambda item: item.source_frame)),
        keyframes=keyframes,
    )


def _revision(correction: ReviewCorrection, **changes: Any) -> ReviewCorrection:
    payload = correction.to_dict()
    payload.update(
        {
            "created_at_utc": media.utc_timestamp(),
            "previous_revision_id": correction.revision_id,
            "revision_id": f"corr_rev_{uuid.uuid4().hex}",
        }
    )
    payload.update(changes)
    if "keyframes" in payload:
        payload["keyframe_boxes"] = [
            keyframe.to_dict() if hasattr(keyframe, "to_dict") else keyframe
            for keyframe in payload.pop("keyframes")
        ]
    if "checkpoints" in payload:
        payload["checkpoint_verifications"] = [
            checkpoint.to_dict() if hasattr(checkpoint, "to_dict") else checkpoint
            for checkpoint in payload.pop("checkpoints")
        ]
    return correction_from_dict(payload)


def write_correction_jsonl(path: Path, correction: ReviewCorrection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(correction.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def load_corrections_jsonl(path: Path) -> list[ReviewCorrection]:
    if not path.exists():
        return []
    corrections: list[ReviewCorrection] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                corrections.append(correction_from_dict(json.loads(line)))
    return corrections
