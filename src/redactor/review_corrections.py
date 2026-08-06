from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redactor import media
from redactor.privacy_records import stable_hash

SCHEMA_VERSION = "review_correction.v1"

OP_ADD = "add_privacy_region"
OP_REPLACE = "replace_privacy_region"
OP_SUPPRESS = "suppress_region"
OP_NOT_SENSITIVE = "mark_not_sensitive"
OPERATIONS = {OP_ADD, OP_REPLACE, OP_SUPPRESS, OP_NOT_SENSITIVE}

INTERPOLATION_LINEAR = "linear"
INTERPOLATION_NONE = "none"
INTERPOLATION_METHODS = {INTERPOLATION_LINEAR, INTERPOLATION_NONE}

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
    input_region_ids: tuple[str, ...] = ()
    correction_id: str = field(default_factory=lambda: f"corr_{uuid.uuid4().hex}")
    created_at_utc: str = field(default_factory=media.utc_timestamp)
    application_version: str = "final_review_workspace.v1"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_version": self.application_version,
            "case_id": self.case_id,
            "category": self.category,
            "configuration_freeze_hash": self.configuration_freeze_hash,
            "correction_id": self.correction_id,
            "created_at_utc": self.created_at_utc,
            "input_region_ids": list(self.input_region_ids),
            "interpolation_method": self.interpolation_method,
            "keyframe_boxes": [keyframe.to_dict() for keyframe in self.keyframes],
            "operation": self.operation,
            "reason": self.reason,
            "reviewer_identity": self.reviewer_identity,
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
        application_version=str(payload.get("application_version") or "final_review_workspace.v1"),
        case_id=str(payload["case_id"]),
        category=str(payload["category"]),
        configuration_freeze_hash=str(payload["configuration_freeze_hash"]),
        correction_id=str(payload.get("correction_id") or f"corr_{uuid.uuid4().hex}"),
        created_at_utc=str(payload.get("created_at_utc") or media.utc_timestamp()),
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
        reason=str(payload.get("reason") or ""),
        reviewer_identity=dict(payload.get("reviewer_identity") or {}),
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
