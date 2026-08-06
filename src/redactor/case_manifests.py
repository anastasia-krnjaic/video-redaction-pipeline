from __future__ import annotations

import html.parser
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redactor import media

SCHEMA_VERSION = "case_manifest.v1"

CASE_STATUS_READY = "ready"
CASE_STATUS_BLOCKED = "blocked"
CASE_STATUSES = {CASE_STATUS_READY, CASE_STATUS_BLOCKED}

FPS_RATIONAL_DEFAULT = "30000/1001"


@dataclass(frozen=True)
class VideoRole:
    role: str
    path: str
    label: str
    source_frame_start: int
    source_frame_end: int
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "source_frame_end": self.source_frame_end,
            "source_frame_start": self.source_frame_start,
        }


@dataclass(frozen=True)
class CaseManifest:
    case_id: str
    display_name: str
    status: str
    source_video_hash: str | None
    source_path: str | None
    source_frame_start: int
    source_frame_end: int
    fps: str
    frame_count: int
    width: int | None
    height: int | None
    duration_seconds: float | None
    video_roles: tuple[VideoRole, ...] = ()
    face_evidence_inputs: tuple[str, ...] = ()
    plate_evidence_inputs: tuple[str, ...] = ()
    reviewed_correction_inputs: tuple[str, ...] = ()
    expected_output_roles: tuple[str, ...] = ("original", "current_result", "final_reviewed")
    parent_source_hash: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in CASE_STATUSES:
            raise ValueError(f"unsupported case status: {self.status}")
        if self.frame_count < 0:
            raise ValueError("frame_count cannot be negative")
        if self.source_frame_end < self.source_frame_start:
            raise ValueError("source_frame_end cannot be before source_frame_start")
        if self.status == CASE_STATUS_READY and not self.video_roles:
            raise ValueError("ready cases require at least one video role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockers": list(self.blockers),
            "case_id": self.case_id,
            "display_name": self.display_name,
            "duration_seconds": self.duration_seconds,
            "expected_output_roles": list(self.expected_output_roles),
            "face_evidence_inputs": list(self.face_evidence_inputs),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "height": self.height,
            "lineage": self.lineage,
            "parent_source_hash": self.parent_source_hash,
            "plate_evidence_inputs": list(self.plate_evidence_inputs),
            "reviewed_correction_inputs": list(self.reviewed_correction_inputs),
            "schema_version": self.schema_version,
            "source_frame_end": self.source_frame_end,
            "source_frame_start": self.source_frame_start,
            "source_path": self.source_path,
            "source_video_hash": self.source_video_hash,
            "status": self.status,
            "video_roles": [role.to_dict() for role in self.video_roles],
            "width": self.width,
        }

    def manifest_hash(self) -> str:
        from redactor.privacy_records import stable_hash

        return stable_hash(self.to_dict())


def write_case_manifest(path: Path, manifest: CaseManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_case_manifest(path: Path) -> CaseManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return case_manifest_from_dict(payload)


def case_manifest_from_dict(payload: dict[str, Any]) -> CaseManifest:
    return CaseManifest(
        blockers=tuple(str(item) for item in payload.get("blockers") or []),
        case_id=str(payload["case_id"]),
        display_name=str(payload["display_name"]),
        duration_seconds=_optional_float(payload.get("duration_seconds")),
        expected_output_roles=tuple(
            str(item) for item in payload.get("expected_output_roles") or []
        ),
        face_evidence_inputs=tuple(str(item) for item in payload.get("face_evidence_inputs") or []),
        fps=str(payload.get("fps") or FPS_RATIONAL_DEFAULT),
        frame_count=int(payload.get("frame_count") or 0),
        height=_optional_int(payload.get("height")),
        lineage=dict(payload.get("lineage") or {}),
        parent_source_hash=_optional_str(payload.get("parent_source_hash")),
        plate_evidence_inputs=tuple(
            str(item) for item in payload.get("plate_evidence_inputs") or []
        ),
        reviewed_correction_inputs=tuple(
            str(item) for item in payload.get("reviewed_correction_inputs") or []
        ),
        source_frame_end=int(payload.get("source_frame_end") or 0),
        source_frame_start=int(payload.get("source_frame_start") or 0),
        source_path=_optional_str(payload.get("source_path")),
        source_video_hash=_optional_str(payload.get("source_video_hash")),
        status=str(payload.get("status") or CASE_STATUS_BLOCKED),
        video_roles=tuple(video_role_from_dict(item) for item in payload.get("video_roles") or []),
        width=_optional_int(payload.get("width")),
    )


def video_role_from_dict(payload: dict[str, Any]) -> VideoRole:
    return VideoRole(
        label=str(payload["label"]),
        path=str(payload["path"]),
        role=str(payload["role"]),
        sha256=_optional_str(payload.get("sha256")),
        source_frame_end=int(payload.get("source_frame_end") or 0),
        source_frame_start=int(payload.get("source_frame_start") or 0),
    )


def case_manifest_from_media(
    *,
    case_id: str,
    display_name: str,
    source_video: Path,
    source_path: str,
    video_roles: list[VideoRole],
    source_frame_start: int = 0,
    parent_source_hash: str | None = None,
    lineage: dict[str, Any] | None = None,
    face_evidence_inputs: tuple[str, ...] = (),
    plate_evidence_inputs: tuple[str, ...] = (),
    reviewed_correction_inputs: tuple[str, ...] = (),
) -> CaseManifest:
    info = media.probe_media(source_video)
    frame_count = int(info.video_stream.frame_count or 0)
    return CaseManifest(
        case_id=case_id,
        display_name=display_name,
        duration_seconds=info.duration_seconds,
        face_evidence_inputs=face_evidence_inputs,
        fps=info.video_stream.nominal_frame_rate or FPS_RATIONAL_DEFAULT,
        frame_count=frame_count,
        height=info.video_stream.height,
        lineage=lineage or {},
        parent_source_hash=parent_source_hash,
        plate_evidence_inputs=plate_evidence_inputs,
        reviewed_correction_inputs=reviewed_correction_inputs,
        source_frame_end=source_frame_start + max(0, frame_count - 1),
        source_frame_start=source_frame_start,
        source_path=source_path,
        source_video_hash=info.sha256,
        status=CASE_STATUS_READY,
        video_roles=tuple(video_roles),
        width=info.video_stream.width,
    )


class MediaUrlParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"src", "href"} and value:
                self.urls.append((name, value))


def validate_relative_media_urls(
    *,
    html_path: Path,
    package_root: Path,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    html_path = html_path.resolve()
    parser = MediaUrlParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    checked: list[dict[str, Any]] = []
    errors: list[str] = []
    for attr, url in parser.urls:
        if _is_non_file_url(url):
            continue
        resolved = (html_path.parent / url.split("#", 1)[0].split("?", 1)[0]).resolve()
        item = {
            "attribute": attr,
            "exists": resolved.exists(),
            "relative_url": url,
            "resolved_package_path": _package_relative_or_placeholder(
                path=resolved,
                package_root=package_root,
            ),
        }
        try:
            resolved.relative_to(package_root)
        except ValueError:
            item["inside_package_root"] = False
            errors.append(f"{url} resolves outside package root")
        else:
            item["inside_package_root"] = True
        if not resolved.exists():
            errors.append(f"{url} does not resolve from {html_path}")
        checked.append(item)
    return {
        "checked_count": len(checked),
        "errors": errors,
        "items": checked,
        "passed": not errors,
        "schema_version": "media_url_audit.v1",
    }


def _is_non_file_url(url: str) -> bool:
    return (
        url.startswith("#")
        or url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("data:")
        or url.startswith("mailto:")
        or url.startswith("javascript:")
    )


def _package_relative_or_placeholder(*, path: Path, package_root: Path) -> str:
    try:
        return path.relative_to(package_root).as_posix()
    except ValueError:
        return "<outside-package-root>"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
