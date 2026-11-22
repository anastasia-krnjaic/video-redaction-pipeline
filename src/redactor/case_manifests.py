from __future__ import annotations

import html.parser
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redactor import media

SCHEMA_VERSION = "case_manifest.v1"
ARTIFACT_REFERENCE_SCHEMA_VERSION = "artifact_reference.v1"
UNIFIED_CASE_MANIFEST_SCHEMA_VERSION = "unified_case_manifest.v1"

CASE_STATUS_READY = "ready"
CASE_STATUS_BLOCKED = "blocked"
CASE_STATUSES = {CASE_STATUS_READY, CASE_STATUS_BLOCKED}

FPS_RATIONAL_DEFAULT = "30000/1001"
NON_SEMANTIC_MANIFEST_KEYS = {
    "case_content_sha256",
    "created_at",
    "created_at_utc",
    "generated_at",
    "generated_at_utc",
    "manifest_hash",
    "timestamp",
    "timestamp_utc",
}


class ManifestValidationError(ValueError):
    """Raised when a case manifest would publish incomplete or stale lineage."""


@dataclass(frozen=True)
class ArtifactReference:
    role: str
    path: str
    sha256: str
    size_bytes: int
    category: str | None = None
    artifact_id: str | None = None
    schema_version: str = ARTIFACT_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.role:
            raise ManifestValidationError("artifact role is required")
        if not self.path:
            raise ManifestValidationError(f"{self.role}: artifact path is required")
        if not self.sha256:
            raise ManifestValidationError(f"{self.role}: artifact sha256 is required")
        if self.size_bytes < 0:
            raise ManifestValidationError(f"{self.role}: artifact size cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id
            or artifact_id_for(
                role=self.role,
                category=self.category,
            ),
            "category": self.category,
            "path": self.path,
            "role": self.role,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def artifact_id_for(*, role: str, category: str | None = None) -> str:
    namespace = category or "case"
    return f"{namespace}:{role}"


def build_artifact_reference(
    path: Path,
    *,
    case_root: Path,
    role: str,
    category: str | None = None,
    required: bool = True,
    allow_external: bool = False,
) -> ArtifactReference | None:
    if not path.exists():
        if required:
            raise ManifestValidationError(f"{role}: required artifact is missing: {path}")
        return None
    if not path.is_file():
        raise ManifestValidationError(f"{role}: expected file artifact, got directory: {path}")

    resolved_root = case_root.resolve()
    resolved_path = path.resolve()
    try:
        artifact_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        if not allow_external:
            raise ManifestValidationError(
                f"{role}: artifact escapes case root: {resolved_path}"
            ) from exc
        artifact_path = resolved_path.as_posix()

    return ArtifactReference(
        artifact_id=artifact_id_for(role=role, category=category),
        category=category,
        path=artifact_path,
        role=role,
        sha256=media.sha256_file(resolved_path),
        size_bytes=resolved_path.stat().st_size,
    )


def artifact_reference_from_dict(payload: dict[str, Any]) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=str(
            payload.get("artifact_id")
            or artifact_id_for(
                role=str(payload.get("role") or ""),
                category=_optional_str(payload.get("category")),
            )
        ),
        category=_optional_str(payload.get("category")),
        path=str(payload.get("path") or ""),
        role=str(payload.get("role") or ""),
        schema_version=str(payload.get("schema_version") or ARTIFACT_REFERENCE_SCHEMA_VERSION),
        sha256=str(payload.get("sha256") or ""),
        size_bytes=int(payload.get("size_bytes") if payload.get("size_bytes") is not None else -1),
    )


def validate_artifact_inventory(
    artifacts: list[ArtifactReference] | tuple[ArtifactReference, ...],
    *,
    case_root: Path,
) -> tuple[ArtifactReference, ...]:
    seen_ids: dict[str, ArtifactReference] = {}
    seen_paths: dict[str, ArtifactReference] = {}
    validated: list[ArtifactReference] = []
    for artifact in artifacts:
        artifact_id = artifact.artifact_id or artifact_id_for(
            role=artifact.role,
            category=artifact.category,
        )
        if artifact_id in seen_ids:
            raise ManifestValidationError(f"duplicate artifact role: {artifact_id}")
        if artifact.path in seen_paths:
            previous = seen_paths[artifact.path]
            if previous.sha256 != artifact.sha256 or previous.role != artifact.role:
                raise ManifestValidationError(
                    f"duplicate incompatible artifact path: {artifact.path}"
                )
            raise ManifestValidationError(f"duplicate artifact path: {artifact.path}")

        artifact_path = Path(artifact.path)
        resolved = artifact_path if artifact_path.is_absolute() else case_root / artifact_path
        if not resolved.exists():
            raise ManifestValidationError(
                f"{artifact_id}: referenced artifact is missing: {artifact.path}"
            )
        if not resolved.is_file():
            raise ManifestValidationError(
                f"{artifact_id}: referenced artifact is not a file: {artifact.path}"
            )
        actual_hash = media.sha256_file(resolved)
        if actual_hash != artifact.sha256:
            raise ManifestValidationError(
                f"{artifact_id}: artifact hash mismatch for {artifact.path}"
            )
        actual_size = resolved.stat().st_size
        if actual_size != artifact.size_bytes:
            raise ManifestValidationError(
                f"{artifact_id}: artifact size mismatch for {artifact.path}"
            )

        seen_ids[artifact_id] = artifact
        seen_paths[artifact.path] = artifact
        validated.append(artifact)
    return tuple(validated)


def validate_unified_case_manifest(payload: dict[str, Any], *, case_root: Path) -> None:
    if payload.get("schema_version") != UNIFIED_CASE_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"unsupported unified case manifest schema: {payload.get('schema_version')}"
        )
    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, list):
        raise ManifestValidationError("unified case manifest artifacts must be a list")
    artifacts = [artifact_reference_from_dict(item) for item in artifacts_payload]
    validate_artifact_inventory(artifacts, case_root=case_root)

    expected_content_hash = semantic_manifest_hash(payload)
    if payload.get("case_content_sha256") != expected_content_hash:
        raise ManifestValidationError("case_content_sha256 does not match semantic manifest")


def semantic_manifest_hash(payload: dict[str, Any]) -> str:
    from redactor.privacy_records import stable_hash

    return stable_hash(_semantic_manifest_payload(payload))


def publish_unified_case_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(payload, sort_keys=True))
    manifest["case_content_sha256"] = semantic_manifest_hash(manifest)
    from redactor.privacy_records import stable_hash

    manifest["manifest_hash"] = stable_hash(manifest)
    validate_unified_case_manifest(manifest, case_root=path.parent)

    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    reopened = json.loads(temp_path.read_text(encoding="utf-8"))
    validate_unified_case_manifest(reopened, case_root=path.parent)
    os.replace(temp_path, path)
    return manifest


def _semantic_manifest_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_manifest_payload(item)
            for key, item in sorted(value.items())
            if key not in NON_SEMANTIC_MANIFEST_KEYS
        }
    if isinstance(value, list):
        return [_semantic_manifest_payload(item) for item in value]
    return value


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
