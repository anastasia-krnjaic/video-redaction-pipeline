from __future__ import annotations

import re
from pathlib import Path

CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class CaseIdError(ValueError):
    """Raised when a case ID is unsafe for filesystem, route, or DOM use."""


def validate_case_id(case_id: str) -> str:
    value = str(case_id or "")
    if not CASE_ID_PATTERN.fullmatch(value):
        raise CaseIdError(f"unsafe case_id: {value!r}")
    return value


def validate_case_ids(case_ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    validated = tuple(validate_case_id(case_id) for case_id in case_ids)
    if len(set(validated)) != len(validated):
        raise CaseIdError("duplicate case_id")
    return validated


def contained_path(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise CaseIdError(f"path escapes allowed root: {candidate}") from exc
    return candidate


def case_manifest_path(package_root: Path, case_id: str) -> Path:
    return contained_path(package_root, "cases", f"{validate_case_id(case_id)}.json")
