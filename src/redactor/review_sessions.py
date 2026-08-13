from __future__ import annotations

import json
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redactor import case_ids, media
from redactor.privacy_records import stable_hash
from redactor.review_corrections import ReviewCorrection, write_correction_jsonl

SCHEMA_VERSION = "review_session.v1"
EVENT_SCHEMA_VERSION = "review_session_event.v1"
SESSION_LAYOUT_DIRS = ("state", "events", "corrections", "checkpoints", "media")

EVENT_CORRECTION_PREVIEWED = "correction_previewed"
EVENT_CORRECTION_SAVED_LEGACY = "correction_saved"
EVENT_MANUAL_CORRECTION_SAVED = "manual_correction_saved"

ACTION_LOOKS_CORRECT = "looks_correct"
ACTION_PRIVACY_LEAK = "privacy_leak"
ACTION_PARTIAL_COVERAGE = "partial_coverage"
ACTION_FALSE_REDACTION = "false_redaction"
ACTION_WRONG_TARGET = "wrong_target"
ACTION_NOT_SENSITIVE = "not_sensitive"
ACTION_COSMETIC_ONLY = "cosmetic_only"
ACTION_ESCALATE = "uncertain_escalate"

DECISION_ACTIONS = {
    ACTION_LOOKS_CORRECT,
    ACTION_PRIVACY_LEAK,
    ACTION_PARTIAL_COVERAGE,
    ACTION_FALSE_REDACTION,
    ACTION_WRONG_TARGET,
    ACTION_NOT_SENSITIVE,
    ACTION_COSMETIC_ONLY,
    ACTION_ESCALATE,
}

GREEN_ACTIONS = {
    ACTION_LOOKS_CORRECT,
    ACTION_NOT_SENSITIVE,
    ACTION_COSMETIC_ONLY,
}
BLOCKING_ACTIONS = {
    ACTION_PRIVACY_LEAK,
    ACTION_PARTIAL_COVERAGE,
    ACTION_FALSE_REDACTION,
    ACTION_WRONG_TARGET,
    ACTION_ESCALATE,
}


def allowed_actions_for_task(task: dict[str, Any]) -> tuple[str, ...]:
    raw_actions = task.get("allowed_actions") or []
    actions: list[str] = []
    for raw_action in raw_actions:
        action = str(raw_action)
        if action not in DECISION_ACTIONS:
            raise ValueError(f"task allows unsupported decision action: {action}")
        if action not in actions:
            actions.append(action)
    return tuple(actions)


def invalid_task_action_reason(task: dict[str, Any] | None, action: str) -> str | None:
    if action not in DECISION_ACTIONS:
        return f"unsupported decision action: {action}"
    if task is None:
        return "decision references an unknown review task"
    task_id = str(task.get("review_task_id") or task.get("task_id") or "")
    try:
        allowed_actions = allowed_actions_for_task(task)
    except ValueError as exc:
        return f"{task_id}: {exc}"
    if not allowed_actions:
        return f"{task_id}: review task has no allowed decision actions"
    if action not in allowed_actions:
        allowed_text = ", ".join(allowed_actions)
        return (
            f"{task_id}: decision action {action} is not allowed; "
            f"allowed actions: {allowed_text}"
        )
    return None


def validate_task_action(task: dict[str, Any] | None, action: str) -> None:
    reason = invalid_task_action_reason(task, action)
    if reason is not None:
        raise ValueError(reason)


def invalid_task_decision_reason(
    task: dict[str, Any] | None,
    decision: dict[str, Any] | None,
) -> str | None:
    if decision is None:
        return None
    return invalid_task_action_reason(task, str(decision.get("action") or ""))


@dataclass(frozen=True)
class ReviewerIdentity:
    display_name: str | None = None
    reviewer_id: str | None = None
    role: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "notes": self.notes,
            "reviewer_id": self.reviewer_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class ReviewSession:
    session_id: str
    reviewer: ReviewerIdentity
    package_root: str
    started_at_utc: str
    source_revision: str | None
    application_version: str
    configuration_freeze_hash: str
    case_manifest_hashes: dict[str, str]
    source_hashes: dict[str, str]
    workspace_identity_hashes: dict[str, str] | None = None
    user_agent: str | None = None
    hostname: str | None = None
    session_status: str = "active"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_version": self.application_version,
            "case_manifest_hashes": self.case_manifest_hashes,
            "configuration_freeze_hash": self.configuration_freeze_hash,
            "hostname": self.hostname,
            "package_root": self.package_root,
            "reviewer": self.reviewer.to_dict(),
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "source_hashes": self.source_hashes,
            "source_revision": self.source_revision,
            "started_at_utc": self.started_at_utc,
            "user_agent": self.user_agent,
            "workspace_identity_hashes": self.workspace_identity_hashes or {},
        }


def create_review_session(
    *,
    package_root: Path,
    reviewer: ReviewerIdentity,
    source_revision: str | None,
    configuration_freeze_hash: str,
    case_manifest_hashes: dict[str, str],
    source_hashes: dict[str, str],
    workspace_identity_hashes: dict[str, str] | None = None,
    user_agent: str | None = None,
    include_hostname: bool = False,
) -> ReviewSession:
    session = ReviewSession(
        application_version="final_review_workspace.v2.3.1",
        case_manifest_hashes=dict(sorted(case_manifest_hashes.items())),
        configuration_freeze_hash=configuration_freeze_hash,
        hostname=socket.gethostname() if include_hostname else None,
        package_root=".",
        reviewer=reviewer,
        session_id=f"session_{uuid.uuid4().hex}",
        source_hashes=dict(sorted(source_hashes.items())),
        source_revision=source_revision,
        started_at_utc=media.utc_timestamp(),
        user_agent=user_agent,
        workspace_identity_hashes=dict(sorted((workspace_identity_hashes or {}).items())),
    )
    session_dir = session_directory(package_root, session.session_id)
    session_dir.mkdir(parents=True, exist_ok=False)
    ensure_session_layout(session_dir)
    write_json(session_dir / "review_session.json", session.to_dict())
    write_json(
        session_dir / "reviewer_attribution_current.json",
        {
            "attributed_at_utc": None,
            "display_name": None,
            "event_hash": None,
            "event_id": None,
            "schema_version": "reviewer_attribution_current.v1",
            "session_id": session.session_id,
        },
    )
    write_json(
        session_dir / "decisions_current.json",
        {"decisions": {}, "schema_version": "review_decisions_current.v1"},
    )
    write_json(
        session_dir / "review_summary.json",
        {
            "correction_count": 0,
            "decision_count": 0,
            "schema_version": "review_summary.v1",
            "session_id": session.session_id,
            "status": "active",
        },
    )
    append_event(
        session_dir,
        event_type="session_started",
        actor=reviewer.to_dict(),
        case_id=None,
        task_id=None,
        before_state=None,
        after_state={"session_status": "active"},
        note=(
            "Review session started anonymously. Reviewer attribution is locally declared "
            "when exporting or finalizing, not enterprise-authenticated."
        ),
        application_hashes={"configuration_freeze_hash": configuration_freeze_hash},
    )
    return session


def append_reviewer_attribution_event(
    session_dir: Path,
    *,
    display_name: str,
    actor: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    reviewer_name = display_name.strip()
    if not reviewer_name:
        raise ValueError("reviewer name is required")
    before = load_reviewer_attribution(session_dir)
    after = {
        "attributed_at_utc": media.utc_timestamp(),
        "display_name": reviewer_name,
        "schema_version": "reviewer_attribution_current.v1",
        "session_id": session_dir.name,
    }
    event = append_event(
        session_dir,
        event_type="reviewer_attribution_set",
        actor=actor or before or {"display_name": None, "reviewer_id": None},
        case_id=None,
        task_id=None,
        before_state=before,
        after_state=after,
        note=note or "Reviewer attribution set for the local review session.",
        application_hashes={},
    )
    write_json(
        session_dir / "reviewer_attribution_current.json",
        after | {"event_hash": event["event_hash"], "event_id": event["event_id"]},
    )
    refresh_session_summary(session_dir)
    return event


def append_decision_event(
    session_dir: Path,
    *,
    actor: dict[str, Any],
    case_id: str,
    task_id: str,
    action: str,
    note: str,
    task_before_state: dict[str, Any] | None = None,
    application_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if action not in DECISION_ACTIONS:
        raise ValueError(f"unsupported decision action: {action}")
    if task_before_state is not None:
        validate_task_action(task_before_state, action)
    current = load_current_decisions(session_dir)
    before = current.get(task_id)
    status = decision_status_for_action(action)
    after = {
        "action": action,
        "case_id": case_id,
        "decision_status": status,
        "note": note,
        "reviewed_at_utc": media.utc_timestamp(),
        "task_id": task_id,
    }
    event = append_event(
        session_dir,
        event_type="decision_recorded",
        actor=actor,
        case_id=case_id,
        task_id=task_id,
        before_state=before or task_before_state,
        after_state=after,
        note=note,
        application_hashes=application_hashes or {},
    )
    current[task_id] = after | {"event_id": event["event_id"], "event_hash": event["event_hash"]}
    write_json(
        session_dir / "decisions_current.json",
        {"decisions": current, "schema_version": "review_decisions_current.v1"},
    )
    refresh_session_summary(session_dir)
    return event


def append_correction_event(
    session_dir: Path,
    *,
    actor: dict[str, Any],
    correction: ReviewCorrection,
    before_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = append_event(
        session_dir,
        event_type=EVENT_MANUAL_CORRECTION_SAVED,
        actor=actor,
        case_id=correction.case_id,
        task_id=correction.task_id,
        before_state=before_state,
        after_state=correction.to_dict() | {"correction_hash": correction.correction_hash()},
        note=correction.reason,
        application_hashes={"configuration_freeze_hash": correction.configuration_freeze_hash},
    )
    write_correction_jsonl(session_dir / "corrections.jsonl", correction)
    write_correction_jsonl(session_dir / "corrections" / "corrections.jsonl", correction)
    refresh_session_summary(session_dir)
    return event


def append_correction_preview_event(
    session_dir: Path,
    *,
    actor: dict[str, Any],
    correction: ReviewCorrection,
    preview: dict[str, Any],
) -> dict[str, Any]:
    event = append_event(
        session_dir,
        event_type=EVENT_CORRECTION_PREVIEWED,
        actor=actor,
        case_id=correction.case_id,
        task_id=correction.task_id,
        before_state=None,
        after_state={
            "case_id": correction.case_id,
            "category": correction.category,
            "correction_id": correction.correction_id,
            "draft_hash": preview.get("draft_hash"),
            "effective_regions_hash": preview.get("effective_regions_hash"),
            "frame_end": correction.source_frame_end,
            "frame_start": correction.source_frame_start,
            "interpolation_method": correction.interpolation_method,
            "preview_sha256": preview.get("preview_sha256"),
            "preview_token": preview.get("preview_token"),
            "source_frame_end": correction.source_frame_end,
            "source_frame_start": correction.source_frame_start,
        },
        note=correction.reason or "Manual correction previewed.",
        application_hashes={"preview_sha256": str(preview.get("preview_sha256") or "")},
    )
    refresh_session_summary(session_dir)
    return event


def append_sweep_acknowledgement(
    session_dir: Path,
    *,
    actor: dict[str, Any],
    case_id: str,
    viewed_ranges: list[dict[str, Any]],
    override_reason: str | None = None,
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    after_state = {
        "override_reason": override_reason,
        "viewed_ranges": viewed_ranges,
    }
    if binding is not None:
        after_state = dict(binding) | after_state
    return append_event(
        session_dir,
        event_type="full_video_sweep_acknowledged",
        actor=actor,
        case_id=case_id,
        task_id=None,
        before_state=None,
        after_state=after_state,
        note=override_reason or "Reviewer acknowledged full-video sweep.",
        application_hashes={},
    )


def append_event(
    session_dir: Path,
    *,
    event_type: str,
    actor: dict[str, Any],
    case_id: str | None,
    task_id: str | None,
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any] | None,
    note: str | None,
    application_hashes: dict[str, str],
) -> dict[str, Any]:
    validated_case_id = None if case_id is None else case_ids.validate_case_id(case_id)
    previous_hash = last_event_hash(session_dir)
    payload = {
        "actor": actor,
        "after_state": after_state,
        "application_hashes": dict(sorted(application_hashes.items())),
        "before_state": before_state,
        "case_id": validated_case_id,
        "event_id": f"event_{uuid.uuid4().hex}",
        "event_type": event_type,
        "note": note,
        "previous_event_hash": previous_hash,
        "schema_version": EVENT_SCHEMA_VERSION,
        "task_id": task_id,
        "timestamp_utc": media.utc_timestamp(),
    }
    payload["event_hash"] = stable_hash(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    events_path = session_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as file:
        file.write(encoded)
    ensure_session_layout(session_dir)
    with (session_dir / "events" / "events.jsonl").open("a", encoding="utf-8") as file:
        file.write(encoded)
    return payload


def validate_event_hash_chain(session_dir: Path) -> dict[str, Any]:
    previous: str | None = None
    errors: list[str] = []
    count = 0
    for index, event in enumerate(load_events(session_dir)):
        count += 1
        if event.get("previous_event_hash") != previous:
            errors.append(f"events[{index}] previous hash mismatch")
        expected_hash = event.get("event_hash")
        without_hash = dict(event)
        without_hash.pop("event_hash", None)
        actual_hash = stable_hash(without_hash)
        if expected_hash != actual_hash:
            errors.append(f"events[{index}] event hash mismatch")
        previous = str(expected_hash)
    return {
        "error_count": len(errors),
        "errors": errors,
        "event_count": count,
        "passed": not errors,
        "schema_version": "review_event_chain_validation.v1",
    }


def load_events(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "events.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_event_hash(session_dir: Path) -> str | None:
    events = load_events(session_dir)
    return str(events[-1]["event_hash"]) if events else None


def load_current_decisions(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "decisions_current.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("decisions") or {})


def ensure_session_layout(session_dir: Path) -> None:
    for dirname in SESSION_LAYOUT_DIRS:
        (session_dir / dirname).mkdir(parents=True, exist_ok=True)


def refresh_session_summary(session_dir: Path) -> dict[str, Any]:
    events = load_events(session_dir)
    decisions = load_current_decisions(session_dir)
    corrections_path = session_dir / "corrections.jsonl"
    correction_count = 0
    if corrections_path.exists():
        correction_count = sum(
            1 for line in corrections_path.read_text(encoding="utf-8").splitlines() if line
        )
    summary = {
        "correction_count": correction_count,
        "decision_count": len(decisions),
        "event_count": len(events),
        "last_event_hash": events[-1]["event_hash"] if events else None,
        "reviewer_attribution": load_reviewer_attribution(session_dir),
        "schema_version": "review_summary.v1",
        "session_id": session_dir.name,
        "status": "active",
    }
    write_json(session_dir / "review_summary.json", summary)
    return summary


def decision_status_for_action(action: str) -> str:
    if action in GREEN_ACTIONS:
        return "green"
    if action in BLOCKING_ACTIONS:
        return "blocking"
    raise ValueError(f"unsupported decision action: {action}")


def session_directory(package_root: Path, session_id: str) -> Path:
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"_", "-"})
    if safe_id != session_id or not safe_id:
        raise ValueError("invalid session id")
    return package_root / "sessions" / safe_id


def reviewer_identity_from_dict(payload: dict[str, Any]) -> ReviewerIdentity:
    display_name = str(payload.get("display_name") or "").strip()
    reviewer_id = str(payload.get("reviewer_id") or payload.get("email") or "").strip()
    return ReviewerIdentity(
        display_name=display_name or None,
        notes=str(payload["notes"]) if payload.get("notes") is not None else None,
        reviewer_id=reviewer_id or None,
        role=str(payload["role"]) if payload.get("role") is not None else None,
    )


def load_reviewer_attribution(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / "reviewer_attribution_current.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
