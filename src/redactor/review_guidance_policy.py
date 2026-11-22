from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from redactor.privacy_records import stable_hash

POLICY_VERSION = "review_guidance_full_sweep.v1"
SWEEP_COVERAGE_STATUS = "COVERED_BY_CURRENT_FULL_VIDEO_SWEEP"
UNADJUDICATED_STATUS = "NOT_INDIVIDUALLY_ADJUDICATED"

_TASK_CACHE_LOCK = threading.Lock()
_TASK_CACHE: dict[tuple[str, int, int], tuple[dict[str, Any], ...]] = {}


def unreviewed_guidance_snapshot(
    *,
    package_root: Path,
    working_state: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    from redactor import review_working_state

    decisions_by_case = review_working_state.task_decisions_by_case_from_payload(working_state)
    resolutions_by_case = review_working_state.task_resolutions_by_case_from_payload(working_state)
    task_ids: list[str] = []
    for task in load_review_tasks(package_root):
        if not is_guidance_task(task, case_id=case_id):
            continue
        task_id = str(task.get("review_task_id") or "")
        if not task_id:
            continue
        decision = review_working_state.effective_task_state_for_case(
            decisions_by_case,
            resolutions_by_case,
            case_id=case_id,
            task_id=task_id,
        )
        if decision is None:
            task_ids.append(task_id)
    task_ids = sorted(set(task_ids))
    identity = {
        "case_id": case_id,
        "policy_version": POLICY_VERSION,
        "unreviewed_guidance_task_ids": task_ids,
    }
    return {
        "case_id": case_id,
        "guidance_identity_hash": stable_hash(identity),
        "guidance_unreviewed_count": len(task_ids),
        "guidance_unreviewed_task_ids": task_ids,
        "policy_version": POLICY_VERSION,
        "schema_version": "review_guidance_snapshot.v1",
        "sweep_coverage_status": SWEEP_COVERAGE_STATUS,
        "unadjudicated_status": UNADJUDICATED_STATUS,
    }


def is_guidance_task(task: dict[str, Any], *, case_id: str) -> bool:
    if str(task.get("case_id") or "") != case_id:
        return False
    if task.get("technical_only") or task.get("optional"):
        return False
    task_id = str(task.get("review_task_id") or "")
    if not task_id:
        return False
    normal_class = str(task.get("normal_task_class") or "")
    if normal_class == "correction_verification":
        return False
    return not (
        task_id.startswith("correction_verify_") or task_id.startswith("object_policy_verify_")
    )


def load_review_tasks(package_root: Path) -> list[dict[str, Any]]:
    path = package_root / "reviewer_tasks.json"
    if not path.exists():
        return []
    stat = path.stat()
    cache_key = (path.resolve().as_posix(), int(stat.st_size), int(stat.st_mtime_ns))
    with _TASK_CACHE_LOCK:
        cached = _TASK_CACHE.get(cache_key)
        if cached is not None:
            return [dict(task) for task in cached]
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = list(payload.get("tasks") or payload.get("reviewer_tasks") or [])
    tasks = tuple(dict(task) for task in raw_tasks)
    with _TASK_CACHE_LOCK:
        _TASK_CACHE.clear()
        _TASK_CACHE[cache_key] = tasks
    return [dict(task) for task in tasks]
