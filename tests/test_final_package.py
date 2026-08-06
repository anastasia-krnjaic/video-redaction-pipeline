from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from redactor import cli as redactor_cli
from redactor import final_package


def source_record(
    *,
    logical_clip_id: str = "night",
    category: str = "face",
    sha: str = "source-sha",
    frame_count: int = 10,
    fps: float = 29.97,
    width: int = 1920,
    height: int = 1080,
) -> final_package.SourceArtifactRecord:
    return final_package.SourceArtifactRecord(
        logical_clip_id=logical_clip_id,
        category=category,
        role="reviewed",
        source_path="source.mp4",
        source_sha256=sha,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=frame_count / fps,
        artifact_path="artifact.mp4",
    )


def audit_task(task_id: str = "audit_0001", *, severity: str = "blocker") -> dict[str, object]:
    return {
        "audit_item_id": task_id,
        "category": "plate",
        "clip_id": "night",
        "issue_type": "visible_plate_leak",
        "notes": None,
        "object_count": None,
        "reviewer_decision": "pending_review",
        "severity": severity,
        "source_frame_end": 20,
        "source_frame_start": 10,
    }


def test_active_configuration_freeze_assertions() -> None:
    final_package.assert_active_configuration_frozen()
    snapshot = final_package.active_configuration_snapshot()

    assert snapshot["face"]["detector_id"] == final_package.FACE_DETECTOR_ID
    assert snapshot["face"]["production_shared_evidence_enabled"] is False
    assert snapshot["plate"]["detector_id"] == final_package.PLATE_DETECTOR_ID
    assert snapshot["plate"]["detector_configuration_hash"] == (
        final_package.PLATE_DETECTOR_CONFIGURATION_HASH
    )
    assert snapshot["plate"]["tracking_configuration_hash"] == (
        final_package.PLATE_TRACKING_CONFIGURATION_HASH
    )
    assert snapshot["plate"]["evidence_configuration_hash"] == (
        final_package.PLATE_EVIDENCE_CONFIGURATION_HASH
    )
    assert snapshot["plate"]["render_configuration_hash"] == (
        final_package.PLATE_RENDER_CONFIGURATION_HASH
    )


def test_source_compatibility_rejects_mismatched_source_hash() -> None:
    face = source_record(category="face", sha="same")
    plate = source_record(category="plate", sha="different")

    result = final_package.source_artifacts_compatible([face, plate])

    assert result["compatible"] is False
    assert "mismatched_source_sha256" in result["reasons"]


def test_source_compatibility_allows_matching_face_and_plate_records() -> None:
    face = source_record(category="face")
    plate = source_record(category="plate")

    matrix = final_package.build_source_compatibility_matrix({"night": [face, plate]})

    assert matrix["groups"]["night"]["can_combine_face_and_plate"] is True
    assert matrix["groups"]["night"]["compatibility"]["reasons"] == []


def test_pending_human_metrics_use_null_not_zero() -> None:
    pending = final_package.pending_human_metrics()

    assert pending["categories"]["face"]["video_frame_metrics"]["leak_frames"] is None
    assert pending["categories"]["plate"]["object_frame_metrics"]["leaked_object_frames"] is None
    assert pending["combined"]["total_blocker_intervals"] is None
    assert "0" not in json.dumps(pending)


def test_final_audit_task_and_decision_validation() -> None:
    tasks = {"schema_version": final_package.AUDIT_SCHEMA_VERSION, "tasks": [audit_task()]}
    decisions = {
        "schema_version": final_package.AUDIT_SCHEMA_VERSION,
        "decisions": [
            {
                "audit_item_id": "audit_0001",
                "reviewer_decision": "confirmed_issue",
                "reviewed_at": "2026-08-05T00:00:00Z",
            }
        ],
    }

    task_result = final_package.validate_audit_tasks(tasks)
    decision_result = final_package.validate_audit_decisions(decisions, tasks)
    summary = final_package.summarize_audit(tasks, decisions)

    assert task_result["passed"] is True
    assert decision_result["passed"] is True
    assert decision_result["unresolved_blocking_items"] == []
    assert summary["human_metrics_ready"] is True


def test_final_audit_validation_rejects_duplicate_decisions() -> None:
    tasks = {"schema_version": final_package.AUDIT_SCHEMA_VERSION, "tasks": [audit_task()]}
    decisions = {
        "schema_version": final_package.AUDIT_SCHEMA_VERSION,
        "decisions": [
            {"audit_item_id": "audit_0001", "reviewer_decision": "confirmed_issue"},
            {"audit_item_id": "audit_0001", "reviewer_decision": "not_an_issue"},
        ],
    }

    result = final_package.validate_audit_decisions(decisions, tasks)

    assert result["passed"] is False
    assert any("duplicate decision" in error for error in result["errors"])


def test_final_audit_cli_helpers(tmp_path: Path) -> None:
    tasks_path = tmp_path / "audit_tasks.json"
    decisions_path = tmp_path / "audit_decisions.json"
    tasks_path.write_text(
        json.dumps({"schema_version": final_package.AUDIT_SCHEMA_VERSION, "tasks": [audit_task()]}),
        encoding="utf-8",
    )
    decisions_path.write_text(
        json.dumps({"schema_version": final_package.AUDIT_SCHEMA_VERSION, "decisions": []}),
        encoding="utf-8",
    )

    runner = CliRunner()
    summary_result = runner.invoke(
        redactor_cli.main,
        ["summarize-final-audit", "--tasks", str(tasks_path)],
    )
    validation_result = runner.invoke(
        redactor_cli.main,
        [
            "validate-final-audit-decisions",
            "--tasks",
            str(tasks_path),
            "--decisions",
            str(decisions_path),
        ],
    )

    assert summary_result.exit_code == 0
    assert "1 pending" in summary_result.output
    assert validation_result.exit_code == 0
    assert "1 unresolved blocking items" in validation_result.output
