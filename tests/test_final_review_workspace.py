from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np

from redactor import (
    case_manifests,
    review_corrections,
    review_finalization,
    review_sessions,
    review_workspace,
)
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    stable_hash,
)


def write_synthetic_video(path: Path, *, frame_count: int = 5) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    try:
        for frame_index in range(frame_count):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (30 + frame_index * 20, 90, 140)
            cv.rectangle(frame, (20, 20), (54, 34), (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def synthetic_case_manifest(package_root: Path) -> case_manifests.CaseManifest:
    source = package_root / "media" / "synthetic" / "original_reference.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_video(source)
    manifest = case_manifests.case_manifest_from_media(
        case_id="synthetic",
        display_name="Synthetic",
        source_path="media/synthetic/original_reference.mp4",
        source_video=source,
        video_roles=[
            case_manifests.VideoRole(
                label="Original",
                path="media/synthetic/original_reference.mp4",
                role="original",
                sha256=None,
                source_frame_end=4,
                source_frame_start=0,
            )
        ],
    )
    case_manifests.write_case_manifest(package_root / "cases" / "synthetic.json", manifest)
    return manifest


def create_session(package_root: Path, manifest: case_manifests.CaseManifest) -> Path:
    freeze = {"schema_version": "freeze.test", "value": 1}
    write_json(package_root / "configuration_freeze.json", freeze)
    session = review_sessions.create_review_session(
        package_root=package_root,
        reviewer=review_sessions.ReviewerIdentity("Analyst", "analyst@example.test"),
        source_revision="test",
        configuration_freeze_hash=stable_hash(freeze),
        case_manifest_hashes={manifest.case_id: manifest.manifest_hash()},
        source_hashes={manifest.case_id: manifest.source_video_hash or ""},
    )
    return review_sessions.session_directory(package_root, session.session_id)


def test_case_manifest_and_workspace_media_url_validation(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "assets").mkdir(parents=True)
    (package / "assets" / "workspace.css").write_text("", encoding="utf-8")
    (package / "assets" / "workspace.js").write_text("", encoding="utf-8")
    (package / "index.html").write_text(
        '<link rel="stylesheet" href="assets/workspace.css">'
        '<script src="assets/workspace.js"></script>',
        encoding="utf-8",
    )

    audit = review_workspace.validate_workspace_media_urls(package)

    assert manifest.case_id == "synthetic"
    assert audit["passed"] is True
    assert {item["relative_url"] for item in audit["items"]} >= {
        "assets/workspace.css",
        "assets/workspace.js",
        "media/synthetic/original_reference.mp4",
    }
    assert all("resolved_path" not in item for item in audit["items"])
    assert all(not item["resolved_package_path"].startswith("/") for item in audit["items"])


def test_relative_media_url_validation_rejects_missing_case_media(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "cases").mkdir(parents=True)
    (package / "index.html").write_text("", encoding="utf-8")
    manifest = case_manifests.CaseManifest(
        case_id="bad",
        display_name="Bad",
        duration_seconds=None,
        fps="10/1",
        frame_count=1,
        height=10,
        source_frame_end=0,
        source_frame_start=0,
        source_path="media/bad/missing.mp4",
        source_video_hash="x",
        status=case_manifests.CASE_STATUS_READY,
        video_roles=(
            case_manifests.VideoRole(
                label="Original",
                path="media/bad/missing.mp4",
                role="original",
                source_frame_end=0,
                source_frame_start=0,
            ),
        ),
        width=10,
    )
    case_manifests.write_case_manifest(package / "cases" / "bad.json", manifest)

    audit = review_workspace.validate_workspace_media_urls(package)

    assert audit["passed"] is False
    assert any("missing.mp4" in error for error in audit["errors"])


def test_day_audit_frames_map_to_parent_source_frames() -> None:
    signals = [
        signal
        for signal in review_workspace.load_raw_audit_signals(
            Path("artifacts/final_pipeline_package_v1/audit/audit_tasks.json")
        )
        if signal["raw_signal_id"] == "day_plate_plate_review_0001"
    ]

    assert signals[0]["source_frame_start"] == 1830
    assert signals[0]["source_frame_end"] == 1851


def test_task_aggregation_preserves_raw_ids_and_separates_technical() -> None:
    raw = [
        {
            "case_id": "night",
            "category": CATEGORY_PLATE,
            "raw_signal_id": "a",
            "root_cause": "unresolved_gap",
            "schema_version": "raw_audit_signal.v1",
            "severity": "blocker",
            "source_clip_id": "night",
            "source_frame_end": 10,
            "source_frame_start": 0,
            "technical": False,
        },
        {
            "case_id": "night",
            "category": CATEGORY_PLATE,
            "raw_signal_id": "b",
            "root_cause": "unresolved_gap",
            "schema_version": "raw_audit_signal.v1",
            "severity": "blocker",
            "source_clip_id": "night",
            "source_frame_end": 18,
            "source_frame_start": 11,
            "technical": False,
        },
        {
            "case_id": "close_plate",
            "category": CATEGORY_PLATE,
            "raw_signal_id": "c",
            "root_cause": "optional_spot_check",
            "schema_version": "raw_audit_signal.v1",
            "severity": "low",
            "source_clip_id": "close_plate",
            "source_frame_end": 2,
            "source_frame_start": 2,
            "technical": True,
        },
    ]

    tasks = review_workspace.aggregate_reviewer_tasks(raw)

    merged = [task for task in tasks if task["case_id"] == "night"][0]
    technical = [task for task in tasks if task.get("technical_only")][0]
    assert merged["raw_signal_ids"] == ["a", "b"]
    assert merged["blocking"] is True
    assert technical["raw_signal_ids"] == ["c"]


def test_review_session_event_chain_and_decision_revision(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    session_dir = create_session(package, manifest)

    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="task_1",
        action=review_sessions.ACTION_PRIVACY_LEAK,
        note="visible",
    )
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="task_1",
        action=review_sessions.ACTION_NOT_SENSITIVE,
        note="not a plate",
    )

    chain = review_sessions.validate_event_hash_chain(session_dir)
    current = review_sessions.load_current_decisions(session_dir)

    assert chain["passed"] is True
    assert chain["event_count"] == 3
    assert current["task_1"]["action"] == review_sessions.ACTION_NOT_SENSITIVE


def test_event_hash_chain_detects_tampering(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    session_dir = create_session(package, manifest)
    events_path = session_dir / "events.jsonl"
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    event["event_type"] = "tampered"
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    chain = review_sessions.validate_event_hash_chain(session_dir)

    assert chain["passed"] is False
    assert any("event hash mismatch" in error for error in chain["errors"])


def test_correction_interpolation_and_no_extrapolation() -> None:
    correction = review_corrections.ReviewCorrection(
        case_id="case",
        category="license_plate",
        configuration_freeze_hash="hash",
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(
            review_corrections.CorrectionKeyframe(10, (10, 10, 30, 20)),
            review_corrections.CorrectionKeyframe(12, (12, 12, 34, 24)),
        ),
        operation=review_corrections.OP_ADD,
        reason="reviewed",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=12,
        source_frame_start=10,
        source_video_hash="source",
        task_id="task",
    )

    boxes = review_corrections.interpolate_correction_boxes(correction)

    assert boxes[11] == (11, 11, 32, 22)


def test_correction_interpolation_rejects_extrapolation() -> None:
    correction = review_corrections.ReviewCorrection(
        case_id="case",
        category="license_plate",
        configuration_freeze_hash="hash",
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(
            review_corrections.CorrectionKeyframe(11, (10, 10, 30, 20)),
            review_corrections.CorrectionKeyframe(12, (12, 12, 34, 24)),
        ),
        operation=review_corrections.OP_ADD,
        reason="reviewed",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=12,
        source_frame_start=10,
        source_video_hash="source",
        task_id="task",
    )

    try:
        review_corrections.interpolate_correction_boxes(correction)
    except ValueError as exc:
        assert "cannot extrapolate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("correction extrapolation was accepted")


def write_base_region(package: Path, case_id: str) -> None:
    record = PrivacyRegionRecord(
        approval_state=APPROVAL_ACCEPTED,
        bbox_xyxy=(20, 20, 54, 34),
        category=CATEGORY_PLATE,
        object_id="plate_1",
        provenance="plate_detector_observation",
        render_eligible=True,
        source_frame=0,
        source_record={"region_id": "region_1"},
    )
    review_finalization.write_privacy_region_records(
        package / "regions" / f"{case_id}_current_regions.jsonl",
        [record],
    )


def test_finalization_blocks_required_unresolved_task(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_base_region(package, manifest.case_id)
    session_dir = create_session(package, manifest)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "review_task_id": "task_1",
                    "technical_only": False,
                }
            ]
        },
    )

    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert result["finalized"] is False
    assert any("required task has no decision" in blocker for blocker in result["blockers"])


def test_synthetic_successful_finalization(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_base_region(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    session_dir = create_session(package, manifest)
    review_sessions.append_sweep_acknowledgement(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        viewed_ranges=[{"start": 0, "end": 4}],
    )

    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert result["finalized"] is True
    assert result["media_validation_passed"] is True
    assert (package / result["final_video"]).exists()


def test_workspace_server_serves_mp4_range_and_rejects_traversal(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(
            f"{base_url}/{manifest.video_roles[0].path}",
            headers={"Range": "bytes=0-15"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
            content_type = response.headers["Content-Type"]
            status = response.status
        assert status == 206
        assert content_type == "video/mp4"
        assert len(body) == 16

        try:
            urllib.request.urlopen(f"{base_url}/../pyproject.toml", timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("path traversal request unexpectedly succeeded")
    finally:
        server.shutdown()
        server.server_close()


def test_workspace_server_persists_reviewed_correction_event(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session_response = post_json(
            f"{base_url}/api/sessions",
            {
                "reviewer": {"display_name": "Analyst", "reviewer_id": "analyst@example.test"},
                "source_revision": "test",
            },
        )
        session = session_response["session"]
        correction = {
            "case_id": manifest.case_id,
            "category": CATEGORY_PLATE,
            "configuration_freeze_hash": session["configuration_freeze_hash"],
            "interpolation_method": review_corrections.INTERPOLATION_NONE,
            "keyframe_boxes": [{"bbox_xyxy": [20, 20, 54, 34], "source_frame": 0}],
            "operation": review_corrections.OP_ADD,
            "reason": "reviewed",
            "reviewer_identity": {"reviewer_id": "analyst@example.test"},
            "source_frame_end": 0,
            "source_frame_start": 0,
            "source_video_hash": manifest.source_video_hash,
            "task_id": "task_1",
        }

        correction_response = post_json(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        session_dir = package / "sessions" / session["session_id"]
        chain = review_sessions.validate_event_hash_chain(session_dir)

        assert correction_response["event"]["event_type"] == "correction_saved"
        assert chain["passed"] is True
        assert (session_dir / "corrections.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()
