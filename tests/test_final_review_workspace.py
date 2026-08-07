from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np

from redactor import (
    case_manifests,
    privacy_entities,
    review_corrections,
    review_finalization,
    review_sessions,
    review_workspace,
)
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
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


def test_version_alignment_audit_accepts_matching_case_versions(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = package / "media" / "day" / "original_reference.mp4"
    current = package / "media" / "day" / "current_combined_result.mp4"
    source.parent.mkdir(parents=True)
    write_synthetic_video(source, frame_count=7)
    shutil.copy2(source, current)
    info = review_workspace.media.probe_media(source)
    manifest = case_manifests.CaseManifest(
        case_id="day",
        display_name="Day",
        duration_seconds=info.duration_seconds,
        fps=info.video_stream.nominal_frame_rate or "10/1",
        frame_count=7,
        height=64,
        source_frame_end=106,
        source_frame_start=100,
        source_path="media/day/original_reference.mp4",
        source_video_hash=info.sha256,
        status=case_manifests.CASE_STATUS_READY,
        video_roles=(
            case_manifests.VideoRole(
                label="Original",
                path="media/day/original_reference.mp4",
                role="original",
                source_frame_end=106,
                source_frame_start=100,
            ),
            case_manifests.VideoRole(
                label="Current result",
                path="media/day/current_combined_result.mp4",
                role="current_result",
                source_frame_end=106,
                source_frame_start=100,
            ),
        ),
        width=96,
    )
    case_manifests.write_case_manifest(package / "cases" / "day.json", manifest)

    audit = review_workspace.write_version_alignment_audit(package)

    assert audit["passed"] is True
    assert audit["cases"]["day"]["roles"][0]["case_to_source_mapping"]["3"] == 103
    assert audit["cases"]["day"]["roles"][1]["decoded_frame_evidence"]["6"]["decoded"] is True


def test_version_alignment_audit_rejects_mismatched_role_mapping(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = package / "media" / "day" / "original_reference.mp4"
    current = package / "media" / "day" / "current_combined_result.mp4"
    source.parent.mkdir(parents=True)
    write_synthetic_video(source, frame_count=5)
    shutil.copy2(source, current)
    info = review_workspace.media.probe_media(source)
    manifest = case_manifests.CaseManifest(
        case_id="day",
        display_name="Day",
        duration_seconds=info.duration_seconds,
        fps=info.video_stream.nominal_frame_rate or "10/1",
        frame_count=5,
        height=64,
        source_frame_end=204,
        source_frame_start=200,
        source_path="media/day/original_reference.mp4",
        source_video_hash=info.sha256,
        status=case_manifests.CASE_STATUS_READY,
        video_roles=(
            case_manifests.VideoRole(
                label="Original",
                path="media/day/original_reference.mp4",
                role="original",
                source_frame_end=204,
                source_frame_start=200,
            ),
            case_manifests.VideoRole(
                label="Current result",
                path="media/day/current_combined_result.mp4",
                role="current_result",
                source_frame_end=205,
                source_frame_start=201,
            ),
        ),
        width=96,
    )
    case_manifests.write_case_manifest(package / "cases" / "day.json", manifest)

    audit = review_workspace.write_version_alignment_audit(package)

    assert audit["passed"] is False
    assert any("role source start" in error for error in audit["errors"])


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


def test_finalization_blocks_reviewer_added_manual_issue(tmp_path: Path) -> None:
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
    review_sessions.append_event(
        session_dir,
        event_type="manual_issue_created",
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id=None,
        before_state=None,
        after_state={
            "task": {
                "blocking": True,
                "case_id": manifest.case_id,
                "review_task_id": "manual_issue_1",
            }
        },
        note="manual issue",
        application_hashes={},
    )

    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert result["finalized"] is False
    assert any("reviewer-added issue has no decision" in blocker for blocker in result["blockers"])


def test_finalization_blocks_not_sensitive_entity_policy_until_verified(tmp_path: Path) -> None:
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
    review_sessions.append_event(
        session_dir,
        event_type="entity_policy_changed",
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id=None,
        before_state=None,
        after_state={
            "entity_id": "entity_plate_1",
            "label": "Plate 01",
            "policy": privacy_entities.POLICY_NOT_SENSITIVE,
            "verification_task_id": "object_policy_verify_synthetic_plate_1_not_sensitive",
        },
        note="not a plate",
        application_hashes={},
    )

    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert result["finalized"] is False
    assert any(
        "Not sensitive policy has not been verified" in blocker for blocker in result["blockers"]
    )


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


def test_workspace_server_creates_anonymous_session_and_late_attribution(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session_response = post_json(f"{base_url}/api/sessions", {})
        session = session_response["session"]

        assert session["reviewer"]["display_name"] is None
        assert session["reviewer"]["reviewer_id"] is None

        attribution_response = post_json(
            f"{base_url}/api/reviewer-attribution",
            {
                "display_name": "Reviewer One",
                "session_id": session["session_id"],
            },
        )
        with urllib.request.urlopen(
            f"{base_url}/api/session-state?session_id={session['session_id']}",
            timeout=5,
        ) as response:
            state_payload = json.loads(response.read().decode("utf-8"))
        session_dir = package / "sessions" / session["session_id"]
        chain = review_sessions.validate_event_hash_chain(session_dir)

        assert attribution_response["event"]["event_type"] == "reviewer_attribution_set"
        assert state_payload["attribution"]["display_name"] == "Reviewer One"
        assert chain["passed"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_workspace_markup_uses_single_player_layout_and_disables_picture_in_picture() -> None:
    html = review_workspace.index_html()
    css = review_workspace.workspace_css()
    js = review_workspace.workspace_js()

    assert "Start review" in html
    assert "reviewerId" not in html
    assert "reviewerRole" not in html
    assert "disablepictureinpicture" in js.lower()
    assert "disablePictureInPicture = true" in js
    assert "requestFullscreen" in js
    assert ".workspace-layout" in css
    assert ".drawer" not in css
    assert "one active video" not in html.lower()
    assert "Play together" not in html + css + js
    assert "Ready" not in html + css + js
    assert "Could not start local review session" in js
    assert "Playback error" in js
    assert "Latest reviewed" in js
    assert "timeline-markers" in js
    assert "function isTextEntryTarget" in js
    assert "'range'" in js
    assert "stepFrame(event.shiftKey ? 10 : 1)" in js
    assert "stepFrame(event.shiftKey ? -10 : -1)" in js
    assert "Final Review Workspace v2.3.1" in html
    assert "final_review_workspace.v2.3.1" in js
    assert "function closestElement" in js
    assert "reviewDelegated" in js
    assert "video.addEventListener('click'" in js
    assert "togglePlay(caseId)" in js
    assert ".decision-correct" in css
    assert ".decision-needs-correction" in css
    assert ".decision-unsure" in css
    assert "Correct', action:ACTIONS.looks" in js
    assert "Needs correction', action:ACTIONS.leak" in js
    assert "Unsure', action:ACTIONS.unsure" in js
    assert 'data-action="filter"' in js
    assert "Saving..." in js
    assert "Could not save:" in js
    assert ".toFixed(3)" not in js
    assert "padStart(2, '0')" in js
    decision_buttons = js[
        js.index("function decisionButtons") : js.index("function bindReviewPanel")
    ]
    assert "Not sensitive" not in decision_buttons


def test_review_entity_registry_generates_thumbnails(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = package / "media" / "day" / "original_reference.mp4"
    source.parent.mkdir(parents=True)
    write_synthetic_video(source, frame_count=3)
    manifest = case_manifests.case_manifest_from_media(
        case_id="day",
        display_name="Day",
        source_frame_start=100,
        source_path="media/day/original_reference.mp4",
        source_video=source,
        video_roles=[
            case_manifests.VideoRole(
                label="Original",
                path="media/day/original_reference.mp4",
                role="original",
                source_frame_end=102,
                source_frame_start=100,
            )
        ],
    )
    case_manifests.write_case_manifest(package / "cases" / "day.json", manifest)
    write_base_region(package, "day")

    review_workspace.write_entity_index(package)
    registry = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    entity = registry["cases"]["day"][0]

    assert entity["label"] == "Plate 01"
    assert entity["entity_uuid"]
    assert entity["representative_source_frame"] == 100
    assert entity["representative_thumbnail_path"]
    assert (package / entity["representative_thumbnail_path"]).exists()


def test_privacy_entity_labels_scale_to_100_without_rendering_forms() -> None:
    entities: list[privacy_entities.PrivacyEntity] = []
    for _ in range(100):
        entities.append(
            privacy_entities.create_entity(
                case_id="synthetic",
                category=CATEGORY_FACE,
                existing=entities,
            )
        )

    assert entities[0].label == "Person 01"
    assert entities[-1].label == "Person 100"
    assert entities[-1].to_dict()["entity_uuid"] == entities[-1].object_uuid


def test_conservative_entity_consolidation_rejects_simultaneous_visibility() -> None:
    first = privacy_entities.create_entity(
        case_id="night",
        category=CATEGORY_FACE,
        provenance={"first_source_frame": 10, "last_source_frame": 30},
    )
    second = privacy_entities.create_entity(
        case_id="night",
        category=CATEGORY_FACE,
        existing=[first],
        provenance={"first_source_frame": 20, "last_source_frame": 40},
    )

    decision = privacy_entities.can_conservatively_merge(first, second)

    assert decision["allowed"] is False
    assert "simultaneously_visible" in decision["reasons"]


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


def test_frozen_model_path_validation_honors_explicit_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adas_xml = tmp_path / "face-detection-adas-0001.xml"
    adas_bin = tmp_path / "face-detection-adas-0001.bin"
    plate_model = tmp_path / "yolo-v9-s-608-license-plates-end2end.onnx"
    adas_xml.write_text("xml", encoding="utf-8")
    adas_bin.write_text("bin", encoding="utf-8")
    plate_model.write_text("onnx", encoding="utf-8")
    monkeypatch.setattr(review_workspace, "EXPECTED_ADAS_XML_SHA256", media_hash(adas_xml))
    monkeypatch.setattr(review_workspace, "EXPECTED_ADAS_BIN_SHA256", media_hash(adas_bin))
    monkeypatch.setattr(
        review_workspace,
        "EXPECTED_OIM_PLATE_ONNX_SHA256",
        media_hash(plate_model),
    )

    paths = review_workspace.resolve_frozen_model_paths(
        adas_xml_path=adas_xml,
        adas_bin_path=adas_bin,
        plate_model_path=plate_model,
    )
    validation = review_workspace.validate_frozen_models(paths)

    assert validation["passed"] is True
    assert validation["files"]["plate_model"]["sha256"] == media_hash(plate_model)


def test_frozen_model_path_validation_honors_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adas_xml = tmp_path / "face-detection-adas-0001.xml"
    adas_bin = tmp_path / "face-detection-adas-0001.bin"
    plate_model = tmp_path / "yolo-v9-s-608-license-plates-end2end.onnx"
    for path, text in ((adas_xml, "xml"), (adas_bin, "bin"), (plate_model, "onnx")):
        path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("REDACTOR_ADAS_XML", adas_xml.as_posix())
    monkeypatch.setenv("REDACTOR_ADAS_BIN", adas_bin.as_posix())
    monkeypatch.setenv("REDACTOR_OIM_PLATE_MODEL", plate_model.as_posix())

    paths = review_workspace.resolve_frozen_model_paths()

    assert paths.adas_xml == adas_xml
    assert paths.adas_bin == adas_bin
    assert paths.plate_model == plate_model


def test_frozen_model_validation_rejects_historical_plate_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adas_xml = tmp_path / "face-detection-adas-0001.xml"
    adas_bin = tmp_path / "face-detection-adas-0001.bin"
    plate_model = tmp_path / "rtdetr" / "yolo-v9-s-608-license-plates-end2end.onnx"
    plate_model.parent.mkdir()
    for path, text in ((adas_xml, "xml"), (adas_bin, "bin"), (plate_model, "onnx")):
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(review_workspace, "EXPECTED_ADAS_XML_SHA256", media_hash(adas_xml))
    monkeypatch.setattr(review_workspace, "EXPECTED_ADAS_BIN_SHA256", media_hash(adas_bin))
    monkeypatch.setattr(
        review_workspace,
        "EXPECTED_OIM_PLATE_ONNX_SHA256",
        media_hash(plate_model),
    )

    try:
        review_workspace.validate_frozen_models(
            review_workspace.FrozenModelPaths(adas_xml, adas_bin, plate_model)
        )
    except ValueError as exc:
        assert "rejected historical model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("historical detector path was accepted")


def test_privacy_entity_creation_policy_merge_split_and_reassign() -> None:
    face = privacy_entities.create_entity(case_id="day", category=CATEGORY_FACE)
    plate_a = privacy_entities.create_entity(case_id="day", category=CATEGORY_PLATE)
    plate_b = privacy_entities.create_entity(
        case_id="day",
        category=CATEGORY_PLATE,
        existing=[plate_a],
        automatic_source_track_ids=("track_b",),
    )

    assert face.label == "Person 01"
    assert plate_a.label == "Plate 01"
    assert plate_b.label == "Plate 02"

    kept = privacy_entities.set_entity_policy(plate_a, privacy_entities.POLICY_KEEP_VISIBLE)
    interval = privacy_entities.set_entity_policy(
        kept,
        privacy_entities.POLICY_REVIEW_REQUIRED,
        frame_start=10,
        frame_end=20,
        reason="occluded",
    )
    locked = privacy_entities.set_entity_lock(interval, True)

    assert locked.default_privacy_policy == privacy_entities.POLICY_KEEP_VISIBLE
    assert locked.interval_policy_overrides[0].policy == privacy_entities.POLICY_REVIEW_REQUIRED
    assert locked.locked is True

    source = privacy_entities.create_entity(
        case_id="day",
        category=CATEGORY_PLATE,
        automatic_source_track_ids=("track_a", "track_c"),
    )
    source_after_split, new_entity = privacy_entities.split_entity(
        entity=source,
        moved_track_ids=("track_c",),
        existing=[source],
    )
    reassigned_source, reassigned_target = privacy_entities.reassign_region_track(
        source=source_after_split,
        target=new_entity,
        track_id="track_a",
    )

    assert "track_c" in new_entity.automatic_source_track_ids
    assert not reassigned_source.automatic_source_track_ids
    assert set(reassigned_target.automatic_source_track_ids) == {"track_a", "track_c"}

    merged, retired = privacy_entities.merge_entities(primary=plate_b, secondary=reassigned_target)
    assert set(merged.automatic_source_track_ids) >= {"track_a", "track_c", "track_b"}
    assert retired.current_status == privacy_entities.STATUS_MERGED


def test_privacy_entities_group_face_regions_by_track_not_region_id() -> None:
    records = [
        PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(1, 1, 10, 10),
            category=CATEGORY_FACE,
            object_id="face_track_6",
            provenance="detector",
            render_eligible=True,
            source_frame=frame,
            source_record={"region_id": f"face_{frame}", "track_id": "6"},
        )
        for frame in (10, 11, 12)
    ]

    entities = privacy_entities.entities_from_region_records("day", records)

    assert len(entities) == 1
    assert entities[0].label == "Person 01"
    assert entities[0].automatic_source_track_ids == ("6",)
    assert (
        entities[0].entity_id
        == privacy_entities.entities_from_region_records("day", records)[0].entity_id
    )
    assert entities[0].object_uuid == entities[0].entity_id.removeprefix("entity_")


def test_privacy_entities_use_first_appearance_order_per_category() -> None:
    records = [
        PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(1, 1, 10, 10),
            category=CATEGORY_PLATE,
            object_id="plate_track_late",
            provenance="detector",
            render_eligible=True,
            source_frame=50,
            source_record={"track_id": "plate_late"},
        ),
        PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(1, 1, 10, 10),
            category=CATEGORY_PLATE,
            object_id="plate_track_early",
            provenance="detector",
            render_eligible=True,
            source_frame=5,
            source_record={"track_id": "plate_early"},
        ),
    ]

    entities = privacy_entities.entities_from_region_records("day", records)

    assert [(entity.label, entity.automatic_source_track_ids) for entity in entities] == [
        ("Plate 01", ("plate_early",)),
        ("Plate 02", ("plate_late",)),
    ]


def test_checkpoint_policy_and_adjustment_revision() -> None:
    checkpoints = review_corrections.default_checkpoints_for_interval(
        source_frame_start=0,
        source_frame_end=90,
        fps=30.0,
    )
    high_motion = review_corrections.required_checkpoint_frames(
        source_frame_start=0,
        source_frame_end=30,
        fps=30.0,
        high_motion=True,
    )
    correction = review_corrections.ReviewCorrection(
        case_id="day",
        category=CATEGORY_PLATE,
        configuration_freeze_hash="hash",
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(
            review_corrections.CorrectionKeyframe(0, (0, 0, 10, 10)),
            review_corrections.CorrectionKeyframe(90, (10, 10, 20, 20)),
        ),
        operation=review_corrections.OP_ADD,
        reason="gap",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=90,
        source_frame_start=0,
        source_video_hash="source",
        task_id="task",
    )

    adjusted = review_corrections.mark_checkpoint(
        correction,
        source_frame=45,
        state=review_corrections.CHECKPOINT_VERIFIED,
        bbox_xyxy=(5, 5, 15, 15),
    )

    assert [checkpoint.source_frame for checkpoint in checkpoints] == [0, 30, 60, 90]
    assert high_motion == (0, 15, 30)
    assert len(adjusted.keyframes) == 3
    assert adjusted.previous_revision_id == correction.revision_id


def test_suggested_keyframe_policy_boundaries_midpoints_and_pages() -> None:
    assert review_corrections.suggested_keyframe_frames(
        source_frame_start=0,
        source_frame_end=9,
        fps=30.0,
    ) == (0, 9)
    assert review_corrections.suggested_keyframe_frames(
        source_frame_start=0,
        source_frame_end=30,
        fps=30.0,
    ) == (0, 15, 30)
    assert review_corrections.suggested_keyframe_frames(
        source_frame_start=0,
        source_frame_end=150,
        fps=30.0,
        evidence_event_frames=(44, 91),
    ) == (0, 30, 44, 60, 90, 91, 120, 150)
    assert review_corrections.review_pages_for_interval(
        source_frame_start=0,
        source_frame_end=359,
        fps=30.0,
    ) == ((0, 149), (150, 299), (300, 359))


def test_interpolation_rejects_not_visible_span() -> None:
    correction = review_corrections.ReviewCorrection(
        case_id="day",
        category=CATEGORY_FACE,
        checkpoints=(
            review_corrections.CorrectionCheckpoint(
                source_frame=5,
                state=review_corrections.CHECKPOINT_NOT_VISIBLE,
            ),
        ),
        configuration_freeze_hash="hash",
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(
            review_corrections.CorrectionKeyframe(0, (0, 0, 10, 10)),
            review_corrections.CorrectionKeyframe(10, (10, 10, 20, 20)),
        ),
        operation=review_corrections.OP_ADD,
        reason="blocked span",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=10,
        source_frame_start=0,
        source_video_hash="source",
        task_id="task",
    )

    try:
        review_corrections.interpolate_correction_boxes(correction)
    except ValueError as exc:
        assert "cannot cross not-visible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("interpolation crossed a not-visible checkpoint")


def test_finalization_blocks_unreviewed_correction_checkpoint(tmp_path: Path) -> None:
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
    correction = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_PLATE,
        checkpoints=(
            review_corrections.CorrectionCheckpoint(
                source_frame=0,
                state=review_corrections.CHECKPOINT_UNREVIEWED,
            ),
        ),
        configuration_freeze_hash=stable_hash({"schema_version": "freeze.test", "value": 1}),
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(review_corrections.CorrectionKeyframe(0, (10, 10, 30, 20)),),
        operation=review_corrections.OP_ADD,
        reason="needs checkpoint",
        reviewer_identity={"reviewer_id": "analyst@example.test"},
        source_frame_end=0,
        source_frame_start=0,
        source_video_hash=manifest.source_video_hash or "",
        task_id="task",
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        correction=correction,
    )

    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert blockers["ready"] is False
    assert any("checkpoint is unreviewed" in blocker for blocker in blockers["blockers"])


def test_finalization_uses_latest_correction_revision(tmp_path: Path) -> None:
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
    original = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_PLATE,
        checkpoints=(
            review_corrections.CorrectionCheckpoint(
                source_frame=0,
                state=review_corrections.CHECKPOINT_UNREVIEWED,
            ),
        ),
        configuration_freeze_hash=stable_hash({"schema_version": "freeze.test", "value": 1}),
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(review_corrections.CorrectionKeyframe(0, (10, 10, 30, 20)),),
        operation=review_corrections.OP_ADD,
        reason="draft",
        reviewer_identity={"reviewer_id": "analyst@example.test"},
        source_frame_end=0,
        source_frame_start=0,
        source_video_hash=manifest.source_video_hash or "",
        task_id="task",
    )
    verified = review_corrections.mark_checkpoint(
        original,
        source_frame=0,
        state=review_corrections.CHECKPOINT_VERIFIED,
        bbox_xyxy=(10, 10, 30, 20),
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        correction=original,
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        correction=verified,
    )

    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert blockers["ready"] is True


def test_finalization_blocks_keep_policy_until_verified(tmp_path: Path) -> None:
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
    review_sessions.append_event(
        session_dir,
        event_type="entity_policy_changed",
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id=None,
        before_state={"default_privacy_policy": "redact"},
        after_state={
            "default_privacy_policy": privacy_entities.POLICY_KEEP_VISIBLE,
            "entity_id": "entity_keep",
            "label": "Person 01",
            "verification_task_id": "object_policy_verify_synthetic_entity_keep",
        },
        note="known non-sensitive officer",
        application_hashes={},
    )

    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )
    assert any("Keep policy has not been verified" in item for item in blockers["blockers"])

    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="object_policy_verify_synthetic_entity_keep",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="verified",
    )
    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )
    assert blockers["ready"] is True


def test_apply_corrections_maps_parent_source_frames_to_case_local_frames() -> None:
    correction = review_corrections.ReviewCorrection(
        case_id="day",
        category=CATEGORY_PLATE,
        checkpoints=(
            review_corrections.CorrectionCheckpoint(
                source_frame=1828,
                state=review_corrections.CHECKPOINT_VERIFIED,
            ),
            review_corrections.CorrectionCheckpoint(
                source_frame=1829,
                state=review_corrections.CHECKPOINT_VERIFIED,
            ),
        ),
        configuration_freeze_hash="hash",
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(
            review_corrections.CorrectionKeyframe(1828, (1, 1, 10, 10)),
            review_corrections.CorrectionKeyframe(1829, (2, 2, 11, 11)),
        ),
        operation=review_corrections.OP_ADD,
        reason="reviewed",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=1829,
        source_frame_start=1828,
        source_video_hash="source",
        task_id="task",
    )

    final_regions, provenance = review_finalization.apply_corrections(
        [],
        [correction],
        case_id="day",
        source_frame_offset=1828,
    )

    assert [region.source_frame for region in final_regions] == [0, 1]
    assert final_regions[0].source_record["source_frame"] == 1828
    assert provenance["final_region_count"] == 2


def test_render_correction_interval_preview(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_base_region(package, manifest.case_id)
    session_dir = create_session(package, manifest)
    correction = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_PLATE,
        checkpoints=(
            review_corrections.CorrectionCheckpoint(
                source_frame=1,
                state=review_corrections.CHECKPOINT_VERIFIED,
            ),
        ),
        configuration_freeze_hash=stable_hash({"schema_version": "freeze.test", "value": 1}),
        interpolation_method=review_corrections.INTERPOLATION_LINEAR,
        keyframes=(
            review_corrections.CorrectionKeyframe(1, (20, 20, 54, 34)),
            review_corrections.CorrectionKeyframe(2, (22, 20, 56, 34)),
        ),
        operation=review_corrections.OP_ADD,
        reason="preview",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=2,
        source_frame_start=1,
        source_video_hash=manifest.source_video_hash or "",
        task_id="task",
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        correction=correction,
    )

    preview = review_workspace.render_correction_interval_preview(
        package,
        session_dir=session_dir,
        correction=correction,
    )

    assert preview["ok"] is True
    assert (package / preview["original_preview"]).exists()
    assert (package / preview["corrected_preview"]).exists()
    assert preview["rendered_frames"] >= 1


def media_hash(path: Path) -> str:
    from redactor import media

    return media.sha256_file(path)
