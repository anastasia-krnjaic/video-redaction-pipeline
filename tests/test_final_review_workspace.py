from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from redactor import (
    case_manifests,
    privacy_entities,
    review_corrections,
    review_finalization,
    review_sessions,
    review_working_state,
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


def post_json_error(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        post_json(url, payload)
    error = exc_info.value
    return error.code, json.loads(error.read().decode("utf-8"))


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


def write_review_entity_shell(package_root: Path, case_id: str) -> None:
    shell = {
        "cases": {case_id: []},
        "schema_version": "review_entities.v2",
        "unresolved_evidence_by_case": {
            case_id: {
                "fragment_count": 0,
                "fragments": [],
                "schema_version": "review_unresolved_evidence.v1",
            }
        },
    }
    write_json(package_root / "review_entities.json", shell)
    write_json(package_root / "review_objects.json", shell)
    write_json(package_root / "privacy_entities.json", shell)
    write_json(
        package_root / "entities" / f"{case_id}_unresolved_evidence.json",
        shell["unresolved_evidence_by_case"][case_id],
    )


def write_explicit_keep_face_policy(
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    *,
    track_id: int,
) -> None:
    policy_root = package_root / "technical" / f"{manifest.case_id}_pipeline" / "face_policy"
    write_json(
        policy_root / "track_census.json",
        {
            "source_video_hash": manifest.source_video_hash,
            "tracks": [
                {
                    "assignment_provenance": "explicit_case_policy",
                    "first_frame": 1,
                    "interpolated_frames": [],
                    "last_frame": 3,
                    "observed_frames": [1, 2, 3],
                    "predicted_frames": [],
                    "quality_status": "auto_confirmed",
                    "representative_frame": 2,
                    "track_id": track_id,
                }
            ],
        },
    )
    write_json(
        policy_root / "privacy_policy.json",
        {
            "assignments": [
                {
                    "decision": "keep",
                    "provenance": "explicit_case_policy",
                    "track_id": track_id,
                }
            ]
        },
    )
    tracks_dir = policy_root / "tracking"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame_index in (1, 2, 3):
        rows.append(
            {
                "frame_index": frame_index,
                "track_boxes": [
                    {
                        "bbox_xyxy": [20 + frame_index, 22, 54 + frame_index, 42],
                        "candidate_id": f"candidate_{track_id}_{frame_index}",
                        "render_eligible": True,
                        "track_id": track_id,
                    }
                ],
            }
        )
    tracks_dir.joinpath("tracks.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


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
    assert merged["normal_task_class"] == "possible_privacy_gap"
    assert merged["issue_type"] == "possible_privacy_gap"
    assert technical["raw_signal_ids"] == ["c"]


def test_long_reviewer_task_is_split_into_traceable_child_chunks(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    manifest = replace(manifest, fps="10/1", frame_count=200, source_frame_end=199)
    case_manifests.write_case_manifest(package / "cases" / "synthetic.json", manifest)
    task = {
        "blocking": True,
        "case_id": manifest.case_id,
        "category": CATEGORY_PLATE,
        "issue_type": "possible_privacy_gap",
        "normal_task_class": "possible_privacy_gap",
        "raw_signal_ids": ["raw_long"],
        "review_task_id": "synthetic_task_long",
        "source_frame_end": 60,
        "source_frame_start": 0,
        "technical_only": False,
        "title": "Possible plate privacy gap",
        "what_to_inspect": "Inspect the interval.",
    }

    children, audit = review_workspace.chunk_reviewer_tasks_for_package(package, [task])

    assert audit["passed"] is True
    assert len(children) == 3
    assert [child["child_index"] for child in children] == [1, 2, 3]
    assert children[0]["title"] == "Possible plate privacy gap - part 1 of 3"
    assert all(child["parent_review_task_id"] == "synthetic_task_long" for child in children)
    assert all(child["parent_raw_signal_ids"] == ["raw_long"] for child in children)
    assert all(
        (child["source_frame_end"] - child["source_frame_start"] + 1) / 10.0 <= 5.0
        for child in children
    )
    assert [(child["source_frame_start"], child["source_frame_end"]) for child in children] == [
        (0, 29),
        (30, 59),
        (60, 60),
    ]


def test_detector_span_audit_requires_every_canonical_day_frame(tmp_path: Path) -> None:
    detections = tmp_path / "detections.jsonl"
    rows = [
        {"frame_index": 0, "detections": []},
        {"frame_index": 2, "detections": []},
    ]
    detections.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    audit = review_workspace.detector_span_audit(
        detections,
        source_frame_count=3,
        parent_source_start=1828,
    )

    assert audit["full_source_processed"] is False
    assert audit["detector_frame_available_for_every_canonical_day_frame"] is False
    assert audit["missing_source_frame_ranges"] == [
        {"frame_start": 1829, "frame_end": 1829, "length": 1}
    ]


def test_normal_review_task_classes_are_actionable() -> None:
    roots = [
        review_workspace.root_cause_for_signal(
            {"issue_type": "visible_plate_leak", "severity": "blocker", "category": "plate"}
        ),
        review_workspace.root_cause_for_signal(
            {"issue_type": "uncertain", "severity": "low", "category": "face"}
        ),
        review_workspace.root_cause_for_signal(
            {"issue_type": "false_plate_redaction", "severity": "medium", "category": "plate"}
        ),
    ]

    assert roots == ["possible_privacy_gap", "spot_check", "possible_false_redaction"]


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


@pytest.mark.parametrize(
    (
        "name",
        "start",
        "end",
        "method",
        "keyframes",
        "expected_frames",
        "sample_frame",
        "sample_box",
    ),
    [
        (
            "single",
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            ((0, (10, 10, 70, 50)),),
            [0],
            0,
            (10, 10, 70, 50),
        ),
        (
            "fixed",
            1,
            3,
            review_corrections.INTERPOLATION_LINEAR,
            ((1, (10, 10, 70, 50)),),
            [1, 2, 3],
            2,
            (10, 10, 70, 50),
        ),
        (
            "interpolate",
            1,
            3,
            review_corrections.INTERPOLATION_LINEAR,
            ((1, (10, 10, 60, 42)), (3, (26, 16, 78, 50))),
            [1, 2, 3],
            2,
            (18, 13, 69, 46),
        ),
    ],
)
def test_saved_manual_correction_materializes_latest_reviewed_regions_and_pixels(
    tmp_path: Path,
    name: str,
    start: int,
    end: int,
    method: str,
    keyframes: tuple[tuple[int, tuple[int, int, int, int]], ...],
    expected_frames: list[int],
    sample_frame: int,
    sample_box: tuple[int, int, int, int],
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    automatic = package / "media" / "synthetic" / "automatic_baseline.mp4"
    shutil.copy2(package / manifest.video_roles[0].path, automatic)
    roles = (
        manifest.video_roles[0],
        case_manifests.VideoRole(
            label="Automatic",
            path=automatic.relative_to(package).as_posix(),
            role="automatic",
            sha256=review_workspace.media.sha256_file(automatic),
            source_frame_end=manifest.source_frame_end,
            source_frame_start=manifest.source_frame_start,
        ),
    )
    manifest = replace(manifest, video_roles=roles)
    case_manifests.write_case_manifest(package / "cases" / "synthetic.json", manifest)
    write_empty_regions(package, manifest.case_id)
    session_dir = create_session(package, manifest)
    correction = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_PLATE,
        configuration_freeze_hash=stable_hash({"schema_version": "freeze.test", "value": 1}),
        interpolation_method=method,
        keyframes=tuple(
            review_corrections.CorrectionKeyframe(frame, box) for frame, box in keyframes
        ),
        operation=review_corrections.OP_ADD,
        reason=f"{name} correction",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=end,
        source_frame_start=start,
        source_video_hash=manifest.source_video_hash or "",
        task_id=f"task_{name}",
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        correction=correction,
    )

    final_regions, provenance = review_working_state.compute_effective_privacy_regions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    correction_regions = [
        record for record in final_regions if record.provenance == "reviewed_correction"
    ]

    assert [
        record.source_record["source_frame"] for record in correction_regions
    ] == expected_frames
    assert provenance["manual_correction_count"] == 1

    materialization = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        actor={"reviewer_id": "analyst"},
        accepted_revision=review_working_state.derive_working_state(package, session_dir)[
            "active_projection_hash"
        ],
    )
    latest_path = package / materialization["latest_reviewed"]["path"]
    automatic_frame = read_video_frame(automatic, sample_frame)
    latest_frame = read_video_frame(latest_path, sample_frame)
    x1, y1, x2, y2 = sample_box
    inside_diff = float(
        np.mean(
            np.abs(
                latest_frame[y1:y2, x1:x2].astype(np.int16)
                - automatic_frame[y1:y2, x1:x2].astype(np.int16)
            )
        )
    )
    control_diff = float(
        np.mean(
            np.abs(
                latest_frame[0:8, 0:8].astype(np.int16) - automatic_frame[0:8, 0:8].astype(np.int16)
            )
        )
    )
    audit = json.loads(
        (package / "manual_correction_persistence_audit.json").read_text(encoding="utf-8")
    )

    assert materialization["manual_correction_count"] == 1
    assert video_frame_count(latest_path) == manifest.frame_count
    assert inside_diff > max(5.0, control_diff * 1.5)
    assert audit["events"][-1]["correction_ids"] == [correction.correction_id]


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


def write_policy_entity_package(
    package: Path,
) -> tuple[case_manifests.CaseManifest, dict[str, object]]:
    manifest = synthetic_case_manifest(package)
    automatic = package / "media" / manifest.case_id / "automatic_baseline.mp4"
    shutil.copy2(package / manifest.video_roles[0].path, automatic)
    manifest = replace(
        manifest,
        video_roles=(
            manifest.video_roles[0],
            case_manifests.VideoRole(
                label="Automatic",
                path=automatic.relative_to(package).as_posix(),
                role="automatic",
                sha256=review_workspace.media.sha256_file(automatic),
                source_frame_end=manifest.source_frame_end,
                source_frame_start=manifest.source_frame_start,
            ),
        ),
    )
    case_manifests.write_case_manifest(package / "cases" / f"{manifest.case_id}.json", manifest)
    write_base_region(package, manifest.case_id)
    entity = {
        "automatic_source_track_ids": ["plate_1"],
        "case_id": manifest.case_id,
        "category": CATEGORY_PLATE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "default_privacy_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_plate_1",
        "first_visible_source_frame": 0,
        "label": "Plate 01",
        "last_visible_source_frame": 0,
        "member_detector_tracker_ids": ["plate_1"],
        "privacy_action": privacy_entities.POLICY_REDACT,
        "representative_source_frame": 0,
    }
    write_json(
        package / "review_entities.json",
        {"cases": {manifest.case_id: [entity]}, "schema_version": "review_entities.v2"},
    )
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_json(package / "workspace_manifest.json", {"case_count": 1})
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    return manifest, entity


def write_empty_regions(package: Path, case_id: str) -> None:
    review_finalization.write_privacy_region_records(
        package / "regions" / f"{case_id}_current_regions.jsonl",
        [],
    )


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv.VideoCapture(str(path))
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise AssertionError(f"could not read frame {frame_index} from {path}")
        return frame
    finally:
        capture.release()


def video_frame_count(path: Path) -> int:
    capture = cv.VideoCapture(str(path))
    try:
        return int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def wait_for_materialization_job(
    server: review_workspace.FinalReviewWorkspaceServer,
    job_id: str,
) -> dict[str, object]:
    deadline = review_workspace.time.monotonic() + 10
    while review_workspace.time.monotonic() < deadline:
        job = server.materialization_jobs.get(job_id)
        if job and job.get("status") in {"completed", "failed", "superseded"}:
            return job
        review_workspace.time.sleep(0.05)
    raise AssertionError(f"materialization job did not finish: {job_id}")


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

        suffix_request = urllib.request.Request(
            f"{base_url}/{manifest.video_roles[0].path}",
            headers={"Range": "bytes=-8"},
        )
        with urllib.request.urlopen(suffix_request, timeout=5) as response:
            suffix_body = response.read()
            suffix_range = response.headers["Content-Range"]
        assert response.status == 206
        assert len(suffix_body) == 8
        assert suffix_range.startswith("bytes ")

        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url}/{manifest.video_roles[0].path}",
                    headers={"Range": "bytes=999999999-1000000000"},
                ),
                timeout=5,
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 416
            assert exc.headers["Content-Range"].startswith("bytes */")
        else:  # pragma: no cover
            raise AssertionError("out-of-bounds range request unexpectedly succeeded")

        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url}/{manifest.video_roles[0].path}",
                    headers={"Range": "bytes=0-1,3-4"},
                ),
                timeout=5,
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 416
        else:  # pragma: no cover
            raise AssertionError("multi-range request unexpectedly succeeded")

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


def test_workspace_server_exposes_versioned_api_contract(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    write_json(
        package / "workspace_manifest.json",
        {
            "case_count": 1,
            "cases": {"synthetic": "cases/synthetic.json"},
            "schema_version": review_workspace.SCHEMA_VERSION,
        },
    )
    write_json(package / "reviewer_tasks.json", {"actions": [], "task_count": 0, "tasks": []})
    write_json(package / "review_entities.json", {"cases": {"synthetic": []}})
    write_json(package / "raw_audit_signals.json", {"raw_signals": []})
    write_json(package / "version_alignment_audit.json", {"passed": True})
    write_json(package / "timeline_marker_audit.json", {"passed": True})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/v1/schema", timeout=5) as response:
            schema = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base_url}/api/v1/workspace/summary", timeout=5) as response:
            summary = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(
            f"{base_url}/api/v1/cases/{manifest.case_id}", timeout=5
        ) as response:
            case_payload = json.loads(response.read().decode("utf-8"))

        assert schema["x-api-schema-version"] == "final_review_workspace_api.v1"
        assert summary["api_schema_version"] == "final_review_workspace_api.v1"
        assert summary["cases"]["synthetic"]["case_id"] == "synthetic"
        assert case_payload["source_frame_start"] == manifest.source_frame_start
    finally:
        server.shutdown()
        server.server_close()


def test_noop_entity_policy_skips_materialization_job(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    session_id = session_dir.name
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = post_json(
            f"{base_url}/api/v1/object-policy",
            {
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "entity_id": entity["entity_id"],
                "policy": privacy_entities.POLICY_REDACT,
                "session_id": session_id,
            },
        )
        audit = json.loads((package / "materialization_job_audit.json").read_text())

        assert response["ok"] is True
        assert response["materialization_required"] is False
        assert response["materialization_job"] is None
        assert response["materialization_plan"]["changed_source_frame_count"] == 0
        assert response["materialization_plan"]["materialization_strategy"] == "no_op"
        assert (
            response["materialization_plan"]["before_effective_regions_hash"]
            == response["materialization_plan"]["after_effective_regions_hash"]
        )
        assert audit["cases"][manifest.case_id]["materialization_jobs_skipped"] == 1
        assert audit["cases"][manifest.case_id]["materialization_jobs_started"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_media_changing_policy_queues_one_target_hashed_job(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = post_json(
            f"{base_url}/api/v1/object-policy",
            {
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "entity_id": entity["entity_id"],
                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                "session_id": session_dir.name,
            },
        )
        job = response["materialization_job"]

        assert response["ok"] is True
        assert response["materialization_required"] is True
        assert job["status"] == "queued"
        assert job["changed_source_frame_count"] == 1
        assert job["current_stage"] == "waiting"
        assert job["materialization_plan"]["materialization_strategy"] == "full_render_fallback"
        assert job["planned_frames_to_render"] == manifest.frame_count
        assert (
            job["target_effective_regions_hash"]
            == response["materialization_plan"]["after_effective_regions_hash"]
        )
        assert job["target_effective_region_count"] == 0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("name", "start", "end", "interpolation", "keyframes", "expected_intervals"),
    [
        (
            "single",
            2,
            2,
            review_corrections.INTERPOLATION_NONE,
            [review_corrections.CorrectionKeyframe(2, (10, 10, 20, 20))],
            [{"source_frame_start": 2, "source_frame_end": 2}],
        ),
        (
            "fixed",
            1,
            3,
            review_corrections.INTERPOLATION_NONE,
            [review_corrections.CorrectionKeyframe(1, (10, 10, 20, 20))],
            [{"source_frame_start": 1, "source_frame_end": 3}],
        ),
        (
            "interpolated",
            1,
            3,
            review_corrections.INTERPOLATION_LINEAR,
            [
                review_corrections.CorrectionKeyframe(1, (10, 10, 20, 20)),
                review_corrections.CorrectionKeyframe(3, (14, 12, 24, 22)),
            ],
            [{"source_frame_start": 1, "source_frame_end": 3}],
        ),
    ],
)
def test_effective_region_change_summary_tracks_manual_correction_frames(
    tmp_path: Path,
    name: str,
    start: int,
    end: int,
    interpolation: str,
    keyframes: list[review_corrections.CorrectionKeyframe],
    expected_intervals: list[dict[str, int]],
) -> None:
    package = tmp_path / f"package_{name}"
    manifest = synthetic_case_manifest(package)
    write_empty_regions(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    session_dir = create_session(package, manifest)
    before = review_working_state.effective_privacy_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    correction = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_PLATE,
        correction_id=f"corr_{name}",
        interpolation_method=interpolation,
        keyframes=tuple(keyframes),
        operation=review_corrections.OP_ADD,
        reason=f"{name} correction",
        reviewer_identity={"reviewer_id": "analyst"},
        configuration_freeze_hash="test_freeze",
        source_frame_end=end,
        source_frame_start=start,
        source_video_hash=manifest.source_video_hash or "",
        task_id=f"task_{name}",
    )

    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        correction=correction,
    )
    after = review_working_state.effective_privacy_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    summary = review_working_state.effective_region_change_summary(
        package,
        case_id=manifest.case_id,
        before_digest=before,
        after_digest=after,
    )

    assert summary["changed_source_frame_intervals"] == expected_intervals
    assert summary["changed_source_frame_count"] == end - start + 1
    assert summary["changed_source_frame_percentage"] == pytest.approx(
        (end - start + 1) / manifest.frame_count
    )


def test_materialization_plan_reports_exact_changed_frames_for_policy_removal(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = post_json(
            f"{base_url}/api/v1/object-policy",
            {
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "entity_id": entity["entity_id"],
                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                "session_id": session_dir.name,
            },
        )
        plan = response["materialization_plan"]

        assert plan["changed_source_frame_count"] == 1
        assert plan["changed_source_frame_intervals"] == [
            {"source_frame_start": 0, "source_frame_end": 0}
        ]
        assert plan["affected_source_frame_intervals"] == plan["changed_source_frame_intervals"]
        assert plan["materialization_strategy"] == "full_render_fallback"
        assert plan["planned_frames_to_render"] == manifest.frame_count
        assert plan["planned_frames_to_encode"] == manifest.frame_count
    finally:
        server.shutdown()
        server.server_close()


def test_repeated_same_policy_reuses_existing_pending_materialization_job(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = {
        "actor": {"reviewer_id": "analyst"},
        "case_id": manifest.case_id,
        "entity_id": entity["entity_id"],
        "policy": privacy_entities.POLICY_KEEP_VISIBLE,
        "session_id": session_dir.name,
    }
    try:
        first = post_json(f"{base_url}/api/v1/object-policy", payload)
        second = post_json(f"{base_url}/api/v1/object-policy", payload)
        jobs = server.materialization_jobs.jobs

        assert first["materialization_required"] is True
        assert second["materialization_required"] is True
        assert first["materialization_job"]["job_id"] == second["materialization_job"]["job_id"]
        assert (
            second["materialization_plan"]["reused_job_id"]
            == first["materialization_job"]["job_id"]
        )
        assert len(jobs) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_review_decision_does_not_start_materialization_job(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, _entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = post_json(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "note": "correct",
                "session_id": session_dir.name,
                "task_id": "task_1",
            },
        )

        assert response["ok"] is True
        assert server.materialization_jobs.jobs == {}
    finally:
        server.shutdown()
        server.server_close()


def test_region_compatible_job_publishes_after_non_media_decision(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 0.01
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        policy_response = post_json(
            f"{base_url}/api/v1/object-policy",
            {
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "entity_id": entity["entity_id"],
                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                "session_id": session_dir.name,
            },
        )
        decision_response = post_json(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "note": "correct",
                "session_id": session_dir.name,
                "task_id": "task_1",
            },
        )
        job = wait_for_materialization_job(
            server,
            str(policy_response["materialization_job"]["job_id"]),
        )
        working_state = review_working_state.derive_working_state(package, session_dir)

        assert decision_response["ok"] is True
        assert job["status"] == "completed"
        assert job["published_working_state_revision"] == working_state["active_projection_hash"]
        assert manifest.case_id in working_state["materialized_latest_reviewed"]
    finally:
        server.shutdown()
        server.server_close()


def test_react_workspace_assets_copy_app_and_legacy_fallback(tmp_path: Path) -> None:
    package = tmp_path / "package"
    frontend = tmp_path / "frontend"
    (frontend / "dist" / "assets").mkdir(parents=True)
    (frontend / "dist" / "index.html").write_text(
        '<script type="module" src="/app/assets/index.js"></script>',
        encoding="utf-8",
    )
    (frontend / "dist" / "assets" / "index.js").write_text("console.log('ok')", encoding="utf-8")
    (frontend / "package-lock.json").write_text('{"lockfileVersion":3}\n', encoding="utf-8")
    (frontend / "package.json").write_text('{"name":"test"}\n', encoding="utf-8")

    review_workspace.write_react_workspace_assets(
        package,
        build_frontend=False,
        frontend_root=frontend,
    )

    assert (package / "app" / "index.html").exists()
    assert (package / "app" / "assets" / "index.js").exists()
    assert "legacy.html" in (package / "index.html").read_text(encoding="utf-8")
    assert "Final Review Workspace v2.3.4" in (package / "legacy.html").read_text(encoding="utf-8")
    manifest = json.loads((package / "frontend_build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_schema_version"] == "final_review_workspace_api.v1"
    assert manifest["selected_frontend"] == "react"


def test_backend_parity_audit_detects_hash_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for root in (reference, candidate):
        write_json(root / "cases" / "day.json", {"case_id": "day", "value": 1})
        write_json(root / "reviewer_tasks.json", {"tasks": []})
        (root / "regions").mkdir(parents=True, exist_ok=True)
        (root / "regions" / "day_current_regions.jsonl").write_text("same\n", encoding="utf-8")

    passed = review_workspace.backend_parity_audit(reference, candidate)
    (candidate / "regions" / "day_current_regions.jsonl").write_text("changed\n", encoding="utf-8")
    failed = review_workspace.backend_parity_audit(reference, candidate)

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert any("day_current_regions" in error for error in failed["errors"])


def test_backend_parity_audit_allows_source_backed_keep_entity_delta(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for root in (reference, candidate):
        write_json(root / "cases" / "night.json", {"case_id": "night", "value": 1})
        write_json(root / "reviewer_tasks.json", {"tasks": []})
        write_json(root / "entities" / "night.json", {"entities": []})
        write_json(root / "review_entities.json", {"cases": {"night": []}})
        write_json(root / "privacy_entities.json", {"cases": {"night": []}})
        (root / "regions").mkdir(parents=True, exist_ok=True)
        (root / "regions" / "night_current_regions.jsonl").write_text("same\n", encoding="utf-8")
    keep_entity = {"entity_id": "entity_keep", "automatic_source_track_ids": ["10"]}
    write_json(candidate / "entities" / "night.json", {"entities": [keep_entity]})
    write_json(candidate / "review_entities.json", {"cases": {"night": [keep_entity]}})
    write_json(candidate / "privacy_entities.json", {"cases": {"night": [keep_entity]}})
    write_json(
        candidate / "review_entity_coverage_audit.json",
        {
            "cases": {
                "night": {
                    "explicit_keep_tracks_added": [
                        {
                            "assignment_provenance": "explicit_case_policy",
                            "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                            "review_entity_uuid": "entity_keep",
                            "track_id": "10",
                            "valid_source_evidence": True,
                        }
                    ]
                }
            },
            "passed": True,
            "schema_version": "review_entity_coverage_audit.v1",
        },
    )

    audit = review_workspace.backend_parity_audit(reference, candidate)

    assert audit["passed"] is True
    assert audit["entity_registry_coverage_delta_count"] == 3
    assert audit["privacy_regions_preserved"] is True


def test_entity_merge_payload_allows_manual_overlap_but_rejects_invalid_targets() -> None:
    source = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "entity_id": "source",
        "first_visible_source_frame": 10,
        "last_visible_source_frame": 20,
    }
    overlap = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "entity_id": "overlap",
        "first_visible_source_frame": 15,
        "last_visible_source_frame": 30,
    }
    safe = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "entity_id": "safe",
        "first_visible_source_frame": 21,
        "last_visible_source_frame": 30,
    }

    assert (
        review_workspace.validate_entity_merge_payload(
            {
                "before_state": {"source": source, "target": overlap},
                "after_state": {"primary": source, "retired_entity_ids": ["overlap"]},
            }
        )
        is None
    )
    assert "retired target" in (
        review_workspace.validate_entity_merge_payload(
            {
                "before_state": {"source": source, "target": safe},
                "after_state": {"primary": source, "retired_entity_ids": []},
            }
        )
        or ""
    )
    assert (
        review_workspace.validate_entity_merge_payload(
            {
                "before_state": {"source": source, "target": safe},
                "after_state": {"primary": source, "retired_entity_ids": ["safe"]},
            }
        )
        is None
    )


def test_entity_merge_event_creates_canonical_active_entity(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    entity_a = {
        "automatic_source_track_ids": ["track_a"],
        "case_id": manifest.case_id,
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_a",
        "evidence_summary": {"region_count": 2, "review_observation_count": 2},
        "first_visible_source_frame": 1,
        "label": "Person 01",
        "last_visible_source_frame": 2,
        "member_detector_tracker_ids": ["track_a"],
        "representative_quality": {"bbox_xyxy": [1, 2, 10, 12], "relative_area": 0.1},
        "representative_source_frame": 1,
    }
    entity_b = {
        "automatic_source_track_ids": ["track_b"],
        "case_id": manifest.case_id,
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_b",
        "evidence_summary": {"region_count": 3, "review_observation_count": 3},
        "first_visible_source_frame": 2,
        "label": "Person 02",
        "last_visible_source_frame": 4,
        "member_detector_tracker_ids": ["track_b"],
        "representative_quality": {"bbox_xyxy": [3, 4, 20, 22], "relative_area": 0.2},
        "representative_source_frame": 2,
    }
    registry = {
        "cases": {manifest.case_id: [entity_a, entity_b]},
        "schema_version": "review_entities.v2",
    }
    write_json(package / "review_entities.json", registry)
    session_dir = create_session(package, manifest)

    merged = review_workspace.canonical_merged_entity(
        [entity_a, entity_b],
        session_dir=session_dir,
    )
    review_sessions.append_event(
        session_dir,
        event_type="entities_merged",
        actor={"display_name": "Analyst"},
        case_id=manifest.case_id,
        task_id=None,
        before_state={"entities": [entity_a, entity_b]},
        after_state={
            "primary": merged,
            "retired_entity_ids": ["entity_a", "entity_b"],
            "retired_entity_uuids": ["entity_a", "entity_b"],
        },
        note="Reviewer confirmed object merge.",
        application_hashes={},
    )

    state = review_working_state.derive_working_state(package, session_dir)
    active = review_working_state.effective_case_entities(
        registry["cases"][manifest.case_id], state
    )

    assert [entity["entity_id"] for entity in active] == [merged["entity_id"]]
    assert active[0]["automatic_source_track_ids"] == ["track_a", "track_b"]
    assert active[0]["member_detector_tracker_ids"] == ["track_a", "track_b"]
    assert active[0]["representative_bbox"] == [3, 4, 20, 22]
    assert active[0]["label"] == "Merged Person 01"
    assert active[0]["current_policy"] == privacy_entities.POLICY_REDACT
    assert (
        active[0]["merge_policy_initialization"]["reason"] == "all_selected_entities_shared_policy"
    )

    entity_c = entity_a | {
        "automatic_source_track_ids": ["track_c"],
        "entity_id": "entity_c",
        "label": "Person 03",
    }
    entity_d = entity_b | {
        "automatic_source_track_ids": ["track_d"],
        "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
        "entity_id": "entity_d",
        "label": "Person 04",
    }
    second = review_workspace.canonical_merged_entity(
        [entity_c, entity_d],
        session_dir=session_dir,
    )
    assert second["label"] == "Merged Person 02"
    assert second["current_policy"] == privacy_entities.POLICY_REVIEW_REQUIRED
    assert second["merge_policy_initialization"]["reason"] == "conflicting_selected_policies"

    extended = review_workspace.canonical_merged_entity(
        [active[0], entity_c],
        session_dir=session_dir,
    )
    assert extended["label"] == "Merged Person 01"

    wrong_case_active = review_working_state.effective_case_entities(
        [
            {
                **entity_a,
                "case_id": "other_case",
                "entity_id": "other_entity_a",
            }
        ],
        state,
    )
    assert [entity["entity_id"] for entity in wrong_case_active] == ["other_entity_a"]


def test_case_restart_resets_only_selected_active_projection(tmp_path: Path) -> None:
    package = tmp_path / "package"
    day_manifest = synthetic_case_manifest(package)
    day_manifest = replace(day_manifest, case_id="day", display_name="Day")
    case_manifests.write_case_manifest(package / "cases" / "day.json", day_manifest)
    (package / "cases" / "synthetic.json").unlink()
    night_source = package / "media" / "night" / "original_reference.mp4"
    night_source.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_video(night_source)
    night_manifest = case_manifests.case_manifest_from_media(
        case_id="night",
        display_name="Night",
        source_path="media/night/original_reference.mp4",
        source_video=night_source,
        video_roles=[
            case_manifests.VideoRole(
                label="Original",
                path="media/night/original_reference.mp4",
                role="original",
                sha256=None,
                source_frame_end=4,
                source_frame_start=0,
            ),
            case_manifests.VideoRole(
                label="Automatic",
                path="media/night/original_reference.mp4",
                role="automatic",
                sha256=None,
                source_frame_end=4,
                source_frame_start=0,
            ),
        ],
    )
    case_manifests.write_case_manifest(package / "cases" / "night.json", night_manifest)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "blocking": True,
                    "case_id": "day",
                    "review_task_id": "day_task",
                    "technical_only": False,
                },
                {
                    "blocking": True,
                    "case_id": "night",
                    "review_task_id": "night_task",
                    "technical_only": False,
                },
            ]
        },
    )
    day_a = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "day_a",
        "label": "Person 01",
        "representative_source_frame": 0,
    }
    day_b = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "day_b",
        "label": "Person 02",
        "representative_source_frame": 1,
    }
    night_entity = {
        "case_id": "night",
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
        "entity_id": "night_keep",
        "label": "Person 01",
        "representative_source_frame": 0,
    }
    registry = {"cases": {"day": [day_a, day_b], "night": [night_entity]}}
    write_json(package / "review_entities.json", registry)
    (package / "regions").mkdir(parents=True, exist_ok=True)
    (package / "regions" / "day_current_regions.jsonl").write_text("", encoding="utf-8")
    review_working_state.write_reset_baseline_state_audit_shell(package)
    day_baseline_summary = review_working_state.baseline_case_active_state_summary(package, "day")
    night_baseline_summary = review_working_state.baseline_case_active_state_summary(
        package, "night"
    )
    freeze = {"schema_version": "freeze.test", "value": 1}
    write_json(package / "configuration_freeze.json", freeze)
    session = review_sessions.create_review_session(
        package_root=package,
        reviewer=review_sessions.ReviewerIdentity("Analyst", "analyst@example.test"),
        source_revision="test",
        configuration_freeze_hash=stable_hash(freeze),
        case_manifest_hashes={
            "day": day_manifest.manifest_hash(),
            "night": night_manifest.manifest_hash(),
        },
        source_hashes={
            "day": day_manifest.source_video_hash or "",
            "night": night_manifest.source_video_hash or "",
        },
    )
    session_dir = review_sessions.session_directory(package, session.session_id)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id="day_task",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="correct",
    )
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="night",
        task_id="night_task",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="correct",
    )
    review_sessions.append_sweep_acknowledgement(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
        viewed_ranges=[{"start": 0, "end": 4}],
    )
    review_sessions.append_sweep_acknowledgement(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="night",
        viewed_ranges=[{"start": 0, "end": 4}],
    )
    correction = review_corrections.ReviewCorrection(
        case_id="day",
        category=CATEGORY_FACE,
        configuration_freeze_hash=stable_hash(freeze),
        interpolation_method=review_corrections.INTERPOLATION_NONE,
        keyframes=(review_corrections.CorrectionKeyframe(0, (1, 2, 10, 12)),),
        operation=review_corrections.OP_ADD,
        reason="reviewed",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=0,
        source_frame_start=0,
        source_video_hash=day_manifest.source_video_hash or "",
        task_id="day_task",
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        correction=correction,
    )
    review_sessions.append_event(
        session_dir,
        event_type="manual_issue_created",
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id="manual_day",
        before_state=None,
        after_state={"task": {"case_id": "day", "review_task_id": "manual_day"}},
        note="manual",
        application_hashes={},
    )
    merged = review_workspace.canonical_merged_entity([day_a, day_b], session_dir=session_dir)
    review_sessions.append_event(
        session_dir,
        event_type="entities_merged",
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id=None,
        before_state={"entities": [day_a, day_b]},
        after_state={"primary": merged, "retired_entity_ids": ["day_a", "day_b"]},
        note="merge",
        application_hashes={},
    )
    review_sessions.append_event(
        session_dir,
        event_type="entity_policy_changed",
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id=None,
        before_state=merged,
        after_state=merged | {"current_policy": privacy_entities.POLICY_KEEP_VISIBLE},
        note="keep",
        application_hashes={},
    )
    checkpoint, _event = review_working_state.create_checkpoint(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        name="Before restart",
    )
    mutated = review_working_state.derive_working_state(package, session_dir)
    day_mutated_summary = review_working_state.normalized_case_active_state_summary(
        package,
        mutated,
        "day",
    )
    night_mutated_summary = review_working_state.normalized_case_active_state_summary(
        package,
        mutated,
        "night",
    )

    assert (
        day_mutated_summary["normalized_state_hash"]
        != day_baseline_summary["normalized_state_hash"]
    )
    assert day_mutated_summary["merged_entity_ids"] == [merged["entity_id"]]
    assert day_mutated_summary["entity_count"] == 1
    assert (
        night_mutated_summary["normalized_state_hash"]
        != night_baseline_summary["normalized_state_hash"]
    )

    restart_event, restarted = review_working_state.restart_review_workflow(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
    )
    restarted_day_active = review_working_state.effective_case_entities(
        registry["cases"]["day"], restarted
    )
    restarted_night_active = review_working_state.effective_case_entities(
        registry["cases"]["night"], restarted
    )

    assert restart_event["event_type"] == "review_workflow_restarted"
    assert "day_task" not in restarted["task_decisions"]
    assert (
        restarted["task_decisions"]["night_task"]["action"] == review_sessions.ACTION_LOOKS_CORRECT
    )
    assert "day" not in restarted["sweep_acknowledgements"]
    assert "night" in restarted["sweep_acknowledgements"]
    assert restarted["active_manual_tasks"] == []
    assert restarted["active_correction_ids"] == [correction.correction_id]
    assert [entity["entity_id"] for entity in restarted_day_active] == [merged["entity_id"]]
    assert restarted_day_active[0]["current_policy"] == privacy_entities.POLICY_KEEP_VISIBLE
    assert restarted_night_active[0]["entity_id"] == "night_keep"
    assert restarted_night_active[0]["current_policy"] == privacy_entities.POLICY_KEEP_VISIBLE
    assert checkpoint["checkpoint_uuid"] in {
        item["checkpoint_uuid"] for item in restarted["checkpoints"]
    }
    pre_reset_day_summary = review_working_state.normalized_case_active_state_summary(
        package,
        restarted,
        "day",
    )

    reset_event, reset = review_working_state.reset_to_automatic_baseline(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
    )
    day_active = review_working_state.effective_case_entities(registry["cases"]["day"], reset)
    night_active = review_working_state.effective_case_entities(registry["cases"]["night"], reset)

    assert reset_event["event_type"] == "case_edits_reset_to_automatic_baseline"
    assert (
        reset_event["after_state"]["baseline_state_hashes"]["day"]
        == day_baseline_summary["normalized_state_hash"]
    )
    assert (
        reset_event["after_state"]["mutated_state_hashes"]["day"]
        == pre_reset_day_summary["normalized_state_hash"]
    )
    assert "day_task" not in reset["task_decisions"]
    assert reset["task_decisions"]["night_task"]["action"] == review_sessions.ACTION_LOOKS_CORRECT
    assert "day" not in reset["sweep_acknowledgements"]
    assert "night" in reset["sweep_acknowledgements"]
    assert reset["active_manual_tasks"] == []
    assert reset["active_correction_ids"] == []
    assert [entity["entity_id"] for entity in day_active] == ["day_a", "day_b"]
    assert night_active[0]["entity_id"] == "night_keep"
    assert night_active[0]["current_policy"] == privacy_entities.POLICY_KEEP_VISIBLE
    assert (
        review_working_state.normalized_case_active_state_summary(
            package,
            reset,
            "day",
        )["normalized_state_hash"]
        == day_baseline_summary["normalized_state_hash"]
    )
    assert (
        review_working_state.normalized_case_active_state_summary(
            package,
            reset,
            "night",
        )["normalized_state_hash"]
        == night_mutated_summary["normalized_state_hash"]
    )
    automatic_regions = review_finalization.load_privacy_region_records(
        package / "regions" / "day_current_regions.jsonl"
    )
    effective_regions, provenance = review_working_state.compute_effective_privacy_regions(
        package,
        session_dir,
        case_id="day",
    )
    assert stable_hash([record.to_dict() for record in effective_regions]) == stable_hash(
        [record.to_dict() for record in automatic_regions]
    )
    assert provenance["privacy_region_changes_by_cause"]["suppression_removed"] == 0
    stale_latest = {
        "label": "Latest reviewed",
        "path": "sessions/stale/media/day/stale.mp4",
        "role": "latest_reviewed",
        "source_frame_end": 4,
        "source_frame_start": 0,
    }
    review_sessions.append_event(
        session_dir,
        event_type="reviewed_media_materialization_completed",
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id=None,
        before_state=None,
        after_state={
            "active_projection_hash": mutated["active_projection_hash"],
            "case_id": "day",
            "latest_reviewed": stale_latest,
        },
        note="stale materialization",
        application_hashes={},
    )
    stale_guarded = review_working_state.derive_working_state(package, session_dir)
    assert "day" not in stale_guarded["materialized_latest_reviewed"]
    audit = json.loads((package / "reset_baseline_state_audit.json").read_text())
    audit_event = audit["events"][-1]
    assert audit_event["case_isolation"] == {"night": True}
    assert audit_event["case_results"][0]["post_reset_equals_baseline"] is True
    assert audit_event["case_results"][0]["entity_counts"] == {
        "after_mutation": 1,
        "after_reset": 2,
        "before_mutation": 2,
    }
    assert audit_event["case_results"][0]["merged_entities"]["after_mutation"] == [
        merged["entity_id"]
    ]
    assert audit_event["case_results"][0]["merged_entities"]["after_reset"] == []
    assert checkpoint["checkpoint_uuid"] in {
        item["checkpoint_uuid"] for item in reset["checkpoints"]
    }

    _checkpoint, _restore_event, restored = review_working_state.restore_checkpoint(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        checkpoint_uuid=checkpoint["checkpoint_uuid"],
    )

    assert restored["task_decisions"]["day_task"]["action"] == review_sessions.ACTION_LOOKS_CORRECT
    assert "day" in restored["sweep_acknowledgements"]
    assert restored["active_correction_ids"] == [correction.correction_id]
    restored_day_active = review_working_state.effective_case_entities(
        registry["cases"]["day"],
        restored,
    )
    assert restored_day_active[0]["entity_id"] == merged["entity_id"]


def test_case_versions_exposes_only_public_review_versions(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    roles = [
        case_manifests.VideoRole(
            label="Original",
            path="media/synthetic/original_reference.mp4",
            role="original",
            sha256=None,
            source_frame_end=4,
            source_frame_start=0,
        ),
        case_manifests.VideoRole(
            label="Current result",
            path="media/synthetic/original_reference.mp4",
            role="current_result",
            sha256=None,
            source_frame_end=4,
            source_frame_start=0,
        ),
        case_manifests.VideoRole(
            label="Correction preview",
            path="media/synthetic/original_reference.mp4",
            role="correction_preview",
            sha256=None,
            source_frame_end=4,
            source_frame_start=0,
        ),
    ]
    manifest = replace(manifest, video_roles=tuple(roles))
    case_manifests.write_case_manifest(package / "cases" / "synthetic.json", manifest)

    versions = review_working_state.case_versions(package, None, case_id="synthetic")["versions"]

    assert [role["role"] for role in versions] == ["original", "automatic", "latest_reviewed"]
    assert versions[1]["path"] == versions[2]["path"]


def test_entity_registry_adds_source_backed_explicit_keep_face_track(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_review_entity_shell(package, "synthetic")
    write_explicit_keep_face_policy(package, manifest, track_id=10)

    audit = review_workspace.apply_review_entity_coverage_patch(package)
    registry = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    entities = registry["cases"]["synthetic"]

    assert audit["passed"] is True
    assert audit["summary"]["before_counts"]["synthetic"] == 0
    assert audit["summary"]["after_counts"]["synthetic"] == 1
    assert audit["summary"]["entity_count_delta"]["synthetic"] == 1
    assert audit["summary"]["explicit_keep_tracks_added"]["synthetic"][0]["track_id"] == "10"
    assert len(entities) == 1
    assert entities[0]["current_policy"] == privacy_entities.POLICY_KEEP_VISIBLE
    assert entities[0]["automatic_source_track_ids"] == ["10"]


def test_entity_focus_overlay_audit_enriches_source_backed_keep_geometry(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_review_entity_shell(package, "synthetic")
    write_explicit_keep_face_policy(package, manifest, track_id=10)
    review_workspace.apply_review_entity_coverage_patch(package)
    registry = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    entity = registry["cases"]["synthetic"][0]
    entity.pop("representative_bbox", None)
    registry["cases"]["synthetic"][0] = entity
    write_json(package / "review_entities.json", registry)

    audit = review_workspace.write_entity_focus_overlay_audit(package)
    updated = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    enriched = updated["cases"]["synthetic"][0]

    assert audit["summary"]["focus_ready_count"]["synthetic"] == 1
    assert audit["summary"]["missing_geometry_count"]["synthetic"] == 0
    assert audit["summary"]["enriched_entity_ids"]["synthetic"] == [entity["entity_id"]]
    assert enriched["representative_source_frame"] == 2
    assert enriched["representative_bbox"] == [22, 22, 56, 42]


def test_entity_registry_fails_if_explicit_keep_track_stays_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_review_entity_shell(package, "synthetic")
    write_explicit_keep_face_policy(package, manifest, track_id=10)

    def broken_entity_payload(**_: object) -> dict[str, object]:
        return {
            "case_id": "synthetic",
            "category": CATEGORY_FACE,
            "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
            "entity_id": "entity_without_track_lineage",
            "label": "Person 01",
        }

    monkeypatch.setattr(
        review_workspace,
        "source_backed_keep_entity_payload",
        broken_entity_payload,
    )

    with pytest.raises(ValueError, match="explicit keep-policy face tracks omitted"):
        review_workspace.apply_review_entity_coverage_patch(package)


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
    assert "Final Review Workspace v2.3.4" in html
    assert "final_review_workspace.v2.3.4" in js
    assert "function closestElement" in js
    assert "reviewDelegated" in js
    assert "function refreshSidePanel" in js
    assert "function taskState" in js
    assert "Needs correction" in js
    assert "Unsure / Escalated" in js
    assert "Open Corrections" in js
    assert "openEntities(caseId)" in js
    assert "entitiesModal" in html
    assert "mergeModal" in html
    assert "Search objects" not in js
    assert "window.prompt" not in js
    assert "entityFilterButtons" in js
    assert "policy-segment" in css
    assert "openMergeModal" in js
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
    assert (
        "renderCase(caseId);"
        not in js[
            js.index("async function recordDecision") : js.index("function setDecisionPending")
        ]
    )
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
    write_synthetic_video(source, frame_count=12)
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
                source_frame_end=111,
                source_frame_start=100,
            )
        ],
    )
    case_manifests.write_case_manifest(package / "cases" / "day.json", manifest)
    records = [
        PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(20, 20, 54, 34),
            category=CATEGORY_PLATE,
            object_id="plate_1",
            provenance="plate_detector_observation",
            render_eligible=True,
            source_frame=frame,
            source_record={
                "region_id": "region_1",
                "score": 0.8,
                "status": "accepted_plate_track",
                "track_uid": "plate_track_0001",
            },
        )
        for frame in range(12)
    ]
    review_finalization.write_privacy_region_records(
        package / "regions" / "day_current_regions.jsonl",
        records,
    )

    review_workspace.write_entity_index(package)
    registry = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    entity = registry["cases"]["day"][0]

    assert entity["label"] == "Plate 01"
    assert entity["entity_uuid"]
    assert 100 <= entity["representative_source_frame"] <= 111
    assert entity["representative_thumbnail_path"]
    assert (package / entity["representative_thumbnail_path"]).exists()
    assert entity["review_entity_state"] == "confirmed"
    assert registry["unresolved_evidence_by_case"]["day"]["fragment_count"] == 0


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


def review_tracklet(
    track_id: str,
    *,
    category: str = CATEGORY_FACE,
    first: int = 0,
    last: int = 39,
    observations: int = 40,
    descriptor: tuple[float, ...] = (0.2, 0.3, 0.5, *([0.0] * 15)),
    evidence_score: float = 0.82,
    bbox: tuple[int, int, int, int] = (100, 100, 160, 180),
) -> review_workspace.ReviewTrackletEvidence:
    components = {
        "appearance_quality": 0.8,
        "aspect_ratio_stability": 0.8,
        "center_motion_continuity": 0.8,
        "confidence_lower_tail": 0.8,
        "confidence_median": 0.8,
        "confidence_peak": 0.9,
        "crop_quality": 0.8,
        "density": 0.8,
        "duration": 0.6,
        "edge_clipping": 0.05,
        "geometry_stability": 0.8,
        "internal_gap_burden": 0.0,
        "observation_count": min(1.0, observations / 40.0),
        "provenance_support": 1.0,
        "scale_motion_continuity": 0.8,
    }
    return review_workspace.ReviewTrackletEvidence(
        appearance_descriptor=descriptor,
        appearance_quality=0.8,
        aspect_ratio_stability=0.8,
        bbox_xyxy=bbox,
        category=category,
        center_motion_continuity=0.8,
        confidence_p10=0.8,
        crop_quality=0.8,
        density=observations / max(1, last - first + 1),
        detector_confidence_max=0.9,
        detector_confidence_median=0.8,
        duration_seconds=(last - first + 1) / 30.0,
        edge_penalty=0.05,
        evidence_components=components,
        evidence_score=evidence_score,
        first_frame=first,
        geometry_stability=0.8,
        jump_penalty=0.05,
        large_internal_gap_penalty=0.0,
        last_frame=last,
        mean_confidence=0.8,
        observation_count=observations,
        provenance_counts={"detector": observations},
        scale_motion_continuity=0.8,
        source_track_id=track_id,
        status="accepted",
    )


def test_review_entity_accumulator_attaches_same_appearance_fragments() -> None:
    profile = review_workspace.review_entity_profile()
    result = review_workspace.ReviewEntityAccumulator(profile).accumulate(
        case_id="synthetic",
        previous_user_entity_count=3,
        tracklets=[
            review_tracklet("person_main"),
            review_tracklet(
                "person_fragment",
                first=45,
                last=48,
                observations=4,
                evidence_score=0.24,
                bbox=(104, 102, 164, 182),
            ),
        ],
    )

    assert result.metrics()["confirmed_entity_count"] == 1
    assert result.metrics()["attached_fragment_count"] == 1
    assert result.metrics()["unresolved_fragment_count"] == 0


def test_review_entity_accumulator_rejects_simultaneous_similar_people() -> None:
    profile = review_workspace.review_entity_profile()
    result = review_workspace.ReviewEntityAccumulator(profile).accumulate(
        case_id="synthetic",
        previous_user_entity_count=2,
        tracklets=[
            review_tracklet("person_main", first=10, last=60),
            review_tracklet(
                "person_overlap",
                first=30,
                last=34,
                observations=4,
                evidence_score=0.24,
                bbox=(102, 101, 162, 181),
            ),
        ],
    )

    assert result.metrics()["attached_fragment_count"] == 0
    assert result.metrics()["simultaneous_overlap_merge_rejection_count"] == 1


def test_review_entity_accumulator_collapses_weak_unrelated_noise() -> None:
    profile = review_workspace.review_entity_profile()
    result = review_workspace.ReviewEntityAccumulator(profile).accumulate(
        case_id="synthetic",
        previous_user_entity_count=1,
        tracklets=[
            review_tracklet(
                "weak_noise",
                first=0,
                last=0,
                observations=1,
                evidence_score=0.05,
                bbox=(10, 10, 14, 14),
            )
        ],
    )

    assert result.metrics()["new_user_facing_entity_count"] == 0
    assert result.metrics()["unresolved_fragment_count"] == 1


def test_review_entity_accumulator_keeps_different_plates_separate() -> None:
    profile = review_workspace.review_entity_profile()
    result = review_workspace.ReviewEntityAccumulator(profile).accumulate(
        case_id="synthetic",
        previous_user_entity_count=2,
        tracklets=[
            review_tracklet(
                "plate_main",
                category=CATEGORY_PLATE,
                descriptor=(0.9, 0.1, *([0.0] * 16)),
                bbox=(100, 300, 180, 330),
            ),
            review_tracklet(
                "plate_other",
                category=CATEGORY_PLATE,
                first=45,
                last=48,
                observations=4,
                descriptor=(0.0, 0.9, 0.1, *([0.0] * 15)),
                evidence_score=0.2,
                bbox=(102, 300, 182, 330),
            ),
        ],
    )

    assert result.metrics()["attached_fragment_count"] == 0
    assert result.metrics()["unresolved_fragment_count"] == 1


def test_review_entity_accumulator_rejects_long_gap_scene_barrier() -> None:
    profile = review_workspace.review_entity_profile()
    result = review_workspace.ReviewEntityAccumulator(profile).accumulate(
        case_id="synthetic",
        previous_user_entity_count=2,
        tracklets=[
            review_tracklet("person_main"),
            review_tracklet(
                "person_after_cut",
                first=300,
                last=304,
                observations=4,
                evidence_score=0.24,
                bbox=(104, 102, 164, 182),
            ),
        ],
    )

    assert result.metrics()["attached_fragment_count"] == 0
    assert any(
        item["reason"] == "gap_too_large_for_review_entity_attachment"
        for item in result.merge_audit()["rejected_merges"]
    )


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

        preview_response = post_json(
            f"{base_url}/api/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        saved_correction = dict(correction)
        saved_correction["reviewer_identity"] = {
            "attributed_at_utc": None,
            "display_name": "Analyst",
            "event_hash": None,
            "event_id": None,
            "reviewer_id": "analyst@example.test",
            "schema_version": "reviewer_attribution_current.v1",
            "session_id": session["session_id"],
        }
        correction_response = post_json(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": saved_correction,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        session_dir = package / "sessions" / session["session_id"]
        chain = review_sessions.validate_event_hash_chain(session_dir)
        events = review_sessions.load_events(session_dir)

        assert preview_response["event"]["event_type"] == "correction_previewed"
        assert correction_response["event"]["event_type"] == "manual_correction_saved"
        assert events[-3]["event_type"] == "correction_previewed"
        assert events[-2]["event_type"] == "manual_correction_saved"
        assert events[-1]["event_type"] == "manual_issue_created"
        assert chain["passed"] is True
        assert (session_dir / "corrections.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_workspace_server_rejects_unpreviewed_manual_correction_save(tmp_path: Path) -> None:
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
        correction = {
            "case_id": manifest.case_id,
            "category": CATEGORY_PLATE,
            "configuration_freeze_hash": session_response["session"]["configuration_freeze_hash"],
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

        status, body = post_json_error(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "reviewed",
                "session_id": session_response["session"]["session_id"],
            },
        )

        assert status == 409
        assert "Preview the correction" in str(body["error"])
    finally:
        server.shutdown()
        server.server_close()


def test_workspace_server_rejects_stale_manual_correction_preview_token(
    tmp_path: Path,
) -> None:
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
        preview_response = post_json(
            f"{base_url}/api/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        changed = dict(correction)
        changed["source_frame_end"] = 1

        status, body = post_json_error(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": changed,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )

        assert status == 409
        assert "draft changed" in str(body["error"])
    finally:
        server.shutdown()
        server.server_close()


def test_workspace_server_reuses_duplicate_manual_correction_save(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
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
        preview_response = post_json(
            f"{base_url}/api/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        payload = {
            "actor": {"reviewer_id": "analyst@example.test"},
            "correction": correction,
            "preview_token": preview_response["preview"]["preview_token"],
            "reason": "reviewed",
            "session_id": session["session_id"],
        }

        first = post_json(f"{base_url}/api/corrections", payload)
        second = post_json(f"{base_url}/api/corrections", payload)
        session_dir = package / "sessions" / session["session_id"]
        state = review_working_state.derive_working_state(package, session_dir)
        save_events = [
            event
            for event in review_sessions.load_events(session_dir)
            if event.get("event_type") == "manual_correction_saved"
        ]

        assert second["idempotent_retry"] is True
        assert first["materialization_job"]["job_id"] == second["materialization_job"]["job_id"]
        assert len(server.materialization_jobs.jobs) == 1
        assert state["active_correction_ids"] == [first["event"]["after_state"]["correction_id"]]
        assert len(save_events) == 1
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


def test_materialization_job_manager_coalesces_queued_case_jobs(tmp_path: Path) -> None:
    package = tmp_path / "package"
    session_dir = package / "sessions" / "session_test"
    session_dir.mkdir(parents=True)
    manager = review_workspace.MaterializationJobManager(package)
    manager.debounce_seconds = 60.0

    first = manager.enqueue(
        session_dir=session_dir,
        case_id="day",
        actor={"reviewer_id": "analyst"},
        reason="first policy change",
        accepted_revision="revision_a",
        affected_entity_ids=["entity_a"],
        requested_policy=privacy_entities.POLICY_KEEP_VISIBLE,
    )
    second = manager.enqueue(
        session_dir=session_dir,
        case_id="day",
        actor={"reviewer_id": "analyst"},
        reason="second policy change",
        accepted_revision="revision_b",
        affected_entity_ids=["entity_b"],
        requested_policy=privacy_entities.POLICY_REDACT,
    )

    with manager.lock:
        state = manager.case_state[("session_test", "day")]
        timer = state.get("debounce_timer")
        if timer is not None:
            timer.cancel()
        state["debounce_timer"] = None
        first_job = manager.jobs[first["job_id"]]
        second_job = manager.jobs[second["job_id"]]

    assert first_job["status"] == "superseded"
    assert first_job["superseded_by"] == second["job_id"]
    assert second_job["status"] == "queued"
    assert state["latest_requested"] == second["job_id"]
    assert manager.metrics["day"]["accepted_state_mutations"] == 2
    assert manager.metrics["day"]["jobs_coalesced"] == 1

    audit = json.loads((package / "materialization_job_audit.json").read_text())
    assert audit["cases"]["day"]["jobs_coalesced"] == 1


def media_hash(path: Path) -> str:
    from redactor import media

    return media.sha256_file(path)
