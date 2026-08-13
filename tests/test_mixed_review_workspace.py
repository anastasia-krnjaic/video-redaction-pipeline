from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest

from redactor import (
    case_ids,
    case_pipeline,
    mixed_review_workspace,
    review_sessions,
    review_workspace,
)
from redactor.models import MediaInfo, VideoStreamInfo
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    stable_hash,
)
from redactor.review_corrections import (
    INTERPOLATION_NONE,
    OP_ADD,
    CorrectionKeyframe,
    ReviewCorrection,
)


def fake_media_info() -> MediaInfo:
    return MediaInfo(
        audio_stream=None,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        created_at="2026-08-09T00:00:00Z",
        duration_seconds=0.2,
        file_size_bytes=12,
        ffprobe_version="ffprobe test",
        schema_version="1.0",
        sha256="source-sha256",
        source_filename="source.mp4",
        source_path="source.mp4",
        video_stream=VideoStreamInfo(
            average_frame_rate="10/1",
            codec="h264",
            frame_count="2",
            height=80,
            index=0,
            nominal_frame_rate="10/1",
            pixel_format="yuv420p",
            time_base="1/10240",
            width=120,
        ),
    )


def write_synthetic_source_video(path: Path, *, frame_count: int = 2) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 80))
    try:
        for frame_index in range(frame_count):
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            frame[:, :] = (30 + frame_index * 35, 80, 150)
            cv.rectangle(frame, (8, 8), (34, 34), (220, 220, 220), -1)
            cv.rectangle(frame, (48, 36), (96, 64), (245, 245, 245), -1)
            writer.write(frame)
    finally:
        writer.release()


def region(category: str, frame: int, object_id: str) -> PrivacyRegionRecord:
    return PrivacyRegionRecord(
        approval_state=APPROVAL_ACCEPTED,
        bbox_xyxy=(10, 10, 30, 30) if category == CATEGORY_FACE else (50, 40, 90, 58),
        category=category,
        object_id=object_id,
        provenance="test",
        render_eligible=True,
        source_frame=frame,
        source_record={"track_uid": object_id},
    )


def workspace_fixture(
    tmp_path: Path,
    *,
    categories: tuple[str, ...] = (CATEGORY_FACE, CATEGORY_PLATE),
) -> dict[str, object]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")
    info = fake_media_info()
    case_root = tmp_path / "case"
    case_root.mkdir()
    context = case_pipeline.CaseContext(
        case_id="case",
        output_dir=case_root,
        selected_categories=categories,
        source_identity=case_pipeline.SourceIdentity.from_video(source, info=info),
        source_info=info,
        source_video=source,
    )
    records = []
    if CATEGORY_FACE in categories:
        records.append(region(CATEGORY_FACE, 0, "face_track_1"))
    if CATEGORY_PLATE in categories:
        records.append(region(CATEGORY_PLATE, 1, "plate_track_1"))
    regions_path = case_root / "render_plan" / "effective_privacy_regions.jsonl"
    case_pipeline.write_privacy_region_records(regions_path, records)
    render_plan = {
        "blockers": [],
        "case_id": "case",
        "effective_region_count": len(records),
        "effective_regions_artifact": case_pipeline.artifact_reference(
            case_root,
            regions_path,
            role="effective_privacy_regions",
        ).to_dict(),
        "render_plan_hash": "render-plan-hash",
        "schema_version": case_pipeline.RENDER_PLAN_SCHEMA_VERSION,
        "selected_categories": list(categories),
    }
    write_category_tasks(case_root)
    write_single_category_previews(case_root, context, rendered_region_count=len(records))
    preview = case_root / "combined_preview" / "automatic_preview.mp4"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"combined preview")
    preview_manifest = {
        "output_video": "combined_preview/automatic_preview.mp4",
        "output_video_hash": case_pipeline.media.sha256_file(preview),
        "rendered_region_count": len(records),
        "source_video_hash": context.source_identity.sha256,
    }
    return {
        "case_root": case_root,
        "combined_preview": {"artifacts": (), "manifest": preview_manifest, "summary": {}},
        "context": context,
        "render_plan": render_plan,
    }


def write_category_tasks(case_root: Path) -> None:
    face_path = case_root / "review" / "review_tasks.json"
    face_path.parent.mkdir(parents=True, exist_ok=True)
    face_path.write_text(
        json.dumps(
            {
                "reviewer_tasks": [
                    {
                        "category": "action_required",
                        "case_id": "case",
                        "final_export_blocking": True,
                        "frame_end": 0,
                        "frame_start": 0,
                        "object_id": "face_track_1",
                        "reviewer_task_id": "face_review_0001",
                        "title": "Face review",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plate_path = case_root / CATEGORY_PLATE / "review" / "review_tasks.json"
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    plate_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "blocking": True,
                        "frame_end": 1,
                        "frame_start": 1,
                        "reason_codes": ["unresolved_internal_gap"],
                        "review_task_id": "plate_review_0001",
                        "severity": "blocker",
                        "source_stage": "plate_tracking",
                        "title": "Plate review",
                        "track_uids": ["plate_track_1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def write_single_category_previews(
    case_root: Path,
    context: case_pipeline.CaseContext,
    *,
    rendered_region_count: int,
) -> None:
    face_preview = case_root / "automatic_preview" / "redacted_preview.mp4"
    face_preview.parent.mkdir(parents=True, exist_ok=True)
    face_preview.write_bytes(b"face automatic preview")
    face_hash = case_pipeline.media.sha256_file(face_preview)
    (face_preview.parent / "render_manifest.json").write_text(
        json.dumps(
            {
                "output_video": "automatic_preview/redacted_preview.mp4",
                "output_video_hash": face_hash,
                "rendered_region_count": rendered_region_count,
                "source_video_hash": context.source_identity.sha256,
            }
        ),
        encoding="utf-8",
    )
    plate_preview = case_root / CATEGORY_PLATE / "automatic_preview" / "automatic_preview.mp4"
    plate_preview.parent.mkdir(parents=True, exist_ok=True)
    plate_preview.write_bytes(b"plate automatic preview")
    plate_hash = case_pipeline.media.sha256_file(plate_preview)
    (plate_preview.parent / "render_manifest.json").write_text(
        json.dumps(
            {
                "output_video": f"{CATEGORY_PLATE}/automatic_preview/automatic_preview.mp4",
                "output_video_hash": plate_hash,
                "rendered_region_count": rendered_region_count,
                "source_video_hash": context.source_identity.sha256,
            }
        ),
        encoding="utf-8",
    )


def write_source_backed_entity_artifacts(case_root: Path) -> None:
    (case_root / "track_census.json").write_text(
        json.dumps(
            {
                "schema_id": "track_census",
                "source_frame_count": 2,
                "source_video_hash": "source-sha256",
                "tracks": [
                    {
                        "assignment_provenance": "automatic_case_policy",
                        "first_frame": 0,
                        "last_frame": 0,
                        "policy_assignment": "redact",
                        "quality_status": "auto_confirmed",
                        "representative_frame": 0,
                        "track_id": 1,
                    },
                    {
                        "assignment_provenance": "automatic_case_policy",
                        "first_frame": 1,
                        "last_frame": 1,
                        "policy_assignment": "redact",
                        "quality_status": "auto_confirmed",
                        "representative_frame": 1,
                        "track_id": 2,
                    },
                    {
                        "assignment_provenance": "explicit_case_policy",
                        "first_frame": 0,
                        "last_frame": 0,
                        "policy_assignment": "keep",
                        "quality_status": "auto_confirmed",
                        "representative_frame": 0,
                        "track_id": 3,
                    },
                    {
                        "assignment_provenance": "explicit_case_policy",
                        "first_frame": 1,
                        "last_frame": 1,
                        "policy_assignment": "review",
                        "quality_status": "review_candidate",
                        "representative_frame": 1,
                        "track_id": 4,
                    },
                    {
                        "assignment_provenance": "automatic_case_policy",
                        "first_frame": 1,
                        "last_frame": 1,
                        "policy_assignment": "ignore",
                        "quality_status": "rejected",
                        "representative_frame": 1,
                        "track_id": 9,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    tracking_dir = case_root / CATEGORY_PLATE / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    (tracking_dir / "track_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "plate_tracking.v1",
                "tracks": [
                    {
                        "first_frame": 0,
                        "last_frame": 0,
                        "review_reasons": [],
                        "status": "accepted_plate_track",
                        "track_id": 1,
                        "track_uid": "plate_track_1",
                    },
                    {
                        "first_frame": 1,
                        "last_frame": 1,
                        "review_reasons": [],
                        "status": "accepted_plate_track",
                        "track_id": 2,
                        "track_uid": "plate_track_2",
                    },
                    {
                        "first_frame": 1,
                        "last_frame": 1,
                        "review_reasons": ["gap_requires_review"],
                        "status": "needs_review",
                        "track_id": 3,
                        "track_uid": "plate_track_3",
                    },
                    {
                        "first_frame": 1,
                        "last_frame": 1,
                        "review_reasons": ["wrong_class"],
                        "status": "rejected_noise_track",
                        "track_id": 9,
                        "track_uid": "plate_track_9",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "frame_index": 0,
            "tracks": [
                {
                    "bbox_xyxy": [50, 40, 90, 58],
                    "provenance": "detector",
                    "status": "accepted_plate_track",
                    "track_id": 1,
                    "track_uid": "plate_track_1",
                }
            ],
        },
        {
            "frame_index": 1,
            "tracks": [
                {
                    "bbox_xyxy": [51, 41, 91, 59],
                    "provenance": "detector",
                    "status": "accepted_plate_track",
                    "track_id": 2,
                    "track_uid": "plate_track_2",
                },
                {
                    "bbox_xyxy": [52, 42, 92, 60],
                    "provenance": "detector",
                    "status": "needs_review",
                    "track_id": 3,
                    "track_uid": "plate_track_3",
                },
            ],
        },
    ]
    (tracking_dir / "tracks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tracking_dir / "review_intervals.json").write_text(
        json.dumps(
            [
                {
                    "frame_end": 1,
                    "frame_start": 1,
                    "reason_codes": ["gap_requires_review"],
                    "track_uids": ["plate_track_3"],
                }
            ]
        ),
        encoding="utf-8",
    )


def write_source_backed_tasks(case_root: Path) -> None:
    face_path = case_root / "review" / "review_tasks.json"
    face_path.write_text(
        json.dumps(
            {
                "reviewer_tasks": [
                    {
                        "allowed_user_actions": [
                            {"action": review_sessions.ACTION_LOOKS_CORRECT},
                            {"action": review_sessions.ACTION_PRIVACY_LEAK},
                        ],
                        "category": "action_required",
                        "case_id": "case",
                        "final_export_blocking": True,
                        "frame_end": 0,
                        "frame_start": 0,
                        "object_id": "face_track_1",
                        "reviewer_task_id": "face_review_0001",
                    },
                    {
                        "allowed_user_actions": [
                            {"action": review_sessions.ACTION_ESCALATE},
                        ],
                        "category": "action_required",
                        "case_id": "case",
                        "final_export_blocking": True,
                        "frame_end": 1,
                        "frame_start": 1,
                        "object_id": "face_track_4",
                        "reviewer_task_id": "face_review_0002",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    plate_path = case_root / CATEGORY_PLATE / "review" / "review_tasks.json"
    plate_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "allowed_actions": [
                            "Looks correct",
                            "Needs correction",
                            "Not a license plate",
                        ],
                        "blocking": True,
                        "frame_end": 0,
                        "frame_start": 0,
                        "review_task_id": "plate_review_0001",
                        "severity": "blocker",
                        "track_uids": ["plate_track_1"],
                    },
                    {
                        "allowed_actions": [
                            "Needs correction",
                            "Not a license plate",
                        ],
                        "blocking": True,
                        "frame_end": 1,
                        "frame_start": 1,
                        "review_task_id": "plate_review_0002",
                        "severity": "blocker",
                        "track_uids": ["plate_track_3"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def source_span_face_task(**overrides: Any) -> dict[str, Any]:
    task: dict[str, Any] = {
        "allowed_user_actions": [
            {"action": review_sessions.ACTION_PRIVACY_LEAK},
            {"action": review_sessions.ACTION_ESCALATE},
        ],
        "case_id": "case",
        "category": "action_required",
        "final_export_blocking": True,
        "frame_end": 1,
        "frame_start": 1,
        "media_roles": [
            {
                "media_audit": {
                    "source_frame_range": {
                        "frame_count": 1,
                        "frame_end": 1,
                        "frame_start": 1,
                    }
                },
                "role": "original",
            }
        ],
        "object_category": CATEGORY_FACE,
        "object_id": None,
        "primary_raw_review_item_ids": ["face_interval_gap_0001"],
        "raw_reason_codes": [
            "insufficient_observed_hits",
            "needs_review_interval",
            "unresolved_privacy_frames",
        ],
        "raw_review_item_ids": ["face_interval_gap_0001"],
        "reviewer_task_id": "face_interval_gap_review",
        "title": "Automatic redaction was uncertain",
        "what_to_check": ["Inspect the original context."],
    }
    for key, value in overrides.items():
        if value is None:
            task.pop(key, None)
        else:
            task[key] = value
    return task


def append_face_task(case_root: Path, task: dict[str, Any]) -> None:
    face_path = case_root / "review" / "review_tasks.json"
    payload = json.loads(face_path.read_text(encoding="utf-8"))
    payload["reviewer_tasks"].append(task)
    face_path.write_text(json.dumps(payload), encoding="utf-8")


def category_results(
    case_root: Path,
    categories: tuple[str, ...] = (CATEGORY_FACE, CATEGORY_PLATE),
) -> list[case_pipeline.CategoryRunResult]:
    return [category_result(case_root, category) for category in categories]


def category_result(case_root: Path, category: str) -> case_pipeline.CategoryRunResult:
    configuration_identity = {
        "category": category,
        "configuration_hash": f"{category}-config",
        "source_sha256": "source-sha256",
    }
    manifest_path = case_root / category / "category_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_hashes": {},
        "artifacts": [],
        "blockers": [],
        "case_id": "case",
        "category": category,
        "configuration_identity": configuration_identity,
        "schema_version": "category_run_manifest.v1",
        "source_lineage": {
            "frame_count": 2,
            "height": 80,
            "processed_span": {
                "first_processed_frame": 0,
                "full_source_processed": True,
                "last_processed_frame": 1,
                "source_frame_count": 2,
            },
            "schema_version": "category_source_lineage.v1",
            "source_sha256": "source-sha256",
            "width": 120,
        },
        "stage_outcomes": [
            {
                "category": category,
                "schema_version": "stage_status.v1",
                "stage": "PRIVACY_REGIONS",
                "status": "succeeded",
            }
        ],
    }
    payload["category_run_content_sha256"] = stable_hash(payload)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return case_pipeline.CategoryRunResult(
        artifacts=(
            case_pipeline.artifact_reference(
                case_root,
                manifest_path,
                role="category_run_manifest",
                category=category,
            ),
        ),
        category=category,
        stages=(),
        configuration_identity=configuration_identity,
    )


def mutated_category_result(
    case_root: Path,
    category: str,
    mutate: Any,
) -> case_pipeline.CategoryRunResult:
    result = category_result(case_root, category)
    manifest_path = case_root / category / "category_run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return case_pipeline.CategoryRunResult(
        artifacts=(
            case_pipeline.artifact_reference(
                case_root,
                manifest_path,
                role="category_run_manifest",
                category=category,
            ),
        ),
        category=category,
        stages=(),
        configuration_identity=result.configuration_identity,
    )


def test_mixed_review_workspace_entities_tasks_and_manifest_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)

    first = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )
    workspace_root = fixture["case_root"] / first["workspace_dir"]
    first_entities = json.loads((workspace_root / "review_entities.json").read_text())
    first_tasks = json.loads((workspace_root / "reviewer_tasks.json").read_text())

    second = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )
    second_entities = json.loads(
        (fixture["case_root"] / second["workspace_dir"] / "review_entities.json").read_text()
    )

    entities = first_entities["cases"]["case"]
    tasks = first_tasks["tasks"]
    assert [entity["label"] for entity in entities] == ["Person 01", "Plate 01"]
    assert [entity["entity_id"] for entity in entities] == [
        entity["entity_id"] for entity in second_entities["cases"]["case"]
    ]
    assert [task["category"] for task in tasks] == [CATEGORY_FACE, CATEGORY_PLATE]
    assert tasks[1]["normal_task_class"] == "plate_track_review_required"
    assert {task["entity_id"] for task in tasks} <= {entity["entity_id"] for entity in entities}
    assert first["validation"]["passed"] is True
    frontend_manifest = json.loads((workspace_root / "frontend_build_manifest.json").read_text())
    manifest = json.loads((workspace_root / "workspace_manifest.json").read_text())
    assert frontend_manifest["selected_frontend"] == "react"
    assert manifest["selected_frontend"] == "react"
    assert manifest["react_app_root"] == "app/"
    assert (workspace_root / "app" / "index.html").exists()
    assert (workspace_root / "legacy.html").exists()
    assert "legacy.html" in (workspace_root / "index.html").read_text(encoding="utf-8")


def test_authoritative_workspace_manifest_is_sole_runtime_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    artifact_roles = {artifact.role for artifact in result["artifacts"]}
    manifest = json.loads((workspace_root / "workspace_manifest.json").read_text())
    assert "review_workspace_manifest" in artifact_roles
    assert "mixed_review_manifest" not in artifact_roles
    assert not (workspace_root / "mixed_review_manifest.json").exists()
    assert manifest["content_identities"]["entity_registry"]["sha256"]
    assert manifest["automatic_preview"]["source_sha256"] == "source-sha256"
    assert manifest["automatic_preview"]["automatic_preview_sha256"]
    assert manifest["automatic_preview"]["region_count"] == 2
    identities = manifest["category_run_manifest_identities"]
    assert set(identities) == {CATEGORY_FACE, CATEGORY_PLATE}
    assert manifest["category_run_manifest_identity_hash"] == stable_hash(identities)
    assert identities[CATEGORY_FACE]["source_sha256"] == "source-sha256"
    assert identities[CATEGORY_FACE]["configuration_hash"] == "face-config"
    assert identities[CATEGORY_PLATE]["configuration_hash"] == "license_plate-config"
    case_payload = json.loads((workspace_root / "cases" / "case.json").read_text())
    assert case_payload["lineage"]["category_run_manifest_identities"] == identities
    validation = json.loads((workspace_root / "validation.json").read_text())
    summary = json.loads((workspace_root / "workspace_summary.json").read_text())
    assert validation["category_run_manifest_identity_count"] == 2
    assert validation["category_run_manifest_identity_hash"] == stable_hash(identities)
    assert validation["automatic_preview_region_count"] == 2
    assert summary["automatic_preview_region_count"] == 2
    assert manifest["lifecycle"]["review_session"]["status"] == "not_started"
    assert manifest["lifecycle"]["latest_reviewed"]["status"] == "not_run"
    assert manifest["lifecycle"]["finalization"]["status"] == "not_run"
    assert manifest["lifecycle"]["evaluation"]["status"] == "not_run"
    baseline_regions = workspace_root / "regions" / "case_current_regions.jsonl"
    assert baseline_regions.exists()
    assert manifest["content_identities"]["automatic_baseline_regions"]["sha256"]
    assert (
        manifest["latest_reviewed_publication"]["publication_mode"]
        == "reviewed_region_compiler.v1"
    )
    assert manifest["latest_reviewed_publication"]["status"] == "supported"


@pytest.mark.parametrize("category", [CATEGORY_FACE, CATEGORY_PLATE])
def test_single_category_canonical_workspace_binds_one_run_identity_and_supports_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path, categories=(category,))

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"], categories=(category,)),
        render_plan=fixture["render_plan"],
        combined_preview=None,
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    manifest = json.loads((workspace_root / "workspace_manifest.json").read_text())
    identities = manifest["category_run_manifest_identities"]
    assert set(identities) == {category}
    assert manifest["category_run_manifest_identity_hash"] == stable_hash(identities)
    assert manifest["latest_reviewed_publication"]["status"] == "supported"
    assert manifest["automatic_preview"]["region_count"] == 1
    assert (workspace_root / "regions" / "case_current_regions.jsonl").exists()


def test_category_run_manifest_identity_failures_block_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    missing_face = case_pipeline.CategoryRunResult(
        category=CATEGORY_FACE,
        stages=(),
        configuration_identity={
            "category": CATEGORY_FACE,
            "configuration_hash": "face-config",
            "source_sha256": "source-sha256",
        },
    )

    with pytest.raises(ValueError, match="exactly match selected categories"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=[
                missing_face,
                category_result(fixture["case_root"], CATEGORY_PLATE),
            ],
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    fixture = workspace_fixture(stale_root)
    results = category_results(fixture["case_root"])
    (fixture["case_root"] / CATEGORY_FACE / "category_run_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="category run manifest hash mismatch"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=results,
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "source",
            lambda payload: payload["source_lineage"].__setitem__(
                "source_sha256",
                "wrong-source",
            ),
            "source hash mismatch",
        ),
        (
            "config",
            lambda payload: payload["configuration_identity"].__setitem__(
                "configuration_hash",
                "wrong-config",
            ),
            "configuration identity mismatch",
        ),
        (
            "category",
            lambda payload: payload.__setitem__("category", CATEGORY_PLATE),
            "category mismatch",
        ),
        (
            "case",
            lambda payload: payload.__setitem__("case_id", "other"),
            "case_id mismatch",
        ),
        (
            "missing_content_hash",
            lambda payload: payload.pop("category_run_content_sha256"),
            "content hash is required",
        ),
    ],
)
def test_category_run_manifest_lineage_mismatches_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mutate: Any,
    message: str,
) -> None:
    del name
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)

    with pytest.raises(ValueError, match=message):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=[
                mutated_category_result(fixture["case_root"], CATEGORY_FACE, mutate),
                category_result(fixture["case_root"], CATEGORY_PLATE),
            ],
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


def test_session_reload_rejects_changed_category_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )
    workspace_root = fixture["case_root"] / result["workspace_dir"]
    freeze_hash = stable_hash(
        json.loads((workspace_root / "configuration_freeze.json").read_text())
    )
    case_payload = json.loads((workspace_root / "cases" / "case.json").read_text())
    session = review_sessions.create_review_session(
        package_root=workspace_root,
        reviewer=review_sessions.ReviewerIdentity(display_name="Reviewer"),
        source_revision=None,
        configuration_freeze_hash=freeze_hash,
        case_manifest_hashes={"case": stable_hash(case_payload)},
        source_hashes={"case": "source-sha256"},
        workspace_identity_hashes=review_workspace.workspace_identity_hashes(workspace_root),
    )
    session_dir = review_sessions.session_directory(workspace_root, session.session_id)
    manifest_path = workspace_root / "workspace_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["category_run_manifest_identities"][CATEGORY_FACE]["sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    binding = review_workspace.validate_session_workspace_binding(workspace_root, session_dir)

    assert binding["ok"] is False
    assert "category_run_manifest_identity_hash" in binding["mismatches"]


def test_source_backed_entities_include_policy_and_no_task_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    records = [
        region(CATEGORY_FACE, 0, "face_track_1"),
        region(CATEGORY_FACE, 1, "face_track_2"),
        region(CATEGORY_PLATE, 0, "plate_track_1"),
        region(CATEGORY_PLATE, 1, "plate_track_2"),
    ]
    regions_path = fixture["case_root"] / "render_plan" / "effective_privacy_regions.jsonl"
    case_pipeline.write_privacy_region_records(regions_path, records)
    fixture["render_plan"]["effective_region_count"] = len(records)
    fixture["render_plan"]["effective_regions_artifact"] = case_pipeline.artifact_reference(
        fixture["case_root"],
        regions_path,
        role="effective_privacy_regions",
    ).to_dict()
    write_source_backed_entity_artifacts(fixture["case_root"])
    write_source_backed_tasks(fixture["case_root"])

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    entities = json.loads((workspace_root / "review_entities.json").read_text())["cases"]["case"]
    tasks = json.loads((workspace_root / "reviewer_tasks.json").read_text())["tasks"]
    by_key = {
        entity["source_lineage"]["technical_id"]: entity
        for entity in entities
    }
    labels = [entity["label"] for entity in entities]
    assert labels == [
        "Person 01",
        "Person 02",
        "Person 03",
        "Person 04",
        "Plate 01",
        "Plate 02",
        "Plate 03",
    ]
    assert by_key["face_track_3"]["current_policy"] == "keep_visible"
    assert by_key["face_track_4"]["current_policy"] == "review_required"
    assert by_key["plate_track_2"]["current_policy"] == "redact"
    assert by_key["plate_track_3"]["current_policy"] == "review_required"
    assert "face_track_9" not in by_key
    assert "plate_track_9" not in by_key
    assert by_key["face_track_2"]["source_lineage"]["source_task_ids"] == []
    assert by_key["plate_track_2"]["source_lineage"]["source_task_ids"] == []
    actions_by_id = {task["source_task_id"]: task["allowed_actions"] for task in tasks}
    labels_by_id = {task["source_task_id"]: task["category_allowed_actions"] for task in tasks}
    assert actions_by_id["face_review_0001"] == [
        review_sessions.ACTION_LOOKS_CORRECT,
        review_sessions.ACTION_PRIVACY_LEAK,
        review_sessions.ACTION_NOT_SENSITIVE,
    ]
    assert labels_by_id["plate_review_0002"] == [
        "Correct",
        "Needs correction",
        "Not sensitive",
    ]
    assert actions_by_id["plate_review_0002"] == [
        review_sessions.ACTION_LOOKS_CORRECT,
        review_sessions.ACTION_PRIVACY_LEAK,
        review_sessions.ACTION_NOT_SENSITIVE,
    ]
    binding_by_id = {task["source_task_id"]: task["target_binding"] for task in tasks}
    assert binding_by_id["face_review_0001"]["kind"] == "entity"
    assert binding_by_id["face_review_0002"]["kind"] == "entity"
    assert all(task["entity_id"] for task in tasks)


def test_source_span_action_required_task_does_not_create_fake_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    append_face_task(fixture["case_root"], source_span_face_task())

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    entities = json.loads((workspace_root / "review_entities.json").read_text())["cases"]["case"]
    tasks = json.loads((workspace_root / "reviewer_tasks.json").read_text())["tasks"]
    source_span = next(
        task for task in tasks if task["source_task_id"] == "face_interval_gap_review"
    )
    assert source_span["target_binding"] == {
        "binding_source": "authoritative_review_task_interval",
        "case_id": "case",
        "category": CATEGORY_FACE,
        "event_ids": ["face_interval_gap_0001", "face_interval_gap_review"],
        "kind": "source_span",
        "missing_entity_keys": ["face_interval_gap_0001"],
        "reason_codes": [
            "insufficient_observed_hits",
            "needs_review_interval",
            "unresolved_privacy_frames",
        ],
        "schema_version": "review_task_target_binding.v1",
        "source_frame_end": 1,
        "source_frame_start": 1,
        "source_review_category": "action_required",
        "source_task_id": "face_interval_gap_review",
        "unresolved_entity_keys": ["face_interval_gap_0001"],
    }
    assert source_span["entity_id"] is None
    assert source_span["target_entity_ids"] == []
    assert source_span["technical_track_ids"] == []
    assert source_span["correction_required"] is True
    assert "face_interval_gap" not in {
        entity["source_lineage"]["technical_id"] for entity in entities
    }


def test_entity_bound_action_required_task_preserves_real_entity_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    tasks = json.loads((workspace_root / "reviewer_tasks.json").read_text())["tasks"]
    face_task = next(task for task in tasks if task["source_task_id"] == "face_review_0001")
    assert face_task["target_binding"]["kind"] == "entity"
    assert face_task["target_binding"]["technical_track_ids"] == ["face_track_1"]
    assert face_task["entity_id"]
    assert face_task["target_entity_ids"] == [face_task["entity_id"]]


@pytest.mark.parametrize(
    ("task", "match"),
    [
        (
            source_span_face_task(media_roles=[]),
            "unbound_review_task",
        ),
        (
            source_span_face_task(case_id="wrong_case"),
            "case mismatch",
        ),
        (
            source_span_face_task(object_category=CATEGORY_PLATE),
            "category mismatch",
        ),
        (
            source_span_face_task(
                media_roles=[
                    {
                        "media_audit": {
                            "source_frame_range": {
                                "frame_count": 1,
                                "frame_end": 0,
                                "frame_start": 0,
                            }
                        },
                        "role": "original",
                    }
                ],
            ),
            "unbound_review_task",
        ),
        (
            source_span_face_task(
                primary_raw_review_item_ids=[],
                raw_review_item_ids=[],
                reviewer_task_id=None,
            ),
            "unbound_review_task",
        ),
        (
            source_span_face_task(category="reviewer_created_issue"),
            "unbound_review_task",
        ),
        (
            source_span_face_task(object_id="face_track_404"),
            "unbound_review_task",
        ),
    ],
)
def test_invalid_source_span_task_binding_attempts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: dict[str, Any],
    match: str,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    append_face_task(fixture["case_root"], task)

    with pytest.raises(ValueError, match=match):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


def test_source_backed_entities_get_real_representative_thumbnails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    write_synthetic_source_video(fixture["context"].source_video)
    records = [
        region(CATEGORY_FACE, 0, "face_track_1"),
        region(CATEGORY_FACE, 1, "face_track_2"),
        region(CATEGORY_PLATE, 0, "plate_track_1"),
        region(CATEGORY_PLATE, 1, "plate_track_2"),
    ]
    regions_path = fixture["case_root"] / "render_plan" / "effective_privacy_regions.jsonl"
    case_pipeline.write_privacy_region_records(regions_path, records)
    fixture["render_plan"]["effective_region_count"] = len(records)
    fixture["render_plan"]["effective_regions_artifact"] = case_pipeline.artifact_reference(
        fixture["case_root"],
        regions_path,
        role="effective_privacy_regions",
    ).to_dict()
    write_source_backed_entity_artifacts(fixture["case_root"])
    write_source_backed_tasks(fixture["case_root"])

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    entities = json.loads((workspace_root / "review_entities.json").read_text())["cases"]["case"]
    geometry_entities = [entity for entity in entities if entity.get("representative_bbox")]
    assert geometry_entities
    for entity in geometry_entities:
        thumbnail = entity.get("representative_thumbnail_path")
        identity = entity.get("representative_thumbnail_identity") or {}
        assert thumbnail
        assert (workspace_root / thumbnail).exists()
        assert identity["source_sha256"] == "source-sha256"
        assert identity["entity_id"] == entity["entity_id"]
        assert identity["category"] == entity["category"]

    plate_review = next(
        entity
        for entity in entities
        if (entity.get("source_lineage") or {}).get("technical_id") == "plate_track_3"
    )
    assert plate_review["representative_thumbnail_path"]
    assert plate_review["evidence_summary"]["review_observation_count"] > 0
    audit = json.loads((workspace_root / "representative_thumbnail_audit.json").read_text())
    assert any(
        row["status"] == "source_backed" and row["entity_id"] == plate_review["entity_id"]
        for row in audit["cases"]["case"]
    )


def test_unbound_or_invalid_review_tasks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    face_tasks = fixture["case_root"] / "review" / "review_tasks.json"
    payload = json.loads(face_tasks.read_text())
    payload["reviewer_tasks"][0].pop("object_id")
    face_tasks.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unbound_review_task"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    fixture = workspace_fixture(invalid_root)
    face_tasks = fixture["case_root"] / "review" / "review_tasks.json"
    payload = json.loads(face_tasks.read_text())
    payload["reviewer_tasks"][0]["frame_start"] = 1
    payload["reviewer_tasks"][0]["frame_end"] = 0
    face_tasks.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid review task frame span"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


def test_media_hash_and_span_validation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    fixture["combined_preview"]["manifest"].pop("source_video_hash")

    with pytest.raises(ValueError, match="source_video_hash is required"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )

    output_hash_root = tmp_path / "output_hash"
    output_hash_root.mkdir()
    fixture = workspace_fixture(output_hash_root)
    fixture["combined_preview"]["manifest"].pop("output_video_hash")
    with pytest.raises(ValueError, match="output hash is required"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


@pytest.mark.parametrize("bad_count", [None, -1, "two", True])
def test_non_noop_preview_region_count_must_be_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_count: object,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    if bad_count is None:
        fixture["combined_preview"]["manifest"].pop("rendered_region_count")
    else:
        fixture["combined_preview"]["manifest"]["rendered_region_count"] = bad_count

    with pytest.raises(ValueError, match="region"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


def test_zero_region_preview_uses_distinct_noop_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    regions_path = fixture["case_root"] / "render_plan" / "effective_privacy_regions.jsonl"
    regions_path.write_text("", encoding="utf-8")
    fixture["render_plan"]["effective_region_count"] = 0
    write_category_tasks(fixture["case_root"])
    for task_file in (
        fixture["case_root"] / "review" / "review_tasks.json",
        fixture["case_root"] / CATEGORY_PLATE / "review" / "review_tasks.json",
    ):
        payload = json.loads(task_file.read_text())
        if "reviewer_tasks" in payload:
            payload["reviewer_tasks"] = []
        if "tasks" in payload:
            payload["tasks"] = []
        task_file.write_text(json.dumps(payload), encoding="utf-8")

    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=None,
        overwrite=True,
    )

    workspace_root = fixture["case_root"] / result["workspace_dir"]
    manifest = json.loads((workspace_root / "workspace_manifest.json").read_text())
    original = workspace_root / "media" / "case" / "original.mp4"
    automatic = workspace_root / "media" / "case" / "automatic_preview.mp4"
    assert original.exists()
    assert automatic.exists()
    assert original != automatic
    assert manifest["automatic_preview"]["renderer_backend"] == "noop_source_copy"
    assert manifest["automatic_preview"]["region_count"] == 0


@pytest.mark.parametrize("value", ["day", "night", "mixed_case-01", "Case_123"])
def test_case_id_validator_accepts_safe_ids(value: str) -> None:
    assert case_ids.validate_case_id(value) == value


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../case", "/absolute", "C:\\case", "bad case", "case:1", "case.sel"],
)
def test_case_id_validator_rejects_unsafe_ids(value: str) -> None:
    with pytest.raises(case_ids.CaseIdError):
        case_ids.validate_case_id(value)


def test_workspace_case_ids_reject_manifest_traversal(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "cases").mkdir(parents=True)
    (package / "workspace_manifest.json").write_text(
        json.dumps(
            {
                "cases": {"../outside": "cases/outside.json"},
                "schema_version": "final_review_workspace.v2.3.4",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(case_ids.CaseIdError):
        review_workspace.workspace_case_ids(package)


def test_public_case_route_rejects_unsafe_or_unknown_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )
    workspace_root = fixture["case_root"] / result["workspace_dir"]
    server = review_workspace.serve_workspace(workspace_root, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/v1/cases/case", timeout=5) as response:
            assert response.status == 200
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(
                f"{base_url}/api/v1/cases/..%2Foutside",
                timeout=5,
            )
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_mixed_review_workspace_rejects_stale_preview_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    fixture["combined_preview"]["manifest"]["output_video_hash"] = "stale"

    with pytest.raises(ValueError, match="automatic preview output hash mismatch"):
        mixed_review_workspace.build_case_review_workspace(
            fixture["context"],
            category_results=category_results(fixture["case_root"]),
            render_plan=fixture["render_plan"],
            combined_preview=fixture["combined_preview"],
            overwrite=True,
        )


def test_mixed_review_session_persists_face_and_plate_corrections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixed_review_workspace.media, "probe_media", lambda _: fake_media_info())
    fixture = workspace_fixture(tmp_path)
    result = mixed_review_workspace.build_case_review_workspace(
        fixture["context"],
        category_results=category_results(fixture["case_root"]),
        render_plan=fixture["render_plan"],
        combined_preview=fixture["combined_preview"],
        overwrite=True,
    )
    workspace_root = fixture["case_root"] / result["workspace_dir"]
    tasks = json.loads((workspace_root / "reviewer_tasks.json").read_text())["tasks"]
    task_by_category = {task["category"]: task for task in tasks}
    freeze_hash = stable_hash(
        json.loads((workspace_root / "configuration_freeze.json").read_text())
    )
    case_payload = json.loads((workspace_root / "cases" / "case.json").read_text())

    session = review_sessions.create_review_session(
        package_root=workspace_root,
        reviewer=review_sessions.ReviewerIdentity(display_name="Reviewer"),
        source_revision=None,
        configuration_freeze_hash=freeze_hash,
        case_manifest_hashes={"case": stable_hash(case_payload)},
        source_hashes={"case": "source-sha256"},
    )
    session_dir = review_sessions.session_directory(workspace_root, session.session_id)
    actor = {"display_name": "Reviewer"}
    for category, action in (
        (CATEGORY_FACE, review_sessions.ACTION_PRIVACY_LEAK),
        (CATEGORY_PLATE, review_sessions.ACTION_LOOKS_CORRECT),
    ):
        task = task_by_category[category]
        review_sessions.append_decision_event(
            session_dir,
            actor=actor,
            case_id="case",
            task_id=task["review_task_id"],
            action=action,
            note=f"{category} decision",
        )
        correction = ReviewCorrection(
            case_id="case",
            category=category,
            configuration_freeze_hash=freeze_hash,
            entity_id=task["entity_id"],
            interpolation_method=INTERPOLATION_NONE,
            keyframes=(
                CorrectionKeyframe(
                    bbox_xyxy=(10, 10, 30, 30),
                    source_frame=task["source_frame_start"],
                ),
            ),
            operation=OP_ADD,
            reason=f"{category} correction",
            reviewer_identity=actor,
            source_frame_end=task["source_frame_start"],
            source_frame_start=task["source_frame_start"],
            source_video_hash="source-sha256",
            task_id=task["review_task_id"],
        )
        review_sessions.append_correction_event(
            session_dir,
            actor=actor,
            correction=correction,
        )

    assert review_sessions.validate_event_hash_chain(session_dir)["passed"] is True
    events = review_sessions.load_events(session_dir)
    correction_categories = {
        event["after_state"]["category"]
        for event in events
        if event["event_type"] == review_sessions.EVENT_MANUAL_CORRECTION_SAVED
    }
    assert correction_categories == {CATEGORY_FACE, CATEGORY_PLATE}
