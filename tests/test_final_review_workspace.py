from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest
from click.testing import CliRunner

from redactor import (
    case_ids,
    case_manifests,
    privacy_entities,
    privacy_rendering,
    review_api_contract,
    review_corrections,
    review_finalization,
    review_sessions,
    review_working_state,
    review_workspace,
    reviewed_regions,
)
from redactor.cli import main
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_OTHER_SENSITIVE,
    CATEGORY_PLATE,
    MANUAL_REDACTION_CATEGORIES,
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


def get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (
            {
                "allowed_actions": ["looks_correct", "privacy_leak", "not_sensitive"],
                "category": "license_plate",
                "issue_type": "plate_track_review_required",
                "normal_task_class": "plate_track_review_required",
                "title": "Check a low-confidence plate track",
                "what_to_inspect": (
                    "Check the plate-like target and choose Needs correction if any "
                    "sensitive plate remains exposed."
                ),
                "source_task": {"raw_reason_codes": ["insufficient_observations"]},
            },
            "TRACKING_UNCERTAIN",
        ),
        (
            {
                "category": "license_plate",
                "issue_type": "plate_coverage_gap",
                "normal_task_class": "plate_coverage_gap",
                "source_task": {"raw_reason_codes": ["gap_exceeds_safe_interpolation_limit"]},
            },
            "PLATE_COVERAGE_GAP",
        ),
        (
            {
                "category": "face",
                "issue_type": "possible_privacy_gap",
                "normal_task_class": "possible_privacy_gap",
                "source_task": {"raw_reason_codes": ["identity_competition"]},
            },
            "IDENTITY_UNCERTAIN",
        ),
        (
            {
                "category": "face",
                "issue_type": "possible_privacy_gap",
                "normal_task_class": "possible_privacy_gap",
                "source_task": {"raw_reason_codes": ["long_detection_gap"]},
            },
            "INTERPOLATION_OR_SCALE_RISK",
        ),
        (
            {
                "category": "license_plate",
                "issue_type": "possible_false_redaction",
                "normal_task_class": "possible_false_redaction",
            },
            "POSSIBLE_FALSE_POSITIVE",
        ),
        (
            {
                "category": "face",
                "issue_type": "optional_spot_check",
                "normal_task_class": "optional_spot_check",
            },
            "OPTIONAL_SPOT_CHECK",
        ),
        (
            {
                "category": "face",
                "correction_id": "corr_123",
                "issue_type": "correction_verification",
                "normal_task_class": "correction_verification",
            },
            "VERIFY_SAVED_CORRECTION",
        ),
        (
            {
                "category": "face",
                "issue_type": "legacy",
                "normal_task_class": "legacy",
                "query_presentation_class": "FACE_COVERAGE_GAP",
            },
            "FACE_COVERAGE_GAP",
        ),
    ],
)
def test_query_presentation_class_uses_authoritative_semantics(
    task: dict[str, object], expected: str
) -> None:
    assert review_api_contract.query_presentation_class(task) == expected


def test_query_presentation_class_does_not_use_needs_correction_copy() -> None:
    task = {
        "allowed_actions": ["privacy_leak"],
        "category": "license_plate",
        "issue_type": "plate_track_review_required",
        "normal_task_class": "plate_track_review_required",
        "title": "Check a low-confidence plate track",
        "what_happened": "A reviewer can choose Needs correction from the buttons.",
        "what_to_inspect": "Choose Needs correction if any plate remains exposed.",
    }

    assert review_api_contract.query_presentation_class(task) == "TRACKING_UNCERTAIN"
    assert review_api_contract.query_presentation_class(task) != "VERIFY_SAVED_CORRECTION"


def get_json_error(url: str) -> tuple[int, dict[str, object]]:
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url, timeout=5)
    error = exc_info.value
    return error.code, json.loads(error.read().decode("utf-8"))


def synthetic_case_manifest_for(
    package_root: Path,
    case_id: str,
    *,
    display_name: str | None = None,
) -> case_manifests.CaseManifest:
    source = package_root / "media" / case_id / "original_reference.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_video(source)
    manifest = case_manifests.case_manifest_from_media(
        case_id=case_id,
        display_name=display_name or case_id.title(),
        source_path=f"media/{case_id}/original_reference.mp4",
        source_video=source,
        video_roles=[
            case_manifests.VideoRole(
                label="Original",
                path=f"media/{case_id}/original_reference.mp4",
                role="original",
                sha256=None,
                source_frame_end=4,
                source_frame_start=0,
            )
        ],
    )
    case_manifests.write_case_manifest(package_root / "cases" / f"{case_id}.json", manifest)
    return manifest


def synthetic_case_manifest(package_root: Path) -> case_manifests.CaseManifest:
    return synthetic_case_manifest_for(package_root, "synthetic", display_name="Synthetic")


def add_automatic_role(
    package_root: Path,
    manifest: case_manifests.CaseManifest,
) -> case_manifests.CaseManifest:
    source = package_root / manifest.source_path
    automatic = package_root / "media" / manifest.case_id / "automatic_preview.mp4"
    automatic.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, automatic)
    roles = [
        *manifest.video_roles,
        case_manifests.VideoRole(
            label="Automatic preview",
            path=f"media/{manifest.case_id}/automatic_preview.mp4",
            role="automatic",
            sha256=None,
            source_frame_end=manifest.source_frame_end,
            source_frame_start=manifest.source_frame_start,
        ),
    ]
    updated = replace(manifest, video_roles=tuple(roles))
    case_manifests.write_case_manifest(
        package_root / "cases" / f"{manifest.case_id}.json",
        updated,
    )
    return updated


def write_deferred_canonical_workspace_manifest(
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    *,
    categories: tuple[str, ...],
) -> None:
    automatic_role = next(role for role in manifest.video_roles if role.role == "automatic")
    write_json(
        package_root / "workspace_manifest.json",
        {
            "automatic_preview": {
                "automatic_preview_sha256": media_sha(package_root / automatic_role.path),
                "region_count": 1,
                "source_sha256": manifest.source_video_hash,
            },
            "case_id": manifest.case_id,
            "cases": {manifest.case_id: f"cases/{manifest.case_id}.json"},
            "latest_reviewed_publication": {
                "error_code": "latest_reviewed_publication_deferred",
                "reason": "mixed_reviewed_region_compilation_not_implemented",
                "schema_version": "latest_reviewed_publication_capability.v1",
                "status": "deferred",
            },
            "lifecycle": {
                "evaluation": {"status": "not_run"},
                "finalization": {"status": "not_run"},
                "latest_reviewed": {"status": "not_run"},
                "review_session": {"status": "not_started"},
                "review_workspace": "ready",
            },
            "schema_version": "final_review_workspace.v2.3.4",
            "selected_categories": list(categories),
            "workspace_kind": "unified_case_review_workspace",
            "workspace_status": "ready",
        },
    )


def write_supported_canonical_workspace_manifest(
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    *,
    categories: tuple[str, ...],
) -> None:
    automatic_role = next(role for role in manifest.video_roles if role.role == "automatic")
    baseline_path = package_root / "regions" / f"{manifest.case_id}_current_regions.jsonl"
    region_count = sum(
        1 for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    category_run_identities = {
        category: write_test_category_run_manifest(package_root, manifest, category)
        for category in sorted(categories)
    }
    configuration_freeze = {
        "case_id": manifest.case_id,
        "category_configuration_identities": {
            category: {
                "category": category,
                "configuration_hash": identity["configuration_hash"],
                "schema_version": "category_configuration_identity.v1",
                "source_sha256": manifest.source_video_hash,
            }
            for category, identity in category_run_identities.items()
        },
        "category_run_manifest_identity_hash": stable_hash(category_run_identities),
        "category_run_manifest_identities": category_run_identities,
        "schema_version": "mixed_review_configuration_freeze.v1",
        "selected_categories": list(categories),
    }
    write_json(package_root / "configuration_freeze.json", configuration_freeze)
    write_json(
        package_root / "workspace_manifest.json",
        {
            "automatic_preview": {
                "automatic_preview_sha256": media_sha(package_root / automatic_role.path),
                "region_count": region_count,
                "source_sha256": manifest.source_video_hash,
            },
            "case_id": manifest.case_id,
            "cases": {manifest.case_id: f"cases/{manifest.case_id}.json"},
            "category_run_manifest_identities": category_run_identities,
            "category_run_manifest_identity_hash": stable_hash(category_run_identities),
            "configuration_freeze": "configuration_freeze.json",
            "configuration_freeze_hash": stable_hash(configuration_freeze),
            "content_identities": {
                "automatic_baseline_regions": {
                    "path": baseline_path.relative_to(package_root).as_posix(),
                    "role": "automatic_baseline_regions",
                    "schema_version": "artifact_reference.v1",
                    "sha256": media_sha(baseline_path),
                    "size_bytes": baseline_path.stat().st_size,
                }
            },
            "latest_reviewed_publication": {
                "publication_mode": "reviewed_region_compiler.v1",
                "schema_version": "latest_reviewed_publication_capability.v1",
                "status": "supported",
            },
            "lifecycle": {
                "evaluation": {"status": "not_run"},
                "finalization": {"status": "not_run"},
                "latest_reviewed": {"status": "not_run"},
                "review_session": {"status": "not_started"},
                "review_workspace": "ready",
            },
            "schema_version": "final_review_workspace.v2.3.4",
            "selected_categories": list(categories),
            "source_identity": {"sha256": manifest.source_video_hash},
            "source_metadata": {"sha256": manifest.source_video_hash},
            "workspace_kind": "unified_case_review_workspace",
            "workspace_status": "ready",
        },
    )


def write_test_category_run_manifest(
    package_root: Path,
    manifest: case_manifests.CaseManifest,
    category: str,
) -> dict[str, object]:
    configuration_identity = {
        "category": category,
        "configuration_hash": stable_hash(
            {
                "category": category,
                "schema_version": "test_category_configuration.v1",
                "source_sha256": manifest.source_video_hash,
            }
        ),
        "schema_version": "category_configuration_identity.v1",
        "source_sha256": manifest.source_video_hash,
    }
    run_manifest = {
        "artifacts": [],
        "blockers": [],
        "case_id": manifest.case_id,
        "category": category,
        "configuration_identity": configuration_identity,
        "schema_version": "category_run_manifest.v1",
        "source_lineage": {
            "processed_span": {
                "first_processed_frame": manifest.source_frame_start,
                "last_processed_frame": manifest.source_frame_end,
                "source_frame_count": manifest.frame_count,
            },
            "schema_version": "category_source_lineage.v1",
            "source_sha256": manifest.source_video_hash,
        },
        "stage_outcomes": [{"stage": "test", "status": "succeeded"}],
    }
    run_manifest["category_run_content_sha256"] = stable_hash(run_manifest)
    path = (
        package_root
        / "category_runs"
        / manifest.case_id
        / category
        / "category_run_manifest.json"
    )
    write_json(path, run_manifest)
    return {
        "artifact_id": f"{manifest.case_id}:{category}:category_run_manifest",
        "artifact_reference_schema_version": "artifact_reference.v1",
        "case_id": manifest.case_id,
        "category": category,
        "category_run_content_sha256": run_manifest["category_run_content_sha256"],
        "configuration_hash": configuration_identity["configuration_hash"],
        "configuration_sha256": configuration_identity["configuration_hash"],
        "path": path.relative_to(package_root).as_posix(),
        "processed_span": run_manifest["source_lineage"]["processed_span"],
        "role": "category_run_manifest",
        "schema_version": "category_run_manifest_identity.v1",
        "sha256": media_sha(path),
        "size_bytes": path.stat().st_size,
        "source_sha256": manifest.source_video_hash,
    }


def media_sha(path: Path) -> str:
    from redactor import media

    return media.sha256_file(path)


def write_supported_multi_case_workspace_manifest(
    package_root: Path,
    manifests: tuple[case_manifests.CaseManifest, ...],
    *,
    categories: tuple[str, ...],
) -> None:
    cases = {manifest.case_id: f"cases/{manifest.case_id}.json" for manifest in manifests}
    automatic_preview_by_case: dict[str, dict[str, object]] = {}
    category_identities_by_case: dict[str, dict[str, object]] = {}
    category_identity_hash_by_case: dict[str, str] = {}
    category_configuration_by_case: dict[str, dict[str, object]] = {}
    content_identities_by_case: dict[str, dict[str, object]] = {}
    source_identity_by_case: dict[str, dict[str, object]] = {}
    source_metadata_by_case: dict[str, dict[str, object]] = {}
    for manifest in manifests:
        automatic_role = next(role for role in manifest.video_roles if role.role == "automatic")
        baseline_path = package_root / "regions" / f"{manifest.case_id}_current_regions.jsonl"
        region_count = sum(
            1 for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        category_identities = {
            category: write_test_category_run_manifest(package_root, manifest, category)
            for category in sorted(categories)
        }
        category_identities_by_case[manifest.case_id] = category_identities
        category_identity_hash_by_case[manifest.case_id] = stable_hash(category_identities)
        category_configuration_by_case[manifest.case_id] = {
            category: {
                "category": category,
                "configuration_hash": identity["configuration_hash"],
                "schema_version": "category_configuration_identity.v1",
                "source_sha256": manifest.source_video_hash,
            }
            for category, identity in category_identities.items()
        }
        automatic_preview_by_case[manifest.case_id] = {
            "automatic_preview_sha256": media_sha(package_root / automatic_role.path),
            "region_count": region_count,
            "source_sha256": manifest.source_video_hash,
        }
        content_identities_by_case[manifest.case_id] = {
            "automatic_baseline_regions": {
                "path": baseline_path.relative_to(package_root).as_posix(),
                "role": "automatic_baseline_regions",
                "schema_version": "artifact_reference.v1",
                "sha256": media_sha(baseline_path),
                "size_bytes": baseline_path.stat().st_size,
            }
        }
        source_identity_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
        source_metadata_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
    configuration_freeze = {
        "category_configuration_identities_by_case": category_configuration_by_case,
        "category_run_manifest_identities_by_case": category_identities_by_case,
        "schema_version": "mixed_review_configuration_freeze.v1",
        "selected_categories": list(categories),
    }
    write_json(package_root / "configuration_freeze.json", configuration_freeze)
    write_json(
        package_root / "workspace_manifest.json",
        {
            "automatic_preview_by_case": automatic_preview_by_case,
            "cases": cases,
            "category_configuration_identities_by_case": category_configuration_by_case,
            "category_run_manifest_identities_by_case": category_identities_by_case,
            "category_run_manifest_identity_hash_by_case": category_identity_hash_by_case,
            "configuration_freeze": "configuration_freeze.json",
            "configuration_freeze_hash": stable_hash(configuration_freeze),
            "content_identities_by_case": content_identities_by_case,
            "latest_reviewed_publication": {
                "publication_mode": "reviewed_region_compiler.v1",
                "schema_version": "latest_reviewed_publication_capability.v1",
                "status": "supported",
            },
            "schema_version": "final_review_workspace.v2.3.4",
            "selected_categories": list(categories),
            "source_identity_by_case": source_identity_by_case,
            "source_metadata_by_case": source_metadata_by_case,
            "workspace_kind": "unified_case_review_workspace",
            "workspace_status": "ready",
        },
    )


def write_multi_case_review_entity_shell(
    package_root: Path,
    case_ids_to_write: tuple[str, ...],
) -> None:
    shell = {
        "cases": {case_id: [] for case_id in case_ids_to_write},
        "schema_version": "review_entities.v2",
        "unresolved_evidence_by_case": {
            case_id: {
                "fragment_count": 0,
                "fragments": [],
                "schema_version": "review_unresolved_evidence.v1",
            }
            for case_id in case_ids_to_write
        },
    }
    write_json(package_root / "review_entities.json", shell)
    write_json(package_root / "review_objects.json", shell)
    write_json(package_root / "privacy_entities.json", shell)
    for case_id in case_ids_to_write:
        write_json(
            package_root / "entities" / f"{case_id}_unresolved_evidence.json",
            shell["unresolved_evidence_by_case"][case_id],
        )


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
    freeze_path = package_root / "configuration_freeze.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    else:
        freeze = {"schema_version": "freeze.test", "value": 1}
        write_json(freeze_path, freeze)
    session = review_sessions.create_review_session(
        package_root=package_root,
        reviewer=review_sessions.ReviewerIdentity("Analyst", "analyst@example.test"),
        source_revision="test",
        configuration_freeze_hash=stable_hash(freeze),
        case_manifest_hashes={manifest.case_id: manifest.manifest_hash()},
        source_hashes={manifest.case_id: manifest.source_video_hash or ""},
        workspace_identity_hashes=review_workspace.workspace_identity_hashes(package_root),
    )
    return review_sessions.session_directory(package_root, session.session_id)


def create_multi_case_session(
    package_root: Path,
    manifests: tuple[case_manifests.CaseManifest, ...],
) -> Path:
    freeze = json.loads((package_root / "configuration_freeze.json").read_text(encoding="utf-8"))
    session = review_sessions.create_review_session(
        package_root=package_root,
        reviewer=review_sessions.ReviewerIdentity("Analyst", "analyst@example.test"),
        source_revision="test",
        configuration_freeze_hash=stable_hash(freeze),
        case_manifest_hashes={
            manifest.case_id: manifest.manifest_hash() for manifest in manifests
        },
        source_hashes={
            manifest.case_id: manifest.source_video_hash or "" for manifest in manifests
        },
        workspace_identity_hashes=review_workspace.workspace_identity_hashes(package_root),
    )
    return review_sessions.session_directory(package_root, session.session_id)


def write_duplicate_task_case_package(
    tmp_path: Path,
) -> tuple[Path, case_manifests.CaseManifest, case_manifests.CaseManifest, Path]:
    package = tmp_path / "package"
    day = add_automatic_role(package, synthetic_case_manifest_for(package, "day"))
    night = add_automatic_role(package, synthetic_case_manifest_for(package, "night"))
    write_canonical_base_regions(package, "day", (CATEGORY_PLATE,))
    write_canonical_base_regions(package, "night", (CATEGORY_PLATE,))
    write_multi_case_review_entity_shell(package, ("day", "night"))
    write_supported_multi_case_workspace_manifest(
        package,
        (day, night),
        categories=(CATEGORY_PLATE,),
    )
    shared_task = {
        "allowed_actions": list(review_sessions.CANONICAL_REVIEW_ACTIONS),
        "blocking": True,
        "category": CATEGORY_PLATE,
        "correction_required": False,
        "review_task_id": "shared_task_001",
        "source_frame_end": 0,
        "source_frame_start": 0,
        "target_binding": {
            "kind": "entity",
            "technical_track_ids": ["plate_track_1"],
        },
        "technical_only": False,
        "technical_track_ids": ["plate_track_1"],
    }
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                dict(shared_task) | {"case_id": "day"},
                dict(shared_task) | {"case_id": "night"},
                dict(shared_task) | {
                    "case_id": "night",
                    "review_task_id": "night_only_task",
                },
            ]
        },
    )
    session_dir = create_multi_case_session(package, (day, night))
    return package, day, night, session_dir


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
    with pytest.raises(ValueError, match="start keyframe at source frame 10"):
        review_corrections.ReviewCorrection(
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


def test_correction_linear_preview_contract_requires_exact_endpoint_keyframes() -> None:
    with pytest.raises(ValueError, match="end keyframe at source frame 1848"):
        review_corrections.ReviewCorrection(
            case_id="day",
            category=CATEGORY_OTHER_SENSITIVE,
            configuration_freeze_hash="hash",
            interpolation_method=review_corrections.INTERPOLATION_LINEAR,
            keyframes=(review_corrections.CorrectionKeyframe(1828, (10, 10, 30, 20)),),
            operation=review_corrections.OP_ADD,
            reason="observed invalid interpolation draft",
            reviewer_identity={"reviewer_id": "analyst"},
            source_frame_end=1848,
            source_frame_start=1828,
            source_video_hash="source",
            task_id="task",
        )


@pytest.mark.parametrize(
    (
        "name",
        "category",
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
            "face_single",
            CATEGORY_FACE,
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            ((0, (10, 10, 70, 50)),),
            [0],
            0,
            (10, 10, 70, 50),
        ),
        (
            "plate_single",
            CATEGORY_PLATE,
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            ((0, (10, 10, 70, 50)),),
            [0],
            0,
            (10, 10, 70, 50),
        ),
        (
            "other_single",
            CATEGORY_OTHER_SENSITIVE,
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            ((0, (10, 10, 70, 50)),),
            [0],
            0,
            (10, 10, 70, 50),
        ),
        (
            "other_fixed",
            CATEGORY_OTHER_SENSITIVE,
            1,
            3,
            review_corrections.INTERPOLATION_NONE,
            ((1, (10, 10, 70, 50)),),
            [1, 2, 3],
            2,
            (10, 10, 70, 50),
        ),
        (
            "other_interpolate",
            CATEGORY_OTHER_SENSITIVE,
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
    category: str,
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
        category=category,
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
    assert {record.category for record in correction_regions} == {category}
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


def write_mixed_base_regions(package: Path, case_id: str) -> list[PrivacyRegionRecord]:
    records = [
        PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(20, 20, 54, 34),
            category=CATEGORY_FACE,
            object_id="face_track_1",
            provenance="face_detector_observation",
            render_eligible=True,
            source_frame=0,
            source_record={"region_id": "face_region_1", "track_uid": "face_track_1"},
        ),
        PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(20, 20, 54, 34),
            category=CATEGORY_PLATE,
            object_id="plate_track_1",
            provenance="plate_detector_observation",
            render_eligible=True,
            source_frame=0,
            source_record={"region_id": "plate_region_1", "track_uid": "plate_track_1"},
        ),
    ]
    review_finalization.write_privacy_region_records(
        package / "regions" / f"{case_id}_current_regions.jsonl",
        records,
    )
    return records


def write_mixed_entity_registry(
    package: Path,
    case_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    face = {
        "automatic_source_track_ids": ["face_track_1"],
        "case_id": case_id,
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "default_privacy_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_face_1",
        "label": "Person 01",
        "member_detector_tracker_ids": ["face_track_1"],
        "privacy_action": privacy_entities.POLICY_REDACT,
    }
    plate = {
        "automatic_source_track_ids": ["plate_track_1"],
        "case_id": case_id,
        "category": CATEGORY_PLATE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "default_privacy_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_plate_1",
        "label": "Plate 01",
        "member_detector_tracker_ids": ["plate_track_1"],
        "privacy_action": privacy_entities.POLICY_REDACT,
    }
    write_json(
        package / "review_entities.json",
        {"cases": {case_id: [face, plate]}, "schema_version": "review_entities.v2"},
    )
    return face, plate


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


def write_canonical_base_regions(
    package: Path,
    case_id: str,
    categories: tuple[str, ...],
) -> None:
    records: list[PrivacyRegionRecord] = []
    if CATEGORY_FACE in categories:
        records.append(
            PrivacyRegionRecord(
                approval_state=APPROVAL_ACCEPTED,
                bbox_xyxy=(18, 18, 50, 36),
                category=CATEGORY_FACE,
                object_id="face_track_1",
                provenance="face_detector_observation",
                render_eligible=True,
                source_frame=0,
                source_record={"region_id": "face_region_1", "track_uid": "face_track_1"},
            )
        )
    if CATEGORY_PLATE in categories:
        records.append(
            PrivacyRegionRecord(
                approval_state=APPROVAL_ACCEPTED,
                bbox_xyxy=(20, 20, 54, 34),
                category=CATEGORY_PLATE,
                object_id="plate_track_1",
                provenance="plate_detector_observation",
                render_eligible=True,
                source_frame=0,
                source_record={"region_id": "plate_region_1", "track_uid": "plate_track_1"},
            )
        )
    review_finalization.write_privacy_region_records(
        package / "regions" / f"{case_id}_current_regions.jsonl",
        records,
    )
    write_review_entity_shell(package, case_id)


def acknowledge_current_latest_reviewed(
    package: Path,
    session_dir: Path,
    case_id: str,
) -> dict[str, object]:
    binding = review_finalization.sweep_acknowledgement_binding(
        package_root=package,
        session_dir=session_dir,
        case_id=case_id,
    )
    return review_sessions.append_sweep_acknowledgement(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=case_id,
        viewed_ranges=[{"start": 0, "end": 4}],
        binding=binding,
    )


def publish_latest_and_acknowledge(
    package: Path,
    session_dir: Path,
    case_id: str,
) -> dict[str, object]:
    state = review_working_state.derive_working_state(package, session_dir)
    materialization = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=case_id,
        accepted_revision=state["active_projection_hash"],
    )
    acknowledge_current_latest_reviewed(package, session_dir, case_id)
    return materialization


def rewrite_latest_reviewed_pointer_as_legacy_projection_v1(
    package: Path,
    session_dir: Path,
    case_id: str,
) -> str:
    projection = review_working_state.derive_active_review_projection(
        package_root=package,
        session_dir=session_dir,
        entities=review_working_state.load_entity_index(package),
    )
    legacy_hash = review_working_state.active_projection_hash(
        projection,
        schema_version=review_working_state.LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION,
    )
    pointer_path = package / "latest_reviewed" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    revision_manifest_path = package / pointer["revision_manifest"]
    revision_manifest = json.loads(revision_manifest_path.read_text(encoding="utf-8"))
    publication_guard = dict(revision_manifest.get("publication_guard") or {})
    for payload in (revision_manifest, publication_guard):
        payload["accepted_working_state_revision"] = legacy_hash
        payload["working_state_projection_hash"] = legacy_hash
        payload.pop("active_projection_hash_schema_version", None)
        payload.pop("working_state_projection_schema_version", None)
        payload.pop("projection_hash_schema_version", None)
    revision_manifest["publication_guard"] = publication_guard
    revision_manifest["accepted_working_state_revision"] = legacy_hash
    revision_manifest.pop("revision_manifest_sha256", None)
    revision_manifest["revision_manifest_sha256"] = stable_hash(revision_manifest)
    write_json(revision_manifest_path, revision_manifest)
    revision_manifest_artifact_sha256 = media_sha(revision_manifest_path)

    pointer["accepted_working_state_revision"] = legacy_hash
    pointer["working_state_generation"] = legacy_hash
    pointer["working_state_projection_hash"] = legacy_hash
    pointer["publication_guard"] = publication_guard
    pointer["revision_manifest_artifact_sha256"] = revision_manifest_artifact_sha256
    pointer["revision_manifest_sha256"] = revision_manifest["revision_manifest_sha256"]
    pointer.pop("active_projection_hash_schema_version", None)
    pointer.pop("working_state_projection_schema_version", None)
    pointer["content_identities"]["latest_reviewed_revision_manifest"][
        "sha256"
    ] = revision_manifest_artifact_sha256
    latest = dict(pointer.get("latest_reviewed") or {})
    latest["accepted_working_state_revision"] = legacy_hash
    latest["published_working_state_revision"] = legacy_hash
    pointer["latest_reviewed"] = latest
    write_json(pointer_path, pointer)
    return legacy_hash


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
    server.materialization_jobs.debounce_seconds = 60
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


def test_manual_redaction_category_contract_is_shared() -> None:
    backend_categories = list(MANUAL_REDACTION_CATEGORIES)
    generated_schema = review_api_contract.final_review_workspace_schema()
    checked_in_schema = json.loads(
        Path("docs/api/final_review_workspace_v1.schema.json").read_text(encoding="utf-8")
    )
    frontend_domain = Path("frontend/src/types/domain.ts").read_text(encoding="utf-8")
    frontend_api_types = Path("frontend/src/types/api.ts").read_text(encoding="utf-8")
    frontend_match = re.search(
        r"MANUAL_REDACTION_CATEGORIES\s*=\s*\[(?P<values>[^\]]+)\]",
        frontend_domain,
    )

    assert frontend_match is not None
    frontend_categories = re.findall(r"'([^']+)'", frontend_match.group("values"))
    assert generated_schema["x-manual-redaction-categories"] == backend_categories
    assert generated_schema["definitions"]["ManualPrivacyCategory"]["enum"] == backend_categories
    assert checked_in_schema["definitions"]["ManualPrivacyCategory"]["enum"] == backend_categories
    assert frontend_categories == backend_categories
    assert (
        "export type ManualPrivacyCategory = "
        + " | ".join(json.dumps(item) for item in backend_categories)
    ) in frontend_api_types


def test_manual_only_category_is_not_required_as_detector_run(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    correction = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_OTHER_SENSITIVE,
        configuration_freeze_hash="freeze",
        interpolation_method=review_corrections.INTERPOLATION_NONE,
        keyframes=(review_corrections.CorrectionKeyframe(0, (20, 20, 54, 34)),),
        operation=review_corrections.OP_ADD,
        reason="manual-only sensitive region",
        reviewer_identity={"reviewer_id": "analyst@example.test"},
        source_frame_end=0,
        source_frame_start=0,
        source_video_hash=manifest.source_video_hash or "",
        task_id="manual_other_sensitive",
    )

    records, provenance = reviewed_regions.apply_review_corrections(
        [],
        [correction],
        manifest=manifest,
        selected_categories=[CATEGORY_FACE, CATEGORY_PLATE],
        effective_entities=[],
    )

    assert [record.category for record in records] == [CATEGORY_OTHER_SENSITIVE]
    assert records[0].source_record["reviewed_region_kind"] == "manual_correction"
    assert provenance["added_region_count"] == 1


def test_manual_only_category_preview_save_ignores_detector_run_selection(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
    )
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session_response = post_json(
            f"{base_url}/api/v1/sessions",
            {
                "reviewer": {"display_name": "Analyst", "reviewer_id": "analyst@example.test"},
                "source_revision": "test",
            },
        )
        session = session_response["session"]
        correction = {
            "case_id": manifest.case_id,
            "category": CATEGORY_OTHER_SENSITIVE,
            "configuration_freeze_hash": session["configuration_freeze_hash"],
            "interpolation_method": review_corrections.INTERPOLATION_NONE,
            "keyframe_boxes": [{"bbox_xyxy": [12, 12, 44, 30], "source_frame": 1}],
            "operation": review_corrections.OP_ADD,
            "reason": "manual-only category preview",
            "reviewer_identity": {"reviewer_id": "analyst@example.test"},
            "source_frame_end": 1,
            "source_frame_start": 1,
            "source_video_hash": manifest.source_video_hash,
            "task_id": "manual_other_sensitive",
        }

        preview_response = post_json(
            f"{base_url}/api/v1/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "manual-only category preview",
                "session_id": session["session_id"],
            },
        )
        correction_response = post_json(
            f"{base_url}/api/v1/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "manual-only category save",
                "session_id": session["session_id"],
            },
        )
        effective_regions, provenance = review_working_state.compute_effective_privacy_regions(
            package,
            package / "sessions" / session["session_id"],
            case_id=manifest.case_id,
        )
        manual_regions = [
            record for record in effective_regions if record.provenance == "reviewed_correction"
        ]

        assert preview_response["ok"] is True
        assert correction_response["ok"] is True
        assert correction_response["event"]["after_state"]["category"] == CATEGORY_OTHER_SENSITIVE
        assert [record.category for record in manual_regions] == [CATEGORY_OTHER_SENSITIVE]
        assert provenance["reviewed_region_identity"]["selected_categories"] == [
            CATEGORY_FACE,
            CATEGORY_PLATE,
        ]
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
    assert after["region_count"] == end - start + 1
    final_regions, _ = review_working_state.compute_effective_privacy_regions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    correction_source_frames = sorted(
        int(record.source_record["source_frame"])
        for record in final_regions
        if record.object_id == correction.correction_id
    )
    assert correction_source_frames == list(range(start, end + 1))
    assert start - 1 not in correction_source_frames
    assert end + 1 not in correction_source_frames
    assert summary["changed_source_frame_percentage"] == pytest.approx(
        (end - start + 1) / manifest.frame_count
    )


def test_fixed_box_long_day_span_expands_to_544_effective_frames(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    manifest = replace(
        manifest,
        frame_count=673,
        source_frame_start=1828,
        source_frame_end=2500,
        video_roles=tuple(
            replace(role, source_frame_start=1828, source_frame_end=2500)
            for role in manifest.video_roles
        ),
    )
    case_manifests.write_case_manifest(package / "cases" / "synthetic.json", manifest)
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
        category=CATEGORY_FACE,
        correction_id="corr_long_fixed",
        interpolation_method=review_corrections.INTERPOLATION_NONE,
        keyframes=(review_corrections.CorrectionKeyframe(1957, (10, 10, 40, 40)),),
        operation=review_corrections.OP_ADD,
        reason="long fixed correction",
        reviewer_identity={"reviewer_id": "analyst"},
        configuration_freeze_hash="test_freeze",
        source_frame_end=2500,
        source_frame_start=1957,
        source_video_hash=manifest.source_video_hash or "",
        task_id="task_long_fixed",
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
    final_regions, _ = review_working_state.compute_effective_privacy_regions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    correction_source_frames = sorted(
        int(record.source_record["source_frame"])
        for record in final_regions
        if record.object_id == correction.correction_id
    )

    assert len(correction_source_frames) == 544
    assert correction_source_frames[0] == 1957
    assert correction_source_frames[-1] == 2500
    assert 1956 not in correction_source_frames
    assert 2501 not in correction_source_frames
    assert after["region_count"] == 544
    assert summary["changed_source_frame_count"] == 544
    assert summary["changed_source_frame_intervals"] == [
        {"source_frame_start": 1957, "source_frame_end": 2500}
    ]


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


def test_workspace_server_shutdown_cancels_pending_materialization_timer(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    timer: review_workspace.threading.Timer | None = None
    try:
        post_json(
            f"{base_url}/api/v1/object-policy",
            {
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "entity_id": entity["entity_id"],
                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                "session_id": session_dir.name,
            },
        )
        state = server.materialization_jobs.case_state[(session_dir.name, manifest.case_id)]
        timer = state["debounce_timer"]
        assert isinstance(timer, review_workspace.threading.Timer)
        assert timer.is_alive()
    finally:
        server.shutdown()
        server.server_close()

    state = server.materialization_jobs.case_state[(session_dir.name, manifest.case_id)]
    assert state["debounce_timer"] is None
    assert timer is not None
    assert not timer.is_alive()


def test_review_decision_does_not_start_materialization_job(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, _entity = write_policy_entity_package(package)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": False,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "correction_required": False,
                    "review_task_id": "task_1",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "technical_only": False,
                }
            ]
        },
    )
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


def test_legacy_only_task_actions_remain_unresolved_for_compatibility() -> None:
    task = {
        "allowed_actions": [review_sessions.ACTION_ESCALATE],
        "review_task_id": "legacy_task",
    }

    assert review_sessions.allowed_actions_for_task(task) == ()
    assert (
        review_sessions.invalid_task_action_reason(
            task,
            review_sessions.ACTION_LOOKS_CORRECT,
        )
        == "legacy_task: review task has no allowed decision actions"
    )
    assert review_sessions.invalid_task_decision_reason(
        task,
        {"action": review_sessions.ACTION_ESCALATE},
    ) == "unsupported decision action: uncertain_escalate"


def test_review_decision_api_enforces_authoritative_task_allowed_actions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    write_json(package / "workspace_manifest.json", {"case_count": 1})
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_FACE,
                    "correction_required": True,
                    "entity_id": None,
                    "review_task_id": "source_span_task",
                    "source_frame_end": 2,
                    "source_frame_start": 1,
                    "target_binding": {"kind": "source_span"},
                    "target_entity_ids": [],
                    "technical_only": False,
                    "technical_track_ids": [],
                },
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": False,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "correction_required": False,
                    "entity_id": "entity_plate_1",
                    "review_task_id": "entity_task",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "target_binding": {"kind": "entity"},
                    "technical_only": False,
                },
            ]
        },
    )
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session_response = post_json(
            f"{base_url}/api/v1/sessions",
            {
                "reviewer": {"display_name": "Analyst", "reviewer_id": "analyst@example.test"},
                "source_revision": "test",
            },
        )
        session_id = str(session_response["session"]["session_id"])

        status, body = post_json_error(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_ESCALATE,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "legacy escalation action",
                "session_id": session_id,
                "task_id": "source_span_task",
            },
        )
        restricted_current = review_sessions.load_current_decisions(
            package / "sessions" / session_id
        )
        legal_privacy = post_json(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_PRIVACY_LEAK,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "needs correction",
                "session_id": session_id,
                "task_id": "source_span_task",
            },
        )
        legal_not_sensitive = post_json(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_NOT_SENSITIVE,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "not sensitive",
                "session_id": session_id,
                "task_id": "source_span_task",
            },
        )
        entity_green = post_json(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "correct",
                "session_id": session_id,
                "task_id": "entity_task",
            },
        )

        assert status == 400
        assert "unsupported decision action" in str(body["error"])
        assert "source_span_task" not in restricted_current
        assert legal_privacy["ok"] is True
        assert legal_not_sensitive["ok"] is True
        assert entity_green["ok"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_region_compatible_job_publishes_after_non_media_decision(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": False,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "correction_required": False,
                    "review_task_id": "task_1",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "technical_only": False,
                }
            ]
        },
    )
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


def test_reset_supersedes_running_materialization_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    manifest, entity = write_policy_entity_package(package)
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 0.01
    entered = review_workspace.threading.Event()
    release = review_workspace.threading.Event()

    def fake_materialize_latest_reviewed(
        package_root: Path,
        job_session_dir: Path,
        *,
        case_id: str,
        accepted_revision: str | None = None,
        actor: dict[str, Any] | None = None,
        event_reason: str | None = None,
    ) -> dict[str, Any]:
        del package_root, job_session_dir, actor, event_reason
        entered.set()
        assert release.wait(5)
        return {
            "accepted_working_state_revision": accepted_revision,
            "case_id": case_id,
            "effective_regions_hash": "old-effective-hash",
            "latest_reviewed": {
                "label": "Latest reviewed",
                "path": "sessions/stale/media/synthetic/latest_reviewed_old.mp4",
                "role": "latest_reviewed",
                "source_frame_end": manifest.source_frame_end,
                "source_frame_start": manifest.source_frame_start,
            },
            "media_validation": {"passed": True},
            "schema_version": "latest_reviewed_materialization.v1",
            "timing": {},
            "video_sha256": "old-video-hash",
        }

    monkeypatch.setattr(
        review_workspace.review_working_state,
        "materialize_latest_reviewed",
        fake_materialize_latest_reviewed,
    )
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
        old_job_id = str(response["materialization_job"]["job_id"])
        assert entered.wait(5)
        server.materialization_jobs.debounce_seconds = 60

        reset = post_json(
            f"{base_url}/api/checkpoints/reset",
            {
                "actor": {"reviewer_id": "analyst"},
                "case_id": manifest.case_id,
                "session_id": session_dir.name,
            },
        )
        release.set()
        old_job = wait_for_materialization_job(server, old_job_id)
        state = review_working_state.derive_working_state(package, session_dir)

        assert reset["ok"] is True
        assert old_job["status"] == "superseded"
        assert old_job["publication_status"] == "superseded"
        assert old_job["queue_state"] == "discarded_stale_generation"
        assert manifest.case_id not in state["materialized_latest_reviewed"]
        assert review_sessions.validate_event_hash_chain(session_dir)["passed"] is True
    finally:
        release.set()
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

    assert "Start new review" in html
    assert "No local review session loaded" in html
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
    assert "Correct" in js
    assert "Needs correction" in js
    assert "Not sensitive" in js
    assert "Completed" not in js
    assert "Unsure / Escalated" not in js
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
    assert ".decision-unsure" not in css
    assert "task.allowed_actions || []" in js
    assert "actionLabel(action)" in js
    assert "actionTone(action)" in js
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
    assert "task.allowed_actions || []" in decision_buttons
    assert "[ACTIONS.looks, ACTIONS.leak, ACTIONS.unsure]" not in decision_buttons
    assert "actionTone(action)" in decision_buttons


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
        assert correction_response["publication_status"] == "queued"
        assert correction_response["target_effective_region_count"] == 1
        assert correction_response["changed_frame_count"] == 1
        assert correction_response["changed_frame_intervals"] == [
            {"source_frame_start": 0, "source_frame_end": 0}
        ]
        verification_task = correction_response["verification_task"]
        assert verification_task["actionable"] is False
        assert verification_task["decision_state"] == "waiting_for_latest_reviewed"

        status, body = post_json_error(
            f"{base_url}/api/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "Correct",
                "session_id": session["session_id"],
                "task_id": verification_task["review_task_id"],
            },
        )

        assert status == 409
        assert "Latest reviewed media" in str(body["error"])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("categories", "correction_category"),
    [
        ((CATEGORY_FACE,), CATEGORY_FACE),
        ((CATEGORY_PLATE,), CATEGORY_PLATE),
        ((CATEGORY_FACE, CATEGORY_PLATE), CATEGORY_PLATE),
    ],
)
def test_deferred_canonical_workspace_persists_correction_without_latest_reviewed(
    tmp_path: Path,
    categories: tuple[str, ...],
    correction_category: str,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_deferred_canonical_workspace_manifest(package, manifest, categories=categories)
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
            "category": correction_category,
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
        correction_response = post_json(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        session_dir = package / "sessions" / session["session_id"]
        state = review_working_state.derive_working_state(package, session_dir)
        with urllib.request.urlopen(
            f"{base_url}/api/v1/cases/{manifest.case_id}/versions?session_id={session['session_id']}",
            timeout=5,
        ) as response:
            versions = json.loads(response.read().decode("utf-8"))["versions"]

        assert preview_response["preview"]["latest_reviewed"] is None
        assert preview_response["preview"]["corrected_preview"]
        assert correction_response["ok"] is True
        assert correction_response["publication_status"] == "deferred"
        assert correction_response["materialization_job"] is None
        assert correction_response["materialization_required"] is False
        assert correction_response["verification_task"]["publication_status"] == "deferred"
        assert correction_response["verification_task"]["actionable"] is True
        assert server.materialization_jobs.jobs == {}
        assert state["active_correction_ids"] == [
            correction_response["event"]["after_state"]["correction_id"]
        ]
        assert state["materialized_latest_reviewed"] == {}
        latest_reviewed_outputs = list(
            (package / "sessions" / session["session_id"]).glob("**/latest_reviewed_*.mp4")
        )
        assert not latest_reviewed_outputs
        assert "latest_reviewed" not in {role["role"] for role in versions}

        status, body = post_json_error(
            f"{base_url}/api/materialization-jobs/retry",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "job_id": "materialize_missing",
                "session_id": session["session_id"],
            },
        )
        assert status == 409
        assert body["error_code"] == "latest_reviewed_publication_deferred"

        finalization = review_finalization.finalization_blockers(
            package_root=package,
            case_id=manifest.case_id,
            session_dir=session_dir,
        )
        assert any(
            "latest_reviewed_publication_deferred" in item
            for item in finalization["blockers"]
        )
        assert (
            json.loads((package / "workspace_manifest.json").read_text())["lifecycle"][
                "latest_reviewed"
            ]["status"]
            == "not_run"
        )
    finally:
        server.shutdown()
        server.server_close()


def test_supported_canonical_compiler_applies_keep_visible_and_replacement(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_mixed_base_regions(package, manifest.case_id)
    face, plate = write_mixed_entity_registry(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
    )
    session_dir = create_session(package, manifest)
    review_sessions.append_event(
        session_dir,
        event_type="entity_policy_changed",
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id=None,
        before_state=plate,
        after_state=plate
        | {
            "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
            "privacy_action": privacy_entities.POLICY_KEEP_VISIBLE,
            "reviewer_override": True,
        },
        note="keep visible",
        application_hashes={},
    )
    correction = review_corrections.ReviewCorrection(
        case_id=manifest.case_id,
        category=CATEGORY_FACE,
        configuration_freeze_hash=stable_hash({"schema_version": "freeze.test", "value": 1}),
        entity_id=str(face["entity_id"]),
        interpolation_method=review_corrections.INTERPOLATION_NONE,
        keyframes=(review_corrections.CorrectionKeyframe(0, (10, 10, 70, 50)),),
        operation=review_corrections.OP_REPLACE,
        reason="replace face geometry",
        reviewer_identity={"reviewer_id": "analyst"},
        source_frame_end=0,
        source_frame_start=0,
        source_video_hash=manifest.source_video_hash or "",
        task_id="task_replace",
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        correction=correction,
    )

    compiled, provenance = reviewed_regions.compile_reviewed_regions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )

    assert [record.category for record in compiled] == [CATEGORY_FACE]
    assert compiled[0].provenance == "reviewed_correction"
    assert compiled[0].bbox_xyxy == (10, 10, 70, 50)
    assert compiled[0].source_record["baseline_replaced_region_ids"] == ["face_region_1"]
    assert provenance["privacy_region_changes_by_cause"]["suppression_removed"] == 1
    assert provenance["privacy_region_changes_by_cause"]["manual_correction_replaced"] == 1
    identity = provenance["reviewed_region_identity"]
    assert identity["selected_categories"] == [CATEGORY_FACE, CATEGORY_PLATE]
    assert identity["reviewed_region_set_sha256"] == provenance["reviewed_region_set_sha256"]
    assert identity["active_corrections_hash"]
    assert identity["object_policy_hash"]


def test_supported_canonical_unresolved_blocking_task_prevents_publication(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "correction_required": False,
                    "review_task_id": "plate_task_1",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "technical_only": False,
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)

    with pytest.raises(reviewed_regions.ReviewedRegionCompilationError, match="required task"):
        review_working_state.materialize_latest_reviewed(
            package,
            session_dir,
            case_id=manifest.case_id,
            accepted_revision=review_working_state.derive_working_state(package, session_dir)[
                "active_projection_hash"
            ],
        )

    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id="plate_task_1",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="resolved",
    )
    materialization = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=review_working_state.derive_working_state(package, session_dir)[
            "active_projection_hash"
        ],
    )

    assert materialization["latest_reviewed"]["revision_id"].startswith("latest_reviewed_")
    assert materialization["media_validation"]["passed"] is True


def test_source_backed_face_entity_regions_match_normalized_track_variants(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_explicit_keep_face_policy(package, manifest, track_id=10)

    regions = review_working_state.source_backed_entity_regions(
        package_root=package,
        manifest=manifest,
        entity={
            "case_id": manifest.case_id,
            "category": CATEGORY_FACE,
            "current_policy": privacy_entities.POLICY_REDACT,
            "entity_id": "entity_face_10",
            "member_detector_tracker_ids": ["face_track_10"],
        },
    )

    assert [record.source_frame for record in regions] == [1, 2, 3]
    assert {record.object_id for record in regions} == {"face_track_10"}
    assert {record.source_record["track_id"] for record in regions} == {"10"}


def test_not_sensitive_task_decision_suppresses_bound_reviewed_regions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_FACE,))
    write_review_entity_shell(package, manifest.case_id)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": list(review_sessions.CANONICAL_REVIEW_ACTIONS),
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_FACE,
                    "correction_required": False,
                    "review_task_id": "face_task_1",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "target_binding": {
                        "kind": "entity",
                        "technical_track_ids": ["face_track_1"],
                    },
                    "technical_only": False,
                    "technical_track_ids": ["face_track_1"],
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE,),
    )
    session_dir = create_session(package, manifest)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="face_task_1",
        action=review_sessions.ACTION_NOT_SENSITIVE,
        note="not sensitive",
    )

    regions, provenance = reviewed_regions.compile_reviewed_regions(
        package,
        session_dir,
        case_id=manifest.case_id,
        require_blockers_resolved=True,
    )

    assert regions == []
    assert provenance["task_not_sensitive_provenance"]["suppressed_region_count"] == 1
    assert provenance["privacy_region_changes_by_cause"]["task_not_sensitive_removed"] == 1


def test_duplicate_task_ids_are_case_scoped_for_api_persistence_and_blockers(
    tmp_path: Path,
) -> None:
    package, _day, _night, session_dir = write_duplicate_task_case_package(tmp_path)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, body = post_json_error(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst"},
                "case_id": "day",
                "note": "wrong case",
                "session_id": session_dir.name,
                "task_id": "night_only_task",
            },
        )
        assert status == 404
        assert body["ok"] is False

        response = post_json(
            f"{base_url}/api/v1/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst"},
                "case_id": "day",
                "note": "day correct",
                "session_id": session_dir.name,
                "task_id": "shared_task_001",
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response["ok"] is True
    decisions_by_case = review_sessions.load_current_decisions_by_case(session_dir)
    assert decisions_by_case["day"]["shared_task_001"]["action"] == (
        review_sessions.ACTION_LOOKS_CORRECT
    )
    assert decisions_by_case.get("night", {}) == {}

    state = review_working_state.derive_working_state(package, session_dir)
    assert state["task_decisions_by_case"]["day"]["shared_task_001"]["action"] == (
        review_sessions.ACTION_LOOKS_CORRECT
    )
    assert state["task_decisions_by_case"].get("night", {}) == {}
    assert not any(
        "shared_task_001" in blocker
        for blocker in reviewed_regions.publication_blockers(
            package_root=package,
            session_dir=session_dir,
            case_id="day",
            state=state,
        )
    )
    night_blockers = reviewed_regions.publication_blockers(
        package_root=package,
        session_dir=session_dir,
        case_id="night",
        state=state,
    )
    assert any("shared_task_001: required task has no decision" in item for item in night_blockers)
    assert any("night_only_task: required task has no decision" in item for item in night_blockers)


def test_duplicate_task_id_not_sensitive_suppression_is_case_and_task_scoped(
    tmp_path: Path,
) -> None:
    package, _day, _night, session_dir = write_duplicate_task_case_package(tmp_path)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id="shared_task_001",
        action=review_sessions.ACTION_NOT_SENSITIVE,
        note="day not sensitive",
    )

    day_regions, day_provenance = reviewed_regions.compile_reviewed_regions(
        package,
        session_dir,
        case_id="day",
        require_blockers_resolved=True,
    )
    night_regions, night_provenance = reviewed_regions.compile_reviewed_regions(
        package,
        session_dir,
        case_id="night",
        require_blockers_resolved=False,
    )

    assert day_regions == []
    assert day_provenance["task_not_sensitive_provenance"]["suppressed_region_count"] == 1
    assert [record.category for record in night_regions] == [CATEGORY_PLATE]
    assert night_provenance["task_not_sensitive_provenance"]["suppressed_region_count"] == 0
    assert any(
        "shared_task_001: required task has no decision" in item
        for item in night_provenance["publication_blockers"]
    )


def test_duplicate_task_id_correction_resolution_is_case_scoped(
    tmp_path: Path,
) -> None:
    package, day, _night, session_dir = write_duplicate_task_case_package(tmp_path)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id="shared_task_001",
        action=review_sessions.ACTION_PRIVACY_LEAK,
        note="needs correction",
    )
    freeze_hash = stable_hash(json.loads((package / "configuration_freeze.json").read_text()))
    correction = review_corrections.ReviewCorrection(
        case_id="day",
        task_id="shared_task_001",
        category=CATEGORY_PLATE,
        operation=review_corrections.OP_ADD,
        source_frame_start=0,
        source_frame_end=0,
        keyframes=(review_corrections.CorrectionKeyframe(0, (10, 10, 30, 20)),),
        interpolation_method="none",
        reviewer_identity={"reviewer_id": "analyst"},
        reason="accepted correction",
        source_video_hash=day.source_video_hash or "",
        configuration_freeze_hash=freeze_hash,
    )
    review_sessions.append_correction_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        correction=correction,
    )
    review_sessions.append_task_resolution_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
        task_id="shared_task_001",
        resolved_by="accepted_correction",
        note="correction accepted",
        resolution_metadata={
            "correction_id": correction.correction_id,
            "correction_revision_id": correction.revision_id,
        },
    )

    state = review_working_state.derive_working_state(package, session_dir)
    assert state["task_decisions_by_case"]["day"]["shared_task_001"]["action"] == (
        review_sessions.ACTION_PRIVACY_LEAK
    )
    assert state["task_resolutions_by_case"]["day"]["shared_task_001"]["resolved_by"] == (
        "accepted_correction"
    )
    assert state["task_resolutions_by_case"].get("night", {}) == {}
    assert review_sessions.refresh_session_summary(session_dir)["decision_count"] == 1
    assert not any(
        "shared_task_001" in blocker
        for blocker in reviewed_regions.publication_blockers(
            package_root=package,
            session_dir=session_dir,
            case_id="day",
            state=state,
        )
    )
    assert any(
        "shared_task_001: required task has no decision" in blocker
        for blocker in reviewed_regions.publication_blockers(
            package_root=package,
            session_dir=session_dir,
            case_id="night",
            state=state,
        )
    )


def test_duplicate_task_id_checkpoint_and_reset_are_case_scoped(
    tmp_path: Path,
) -> None:
    package, _day, _night, session_dir = write_duplicate_task_case_package(tmp_path)
    for case_id, action in (
        ("day", review_sessions.ACTION_LOOKS_CORRECT),
        ("night", review_sessions.ACTION_NOT_SENSITIVE),
    ):
        review_sessions.append_decision_event(
            session_dir,
            actor={"reviewer_id": "analyst"},
            case_id=case_id,
            task_id="shared_task_001",
            action=action,
            note=f"{case_id} decision",
        )
    checkpoint, _checkpoint_event = review_working_state.create_checkpoint(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        name="Both cases",
    )

    _event, restarted = review_working_state.restart_review_workflow(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id="day",
    )

    assert "day" not in restarted["task_decisions_by_case"]
    assert restarted["task_decisions_by_case"]["night"]["shared_task_001"]["action"] == (
        review_sessions.ACTION_NOT_SENSITIVE
    )

    _checkpoint, _restore_event, restored = review_working_state.restore_checkpoint(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        checkpoint_uuid=checkpoint["checkpoint_uuid"],
    )

    assert restored["task_decisions_by_case"]["day"]["shared_task_001"]["action"] == (
        review_sessions.ACTION_LOOKS_CORRECT
    )
    assert restored["task_decisions_by_case"]["night"]["shared_task_001"]["action"] == (
        review_sessions.ACTION_NOT_SENSITIVE
    )


def test_invalid_stored_task_decision_cannot_satisfy_publication_or_finalization(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_FACE,
                    "correction_required": True,
                    "entity_id": None,
                    "review_task_id": "source_span_task",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "target_binding": {"kind": "source_span"},
                    "target_entity_ids": [],
                    "technical_only": False,
                    "technical_track_ids": [],
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
    )
    session_dir = create_session(package, manifest)
    review_sessions.append_event(
        session_dir,
        event_type="decision_recorded",
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id="source_span_task",
        before_state=None,
        after_state={
            "action": review_sessions.ACTION_ESCALATE,
            "case_id": manifest.case_id,
            "decision_status": "blocking",
            "note": "forged historical legacy decision",
            "task_id": "source_span_task",
        },
        note="forged historical green decision",
        application_hashes={},
    )

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="invalid task decision",
    ):
        review_working_state.materialize_latest_reviewed(
            package,
            session_dir,
            case_id=manifest.case_id,
            accepted_revision=review_working_state.derive_working_state(package, session_dir)[
                "active_projection_hash"
            ],
        )
    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert blockers["ready"] is False
    assert any("invalid task decision" in blocker for blocker in blockers["blockers"])


def test_supported_canonical_reviewed_region_hash_uses_semantic_decisions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": False,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "correction_required": False,
                    "review_task_id": "plate_task_1",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "technical_only": False,
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)

    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id="plate_task_1",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="same decision",
    )
    first = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id="plate_task_1",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="same decision",
    )
    second = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id="plate_task_1",
        action=review_sessions.ACTION_NOT_SENSITIVE,
        note="changed decision",
    )
    third = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )

    assert first["effective_regions_hash"] == second["effective_regions_hash"]
    assert third["effective_regions_hash"] != second["effective_regions_hash"]


def test_supported_canonical_correction_semantic_hash_excludes_audit_metadata(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_a = create_session(package, manifest)
    session_b = create_session(package, manifest)
    base_payload = {
        "case_id": manifest.case_id,
        "category": CATEGORY_PLATE,
        "configuration_freeze_hash": json.loads(
            (package / "workspace_manifest.json").read_text()
        )["configuration_freeze_hash"],
        "entity_id": "plate_entity_1",
        "interpolation_method": review_corrections.INTERPOLATION_NONE,
        "keyframes": (review_corrections.CorrectionKeyframe(0, (20, 20, 54, 34)),),
        "operation": review_corrections.OP_ADD,
        "reason": "reviewed",
        "reviewer_identity": {"reviewer_id": "analyst"},
        "source_frame_end": 0,
        "source_frame_start": 0,
        "source_video_hash": manifest.source_video_hash or "",
        "task_id": "task_add",
    }
    first = review_corrections.ReviewCorrection(
        **base_payload,
        correction_id="corr_a",
        created_at_utc="2026-01-01T00:00:00Z",
        revision_id="corr_rev_a",
    )
    second = review_corrections.ReviewCorrection(
        **base_payload,
        correction_id="corr_b",
        created_at_utc="2026-01-02T00:00:00Z",
        revision_id="corr_rev_b",
    )

    assert first.correction_hash() != second.correction_hash()
    assert reviewed_regions.correction_semantic_sha256(first) == (
        reviewed_regions.correction_semantic_sha256(second)
    )

    review_sessions.append_correction_event(
        session_a,
        actor={"reviewer_id": "analyst"},
        correction=first,
    )
    review_sessions.append_correction_event(
        session_b,
        actor={"reviewer_id": "analyst"},
        correction=second,
    )
    digest_a = reviewed_regions.reviewed_region_digest(
        package,
        session_a,
        case_id=manifest.case_id,
    )
    digest_b = reviewed_regions.reviewed_region_digest(
        package,
        session_b,
        case_id=manifest.case_id,
    )
    changed_keyframes = (review_corrections.CorrectionKeyframe(0, (22, 20, 56, 34)),)
    changed = review_corrections.ReviewCorrection(
        **(base_payload | {"keyframes": changed_keyframes}),
        correction_id="corr_c",
        created_at_utc="2026-01-03T00:00:00Z",
        revision_id="corr_rev_c",
    )
    changed_span = review_corrections.ReviewCorrection(
        **(base_payload | {"source_frame_end": 1}),
        correction_id="corr_span",
        created_at_utc="2026-01-04T00:00:00Z",
        revision_id="corr_rev_span",
    )
    fixed_multi_frame = review_corrections.ReviewCorrection(
        **(
            base_payload
            | {
                "keyframes": (
                    review_corrections.CorrectionKeyframe(0, (20, 20, 54, 34)),
                    review_corrections.CorrectionKeyframe(2, (30, 22, 70, 44)),
                ),
                "source_frame_end": 2,
            }
        ),
        correction_id="corr_fixed",
        created_at_utc="2026-01-05T00:00:00Z",
        revision_id="corr_rev_fixed",
    )
    linear_multi_frame = review_corrections.ReviewCorrection(
        **(
            base_payload
            | {
                "interpolation_method": review_corrections.INTERPOLATION_LINEAR,
                "keyframes": (
                    review_corrections.CorrectionKeyframe(0, (20, 20, 54, 34)),
                    review_corrections.CorrectionKeyframe(2, (30, 22, 70, 44)),
                ),
                "source_frame_end": 2,
            }
        ),
        correction_id="corr_linear",
        created_at_utc="2026-01-06T00:00:00Z",
        revision_id="corr_rev_linear",
    )
    changed_source = review_corrections.ReviewCorrection(
        **(base_payload | {"source_video_hash": "other-source-sha256"}),
        correction_id="corr_source",
        created_at_utc="2026-01-07T00:00:00Z",
        revision_id="corr_rev_source",
    )

    assert digest_a["effective_regions_hash"] == digest_b["effective_regions_hash"]
    assert reviewed_regions.correction_semantic_sha256(changed) != (
        reviewed_regions.correction_semantic_sha256(first)
    )
    assert reviewed_regions.correction_semantic_sha256(changed_span) != (
        reviewed_regions.correction_semantic_sha256(first)
    )
    assert reviewed_regions.correction_semantic_sha256(linear_multi_frame) != (
        reviewed_regions.correction_semantic_sha256(fixed_multi_frame)
    )
    assert reviewed_regions.correction_semantic_sha256(changed_source) != (
        reviewed_regions.correction_semantic_sha256(first)
    )


def test_supported_canonical_repeated_compilation_is_deterministic_with_reordered_inputs(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_mixed_base_regions(package, manifest.case_id)
    write_mixed_entity_registry(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
    )
    session_dir = create_session(package, manifest)

    first = reviewed_regions.reviewed_region_digest(package, session_dir, case_id=manifest.case_id)
    baseline_path = package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
    rows = baseline_path.read_text(encoding="utf-8").splitlines()
    baseline_path.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    workspace = json.loads((package / "workspace_manifest.json").read_text())
    workspace["content_identities"]["automatic_baseline_regions"]["sha256"] = media_sha(
        baseline_path
    )
    write_json(package / "workspace_manifest.json", workspace)
    session_dir = create_session(package, manifest)
    second = reviewed_regions.reviewed_region_digest(package, session_dir, case_id=manifest.case_id)

    assert first["effective_regions_hash"] == second["effective_regions_hash"]


def test_supported_canonical_region_hash_tracks_policy_merge_baseline_and_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_mixed_base_regions(package, manifest.case_id)
    face, plate = write_mixed_entity_registry(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
    )
    session_dir = create_session(package, manifest)
    base = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )

    review_sessions.append_event(
        session_dir,
        event_type="entity_policy_changed",
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id=None,
        before_state=plate,
        after_state=plate
        | {
            "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
            "privacy_action": privacy_entities.POLICY_KEEP_VISIBLE,
            "reviewer_override": True,
        },
        note="keep plate visible",
        application_hashes={},
    )
    policy_changed = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )
    review_sessions.append_event(
        session_dir,
        event_type="entities_merged",
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id=None,
        before_state=None,
        after_state={
            "primary": face
            | {
                "entity_id": "merged_face_plate",
                "label": "Merged Face Plate",
                "member_entity_ids": [face["entity_id"], plate["entity_id"]],
                "merge_split_lineage": {"source_entity_ids": [face["entity_id"]]},
            },
            "retired_entity_ids": [face["entity_id"]],
        },
        note="merge entities",
        application_hashes={},
    )
    merged = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )

    baseline_path = package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
    baseline_records = review_finalization.load_privacy_region_records(baseline_path)
    baseline_records[0] = replace(baseline_records[0], bbox_xyxy=(24, 20, 58, 34))
    review_finalization.write_privacy_region_records(baseline_path, baseline_records)
    workspace = json.loads((package / "workspace_manifest.json").read_text())
    workspace["content_identities"]["automatic_baseline_regions"]["sha256"] = media_sha(
        baseline_path
    )
    write_json(package / "workspace_manifest.json", workspace)
    baseline_changed = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )

    original_render_settings = review_working_state.render_settings

    def larger_pixel_settings() -> Any:
        return replace(original_render_settings(), pixel_size=32)

    monkeypatch.setattr(review_working_state, "render_settings", larger_pixel_settings)
    renderer_changed = reviewed_regions.reviewed_region_digest(
        package,
        session_dir,
        case_id=manifest.case_id,
    )

    assert policy_changed["effective_regions_hash"] != base["effective_regions_hash"]
    assert merged["effective_regions_hash"] != policy_changed["effective_regions_hash"]
    assert baseline_changed["effective_regions_hash"] != merged["effective_regions_hash"]
    assert renderer_changed["effective_regions_hash"] != baseline_changed[
        "effective_regions_hash"
    ]


def test_supported_canonical_category_run_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    workspace_path = package / "workspace_manifest.json"
    workspace = json.loads(workspace_path.read_text())
    workspace["category_run_manifest_identities"][CATEGORY_PLATE]["configuration_hash"] = (
        "changed"
    )
    workspace["category_run_manifest_identity_hash"] = stable_hash(
        workspace["category_run_manifest_identities"]
    )
    write_json(workspace_path, workspace)

    with pytest.raises(
        reviewed_regions.ReviewedRegionCompilationError,
        match="configuration mismatch",
    ):
        reviewed_regions.reviewed_region_digest(package, session_dir, case_id=manifest.case_id)


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "category_identity_source",
            lambda package, workspace, _manifest: workspace[
                "category_run_manifest_identities"
            ][CATEGORY_PLATE].__setitem__("source_sha256", "changed-source"),
            "category identity source hash mismatch",
        ),
        (
            "category_run_content",
            lambda package, workspace, _manifest: workspace[
                "category_run_manifest_identities"
            ][CATEGORY_PLATE].__setitem__("category_run_content_sha256", "changed-content"),
            "category run content hash mismatch",
        ),
        (
            "category_set",
            lambda _package, workspace, _manifest: workspace.__setitem__(
                "selected_categories",
                [CATEGORY_FACE, CATEGORY_PLATE],
            ),
            "category run identities must match selected categories",
        ),
        (
            "category_run_artifact",
            lambda _package, workspace, _manifest: workspace[
                "category_run_manifest_identities"
            ][CATEGORY_PLATE].__setitem__("sha256", "changed-artifact"),
            "category run artifact hash mismatch",
        ),
        (
            "baseline_artifact",
            lambda package, _workspace, manifest: (
                package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
            ).write_text(
                (
                    package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
                ).read_text(encoding="utf-8")
                * 2,
                encoding="utf-8",
            ),
            "baseline region hash identity mismatch",
        ),
    ],
)
def test_supported_canonical_baseline_binding_tampering_fails_closed(
    tmp_path: Path,
    name: str,
    mutate: Any,
    message: str,
) -> None:
    del name
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    workspace_path = package / "workspace_manifest.json"
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    mutate(package, workspace, manifest)
    if "category_run_manifest_identities" in workspace:
        workspace["category_run_manifest_identity_hash"] = stable_hash(
            workspace["category_run_manifest_identities"]
        )
    write_json(workspace_path, workspace)
    session_dir = create_session(package, manifest)

    with pytest.raises(reviewed_regions.ReviewedRegionCompilationError, match=message):
        reviewed_regions.reviewed_region_digest(package, session_dir, case_id=manifest.case_id)


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_reviewed_regions_write",
        "after_video_render",
        "after_media_validation",
        "after_region_validation",
        "after_revision_manifest_write",
        "after_immutable_revision_move",
        "before_current_pointer_update",
        "during_current_pointer_update",
    ],
)
def test_supported_canonical_publication_failure_never_writes_current_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    accepted = review_working_state.derive_working_state(package, session_dir)[
        "active_projection_hash"
    ]
    monkeypatch.setenv("REDACTOR_REVIEWED_REGIONS_FAIL_AT", failure_stage)

    with pytest.raises(reviewed_regions.ReviewedRegionPublicationStaleError):
        review_working_state.materialize_latest_reviewed(
            package,
            session_dir,
            case_id=manifest.case_id,
            accepted_revision=accepted,
        )

    assert not (package / "latest_reviewed" / "current.json").exists()
    monkeypatch.delenv("REDACTOR_REVIEWED_REGIONS_FAIL_AT")
    retry = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=accepted,
    )

    assert retry["latest_reviewed"]["revision_id"].startswith("latest_reviewed_")
    assert json.loads((package / "latest_reviewed" / "current.json").read_text())[
        "status"
    ] == "ready"


def test_generated_browser_code_keeps_correction_preview_distinct() -> None:
    script = review_workspace.workspace_js()
    legacy_mapping = (
        "roleKey = latestReviewedPublicationDeferred() ? "
        "'correction_preview' : 'latest_reviewed'"
    )

    assert legacy_mapping not in script
    assert "role: 'correction_preview'" in script
    assert "state.workingState?.materialized_latest_reviewed" in script


def test_supported_canonical_public_preview_api_returns_correction_preview_role(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
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
            "reason": "preview",
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
                "reason": "preview",
                "session_id": session["session_id"],
            },
        )

        assert preview_response["preview"]["role"] == "correction_preview"
        assert preview_response["preview"]["latest_reviewed"] is None
    finally:
        server.shutdown()
        server.server_close()


def test_supported_canonical_public_workflow_publishes_authoritative_latest_only(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 0.01
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
        before_versions = json.loads(
            urllib.request.urlopen(
                (
                    f"{base_url}/api/v1/cases/{manifest.case_id}/versions"
                    f"?session_id={session['session_id']}"
                ),
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )["versions"]
        correction = {
            "case_id": manifest.case_id,
            "category": CATEGORY_PLATE,
            "configuration_freeze_hash": session["configuration_freeze_hash"],
            "interpolation_method": review_corrections.INTERPOLATION_NONE,
            "keyframe_boxes": [{"bbox_xyxy": [22, 22, 58, 38], "source_frame": 1}],
            "operation": review_corrections.OP_ADD,
            "reason": "preview",
            "reviewer_identity": {"reviewer_id": "analyst@example.test"},
            "source_frame_end": 1,
            "source_frame_start": 1,
            "source_video_hash": manifest.source_video_hash,
            "task_id": "task_1",
        }
        preview_response = post_json(
            f"{base_url}/api/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "preview",
                "session_id": session["session_id"],
            },
        )
        correction_response = post_json(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "save",
                "session_id": session["session_id"],
            },
        )
        job = wait_for_materialization_job(
            server,
            str(correction_response["materialization_job"]["job_id"]),
        )
        after_versions = json.loads(
            urllib.request.urlopen(
                (
                    f"{base_url}/api/v1/cases/{manifest.case_id}/versions"
                    f"?session_id={session['session_id']}"
                ),
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )["versions"]
        reloaded = json.loads(
            urllib.request.urlopen(
                f"{base_url}/api/session-state?session_id={session['session_id']}",
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )
        pointer = json.loads((package / "latest_reviewed" / "current.json").read_text())
        latest = next(role for role in after_versions if role["role"] == "latest_reviewed")
        finalization = json.loads((package / "workspace_manifest.json").read_text())[
            "lifecycle"
        ]["finalization"]

        assert [role["role"] for role in before_versions] == ["original", "automatic"]
        assert preview_response["preview"]["role"] == "correction_preview"
        assert preview_response["preview"]["latest_reviewed"] is None
        assert job["status"] == "completed"
        assert "correction_preview" not in {role["role"] for role in after_versions}
        assert [role["role"] for role in after_versions] == [
            "original",
            "automatic",
            "latest_reviewed",
        ]
        assert pointer["status"] == "ready"
        assert latest["revision_id"] == pointer["revision_id"]
        assert manifest.case_id in reloaded["working_state"]["materialized_latest_reviewed"]
        assert finalization["status"] == "not_run"
        assert not list(package.glob("**/final_reviewed*.mp4"))
    finally:
        server.shutdown()
        server.server_close()


def test_supported_canonical_public_sweep_finalize_and_versions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
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
        session_dir = review_sessions.session_directory(package, session["session_id"])
        state = review_working_state.derive_working_state(package, session_dir)
        materialization = review_working_state.materialize_latest_reviewed(
            package,
            session_dir,
            case_id=manifest.case_id,
            accepted_revision=state["active_projection_hash"],
        )
        before_finalize = json.loads(
            urllib.request.urlopen(
                (
                    f"{base_url}/api/v1/cases/{manifest.case_id}/versions"
                    f"?session_id={session['session_id']}"
                ),
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )["versions"]
        sweep_response = post_json(
            f"{base_url}/api/events/sweep",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "session_id": session["session_id"],
                "viewed_ranges": [{"start": 0, "end": 4}],
            },
        )
        finalize_response = post_json(
            f"{base_url}/api/finalize",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "session_id": session["session_id"],
            },
        )
        after_finalize = json.loads(
            urllib.request.urlopen(
                (
                    f"{base_url}/api/v1/cases/{manifest.case_id}/versions"
                    f"?session_id={session['session_id']}"
                ),
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )["versions"]

        assert "final_reviewed" not in {role["role"] for role in before_finalize}
        assert sweep_response["event"]["after_state"]["schema_version"] == (
            "full_video_sweep_acknowledgement.v2"
        )
        assert finalize_response["ok"] is True
        assert finalize_response["result"]["final_video_sha256"] == materialization["video_sha256"]
        assert "final_reviewed" in {role["role"] for role in after_finalize}
    finally:
        server.shutdown()
        server.server_close()


def test_supported_canonical_legacy_projection_pointer_survives_sweep_and_finalization(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    state = review_working_state.derive_working_state(package, session_dir)
    materialization = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=state["active_projection_hash"],
    )
    legacy_hash = rewrite_latest_reviewed_pointer_as_legacy_projection_v1(
        package,
        session_dir,
        manifest.case_id,
    )
    latest_path = package / materialization["latest_reviewed"]["path"]

    before_sweep = review_working_state.derive_working_state(package, session_dir)
    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )
    acknowledge_current_latest_reviewed(package, session_dir, manifest.case_id)
    after_sweep = review_working_state.derive_working_state(package, session_dir)
    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )
    after_final = review_working_state.derive_working_state(package, session_dir)
    versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]

    assert legacy_hash == before_sweep["active_projection_hashes"][
        review_working_state.LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION
    ]
    assert before_sweep["active_projection_hash"] == after_sweep["active_projection_hash"]
    assert before_sweep["active_projection_hash"] == after_final["active_projection_hash"]
    assert before_sweep["active_projection_hashes"][
        review_working_state.LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION
    ] != after_sweep["active_projection_hashes"][
        review_working_state.LEGACY_ACTIVE_PROJECTION_HASH_SCHEMA_VERSION
    ]
    assert manifest.case_id in before_sweep["materialized_latest_reviewed"]
    assert manifest.case_id in after_sweep["materialized_latest_reviewed"]
    assert manifest.case_id in after_final["materialized_latest_reviewed"]
    assert any("full-video sweep" in item for item in blockers["blockers"])
    assert result["finalized"] is True
    assert result["latest_reviewed_revision_id"] == materialization["revision_id"]
    assert result["final_video_sha256"] == media_sha(latest_path)
    assert "final_reviewed" in {role["role"] for role in versions}


def test_case_media_download_endpoint_enforces_role_and_delivery_readiness(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        download_url = (
            f"{base_url}/api/v1/cases/{manifest.case_id}/download"
            f"?role=automatic&session_id={session_dir.name}"
        )
        with urllib.request.urlopen(download_url, timeout=5) as response:
            downloaded = response.read()
            content_disposition = response.headers.get("Content-Disposition", "")
            content_type = response.headers.get("Content-Type", "")
        delivery = get_json(
            f"{base_url}/api/v1/cases/{manifest.case_id}/delivery-package"
            f"?session_id={session_dir.name}"
        )
        unsafe_status, unsafe_body = get_json_error(
            f"{base_url}/api/v1/cases/{manifest.case_id}/download?role=../../etc/passwd"
        )
        final_status, final_body = get_json_error(
            f"{base_url}/api/v1/cases/{manifest.case_id}/download"
            f"?role=final&session_id={session_dir.name}"
        )

        automatic_path = package / "media" / manifest.case_id / "automatic_preview.mp4"
        assert downloaded == automatic_path.read_bytes()
        assert "attachment;" in content_disposition
        assert "automatic" in content_disposition
        assert content_type == "video/mp4"
        assert delivery["ready_for_delivery"] is False
        assert delivery["roles"]["automatic"]["available"] is True
        assert delivery["roles"]["final_reviewed"]["available"] is False
        assert delivery["manifest"]["case"] == manifest.case_id
        assert delivery["manifest"]["available_media_roles"] == ["original", "automatic"]
        assert delivery["manifest"]["media_sha256"]["automatic"]
        assert delivery["manifest"]["review_state_identity"]["decision_identity"] == (
            review_sessions.REVIEW_DECISION_IDENTITY_V2
        )
        assert delivery["manifest"]["status"] == "non-final"
        assert delivery["manifest"]["validation_state"]["latest_reviewed_current"] is False
        assert unsafe_status == 400
        assert "unsupported downloadable" in str(unsafe_body["error"])
        assert final_status == 404
        assert "no current final_reviewed version" in str(final_body["error"])
    finally:
        server.shutdown()
        server.server_close()


def test_supported_canonical_unknown_projection_schema_blocks_pointer(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    state = review_working_state.derive_working_state(package, session_dir)
    review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=state["active_projection_hash"],
    )
    pointer_path = package / "latest_reviewed" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["active_projection_hash_schema_version"] = "active_projection.future"
    write_json(pointer_path, pointer)

    stale_state = review_working_state.derive_working_state(package, session_dir)
    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert manifest.case_id not in stale_state["materialized_latest_reviewed"]
    assert result["finalized"] is False
    assert any("unknown active projection hash schema" in item for item in result["blockers"])


def test_supported_canonical_projection_hash_is_copy_independent_and_corrupt_hash_blocks(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    state = review_working_state.derive_working_state(package, session_dir)
    review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=state["active_projection_hash"],
    )
    copied = tmp_path / "copied" / "package"
    shutil.copytree(package, copied)
    copied_session_dir = copied / session_dir.relative_to(package)

    original = review_working_state.derive_working_state(package, session_dir)
    copied_state = review_working_state.derive_working_state(copied, copied_session_dir)
    pointer_path = copied / "latest_reviewed" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["working_state_projection_hash"] = "0" * 64
    pointer["accepted_working_state_revision"] = "0" * 64
    write_json(pointer_path, pointer)
    corrupted = review_working_state.derive_working_state(copied, copied_session_dir)

    assert original["active_projection_hashes"] == copied_state["active_projection_hashes"]
    assert manifest.case_id in copied_state["materialized_latest_reviewed"]
    assert manifest.case_id not in corrupted["materialized_latest_reviewed"]


def test_supported_canonical_public_required_task_blocks_then_allows_publication(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_PRIVACY_LEAK,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "correction_required": False,
                    "review_task_id": "plate_task_1",
                    "source_frame_end": 0,
                    "source_frame_start": 0,
                    "technical_only": False,
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 0.01
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
            "keyframe_boxes": [{"bbox_xyxy": [22, 22, 58, 38], "source_frame": 1}],
            "operation": review_corrections.OP_ADD,
            "reason": "preview",
            "reviewer_identity": {"reviewer_id": "analyst@example.test"},
            "source_frame_end": 1,
            "source_frame_start": 1,
            "source_video_hash": manifest.source_video_hash,
            "task_id": "task_1",
        }
        preview_response = post_json(
            f"{base_url}/api/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "preview",
                "session_id": session["session_id"],
            },
        )
        save_payload = {
            "actor": {"reviewer_id": "analyst@example.test"},
            "correction": correction,
            "preview_token": preview_response["preview"]["preview_token"],
            "reason": "save",
            "session_id": session["session_id"],
        }
        blocked_response = post_json(f"{base_url}/api/corrections", save_payload)
        blocked_job = wait_for_materialization_job(
            server,
            str(blocked_response["materialization_job"]["job_id"]),
        )
        blocked_versions = json.loads(
            urllib.request.urlopen(
                (
                    f"{base_url}/api/v1/cases/{manifest.case_id}/versions"
                    f"?session_id={session['session_id']}"
                ),
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )["versions"]
        blocked_lifecycle = json.loads((package / "workspace_manifest.json").read_text())[
            "lifecycle"
        ]

        decision_response = post_json(
            f"{base_url}/api/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "Resolved",
                "session_id": session["session_id"],
                "task_id": "plate_task_1",
            },
        )
        retry_response = post_json(f"{base_url}/api/corrections", save_payload)
        retry_job = wait_for_materialization_job(
            server,
            str(retry_response["materialization_job"]["job_id"]),
        )
        after_versions = json.loads(
            urllib.request.urlopen(
                (
                    f"{base_url}/api/v1/cases/{manifest.case_id}/versions"
                    f"?session_id={session['session_id']}"
                ),
                timeout=5,
            )
            .read()
            .decode("utf-8")
        )["versions"]
        pointer = json.loads((package / "latest_reviewed" / "current.json").read_text())
        latest = next(role for role in after_versions if role["role"] == "latest_reviewed")
        lifecycle = json.loads((package / "workspace_manifest.json").read_text())[
            "lifecycle"
        ]

        assert blocked_job["status"] == "failed"
        assert "required task has no decision" in str(blocked_job["error"])
        assert "latest_reviewed" not in {role["role"] for role in blocked_versions}
        assert blocked_lifecycle["latest_reviewed"]["status"] == "not_run"
        assert decision_response["ok"] is True
        assert retry_response["idempotent_retry"] is True
        assert retry_job["status"] == "completed"
        assert pointer["status"] == "ready"
        assert latest["revision_id"] == pointer["revision_id"]
        assert lifecycle["finalization"]["status"] == "not_run"
    finally:
        server.shutdown()
        server.server_close()


def test_supported_canonical_materializes_revision_versions_and_reset(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    before_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]

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
    after_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]
    workspace_manifest = json.loads((package / "workspace_manifest.json").read_text())
    case_manifest = case_manifests.load_case_manifest(
        package / "cases" / f"{manifest.case_id}.json"
    )
    current_pointer = json.loads((package / "latest_reviewed" / "current.json").read_text())

    assert "latest_reviewed" not in {role["role"] for role in before_versions}
    assert "latest_reviewed" in {role["role"] for role in after_versions}
    assert latest_path.exists()
    assert media_sha(latest_path) != media_sha(package / manifest.video_roles[1].path)
    assert video_frame_count(latest_path) == manifest.frame_count
    assert workspace_manifest["latest_reviewed_publication"]["status"] == "supported"
    assert "latest_reviewed_current" not in workspace_manifest
    assert current_pointer["status"] == "ready"
    assert current_pointer["revision_id"] == materialization["revision_id"]
    assert current_pointer["reviewed_region_set_sha256"] == materialization[
        "reviewed_region_set_sha256"
    ]
    assert not any(role.role == "latest_reviewed" for role in case_manifest.video_roles)
    assert (package / materialization["revision_manifest"]).exists()
    assert materialization["media_validation"]["expected_frame_count"] == manifest.frame_count

    _event, reset_state = review_working_state.reset_to_automatic_baseline(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
    )
    reset_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]
    reset_manifest = json.loads((package / "workspace_manifest.json").read_text())
    reset_pointer = json.loads((package / "latest_reviewed" / "current.json").read_text())

    assert reset_state["materialized_latest_reviewed"] == {}
    assert "latest_reviewed" not in {role["role"] for role in reset_versions}
    assert reset_manifest["latest_reviewed_publication"]["status"] == "supported"
    assert "latest_reviewed_current" not in reset_manifest
    assert reset_pointer["status"] == "not_current"
    assert latest_path.exists()


@pytest.mark.parametrize(
    "categories",
    [
        (CATEGORY_FACE,),
        (CATEGORY_PLATE,),
        (CATEGORY_FACE, CATEGORY_PLATE),
    ],
)
def test_supported_canonical_finalization_approves_latest_reviewed_bytes(
    tmp_path: Path,
    categories: tuple[str, ...],
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, categories)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=categories,
    )
    session_dir = create_session(package, manifest)
    materialization = publish_latest_and_acknowledge(package, session_dir, manifest.case_id)
    latest_path = package / materialization["latest_reviewed"]["path"]
    before_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]

    result = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
        actor={"reviewer_id": "analyst@example.test"},
    )
    final_path = package / result["final_video"]
    after_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]
    pointer = json.loads((package / "final" / "current.json").read_text(encoding="utf-8"))
    repeat = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
        actor={"reviewer_id": "analyst@example.test"},
    )

    assert "final_reviewed" not in {role["role"] for role in before_versions}
    assert result["finalized"] is True
    assert result["byte_equal_to_latest_reviewed"] is True
    assert result["latest_reviewed_video_sha256"] == media_sha(latest_path)
    assert result["final_video_sha256"] == media_sha(final_path)
    assert result["final_video_sha256"] == result["latest_reviewed_video_sha256"]
    assert result["final_video_materialization_mode"] in {"hardlink", "verified_copy"}
    assert pointer["finalization_id"] == result["finalization_id"]
    assert pointer["finalization_input_sha256"] == result["finalization_input_sha256"]
    assert [role["role"] for role in after_versions] == [
        "original",
        "automatic",
        "latest_reviewed",
        "final_reviewed",
    ]
    assert repeat["reused_existing_finalization"] is True
    assert repeat["finalization_id"] == result["finalization_id"]
    assert len([path for path in (package / "final" / "revisions").iterdir() if path.is_dir()]) == 1


def test_supported_canonical_sweep_acknowledgement_is_revision_bound(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "review_task_id": "task_plate_boundary",
                    "technical_only": False,
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="task_plate_boundary",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="resolved",
    )
    first = publish_latest_and_acknowledge(package, session_dir, manifest.case_id)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="task_plate_boundary",
        action=review_sessions.ACTION_NOT_SENSITIVE,
        note="still resolved",
    )
    second_state = review_working_state.derive_working_state(package, session_dir)
    second = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=second_state["active_projection_hash"],
    )

    blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert first["revision_id"] != second["revision_id"]
    assert blockers["ready"] is False
    assert any(
        "sweep_acknowledgement:latest_reviewed_revision_id mismatch" in item
        for item in blockers["blockers"]
    )


def test_supported_canonical_review_change_invalidates_current_final(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(
        package / "reviewer_tasks.json",
        {
            "tasks": [
                {
                    "allowed_actions": [
                        review_sessions.ACTION_LOOKS_CORRECT,
                        review_sessions.ACTION_NOT_SENSITIVE,
                    ],
                    "blocking": True,
                    "case_id": manifest.case_id,
                    "category": CATEGORY_PLATE,
                    "review_task_id": "task_plate_boundary",
                    "technical_only": False,
                }
            ]
        },
    )
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="task_plate_boundary",
        action=review_sessions.ACTION_LOOKS_CORRECT,
        note="resolved",
    )
    publish_latest_and_acknowledge(package, session_dir, manifest.case_id)
    finalized = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )
    current_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]

    review_sessions.append_decision_event(
        session_dir,
        actor={"reviewer_id": "analyst@example.test"},
        case_id=manifest.case_id,
        task_id="task_plate_boundary",
        action=review_sessions.ACTION_NOT_SENSITIVE,
        note="changed after final",
    )
    stale_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]
    pointer = json.loads((package / "final" / "current.json").read_text(encoding="utf-8"))

    assert finalized["finalized"] is True
    assert "final_reviewed" in {role["role"] for role in current_versions}
    assert "final_reviewed" not in {role["role"] for role in stale_versions}
    assert pointer["finalization_id"] == finalized["finalization_id"]


def test_supported_canonical_finalization_failure_does_not_publish_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_canonical_base_regions(package, manifest.case_id, (CATEGORY_PLATE,))
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    materialization = publish_latest_and_acknowledge(package, session_dir, manifest.case_id)
    latest_path = package / materialization["latest_reviewed"]["path"]
    latest_hash = media_sha(latest_path)
    monkeypatch.setenv(
        "REDACTOR_FINALIZATION_FAIL_AT",
        "after_final_video_link_copy",
    )

    with pytest.raises(ValueError, match="injected finalization failure"):
        review_finalization.finalize_reviewed_case(
            package_root=package,
            case_id=manifest.case_id,
            session_dir=session_dir,
        )
    current_after_failure = package / "final" / "current.json"
    assert not current_after_failure.exists()

    monkeypatch.delenv("REDACTOR_FINALIZATION_FAIL_AT")
    retry = review_finalization.finalize_reviewed_case(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )

    assert media_sha(latest_path) == latest_hash
    assert retry["finalized"] is True
    assert retry["final_video_sha256"] == latest_hash


def test_supported_canonical_new_event_invalidates_current_latest_until_republish(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_mixed_base_regions(package, manifest.case_id)
    _face, plate = write_mixed_entity_registry(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
    )
    session_dir = create_session(package, manifest)
    original_revision = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=review_working_state.derive_working_state(package, session_dir)[
            "active_projection_hash"
        ],
    )
    original_path = package / original_revision["latest_reviewed"]["path"]

    current_state = review_working_state.derive_working_state(package, session_dir)
    current_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]

    review_sessions.append_event(
        session_dir,
        event_type="entity_policy_changed",
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
        task_id=None,
        before_state=plate,
        after_state=plate
        | {
            "current_policy": privacy_entities.POLICY_KEEP_VISIBLE,
            "privacy_action": privacy_entities.POLICY_KEEP_VISIBLE,
            "reviewer_override": True,
        },
        note="keep plate visible",
        application_hashes={},
    )
    stale_state = review_working_state.derive_working_state(package, session_dir)
    stale_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]
    stale_blockers = review_finalization.finalization_blockers(
        package_root=package,
        case_id=manifest.case_id,
        session_dir=session_dir,
    )
    pointer_after_change = json.loads(
        (package / "latest_reviewed" / "current.json").read_text()
    )

    republished = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=stale_state["active_projection_hash"],
    )
    republished_versions = review_working_state.case_versions(
        package,
        session_dir,
        case_id=manifest.case_id,
    )["versions"]

    assert manifest.case_id in current_state["materialized_latest_reviewed"]
    assert "latest_reviewed" in {role["role"] for role in current_versions}
    assert stale_state["active_projection_hash"] != current_state["active_projection_hash"]
    assert manifest.case_id not in stale_state["materialized_latest_reviewed"]
    assert "latest_reviewed" not in {role["role"] for role in stale_versions}
    assert any(
        "Latest reviewed has not been published" in item
        for item in stale_blockers["blockers"]
    )
    assert pointer_after_change["revision_id"] == original_revision["revision_id"]
    assert original_path.exists()
    assert republished["revision_id"] != original_revision["revision_id"]
    assert "latest_reviewed" in {role["role"] for role in republished_versions}


def test_supported_canonical_session_binding_survives_publication_and_reset(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)

    materialization = review_working_state.materialize_latest_reviewed(
        package,
        session_dir,
        case_id=manifest.case_id,
        accepted_revision=review_working_state.derive_working_state(package, session_dir)[
            "active_projection_hash"
        ],
    )
    binding_after_publish = review_workspace.validate_session_workspace_binding(
        package,
        session_dir,
    )
    _event, _reset_state = review_working_state.reset_to_automatic_baseline(
        package,
        session_dir,
        actor={"reviewer_id": "analyst"},
        case_id=manifest.case_id,
    )
    binding_after_reset = review_workspace.validate_session_workspace_binding(
        package,
        session_dir,
    )

    assert materialization["latest_reviewed"]["revision_id"].startswith("latest_reviewed_")
    assert binding_after_publish["ok"] is True
    assert binding_after_reset["ok"] is True


def test_supported_canonical_publication_guard_rejects_workspace_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    manifest = add_automatic_role(package, synthetic_case_manifest(package))
    write_base_region(package, manifest.case_id)
    write_review_entity_shell(package, manifest.case_id)
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    write_supported_canonical_workspace_manifest(
        package,
        manifest,
        categories=(CATEGORY_PLATE,),
    )
    session_dir = create_session(package, manifest)
    original_verify = privacy_rendering.verify_rendered_regions

    def mutate_workspace_manifest_after_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_verify(*args, **kwargs)
        workspace_manifest_path = package / "workspace_manifest.json"
        workspace_manifest = json.loads(workspace_manifest_path.read_text(encoding="utf-8"))
        workspace_manifest["guard_mutation"] = "changed-after-render"
        write_json(workspace_manifest_path, workspace_manifest)
        return result

    monkeypatch.setattr(
        privacy_rendering,
        "verify_rendered_regions",
        mutate_workspace_manifest_after_validation,
    )

    with pytest.raises(
        reviewed_regions.ReviewedRegionPublicationStaleError,
        match="publication guard is stale",
    ):
        review_working_state.materialize_latest_reviewed(
            package,
            session_dir,
            case_id=manifest.case_id,
            accepted_revision=review_working_state.derive_working_state(package, session_dir)[
                "active_projection_hash"
            ],
        )

    workspace_manifest = json.loads((package / "workspace_manifest.json").read_text())
    case_manifest = case_manifests.load_case_manifest(
        package / "cases" / f"{manifest.case_id}.json"
    )

    assert "latest_reviewed_current" not in workspace_manifest
    assert workspace_manifest["latest_reviewed_publication"]["status"] == "supported"
    assert not any(role.role == "latest_reviewed" for role in case_manifest.video_roles)
    assert not (package / "latest_reviewed" / "current.json").exists()


@pytest.mark.parametrize("route", ["/api/entity-events", "/api/finalize"])
@pytest.mark.parametrize("case_id", ["../outside", "/tmp/outside", "unknown_case"])
def test_mutation_routes_reject_unsafe_or_unknown_case_ids_without_events(
    tmp_path: Path,
    route: str,
    case_id: str,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload: dict[str, object]
    if route == "/api/entity-events":
        payload = {
            "actor": {"reviewer_id": "analyst@example.test"},
            "case_id": case_id,
            "event": {
                "case_id": case_id,
                "event_type": "manual_issue_created",
                "payload": {"after": {"case_id": case_id}},
            },
            "session_id": session_dir.name,
        }
    else:
        payload = {
            "actor": {"reviewer_id": "analyst@example.test"},
            "case_id": case_id,
            "session_id": session_dir.name,
        }
    try:
        before = review_sessions.load_events(session_dir)
        status, body = post_json_error(f"{base_url}{route}", payload)
        after = review_sessions.load_events(session_dir)

        assert status == 400
        assert body["ok"] is False
        assert after == before
        assert case_id not in {str(event.get("case_id") or "") for event in after}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "route_name",
    [
        "decisions",
        "sweep",
        "correction_preview",
        "correction_save",
        "object_policy",
        "bulk_object_policy",
        "checkpoint_restore",
        "checkpoint_restart",
        "checkpoint_reset",
        "entity_merge",
        "entity_split",
    ],
)
@pytest.mark.parametrize("case_id", ["../outside", "/tmp/outside", "unknown_case"])
def test_all_case_mutation_routes_reject_unsafe_or_unknown_case_ids_without_events(
    tmp_path: Path,
    route_name: str,
    case_id: str,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    session_dir = create_session(package, manifest)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    valid_correction = {
        "case_id": manifest.case_id,
        "category": CATEGORY_PLATE,
        "configuration_freeze_hash": stable_hash({"schema_version": "freeze.test", "value": 1}),
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

    try:
        preview_response = post_json(
            f"{base_url}/api/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": valid_correction,
                "reason": "reviewed",
                "session_id": session_dir.name,
            },
        )
        unsafe_correction = dict(valid_correction)
        unsafe_correction["case_id"] = case_id
        payloads = {
            "decisions": (
                "/api/decisions",
                {
                    "action": review_sessions.ACTION_LOOKS_CORRECT,
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "note": "correct",
                    "session_id": session_dir.name,
                    "task_id": "task_1",
                },
            ),
            "sweep": (
                "/api/events/sweep",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "session_id": session_dir.name,
                    "viewed_ranges": [{"start": 0, "end": 4}],
                },
            ),
            "correction_preview": (
                "/api/corrections/preview",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "correction": unsafe_correction,
                    "reason": "reviewed",
                    "session_id": session_dir.name,
                },
            ),
            "correction_save": (
                "/api/corrections",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "correction": unsafe_correction,
                    "preview_token": preview_response["preview"]["preview_token"],
                    "reason": "reviewed",
                    "session_id": session_dir.name,
                },
            ),
            "object_policy": (
                "/api/object-policy",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "entity_id": "entity_1",
                    "policy": privacy_entities.POLICY_REDACT,
                    "session_id": session_dir.name,
                },
            ),
            "bulk_object_policy": (
                "/api/bulk-object-policy",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "entity_ids": ["entity_1"],
                    "policy": privacy_entities.POLICY_REDACT,
                    "session_id": session_dir.name,
                },
            ),
            "checkpoint_restore": (
                "/api/checkpoints/restore",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "checkpoint_uuid": "checkpoint_missing",
                    "session_id": session_dir.name,
                },
            ),
            "checkpoint_restart": (
                "/api/checkpoints/restart-review",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "session_id": session_dir.name,
                },
            ),
            "checkpoint_reset": (
                "/api/checkpoints/reset",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": case_id,
                    "session_id": session_dir.name,
                },
            ),
            "entity_merge": (
                "/api/entities/merge",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "after_state": {
                        "primary": {"case_id": case_id, "category": CATEGORY_PLATE},
                        "retired_entity_ids": ["entity_2"],
                    },
                    "before_state": {
                        "source": {"case_id": case_id, "category": CATEGORY_PLATE},
                        "target": {
                            "case_id": case_id,
                            "category": CATEGORY_PLATE,
                            "entity_id": "entity_2",
                        },
                    },
                    "case_id": case_id,
                    "reason": "merge",
                    "session_id": session_dir.name,
                },
            ),
            "entity_split": (
                "/api/entities/split",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "after_state": {"case_id": case_id},
                    "before_state": {"case_id": case_id},
                    "case_id": case_id,
                    "reason": "split",
                    "session_id": session_dir.name,
                },
            ),
        }
        route, payload = payloads[route_name]
        before = review_sessions.load_events(session_dir)
        status, body = post_json_error(f"{base_url}{route}", payload)
        after = review_sessions.load_events(session_dir)

        assert status in {400, 404}
        assert body["ok"] is False
        assert after == before
    finally:
        server.shutdown()
        server.server_close()


def test_lower_level_and_cli_finalization_reject_unsafe_case_id(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    write_json(package / "reviewer_tasks.json", {"tasks": []})
    session_dir = create_session(package, manifest)

    with pytest.raises(case_ids.CaseIdError):
        review_finalization.finalize_reviewed_case(
            package_root=package,
            case_id="../outside",
            session_dir=session_dir,
        )
    with pytest.raises(case_ids.CaseIdError):
        review_sessions.append_event(
            session_dir,
            event_type="manual_issue_created",
            actor={"reviewer_id": "analyst@example.test"},
            case_id="../outside",
            task_id=None,
            before_state=None,
            after_state={},
            note="invalid",
            application_hashes={},
        )

    result = CliRunner().invoke(
        main,
        [
            "finalize-reviewed-case",
            "--package-root",
            str(package),
            "--case-id",
            "../outside",
            "--session-dir",
            str(session_dir),
        ],
    )

    assert result.exit_code != 0
    assert "unsafe case_id" in result.output
    assert "../outside" not in {
        str(event.get("case_id") or "") for event in review_sessions.load_events(session_dir)
    }


@pytest.mark.parametrize(
    ("name", "category", "start", "end", "method", "keyframes", "expected_frames"),
    [
        (
            "face_single",
            CATEGORY_FACE,
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            [{"bbox_xyxy": [20, 20, 54, 34], "source_frame": 0}],
            [0],
        ),
        (
            "plate_single",
            CATEGORY_PLATE,
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            [{"bbox_xyxy": [20, 20, 54, 34], "source_frame": 0}],
            [0],
        ),
        (
            "other_single",
            CATEGORY_OTHER_SENSITIVE,
            0,
            0,
            review_corrections.INTERPOLATION_NONE,
            [{"bbox_xyxy": [20, 20, 54, 34], "source_frame": 0}],
            [0],
        ),
        (
            "other_fixed",
            CATEGORY_OTHER_SENSITIVE,
            1,
            3,
            review_corrections.INTERPOLATION_NONE,
            [{"bbox_xyxy": [18, 18, 58, 38], "source_frame": 1}],
            [1, 2, 3],
        ),
        (
            "other_interpolate",
            CATEGORY_OTHER_SENSITIVE,
            1,
            3,
            review_corrections.INTERPOLATION_LINEAR,
            [
                {"bbox_xyxy": [18, 18, 58, 38], "source_frame": 1},
                {"bbox_xyxy": [26, 22, 66, 42], "source_frame": 3},
            ],
            [1, 2, 3],
        ),
    ],
)
def test_workspace_server_accepts_manual_correction_categories_and_modes(
    tmp_path: Path,
    name: str,
    category: str,
    start: int,
    end: int,
    method: str,
    keyframes: list[dict[str, Any]],
    expected_frames: list[int],
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session_response = post_json(
            f"{base_url}/api/v1/sessions",
            {
                "reviewer": {"display_name": "Analyst", "reviewer_id": "analyst@example.test"},
                "source_revision": "test",
            },
        )
        session = session_response["session"]
        correction = {
            "case_id": manifest.case_id,
            "category": category,
            "configuration_freeze_hash": session["configuration_freeze_hash"],
            "interpolation_method": method,
            "keyframe_boxes": keyframes,
            "operation": review_corrections.OP_ADD,
            "reason": f"{name} reviewed",
            "reviewer_identity": {"reviewer_id": "analyst@example.test"},
            "source_frame_end": end,
            "source_frame_start": start,
            "source_video_hash": manifest.source_video_hash,
            "task_id": f"task_{name}",
        }

        preview_response = post_json(
            f"{base_url}/api/v1/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": f"{name} preview",
                "session_id": session["session_id"],
            },
        )
        correction_response = post_json(
            f"{base_url}/api/v1/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction
                | {
                    "reviewer_identity": {
                        "display_name": "Analyst",
                        "reviewer_id": "analyst@example.test",
                        "session_id": session["session_id"],
                    }
                },
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": f"{name} save",
                "session_id": session["session_id"],
            },
        )
        session_dir = package / "sessions" / session["session_id"]
        state = review_working_state.derive_working_state(package, session_dir)
        effective_regions, _provenance = review_working_state.compute_effective_privacy_regions(
            package,
            session_dir,
            case_id=manifest.case_id,
        )
        correction_regions = [
            record for record in effective_regions if record.provenance == "reviewed_correction"
        ]
        with urllib.request.urlopen(
            f"{base_url}/api/v1/session-state?session_id={session['session_id']}",
            timeout=5,
        ) as response:
            reloaded = json.loads(response.read().decode("utf-8"))

        assert preview_response["ok"] is True
        assert correction_response["ok"] is True
        assert correction_response["event"]["after_state"]["category"] == category
        assert correction_response["verification_task"]["category"] == category
        if category == CATEGORY_OTHER_SENSITIVE:
            assert "Other sensitive region" in correction_response["verification_task"]["title"]
        assert [
            record.source_record["source_frame"] for record in correction_regions
        ] == expected_frames
        assert {record.category for record in correction_regions} == {category}
        assert state["active_correction_ids"] == [
            correction_response["event"]["after_state"]["correction_id"]
        ]
        assert reloaded["working_state"]["manual_corrections"][-1]["category"] == category

        if name == "other_single":
            reset_response = post_json(
                f"{base_url}/api/v1/checkpoints/reset",
                {
                    "actor": {"reviewer_id": "analyst@example.test"},
                    "case_id": manifest.case_id,
                    "session_id": session["session_id"],
                },
            )
            reset_regions, _reset_provenance = (
                review_working_state.compute_effective_privacy_regions(
                    package,
                    session_dir,
                    case_id=manifest.case_id,
                )
            )
            assert reset_response["ok"] is True
            assert reset_response["working_state"]["active_correction_ids"] == []
            assert [
                record for record in reset_regions if record.provenance == "reviewed_correction"
            ] == []
    finally:
        server.shutdown()
        server.server_close()


def test_workspace_server_rejects_interpolation_preview_without_end_keyframe(
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
            f"{base_url}/api/v1/sessions",
            {
                "reviewer": {"display_name": "Analyst", "reviewer_id": "analyst@example.test"},
                "source_revision": "test",
            },
        )
        session = session_response["session"]
        correction = {
            "case_id": manifest.case_id,
            "category": CATEGORY_OTHER_SENSITIVE,
            "configuration_freeze_hash": session["configuration_freeze_hash"],
            "interpolation_method": review_corrections.INTERPOLATION_LINEAR,
            "keyframe_boxes": [{"bbox_xyxy": [20, 20, 54, 34], "source_frame": 0}],
            "operation": review_corrections.OP_ADD,
            "reason": "invalid interpolation preview",
            "reviewer_identity": {"reviewer_id": "analyst@example.test"},
            "source_frame_end": 3,
            "source_frame_start": 0,
            "source_video_hash": manifest.source_video_hash,
            "task_id": "task_invalid_interpolation",
        }

        status, body = post_json_error(
            f"{base_url}/api/v1/corrections/preview",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "reason": "invalid interpolation preview",
                "session_id": session["session_id"],
            },
        )

        assert status == 400
        assert body["ok"] is False
        assert "end keyframe at source frame 3" in body["error"]
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
        changed["keyframe_boxes"] = [{"bbox_xyxy": [22, 22, 60, 40], "source_frame": 0}]

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


def test_workspace_server_rejects_cross_case_manual_correction_preview_token(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    other_manifest = synthetic_case_manifest_for(package, "other", display_name="Other")
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
        cross_case = dict(correction)
        cross_case["case_id"] = other_manifest.case_id
        cross_case["source_video_hash"] = other_manifest.source_video_hash

        status, body = post_json_error(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": cross_case,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )

        assert status == 409
        assert "different case" in str(body["error"])
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


def test_workspace_server_unlocks_correction_verification_after_publication(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = synthetic_case_manifest(package)
    write_json(package / "configuration_freeze.json", {"schema_version": "test"})
    (package / "index.html").write_text("ok", encoding="utf-8")
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 0.01
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
            "keyframe_boxes": [{"bbox_xyxy": [0, 0, 96, 64], "source_frame": 0}],
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
        correction_response = post_json(
            f"{base_url}/api/corrections",
            {
                "actor": {"reviewer_id": "analyst@example.test"},
                "correction": correction,
                "preview_token": preview_response["preview"]["preview_token"],
                "reason": "reviewed",
                "session_id": session["session_id"],
            },
        )
        job = wait_for_materialization_job(
            server,
            str(correction_response["materialization_job"]["job_id"]),
        )
        assert job["status"] == "completed"
        session_dir = package / "sessions" / session["session_id"]
        state = review_working_state.derive_working_state(package, session_dir)
        verification_task = review_working_state.find_correction_verification_task(
            state,
            correction_id=correction_response["event"]["after_state"]["correction_id"],
        )

        assert verification_task is not None
        assert verification_task["actionable"] is True
        decision_response = post_json(
            f"{base_url}/api/decisions",
            {
                "action": review_sessions.ACTION_LOOKS_CORRECT,
                "actor": {"reviewer_id": "analyst@example.test"},
                "case_id": manifest.case_id,
                "note": "Correct",
                "session_id": session["session_id"],
                "task_id": verification_task["review_task_id"],
            },
        )

        assert decision_response["ok"] is True
        assert decision_response["event"]["event_type"] == "decision_recorded"
    finally:
        server.shutdown()
        server.server_close()


def test_correction_verification_unlocks_on_newer_publication_that_contains_correction(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    latest_path = package / "media" / "day" / "latest_reviewed_new.mp4"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_bytes(b"media")
    task = {
        "case_id": "day",
        "correction_id": "corr_1",
        "normal_task_class": "correction_verification",
        "review_task_id": "correction_verify_day_corr_1",
        "target_effective_regions_hash": "old_hash",
    }
    status = review_working_state.correction_verification_status(
        package_root=package,
        task=task,
        materialized={
            "day": {
                "correction_render_verifications": {"corr_1": {"passed": True}},
                "effective_regions_hash": "new_hash",
                "media_validation_passed": True,
                "path": "media/day/latest_reviewed_new.mp4",
                "published_working_state_revision": "new_revision",
            }
        },
    )

    assert status["actionable"] is True
    assert status["published_effective_regions_hash"] == "new_hash"
    assert status["published_superseded_target_hash"] == "old_hash"


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
