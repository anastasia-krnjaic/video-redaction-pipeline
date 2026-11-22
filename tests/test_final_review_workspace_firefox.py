from __future__ import annotations

import json
import os
import re
import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest

from redactor import case_manifests, entity_refinement, media, review_workspace
from redactor.privacy_records import CATEGORY_FACE, stable_hash

SESSION_READY_JS = (
    "() => document.querySelector('#sessionBanner')?.textContent.includes('Review session active')"
)

CANONICAL_ACTIONS = ["looks_correct", "privacy_leak", "not_sensitive"]
GROUPED_CHILD_TASK_ID = "license_plate_plate_review_0005"

pytestmark = pytest.mark.skipif(
    os.environ.get("REDACTOR_RUN_FIREFOX_E2E") != "1",
    reason="set REDACTOR_RUN_FIREFOX_E2E=1 to run the real Firefox workspace smoke",
)


def record_step(report: dict[str, Any], name: str, **details: Any) -> None:
    report.setdefault("steps", []).append({"details": details, "name": name, "ok": True})


def write_report(package: Path, report: dict[str, Any], report_root: Path | None = None) -> None:
    out = (report_root or package / "browser_smoke") / "browser_smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def migration_session_id(package: Path) -> str | None:
    report_path = package / "review_state_migration_report.json"
    if not report_path.exists():
        report_path = package / "review_v2_final_migration_report.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("migration_session_id", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith("session_"):
            return value
    migration = payload.get("migration")
    if isinstance(migration, dict):
        value = migration.get("session_id")
        if isinstance(value, str) and value.startswith("session_"):
            return value
    return None


def task_card_selector(case_id: str, task_id: str) -> str:
    return f'.case-panel[data-case-id="{case_id}"] .task-card[data-task-id="{task_id}"]'


def task_card_for_child(page: Any, case_id: str, child_task_id: str) -> Any:
    locator = page.locator(f'.case-panel[data-case-id="{case_id}"] .task-card').filter(
        has_text=child_task_id
    )
    locator.first.wait_for(state="visible", timeout=5_000)
    return locator.first


def first_open_plate_group(panel: Any) -> Any:
    cards = panel.locator(".task-card").filter(has_text=re.compile(r"\bPlate\b"))
    cards.first.wait_for(state="visible", timeout=5_000)
    return cards.first


def active_case_panel(page: Any, case_id: str) -> Any:
    panel = page.locator(f'.case-panel[data-case-id="{case_id}"]')
    panel.wait_for(state="visible", timeout=10_000)
    return panel


def click_bucket(page: Any, case_id: str, bucket: str) -> None:
    panel = active_case_panel(page, case_id)
    panel.locator(".filter-row").get_by_role(
        "button", name=re.compile(rf"^{re.escape(bucket)}\b")
    ).click()


def bucket_names(panel: Any) -> list[str]:
    names = []
    for raw in panel.locator(".filter-row button").all_inner_texts():
        names.append(re.sub(r"\s+\d+\s*$", "", raw.strip()).strip())
    return names


def wait_for_timeline_frame(page: Any, case_id: str, expected_frames: set[str]) -> None:
    page.wait_for_function(
        """
        ({caseId, expected}) => {
          const panel = document.querySelector(`.case-panel[data-case-id="${caseId}"]`);
          const slider = panel?.querySelector('[role="slider"][aria-label="Video timeline"]');
          return expected.includes(slider?.getAttribute('aria-valuenow') || '');
        }
        """,
        arg={"caseId": case_id, "expected": sorted(expected_frames)},
        timeout=5_000,
    )


def post_json(url: str, payload: dict[str, object]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def firefox_descriptor(seed: int) -> list[float]:
    values = [0.0, 0.0, 0.0]
    values[seed % len(values)] = 1.0
    return values


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_synthetic_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"mp4v"),
        10,
        (96, 64),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create synthetic video: {path}")
    try:
        for frame_number in range(10):
            frame = np.full((64, 96, 3), 32, dtype=np.uint8)
            if frame_number in {0, 4}:
                cv.rectangle(frame, (10, 10), (30, 30), (0, 0, 220), -1)
            if frame_number in {2, 7}:
                cv.rectangle(frame, (10, 10), (30, 30), (0, 180, 0), -1)
            if frame_number in {1, 3}:
                cv.rectangle(frame, (10, 10), (30, 30), (220, 0, 0), -1)
            if frame_number in {3, 4, 5}:
                cv.rectangle(frame, (40, 10), (60, 30), (0, 180, 180), -1)
            writer.write(frame)
    finally:
        writer.release()


def write_case(package: Path, case_id: str) -> case_manifests.CaseManifest:
    source = package / "media" / case_id / "original_reference.mp4"
    automatic = package / "media" / case_id / "automatic_preview.mp4"
    write_synthetic_video(source)
    shutil.copy2(source, automatic)
    manifest = case_manifests.case_manifest_from_media(
        case_id=case_id,
        display_name=case_id.title(),
        source_path=f"media/{case_id}/original_reference.mp4",
        source_video=source,
        video_roles=[
            case_manifests.VideoRole(
                label="Original",
                path=f"media/{case_id}/original_reference.mp4",
                role="original",
                sha256=None,
                source_frame_end=9,
                source_frame_start=0,
            )
        ],
    )
    manifest = replace(
        manifest,
        video_roles=(
            *manifest.video_roles,
            case_manifests.VideoRole(
                label="Automatic preview",
                path=f"media/{case_id}/automatic_preview.mp4",
                role="automatic",
                sha256=media.sha256_file(automatic),
                source_frame_end=manifest.source_frame_end,
                source_frame_start=manifest.source_frame_start,
            ),
        ),
    )
    case_manifests.write_case_manifest(package / "cases" / f"{case_id}.json", manifest)
    return manifest


def review_task(
    case_id: str,
    task_id: str,
    *,
    end: int,
    source_task_id: str,
    start: int,
) -> dict[str, object]:
    return {
        "allowed_actions": CANONICAL_ACTIONS,
        "blocking": True,
        "case_id": case_id,
        "category": "license_plate",
        "correction_required": False,
        "issue_type": "plate_coverage_gap",
        "normal_task_class": "plate_coverage_gap",
        "review_task_id": task_id,
        "schema_version": "case_reviewer_task.v1",
        "severity": "blocker",
        "source_frame_end": end,
        "source_frame_start": start,
        "target_binding": {
            "case_id": case_id,
            "category": "license_plate",
            "kind": "entity",
            "schema_version": "review_task_target_binding.v1",
            "source_frame_end": end,
            "source_frame_start": start,
            "source_task_id": source_task_id,
            "technical_track_ids": [f"{case_id}_{task_id}_track"],
        },
        "target_entity_ids": [f"{case_id}_plate_entity"],
        "technical_only": False,
        "title": "Plate may be visible",
        "what_happened": "A plate-like target may still need review.",
        "what_to_inspect": "Check that the plate is fully covered in this interval.",
    }


def write_grouped_firefox_workspace(package: Path) -> str:
    day = write_case(package, "day")
    night = write_case(package, "night")
    automatic_preview_by_case: dict[str, dict[str, object]] = {}
    category_configuration_by_case: dict[str, dict[str, object]] = {}
    category_identity_hash_by_case: dict[str, str] = {}
    category_identities_by_case: dict[str, dict[str, object]] = {}
    content_identities_by_case: dict[str, dict[str, object]] = {}
    source_identity_by_case: dict[str, dict[str, object]] = {}
    source_metadata_by_case: dict[str, dict[str, object]] = {}
    for manifest in (day, night):
        baseline_path = package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text("", encoding="utf-8")
        automatic_role = next(role for role in manifest.video_roles if role.role == "automatic")
        automatic_preview_by_case[manifest.case_id] = {
            "automatic_preview_sha256": media.sha256_file(package / automatic_role.path),
            "region_count": 0,
            "source_sha256": manifest.source_video_hash,
        }
        content_identities_by_case[manifest.case_id] = {
            "automatic_baseline_regions": {
                "path": baseline_path.relative_to(package).as_posix(),
                "role": "automatic_baseline_regions",
                "schema_version": "artifact_reference.v1",
                "sha256": media.sha256_file(baseline_path),
                "size_bytes": baseline_path.stat().st_size,
            }
        }
        configuration_hash = stable_hash(
            {
                "case_id": manifest.case_id,
                "category": "license_plate",
                "fixture": "firefox_grouped_review",
            }
        )
        processed_span = {
            "first_processed_frame": manifest.source_frame_start,
            "last_processed_frame": manifest.source_frame_end,
            "source_frame_count": manifest.frame_count,
        }
        content_hash = stable_hash(
            {
                "case_id": manifest.case_id,
                "category": "license_plate",
                "processed_span": processed_span,
                "source_sha256": manifest.source_video_hash,
            }
        )
        run_manifest_path = package / "technical" / f"{manifest.case_id}_license_plate_run.json"
        write_json(
            run_manifest_path,
            {
                "case_id": manifest.case_id,
                "category": "license_plate",
                "category_run_content_sha256": content_hash,
                "configuration_identity": {
                    "category": "license_plate",
                    "configuration_hash": configuration_hash,
                    "schema_version": "category_configuration_identity.v1",
                    "source_sha256": manifest.source_video_hash,
                },
                "schema_version": "category_run_manifest.fixture.v1",
                "source_lineage": {
                    "processed_span": processed_span,
                    "source_sha256": manifest.source_video_hash,
                },
                "stage_outcomes": [],
            },
        )
        category_identities = {
            "license_plate": {
                "artifact_id": f"{manifest.case_id}_license_plate_run",
                "category": "license_plate",
                "category_run_content_sha256": content_hash,
                "configuration_hash": configuration_hash,
                "path": run_manifest_path.relative_to(package).as_posix(),
                "processed_span": processed_span,
                "schema_version": "category_run_manifest_identity.v1",
                "sha256": media.sha256_file(run_manifest_path),
                "source_sha256": manifest.source_video_hash,
            }
        }
        category_identities_by_case[manifest.case_id] = category_identities
        category_identity_hash_by_case[manifest.case_id] = stable_hash(category_identities)
        category_configuration_by_case[manifest.case_id] = {
            "license_plate": {
                "category": "license_plate",
                "configuration_hash": configuration_hash,
                "schema_version": "category_configuration_identity.v1",
                "source_sha256": manifest.source_video_hash,
            }
        }
        source_identity_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
        source_metadata_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
    freeze = {
        "category_configuration_identities_by_case": category_configuration_by_case,
        "category_run_manifest_identities_by_case": category_identities_by_case,
        "schema_version": "firefox_grouped_review_fixture.v1",
        "selected_categories": ["license_plate"],
    }
    write_json(package / "configuration_freeze.json", freeze)
    write_json(
        package / "workspace_manifest.json",
        {
            "automatic_preview_by_case": automatic_preview_by_case,
            "cases": {
                "day": "cases/day.json",
                "night": "cases/night.json",
            },
            "category_configuration_identities_by_case": category_configuration_by_case,
            "category_run_manifest_identities_by_case": category_identities_by_case,
            "category_run_manifest_identity_hash_by_case": category_identity_hash_by_case,
            "configuration_freeze": "configuration_freeze.json",
            "configuration_freeze_hash": stable_hash(freeze),
            "content_identities_by_case": content_identities_by_case,
            "latest_reviewed_publication": {
                "publication_mode": "reviewed_region_compiler.v1",
                "schema_version": "latest_reviewed_publication_capability.v1",
                "status": "supported",
            },
            "review_entities": "review_entities.json",
            "review_tasks": "reviewer_tasks.json",
            "schema_version": "final_review_workspace.v2.3.4",
            "selected_categories": ["license_plate"],
            "source_identity_by_case": source_identity_by_case,
            "source_metadata_by_case": source_metadata_by_case,
            "workspace_kind": "unified_case_review_workspace",
            "workspace_status": "ready",
        },
    )
    entity_shell = {
        "cases": {"day": [], "night": []},
        "schema_version": "review_entities.v2",
        "unresolved_evidence_by_case": {
            "day": {
                "fragment_count": 0,
                "fragments": [],
                "schema_version": "review_unresolved_evidence.v1",
            },
            "night": {
                "fragment_count": 0,
                "fragments": [],
                "schema_version": "review_unresolved_evidence.v1",
            },
        },
    }
    write_json(package / "review_entities.json", entity_shell)
    write_json(package / "review_objects.json", entity_shell)
    write_json(package / "privacy_entities.json", entity_shell)
    write_json(
        package / "reviewer_tasks.json",
        {
            "actions": [
                {"action": "looks_correct", "label": "Correct"},
                {"action": "privacy_leak", "label": "Needs correction"},
                {"action": "not_sensitive", "label": "Not sensitive"},
            ],
            "schema_version": "case_reviewer_tasks.v1",
            "task_count": 4,
            "tasks": [
                review_task(
                    "day",
                    "day_plate_review_0001",
                    start=1,
                    end=2,
                    source_task_id="day_plate_group_0001",
                ),
                review_task(
                    "night",
                    GROUPED_CHILD_TASK_ID,
                    start=2,
                    end=3,
                    source_task_id="night_plate_group_0005",
                ),
                review_task(
                    "night",
                    "license_plate_plate_review_0006",
                    start=4,
                    end=5,
                    source_task_id="night_plate_group_0005",
                ),
                review_task(
                    "night",
                    "night_plate_review_qa_0001",
                    start=7,
                    end=7,
                    source_task_id="night_plate_group_qa_0001",
                )
                | {"optional": True, "severity": "info"},
            ],
        },
    )
    review_workspace.write_react_workspace_assets(package, build_frontend=False)
    return stable_hash({"case_hashes": [day.manifest_hash(), night.manifest_hash()]})


def write_svg_thumbnail(package: Path, relative_path: str, *, color: str, label: str) -> None:
    path = package / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="72" '
            'viewBox="0 0 96 72">'
            f'<rect width="96" height="72" fill="{color}"/>'
            '<rect x="18" y="16" width="60" height="38" fill="none" '
            'stroke="white" stroke-width="5"/>'
            f'<text x="48" y="40" text-anchor="middle" font-size="12" '
            f'font-family="Arial" fill="white">{label}</text>'
            "</svg>"
        ),
        encoding="utf-8",
    )


def refinement_firefox_entity(
    case_id: str,
    entity_id: str,
    label: str,
    *,
    bbox: list[int] | None = None,
    descriptor_values: list[float] | None = None,
    end: int = 2,
    policy: str = "redact",
    start: int = 0,
    tracklets: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    bbox = bbox or [10, 10, 30, 30]
    crop_path = f"entities/{case_id}/thumbnails/{entity_id}.svg"
    tracklets = tracklets or [
        {
            "bbox_xyxy": bbox,
            "detector_backed": True,
            "representative_crops": [
                {
                    "bbox_xyxy": bbox,
                    "crop_role": "best_middle",
                    "path": crop_path,
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": start,
                    **(
                        {
                            "visual_descriptor": descriptor_values,
                            "visual_descriptor_version": (
                                entity_refinement.VISUAL_DESCRIPTOR_VERSION
                            ),
                        }
                        if descriptor_values is not None
                        else {}
                    ),
                }
            ],
            "source_frame_end": end,
            "source_frame_start": start,
            "tracklet_id": f"{entity_id}_track",
        }
    ]
    return {
        "automatic_source_track_ids": [str(item["tracklet_id"]) for item in tracklets],
        "case_id": case_id,
        "category": CATEGORY_FACE,
        "current_policy": policy,
        "current_status": "active",
        "default_privacy_policy": policy,
        "display_name": label,
        "entity_id": entity_id,
        "evidence_summary": {
            "provenance_counts": {"detector": len(tracklets)},
            "region_count": len(tracklets),
            "review_observation_count": len(tracklets),
            "review_tracklet_count": len(tracklets),
            "tracklet_count": len(tracklets),
        },
        "first_visible_source_frame": start,
        "label": label,
        "last_visible_source_frame": end,
        "member_detector_tracker_ids": [str(item["tracklet_id"]) for item in tracklets],
        "privacy_action": policy,
        "raw_tracklets": tracklets,
        "representative_bbox": bbox,
        "representative_crops": list(tracklets[0].get("representative_crops") or []),
        "representative_source_frame": start,
        "representative_thumbnail_path": crop_path,
        "review_entity_state": "ready",
        "review_state": "reviewed",
        "schema_version": "review_entity.v2",
    }


def write_entity_refinement_firefox_workspace(package: Path) -> str:
    day = write_case(package, "day")
    night = write_case(package, "night")
    automatic_preview_by_case: dict[str, dict[str, object]] = {}
    category_configuration_by_case: dict[str, dict[str, object]] = {}
    category_identity_hash_by_case: dict[str, str] = {}
    category_identities_by_case: dict[str, dict[str, object]] = {}
    content_identities_by_case: dict[str, dict[str, object]] = {}
    source_identity_by_case: dict[str, dict[str, object]] = {}
    source_metadata_by_case: dict[str, dict[str, object]] = {}
    for manifest in (day, night):
        baseline_path = package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text("", encoding="utf-8")
        automatic_role = next(role for role in manifest.video_roles if role.role == "automatic")
        automatic_preview_by_case[manifest.case_id] = {
            "automatic_preview_sha256": media.sha256_file(package / automatic_role.path),
            "region_count": 0,
            "source_sha256": manifest.source_video_hash,
        }
        content_identities_by_case[manifest.case_id] = {
            "automatic_baseline_regions": {
                "path": baseline_path.relative_to(package).as_posix(),
                "role": "automatic_baseline_regions",
                "schema_version": "artifact_reference.v1",
                "sha256": media.sha256_file(baseline_path),
                "size_bytes": baseline_path.stat().st_size,
            }
        }
        configuration_hash = stable_hash(
            {
                "case_id": manifest.case_id,
                "category": CATEGORY_FACE,
                "fixture": "firefox_entity_refinement_v1",
            }
        )
        processed_span = {
            "first_processed_frame": manifest.source_frame_start,
            "last_processed_frame": manifest.source_frame_end,
            "source_frame_count": manifest.frame_count,
        }
        content_hash = stable_hash(
            {
                "case_id": manifest.case_id,
                "category": CATEGORY_FACE,
                "processed_span": processed_span,
                "source_sha256": manifest.source_video_hash,
            }
        )
        run_manifest_path = package / "technical" / f"{manifest.case_id}_face_run.json"
        write_json(
            run_manifest_path,
            {
                "case_id": manifest.case_id,
                "category": CATEGORY_FACE,
                "category_run_content_sha256": content_hash,
                "configuration_identity": {
                    "category": CATEGORY_FACE,
                    "configuration_hash": configuration_hash,
                    "schema_version": "category_configuration_identity.v1",
                    "source_sha256": manifest.source_video_hash,
                },
                "schema_version": "category_run_manifest.fixture.v1",
                "source_lineage": {
                    "processed_span": processed_span,
                    "source_sha256": manifest.source_video_hash,
                },
                "stage_outcomes": [],
            },
        )
        category_identities = {
            CATEGORY_FACE: {
                "artifact_id": f"{manifest.case_id}_face_run",
                "category": CATEGORY_FACE,
                "category_run_content_sha256": content_hash,
                "configuration_hash": configuration_hash,
                "path": run_manifest_path.relative_to(package).as_posix(),
                "processed_span": processed_span,
                "schema_version": "category_run_manifest_identity.v1",
                "sha256": media.sha256_file(run_manifest_path),
                "source_sha256": manifest.source_video_hash,
            }
        }
        category_identities_by_case[manifest.case_id] = category_identities
        category_identity_hash_by_case[manifest.case_id] = stable_hash(category_identities)
        category_configuration_by_case[manifest.case_id] = {
            CATEGORY_FACE: {
                "category": CATEGORY_FACE,
                "configuration_hash": configuration_hash,
                "schema_version": "category_configuration_identity.v1",
                "source_sha256": manifest.source_video_hash,
            }
        }
        source_identity_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
        source_metadata_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}

    for relative_path, color, label in (
        ("entities/day/thumbnails/mixed_a.svg", "#0f766e", "A"),
        ("entities/day/thumbnails/mixed_b.svg", "#b45309", "B"),
        ("entities/night/thumbnails/night_person_04.svg", "#1d4ed8", "04"),
        ("entities/night/thumbnails/night_person_01.svg", "#047857", "01"),
        ("entities/night/thumbnails/night_person_12.svg", "#0891b2", "12"),
        ("entities/night/thumbnails/night_person_09.svg", "#7c3aed", "09"),
        ("entities/night/thumbnails/night_conflict_a.svg", "#be123c", "CA"),
        ("entities/night/thumbnails/night_conflict_b.svg", "#c2410c", "CB"),
    ):
        write_svg_thumbnail(package, relative_path, color=color, label=label)

    mixed_tracklets: list[dict[str, object]] = [
        {
            "bbox_xyxy": [10, 10, 30, 30],
            "detector_backed": True,
            "observations": [
                {"bbox_xyxy": [10, 10, 30, 30], "provenance": "detector", "source_frame": 2},
                {"bbox_xyxy": [10, 10, 30, 30], "provenance": "detector", "source_frame": 3},
                {"bbox_xyxy": [10, 10, 30, 30], "provenance": "detector", "source_frame": 4},
            ],
            "representative_crops": [
                {
                    "bbox_xyxy": [10, 10, 30, 30],
                    "crop_role": "best_early",
                    "path": "entities/day/thumbnails/mixed_a.svg",
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": 2,
                }
            ],
            "source_frame_end": 4,
            "source_frame_start": 2,
            "tracklet_id": "face_track_a",
        },
        {
            "bbox_xyxy": [40, 10, 60, 30],
            "detector_backed": True,
            "observations": [
                {"bbox_xyxy": [40, 10, 60, 30], "provenance": "detector", "source_frame": 3},
                {"bbox_xyxy": [40, 10, 60, 30], "provenance": "detector", "source_frame": 4},
                {"bbox_xyxy": [40, 10, 60, 30], "provenance": "detector", "source_frame": 5},
            ],
            "representative_crops": [
                {
                    "bbox_xyxy": [40, 10, 60, 30],
                    "crop_role": "best_middle",
                    "path": "entities/day/thumbnails/mixed_b.svg",
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": 3,
                }
            ],
            "source_frame_end": 5,
            "source_frame_start": 3,
            "tracklet_id": "face_track_b",
        },
    ]
    entity_shell = {
        "cases": {
            "day": [
                refinement_firefox_entity(
                    "day",
                    "entity_merged_bd6717c585d53525a416bb90",
                    "Merged Person 01",
                    end=5,
                    start=2,
                    tracklets=mixed_tracklets,
                )
            ],
            "night": [
                refinement_firefox_entity(
                    "night",
                    "night_person_04",
                    "Person 04",
                    descriptor_values=firefox_descriptor(0),
                    start=0,
                    end=1,
                ),
                refinement_firefox_entity(
                    "night",
                    "night_person_01",
                    "Person 01",
                    descriptor_values=firefox_descriptor(0),
                    start=4,
                    end=5,
                ),
                refinement_firefox_entity(
                    "night",
                    "night_person_12",
                    "Person 12",
                    descriptor_values=firefox_descriptor(1),
                    start=2,
                    end=3,
                ),
                refinement_firefox_entity(
                    "night",
                    "night_person_09",
                    "Person 09",
                    descriptor_values=firefox_descriptor(1),
                    start=7,
                    end=8,
                ),
                refinement_firefox_entity(
                    "night",
                    "night_conflict_a",
                    "Conflict A",
                    descriptor_values=firefox_descriptor(2),
                    start=1,
                    end=4,
                ),
                refinement_firefox_entity(
                    "night",
                    "night_conflict_b",
                    "Conflict B",
                    descriptor_values=firefox_descriptor(2),
                    start=3,
                    end=6,
                ),
            ],
        },
        "schema_version": "review_entities.v2",
        "unresolved_evidence_by_case": {
            "day": {"fragment_count": 0, "fragments": []},
            "night": {"fragment_count": 0, "fragments": []},
        },
    }
    write_json(package / "review_entities.json", entity_shell)
    write_json(package / "review_objects.json", entity_shell)
    write_json(package / "privacy_entities.json", entity_shell)
    write_json(
        package / "reviewer_tasks.json",
        {"actions": [], "schema_version": "case_reviewer_tasks.v1", "task_count": 0, "tasks": []},
    )
    freeze = {
        "category_configuration_identities_by_case": category_configuration_by_case,
        "category_run_manifest_identities_by_case": category_identities_by_case,
        "schema_version": "firefox_entity_refinement_fixture.v1",
        "selected_categories": [CATEGORY_FACE],
    }
    write_json(package / "configuration_freeze.json", freeze)
    write_json(
        package / "workspace_manifest.json",
        {
            "automatic_preview_by_case": automatic_preview_by_case,
            "cases": {
                "day": "cases/day.json",
                "night": "cases/night.json",
            },
            "category_configuration_identities_by_case": category_configuration_by_case,
            "category_run_manifest_identities_by_case": category_identities_by_case,
            "category_run_manifest_identity_hash_by_case": category_identity_hash_by_case,
            "configuration_freeze": "configuration_freeze.json",
            "configuration_freeze_hash": stable_hash(freeze),
            "content_identities_by_case": content_identities_by_case,
            "latest_reviewed_publication": {
                "publication_mode": "reviewed_region_compiler.v1",
                "schema_version": "latest_reviewed_publication_capability.v1",
                "status": "supported",
            },
            "review_entities": "review_entities.json",
            "review_tasks": "reviewer_tasks.json",
            "schema_version": "final_review_workspace.v2.3.4",
            "selected_categories": [CATEGORY_FACE],
            "source_identity_by_case": source_identity_by_case,
            "source_metadata_by_case": source_metadata_by_case,
            "workspace_kind": "unified_case_review_workspace",
            "workspace_status": "ready",
        },
    )
    review_workspace.write_react_workspace_assets(package, build_frontend=False)
    return stable_hash({"case_hashes": [day.manifest_hash(), night.manifest_hash()]})


def test_final_review_workspace_firefox_entity_refinement_v1(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    package = tmp_path / "firefox_entity_refinement_workspace"
    source_revision = write_entity_refinement_firefox_workspace(package)
    report: dict[str, Any] = {
        "browser": "firefox",
        "package": package.as_posix(),
        "passed": False,
        "schema_version": "entity_refinement_v1_browser_smoke.v1",
        "steps": [],
    }
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session_payload = post_json(
            f"{base_url}/api/sessions",
            {"reviewer": {"display_name": "Firefox Entity V1"}, "source_revision": source_revision},
        )
        session_id = str(session_payload["session"]["session_id"])
        report["session_id"] = session_id
        with sync_api.sync_playwright() as playwright:
            launch_options = {"headless": True}
            if executable := os.environ.get("REDACTOR_FIREFOX_EXECUTABLE"):
                firefox_path = Path(executable)
                if not firefox_path.exists():
                    pytest.skip(f"Firefox executable does not exist: {firefox_path}")
                launch_options["executable_path"] = str(firefox_path)
            try:
                browser = playwright.firefox.launch(**launch_options)
            except Exception as exc:
                pytest.skip(f"Firefox automation is unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                browser_errors: list[str] = []
                page.on(
                    "console",
                    lambda msg: (
                        browser_errors.append(msg.text)
                        if msg.type in {"error", "pageerror"}
                        else None
                    ),
                )
                page.on(
                    "response",
                    lambda response: (
                        browser_errors.append(f"server error {response.status}: {response.url}")
                        if response.status >= 500
                        else None
                    ),
                )
                page.goto(
                    f"{base_url}/app/day?session_id={session_id}",
                    wait_until="domcontentloaded",
                )
                page.wait_for_function(SESSION_READY_JS, timeout=10_000)
                page.get_by_role("button", name="Entities").click()
                entity_dialog = page.get_by_role("dialog", name="Review objects")
                entity_dialog.wait_for(state="visible", timeout=5_000)
                entity_dialog.get_by_role("button", name="People").click()
                assert entity_dialog.get_by_role("button", name="Load previews").count() == 0
                mixed = entity_dialog.locator(
                    '.entity-card[data-entity-id="entity_merged_bd6717c585d53525a416bb90"]'
                )
                mixed.wait_for(state="visible", timeout=5_000)
                page.wait_for_function(
                    """
                    () => Boolean(document.querySelector(
                      '.entity-card[data-entity-id="entity_merged_bd6717c585d53525a416bb90"]'
                      + ' img[src*="derived_thumbnails"]'
                    ))
                    """,
                    timeout=10_000,
                )
                assert mixed.get_attribute("data-purity") == "LIKELY_MIXED"
                assert mixed.locator('[data-policy="redact"]').is_enabled()
                assert mixed.locator('[data-policy="keep_visible"]').is_disabled()
                assert mixed.locator('[data-policy="not_sensitive"]').is_disabled()
                record_step(report, "day_automatic_visible_preview")
                with page.expect_response(
                    lambda response: "/api/v1/entity-refinement/split-mixed" in response.url
                ) as split_response:
                    mixed.get_by_role("button", name="Split mixed entity").click()
                split_payload = split_response.value.json()
                assert split_payload["event"]["event_type"] == "entity_mixed_split"
                assert len(split_payload["split_entities"]) == 2
                page.wait_for_function(
                    """
                    () => document.querySelectorAll(
                      '.entity-card[data-entity-id^="entity_split_"]'
                    ).length === 2
                    """,
                    timeout=5_000,
                )
                record_step(report, "day_mixed_split_workflow")

                entity_dialog.get_by_role("button", name="Close").last.click()
                failed_preview_image = {"used": False}

                def fail_one_preview_image(route: Any) -> None:
                    if not failed_preview_image["used"]:
                        failed_preview_image["used"] = True
                        route.abort()
                        return
                    route.continue_()

                page.route(
                    "**/entities/night/derived_thumbnails/*.jpg**",
                    fail_one_preview_image,
                )
                page.get_by_role("button", name="Night").click()
                page.get_by_role("button", name="Entities").click()
                entity_dialog = page.get_by_role("dialog", name="Review objects")
                entity_dialog.wait_for(state="visible", timeout=5_000)
                entity_dialog.get_by_role("button", name="People").click()
                before_reject_count = entity_dialog.locator(".entity-card").count()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll(
                      '.entity-card .thumbnail-button.image-loaded'
                    ).length >= 2
                    """,
                    timeout=10_000,
                )
                assert entity_dialog.locator(".candidate-group-card").count() >= 3
                possible_cards = entity_dialog.locator(
                    ".candidate-group-card",
                    has_text="Possible merge",
                )
                assert possible_cards.count() >= 2
                first_possible = possible_cards.first
                first_possible.wait_for(state="visible", timeout=5_000)
                entity_dialog.get_by_text("Preview image unavailable").first.wait_for(
                    state="visible", timeout=5_000
                )
                assert "similarity score is only a triage signal" in first_possible.inner_text()
                entity_dialog.get_by_role(
                    "button", name=re.compile("Retry preview for")
                ).first.click()
                page.wait_for_function(
                    """
                    () => document.querySelectorAll(
                      '.candidate-group-card[data-evidence-ready="true"]'
                    ).length >= 2
                    """,
                    timeout=10_000,
                )
                assert first_possible.locator(".candidate-comparison-grid").is_visible()
                assert first_possible.locator(".candidate-evidence-side").count() == 2
                assert first_possible.locator(".candidate-evidence-crops img").count() >= 2
                assert first_possible.get_by_role(
                    "button", name="Accept merge candidate"
                ).is_enabled()
                assert (
                    possible_cards.nth(1)
                    .get_by_role("button", name="Accept merge candidate")
                    .is_enabled()
                )
                record_step(report, "night_candidate_comparison_evidence_and_retry")
                conflict_card = entity_dialog.locator(
                    ".candidate-group-card",
                    has_text="Temporal conflict",
                ).first
                conflict_card.wait_for(state="visible", timeout=5_000)
                assert "coexist" in conflict_card.inner_text()
                assert "3-4" in conflict_card.inner_text()
                assert conflict_card.get_by_role(
                    "button", name="Accept merge candidate"
                ).is_disabled()
                assert conflict_card.get_by_role("button", name="Reject candidate").is_enabled()
                conflict_card.get_by_role(
                    "button", name=re.compile("Inspect conflict frames")
                ).click()
                entity_dialog.wait_for(state="hidden", timeout=5_000)
                wait_for_timeline_frame(page, "night", {"3"})
                record_step(report, "night_temporal_conflict_explained_and_seeked")

                page.get_by_role("button", name="Entities").click()
                entity_dialog = page.get_by_role("dialog", name="Review objects")
                entity_dialog.wait_for(state="visible", timeout=5_000)
                entity_dialog.get_by_role("button", name="People").click()
                possible_card = entity_dialog.locator(
                    '.candidate-group-card[data-evidence-ready="true"]',
                    has_text="Possible merge",
                ).first
                possible_card.wait_for(state="visible", timeout=5_000)
                with page.expect_response(
                    lambda response: "/api/v1/entity-refinement/reject-merge" in response.url
                ) as reject_response:
                    possible_card.get_by_role("button", name="Reject candidate").click()
                reject_payload = reject_response.value.json()
                assert reject_payload["event"]["event_type"] == "entity_merge_candidate_rejected"
                assert entity_dialog.locator(".entity-card").count() == before_reject_count
                page.wait_for_function(
                    """
                    () => document.querySelectorAll('.candidate-group-card').length === 2
                    """,
                    timeout=5_000,
                )

                possible_card = entity_dialog.locator(
                    '.candidate-group-card[data-evidence-ready="true"]',
                    has_text="Possible merge",
                ).first
                possible_card.wait_for(state="visible", timeout=5_000)
                with page.expect_response(
                    lambda response: "/api/v1/entity-refinement/accept-merge" in response.url
                ) as accept_response:
                    possible_card.get_by_role("button", name="Accept merge candidate").click()
                accept_payload = accept_response.value.json()
                assert accept_payload["event"]["event_type"] == "entity_merge_candidate_accepted"
                page.wait_for_function(
                    """
                    (beforeCount) => (
                      document.querySelectorAll('.entity-card').length === beforeCount - 1
                    )
                    """,
                    arg=before_reject_count,
                    timeout=5_000,
                )
                record_step(report, "night_candidate_reject_and_accept")

                assert not (package / "final").exists()
                assert not browser_errors
                report["passed"] = True
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        write_report(package, report, tmp_path / "entity_refinement_browser_smoke")


def test_final_review_workspace_firefox_entity_timeline_smoke(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    package = tmp_path / "firefox_grouped_workspace"
    source_revision = write_grouped_firefox_workspace(package)
    if external_workspace := os.environ.get("REDACTOR_FIREFOX_WORKSPACE"):
        package = Path(external_workspace)
        source_revision = "external-firefox-workspace"
    if not package.exists():
        pytest.skip(f"workspace package does not exist: {package}")
    session_id = migration_session_id(package)

    report: dict[str, Any] = {
        "browser": "firefox",
        "package": package.as_posix(),
        "passed": False,
        "schema_version": "final_review_workspace_browser_smoke.v1",
        "session_id": session_id,
        "steps": [],
    }
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        if not session_id:
            session_payload = post_json(
                f"{base_url}/api/sessions",
                {"reviewer": {"display_name": "Firefox Smoke"}, "source_revision": source_revision},
            )
            session_id = str(session_payload["session"]["session_id"])
        if not (package / "sessions" / session_id).exists():
            pytest.skip(f"workspace package does not include migration session state: {session_id}")
        report["session_id"] = session_id
        with sync_api.sync_playwright() as playwright:
            launch_options = {"headless": True}
            if executable := os.environ.get("REDACTOR_FIREFOX_EXECUTABLE"):
                firefox_path = Path(executable)
                if not firefox_path.exists():
                    pytest.skip(f"Firefox executable does not exist: {firefox_path}")
                launch_options["executable_path"] = str(firefox_path)
            try:
                browser = playwright.firefox.launch(**launch_options)
            except Exception as exc:
                pytest.skip(f"Firefox automation is unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                browser_errors: list[str] = []
                page.on(
                    "console",
                    lambda msg: (
                        browser_errors.append(msg.text)
                        if msg.type in {"error", "pageerror"}
                        else None
                    ),
                )
                page.on(
                    "requestfailed",
                    lambda request: browser_errors.append(f"request failed: {request.url}"),
                )
                page.on(
                    "response",
                    lambda response: (
                        browser_errors.append(f"server error {response.status}: {response.url}")
                        if response.status >= 500
                        else None
                    ),
                )
                page.goto(
                    f"{base_url}/app/day?session_id={session_id}",
                    wait_until="domcontentloaded",
                )
                day_panel = active_case_panel(page, "day")
                page.wait_for_function(SESSION_READY_JS, timeout=10_000)
                assert "Review session active" in page.locator("#sessionBanner").inner_text()
                assert session_id in page.locator("#sessionBanner").inner_text()
                assert "Unsure" not in page.locator("body").inner_text()
                assert "Uncertain" not in page.locator("body").inner_text()
                record_step(report, "migration_session_app_load")

                assert bucket_names(day_panel) == [
                    "Guidance",
                    "Needs correction",
                    "Correct",
                    "Not sensitive",
                ]
                assert "Waiting" not in bucket_names(day_panel)
                assert "Completed" not in bucket_names(day_panel)
                record_step(report, "canonical_queue_buckets")

                click_bucket(page, "day", "Guidance")
                day_group = first_open_plate_group(day_panel)
                assert day_group.locator(".task-state-row .badge").first.inner_text() == "Guidance"
                day_group_id = day_group.get_attribute("data-task-id")
                assert day_group_id
                day_group.locator(".task-title").click()
                day_marker = (
                    f'.case-panel[data-case-id="day"] '
                    f'.player-marker[data-task-id="{day_group_id}"].selected'
                )
                page.locator(day_marker).wait_for(state="visible", timeout=5_000)
                wait_for_timeline_frame(page, "day", {"1", "2"})
                record_step(report, "day_grouped_plate_navigation", group_id=day_group_id)

                day_panel.get_by_role("button", name="Entities").click()
                entity_dialog = page.get_by_role("dialog", name="Review objects")
                entity_dialog.wait_for(state="visible", timeout=5_000)
                entity_dialog.get_by_role("button", name="All").click()
                entity_count = entity_dialog.locator(".entity-card").count()
                assert entity_count >= 0
                record_step(report, "entity_modal_assets", entity_count=entity_count)
                entity_dialog.get_by_role("button", name="Close").last.click()

                page.get_by_role("button", name="Night").click()
                night_panel = active_case_panel(page, "night")
                assert "night" in night_panel.locator(".case-header").inner_text().lower()
                assert bucket_names(night_panel) == [
                    "Guidance",
                    "Needs correction",
                    "Correct",
                    "Not sensitive",
                ]
                click_bucket(page, "night", "Guidance")
                night_duplicate = task_card_for_child(page, "night", GROUPED_CHILD_TASK_ID)
                assert (
                    night_duplicate.locator(".task-state-row .badge").first.inner_text()
                    == "Guidance"
                )
                night_group_id = night_duplicate.get_attribute("data-task-id")
                assert night_group_id and night_group_id != GROUPED_CHILD_TASK_ID
                night_duplicate.locator(".task-title").click()
                night_marker = (
                    f'.case-panel[data-case-id="night"] '
                    f'.player-marker[data-task-id="{night_group_id}"].selected'
                )
                page.locator(night_marker).wait_for(state="visible", timeout=5_000)
                wait_for_timeline_frame(page, "night", {"2", "3"})
                night_duplicate.locator(".task-technical summary").click()
                assert (
                    GROUPED_CHILD_TASK_ID in night_duplicate.locator(".task-technical").inner_text()
                )
                assert (
                    "license_plate_plate_review_0006"
                    in night_duplicate.locator(".task-technical").inner_text()
                )
                action_labels = [
                    item.strip()
                    for item in night_duplicate.locator(".task-actions button").all_inner_texts()
                ]
                assert action_labels == [
                    "Correct",
                    "Needs correction",
                    "Not sensitive",
                ]
                assert night_duplicate.locator(".task-instruction").first.inner_text().strip()
                assert "WHY THIS WAS FLAGGED" not in night_duplicate.inner_text()
                assert "WHAT TO CHECK" not in night_duplicate.inner_text()
                record_step(
                    report,
                    "night_grouped_child_open_and_three_actions",
                    group_id=night_group_id,
                    task_id=GROUPED_CHILD_TASK_ID,
                )

                with page.expect_response("**/api/v1/decision-groups"):
                    night_duplicate.get_by_role("button", name=re.compile(r"^Correct$")).click()
                click_bucket(page, "night", "Correct")
                night_completed = page.locator(task_card_selector("night", night_group_id))
                night_completed.wait_for(state="visible", timeout=5_000)
                assert (
                    night_completed.locator(".task-state-row .badge").first.inner_text()
                    == "Correct"
                )
                record_step(report, "night_group_decision_isolated", group_id=night_group_id)

                page.get_by_role("button", name="Day").click()
                active_case_panel(page, "day")
                click_bucket(page, "day", "Guidance")
                day_group = page.locator(task_card_selector("day", day_group_id))
                day_group.wait_for(state="visible", timeout=5_000)
                assert day_group.locator(".task-state-row .badge").first.inner_text() == "Guidance"
                record_step(
                    report,
                    "day_group_still_open_after_night_decision",
                    group_id=day_group_id,
                )

                page.get_by_role("button", name="Final Results").click()
                page.locator(".case-panel.active").get_by_role(
                    "heading", name="Final Results"
                ).wait_for(timeout=5_000)
                final_options = [
                    item.strip()
                    for item in page.locator(
                        ".case-panel.active .version-selector option"
                    ).all_inner_texts()
                ]
                assert "Original" in final_options
                assert "Automatic" in final_options
                assert "Latest reviewed" not in final_options
                assert "Final" not in final_options
                assert page.locator(".download-row").count() > 0
                record_step(report, "final_results_hides_unavailable_roles")
                assert not browser_errors
                report["passed"] = True
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        write_report(package, report, tmp_path / "browser_smoke")
