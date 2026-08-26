from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import (
    case_manifests,
    entity_refinement,
    media,
    mixed_review_workspace,
    privacy_entities,
    review_api_contract,
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get_json(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def descriptor(seed: int) -> list[float]:
    values = [0.0, 0.0, 0.0]
    values[seed % len(values)] = 1.0
    return values


def observation(frame: int, bbox: list[int], *, score: float = 0.9) -> dict[str, Any]:
    return {
        "bbox_xyxy": bbox,
        "detector_score": score,
        "provenance": "detector",
        "source_frame": frame,
    }


def case_manifest(case_id: str, *, source_hash: str) -> case_manifests.CaseManifest:
    return case_manifests.CaseManifest(
        case_id=case_id,
        display_name=case_id.title(),
        duration_seconds=1.0,
        fps="10/1",
        frame_count=10,
        height=64,
        source_frame_end=9,
        source_frame_start=0,
        source_path=f"media/{case_id}/original.mp4",
        source_video_hash=source_hash,
        status=case_manifests.CASE_STATUS_READY,
        video_roles=(
            case_manifests.VideoRole(
                label="Original",
                path=f"media/{case_id}/original.mp4",
                role="original",
                source_frame_end=9,
                source_frame_start=0,
            ),
            case_manifests.VideoRole(
                label="Automatic preview",
                path=f"media/{case_id}/automatic.mp4",
                role="automatic",
                source_frame_end=9,
                source_frame_start=0,
            ),
        ),
        width=96,
    )


def refinement_entity(
    case_id: str,
    entity_id: str,
    label: str,
    *,
    category: str = CATEGORY_FACE,
    descriptor_values: list[float] | None = None,
    policy: str = privacy_entities.POLICY_REDACT,
    start: int = 0,
    end: int = 2,
    bbox: list[int] | None = None,
    tracklets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bbox = bbox or [10, 10, 30, 30]
    tracklets = tracklets or [
        {
            "bbox_xyxy": bbox,
            "detector_backed": True,
            "observations": [observation(start, bbox)],
            "representative_crops": [
                {
                    "bbox_xyxy": bbox,
                    "crop_role": "best_middle",
                    "path": f"entities/{case_id}/thumbnails/{entity_id}.jpg",
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
        "category": category,
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
        "representative_crops": tracklets[0].get("representative_crops", []),
        "representative_source_frame": start,
        "representative_thumbnail_path": f"entities/{case_id}/thumbnails/{entity_id}.jpg",
        "review_entity_state": "ready",
        "review_state": "reviewed",
        "schema_version": "review_entity.v2",
    }


def source_backed_review_entity(
    case_id: str,
    entity_id: str,
    label: str,
    *,
    category: str = CATEGORY_FACE,
    policy: str = privacy_entities.POLICY_REDACT,
    render_region_sources: list[dict[str, Any]] | None = None,
    start: int = 0,
    end: int = 2,
    track_id: str | None = None,
) -> dict[str, Any]:
    track_id = track_id or f"{entity_id}_track"
    records = list(render_region_sources or [])
    return {
        "automatic_source_track_ids": [track_id],
        "case_id": case_id,
        "category": category,
        "current_policy": policy,
        "current_status": "active",
        "default_privacy_policy": policy,
        "display_name": label,
        "entity_id": entity_id,
        "evidence_summary": {
            "provenance_counts": {"detector": len(records)} if records else {},
            "region_count": len(records),
            "review_observation_count": len(records),
            "review_tracklet_count": 1,
            "tracklet_count": 1,
        },
        "first_visible_source_frame": start,
        "label": label,
        "last_visible_source_frame": end,
        "member_detector_tracker_ids": [track_id],
        "privacy_action": policy,
        "review_entity_state": "ready",
        "review_state": "reviewed",
        "schema_version": "review_entity.v2",
        "source_lineage": {
            "render_region_sources": records,
            "schema_version": "mixed_review_entity_source_lineage.v1",
            "technical_id": track_id,
            "technical_track_ids": [track_id],
        },
    }


def write_refinement_workspace(
    package: Path,
    *,
    extra_entities_by_case: dict[str, list[dict[str, Any]]] | None = None,
) -> Path:
    package.mkdir(parents=True)
    manifests: dict[str, case_manifests.CaseManifest] = {}
    for case_id in ("day", "night"):
        for relative_path in (
            f"media/{case_id}/original.mp4",
            f"media/{case_id}/automatic.mp4",
        ):
            media_path = package / relative_path
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(b"entity-refinement-fixture-media")
        manifests[case_id] = case_manifest(
            case_id,
            source_hash=media.sha256_file(package / f"media/{case_id}/original.mp4"),
        )
    for manifest in manifests.values():
        case_manifests.write_case_manifest(
            package / "cases" / f"{manifest.case_id}.json",
            manifest,
        )
        review_finalization.write_privacy_region_records(
            package / "regions" / f"{manifest.case_id}_current_regions.jsonl",
            [
                PrivacyRegionRecord(
                    approval_state=APPROVAL_ACCEPTED,
                    bbox_xyxy=(10, 10, 30, 30),
                    category=CATEGORY_FACE,
                    object_id=f"{manifest.case_id}_baseline_track",
                    provenance="detector",
                    render_eligible=True,
                    source_frame=0,
                    source_record={"track_id": f"{manifest.case_id}_baseline_track"},
                )
            ],
        )
    automatic_preview_by_case: dict[str, dict[str, Any]] = {}
    category_configuration_by_case: dict[str, dict[str, Any]] = {}
    category_identity_hash_by_case: dict[str, str] = {}
    category_identities_by_case: dict[str, dict[str, Any]] = {}
    content_identities_by_case: dict[str, dict[str, Any]] = {}
    source_identity_by_case: dict[str, dict[str, Any]] = {}
    source_metadata_by_case: dict[str, dict[str, Any]] = {}
    for manifest in manifests.values():
        automatic_role = next(role for role in manifest.video_roles if role.role == "automatic")
        baseline_path = package / "regions" / f"{manifest.case_id}_current_regions.jsonl"
        automatic_preview_by_case[manifest.case_id] = {
            "automatic_preview_sha256": media.sha256_file(package / automatic_role.path),
            "region_count": 1,
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
        source_identity_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
        source_metadata_by_case[manifest.case_id] = {"sha256": manifest.source_video_hash}
        configuration_hash = stable_hash(
            {
                "case_id": manifest.case_id,
                "category": CATEGORY_FACE,
                "fixture": "entity_refinement_v1",
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
    mixed_tracklets = [
        {
            "bbox_xyxy": [10, 10, 30, 30],
            "detector_backed": True,
            "observations": [
                observation(2, [10, 10, 30, 30]),
                observation(3, [10, 10, 30, 30]),
                observation(4, [10, 10, 30, 30]),
            ],
            "representative_crops": [
                {
                    "bbox_xyxy": [10, 10, 30, 30],
                    "crop_role": "best_early",
                    "path": "entities/day/thumbnails/mixed_a.jpg",
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": 2,
                    "visual_descriptor": descriptor(0),
                    "visual_descriptor_version": entity_refinement.VISUAL_DESCRIPTOR_VERSION,
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
                observation(3, [40, 10, 60, 30]),
                observation(4, [40, 10, 60, 30]),
                observation(5, [40, 10, 60, 30]),
            ],
            "representative_crops": [
                {
                    "bbox_xyxy": [40, 10, 60, 30],
                    "crop_role": "best_middle",
                    "path": "entities/day/thumbnails/mixed_b.jpg",
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": 3,
                    "visual_descriptor": descriptor(1),
                    "visual_descriptor_version": entity_refinement.VISUAL_DESCRIPTOR_VERSION,
                }
            ],
            "source_frame_end": 5,
            "source_frame_start": 3,
            "tracklet_id": "face_track_b",
        },
    ]
    entities = {
        "day": [
            refinement_entity(
                "day",
                "entity_merged_bd6717c585d53525a416bb90",
                "Merged Person 01",
                tracklets=mixed_tracklets,
            )
        ],
        "night": [
            refinement_entity(
                "night",
                "night_person_04",
                "Person 04",
                descriptor_values=descriptor(0),
                start=0,
                end=1,
            ),
            refinement_entity(
                "night",
                "night_person_01",
                "Person 01",
                descriptor_values=descriptor(0),
                start=4,
                end=5,
            ),
            refinement_entity(
                "night",
                "night_keep_visible",
                "Person 09",
                descriptor_values=descriptor(1),
                policy=privacy_entities.POLICY_KEEP_VISIBLE,
                start=7,
                end=8,
            ),
        ],
    }
    for case_id, extra_entities in dict(extra_entities_by_case or {}).items():
        entities.setdefault(case_id, []).extend(extra_entities)
    write_json(
        package / "review_entities.json",
        {
            "cases": entities,
            "schema_version": "review_entities.v2",
            "unresolved_evidence_by_case": {
                "day": {"fragment_count": 0},
                "night": {"fragment_count": 0},
            },
        },
    )
    write_json(
        package / "reviewer_tasks.json",
        {"actions": [], "schema_version": "case_reviewer_tasks.v1", "task_count": 0, "tasks": []},
    )
    freeze = {
        "category_configuration_identities_by_case": category_configuration_by_case,
        "category_run_manifest_identities_by_case": category_identities_by_case,
        "schema_version": "entity_refinement_fixture.freeze.v1",
        "selected_categories": [CATEGORY_FACE],
    }
    write_json(package / "configuration_freeze.json", freeze)
    write_json(
        package / "workspace_manifest.json",
        {
            "automatic_preview_by_case": automatic_preview_by_case,
            "cases": {"day": "cases/day.json", "night": "cases/night.json"},
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
    (package / "index.html").write_text("ok", encoding="utf-8")
    session = review_sessions.create_review_session(
        package_root=package,
        reviewer=review_sessions.ReviewerIdentity(display_name="Entity Reviewer"),
        source_revision=None,
        configuration_freeze_hash=stable_hash(freeze),
        case_manifest_hashes={
            case_id: manifest.manifest_hash() for case_id, manifest in manifests.items()
        },
        source_hashes={case_id: f"{case_id}-source-sha" for case_id in manifests},
        workspace_identity_hashes=review_workspace.workspace_identity_hashes(package),
    )
    return review_sessions.session_directory(package, session.session_id)


def write_valid_preview_video(package: Path, case_id: str = "day") -> None:
    source_path = package / f"media/{case_id}/original.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(source_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (96, 64),
    )
    try:
        for index in range(10):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (20, 20 + index, 40)
            frame[12:28, 42:70] = (240, 240, 240)
            frame[34:50, 14:34] = (30, 220, 180)
            writer.write(frame)
    finally:
        writer.release()
    manifest_path = package / "cases" / f"{case_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_video_hash"] = media.sha256_file(source_path)
    write_json(manifest_path, manifest)


def create_bound_refinement_session(package: Path) -> Path:
    freeze = json.loads((package / "configuration_freeze.json").read_text(encoding="utf-8"))
    manifests = {
        case_path.stem: case_manifests.load_case_manifest(case_path)
        for case_path in sorted((package / "cases").glob("*.json"))
    }
    session = review_sessions.create_review_session(
        package_root=package,
        reviewer=review_sessions.ReviewerIdentity(display_name="Entity Reviewer"),
        source_revision=None,
        configuration_freeze_hash=stable_hash(freeze),
        case_manifest_hashes={
            case_id: manifest.manifest_hash() for case_id, manifest in manifests.items()
        },
        source_hashes={
            case_id: str(manifest.source_video_hash)
            for case_id, manifest in manifests.items()
            if manifest.source_video_hash
        },
        workspace_identity_hashes=review_workspace.workspace_identity_hashes(package),
    )
    return review_sessions.session_directory(package, session.session_id)


def test_production_sources_have_no_known_entity_or_audit_artifact_logic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    production_files = [
        *sorted((repo_root / "src/redactor").glob("*.py")),
        *sorted((repo_root / "frontend/src").rglob("*.ts")),
        *sorted((repo_root / "frontend/src").rglob("*.tsx")),
    ]
    production_text = "\n".join(path.read_text(encoding="utf-8") for path in production_files)

    assert "entity_merged_bd6717c585d53525a416bb90" not in production_text
    assert "entity_merged_940643ae728fa471665fff2c" not in production_text
    assert "entity_purity_similarity_audit" not in production_text


def test_entity_refinement_classifies_cannot_link_and_disables_reveal() -> None:
    topology = [
        {
            "bbox_xyxy": [10, 10, 30, 30],
            "detector_backed": True,
            "observations": [observation(11, [10, 10, 30, 30])],
            "representative_crops": [
                {
                    "bbox_xyxy": [10, 10, 30, 30],
                    "path": "entities/day/thumbnails/a.jpg",
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": 11,
                }
            ],
            "source_frame_end": 12,
            "source_frame_start": 10,
            "tracklet_id": "a",
        },
        {
            "bbox_xyxy": [55, 10, 75, 30],
            "detector_backed": True,
            "observations": [observation(11, [55, 10, 75, 30])],
            "representative_crops": [
                {
                    "bbox_xyxy": [55, 10, 75, 30],
                    "path": "entities/day/thumbnails/b.jpg",
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": 11,
                }
            ],
            "source_frame_end": 13,
            "source_frame_start": 11,
            "tracklet_id": "b",
        },
    ]
    outcomes = []
    for entity_id in (
        "entity_merged_bd6717c585d53525a416bb90",
        "entity_merged_940643ae728fa471665fff2c",
        "entity_randomized_merge_fixture_6f97",
    ):
        entity = refinement_entity(
            "day",
            entity_id,
            "Merged Person 01",
            tracklets=[dict(item) for item in topology],
        )

        enriched = entity_refinement.enrich_entity(entity)
        outcomes.append(
            (
                enriched["purity_classification"],
                enriched["entity_refinement"]["cannot_link_conflicts"],
            )
        )

        assert enriched["purity_classification"] == entity_refinement.PURITY_LIKELY_MIXED
        assert enriched["entity_refinement"]["cannot_link_conflicts"]
        assert entity_refinement.validate_policy_action(
            enriched,
            privacy_entities.POLICY_KEEP_VISIBLE,
        )
        assert entity_refinement.validate_policy_action(
            enriched,
            privacy_entities.POLICY_REDACT,
        ) is None
    assert [item[0] for item in outcomes] == [entity_refinement.PURITY_LIKELY_MIXED] * 3
    assert outcomes[0][1][0]["raw_child_ids"] == outcomes[-1][1][0]["raw_child_ids"]


def test_conservative_policy_inheritance_prefers_redact() -> None:
    assert entity_refinement.most_protective_policy(
        [
            privacy_entities.POLICY_KEEP_VISIBLE,
            privacy_entities.POLICY_REVIEW_REQUIRED,
            privacy_entities.POLICY_REDACT,
        ]
    ) == privacy_entities.POLICY_REDACT


def test_thumbnail_ranking_binds_bbox_crops_to_source_frames() -> None:
    crops = mixed_review_workspace.ranked_source_crop_candidates(
        [
            (12, (10, 10, 18, 18)),
            (2, (20, 10, 52, 26)),
            (7, (8, 8, 22, 22)),
            (20, (40, 12, 66, 30)),
        ]
    )

    assert [crop["crop_role"] for crop in crops] == ["best_early", "best_middle", "best_late"]
    assert [crop["source_frame"] for crop in crops] == [2, 12, 20]
    assert crops[0]["bbox_xyxy"] == [20, 10, 52, 26]
    assert crops[0]["bbox_area"] == 512


def test_no_reliable_crop_classifies_low_quality() -> None:
    entity = refinement_entity(
        "day",
        "tiny_face_fragment",
        "Tiny fragment",
        tracklets=[
            {
                "detector_backed": False,
                "observations": [observation(12, [11, 11, 18, 18])],
                "representative_crops": [],
                "source_frame_end": 12,
                "source_frame_start": 12,
                "tracklet_id": "tiny_fragment_track",
            }
        ],
    )
    entity.pop("representative_bbox", None)
    entity.pop("representative_source_frame", None)
    entity.pop("representative_thumbnail_path", None)
    entity["representative_crops"] = []
    entity["evidence_summary"] = {
        "provenance_counts": {},
        "region_count": 3,
        "review_observation_count": 3,
        "review_tracklet_count": 1,
        "tracklet_count": 1,
    }

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_LOW_QUALITY
    assert enriched["thumbnail_unavailable_reason"] == "no_reliable_representative_crop"
    assert not entity_refinement.policy_action_availability(enriched)["actions"][
        privacy_entities.POLICY_KEEP_VISIBLE
    ]["allowed"]


def test_multi_tracklet_missing_geometry_is_unverified_not_fabricated_mixed() -> None:
    entity = refinement_entity(
        "day",
        "multi_tracklet_no_geometry",
        "Unverified group",
        tracklets=[
            {
                "detector_backed": False,
                "representative_crops": [],
                "source_frame_end": 12,
                "source_frame_start": 10,
                "tracklet_id": "raw_without_geometry_a",
            },
            {
                "detector_backed": False,
                "representative_crops": [],
                "source_frame_end": 13,
                "source_frame_start": 11,
                "tracklet_id": "raw_without_geometry_b",
            },
        ],
    )

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_UNVERIFIED
    assert enriched["entity_refinement"]["cannot_link_conflicts"] == []
    assert entity_refinement.validate_policy_action(
        enriched,
        privacy_entities.POLICY_NOT_SENSITIVE,
    )


def test_no_resolvable_raw_descendant_is_unverified() -> None:
    entity = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "current_status": "active",
        "default_privacy_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_without_raw_descendant",
        "label": "No raw descendant",
        "privacy_action": privacy_entities.POLICY_REDACT,
        "schema_version": "review_entity.v2",
    }

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_UNVERIFIED
    assert enriched["entity_refinement"]["child_tracklet_count"] == 0
    assert enriched["entity_refinement"]["reasons"] == ["no_resolvable_raw_tracklet"]


def test_source_backed_singleton_without_observations_is_unverified() -> None:
    entity = source_backed_review_entity(
        "day",
        "source_backed_no_observations",
        "Source backed without observations",
    )

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_UNVERIFIED
    assert enriched["entity_refinement"]["child_tracklet_count"] == 1
    assert enriched["entity_refinement"]["reasons"] == [
        "raw_tracklet_missing_durable_geometry"
    ]
    assert entity_refinement.validate_policy_action(
        enriched,
        privacy_entities.POLICY_KEEP_VISIBLE,
    )
    assert entity_refinement.validate_policy_action(
        enriched,
        privacy_entities.POLICY_NOT_SENSITIVE,
    )
    assert entity_refinement.validate_policy_action(
        enriched,
        privacy_entities.POLICY_REDACT,
    ) is None


def test_source_backed_singleton_with_malformed_geometry_is_unverified() -> None:
    entity = source_backed_review_entity(
        "day",
        "source_backed_malformed_geometry",
        "Source backed malformed geometry",
        render_region_sources=[
            {
                "bbox_xyxy": ["bad", 10, 30, 30],
                "provenance": "detector",
                "source_frame": 4,
                "source_record": {"metrics": {"score": 0.95}},
            }
        ],
    )

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_UNVERIFIED
    assert enriched["entity_refinement"]["reasons"] == [
        "raw_tracklet_missing_durable_geometry"
    ]


def test_singleton_raw_id_with_unresolved_geometry_is_unverified() -> None:
    entity = refinement_entity(
        "day",
        "raw_id_geometry_lookup_fails",
        "Raw id lookup fails",
        tracklets=[
            {
                "detector_backed": True,
                "observations": [
                    {"bbox_xyxy": ["bad", 10, 30, 30], "source_frame": 5}
                ],
                "source_frame_end": 5,
                "source_frame_start": 5,
                "tracklet_id": "raw_geometry_lookup_fails",
            }
        ],
    )

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_UNVERIFIED
    assert enriched["raw_tracklets"][0]["tracklet_id"] == "raw_geometry_lookup_fails"
    assert enriched["raw_tracklets"][0]["observations"] == []


def test_source_backed_singleton_with_valid_geometry_remains_singleton_pure() -> None:
    entity = source_backed_review_entity(
        "day",
        "source_backed_valid_geometry",
        "Source backed valid geometry",
        render_region_sources=[
            {
                "bbox_xyxy": [12, 10, 32, 30],
                "provenance": "detector",
                "source_frame": 4,
                "source_record": {"metrics": {"score": 0.95}},
            }
        ],
    )

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["purity_classification"] == entity_refinement.PURITY_PURE
    assert enriched["entity_refinement"]["reasons"] == ["single_tracklet"]
    assert enriched["raw_tracklets"][0]["observations"] == [
        {
            "bbox_xyxy": [12, 10, 32, 30],
            "detector_score": 0.95,
            "provenance": "detector",
            "source_frame": 4,
        }
    ]


def test_missing_geometry_singleton_result_is_entity_id_independent() -> None:
    results = []
    for entity_id in ("missing_geometry_alpha", "missing_geometry_beta"):
        enriched = entity_refinement.enrich_entity(
            source_backed_review_entity(
                "day",
                entity_id,
                f"Missing geometry {entity_id}",
            )
        )
        results.append(
            (
                enriched["purity_classification"],
                enriched["entity_refinement"]["reasons"],
                enriched["policy_action_availability"]["actions"][
                    privacy_entities.POLICY_KEEP_VISIBLE
                ]["allowed"],
            )
        )

    assert results == [
        (
            entity_refinement.PURITY_UNVERIFIED,
            ["raw_tracklet_missing_durable_geometry"],
            False,
        ),
        (
            entity_refinement.PURITY_UNVERIFIED,
            ["raw_tracklet_missing_durable_geometry"],
            False,
        ),
    ]


def test_recursive_historical_merge_lineage_resolves_raw_children(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    child_a = refinement_entity(
        "day",
        "raw_face_child_a",
        "Raw A",
        start=10,
        end=12,
        bbox=[10, 10, 30, 30],
    )
    child_b = refinement_entity(
        "day",
        "raw_face_child_b",
        "Raw B",
        start=11,
        end=13,
        bbox=[55, 10, 75, 30],
    )
    child_a["raw_tracklets"][0]["observations"] = [observation(11, [10, 10, 30, 30])]
    child_b["raw_tracklets"][0]["observations"] = [observation(11, [55, 10, 75, 30])]
    index = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    index["cases"]["day"] = [child_a, child_b]
    write_json(package / "review_entities.json", index)
    primary = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "current_status": "active",
        "default_privacy_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_historical_merge_without_embedded_children",
        "first_visible_source_frame": 10,
        "label": "Historical merged person",
        "last_visible_source_frame": 13,
        "privacy_action": privacy_entities.POLICY_REDACT,
        "schema_version": "review_entity.v2",
    }
    review_sessions.append_event(
        session_dir,
        event_type="entities_merged",
        actor={"display_name": "Historical Reviewer"},
        case_id="day",
        task_id=None,
        before_state={"entities": [child_a, child_b]},
        after_state={
            "primary": primary,
            "retired_entity_ids": ["raw_face_child_a", "raw_face_child_b"],
            "source_entity_ids": ["raw_face_child_a", "raw_face_child_b"],
        },
        note="Historical merge fixture.",
        application_hashes={},
    )

    enriched = review_workspace.find_active_review_entity(
        package,
        session_dir=session_dir,
        case_id="day",
        entity_id=primary["entity_id"],
        lazy_crops=False,
    )

    assert enriched is not None
    assert enriched["purity_classification"] == entity_refinement.PURITY_LIKELY_MIXED
    assert {
        item["tracklet_id"]
        for item in enriched["entity_refinement"]["raw_tracklets"]
    } == {"raw_face_child_a_track", "raw_face_child_b_track"}


def test_historical_replayed_singleton_with_valid_geometry_stays_pure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    child = refinement_entity(
        "day",
        "entity_split_history_valid_singleton",
        "History split valid singleton",
        start=6,
        end=6,
        bbox=[18, 14, 38, 34],
    )
    parent = refinement_entity(
        "day",
        "entity_split_history_parent",
        "History split parent",
        tracklets=child["raw_tracklets"],
    )
    review_sessions.append_event(
        session_dir,
        event_type="entity_tracklet_split",
        actor={"display_name": "Historical Reviewer"},
        case_id="day",
        task_id=None,
        before_state={"entity": parent, "raw_tracklets": child["raw_tracklets"]},
        after_state={
            "parent": parent | {"current_status": "split"},
            "source_entity_ids": [parent["entity_id"]],
            "split_entities": [child],
            "tracklet_id": child["raw_tracklets"][0]["tracklet_id"],
        },
        note="Historical singleton split fixture.",
        application_hashes={},
    )
    state = review_workspace.review_working_state.derive_working_state(
        package,
        session_dir,
    )
    presentation = review_workspace.presentation_working_state(
        package,
        session_dir,
        state,
        lazy_crops=False,
    )

    enriched = dict(presentation["object_policy_overrides"])[child["entity_id"]]

    assert enriched["purity_classification"] == entity_refinement.PURITY_PURE
    assert enriched["entity_refinement"]["reasons"] == ["single_tracklet"]
    assert enriched["entity_refinement"]["raw_tracklets"][0]["observations"]


def test_raw_descendant_aliases_are_deduplicated() -> None:
    entity = refinement_entity(
        "day",
        "entity_alias_dedupe",
        "Alias group",
        tracklets=[
            {
                "raw_source_id": "immutable_raw_a",
                "source_frame_end": 5,
                "source_frame_start": 5,
                "tracklet_id": "track_alias_a",
            },
            {
                "raw_source_id": "immutable_raw_a",
                "source_frame_end": 5,
                "source_frame_start": 5,
                "tracklet_id": "track_alias_b",
            },
        ],
    )

    enriched = entity_refinement.enrich_entity(entity)

    assert enriched["entity_refinement"]["child_tracklet_count"] == 1


def test_explicit_preview_generation_uses_existing_source_geometry_and_raw_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    del session_dir
    write_valid_preview_video(package)
    entity = refinement_entity(
        "day",
        "plate_entity_lazy_crop",
        "Plate 01",
        category=CATEGORY_PLATE,
        bbox=[42, 12, 70, 28],
        tracklets=[
            {
                "bbox_xyxy": [42, 12, 70, 28],
                "detector_backed": True,
                "observations": [
                    observation(2, [42, 12, 70, 28]),
                    observation(4, [42, 12, 70, 28]),
                    observation(7, [42, 12, 70, 28]),
                ],
                "representative_crops": [],
                "source_frame_end": 7,
                "source_frame_start": 2,
                "tracklet_id": "plate_track_01",
            }
        ],
    )
    entity["representative_crops"] = []
    entity.pop("representative_thumbnail_path", None)
    real_capture = entity_refinement.cv.VideoCapture
    calls = {"opens": 0, "reads": 0}

    class CountingCapture:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["opens"] += 1
            self.capture = real_capture(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.capture, name)

        def read(self) -> Any:
            calls["reads"] += 1
            return self.capture.read()

        def release(self) -> None:
            self.capture.release()

    monkeypatch.setattr(entity_refinement.cv, "VideoCapture", CountingCapture)

    metadata_only = entity_refinement.enrich_entity(entity, package_root=package)
    enriched = entity_refinement.prepare_entity_previews(entity=entity, package_root=package)

    assert metadata_only["preview_status"]["status"] == entity_refinement.PREVIEW_NOT_PREPARED
    assert metadata_only["representative_crops"] == []
    crops = enriched["representative_crops"]
    assert [crop["crop_role"] for crop in crops] == ["best_early", "best_middle", "best_late"]
    assert {crop["source_frame"] for crop in crops} == {2, 4, 7}
    assert all(crop["bbox_xyxy"] == [42, 12, 70, 28] for crop in crops)
    assert {crop["raw_tracklet_id"] for crop in crops} == {"plate_track_01"}
    assert all((package / crop["path"]).exists() for crop in crops)
    assert all(crop["path"].startswith("entities/day/derived_thumbnails/") for crop in crops)
    assert calls == {"opens": 1, "reads": 3}

    calls.update({"opens": 0, "reads": 0})
    cached = entity_refinement.prepare_entity_previews(entity=entity, package_root=package)

    assert cached["representative_crops"] == crops
    assert calls == {"opens": 0, "reads": 0}


def test_routine_read_paths_do_not_open_source_video(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    large_tracklets = [
        {
            "bbox_xyxy": [10, 10, 30, 30],
            "detector_backed": True,
            "observations": [observation(index % 10, [10, 10, 30, 30])],
            "representative_crops": [],
            "source_frame_end": index % 10,
            "source_frame_start": index % 10,
            "tracklet_id": f"raw_large_{index}",
        }
        for index in range(120)
    ]
    index = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    index["cases"]["day"].append(
        refinement_entity(
            "day",
            "entity_many_raw_tracklets",
            "Many tracklets",
            tracklets=large_tracklets,
        )
    )
    write_json(package / "review_entities.json", index)
    session_dir = create_bound_refinement_session(package)

    def fail_capture(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("routine read opened source video")

    monkeypatch.setattr(entity_refinement.cv, "VideoCapture", fail_capture)

    summary = review_api_contract.workspace_summary(package)
    enriched_index = entity_refinement.enrich_entity_index(
        package,
        json.loads((package / "review_entities.json").read_text(encoding="utf-8")),
        session_events=review_sessions.load_events(session_dir),
    )
    state = review_workspace.presentation_working_state(
        package,
        session_dir,
        review_workspace.review_working_state.derive_working_state(package, session_dir),
    )

    many = next(
        entity
        for entity in enriched_index["cases"]["day"]
        if entity["entity_id"] == "entity_many_raw_tracklets"
    )
    assert summary["entities"]["cases"]["day"]
    assert state["presentation_enrichment"]["entity_refinement"] is True
    assert many["representative_crops"] == []
    assert many["preview_status"]["status"] == entity_refinement.PREVIEW_NOT_PREPARED

    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for url in (
            f"{base_url}/api/v1/workspace/summary",
            f"{base_url}/api/v1/entities?session_id={session_dir.name}",
            f"{base_url}/api/session-state?session_id={session_dir.name}",
        ):
            status, payload = get_json(url)
            assert status == 200
            assert payload
    finally:
        server.shutdown()
        server.server_close()


def test_merged_preview_crops_carry_each_descendant_raw_identity(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    write_refinement_workspace(package)
    write_valid_preview_video(package)
    tracklets = [
        {
            "bbox_xyxy": [42, 12, 70, 28],
            "detector_backed": True,
            "observations": [observation(frame, [42, 12, 70, 28])],
            "representative_crops": [],
            "source_frame_end": frame,
            "source_frame_start": frame,
            "tracklet_id": tracklet_id,
        }
        for frame, tracklet_id in [(2, "raw_early"), (4, "raw_middle"), (7, "raw_late")]
    ]
    entity = refinement_entity(
        "day",
        "entity_merged_preview_identity",
        "Merged preview identity",
        tracklets=tracklets,
    )
    entity["representative_crops"] = []

    prepared = entity_refinement.prepare_entity_previews(package, entity)

    crops = prepared["representative_crops"]
    assert [crop["raw_tracklet_id"] for crop in crops] == [
        "raw_early",
        "raw_middle",
        "raw_late",
    ]
    raw_lookup = {
        (
            tracklet["tracklet_id"],
            item["source_frame"],
            tuple(item["bbox_xyxy"]),
        )
        for tracklet in tracklets
        for item in tracklet["observations"]
    }
    assert {
        (
            crop["raw_tracklet_id"],
            crop["source_frame"],
            tuple(crop["bbox_xyxy"]),
        )
        for crop in crops
    } <= raw_lookup


def test_preview_endpoint_is_explicit_bounded_and_cacheable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    write_valid_preview_video(package)
    tracklets = [
        {
            "bbox_xyxy": [42, 12, 70, 28],
            "detector_backed": True,
            "observations": [observation(frame, [42, 12, 70, 28])],
            "representative_crops": [],
            "source_frame_end": frame,
            "source_frame_start": frame,
            "tracklet_id": tracklet_id,
        }
        for frame, tracklet_id in [
            (2, "endpoint_raw_a"),
            (4, "endpoint_raw_b"),
            (7, "endpoint_raw_c"),
        ]
    ]
    index = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    index["cases"]["day"].append(
        refinement_entity(
            "day",
            "entity_endpoint_preview",
            "Endpoint preview",
            tracklets=tracklets,
        )
    )
    index["cases"]["day"][-1]["representative_crops"] = []
    write_json(package / "review_entities.json", index)
    session_dir = create_bound_refinement_session(package)
    real_capture = entity_refinement.cv.VideoCapture
    calls = {"opens": 0, "reads": 0}

    class CountingCapture:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["opens"] += 1
            self.capture = real_capture(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.capture, name)

        def read(self) -> Any:
            calls["reads"] += 1
            return self.capture.read()

        def release(self) -> None:
            self.capture.release()

    monkeypatch.setattr(entity_refinement.cv, "VideoCapture", CountingCapture)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = {
        "actor": {"display_name": "Entity Reviewer"},
        "case_id": "day",
        "entity_id": "entity_endpoint_preview",
        "session_id": session_dir.name,
    }
    try:
        status, first = post_json(f"{base_url}/api/v1/entity-refinement/previews", payload)
        assert status == 200
        assert first["preview_status"]["status"] == entity_refinement.PREVIEW_READY
        assert [crop["raw_tracklet_id"] for crop in first["representative_crops"]] == [
            "endpoint_raw_a",
            "endpoint_raw_b",
            "endpoint_raw_c",
        ]
        assert calls == {"opens": 1, "reads": 3}
        before_events = review_sessions.load_events(session_dir)

        calls.update({"opens": 0, "reads": 0})
        status, second = post_json(f"{base_url}/api/v1/entity-refinement/previews", payload)
        assert status == 200
        assert second["representative_crops"] == first["representative_crops"]
        assert calls == {"opens": 0, "reads": 0}
        assert review_sessions.load_events(session_dir) == before_events
    finally:
        server.shutdown()
        server.server_close()


def test_historical_merge_preview_resolves_raw_descendant_identity(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    write_valid_preview_video(package)
    child_a = refinement_entity(
        "day",
        "raw_preview_child_a",
        "Raw A",
        tracklets=[
            {
                "bbox_xyxy": [42, 12, 70, 28],
                "detector_backed": True,
                "observations": [observation(2, [42, 12, 70, 28])],
                "representative_crops": [],
                "source_frame_end": 2,
                "source_frame_start": 2,
                "tracklet_id": "raw_preview_track_a",
            }
        ],
    )
    child_b = refinement_entity(
        "day",
        "raw_preview_child_b",
        "Raw B",
        tracklets=[
            {
                "bbox_xyxy": [42, 12, 70, 28],
                "detector_backed": True,
                "observations": [observation(7, [42, 12, 70, 28])],
                "representative_crops": [],
                "source_frame_end": 7,
                "source_frame_start": 7,
                "tracklet_id": "raw_preview_track_b",
            }
        ],
    )
    index = json.loads((package / "review_entities.json").read_text(encoding="utf-8"))
    index["cases"]["day"] = [child_a, child_b]
    write_json(package / "review_entities.json", index)
    primary = {
        "case_id": "day",
        "category": CATEGORY_FACE,
        "current_policy": privacy_entities.POLICY_REDACT,
        "current_status": "active",
        "default_privacy_policy": privacy_entities.POLICY_REDACT,
        "entity_id": "entity_historical_preview_merge",
        "first_visible_source_frame": 2,
        "label": "Historical preview merge",
        "last_visible_source_frame": 7,
        "privacy_action": privacy_entities.POLICY_REDACT,
        "schema_version": "review_entity.v2",
    }
    review_sessions.append_event(
        session_dir,
        event_type="entities_merged",
        actor={"display_name": "Historical Reviewer"},
        case_id="day",
        task_id=None,
        before_state={"entities": [child_a, child_b]},
        after_state={
            "primary": primary,
            "retired_entity_ids": ["raw_preview_child_a", "raw_preview_child_b"],
            "source_entity_ids": ["raw_preview_child_a", "raw_preview_child_b"],
        },
        note="Historical preview merge fixture.",
        application_hashes={},
    )
    active = review_workspace.find_active_review_entity(
        package,
        session_dir=session_dir,
        case_id="day",
        entity_id=primary["entity_id"],
        lazy_crops=False,
    )
    assert active is not None

    prepared = entity_refinement.prepare_entity_previews(
        package,
        active,
        entity_index=json.loads((package / "review_entities.json").read_text(encoding="utf-8")),
        session_events=review_sessions.load_events(session_dir),
    )

    assert {
        crop["raw_tracklet_id"] for crop in prepared["representative_crops"]
    } == {"raw_preview_track_a", "raw_preview_track_b"}


def test_split_child_preview_preserves_raw_tracklet_identity(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    write_refinement_workspace(package)
    write_valid_preview_video(package)
    parent = refinement_entity(
        "day",
        "entity_split_preview_parent",
        "Split preview parent",
        tracklets=[
            {
                "bbox_xyxy": [42, 12, 70, 28],
                "detector_backed": True,
                "observations": [observation(4, [42, 12, 70, 28])],
                "representative_crops": [],
                "source_frame_end": 4,
                "source_frame_start": 4,
                "tracklet_id": "split_preview_raw_track",
            }
        ],
    )
    parent["representative_crops"] = []
    child = review_workspace.split_entities_for_tracklets(
        parent,
        entity_refinement.raw_tracklets_for_entity(parent),
    )[0]

    prepared = entity_refinement.prepare_entity_previews(package, child)

    assert [crop["raw_tracklet_id"] for crop in prepared["representative_crops"]] == [
        "split_preview_raw_track"
    ]


def test_unresolved_preview_provenance_returns_no_crop(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    write_refinement_workspace(package)
    write_valid_preview_video(package)
    entity = refinement_entity(
        "day",
        "entity_unresolved_preview",
        "Unresolved preview",
        tracklets=[
            {
                "bbox_xyxy": [42, 12, 70, 28],
                "detector_backed": True,
                "observations": [observation(2, [42, 12, 70, 28])],
                "representative_crops": [],
                "source_frame_end": 2,
                "source_frame_start": 2,
                "tracklet_id": "",
            }
        ],
    )
    entity["representative_crops"] = []
    entity["automatic_source_track_ids"] = []
    entity["member_detector_tracker_ids"] = []

    prepared = entity_refinement.prepare_entity_previews(package, entity)

    assert prepared["representative_crops"] == []
    assert prepared["preview_status"]["status"] == entity_refinement.PREVIEW_NO_RELIABLE_GEOMETRY


def test_visual_descriptor_and_candidate_selectivity_are_content_based(tmp_path: Path) -> None:
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)
    blue = np.zeros((32, 32, 3), dtype=np.uint8)
    blue[:, :] = (255, 0, 0)
    red_descriptor = entity_refinement.visual_descriptor_from_image(red)
    blue_descriptor = entity_refinement.visual_descriptor_from_image(blue)
    assert red_descriptor is not None
    assert blue_descriptor is not None
    assert entity_refinement.cosine_similarity(
        tuple(red_descriptor),
        tuple(red_descriptor),
    ) > entity_refinement.cosine_similarity(tuple(red_descriptor), tuple(blue_descriptor))

    entities = [
        refinement_entity(
            "night",
            "night_face_a",
            "Person A",
            descriptor_values=descriptor(0),
            start=0,
            end=1,
        ),
        refinement_entity(
            "night",
            "night_face_b",
            "Person B",
            descriptor_values=descriptor(0),
            start=4,
            end=5,
        ),
        refinement_entity(
            "night",
            "night_face_c",
            "Person C",
            descriptor_values=descriptor(1),
            start=8,
            end=9,
        ),
        refinement_entity(
            "night",
            "night_plate_a",
            "Plate A",
            category=CATEGORY_PLATE,
            descriptor_values=descriptor(0),
            start=0,
            end=1,
        ),
        refinement_entity(
            "night",
            "night_plate_b",
            "Plate B",
            category=CATEGORY_PLATE,
            descriptor_values=descriptor(0),
            start=4,
            end=5,
        ),
    ]

    candidates = entity_refinement.advisory_candidate_pairs(
        tmp_path,
        entities,
        case_id="night",
    )

    assert [
        candidate["entity_ids"]
        for candidate in candidates
        if candidate["classification"] == entity_refinement.CANDIDATE_POSSIBLE
    ] == [["night_face_a", "night_face_b"]]
    assert all(candidate["category"] == CATEGORY_FACE for candidate in candidates)


def test_temporal_conflict_dominates_visual_similarity(tmp_path: Path) -> None:
    left = refinement_entity(
        "night",
        "overlap_face_a",
        "Person A",
        descriptor_values=descriptor(0),
        start=0,
        end=5,
    )
    right = refinement_entity(
        "night",
        "overlap_face_b",
        "Person B",
        descriptor_values=descriptor(0),
        start=3,
        end=7,
    )

    candidates = entity_refinement.advisory_candidate_pairs(
        tmp_path,
        [left, right],
        case_id="night",
    )

    assert len(candidates) == 1
    assert candidates[0]["classification"] == entity_refinement.CANDIDATE_REJECT_TEMPORAL_CONFLICT


def test_refinement_endpoints_are_append_only_and_case_isolated(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    actor = {"display_name": "Entity Reviewer"}
    try:
        baseline_state = review_workspace.review_working_state.derive_working_state(
            package,
            session_dir,
        )
        night_before = {
            entity_id: entity
            for entity_id, entity in dict(baseline_state["object_policy_overrides"]).items()
            if dict(entity).get("case_id") == "night"
        }
        status, reveal = post_json(
            f"{base_url}/api/v1/object-policy",
            {
                "actor": actor,
                "case_id": "day",
                "entity_id": "entity_merged_bd6717c585d53525a416bb90",
                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                "session_id": session_dir.name,
            },
        )
        assert status == 409
        assert "disabled" in reveal["error"]

        status, split = post_json(
            f"{base_url}/api/v1/entity-refinement/split-mixed",
            {
                "actor": actor,
                "case_id": "day",
                "entity_id": "entity_merged_bd6717c585d53525a416bb90",
                "session_id": session_dir.name,
            },
        )
        assert status == 200
        assert split["event"]["event_type"] == "entity_mixed_split"
        state_after_split = split["working_state"]
        day_overrides = dict(state_after_split["object_policy_overrides"])
        assert day_overrides["entity_merged_bd6717c585d53525a416bb90"]["current_status"] == "split"
        assert {child["member_detector_tracker_ids"][0] for child in split["split_entities"]} == {
            "face_track_a",
            "face_track_b",
        }
        assert all(
            child["current_policy"] == privacy_entities.POLICY_REDACT
            for child in split["split_entities"]
        )
        raw_state_after_split = review_workspace.review_working_state.derive_working_state(
            package,
            session_dir,
        )
        night_after = {
            entity_id: entity
            for entity_id, entity in dict(
                raw_state_after_split["object_policy_overrides"]
            ).items()
            if dict(entity).get("case_id") == "night"
        }
        assert night_after == night_before
        assert not [
            event
            for event in review_sessions.load_events(session_dir)
            if event["event_type"] == "entity_mixed_split" and event.get("case_id") == "night"
        ]
        assert [
                entity
                for entity in day_overrides.values()
                if dict(entity).get("case_id") == "day"
            ]

        summary_before_candidate = entity_refinement.enrich_entity_index(
            package,
            json.loads((package / "review_entities.json").read_text(encoding="utf-8")),
            session_events=review_sessions.load_events(session_dir),
        )
        night_candidates = [
            candidate
            for entity in summary_before_candidate["cases"]["night"]
            for candidate in list(entity.get("advisory_candidate_matches") or [])
            if candidate["classification"] == entity_refinement.CANDIDATE_POSSIBLE
        ]
        assert night_candidates
        candidate = night_candidates[0]
        status, reject = post_json(
            f"{base_url}/api/v1/entity-refinement/reject-merge",
            {
                "actor": actor,
                "candidate_id": "candidate_supplied_noncanonical_fixture",
                "case_id": "night",
                "entity_ids": list(reversed(candidate["entity_ids"])),
                "session_id": session_dir.name,
            },
        )
        assert status == 200
        assert reject["event"]["event_type"] == "entity_merge_candidate_rejected"
        assert reject["event"]["after_state"]["candidate_id"] == candidate["candidate_id"]
        assert (
            reject["event"]["after_state"]["supplied_candidate_id"]
            == "candidate_supplied_noncanonical_fixture"
        )
        summary_after_reject = entity_refinement.enrich_entity_index(
            package,
            json.loads((package / "review_entities.json").read_text(encoding="utf-8")),
            session_events=review_sessions.load_events(session_dir),
        )
        assert candidate["candidate_id"] not in {
            item["candidate_id"]
            for entity in summary_after_reject["cases"]["night"]
            for item in list(entity.get("advisory_candidate_matches") or [])
        }
    finally:
        server.shutdown()
        server.server_close()


def test_missing_geometry_singleton_policy_routes_fail_closed(tmp_path: Path) -> None:
    missing_entity = source_backed_review_entity(
        "day",
        "entity_missing_geometry_policy_route",
        "Missing geometry policy route",
        policy=privacy_entities.POLICY_REVIEW_REQUIRED,
    )
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(
        package,
        extra_entities_by_case={"day": [missing_entity]},
    )
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    actor = {"display_name": "Entity Reviewer"}
    base_payload = {
        "actor": actor,
        "case_id": "day",
        "entity_id": missing_entity["entity_id"],
        "session_id": session_dir.name,
    }
    try:
        status, keep_visible = post_json(
            f"{base_url}/api/v1/object-policy",
            base_payload | {"policy": privacy_entities.POLICY_KEEP_VISIBLE},
        )
        assert status == 409
        assert "disabled" in keep_visible["error"]

        status, not_sensitive = post_json(
            f"{base_url}/api/v1/object-policy",
            base_payload | {"policy": privacy_entities.POLICY_NOT_SENSITIVE},
        )
        assert status == 409
        assert "disabled" in not_sensitive["error"]

        status, bulk_keep_visible = post_json(
            f"{base_url}/api/v1/bulk-object-policy",
            {
                "actor": actor,
                "case_id": "day",
                "entity_ids": [missing_entity["entity_id"]],
                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                "session_id": session_dir.name,
            },
        )
        assert status == 409
        assert "disabled" in bulk_keep_visible["error"]
        assert not [
            event
            for event in review_sessions.load_events(session_dir)
            if event["event_type"]
            in {"bulk_entity_policy_changed", "entity_policy_changed"}
        ]

        status, redact = post_json(
            f"{base_url}/api/v1/object-policy",
            base_payload | {"policy": privacy_entities.POLICY_REDACT},
        )
        assert status == 200
        assert redact["event"]["event_type"] == "entity_policy_changed"
        assert redact["entity"]["current_policy"] == privacy_entities.POLICY_REDACT
    finally:
        server.shutdown()
        server.server_close()


def test_split_tracklet_out_removes_child_and_repeated_split_fails_closed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = {
        "actor": {"display_name": "Entity Reviewer"},
        "case_id": "day",
        "entity_id": "entity_merged_bd6717c585d53525a416bb90",
        "session_id": session_dir.name,
        "tracklet_id": "face_track_b",
    }
    try:
        status, split = post_json(
            f"{base_url}/api/v1/entity-refinement/split-tracklet",
            payload,
        )
        assert status == 200
        assert split["event"]["event_type"] == "entity_tracklet_split"
        parent = dict(split["working_state"]["object_policy_overrides"][payload["entity_id"]])
        assert parent["current_status"] == "active"
        assert parent["automatic_source_track_ids"] == ["face_track_a"]
        assert {
            item["tracklet_id"]
            for item in parent["entity_refinement"]["raw_tracklets"]
        } == {"face_track_a"}
        assert split["split_entities"][0]["member_detector_tracker_ids"] == ["face_track_b"]
        status, repeated = post_json(
            f"{base_url}/api/v1/entity-refinement/split-tracklet",
            payload,
        )
        assert status == 409
        assert "tracklet not found" in repeated["error"]
        events = [
            event
            for event in review_sessions.load_events(session_dir)
            if event["event_type"] == "entity_tracklet_split"
        ]
        assert len(events) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_accept_candidate_does_not_auto_merge_before_human_event(tmp_path: Path) -> None:
    package = tmp_path / "workspace"
    session_dir = write_refinement_workspace(package)
    baseline_events = review_sessions.load_events(session_dir)
    baseline_state = review_workspace.review_working_state.derive_working_state(
        package,
        session_dir,
    )
    baseline_projection_hash = baseline_state["active_projection_hash"]
    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        index = entity_refinement.enrich_entity_index(
            package,
            json.loads((package / "review_entities.json").read_text(encoding="utf-8")),
            session_events=baseline_events,
        )
        candidate = next(
            item
            for entity in index["cases"]["night"]
            for item in list(entity.get("advisory_candidate_matches") or [])
            if item["classification"] == entity_refinement.CANDIDATE_POSSIBLE
        )
        pre_events = review_sessions.load_events(session_dir)
        assert not [
            event for event in pre_events if event["event_type"].startswith("entity_")
        ]
        state_after_suggestions = review_workspace.review_working_state.derive_working_state(
            package,
            session_dir,
        )
        assert state_after_suggestions["active_projection_hash"] == baseline_projection_hash
        assert review_sessions.load_events(session_dir) == baseline_events
        assert baseline_state["object_policy_overrides"]

        status, accepted = post_json(
            f"{base_url}/api/v1/entity-refinement/accept-merge",
            {
                "actor": {"display_name": "Entity Reviewer"},
                "candidate_id": candidate["candidate_id"],
                "case_id": "night",
                "entity_ids": candidate["entity_ids"],
                "session_id": session_dir.name,
            },
        )
        assert status == 200
        assert accepted["event"]["event_type"] == "entity_merge_candidate_accepted"
        assert accepted["resulting_policy"] == privacy_entities.POLICY_REDACT
        events = review_sessions.load_events(session_dir)
        assert [
            event["event_type"]
            for event in events
            if event["event_type"].startswith("entity_")
        ] == ["entity_merge_candidate_accepted"]
        state = accepted["working_state"]
        assert accepted["merged_entity"]["entity_id"] in state["object_policy_overrides"]
        assert all(
            dict(state["object_policy_overrides"][entity_id]).get("current_status") == "merged"
            for entity_id in candidate["entity_ids"]
        )
    finally:
        server.shutdown()
        server.server_close()
