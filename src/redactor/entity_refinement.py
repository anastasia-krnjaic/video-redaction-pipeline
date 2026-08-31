from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import privacy_entities
from redactor.privacy_records import CATEGORY_FACE, stable_hash

PURITY_PURE = "PURE"
PURITY_LIKELY_MIXED = "LIKELY_MIXED"
PURITY_LOW_QUALITY = "LOW_QUALITY"
PURITY_UNVERIFIED = "UNVERIFIED"
PURITY_STATES = {
    PURITY_PURE,
    PURITY_LIKELY_MIXED,
    PURITY_LOW_QUALITY,
    PURITY_UNVERIFIED,
}

CANDIDATE_POSSIBLE = "POSSIBLE_CANDIDATE"
CANDIDATE_REJECT_TEMPORAL_CONFLICT = "REJECT_TEMPORAL_CONFLICT"
CANDIDATE_REJECT_VISUAL_MISMATCH = "REJECT_VISUAL_MISMATCH"
CANDIDATE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

TRACKLET_INTERVAL_EXPLICIT = "explicit_tracklet_interval"
TRACKLET_INTERVAL_ENTITY_SPAN = "entity_span_fallback"
TRACKLET_INTERVAL_SOURCE_BACKED = "source_backed_tracklet"

THUMBNAIL_ALGORITHM_VERSION = "entity_refinement_lazy_crop.v1"
DERIVED_PREVIEW_METADATA_VERSION = "entity_refinement_preview_cache.v1"
VISUAL_DESCRIPTOR_VERSION = "opencv_hsv_luma_gradient.v1"

PREVIEW_READY = "READY"
PREVIEW_NOT_PREPARED = "NOT_PREPARED"
PREVIEW_NO_RELIABLE_GEOMETRY = "NO_RELIABLE_GEOMETRY"

POLICY_RANK = {
    privacy_entities.POLICY_NOT_SENSITIVE: 0,
    privacy_entities.POLICY_KEEP_VISIBLE: 0,
    privacy_entities.POLICY_REVIEW_REQUIRED: 1,
    privacy_entities.POLICY_REDACT: 2,
}


def most_protective_policy(policies: Any) -> str:
    values = [
        str(policy)
        for policy in list(policies or [])
        if str(policy) in privacy_entities.POLICIES
    ]
    if not values:
        return privacy_entities.POLICY_REVIEW_REQUIRED
    return max(values, key=lambda policy: (POLICY_RANK.get(policy, -1), policy))


def enrich_entity_index(
    package_root: Path,
    entity_index: dict[str, Any],
    *,
    session_events: list[dict[str, Any]] | None = None,
    lazy_crops: bool = False,
) -> dict[str, Any]:
    output = dict(entity_index)
    cases: dict[str, list[dict[str, Any]]] = {}
    rejected = rejected_candidate_ids(session_events or [])
    for case_id, entities in dict(entity_index.get("cases") or {}).items():
        enriched = [
            enrich_entity(
                dict(entity),
                package_root=package_root,
                entity_index=entity_index,
                session_events=session_events,
                lazy_crops=lazy_crops,
            )
            for entity in list(entities or [])
            if isinstance(entity, dict)
        ]
        pairs = advisory_candidate_pairs(
            package_root,
            enriched,
            case_id=str(case_id),
            rejected_candidate_ids=rejected,
        )
        by_entity: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            for entity_id in list(pair.get("entity_ids") or []):
                by_entity.setdefault(str(entity_id), []).append(pair)
        cases[str(case_id)] = [
            entity | {"advisory_candidate_matches": by_entity.get(entity["entity_id"], [])}
            for entity in enriched
        ]
    output["cases"] = cases
    output["entity_refinement"] = {
        "advisory_only": True,
        "candidate_count": sum(
            len(entity.get("advisory_candidate_matches") or [])
            for entities in cases.values()
            for entity in entities
        )
        // 2,
        "purity_counts": purity_counts(cases),
        "schema_version": "entity_refinement_index.v1",
    }
    return output


def enrich_entity(
    entity: dict[str, Any],
    *,
    package_root: Path | None = None,
    entity_index: dict[str, Any] | None = None,
    session_events: list[dict[str, Any]] | None = None,
    lazy_crops: bool = False,
) -> dict[str, Any]:
    context = lineage_context(entity_index, session_events)
    tracklets = raw_tracklets_for_entity(
        entity,
        package_root=package_root,
        entity_index=entity_index,
        session_events=session_events,
        lazy_crops=False,
        _context=context,
    )
    crops = representative_crops(
        entity,
        package_root=package_root,
        tracklets=tracklets,
        prepare_missing=lazy_crops,
    )
    preview_status = preview_status_for_tracklets(tracklets, crops)
    purity = classify_entity_purity(entity | {"raw_tracklets": tracklets})
    enriched = dict(entity)
    enriched["raw_tracklets"] = tracklets
    enriched["representative_crops"] = crops
    enriched["preview_status"] = preview_status
    if crops:
        enriched.pop("thumbnail_unavailable_reason", None)
        enriched["representative_bbox"] = crops[0].get("bbox_xyxy")
        enriched["representative_source_frame"] = crops[0].get("source_frame")
        enriched["representative_thumbnail_path"] = crops[0].get("path")
    else:
        enriched.setdefault("thumbnail_unavailable_reason", "no_reliable_representative_crop")
    enriched["purity_classification"] = purity["purity"]
    enriched["entity_refinement"] = {
        **dict(enriched.get("entity_refinement") or {}),
        **purity,
        "advisory_similarity_only": True,
        "raw_lineage_resolution": {
            "schema_version": "entity_raw_lineage_resolution.v1",
            "source": "append_only_events_and_source_backed_entity_registry",
        },
        "raw_tracklets": tracklets,
    }
    enriched["policy_action_availability"] = policy_action_availability(enriched)
    if purity["purity"] == PURITY_LIKELY_MIXED:
        enriched["mixed_warning"] = "MIXED - contains conflicting simultaneous tracklets"
    return enriched


def enrich_working_state_entities(
    package_root: Path,
    state: dict[str, Any],
    *,
    entity_index: dict[str, Any],
    session_events: list[dict[str, Any]],
    lazy_crops: bool = False,
) -> dict[str, Any]:
    output = dict(state)
    output["object_policy_overrides"] = {
        str(entity_id): enrich_entity(
            dict(entity),
            package_root=package_root,
            entity_index=entity_index,
            session_events=session_events,
            lazy_crops=lazy_crops,
        )
        for entity_id, entity in dict(state.get("object_policy_overrides") or {}).items()
        if isinstance(entity, dict)
    }
    output["presentation_enrichment"] = {
        "entity_refinement": True,
        "schema_version": "working_state_presentation_enrichment.v1",
        "state_hash_unchanged": state.get("state_hash"),
    }
    return output


def raw_tracklets_for_entity(
    entity: dict[str, Any],
    *,
    package_root: Path | None = None,
    entity_index: dict[str, Any] | None = None,
    session_events: list[dict[str, Any]] | None = None,
    lazy_crops: bool = False,
    _context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    context = _context or lineage_context(entity_index, session_events)
    return dedupe_tracklets(
        resolve_raw_tracklets(
            entity,
            package_root=package_root,
            context=context,
            lazy_crops=lazy_crops,
            seen_entity_ids=set(),
        )
    )


def lineage_context(
    entity_index: dict[str, Any] | None,
    session_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    lineage: dict[str, list[str]] = {}

    for entities in dict((entity_index or {}).get("cases") or {}).values():
        for entity in list(entities or []):
            if isinstance(entity, dict):
                entity_id = str(entity.get("entity_id") or "")
                if entity_id:
                    by_id.setdefault(entity_id, dict(entity))

    for event in list(session_events or []):
        event_type = str(event.get("event_type") or "")
        before = dict(event.get("before_state") or {})
        after = dict(event.get("after_state") or {})
        before_entities = [
            dict(item)
            for item in list(before.get("entities") or [])
            if isinstance(item, dict) and item.get("entity_id")
        ]
        for item in before_entities:
            by_id.setdefault(str(item["entity_id"]), item)
        if event_type in {"entities_merged", "entity_merge_candidate_accepted"}:
            primary = dict(after.get("primary") or {})
            primary_id = str(primary.get("entity_id") or "")
            if primary_id:
                by_id[primary_id] = primary
                sources = [
                    str(item)
                    for item in list(
                        after.get("source_entity_ids")
                        or after.get("retired_entity_ids")
                        or primary.get("source_entity_ids")
                        or primary.get("aliases")
                        or []
                    )
                    if item
                ]
                if not sources:
                    sources = [str(item["entity_id"]) for item in before_entities]
                lineage[primary_id] = sorted(set(sources))
        if event_type in {"entity_mixed_split", "entity_tracklet_split"}:
            parent = dict(after.get("parent") or {})
            parent_id = str(parent.get("entity_id") or "")
            if parent_id:
                by_id[parent_id] = parent
            for child in list(after.get("split_entities") or []):
                if isinstance(child, dict) and child.get("entity_id"):
                    by_id[str(child["entity_id"])] = dict(child)
        if event_type == "entity_policy_changed":
            entity_id = str(after.get("entity_id") or "")
            if entity_id:
                by_id[entity_id] = dict(by_id.get(entity_id) or {}) | dict(after)
        if event_type == "bulk_entity_policy_changed":
            for item in list(after.get("entities") or []):
                if isinstance(item, dict) and item.get("entity_id"):
                    entity_id = str(item["entity_id"])
                    by_id[entity_id] = dict(by_id.get(entity_id) or {}) | dict(item)

    return {"entity_by_id": by_id, "lineage_by_id": lineage}


def resolve_raw_tracklets(
    entity: dict[str, Any],
    *,
    package_root: Path | None,
    context: dict[str, Any],
    lazy_crops: bool,
    seen_entity_ids: set[str],
) -> list[dict[str, Any]]:
    entity_id = str(entity.get("entity_id") or "")
    if entity_id:
        if entity_id in seen_entity_ids:
            return []
        seen_entity_ids = {*seen_entity_ids, entity_id}

    explicit = list(
        dict(entity.get("entity_refinement") or {}).get("raw_tracklets")
        or entity.get("raw_tracklets")
        or []
    )
    if explicit:
        return [
            populate_tracklet_crops(
                normalize_tracklet(
                    item,
                    entity=entity,
                    interval_source=TRACKLET_INTERVAL_EXPLICIT,
                ),
                entity=entity,
                package_root=package_root,
                lazy_crops=lazy_crops,
            )
            for item in explicit
            if isinstance(item, dict)
        ]

    descendants: list[dict[str, Any]] = []
    by_id = dict(context.get("entity_by_id") or {})
    for source_id in raw_lineage_source_ids(entity, context):
        source = by_id.get(source_id)
        if not source or str(source.get("entity_id") or "") == entity_id:
            continue
        descendants.extend(
            resolve_raw_tracklets(
                source,
                package_root=package_root,
                context=context,
                lazy_crops=lazy_crops,
                seen_entity_ids=seen_entity_ids,
            )
        )
    if descendants:
        return dedupe_tracklets(descendants)

    source_backed = source_backed_tracklet_from_entity(entity)
    if source_backed is not None:
        return [
            populate_tracklet_crops(
                source_backed,
                entity=entity,
                package_root=package_root,
                lazy_crops=lazy_crops,
            )
        ]

    return fallback_span_tracklets(entity)


def raw_lineage_source_ids(entity: dict[str, Any], context: dict[str, Any]) -> list[str]:
    entity_id = str(entity.get("entity_id") or "")
    values: list[str] = list(dict(context.get("lineage_by_id") or {}).get(entity_id) or [])
    for key in ("source_entity_ids", "aliases", "retired_entity_ids"):
        values.extend(str(item) for item in list(entity.get(key) or []) if item)
    for entry in list(entity.get("merge_split_lineage") or []):
        if isinstance(entry, dict):
            values.extend(str(item) for item in list(entry.get("source_entity_ids") or []) if item)
            parent = str(entry.get("parent_entity_id") or "")
            if parent:
                values.append(parent)
    return sorted({value for value in values if value and value != entity_id})


def source_backed_tracklet_from_entity(entity: dict[str, Any]) -> dict[str, Any] | None:
    category = str(entity.get("category") or "")
    track_ids = sorted(entity_track_ids(entity))
    source_lineage = dict(entity.get("source_lineage") or {})
    records = [
        dict(item)
        for item in list(source_lineage.get("render_region_sources") or [])
        if isinstance(item, dict)
    ]
    observations: list[dict[str, Any]] = []
    detector_scores: list[float] = []
    provenance_counts: dict[str, int] = {}
    for record in records:
        bbox = parse_bbox(record.get("bbox_xyxy"))
        frame = optional_int(record.get("source_frame"))
        if bbox is None or frame is None:
            continue
        source_record = dict(record.get("source_record") or {})
        metrics = dict(source_record.get("metrics") or {})
        score = optional_float(metrics.get("score") or source_record.get("score"))
        provenance = str(record.get("provenance") or source_record.get("provenance") or "")
        if provenance:
            provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
        if score is not None:
            detector_scores.append(score)
        observations.append(
            {
                "bbox_xyxy": list(bbox),
                "detector_score": score,
                "provenance": provenance,
                "source_frame": frame,
            }
        )
    if not observations:
        bbox = parse_bbox(entity.get("representative_bbox"))
        frame = optional_int(entity.get("representative_source_frame"))
        if bbox is not None and frame is not None:
            observations.append(
                {
                    "bbox_xyxy": list(bbox),
                    "detector_score": None,
                    "provenance": "representative_entity_geometry",
                    "source_frame": frame,
                }
            )
    if not observations and not track_ids:
        return None

    technical_ids = [
        str(item)
        for item in list(source_lineage.get("technical_track_ids") or [])
        if item is not None
    ]
    if not technical_ids:
        technical_id = str(source_lineage.get("technical_id") or "")
        if technical_id:
            technical_ids.append(technical_id)
    tracklet_id = sorted(set(technical_ids or track_ids or [str(entity.get("entity_id") or "")]))[0]
    frames = [
        int(item["source_frame"])
        for item in observations
        if isinstance(item.get("source_frame"), int)
    ]
    start = min(frames) if frames else optional_int(entity.get("first_visible_source_frame"))
    end = max(frames) if frames else optional_int(entity.get("last_visible_source_frame"))
    raw_source_id = stable_hash(
        {
            "case_id": entity.get("case_id"),
            "category": category,
            "tracklet_id": tracklet_id,
            "technical_ids": sorted(set(technical_ids or track_ids)),
        }
    )
    return {
        "category": category,
        "detector_backed": bool(
            detector_scores
            or provenance_counts.get("detector")
            or source_lineage.get("technical_detector_ids")
        ),
        "detector_score": max(detector_scores) if detector_scores else None,
        "geometry_hash": stable_hash(observations),
        "interval_source": TRACKLET_INTERVAL_SOURCE_BACKED,
        "observation_count": len(observations),
        "observations": sorted(observations, key=lambda item: int(item["source_frame"])),
        "parent_entity_id": entity.get("entity_id"),
        "provenance": {
            "created_provenance": dict(entity.get("created_provenance") or {}),
            "detector_backed": detector_backed(entity),
            "provenance_counts": provenance_counts,
            "source_lineage": source_lineage,
        },
        "quality_state": "REPRESENTATIVE" if observations else "NO_RELIABLE_CROP",
        "raw_source_id": raw_source_id,
        "representative_crops": existing_representative_crops(entity),
        "source_frame_end": end,
        "source_frame_start": start,
        "tracklet_id": tracklet_id,
    }


def fallback_span_tracklets(entity: dict[str, Any]) -> list[dict[str, Any]]:
    track_ids = sorted(entity_track_ids(entity))
    first = optional_int(entity.get("first_visible_source_frame"))
    last = optional_int(entity.get("last_visible_source_frame"))
    crops = existing_representative_crops(entity)
    return [
        {
            "category": str(entity.get("category") or ""),
            "detector_backed": detector_backed(entity),
            "detector_score": None,
            "geometry_hash": None,
            "interval_source": TRACKLET_INTERVAL_ENTITY_SPAN,
            "observation_count": 0,
            "observations": [],
            "provenance": tracklet_provenance(entity),
            "quality_state": crop_quality_state_from_crops(crops),
            "raw_source_id": f"{entity.get('case_id')}:{entity.get('category')}:{track_id}",
            "representative_crops": crops,
            "source_frame_end": last,
            "source_frame_start": first,
            "tracklet_id": track_id,
        }
        for track_id in track_ids
    ]


def normalize_tracklet(
    item: dict[str, Any],
    *,
    entity: dict[str, Any],
    interval_source: str,
) -> dict[str, Any]:
    tracklet_id = str(
        item.get("tracklet_id")
        or item.get("track_id")
        or item.get("track_uid")
        or item.get("id")
        or ""
    )
    start = optional_int(
        item.get("source_frame_start")
        if item.get("source_frame_start") is not None
        else item.get("first_visible_source_frame")
        if item.get("first_visible_source_frame") is not None
        else item.get("first_frame")
    )
    end = optional_int(
        item.get("source_frame_end")
        if item.get("source_frame_end") is not None
        else item.get("last_visible_source_frame")
        if item.get("last_visible_source_frame") is not None
        else item.get("last_frame")
    )
    observations = normalize_observations(item)
    if not observations:
        for crop in list(item.get("representative_crops") or []):
            if not isinstance(crop, dict):
                continue
            bbox = parse_bbox(crop.get("bbox_xyxy"))
            frame = optional_int(crop.get("source_frame"))
            if bbox is None or frame is None:
                continue
            observations.append(
                {
                    "bbox_xyxy": list(bbox),
                    "detector_score": optional_float(crop.get("detector_score")),
                    "provenance": crop.get("provenance") or "representative_crop",
                    "source_frame": frame,
                }
            )
    if not observations:
        bbox = parse_bbox(
            item.get("bbox_xyxy")
            or item.get("bbox")
            or item.get("representative_bbox")
        )
        frame_value = (
            item.get("source_frame")
            if item.get("source_frame") is not None
            else item.get("representative_source_frame")
        )
        frame = optional_int(frame_value if frame_value is not None else start)
        if bbox is not None and frame is not None:
            observations = [
                {"bbox_xyxy": list(bbox), "detector_score": None, "source_frame": frame}
            ]
    if observations:
        frames = [int(obs["source_frame"]) for obs in observations]
        start = min(frames) if start is None else min(start, min(frames))
        end = max(frames) if end is None else max(end, max(frames))
    crops = [
        crop
        for crop in list(item.get("representative_crops") or [])
        if isinstance(crop, dict)
    ]
    if not crops:
        crops = existing_representative_crops(entity)
    raw_source_id = str(
        item.get("raw_source_id")
        or stable_hash(
            {
                "case_id": item.get("case_id") or entity.get("case_id"),
                "category": item.get("category") or entity.get("category"),
                "tracklet_id": tracklet_id,
            }
        )
    )
    return {
        "category": str(item.get("category") or entity.get("category") or ""),
        "detector_backed": bool(
            item.get("detector_backed")
            if item.get("detector_backed") is not None
            else detector_backed(entity)
        ),
        "detector_score": item.get("detector_score"),
        "geometry_hash": item.get("geometry_hash") or stable_hash(observations),
        "interval_source": str(item.get("interval_source") or interval_source),
        "observation_count": int(item.get("observation_count") or len(observations)),
        "observations": observations,
        "parent_entity_id": item.get("parent_entity_id") or entity.get("entity_id"),
        "provenance": dict(item.get("provenance") or tracklet_provenance(entity)),
        "quality_state": str(item.get("quality_state") or crop_quality_state_from_crops(crops)),
        "raw_source_id": raw_source_id,
        "representative_crops": crops,
        "source_frame_end": end,
        "source_frame_start": start,
        "tracklet_id": tracklet_id,
    }


def normalize_observations(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = list(item.get("observations") or item.get("per_frame_bboxes") or [])
    for row in values:
        if not isinstance(row, dict):
            continue
        frame_value = (
            row.get("source_frame")
            if row.get("source_frame") is not None
            else row.get("frame")
        )
        frame = optional_int(frame_value)
        bbox = parse_bbox(row.get("bbox_xyxy") or row.get("bbox"))
        if frame is None or bbox is None:
            continue
        rows.append(
            {
                "bbox_xyxy": list(bbox),
                "detector_score": optional_float(row.get("detector_score") or row.get("score")),
                "provenance": row.get("provenance"),
                "source_frame": frame,
            }
        )
    return sorted(rows, key=lambda row: int(row["source_frame"]))


def populate_tracklet_crops(
    tracklet: dict[str, Any],
    *,
    entity: dict[str, Any],
    package_root: Path | None,
    lazy_crops: bool,
) -> dict[str, Any]:
    crops = [
        crop
        for crop in list(tracklet.get("representative_crops") or [])
        if isinstance(crop, dict)
        and str(crop.get("quality_state") or "REPRESENTATIVE")
        not in {"WRONG_OBJECT", "BACKGROUND_OR_NOISE"}
    ]
    crops = [ensure_crop_descriptor(crop, package_root=package_root) for crop in crops]
    crops = [crop for crop in crops if crop.get("path") and crop.get("source_frame") is not None]
    if crops:
        return tracklet | {
            "quality_state": crop_quality_state_from_crops(crops),
            "representative_crops": crops[:3],
        }
    if not lazy_crops or package_root is None:
        return tracklet | {"representative_crops": []}
    generated = generate_lazy_tracklet_crops(package_root, entity=entity, tracklet=tracklet)
    return tracklet | {
        "quality_state": crop_quality_state_from_crops(generated),
        "representative_crops": generated,
    }


def generate_lazy_tracklet_crops(
    package_root: Path,
    *,
    entity: dict[str, Any],
    tracklet: dict[str, Any],
) -> list[dict[str, Any]]:
    plan = lazy_crop_plan(package_root, entity=entity, tracklet=tracklet)
    if not plan:
        return []
    rows: list[dict[str, Any] | None] = []
    missing: list[tuple[int, dict[str, Any]]] = []
    for crop_plan in plan["crop_plans"]:
        cached = read_cached_crop_metadata(package_root, crop_plan)
        rows.append(cached)
        if cached is None:
            missing.append((len(rows) - 1, crop_plan))
    if not missing:
        return [row for row in rows if row is not None][:3]

    capture = cv.VideoCapture(str(plan["source_path"]))
    try:
        if not capture.isOpened():
            return [row for row in rows if row is not None][:3]
        for row_index, crop_plan in missing:
            frame_index = int(crop_plan["source_frame_index"])
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_height, frame_width = frame.shape[:2]
            clamped = clamp_bbox(
                tuple(crop_plan["bbox_xyxy"]),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if clamped is None:
                continue
            if list(clamped) != list(crop_plan["bbox_xyxy"]):
                continue
            crop = padded_crop(frame, clamped, category=str(entity.get("category") or ""))
            if crop.size == 0:
                continue
            descriptor = visual_descriptor_from_image(crop)
            if descriptor is None:
                continue
            out_path = package_root / str(crop_plan["path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                if not cv.imwrite(
                    str(out_path),
                    resize_crop(crop),
                    [int(cv.IMWRITE_JPEG_QUALITY), 88],
                ):
                    continue
            row = dict(crop_plan["metadata"]) | {
                "path": str(crop_plan["path"]),
                "visual_descriptor": descriptor,
                "visual_descriptor_version": VISUAL_DESCRIPTOR_VERSION,
            }
            write_json_atomic(package_root / str(crop_plan["metadata_path"]), row)
            rows[row_index] = row
        return [row for row in rows if row is not None][:3]
    finally:
        capture.release()


def cached_lazy_tracklet_crops(
    package_root: Path,
    *,
    entity: dict[str, Any],
    tracklet: dict[str, Any],
) -> list[dict[str, Any]]:
    plan = lazy_crop_plan(package_root, entity=entity, tracklet=tracklet)
    if not plan:
        return []
    rows = [
        cached
        for crop_plan in plan["crop_plans"]
        for cached in [read_cached_crop_metadata(package_root, crop_plan)]
        if cached is not None
    ]
    return rows[:3]


def lazy_crop_plan(
    package_root: Path,
    *,
    entity: dict[str, Any],
    tracklet: dict[str, Any],
) -> dict[str, Any] | None:
    case_id = str(entity.get("case_id") or tracklet.get("case_id") or "")
    if not case_id:
        return None
    manifest_path = package_root / "cases" / f"{case_id}.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    source_path = package_root / str(manifest.get("source_path") or "")
    if not source_path.exists():
        return None
    observations = [
        item
        for item in list(tracklet.get("observations") or [])
        if isinstance(item, dict)
        and isinstance(item.get("source_frame"), int)
        and parse_bbox(item.get("bbox_xyxy")) is not None
    ]
    if not observations:
        return None
    width = optional_int(manifest.get("width"))
    height = optional_int(manifest.get("height"))
    selected = selected_crop_observations(observations, width=width, height=height)
    if not selected:
        return None
    source_start = int(manifest.get("source_frame_start") or 0)
    source_sha256 = manifest.get("source_video_hash")
    crop_plans: list[dict[str, Any]] = []
    for role, observation in selected:
        bbox = parse_bbox(observation.get("bbox_xyxy"))
        source_frame = optional_int(observation.get("source_frame"))
        if bbox is None or source_frame is None:
            continue
        raw_tracklet_id = str(
            observation.get("raw_tracklet_id")
            or observation.get("tracklet_id")
            or tracklet.get("tracklet_id")
            or ""
        )
        if not raw_tracklet_id or raw_tracklet_id.startswith("entity_preview_"):
            continue
        frame_index = source_frame - source_start
        if frame_index < 0:
            continue
        clamped = (
            clamp_bbox(bbox, frame_width=width, frame_height=height)
            if width is not None and height is not None
            else bbox
        )
        if clamped is None:
            continue
        raw_source_id = str(
            observation.get("raw_source_id") or tracklet.get("raw_source_id") or ""
        )
        source_entity_id = str(
            observation.get("source_entity_id")
            or observation.get("parent_entity_id")
            or tracklet.get("parent_entity_id")
            or ""
        )
        geometry_hash = stable_hash(
            {
                "bbox_xyxy": list(clamped),
                "raw_source_id": raw_source_id,
                "raw_tracklet_id": raw_tracklet_id,
                "source_frame": source_frame,
            }
        )
        cache_key = stable_hash(
            {
                "algorithm": THUMBNAIL_ALGORITHM_VERSION,
                "geometry_hash": geometry_hash,
                "raw_source_id": raw_source_id,
                "raw_tracklet_id": raw_tracklet_id,
                "source_sha256": source_sha256,
            }
        )
        relative_path = (
            Path("entities")
            / case_id
            / "derived_thumbnails"
            / f"{cache_key[:24]}.jpg"
        ).as_posix()
        metadata_path = (
            Path("entities")
            / case_id
            / "derived_thumbnails"
            / f"{cache_key[:24]}.json"
        ).as_posix()
        metadata = {
            "bbox_xyxy": list(clamped),
            "cache_key": cache_key,
            "crop_role": role,
            "geometry_hash": geometry_hash,
            "metadata_schema_version": DERIVED_PREVIEW_METADATA_VERSION,
            "path": relative_path,
            "quality_state": "REPRESENTATIVE",
            "raw_source_id": raw_source_id or None,
            "raw_tracklet_id": raw_tracklet_id,
            "schema_version": THUMBNAIL_ALGORITHM_VERSION,
            "source_entity_id": source_entity_id or None,
            "source_frame": source_frame,
            "source_frame_index": frame_index,
        }
        crop_plans.append(
            {
                "bbox_xyxy": list(clamped),
                "metadata": metadata,
                "metadata_path": metadata_path,
                "path": relative_path,
                "source_frame_index": frame_index,
            }
        )
    if not crop_plans:
        return None
    return {"crop_plans": crop_plans, "source_path": source_path}


def read_cached_crop_metadata(
    package_root: Path,
    crop_plan: dict[str, Any],
) -> dict[str, Any] | None:
    metadata_path = package_root / str(crop_plan.get("metadata_path") or "")
    image_path = package_root / str(crop_plan.get("path") or "")
    if not metadata_path.exists() or not image_path.exists():
        return None
    payload = read_json(metadata_path)
    expected = dict(crop_plan.get("metadata") or {})
    for key in (
        "bbox_xyxy",
        "cache_key",
        "crop_role",
        "geometry_hash",
        "path",
        "raw_tracklet_id",
        "schema_version",
        "source_frame",
        "source_frame_index",
    ):
        if payload.get(key) != expected.get(key):
            return None
    if not payload.get("raw_tracklet_id"):
        return None
    return payload


def selected_crop_observations(
    observations: list[dict[str, Any]],
    *,
    width: int | None,
    height: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(observations, key=lambda item: int(item["source_frame"]))
    if not ordered:
        return []
    roles = [
        ("best_early", ordered[: max(1, len(ordered) // 3)]),
        ("best_middle", ordered[max(0, len(ordered) // 3) : max(1, (2 * len(ordered)) // 3)]),
        ("best_late", ordered[max(0, (2 * len(ordered)) // 3) :]),
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    seen_frames: set[int] = set()
    for role, bucket in roles:
        candidates = bucket or ordered
        best = max(
            candidates,
            key=lambda item: observation_quality(item, width=width, height=height),
        )
        frame = int(best["source_frame"])
        if frame in seen_frames and len(ordered) > len(seen_frames):
            alternatives = [
                item for item in ordered if int(item["source_frame"]) not in seen_frames
            ]
            if alternatives:
                best = max(
                    alternatives,
                    key=lambda item: observation_quality(item, width=width, height=height),
                )
                frame = int(best["source_frame"])
        seen_frames.add(frame)
        selected.append((role, best))
    output: list[tuple[str, dict[str, Any]]] = []
    seen_keys: set[tuple[int, tuple[int, int, int, int]]] = set()
    for role, item in selected:
        bbox = parse_bbox(item.get("bbox_xyxy"))
        if bbox is None:
            continue
        key = (int(item["source_frame"]), bbox)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        output.append((role, item))
    return output


def observation_quality(
    item: dict[str, Any],
    *,
    width: int | None,
    height: int | None,
) -> tuple[float, float, float, int]:
    bbox = parse_bbox(item.get("bbox_xyxy"))
    if bbox is None:
        return (0.0, 0.0, 0.0, 0)
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    score = optional_float(item.get("detector_score")) or 0.0
    frame_width = max(1, width or max(x2, 1))
    frame_height = max(1, height or max(y2, 1))
    border_margin = min(x1, y1, frame_width - x2, frame_height - y2) / float(
        max(frame_width, frame_height)
    )
    detector_bonus = 1.0 if str(item.get("provenance") or "") == "detector" else 0.0
    return (
        detector_bonus,
        score,
        area / float(frame_width * frame_height),
        int(border_margin * 1000),
    )


def classify_entity_purity(entity: dict[str, Any]) -> dict[str, Any]:
    tracklets = raw_tracklets_for_entity(entity, lazy_crops=False)
    child_count = len(tracklets)
    conflicts = cannot_link_conflicts(tracklets)
    verification = dict(entity.get("entity_refinement_verification") or {})
    human_verified = str(verification.get("status") or "") in {
        "human_verified",
        "human_verified_merge",
    }

    reasons: list[str] = []
    if conflicts:
        purity = PURITY_LIKELY_MIXED
        reasons.append("cannot_link_simultaneous_spatially_distinct_raw_tracklets")
    elif child_count == 0:
        purity = PURITY_UNVERIFIED
        reasons.append("no_resolvable_raw_tracklet")
    elif missing_durable_geometry(tracklets):
        purity = PURITY_UNVERIFIED
        reasons.append(
            "raw_tracklet_missing_durable_geometry"
            if child_count <= 1
            else "multi_tracklet_missing_durable_geometry"
        )
    elif human_verified:
        purity = PURITY_PURE
        reasons.append("human_verified_group")
    elif child_count <= 1:
        if not has_reliable_crop(entity) and has_explicit_low_quality_crop_signal(
            entity,
            tracklets,
        ):
            purity = PURITY_LOW_QUALITY
            reasons.append("explicit_low_quality_or_missing_crop")
        else:
            purity = PURITY_PURE
            reasons.append("single_tracklet")
    elif not has_reliable_crop(entity) and has_explicit_low_quality_crop_signal(
        entity,
        tracklets,
    ):
        purity = PURITY_LOW_QUALITY
        reasons.append("explicit_multi_tracklet_low_quality_or_missing_crop")
    else:
        purity = PURITY_UNVERIFIED
        reasons.append("multi_tracklet_without_human_verification")

    return {
        "cannot_link_conflicts": conflicts,
        "child_tracklet_count": child_count,
        "purity": purity,
        "reasons": reasons,
        "schema_version": "entity_purity.v1",
    }


def cannot_link_conflicts(tracklets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    unique = dedupe_tracklets(tracklets)
    for index, left in enumerate(unique):
        left_by_frame = observations_by_frame(left)
        if not left_by_frame:
            continue
        for right in unique[index + 1 :]:
            if raw_tracklet_key(left) == raw_tracklet_key(right):
                continue
            right_by_frame = observations_by_frame(right)
            if not right_by_frame:
                continue
            for frame in sorted(set(left_by_frame) & set(right_by_frame)):
                left_bbox = parse_bbox(left_by_frame[frame].get("bbox_xyxy"))
                right_bbox = parse_bbox(right_by_frame[frame].get("bbox_xyxy"))
                if left_bbox is None or right_bbox is None:
                    continue
                iou_value = bbox_iou(left_bbox, right_bbox)
                separation = center_separation_norm(left_bbox, right_bbox)
                if iou_value < 0.50:
                    conflicts.append(
                        {
                            "bboxes": {
                                str(left.get("tracklet_id") or ""): list(left_bbox),
                                str(right.get("tracklet_id") or ""): list(right_bbox),
                            },
                            "center_separation_norm": round(separation, 4),
                            "iou": round(iou_value, 4),
                            "raw_child_ids": [
                                str(left.get("tracklet_id") or ""),
                                str(right.get("tracklet_id") or ""),
                            ],
                            "schema_version": "entity_cannot_link_evidence.v1",
                            "source_frame": frame,
                        }
                    )
                    break
    return conflicts[:20]


def observations_by_frame(tracklet: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for item in list(tracklet.get("observations") or []):
        if not isinstance(item, dict):
            continue
        frame = optional_int(item.get("source_frame"))
        bbox = parse_bbox(item.get("bbox_xyxy"))
        if frame is None or bbox is None:
            continue
        existing = rows.get(frame)
        existing_bbox = parse_bbox((existing or {}).get("bbox_xyxy"))
        if existing is None or bbox_area(bbox) > bbox_area(existing_bbox or (0, 0, 0, 0)):
            rows[frame] = item | {"bbox_xyxy": list(bbox), "source_frame": frame}
    return rows


def advisory_candidate_pairs(
    package_root: Path,
    entities: list[dict[str, Any]],
    *,
    case_id: str,
    rejected_candidate_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    del package_root
    rejected_candidate_ids = rejected_candidate_ids or set()
    candidates: list[dict[str, Any]] = []
    face_entities = [entity for entity in entities if entity.get("category") == CATEGORY_FACE]
    for index, left in enumerate(face_entities):
        for right in face_entities[index + 1 :]:
            pair_id = candidate_pair_id(case_id, left, right)
            if pair_id in rejected_candidate_ids:
                continue
            candidates.append(deterministic_candidate(pair_id, left, right))

    similarity_by_entity: dict[str, list[float]] = {}
    for candidate in candidates:
        if candidate["classification"] == CANDIDATE_REJECT_TEMPORAL_CONFLICT:
            continue
        if not candidate.get("has_visual_evidence"):
            continue
        for entity_id in list(candidate.get("entity_ids") or []):
            similarity_by_entity.setdefault(str(entity_id), []).append(
                float(candidate.get("similarity") or 0.0)
            )

    output: list[dict[str, Any]] = []
    for candidate in candidates:
        classification = str(candidate.get("classification") or "")
        similarity = float(candidate.get("similarity") or 0.0)
        if classification == CANDIDATE_REJECT_TEMPORAL_CONFLICT:
            if similarity >= 0.82:
                output.append(candidate)
            continue
        if classification not in {CANDIDATE_INSUFFICIENT, CANDIDATE_REJECT_VISUAL_MISMATCH}:
            continue
        if not candidate.get("has_visual_evidence") or similarity < 0.86:
            continue
        entity_ids = [str(item) for item in list(candidate.get("entity_ids") or [])]
        margins = []
        for entity_id in entity_ids:
            scores = sorted(similarity_by_entity.get(entity_id) or [], reverse=True)
            best = scores[0] if scores else 0.0
            second = scores[1] if len(scores) > 1 else 0.0
            margins.append(best - second)
            if abs(similarity - best) > 1e-9:
                break
        else:
            if all(margin >= 0.035 or similarity >= 0.93 for margin in margins):
                output.append(candidate | {"classification": CANDIDATE_POSSIBLE})
    return sorted(
        output,
        key=lambda item: (
            classification_rank(str(item.get("classification") or "")),
            -float(item.get("similarity") or 0),
            str(item.get("candidate_id") or ""),
        ),
    )[:24]


def deterministic_candidate(
    pair_id: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    conflict_frames = temporal_conflict_frame_count(left, right)
    similarity = deterministic_visual_similarity(left, right)
    has_visual_evidence = bool(entity_visual_descriptors(left) and entity_visual_descriptors(right))
    if conflict_frames > 0:
        classification = CANDIDATE_REJECT_TEMPORAL_CONFLICT
    elif not has_visual_evidence:
        classification = CANDIDATE_INSUFFICIENT
    elif similarity <= 0.55:
        classification = CANDIDATE_REJECT_VISUAL_MISMATCH
    else:
        classification = CANDIDATE_INSUFFICIENT
    return {
        "candidate_id": pair_id,
        "category": left.get("category"),
        "classification": classification,
        "descriptor_version": VISUAL_DESCRIPTOR_VERSION,
        "entity_ids": [left["entity_id"], right["entity_id"]],
        "has_visual_evidence": has_visual_evidence,
        "labels": [left.get("label"), right.get("label")],
        "representative_crops": {
            left["entity_id"]: representative_crops(left),
            right["entity_id"]: representative_crops(right),
        },
        "schema_version": "entity_similarity_candidate.v1",
        "similarity": round(similarity, 4),
        "source_intervals": {
            left["entity_id"]: [
                left.get("first_visible_source_frame"),
                left.get("last_visible_source_frame"),
            ],
            right["entity_id"]: [
                right.get("first_visible_source_frame"),
                right.get("last_visible_source_frame"),
            ],
        },
        "temporal_conflict_frame_count": conflict_frames,
        "temporal_conflict_sample_frames": temporal_conflict_sample_frames(left, right),
    }


def candidate_pair_id(case_id: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    return candidate_pair_id_from_ids(case_id, str(left["entity_id"]), str(right["entity_id"]))


def candidate_pair_id_from_ids(case_id: str, left_id: str, right_id: str) -> str:
    ordered = sorted([left_id, right_id])
    return f"candidate_{stable_hash({'case_id': case_id, 'entity_ids': ordered})[:24]}"


def rejected_candidate_ids(events: list[dict[str, Any]]) -> set[str]:
    rejected: set[str] = set()
    for event in events:
        if str(event.get("event_type") or "") != "entity_merge_candidate_rejected":
            continue
        after = dict(event.get("after_state") or {})
        candidate_id = str(after.get("candidate_id") or "")
        if candidate_id:
            rejected.add(candidate_id)
    return rejected


def representative_crops(
    entity: dict[str, Any],
    *,
    package_root: Path | None = None,
    tracklets: list[dict[str, Any]] | None = None,
    prepare_missing: bool = False,
) -> list[dict[str, Any]]:
    if package_root is not None and tracklets:
        prepared = (
            generate_entity_lazy_crops(package_root, entity=entity, tracklets=tracklets)
            if prepare_missing
            else cached_entity_lazy_crops(package_root, entity=entity, tracklets=tracklets)
        )
        if prepared:
            return prepared
        return []
    crops = existing_representative_crops(entity, package_root=package_root)
    for tracklet in list(tracklets or []):
        crops.extend(
            crop
            for crop in list(tracklet.get("representative_crops") or [])
            if isinstance(crop, dict)
        )
    filtered = [
        ensure_crop_descriptor(crop, package_root=package_root)
        for crop in crops
        if str(crop.get("quality_state") or "REPRESENTATIVE")
        not in {"WRONG_OBJECT", "BACKGROUND_OR_NOISE"}
        and crop.get("path")
        and crop.get("source_frame") is not None
    ]
    return unique_crops(filtered)[:3]


def generate_entity_lazy_crops(
    package_root: Path,
    *,
    entity: dict[str, Any],
    tracklets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preview_tracklet = entity_preview_tracklet(entity, tracklets)
    if preview_tracklet is None:
        return []
    return generate_lazy_tracklet_crops(package_root, entity=entity, tracklet=preview_tracklet)


def cached_entity_lazy_crops(
    package_root: Path,
    *,
    entity: dict[str, Any],
    tracklets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preview_tracklet = entity_preview_tracklet(entity, tracklets)
    if preview_tracklet is None:
        return []
    return cached_lazy_tracklet_crops(package_root, entity=entity, tracklet=preview_tracklet)


def entity_preview_tracklet(
    entity: dict[str, Any],
    tracklets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    observations = preview_observations_for_tracklets(tracklets)
    if not observations:
        return None
    raw_keys = [raw_tracklet_key(tracklet) for tracklet in tracklets if isinstance(tracklet, dict)]
    return {
        "observations": observations,
        "raw_source_id": stable_hash(
            {
                "algorithm": THUMBNAIL_ALGORITHM_VERSION,
                "entity_id": entity.get("entity_id"),
                "raw_tracklet_keys": sorted(set(raw_keys)),
            }
        ),
        "tracklet_id": f"entity_preview_{entity.get('entity_id') or ''}",
    }


def preview_observations_for_tracklets(tracklets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for tracklet in tracklets:
        if not isinstance(tracklet, dict):
            continue
        raw_tracklet_id = str(tracklet.get("tracklet_id") or "")
        if not raw_tracklet_id:
            continue
        for observation in list(tracklet.get("observations") or []):
            if not isinstance(observation, dict):
                continue
            bbox = parse_bbox(observation.get("bbox_xyxy"))
            frame = optional_int(observation.get("source_frame"))
            if bbox is None or frame is None:
                continue
            observations.append(
                dict(observation)
                | {
                    "bbox_xyxy": list(bbox),
                    "parent_entity_id": tracklet.get("parent_entity_id"),
                    "raw_source_id": tracklet.get("raw_source_id"),
                    "raw_tracklet_id": raw_tracklet_id,
                    "source_entity_id": tracklet.get("parent_entity_id"),
                    "source_frame": frame,
                    "tracklet_id": raw_tracklet_id,
                }
            )
    return sorted(observations, key=lambda item: int(item["source_frame"]))


def preview_status_for_tracklets(
    tracklets: list[dict[str, Any]],
    crops: list[dict[str, Any]],
) -> dict[str, Any]:
    if crops:
        status = PREVIEW_READY
        reason = "derived_preview_cache_ready"
    elif preview_observations_for_tracklets(tracklets):
        status = PREVIEW_NOT_PREPARED
        reason = "derived_preview_not_prepared"
    else:
        status = PREVIEW_NO_RELIABLE_GEOMETRY
        reason = "no_durable_raw_geometry_for_preview"
    return {
        "reason": reason,
        "schema_version": "entity_preview_status.v1",
        "status": status,
    }


def prepare_entity_previews(
    package_root: Path,
    entity: dict[str, Any],
    *,
    entity_index: dict[str, Any] | None = None,
    session_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return enrich_entity(
        entity,
        package_root=package_root,
        entity_index=entity_index,
        session_events=session_events,
        lazy_crops=True,
    )


def existing_representative_crops(
    entity: dict[str, Any],
    *,
    package_root: Path | None = None,
) -> list[dict[str, Any]]:
    crops = [
        dict(item)
        for item in list(entity.get("representative_crops") or [])
        if isinstance(item, dict)
    ]
    if crops:
        return [ensure_crop_descriptor(crop, package_root=package_root) for crop in crops]
    identity = dict(entity.get("representative_thumbnail_identity") or {})
    path = entity.get("representative_thumbnail_path")
    bbox = entity.get("representative_bbox") or identity.get("bbox_xyxy")
    source_frame = entity.get("representative_source_frame") or identity.get("source_frame")
    if path and bbox and source_frame is not None:
        return [
            ensure_crop_descriptor(
                {
                    "bbox_xyxy": list(bbox),
                    "crop_role": "best_middle",
                    "path": path,
                    "quality_state": "REPRESENTATIVE",
                    "source_frame": source_frame,
                    "thumbnail_sha256": identity.get("thumbnail_sha256"),
                },
                package_root=package_root,
            )
        ]
    return []


def unique_crops(crops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for crop in crops:
        key = (crop.get("path"), crop.get("source_frame"), tuple(crop.get("bbox_xyxy") or []))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(crop))
    return output


def policy_action_availability(entity: dict[str, Any]) -> dict[str, Any]:
    purity = str(
        entity.get("purity_classification")
        or dict(entity.get("entity_refinement") or {}).get("purity")
        or classify_entity_purity(entity)["purity"]
    )
    verification = dict(entity.get("entity_refinement_verification") or {})
    child_review_complete = bool(
        verification.get("child_review_complete")
        or verification.get("complete_child_review")
        or verification.get("status") == "human_verified"
    )
    availability: dict[str, dict[str, Any]] = {
        privacy_entities.POLICY_REDACT: {"allowed": True, "reason": "most_protective"},
        privacy_entities.POLICY_REVIEW_REQUIRED: {
            "allowed": True,
            "reason": "conservative_review_required",
        },
        privacy_entities.POLICY_KEEP_VISIBLE: {"allowed": True, "reason": "verified_or_single"},
        privacy_entities.POLICY_NOT_SENSITIVE: {"allowed": True, "reason": "verified_or_single"},
    }
    if purity in {PURITY_LIKELY_MIXED, PURITY_UNVERIFIED}:
        reason = "mixed_or_unverified_entities_cannot_be_revealed_without_explicit_verification"
        availability[privacy_entities.POLICY_KEEP_VISIBLE] = {
            "allowed": False,
            "reason": reason,
        }
        availability[privacy_entities.POLICY_NOT_SENSITIVE] = {
            "allowed": False,
            "reason": reason,
        }
    elif purity == PURITY_LOW_QUALITY and not child_review_complete:
        reason = "low_quality_entities_require_complete_child_review_before_reveal"
        availability[privacy_entities.POLICY_KEEP_VISIBLE] = {
            "allowed": False,
            "reason": reason,
        }
        availability[privacy_entities.POLICY_NOT_SENSITIVE] = {
            "allowed": False,
            "reason": reason,
        }
    return {
        "actions": availability,
        "purity": purity,
        "schema_version": "entity_policy_action_availability.v1",
    }


def validate_policy_action(entity: dict[str, Any], policy: str) -> str | None:
    availability = policy_action_availability(entity)
    action = dict(dict(availability.get("actions") or {}).get(policy) or {})
    if action and action.get("allowed") is False:
        return str(action.get("reason") or "entity policy action disabled")
    return None


def entity_track_ids(entity: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "member_detector_tracker_ids",
        "automatic_source_track_ids",
        "reviewed_region_track_ids",
        "attached_fragment_track_ids",
    ):
        values.update(str(item) for item in list(entity.get(key) or []) if item is not None)
    return values


def dedupe_tracklets(tracklets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tracklet in tracklets:
        key = raw_tracklet_key(tracklet)
        if key in seen:
            continue
        seen.add(key)
        output.append(tracklet)
    return output


def raw_tracklet_key(tracklet: dict[str, Any]) -> str:
    return str(
        tracklet.get("raw_source_id")
        or stable_hash(
            {
                "category": tracklet.get("category"),
                "tracklet_id": tracklet.get("tracklet_id"),
            }
        )
    )


def missing_durable_geometry(tracklets: list[dict[str, Any]]) -> bool:
    return any(not observations_by_frame(tracklet) for tracklet in tracklets)


def has_reliable_crop(entity: dict[str, Any]) -> bool:
    return bool(representative_crops(entity))


def has_explicit_low_quality_crop_signal(
    entity: dict[str, Any],
    tracklets: list[dict[str, Any]],
) -> bool:
    if entity.get("thumbnail_unavailable_reason"):
        return True
    quality = dict(entity.get("representative_quality") or {})
    if str(quality.get("quality_state") or "") in {
        "BACKGROUND_OR_NOISE",
        "BORDER_CLIPPED",
        "NO_RELIABLE_CROP",
        "TOO_SMALL_OR_BLURRED",
        "WRONG_OBJECT",
    }:
        return True
    if "representative_crops" in entity and not representative_crops(entity):
        return True
    raw_tracklets = entity.get("raw_tracklets")
    if isinstance(raw_tracklets, list):
        return any(
            isinstance(tracklet, dict)
            and str(tracklet.get("interval_source") or "") == TRACKLET_INTERVAL_EXPLICIT
            and "representative_crops" in tracklet
            and not list(tracklet.get("representative_crops") or [])
            for tracklet in raw_tracklets
        )
    return False


def crop_quality_state_from_crops(crops: list[dict[str, Any]]) -> str:
    if not crops:
        return "NO_RELIABLE_CROP"
    return "REPRESENTATIVE"


def detector_backed(entity: dict[str, Any]) -> bool:
    provenance = dict(dict(entity.get("evidence_summary") or {}).get("provenance_counts") or {})
    if int(provenance.get("detector") or 0) > 0:
        return True
    lineage = dict(entity.get("source_lineage") or {})
    return bool(lineage.get("technical_detector_ids"))


def tracklet_provenance(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_provenance": dict(entity.get("created_provenance") or {}),
        "detector_backed": detector_backed(entity),
        "source_lineage": dict(entity.get("source_lineage") or {}),
    }


def temporal_conflict_frame_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    l1 = optional_int(left.get("first_visible_source_frame"))
    l2 = optional_int(left.get("last_visible_source_frame"))
    r1 = optional_int(right.get("first_visible_source_frame"))
    r2 = optional_int(right.get("last_visible_source_frame"))
    if None in {l1, l2, r1, r2}:
        return 0
    start = max(int(l1), int(r1))
    end = min(int(l2), int(r2))
    return max(0, end - start + 1)


def temporal_conflict_sample_frames(left: dict[str, Any], right: dict[str, Any]) -> list[int]:
    l1 = optional_int(left.get("first_visible_source_frame"))
    l2 = optional_int(left.get("last_visible_source_frame"))
    r1 = optional_int(right.get("first_visible_source_frame"))
    r2 = optional_int(right.get("last_visible_source_frame"))
    if None in {l1, l2, r1, r2}:
        return []
    start = max(int(l1), int(r1))
    end = min(int(l2), int(r2))
    if start > end:
        return []
    return list(range(start, min(end, start + 4) + 1))


def deterministic_visual_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_desc = entity_visual_descriptors(left)
    right_desc = entity_visual_descriptors(right)
    if not left_desc or not right_desc:
        return 0.0
    scores = [
        cosine_similarity(left_item, right_item)
        for left_item in left_desc
        for right_item in right_desc
    ]
    if not scores:
        return 0.0
    return max(0.0, min(1.0, float(np.median(sorted(scores)[-min(3, len(scores)) :]))))


def entity_visual_descriptors(entity: dict[str, Any]) -> list[tuple[float, ...]]:
    stable_descriptors: list[tuple[float, ...]] = []
    preview_descriptors: list[tuple[float, ...]] = []
    collect_visual_descriptors(
        representative_crops(entity),
        stable_descriptors=stable_descriptors,
        preview_descriptors=preview_descriptors,
    )
    for tracklet in list(entity.get("raw_tracklets") or []):
        if not isinstance(tracklet, dict):
            continue
        collect_visual_descriptors(
            representative_crops(tracklet),
            stable_descriptors=stable_descriptors,
            preview_descriptors=preview_descriptors,
        )
    return (stable_descriptors or preview_descriptors)[:9]


def collect_visual_descriptors(
    crops: list[dict[str, Any]],
    *,
    stable_descriptors: list[tuple[float, ...]],
    preview_descriptors: list[tuple[float, ...]],
) -> None:
    for crop in crops:
        descriptor = crop.get("visual_descriptor")
        if not isinstance(descriptor, list) or not descriptor:
            continue
        try:
            parsed = tuple(float(item) for item in descriptor)
        except (TypeError, ValueError):
            continue
        if crop_uses_derived_preview_cache(crop):
            preview_descriptors.append(parsed)
        else:
            stable_descriptors.append(parsed)


def crop_uses_derived_preview_cache(crop: dict[str, Any]) -> bool:
    return str(crop.get("metadata_schema_version") or "") == DERIVED_PREVIEW_METADATA_VERSION


def ensure_crop_descriptor(crop: dict[str, Any], *, package_root: Path | None) -> dict[str, Any]:
    if crop.get("visual_descriptor"):
        return dict(crop)
    if package_root is None:
        return dict(crop)
    path = crop.get("path")
    if not path:
        return dict(crop)
    crop_path = package_root / str(path)
    if not crop_path.exists():
        return dict(crop)
    image = cv.imread(str(crop_path), cv.IMREAD_COLOR)
    descriptor = visual_descriptor_from_image(image)
    if descriptor is None:
        return dict(crop)
    return dict(crop) | {
        "visual_descriptor": descriptor,
        "visual_descriptor_version": VISUAL_DESCRIPTOR_VERSION,
    }


def visual_descriptor_from_image(image: Any) -> list[float] | None:
    if image is None or getattr(image, "size", 0) == 0:
        return None
    resized = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()
    sobel_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    magnitude = cv.magnitude(sobel_x, sobel_y)
    gradient_hist = cv.calcHist([magnitude.astype("float32")], [0], None, [16], [0, 255])
    gradient_hist = cv.normalize(gradient_hist, gradient_hist).flatten()
    descriptor = np.concatenate([hist, gradient_hist]).astype("float32")
    norm = float(np.linalg.norm(descriptor))
    if norm <= 0:
        return None
    descriptor = descriptor / norm
    return [round(float(value), 6) for value in descriptor.tolist()]


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def resize_crop(crop: Any) -> Any:
    target_height = 120
    scale = target_height / max(1, crop.shape[0])
    return cv.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), target_height),
        interpolation=cv.INTER_AREA,
    )


def padded_crop(frame: Any, bbox: tuple[int, int, int, int], *, category: str) -> Any:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_fraction = 0.42 if category == CATEGORY_FACE else 0.55
    pad = max(10, int(max(x2 - x1, y2 - y1) * pad_fraction))
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(frame_width, x2 + pad)
    bottom = min(frame_height, y2 + pad)
    return frame[top:bottom, left:right]


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    left = max(0, min(frame_width, min(x1, x2)))
    right = max(0, min(frame_width, max(x1, x2)))
    top = max(0, min(frame_height, min(y1, y2)))
    bottom = max(0, min(frame_height, max(y1, y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(left) + bbox_area(right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def center_separation_norm(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    lcx = (lx1 + lx2) / 2.0
    lcy = (ly1 + ly2) / 2.0
    rcx = (rx1 + rx2) / 2.0
    rcy = (ry1 + ry2) / 2.0
    scale = max(lx2 - lx1, ly2 - ly1, rx2 - rx1, ry2 - ry1, 1)
    return math.hypot(lcx - rcx, lcy - rcy) / scale


def parse_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def classification_rank(classification: str) -> int:
    return {
        CANDIDATE_POSSIBLE: 0,
        CANDIDATE_REJECT_TEMPORAL_CONFLICT: 1,
        CANDIDATE_REJECT_VISUAL_MISMATCH: 2,
        CANDIDATE_INSUFFICIENT: 3,
    }.get(classification, 4)


def purity_counts(cases: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = {state: 0 for state in sorted(PURITY_STATES)}
    for entities in cases.values():
        for entity in entities:
            purity = str(entity.get("purity_classification") or PURITY_UNVERIFIED)
            counts[purity] = counts.get(purity, 0) + 1
    return counts
