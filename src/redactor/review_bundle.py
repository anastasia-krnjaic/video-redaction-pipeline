# ruff: noqa: E501

from __future__ import annotations

import json
import math
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import media, rendering
from redactor.detection import rational_to_float

SCHEMA_VERSION = "1.0"
REVIEW_QUEUE_SCHEMA_ID = "face_review_queue_schema_v1.0"
REVIEW_BUNDLE_ID = "automated_review_bundle_v1"

SEVERITY_ORDER = {
    "blocking": 0,
    "high": 1,
    "medium": 2,
    "informational": 3,
    "random_audit": 4,
}
VALID_SEVERITIES = set(SEVERITY_ORDER)
VALID_REVIEW_STATUSES = {
    "pending",
    "accepted_current",
    "correction_required",
    "corrected",
    "false_positive",
    "intentionally_unredacted",
    "deferred",
    "escalated",
}
DEFAULT_ALLOWED_ACTIONS = [
    "accepted_current",
    "correction_required",
    "false_positive",
    "intentionally_unredacted",
    "deferred",
    "escalated",
]
CORRECTION_ONLY_ACTIONS = [
    "correction_required",
    "corrected",
    "deferred",
    "escalated",
]
BLOCKING_REASON_CODES = {
    "needs_review_interval",
    "rejected_interval",
    "unresolved_privacy_frames",
    "invalid_box",
    "render_validation_failure",
    "media_validation_failure",
    "ground_truth_privacy_miss",
    "possible_keep_person_overlap",
}
CORRECTION_REQUIRED_REASON_CODES = {
    "rejected_interval",
    "unresolved_privacy_frames",
    "ground_truth_privacy_miss",
    "invalid_box",
}
GROUND_TRUTH_REASON_MAP = {
    "detector_scored_frame_without_adjudicated_candidate": "ground_truth_detector_miss",
    "adjudicated_candidate_below_detector_threshold": "ground_truth_detector_miss",
    "adjudicated_candidate_missing_from_detector_output": "ground_truth_detector_miss",
    "privacy_required_frame_uncovered": "ground_truth_privacy_miss",
    "invalid_privacy_box": "invalid_box",
    "keep_person_false_redaction_overlap": "possible_keep_person_overlap",
    "identity_switch_event": "possible_identity_competition",
    "source_video_hash_mismatch": "render_validation_failure",
    "privacy_region_input_hash_mismatch": "render_validation_failure",
    "output_video_hash_mismatch": "render_validation_failure",
    "final_approval_false": "render_validation_failure",
    "media_validation_failed": "media_validation_failure",
}
INTERVAL_REASON_MAP = {
    "excessive_scale_change": "excessive_scale_change",
    "excessive_center_jump": "excessive_center_jump",
    "excessive_refined_disagreement": "excessive_bidirectional_disagreement",
    "excessive_raw_disagreement": "excessive_bidirectional_disagreement",
    "gap_duration_exceeds_max_one_sided_gap_frames": "one_sided_recovery",
    "scene_cut_detected": "scene_cut_inside_track",
    "keep_overlap": "possible_keep_person_overlap",
    "identity_competition": "possible_identity_competition",
    "invalid_box": "invalid_box",
}
TRACK_REASON_MAP = {
    "late_track_start": "track_start",
    "early_track_end": "track_end",
    "review_candidate_track": "low_confidence_observations",
    "rejected_track": "rejected_interval",
    "long_gap": "long_detection_gap",
    "unresolved_interval": "unresolved_privacy_frames",
    "interpolated_span": "suspicious_provenance_transition",
    "predicted_span": "one_sided_recovery",
    "mandatory_scene_start_review": "track_start",
}

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class ReviewBundleConfig:
    case_id: str | None = None
    object_category: str = "face"
    random_sample_count: int = 3
    random_sample_frame_count: int = 24
    random_seed: int = 20260803
    merge_gap_frames: int = 12
    context_frames: int = 18
    contact_sheet_max_frames: int = 9
    clip_context_seconds: float = 0.50
    generate_visuals: bool = True
    created_at: str | None = None


@dataclass
class ReviewDraft:
    item_origin: str
    source_stage: str
    frame_start: int
    frame_end: int
    reason_codes: set[str]
    severity: str
    explanation: str
    object_id: str | None = None
    track_ids: set[int] = field(default_factory=set)
    metrics: dict[str, Any] = field(default_factory=dict)
    unresolved_frame_count: int = 0
    candidate_region_status: str = "unknown"
    frame_references: set[int] = field(default_factory=set)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking" or bool(self.reason_codes & BLOCKING_REASON_CODES)

    @property
    def requires_corrected_regions(self) -> bool:
        return bool(self.reason_codes & CORRECTION_REQUIRED_REASON_CODES)


def build_review_bundle(
    *,
    source_video_path: Path,
    output_dir: Path,
    config: ReviewBundleConfig,
    detections_path: Path | None = None,
    tracks_path: Path | None = None,
    track_quality_path: Path | None = None,
    track_review_flags_path: Path | None = None,
    track_review_windows_path: Path | None = None,
    privacy_regions_path: Path | None = None,
    interval_decisions_path: Path | None = None,
    render_manifest_path: Path | None = None,
    render_validation_path: Path | None = None,
    automatic_preview_video_path: Path | None = None,
    reviewed_final_video_path: Path | None = None,
    evaluation_dir: Path | None = None,
    ground_truth_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    ensure_output_dir(output_dir, overwrite=overwrite)
    bundle_paths = create_bundle_dirs(output_dir)
    source_info = media.probe_media(source_video_path)
    fps = rational_to_float(source_info.video_stream.nominal_frame_rate) or 30.0
    frame_count = int(source_info.video_stream.frame_count or 0)
    if frame_count <= 0:
        frame_count = round(float(source_info.duration_seconds or 0) * fps)
    created_at = config.created_at or media.utc_timestamp()
    case_id = config.case_id or source_video_path.stem

    privacy_input = (
        rendering.load_privacy_regions(privacy_regions_path)
        if privacy_regions_path is not None
        else rendering.PrivacyRegionInput({}, (), "not_supplied")
    )
    intervals = rendering.load_interval_decisions(interval_decisions_path)
    evaluation_payload = load_evaluation_payload(evaluation_dir)
    render_manifest = load_json(render_manifest_path)
    render_validation = load_json(render_validation_path)
    track_quality = load_json(track_quality_path)
    track_review_flags = load_json(track_review_flags_path)
    track_review_windows = load_json(track_review_windows_path)
    ground_truth = load_json(ground_truth_path)

    drafts = build_review_drafts(
        case_id=case_id,
        source_video_hash=source_info.sha256,
        frame_count=frame_count,
        fps=fps,
        config=config,
        privacy_input=privacy_input,
        intervals=intervals,
        render_manifest=render_manifest,
        render_validation=render_validation,
        evaluation_payload=evaluation_payload,
        track_quality=track_quality,
        track_review_flags=track_review_flags,
        track_review_windows=track_review_windows,
        ground_truth=ground_truth,
    )
    items = review_items_from_drafts(
        drafts,
        case_id=case_id,
        source_video_hash=source_info.sha256,
        fps=fps,
        created_at=created_at,
        config=config,
    )
    if config.generate_visuals:
        add_visual_evidence(
            items,
            source_video_path=source_video_path,
            automatic_preview_video_path=automatic_preview_video_path,
            reviewed_final_video_path=reviewed_final_video_path,
            privacy_input=privacy_input,
            output_dir=output_dir,
            bundle_paths=bundle_paths,
            fps=fps,
            frame_count=frame_count,
            config=config,
            overwrite=overwrite,
        )

    queue = queue_payload(
        items,
        case_id=case_id,
        source_info=source_info,
        config=config,
        created_at=created_at,
        paths={
            "automatic_preview_video": automatic_preview_video_path,
            "detections": detections_path,
            "evaluation_dir": evaluation_dir,
            "ground_truth": ground_truth_path,
            "interval_decisions": interval_decisions_path,
            "privacy_regions": privacy_regions_path,
            "render_manifest": render_manifest_path,
            "render_validation": render_validation_path,
            "reviewed_final_video": reviewed_final_video_path,
            "source_video": source_video_path,
            "track_quality": track_quality_path,
            "track_review_flags": track_review_flags_path,
            "track_review_windows": track_review_windows_path,
            "tracks": tracks_path,
        },
    )
    summary = review_summary(queue)
    write_bundle_outputs(
        output_dir,
        queue,
        summary,
        config=config,
        source_info=source_info,
        created_at=created_at,
    )
    return summary


def build_review_drafts(
    *,
    case_id: str,
    source_video_hash: str,
    frame_count: int,
    fps: float,
    config: ReviewBundleConfig,
    privacy_input: rendering.PrivacyRegionInput,
    intervals: list[dict[str, object]],
    render_manifest: dict[str, Any],
    render_validation: dict[str, Any],
    evaluation_payload: dict[str, Any],
    track_quality: dict[str, Any],
    track_review_flags: dict[str, Any],
    track_review_windows: dict[str, Any],
    ground_truth: dict[str, Any],
) -> list[ReviewDraft]:
    del case_id, source_video_hash, ground_truth
    drafts: list[ReviewDraft] = []
    drafts.extend(interval_decision_drafts(intervals))
    drafts.extend(unresolved_privacy_drafts(privacy_input))
    drafts.extend(invalid_box_drafts(privacy_input))
    drafts.extend(render_validation_drafts(render_manifest, render_validation, frame_count))
    drafts.extend(evaluation_failure_drafts(evaluation_payload))
    drafts.extend(evaluation_metric_drafts(evaluation_payload))
    drafts.extend(track_flag_drafts(track_review_flags))
    drafts.extend(track_window_drafts(track_review_windows))
    drafts.extend(track_quality_drafts(track_quality))
    drafts.extend(provenance_transition_drafts(privacy_input))

    merged = merge_review_drafts(drafts, merge_gap_frames=config.merge_gap_frames)
    blocking_ranges = [(draft.frame_start, draft.frame_end) for draft in merged if draft.blocking]
    merged.extend(
        random_audit_drafts(
            frame_count=frame_count,
            fps=fps,
            config=config,
            blocking_ranges=blocking_ranges,
        )
    )
    return sorted(merged, key=draft_sort_key)


def interval_decision_drafts(intervals: list[dict[str, object]]) -> list[ReviewDraft]:
    drafts: list[ReviewDraft] = []
    for interval in intervals:
        status = str(interval.get("status") or "unknown")
        strategy = str(interval.get("strategy") or "unknown")
        frame_start = as_int(interval.get("frame_start"), 0)
        frame_end = as_int(interval.get("frame_end"), frame_start)
        reasons = stable_reason_codes(interval.get("reasons") or [], INTERVAL_REASON_MAP)
        if status == "needs_review":
            reasons.add("needs_review_interval")
        elif status == "rejected":
            reasons.add("rejected_interval")
        elif strategy == "one_sided_tracking":
            reasons.add("one_sided_recovery")
        if strategy == "preserved_privacy_region":
            reasons.add("preserved_reviewed_region")
        if not reasons:
            continue
        severity = "informational"
        if status in {"needs_review", "rejected"}:
            severity = "blocking"
        elif "one_sided_recovery" in reasons:
            severity = "medium"
        explanation = interval_explanation(status, strategy, reasons)
        drafts.append(
            ReviewDraft(
                item_origin="pipeline",
                source_stage="privacy_gap_recovery",
                frame_start=frame_start,
                frame_end=frame_end,
                reason_codes=reasons,
                severity=severity,
                explanation=explanation,
                track_ids={int(value) for value in interval.get("target_track_ids", [])},
                metrics=dict(interval.get("metrics") or {}),
                unresolved_frame_count=as_int(interval.get("unresolved_frames"), 0),
                candidate_region_status="available"
                if as_int(interval.get("recovered_frames"), 0) > 0
                else "unresolved",
                frame_references=set(range(frame_start, frame_end + 1)),
                provenance={
                    "before": "ordinary_tracker_or_detector_anchor",
                    "during": strategy,
                    "after": status,
                },
            )
        )
    return drafts


def unresolved_privacy_drafts(
    privacy_input: rendering.PrivacyRegionInput,
) -> list[ReviewDraft]:
    frames = [
        frame
        for frame in privacy_input.required_frames
        if any(not region.is_renderable for region in privacy_input.regions_by_frame.get(frame, []))
        or not privacy_input.regions_by_frame.get(frame)
    ]
    return [
        ReviewDraft(
            item_origin="pipeline",
            source_stage="privacy_gap_recovery",
            frame_start=start,
            frame_end=end,
            reason_codes={"unresolved_privacy_frames"},
            severity="blocking",
            explanation=(
                "Privacy-required frames have no valid region. Final export must remain blocked "
                "until reviewed regions are supplied."
            ),
            unresolved_frame_count=end - start + 1,
            candidate_region_status="unresolved",
            frame_references=set(range(start, end + 1)),
            provenance={"during": "unresolved"},
        )
        for start, end in contiguous_ranges(frames)
    ]


def invalid_box_drafts(privacy_input: rendering.PrivacyRegionInput) -> list[ReviewDraft]:
    frames: list[int] = []
    for frame, regions in privacy_input.regions_by_frame.items():
        for region in regions:
            if region.bbox_xyxy is not None and not valid_box(region.bbox_xyxy):
                frames.append(frame)
    return [
        ReviewDraft(
            item_origin="pipeline",
            source_stage="privacy_gap_recovery",
            frame_start=start,
            frame_end=end,
            reason_codes={"invalid_box"},
            severity="blocking",
            explanation="A privacy region has invalid or zero-area coordinates.",
            unresolved_frame_count=0,
            candidate_region_status="invalid",
            frame_references=set(range(start, end + 1)),
        )
        for start, end in contiguous_ranges(frames)
    ]


def render_validation_drafts(
    render_manifest: dict[str, Any],
    render_validation: dict[str, Any],
    frame_count: int,
) -> list[ReviewDraft]:
    if not render_manifest and not render_validation:
        return []
    drafts: list[ReviewDraft] = []
    review_status = render_manifest.get("review_status") or {}
    unresolved = as_int(review_status.get("unresolved_frame_count"), 0)
    if unresolved and not render_manifest.get("final_approval_status"):
        unresolved_intervals = [
            interval
            for interval in render_manifest.get("unresolved_intervals", [])
            if isinstance(interval, dict)
        ]
        if unresolved_intervals:
            for interval in unresolved_intervals:
                reasons = {
                    *stable_reason_codes(interval.get("reasons", []), {}),
                    "render_validation_failure",
                    "unresolved_privacy_frames",
                }
                drafts.append(
                    ReviewDraft(
                        item_origin="pipeline",
                        source_stage="rendering",
                        frame_start=as_int(interval.get("frame_start"), 0),
                        frame_end=as_int(interval.get("frame_end"), 0),
                        reason_codes=reasons,
                        severity="blocking",
                        explanation="The render manifest reports an unresolved privacy interval.",
                        metrics={
                            "interval_status": interval.get("status"),
                            "review_status": review_status,
                            "strategy": interval.get("strategy"),
                        },
                        unresolved_frame_count=as_int(
                            interval.get("unresolved_frames"),
                            max(
                                0,
                                as_int(interval.get("frame_end"), 0)
                                - as_int(interval.get("frame_start"), 0)
                                + 1,
                            ),
                        ),
                        candidate_region_status="unresolved",
                        frame_references=set(
                            range(
                                as_int(interval.get("frame_start"), 0),
                                as_int(interval.get("frame_end"), 0) + 1,
                            )
                        ),
                    )
                )
        else:
            drafts.append(
                ReviewDraft(
                    item_origin="pipeline",
                    source_stage="rendering",
                    frame_start=0,
                    frame_end=max(0, frame_count - 1),
                    reason_codes={"unresolved_privacy_frames", "render_validation_failure"},
                    severity="blocking",
                    explanation="The render manifest reports unresolved privacy frames.",
                    metrics={"review_status": review_status},
                    unresolved_frame_count=unresolved,
                    candidate_region_status="unresolved",
                )
            )
    media_validation = render_manifest.get("media_validation")
    if not media_validation:
        media_validation = render_validation.get("media_validation")
    if isinstance(media_validation, dict) and media_validation.get("passed") is False:
        drafts.append(
            ReviewDraft(
                item_origin="pipeline",
                source_stage="rendering",
                frame_start=0,
                frame_end=max(0, frame_count - 1),
                reason_codes={"media_validation_failure"},
                severity="blocking",
                explanation="Rendered media failed validation.",
                metrics={"media_validation": media_validation},
            )
        )
    return drafts


def evaluation_failure_drafts(evaluation_payload: dict[str, Any]) -> list[ReviewDraft]:
    failures = (evaluation_payload.get("failures") or {}).get("failures") or []
    drafts: list[ReviewDraft] = []
    for item in failures:
        reason = str(item.get("failure_reason") or "unknown")
        reason_code = GROUND_TRUTH_REASON_MAP.get(reason, reason)
        stage = str(item.get("evaluated_stage") or "evaluation")
        frame = as_optional_int(item.get("frame"))
        frame_start = as_optional_int(item.get("frame_start"))
        frame_end = as_optional_int(item.get("frame_end"))
        if frame_start is None:
            frame_start = 0 if frame is None else frame
        if frame_end is None:
            frame_end = frame_start if frame is None else frame
        severity = "medium"
        if reason_code in {
            "ground_truth_privacy_miss",
            "possible_keep_person_overlap",
            "invalid_box",
        }:
            severity = "blocking"
        elif reason_code == "ground_truth_detector_miss":
            severity = "high"
        drafts.append(
            ReviewDraft(
                item_origin="ground_truth_evaluation",
                source_stage=stage,
                frame_start=frame_start,
                frame_end=frame_end,
                reason_codes={reason_code},
                severity=severity,
                explanation=evaluation_explanation(reason_code, item),
                object_id=none_if_empty(item.get("object_id")),
                metrics={"failure_reason": reason, "expected_policy": item.get("expected_policy")},
                unresolved_frame_count=1 if reason_code == "ground_truth_privacy_miss" else 0,
                candidate_region_status=str(item.get("observed_provenance") or "unknown"),
                frame_references={frame_start},
                provenance={
                    "during": item.get("observed_provenance"),
                    "review_status": item.get("review_status"),
                },
            )
        )
    return drafts


def evaluation_metric_drafts(evaluation_payload: dict[str, Any]) -> list[ReviewDraft]:
    objects = (evaluation_payload.get("object_metrics") or {}).get("objects") or {}
    drafts: list[ReviewDraft] = []
    for object_id, item in objects.items():
        tracking = item.get("tracking") or {}
        stability = item.get("stability") or {}
        privacy = item.get("privacy") or {}
        detector = item.get("detector") or {}
        if as_int(tracking.get("track_fragmentation"), 0) > 1:
            frames = [
                int(start["frame"])
                for start in tracking.get("track_starts", [])
                if start.get("frame") is not None
            ]
            frame_start, frame_end = bounds_or_zero(frames)
            drafts.append(
                ReviewDraft(
                    item_origin="pipeline",
                    source_stage="tracking",
                    frame_start=frame_start,
                    frame_end=frame_end,
                    reason_codes={"track_fragmentation"},
                    severity="medium",
                    explanation="The target is represented by multiple track fragments.",
                    object_id=str(object_id),
                    track_ids={int(tid) for tid in tracking.get("fragment_track_ids", [])},
                    metrics={"track_fragmentation": tracking.get("track_fragmentation")},
                    frame_references=set(frames),
                )
            )
        if as_int(stability.get("suspicious_provenance_transitions"), 0) > 0:
            frames = privacy.get("detector_scored_visible_uncovered_frames") or []
            if not frames:
                frames = detector.get("missed_detector_scored_frames") or []
            frame_refs = sorted(int(frame) for frame in frames)
            frame_start = frame_refs[0] if frame_refs else 0
            drafts.append(
                ReviewDraft(
                    item_origin="pipeline",
                    source_stage="privacy_gap_recovery",
                    frame_start=frame_start,
                    frame_end=frame_start,
                    reason_codes={"suspicious_provenance_transition"},
                    severity="informational",
                    explanation="The privacy evidence changes provenance near this interval.",
                    object_id=str(object_id),
                    metrics={
                        "suspicious_provenance_transitions": stability.get(
                            "suspicious_provenance_transitions"
                        )
                    },
                    frame_references=set(frame_refs[:25]),
                )
            )
    return drafts


def track_window_drafts(track_review_windows: dict[str, Any]) -> list[ReviewDraft]:
    windows = track_review_windows.get("review_windows") or []
    drafts: list[ReviewDraft] = []
    for window in windows:
        flag_types = [str(value) for value in window.get("flag_types", [])]
        reasons = stable_reason_codes(flag_types, TRACK_REASON_MAP)
        if not reasons:
            continue
        severity = "medium"
        if "rejected_interval" in reasons or "unresolved_privacy_frames" in reasons:
            severity = "high"
        drafts.append(
            ReviewDraft(
                item_origin="pipeline",
                source_stage="tracking",
                frame_start=as_int(window.get("frame_start"), 0),
                frame_end=as_int(window.get("frame_end"), 0),
                reason_codes=reasons,
                severity=severity,
                explanation="The tracker marked this interval for diagnostic review.",
                track_ids={int(value) for value in window.get("track_ids", [])},
                metrics={"flag_types": flag_types, "reasons": window.get("reasons", [])},
                candidate_region_status="not_applicable",
            )
        )
    return drafts


def track_flag_drafts(track_review_flags: dict[str, Any]) -> list[ReviewDraft]:
    flags = track_review_flags.get("review_flags") or []
    drafts: list[ReviewDraft] = []
    for flag in flags:
        flag_type = str(flag.get("type") or "")
        reasons = stable_reason_codes([flag_type], TRACK_REASON_MAP)
        if not reasons:
            continue
        severity = "medium"
        if "rejected_interval" in reasons or "unresolved_privacy_frames" in reasons:
            severity = "high"
        drafts.append(
            ReviewDraft(
                item_origin="pipeline",
                source_stage="tracking",
                frame_start=as_int(flag.get("frame_start"), 0),
                frame_end=as_int(flag.get("frame_end"), 0),
                reason_codes=reasons,
                severity=severity,
                explanation=str(flag.get("reason") or "The tracker marked this frame for review."),
                track_ids={as_int(flag.get("track_id"), 0)}
                if flag.get("track_id") is not None
                else set(),
                metrics={"flag_type": flag_type},
                candidate_region_status="not_applicable",
            )
        )
    return drafts


def track_quality_drafts(track_quality: dict[str, Any]) -> list[ReviewDraft]:
    tracks = track_quality.get("tracks") or []
    drafts: list[ReviewDraft] = []
    for track in tracks:
        status = str(track.get("quality_status") or "")
        failures = [str(value) for value in track.get("quality_gate_failures", [])]
        if (
            status not in {"review_candidate", "rejected"}
            and "low_median_observed_score" not in failures
        ):
            continue
        reason_codes = {"low_confidence_observations"}
        if status == "rejected":
            reason_codes.add("rejected_interval")
        if as_int(track.get("weak_hits"), 0) and not as_int(track.get("strong_hits"), 0):
            reason_codes.add("weak_only_observations")
        severity = "high" if status == "rejected" else "medium"
        drafts.append(
            ReviewDraft(
                item_origin="pipeline",
                source_stage="tracking",
                frame_start=as_int(track.get("first_frame"), 0),
                frame_end=as_int(track.get("last_frame"), 0),
                reason_codes=reason_codes,
                severity=severity,
                explanation="A track failed or nearly failed automatic confidence gates.",
                track_ids={as_int(track.get("track_id"), 0)},
                metrics=dict(track),
                candidate_region_status="not_applicable",
            )
        )
    return drafts


def provenance_transition_drafts(
    privacy_input: rendering.PrivacyRegionInput,
) -> list[ReviewDraft]:
    ordered: list[tuple[int, str]] = []
    for frame in sorted(privacy_input.regions_by_frame):
        regions = [
            region
            for region in privacy_input.regions_by_frame.get(frame, [])
            if region.is_renderable
        ]
        if regions:
            ordered.append((frame, regions[0].provenance))
    drafts: list[ReviewDraft] = []
    previous: tuple[int, str] | None = None
    for frame, provenance in ordered:
        if previous is not None and frame == previous[0] + 1 and provenance != previous[1]:
            drafts.append(
                ReviewDraft(
                    item_origin="pipeline",
                    source_stage="privacy_gap_recovery",
                    frame_start=frame,
                    frame_end=frame,
                    reason_codes={"suspicious_provenance_transition"},
                    severity="informational",
                    explanation="The privacy-region source changes at this frame.",
                    metrics={"from": previous[1], "to": provenance},
                    candidate_region_status="available",
                    frame_references={frame},
                    provenance={"before": previous[1], "during": provenance},
                )
            )
        previous = (frame, provenance)
    return drafts


def random_audit_drafts(
    *,
    frame_count: int,
    fps: float,
    config: ReviewBundleConfig,
    blocking_ranges: list[tuple[int, int]],
) -> list[ReviewDraft]:
    del fps
    if frame_count <= 0 or config.random_sample_count <= 0:
        return []
    sample_len = max(1, min(config.random_sample_frame_count, frame_count))
    rng = random.Random(config.random_seed)
    blocked = set()
    for start, end in blocking_ranges:
        blocked.update(range(max(0, start), min(frame_count - 1, end) + 1))
    drafts: list[ReviewDraft] = []
    attempts = 0
    used_ranges: list[tuple[int, int]] = []
    while len(drafts) < config.random_sample_count and attempts < config.random_sample_count * 100:
        attempts += 1
        start = rng.randint(0, max(0, frame_count - sample_len))
        end = start + sample_len - 1
        frames = set(range(start, end + 1))
        if frames & blocked:
            continue
        if any(
            ranges_overlap(start, end, used_start, used_end) for used_start, used_end in used_ranges
        ):
            continue
        used_ranges.append((start, end))
        drafts.append(
            ReviewDraft(
                item_origin="random_audit",
                source_stage="random_audit",
                frame_start=start,
                frame_end=end,
                reason_codes={"random_audit_sample"},
                severity="random_audit",
                explanation=(
                    "Deterministic audit sample from an apparently normal interval. "
                    "This helps reduce silent-failure risk but cannot prove there were no misses."
                ),
                metrics={
                    "sample_frame_count": sample_len,
                    "seed": config.random_seed,
                    "selection_policy": "avoid_known_blocking_intervals",
                },
                candidate_region_status="not_applicable",
                frame_references={start, end},
            )
        )
    return drafts


def merge_review_drafts(
    drafts: list[ReviewDraft],
    *,
    merge_gap_frames: int,
) -> list[ReviewDraft]:
    non_random = [draft for draft in drafts if "random_audit_sample" not in draft.reason_codes]
    random_items = [draft for draft in drafts if "random_audit_sample" in draft.reason_codes]
    merged: list[ReviewDraft] = []
    for draft in sorted(non_random, key=lambda item: (item.frame_start, item.frame_end)):
        merge_index = next(
            (
                index
                for index in range(len(merged) - 1, -1, -1)
                if should_merge(merged[index], draft, merge_gap_frames)
            ),
            None,
        )
        if merge_index is None:
            merged.append(draft)
            continue
        merged[merge_index] = merge_two_drafts(merged[merge_index], draft)
    return [*merged, *random_items]


def should_merge(left: ReviewDraft, right: ReviewDraft, merge_gap_frames: int) -> bool:
    if left.item_origin == "random_audit" or right.item_origin == "random_audit":
        return False
    if left.object_id and right.object_id and left.object_id != right.object_id:
        return False
    detector_miss_only = {"ground_truth_detector_miss"}
    if (
        left.reason_codes == detector_miss_only or right.reason_codes == detector_miss_only
    ) and left.reason_codes != right.reason_codes:
        return False
    if left.frame_end + merge_gap_frames < right.frame_start:
        return False
    shared_reasons = left.reason_codes & right.reason_codes
    preserved_reasons = {
        "preserved_existing_accepted_privacy_region",
        "preserved_reviewed_region",
    }
    if (left.reason_codes | right.reason_codes) & preserved_reasons:
        return bool(
            (left.reason_codes & preserved_reasons)
            and (right.reason_codes & preserved_reasons)
            and ranges_overlap(
                left.frame_start,
                left.frame_end,
                right.frame_start,
                right.frame_end,
            )
        )
    interval_reasons = {
        "needs_review_interval",
        "rejected_interval",
    }
    if (left.reason_codes | right.reason_codes) & interval_reasons:
        return ranges_overlap(left.frame_start, left.frame_end, right.frame_start, right.frame_end)
    if shared_reasons:
        return True
    if left.blocking or right.blocking:
        if not (left.blocking and right.blocking):
            return False
        return ranges_overlap(left.frame_start, left.frame_end, right.frame_start, right.frame_end)
    return left.source_stage == right.source_stage


def merge_two_drafts(left: ReviewDraft, right: ReviewDraft) -> ReviewDraft:
    severity = min([left.severity, right.severity], key=lambda value: SEVERITY_ORDER[value])
    merged_metrics = {
        "merged_events": [
            compact_draft_metrics(left),
            compact_draft_metrics(right),
        ]
    }
    return ReviewDraft(
        item_origin=left.item_origin
        if left.item_origin == right.item_origin
        else "mixed_pipeline_and_evaluation",
        source_stage=left.source_stage
        if left.source_stage == right.source_stage
        else "multiple_stages",
        frame_start=min(left.frame_start, right.frame_start),
        frame_end=max(left.frame_end, right.frame_end),
        reason_codes=left.reason_codes | right.reason_codes,
        severity=severity,
        explanation=combine_explanations(left.explanation, right.explanation),
        object_id=left.object_id or right.object_id,
        track_ids=left.track_ids | right.track_ids,
        metrics=merged_metrics,
        unresolved_frame_count=max(left.unresolved_frame_count, right.unresolved_frame_count),
        candidate_region_status=combine_candidate_status(
            left.candidate_region_status,
            right.candidate_region_status,
        ),
        frame_references=left.frame_references | right.frame_references,
        provenance={**left.provenance, **right.provenance},
    )


def review_items_from_drafts(
    drafts: list[ReviewDraft],
    *,
    case_id: str,
    source_video_hash: str,
    fps: float,
    created_at: str,
    config: ReviewBundleConfig,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, draft in enumerate(sorted(drafts, key=draft_sort_key), start=1):
        reason_codes = sorted(draft.reason_codes)
        item_id = stable_item_id(case_id, draft, index)
        blocking = draft.blocking
        requires_corrected_regions = draft.requires_corrected_regions
        allowed_actions = (
            CORRECTION_ONLY_ACTIONS if requires_corrected_regions else DEFAULT_ALLOWED_ACTIONS
        )
        item = {
            "allowed_review_actions": allowed_actions,
            "available_candidate_region_status": draft.candidate_region_status,
            "blocking_status": blocking,
            "case_id": case_id,
            "creation_metadata": {
                "created_at": created_at,
                "generator": REVIEW_BUNDLE_ID,
                "merge_gap_frames": config.merge_gap_frames,
                "random_seed": config.random_seed,
            },
            "explanation": draft.explanation,
            "frame_end": draft.frame_end,
            "frame_references": sorted(draft.frame_references),
            "frame_start": draft.frame_start,
            "item_origin": draft.item_origin,
            "metrics": draft.metrics,
            "object_category": config.object_category,
            "object_id": draft.object_id,
            "provenance": {
                "after": draft.provenance.get("after"),
                "before": draft.provenance.get("before"),
                "during": draft.provenance.get("during"),
                "raw": draft.provenance,
            },
            "reason_codes": reason_codes,
            "requires_corrected_regions": requires_corrected_regions,
            "review_item_id": item_id,
            "review_status": "pending",
            "schema_version": SCHEMA_VERSION,
            "severity": draft.severity,
            "source_stage": draft.source_stage,
            "source_video_hash": source_video_hash,
            "timestamps": {
                "end_seconds": frame_to_seconds(draft.frame_end, fps),
                "start_seconds": frame_to_seconds(draft.frame_start, fps),
            },
            "track_ids": sorted(draft.track_ids),
            "unresolved_frame_count": draft.unresolved_frame_count,
            "visual_evidence": {},
        }
        items.append(item)
    return items


def queue_payload(
    items: list[dict[str, Any]],
    *,
    case_id: str,
    source_info: Any,
    config: ReviewBundleConfig,
    created_at: str,
    paths: dict[str, Path | None],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "created_at": created_at,
        "generator": REVIEW_BUNDLE_ID,
        "input_hashes": {
            key: media.sha256_file(path) for key, path in paths.items() if path and path.is_file()
        },
        "input_paths": {
            key: path.name if path and path.is_file() else None for key, path in paths.items()
        },
        "items": items,
        "random_audit_policy": {
            "avoid_blocking_intervals": True,
            "interval_length_frames": config.random_sample_frame_count,
            "sample_count": config.random_sample_count,
            "seed": config.random_seed,
        },
        "schema_id": REVIEW_QUEUE_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source_video": {
            "duration_seconds": source_info.duration_seconds,
            "fps": source_info.video_stream.nominal_frame_rate,
            "frame_count": source_info.video_stream.frame_count,
            "hash": source_info.sha256,
            "height": source_info.video_stream.height,
            "width": source_info.video_stream.width,
        },
    }


def review_summary(queue: dict[str, Any]) -> dict[str, Any]:
    items = queue.get("items", [])
    counts_by_severity = count_values(item["severity"] for item in items)
    blocking_items = [item for item in items if item["blocking_status"]]
    random_items = [item for item in items if item["severity"] == "random_audit"]
    reason_counts = count_values(
        reason for item in items for reason in item.get("reason_codes", [])
    )
    return {
        "accepted_items": 0,
        "blocking_items": len(blocking_items),
        "blocker_reasons": sorted(
            {reason for item in blocking_items for reason in item.get("reason_codes", [])}
        ),
        "case_id": queue.get("case_id"),
        "correction_required_items": 0,
        "false_positive_items": 0,
        "final_export_blocked": bool(blocking_items),
        "pending_items": len(items),
        "random_audit_items": len(random_items),
        "reason_counts": reason_counts,
        "review_completeness_status": "pending" if items else "no_review_items",
        "schema_version": SCHEMA_VERSION,
        "severity_counts": counts_by_severity,
        "total_items": len(items),
    }


def validate_review_queue_payload(queue: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if queue.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported review queue schema_version")
    items = queue.get("items")
    if not isinstance(items, list):
        return [*errors, "items must be a list"]
    seen_ids: set[str] = set()
    for item in items:
        item_id = str(item.get("review_item_id") or "")
        if not item_id:
            errors.append("review item missing review_item_id")
        elif item_id in seen_ids:
            errors.append(f"duplicate review_item_id: {item_id}")
        seen_ids.add(item_id)
        if item.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{item_id}: unsupported item schema_version")
        if item.get("severity") not in VALID_SEVERITIES:
            errors.append(f"{item_id}: invalid severity")
        if item.get("review_status") not in VALID_REVIEW_STATUSES:
            errors.append(f"{item_id}: invalid review_status")
        if as_int(item.get("frame_start"), -1) > as_int(item.get("frame_end"), -1):
            errors.append(f"{item_id}: frame_start greater than frame_end")
        if not item.get("reason_codes"):
            errors.append(f"{item_id}: missing reason_codes")
        for key in (
            "contact_sheet",
            "source_clip",
            "automatic_preview_clip",
            "reviewed_final_clip",
        ):
            value = (item.get("visual_evidence") or {}).get(key)
            if value and Path(str(value)).is_absolute():
                errors.append(f"{item_id}: visual_evidence contains absolute path")
    return errors


def validate_review_decisions(
    queue_path: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    queue = load_json(queue_path)
    decisions = load_json(decisions_path)
    errors = validate_review_queue_payload(queue)
    if decisions.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported decisions schema_version")
    items_by_id = {item["review_item_id"]: item for item in queue.get("items", [])}
    seen: set[str] = set()
    decision_rows = decisions.get("decisions")
    if not isinstance(decision_rows, list):
        decision_rows = []
        errors.append("decisions must be a list")
    unresolved_blockers: list[str] = []
    correction_files_required: list[str] = []
    for decision in decision_rows:
        item_id = str(decision.get("review_item_id") or "")
        if item_id in seen:
            errors.append(f"duplicate decision for review item: {item_id}")
            continue
        seen.add(item_id)
        item = items_by_id.get(item_id)
        if item is None:
            errors.append(f"unknown review_item_id: {item_id}")
            continue
        status = str(decision.get("review_status") or "")
        if status not in VALID_REVIEW_STATUSES:
            errors.append(f"{item_id}: invalid review_status")
            continue
        if status not in set(item.get("allowed_review_actions", [])) | {"pending"}:
            errors.append(f"{item_id}: review_status not allowed for this item")
        correction_artifact = decision.get("correction_artifact")
        if (
            item.get("requires_corrected_regions")
            and status == "corrected"
            and not correction_artifact
        ):
            correction_files_required.append(item_id)
    for item_id, item in items_by_id.items():
        decision = next(
            (row for row in decision_rows if row.get("review_item_id") == item_id),
            None,
        )
        status = str((decision or {}).get("review_status") or "pending")
        if not item.get("blocking_status"):
            continue
        if status in {"pending", "correction_required", "deferred", "escalated"}:
            unresolved_blockers.append(item_id)
        elif item.get("requires_corrected_regions") and status != "corrected":
            unresolved_blockers.append(item_id)
        elif item.get("requires_corrected_regions") and not (decision or {}).get(
            "correction_artifact"
        ):
            unresolved_blockers.append(item_id)
    return {
        "correction_files_required": sorted(set(correction_files_required)),
        "decision_count": len(decision_rows),
        "duplicate_decision_count": len(decision_rows) - len(seen),
        "error_count": len(errors),
        "errors": errors,
        "final_export_blocked": bool(unresolved_blockers or correction_files_required or errors),
        "known_review_item_count": len(items_by_id),
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if not errors else "invalid",
        "unresolved_blocking_items": sorted(set(unresolved_blockers)),
        "unresolved_blocking_item_count": len(set(unresolved_blockers)),
    }


def add_visual_evidence(
    items: list[dict[str, Any]],
    *,
    source_video_path: Path,
    automatic_preview_video_path: Path | None,
    reviewed_final_video_path: Path | None,
    privacy_input: rendering.PrivacyRegionInput,
    output_dir: Path,
    bundle_paths: dict[str, Path],
    fps: float,
    frame_count: int,
    config: ReviewBundleConfig,
    overwrite: bool,
) -> None:
    for item in items:
        item_id = item["review_item_id"]
        start = max(0, int(item["frame_start"]) - config.context_frames)
        end = min(frame_count - 1, int(item["frame_end"]) + config.context_frames)
        visual: dict[str, str] = {}
        sheet_path = bundle_paths["contact_sheets"] / f"{item_id}.jpg"
        create_contact_sheet_for_item(
            source_video_path,
            sheet_path,
            item,
            privacy_input,
            frame_count=frame_count,
            overwrite=overwrite,
            max_frames=config.contact_sheet_max_frames,
        )
        visual["contact_sheet"] = relative_to(sheet_path, output_dir)
        source_clip = bundle_paths["clips"] / f"{item_id}_source.mp4"
        trim_clip_by_frames(
            source_video_path,
            source_clip,
            start,
            end,
            fps=fps,
            overwrite=overwrite,
        )
        visual["source_clip"] = relative_to(source_clip, output_dir)
        if automatic_preview_video_path is not None:
            preview_clip = bundle_paths["clips"] / f"{item_id}_automatic_preview.mp4"
            trim_clip_by_frames(
                automatic_preview_video_path,
                preview_clip,
                start,
                end,
                fps=fps,
                overwrite=overwrite,
            )
            visual["automatic_preview_clip"] = relative_to(preview_clip, output_dir)
        if reviewed_final_video_path is not None:
            final_clip = bundle_paths["clips"] / f"{item_id}_reviewed_final.mp4"
            trim_clip_by_frames(
                reviewed_final_video_path,
                final_clip,
                start,
                end,
                fps=fps,
                overwrite=overwrite,
            )
            visual["reviewed_final_clip"] = relative_to(final_clip, output_dir)
        item["visual_evidence"] = visual


def create_contact_sheet_for_item(
    source_video_path: Path,
    output_path: Path,
    item: dict[str, Any],
    privacy_input: rendering.PrivacyRegionInput,
    *,
    frame_count: int,
    overwrite: bool,
    max_frames: int,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames_for_contact_sheet(item, frame_count=frame_count, max_frames=max_frames)
    capture = cv.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source_video_path}")
    tiles: list[np.ndarray] = []
    try:
        for frame_index in frames:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            tiles.append(
                resize_tile(
                    draw_review_overlay(frame, frame_index, item, privacy_input),
                    width=360,
                )
            )
    finally:
        capture.release()
    if not tiles:
        return
    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    cv.imwrite(str(output_path), sheet)


def draw_review_overlay(
    frame: np.ndarray,
    frame_index: int,
    item: dict[str, Any],
    privacy_input: rendering.PrivacyRegionInput,
) -> np.ndarray:
    output = frame.copy()
    color = (0, 0, 255) if item.get("blocking_status") else (0, 180, 255)
    if item.get("severity") == "random_audit":
        color = (255, 128, 0)
    for region in privacy_input.regions_by_frame.get(frame_index, []):
        if not region.bbox_xyxy:
            continue
        x1, y1, x2, y2 = region.bbox_xyxy
        region_color = (0, 180, 0) if region.is_renderable else (0, 0, 255)
        cv.rectangle(output, (x1, y1), (x2, y2), region_color, 2)
        cv.putText(
            output,
            region.provenance[:28],
            (x1, max(18, y1 - 6)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            region_color,
            2,
            cv.LINE_AA,
        )
    cv.rectangle(output, (10, 10), (520, 86), (255, 255, 255), -1)
    cv.rectangle(output, (10, 10), (520, 86), color, 2)
    cv.putText(
        output,
        f"frame {frame_index} | {item.get('severity')}",
        (22, 38),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv.LINE_AA,
    )
    cv.putText(
        output,
        ", ".join(item.get("reason_codes", []))[:52],
        (22, 68),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 40, 40),
        2,
        cv.LINE_AA,
    )
    return output


def trim_clip_by_frames(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    overwrite: bool,
) -> None:
    start_seconds = frame_to_seconds(start_frame, fps)
    duration_seconds = max(frame_to_seconds(end_frame - start_frame + 1, fps), 1 / fps)
    media.trim_video(
        input_path,
        output_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        overwrite=overwrite,
    )


def write_bundle_outputs(
    output_dir: Path,
    queue: dict[str, Any],
    summary: dict[str, Any],
    *,
    config: ReviewBundleConfig,
    source_info: Any,
    created_at: str,
) -> None:
    write_json(output_dir / "review_queue.json", queue)
    write_json(output_dir / "review_summary.json", summary)
    write_json(output_dir / "review_config.json", config_payload(config, created_at))
    write_json(output_dir / "review_queue_schema.json", review_queue_schema())
    write_json(
        output_dir / "decisions" / "analyst_decisions.template.json", decisions_template(queue)
    )
    write_text(output_dir / "assets" / "review.css", review_css())
    write_text(output_dir / "assets" / "review.js", review_js())
    write_text(output_dir / "index.html", review_html(queue, summary))
    write_text(output_dir / "INDEX.md", index_markdown(queue, summary))
    write_text(output_dir / "REVIEW_CHECKLIST.md", checklist_markdown())
    write_text(output_dir / "README.md", bundle_readme(source_info))


def review_queue_schema() -> dict[str, Any]:
    return {
        "schema_id": REVIEW_QUEUE_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "required_item_fields": [
            "review_item_id",
            "case_id",
            "source_video_hash",
            "object_category",
            "frame_start",
            "frame_end",
            "severity",
            "blocking_status",
            "source_stage",
            "reason_codes",
            "explanation",
            "review_status",
            "allowed_review_actions",
            "visual_evidence",
        ],
        "severity_values": sorted(VALID_SEVERITIES),
        "review_status_values": sorted(VALID_REVIEW_STATUSES),
    }


def decisions_template(queue: dict[str, Any]) -> dict[str, Any]:
    return {
        "decisions": [],
        "export_note": "Use Export Review Decisions in index.html, or fill this manually.",
        "review_queue_id": queue.get("case_id"),
        "review_queue_item_count": len(queue.get("items", [])),
        "schema_version": SCHEMA_VERSION,
    }


def review_html(queue: dict[str, Any], summary: dict[str, Any]) -> str:
    queue_json = json.dumps(queue, sort_keys=True)
    summary_json = json.dumps(summary, sort_keys=True)
    status_text = (
        "REVIEW REQUIRED - NOT FINAL"
        if summary["final_export_blocked"]
        else "No blocking items generated"
    )
    status_class = "status blocked" if summary["final_export_blocked"] else "status clear"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Review Bundle - {escape_html(str(queue.get("case_id")))}</title>
  <link rel="stylesheet" href="assets/review.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Face Review Bundle</h1>
      <p>{escape_html(str(queue.get("case_id")))}</p>
    </div>
    <div class="{status_class}" id="final-status">{status_text}</div>
  </header>
  <main>
    <section class="summary-grid">
      <div><strong>Total</strong><span>{summary["total_items"]}</span></div>
      <div><strong>Blocking</strong><span>{summary["blocking_items"]}</span></div>
      <div><strong>Random audit</strong><span>{summary["random_audit_items"]}</span></div>
      <div><strong>Progress</strong><span id="progress-count">0 reviewed</span></div>
    </section>
    <section class="notice">
      <strong id="notice-title"></strong>
      <p>The review queue highlights detected risk and deterministic audit samples. No queue can prove that every possible missed face was found.</p>
      <p>Export decisions before closing this page or moving to another computer.</p>
    </section>
    <section class="filters">
      <label>Severity <select id="severity-filter"><option value="all">All</option></select></label>
      <label>Status <select id="status-filter"><option value="all">All</option></select></label>
      <button id="export-decisions">Export Review Decisions</button>
      <button id="clear-local">Clear Local Choices</button>
    </section>
    <section id="items" class="item-list"></section>
  </main>
  <script id="review-data" type="application/json">{queue_json}</script>
  <script id="review-summary" type="application/json">{summary_json}</script>
  <script src="assets/review.js"></script>
</body>
</html>
"""


def review_css() -> str:
    return """
:root { color-scheme: light; font-family: Inter, system-ui, -apple-system, sans-serif; }
body { margin: 0; background: #f6f7f9; color: #1f2933; }
.topbar { display: flex; justify-content: space-between; gap: 20px; padding: 22px 28px; background: #132238; color: white; }
h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
.topbar p { margin: 4px 0 0; color: #cfd8e3; }
.status { align-self: center; padding: 10px 14px; border-radius: 6px; font-weight: 700; background: #37506f; }
.status.blocked { background: #9f1d1d; }
.status.clear { background: #17633a; }
main { max-width: 1180px; margin: 0 auto; padding: 24px; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.summary-grid div, .notice, .filters, .card { background: white; border: 1px solid #d8dee8; border-radius: 8px; }
.summary-grid div { padding: 14px; }
.summary-grid strong { display: block; font-size: 13px; color: #52616f; }
.summary-grid span { font-size: 26px; font-weight: 700; }
.notice { margin-top: 16px; padding: 16px; }
.notice p { margin: 8px 0 0; color: #52616f; }
.filters { margin: 16px 0; padding: 12px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
select, button { min-height: 36px; border: 1px solid #b8c2cc; border-radius: 6px; background: white; padding: 0 10px; }
button { cursor: pointer; background: #1f4e79; color: white; border-color: #1f4e79; }
button.secondary { background: white; color: #1f4e79; }
.item-list { display: grid; gap: 14px; }
.card { padding: 16px; border-left: 6px solid #7b8794; }
.card.blocking { border-left-color: #c62828; }
.card.high { border-left-color: #ef6c00; }
.card.medium { border-left-color: #f9a825; }
.card.random_audit { border-left-color: #0277bd; }
.card-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.card h2 { margin: 0; font-size: 18px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.badge { background: #edf2f7; border-radius: 999px; padding: 4px 9px; font-size: 12px; }
.badge.blocking { background: #fdecea; color: #8a1c1c; }
.evidence { display: grid; grid-template-columns: minmax(240px, 360px) 1fr; gap: 14px; margin-top: 12px; }
.evidence img { width: 100%; border: 1px solid #d8dee8; border-radius: 6px; }
.links { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.links a { color: #1f4e79; font-weight: 600; }
.decision { margin-top: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
details { margin-top: 10px; }
pre { overflow: auto; background: #f1f5f9; padding: 10px; border-radius: 6px; }
@media (max-width: 760px) { .evidence { grid-template-columns: 1fr; } .topbar { display: block; } }
"""


def review_js() -> str:
    return """
const queue = JSON.parse(document.getElementById('review-data').textContent);
const summary = JSON.parse(document.getElementById('review-summary').textContent);
const storageKey = `review-decisions:${queue.case_id}:${queue.source_video.hash}`;
let decisions = JSON.parse(localStorage.getItem(storageKey) || '{}');

function save() { localStorage.setItem(storageKey, JSON.stringify(decisions)); render(); }
function statusFor(id) { return decisions[id]?.review_status || 'pending'; }
function setDecision(id, status) {
  decisions[id] = { review_item_id: id, review_status: status, decided_at: new Date().toISOString() };
  save();
}
function populateFilters() {
  const severity = document.getElementById('severity-filter');
  const status = document.getElementById('status-filter');
  [...new Set(queue.items.map(i => i.severity))].forEach(value => severity.add(new Option(value, value)));
  ['pending','accepted_current','correction_required','corrected','false_positive','intentionally_unredacted','deferred','escalated'].forEach(value => status.add(new Option(value, value)));
  severity.onchange = render;
  status.onchange = render;
}
function render() {
  const severityFilter = document.getElementById('severity-filter').value;
  const statusFilter = document.getElementById('status-filter').value;
  const reviewed = queue.items.filter(item => statusFor(item.review_item_id) !== 'pending').length;
  document.getElementById('progress-count').textContent = `${reviewed}/${queue.items.length} reviewed`;
  const finalStatus = document.getElementById('final-status');
  finalStatus.className = summary.final_export_blocked ? 'status blocked' : 'status clear';
  finalStatus.textContent = summary.final_export_blocked ? 'REVIEW REQUIRED - NOT FINAL' : 'No blocking items generated';
  document.getElementById('notice-title').textContent = finalStatus.textContent;
  const container = document.getElementById('items');
  container.innerHTML = '';
  queue.items
    .filter(item => severityFilter === 'all' || item.severity === severityFilter)
    .filter(item => statusFilter === 'all' || statusFor(item.review_item_id) === statusFilter)
    .forEach(item => container.appendChild(card(item)));
}
function card(item) {
  const node = document.createElement('article');
  node.className = `card ${item.severity}`;
  if (item.blocking_status) node.classList.add('blocking');
  const evidence = item.visual_evidence || {};
  node.innerHTML = `
    <div class="card-head">
      <div>
        <h2>${item.frame_start}-${item.frame_end}: ${item.source_stage}</h2>
        <div class="badges">
          <span class="badge ${item.blocking_status ? 'blocking' : ''}">${item.severity}</span>
          <span class="badge">${item.item_origin}</span>
          ${item.reason_codes.map(reason => `<span class="badge">${reason}</span>`).join('')}
        </div>
      </div>
      <strong>${statusFor(item.review_item_id)}</strong>
    </div>
    <p>${item.explanation}</p>
    <div class="evidence">
      <div>${evidence.contact_sheet ? `<img src="${evidence.contact_sheet}" alt="Contact sheet for ${item.review_item_id}">` : '<p>No contact sheet generated.</p>'}</div>
      <div>
        <p><strong>Frames:</strong> ${item.frame_start}-${item.frame_end}</p>
        <p><strong>Unresolved frames:</strong> ${item.unresolved_frame_count}</p>
        <p><strong>Candidate region:</strong> ${item.available_candidate_region_status}</p>
        <div class="links">
          ${evidence.source_clip ? `<a href="${evidence.source_clip}">Source clip</a>` : ''}
          ${evidence.automatic_preview_clip ? `<a href="${evidence.automatic_preview_clip}">Automatic preview</a>` : ''}
          ${evidence.reviewed_final_clip ? `<a href="${evidence.reviewed_final_clip}">Reviewed final</a>` : ''}
        </div>
        <div class="decision">
          <label>Decision
            <select data-item="${item.review_item_id}">
              <option value="pending">pending</option>
              ${item.allowed_review_actions.map(action => `<option value="${action}">${action}</option>`).join('')}
            </select>
          </label>
        </div>
        <details><summary>Technical details</summary><pre>${JSON.stringify(item, null, 2)}</pre></details>
      </div>
    </div>`;
  const select = node.querySelector('select');
  select.value = statusFor(item.review_item_id);
  select.onchange = event => setDecision(item.review_item_id, event.target.value);
  return node;
}
function exportDecisions() {
  const payload = {
    schema_version: queue.schema_version,
    review_queue_id: queue.case_id,
    source_video_hash: queue.source_video.hash,
    exported_at: new Date().toISOString(),
    decisions: Object.values(decisions).sort((a, b) => a.review_item_id.localeCompare(b.review_item_id)),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `review_decisions_${queue.case_id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}
document.getElementById('export-decisions').onclick = exportDecisions;
document.getElementById('clear-local').onclick = () => { decisions = {}; localStorage.removeItem(storageKey); render(); };
populateFilters();
render();
"""


def index_markdown(queue: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Review Bundle Index",
        "",
        f"- Case: `{queue.get('case_id')}`",
        f"- Total review items: `{summary['total_items']}`",
        f"- Blocking items: `{summary['blocking_items']}`",
        f"- Final export blocked by review queue: `{summary['final_export_blocked']}`",
        "",
        "Open `index.html` locally to review clips, contact sheets, and decisions.",
        "",
        "Absence of review items does not prove that every possible missed face was found.",
    ]
    return "\n".join(lines) + "\n"


def checklist_markdown() -> str:
    return """# Review Checklist

1. Open `index.html`.
2. Review every blocking item first.
3. Inspect source and preview/final clips, not just the contact sheet.
4. Mark each item with the appropriate decision.
5. Use `Correction required` when valid privacy regions are missing.
6. Export decisions before closing the browser or moving to another computer.
7. Validate exported decisions with `redactor validate-review-decisions`.

This bundle does not implement box drawing or keyframe correction.
"""


def bundle_readme(source_info: Any) -> str:
    return f"""# Automated Face Review Bundle

This directory is a local, self-contained review package. It contains no remote assets,
analytics, or network calls.

Source video hash: `{source_info.sha256}`

Open `index.html` in a browser. Browser-local choices are temporary until exported.
"""


def config_payload(config: ReviewBundleConfig, created_at: str) -> dict[str, Any]:
    return {
        "case_id": config.case_id,
        "clip_context_seconds": config.clip_context_seconds,
        "contact_sheet_max_frames": config.contact_sheet_max_frames,
        "context_frames": config.context_frames,
        "created_at": created_at,
        "generate_visuals": config.generate_visuals,
        "merge_gap_frames": config.merge_gap_frames,
        "object_category": config.object_category,
        "random_sample_count": config.random_sample_count,
        "random_sample_frame_count": config.random_sample_frame_count,
        "random_seed": config.random_seed,
        "schema_version": SCHEMA_VERSION,
    }


def create_bundle_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "assets": output_dir / "assets",
        "clips": output_dir / "clips",
        "contact_sheets": output_dir / "contact_sheets",
        "decisions": output_dir / "decisions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_evaluation_payload(evaluation_dir: Path | None) -> dict[str, Any]:
    if evaluation_dir is None:
        return {}
    payload: dict[str, Any] = {}
    for name in (
        "failures",
        "metrics",
        "object_metrics",
        "interval_decisions",
        "render_validation",
    ):
        path = evaluation_dir / f"{name}.json"
        if path.exists():
            payload[name] = load_json(path)
    return payload


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def stable_reason_codes(raw_reasons: Any, mapping: dict[str, str]) -> set[str]:
    codes: set[str] = set()
    for reason in raw_reasons or []:
        reason_text = str(reason)
        codes.add(mapping.get(reason_text, normalize_reason_code(reason_text)))
    return codes


def normalize_reason_code(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def interval_explanation(status: str, strategy: str, reasons: set[str]) -> str:
    if "preserved_reviewed_region" in reasons:
        return (
            "This interval uses previously reviewed and preserved privacy regions. "
            "It should be spot-checked, but it is not automatic detector evidence."
        )
    if status == "needs_review":
        return (
            "Automatic privacy recovery could not confidently clear this interval. "
            "A reviewer should inspect it before final export."
        )
    if status == "rejected":
        return "Automatic privacy recovery rejected this interval."
    if "one_sided_recovery" in reasons:
        return "The interval used one-sided recovery, which carries higher uncertainty."
    return f"Privacy recovery produced a {status} decision using {strategy}."


def evaluation_explanation(reason_code: str, item: dict[str, Any]) -> str:
    if reason_code == "ground_truth_detector_miss":
        return (
            "Ground Truth shows a detector-scored face frame where ADAS did not provide a "
            "valid raw detector hit. This is detector evidence only."
        )
    if reason_code == "ground_truth_privacy_miss":
        return "Ground Truth shows a privacy-required frame that automatic privacy coverage missed."
    if reason_code == "possible_keep_person_overlap":
        return "Evaluation found possible redaction overlap with a keep-person target."
    return f"Evaluation reported {item.get('failure_reason')}."


def compact_draft_metrics(draft: ReviewDraft) -> dict[str, Any]:
    return {
        "frame_end": draft.frame_end,
        "frame_start": draft.frame_start,
        "item_origin": draft.item_origin,
        "reason_codes": sorted(draft.reason_codes),
        "severity": draft.severity,
        "source_stage": draft.source_stage,
    }


def combine_explanations(left: str, right: str) -> str:
    if left == right:
        return left
    return f"{left} Additional related evidence: {right}"


def combine_candidate_status(left: str, right: str) -> str:
    if left == right:
        return left
    if "invalid" in {left, right}:
        return "invalid"
    if "unresolved" in {left, right}:
        return "unresolved"
    if "available" in {left, right}:
        return "available"
    return "mixed"


def draft_sort_key(draft: ReviewDraft) -> tuple[int, int, int, str]:
    privacy_priority = 0 if "privacy" in draft.source_stage else 1
    return (SEVERITY_ORDER[draft.severity], privacy_priority, draft.frame_start, draft.source_stage)


def stable_item_id(case_id: str, draft: ReviewDraft, index: int) -> str:
    reason = "_".join(sorted(draft.reason_codes))[:48]
    raw = f"{case_id}:{draft.source_stage}:{draft.frame_start}:{draft.frame_end}:{reason}:{index}"
    slug = normalize_reason_code(raw)
    return f"review_{slug[:96]}"


def frames_for_contact_sheet(
    item: dict[str, Any],
    *,
    frame_count: int,
    max_frames: int,
) -> list[int]:
    references = [int(value) for value in item.get("frame_references", [])]
    start = int(item["frame_start"])
    end = int(item["frame_end"])
    if not references:
        references = [start, (start + end) // 2, end]
    references.extend([start, end])
    selected = sorted({max(0, min(frame_count - 1, frame)) for frame in references})
    if len(selected) <= max_frames:
        return selected
    step = max(1, math.ceil(len(selected) / max_frames))
    return selected[::step][:max_frames]


def resize_tile(frame: np.ndarray, *, width: int) -> np.ndarray:
    scale = width / frame.shape[1]
    return cv.resize(frame, (width, round(frame.shape[0] * scale)), interpolation=cv.INTER_AREA)


def contiguous_ranges(frames: list[int] | set[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append((start, previous))
        start = previous = frame
    ranges.append((start, previous))
    return ranges


def bounds_or_zero(frames: list[int]) -> tuple[int, int]:
    if not frames:
        return (0, 0)
    return min(frames), max(frames)


def ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def valid_box(box: tuple[int, ...]) -> bool:
    return len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def frame_to_seconds(frame: int, fps: float) -> float:
    return round(frame / fps, 6) if fps > 0 else 0.0


def as_int(value: Any, default: int | None = None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0 if default is None else default


def as_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def relative_to(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def assert_no_external_assets(bundle_dir: Path) -> list[str]:
    findings: list[str] = []
    pattern = re.compile(r"https?://|//[A-Za-z0-9.-]+|fetch\(|XMLHttpRequest|navigator\.sendBeacon")
    for path in [
        bundle_dir / "index.html",
        bundle_dir / "assets" / "review.css",
        bundle_dir / "assets" / "review.js",
    ]:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                findings.append(
                    f"{path.relative_to(bundle_dir)}:{line_no}: external/network reference"
                )
    return findings


def assert_relative_bundle_paths(bundle_dir: Path) -> list[str]:
    findings: list[str] = []
    for path in bundle_dir.rglob("*.json"):
        data = path.read_text(encoding="utf-8")
        if re.search(r'"/(?:Users|private|var|tmp)/', data):
            findings.append(f"{path.relative_to(bundle_dir)} contains an absolute local path")
    queue = load_json(bundle_dir / "review_queue.json")
    for item in queue.get("items", []):
        for value in (item.get("visual_evidence") or {}).values():
            if Path(str(value)).is_absolute():
                findings.append(f"{item['review_item_id']} has absolute visual path")
    return findings
