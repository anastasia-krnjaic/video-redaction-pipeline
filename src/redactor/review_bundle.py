# ruff: noqa: E501

from __future__ import annotations

import hashlib
import http.server
import json
import math
import mimetypes
import posixpath
import random
import re
import shutil
import socketserver
import subprocess
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import media, rendering
from redactor.detection import rational_to_float

SCHEMA_VERSION = "1.0"
REVIEW_TASK_SCHEMA_VERSION = "1.2.3"
REVIEW_QUEUE_SCHEMA_ID = "face_review_queue_schema_v1.0"
REVIEW_TASK_SCHEMA_ID = "face_reviewer_task_schema_v1.2.3"
REVIEW_BUNDLE_ID = "automated_review_bundle_v1_2_3"
SUPPORTED_REVIEW_TASK_SCHEMA_VERSIONS = {"1.1", "1.2", "1.2.1", "1.2.2", "1.2.3"}
REVIEW_CLIP_EXPECTED_RENDER_MAX_MEAN_ABSDIFF = 4.0
REVIEW_CLIP_EXPECTED_RENDER_MAX_CHANGED_PIXEL_FRACTION = 0.02
REVIEW_CLIP_EXPECTED_RENDER_MAX_CHANGED_PIXEL_FRACTION_RATIO = 0.2

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
TASK_CATEGORY_ORDER = {
    "action_required": 0,
    "corrected_result_check": 1,
    "automatic_continuity_check": 2,
    "optional_spot_check": 3,
    "technical_only": 4,
}
USER_ACTIONS = {
    "looks_correct": {
        "label": "Correct",
        "review_status": "accepted_current",
    },
    "needs_correction": {
        "label": "Needs correction",
        "review_status": "correction_required",
    },
    "not_privacy_target": {
        "label": "Not sensitive",
        "review_status": "intentionally_unredacted",
    },
    "escalate": {
        "label": "Escalate",
        "review_status": "escalated",
    },
    "no_issue_found": {
        "label": "Correct",
        "review_status": "accepted_current",
    },
    "needs_review": {
        "label": "Needs correction",
        "review_status": "escalated",
    },
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
    "reviewer_found_privacy_miss",
    "silent_miss_found_by_random_audit",
}
CORRECTION_REQUIRED_REASON_CODES = {
    "rejected_interval",
    "unresolved_privacy_frames",
    "ground_truth_privacy_miss",
    "invalid_box",
    "reviewer_found_privacy_miss",
    "silent_miss_found_by_random_audit",
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
class TaskFrameMapping:
    source_frame_start: int
    source_frame_end: int

    @property
    def frame_count(self) -> int:
        return self.source_frame_end - self.source_frame_start + 1

    @property
    def task_clip_frame_start(self) -> int:
        return 0

    @property
    def task_clip_frame_end(self) -> int:
        return self.frame_count - 1

    def local_for_source(self, source_frame_index: int) -> int:
        return source_frame_index - self.source_frame_start

    def source_for_local(self, task_clip_frame_index: int) -> int:
        return self.source_frame_start + task_clip_frame_index

    def payload(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "mapping": [
                {
                    "source_frame_index": source_frame,
                    "task_clip_frame_index": self.local_for_source(source_frame),
                }
                for source_frame in range(self.source_frame_start, self.source_frame_end + 1)
            ],
            "source_frame_end": self.source_frame_end,
            "source_frame_start": self.source_frame_start,
            "task_clip_frame_end": self.task_clip_frame_end,
            "task_clip_frame_start": self.task_clip_frame_start,
        }


@dataclass(frozen=True)
class ReviewRenderConfig:
    method: str = rendering.METHOD_PIXELATE
    margin_pixels: int = 0
    margin_fraction: float = 0.15
    blur_kernel: int = 51
    pixel_size: int = 16

    def validate(self) -> None:
        self.render_settings().validate()

    def render_settings(self) -> rendering.RenderSettings:
        return rendering.RenderSettings(
            mode=rendering.MODE_FINAL,
            method=self.method,
            margin_pixels=self.margin_pixels,
            margin_fraction=self.margin_fraction,
            blur_kernel=self.blur_kernel,
            pixel_size=self.pixel_size,
            preview_watermark=False,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "blur_kernel": self.render_settings().normalized_blur_kernel,
            "margin_fraction": self.margin_fraction,
            "margin_pixels": self.margin_pixels,
            "method": self.method,
            "pixel_size": self.pixel_size,
        }

    def stable_hash(self) -> str:
        return stable_payload_hash(self.payload())


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
    one_frame_miss_context_frames: int = 12
    review_render: ReviewRenderConfig = field(default_factory=ReviewRenderConfig)


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
    corrected_privacy_regions_path: Path | None = None,
    corrected_interval_decisions_path: Path | None = None,
    corrected_render_manifest_path: Path | None = None,
    corrected_render_validation_path: Path | None = None,
    corrected_evaluation_dir: Path | None = None,
    ground_truth_path: Path | None = None,
    reviewer_findings_path: Path | None = None,
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
    corrected_privacy_input = (
        rendering.load_privacy_regions(corrected_privacy_regions_path)
        if corrected_privacy_regions_path is not None
        else None
    )
    intervals = rendering.load_interval_decisions(interval_decisions_path)
    evaluation_payload = load_evaluation_payload(evaluation_dir)
    render_manifest = load_json(render_manifest_path)
    render_validation = load_json(render_validation_path)
    corrected_intervals = rendering.load_interval_decisions(corrected_interval_decisions_path)
    corrected_render_manifest = load_json(corrected_render_manifest_path)
    corrected_render_validation = load_json(corrected_render_validation_path)
    corrected_evaluation_payload = load_evaluation_payload(corrected_evaluation_dir)
    track_quality = load_json(track_quality_path)
    track_review_flags = load_json(track_review_flags_path)
    track_review_windows = load_json(track_review_windows_path)
    ground_truth = load_json(ground_truth_path)
    reviewer_findings = load_json(reviewer_findings_path)

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
        reviewer_findings=reviewer_findings,
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
            privacy_input=privacy_input,
            privacy_regions_path=privacy_regions_path,
            corrected_privacy_input=corrected_privacy_input,
            corrected_privacy_regions_path=corrected_privacy_regions_path,
            interval_decisions_path=interval_decisions_path,
            corrected_interval_decisions_path=corrected_interval_decisions_path,
            output_dir=output_dir,
            bundle_paths=bundle_paths,
            fps=fps,
            frame_count=frame_count,
            source_width=source_info.video_stream.width,
            source_height=source_info.video_stream.height,
            source_has_audio=source_info.audio_stream is not None,
            config=config,
            overwrite=overwrite,
        )
    assign_media_roles(items, output_dir)

    queue = queue_payload(
        items,
        case_id=case_id,
        source_info=source_info,
        config=config,
        created_at=created_at,
        corrected_context=corrected_context_payload(
            corrected_intervals=corrected_intervals,
            corrected_render_manifest=corrected_render_manifest,
            corrected_render_validation=corrected_render_validation,
            corrected_evaluation_payload=corrected_evaluation_payload,
        ),
        paths={
            "automatic_preview_video": automatic_preview_video_path,
            "corrected_evaluation_dir": corrected_evaluation_dir,
            "corrected_interval_decisions": corrected_interval_decisions_path,
            "corrected_privacy_regions": corrected_privacy_regions_path,
            "corrected_render_manifest": corrected_render_manifest_path,
            "corrected_render_validation": corrected_render_validation_path,
            "detections": detections_path,
            "evaluation_dir": evaluation_dir,
            "ground_truth": ground_truth_path,
            "interval_decisions": interval_decisions_path,
            "privacy_regions": privacy_regions_path,
            "render_manifest": render_manifest_path,
            "render_validation": render_validation_path,
            "reviewed_final_video": reviewed_final_video_path,
            "reviewer_findings": reviewer_findings_path,
            "source_video": source_video_path,
            "track_quality": track_quality_path,
            "track_review_flags": track_review_flags_path,
            "track_review_windows": track_review_windows_path,
            "tracks": tracks_path,
        },
    )
    tasks_payload = reviewer_tasks_payload(queue, config=config, fps=fps)
    queue["reviewer_tasks"] = tasks_payload["reviewer_tasks"]
    queue["technical_audit"] = tasks_payload["technical_audit"]
    queue["presentation_schema_id"] = REVIEW_TASK_SCHEMA_ID
    queue["presentation_schema_version"] = REVIEW_TASK_SCHEMA_VERSION
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
    reviewer_findings: dict[str, Any],
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
    merged.extend(reviewer_finding_drafts(reviewer_findings))
    return sorted(merged, key=draft_sort_key)


def reviewer_finding_drafts(payload: dict[str, Any]) -> list[ReviewDraft]:
    drafts: list[ReviewDraft] = []
    findings = payload.get("findings") if isinstance(payload, dict) else []
    for finding in findings or []:
        if not isinstance(finding, dict):
            continue
        frame_start = as_int(finding.get("source_frame_start"), 0)
        frame_end = as_int(finding.get("source_frame_end"), frame_start)
        reason_codes = {normalize_reason_code(reason) for reason in finding.get("reason_codes", [])}
        reason_codes.update({"reviewer_found_privacy_miss", "silent_miss_found_by_random_audit"})
        source_task_id = none_if_empty(finding.get("source_reviewer_task_id"))
        source_item_id = none_if_empty(finding.get("source_review_item_id"))
        drafts.append(
            ReviewDraft(
                item_origin="human_review",
                source_stage="reviewer_finding",
                frame_start=frame_start,
                frame_end=frame_end,
                reason_codes=reason_codes,
                severity="blocking",
                explanation=(
                    "A reviewer found a possible missed privacy target in a deterministic "
                    "random-audit sample. Final export remains blocked until this interval is "
                    "corrected and confirmed."
                ),
                object_id=none_if_empty(finding.get("affected_object_id")),
                track_ids={as_int(finding.get("target_track_id"), -1)}
                if finding.get("target_track_id") is not None
                else set(),
                metrics={
                    "affected_frames": finding.get("affected_frames", []),
                    "correction_status": finding.get("correction_status"),
                    "displayed_clip_frame_end": finding.get("displayed_clip_frame_end"),
                    "displayed_clip_frame_start": finding.get("displayed_clip_frame_start"),
                    "displayed_uncovered_frames_with_track": finding.get(
                        "displayed_uncovered_frames_with_track",
                        [],
                    ),
                    "event_uncovered_frames_with_track": finding.get(
                        "event_uncovered_frames_with_track",
                        [],
                    ),
                    "expected_privacy_policy": finding.get("expected_privacy_policy"),
                    "finding_id": finding.get("finding_id"),
                    "reviewer_finding_type": finding.get("reviewer_finding_type"),
                    "source_review_item_id": source_item_id,
                    "source_reviewer_task_id": source_task_id,
                },
                unresolved_frame_count=len(finding.get("affected_frames") or []),
                candidate_region_status=(
                    "available"
                    if str(finding.get("correction_status") or "").endswith("_available")
                    else "unresolved"
                ),
                frame_references={
                    int(frame) for frame in finding.get("affected_frames", []) if frame is not None
                }
                or set(range(frame_start, frame_end + 1)),
                provenance={
                    "before": "random_audit",
                    "during": "reviewer_found_privacy_miss",
                    "after": finding.get("correction_status"),
                    "raw": finding,
                },
            )
        )
    return drafts


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
            *merged_event_metrics(left),
            *merged_event_metrics(right),
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
    corrected_context: dict[str, Any],
    paths: dict[str, Path | None],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "corrected_context": corrected_context,
        "created_at": created_at,
        "generator": REVIEW_BUNDLE_ID,
        "input_hashes": {
            key: media.sha256_file(path) for key, path in paths.items() if path and path.is_file()
        },
        "input_paths": {
            key: safe_input_path(path) if path and (path.is_file() or path.is_dir()) else None
            for key, path in paths.items()
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


def corrected_context_payload(
    *,
    corrected_intervals: list[dict[str, object]],
    corrected_render_manifest: dict[str, Any],
    corrected_render_validation: dict[str, Any],
    corrected_evaluation_payload: dict[str, Any],
) -> dict[str, Any]:
    privacy_summary = (
        corrected_render_manifest.get("privacy_region_summary")
        or corrected_render_manifest.get("privacy")
        or {}
    )
    if not privacy_summary:
        privacy_summary = (corrected_evaluation_payload.get("metrics") or {}).get("privacy") or {}
    return {
        "corrected_interval_count": len(corrected_intervals),
        "corrected_intervals": [dict(interval) for interval in corrected_intervals],
        "evaluation_status": (
            (corrected_evaluation_payload.get("metrics") or {}).get("acceptance") or {}
        ).get("final_status"),
        "final_approval_status": corrected_render_manifest.get("final_approval_status"),
        "media_validation_passed": (
            (corrected_render_manifest.get("media_validation") or {}).get("passed")
            if corrected_render_manifest
            else (corrected_render_validation.get("media_validation") or {}).get("passed")
        ),
        "privacy_region_summary": privacy_summary,
        "provenance_counts": privacy_summary.get("provenance_counts", {}),
    }


def assign_media_roles(items: list[dict[str, Any]], output_dir: Path) -> None:
    role_map = {
        "source_clip": ("original", "Original"),
        "automatic_preview_clip": ("automatic", "Automatic"),
        "reviewed_final_clip": ("corrected_final", "Corrected final"),
    }
    for item in items:
        visual = item.get("visual_evidence") or {}
        role_audits = (item.get("media_comparison_audit") or {}).get("roles", {})
        roles: list[dict[str, Any]] = []
        seen_hashes: dict[str, str] = {}
        for key, (role, label) in role_map.items():
            rel_path = visual.get(key)
            if not rel_path:
                continue
            full_path = output_dir / rel_path
            file_hash = media.sha256_file(full_path) if full_path.exists() else None
            duplicate_of = seen_hashes.get(file_hash or "")
            display = duplicate_of is None
            if file_hash and duplicate_of is None:
                seen_hashes[file_hash] = role
            roles.append(
                {
                    "display": display,
                    "duplicate_of": duplicate_of,
                    "label": label,
                    "provenance_statement": role_provenance_statement(role, item),
                    "path": rel_path,
                    "role": role,
                    "sha256": file_hash,
                    "subtitle": role_subtitle(role),
                    "source_visual_key": key,
                    "warning": role_warning(role, item),
                    "media_audit": role_audits.get(role, {}),
                }
            )
        if visual.get("contact_sheet"):
            roles.append(
                {
                    "display": True,
                    "duplicate_of": None,
                    "label": "Contact sheet",
                    "path": visual["contact_sheet"],
                    "role": "diagnostic",
                    "sha256": media.sha256_file(output_dir / visual["contact_sheet"])
                    if (output_dir / visual["contact_sheet"]).exists()
                    else None,
                    "source_visual_key": "contact_sheet",
                }
            )
        item["media_roles"] = roles
        item["media_role_validation"] = {
            "duplicate_roles": [
                {
                    "duplicate_of": role["duplicate_of"],
                    "role": role["role"],
                    "sha256": role["sha256"],
                }
                for role in roles
                if role.get("duplicate_of")
            ],
            "distinct_display_roles": [
                role["role"]
                for role in roles
                if role.get("display") and role.get("role") != "diagnostic"
            ],
        }


def reviewer_tasks_payload(
    queue: dict[str, Any],
    *,
    config: ReviewBundleConfig,
    fps: float,
) -> dict[str, Any]:
    items = queue.get("items", [])
    corrected_context = queue.get("corrected_context") or {}
    corrected_intervals = [
        (as_int(interval.get("frame_start"), 0), as_int(interval.get("frame_end"), 0))
        for interval in corrected_context.get("corrected_intervals", [])
    ]
    has_corrected_final = any(
        any(role.get("role") == "corrected_final" for role in item.get("media_roles", []))
        for item in items
    )
    raw_categories = {
        item["review_item_id"]: classify_review_item(
            item,
            has_corrected_final=has_corrected_final,
            corrected_intervals=corrected_intervals,
        )
        for item in items
    }
    promoted_random_audit_intervals = [
        (int(item["frame_start"]), int(item["frame_end"]))
        for item in items
        if set(item.get("reason_codes", []))
        & {"reviewer_found_privacy_miss", "silent_miss_found_by_random_audit"}
    ]
    for item in items:
        if "random_audit_sample" not in set(item.get("reason_codes", [])):
            continue
        if any(
            ranges_overlap(
                int(item["frame_start"]),
                int(item["frame_end"]),
                finding_start,
                finding_end,
            )
            for finding_start, finding_end in promoted_random_audit_intervals
        ):
            raw_categories[item["review_item_id"]] = "technical_only"
    primary_items = [
        item for item in items if raw_categories.get(item["review_item_id"]) != "technical_only"
    ]
    technical_items = [
        item for item in items if raw_categories.get(item["review_item_id"]) == "technical_only"
    ]
    groups = group_reviewer_task_items(primary_items, raw_categories, config.merge_gap_frames)
    tasks = [
        reviewer_task_from_group(
            group,
            index=index,
            case_id=str(queue.get("case_id") or ""),
            fps=fps,
            raw_items=items,
            raw_categories=raw_categories,
            corrected_context=corrected_context,
        )
        for index, group in enumerate(groups, start=1)
    ]
    tasks = sorted(
        tasks,
        key=lambda task: (
            TASK_CATEGORY_ORDER[task["category"]],
            task["frame_start"],
            task["frame_end"],
            task["reviewer_task_id"],
        ),
    )
    for index, task in enumerate(tasks, start=1):
        task["task_number"] = index
        task["task_count"] = len(tasks)
    return {
        "reviewer_tasks": tasks,
        "technical_audit": technical_audit_payload(technical_items, raw_categories),
    }


def classify_review_item(
    item: dict[str, Any],
    *,
    has_corrected_final: bool,
    corrected_intervals: list[tuple[int, int]],
) -> str:
    reasons = set(item.get("reason_codes", []))
    if "random_audit_sample" in reasons:
        return "optional_spot_check"
    if reasons & {"preserved_reviewed_region", "preserved_existing_accepted_privacy_region"}:
        return "corrected_result_check"
    if item.get("blocking_status"):
        if has_corrected_final and (
            reasons
            & {
                "needs_review_interval",
                "ground_truth_privacy_miss",
                "reviewer_found_privacy_miss",
                "silent_miss_found_by_random_audit",
                "unresolved_privacy_frames",
            }
        ):
            return "corrected_result_check"
        return "action_required"
    if reasons == {"ground_truth_detector_miss"}:
        if corrected_intervals:
            return "technical_only"
        return "automatic_continuity_check"
    if reasons & {
        "suspicious_provenance_transition",
        "track_fragmentation",
        "render_validation_failure",
        "media_validation_failure",
    }:
        return "technical_only"
    return "technical_only"


def group_reviewer_task_items(
    items: list[dict[str, Any]],
    raw_categories: dict[str, str],
    merge_gap_frames: int,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda value: (value["frame_start"], value["frame_end"])):
        category = raw_categories[item["review_item_id"]]
        merge_index = next(
            (
                index
                for index in range(len(groups) - 1, -1, -1)
                if should_merge_task_group(
                    groups[index], item, category, raw_categories, merge_gap_frames
                )
            ),
            None,
        )
        if merge_index is None:
            groups.append([item])
        else:
            groups[merge_index].append(item)
    return groups


def should_merge_task_group(
    group: list[dict[str, Any]],
    item: dict[str, Any],
    category: str,
    raw_categories: dict[str, str],
    merge_gap_frames: int,
) -> bool:
    first = group[0]
    if raw_categories[first["review_item_id"]] != category:
        return False
    if first.get("object_id") and item.get("object_id") and first["object_id"] != item["object_id"]:
        return False
    start = min(int(value["frame_start"]) for value in group)
    end = max(int(value["frame_end"]) for value in group)
    reasons = {reason for value in group for reason in value.get("reason_codes", [])}
    item_reasons = set(item.get("reason_codes", []))
    interval_reasons = {
        "needs_review_interval",
        "preserved_existing_accepted_privacy_region",
        "preserved_reviewed_region",
        "rejected_interval",
    }
    if (reasons | item_reasons) & interval_reasons:
        return ranges_overlap(start, end, int(item["frame_start"]), int(item["frame_end"]))
    return end + merge_gap_frames >= int(item["frame_start"])


def reviewer_task_from_group(
    group: list[dict[str, Any]],
    *,
    index: int,
    case_id: str,
    fps: float,
    raw_items: list[dict[str, Any]],
    raw_categories: dict[str, str],
    corrected_context: dict[str, Any],
) -> dict[str, Any]:
    del raw_categories
    frame_start = min(int(item["frame_start"]) for item in group)
    frame_end = max(int(item["frame_end"]) for item in group)
    reason_codes = sorted({reason for item in group for reason in item.get("reason_codes", [])})
    category = classify_task_group(group)
    task_raw_ids = [item["review_item_id"] for item in group]
    related_raw = related_raw_items(raw_items, frame_start, frame_end)
    related_ids = sorted({item["review_item_id"] for item in related_raw})
    item_media = first_media_roles(group)
    task_id = stable_task_id(case_id, category, frame_start, frame_end, reason_codes, index)
    actions = task_actions(category, item_media)
    return {
        "allowed_user_actions": actions,
        "category": category,
        "case_id": case_id,
        "corrected_context": compact_corrected_context(corrected_context, frame_start, frame_end),
        "explanation": task_explanation(
            category, reason_codes, has_role(item_media, "corrected_final")
        ),
        "final_export_blocking": category
        in {
            "action_required",
            "automatic_continuity_check",
            "corrected_result_check",
        },
        "frame_end": frame_end,
        "frame_start": frame_start,
        "media_roles": item_media,
        "object_category": group[0].get("object_category"),
        "object_id": first_non_empty(item.get("object_id") for item in group),
        "primary_raw_review_item_ids": task_raw_ids,
        "raw_reason_codes": reason_codes,
        "raw_review_item_ids": related_ids,
        "raw_signal_count": len(related_ids),
        "requires_decision": category
        in {"action_required", "automatic_continuity_check", "corrected_result_check"},
        "review_status": "pending",
        "reviewer_task_id": task_id,
        "schema_version": REVIEW_TASK_SCHEMA_VERSION,
        "section": task_section(category),
        "technical_signal_count": sum(
            1 for item in related_raw if classify_review_item_for_display(item) == "technical_only"
        ),
        "timestamps": {
            "end_seconds": frame_to_seconds(frame_end, fps),
            "start_seconds": frame_to_seconds(frame_start, fps),
        },
        "title": task_title(category, reason_codes, has_role(item_media, "corrected_final")),
        "user_action_guidance": task_action_guidance(
            category, has_role(item_media, "corrected_final")
        ),
        "what_happened": task_what_happened(category, reason_codes),
        "what_to_check": task_checklist(category, has_role(item_media, "corrected_final")),
    }


def classify_task_group(group: list[dict[str, Any]]) -> str:
    categories = [
        classify_review_item(
            item,
            has_corrected_final=has_role(item.get("media_roles", []), "corrected_final"),
            corrected_intervals=[],
        )
        for item in group
    ]
    return min(categories, key=lambda value: TASK_CATEGORY_ORDER[value])


def classify_review_item_for_display(item: dict[str, Any]) -> str:
    return classify_review_item(item, has_corrected_final=False, corrected_intervals=[])


def related_raw_items(
    raw_items: list[dict[str, Any]],
    frame_start: int,
    frame_end: int,
) -> list[dict[str, Any]]:
    return [
        item
        for item in raw_items
        if ranges_overlap(frame_start, frame_end, int(item["frame_start"]), int(item["frame_end"]))
    ]


def first_media_roles(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in group:
        roles = item.get("media_roles") or []
        if roles:
            return roles
    return []


def has_role(roles: list[dict[str, Any]], role_name: str) -> bool:
    return any(role.get("role") == role_name and role.get("display", True) for role in roles)


def role_subtitle(role: str) -> str:
    if role == "original":
        return "Unredacted source context"
    if role == "automatic":
        return "Result before reviewed corrections"
    if role == "corrected_final":
        return "Same rendering style with reviewed corrections applied"
    return "Diagnostic evidence"


def role_provenance_statement(role: str, item: dict[str, Any]) -> str | None:
    if role == "corrected_final":
        if item.get("blocking_status") or set(item.get("reason_codes", [])) & {
            "ground_truth_privacy_miss",
            "needs_review_interval",
            "preserved_reviewed_region",
            "reviewer_found_privacy_miss",
            "silent_miss_found_by_random_audit",
            "unresolved_privacy_frames",
        }:
            return "Uses reviewed privacy regions for this interval."
        return "Uses the reviewed-final privacy-region set."
    if role == "automatic" and item.get("blocking_status"):
        return "Automatic result is shown to explain why review was required."
    return None


def role_warning(role: str, item: dict[str, Any]) -> str | None:
    if role == "automatic" and item.get("blocking_status"):
        return "Automatic result is not final."
    if role == "automatic" and set(item.get("reason_codes", [])) & {
        "reviewer_found_privacy_miss",
        "silent_miss_found_by_random_audit",
    }:
        return "Previous automatic result is shown to explain the reviewer-discovered miss."
    return None


def task_title(category: str, reasons: list[str], has_corrected_final: bool) -> str:
    if category == "corrected_result_check":
        return (
            "Check the corrected redaction"
            if has_corrected_final
            else "Correction evidence is missing"
        )
    if category == "action_required":
        return "Automatic redaction needs human confirmation"
    if category == "automatic_continuity_check":
        return "Check automatic continuity"
    if category == "optional_spot_check":
        return "Optional spot check"
    if "track_fragmentation" in reasons:
        return "Technical tracking note"
    return "Technical audit item"


def task_explanation(category: str, reasons: list[str], has_corrected_final: bool) -> str:
    reason_text = user_reason_summary(reasons)
    if category == "corrected_result_check":
        if has_corrected_final:
            if {"reviewer_found_privacy_miss", "silent_miss_found_by_random_audit"} & set(reasons):
                return (
                    "A possible missed privacy target was found. A corrected final clip is "
                    "available; confirm that the face is covered for the whole interval."
                )
            return f"{reason_text} A corrected final clip is available; confirm that the face is covered for the whole interval."
        return f"{reason_text} No corrected final clip is available, so this interval still needs corrected privacy regions."
    if category == "action_required":
        return f"{reason_text} Final export remains blocked until corrected privacy regions are supplied."
    if category == "automatic_continuity_check":
        return "The raw detector missed a face briefly, but downstream tracking or reviewed rendering appears to preserve continuity. Confirm the face remains covered."
    if category == "optional_spot_check":
        return "This interval was selected as a deterministic audit sample from an apparently normal part of the video."
    return reason_text


def task_what_happened(category: str, reasons: list[str]) -> str:
    if category == "optional_spot_check":
        return "The system did not flag this interval; it is included as a sanity check."
    if "reviewer_found_privacy_miss" in reasons or "silent_miss_found_by_random_audit" in reasons:
        return "A reviewer found a visible privacy target in a random-audit sample."
    if category == "automatic_continuity_check":
        return "The detector briefly missed the face, but later stages supplied coverage."
    if "needs_review_interval" in reasons:
        return "Automatic recovery could not safely approve this interval by itself."
    if "preserved_reviewed_region" in reasons:
        return "A previously reviewed correction is being used for this interval."
    return user_reason_summary(reasons)


def task_checklist(category: str, has_corrected_final: bool) -> list[str]:
    if category == "corrected_result_check":
        if has_corrected_final:
            return [
                "Compare Original, Automatic, and Corrected final clips.",
                "Confirm the privacy target is covered throughout the interval.",
                "Confirm the keep-person is not redacted by mistake.",
            ]
        return ["Confirm that no valid corrected region is available yet."]
    if category == "action_required":
        return [
            "Inspect the original context.",
            "Do not approve this interval unless corrected privacy regions are supplied later.",
        ]
    if category == "automatic_continuity_check":
        return [
            "Compare Original and Automatic evidence.",
            "Confirm that the redaction remains continuous through the one-frame detector miss.",
        ]
    if category == "optional_spot_check":
        return [
            "Play the sample.",
            "Confirm there is no obvious missed privacy target.",
        ]
    return ["Open technical details only if audit context is needed."]


def task_action_guidance(category: str, has_corrected_final: bool) -> str:
    if category == "corrected_result_check" and has_corrected_final:
        return "Correct confirms the corrected final result shown above. Needs correction keeps final export blocked until new privacy regions are supplied."
    if category == "corrected_result_check":
        return "This interval cannot be approved from the bundle because no corrected final evidence is available."
    if category == "action_required":
        return "Final export remains blocked until corrected privacy regions are supplied."
    if category == "automatic_continuity_check":
        return "Correct confirms that continuity was preserved despite the detector miss."
    if category == "optional_spot_check":
        return "Correct records a clean audit sample. Needs correction flags the sample for follow-up."
    return "Technical audit items do not require a normal review action."


def task_actions(category: str, media_roles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_corrected_final = has_role(media_roles, "corrected_final")
    if category == "corrected_result_check" and has_corrected_final:
        return [
            user_action("looks_correct", reviewed_media_role="corrected_final"),
            user_action("needs_correction", reviewed_media_role="corrected_final"),
            user_action("not_privacy_target", reviewed_media_role="original_context"),
        ]
    if category == "corrected_result_check":
        return [
            user_action("needs_correction", reviewed_media_role="original_context"),
            user_action("escalate", reviewed_media_role="original_context"),
        ]
    if category == "action_required":
        return [
            user_action("needs_correction", reviewed_media_role="original_context"),
            user_action("escalate", reviewed_media_role="original_context"),
        ]
    if category == "automatic_continuity_check":
        role = "corrected_final" if has_corrected_final else "automatic"
        return [
            user_action("looks_correct", reviewed_media_role=role),
            user_action("needs_correction", reviewed_media_role=role),
        ]
    if category == "optional_spot_check":
        return [
            user_action("no_issue_found", reviewed_media_role="original_context"),
            user_action("needs_review", reviewed_media_role="original_context"),
        ]
    return []


def user_action(action_id: str, *, reviewed_media_role: str) -> dict[str, Any]:
    action = USER_ACTIONS[action_id]
    return {
        "action": action_id,
        "label": action["label"],
        "review_status": action["review_status"],
        "reviewed_media_role": reviewed_media_role,
    }


def task_section(category: str) -> str:
    if category in {"action_required", "automatic_continuity_check"}:
        return "Needs your attention"
    if category == "corrected_result_check":
        return "Corrected results to confirm"
    if category == "optional_spot_check":
        return "Optional spot checks"
    return "Technical audit"


def user_reason_summary(reasons: list[str]) -> str:
    mapped = []
    if "excessive_scale_change" in reasons:
        mapped.append("the region changed size too abruptly")
    if "excessive_center_jump" in reasons:
        mapped.append("the region moved too abruptly")
    if "one_sided_recovery" in reasons:
        mapped.append("the interval relied on one-sided recovery")
    if "needs_review_interval" in reasons:
        mapped.append("automatic recovery could not approve the interval")
    if "unresolved_privacy_frames" in reasons or "ground_truth_privacy_miss" in reasons:
        mapped.append("privacy-required frames were not covered automatically")
    if "preserved_reviewed_region" in reasons:
        mapped.append("a reviewed correction is available")
    if "reviewer_found_privacy_miss" in reasons or "silent_miss_found_by_random_audit" in reasons:
        mapped.append("a reviewer found a possible missed privacy target")
    if "ground_truth_detector_miss" in reasons:
        mapped.append("the detector missed at least one annotated visible frame")
    if not mapped:
        mapped.append("the system recorded related audit evidence")
    return "The item was selected because " + "; ".join(dict.fromkeys(mapped)) + "."


def compact_corrected_context(
    corrected_context: dict[str, Any],
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    intervals = [
        interval
        for interval in corrected_context.get("corrected_intervals", [])
        if ranges_overlap(
            frame_start,
            frame_end,
            as_int(interval.get("frame_start"), 0),
            as_int(interval.get("frame_end"), 0),
        )
    ]
    return {
        "final_approval_status": corrected_context.get("final_approval_status"),
        "matching_corrected_intervals": intervals,
        "provenance_counts": corrected_context.get("provenance_counts", {}),
    }


def technical_audit_payload(
    technical_items: list[dict[str, Any]],
    raw_categories: dict[str, str],
) -> dict[str, Any]:
    return {
        "hidden_from_default_task_count": len(technical_items),
        "items": [
            {
                "category": raw_categories[item["review_item_id"]],
                "frame_end": item["frame_end"],
                "frame_start": item["frame_start"],
                "reason_codes": item.get("reason_codes", []),
                "review_item_id": item["review_item_id"],
                "severity": item["severity"],
                "source_stage": item["source_stage"],
            }
            for item in technical_items
        ],
    }


def first_non_empty(values) -> Any:
    for value in values:
        if value:
            return value
    return None


def stable_task_id(
    case_id: str,
    category: str,
    frame_start: int,
    frame_end: int,
    reasons: list[str],
    index: int,
) -> str:
    reason_slug = normalize_reason_code("_".join(reasons) or "review")
    raw = f"task_{case_id}_{category}_{frame_start}_{frame_end}_{reason_slug}_{index}"
    return normalize_reason_code(raw)[:96]


def initial_status_text(summary: dict[str, Any]) -> str:
    required = as_int(summary.get("required_review_tasks"), 0)
    corrected = as_int(summary.get("corrected_result_tasks"), 0)
    if required + corrected > 0:
        return f"{required + corrected} items need your review"
    if as_int(summary.get("optional_spot_checks"), 0) > 0:
        return "Optional spot checks available"
    return "No reviewer tasks generated"


def review_summary(queue: dict[str, Any]) -> dict[str, Any]:
    items = queue.get("items", [])
    tasks = queue.get("reviewer_tasks", [])
    counts_by_severity = count_values(item["severity"] for item in items)
    blocking_items = [item for item in items if item["blocking_status"]]
    random_items = [item for item in items if item["severity"] == "random_audit"]
    reason_counts = count_values(
        reason for item in items for reason in item.get("reason_codes", [])
    )
    if tasks:
        task_counts = count_values(task["category"] for task in tasks)
        required_task_categories = {
            "action_required",
            "automatic_continuity_check",
            "corrected_result_check",
        }
        required_tasks = [task for task in tasks if task["category"] in required_task_categories]
        return {
            "accepted_items": 0,
            "blocking_items": len(blocking_items),
            "blocker_reasons": sorted(
                {reason for item in blocking_items for reason in item.get("reason_codes", [])}
            ),
            "case_id": queue.get("case_id"),
            "completed_required_reviews": 0,
            "corrected_result_tasks": task_counts.get("corrected_result_check", 0),
            "correction_required_items": 0,
            "false_positive_items": 0,
            "final_export_blocked": bool(required_tasks),
            "optional_spot_checks": task_counts.get("optional_spot_check", 0),
            "pending_items": len(items),
            "random_audit_items": len(random_items),
            "reason_counts": reason_counts,
            "required_review_tasks": task_counts.get("action_required", 0)
            + task_counts.get("automatic_continuity_check", 0),
            "review_completeness_status": "pending" if required_tasks else "no_required_tasks",
            "reviewer_task_counts": task_counts,
            "reviewer_task_count": len(tasks),
            "schema_version": SCHEMA_VERSION,
            "severity_counts": counts_by_severity,
            "technical_only_items": (queue.get("technical_audit") or {}).get(
                "hidden_from_default_task_count", 0
            ),
            "total_items": len(items),
        }
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
        for role in item.get("media_roles", []) or []:
            role_path = role.get("path")
            if role_path and Path(str(role_path)).is_absolute():
                errors.append(f"{item_id}: media_roles contains absolute path")
        media_audit = item.get("media_comparison_audit") or {}
        if media_audit:
            validation = validate_media_comparison_audit(media_audit.get("roles", {}))
            for audit_error in validation["errors"]:
                errors.append(f"{item_id}: {audit_error}")
    tasks = queue.get("reviewer_tasks", [])
    if tasks is None:
        tasks = []
    if not isinstance(tasks, list):
        errors.append("reviewer_tasks must be a list")
        return errors
    seen_task_ids: set[str] = set()
    item_ids = seen_ids
    for task in tasks:
        task_id = str(task.get("reviewer_task_id") or "")
        if not task_id:
            errors.append("reviewer task missing reviewer_task_id")
        elif task_id in seen_task_ids:
            errors.append(f"duplicate reviewer_task_id: {task_id}")
        seen_task_ids.add(task_id)
        if task.get("schema_version") not in SUPPORTED_REVIEW_TASK_SCHEMA_VERSIONS:
            errors.append(f"{task_id}: unsupported reviewer task schema_version")
        if task.get("category") not in TASK_CATEGORY_ORDER:
            errors.append(f"{task_id}: invalid reviewer task category")
        if as_int(task.get("frame_start"), -1) > as_int(task.get("frame_end"), -1):
            errors.append(f"{task_id}: frame_start greater than frame_end")
        raw_ids = task.get("raw_review_item_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            errors.append(f"{task_id}: missing raw_review_item_ids")
        else:
            for raw_id in raw_ids:
                if raw_id not in item_ids:
                    errors.append(f"{task_id}: unknown raw_review_item_id: {raw_id}")
        actions = task.get("allowed_user_actions")
        if task.get("requires_decision") and not actions:
            errors.append(f"{task_id}: missing allowed_user_actions")
        for action in actions or []:
            if action.get("action") not in USER_ACTIONS:
                errors.append(f"{task_id}: invalid user action")
            if action.get("review_status") not in VALID_REVIEW_STATUSES:
                errors.append(f"{task_id}: invalid action review_status")
            if not action.get("reviewed_media_role"):
                errors.append(f"{task_id}: action missing reviewed_media_role")
        displayed_hashes: dict[str, str] = {}
        for role in task.get("media_roles", []) or []:
            role_path = role.get("path")
            if role_path and Path(str(role_path)).is_absolute():
                errors.append(f"{task_id}: media_roles contains absolute path")
            role_name = str(role.get("role") or "")
            role_hash = str(role.get("sha256") or "")
            if role.get("display") and role_name != "diagnostic" and role_hash:
                previous_role = displayed_hashes.get(role_hash)
                if previous_role and previous_role != role_name:
                    errors.append(
                        f"{task_id}: duplicate displayed media hash for {previous_role} and {role_name}"
                    )
                displayed_hashes[role_hash] = role_name
    return errors


def validate_review_decisions(
    queue_path: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    queue = load_json(queue_path)
    decisions = load_json(decisions_path)
    errors = validate_review_queue_payload(queue)
    schema_version = decisions.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, *SUPPORTED_REVIEW_TASK_SCHEMA_VERSIONS}:
        errors.append("unsupported decisions schema_version")
    items_by_id = {item["review_item_id"]: item for item in queue.get("items", [])}
    tasks_by_id = {task["reviewer_task_id"]: task for task in queue.get("reviewer_tasks", [])}
    seen: set[str] = set()
    decision_rows = decisions.get("decisions")
    if not isinstance(decision_rows, list):
        decision_rows = []
        errors.append("decisions must be a list")
    unresolved_blockers: list[str] = []
    correction_files_required: list[str] = []
    for decision in decision_rows:
        task_id = str(decision.get("reviewer_task_id") or "")
        item_id = str(decision.get("review_item_id") or "")
        decision_key = task_id or item_id
        if decision_key in seen:
            errors.append(f"duplicate decision for review target: {decision_key}")
            continue
        seen.add(decision_key)
        if task_id:
            task = tasks_by_id.get(task_id)
            if task is None:
                errors.append(f"unknown reviewer_task_id: {task_id}")
                continue
            status = str(decision.get("review_status") or "")
            if status not in VALID_REVIEW_STATUSES:
                errors.append(f"{task_id}: invalid review_status")
                continue
            allowed = task.get("allowed_user_actions", [])
            allowed_statuses = {action.get("review_status") for action in allowed}
            allowed_actions = {action.get("action") for action in allowed}
            selected_action = str(decision.get("selected_user_action") or "")
            if status not in allowed_statuses:
                errors.append(f"{task_id}: review_status not allowed for this reviewer task")
            if selected_action and selected_action not in allowed_actions:
                errors.append(f"{task_id}: selected_user_action not allowed for this reviewer task")
            if not decision.get("raw_review_item_ids"):
                errors.append(f"{task_id}: missing raw_review_item_ids")
            if not decision.get("reviewed_media_role"):
                errors.append(f"{task_id}: missing reviewed_media_role")
            continue
        if not item_id:
            errors.append("decision missing review_item_id or reviewer_task_id")
            continue
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
    if tasks_by_id:
        for task_id, task in tasks_by_id.items():
            if not task.get("requires_decision"):
                decision = next(
                    (row for row in decision_rows if row.get("reviewer_task_id") == task_id),
                    None,
                )
                if (
                    task.get("category") == "optional_spot_check"
                    and str((decision or {}).get("selected_user_action") or "") == "needs_review"
                ):
                    unresolved_blockers.append(task_id)
                continue
            decision = next(
                (row for row in decision_rows if row.get("reviewer_task_id") == task_id),
                None,
            )
            status = str((decision or {}).get("review_status") or "pending")
            if status in {"pending", "correction_required", "deferred", "escalated"}:
                unresolved_blockers.append(task_id)
            elif task.get("category") == "action_required" and status != "corrected":
                unresolved_blockers.append(task_id)
    else:
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
        "known_reviewer_task_count": len(tasks_by_id),
        "schema_version": schema_version,
        "status": "valid" if not errors else "invalid",
        "unresolved_blocking_items": sorted(set(unresolved_blockers)),
        "unresolved_blocking_item_count": len(set(unresolved_blockers)),
    }


def add_visual_evidence(
    items: list[dict[str, Any]],
    *,
    source_video_path: Path,
    privacy_input: rendering.PrivacyRegionInput,
    privacy_regions_path: Path | None,
    corrected_privacy_input: rendering.PrivacyRegionInput | None,
    corrected_privacy_regions_path: Path | None,
    interval_decisions_path: Path | None,
    corrected_interval_decisions_path: Path | None,
    output_dir: Path,
    bundle_paths: dict[str, Path],
    fps: float,
    frame_count: int,
    source_width: int,
    source_height: int,
    source_has_audio: bool,
    config: ReviewBundleConfig,
    overwrite: bool,
) -> None:
    config.review_render.validate()
    render_settings = config.review_render.render_settings()
    renderer_config_hash = config.review_render.stable_hash()
    renderer_config = config.review_render.payload()
    for item in items:
        item_id = item["review_item_id"]
        start, end = clip_range_for_item(item, frame_count=frame_count, config=config)
        frame_mapping = TaskFrameMapping(start, end)
        interval = {
            "duration_seconds": round(frame_to_seconds(end - start + 1, fps), 6),
            "event_frame_end": int(item["frame_end"]),
            "event_frame_start": int(item["frame_start"]),
            "fps": round(fps, 6),
            "frame_mapping": frame_mapping.payload(),
            "frame_count": end - start + 1,
            "frame_end": end,
            "frame_start": start,
            "height": source_height,
            "start_seconds": frame_to_seconds(start, fps),
            "width": source_width,
        }
        visual: dict[str, str] = {}
        role_audits: dict[str, Any] = {}
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
        source_audit = render_review_clip_by_frames(
            source_video_path,
            source_clip,
            start,
            end,
            fps=fps,
            width=source_width,
            height=source_height,
            source_has_audio=source_has_audio,
            role="original",
            regions_by_frame={},
            settings=None,
            privacy_regions_path=None,
            interval_decisions_path=None,
            renderer_config=renderer_config,
            renderer_config_hash=renderer_config_hash,
            overwrite=overwrite,
            decoded_audit_dir=bundle_paths["decoded_frame_audit"],
            media_slug=f"{item_id}_source",
        )
        visual["source_clip"] = relative_to(source_clip, output_dir)
        role_audits["original"] = source_audit
        if privacy_regions_path is not None and privacy_input.regions_by_frame:
            preview_clip = bundle_paths["clips"] / f"{item_id}_automatic_preview.mp4"
            automatic_audit = render_review_clip_by_frames(
                source_video_path,
                preview_clip,
                start,
                end,
                fps=fps,
                width=source_width,
                height=source_height,
                source_has_audio=source_has_audio,
                role="automatic",
                regions_by_frame=privacy_input.regions_by_frame,
                settings=render_settings,
                privacy_regions_path=privacy_regions_path,
                interval_decisions_path=interval_decisions_path,
                renderer_config=renderer_config,
                renderer_config_hash=renderer_config_hash,
                overwrite=overwrite,
                decoded_audit_dir=bundle_paths["decoded_frame_audit"],
                media_slug=f"{item_id}_automatic_preview",
            )
            visual["automatic_preview_clip"] = relative_to(preview_clip, output_dir)
            automatic_application = validate_rendered_region_application(
                original_clip_path=source_clip,
                rendered_clip_path=preview_clip,
                frame_mapping=frame_mapping,
                regions_by_frame=privacy_input.regions_by_frame,
                settings=render_settings,
                source_video_path=source_video_path,
                decoded_audit_dir=bundle_paths["decoded_frame_audit"],
                media_slug=f"{item_id}_automatic_preview_application",
            )
            if not automatic_application["passed"]:
                raise ValueError(
                    "automatic clip rendered-region validation failed: "
                    + "; ".join(automatic_application["errors"])
                )
            role_audits["automatic"] = automatic_audit | {
                "coverage": privacy_coverage_for_interval(
                    privacy_input,
                    int(item["frame_start"]),
                    int(item["frame_end"]),
                ),
                "displayed_clip_coverage": privacy_coverage_for_interval(privacy_input, start, end),
                "rendered_region_application_validation": automatic_application,
                "warning": "Automatic result is shown to explain why review was required. It is not final."
                if item.get("blocking_status")
                else None,
            }
        if corrected_privacy_input is not None and corrected_privacy_regions_path is not None:
            final_clip = bundle_paths["clips"] / f"{item_id}_reviewed_final.mp4"
            corrected_audit = render_review_clip_by_frames(
                source_video_path,
                final_clip,
                start,
                end,
                fps=fps,
                width=source_width,
                height=source_height,
                source_has_audio=source_has_audio,
                role="corrected_final",
                regions_by_frame=corrected_privacy_input.regions_by_frame,
                settings=render_settings,
                privacy_regions_path=corrected_privacy_regions_path,
                interval_decisions_path=corrected_interval_decisions_path,
                renderer_config=renderer_config,
                renderer_config_hash=renderer_config_hash,
                overwrite=overwrite,
                decoded_audit_dir=bundle_paths["decoded_frame_audit"],
                media_slug=f"{item_id}_reviewed_final",
            )
            visual["reviewed_final_clip"] = relative_to(final_clip, output_dir)
            corrected_application = validate_rendered_region_application(
                original_clip_path=source_clip,
                rendered_clip_path=final_clip,
                frame_mapping=frame_mapping,
                regions_by_frame=corrected_privacy_input.regions_by_frame,
                settings=render_settings,
                source_video_path=source_video_path,
                decoded_audit_dir=bundle_paths["decoded_frame_audit"],
                media_slug=f"{item_id}_reviewed_final_application",
            )
            if not corrected_application["passed"]:
                raise ValueError(
                    "corrected final clip rendered-region validation failed: "
                    + "; ".join(corrected_application["errors"])
                )
            role_audits["corrected_final"] = corrected_audit | {
                "coverage": privacy_coverage_for_interval(
                    corrected_privacy_input,
                    int(item["frame_start"]),
                    int(item["frame_end"]),
                ),
                "displayed_clip_coverage": privacy_coverage_for_interval(
                    corrected_privacy_input,
                    start,
                    end,
                ),
                "rendered_region_application_validation": corrected_application,
                "traceable_reviewed_region_source": corrected_final_is_traceable(
                    corrected_privacy_input,
                    int(item["frame_start"]),
                    int(item["frame_end"]),
                ),
            }
        item["clip_interval"] = interval
        item["visual_evidence"] = visual
        item["media_comparison_audit"] = {
            "renderer_configuration": renderer_config,
            "renderer_configuration_hash": renderer_config_hash,
            "roles": role_audits,
            "source_interval": interval,
            "validation": validate_media_comparison_audit(role_audits),
        }
        validation_errors = item["media_comparison_audit"]["validation"]["errors"]
        if validation_errors:
            raise ValueError("; ".join(validation_errors))


def clip_range_for_item(
    item: dict[str, Any],
    *,
    frame_count: int,
    config: ReviewBundleConfig,
) -> tuple[int, int]:
    context = config.context_frames
    if set(item.get("reason_codes", [])) == {"ground_truth_detector_miss"} and int(
        item["frame_start"]
    ) == int(item["frame_end"]):
        context = min(context, config.one_frame_miss_context_frames)
    start = max(0, int(item["frame_start"]) - context)
    end = min(frame_count - 1, int(item["frame_end"]) + context)
    return start, max(start, end)


def render_review_clip_by_frames(
    source_video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    width: int,
    height: int,
    source_has_audio: bool,
    role: str,
    regions_by_frame: dict[int, list[rendering.RenderRegion]],
    settings: rendering.RenderSettings | None,
    privacy_regions_path: Path | None,
    interval_decisions_path: Path | None,
    renderer_config: dict[str, Any],
    renderer_config_hash: str,
    overwrite: bool,
    decoded_audit_dir: Path | None = None,
    media_slug: str | None = None,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_only_path = output_path.with_name(f".{output_path.stem}.video_only.mp4")
    if video_only_path.exists():
        video_only_path.unlink()

    capture = cv.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source_video_path}")
    writer = cv.VideoWriter(
        str(video_only_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create video: {video_only_path}")

    source_frame_hashes: list[dict[str, Any]] = []
    sample_frames = {
        start_frame,
        start_frame + ((end_frame - start_frame) // 2),
        end_frame,
    }
    rendered_frames = 0
    rendered_regions = 0
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in sample_frames:
                source_frame_hashes.append(
                    {
                        "frame_index": frame_index,
                        "sha256_bgr": stable_frame_hash(frame),
                    }
                )
            if settings is None:
                output = frame
            else:
                output = rendering.render_frame(frame, frame_index, regions_by_frame, settings)
                rendered_regions += rendering.count_renderable_regions(
                    regions_by_frame.get(frame_index, []),
                    frame_width=width,
                    frame_height=height,
                    settings=settings,
                )
            writer.write(output)
            rendered_frames += 1
    finally:
        capture.release()
        writer.release()

    if rendered_frames != end_frame - start_frame + 1:
        if video_only_path.exists():
            video_only_path.unlink()
        raise ValueError(
            f"{role} clip frame read failed: expected {end_frame - start_frame + 1}, "
            f"rendered {rendered_frames}"
        )

    audio_handling = mux_review_clip_audio(
        source_video_path,
        video_only_path,
        output_path,
        source_has_audio=source_has_audio,
        start_seconds=frame_to_seconds(start_frame, fps),
        duration_seconds=frame_to_seconds(rendered_frames, fps),
    )
    if video_only_path.exists():
        video_only_path.unlink()

    output_info = media.probe_media(output_path)
    output_fps = rational_to_float(output_info.video_stream.nominal_frame_rate) or 0.0
    visual_content_validation = validate_review_clip_visual_content(
        source_video_path=source_video_path,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        role=role,
        expected_fps=fps,
        expected_width=width,
        expected_height=height,
        expected_frame_count=rendered_frames,
        output_info=output_info,
        decoded_audit_dir=decoded_audit_dir,
        media_slug=media_slug or output_path.stem,
    )
    if not visual_content_validation["passed"]:
        raise ValueError(
            f"{role} clip visual-content validation failed: "
            + "; ".join(visual_content_validation["errors"])
        )
    return {
        "audio_handling": audio_handling,
        "duration_seconds": output_info.duration_seconds,
        "codec": output_info.video_stream.codec,
        "decoded_visual_content_validation": visual_content_validation,
        "interval_decision_input_hash": media.sha256_file(interval_decisions_path)
        if interval_decisions_path is not None
        else None,
        "interval_decision_input_path": safe_input_path(interval_decisions_path),
        "output_frame_count": as_optional_int(output_info.video_stream.frame_count)
        or rendered_frames,
        "output_fps": round(output_fps, 6),
        "output_hash": media.sha256_file(output_path),
        "output_path": output_path.name,
        "pixel_format": output_info.video_stream.pixel_format,
        "output_resolution": {
            "height": output_info.video_stream.height,
            "width": output_info.video_stream.width,
        },
        "privacy_region_input_hash": media.sha256_file(privacy_regions_path)
        if privacy_regions_path is not None
        else None,
        "privacy_region_input_path": safe_input_path(privacy_regions_path),
        "redaction_rendered_region_count": rendered_regions,
        "rendered_privacy_regions": settings is not None,
        "renderer_configuration": renderer_config,
        "renderer_configuration_hash": renderer_config_hash,
        "role": role,
        "source_frame_hashes": source_frame_hashes,
        "source_frame_range": {
            "frame_count": rendered_frames,
            "frame_end": end_frame,
            "frame_start": start_frame,
        },
    }


def mux_review_clip_audio(
    source_video_path: Path,
    video_only_path: Path,
    output_path: Path,
    *,
    source_has_audio: bool,
    start_seconds: float,
    duration_seconds: float,
) -> str:
    if output_path.exists():
        output_path.unlink()
    command = [
        media.executable_path("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
    ]
    if source_has_audio:
        command.extend(
            [
                "-ss",
                f"{start_seconds:.6f}",
                "-t",
                f"{duration_seconds:.6f}",
                "-i",
                str(source_video_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
            ]
        )
    else:
        command.extend(["-map", "0:v:0"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if source_has_audio:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    command.append(str(output_path))
    subprocess.run(command, capture_output=True, text=True, check=True)
    if not source_has_audio:
        return "h264_yuv420p_video_only_output"
    return "h264_yuv420p_with_source_audio_segment"


def validate_review_clip_visual_content(
    *,
    source_video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    role: str,
    expected_fps: float,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
    output_info: media.MediaInfo,
    decoded_audit_dir: Path | None = None,
    media_slug: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    decoded_frames = decode_video_frames(output_path)
    decoded_count = len(decoded_frames)
    if not decoded_frames:
        errors.append("no frames could be decoded from generated MP4")

    video_stream = output_info.video_stream
    output_fps = rational_to_float(video_stream.nominal_frame_rate) or 0.0
    if video_stream.codec != "h264":
        errors.append(f"browser-compatible codec required: expected h264, got {video_stream.codec}")
    if video_stream.pixel_format != "yuv420p":
        errors.append(
            f"browser-compatible pixel format required: expected yuv420p, got {video_stream.pixel_format}"
        )
    if decoded_count != expected_frame_count:
        errors.append(
            f"decoded frame count mismatch: expected {expected_frame_count}, got {decoded_count}"
        )
    if video_stream.width != expected_width or video_stream.height != expected_height:
        errors.append(
            "resolution mismatch: "
            f"expected {expected_width}x{expected_height}, got {video_stream.width}x{video_stream.height}"
        )
    if abs(output_fps - expected_fps) > 0.02:
        errors.append(f"fps mismatch: expected {expected_fps:.6f}, got {output_fps:.6f}")

    output_sample_offsets = sample_offsets(decoded_count)
    source_sample_indices = {
        label: start_frame + offset for label, offset in output_sample_offsets.items()
    }
    source_frames = read_source_sample_frames(source_video_path, source_sample_indices)
    sample_payloads: dict[str, Any] = {}
    for label, output_offset in output_sample_offsets.items():
        source_frame = source_frames.get(label)
        output_frame = decoded_frames[output_offset] if 0 <= output_offset < decoded_count else None
        sample_payloads[label] = visual_sample_payload(
            source_frame=source_frame,
            output_frame=output_frame,
            output_offset=output_offset,
            source_frame_index=source_sample_indices[label],
        )

    output_hashes = [
        payload["output_metrics"]["sha256_bgr"]
        for payload in sample_payloads.values()
        if payload.get("output_metrics")
    ]
    source_hashes = [
        payload["source_metrics"]["sha256_bgr"]
        for payload in sample_payloads.values()
        if payload.get("source_metrics")
    ]
    all_sampled_output_identical = bool(output_hashes) and len(set(output_hashes)) == 1
    all_sampled_source_identical = bool(source_hashes) and len(set(source_hashes)) == 1
    all_decoded_frames_identical = all_frames_identical(decoded_frames)

    black_failures: list[str] = []
    source_alignment_diffs: list[float] = []
    for label, payload in sample_payloads.items():
        source_metrics = payload.get("source_metrics")
        output_metrics = payload.get("output_metrics")
        if not source_metrics or not output_metrics:
            errors.append(f"{label}: missing source or output sample")
            continue
        if output_is_unexpectedly_black(source_metrics, output_metrics):
            black_failures.append(label)
        mean_absdiff = payload.get("output_vs_source_mean_absdiff")
        if isinstance(mean_absdiff, (int, float)):
            source_alignment_diffs.append(float(mean_absdiff))

    if black_failures:
        errors.append(
            "generated clip samples are near-black while matching source frames contain visible structure: "
            + ", ".join(black_failures)
        )
    if all_sampled_output_identical and not all_sampled_source_identical and len(output_hashes) > 1:
        errors.append("sampled output frames are unexpectedly identical")
    if all_decoded_frames_identical and not all_sampled_source_identical and decoded_count > 1:
        errors.append("all decoded output frames are unexpectedly identical")
    if role == "original" and source_alignment_diffs:
        max_diff = max(source_alignment_diffs)
        if max_diff > 30.0:
            errors.append(
                f"original clip does not visually match selected source interval; max mean absdiff {max_diff:.3f}"
            )
    elif source_alignment_diffs and min(source_alignment_diffs) > 75.0:
        warnings.append(
            "all sampled redacted frames differ substantially from source; verify interval alignment"
        )

    contact_sheet_path: Path | None = None
    if decoded_audit_dir is not None and decoded_frames:
        decoded_audit_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet_path = (
            decoded_audit_dir / f"{normalize_reason_code(media_slug or output_path.stem)}.jpg"
        )
        write_decoded_frame_contact_sheet(
            contact_sheet_path,
            [
                (label, source_sample_indices[label], decoded_frames[offset])
                for label, offset in output_sample_offsets.items()
            ],
        )

    return {
        "all_decoded_frames_identical": all_decoded_frames_identical,
        "all_sampled_output_frames_identical": all_sampled_output_identical,
        "all_sampled_source_frames_identical": all_sampled_source_identical,
        "browser_compatible": video_stream.codec == "h264"
        and video_stream.pixel_format == "yuv420p",
        "contact_sheet_path": relative_to(contact_sheet_path, decoded_audit_dir.parent)
        if contact_sheet_path is not None and decoded_audit_dir is not None
        else None,
        "decoded_frame_count": decoded_count,
        "errors": errors,
        "expected_frame_count": expected_frame_count,
        "output_codec": video_stream.codec,
        "output_fps": round(output_fps, 6),
        "output_pixel_format": video_stream.pixel_format,
        "output_resolution": {
            "height": video_stream.height,
            "width": video_stream.width,
        },
        "passed": not errors,
        "sample_metrics": sample_payloads,
        "source_frame_range": {
            "frame_count": expected_frame_count,
            "frame_end": end_frame,
            "frame_start": start_frame,
        },
        "warnings": warnings,
    }


def decode_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def sample_offsets(frame_count: int) -> dict[str, int]:
    if frame_count <= 0:
        return {}
    return {
        "first": 0,
        "middle": (frame_count - 1) // 2,
        "final": frame_count - 1,
    }


def read_source_sample_frames(
    source_video_path: Path,
    source_sample_indices: dict[str, int],
) -> dict[str, np.ndarray]:
    capture = cv.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        return {}
    frames: dict[str, np.ndarray] = {}
    try:
        for label, frame_index in source_sample_indices.items():
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                frames[label] = frame
    finally:
        capture.release()
    return frames


def visual_sample_payload(
    *,
    source_frame: np.ndarray | None,
    output_frame: np.ndarray | None,
    output_offset: int,
    source_frame_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "decoded_output_frame_offset": output_offset,
        "source_frame_index": source_frame_index,
    }
    if source_frame is not None:
        payload["source_metrics"] = decoded_frame_metrics(source_frame)
    if output_frame is not None:
        payload["output_metrics"] = decoded_frame_metrics(output_frame)
    if (
        source_frame is not None
        and output_frame is not None
        and source_frame.shape == output_frame.shape
    ):
        diff = cv.absdiff(source_frame, output_frame)
        payload["output_vs_source_mean_absdiff"] = round(float(diff.mean()), 6)
        payload["output_vs_source_max_absdiff"] = int(diff.max())
    else:
        payload["output_vs_source_mean_absdiff"] = None
        payload["output_vs_source_max_absdiff"] = None
    return payload


def decoded_frame_metrics(frame: np.ndarray) -> dict[str, Any]:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    near_black = gray < 8
    return {
        "mean_luminance": round(float(gray.mean()), 6),
        "near_black_percent": round(float(near_black.mean() * 100.0), 6),
        "non_black_percent": round(float((~near_black).mean() * 100.0), 6),
        "sha256_bgr": stable_frame_hash(frame),
        "std_luminance": round(float(gray.std()), 6),
    }


def output_is_unexpectedly_black(
    source_metrics: dict[str, Any],
    output_metrics: dict[str, Any],
) -> bool:
    output_black = (
        float(output_metrics["mean_luminance"]) < 8.0
        and float(output_metrics["std_luminance"]) < 2.0
        and float(output_metrics["near_black_percent"]) > 98.0
    )
    source_has_visible_structure = (
        float(source_metrics["mean_luminance"]) >= 8.0
        or float(source_metrics["std_luminance"]) >= 4.0
        or float(source_metrics["near_black_percent"]) <= 95.0
    )
    return output_black and source_has_visible_structure


def all_frames_identical(frames: list[np.ndarray]) -> bool:
    if len(frames) <= 1:
        return False
    first_hash = stable_frame_hash(frames[0])
    return all(stable_frame_hash(frame) == first_hash for frame in frames[1:])


def write_decoded_frame_contact_sheet(
    output_path: Path,
    samples: list[tuple[str, int, np.ndarray]],
) -> None:
    tiles: list[np.ndarray] = []
    for label, frame_index, frame in samples:
        tile = resize_tile(frame, width=360)
        cv.rectangle(tile, (0, 0), (tile.shape[1] - 1, 40), (255, 255, 255), -1)
        cv.putText(
            tile,
            f"{label} | frame {frame_index}",
            (12, 27),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    height = max(tile.shape[0] for tile in tiles)
    width = sum(tile.shape[1] for tile in tiles)
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 0
    for tile in tiles:
        sheet[: tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1]
    cv.imwrite(str(output_path), sheet)


def validate_rendered_region_application(
    *,
    original_clip_path: Path,
    rendered_clip_path: Path,
    frame_mapping: TaskFrameMapping,
    regions_by_frame: dict[int, list[rendering.RenderRegion]],
    settings: rendering.RenderSettings,
    source_video_path: Path | None = None,
    decoded_audit_dir: Path | None = None,
    media_slug: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    original_frames = decode_video_frames(original_clip_path)
    rendered_frames = decode_video_frames(rendered_clip_path)
    if len(original_frames) != frame_mapping.frame_count:
        errors.append(
            f"original clip decoded frame count mismatch: expected {frame_mapping.frame_count}, "
            f"got {len(original_frames)}"
        )
    if len(rendered_frames) != frame_mapping.frame_count:
        errors.append(
            f"rendered clip decoded frame count mismatch: expected {frame_mapping.frame_count}, "
            f"got {len(rendered_frames)}"
        )

    region_results: list[dict[str, Any]] = []
    expected_region_frames: set[int] = set()
    successful_region_frames: set[int] = set()
    missing_region_frames: set[int] = set()
    unexpected_rendered_region_frames: set[int] = set()
    expected_region_count = 0
    successful_region_count = 0
    source_capture = cv.VideoCapture(str(source_video_path)) if source_video_path else None
    if source_capture is not None and not source_capture.isOpened():
        source_capture.release()
        source_capture = None
    source_capture_next_frame_index = frame_mapping.source_frame_start
    if source_capture is not None:
        source_capture.set(cv.CAP_PROP_POS_FRAMES, frame_mapping.source_frame_start)

    def read_source_frame_for_expected_render(source_frame_index: int) -> np.ndarray | None:
        nonlocal source_capture_next_frame_index
        if source_capture is None:
            return None
        if source_capture_next_frame_index != source_frame_index:
            source_capture.set(cv.CAP_PROP_POS_FRAMES, source_frame_index)
            source_capture_next_frame_index = source_frame_index
        ok, frame = source_capture.read()
        source_capture_next_frame_index = source_frame_index + 1
        return frame if ok else None

    try:
        for source_frame_index in range(
            frame_mapping.source_frame_start,
            frame_mapping.source_frame_end + 1,
        ):
            local_index = frame_mapping.local_for_source(source_frame_index)
            if local_index >= len(original_frames) or local_index >= len(rendered_frames):
                continue
            original = original_frames[local_index]
            rendered = rendered_frames[local_index]
            renderable_regions = [
                region
                for region in regions_by_frame.get(source_frame_index, [])
                if region.is_renderable
            ]
            frame_region_successes: list[bool] = []
            if not renderable_regions:
                if unexpected_frame_change(original, rendered):
                    unexpected_rendered_region_frames.add(source_frame_index)
                continue
            expected_region_frames.add(source_frame_index)
            expected_render_source_frame: np.ndarray | None = None
            expected_rendered_frame: np.ndarray | None = None
            expected_render_source = "original_clip"
            for region_index, region in enumerate(renderable_regions):
                expected_region_count += 1
                box = rendering.expanded_clipped_box(
                    region.bbox_xyxy,
                    frame_width=original.shape[1],
                    frame_height=original.shape[0],
                    margin_pixels=settings.margin_pixels,
                    margin_fraction=settings.margin_fraction,
                )
                target_box = rendering.clipped_box(
                    region.bbox_xyxy,
                    frame_width=original.shape[1],
                    frame_height=original.shape[0],
                )
                if box is None or target_box is None:
                    result = {
                        "bbox_xyxy": region.bbox_xyxy,
                        "error": "invalid expanded bbox",
                        "local_task_frame_index": local_index,
                        "provenance": region.provenance,
                        "region_index": region_index,
                        "render_applied": False,
                        "source_frame_index": source_frame_index,
                    }
                    region_results.append(result)
                    frame_region_successes.append(False)
                    continue
                metrics = region_difference_metrics(original, rendered, box)
                target_metrics = region_difference_metrics(original, rendered, target_box)
                target_region_changed = bool(target_metrics["region_changed"])
                legacy_expanded_render_applied = region_redaction_applied(metrics)
                legacy_render_applied = legacy_expanded_render_applied and target_region_changed
                target_region_privacy_signal = legacy_render_applied or target_region_changed
                source_backed_expected_match_required = source_capture is not None
                render_applied = False if source_backed_expected_match_required else legacy_render_applied
                expected_render_match: dict[str, Any] | None = None
                if source_backed_expected_match_required or (
                    target_region_privacy_signal and not render_applied
                    ):
                    if expected_rendered_frame is None:
                        expected_render_source_frame = read_source_frame_for_expected_render(
                            source_frame_index
                        )
                        if expected_render_source_frame is None:
                            expected_render_source_frame = original
                            expected_render_source = "original_clip"
                        else:
                            expected_render_source = "source_video"
                        expected_rendered_frame = rendering.render_frame(
                            expected_render_source_frame,
                            source_frame_index,
                            regions_by_frame,
                            settings,
                        )
                    expected_render_match = expected_render_match_metrics(
                        expected_rendered_frame,
                        rendered,
                        target_box,
                        source_frame=expected_render_source_frame,
                    )
                    render_applied = (
                        expected_render_matches_clip(expected_render_match)
                        if source_backed_expected_match_required
                        else legacy_render_applied
                        or expected_render_matches_clip(expected_render_match)
                    )
                if render_applied:
                    successful_region_count += 1
                result = {
                    "bbox_xyxy": list(region.bbox_xyxy) if region.bbox_xyxy else None,
                    "changed_pixel_fraction_inside_bbox": metrics[
                        "changed_pixel_fraction_inside_bbox"
                    ],
                    "expanded_render_bbox_xyxy": list(box),
                    "expected_render_changed_pixel_fraction_inside_bbox": None
                    if expected_render_match is None
                    else expected_render_match["changed_pixel_fraction_inside_bbox"],
                    "expected_render_matches_clip": None
                    if expected_render_match is None
                    else expected_render_match["matches_clip"],
                    "expected_render_mean_absdiff_inside_bbox": None
                    if expected_render_match is None
                    else expected_render_match["mean_absdiff_inside_bbox"],
                    "expected_render_source": None
                    if expected_render_match is None
                    else expected_render_source,
                    "expected_render_vs_source_changed_pixel_fraction": None
                    if expected_render_match is None
                    else expected_render_match[
                        "expected_vs_source_changed_pixel_fraction_inside_bbox"
                    ],
                    "expected_render_vs_source_mean_absdiff_inside_bbox": None
                    if expected_render_match is None
                    else expected_render_match["expected_vs_source_mean_absdiff_inside_bbox"],
                    "localized_mean_absdiff_inside_bbox": metrics[
                        "localized_mean_absdiff_inside_bbox"
                    ],
                    "local_task_frame_index": local_index,
                    "legacy_expanded_inside_outside_render_applied": (
                        legacy_expanded_render_applied
                    ),
                    "legacy_inside_outside_render_applied": legacy_render_applied,
                    "max_absdiff_inside_bbox": metrics["max_absdiff_inside_bbox"],
                    "mean_absdiff_inside_bbox": metrics["mean_absdiff_inside_bbox"],
                    "mean_absdiff_outside_bbox": metrics["mean_absdiff_outside_bbox"],
                    "provenance": region.provenance,
                    "region_index": region_index,
                    "render_applied": render_applied,
                    "source_frame_index": source_frame_index,
                    "target_region_privacy_signal": target_region_privacy_signal,
                    "target_bbox_changed_pixel_fraction": target_metrics[
                        "changed_pixel_fraction_inside_bbox"
                    ],
                    "target_bbox_mean_absdiff": target_metrics["localized_mean_absdiff_inside_bbox"],
                }
                region_results.append(result)
                frame_region_successes.append(render_applied)
            if frame_region_successes and all(frame_region_successes):
                successful_region_frames.add(source_frame_index)
            else:
                missing_region_frames.add(source_frame_index)
    finally:
        if source_capture is not None:
            source_capture.release()

    if missing_region_frames:
        errors.append(
            "expected privacy regions were not visibly applied on source frames: "
            + frame_ranges_text(sorted(missing_region_frames))
        )
    if unexpected_rendered_region_frames:
        errors.append(
            "rendered clip changed frames with no expected privacy region: "
            + frame_ranges_text(sorted(unexpected_rendered_region_frames))
        )
    if expected_region_count and successful_region_count != expected_region_count:
        errors.append(
            f"rendered region count mismatch: expected {expected_region_count}, "
            f"verified {successful_region_count}"
        )

    contact_sheet_path: Path | None = None
    if decoded_audit_dir is not None and expected_region_frames:
        decoded_audit_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet_path = (
            decoded_audit_dir
            / f"{normalize_reason_code(media_slug or rendered_clip_path.stem)}.jpg"
        )
        write_application_contact_sheet(
            contact_sheet_path,
            original_frames=original_frames,
            rendered_frames=rendered_frames,
            frame_mapping=frame_mapping,
            regions_by_frame=regions_by_frame,
            settings=settings,
        )

    return {
        "contact_sheet_path": relative_to(contact_sheet_path, decoded_audit_dir.parent)
        if contact_sheet_path is not None and decoded_audit_dir is not None
        else None,
        "errors": errors,
        "expected_region_count": expected_region_count,
        "expected_region_frames": sorted(expected_region_frames),
        "frame_mapping": frame_mapping.payload(),
        "missing_rendered_region_frames": sorted(missing_region_frames),
        "passed": not errors,
        "region_results": region_results,
        "successfully_rendered_region_count": successful_region_count,
        "successfully_rendered_region_frames": sorted(successful_region_frames),
        "unexpected_rendered_region_frames": sorted(unexpected_rendered_region_frames),
    }


def region_difference_metrics(
    original: np.ndarray,
    rendered: np.ndarray,
    box: Box,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    original_roi = original[y1:y2, x1:x2]
    rendered_roi = rendered[y1:y2, x1:x2]
    inside_diff = cv.absdiff(original_roi, rendered_roi)
    outside_mask = np.ones(original.shape[:2], dtype=np.uint8)
    outside_mask[y1:y2, x1:x2] = 0
    outside_diff = cv.absdiff(original, rendered)
    outside_values = outside_diff[outside_mask.astype(bool)]
    localized_mean_absdiff, changed_pixel_fraction = rendering.redaction_change_metrics(
        original_roi,
        rendered_roi,
    )
    return {
        "changed_pixel_fraction_inside_bbox": round(changed_pixel_fraction, 6),
        "localized_mean_absdiff_inside_bbox": round(localized_mean_absdiff, 6),
        "max_absdiff_inside_bbox": int(inside_diff.max()) if inside_diff.size else 0,
        "mean_absdiff_inside_bbox": round(float(inside_diff.mean()), 6)
        if inside_diff.size
        else 0.0,
        "mean_absdiff_outside_bbox": round(float(outside_values.mean()), 6)
        if outside_values.size
        else 0.0,
        "region_changed": rendering.region_changed_enough(original_roi, rendered_roi),
    }


def region_redaction_applied(metrics: dict[str, Any]) -> bool:
    inside_mean = float(metrics["mean_absdiff_inside_bbox"])
    outside_mean = float(metrics["mean_absdiff_outside_bbox"])
    inside_max = int(metrics["max_absdiff_inside_bbox"])
    return inside_mean >= max(4.0, outside_mean + 2.0) and inside_max >= 40


def expected_render_match_metrics(
    expected_rendered: np.ndarray,
    actual_rendered: np.ndarray,
    box: Box,
    *,
    source_frame: np.ndarray | None = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    expected_roi = expected_rendered[y1:y2, x1:x2]
    actual_roi = actual_rendered[y1:y2, x1:x2]
    mean_absdiff, changed_pixel_fraction = rendering.redaction_change_metrics(
        expected_roi,
        actual_roi,
    )
    expected_vs_source_mean_absdiff: float | None = None
    expected_vs_source_changed_pixel_fraction: float | None = None
    if source_frame is not None:
        source_roi = source_frame[y1:y2, x1:x2]
        expected_vs_source_mean_absdiff, expected_vs_source_changed_pixel_fraction = (
            rendering.redaction_change_metrics(source_roi, expected_roi)
        )
    metrics = {
        "changed_pixel_fraction_inside_bbox": round(changed_pixel_fraction, 6),
        "expected_vs_source_changed_pixel_fraction_inside_bbox": None
        if expected_vs_source_changed_pixel_fraction is None
        else round(expected_vs_source_changed_pixel_fraction, 6),
        "expected_vs_source_mean_absdiff_inside_bbox": None
        if expected_vs_source_mean_absdiff is None
        else round(expected_vs_source_mean_absdiff, 6),
        "mean_absdiff_inside_bbox": round(mean_absdiff, 6),
    }
    return metrics | {"matches_clip": expected_render_matches_clip(metrics)}


def expected_render_matches_clip(metrics: dict[str, Any]) -> bool:
    mean_absdiff = float(metrics["mean_absdiff_inside_bbox"])
    changed_pixel_fraction = float(metrics["changed_pixel_fraction_inside_bbox"])
    if (
        mean_absdiff <= REVIEW_CLIP_EXPECTED_RENDER_MAX_MEAN_ABSDIFF
        and changed_pixel_fraction <= REVIEW_CLIP_EXPECTED_RENDER_MAX_CHANGED_PIXEL_FRACTION
    ):
        return True
    expected_vs_source_mean = metrics.get("expected_vs_source_mean_absdiff_inside_bbox")
    expected_vs_source_fraction = metrics.get(
        "expected_vs_source_changed_pixel_fraction_inside_bbox"
    )
    if expected_vs_source_mean is None or expected_vs_source_fraction is None:
        return False
    return (
        mean_absdiff <= REVIEW_CLIP_EXPECTED_RENDER_MAX_MEAN_ABSDIFF
        and float(expected_vs_source_mean) >= rendering.MIN_REDACTION_MEAN_ABSDIFF
        and float(expected_vs_source_fraction)
        >= rendering.MIN_REDACTION_CHANGED_PIXEL_FRACTION
        and changed_pixel_fraction
        <= float(expected_vs_source_fraction)
        * REVIEW_CLIP_EXPECTED_RENDER_MAX_CHANGED_PIXEL_FRACTION_RATIO
    )


def unexpected_frame_change(original: np.ndarray, rendered: np.ndarray) -> bool:
    diff = cv.absdiff(original, rendered)
    return float(diff.mean()) > 8.0 and int(diff.max()) > 100


def write_application_contact_sheet(
    output_path: Path,
    *,
    original_frames: list[np.ndarray],
    rendered_frames: list[np.ndarray],
    frame_mapping: TaskFrameMapping,
    regions_by_frame: dict[int, list[rendering.RenderRegion]],
    settings: rendering.RenderSettings,
) -> None:
    expected_frames = [
        frame
        for frame in range(frame_mapping.source_frame_start, frame_mapping.source_frame_end + 1)
        if any(region.is_renderable for region in regions_by_frame.get(frame, []))
    ]
    if not expected_frames:
        return
    sampled_frames = [
        expected_frames[0],
        expected_frames[(len(expected_frames) - 1) // 2],
        expected_frames[-1],
    ]
    tiles: list[np.ndarray] = []
    for source_frame_index in sampled_frames:
        local_index = frame_mapping.local_for_source(source_frame_index)
        if local_index >= len(original_frames) or local_index >= len(rendered_frames):
            continue
        original = original_frames[local_index].copy()
        rendered = rendered_frames[local_index].copy()
        for region in regions_by_frame.get(source_frame_index, []):
            if not region.is_renderable:
                continue
            box = rendering.expanded_clipped_box(
                region.bbox_xyxy,
                frame_width=original.shape[1],
                frame_height=original.shape[0],
                margin_pixels=settings.margin_pixels,
                margin_fraction=settings.margin_fraction,
            )
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cv.rectangle(original, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 255), 3)
        pair = np.hstack([original, rendered])
        tile = resize_tile(pair, width=720)
        cv.rectangle(tile, (0, 0), (tile.shape[1] - 1, 44), (255, 255, 255), -1)
        cv.putText(
            tile,
            f"source {source_frame_index} | local {local_index} | Original left, rendered right",
            (10, 29),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    width = max(tile.shape[1] for tile in tiles)
    height = sum(tile.shape[0] for tile in tiles)
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 0
    for tile in tiles:
        sheet[y : y + tile.shape[0], : tile.shape[1]] = tile
        y += tile.shape[0]
    cv.imwrite(str(output_path), sheet)


def frame_ranges_text(frames: list[int]) -> str:
    return ", ".join(
        f"{start}-{end}" if start != end else str(start) for start, end in contiguous_ranges(frames)
    )


def validate_media_comparison_audit(role_audits: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    automatic = role_audits.get("automatic")
    corrected = role_audits.get("corrected_final")
    roles = [role for role in ("original", "automatic", "corrected_final") if role in role_audits]
    for role in roles:
        audit = role_audits[role]
        expected_count = audit["source_frame_range"]["frame_count"]
        if audit.get("output_frame_count") != expected_count:
            errors.append(f"{role}: output frame count does not match source-frame range")
        if audit.get("codec") != "h264":
            errors.append(f"{role}: generated MP4 must use h264 video")
        if audit.get("pixel_format") != "yuv420p":
            errors.append(f"{role}: generated MP4 must use yuv420p pixel format")
        visual_validation = audit.get("decoded_visual_content_validation")
        if visual_validation and not visual_validation.get("passed"):
            errors.append(f"{role}: decoded visual-content validation failed")
        application_validation = audit.get("rendered_region_application_validation")
        if application_validation and not application_validation.get("passed"):
            errors.append(f"{role}: rendered-region application validation failed")
    if automatic and corrected:
        if automatic.get("source_frame_range") != corrected.get("source_frame_range"):
            errors.append("Automatic and Corrected final source-frame ranges differ")
        if automatic.get("renderer_configuration_hash") != corrected.get(
            "renderer_configuration_hash"
        ):
            errors.append("Automatic and Corrected final renderer configurations differ")
        if automatic.get("output_frame_count") != corrected.get("output_frame_count"):
            errors.append("Automatic and Corrected final output frame counts differ")
        duration_delta = abs(
            float(automatic.get("duration_seconds") or 0.0)
            - float(corrected.get("duration_seconds") or 0.0)
        )
        duration_tolerance = 2.0 / max(
            float(automatic.get("output_fps") or corrected.get("output_fps") or 1.0),
            1.0,
        )
        if duration_delta > duration_tolerance:
            errors.append("Automatic and Corrected final durations differ")
        if not corrected.get("privacy_region_input_hash"):
            errors.append("Corrected final has no traceable privacy-region input hash")
        if not corrected.get("traceable_reviewed_region_source"):
            errors.append("Corrected final has no reviewed/preserved region provenance")
    return {
        "errors": errors,
        "passed": not errors,
        "roles": roles,
    }


def privacy_coverage_for_interval(
    privacy_input: rendering.PrivacyRegionInput,
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    covered_frames: list[int] = []
    unresolved_frames: list[int] = []
    provenance_counts: dict[str, int] = {}
    for frame in range(frame_start, frame_end + 1):
        regions = privacy_input.regions_by_frame.get(frame, [])
        renderable = [region for region in regions if region.is_renderable]
        if renderable:
            covered_frames.append(frame)
            for region in renderable:
                provenance_counts[region.provenance] = (
                    provenance_counts.get(region.provenance, 0) + 1
                )
        else:
            unresolved_frames.append(frame)
    return {
        "covered_frame_count": len(covered_frames),
        "event_frame_count": frame_end - frame_start + 1,
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "unresolved_frame_count": len(unresolved_frames),
        "unresolved_frames": unresolved_frames,
    }


def corrected_final_is_traceable(
    corrected_privacy_input: rendering.PrivacyRegionInput,
    frame_start: int,
    frame_end: int,
) -> bool:
    del frame_start, frame_end
    reviewed_provenance = {
        "manual_correction",
        "preserved_privacy_region",
        "preserved_reviewed_correction",
        "reviewer_discovered_correction",
        "reviewed_privacy_region",
        "synthetic_privacy_bridge",
    }
    all_provenance = {
        region.provenance
        for regions in corrected_privacy_input.regions_by_frame.values()
        for region in regions
        if region.is_renderable
    }
    return bool(all_provenance & reviewed_provenance)


def stable_frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def stable_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_input_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


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
    write_json(output_dir / "review_media_audit.json", media_audit_payload(queue))
    write_json(
        output_dir / "review_tasks.json",
        {
            "case_id": queue.get("case_id"),
            "reviewer_tasks": queue.get("reviewer_tasks", []),
            "schema_id": REVIEW_TASK_SCHEMA_ID,
            "schema_version": REVIEW_TASK_SCHEMA_VERSION,
            "technical_audit": queue.get("technical_audit", {}),
        },
    )
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
        "presentation_schema_id": REVIEW_TASK_SCHEMA_ID,
        "presentation_schema_version": REVIEW_TASK_SCHEMA_VERSION,
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


def media_audit_payload(queue: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": queue.get("case_id"),
        "items": [
            {
                "media_comparison_audit": item.get("media_comparison_audit", {}),
                "review_item_id": item.get("review_item_id"),
            }
            for item in queue.get("items", [])
            if item.get("media_comparison_audit")
        ],
        "reviewer_tasks": [
            {
                "media_roles": task.get("media_roles", []),
                "reviewer_task_id": task.get("reviewer_task_id"),
            }
            for task in queue.get("reviewer_tasks", [])
        ],
        "schema_version": REVIEW_TASK_SCHEMA_VERSION,
    }


def decisions_template(queue: dict[str, Any]) -> dict[str, Any]:
    return {
        "decisions": [],
        "export_note": "Use Export Review Decisions in index.html, or fill this manually. Decisions should reference reviewer_task_id for v1.1+ bundles.",
        "review_queue_id": queue.get("case_id"),
        "review_queue_item_count": len(queue.get("items", [])),
        "reviewer_task_count": len(queue.get("reviewer_tasks", [])),
        "schema_version": REVIEW_TASK_SCHEMA_VERSION,
    }


def review_html(queue: dict[str, Any], summary: dict[str, Any]) -> str:
    queue_json = json.dumps(queue, sort_keys=True)
    summary_json = json.dumps(summary, sort_keys=True)
    status_text = initial_status_text(summary)
    status_class = "status blocked" if summary["final_export_blocked"] else "status pending"
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
    <section class="next-action">
      <h2 id="next-action-title">{status_text}</h2>
      <p id="next-action-copy">Review required tasks first. Export decisions before closing this page or moving to another computer.</p>
    </section>
    <section class="summary-grid">
      <div><strong>Required review</strong><span id="required-count">0/0</span></div>
      <div><strong>Corrected results</strong><span id="corrected-count">0/0</span></div>
      <div><strong>Optional spot checks</strong><span id="optional-count">0/0</span></div>
      <div><strong>Technical audit</strong><span>{summary.get("technical_only_items", 0)}</span></div>
    </section>
    <section class="notice">
      <strong>Your choices stay only in this browser until you export them.</strong>
      <p>The default workflow shows reviewer tasks, not every raw pipeline signal. Technical signals remain available below for audit.</p>
      <p>No review queue can prove that every possible missed face was found.</p>
      <p id="file-protocol-note" class="file-protocol-note" hidden>For reliable embedded playback, open this bundle with <code>redactor serve-review-bundle BUNDLE_DIRECTORY --open</code>.</p>
    </section>
    <section class="toolbar">
      <button id="guided-mode" class="active">Guided review</button>
      <button id="all-mode" class="secondary">View all tasks</button>
      <button id="sync-media" class="secondary">Play together</button>
      <button id="restart-media" class="secondary">Restart media</button>
      <button id="export-decisions">Export Review Decisions</button>
      <button id="clear-local">Clear Local Choices</button>
      <span id="media-error" class="media-error" aria-live="polite"></span>
    </section>
    <section id="guided-panel" class="guided-panel"></section>
    <section id="all-tasks" class="task-sections"></section>
    <section id="technical-audit" class="technical-audit"></section>
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
.status { align-self: center; padding: 10px 14px; border-radius: 6px; font-weight: 700; background: #37506f; text-align: center; }
.status.blocked { background: #9f1d1d; }
.status.pending { background: #7a4b00; }
.status.clear { background: #17633a; }
main { max-width: 1180px; margin: 0 auto; padding: 24px; }
.next-action { background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 18px; margin-bottom: 16px; }
.next-action h2 { margin: 0; font-size: 24px; }
.next-action p { margin: 8px 0 0; max-width: 780px; color: #52616f; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.summary-grid div, .notice, .toolbar, .task-card, .section, .technical-audit details { background: white; border: 1px solid #d8dee8; border-radius: 8px; }
.summary-grid div { padding: 14px; }
.summary-grid strong { display: block; font-size: 13px; color: #52616f; }
.summary-grid span { font-size: 26px; font-weight: 700; }
.notice { margin-top: 16px; padding: 16px; }
.notice p { margin: 8px 0 0; color: #52616f; }
.file-protocol-note { background: #fff7e6; border-left: 4px solid #b7791f; padding: 8px 10px; }
.media-error { color: #9f1d1d; font-weight: 700; align-self: center; }
.media-note { color: #52616f; }
.toolbar { position: sticky; top: 0; z-index: 2; margin: 16px 0; padding: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
select, button { min-height: 38px; border: 1px solid #b8c2cc; border-radius: 6px; background: white; padding: 0 12px; }
button { cursor: pointer; background: #1f4e79; color: white; border-color: #1f4e79; }
button.secondary { background: white; color: #1f4e79; }
button:disabled { cursor: not-allowed; opacity: 0.55; }
button.active { background: #16324f; }
button.action { background: #17633a; border-color: #17633a; }
button.warn { background: #9f1d1d; border-color: #9f1d1d; }
button.selected-action { outline: 3px solid #f6ad55; outline-offset: 2px; }
.task-sections { display: grid; gap: 18px; }
.section { padding: 16px; }
.section h2 { margin: 0 0 12px; font-size: 20px; }
.task-list { display: grid; gap: 14px; }
.task-card { padding: 18px; border-left: 6px solid #7b8794; }
.task-card.action_required, .task-card.corrected_result_check, .task-card.automatic_continuity_check { border-left-color: #c62828; }
.task-card.optional_spot_check { border-left-color: #0277bd; }
.task-head { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
.task-head h3 { margin: 0; font-size: 22px; }
.task-meta { color: #52616f; margin: 6px 0 0; }
.task-copy { max-width: 760px; line-height: 1.45; }
.checklist { margin: 8px 0 14px; padding-left: 20px; }
.media-grid { display: grid; gap: 12px; margin-top: 12px; }
.media-grid.cols-1 { grid-template-columns: 1fr; }
.media-grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.media-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.media-card { border: 1px solid #d8dee8; border-radius: 8px; overflow: hidden; background: #f9fafb; }
.media-card h4 { margin: 0; padding: 9px 10px; background: #edf2f7; font-size: 14px; }
.media-subtitle, .media-provenance, .media-warning { margin: 7px 10px; font-size: 13px; line-height: 1.35; color: #52616f; }
.media-warning { color: #9f1d1d; font-weight: 700; }
.media-card video, .media-card img { width: 100%; display: block; background: #111; }
.media-card a { display: inline-block; padding: 8px 10px; }
.duplicate-note { color: #52616f; font-size: 13px; margin: 8px 0 0; }
.actions { position: sticky; bottom: 0; background: rgba(255,255,255,0.96); border-top: 1px solid #d8dee8; padding: 12px 0 0; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.guidance { color: #52616f; width: 100%; margin: 4px 0 0; }
.nav { display: flex; gap: 10px; justify-content: space-between; margin: 12px 0; }
.nav span { align-self: center; color: #52616f; font-weight: 700; }
.links { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.links a { color: #1f4e79; font-weight: 600; }
.technical-audit { margin-top: 18px; }
.technical-audit details { padding: 16px; }
details { margin-top: 14px; }
pre { overflow: auto; background: #f1f5f9; padding: 10px; border-radius: 6px; }
@media (max-width: 900px) { .media-grid.cols-2, .media-grid.cols-3 { grid-template-columns: 1fr; } .topbar { display: block; } }
"""


def review_js() -> str:
    return """
const queue = JSON.parse(document.getElementById('review-data').textContent);
const summary = JSON.parse(document.getElementById('review-summary').textContent);
const storageKey = `review-decisions:${queue.case_id}:${queue.source_video.hash}`;
let decisions = JSON.parse(localStorage.getItem(storageKey) || '{}');
let mode = 'guided';
let guidedIndex = 0;
const tasks = queue.reviewer_tasks || [];
const requiredCategories = new Set(['action_required', 'automatic_continuity_check', 'corrected_result_check']);
const requiredTasks = tasks.filter(task => requiredCategories.has(task.category));
const optionalTasks = tasks.filter(task => task.category === 'optional_spot_check');
const guidedTasks = [...requiredTasks, ...optionalTasks];

function persist() { localStorage.setItem(storageKey, JSON.stringify(decisions)); }
function decisionFor(id) { return decisions[id] || null; }
function isDone(task) { return Boolean(decisionFor(task.reviewer_task_id)); }
function isBlockingDecision(decision) {
  if (!decision) return false;
  return ['needs_review', 'needs_correction', 'escalate'].includes(decision.selected_user_action)
    || ['correction_required', 'escalated', 'deferred'].includes(decision.review_status);
}
function nextPendingIndex() {
  const requiredPending = guidedTasks.findIndex(task => requiredCategories.has(task.category) && !isDone(task));
  if (requiredPending >= 0) return requiredPending;
  const optionalPending = guidedTasks.findIndex(task => !requiredCategories.has(task.category) && !isDone(task));
  if (optionalPending >= 0) return optionalPending;
  return Math.min(guidedIndex, Math.max(0, guidedTasks.length - 1));
}
function setDecision(task, action) {
  decisions[task.reviewer_task_id] = {
    reviewer_task_id: task.reviewer_task_id,
    raw_review_item_ids: task.raw_review_item_ids,
    selected_user_action: action.action,
    selected_user_action_label: action.label,
    review_status: action.review_status,
    reviewed_media_role: action.reviewed_media_role,
    decided_at: new Date().toISOString(),
  };
  persist();
  render();
  if (mode === 'guided') {
    window.setTimeout(() => {
      guidedIndex = nextPendingIndex();
      render();
    }, 350);
  }
}
function progress(items) {
  const total = items.length;
  const done = items.filter(isDone).length;
  return `${done}/${total}`;
}
function render() {
  document.getElementById('required-count').textContent = progress(tasks.filter(task => task.category === 'action_required' || task.category === 'automatic_continuity_check'));
  document.getElementById('corrected-count').textContent = progress(tasks.filter(task => task.category === 'corrected_result_check'));
  document.getElementById('optional-count').textContent = progress(optionalTasks);
  const requiredDone = requiredTasks.filter(isDone).length;
  const finalStatus = document.getElementById('final-status');
  const title = document.getElementById('next-action-title');
  const copy = document.getElementById('next-action-copy');
  const escalatedOptional = optionalTasks.some(task => isBlockingDecision(decisionFor(task.reviewer_task_id)));
  if (escalatedOptional) {
    finalStatus.className = 'status blocked';
    finalStatus.textContent = 'Review required - final export remains blocked';
    title.textContent = finalStatus.textContent;
    copy.textContent = 'A possible missed privacy target was found. Final export remains blocked until this interval is corrected and confirmed.';
  } else if (requiredTasks.length && requiredDone < requiredTasks.length) {
    finalStatus.className = 'status blocked';
    finalStatus.textContent = `${requiredTasks.length - requiredDone} items need your review`;
    title.textContent = finalStatus.textContent;
    copy.textContent = 'Review required and corrected-result tasks first. Export decisions after completing the review.';
  } else if (optionalTasks.some(task => !isDone(task))) {
    finalStatus.className = 'status pending';
    finalStatus.textContent = 'Required review complete';
    title.textContent = 'Required review complete';
    copy.textContent = 'Optional spot checks are available. Export decisions when you are finished.';
  } else {
    finalStatus.className = 'status clear';
    finalStatus.textContent = 'Review complete - decisions must be exported';
    title.textContent = finalStatus.textContent;
    copy.textContent = 'Export your decisions JSON before closing this browser.';
  }
  document.getElementById('guided-panel').style.display = mode === 'guided' ? 'block' : 'none';
  document.getElementById('all-tasks').style.display = mode === 'all' ? 'grid' : 'none';
  document.getElementById('guided-mode').className = mode === 'guided' ? 'active' : 'secondary';
  document.getElementById('all-mode').className = mode === 'all' ? 'active' : 'secondary';
  renderGuided();
  renderAllTasks();
  renderTechnicalAudit();
  updateMediaControlState();
}
function renderGuided() {
  const panel = document.getElementById('guided-panel');
  panel.innerHTML = '';
  if (!guidedTasks.length) {
    panel.innerHTML = '<section class="section"><h2>No reviewer tasks</h2><p>No required or optional reviewer tasks were generated.</p></section>';
    return;
  }
  guidedIndex = Math.max(0, Math.min(guidedIndex, guidedTasks.length - 1));
  const task = guidedTasks[guidedIndex];
  const nav = document.createElement('div');
  nav.className = 'nav';
  nav.innerHTML = `<button class="secondary" id="prev-task">Previous</button><span>Task ${guidedIndex + 1} of ${guidedTasks.length}</span><button class="secondary" id="next-task">Next</button>`;
  panel.appendChild(nav);
  panel.appendChild(taskCard(task));
  document.getElementById('prev-task').onclick = () => { guidedIndex = Math.max(0, guidedIndex - 1); render(); };
  document.getElementById('next-task').onclick = () => { guidedIndex = Math.min(guidedTasks.length - 1, guidedIndex + 1); render(); };
}
function renderAllTasks() {
  const container = document.getElementById('all-tasks');
  container.innerHTML = '';
  const sections = ['Needs your attention', 'Corrected results to confirm', 'Optional spot checks'];
  sections.forEach(section => {
    const sectionTasks = tasks.filter(task => task.section === section);
    if (!sectionTasks.length) return;
    const node = document.createElement('section');
    node.className = 'section';
    node.innerHTML = `<h2>${section}</h2><div class="task-list"></div>`;
    const list = node.querySelector('.task-list');
    sectionTasks.forEach(task => list.appendChild(taskCard(task)));
    container.appendChild(node);
  });
}
function renderTechnicalAudit() {
  const container = document.getElementById('technical-audit');
  const audit = queue.technical_audit || { items: [] };
  container.innerHTML = `<details><summary>Technical audit (${audit.hidden_from_default_task_count || 0} hidden raw signals)</summary><pre>${JSON.stringify({technical_audit: audit, raw_review_queue: queue.items}, null, 2)}</pre></details>`;
}
function taskCard(task) {
  const node = document.createElement('article');
  node.className = `task-card ${task.category}`;
  node.dataset.reviewerTaskId = task.reviewer_task_id;
  const decision = decisionFor(task.reviewer_task_id);
  node.innerHTML = `
    <div class="task-head">
      <div>
        <h3>${task.title}</h3>
        <p class="task-meta">Frames ${task.frame_start}-${task.frame_end} · ${task.timestamps.start_seconds}s-${task.timestamps.end_seconds}s</p>
      </div>
      <strong>${decision ? decision.selected_user_action_label : 'Pending'}</strong>
    </div>
    <div class="task-copy">
      <h4>What happened</h4>
      <p>${task.what_happened}</p>
      <h4>What to check</h4>
      <ul class="checklist">${task.what_to_check.map(value => `<li>${value}</li>`).join('')}</ul>
    </div>
    ${mediaGrid(task)}
    ${duplicateNotes(task)}
    <details><summary>Diagnostic contact sheet and technical details</summary>${diagnosticMedia(task)}<pre>${JSON.stringify(task, null, 2)}</pre></details>
    <div class="actions">
      ${task.allowed_user_actions.map(action => `<button class="${buttonClass(action, decision)}" data-task="${task.reviewer_task_id}" data-action="${action.action}">${action.label}</button>`).join('')}
      <p class="guidance">${task.user_action_guidance}</p>
    </div>`;
  node.querySelectorAll('button[data-action]').forEach(button => {
    const action = task.allowed_user_actions.find(value => value.action === button.dataset.action);
    button.onclick = () => setDecision(task, action);
  });
  return node;
}
function buttonClass(action, decision) {
  const base = action.review_status === 'correction_required' || action.review_status === 'escalated' ? 'warn' : 'action';
  return decision && decision.selected_user_action === action.action ? `${base} selected-action` : base;
}
function mediaGrid(task) {
  const roles = (task.media_roles || []).filter(role => role.display && role.role !== 'diagnostic');
  if (!roles.length) return '<p>No comparison video was generated for this task.</p>';
  const columns = Math.min(3, roles.length);
  return `<div class="media-grid cols-${columns}">${roles.map(role => `
    <div class="media-card">
      <h4>${role.label}</h4>
      ${role.subtitle ? `<p class="media-subtitle">${role.subtitle}</p>` : ''}
      ${role.provenance_statement ? `<p class="media-provenance">${role.provenance_statement}</p>` : ''}
      ${role.warning ? `<p class="media-warning">${role.warning}</p>` : ''}
      <video controls playsinline preload="metadata" src="${role.path}"></video>
      <a href="${role.path}">Open ${role.label}</a>
    </div>`).join('')}</div>`;
}
function diagnosticMedia(task) {
  const role = (task.media_roles || []).find(value => value.role === 'diagnostic');
  if (!role) return '';
  return `<div class="media-card"><h4>${role.label}</h4><img src="${role.path}" alt="Diagnostic contact sheet"><a href="${role.path}">Open contact sheet</a></div>`;
}
function duplicateNotes(task) {
  const duplicates = (task.media_roles || []).filter(role => role.duplicate_of);
  if (!duplicates.length) return '';
  return `<p class="duplicate-note">${duplicates.map(role => `${role.label} matches ${role.duplicate_of} and is shown once.`).join(' ')}</p>`;
}
function exportDecisions() {
  const rows = Object.values(decisions).sort((a, b) => a.reviewer_task_id.localeCompare(b.reviewer_task_id));
  const payload = {
    schema_version: queue.presentation_schema_version || '1.1',
    review_queue_id: queue.case_id,
    source_video_hash: queue.source_video.hash,
    exported_at: new Date().toISOString(),
    decisions: rows,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `review_decisions_${queue.case_id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}
function mediaErrorNode() { return document.getElementById('media-error'); }
function setMediaMessage(message, isError = false) {
  const node = mediaErrorNode();
  node.textContent = message;
  node.className = isError ? 'media-error' : 'media-error media-note';
}
function activeTaskCard() {
  if (mode !== 'guided') return null;
  return document.querySelector('#guided-panel .task-card');
}
function activeComparisonVideos() {
  const card = activeTaskCard();
  if (!card) return [];
  return [...card.querySelectorAll('video')];
}
function clampTime(video, time) {
  if (!Number.isFinite(video.duration) || video.duration <= 0) return Math.max(0, time);
  return Math.max(0, Math.min(time, Math.max(0, video.duration - 0.05)));
}
function waitForMetadata(video) {
  if (video.readyState >= 1) return Promise.resolve();
  video.load();
  return new Promise(resolve => {
    const finish = () => {
      video.removeEventListener('loadedmetadata', finish);
      video.removeEventListener('error', finish);
      resolve();
    };
    video.addEventListener('loadedmetadata', finish, {once: true});
    video.addEventListener('error', finish, {once: true});
    window.setTimeout(finish, 1200);
  });
}
function updateMediaControlState() {
  const syncButton = document.getElementById('sync-media');
  const restartButton = document.getElementById('restart-media');
  const videos = activeComparisonVideos();
  const noGuidedMedia = mode !== 'guided' || !videos.length;
  syncButton.disabled = noGuidedMedia;
  restartButton.disabled = noGuidedMedia;
  if (mode !== 'guided') {
    syncButton.textContent = 'Play together';
    setMediaMessage('Group media controls are available in Guided review.');
    return;
  }
  if (!videos.length) {
    syncButton.textContent = 'Play together';
    setMediaMessage('No comparison media is available for this task.');
    return;
  }
  const anyPlaying = videos.some(video => !video.paused && !video.ended);
  syncButton.textContent = anyPlaying ? 'Pause together' : 'Play together';
  if (!mediaErrorNode().textContent.includes('failed') && !mediaErrorNode().textContent.includes('blocked')) {
    setMediaMessage('');
  }
}
async function alignVideos(videos, time) {
  await Promise.allSettled(videos.map(video => waitForMetadata(video)));
  videos.forEach(video => {
    video.muted = true;
    video.currentTime = clampTime(video, time);
  });
}
function warnIfDrifted(videos) {
  const playing = videos.filter(video => !video.paused && !video.ended);
  if (playing.length < 2) return;
  const times = playing.map(video => video.currentTime);
  const drift = Math.max(...times) - Math.min(...times);
  if (drift > 0.25) {
    setMediaMessage('Comparison clips drifted slightly; restart media to realign them.', true);
  }
}
async function syncMedia(event) {
  if (event) event.preventDefault();
  const videos = activeComparisonVideos();
  if (!videos.length) {
    updateMediaControlState();
    return;
  }
  setMediaMessage('');
  const anyPlaying = videos.some(video => !video.paused && !video.ended);
  if (anyPlaying) {
    videos.forEach(video => video.pause());
    updateMediaControlState();
    return;
  }
  const reference = videos[0];
  const sharedTime = reference.ended ? 0 : (reference.currentTime || 0);
  await alignVideos(videos, sharedTime);
  const results = await Promise.allSettled(videos.map(video => video.play()));
  const rejected = results.find(result => result.status === 'rejected');
  if (rejected) {
    setMediaMessage('Embedded playback failed. Use the direct links, or reload the localhost review page and try again.', true);
  }
  updateMediaControlState();
  window.setTimeout(() => warnIfDrifted(videos), 750);
}
async function restartMedia(event) {
  if (event) event.preventDefault();
  const videos = activeComparisonVideos();
  if (!videos.length) {
    updateMediaControlState();
    return;
  }
  await Promise.allSettled(videos.map(video => waitForMetadata(video)));
  videos.forEach(video => {
    video.pause();
    video.currentTime = 0;
  });
  setMediaMessage('');
  updateMediaControlState();
}
document.getElementById('guided-mode').onclick = () => { mode = 'guided'; render(); };
document.getElementById('all-mode').onclick = () => { mode = 'all'; render(); };
document.getElementById('sync-media').onclick = syncMedia;
document.getElementById('restart-media').onclick = restartMedia;
document.getElementById('export-decisions').onclick = exportDecisions;
document.getElementById('clear-local').onclick = () => {
  if (!window.confirm('Clear local choices for this review bundle?')) return;
  decisions = {};
  localStorage.removeItem(storageKey);
  guidedIndex = nextPendingIndex();
  render();
};
if (window.location.protocol === 'file:') {
  document.getElementById('file-protocol-note').hidden = false;
}
render();
"""


def index_markdown(queue: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Review Bundle Index",
        "",
        f"- Case: `{queue.get('case_id')}`",
        f"- Required review tasks: `{summary.get('required_review_tasks', summary['blocking_items'])}`",
        f"- Corrected results to confirm: `{summary.get('corrected_result_tasks', 0)}`",
        f"- Optional spot checks: `{summary.get('optional_spot_checks', summary['random_audit_items'])}`",
        f"- Technical-only raw signals: `{summary.get('technical_only_items', 0)}`",
        f"- Final export blocked by review queue: `{summary['final_export_blocked']}`",
        "",
        "Open `index.html` locally for the guided reviewer-task workflow.",
        "",
        "Raw pipeline and evaluation signals are preserved in `review_queue.json` and the Technical audit section.",
        "",
        "Absence of review tasks does not prove that every possible missed face was found.",
    ]
    return "\n".join(lines) + "\n"


def checklist_markdown() -> str:
    return """# Review Checklist

1. Open `index.html`.
2. Complete required tasks and corrected-result confirmations first.
3. Compare the embedded Original, Automatic, and Corrected final clips when available.
4. Use `Correct` only when the displayed result covers the privacy target for the whole interval.
5. Use `Needs correction` when valid privacy regions are missing or the result is not acceptable.
6. Complete optional spot checks after required tasks.
7. Export decisions before closing the browser or moving to another computer.
8. Validate exported decisions with `redactor validate-review-decisions`.

This bundle does not implement box drawing or keyframe correction.
"""


def bundle_readme(source_info: Any) -> str:
    return f"""# Automated Face Review Bundle

This directory is a local, self-contained review package. It contains no remote assets,
analytics, or network calls.

Source video hash: `{source_info.sha256}`

Open `index.html` in a browser. The default view shows reviewer tasks, while raw
signals remain available in the Technical audit section. Browser-local choices are
temporary until exported.
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
        "one_frame_miss_context_frames": config.one_frame_miss_context_frames,
        "random_sample_count": config.random_sample_count,
        "random_sample_frame_count": config.random_sample_frame_count,
        "random_seed": config.random_seed,
        "review_render_configuration": config.review_render.payload(),
        "review_render_configuration_hash": config.review_render.stable_hash(),
        "schema_version": SCHEMA_VERSION,
    }


def create_bundle_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "assets": output_dir / "assets",
        "clips": output_dir / "clips",
        "contact_sheets": output_dir / "contact_sheets",
        "decoded_frame_audit": output_dir / "decoded_frame_audit",
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


def merged_event_metrics(draft: ReviewDraft) -> list[dict[str, Any]]:
    merged_events = draft.metrics.get("merged_events")
    if isinstance(merged_events, list):
        return [event for event in merged_events if isinstance(event, dict)]
    return [compact_draft_metrics(draft)]


def combine_explanations(left: str, right: str) -> str:
    parts: list[str] = []
    for value in (left, right):
        for part in re.split(r"\s*Additional related evidence:\s*", value):
            cleaned = part.strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} Related evidence includes {len(parts) - 1} additional unique signal(s); see metrics.merged_events."


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

    def check_media_roles(owner_id: str, roles: list[dict[str, Any]]) -> None:
        for role in roles or []:
            value = role.get("path")
            if value and Path(str(value)).is_absolute():
                findings.append(f"{owner_id} has absolute media role path")

    for item in queue.get("items", []):
        for value in (item.get("visual_evidence") or {}).values():
            if Path(str(value)).is_absolute():
                findings.append(f"{item['review_item_id']} has absolute visual path")
        check_media_roles(str(item.get("review_item_id")), item.get("media_roles", []))
    for task in queue.get("reviewer_tasks", []):
        check_media_roles(str(task.get("reviewer_task_id")), task.get("media_roles", []))
    return findings


def resolve_bundle_request_path(bundle_dir: Path, request_path: str) -> Path:
    root = bundle_dir.resolve()
    parsed = urllib.parse.urlparse(request_path)
    decoded = urllib.parse.unquote(parsed.path)
    normalized = posixpath.normpath(decoded)
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    candidate = root.joinpath(*parts) if parts else root / "index.html"
    resolved = candidate.resolve()
    resolved.relative_to(root)
    return resolved


class SafeReviewBundleHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, bundle_dir: Path, **kwargs: Any) -> None:
        self.bundle_dir = bundle_dir.resolve()
        super().__init__(*args, directory=str(self.bundle_dir), **kwargs)

    def copyfile(self, source: Any, outputfile: Any) -> None:
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            return

    def translate_path(self, path: str) -> str:
        try:
            return str(resolve_bundle_request_path(self.bundle_dir, path))
        except ValueError:
            return str(self.bundle_dir / "__forbidden__")

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def create_review_bundle_server(
    bundle_dir: Path,
    *,
    port: int = 0,
) -> tuple[socketserver.TCPServer, str]:
    root = bundle_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{bundle_dir} is not a directory")
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"{bundle_dir} does not contain index.html")
    mimetypes.add_type("video/mp4", ".mp4")
    handler = lambda *args, **kwargs: SafeReviewBundleHandler(  # noqa: E731
        *args,
        bundle_dir=root,
        **kwargs,
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    host, bound_port = server.server_address
    return server, f"http://{host}:{bound_port}/index.html"


def serve_review_bundle(
    bundle_dir: Path,
    *,
    port: int = 0,
    open_browser: bool = False,
) -> str:
    server, url = create_review_bundle_server(bundle_dir, port=port)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return url
