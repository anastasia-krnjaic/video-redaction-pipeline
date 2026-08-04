from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redactor import media, privacy_gap_recovery, rendering, tracking
from redactor.detection import DETECTOR_BACKEND_ADAS, rational_to_float

SCHEMA_VERSION = "1.0"
POLICY_SCHEMA_ID = "face_privacy_policy_v1"
TRACK_CENSUS_SCHEMA_ID = "face_track_census_v1"
COMPILER_ID = "full_video_face_privacy_policy_compiler_v1"

MODE_REDACT_ALL = "redact_all"
MODE_REDACT_ALL_EXCEPT_KEEP = "redact_all_except_keep"
MODE_REDACT_SELECTED = "redact_selected"
MODE_REVIEW_ONLY = "review_only"
SUPPORTED_POLICY_MODES = {
    MODE_REDACT_ALL,
    MODE_REDACT_ALL_EXCEPT_KEEP,
    MODE_REDACT_SELECTED,
    MODE_REVIEW_ONLY,
}

POLICY_REDACT = "redact"
POLICY_KEEP = "keep"
POLICY_IGNORE = "ignore"
POLICY_REVIEW = "review"

AUTO_CONFIRMED = "auto_confirmed"
REVIEW_CANDIDATE = "review_candidate"
REJECTED = "rejected"
TRACK_QUALITIES = {AUTO_CONFIRMED, REVIEW_CANDIDATE, REJECTED}

PROVENANCE_POLICY_AUTOMATIC = "automatic_case_policy"
PROVENANCE_POLICY_EXPLICIT = "explicit_case_policy"

DEFAULT_PREVIEW_RENDER_SETTINGS = rendering.RenderSettings(
    mode=rendering.MODE_PREVIEW,
    method=rendering.METHOD_PIXELATE,
    margin_fraction=0.15,
    margin_pixels=0,
    pixel_size=16,
)


@dataclass(frozen=True)
class PolicyAssignment:
    track_id: int
    decision: str
    provenance: str
    review_status: str = "not_required"
    identity_group_id: str | None = None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "identity_group_id": self.identity_group_id,
            "provenance": self.provenance,
            "reasons": list(self.reasons),
            "review_status": self.review_status,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class FacePrivacyPolicy:
    case_id: str
    source_video_hash: str
    source_frame_count: int
    mode: str = MODE_REDACT_ALL
    object_category: str = "face"
    assignments: tuple[PolicyAssignment, ...] = ()
    full_source_frame_span: tuple[int, int] | None = None
    policy_provenance: str = PROVENANCE_POLICY_AUTOMATIC
    review_status: str = "not_required"
    unresolved_policy_assignments: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.mode not in SUPPORTED_POLICY_MODES:
            raise ValueError(f"unsupported policy mode: {self.mode}")
        if self.object_category != "face":
            raise ValueError("FacePrivacyPolicy currently supports object_category='face'")
        if self.source_frame_count <= 0:
            raise ValueError("source_frame_count must be positive")
        span = self.full_source_frame_span or (0, self.source_frame_count - 1)
        if span[0] < 0 or span[1] < span[0] or span[1] >= self.source_frame_count:
            raise ValueError("full_source_frame_span is outside the source video")
        seen: set[int] = set()
        for assignment in self.assignments:
            if assignment.track_id in seen:
                raise ValueError(f"duplicate policy assignment for track {assignment.track_id}")
            seen.add(assignment.track_id)
            if assignment.decision not in {
                POLICY_REDACT,
                POLICY_KEEP,
                POLICY_IGNORE,
                POLICY_REVIEW,
            }:
                raise ValueError(f"unsupported policy decision: {assignment.decision}")

    def assignment_for(self, track_id: int) -> PolicyAssignment | None:
        for assignment in self.assignments:
            if assignment.track_id == track_id:
                return assignment
        return None

    def to_dict(self) -> dict[str, object]:
        self.validate()
        span = self.full_source_frame_span or (0, self.source_frame_count - 1)
        return {
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "case_id": self.case_id,
            "full_source_frame_span": {"frame_end": span[1], "frame_start": span[0]},
            "mode": self.mode,
            "object_category": self.object_category,
            "policy_provenance": self.policy_provenance,
            "review_status": self.review_status,
            "schema_id": POLICY_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "source_frame_count": self.source_frame_count,
            "source_video_hash": self.source_video_hash,
            "unresolved_policy_assignments": list(self.unresolved_policy_assignments),
        }


@dataclass(frozen=True)
class TrackCensusEntry:
    track_id: int
    quality_status: str
    render_eligible: bool
    first_frame: int
    last_frame: int
    observed_frames: tuple[int, ...]
    weak_observation_frames: tuple[int, ...]
    interpolated_frames: tuple[int, ...]
    predicted_frames: tuple[int, ...]
    observed_hits: int
    strong_hits: int
    weak_hits: int
    median_observed_score: float | None
    quality_gate_failures: tuple[str, ...]
    representative_frame: int | None
    policy_assignment: str = POLICY_REVIEW
    assignment_provenance: str = PROVENANCE_POLICY_AUTOMATIC

    @property
    def frame_count(self) -> int:
        return self.last_frame - self.first_frame + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_provenance": self.assignment_provenance,
            "first_frame": self.first_frame,
            "fragment_count": 1,
            "interpolated_frames": list(self.interpolated_frames),
            "last_frame": self.last_frame,
            "median_observed_score": self.median_observed_score,
            "observed_frames": list(self.observed_frames),
            "observed_hits": self.observed_hits,
            "policy_assignment": self.policy_assignment,
            "predicted_frames": list(self.predicted_frames),
            "quality_gate_failures": list(self.quality_gate_failures),
            "quality_status": self.quality_status,
            "render_eligible": self.render_eligible,
            "representative_frame": self.representative_frame,
            "strong_hits": self.strong_hits,
            "track_id": self.track_id,
            "weak_hits": self.weak_hits,
            "weak_observation_frames": list(self.weak_observation_frames),
        }


@dataclass(frozen=True)
class PolicyCompilerSettings:
    recovery_settings: privacy_gap_recovery.PrivacyGapRecoverySettings = field(
        default_factory=lambda: privacy_gap_recovery.PrivacyGapRecoverySettings(
            render_overlay=False,
            preserve_existing_regions=False,
        )
    )
    include_review_candidate_tracks: bool = True
    include_rejected_tracks: bool = True


@dataclass(frozen=True)
class PolicyCompileResult:
    privacy_regions_by_frame: dict[int, list[privacy_gap_recovery.PrivacyRegion]]
    interval_decisions: list[privacy_gap_recovery.IntervalDecision]
    redacted_track_ids: tuple[int, ...]
    keep_track_ids: tuple[int, ...]
    review_track_ids: tuple[int, ...]
    unresolved_policy_assignments: tuple[int, ...]
    processing_fps: float


def build_track_census(
    *,
    tracks_path: Path,
    track_quality_path: Path,
    source_video_hash: str,
    source_frame_count: int,
    policy: FacePrivacyPolicy | None = None,
) -> dict[str, object]:
    records = privacy_gap_recovery.load_track_records(tracks_path)
    quality_by_id = load_track_quality(track_quality_path)
    track_frames: dict[int, list[privacy_gap_recovery.TrackSourceBox]] = {}
    for boxes in records.values():
        for box in boxes:
            track_frames.setdefault(box.track_id, []).append(box)

    entries: list[TrackCensusEntry] = []
    for track_id in sorted(set(track_frames) | set(quality_by_id)):
        boxes = sorted(
            track_frames.get(track_id, []),
            key=lambda item: (item.frame_index, item.source_provenance),
        )
        quality = quality_by_id.get(track_id, {})
        if boxes:
            first_frame = boxes[0].frame_index
            last_frame = boxes[-1].frame_index
        else:
            first_frame = int(quality.get("first_frame", 0))
            last_frame = int(quality.get("last_frame", first_frame))
        assignment = policy.assignment_for(track_id) if policy is not None else None
        entries.append(
            TrackCensusEntry(
                track_id=track_id,
                quality_status=str(
                    quality.get("quality_status") or _box_quality(boxes) or REVIEW_CANDIDATE
                ),
                render_eligible=bool(quality.get("render_eligible", False)),
                first_frame=first_frame,
                last_frame=last_frame,
                observed_frames=tuple(
                    sorted(
                        {
                            box.frame_index
                            for box in boxes
                            if box.source_provenance.startswith("observed")
                        }
                    )
                ),
                weak_observation_frames=tuple(
                    sorted(
                        {
                            box.frame_index
                            for box in boxes
                            if box.source_provenance == "observed_weak"
                        }
                    )
                ),
                interpolated_frames=tuple(
                    sorted(
                        {
                            box.frame_index
                            for box in boxes
                            if box.source_provenance == "interpolated"
                        }
                    )
                ),
                predicted_frames=tuple(
                    sorted(
                        {box.frame_index for box in boxes if box.source_provenance == "predicted"}
                    )
                ),
                observed_hits=int(quality.get("observed_hits", 0)),
                strong_hits=int(quality.get("strong_hits", 0)),
                weak_hits=int(quality.get("weak_hits", 0)),
                median_observed_score=quality.get("median_observed_score"),
                quality_gate_failures=tuple(
                    str(value) for value in quality.get("quality_gate_failures", [])
                ),
                representative_frame=representative_frame(boxes),
                policy_assignment=assignment.decision if assignment else POLICY_REVIEW,
                assignment_provenance=assignment.provenance
                if assignment
                else PROVENANCE_POLICY_AUTOMATIC,
            )
        )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.quality_status] = counts.get(entry.quality_status, 0) + 1
    return {
        "case_policy_mode": policy.mode if policy else None,
        "schema_id": TRACK_CENSUS_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": source_frame_count,
        "source_video_hash": source_video_hash,
        "summary": {
            "auto_confirmed_tracks": counts.get(AUTO_CONFIRMED, 0),
            "rejected_tracks": counts.get(REJECTED, 0),
            "review_candidate_tracks": counts.get(REVIEW_CANDIDATE, 0),
            "total_tracks": len(entries),
        },
        "tracks": [entry.to_dict() for entry in entries],
    }


def policy_from_census(
    *,
    case_id: str,
    source_video_hash: str,
    source_frame_count: int,
    census: dict[str, object],
    mode: str = MODE_REDACT_ALL,
    keep_track_ids: tuple[int, ...] = (),
    selected_track_ids: tuple[int, ...] = (),
) -> FacePrivacyPolicy:
    if mode not in SUPPORTED_POLICY_MODES:
        raise ValueError(f"unsupported policy mode: {mode}")
    keep_ids = set(keep_track_ids)
    selected_ids = set(selected_track_ids)
    assignments: list[PolicyAssignment] = []
    unresolved: list[int] = []
    for track in census.get("tracks", []):
        track_id = int(track["track_id"])
        quality_status = str(track["quality_status"])
        reasons = tuple(str(value) for value in track.get("quality_gate_failures", []))
        if mode == MODE_REDACT_ALL:
            decision = POLICY_REVIEW if quality_status == REJECTED else POLICY_REDACT
            provenance = PROVENANCE_POLICY_AUTOMATIC
            review_status = "required" if quality_status != AUTO_CONFIRMED else "not_required"
        elif mode == MODE_REDACT_ALL_EXCEPT_KEEP:
            if track_id in keep_ids:
                decision = POLICY_KEEP
                provenance = PROVENANCE_POLICY_EXPLICIT
                review_status = "not_required"
            else:
                decision = POLICY_REVIEW if quality_status == REJECTED else POLICY_REDACT
                provenance = PROVENANCE_POLICY_AUTOMATIC
                review_status = "required" if quality_status != AUTO_CONFIRMED else "not_required"
        elif mode == MODE_REDACT_SELECTED:
            decision = POLICY_REDACT if track_id in selected_ids else POLICY_IGNORE
            provenance = (
                PROVENANCE_POLICY_EXPLICIT
                if track_id in selected_ids
                else PROVENANCE_POLICY_AUTOMATIC
            )
            review_status = (
                "required"
                if decision == POLICY_REDACT and quality_status != AUTO_CONFIRMED
                else "not_required"
            )
        else:
            decision = POLICY_REVIEW
            provenance = PROVENANCE_POLICY_AUTOMATIC
            review_status = "required"
        if review_status == "required":
            unresolved.append(track_id)
        assignments.append(
            PolicyAssignment(
                track_id=track_id,
                decision=decision,
                provenance=provenance,
                review_status=review_status,
                reasons=((f"quality_status:{quality_status}",) + reasons),
            )
        )
    policy = FacePrivacyPolicy(
        case_id=case_id,
        source_video_hash=source_video_hash,
        source_frame_count=source_frame_count,
        mode=mode,
        assignments=tuple(assignments),
        review_status="required" if unresolved else "not_required",
        unresolved_policy_assignments=tuple(unresolved),
    )
    policy.validate()
    return policy


def load_policy(path: Path) -> FacePrivacyPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = tuple(
        PolicyAssignment(
            track_id=int(item["track_id"]),
            decision=str(item["decision"]),
            provenance=str(item.get("provenance") or PROVENANCE_POLICY_EXPLICIT),
            review_status=str(item.get("review_status") or "not_required"),
            identity_group_id=item.get("identity_group_id"),
            reasons=tuple(str(value) for value in item.get("reasons", [])),
        )
        for item in payload.get("assignments", [])
    )
    span = payload.get("full_source_frame_span") or {}
    policy = FacePrivacyPolicy(
        case_id=str(payload["case_id"]),
        source_video_hash=str(payload["source_video_hash"]),
        source_frame_count=int(payload["source_frame_count"]),
        mode=str(payload["mode"]),
        object_category=str(payload.get("object_category", "face")),
        assignments=assignments,
        full_source_frame_span=(
            int(span["frame_start"]),
            int(span["frame_end"]),
        )
        if span
        else None,
        policy_provenance=str(payload.get("policy_provenance") or PROVENANCE_POLICY_EXPLICIT),
        review_status=str(payload.get("review_status") or "not_required"),
        unresolved_policy_assignments=tuple(
            int(value) for value in payload.get("unresolved_policy_assignments", [])
        ),
    )
    policy.validate()
    return policy


def compile_policy(
    input_video: Path,
    *,
    tracks_path: Path,
    track_quality_path: Path,
    policy: FacePrivacyPolicy,
    settings: PolicyCompilerSettings | None = None,
) -> PolicyCompileResult:
    settings = settings or PolicyCompilerSettings()
    info = media.probe_media(input_video)
    if info.sha256 != policy.source_video_hash:
        raise ValueError("privacy policy source_video_hash does not match input video")
    frame_count = _source_frame_count(info)
    if frame_count != policy.source_frame_count:
        raise ValueError("privacy policy source_frame_count does not match input video")
    records = privacy_gap_recovery.load_track_records(tracks_path)
    quality_by_id = load_track_quality(track_quality_path)
    all_track_ids = tuple(sorted({int(item["track_id"]) for item in quality_by_id.values()}))
    redacted_track_ids = tuple(
        assignment.track_id
        for assignment in policy.assignments
        if assignment.decision == POLICY_REDACT
    )
    keep_track_ids = tuple(
        assignment.track_id
        for assignment in policy.assignments
        if assignment.decision == POLICY_KEEP
    )
    review_track_ids = tuple(
        assignment.track_id
        for assignment in policy.assignments
        if assignment.review_status == "required"
    )
    started_at = time.perf_counter()
    regions_by_frame: dict[int, list[privacy_gap_recovery.PrivacyRegion]] = {
        frame_index: [] for frame_index in range(frame_count)
    }
    decisions: list[privacy_gap_recovery.IntervalDecision] = []
    provider = privacy_gap_recovery.VideoFrameProvider(input_video)
    try:
        for track_id in redacted_track_ids:
            target_boxes = privacy_gap_recovery.select_track_boxes_by_frame(records, (track_id,))
            if not target_boxes:
                decisions.append(
                    policy_decision_for_track(track_id, 0, frame_count - 1, ["no_track_boxes"])
                )
                continue
            competing_ids = tuple(value for value in all_track_ids if value != track_id)
            recovery = privacy_gap_recovery.recover_privacy_gaps(
                frame_provider=provider,
                target_boxes=target_boxes,
                target_track_ids=(track_id,),
                keep_boxes_by_frame=privacy_gap_recovery.boxes_for_track_ids(
                    records, keep_track_ids
                ),
                competing_boxes_by_frame=privacy_gap_recovery.boxes_for_track_ids(
                    records, competing_ids
                ),
                preserved_regions={},
                frame_start=min(target_boxes),
                frame_end=max(target_boxes),
                frame_size=(info.video_stream.width, info.video_stream.height),
                settings=settings.recovery_settings,
                tracker_factory=privacy_gap_recovery.OpenCVMILTracker,
            )
            for frame_index, frame_regions in recovery.regions_by_frame.items():
                regions_by_frame.setdefault(frame_index, []).extend(frame_regions)
            decisions.extend(recovery.interval_decisions)
            quality = quality_by_id.get(track_id, {})
            quality_status = str(quality.get("quality_status") or AUTO_CONFIRMED)
            if quality_status != AUTO_CONFIRMED:
                decisions.append(
                    policy_decision_for_track(
                        track_id,
                        int(quality.get("first_frame", min(target_boxes))),
                        int(quality.get("last_frame", max(target_boxes))),
                        ["policy_track_quality_review", f"quality_status:{quality_status}"]
                        + [str(value) for value in quality.get("quality_gate_failures", [])],
                    )
                )
        for assignment in policy.assignments:
            if assignment.decision != POLICY_REVIEW:
                continue
            quality = quality_by_id.get(assignment.track_id, {})
            first_frame = int(quality.get("first_frame", 0))
            last_frame = int(quality.get("last_frame", first_frame))
            decisions.append(
                policy_decision_for_track(
                    assignment.track_id,
                    first_frame,
                    last_frame,
                    ["policy_assignment_review"] + [str(value) for value in assignment.reasons],
                )
            )
    finally:
        provider.release()
    elapsed = time.perf_counter() - started_at
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    compact_regions = {frame: regions for frame, regions in regions_by_frame.items() if regions}
    unresolved = tuple(sorted(set(policy.unresolved_policy_assignments) | set(review_track_ids)))
    return PolicyCompileResult(
        privacy_regions_by_frame=compact_regions,
        interval_decisions=sorted(
            decisions, key=lambda item: (item.frame_start, item.frame_end, item.target_track_ids)
        ),
        redacted_track_ids=redacted_track_ids,
        keep_track_ids=keep_track_ids,
        review_track_ids=review_track_ids,
        unresolved_policy_assignments=unresolved,
        processing_fps=fps,
    )


def write_compiled_outputs(
    output_dir: Path,
    *,
    result: PolicyCompileResult,
    policy: FacePrivacyPolicy,
    census: dict[str, object],
    source_video_hash: str,
    tracks_hash: str,
    track_quality_hash: str,
    overwrite: bool = False,
) -> dict[str, object]:
    ensure_output_dir(output_dir, overwrite=overwrite)
    privacy_gap_recovery.write_privacy_regions(
        output_dir / "privacy_regions.jsonl",
        result.privacy_regions_by_frame,
    )
    interval_payload = {
        "compiler_id": COMPILER_ID,
        "interval_decisions": [decision.to_dict() for decision in result.interval_decisions],
        "schema_version": SCHEMA_VERSION,
    }
    write_json(output_dir / "interval_decisions.json", interval_payload)
    summary = {
        "compiler_id": COMPILER_ID,
        "keep_track_count": len(result.keep_track_ids),
        "keep_track_ids": list(result.keep_track_ids),
        "policy_mode": policy.mode,
        "privacy_region_frame_count": len(result.privacy_regions_by_frame),
        "privacy_region_count": sum(
            len(regions) for regions in result.privacy_regions_by_frame.values()
        ),
        "processing_fps": round(result.processing_fps, 6),
        "redact_track_count": len(result.redacted_track_ids),
        "redact_track_ids": list(result.redacted_track_ids),
        "review_required_interval_count": sum(
            1
            for decision in result.interval_decisions
            if decision.status == privacy_gap_recovery.DECISION_NEEDS_REVIEW
        ),
        "review_track_count": len(result.review_track_ids),
        "review_track_ids": list(result.review_track_ids),
        "schema_version": SCHEMA_VERSION,
        "source_video_hash": source_video_hash,
        "track_census_summary": census.get("summary", {}),
        "track_quality_sha256": track_quality_hash,
        "tracks_sha256": tracks_hash,
        "unresolved_policy_assignments": list(result.unresolved_policy_assignments),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_full_video_policy_flow(
    input_video: Path,
    *,
    output_dir: Path,
    detections_path: Path,
    mode: str = MODE_REDACT_ALL,
    policy_path: Path | None = None,
    keep_track_ids: tuple[int, ...] = (),
    selected_track_ids: tuple[int, ...] = (),
    overwrite: bool = False,
) -> dict[str, object]:
    ensure_case_output_dir(output_dir, overwrite=overwrite)
    source_info = media.probe_media(input_video)
    source_frame_count = _source_frame_count(source_info)
    detection_manifest_dir = output_dir / "detection"
    detection_manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        detection_manifest_dir / "input_detections_manifest.json",
        {
            "detections_path": str(detections_path),
            "detections_sha256": media.sha256_file(detections_path),
            "detector_id": DETECTOR_BACKEND_ADAS,
            "note": (
                "Face policy compilation uses the supplied ADAS detections as detector evidence."
            ),
            "schema_version": SCHEMA_VERSION,
        },
    )
    tracking_dir = output_dir / "tracking"
    track_summary = tracking.run_tracking(
        input_video,
        detections_path=detections_path,
        output_dir=tracking_dir,
        overwrite=overwrite,
    )
    source_manifest = {
        "case_input_video": input_video.name,
        "detector_id": DETECTOR_BACKEND_ADAS,
        "fps": rational_to_float(source_info.video_stream.nominal_frame_rate),
        "height": source_info.video_stream.height,
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": source_frame_count,
        "source_video_hash": source_info.sha256,
        "width": source_info.video_stream.width,
    }
    write_json(output_dir / "source_manifest.json", source_manifest)

    draft_census = build_track_census(
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        source_video_hash=source_info.sha256,
        source_frame_count=source_frame_count,
    )
    if policy_path is not None:
        policy = load_policy(policy_path)
    else:
        policy = policy_from_census(
            case_id=output_dir.name,
            source_video_hash=source_info.sha256,
            source_frame_count=source_frame_count,
            census=draft_census,
            mode=mode,
            keep_track_ids=keep_track_ids,
            selected_track_ids=selected_track_ids,
        )
    write_json(output_dir / "privacy_policy.json", policy.to_dict())
    census = build_track_census(
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        source_video_hash=source_info.sha256,
        source_frame_count=source_frame_count,
        policy=policy,
    )
    write_json(output_dir / "track_census.json", census)
    write_track_gallery(output_dir / "track_gallery", census)

    compile_result = compile_policy(
        input_video,
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        policy=policy,
    )
    privacy_dir = output_dir / "privacy_regions"
    privacy_summary = write_compiled_outputs(
        privacy_dir,
        result=compile_result,
        policy=policy,
        census=census,
        source_video_hash=source_info.sha256,
        tracks_hash=media.sha256_file(tracking_dir / "tracks.jsonl"),
        track_quality_hash=media.sha256_file(tracking_dir / "track_quality.json"),
        overwrite=overwrite,
    )
    preview_dir = output_dir / "automatic_preview"
    render_summary = rendering.run_face_render(
        input_video,
        privacy_regions_path=privacy_dir / "privacy_regions.jsonl",
        interval_decisions_path=privacy_dir / "interval_decisions.json",
        output_dir=preview_dir,
        settings=DEFAULT_PREVIEW_RENDER_SETTINGS,
        overwrite=overwrite,
    )
    review_summary = build_policy_review_bundle(
        input_video,
        output_dir=output_dir / "review",
        detections_path=detections_path,
        tracks_path=tracking_dir / "tracks.jsonl",
        track_quality_path=tracking_dir / "track_quality.json",
        track_review_flags_path=tracking_dir / "review_flags.json",
        track_review_windows_path=tracking_dir / "review_windows.json",
        privacy_regions_path=privacy_dir / "privacy_regions.jsonl",
        interval_decisions_path=privacy_dir / "interval_decisions.json",
        render_manifest_path=preview_dir / "render_manifest.json",
        render_validation_path=preview_dir / "validation.json",
        automatic_preview_video_path=preview_dir / "redacted_preview.mp4",
        case_id=output_dir.name,
        overwrite=overwrite,
    )
    final_dir = output_dir / "final"
    final_status = write_or_run_final_render(
        input_video,
        output_dir=final_dir,
        privacy_regions_path=privacy_dir / "privacy_regions.jsonl",
        interval_decisions_path=privacy_dir / "interval_decisions.json",
        policy=policy,
        privacy_summary=privacy_summary,
        overwrite=overwrite,
    )
    audit = {
        "compiler_id": COMPILER_ID,
        "detection_input_sha256": media.sha256_file(detections_path),
        "final_export_status": final_status["status"],
        "policy": policy.to_dict(),
        "review_summary": review_summary.get("summary", review_summary),
        "privacy_summary": privacy_summary,
        "render_summary": render_summary,
        "schema_version": SCHEMA_VERSION,
        "source_manifest": source_manifest,
        "tracking_summary": track_summary,
    }
    write_json(output_dir / "audit_manifest.json", audit)
    return audit


def build_policy_review_bundle(
    input_video: Path,
    *,
    output_dir: Path,
    detections_path: Path,
    tracks_path: Path,
    track_quality_path: Path,
    track_review_flags_path: Path,
    track_review_windows_path: Path,
    privacy_regions_path: Path,
    interval_decisions_path: Path,
    render_manifest_path: Path,
    render_validation_path: Path,
    automatic_preview_video_path: Path,
    case_id: str,
    overwrite: bool,
) -> dict[str, object]:
    from redactor import review_bundle

    return review_bundle.build_review_bundle(
        source_video_path=input_video,
        output_dir=output_dir,
        config=review_bundle.ReviewBundleConfig(case_id=case_id),
        detections_path=detections_path,
        tracks_path=tracks_path,
        track_quality_path=track_quality_path,
        track_review_flags_path=track_review_flags_path,
        track_review_windows_path=track_review_windows_path,
        privacy_regions_path=privacy_regions_path,
        interval_decisions_path=interval_decisions_path,
        render_manifest_path=render_manifest_path,
        render_validation_path=render_validation_path,
        automatic_preview_video_path=automatic_preview_video_path,
        overwrite=overwrite,
    )


def write_or_run_final_render(
    input_video: Path,
    *,
    output_dir: Path,
    privacy_regions_path: Path,
    interval_decisions_path: Path,
    policy: FacePrivacyPolicy,
    privacy_summary: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    ensure_output_dir(output_dir, overwrite=overwrite)
    blocked = (
        bool(privacy_summary.get("review_required_interval_count"))
        or policy.review_status == "required"
    )
    if blocked:
        status = {
            "blocking_reasons": ["review_required_policy_or_track_intervals"],
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_review_required",
        }
        write_json(output_dir / "final_status.json", status)
        return status
    try:
        summary = rendering.run_face_render(
            input_video,
            privacy_regions_path=privacy_regions_path,
            interval_decisions_path=interval_decisions_path,
            output_dir=output_dir,
            settings=rendering.RenderSettings(
                mode=rendering.MODE_FINAL,
                method=DEFAULT_PREVIEW_RENDER_SETTINGS.method,
                margin_pixels=DEFAULT_PREVIEW_RENDER_SETTINGS.margin_pixels,
                margin_fraction=DEFAULT_PREVIEW_RENDER_SETTINGS.margin_fraction,
                pixel_size=DEFAULT_PREVIEW_RENDER_SETTINGS.pixel_size,
            ),
            overwrite=overwrite,
        )
    except ValueError as exc:
        status = {
            "blocking_reasons": [str(exc)],
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_renderer_validation",
        }
        write_json(output_dir / "final_status.json", status)
        return status
    status = {
        "render_summary": summary,
        "schema_version": SCHEMA_VERSION,
        "status": "final_rendered",
    }
    write_json(output_dir / "final_status.json", status)
    return status


def policy_decision_for_track(
    track_id: int,
    frame_start: int,
    frame_end: int,
    reasons: list[str],
) -> privacy_gap_recovery.IntervalDecision:
    return privacy_gap_recovery.IntervalDecision(
        frame_start=frame_start,
        frame_end=frame_end,
        target_track_ids=(track_id,),
        strategy="policy_track_review",
        status=privacy_gap_recovery.DECISION_NEEDS_REVIEW,
        reasons=reasons,
        metrics={"track_id": track_id},
        regions=[],
    )


def load_track_quality(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["track_id"]): dict(item) for item in payload.get("tracks", [])}


def write_track_gallery(output_dir: Path, census: dict[str, object]) -> None:
    ensure_output_dir(output_dir, overwrite=True)
    write_json(output_dir / "track_index.json", census)
    rows = []
    for track in census.get("tracks", []):
        rows.append(
            "<tr>"
            f"<td>{track['track_id']}</td>"
            f"<td>{track['policy_assignment']}</td>"
            f"<td>{track['quality_status']}</td>"
            f"<td>{track['first_frame']}-{track['last_frame']}</td>"
            f"<td>{track['observed_hits']}</td>"
            f"<td>{', '.join(str(value) for value in track.get('quality_gate_failures', []))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Track Census</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee8; padding: 8px; text-align: left; }}
    th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  <h1>Face Track Census</h1>
  <p>
    Default production policy redacts all qualifying face tracks unless a case policy
    explicitly marks a track as keep.
  </p>
  <table>
    <thead>
      <tr>
        <th>Track</th><th>Policy</th><th>Quality</th>
        <th>Frames</th><th>Observed</th><th>Review reasons</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def validate_policy_file(path: Path, *, source_video: Path | None = None) -> dict[str, object]:
    policy = load_policy(path)
    errors: list[str] = []
    if source_video is not None:
        info = media.probe_media(source_video)
        if info.sha256 != policy.source_video_hash:
            errors.append("source_video_hash_mismatch")
        if _source_frame_count(info) != policy.source_frame_count:
            errors.append("source_frame_count_mismatch")
    return {
        "errors": errors,
        "mode": policy.mode,
        "schema_id": POLICY_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if not errors else "invalid",
        "track_assignment_count": len(policy.assignments),
    }


def _box_quality(boxes: list[privacy_gap_recovery.TrackSourceBox]) -> str | None:
    qualities = [box.quality_status for box in boxes if box.quality_status]
    return str(qualities[0]) if qualities else None


def representative_frame(boxes: list[privacy_gap_recovery.TrackSourceBox]) -> int | None:
    observed = [box.frame_index for box in boxes if box.source_provenance.startswith("observed")]
    if observed:
        return observed[len(observed) // 2]
    frames = [box.frame_index for box in boxes]
    return frames[len(frames) // 2] if frames else None


def _source_frame_count(info: media.MediaInfo) -> int:
    value = info.video_stream.frame_count
    if value is None:
        raise ValueError("source video frame count is required for full-video policy")
    return int(value)


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_case_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        existing_names = {path.name for path in output_dir.iterdir()}
        unexpected = sorted(existing_names - {"detection"})
        if unexpected:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
