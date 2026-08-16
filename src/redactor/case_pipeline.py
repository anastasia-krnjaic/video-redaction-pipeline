from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from redactor import (
    case_ids,
    case_manifests,
    detection,
    face_policy,
    face_rendering,
    media,
    mixed_review_workspace,
    plate_detection,
    plate_rendering,
    plate_tracking,
    privacy_profiles,
    privacy_rendering,
    rendering,
)
from redactor.detection import rational_to_float
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    APPROVAL_MANUAL,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
    stable_hash,
)

SCHEMA_VERSION = "case_pipeline.v1"
CASE_MANIFEST_SCHEMA_VERSION = "unified_case_manifest.v1"
RENDER_PLAN_SCHEMA_VERSION = "unified_render_plan.v1"
RESUMABLE_STAGE_MANIFEST_SCHEMA_VERSION = "case_pipeline_resumable_stage_manifest.v1"

STAGE_INGEST = "INGEST"
STAGE_DETECT = "DETECT"
STAGE_TEMPORAL_ASSOCIATION = "TEMPORAL_ASSOCIATION"
STAGE_BOUNDED_RECOVERY = "BOUNDED_RECOVERY"
STAGE_POLICY = "POLICY"
STAGE_PRIVACY_REGIONS = "PRIVACY_REGIONS"
STAGE_REVIEW_TASKS = "REVIEW_TASKS"
STAGE_AUTOMATIC_PREVIEW = "AUTOMATIC_PREVIEW"
STAGE_REVIEW_WORKSPACE = "REVIEW_WORKSPACE"
STAGE_REVIEW_SESSION = "REVIEW_SESSION"
STAGE_LATEST_REVIEWED = "LATEST_REVIEWED"
STAGE_FINALIZATION = "FINALIZATION"
STAGE_EVALUATION = "EVALUATION"

STATUS_SUCCEEDED = "succeeded"
STATUS_READY = "ready"
STATUS_NOT_STARTED = "not_started"
STATUS_BLOCKED = "blocked"
STATUS_SKIPPED = "skipped"
STATUS_NOT_RUN = "not_run"
STATUS_FAILED = "failed"

CASE_PHASE_AUTOMATIC_PREVIEW_READY = "automatic_preview_ready"
CASE_PHASE_AUTOMATIC_PREVIEW_BLOCKED = "automatic_preview_blocked"
CASE_PHASE_AUTOMATIC_PREVIEW_FAILED = "automatic_preview_failed"

APPROVED_RENDER_STATES = {APPROVAL_ACCEPTED, APPROVAL_MANUAL}

VALID_CATEGORIES = (CATEGORY_FACE, CATEGORY_PLATE)
CATEGORY_ALIASES = {
    "all": "all",
    "face": CATEGORY_FACE,
    "faces": CATEGORY_FACE,
    "person": CATEGORY_FACE,
    "people": CATEGORY_FACE,
    "license-plate": CATEGORY_PLATE,
    "license_plate": CATEGORY_PLATE,
    "license-plates": CATEGORY_PLATE,
    "license_plates": CATEGORY_PLATE,
    "plate": CATEGORY_PLATE,
    "plates": CATEGORY_PLATE,
}


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    sha256: str
    frame_count: int | None
    width: int
    height: int
    fps: str | None
    duration_seconds: float | None
    has_audio: bool
    schema_version: str = "source_identity.v1"

    @classmethod
    def from_video(cls, path: Path, *, info: media.MediaInfo) -> SourceIdentity:
        return cls(
            duration_seconds=info.duration_seconds,
            fps=info.video_stream.nominal_frame_rate,
            frame_count=_optional_int(info.video_stream.frame_count),
            has_audio=info.audio_stream is not None,
            height=info.video_stream.height,
            path=media.portable_source_path(path),
            sha256=info.sha256,
            width=info.video_stream.width,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "has_audio": self.has_audio,
            "height": self.height,
            "path": self.path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "width": self.width,
        }


ArtifactReference = case_manifests.ArtifactReference


@dataclass(frozen=True)
class StageStatus:
    stage: str
    status: str
    category: str | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    reason: str | None = None
    schema_version: str = "case_stage_status.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "blockers": list(self.blockers),
            "category": self.category,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "status": self.status,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CaseContext:
    case_id: str
    source_video: Path
    output_dir: Path
    source_info: media.MediaInfo
    source_identity: SourceIdentity
    selected_categories: tuple[str, ...]


@dataclass(frozen=True)
class CasePipelineOptions:
    face_policy_mode: str = face_policy.MODE_REDACT_ALL
    face_policy_path: Path | None = None
    keep_track_ids: tuple[int, ...] = ()
    selected_track_ids: tuple[int, ...] = ()
    face_detections_path: Path | None = None
    face_model_path: Path | None = None
    adas_xml_path: Path | None = None
    adas_bin_path: Path | None = None
    plate_model_path: Path | None = None
    proposal_threshold: float = detection.DEFAULT_ADAS_PROPOSAL_THRESHOLD
    primary_threshold: float = detection.DEFAULT_ADAS_PRIMARY_THRESHOLD
    no_detection_overlay: bool = False
    allow_partial_plate_preview: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class CategoryRunResult:
    category: str
    stages: tuple[StageStatus, ...]
    privacy_regions: tuple[PrivacyRegionRecord, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    source_lineage: dict[str, Any] = field(default_factory=dict)
    configuration_identity: dict[str, Any] = field(default_factory=dict)


class CategoryAdapter(Protocol):
    category: str

    def run(self, context: CaseContext, options: CasePipelineOptions) -> CategoryRunResult: ...


class FaceCategoryAdapter:
    category = CATEGORY_FACE

    def run(self, context: CaseContext, options: CasePipelineOptions) -> CategoryRunResult:
        resolved_detections_path = options.face_detections_path
        generated_detections = resolved_detections_path is None
        stages: list[StageStatus] = []
        artifacts: list[ArtifactReference] = []
        started_at = time.perf_counter()
        if resolved_detections_path is None:
            detection_dir = context.output_dir / "detection"
            backend = detection.resolve_detector_backend(CATEGORY_FACE, None)
            model_path = resolve_face_model_path(options)
            adas_bin_path = options.adas_bin_path or detection.default_adas_bin_path(model_path)
            detection_summary = detection.run_detection(
                context.source_video,
                category=CATEGORY_FACE,
                model_path=model_path,
                threshold=options.proposal_threshold,
                output_dir=detection_dir,
                overwrite=options.overwrite,
                detector_backend=backend,
                adas_bin_path=adas_bin_path,
                primary_threshold=options.primary_threshold,
                write_overlay_video=not options.no_detection_overlay,
            )
            resolved_detections_path = detection_dir / "detections.jsonl"
            stages.append(
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_DETECT,
                    status=STATUS_SUCCEEDED,
                    artifacts=(
                        artifact_reference(
                            context.output_dir,
                            resolved_detections_path,
                            role="direct_observations",
                            category=CATEGORY_FACE,
                        ),
                    ),
                    summary={
                        "detector_backend": detection_summary["detector_backend"],
                        "frames_processed": detection_summary["frames_processed"],
                        "total_detections": detection_summary["total_detections"],
                    },
                )
            )
            artifacts.extend(stages[-1].artifacts)
        else:
            stages.append(
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_DETECT,
                    status=STATUS_SUCCEEDED,
                    artifacts=(
                        artifact_reference(
                            context.output_dir,
                            resolved_detections_path,
                            role="direct_observations",
                            category=CATEGORY_FACE,
                            allow_external=True,
                        ),
                    ),
                    summary={"reused_input": True},
                )
            )
            artifacts.extend(stages[-1].artifacts)

        audit = face_policy.run_full_video_policy_flow(
            context.source_video,
            output_dir=context.output_dir,
            detections_path=resolved_detections_path,
            mode=options.face_policy_mode,
            policy_path=options.face_policy_path,
            keep_track_ids=options.keep_track_ids,
            selected_track_ids=options.selected_track_ids,
            overwrite=options.overwrite or generated_detections,
        )
        tracking_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "tracking" / "tracks.jsonl",
            role="temporal_tracks",
            category=CATEGORY_FACE,
        )
        track_quality_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "tracking" / "track_quality.json",
            role="track_quality",
            category=CATEGORY_FACE,
        )
        privacy_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "privacy_regions" / "privacy_regions.jsonl",
            role="automatic_privacy_regions",
            category=CATEGORY_FACE,
        )
        interval_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "privacy_regions" / "interval_decisions.json",
            role="privacy_interval_decisions",
            category=CATEGORY_FACE,
        )
        review_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "review" / "review_tasks.json",
            role="review_tasks",
            category=CATEGORY_FACE,
        )
        automatic_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "automatic_preview" / "redacted_preview.mp4",
            role="automatic_preview",
            category=CATEGORY_FACE,
        )
        render_manifest_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "automatic_preview" / "render_manifest.json",
            role="render_manifest",
            category=CATEGORY_FACE,
        )
        render_validation_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "automatic_preview" / "validation.json",
            role="render_validation",
            category=CATEGORY_FACE,
        )
        final_status_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "final" / "final_status.json",
            role="legacy_face_final_status",
            category=CATEGORY_FACE,
        )
        source_manifest_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "source_manifest.json",
            role="source_lineage",
            category=CATEGORY_FACE,
        )
        policy_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "privacy_policy.json",
            role="privacy_policy",
            category=CATEGORY_FACE,
        )
        census_artifact = artifact_reference(
            context.output_dir,
            context.output_dir / "track_census.json",
            role="track_census",
            category=CATEGORY_FACE,
        )
        final_status_path = context.output_dir / "final" / "final_status.json"
        final_status = json.loads(final_status_path.read_text(encoding="utf-8"))
        if not isinstance(final_status, dict):
            final_status = {}
        legacy_final_blockers = tuple(
            str(item) for item in final_status.get("blocking_reasons", [])
        )
        stages.extend(
            [
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_TEMPORAL_ASSOCIATION,
                    status=STATUS_SUCCEEDED,
                    artifacts=(tracking_artifact, track_quality_artifact),
                    summary=audit.get("tracking_summary", {}),
                ),
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_POLICY,
                    status=STATUS_SUCCEEDED,
                    artifacts=(policy_artifact, census_artifact),
                    summary={"policy_mode": options.face_policy_mode},
                ),
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_PRIVACY_REGIONS,
                    status=STATUS_SUCCEEDED,
                    artifacts=(privacy_artifact, interval_artifact),
                    summary=audit.get("privacy_summary", {}),
                ),
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_REVIEW_TASKS,
                    status=STATUS_SUCCEEDED,
                    artifacts=(review_artifact,),
                    summary=audit.get("review_summary", {}),
                    blockers=legacy_final_blockers,
                ),
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_AUTOMATIC_PREVIEW,
                    status=STATUS_SUCCEEDED,
                    artifacts=(
                        automatic_artifact,
                        render_manifest_artifact,
                        render_validation_artifact,
                    ),
                    summary=audit.get("render_summary", {}),
                ),
                StageStatus(
                    category=CATEGORY_FACE,
                    stage=STAGE_FINALIZATION,
                    status=STATUS_NOT_RUN,
                    summary={
                        "legacy_face_final_status": audit.get("final_export_status"),
                        "reason": (
                            "case-level finalization is deferred for this "
                            "automatic-preview checkpoint"
                        ),
                    },
                    reason="case_level_finalization_deferred",
                ),
            ]
        )
        artifacts.extend(
            (
                source_manifest_artifact,
                tracking_artifact,
                track_quality_artifact,
                policy_artifact,
                census_artifact,
                privacy_artifact,
                interval_artifact,
                review_artifact,
                automatic_artifact,
                render_manifest_artifact,
                render_validation_artifact,
                final_status_artifact,
                artifact_reference(
                    context.output_dir,
                    context.output_dir / "audit_manifest.json",
                    role="category_audit_manifest",
                    category=CATEGORY_FACE,
                ),
            )
        )
        privacy_records = face_rendering.face_privacy_region_records(
            context.output_dir / "privacy_regions" / "privacy_regions.jsonl"
        )
        blockers = tuple(
            sorted(
                {
                    blocker
                    for stage in stages
                    for blocker in stage.blockers
                    if stage.status == STATUS_BLOCKED
                }
            )
        )
        return CategoryRunResult(
            artifacts=tuple(artifacts),
            blockers=blockers,
            category=CATEGORY_FACE,
            privacy_regions=tuple(privacy_records),
            stages=tuple(stages),
            source_lineage=source_lineage_for_context(context),
            configuration_identity=category_configuration_identity(
                context,
                options,
                category=CATEGORY_FACE,
            ),
            summary={
                "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                "legacy_face_final_status": audit.get("final_export_status"),
                "privacy_summary": audit.get("privacy_summary", {}),
            },
        )


class PlateCategoryAdapter:
    category = CATEGORY_PLATE

    def run(self, context: CaseContext, options: CasePipelineOptions) -> CategoryRunResult:
        category_root = context.output_dir / CATEGORY_PLATE
        detection_dir = category_root / "detection"
        tracking_dir = category_root / "tracking"
        render_dir = category_root / "automatic_preview"
        review_dir = category_root / "review"
        stages: list[StageStatus] = []
        started_at = time.perf_counter()
        model_path = options.plate_model_path or plate_detection.default_model_path()
        detection_summary = detection.run_detection(
            context.source_video,
            category=CATEGORY_PLATE,
            model_path=model_path,
            threshold=plate_detection.DEFAULT_PROPOSAL_THRESHOLD,
            output_dir=detection_dir,
            overwrite=options.overwrite,
            detector_backend=detection.DETECTOR_BACKEND_OIM_YOLOV9_S_608,
            primary_threshold=plate_detection.DEFAULT_PRIMARY_THRESHOLD,
            write_overlay_video=not options.no_detection_overlay,
        )
        detections_path = detection_dir / "detections.jsonl"
        detection_artifact = artifact_reference(
            context.output_dir,
            detections_path,
            role="direct_observations",
            category=CATEGORY_PLATE,
        )
        stages.append(
            StageStatus(
                category=CATEGORY_PLATE,
                stage=STAGE_DETECT,
                status=STATUS_SUCCEEDED,
                artifacts=(detection_artifact,),
                summary={
                    "detector_backend": detection_summary["detector_backend"],
                    "frames_processed": detection_summary["frames_processed"],
                    "total_detections": detection_summary["total_detections"],
                },
            )
        )
        tracking_summary = plate_tracking.run_plate_tracking(
            context.source_video,
            detections_path=detections_path,
            output_dir=tracking_dir,
            overwrite=options.overwrite,
        )
        tracking_artifact = artifact_reference(
            context.output_dir,
            tracking_dir / "tracks.jsonl",
            role="temporal_tracks",
            category=CATEGORY_PLATE,
        )
        review_intervals_artifact = artifact_reference(
            context.output_dir,
            tracking_dir / "review_intervals.json",
            role="review_intervals",
            category=CATEGORY_PLATE,
        )
        gap_decisions_artifact = artifact_reference(
            context.output_dir,
            tracking_dir / "gap_decisions.json",
            role="recovery_gap_decisions",
            category=CATEGORY_PLATE,
        )
        frame_span_artifact = artifact_reference(
            context.output_dir,
            tracking_dir / "frame_span_report.json",
            role="processing_span_report",
            category=CATEGORY_PLATE,
        )
        track_summary_artifact = artifact_reference(
            context.output_dir,
            tracking_dir / "track_summary.json",
            role="track_summary",
            category=CATEGORY_PLATE,
        )
        stages.extend(
            [
                StageStatus(
                    category=CATEGORY_PLATE,
                    stage=STAGE_TEMPORAL_ASSOCIATION,
                    status=STATUS_SUCCEEDED,
                    artifacts=(tracking_artifact,),
                    summary=tracking_summary.get("tracking_summary", {}),
                ),
                StageStatus(
                    category=CATEGORY_PLATE,
                    stage=STAGE_BOUNDED_RECOVERY,
                    status=STATUS_SUCCEEDED,
                    artifacts=(gap_decisions_artifact, frame_span_artifact),
                    summary={
                        "review_interval_count": tracking_summary.get(
                            "tracking_summary", {}
                        ).get("review_interval_count"),
                        "full_source_processed": tracking_summary.get("full_source_processed"),
                    },
                ),
            ]
        )
        render_summary = plate_rendering.run_plate_render(
            context.source_video,
            tracks_path=tracking_dir / "tracks.jsonl",
            review_intervals_path=tracking_dir / "review_intervals.json",
            output_dir=render_dir,
            allow_partial_preview=options.allow_partial_plate_preview,
            overwrite=options.overwrite,
        )
        preview_artifact = artifact_reference(
            context.output_dir,
            render_dir / "automatic_preview.mp4",
            role="automatic_preview",
            category=CATEGORY_PLATE,
        )
        render_manifest_artifact = artifact_reference(
            context.output_dir,
            render_dir / "render_manifest.json",
            role="render_manifest",
            category=CATEGORY_PLATE,
        )
        render_validation_artifact = artifact_reference(
            context.output_dir,
            render_dir / "validation.json",
            role="render_validation",
            category=CATEGORY_PLATE,
        )
        render_summary_artifact = artifact_reference(
            context.output_dir,
            render_dir / "render_summary.json",
            role="render_summary",
            category=CATEGORY_PLATE,
        )
        stages.append(
            StageStatus(
                category=CATEGORY_PLATE,
                stage=STAGE_AUTOMATIC_PREVIEW,
                status=STATUS_SUCCEEDED,
                artifacts=(
                    preview_artifact,
                    render_manifest_artifact,
                    render_validation_artifact,
                    render_summary_artifact,
                ),
                summary=render_summary,
            )
        )
        review_summary = plate_rendering.build_plate_review_bundle(
            context.source_video,
            tracks_path=tracking_dir / "tracks.jsonl",
            review_intervals_path=tracking_dir / "review_intervals.json",
            automatic_preview_path=render_dir / "automatic_preview.mp4",
            diagnostic_overlay_path=tracking_dir / "tracking_overlay.mp4",
            output_dir=review_dir,
            overwrite=options.overwrite,
        )
        review_artifact = artifact_reference(
            context.output_dir,
            review_dir / "review_tasks.json",
            role="review_tasks",
            category=CATEGORY_PLATE,
        )
        stages.append(
            StageStatus(
                category=CATEGORY_PLATE,
                stage=STAGE_REVIEW_TASKS,
                status=STATUS_SUCCEEDED,
                artifacts=(review_artifact,),
                summary=review_summary,
                blockers=(
                    ("plate_review_intervals",)
                    if review_summary.get("final_export_blocked")
                    else ()
                ),
            )
        )
        stages.append(
            StageStatus(
                category=CATEGORY_PLATE,
                stage=STAGE_FINALIZATION,
                status=STATUS_NOT_RUN,
                summary={
                    "automatic_plate_preview_ready": True,
                    "reason": (
                        "case-level finalization is deferred for this "
                        "automatic-preview checkpoint"
                    ),
                },
                reason="case_level_finalization_deferred",
            )
        )
        records = plate_privacy_region_records(tracking_dir / "tracks.jsonl")
        artifacts = (
            detection_artifact,
            tracking_artifact,
            review_intervals_artifact,
            gap_decisions_artifact,
            frame_span_artifact,
            track_summary_artifact,
            preview_artifact,
            review_artifact,
            render_manifest_artifact,
            render_validation_artifact,
            render_summary_artifact,
            artifact_reference(
                context.output_dir,
                tracking_dir / "tracking_manifest.json",
                role="category_tracking_manifest",
                category=CATEGORY_PLATE,
            ),
        )
        blockers = tuple(
            sorted({blocker for stage in stages for blocker in stage.blockers})
        )
        return CategoryRunResult(
            artifacts=artifacts,
            blockers=blockers,
            category=CATEGORY_PLATE,
            privacy_regions=tuple(records),
            stages=tuple(stages),
            source_lineage=source_lineage_for_context(context),
            configuration_identity=category_configuration_identity(
                context,
                options,
                category=CATEGORY_PLATE,
            ),
            summary={
                "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                "render_summary": render_summary,
                "review_summary": review_summary,
            },
        )


def run_case_pipeline(
    input_video: Path,
    *,
    output_dir: Path,
    categories: tuple[str, ...],
    options: CasePipelineOptions | None = None,
    adapters: dict[str, CategoryAdapter] | None = None,
) -> dict[str, Any]:
    if not categories:
        raise ValueError("at least one privacy category must be selected")
    options = options or CasePipelineOptions()
    source_info = media.probe_media(input_video)
    source_identity = SourceIdentity.from_video(input_video, info=source_info)
    if not options.overwrite:
        reusable = reusable_case_pipeline_result(
            input_video=input_video,
            output_dir=output_dir,
            categories=categories,
            options=options,
            source_info=source_info,
            source_identity=source_identity,
        )
        if reusable is not None:
            return reusable
    ensure_case_output_dir(output_dir, overwrite=options.overwrite)
    case_id = case_ids.validate_case_id(output_dir.name)
    context = CaseContext(
        case_id=case_id,
        output_dir=output_dir,
        selected_categories=categories,
        source_identity=source_identity,
        source_info=source_info,
        source_video=input_video,
    )
    source_identity_path = output_dir / "case_source_identity.json"
    write_json(source_identity_path, source_identity.to_dict())
    source_identity_artifact = artifact_reference(
        output_dir,
        source_identity_path,
        role="source_identity",
    )
    registry: dict[str, CategoryAdapter] = {
        CATEGORY_FACE: FaceCategoryAdapter(),
        CATEGORY_PLATE: PlateCategoryAdapter(),
    }
    if adapters is not None:
        registry.update(adapters)
    stage_statuses: list[StageStatus] = [
        StageStatus(
            stage=STAGE_INGEST,
            status=STATUS_SUCCEEDED,
            artifacts=(source_identity_artifact,),
            summary=source_identity.to_dict(),
        )
    ]
    category_results: list[CategoryRunResult] = []
    category_failures: list[dict[str, Any]] = []
    for category in categories:
        adapter = registry.get(category)
        if adapter is None:
            raise ValueError(f"unsupported privacy category: {category}")
        try:
            result = adapter.run(context, options)
            result = bind_category_run_manifest(context, options, result)
        except Exception as exc:
            blocker = f"{category}_automatic_preview_failed"
            category_failures.append(
                {
                    "blocker": blocker,
                    "category": category,
                    "error": str(exc),
                }
            )
            stage_statuses.append(
                StageStatus(
                    category=category,
                    stage=STAGE_AUTOMATIC_PREVIEW,
                    status=STATUS_FAILED,
                    blockers=(blocker,),
                    reason="category_adapter_failed",
                    summary={"error": str(exc)},
                )
            )
            break
        category_results.append(result)
        stage_statuses.extend(result.stages)

    render_plan = write_render_plan(context, category_results)
    combined_preview = None
    render_plan_blockers = tuple(str(item) for item in render_plan.get("blockers", []))
    category_failure_blockers = tuple(item["blocker"] for item in category_failures)
    if category_failures:
        automatic_stage = StageStatus(
            stage=STAGE_AUTOMATIC_PREVIEW,
            status=STATUS_FAILED,
            artifacts=render_plan_artifacts(render_plan),
            blockers=category_failure_blockers,
            reason="requested_category_failed",
            summary={"case_phase": CASE_PHASE_AUTOMATIC_PREVIEW_FAILED},
        )
    elif render_plan_blockers:
        automatic_stage = StageStatus(
            stage=STAGE_AUTOMATIC_PREVIEW,
            status=STATUS_BLOCKED,
            artifacts=render_plan_artifacts(render_plan),
            blockers=render_plan_blockers,
            reason="render_plan_validation_blocked",
            summary={"case_phase": CASE_PHASE_AUTOMATIC_PREVIEW_BLOCKED},
        )
    elif len(categories) > 1 and render_plan["effective_region_count"] > 0:
        try:
            combined_preview = render_combined_automatic_preview(
                context,
                effective_regions_path=output_dir
                / render_plan["effective_regions_artifact"]["path"],
                overwrite=options.overwrite,
            )
        except Exception as exc:
            blocker = "combined_automatic_preview_failed"
            stage_statuses.append(
                StageStatus(
                    stage=STAGE_AUTOMATIC_PREVIEW,
                    status=STATUS_FAILED,
                    artifacts=render_plan_artifacts(render_plan),
                    blockers=(blocker,),
                    reason="combined_preview_render_failed",
                    summary={"error": str(exc)},
                )
            )
            write_unified_case_manifest(
                context,
                category_results=category_results,
                render_plan=render_plan,
                combined_preview=None,
                review_workspace=None,
                stage_statuses=deferred_case_stage_statuses(stage_statuses),
                case_phase=CASE_PHASE_AUTOMATIC_PREVIEW_FAILED,
                case_blockers=(blocker,),
                source_identity_artifact=source_identity_artifact,
                category_failures=category_failures,
            )
            raise RuntimeError(f"combined automatic preview failed: {exc}") from exc
        stage_statuses.append(
            StageStatus(
                stage=STAGE_AUTOMATIC_PREVIEW,
                status=STATUS_SUCCEEDED,
                artifacts=render_plan_artifacts(render_plan) + tuple(
                    combined_preview["artifacts"]
                ),
                reason="combined_automatic_preview_rendered",
                summary={
                    **combined_preview["summary"],
                    "case_phase": CASE_PHASE_AUTOMATIC_PREVIEW_READY,
                },
            )
        )
        automatic_stage = None
    else:
        automatic_stage = StageStatus(
            stage=STAGE_AUTOMATIC_PREVIEW,
            status=STATUS_SUCCEEDED,
            artifacts=render_plan_artifacts(render_plan),
            reason="category_automatic_preview_ready",
            summary={
                "case_phase": CASE_PHASE_AUTOMATIC_PREVIEW_READY,
                "effective_region_count": render_plan["effective_region_count"],
            },
        )
    if automatic_stage is not None:
        stage_statuses.append(automatic_stage)

    case_blockers = tuple(
        sorted(
            {
                blocker
                for result in category_results
                for blocker in result.blockers
            }
            | set(render_plan_blockers)
            | set(category_failure_blockers)
        )
    )
    if category_failures:
        case_phase = CASE_PHASE_AUTOMATIC_PREVIEW_FAILED
    elif render_plan_blockers:
        case_phase = CASE_PHASE_AUTOMATIC_PREVIEW_BLOCKED
    else:
        case_phase = CASE_PHASE_AUTOMATIC_PREVIEW_READY
    review_workspace_result = None
    if category_failures:
        stage_statuses.append(
            StageStatus(
                stage=STAGE_REVIEW_WORKSPACE,
                status=STATUS_FAILED,
                blockers=category_failure_blockers,
                reason="requested_category_failed",
                summary={"review_workspace": "not_created"},
            )
        )
    elif render_plan_blockers:
        stage_statuses.append(
            StageStatus(
                stage=STAGE_REVIEW_WORKSPACE,
                status=STATUS_BLOCKED,
                blockers=render_plan_blockers,
                reason="render_plan_validation_blocked",
                summary={"review_workspace": "not_created"},
            )
        )
    else:
        try:
            review_workspace_result = mixed_review_workspace.build_case_review_workspace(
                context,
                category_results=category_results,
                render_plan=render_plan,
                combined_preview=combined_preview,
                case_blockers=case_blockers,
                overwrite=options.overwrite,
            )
        except Exception as exc:
            blocker = "review_workspace_generation_failed"
            stage_statuses.append(
                StageStatus(
                    stage=STAGE_REVIEW_WORKSPACE,
                    status=STATUS_FAILED,
                    blockers=(blocker,),
                    reason="review_workspace_generation_failed",
                    summary={"error": str(exc)},
                )
            )
            stage_statuses = deferred_case_stage_statuses(stage_statuses)
            write_unified_case_manifest(
                context,
                category_results=category_results,
                render_plan=render_plan,
                combined_preview=combined_preview,
                review_workspace=None,
                stage_statuses=stage_statuses,
                case_phase=case_phase,
                case_blockers=tuple(sorted((*case_blockers, blocker))),
                source_identity_artifact=source_identity_artifact,
                category_failures=category_failures,
            )
            raise RuntimeError(f"review workspace generation failed: {exc}") from exc
        stage_statuses.append(
            StageStatus(
                stage=STAGE_REVIEW_WORKSPACE,
                status=STATUS_READY,
                artifacts=tuple(review_workspace_result["artifacts"]),
                reason="review_workspace_ready",
                summary=review_workspace_result["summary"],
            )
        )
        stage_statuses.append(
            StageStatus(
                stage=STAGE_REVIEW_SESSION,
                status=STATUS_NOT_STARTED,
                artifacts=tuple(
                    artifact
                    for artifact in review_workspace_result["artifacts"]
                    if artifact.role == "mixed_review_session_state"
                ),
                reason="review_session_not_started",
                summary={
                    "history_namespace": "review_workspace/sessions",
                    "reason": "review sessions are created only when a reviewer starts review",
                },
            )
        )
    stage_statuses = deferred_case_stage_statuses(stage_statuses)
    manifest = write_unified_case_manifest(
        context,
        category_results=category_results,
        render_plan=render_plan,
        combined_preview=combined_preview,
        review_workspace=review_workspace_result,
        stage_statuses=stage_statuses,
        case_phase=case_phase,
        case_blockers=case_blockers,
        source_identity_artifact=source_identity_artifact,
        category_failures=category_failures,
    )
    resumable_manifest = write_case_pipeline_resumable_stage_manifest(
        context=context,
        options=options,
        categories=categories,
        case_manifest=manifest,
    )
    if category_failures:
        details = "; ".join(
            f"{item['category']}: {item['error']}" for item in category_failures
        )
        raise RuntimeError(f"case automatic preview failed: {details}")
    return {
        "case_id": context.case_id,
        "categories": list(categories),
        "combined_preview": combined_preview,
        "manifest": manifest,
        "render_plan": render_plan,
        "resumable_stage_manifest": resumable_manifest,
        "review_workspace": review_workspace_result,
        "schema_version": SCHEMA_VERSION,
        "stage_statuses": [stage.to_dict() for stage in stage_statuses],
    }


def reusable_case_pipeline_result(
    *,
    input_video: Path,
    output_dir: Path,
    categories: tuple[str, ...],
    options: CasePipelineOptions,
    source_info: media.MediaInfo,
    source_identity: SourceIdentity,
) -> dict[str, Any] | None:
    if not output_dir.exists() or not output_dir.is_dir():
        return None
    case_id = case_ids.validate_case_id(output_dir.name)
    context = CaseContext(
        case_id=case_id,
        output_dir=output_dir,
        selected_categories=categories,
        source_identity=source_identity,
        source_info=source_info,
        source_video=input_video,
    )
    manifest_path = output_dir / "case_manifest.json"
    stage_manifest_path = output_dir / "case_pipeline_stage_manifest.json"
    if not manifest_path.exists() or not stage_manifest_path.exists():
        return None
    case_manifest = load_json_if_exists(manifest_path)
    decision = resumable_stage_reuse_decision(
        stage_manifest_path,
        expected_identity=case_pipeline_stage_identity(
            context=context,
            options=options,
            categories=categories,
            case_manifest=case_manifest,
        ),
        output_dir=output_dir,
    )
    if not decision["reusable"]:
        return None
    return {
        "case_id": case_id,
        "categories": list(categories),
        "combined_preview": None,
        "manifest": case_manifest,
        "render_plan": {},
        "resumable_reuse": True,
        "resumable_stage_manifest": decision["manifest"],
        "review_workspace": None,
        "schema_version": SCHEMA_VERSION,
        "stage_statuses": list(case_manifest.get("stages") or []),
    }


def write_case_pipeline_resumable_stage_manifest(
    *,
    context: CaseContext,
    options: CasePipelineOptions,
    categories: tuple[str, ...],
    case_manifest: dict[str, Any],
) -> dict[str, Any]:
    output_hashes = case_pipeline_output_hashes(context.output_dir, case_manifest)
    payload = {
        "case_id": context.case_id,
        "completed_at_utc": media.utc_timestamp(),
        "configuration_identity": case_pipeline_configuration_identity(context, options),
        "implementation_identity": case_pipeline_implementation_identity(),
        "input_hashes": {
            "source_video_sha256": context.source_identity.sha256,
        },
        "output_hashes": output_hashes,
        "schema_version": RESUMABLE_STAGE_MANIFEST_SCHEMA_VERSION,
        "source_identity": context.source_identity.to_dict(),
        "stage": "CASE_PIPELINE",
        "status": STATUS_SUCCEEDED,
    }
    payload["stage_identity"] = stable_hash(stage_identity_payload(payload))
    publish_resumable_stage_manifest(
        context.output_dir / "case_pipeline_stage_manifest.json",
        payload,
    )
    return payload


def resumable_stage_reuse_decision(
    manifest_path: Path,
    *,
    expected_identity: str,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        payload = load_json_if_exists(manifest_path)
    except json.JSONDecodeError as exc:
        return {
            "manifest": {},
            "reason": f"malformed resumable stage manifest: {exc}",
            "reusable": False,
        }
    if payload.get("schema_version") != RESUMABLE_STAGE_MANIFEST_SCHEMA_VERSION:
        return {"manifest": payload, "reason": "schema mismatch", "reusable": False}
    if payload.get("status") != STATUS_SUCCEEDED:
        return {"manifest": payload, "reason": "stage incomplete", "reusable": False}
    if payload.get("stage_identity") != expected_identity:
        return {"manifest": payload, "reason": "stage identity mismatch", "reusable": False}
    actual_hashes = case_pipeline_output_hashes(
        output_dir,
        load_json_if_exists(output_dir / "case_manifest.json"),
    )
    if dict(payload.get("output_hashes") or {}) != actual_hashes:
        return {"manifest": payload, "reason": "output hash mismatch", "reusable": False}
    return {"manifest": payload, "reason": "completed stage matches", "reusable": True}


def case_pipeline_stage_identity(
    *,
    context: CaseContext,
    options: CasePipelineOptions,
    categories: tuple[str, ...],
    case_manifest: dict[str, Any],
) -> str:
    payload = {
        "case_id": context.case_id,
        "configuration_identity": case_pipeline_configuration_identity(context, options),
        "implementation_identity": case_pipeline_implementation_identity(),
        "input_hashes": {"source_video_sha256": context.source_identity.sha256},
        "output_hashes": case_pipeline_output_hashes(context.output_dir, case_manifest),
        "selected_categories": list(categories),
        "source_identity": context.source_identity.to_dict(),
        "stage": "CASE_PIPELINE",
    }
    return stable_hash(payload)


def stage_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": payload.get("case_id"),
        "configuration_identity": payload.get("configuration_identity"),
        "implementation_identity": payload.get("implementation_identity"),
        "input_hashes": payload.get("input_hashes"),
        "output_hashes": payload.get("output_hashes"),
        "selected_categories": list(
            dict(payload.get("configuration_identity") or {}).get("selected_categories") or []
        ),
        "source_identity": payload.get("source_identity"),
        "stage": payload.get("stage"),
    }


def case_pipeline_output_hashes(output_dir: Path, case_manifest: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    case_manifest_path = output_dir / "case_manifest.json"
    if case_manifest_path.exists():
        hashes["case_manifest.json"] = media.sha256_file(case_manifest_path)
    for artifact in list(case_manifest.get("artifacts") or []):
        artifact_path = output_dir / str(dict(artifact).get("path") or "")
        if artifact_path.exists() and artifact_path.is_file():
            hashes[str(dict(artifact).get("path") or "")] = media.sha256_file(artifact_path)
    return dict(sorted(hashes.items()))


def case_pipeline_configuration_identity(
    context: CaseContext,
    options: CasePipelineOptions,
) -> dict[str, Any]:
    payload = {
        "allow_partial_plate_preview": options.allow_partial_plate_preview,
        "face_policy_mode": options.face_policy_mode,
        "no_detection_overlay": options.no_detection_overlay,
        "primary_threshold": options.primary_threshold,
        "proposal_threshold": options.proposal_threshold,
        "schema_version": "case_pipeline_configuration_identity.v1",
        "selected_categories": list(context.selected_categories),
        "source_sha256": context.source_identity.sha256,
    }
    payload["category_configuration_hashes"] = {
        category: category_configuration_identity(context, options, category=category)[
            "configuration_hash"
        ]
        for category in context.selected_categories
    }
    payload["configuration_hash"] = stable_hash(payload)
    return payload


def case_pipeline_implementation_identity() -> dict[str, Any]:
    return {
        "case_pipeline_schema_version": SCHEMA_VERSION,
        "case_manifest_schema_version": CASE_MANIFEST_SCHEMA_VERSION,
        "render_plan_schema_version": RENDER_PLAN_SCHEMA_VERSION,
        "resumable_manifest_schema_version": RESUMABLE_STAGE_MANIFEST_SCHEMA_VERSION,
        "schema_version": "case_pipeline_implementation_identity.v1",
    }


def publish_resumable_stage_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_categories(value: str | None, *, legacy_faces_enabled: bool = True) -> tuple[str, ...]:
    if value is None:
        return (CATEGORY_FACE,) if legacy_faces_enabled else ()
    selected: set[str] = set()
    raw_items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("--categories cannot be empty")
    if "all" in raw_items:
        return VALID_CATEGORIES
    for item in raw_items:
        normalized = CATEGORY_ALIASES.get(item)
        if normalized is None or normalized == "all":
            raise ValueError(f"unsupported privacy category: {item}")
        selected.add(normalized)
    return tuple(category for category in VALID_CATEGORIES if category in selected)


def deferred_case_stage_statuses(stage_statuses: list[StageStatus]) -> list[StageStatus]:
    return [
        *stage_statuses,
        StageStatus(
            stage=STAGE_LATEST_REVIEWED,
            status=STATUS_NOT_RUN,
            reason="latest_reviewed_deferred",
            summary={
                "reason": (
                    "Latest reviewed publication requires a review session and is "
                    "not run by this automatic-preview checkpoint"
                )
            },
        ),
        StageStatus(
            stage=STAGE_FINALIZATION,
            status=STATUS_NOT_RUN,
            reason="case_level_finalization_deferred",
            summary={
                "reason": "case-level finalization requires explicit reviewed-case finalization"
            },
        ),
        StageStatus(
            stage=STAGE_EVALUATION,
            status=STATUS_NOT_RUN,
            reason="evaluation_requires_explicit_ground_truth",
            summary={"reason": "evaluation requires explicit ground truth inputs"},
        ),
    ]


def render_plan_artifacts(render_plan: dict[str, Any]) -> tuple[ArtifactReference, ...]:
    return tuple(
        case_manifests.artifact_reference_from_dict(render_plan[key])
        for key in ("render_plan_artifact", "effective_regions_artifact")
        if key in render_plan
    )


def write_render_plan(
    context: CaseContext,
    category_results: list[CategoryRunResult],
) -> dict[str, Any]:
    records, blockers = validate_effective_render_plan(
        context,
        [
            record
            for result in category_results
            for record in result.privacy_regions
        ],
    )
    records = sorted(
        records,
        key=lambda item: (
            item.source_frame,
            item.category,
            privacy_rendering.render_order(item),
            item.object_id or "",
            item.provenance,
            item.bbox_xyxy or (-1, -1, -1, -1),
        ),
    )
    plan_dir = context.output_dir / "render_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    regions_path = plan_dir / "effective_privacy_regions.jsonl"
    write_privacy_region_records(regions_path, records)
    category_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    for record in records:
        category_counts[record.category] = category_counts.get(record.category, 0) + 1
        provenance_counts[record.provenance] = provenance_counts.get(record.provenance, 0) + 1
    plan = {
        "case_id": context.case_id,
        "blockers": list(blockers),
        "category_counts": dict(sorted(category_counts.items())),
        "effective_region_count": len(records),
        "effective_regions_artifact": artifact_reference(
            context.output_dir,
            regions_path,
            role="effective_privacy_regions",
        ).to_dict(),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "schema_version": RENDER_PLAN_SCHEMA_VERSION,
        "selected_categories": list(context.selected_categories),
        "source_identity_sha256": context.source_identity.sha256,
    }
    plan["render_plan_hash"] = stable_hash(plan)
    plan_path = plan_dir / "render_plan.json"
    write_json(plan_path, plan)
    plan["render_plan_artifact"] = artifact_reference(
        context.output_dir,
        plan_path,
        role="render_plan",
    ).to_dict()
    return plan


def validate_effective_render_plan(
    context: CaseContext,
    records: list[PrivacyRegionRecord],
) -> tuple[list[PrivacyRegionRecord], tuple[str, ...]]:
    blockers: set[str] = set()
    selected = set(context.selected_categories)
    source_frame_count = context.source_identity.frame_count
    accepted: list[PrivacyRegionRecord] = []
    seen: set[tuple[Any, ...]] = set()
    keep_visible_regions = [
        record
        for record in records
        if record.source_record.get("privacy_disposition") == "keep_visible"
        or record.approval_state == "keep_visible"
    ]
    for record in records:
        if record.category not in selected:
            raise ValueError(f"render region category was not selected: {record.category}")
        if source_frame_count is not None and not (0 <= record.source_frame < source_frame_count):
            raise ValueError(
                f"render region frame outside source span: {record.category}:{record.source_frame}"
            )
        if not record.render_eligible or record.bbox_xyxy is None:
            continue
        if record.approval_state not in APPROVED_RENDER_STATES:
            raise ValueError(
                f"render-eligible region is not approved for redaction: "
                f"{record.category}:{record.source_frame}:{record.approval_state}"
            )
        if rendering.clipped_box(
            record.bbox_xyxy,
            context.source_info.video_stream.width,
            context.source_info.video_stream.height,
        ) is None:
            raise ValueError(
                f"render region has invalid geometry: {record.category}:{record.source_frame}"
            )
        key = (
            record.category,
            record.source_frame,
            record.object_id,
            record.provenance,
            record.bbox_xyxy,
        )
        if key in seen:
            raise ValueError(
                f"duplicate effective render region: {record.category}:{record.source_frame}"
            )
        seen.add(key)
        accepted.append(record)

    for keep in keep_visible_regions:
        if keep.bbox_xyxy is None:
            continue
        for redact in accepted:
            if keep.source_frame != redact.source_frame or keep.category == redact.category:
                continue
            if boxes_overlap(keep.bbox_xyxy, redact.bbox_xyxy):
                blockers.add("policy_overlap_conflict")
    return accepted, tuple(sorted(blockers))


def boxes_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
        left[1],
        right[1],
    )


def render_combined_automatic_preview(
    context: CaseContext,
    *,
    effective_regions_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    records = load_privacy_region_records(effective_regions_path)
    output_dir = context.output_dir / "combined_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "automatic_preview.mp4"
    fps = rational_to_float(context.source_info.video_stream.nominal_frame_rate) or 30.0
    settings = rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_fraction=0.15,
        margin_pixels=0,
        pixel_size=16,
        blur_kernel=51,
        preview_watermark=False,
    )
    rendered_frames, rendered_regions, audio_handling = privacy_rendering.render_privacy_video(
        context.source_video,
        output_video,
        privacy_rendering.regions_by_frame(records),
        settings=settings,
        fps=fps,
        width=context.source_info.video_stream.width,
        height=context.source_info.video_stream.height,
        source_has_audio=context.source_info.audio_stream is not None,
        overwrite=overwrite,
        log_path=output_dir / "processing_log.jsonl",
    )
    media_validation = rendering.validate_output_media(
        context.source_video,
        context.source_info,
        output_video,
        expected_frame_count=rendered_frames,
        expected_width=context.source_info.video_stream.width,
        expected_height=context.source_info.video_stream.height,
        expected_fps=fps,
        source_has_audio=context.source_info.audio_stream is not None,
        settings=settings,
    )
    region_validation = privacy_rendering.verify_rendered_regions(
        context.source_video,
        output_video,
        [record.to_render_region() for record in records],
        settings=settings,
        frame_width=context.source_info.video_stream.width,
        frame_height=context.source_info.video_stream.height,
    )
    if media_validation["passed"] is not True:
        raise RuntimeError("combined automatic preview media validation failed")
    if region_validation["passed"] is not True:
        raise RuntimeError("combined automatic preview rendered-region validation failed")
    output_hash = media.sha256_file(output_video)
    category_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    for record in records:
        category_counts[record.category] = category_counts.get(record.category, 0) + 1
        provenance_counts[record.provenance] = provenance_counts.get(record.provenance, 0) + 1
    render_configuration = {
        "blur_kernel": settings.normalized_blur_kernel,
        "margin_fraction": settings.margin_fraction,
        "margin_pixels": settings.margin_pixels,
        "method": settings.method,
        "pixel_size": settings.pixel_size,
    }
    manifest = {
        "audio_handling": audio_handling,
        "case_id": context.case_id,
        "category_counts": dict(sorted(category_counts.items())),
        "created_at_utc": media.utc_timestamp(),
        "effective_regions_hash": media.sha256_file(effective_regions_path),
        "final_approval_status": False,
        "frame_span": {
            "frame_count": context.source_identity.frame_count,
            "first_frame": 0,
            "last_frame": None
            if context.source_identity.frame_count is None
            else max(0, context.source_identity.frame_count - 1),
        },
        "output_video": output_video.relative_to(context.output_dir).as_posix(),
        "output_video_hash": output_hash,
        "preview_non_final": True,
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "render_configuration": render_configuration,
        "render_configuration_hash": stable_hash(render_configuration),
        "rendered_frame_count": rendered_frames,
        "rendered_region_count": rendered_regions,
        "schema_version": "combined_automatic_preview.v1",
        "source_video_hash": context.source_identity.sha256,
    }
    summary = {
        "media_validation_passed": media_validation["passed"],
        "output_video": manifest["output_video"],
        "preview_non_final": True,
        "region_counts_by_category": manifest["category_counts"],
        "region_counts_by_provenance": manifest["provenance_counts"],
        "rendered_frame_count": rendered_frames,
        "rendered_region_count": rendered_regions,
        "rendered_region_validation_passed": region_validation["passed"],
        "schema_version": "combined_automatic_preview_summary.v1",
    }
    write_json(output_dir / "render_manifest.json", manifest)
    write_json(
        output_dir / "validation.json",
        {
            "media_validation": media_validation,
            "rendered_region_validation": region_validation,
            "schema_version": "combined_automatic_preview_validation.v1",
        },
    )
    write_json(output_dir / "summary.json", summary)
    artifacts = (
        artifact_reference(
            context.output_dir,
            output_video,
            role="combined_automatic_preview_video",
        ),
        artifact_reference(
            context.output_dir,
            output_dir / "render_manifest.json",
            role="combined_automatic_preview_manifest",
        ),
        artifact_reference(
            context.output_dir,
            output_dir / "validation.json",
            role="combined_automatic_preview_validation",
        ),
        artifact_reference(
            context.output_dir,
            output_dir / "summary.json",
            role="combined_automatic_preview_summary",
        ),
    )
    return {"artifacts": artifacts, "manifest": manifest, "summary": summary}


def write_unified_case_manifest(
    context: CaseContext,
    *,
    category_results: list[CategoryRunResult],
    render_plan: dict[str, Any],
    combined_preview: dict[str, Any] | None,
    review_workspace: dict[str, Any] | None,
    stage_statuses: list[StageStatus],
    case_phase: str,
    case_blockers: tuple[str, ...],
    source_identity_artifact: ArtifactReference,
    category_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    categories = {result.category: result for result in category_results}
    blockers = sorted(set(case_blockers))
    artifacts = [
        source_identity_artifact,
        *[
            artifact
            for result in category_results
            for artifact in result.artifacts
        ],
        *render_plan_artifacts(render_plan),
    ]
    if combined_preview is not None:
        artifacts.extend(combined_preview["artifacts"])
    if review_workspace is not None:
        artifacts.extend(review_workspace["artifacts"])
    artifacts = list(
        case_manifests.validate_artifact_inventory(
            artifacts,
            case_root=context.output_dir,
        )
    )
    manifest = {
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "blockers": blockers,
        "case_id": context.case_id,
        "case_phase": case_phase,
        "category_profiles": {
            category: privacy_profiles.profile_for(category).to_dict()
            for category in context.selected_categories
        },
        "category_failures": category_failures or [],
        "category_summaries": {
            category: result.summary for category, result in sorted(categories.items())
        },
        "combined_preview": None
        if combined_preview is None
        else {
            "artifact_ids": [
                artifact.artifact_id for artifact in combined_preview["artifacts"]
            ],
            "preview_non_final": True,
            "rendered_frame_count": combined_preview["summary"]["rendered_frame_count"],
            "rendered_region_count": combined_preview["summary"]["rendered_region_count"],
        },
        "created_at_utc": media.utc_timestamp(),
        "finalization": {
            "case_level": True,
            "status": STATUS_NOT_RUN,
            "reason": (
                "case-level finalization is deferred until reviewed-case finalization "
                "is explicitly requested"
            ),
        },
        "latest_reviewed": {
            "status": STATUS_NOT_RUN,
            "reason": (
                "Latest reviewed publication requires a review session and is not "
                "run by redact-video in this checkpoint"
            ),
        },
        "review": review_manifest_payload(
            context,
            review_workspace=review_workspace,
            category_failures=category_failures or [],
            render_plan_blockers=tuple(str(item) for item in render_plan.get("blockers", [])),
        ),
        "review_session": review_session_manifest_payload(review_workspace),
        "evaluation": {
            "status": STATUS_NOT_RUN,
            "reason": "evaluation requires explicit ground truth inputs",
        },
        "render_plan": render_plan,
        "schema_version": CASE_MANIFEST_SCHEMA_VERSION,
        "selected_categories": list(context.selected_categories),
        "source_identity": context.source_identity.to_dict(),
        "stages": [stage.to_dict() for stage in stage_statuses],
    }
    return case_manifests.publish_unified_case_manifest(
        context.output_dir / "case_manifest.json",
        manifest,
    )


def review_manifest_payload(
    context: CaseContext,
    *,
    review_workspace: dict[str, Any] | None,
    category_failures: list[dict[str, Any]],
    render_plan_blockers: tuple[str, ...],
) -> dict[str, Any]:
    if review_workspace is not None:
        summary = dict(review_workspace.get("summary") or {})
        workspace_manifest = dict(review_workspace.get("manifest") or {})
        category_run_identities = dict(
            workspace_manifest.get("category_run_manifest_identities") or {}
        )
        return {
            "artifact_ids": [
                artifact.artifact_id for artifact in review_workspace.get("artifacts", ())
            ],
            "category_run_manifest_identities": category_run_identities,
            "category_run_manifest_identity_hash": stable_hash(category_run_identities),
            "category_review_artifacts_created": True,
            "entity_count": summary.get("entity_count", 0),
            "latest_reviewed_publication": workspace_manifest.get(
                "latest_reviewed_publication"
            ),
            "mixed_review_status": STATUS_READY,
            "reason": "review workspace is ready",
            "selected_categories": list(context.selected_categories),
            "task_count": summary.get("task_count", 0),
            "validation": review_workspace.get("validation", {}),
            "workspace_path": review_workspace.get("workspace_dir"),
        }
    if category_failures:
        return {
            "category_review_artifacts_created": False,
            "mixed_review_status": STATUS_FAILED,
            "reason": "requested category failed before review workspace generation",
            "selected_categories": list(context.selected_categories),
        }
    if render_plan_blockers:
        return {
            "category_review_artifacts_created": True,
            "mixed_review_status": STATUS_BLOCKED,
            "reason": "render plan validation blocked review workspace generation",
            "render_plan_blockers": list(render_plan_blockers),
            "selected_categories": list(context.selected_categories),
        }
    return {
        "category_review_artifacts_created": True,
        "mixed_review_status": STATUS_NOT_RUN,
        "reason": "review workspace was not generated",
        "selected_categories": list(context.selected_categories),
    }


def review_session_manifest_payload(review_workspace: dict[str, Any] | None) -> dict[str, Any]:
    if review_workspace is None:
        return {
            "reason": "review workspace is not ready",
            "status": STATUS_NOT_RUN,
        }
    return {
        "history_namespace": "review_workspace/sessions",
        "reason": "review sessions are created only when a reviewer starts review",
        "session_state_artifact_id": "case:mixed_review_session_state",
        "status": STATUS_NOT_STARTED,
    }


def bind_category_run_manifest(
    context: CaseContext,
    options: CasePipelineOptions,
    result: CategoryRunResult,
) -> CategoryRunResult:
    source_lineage = result.source_lineage or source_lineage_for_context(context)
    validate_category_source_lineage(context, source_lineage)
    validate_existing_category_artifact_lineage(context, result.category)
    configuration_identity = result.configuration_identity or category_configuration_identity(
        context,
        options,
        category=result.category,
    )
    validate_category_configuration_identity(
        configuration_identity,
        category_configuration_identity(context, options, category=result.category),
    )
    artifacts = case_manifests.validate_artifact_inventory(
        result.artifacts,
        case_root=context.output_dir,
    )
    manifest_dir = context.output_dir / result.category
    manifest_path = manifest_dir / "category_run_manifest.json"
    payload = {
        "artifact_hashes": {
            artifact.artifact_id or case_manifests.artifact_id_for(
                role=artifact.role,
                category=artifact.category,
            ): artifact.sha256
            for artifact in artifacts
        },
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "blockers": list(result.blockers),
        "case_id": context.case_id,
        "category": result.category,
        "configuration_identity": configuration_identity,
        "created_at_utc": media.utc_timestamp(),
        "schema_version": "category_run_manifest.v1",
        "source_lineage": source_lineage,
        "stage_outcomes": [stage.to_dict() for stage in result.stages],
    }
    payload["category_run_content_sha256"] = stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at_utc", "category_run_content_sha256"}
        }
    )
    write_json(manifest_path, payload)
    run_manifest_artifact = artifact_reference(
        context.output_dir,
        manifest_path,
        role="category_run_manifest",
        category=result.category,
    )
    return replace(
        result,
        artifacts=(*artifacts, run_manifest_artifact),
        configuration_identity=configuration_identity,
        source_lineage=source_lineage,
        summary={
            **result.summary,
            "category_run_manifest": run_manifest_artifact.path,
            "configuration_hash": configuration_identity["configuration_hash"],
        },
    )


def source_lineage_for_context(context: CaseContext) -> dict[str, Any]:
    frame_count = context.source_identity.frame_count
    return {
        "duration_seconds": context.source_identity.duration_seconds,
        "fps": context.source_identity.fps,
        "frame_count": frame_count,
        "has_audio": context.source_identity.has_audio,
        "height": context.source_identity.height,
        "path": context.source_identity.path,
        "processed_span": {
            "first_processed_frame": 0 if frame_count else None,
            "full_source_processed": True,
            "last_processed_frame": None if frame_count is None else max(0, frame_count - 1),
            "source_frame_count": frame_count,
        },
        "schema_version": "category_source_lineage.v1",
        "source_sha256": context.source_identity.sha256,
        "width": context.source_identity.width,
    }


def category_configuration_identity(
    context: CaseContext,
    options: CasePipelineOptions,
    *,
    category: str,
) -> dict[str, Any]:
    profile = privacy_profiles.profile_for(category)
    payload: dict[str, Any] = {
        "category": category,
        "detector_configuration_hash": profile.detector_configuration_hash,
        "detector_id": profile.detector_id,
        "profile_hash": profile.profile_hash(),
        "render_profile_hash": profile.render_profile_hash,
        "render_profile_id": profile.render_profile_id,
        "schema_version": "category_configuration_identity.v1",
        "source_sha256": context.source_identity.sha256,
        "tracking_profile_hash": profile.tracking_profile_hash,
        "tracking_profile_id": profile.tracking_profile_id,
    }
    if category == CATEGORY_FACE:
        payload["policy_mode"] = options.face_policy_mode
        payload["proposal_threshold"] = options.proposal_threshold
        payload["primary_threshold"] = options.primary_threshold
    if category == CATEGORY_PLATE:
        payload["allow_partial_plate_preview"] = options.allow_partial_plate_preview
    payload["configuration_hash"] = stable_hash(payload)
    return payload


def validate_category_configuration_identity(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if actual.get("configuration_hash") != expected.get("configuration_hash"):
        raise ValueError(
            "category configuration hash mismatch: "
            f"{actual.get('configuration_hash')} != {expected.get('configuration_hash')}"
        )


def validate_category_source_lineage(context: CaseContext, lineage: dict[str, Any]) -> None:
    expected = source_lineage_for_context(context)
    comparisons = (
        ("source_sha256", expected["source_sha256"]),
        ("frame_count", expected["frame_count"]),
        ("width", expected["width"]),
        ("height", expected["height"]),
    )
    for key, expected_value in comparisons:
        actual_value = lineage.get(key)
        if actual_value is not None and actual_value != expected_value:
            raise ValueError(
                f"category source lineage mismatch for {key}: {actual_value} != {expected_value}"
            )
    if lineage.get("fps") is not None:
        expected_fps = rational_to_float(str(expected["fps"])) if expected["fps"] else None
        actual_fps = rational_to_float(str(lineage["fps"]))
        if expected_fps is not None and actual_fps is not None:
            if round(expected_fps, 6) != round(actual_fps, 6):
                raise ValueError(
                    f"category source lineage mismatch for fps: {actual_fps} != {expected_fps}"
                )
    processed_span = lineage.get("processed_span") or {}
    expected_span = expected["processed_span"]
    for key in ("source_frame_count", "first_processed_frame", "last_processed_frame"):
        actual_value = processed_span.get(key)
        expected_value = expected_span.get(key)
        if (
            actual_value is not None
            and expected_value is not None
            and actual_value != expected_value
        ):
            raise ValueError(
                f"category processed-span mismatch for {key}: {actual_value} != {expected_value}"
            )


def validate_existing_category_artifact_lineage(context: CaseContext, category: str) -> None:
    if category == CATEGORY_FACE:
        validate_face_artifact_lineage(context)
    elif category == CATEGORY_PLATE:
        validate_plate_artifact_lineage(context)


def validate_face_artifact_lineage(context: CaseContext) -> None:
    source_manifest = load_json_if_exists(context.output_dir / "source_manifest.json")
    if source_manifest:
        validate_source_hash(
            context,
            source_manifest.get("source_video_hash"),
            label="face source_manifest.source_video_hash",
        )
        validate_optional_int(
            context.source_identity.frame_count,
            source_manifest.get("source_frame_count"),
            label="face source_manifest.source_frame_count",
        )
        validate_optional_int(
            context.source_identity.width,
            source_manifest.get("width"),
            label="face source_manifest.width",
        )
        validate_optional_int(
            context.source_identity.height,
            source_manifest.get("height"),
            label="face source_manifest.height",
        )
        validate_optional_fps(
            context,
            source_manifest.get("fps"),
            label="face source_manifest.fps",
        )
    render_manifest = load_json_if_exists(
        context.output_dir / "automatic_preview" / "render_manifest.json"
    )
    if render_manifest:
        validate_source_hash(
            context,
            render_manifest.get("source_video_hash"),
            label="face render_manifest.source_video_hash",
        )


def validate_plate_artifact_lineage(context: CaseContext) -> None:
    category_root = context.output_dir / CATEGORY_PLATE
    tracking_manifest = load_json_if_exists(category_root / "tracking" / "tracking_manifest.json")
    if tracking_manifest:
        validate_source_hash(
            context,
            tracking_manifest.get("source_sha256"),
            label="plate tracking_manifest.source_sha256",
        )
        validate_optional_int(
            context.source_identity.frame_count,
            tracking_manifest.get("source_frame_count"),
            label="plate tracking_manifest.source_frame_count",
        )
        validate_optional_int(
            context.source_identity.width,
            tracking_manifest.get("input_width"),
            label="plate tracking_manifest.input_width",
        )
        validate_optional_int(
            context.source_identity.height,
            tracking_manifest.get("input_height"),
            label="plate tracking_manifest.input_height",
        )
        validate_optional_fps(
            context,
            tracking_manifest.get("nominal_fps"),
            label="plate tracking_manifest.nominal_fps",
        )
    render_manifest = load_json_if_exists(
        category_root / "automatic_preview" / "render_manifest.json"
    )
    if render_manifest:
        validate_source_hash(
            context,
            render_manifest.get("source_video_hash"),
            label="plate render_manifest.source_video_hash",
        )
        validate_optional_int(
            context.source_identity.frame_count,
            render_manifest.get("source_frame_count"),
            label="plate render_manifest.source_frame_count",
        )
        validate_optional_int(
            context.source_identity.width,
            render_manifest.get("input_width"),
            label="plate render_manifest.input_width",
        )
        validate_optional_int(
            context.source_identity.height,
            render_manifest.get("input_height"),
            label="plate render_manifest.input_height",
        )


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def validate_source_hash(context: CaseContext, actual: Any, *, label: str) -> None:
    if actual is not None and str(actual) != context.source_identity.sha256:
        raise ValueError(f"{label} mismatch: {actual} != {context.source_identity.sha256}")


def validate_optional_int(expected: int | None, actual: Any, *, label: str) -> None:
    if expected is None or actual is None:
        return
    if int(actual) != int(expected):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def validate_optional_fps(context: CaseContext, actual: Any, *, label: str) -> None:
    if actual is None or context.source_identity.fps is None:
        return
    expected_fps = rational_to_float(context.source_identity.fps)
    actual_fps = float(actual)
    if expected_fps is not None and round(actual_fps, 6) != round(expected_fps, 6):
        raise ValueError(f"{label} mismatch: {actual_fps} != {expected_fps}")


def plate_privacy_region_records(tracks_path: Path) -> list[PrivacyRegionRecord]:
    if not tracks_path.exists():
        return []
    rows = plate_rendering.load_track_rows(tracks_path)
    regions = plate_rendering.eligible_regions_from_rows(
        rows,
        config=plate_rendering.default_config(),
    )
    return [region.to_privacy_region() for region in regions]


def load_privacy_region_records(path: Path) -> list[PrivacyRegionRecord]:
    from redactor import review_finalization

    return review_finalization.load_privacy_region_records(path)


def write_privacy_region_records(path: Path, records: list[PrivacyRegionRecord]) -> None:
    from redactor import review_finalization

    review_finalization.write_privacy_region_records(path, records)


def resolve_face_model_path(options: CasePipelineOptions) -> Path:
    return options.adas_xml_path or options.face_model_path or detection.default_adas_xml_path()


def ensure_case_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        existing_names = {path.name for path in output_dir.iterdir()}
        unexpected = sorted(existing_names - {"detection"})
        if unexpected:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def artifact_reference(
    case_root: Path,
    path: Path,
    *,
    role: str,
    category: str | None = None,
    required: bool = True,
    allow_external: bool = False,
) -> ArtifactReference:
    artifact = case_manifests.build_artifact_reference(
        path,
        case_root=case_root,
        role=role,
        category=category,
        required=required,
        allow_external=allow_external,
    )
    if artifact is None:
        raise case_manifests.ManifestValidationError(f"{role}: optional artifact is absent")
    return artifact


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
