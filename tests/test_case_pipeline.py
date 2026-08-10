from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from redactor import case_manifests, case_pipeline
from redactor.cli import main
from redactor.models import MediaInfo, VideoStreamInfo
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    PrivacyRegionRecord,
)


def fake_media_info() -> MediaInfo:
    return MediaInfo(
        audio_stream=None,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        created_at="2026-08-09T00:00:00Z",
        duration_seconds=0.2,
        ffprobe_version="ffprobe test",
        file_size_bytes=12,
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


def region(category: str, frame: int, box: tuple[int, int, int, int]) -> PrivacyRegionRecord:
    return PrivacyRegionRecord(
        approval_state=APPROVAL_ACCEPTED,
        bbox_xyxy=box,
        category=category,
        object_id=f"{category}_track_1",
        provenance="test_adapter",
        render_eligible=True,
        source_frame=frame,
        source_record={"category": category},
    )


class FakeAdapter:
    def __init__(self, category: str, record: PrivacyRegionRecord) -> None:
        self.category = category
        self.record = record

    def run(
        self,
        context: case_pipeline.CaseContext,
        options: case_pipeline.CasePipelineOptions,
    ) -> case_pipeline.CategoryRunResult:
        assert context.source_identity.sha256 == "source-sha256"
        assert options.overwrite is True
        review_artifact = self.write_review_tasks(context)
        return case_pipeline.CategoryRunResult(
            category=self.category,
            privacy_regions=(self.record,),
            artifacts=(review_artifact,),
            stages=(
                case_pipeline.StageStatus(
                    category=self.category,
                    stage=case_pipeline.STAGE_PRIVACY_REGIONS,
                    status=case_pipeline.STATUS_SUCCEEDED,
                    artifacts=(review_artifact,),
                    summary={"fake": True},
                ),
            ),
            summary={"fake_category": self.category},
        )

    def write_review_tasks(
        self,
        context: case_pipeline.CaseContext,
    ) -> case_pipeline.ArtifactReference:
        if self.category == CATEGORY_FACE:
            path = context.output_dir / "review" / "review_tasks.json"
            payload = {
                "reviewer_tasks": [
                    {
                        "category": "action_required",
                        "case_id": context.case_id,
                        "final_export_blocking": True,
                        "frame_end": self.record.source_frame,
                        "frame_start": self.record.source_frame,
                        "object_id": self.record.object_id,
                        "reviewer_task_id": "face_review_0001",
                        "title": "Face review",
                    }
                ],
                "schema_version": "test_face_review_tasks.v1",
            }
        else:
            path = context.output_dir / CATEGORY_PLATE / "review" / "review_tasks.json"
            payload = {
                "schema_version": "test_plate_review_tasks.v1",
                "tasks": [
                    {
                        "blocking": True,
                        "frame_end": self.record.source_frame,
                        "frame_start": self.record.source_frame,
                        "review_task_id": "plate_review_0001",
                        "severity": "blocker",
                        "title": "Plate review",
                        "track_uids": [self.record.object_id],
                    }
                ],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return case_pipeline.artifact_reference(
            context.output_dir,
            path,
            role="review_tasks",
            category=self.category,
        )


class FailingAdapter:
    category = CATEGORY_PLATE

    def run(
        self,
        context: case_pipeline.CaseContext,
        options: case_pipeline.CasePipelineOptions,
    ) -> case_pipeline.CategoryRunResult:
        raise RuntimeError("synthetic category failure")


def fake_context(tmp_path: Path) -> case_pipeline.CaseContext:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")
    info = fake_media_info()
    identity = case_pipeline.SourceIdentity.from_video(source, info=info)
    output_dir = tmp_path / "case"
    output_dir.mkdir()
    return case_pipeline.CaseContext(
        case_id="case",
        output_dir=output_dir,
        selected_categories=(CATEGORY_FACE, CATEGORY_PLATE),
        source_identity=identity,
        source_info=info,
        source_video=source,
    )


def test_case_pipeline_writes_one_manifest_and_combined_render_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(case_pipeline.media, "probe_media", lambda _: fake_media_info())

    def fake_combined_preview(
        context: case_pipeline.CaseContext,
        *,
        effective_regions_path: Path,
        overwrite: bool,
    ) -> dict[str, Any]:
        captured["effective_regions_path"] = effective_regions_path
        captured["overwrite"] = overwrite
        records = case_pipeline.load_privacy_region_records(effective_regions_path)
        output_dir = context.output_dir / "combined_preview"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "automatic_preview.mp4"
        render_manifest = output_dir / "render_manifest.json"
        validation = output_dir / "validation.json"
        summary = output_dir / "summary.json"
        output_video.write_bytes(b"combined preview")
        render_manifest.write_text("{}", encoding="utf-8")
        validation.write_text("{}", encoding="utf-8")
        summary.write_text("{}", encoding="utf-8")
        output_hash = case_pipeline.media.sha256_file(output_video)
        return {
            "artifacts": (
                case_pipeline.artifact_reference(
                    context.output_dir,
                    output_video,
                    role="combined_automatic_preview_video",
                ),
                case_pipeline.artifact_reference(
                    context.output_dir,
                    render_manifest,
                    role="combined_automatic_preview_manifest",
                ),
                case_pipeline.artifact_reference(
                    context.output_dir,
                    validation,
                    role="combined_automatic_preview_validation",
                ),
                case_pipeline.artifact_reference(
                    context.output_dir,
                    summary,
                    role="combined_automatic_preview_summary",
                ),
            ),
            "manifest": {
                "output_video": "combined_preview/automatic_preview.mp4",
                "output_video_hash": output_hash,
                "rendered_region_count": len(records),
                "source_video_hash": context.source_identity.sha256,
            },
            "summary": {
                "rendered_frame_count": 2,
                "rendered_region_count": len(records),
                "schema_version": "test_combined_preview.v1",
            },
        }

    monkeypatch.setattr(
        case_pipeline,
        "render_combined_automatic_preview",
        fake_combined_preview,
    )

    result = case_pipeline.run_case_pipeline(
        source,
        output_dir=tmp_path / "case",
        categories=(CATEGORY_FACE, CATEGORY_PLATE),
        options=case_pipeline.CasePipelineOptions(overwrite=True),
        adapters={
            CATEGORY_FACE: FakeAdapter(CATEGORY_FACE, region(CATEGORY_FACE, 1, (1, 2, 10, 20))),
            CATEGORY_PLATE: FakeAdapter(
                CATEGORY_PLATE,
                region(CATEGORY_PLATE, 1, (30, 40, 70, 55)),
            ),
        },
    )

    manifest = json.loads((tmp_path / "case" / "case_manifest.json").read_text())
    render_plan = json.loads((tmp_path / "case" / "render_plan" / "render_plan.json").read_text())
    region_rows = [
        json.loads(line)
        for line in (tmp_path / "case" / "render_plan" / "effective_privacy_regions.jsonl")
        .read_text()
        .splitlines()
    ]

    assert result["categories"] == [CATEGORY_FACE, CATEGORY_PLATE]
    assert manifest["source_identity"]["sha256"] == "source-sha256"
    assert manifest["selected_categories"] == [CATEGORY_FACE, CATEGORY_PLATE]
    assert set(manifest["category_profiles"]) == {CATEGORY_FACE, CATEGORY_PLATE}
    assert manifest["case_phase"] == case_pipeline.CASE_PHASE_AUTOMATIC_PREVIEW_READY
    assert manifest["review"]["mixed_review_status"] == case_pipeline.STATUS_READY
    assert manifest["review"]["entity_count"] == 2
    assert manifest["review"]["task_count"] == 2
    assert manifest["review_session"]["status"] == case_pipeline.STATUS_NOT_STARTED
    assert manifest["latest_reviewed"]["status"] == case_pipeline.STATUS_NOT_RUN
    assert manifest["finalization"]["status"] == case_pipeline.STATUS_NOT_RUN
    assert manifest["evaluation"]["status"] == case_pipeline.STATUS_NOT_RUN
    assert render_plan["effective_region_count"] == 2
    assert render_plan["category_counts"] == {CATEGORY_FACE: 1, CATEGORY_PLATE: 1}
    assert [row["category"] for row in region_rows] == [CATEGORY_FACE, CATEGORY_PLATE]
    assert captured["effective_regions_path"].name == "effective_privacy_regions.jsonl"
    assert captured["overwrite"] is True
    assert result["combined_preview"]["summary"]["rendered_region_count"] == 2
    artifact_roles = {item["role"] for item in manifest["artifacts"]}
    assert {
        "combined_automatic_preview_video",
        "combined_automatic_preview_manifest",
        "combined_automatic_preview_validation",
        "effective_privacy_regions",
        "mixed_review_entities",
        "mixed_review_media_inventory",
        "mixed_review_session_state",
        "mixed_review_tasks",
        "mixed_review_validation",
        "mixed_review_workspace_summary",
        "review_workspace_manifest",
        "render_plan",
        "source_identity",
    } <= artifact_roles
    assert "mixed_review_manifest" not in artifact_roles
    assert all(item["sha256"] for item in manifest["artifacts"])
    assert all((tmp_path / "case" / item["path"]).exists() for item in manifest["artifacts"])
    workspace_root = tmp_path / "case" / "review_workspace"
    workspace_manifest = json.loads((workspace_root / "workspace_manifest.json").read_text())
    category_run_identities = workspace_manifest["category_run_manifest_identities"]
    assert not (workspace_root / "mixed_review_manifest.json").exists()
    assert set(category_run_identities) == {CATEGORY_FACE, CATEGORY_PLATE}
    assert (
        manifest["review"]["category_run_manifest_identities"]
        == category_run_identities
    )
    assert (
        manifest["review"]["category_run_manifest_identity_hash"]
        == workspace_manifest["category_run_manifest_identity_hash"]
    )
    assert workspace_manifest["automatic_preview"]["region_count"] == 2
    assert workspace_manifest["latest_reviewed_publication"]["status"] == "deferred"
    assert manifest["review"]["latest_reviewed_publication"]["status"] == "deferred"
    entities = json.loads((workspace_root / "review_entities.json").read_text())
    tasks = json.loads((workspace_root / "reviewer_tasks.json").read_text())
    assert workspace_manifest["lifecycle"]["review_workspace"] == "ready"
    assert workspace_manifest["lifecycle"]["review_session"]["status"] == "not_started"
    assert {entity["category"] for entity in entities["cases"]["case"]} == {
        CATEGORY_FACE,
        CATEGORY_PLATE,
    }
    assert {task["category"] for task in tasks["tasks"]} == {CATEGORY_FACE, CATEGORY_PLATE}
    assert all(task["entity_id"] for task in tasks["tasks"])


def test_manifest_artifact_references_fail_closed(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    artifact = case_root / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    reference = case_manifests.build_artifact_reference(
        artifact,
        case_root=case_root,
        role="example",
    )

    assert reference is not None
    assert reference.sha256
    assert reference.size_bytes == 2
    assert reference.path == "artifact.json"
    assert (
        case_manifests.build_artifact_reference(
            case_root / "optional.json",
            case_root=case_root,
            role="optional",
            required=False,
        )
        is None
    )
    with pytest.raises(case_manifests.ManifestValidationError, match="missing"):
        case_manifests.build_artifact_reference(
            case_root / "missing.json",
            case_root=case_root,
            role="missing",
        )
    with pytest.raises(case_manifests.ManifestValidationError, match="sha256"):
        case_manifests.ArtifactReference(
            role="bad",
            path="bad.json",
            sha256="",
            size_bytes=0,
        )
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(case_manifests.ManifestValidationError, match="escapes"):
        case_manifests.build_artifact_reference(
            outside,
            case_root=case_root,
            role="outside",
        )


def test_manifest_inventory_rejects_duplicates_and_hash_mismatch(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    artifact = case_root / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    reference = case_manifests.build_artifact_reference(
        artifact,
        case_root=case_root,
        role="example",
    )
    assert reference is not None

    with pytest.raises(case_manifests.ManifestValidationError, match="duplicate"):
        case_manifests.validate_artifact_inventory(
            (reference, reference),
            case_root=case_root,
        )

    artifact.write_text('{"stale": true}', encoding="utf-8")
    with pytest.raises(case_manifests.ManifestValidationError, match="hash mismatch"):
        case_manifests.validate_artifact_inventory((reference,), case_root=case_root)


def test_semantic_case_hash_ignores_generated_timestamp(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    artifact = case_root / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    reference = case_manifests.build_artifact_reference(
        artifact,
        case_root=case_root,
        role="example",
    )
    assert reference is not None
    payload = {
        "artifacts": [reference.to_dict()],
        "case_content_sha256": "",
        "case_id": "case",
        "created_at_utc": "2026-08-09T00:00:00Z",
        "schema_version": case_manifests.UNIFIED_CASE_MANIFEST_SCHEMA_VERSION,
    }
    first = case_manifests.semantic_manifest_hash(payload)
    payload["created_at_utc"] = "2026-08-09T00:01:00Z"
    second = case_manifests.semantic_manifest_hash(payload)

    assert first == second


def test_atomic_manifest_publication_rejects_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "case" / "case_manifest.json"

    with pytest.raises(case_manifests.ManifestValidationError):
        case_manifests.publish_unified_case_manifest(
            path,
            {"artifacts": [], "schema_version": "wrong"},
        )

    assert not path.exists()


def test_requested_category_failure_blocks_case_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")
    monkeypatch.setattr(case_pipeline.media, "probe_media", lambda _: fake_media_info())

    with pytest.raises(RuntimeError, match="synthetic category failure"):
        case_pipeline.run_case_pipeline(
            source,
            output_dir=tmp_path / "case",
            categories=(CATEGORY_PLATE,),
            options=case_pipeline.CasePipelineOptions(overwrite=True),
            adapters={CATEGORY_PLATE: FailingAdapter()},
        )

    manifest = json.loads((tmp_path / "case" / "case_manifest.json").read_text())
    assert manifest["case_phase"] == case_pipeline.CASE_PHASE_AUTOMATIC_PREVIEW_FAILED
    assert manifest["category_failures"][0]["category"] == CATEGORY_PLATE
    assert manifest["finalization"]["status"] == case_pipeline.STATUS_NOT_RUN


def test_lineage_and_configuration_mismatch_fail_closed(tmp_path: Path) -> None:
    context = fake_context(tmp_path)
    lineage = case_pipeline.source_lineage_for_context(context)
    lineage["source_sha256"] = "wrong"

    with pytest.raises(ValueError, match="source_sha256"):
        case_pipeline.validate_category_source_lineage(context, lineage)

    expected = case_pipeline.category_configuration_identity(
        context,
        case_pipeline.CasePipelineOptions(),
        category=CATEGORY_FACE,
    )
    actual = {**expected, "configuration_hash": "wrong"}
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        case_pipeline.validate_category_configuration_identity(actual, expected)


def test_processed_span_mismatch_fails_closed(tmp_path: Path) -> None:
    context = fake_context(tmp_path)
    lineage = case_pipeline.source_lineage_for_context(context)
    lineage["processed_span"] = {
        **lineage["processed_span"],
        "source_frame_count": 999,
    }

    with pytest.raises(ValueError, match="processed-span mismatch"):
        case_pipeline.validate_category_source_lineage(context, lineage)


def test_render_plan_rejects_duplicates_and_blocks_keep_overlap(tmp_path: Path) -> None:
    context = fake_context(tmp_path)
    accepted_face = region(CATEGORY_FACE, 1, (1, 2, 10, 20))
    duplicate_face = region(CATEGORY_FACE, 1, (1, 2, 10, 20))

    with pytest.raises(ValueError, match="duplicate effective render region"):
        case_pipeline.validate_effective_render_plan(
            context,
            [accepted_face, duplicate_face],
        )

    keep_face = PrivacyRegionRecord(
        approval_state="keep_visible",
        bbox_xyxy=(30, 30, 70, 70),
        category=CATEGORY_FACE,
        object_id="face_keep_1",
        provenance="policy",
        render_eligible=False,
        source_frame=1,
        source_record={"privacy_disposition": "keep_visible"},
    )
    accepted_plate = region(CATEGORY_PLATE, 1, (40, 40, 80, 80))

    records, blockers = case_pipeline.validate_effective_render_plan(
        context,
        [keep_face, accepted_plate],
    )

    assert records == [accepted_plate]
    assert blockers == ("policy_overlap_conflict",)


def test_parse_categories_keeps_legacy_face_default() -> None:
    assert case_pipeline.parse_categories(None) == (CATEGORY_FACE,)
    assert case_pipeline.parse_categories("all") == (CATEGORY_FACE, CATEGORY_PLATE)
    assert case_pipeline.parse_categories("plates,face") == (CATEGORY_FACE, CATEGORY_PLATE)

    with pytest.raises(ValueError, match="unsupported privacy category"):
        case_pipeline.parse_categories("badge")


def test_redact_video_no_faces_without_categories_keeps_legacy_error(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not decoded before legacy error")

    result = CliRunner().invoke(
        main,
        [
            "redact-video",
            str(source),
            "--no-faces",
            "--output-dir",
            str(tmp_path / "case"),
        ],
    )

    assert result.exit_code != 0
    assert "redact-video currently supports --faces only" in result.output
