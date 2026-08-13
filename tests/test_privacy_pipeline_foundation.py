from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from redactor import (
    case_pipeline,
    detection,
    media,
    plate_detection,
    plate_rendering,
    plate_tracking,
    privacy_lifecycle,
    privacy_profiles,
    privacy_rendering,
    rendering,
)
from redactor.privacy_records import (
    APPROVAL_ACCEPTED,
    CATEGORY_FACE,
    CATEGORY_PLATE,
    GAP_ACCEPTED,
    ISSUE_UNRESOLVED_GAP,
    LIFECYCLE_CONFIRMED,
    LIFECYCLE_TENTATIVE,
    PrivacyObservation,
    PrivacyRegionRecord,
    ProcessingSpan,
    ReviewTaskRecord,
    TemporalRegion,
    normalize_evidence_disposition,
    normalize_privacy_disposition,
    normalize_review_action,
    normalize_scoring_disposition,
    privacy_regions_from_render_regions,
)


def write_synthetic_video(path: Path, *, frame_count: int = 4) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 80))
    try:
        for frame_index in range(frame_count):
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            frame[:, :] = (25 + frame_index * 8, 50 + frame_index * 10, 95)
            cv.rectangle(frame, (10, 10), (36, 36), (230, 230, 230), -1)
            cv.putText(
                frame,
                "FACE",
                (11, 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 0, 0),
                1,
            )
            cv.rectangle(frame, (55, 50), (100, 66), (245, 245, 245), -1)
            cv.putText(
                frame,
                "ABC123",
                (57, 63),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()


def render_settings() -> rendering.RenderSettings:
    return rendering.RenderSettings(
        mode=rendering.MODE_PREVIEW,
        method=rendering.METHOD_PIXELATE,
        margin_pixels=0,
        margin_fraction=0.0,
        pixel_size=4,
        preview_watermark=False,
    )


def face_record(frame_index: int) -> PrivacyRegionRecord:
    return PrivacyRegionRecord(
        category=CATEGORY_FACE,
        source_frame=frame_index,
        bbox_xyxy=(10, 10, 36, 36),
        provenance="detector",
        approval_state=APPROVAL_ACCEPTED,
        render_eligible=True,
        object_id="face_track_0001",
    )


def plate_record(frame_index: int) -> PrivacyRegionRecord:
    return PrivacyRegionRecord(
        category=CATEGORY_PLATE,
        source_frame=frame_index,
        bbox_xyxy=(55, 50, 100, 66),
        provenance=plate_tracking.PROVENANCE_DETECTOR,
        approval_state=APPROVAL_ACCEPTED,
        render_eligible=True,
        object_id="plate_track_0001",
    )


def test_common_observation_serialization_is_deterministic() -> None:
    observation = PrivacyObservation(
        category=CATEGORY_PLATE,
        detector_id=plate_detection.DETECTOR_ID,
        source_frame=7,
        score=0.3333339,
        bbox_xyxy=(1, 2, 30, 12),
        detector_provenance="model_observation",
        configuration_hash="abc123",
        timing_metadata={"inference_ms": 2.5},
    )

    payload = observation.to_dict()

    assert payload["schema_version"] == "privacy_pipeline_records.v1"
    assert payload["bbox_xyxy"] == [1, 2, 30, 12]
    assert payload["score"] == 0.333334
    assert observation.record_hash() == PrivacyObservation(**observation.__dict__).record_hash()


def test_category_profiles_preserve_selected_detector_separation() -> None:
    face = privacy_profiles.face_profile()
    plate = privacy_profiles.plate_profile()

    assert face.category == CATEGORY_FACE
    assert face.detector_id == detection.DETECTOR_BACKEND_ADAS
    assert plate.category == CATEGORY_PLATE
    assert plate.detector_id == plate_detection.DETECTOR_ID
    assert plate.detector_configuration_hash == (
        "7d914438255905e2a249e5bbd30fd6e70eace1e91b57e205bd572d004ffb5a55"
    )
    assert face.profile_hash() != plate.profile_hash()
    assert face.tracking_profile_hash != plate.tracking_profile_hash


def test_semantic_dispositions_do_not_cross_normalize() -> None:
    assert normalize_privacy_disposition("keep") == "keep_visible"
    assert normalize_evidence_disposition("not_a_license_plate") == "rejected_wrong_class"
    assert normalize_review_action("correct_geometry") == "correct_geometry"
    assert normalize_scoring_disposition("ignore_for_detector") == "detector_ignore"

    with pytest.raises(ValueError, match="scoring disposition"):
        normalize_privacy_disposition("detector_ignore")
    with pytest.raises(ValueError, match="privacy disposition"):
        normalize_scoring_disposition("redact")


def test_common_lifecycle_state_validation_rejects_unknown_state() -> None:
    with pytest.raises(ValueError, match="unsupported lifecycle state"):
        TemporalRegion(
            category=CATEGORY_FACE,
            object_id="face_track_0001",
            source_frame=0,
            bbox_xyxy=(1, 1, 10, 10),
            source_provenance="detector",
            track_state="active",
            render_eligible=True,
            review_status="accepted",
        )


def test_shared_lifecycle_protocols_support_future_policy_implementations() -> None:
    observation = PrivacyObservation(
        category=CATEGORY_PLATE,
        detector_id=plate_detection.DETECTOR_ID,
        source_frame=12,
        score=0.8,
        bbox_xyxy=(10, 10, 40, 20),
        detector_provenance="model_observation",
        configuration_hash="hash",
    )

    class MockAssociationPolicy:
        def associate(
            self,
            *,
            track_id: str,
            observations: tuple[PrivacyObservation, ...],
            source_frame: int,
        ) -> privacy_lifecycle.AssociationDecision:
            return privacy_lifecycle.AssociationDecision(
                category=CATEGORY_PLATE,
                track_id=track_id,
                source_frame=source_frame,
                observation_hashes=tuple(item.record_hash() for item in observations),
                accepted=True,
                reason="synthetic_iou_match",
            )

    policy = MockAssociationPolicy()
    decision = policy.associate(
        track_id="plate_track_0001",
        observations=(observation,),
        source_frame=12,
    )
    transition = privacy_lifecycle.HysteresisDecision(
        category=CATEGORY_PLATE,
        track_id="plate_track_0001",
        source_frame=12,
        from_state=LIFECYCLE_TENTATIVE,
        to_state=LIFECYCLE_CONFIRMED,
        reason="enough_evidence",
    )
    stitch = privacy_lifecycle.TrackletStitchDecision(
        category=CATEGORY_PLATE,
        left_track_id="plate_track_0001",
        right_track_id="plate_track_0002",
        state=GAP_ACCEPTED,
        reason="bounded_gap",
    )

    assert isinstance(policy, privacy_lifecycle.AssociationPolicy)
    assert decision.to_dict()["observation_hashes"] == [observation.record_hash()]
    assert transition.to_dict()["to_state"] == LIFECYCLE_CONFIRMED
    assert stitch.to_dict()["state"] == GAP_ACCEPTED


def test_processing_span_wraps_shared_missing_range_accounting() -> None:
    span = ProcessingSpan.from_frame_indices([0, 2, 3], source_frame_count=5)

    assert span.full_source_processed is False
    assert span.missing_processed_ranges == ((1, 1), (4, 4))
    assert span.to_dict()["missing_processed_ranges"] == [
        {"frame_start": 1, "frame_end": 1, "length": 1},
        {"frame_start": 4, "frame_end": 4, "length": 1},
    ]


def test_shared_review_task_record_keeps_category_and_actions() -> None:
    task = ReviewTaskRecord(
        category=CATEGORY_PLATE,
        task_id="plate_review_0001",
        object_id="plate_track_0001",
        frame_start=10,
        frame_end=20,
        issue_type=ISSUE_UNRESOLVED_GAP,
        blocking=True,
        allowed_actions=("Correct", "Needs correction", "Not sensitive"),
        correction_required=True,
        reasons=("gap_exceeds_safe_interpolation_limit",),
    )

    payload = task.to_dict()

    assert payload["category"] == CATEGORY_PLATE
    assert payload["issue_type"] == ISSUE_UNRESOLVED_GAP
    assert payload["allowed_actions"] == ["Correct", "Needs correction", "Not sensitive"]


def test_shared_renderer_handles_face_and_plate_regions_together(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "shared.mp4"
    write_synthetic_video(source, frame_count=4)
    records = [plate_record(1), face_record(1), plate_record(2)]

    rendered_frames, rendered_regions, audio_handling = privacy_rendering.render_privacy_video(
        source,
        output,
        privacy_rendering.regions_by_frame(records),
        settings=render_settings(),
        fps=10.0,
        width=120,
        height=80,
        source_has_audio=False,
        overwrite=True,
    )
    verification = privacy_rendering.verify_rendered_regions(
        source,
        output,
        [record.to_render_region() for record in records],
        settings=render_settings(),
        frame_width=120,
        frame_height=80,
    )
    info = media.probe_media(output)

    assert rendered_frames == 4
    assert rendered_regions == 3
    assert audio_handling == "source_had_no_audio"
    assert verification["passed"] is True
    assert info.video_stream.codec == "h264"
    assert info.video_stream.pixel_format == "yuv420p"
    assert privacy_rendering.provenance_counts(records) == {
        "detector": 1,
        plate_tracking.PROVENANCE_DETECTOR: 2,
    }


def test_shared_region_order_is_deterministic_for_overlapping_categories() -> None:
    records = [plate_record(1), face_record(1)]

    grouped = privacy_rendering.regions_by_frame(records)

    assert [region.source_record["category"] for region in grouped[1]] == [
        CATEGORY_FACE,
        CATEGORY_PLATE,
    ]


def test_plate_render_region_adapts_to_common_privacy_region() -> None:
    region = plate_rendering.PlateRenderRegion(
        frame_index=3,
        bbox_xyxy=(55, 50, 100, 66),
        provenance=plate_tracking.PROVENANCE_INTERPOLATION,
        status=plate_tracking.ACCEPTED_TRACK,
        track_uid="plate_track_0007",
        track_id=7,
        score=None,
        source_record={"status": plate_tracking.ACCEPTED_TRACK},
    )

    common = region.to_privacy_region()

    assert common.category == CATEGORY_PLATE
    assert common.object_id == "plate_track_0007"
    assert common.source_frame == 3
    assert common.render_eligible is True
    assert common.to_render_region().source_record["category"] == CATEGORY_PLATE


def test_face_renderer_regions_have_shared_renderer_parity(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, frame_count=3)
    privacy_regions = tmp_path / "face_regions.jsonl"
    privacy_regions.write_text(
        json.dumps(
            {
                "bbox_xyxy": [10, 10, 36, 36],
                "frame_index": 1,
                "provenance": "detector",
                "state": "covered",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    existing_output_dir = tmp_path / "face_existing"
    existing_summary = rendering.run_face_render(
        source,
        privacy_regions_path=privacy_regions,
        output_dir=existing_output_dir,
        settings=render_settings(),
    )
    shared_output = tmp_path / "face_shared.mp4"
    common_records = privacy_regions_from_render_regions(
        CATEGORY_FACE,
        rendering.load_privacy_regions(privacy_regions).regions_by_frame,
        object_id="face_track_0001",
    )
    record = common_records[0]
    privacy_rendering.render_privacy_video(
        source,
        shared_output,
        privacy_rendering.regions_by_frame([record]),
        settings=render_settings(),
        fps=10.0,
        width=120,
        height=80,
        source_has_audio=False,
        overwrite=True,
    )

    existing_verification = privacy_rendering.verify_rendered_regions(
        source,
        existing_output_dir / "redacted_preview.mp4",
        [record.to_render_region()],
        settings=render_settings(),
        frame_width=120,
        frame_height=80,
    )
    shared_verification = privacy_rendering.verify_rendered_regions(
        source,
        shared_output,
        [record.to_render_region()],
        settings=render_settings(),
        frame_width=120,
        frame_height=80,
    )

    assert existing_summary["processed_frame_count"] == 3
    assert existing_verification["passed"] is True
    assert shared_verification["passed"] is True


def test_frozen_and_shared_face_renderer_frame_parity_for_edge_cases() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            frame[y, x] = ((x * 3) % 255, (y * 5) % 255, (x + y) % 255)
    regions_by_frame = {
        0: [rendering.RenderRegion(0, (10, 10, 36, 36), "detector", "covered", {})],
        1: [rendering.RenderRegion(1, (0, 0, 18, 18), "detector", "covered", {})],
        2: [rendering.RenderRegion(2, (101, 61, 120, 80), "detector", "covered", {})],
        3: [rendering.RenderRegion(3, (30, 30, 32, 32), "detector", "covered", {})],
        4: [
            rendering.RenderRegion(4, (10, 10, 35, 35), "detector", "covered", {}),
            rendering.RenderRegion(
                4,
                (22, 22, 50, 50),
                "ordinary_tracker",
                "covered",
                {},
            ),
        ],
    }
    records = privacy_regions_from_render_regions(CATEGORY_FACE, regions_by_frame)
    shared_regions = privacy_rendering.regions_by_frame(records)

    for frame_index in range(6):
        frozen = rendering.render_frame(
            frame,
            frame_index,
            regions_by_frame,
            render_settings(),
        )
        shared = rendering.render_frame(
            frame,
            frame_index,
            shared_regions,
            render_settings(),
        )
        assert np.array_equal(frozen, shared)


def test_combined_face_component_matches_frozen_output_outside_plate_mask() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:, :] = (30, 80, 120)
    cv.rectangle(frame, (10, 10), (36, 36), (230, 230, 230), -1)
    cv.rectangle(frame, (55, 50), (100, 66), (245, 245, 245), -1)
    cv.putText(frame, "ABC123", (57, 63), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    face = face_record(0)
    plate = plate_record(0)
    frozen_face = rendering.render_frame(
        frame,
        0,
        privacy_rendering.regions_by_frame([face]),
        render_settings(),
    )
    combined = rendering.render_frame(
        frame,
        0,
        privacy_rendering.regions_by_frame([face, plate]),
        render_settings(),
    )
    plate_box = rendering.expanded_clipped_box(
        plate.bbox_xyxy,
        frame_width=120,
        frame_height=80,
        margin_pixels=render_settings().margin_pixels,
        margin_fraction=render_settings().margin_fraction,
    )
    assert plate_box is not None
    mask = np.zeros(frame.shape[:2], dtype=bool)
    x1, y1, x2, y2 = plate_box
    mask[y1:y2, x1:x2] = True

    assert np.array_equal(frozen_face[~mask], combined[~mask])
    assert not np.array_equal(frozen_face[mask], combined[mask])


def test_combined_render_plan_filters_keep_reject_and_review_required(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, frame_count=4)
    source_info = media.probe_media(source)
    identity = case_pipeline.SourceIdentity.from_video(source, info=source_info)
    context = case_pipeline.CaseContext(
        case_id="case",
        output_dir=tmp_path / "case",
        selected_categories=(CATEGORY_FACE, CATEGORY_PLATE),
        source_identity=identity,
        source_info=source_info,
        source_video=source,
    )
    accepted_plate = plate_record(1)
    rejected_plate = PrivacyRegionRecord(
        approval_state="rejected",
        bbox_xyxy=(20, 20, 50, 35),
        category=CATEGORY_PLATE,
        object_id="plate_rejected",
        provenance="plate_detector_observation",
        render_eligible=False,
        source_frame=1,
    )
    review_plate = PrivacyRegionRecord(
        approval_state="needs_review",
        bbox_xyxy=(60, 20, 90, 35),
        category=CATEGORY_PLATE,
        object_id="plate_review",
        provenance="plate_detector_observation",
        render_eligible=False,
        source_frame=1,
    )
    keep_face = PrivacyRegionRecord(
        approval_state="keep_visible",
        bbox_xyxy=(1, 1, 10, 10),
        category=CATEGORY_FACE,
        object_id="face_keep",
        provenance="policy",
        render_eligible=False,
        source_frame=1,
        source_record={"privacy_disposition": "keep_visible"},
    )

    records, blockers = case_pipeline.validate_effective_render_plan(
        context,
        [keep_face, accepted_plate, rejected_plate, review_plate],
    )

    assert records == [accepted_plate]
    assert blockers == ()


def test_future_category_can_use_common_records_without_core_changes() -> None:
    profile = privacy_profiles.CategoryProfile(
        category="badge",
        detector_id="badge_detector_v1",
        detector_configuration_hash="detector-hash",
        tracking_profile_id="badge_tracking_v1",
        tracking_profile_hash="tracking-hash",
        render_profile_id="badge_pixelate_v1",
        render_profile_hash="render-hash",
        geometry={"minimum_area_fraction": 0.001},
    )
    observation = PrivacyObservation(
        category=profile.category,
        detector_id=profile.detector_id,
        source_frame=1,
        score=0.9,
        bbox_xyxy=(5, 5, 15, 15),
        detector_provenance="model_observation",
        configuration_hash=profile.detector_configuration_hash,
    )

    assert profile.profile_hash()
    assert observation.to_dict()["category"] == "badge"


def test_production_code_does_not_import_archived_detector_experiments() -> None:
    production_files = sorted(Path("src/redactor").glob("*.py"))
    forbidden = ("experiments.archive", "docs.archive", "license_plate_detector_bakeoff")

    offenders = {
        path.as_posix(): token
        for path in production_files
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }

    assert offenders == {}
