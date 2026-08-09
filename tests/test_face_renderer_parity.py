from __future__ import annotations

from typing import Any

from redactor import face_renderer_parity


def test_identical_source_frozen_shared_counts_classifies_pass() -> None:
    report = build_case(source_count=10, frozen_count=10, shared_count=10)

    assert report["classification"] == face_renderer_parity.CLASSIFICATION_PASS
    assert report["exact_encoded_output_parity_passed"] is True
    assert report["shared_renderer_migration_safe"] is True


def test_frozen_terminal_frame_loss_is_baseline_defect_when_shared_preserves_source() -> None:
    report = build_case(
        source_count=10,
        frozen_count=9,
        shared_count=10,
        decoded_outputs_ended_together=False,
    )

    assert report["classification"] == face_renderer_parity.CLASSIFICATION_BASELINE_DEFECT
    assert report["exact_encoded_output_parity_passed"] is False
    assert report["decoded_first_difference_frame"] == 9
    assert report["legacy_source_span_preserved"] is False
    assert report["shared_source_span_preserved"] is True
    assert report["shared_renderer_migration_safe"] is True


def test_shared_terminal_frame_loss_is_shared_renderer_defect() -> None:
    report = build_case(source_count=10, frozen_count=10, shared_count=9)

    assert report["classification"] == face_renderer_parity.CLASSIFICATION_SHARED_RENDERER_DEFECT
    assert report["shared_renderer_migration_safe"] is False


def test_both_outputs_losing_terminal_frame_is_common_pipeline_defect() -> None:
    report = build_case(source_count=10, frozen_count=9, shared_count=9)

    assert report["classification"] == face_renderer_parity.CLASSIFICATION_COMMON_PIPELINE_DEFECT
    assert report["shared_renderer_migration_safe"] is False


def test_output_count_mismatch_fails_exact_encoded_output_parity_with_raw_parity() -> None:
    report = build_case(
        source_count=10,
        frozen_count=9,
        shared_count=10,
        raw_render_parity_passed=True,
        decoded_outputs_ended_together=False,
    )

    assert report["raw_render_parity_passed"] is True
    assert report["exact_encoded_output_parity_passed"] is False


def test_outputs_not_ending_together_cannot_pass_exact_encoded_output_parity() -> None:
    report = build_case(
        source_count=10,
        frozen_count=10,
        shared_count=10,
        decoded_outputs_ended_together=False,
    )

    assert report["decoded_outputs_ended_together"] is False
    assert report["exact_encoded_output_parity_passed"] is False


def test_v2_schema_contains_required_explicit_fields() -> None:
    report = build_case(source_count=10, frozen_count=10, shared_count=10)

    required_fields = {
        "schema_version",
        "source_video",
        "frozen_output_video",
        "shared_output_video",
        "source_sequential_decoded_frame_count",
        "frozen_sequential_decoded_frame_count",
        "shared_sequential_decoded_frame_count",
        "source_first_timestamp",
        "source_last_timestamp",
        "frozen_first_timestamp",
        "frozen_last_timestamp",
        "shared_first_timestamp",
        "shared_last_timestamp",
        "raw_render_parity_passed",
        "raw_render_frame_count_compared",
        "raw_render_first_difference_frame",
        "raw_render_max_abs_diff",
        "exact_encoded_output_parity_passed",
        "decoded_frames_compared",
        "decoded_first_difference_frame",
        "decoded_max_abs_diff",
        "decoded_outputs_ended_together",
        "legacy_source_span_preserved",
        "shared_source_span_preserved",
        "media_validation_passed",
        "rendered_region_validation_passed",
        "classification",
        "shared_renderer_migration_safe",
        "blockers",
        "notes",
    }

    assert report["schema_version"] == face_renderer_parity.CASE_SCHEMA_VERSION
    assert required_fields <= set(report)


def test_summary_records_documented_baseline_defect_status() -> None:
    cases = {
        "day_face": build_case(source_count=10, frozen_count=10, shared_count=10),
        "night_face": build_case(
            source_count=10,
            frozen_count=9,
            shared_count=10,
            decoded_outputs_ended_together=False,
        ),
    }

    summary = face_renderer_parity.build_summary_report(cases)

    assert summary["schema_version"] == face_renderer_parity.SUMMARY_SCHEMA_VERSION
    assert summary["status"] == face_renderer_parity.STATUS_PASS_WITH_DOCUMENTED_BASELINE_DEFECT
    assert summary["all_shared_source_spans_preserved"] is True
    assert summary["exact_encoded_output_parity_for_all_cases"] is False
    assert summary["baseline_defect_detected"] is True
    assert summary["shared_renderer_migration_safe"] is True


def build_case(
    *,
    source_count: int,
    frozen_count: int,
    shared_count: int,
    raw_render_parity_passed: bool = True,
    decoded_outputs_ended_together: bool = True,
) -> dict[str, Any]:
    return face_renderer_parity.build_case_report(
        case_id="case",
        source_video="source.mp4",
        frozen_output_video="frozen.mp4",
        shared_output_video="shared.mp4",
        source_summary=summary("source.mp4", source_count),
        frozen_summary=summary("frozen.mp4", frozen_count),
        shared_summary=summary("shared.mp4", shared_count),
        raw_render_comparison=face_renderer_parity.RawRenderComparison(
            passed=raw_render_parity_passed,
            frame_count_compared=max(source_count, frozen_count, shared_count),
            first_difference_frame=None,
            max_abs_diff=0,
        ),
        decoded_output_comparison=face_renderer_parity.DecodedOutputComparison(
            frames_compared=min(frozen_count, shared_count),
            first_difference_frame=min(frozen_count, shared_count)
            if frozen_count != shared_count
            else None,
            max_abs_diff=0,
            outputs_ended_together=decoded_outputs_ended_together,
        ),
        media_validation_passed=True,
        rendered_region_validation_passed=True,
    )


def summary(path: str, frame_count: int) -> face_renderer_parity.SequentialDecodeSummary:
    return face_renderer_parity.SequentialDecodeSummary(
        path=path,
        frame_count=frame_count,
        first_timestamp=0.0,
        last_timestamp=round((frame_count - 1) / 10.0, 6) if frame_count else None,
    )
