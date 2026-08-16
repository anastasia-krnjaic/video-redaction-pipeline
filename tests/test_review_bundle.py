from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest

from redactor import review_bundle
from redactor.models import MediaInfo, VideoStreamInfo


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records]
    path.write_text(
        "".join(lines),
        encoding="utf-8",
    )
    return path


def media_info_for(path: Path, frame_count: int = 100) -> MediaInfo:
    return MediaInfo(
        schema_version="1.0",
        created_at="2026-01-01T00:00:00Z",
        source_filename=path.name,
        source_path=path.name,
        file_size_bytes=path.stat().st_size,
        sha256=review_bundle.media.sha256_file(path),
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=frame_count / 30,
        video_stream=VideoStreamInfo(
            index=0,
            codec="h264",
            width=100,
            height=80,
            pixel_format="yuv420p",
            nominal_frame_rate="30/1",
            average_frame_rate="30/1",
            time_base="1/30",
            frame_count=str(frame_count),
        ),
        audio_stream=None,
        ffprobe_version="ffprobe version test",
    )


def source_video(tmp_path: Path) -> Path:
    path = tmp_path / "source.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-source")
    return path


def synthetic_frame(index: int, *, dark: bool = False) -> np.ndarray:
    if dark:
        return np.zeros((48, 64, 3), dtype=np.uint8)
    y, x = np.indices((48, 64))
    base = ((x * 7 + y * 11 + index * 29) % 256).astype(np.uint8)
    return np.dstack(
        [
            base,
            np.roll(base, index % 9, axis=1),
            np.roll(base, index % 7, axis=0),
        ]
    )


def synthetic_large_frame(index: int, *, value: int | None = None) -> np.ndarray:
    if value is not None:
        return np.full((120, 160, 3), value, dtype=np.uint8)
    y, x = np.indices((120, 160))
    base = ((x * 7 + y * 11 + index * 23) % 256).astype(np.uint8)
    return np.dstack(
        [
            base,
            np.roll(base, index % 11, axis=1),
            np.roll(base, index % 13, axis=0),
        ]
    )


def write_synthetic_video(path: Path, frames: list[np.ndarray], *, fps: float = 30.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def single_frame_render_validation(
    tmp_path: Path,
    *,
    candidate_regions: dict[int, list[review_bundle.rendering.RenderRegion]],
    expected_regions: dict[int, list[review_bundle.rendering.RenderRegion]],
    frame: np.ndarray | None = None,
    settings: review_bundle.rendering.RenderSettings | None = None,
    source_frame: np.ndarray | None = None,
) -> dict[str, Any]:
    source = frame if frame is not None else synthetic_large_frame(0)
    authoritative_source = source_frame if source_frame is not None else source
    render_settings = settings or review_bundle.ReviewRenderConfig(
        margin_fraction=0.5,
        pixel_size=8,
    ).render_settings()
    source_path = write_synthetic_video(tmp_path / "source.mp4", [authoritative_source])
    candidate_source = (
        review_bundle.decode_video_frames(source_path)[0]
        if source_frame is None
        else source
    )
    original_path = write_synthetic_video(tmp_path / "original.mp4", [candidate_source])
    rendered = review_bundle.rendering.render_frame(
        candidate_source,
        0,
        candidate_regions,
        render_settings,
    )
    rendered_path = write_synthetic_video(tmp_path / "rendered.mp4", [rendered])
    return review_bundle.validate_rendered_region_application(
        original_clip_path=original_path,
        rendered_clip_path=rendered_path,
        frame_mapping=review_bundle.TaskFrameMapping(0, 0),
        regions_by_frame=expected_regions,
        settings=render_settings,
        source_video_path=source_path,
    )


def patch_probe(monkeypatch: pytest.MonkeyPatch, frame_count: int = 100) -> None:
    monkeypatch.setattr(
        review_bundle.media,
        "probe_media",
        lambda path: media_info_for(path, frame_count=frame_count),
    )


def privacy_regions(tmp_path: Path) -> Path:
    records = [
        record(10, [10, 10, 20, 20], "detector"),
        record(11, None, "bidirectional_privacy_bridge", state="unresolved"),
        record(12, [12, 10, 22, 20], "ordinary_tracker"),
        record(13, [13, 10, 23, 20], "preserved_privacy_region"),
        record(14, [14, 10, 14, 20], "detector"),
    ]
    return write_jsonl(tmp_path / "privacy_regions.jsonl", records)


def record(
    frame: int,
    bbox: list[int] | None,
    provenance: str,
    *,
    state: str = "covered",
) -> dict[str, Any]:
    return {
        "frame_index": frame,
        "privacy_regions": [
            {
                "bbox_xyxy": bbox,
                "provenance": provenance,
                "state": state,
                "target_track_ids": [7],
            }
        ],
    }


def intervals(tmp_path: Path, *, status: str = "needs_review") -> Path:
    return write_json(
        tmp_path / "interval_decisions.json",
        {
            "interval_decisions": [
                {
                    "frame_start": 11,
                    "frame_end": 12,
                    "metrics": {
                        "maximum_normalized_raw_disagreement": 1.2,
                        "maximum_scale_jump": 0.9,
                    },
                    "reasons": ["excessive_scale_change"],
                    "recovered_frames": 1,
                    "status": status,
                    "strategy": "bidirectional_privacy_bridge",
                    "target_track_ids": [7],
                    "unresolved_frames": 1,
                }
            ]
        },
    )


def evaluation_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evaluation"
    write_json(
        directory / "failures.json",
        {
            "failures": [
                {
                    "clip": "synthetic",
                    "evaluated_stage": "detector",
                    "expected_policy": "redact",
                    "failure_reason": "detector_scored_frame_without_adjudicated_candidate",
                    "frame": 20,
                    "object_id": "face_a",
                    "observed_provenance": "none",
                    "review_status": "synthetic",
                },
                {
                    "clip": "synthetic",
                    "evaluated_stage": "privacy",
                    "expected_policy": "redact",
                    "failure_reason": "privacy_required_frame_uncovered",
                    "frame": 21,
                    "object_id": "face_a",
                    "observed_provenance": "none",
                    "review_status": "synthetic",
                },
            ],
            "schema_version": "1.0",
        },
    )
    write_json(
        directory / "object_metrics.json",
        {
            "objects": {
                "face_a": {
                    "detector": {"missed_detector_scored_frames": [20]},
                    "privacy": {"detector_scored_visible_uncovered_frames": [21]},
                    "stability": {"suspicious_provenance_transitions": 1},
                    "tracking": {
                        "fragment_track_ids": [7, 8],
                        "track_fragmentation": 2,
                        "track_starts": [
                            {"frame": 10, "track_id": 7},
                            {"frame": 30, "track_id": 8},
                        ],
                    },
                }
            },
            "schema_version": "1.0",
        },
    )
    return directory


def track_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    quality = write_json(
        tmp_path / "track_quality.json",
        {
            "tracks": [
                {
                    "first_frame": 40,
                    "last_frame": 42,
                    "quality_gate_failures": ["low_median_observed_score"],
                    "quality_status": "review_candidate",
                    "strong_hits": 0,
                    "track_id": 4,
                    "weak_hits": 3,
                }
            ]
        },
    )
    flags = write_json(
        tmp_path / "review_flags.json",
        {
            "review_flags": [
                {
                    "frame_start": 44,
                    "frame_end": 45,
                    "reason": "track begins more than half a second after scene start",
                    "track_id": 4,
                    "type": "late_track_start",
                }
            ]
        },
    )
    windows = write_json(
        tmp_path / "review_windows.json",
        {
            "review_windows": [
                {
                    "flag_types": ["early_track_end", "long_gap"],
                    "frame_start": 50,
                    "frame_end": 55,
                    "reasons": ["gap exceeds configured maximum"],
                    "track_ids": [4],
                }
            ]
        },
    )
    return quality, flags, windows


def build_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    interval_status: str = "needs_review",
    random_sample_count: int = 2,
    include_evaluation: bool = False,
    include_tracking: bool = False,
    include_privacy: bool = True,
) -> Path:
    patch_probe(monkeypatch)
    source = source_video(tmp_path)
    privacy = privacy_regions(tmp_path) if include_privacy else None
    interval_path = intervals(tmp_path, status=interval_status) if include_privacy else None
    quality = flags = windows = None
    if include_tracking:
        quality, flags, windows = track_artifacts(tmp_path)
    review_bundle.build_review_bundle(
        source_video_path=source,
        output_dir=tmp_path / "review",
        config=review_bundle.ReviewBundleConfig(
            case_id="synthetic",
            created_at="2026-01-01T00:00:00Z",
            generate_visuals=False,
            random_sample_count=random_sample_count,
            random_sample_frame_count=5,
            random_seed=7,
            merge_gap_frames=2,
        ),
        privacy_regions_path=privacy,
        interval_decisions_path=interval_path,
        evaluation_dir=evaluation_dir(tmp_path) if include_evaluation else None,
        track_quality_path=quality,
        track_review_flags_path=flags,
        track_review_windows_path=windows,
        overwrite=True,
    )
    return tmp_path / "review"


def load_queue(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / "review_queue.json").read_text(encoding="utf-8"))


def reason_codes(queue: dict[str, Any]) -> set[str]:
    return {reason for item in queue["items"] for reason in item["reason_codes"]}


def raw_item(
    item_id: str,
    start: int,
    end: int,
    reasons: list[str],
    *,
    blocking: bool = False,
    severity: str = "medium",
    media_roles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "allowed_review_actions": ["accepted_current", "correction_required"],
        "available_candidate_region_status": "available",
        "blocking_status": blocking,
        "case_id": "synthetic",
        "explanation": "raw",
        "frame_end": end,
        "frame_references": [start],
        "frame_start": start,
        "item_origin": "pipeline",
        "media_roles": media_roles or [],
        "metrics": {},
        "object_category": "face",
        "object_id": "face_a",
        "reason_codes": reasons,
        "requires_corrected_regions": blocking,
        "review_item_id": item_id,
        "review_status": "pending",
        "schema_version": "1.0",
        "severity": severity,
        "source_stage": "multiple_stages",
        "source_video_hash": "hash",
        "timestamps": {"end_seconds": end / 30, "start_seconds": start / 30},
        "track_ids": [],
        "unresolved_frame_count": 0,
        "visual_evidence": {},
    }


def media_roles(*roles: str) -> list[dict[str, Any]]:
    return [
        {
            "display": True,
            "duplicate_of": None,
            "label": role.replace("_", " ").title(),
            "path": f"clips/{role}.mp4",
            "role": role,
            "sha256": role,
        }
        for role in roles
    ]


def test_review_item_schema_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = build_bundle(tmp_path, monkeypatch)
    queue = load_queue(bundle)

    assert review_bundle.validate_review_queue_payload(queue) == []
    assert queue["schema_id"] == review_bundle.REVIEW_QUEUE_SCHEMA_ID
    assert queue["presentation_schema_version"] == review_bundle.REVIEW_TASK_SCHEMA_VERSION


def test_reviewer_task_aggregation_hides_technical_signals() -> None:
    queue = {
        "case_id": "synthetic",
        "corrected_context": {"corrected_intervals": []},
        "items": [
            raw_item(
                "blocker",
                10,
                12,
                ["needs_review_interval", "unresolved_privacy_frames"],
                blocking=True,
                severity="blocking",
                media_roles=media_roles("original", "automatic", "corrected_final"),
            ),
            raw_item(
                "technical",
                11,
                11,
                ["suspicious_provenance_transition"],
                severity="informational",
            ),
            raw_item(
                "random",
                60,
                64,
                ["random_audit_sample"],
                severity="random_audit",
            ),
        ],
    }

    payload = review_bundle.reviewer_tasks_payload(
        queue,
        config=review_bundle.ReviewBundleConfig(case_id="synthetic"),
        fps=30.0,
    )

    tasks = payload["reviewer_tasks"]
    assert [task["category"] for task in tasks] == [
        "corrected_result_check",
        "optional_spot_check",
    ]
    assert tasks[0]["raw_signal_count"] == 2
    assert payload["technical_audit"]["hidden_from_default_task_count"] == 1
    assert "Additional related evidence" not in tasks[0]["explanation"]


def test_detector_miss_becomes_continuity_check_without_corrected_intervals() -> None:
    queue = {
        "case_id": "synthetic",
        "corrected_context": {"corrected_intervals": []},
        "items": [
            raw_item(
                "detector_miss",
                20,
                20,
                ["ground_truth_detector_miss"],
                severity="high",
                media_roles=media_roles("original", "corrected_final"),
            )
        ],
    }

    payload = review_bundle.reviewer_tasks_payload(
        queue,
        config=review_bundle.ReviewBundleConfig(case_id="synthetic"),
        fps=30.0,
    )

    task = payload["reviewer_tasks"][0]
    assert task["category"] == "automatic_continuity_check"
    assert task["title"] == "Check automatic continuity"


def test_media_role_hash_deduplication(tmp_path: Path) -> None:
    source = tmp_path / "clips" / "same_source.mp4"
    final = tmp_path / "clips" / "same_final.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"same bytes")
    final.write_bytes(b"same bytes")
    item = {
        "visual_evidence": {
            "reviewed_final_clip": "clips/same_final.mp4",
            "source_clip": "clips/same_source.mp4",
        }
    }

    review_bundle.assign_media_roles([item], tmp_path)

    roles = item["media_roles"]
    assert roles[0]["role"] == "original"
    assert roles[1]["role"] == "corrected_final"
    assert roles[1]["display"] is False
    assert roles[1]["duplicate_of"] == "original"


def role_audit(
    role: str,
    *,
    source_start: int = 10,
    source_end: int = 20,
    renderer_hash: str = "render_hash",
    frame_count: int = 11,
    duration_seconds: float = 11 / 30,
    privacy_hash: str | None = "privacy_hash",
    traceable: bool = True,
) -> dict[str, Any]:
    return {
        "codec": "h264",
        "decoded_visual_content_validation": {"passed": True},
        "duration_seconds": duration_seconds,
        "output_fps": 30.0,
        "output_frame_count": frame_count,
        "pixel_format": "yuv420p",
        "privacy_region_input_hash": privacy_hash,
        "rendered_region_application_validation": {"passed": True} if role != "original" else None,
        "rendered_privacy_regions": role != "original",
        "renderer_configuration_hash": renderer_hash,
        "role": role,
        "source_frame_range": {
            "frame_count": source_end - source_start + 1,
            "frame_end": source_end,
            "frame_start": source_start,
        },
        "traceable_reviewed_region_source": traceable,
    }


def test_review_render_config_defaults_to_canonical_pixelation() -> None:
    config = review_bundle.ReviewRenderConfig()

    assert config.payload() == {
        "blur_kernel": 51,
        "margin_fraction": 0.15,
        "margin_pixels": 0,
        "method": "pixelate",
        "pixel_size": 16,
    }
    assert config.stable_hash() == review_bundle.stable_payload_hash(config.payload())


def test_media_audit_accepts_aligned_comparison() -> None:
    validation = review_bundle.validate_media_comparison_audit(
        {
            "automatic": role_audit("automatic"),
            "corrected_final": role_audit("corrected_final"),
            "original": role_audit("original", privacy_hash=None),
        }
    )

    assert validation["passed"] is True
    assert validation["errors"] == []


def test_media_audit_rejects_mismatched_renderer_config() -> None:
    validation = review_bundle.validate_media_comparison_audit(
        {
            "automatic": role_audit("automatic", renderer_hash="a"),
            "corrected_final": role_audit("corrected_final", renderer_hash="b"),
        }
    )

    assert "renderer configurations differ" in " ".join(validation["errors"])


def test_media_audit_rejects_mismatched_source_interval_and_duration() -> None:
    validation = review_bundle.validate_media_comparison_audit(
        {
            "automatic": role_audit("automatic", source_start=10, source_end=20),
            "corrected_final": role_audit(
                "corrected_final",
                source_start=11,
                source_end=21,
                duration_seconds=5.0,
            ),
        }
    )

    errors = " ".join(validation["errors"])
    assert "source-frame ranges differ" in errors
    assert "durations differ" in errors


def test_media_audit_requires_traceable_corrected_region_source() -> None:
    validation = review_bundle.validate_media_comparison_audit(
        {
            "automatic": role_audit("automatic"),
            "corrected_final": role_audit(
                "corrected_final",
                privacy_hash=None,
                traceable=False,
            ),
        }
    )

    errors = " ".join(validation["errors"])
    assert "no traceable privacy-region input hash" in errors
    assert "no reviewed/preserved region provenance" in errors


def test_media_audit_rejects_non_browser_compatible_codec() -> None:
    bad = role_audit("automatic")
    bad["codec"] = "mpeg4"

    validation = review_bundle.validate_media_comparison_audit({"automatic": bad})

    assert "h264" in " ".join(validation["errors"])


def test_render_review_clip_outputs_browser_compatible_visible_mp4(tmp_path: Path) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(6)],
    )
    output = tmp_path / "clips" / "review.mp4"
    audit = review_bundle.render_review_clip_by_frames(
        source,
        output,
        1,
        3,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="original",
        regions_by_frame={},
        settings=None,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=review_bundle.ReviewRenderConfig().payload(),
        renderer_config_hash=review_bundle.ReviewRenderConfig().stable_hash(),
        overwrite=True,
        decoded_audit_dir=tmp_path / "decoded_frame_audit",
        media_slug="review",
    )

    assert audit["codec"] == "h264"
    assert audit["pixel_format"] == "yuv420p"
    assert audit["output_frame_count"] == 3
    assert audit["decoded_visual_content_validation"]["passed"] is True
    assert audit["decoded_visual_content_validation"]["decoded_frame_count"] == 3
    assert (tmp_path / audit["decoded_visual_content_validation"]["contact_sheet_path"]).exists()


def test_decoded_visual_validation_rejects_black_output_against_visible_source(
    tmp_path: Path,
) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(3)],
    )
    output = write_synthetic_video(
        tmp_path / "black.mp4",
        [synthetic_frame(index, dark=True) for index in range(3)],
    )
    output_h264 = tmp_path / "black_h264.mp4"
    review_bundle.mux_review_clip_audio(
        source,
        output,
        output_h264,
        source_has_audio=False,
        start_seconds=0.0,
        duration_seconds=3 / 30,
    )

    validation = review_bundle.validate_review_clip_visual_content(
        source_video_path=source,
        output_path=output_h264,
        start_frame=0,
        end_frame=2,
        role="original",
        expected_fps=30.0,
        expected_width=64,
        expected_height=48,
        expected_frame_count=3,
        output_info=review_bundle.media.probe_media(output_h264),
    )

    assert validation["passed"] is False
    assert "near-black" in " ".join(validation["errors"])


def test_decoded_visual_validation_accepts_source_relative_dark_scene(tmp_path: Path) -> None:
    dark_frames = [synthetic_frame(index, dark=True) for index in range(3)]
    source = write_synthetic_video(tmp_path / "source.mp4", dark_frames)
    output = tmp_path / "output.mp4"
    review_bundle.render_review_clip_by_frames(
        source,
        output,
        0,
        2,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="original",
        regions_by_frame={},
        settings=None,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=review_bundle.ReviewRenderConfig().payload(),
        renderer_config_hash=review_bundle.ReviewRenderConfig().stable_hash(),
        overwrite=True,
    )

    audit = review_bundle.validate_review_clip_visual_content(
        source_video_path=source,
        output_path=output,
        start_frame=0,
        end_frame=2,
        role="original",
        expected_fps=30.0,
        expected_width=64,
        expected_height=48,
        expected_frame_count=3,
        output_info=review_bundle.media.probe_media(output),
    )

    assert audit["passed"] is True


def test_decoded_visual_validation_rejects_unexpected_identical_output(tmp_path: Path) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(4)],
    )
    repeated = [synthetic_frame(0) for _ in range(4)]
    output_video = write_synthetic_video(tmp_path / "repeated.mp4", repeated)
    output_h264 = tmp_path / "repeated_h264.mp4"
    review_bundle.mux_review_clip_audio(
        source,
        output_video,
        output_h264,
        source_has_audio=False,
        start_seconds=0.0,
        duration_seconds=4 / 30,
    )

    validation = review_bundle.validate_review_clip_visual_content(
        source_video_path=source,
        output_path=output_h264,
        start_frame=0,
        end_frame=3,
        role="original",
        expected_fps=30.0,
        expected_width=64,
        expected_height=48,
        expected_frame_count=4,
        output_info=review_bundle.media.probe_media(output_h264),
    )

    assert validation["passed"] is False
    assert "identical" in " ".join(validation["errors"])


def test_render_review_clip_fails_before_publication_on_visual_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(3)],
    )
    monkeypatch.setattr(
        review_bundle,
        "validate_review_clip_visual_content",
        lambda **kwargs: {"errors": ["synthetic validation failure"], "passed": False},
    )

    with pytest.raises(ValueError, match="visual-content validation failed"):
        review_bundle.render_review_clip_by_frames(
            source,
            tmp_path / "clips" / "review.mp4",
            0,
            2,
            fps=30.0,
            width=64,
            height=48,
            source_has_audio=False,
            role="original",
            regions_by_frame={},
            settings=None,
            privacy_regions_path=None,
            interval_decisions_path=None,
            renderer_config=review_bundle.ReviewRenderConfig().payload(),
            renderer_config_hash=review_bundle.ReviewRenderConfig().stable_hash(),
            overwrite=True,
        )


def test_one_frame_miss_clip_context_is_short() -> None:
    item = raw_item("miss", 481, 481, ["ground_truth_detector_miss"], severity="high")

    assert review_bundle.clip_range_for_item(
        item,
        frame_count=1000,
        config=review_bundle.ReviewBundleConfig(context_frames=18),
    ) == (469, 493)


def test_task_frame_mapping_uses_source_indices_authoritatively() -> None:
    mapping = review_bundle.TaskFrameMapping(469, 493)

    assert mapping.frame_count == 25
    assert mapping.local_for_source(469) == 0
    assert mapping.local_for_source(481) == 12
    assert mapping.local_for_source(493) == 24
    assert mapping.source_for_local(0) == 469
    assert mapping.source_for_local(24) == 493
    assert mapping.payload()["mapping"][12] == {
        "source_frame_index": 481,
        "task_clip_frame_index": 12,
    }


def test_rendered_region_application_validates_nonzero_start_mapping(
    tmp_path: Path,
) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(12)],
    )
    mapping = review_bundle.TaskFrameMapping(5, 8)
    settings = review_bundle.ReviewRenderConfig(pixel_size=4).render_settings()
    region = review_bundle.rendering.RenderRegion(
        frame_index=6,
        bbox_xyxy=(10, 10, 30, 30),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    source_clip = tmp_path / "source_clip.mp4"
    rendered_clip = tmp_path / "rendered_clip.mp4"
    config = review_bundle.ReviewRenderConfig(pixel_size=4)
    review_bundle.render_review_clip_by_frames(
        source,
        source_clip,
        5,
        8,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="original",
        regions_by_frame={},
        settings=None,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=config.payload(),
        renderer_config_hash=config.stable_hash(),
        overwrite=True,
    )
    review_bundle.render_review_clip_by_frames(
        source,
        rendered_clip,
        5,
        8,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="automatic",
        regions_by_frame={6: [region]},
        settings=settings,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=config.payload(),
        renderer_config_hash=config.stable_hash(),
        overwrite=True,
    )

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=source_clip,
        rendered_clip_path=rendered_clip,
        frame_mapping=mapping,
        regions_by_frame={6: [region]},
        settings=settings,
    )

    assert validation["passed"] is True
    assert validation["expected_region_frames"] == [6]
    assert validation["successfully_rendered_region_frames"] == [6]
    assert validation["region_results"][0]["local_task_frame_index"] == 1


def test_rendered_region_application_streams_video_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = review_bundle.TaskFrameMapping(0, 5)
    settings = review_bundle.ReviewRenderConfig(pixel_size=4).render_settings()
    region = review_bundle.rendering.RenderRegion(
        frame_index=2,
        bbox_xyxy=(10, 10, 30, 30),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    original_clip = write_synthetic_video(
        tmp_path / "original.mp4",
        [synthetic_frame(index) for index in range(6)],
    )
    rendered_frames = [synthetic_frame(index) for index in range(6)]
    rendered_frames[2] = review_bundle.rendering.render_frame(
        rendered_frames[2],
        2,
        {2: [region]},
        settings,
    )
    rendered_clip = write_synthetic_video(tmp_path / "rendered.mp4", rendered_frames)

    def fail_full_decode(path: Path) -> list[np.ndarray]:
        raise AssertionError(f"full decode should not be used for {path}")

    monkeypatch.setattr(review_bundle, "decode_video_frames", fail_full_decode)

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=original_clip,
        rendered_clip_path=rendered_clip,
        frame_mapping=mapping,
        regions_by_frame={2: [region]},
        settings=settings,
    )

    assert validation["passed"] is True
    assert validation["expected_region_frames"] == [2]


def test_rendered_region_application_uses_shared_visibility_contract_for_h264_overlap(
    tmp_path: Path,
) -> None:
    frame_count = 36
    height = 96
    width = 128
    settings = review_bundle.ReviewRenderConfig(pixel_size=4).render_settings()
    broad_region = review_bundle.rendering.RenderRegion(
        frame_index=22,
        bbox_xyxy=(0, 0, width, height),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    specific_region = review_bundle.rendering.RenderRegion(
        frame_index=22,
        bbox_xyxy=(84, 68, 126, 94),
        provenance="detector",
        state="covered",
        source_record={},
    )
    regions_by_frame = {
        10: [
            review_bundle.rendering.RenderRegion(
                frame_index=10,
                bbox_xyxy=(0, 0, 30, 24),
                provenance="detector",
                state="covered",
                source_record={},
            )
        ],
        14: [
            review_bundle.rendering.RenderRegion(
                frame_index=14,
                bbox_xyxy=(22, 18, 58, 48),
                provenance="ordinary_tracker",
                state="covered",
                source_record={},
            )
        ],
        15: [
            review_bundle.rendering.RenderRegion(
                frame_index=15,
                bbox_xyxy=(22, 18, 58, 48),
                provenance="ordinary_tracker",
                state="covered",
                source_record={},
            )
        ],
        16: [
            review_bundle.rendering.RenderRegion(
                frame_index=16,
                bbox_xyxy=(22, 18, 58, 48),
                provenance="ordinary_tracker",
                state="covered",
                source_record={},
            )
        ],
        22: [broad_region, specific_region],
        35: [
            review_bundle.rendering.RenderRegion(
                frame_index=35,
                bbox_xyxy=(98, 70, 128, 96),
                provenance="bidirectional_privacy_bridge",
                state="covered",
                source_record={},
            )
        ],
    }
    originals: list[np.ndarray] = []
    rendered: list[np.ndarray] = []
    for index in range(frame_count):
        base = np.full((height, width, 3), 72 + (index % 5), dtype=np.uint8)
        originals.append(base)
        rendered.append(
            review_bundle.rendering.render_frame(base, index, regions_by_frame, settings)
        )

    source_video_only = write_synthetic_video(tmp_path / "source_video_only.mp4", originals)
    rendered_video_only = write_synthetic_video(tmp_path / "rendered_video_only.mp4", rendered)
    source_clip = tmp_path / "source_clip.mp4"
    rendered_clip = tmp_path / "rendered_clip.mp4"
    review_bundle.mux_review_clip_audio(
        source_video_only,
        source_video_only,
        source_clip,
        source_has_audio=False,
        start_seconds=0.0,
        duration_seconds=frame_count / 30.0,
    )
    review_bundle.mux_review_clip_audio(
        source_video_only,
        rendered_video_only,
        rendered_clip,
        source_has_audio=False,
        start_seconds=0.0,
        duration_seconds=frame_count / 30.0,
    )

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=source_clip,
        rendered_clip_path=rendered_clip,
        frame_mapping=review_bundle.TaskFrameMapping(0, frame_count - 1),
        regions_by_frame=regions_by_frame,
        settings=settings,
    )

    assert validation["passed"] is True
    assert validation["expected_region_count"] == 7
    assert validation["successfully_rendered_region_count"] == 7
    assert validation["successfully_rendered_region_frames"] == [10, 14, 15, 16, 22, 35]
    assert validation["region_results"][0]["local_task_frame_index"] == 10
    assert validation["region_results"][-1]["local_task_frame_index"] == frame_count - 1
    overlap_result = next(
        result
        for result in validation["region_results"]
        if result["source_frame_index"] == 22 and result["bbox_xyxy"] == [84, 68, 126, 94]
    )
    legacy_inside_outside_pass = (
        overlap_result["mean_absdiff_inside_bbox"]
        >= max(4.0, overlap_result["mean_absdiff_outside_bbox"] + 2.0)
        and overlap_result["max_absdiff_inside_bbox"] >= 40
    )
    assert legacy_inside_outside_pass is False
    assert overlap_result["changed_pixel_fraction_inside_bbox"] >= 0.01
    assert overlap_result["localized_mean_absdiff_inside_bbox"] >= 1.0


def test_rendered_region_application_uses_source_video_for_expected_render_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [np.full((48, 64, 3), 150, dtype=np.uint8) for _ in range(9)]
    source_video = write_synthetic_video(tmp_path / "source.mp4", frames)
    settings = review_bundle.ReviewRenderConfig(pixel_size=4).render_settings()
    region = review_bundle.rendering.RenderRegion(
        frame_index=6,
        bbox_xyxy=(18, 12, 50, 40),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    regions_by_frame = {6: [region]}
    rendered_clip = tmp_path / "rendered_clip.mp4"
    config = review_bundle.ReviewRenderConfig(pixel_size=4)
    review_bundle.render_review_clip_by_frames(
        source_video,
        rendered_clip,
        5,
        8,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="automatic",
        regions_by_frame=regions_by_frame,
        settings=settings,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=config.payload(),
        renderer_config_hash=config.stable_hash(),
        overwrite=True,
    )

    source_clip_frames = [frame.copy() for frame in frames[5:9]]
    source_clip_frames[1][12:40, 18:50] = 40
    source_clip = tmp_path / "source_clip.mp4"
    source_clip_video_only = write_synthetic_video(
        tmp_path / "source_clip_video_only.mp4",
        source_clip_frames,
    )
    review_bundle.mux_review_clip_audio(
        source_video,
        source_clip_video_only,
        source_clip,
        source_has_audio=False,
        start_seconds=5 / 30.0,
        duration_seconds=4 / 30.0,
    )
    monkeypatch.setattr(review_bundle, "region_redaction_applied", lambda metrics: False)

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=source_clip,
        rendered_clip_path=rendered_clip,
        source_video_path=source_video,
        frame_mapping=review_bundle.TaskFrameMapping(5, 8),
        regions_by_frame=regions_by_frame,
        settings=settings,
    )

    assert validation["passed"] is True
    assert validation["successfully_rendered_region_count"] == 1
    assert validation["region_results"][0]["expected_render_matches_clip"] is True
    assert validation["region_results"][0]["expected_render_source"] == "source_video"


def test_rendered_region_application_rejects_unrelated_redaction_inside_expanded_target_roi(
    tmp_path: Path,
) -> None:
    target = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(40, 40, 80, 80),
        provenance="target",
        state="covered",
        source_record={},
    )
    unrelated_inside_target_margin = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(22, 22, 36, 36),
        provenance="wrong_region",
        state="covered",
        source_record={},
    )

    validation = single_frame_render_validation(
        tmp_path,
        candidate_regions={0: [unrelated_inside_target_margin]},
        expected_regions={0: [target]},
    )

    result = validation["region_results"][0]
    assert validation["passed"] is False
    assert validation["missing_rendered_region_frames"] == [0]
    assert result["legacy_expanded_inside_outside_render_applied"] is True
    assert result["legacy_inside_outside_render_applied"] is False
    assert result["target_region_privacy_signal"] is False
    assert (
        result["target_bbox_changed_pixel_fraction"]
        < review_bundle.rendering.MIN_REDACTION_CHANGED_PIXEL_FRACTION
    )
    assert result["target_bbox_mean_absdiff"] < review_bundle.rendering.MIN_REDACTION_MEAN_ABSDIFF
    assert result["expected_render_matches_clip"] is False
    assert "rendered region count mismatch" in " ".join(validation["errors"])


def test_rendered_region_application_accepts_target_with_unrelated_extra_redaction(
    tmp_path: Path,
) -> None:
    target = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(40, 40, 80, 80),
        provenance="target",
        state="covered",
        source_record={},
    )
    unrelated_elsewhere = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(118, 72, 148, 108),
        provenance="reviewer_extra",
        state="covered",
        source_record={},
    )

    validation = single_frame_render_validation(
        tmp_path,
        candidate_regions={0: [target, unrelated_elsewhere]},
        expected_regions={0: [target]},
    )

    assert validation["passed"] is True
    assert validation["successfully_rendered_region_count"] == 1
    assert validation["region_results"][0]["legacy_inside_outside_render_applied"] is True
    assert validation["region_results"][0]["expected_render_matches_clip"] is True


def test_rendered_region_application_accepts_authoritative_overlapping_regions(
    tmp_path: Path,
) -> None:
    broad = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(32, 30, 94, 94),
        provenance="broad",
        state="covered",
        source_record={"render_order": 0},
    )
    specific = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(52, 48, 82, 76),
        provenance="specific",
        state="covered",
        source_record={"render_order": 1},
    )

    validation = single_frame_render_validation(
        tmp_path,
        candidate_regions={0: [broad, specific]},
        expected_regions={0: [broad, specific]},
        settings=review_bundle.ReviewRenderConfig(
            margin_fraction=0.0,
            pixel_size=8,
        ).render_settings(),
        frame=synthetic_large_frame(0, value=150),
    )

    assert validation["passed"] is True
    assert validation["expected_region_count"] == 2
    assert validation["successfully_rendered_region_count"] == 2
    assert all(result["expected_render_matches_clip"] for result in validation["region_results"])


def test_rendered_region_application_accepts_low_texture_target_privacy_signal(
    tmp_path: Path,
) -> None:
    target = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(40, 40, 80, 80),
        provenance="low_texture",
        state="covered",
        source_record={},
    )

    validation = single_frame_render_validation(
        tmp_path,
        candidate_regions={0: [target]},
        expected_regions={0: [target]},
        frame=synthetic_large_frame(0, value=150),
    )

    assert validation["passed"] is True
    assert validation["region_results"][0]["target_region_privacy_signal"] is True
    assert validation["region_results"][0]["expected_render_matches_clip"] is True


def test_rendered_region_application_rejects_shifted_wrong_bbox_and_source_mismatch(
    tmp_path: Path,
) -> None:
    target = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(40, 40, 80, 80),
        provenance="target",
        state="covered",
        source_record={},
    )
    shifted = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(52, 52, 92, 92),
        provenance="shifted",
        state="covered",
        source_record={},
    )
    shifted_validation = single_frame_render_validation(
        tmp_path / "shifted",
        candidate_regions={0: [shifted]},
        expected_regions={0: [target]},
    )
    source_mismatch_validation = single_frame_render_validation(
        tmp_path / "source_mismatch",
        candidate_regions={0: [target]},
        expected_regions={0: [target]},
        source_frame=synthetic_large_frame(5),
    )

    assert shifted_validation["passed"] is False
    assert shifted_validation["region_results"][0]["expected_render_matches_clip"] is False
    assert source_mismatch_validation["passed"] is False
    assert source_mismatch_validation["region_results"][0]["expected_render_matches_clip"] is False


def test_rendered_region_application_rejects_materially_wrong_privacy_transform(
    tmp_path: Path,
) -> None:
    source = synthetic_large_frame(0)
    target = review_bundle.rendering.RenderRegion(
        frame_index=0,
        bbox_xyxy=(40, 40, 80, 80),
        provenance="target",
        state="covered",
        source_record={},
    )
    settings = review_bundle.ReviewRenderConfig(
        margin_fraction=0.5,
        pixel_size=8,
    ).render_settings()
    source_path = write_synthetic_video(tmp_path / "source.mp4", [source])
    original_path = write_synthetic_video(tmp_path / "original.mp4", [source])
    wrong = source.copy()
    x1, y1, x2, y2 = review_bundle.rendering.expanded_clipped_box(
        target.bbox_xyxy,
        frame_width=source.shape[1],
        frame_height=source.shape[0],
        margin_pixels=settings.margin_pixels,
        margin_fraction=settings.margin_fraction,
    )
    wrong[y1:y2, x1:x2] = 0
    rendered_path = write_synthetic_video(tmp_path / "wrong_transform.mp4", [wrong])

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=original_path,
        rendered_clip_path=rendered_path,
        frame_mapping=review_bundle.TaskFrameMapping(0, 0),
        regions_by_frame={0: [target]},
        settings=settings,
        source_video_path=source_path,
    )

    assert validation["passed"] is False
    assert validation["region_results"][0]["expected_render_matches_clip"] is False


def test_rendered_region_application_fails_when_expected_bbox_is_unchanged(
    tmp_path: Path,
) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(8)],
    )
    mapping = review_bundle.TaskFrameMapping(3, 5)
    settings = review_bundle.ReviewRenderConfig(pixel_size=4).render_settings()
    region = review_bundle.rendering.RenderRegion(
        frame_index=4,
        bbox_xyxy=(10, 10, 30, 30),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    source_clip = tmp_path / "source_clip.mp4"
    rendered_clip = tmp_path / "rendered_clip.mp4"
    config = review_bundle.ReviewRenderConfig(pixel_size=4)
    for path in (source_clip, rendered_clip):
        review_bundle.render_review_clip_by_frames(
            source,
            path,
            3,
            5,
            fps=30.0,
            width=64,
            height=48,
            source_has_audio=False,
            role="original",
            regions_by_frame={},
            settings=None,
            privacy_regions_path=None,
            interval_decisions_path=None,
            renderer_config=config.payload(),
            renderer_config_hash=config.stable_hash(),
            overwrite=True,
        )

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=source_clip,
        rendered_clip_path=rendered_clip,
        frame_mapping=mapping,
        regions_by_frame={4: [region]},
        settings=settings,
    )

    assert validation["passed"] is False
    assert validation["missing_rendered_region_frames"] == [4]


def test_rendered_region_application_detects_temporal_shift(tmp_path: Path) -> None:
    source = write_synthetic_video(
        tmp_path / "source.mp4",
        [synthetic_frame(index) for index in range(8)],
    )
    mapping = review_bundle.TaskFrameMapping(3, 5)
    settings = review_bundle.ReviewRenderConfig(pixel_size=4).render_settings()
    expected_region = review_bundle.rendering.RenderRegion(
        frame_index=4,
        bbox_xyxy=(10, 10, 30, 30),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    shifted_region = review_bundle.rendering.RenderRegion(
        frame_index=5,
        bbox_xyxy=(10, 10, 30, 30),
        provenance="ordinary_tracker",
        state="covered",
        source_record={},
    )
    source_clip = tmp_path / "source_clip.mp4"
    shifted_clip = tmp_path / "shifted_clip.mp4"
    config = review_bundle.ReviewRenderConfig(pixel_size=4)
    review_bundle.render_review_clip_by_frames(
        source,
        source_clip,
        3,
        5,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="original",
        regions_by_frame={},
        settings=None,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=config.payload(),
        renderer_config_hash=config.stable_hash(),
        overwrite=True,
    )
    review_bundle.render_review_clip_by_frames(
        source,
        shifted_clip,
        3,
        5,
        fps=30.0,
        width=64,
        height=48,
        source_has_audio=False,
        role="automatic",
        regions_by_frame={5: [shifted_region]},
        settings=settings,
        privacy_regions_path=None,
        interval_decisions_path=None,
        renderer_config=config.payload(),
        renderer_config_hash=config.stable_hash(),
        overwrite=True,
    )

    validation = review_bundle.validate_rendered_region_application(
        original_clip_path=source_clip,
        rendered_clip_path=shifted_clip,
        frame_mapping=mapping,
        regions_by_frame={4: [expected_region]},
        settings=settings,
    )

    assert validation["passed"] is False
    assert validation["missing_rendered_region_frames"] == [4]
    assert validation["unexpected_rendered_region_frames"] == [5]


def test_role_subtitles_warnings_and_corrected_action_text(tmp_path: Path) -> None:
    automatic = tmp_path / "clips" / "automatic.mp4"
    corrected = tmp_path / "clips" / "corrected.mp4"
    automatic.parent.mkdir(parents=True)
    automatic.write_bytes(b"automatic")
    corrected.write_bytes(b"corrected")
    item = raw_item(
        "blocker",
        10,
        12,
        ["needs_review_interval", "unresolved_privacy_frames"],
        blocking=True,
        severity="blocking",
    )
    item["visual_evidence"] = {
        "automatic_preview_clip": "clips/automatic.mp4",
        "reviewed_final_clip": "clips/corrected.mp4",
    }

    review_bundle.assign_media_roles([item], tmp_path)
    roles = {role["role"]: role for role in item["media_roles"]}
    task = review_bundle.reviewer_task_from_group(
        [item],
        index=1,
        case_id="synthetic",
        fps=30.0,
        raw_items=[item],
        raw_categories={},
        corrected_context={},
    )

    assert roles["automatic"]["subtitle"] == "Result before reviewed corrections"
    assert roles["automatic"]["warning"] == "Automatic result is not final."
    assert roles["corrected_final"]["subtitle"] == (
        "Same rendering style with reviewed corrections applied"
    )
    assert "Correct confirms the corrected final result shown above" in task["user_action_guidance"]


def test_duplicate_semantic_corrected_media_is_suppressed(tmp_path: Path) -> None:
    automatic = tmp_path / "clips" / "automatic.mp4"
    corrected = tmp_path / "clips" / "corrected.mp4"
    automatic.parent.mkdir(parents=True)
    automatic.write_bytes(b"same")
    corrected.write_bytes(b"same")
    item = {
        "visual_evidence": {
            "automatic_preview_clip": "clips/automatic.mp4",
            "reviewed_final_clip": "clips/corrected.mp4",
        }
    }

    review_bundle.assign_media_roles([item], tmp_path)
    roles = item["media_roles"]

    assert roles[0]["role"] == "automatic"
    assert roles[1]["role"] == "corrected_final"
    assert roles[1]["display"] is False
    assert roles[1]["duplicate_of"] == "automatic"


def test_preserved_region_provenance_count_and_interval_coverage(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "corrected.jsonl",
        [
            record(10, [1, 1, 5, 5], "detector"),
            record(11, [1, 1, 5, 5], "preserved_privacy_region"),
            record(12, [1, 1, 5, 5], "preserved_privacy_region"),
        ],
    )
    privacy_input = review_bundle.rendering.load_privacy_regions(path)

    coverage = review_bundle.privacy_coverage_for_interval(privacy_input, 10, 12)

    assert coverage["covered_frame_count"] == 3
    assert coverage["provenance_counts"] == {"detector": 1, "preserved_privacy_region": 2}
    assert review_bundle.corrected_final_is_traceable(privacy_input, 10, 12) is True


def test_review_media_audit_payload_is_deterministic() -> None:
    queue = {
        "case_id": "synthetic",
        "items": [
            {
                "media_comparison_audit": {
                    "roles": {"automatic": role_audit("automatic")},
                    "validation": {"passed": True},
                },
                "review_item_id": "item_a",
            }
        ],
        "reviewer_tasks": [{"media_roles": [], "reviewer_task_id": "task_a"}],
    }

    assert review_bundle.media_audit_payload(queue) == review_bundle.media_audit_payload(queue)


def test_reviewer_actions_match_task_type() -> None:
    action_required = raw_item(
        "blocker",
        1,
        2,
        ["needs_review_interval", "unresolved_privacy_frames"],
        blocking=True,
        severity="blocking",
    )
    corrected = raw_item(
        "corrected",
        10,
        12,
        ["needs_review_interval", "unresolved_privacy_frames"],
        blocking=True,
        severity="blocking",
        media_roles=media_roles("original", "automatic", "corrected_final"),
    )
    optional = raw_item("random", 20, 24, ["random_audit_sample"], severity="random_audit")

    action_task = review_bundle.reviewer_task_from_group(
        [action_required],
        index=1,
        case_id="synthetic",
        fps=30.0,
        raw_items=[action_required],
        raw_categories={},
        corrected_context={},
    )
    corrected_task = review_bundle.reviewer_task_from_group(
        [corrected],
        index=2,
        case_id="synthetic",
        fps=30.0,
        raw_items=[corrected],
        raw_categories={},
        corrected_context={},
    )
    optional_task = review_bundle.reviewer_task_from_group(
        [optional],
        index=3,
        case_id="synthetic",
        fps=30.0,
        raw_items=[optional],
        raw_categories={},
        corrected_context={},
    )

    assert "looks_correct" not in {
        action["action"] for action in action_task["allowed_user_actions"]
    }
    assert "looks_correct" in {
        action["action"] for action in corrected_task["allowed_user_actions"]
    }
    assert "no_issue_found" in {
        action["action"] for action in optional_task["allowed_user_actions"]
    }


def test_task_decision_export_validation(tmp_path: Path) -> None:
    raw = raw_item("raw_a", 1, 2, ["needs_review_interval"], blocking=True)
    task = {
        "allowed_user_actions": [
            {
                "action": "looks_correct",
                "review_status": "accepted_current",
                "reviewed_media_role": "corrected_final",
            }
        ],
        "category": "corrected_result_check",
        "frame_end": 2,
        "frame_start": 1,
        "raw_review_item_ids": ["raw_a"],
        "requires_decision": True,
        "reviewer_task_id": "task_a",
        "schema_version": "1.1",
    }
    queue = write_json(
        tmp_path / "review_queue.json",
        {
            "case_id": "synthetic",
            "items": [raw],
            "reviewer_tasks": [task],
            "schema_version": "1.0",
        },
    )
    decisions = write_json(
        tmp_path / "decisions.json",
        {
            "decisions": [
                {
                    "raw_review_item_ids": ["raw_a"],
                    "review_status": "accepted_current",
                    "reviewed_media_role": "corrected_final",
                    "reviewer_task_id": "task_a",
                    "selected_user_action": "looks_correct",
                }
            ],
            "schema_version": "1.1",
        },
    )

    result = review_bundle.validate_review_decisions(queue, decisions)

    assert result["status"] == "valid"
    assert result["final_export_blocked"] is False


def test_v1_1_html_contains_guided_workflow_sections() -> None:
    html = review_bundle.review_html(
        {
            "case_id": "synthetic",
            "items": [],
            "reviewer_tasks": [],
            "source_video": {"hash": "hash"},
        },
        {
            "corrected_result_tasks": 0,
            "final_export_blocked": False,
            "optional_spot_checks": 0,
            "required_review_tasks": 0,
            "technical_only_items": 0,
        },
    )

    assert "Guided review" in html
    assert "View all tasks" in html
    assert "Technical audit" in html
    css = review_bundle.review_css()
    assert ".media-grid.cols-2" in css
    assert ".media-grid.cols-3" in css


def test_queue_generation_from_needs_review_and_unresolved_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch))
    codes = reason_codes(queue)

    assert "needs_review_interval" in codes
    assert "unresolved_privacy_frames" in codes
    assert queue["items"][0]["severity"] == "blocking"


def test_rejected_intervals_are_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch, interval_status="rejected"))

    assert "rejected_interval" in reason_codes(queue)
    assert any(item["blocking_status"] for item in queue["items"])


def test_unresolved_and_invalid_boxes_are_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch))
    codes = reason_codes(queue)

    assert "unresolved_privacy_frames" in codes
    assert "invalid_box" in codes


def test_evaluator_only_ground_truth_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch, include_evaluation=True))
    codes = reason_codes(queue)

    assert "ground_truth_detector_miss" in codes
    assert "ground_truth_privacy_miss" in codes


def test_queue_generation_without_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch, include_evaluation=False))

    assert "ground_truth_detector_miss" not in reason_codes(queue)
    assert any(item["item_origin"] == "pipeline" for item in queue["items"])


def test_merging_and_deduplication_of_nearby_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch))
    blocking = [item for item in queue["items"] if item["blocking_status"]]
    merged = [
        item
        for item in blocking
        if {"needs_review_interval", "unresolved_privacy_frames"} <= set(item["reason_codes"])
    ]

    assert len(merged) == 1
    assert all("Additional related evidence" not in item["explanation"] for item in queue["items"])
    assert queue["reviewer_tasks"][0]["explanation"].count("privacy-required frames") == 1


def test_distinct_needs_review_intervals_are_not_merged() -> None:
    drafts = [
        review_bundle.ReviewDraft(
            item_origin="pipeline",
            source_stage="privacy_gap_recovery",
            frame_start=10,
            frame_end=12,
            reason_codes={"needs_review_interval"},
            severity="blocking",
            explanation="first",
        ),
        review_bundle.ReviewDraft(
            item_origin="pipeline",
            source_stage="privacy_gap_recovery",
            frame_start=15,
            frame_end=16,
            reason_codes={"needs_review_interval"},
            severity="blocking",
            explanation="second",
        ),
    ]

    merged = review_bundle.merge_review_drafts(drafts, merge_gap_frames=10)

    assert [(draft.frame_start, draft.frame_end) for draft in merged] == [(10, 12), (15, 16)]


def test_evaluation_failure_frame_is_preserved() -> None:
    drafts = review_bundle.evaluation_failure_drafts(
        {
            "failures": {
                "failures": [
                    {
                        "evaluated_stage": "detector",
                        "failure_reason": "detector_scored_frame_without_adjudicated_candidate",
                        "frame": 42,
                        "object_id": "face_a",
                    }
                ]
            }
        }
    )

    assert [(draft.frame_start, draft.frame_end) for draft in drafts] == [(42, 42)]


def test_severity_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch, include_evaluation=True))
    severities = [item["severity"] for item in queue["items"]]

    assert severities == sorted(severities, key=lambda value: review_bundle.SEVERITY_ORDER[value])


def test_deterministic_random_sampling_and_blocking_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = load_queue(build_bundle(tmp_path / "first", monkeypatch))
    second = load_queue(build_bundle(tmp_path / "second", monkeypatch))
    first_random = [item for item in first["items"] if item["severity"] == "random_audit"]
    second_random = [item for item in second["items"] if item["severity"] == "random_audit"]
    blocking_frames = {
        frame
        for item in first["items"]
        if item["blocking_status"]
        for frame in range(item["frame_start"], item["frame_end"] + 1)
    }

    assert [(item["frame_start"], item["frame_end"]) for item in first_random] == [
        (item["frame_start"], item["frame_end"]) for item in second_random
    ]
    assert all(
        not (set(range(item["frame_start"], item["frame_end"] + 1)) & blocking_frames)
        for item in first_random
    )


def test_provenance_transition_and_tracking_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(build_bundle(tmp_path, monkeypatch, include_tracking=True))
    codes = reason_codes(queue)

    assert "suspicious_provenance_transition" in codes
    assert "low_confidence_observations" in codes
    assert "weak_only_observations" in codes
    assert "track_start" in codes
    assert "track_end" in codes
    assert "long_detection_gap" in codes


def test_relative_artifact_paths_and_local_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle(tmp_path, monkeypatch)

    assert review_bundle.assert_no_external_assets(bundle) == []
    assert review_bundle.assert_relative_bundle_paths(bundle) == []
    html = (bundle / "index.html").read_text(encoding="utf-8")
    assert "https://" not in html
    assert "items need your review" in html


def test_valid_decision_export_schema_for_nonblocking_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle(tmp_path, monkeypatch)
    queue = load_queue(bundle)
    random_item = next(item for item in queue["items"] if item["severity"] == "random_audit")
    decisions = write_json(
        bundle / "decisions.json",
        {
            "schema_version": "1.0",
            "decisions": [
                {
                    "review_item_id": random_item["review_item_id"],
                    "review_status": "accepted_current",
                }
            ],
        },
    )

    result = review_bundle.validate_review_decisions(bundle / "review_queue.json", decisions)

    assert result["status"] == "valid"


def test_invalid_and_duplicate_decision_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle(tmp_path, monkeypatch)
    item_id = load_queue(bundle)["items"][0]["review_item_id"]
    decisions = write_json(
        bundle / "bad_decisions.json",
        {
            "schema_version": "1.0",
            "decisions": [
                {"review_item_id": item_id, "review_status": "accepted_current"},
                {"review_item_id": item_id, "review_status": "accepted_current"},
                {"review_item_id": "missing", "review_status": "accepted_current"},
            ],
        },
    )

    result = review_bundle.validate_review_decisions(bundle / "review_queue.json", decisions)

    assert result["status"] == "invalid"
    assert any("duplicate decision" in error for error in result["errors"])
    assert any("unknown review_item_id" in error for error in result["errors"])
    assert any("not allowed" in error for error in result["errors"])


def test_unresolved_blocker_and_correction_required_preserve_final_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle(tmp_path, monkeypatch)
    blocker = next(item for item in load_queue(bundle)["items"] if item["blocking_status"])
    decisions = write_json(
        bundle / "decisions.json",
        {
            "schema_version": "1.0",
            "decisions": [
                {
                    "review_item_id": blocker["review_item_id"],
                    "review_status": "correction_required",
                }
            ],
        },
    )

    result = review_bundle.validate_review_decisions(bundle / "review_queue.json", decisions)

    assert result["final_export_blocked"] is True
    assert result["unresolved_blocking_item_count"] > 0


def test_empty_clean_run_and_missing_optional_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue(
        build_bundle(
            tmp_path,
            monkeypatch,
            include_privacy=False,
            random_sample_count=0,
        )
    )

    assert queue["items"] == []


def test_deterministic_summary_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_bundle = build_bundle(tmp_path / "first", monkeypatch)
    second_bundle = build_bundle(tmp_path / "second", monkeypatch)

    first = json.loads((first_bundle / "review_summary.json").read_text(encoding="utf-8"))
    second = json.loads((second_bundle / "review_summary.json").read_text(encoding="utf-8"))

    assert first == second


def test_reviewer_finding_becomes_corrected_result_task_and_hides_source_random() -> None:
    random_item = raw_item(
        "random",
        290,
        313,
        ["random_audit_sample"],
        severity="random_audit",
        media_roles=media_roles("original", "automatic", "corrected_final"),
    )
    finding = raw_item(
        "finding",
        290,
        299,
        ["reviewer_found_privacy_miss", "silent_miss_found_by_random_audit"],
        blocking=True,
        severity="blocking",
        media_roles=media_roles("original", "automatic", "corrected_final"),
    )
    queue = {
        "case_id": "synthetic",
        "corrected_context": {},
        "items": [random_item, finding],
        "source_video": {"hash": "hash"},
    }

    payload = review_bundle.reviewer_tasks_payload(
        queue,
        config=review_bundle.ReviewBundleConfig(),
        fps=30.0,
    )

    tasks = payload["reviewer_tasks"]
    assert [task["category"] for task in tasks] == ["corrected_result_check"]
    assert tasks[0]["primary_raw_review_item_ids"] == ["finding"]
    assert set(tasks[0]["raw_review_item_ids"]) == {"finding", "random"}
    assert payload["technical_audit"]["hidden_from_default_task_count"] == 1


def test_reviewer_finding_draft_records_random_audit_provenance() -> None:
    drafts = review_bundle.reviewer_finding_drafts(
        {
            "findings": [
                {
                    "affected_frames": [10, 11],
                    "correction_status": "candidate_corrected_regions_available",
                    "finding_id": "finding_a",
                    "reason_codes": ["reviewer_found_privacy_miss"],
                    "reviewer_finding_type": "silent_miss_found_by_random_audit",
                    "source_frame_end": 11,
                    "source_frame_start": 10,
                    "source_review_item_id": "raw_random",
                    "source_reviewer_task_id": "task_random",
                    "target_track_id": 6,
                }
            ]
        }
    )

    assert len(drafts) == 1
    assert drafts[0].blocking is True
    assert drafts[0].requires_corrected_regions is True
    assert drafts[0].candidate_region_status == "available"
    assert drafts[0].metrics["source_reviewer_task_id"] == "task_random"
    assert drafts[0].provenance["before"] == "random_audit"


def test_optional_random_audit_needs_review_decision_blocks_final(tmp_path: Path) -> None:
    random_item = raw_item("random", 20, 24, ["random_audit_sample"], severity="random_audit")
    task = review_bundle.reviewer_task_from_group(
        [random_item],
        index=1,
        case_id="synthetic",
        fps=30.0,
        raw_items=[random_item],
        raw_categories={},
        corrected_context={},
    )
    queue = write_json(
        tmp_path / "review_queue.json",
        {
            "case_id": "synthetic",
            "items": [random_item],
            "reviewer_tasks": [task],
            "schema_version": "1.0",
        },
    )
    decisions = write_json(
        tmp_path / "decisions.json",
        {
            "decisions": [
                {
                    "raw_review_item_ids": [random_item["review_item_id"]],
                    "review_status": "escalated",
                    "reviewed_media_role": "original_context",
                    "reviewer_task_id": task["reviewer_task_id"],
                    "selected_user_action": "needs_review",
                }
            ],
            "schema_version": review_bundle.REVIEW_TASK_SCHEMA_VERSION,
        },
    )

    result = review_bundle.validate_review_decisions(queue, decisions)

    assert result["status"] == "valid"
    assert result["final_export_blocked"] is True
    assert result["unresolved_blocking_items"] == [task["reviewer_task_id"]]


def test_reviewer_discovered_correction_is_traceable(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "privacy_regions.jsonl",
        [record(10, [1, 2, 9, 10], "reviewer_discovered_correction")],
    )
    privacy_input = review_bundle.rendering.load_privacy_regions(path)

    assert review_bundle.corrected_final_is_traceable(privacy_input, 10, 10) is True


def test_review_bundle_server_is_loopback_and_serves_mp4_mime(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text("<!doctype html><title>Review</title>", encoding="utf-8")
    (bundle / "clip.mp4").write_bytes(b"not really a movie")

    resolved = review_bundle.resolve_bundle_request_path(bundle, "/../clip.mp4")
    assert resolved == (bundle / "clip.mp4").resolve()
    server, url = review_bundle.create_review_bundle_server(bundle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        clip_url = url.replace("/index.html", "/clip.mp4")
        with urllib.request.urlopen(clip_url, timeout=5) as response:
            assert response.headers.get_content_type() == "video/mp4"
            assert response.read() == b"not really a movie"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_review_js_contains_playback_and_auto_advance_hooks() -> None:
    script = review_bundle.review_js()

    assert "Promise.allSettled" in script
    assert "window.setTimeout" in script
    assert "file-protocol-note" in script
    assert "selected-action" in script
    assert "Clear local choices for this review bundle" in script


def test_review_js_group_media_controls_are_scoped_to_guided_task() -> None:
    script = review_bundle.review_js()

    assert "function activeTaskCard()" in script
    assert "document.querySelector('#guided-panel .task-card')" in script
    assert "[...card.querySelectorAll('video')]" in script
    assert "mode !== 'guided'" in script
    assert "Group media controls are available in Guided review." in script
    assert "document.querySelectorAll('video')" not in script


def test_review_js_group_media_controls_handle_play_pause_and_restart() -> None:
    script = review_bundle.review_js()

    assert "async function syncMedia(event)" in script
    assert "async function restartMedia(event)" in script
    assert "waitForMetadata(video)" in script
    assert "video.muted = true" in script
    assert "Promise.allSettled(videos.map(video => video.play()))" in script
    assert "videos.forEach(video => video.pause())" in script
    assert "video.currentTime = 0" in script
    assert "syncButton.textContent = anyPlaying ? 'Pause together' : 'Play together'" in script
    assert "Embedded playback failed" in script


def test_review_html_keeps_native_controls_and_stable_group_buttons() -> None:
    html = review_bundle.review_html(
        {"case_id": "synthetic", "source_video": {"hash": "hash"}, "reviewer_tasks": []},
        {"final_export_blocked": False, "technical_only_items": 0},
    )

    assert '<button id="sync-media" class="secondary">Play together</button>' in html
    assert '<button id="restart-media" class="secondary">Restart media</button>' in html
    assert 'video controls playsinline preload="metadata"' in review_bundle.review_js()
