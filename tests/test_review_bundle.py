from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_review_item_schema_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = build_bundle(tmp_path, monkeypatch)
    queue = load_queue(bundle)

    assert review_bundle.validate_review_queue_payload(queue) == []
    assert queue["schema_id"] == review_bundle.REVIEW_QUEUE_SCHEMA_ID


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
    assert "REVIEW REQUIRED - NOT FINAL" in html


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
    assert blocker["review_item_id"] in result["unresolved_blocking_items"]


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
