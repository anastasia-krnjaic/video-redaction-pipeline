from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from redactor import evaluation

SOURCE_BYTES = b"source-video"
RENDERED_BYTES = b"rendered-video"


def fixture_path() -> Path:
    return Path("tests/fixtures/redaction_ground_truth_synthetic.json")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return path


def copy_ground_truth(tmp_path: Path, mutator=None) -> Path:
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    if mutator is not None:
        mutator(payload)
    return write_json(tmp_path / "ground_truth.json", payload)


def source_video(tmp_path: Path, content: bytes = SOURCE_BYTES) -> Path:
    path = tmp_path / "source.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def detections_path(tmp_path: Path, *, empty: bool = False) -> Path:
    records = []
    hit_frames = {} if empty else {0: 0.90, 3: 0.80, 4: 0.31, 6: 0.70}
    for frame in range(10):
        detections = []
        if frame in hit_frames:
            detections.append(
                {
                    "bbox_xyxy": [10 + frame, 10, 30 + frame, 30],
                    "candidate_id": f"synthetic:f{frame:04d}:d00",
                    "category": "face",
                    "detector_id": "openvino_face_adas_0001",
                    "score": hit_frames[frame],
                }
            )
        records.append({"detections": detections, "frame_index": frame})
    return write_jsonl(tmp_path / "detections.jsonl", records)


def adjudication_path(tmp_path: Path, *, empty: bool = False) -> Path:
    items = []
    for frame in [0, 1, 2, 3, 4, 6]:
        choice = "none" if empty or frame in {1, 2} else "candidate"
        decision: dict[str, Any] = {"choice": choice}
        if choice == "candidate":
            decision.update(
                {
                    "bbox_xyxy": [10 + frame, 10, 30 + frame, 30],
                    "candidate_id": f"synthetic:f{frame:04d}:d00",
                    "score": {0: 0.90, 3: 0.80, 4: 0.31, 6: 0.70}[frame],
                }
            )
        items.append(
            {
                "clip": "synthetic_clip",
                "target_id": "redact_face",
                "frame": frame,
                "decision": decision,
            }
        )
    for frame in range(10):
        items.append(
            {
                "clip": "synthetic_clip",
                "target_id": "keep_face",
                "frame": frame,
                "decision": {
                    "bbox_xyxy": [60, 10, 82, 32],
                    "candidate_id": f"keep:f{frame:04d}:d00",
                    "choice": "candidate",
                    "score": 0.9,
                },
            }
        )
    return write_json(tmp_path / "adjudication.json", {"items": items})


def tracks_path(tmp_path: Path) -> Path:
    records = []
    for frame in range(10):
        boxes = [
            {
                "bbox_xyxy": [10 + frame, 10, 30 + frame, 30],
                "candidate_id": f"synthetic:f{frame:04d}:d00",
                "provenance": "observed_primary",
                "quality_status": "auto_confirmed",
                "score": 0.8,
                "track_id": 1,
            },
            {
                "bbox_xyxy": [60, 10, 82, 32],
                "candidate_id": f"keep:f{frame:04d}:d00",
                "provenance": "observed_primary",
                "quality_status": "auto_confirmed",
                "score": 0.9,
                "track_id": 2,
            },
        ]
        records.append({"frame_index": frame, "track_boxes": boxes})
    return write_jsonl(tmp_path / "tracks.jsonl", records)


def privacy_regions_path(tmp_path: Path, *, overlap_keep: bool = False) -> Path:
    records = [
        privacy_record(0, [10, 10, 30, 30], "detector"),
        privacy_record(1, [11, 10, 31, 30], "ordinary_tracker"),
        privacy_record(2, [12, 10, 32, 30], "bidirectional_privacy_bridge"),
        privacy_record(3, [13, 10, 33, 30], "preserved_privacy_region"),
        privacy_record(4, [14, 10, 34, 30], "bidirectional_privacy_bridge"),
        privacy_record(5, None, "bidirectional_privacy_bridge", state="unresolved"),
        privacy_record(6, [16, 10, 36, 30], "ordinary_tracker"),
    ]
    if overlap_keep:
        records.append(privacy_record(7, [60, 10, 82, 32], "ordinary_tracker"))
    return write_jsonl(tmp_path / "privacy_regions.jsonl", records)


def privacy_record(
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
                "target_track_ids": [1],
            }
        ],
    }


def intervals_path(tmp_path: Path, *, needs_review: bool = True) -> Path:
    intervals = [
        {
            "frame_start": 2,
            "frame_end": 2,
            "status": "accepted_after_refinement",
            "strategy": "bidirectional",
        },
    ]
    if needs_review:
        intervals.append(
            {
                "frame_start": 4,
                "frame_end": 5,
                "status": "needs_review",
                "strategy": "bidirectional",
            }
        )
    return write_json(tmp_path / "interval_decisions.json", {"interval_decisions": intervals})


def render_manifest_path(
    tmp_path: Path, *, source_hash: str, privacy_hash: str, regions: int = 10
) -> Path:
    return write_json(
        tmp_path / "render_manifest.json",
        {
            "final_approval_status": True,
            "output_video_hash": "5c7102036ad65a84771fe6a5292bec988e56e4fd2eb84ccb122b4d2f3ea40dbb",
            "privacy_region_input_hash": privacy_hash,
            "rendered_region_count": regions,
            "rendering_mode": "final",
            "source_video_hash": source_hash,
        },
    )


def render_validation_path(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "validation.json",
        {"media_validation": {"audio_ok": True, "passed": True}},
    )


def run_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = evaluation.MODE_DETECTOR,
    empty_detections: bool = False,
    needs_review: bool = True,
    overlap_keep: bool = False,
    render_manifest: Path | None = None,
    render_validation: Path | None = None,
    rendered_video: Path | None = None,
) -> dict[str, Any]:
    monkeypatch.setattr(evaluation, "write_failure_visuals", lambda **_: {})
    gt = copy_ground_truth(tmp_path)
    source = source_video(tmp_path)
    detections = detections_path(tmp_path, empty=empty_detections)
    adjudication = adjudication_path(tmp_path, empty=empty_detections)
    tracks = tracks_path(tmp_path)
    regions = privacy_regions_path(tmp_path, overlap_keep=overlap_keep)
    intervals = intervals_path(tmp_path, needs_review=needs_review)
    return evaluation.run_face_evaluation(
        ground_truth_path=gt,
        clip_id="synthetic_clip",
        mode=mode,
        source_video_path=source,
        detections_path=detections,
        output_dir=tmp_path / "out",
        adjudication_path=adjudication,
        tracks_path=tracks,
        privacy_regions_path=regions,
        interval_decisions_path=intervals,
        rendered_video_path=rendered_video,
        render_manifest_path=render_manifest,
        render_validation_path=render_validation,
        overwrite=True,
    )


def test_schema_validation_accepts_synthetic_fixture(tmp_path: Path) -> None:
    result = evaluation.validate_ground_truth_file(
        fixture_path(),
        source_video_paths={"synthetic_clip": source_video(tmp_path)},
    )

    assert result["status"] == "valid"
    assert result["error_count"] == 0


def test_invalid_frame_ranges_are_rejected(tmp_path: Path) -> None:
    gt = copy_ground_truth(
        tmp_path,
        lambda payload: payload["clips"][0]["objects"][0].__setitem__(
            "detector_scored_ranges", [[8, 99]]
        ),
    )

    result = evaluation.validate_ground_truth_file(gt)

    assert result["status"] == "invalid"
    assert any("outside source frame bounds" in error for error in result["errors"])


def test_contradictory_detector_ranges_are_rejected(tmp_path: Path) -> None:
    gt = copy_ground_truth(
        tmp_path,
        lambda payload: payload["clips"][0]["objects"][0].__setitem__(
            "detector_ignore_ranges", [[4, 5]]
        ),
    )

    result = evaluation.validate_ground_truth_file(gt)

    assert any(
        "detector_scored_ranges overlap detector_ignore_ranges" in error
        for error in result["errors"]
    )


def test_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    result = evaluation.validate_ground_truth_file(
        fixture_path(),
        source_video_paths={"synthetic_clip": source_video(tmp_path, b"wrong")},
    )

    assert result["status"] == "invalid"
    assert result["errors"] == ["synthetic_clip: source hash mismatch"]


def test_detector_metrics_use_detector_scored_frames_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = run_eval(tmp_path, monkeypatch)
    detector = metrics["detector"]
    obj = json.loads((tmp_path / "out" / "object_metrics.json").read_text())["objects"][
        "redact_face"
    ]

    assert detector["frames_with_valid_detection"] == 4
    assert detector["detector_scored_object_frames"] == 16
    assert obj["detector"]["detector_scored_object_frames"] == 6
    assert obj["detector"]["missed_detector_scored_frames"] == [1, 2]
    assert obj["detector"]["longest_consecutive_missed_run"] == 2
    assert obj["ground_truth"]["detector_ignore_frames"] == 1
    assert obj["privacy"]["not_applicable"] is True


def test_empty_detections_report_zero_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_eval(tmp_path, monkeypatch, empty_detections=True)
    obj = json.loads((tmp_path / "out" / "object_metrics.json").read_text())["objects"][
        "redact_face"
    ]

    assert obj["detector"]["frames_with_valid_detection"] == 0
    assert obj["detector"]["object_frame_recall"] == 0.0


def test_automatic_mode_excludes_reviewed_corrections_and_blocks_needs_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = run_eval(tmp_path, monkeypatch, mode=evaluation.MODE_AUTOMATIC)
    obj = json.loads((tmp_path / "out" / "object_metrics.json").read_text())["objects"][
        "redact_face"
    ]

    assert metrics["acceptance"]["final_status"] == evaluation.STATUS_REVIEW_REQUIRED
    assert "needs_review_intervals" in metrics["acceptance"]["blocking_reasons"]
    assert obj["privacy"]["privacy_required_frames"] == 10
    assert obj["privacy"]["covered_frame_count"] == 4
    assert obj["privacy"]["preserved_or_manual_frame_count"] == 0
    assert obj["privacy"]["detector_scored_visible_uncovered_frames"] == [3, 4]
    assert obj["privacy"]["unresolved_input_privacy_frame_count"] == 1


def test_reviewed_final_counts_preserved_regions_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regions = privacy_regions_path(tmp_path)
    source = source_video(tmp_path)
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(RENDERED_BYTES)
    manifest = render_manifest_path(
        tmp_path,
        source_hash="04c767fd3d68b42476a64bb0c0edec569938a96cf2af218e0af8973e52792233",
        privacy_hash=evaluation.media.sha256_file(regions),
        regions=4,
    )
    monkeypatch.setattr(evaluation, "write_failure_visuals", lambda **_: {})
    metrics = evaluation.run_face_evaluation(
        ground_truth_path=copy_ground_truth(tmp_path),
        clip_id="synthetic_clip",
        mode=evaluation.MODE_REVIEWED_FINAL,
        source_video_path=source,
        detections_path=detections_path(tmp_path),
        output_dir=tmp_path / "out",
        adjudication_path=adjudication_path(tmp_path),
        tracks_path=tracks_path(tmp_path),
        privacy_regions_path=regions,
        interval_decisions_path=intervals_path(tmp_path, needs_review=False),
        rendered_video_path=rendered,
        render_manifest_path=manifest,
        render_validation_path=render_validation_path(tmp_path),
        overwrite=True,
    )
    obj = json.loads((tmp_path / "out" / "object_metrics.json").read_text())["objects"][
        "redact_face"
    ]

    assert metrics["acceptance"]["final_status"] == evaluation.STATUS_FAILED_PRIVACY_ACCEPTANCE
    assert obj["privacy"]["preserved_or_manual_frame_count"] == 1
    assert obj["privacy"]["provenance_counts"]["preserved_privacy_region"] == 1


def test_final_approval_blocks_manifest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(RENDERED_BYTES)
    manifest = render_manifest_path(
        tmp_path, source_hash="0" * 64, privacy_hash="0" * 64, regions=4
    )
    metrics = run_eval(
        tmp_path,
        monkeypatch,
        mode=evaluation.MODE_REVIEWED_FINAL,
        needs_review=False,
        render_manifest=manifest,
        render_validation=render_validation_path(tmp_path),
        rendered_video=rendered,
    )

    assert metrics["acceptance"]["final_status"] == evaluation.STATUS_FAILED_PRIVACY_ACCEPTANCE
    assert "render_manifest_inconsistent" in metrics["acceptance"]["blocking_reasons"]
    failures = json.loads((tmp_path / "out" / "failures.json").read_text())["failures"]
    assert any(item["failure_reason"] == "source_video_hash_mismatch" for item in failures)


def test_keep_person_false_redaction_is_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = run_eval(
        tmp_path,
        monkeypatch,
        mode=evaluation.MODE_AUTOMATIC,
        needs_review=False,
        overlap_keep=True,
    )

    assert "keep_person_false_redaction" in metrics["acceptance"]["blocking_reasons"]


def test_deterministic_object_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = run_eval(tmp_path / "first", monkeypatch, mode=evaluation.MODE_AUTOMATIC)
    second = run_eval(tmp_path / "second", monkeypatch, mode=evaluation.MODE_AUTOMATIC)
    first_objects = json.loads((tmp_path / "first" / "out" / "object_metrics.json").read_text())
    second_objects = json.loads((tmp_path / "second" / "out" / "object_metrics.json").read_text())

    assert first["acceptance"] == second["acceptance"]
    assert first_objects == second_objects


def test_category_evaluation_adapter_preserves_face_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_face_impl(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(evaluation, "_run_face_evaluation_impl", fake_face_impl)
    metrics = evaluation.run_face_evaluation(
        ground_truth_path=tmp_path / "ground_truth.json",
        clip_id="synthetic_clip",
        mode=evaluation.MODE_AUTOMATIC,
        source_video_path=tmp_path / "source.mp4",
        detections_path=tmp_path / "detections.jsonl",
        output_dir=tmp_path / "out",
        detector_threshold=0.42,
        overwrite=True,
    )

    assert metrics == {"status": "ok"}
    assert captured["clip_id"] == "synthetic_clip"
    assert captured["mode"] == evaluation.MODE_AUTOMATIC
    assert captured["source_video_path"] == tmp_path / "source.mp4"
    assert captured["detections_path"] == tmp_path / "detections.jsonl"
    assert captured["detector_threshold"] == 0.42
    assert captured["overwrite"] is True


def test_plate_evaluation_reports_capability_gap_without_scores(tmp_path: Path) -> None:
    metrics = evaluation.run_plate_evaluation(
        ground_truth_path=Path("data/ground_truth/redaction_ground_truth_v1.0.json"),
        clip_id="day_face",
        output_dir=tmp_path / "plate_eval",
    )
    written_metrics = json.loads((tmp_path / "plate_eval" / "metrics.json").read_text())
    report = (tmp_path / "plate_eval" / "report.md").read_text()

    assert metrics["category"] == "license_plate"
    assert metrics["publishable_metrics"] is False
    assert metrics["detector"]["object_frame_recall"] is None
    assert metrics["privacy"]["privacy_coverage"] is None
    assert "machine_readable_ground_truth_has_no_license_plate_objects" in metrics[
        "capability"
    ]["blockers"]
    assert written_metrics == metrics
    assert "not an evaluation scorecard" in report
