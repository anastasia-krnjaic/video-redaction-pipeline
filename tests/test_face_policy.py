from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np
from click.testing import CliRunner

from redactor import face_policy, face_rendering, media, rendering
from redactor.cli import main


def synthetic_video(path: Path, *, frame_count: int = 5) -> Path:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("could not create synthetic video")
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), 30 + index * 20, dtype=np.uint8)
            cv.rectangle(frame, (10 + index, 8), (24 + index, 24), (200, 210, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_tracks(path: Path) -> Path:
    records = {
        0: [
            track_box(1, 0, [10, 10, 24, 24], "observed_strong", "auto_confirmed"),
            track_box(2, 0, [40, 10, 54, 24], "observed_strong", "auto_confirmed"),
        ],
        1: [],
        2: [
            track_box(1, 2, [12, 10, 26, 24], "observed_strong", "auto_confirmed"),
            track_box(2, 2, [42, 10, 56, 24], "observed_strong", "auto_confirmed"),
        ],
        3: [track_box(3, 3, [20, 20, 30, 30], "observed_weak", "review_candidate")],
        4: [],
    }
    with path.open("w", encoding="utf-8") as file:
        for frame_index in range(5):
            file.write(
                json.dumps(
                    {"frame_index": frame_index, "track_boxes": records.get(frame_index, [])},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return path


def track_box(
    track_id: int,
    frame_index: int,
    bbox: list[int],
    provenance: str,
    quality_status: str,
) -> dict[str, object]:
    return {
        "action": None,
        "bbox_xyxy": bbox,
        "candidate_id": f"candidate-{track_id}-{frame_index}",
        "detector_id": "openvino_face_adas_0001",
        "frame_index": frame_index,
        "provenance": provenance,
        "quality_status": quality_status,
        "render_eligible": quality_status == "auto_confirmed",
        "score": 0.8 if provenance == "observed_strong" else 0.24,
        "track_id": track_id,
        "track_state": "ended",
    }


def write_quality(path: Path) -> Path:
    payload = {
        "schema_version": "1.0",
        "tracks": [
            quality(1, 0, 2, "auto_confirmed"),
            quality(2, 0, 2, "auto_confirmed"),
            quality(3, 3, 3, "review_candidate", failures=["low_median_observed_score"]),
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def quality(
    track_id: int,
    first_frame: int,
    last_frame: int,
    quality_status: str,
    *,
    failures: list[str] | None = None,
) -> dict[str, object]:
    return {
        "first_frame": first_frame,
        "interpolated_boxes": 0,
        "last_frame": last_frame,
        "lifecycle_state": "ended",
        "median_area_fraction": 0.01,
        "median_observed_score": 0.8,
        "observed_hits": 2 if track_id in {1, 2} else 1,
        "predicted_boxes": 0,
        "quality_gate_failures": failures or [],
        "quality_status": quality_status,
        "render_eligible": quality_status == "auto_confirmed",
        "strong_hits": 2 if track_id in {1, 2} else 0,
        "synthetic_fraction": 0.0,
        "track_id": track_id,
        "weak_hits": 0 if track_id in {1, 2} else 1,
    }


def build_census(tmp_path: Path) -> tuple[dict[str, object], str, Path, Path, Path]:
    video = synthetic_video(tmp_path / "source.mp4")
    info = media.probe_media(video)
    frame_count = int(info.video_stream.frame_count or 5)
    tracks = write_tracks(tmp_path / "tracks.jsonl")
    quality_path = write_quality(tmp_path / "track_quality.json")
    census = face_policy.build_track_census(
        tracks_path=tracks,
        track_quality_path=quality_path,
        source_video_hash=info.sha256,
        source_frame_count=frame_count,
    )
    return census, info.sha256, video, tracks, quality_path


def test_redact_all_assigns_every_qualifying_face_track(tmp_path: Path) -> None:
    census, source_hash, _, _, _ = build_census(tmp_path)

    policy = face_policy.policy_from_census(
        case_id="case",
        source_video_hash=source_hash,
        source_frame_count=5,
        census=census,
    )

    assert policy.mode == "redact_all"
    assert {item.track_id for item in policy.assignments if item.decision == "redact"} == {
        1,
        2,
        3,
    }
    assert policy.unresolved_policy_assignments == (3,)


def test_policy_compile_uses_full_track_lifetime_without_ground_truth(tmp_path: Path) -> None:
    census, source_hash, video, tracks, quality_path = build_census(tmp_path)
    policy = face_policy.policy_from_census(
        case_id="case",
        source_video_hash=source_hash,
        source_frame_count=5,
        census=census,
        mode="redact_all",
    )

    result = face_policy.compile_policy(
        video,
        tracks_path=tracks,
        track_quality_path=quality_path,
        policy=policy,
    )

    track_one_frames = {
        frame
        for frame, regions in result.privacy_regions_by_frame.items()
        if any(1 in region.target_track_ids for region in regions)
    }
    assert track_one_frames == {0, 1, 2}
    assert result.unresolved_policy_assignments == (3,)


def test_redact_all_except_keep_excludes_explicit_keep_track(tmp_path: Path) -> None:
    census, source_hash, video, tracks, quality_path = build_census(tmp_path)
    policy = face_policy.policy_from_census(
        case_id="case",
        source_video_hash=source_hash,
        source_frame_count=5,
        census=census,
        mode="redact_all_except_keep",
        keep_track_ids=(2,),
    )

    result = face_policy.compile_policy(
        video,
        tracks_path=tracks,
        track_quality_path=quality_path,
        policy=policy,
    )

    assert result.keep_track_ids == (2,)
    redacted_ids = {
        track_id
        for regions in result.privacy_regions_by_frame.values()
        for region in regions
        for track_id in region.target_track_ids
    }
    assert 1 in redacted_ids
    assert 2 not in redacted_ids


def test_rejected_tracks_remain_blocking_review_without_automatic_region(tmp_path: Path) -> None:
    census, source_hash, video, tracks, quality_path = build_census(tmp_path)
    census["tracks"].append(
        {
            "first_frame": 4,
            "last_frame": 4,
            "policy_assignment": "review",
            "quality_gate_failures": ["insufficient_observed_hits"],
            "quality_status": "rejected",
            "render_eligible": False,
            "track_id": 99,
        }
    )
    policy = face_policy.policy_from_census(
        case_id="case",
        source_video_hash=source_hash,
        source_frame_count=5,
        census=census,
    )

    rejected_assignment = policy.assignment_for(99)
    assert rejected_assignment is not None
    assert rejected_assignment.decision == "review"
    assert 99 in policy.unresolved_policy_assignments

    result = face_policy.compile_policy(
        video,
        tracks_path=tracks,
        track_quality_path=quality_path,
        policy=policy,
    )

    assert all(
        99 not in region.target_track_ids
        for regions in result.privacy_regions_by_frame.values()
        for region in regions
    )
    assert any(
        decision.status == "needs_review" and decision.target_track_ids == (99,)
        for decision in result.interval_decisions
    )


def test_review_candidate_track_creates_blocking_interval_decision(tmp_path: Path) -> None:
    census, source_hash, video, tracks, quality_path = build_census(tmp_path)
    policy = face_policy.policy_from_census(
        case_id="case",
        source_video_hash=source_hash,
        source_frame_count=5,
        census=census,
    )

    result = face_policy.compile_policy(
        video,
        tracks_path=tracks,
        track_quality_path=quality_path,
        policy=policy,
    )

    assert any(
        decision.status == "needs_review" and decision.target_track_ids == (3,)
        for decision in result.interval_decisions
    )


def test_face_policy_validation_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    census, source_hash, video, _, _ = build_census(tmp_path)
    policy = face_policy.policy_from_census(
        case_id="case",
        source_video_hash=source_hash,
        source_frame_count=5,
        census=census,
    )
    payload = policy.to_dict() | {"source_video_hash": "not-the-video-hash"}
    policy_path = tmp_path / "privacy_policy.json"
    policy_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    result = face_policy.validate_policy_file(policy_path, source_video=video)

    assert result["status"] == "invalid"
    assert result["errors"] == ["source_video_hash_mismatch"]


def test_redact_video_help_is_full_video_policy_oriented() -> None:
    result = CliRunner().invoke(main, ["redact-video", "--help"])

    assert result.exit_code == 0
    assert "--categories" in result.output
    assert "--policy" in result.output
    assert "redact_all" in result.output
    assert "--target-track-id" not in result.output
    assert "--frame-start" not in result.output
    assert "--frame-end" not in result.output


def test_redact_video_runs_policy_flow_without_target_ids(
    tmp_path: Path,
    monkeypatch,
    synthetic_frontend_assets: Path,
) -> None:
    input_video = tmp_path / "input.mp4"
    synthetic_video(input_video)
    detections = tmp_path / "detections.jsonl"
    detections.write_text("", encoding="utf-8")
    output_dir = tmp_path / "case"
    captured: dict[str, object] = {}

    def fake_run_full_video_policy_flow(input_video_arg, **kwargs):
        captured["input_video"] = input_video_arg
        captured.update(kwargs)
        info = media.probe_media(input_video_arg)
        case_dir = Path(kwargs["output_dir"])
        source_frame_count = int(info.video_stream.frame_count or 0)

        def write_json(path: Path, payload: dict[str, object]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        write_json(
            case_dir / "source_manifest.json",
            {
                "fps": 5.0,
                "height": info.video_stream.height,
                "source_frame_count": source_frame_count,
                "source_video_hash": info.sha256,
                "width": info.video_stream.width,
            },
        )
        write_json(case_dir / "privacy_policy.json", {"mode": "redact_all"})
        write_json(case_dir / "track_census.json", {"summary": {}})
        write_json(case_dir / "tracking" / "track_quality.json", {"tracks": []})
        (case_dir / "tracking").mkdir(parents=True, exist_ok=True)
        (case_dir / "tracking" / "tracks.jsonl").write_text("", encoding="utf-8")
        (case_dir / "privacy_regions").mkdir(parents=True, exist_ok=True)
        (case_dir / "privacy_regions" / "privacy_regions.jsonl").write_text(
            "",
            encoding="utf-8",
        )
        write_json(case_dir / "privacy_regions" / "interval_decisions.json", {})
        write_json(case_dir / "review" / "review_tasks.json", {"tasks": []})
        (case_dir / "automatic_preview").mkdir(parents=True, exist_ok=True)
        preview_path = case_dir / "automatic_preview" / "redacted_preview.mp4"
        preview_path.write_bytes(Path(input_video_arg).read_bytes())
        write_json(
            case_dir / "automatic_preview" / "render_manifest.json",
            {
                "output_video_hash": media.sha256_file(preview_path),
                "rendered_region_count": 0,
                "source_video_hash": info.sha256,
            },
        )
        write_json(case_dir / "automatic_preview" / "validation.json", {"passed": True})
        write_json(
            case_dir / "final" / "final_status.json",
            {
                "blocking_reasons": ["review_required_policy_or_track_intervals"],
                "status": "blocked_review_required",
            },
        )
        write_json(
            case_dir / "audit_manifest.json",
            {"source_manifest": {"source_video_hash": info.sha256}},
        )
        return {
            "final_export_status": "blocked_review_required",
            "privacy_summary": {
                "redact_track_count": 0,
                "review_required_interval_count": 0,
            },
        }

    monkeypatch.setattr(face_policy, "run_full_video_policy_flow", fake_run_full_video_policy_flow)

    result = CliRunner().invoke(
        main,
        [
            "redact-video",
            str(input_video),
            "--detections",
            str(detections),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert captured["input_video"] == input_video
    assert captured["detections_path"] == detections
    assert captured["mode"] == "redact_all"
    assert captured["keep_track_ids"] == ()
    assert captured["selected_track_ids"] == ()


def test_render_faces_preview_uses_shared_renderer_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_video = tmp_path / "input.mp4"
    synthetic_video(input_video)
    privacy_regions = tmp_path / "privacy_regions.jsonl"
    privacy_regions.write_text(
        json.dumps(
            {
                "bbox_xyxy": [10, 10, 24, 24],
                "frame_index": 1,
                "provenance": "detector",
                "state": "covered",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "render"
    captured: dict[str, object] = {}

    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy face renderer called for automatic preview")

    def fake_shared(input_video_arg, **kwargs):
        captured["input_video"] = input_video_arg
        captured.update(kwargs)
        return {
            "final_approval_status": False,
            "output_video": "redacted_preview.mp4",
            "preview_non_final": True,
            "processed_frame_count": 5,
            "rendered_region_count": 1,
        }

    monkeypatch.setattr(rendering, "run_face_render", fail_legacy)
    monkeypatch.setattr(face_rendering, "run_shared_face_preview", fake_shared)

    result = CliRunner().invoke(
        main,
        [
            "render-faces",
            str(input_video),
            "--privacy-regions",
            str(privacy_regions),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert captured["input_video"] == input_video
    assert captured["privacy_regions_path"] == privacy_regions
    assert captured["output_dir"] == output_dir
    assert captured["settings"].mode == rendering.MODE_PREVIEW
    assert "non-final preview" in result.output


def test_render_faces_final_keeps_legacy_renderer_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_video = tmp_path / "input.mp4"
    synthetic_video(input_video)
    privacy_regions = tmp_path / "privacy_regions.jsonl"
    privacy_regions.write_text("", encoding="utf-8")
    output_dir = tmp_path / "render"
    captured: dict[str, object] = {}

    def fail_shared(*args, **kwargs):
        raise AssertionError("shared preview renderer called for final render")

    def fake_legacy(input_video_arg, **kwargs):
        captured["input_video"] = input_video_arg
        captured.update(kwargs)
        return {
            "final_approval_status": True,
            "output_video": "redacted_final.mp4",
            "preview_non_final": False,
            "processed_frame_count": 5,
            "rendered_region_count": 0,
        }

    monkeypatch.setattr(face_rendering, "run_shared_face_preview", fail_shared)
    monkeypatch.setattr(rendering, "run_face_render", fake_legacy)

    result = CliRunner().invoke(
        main,
        [
            "render-faces",
            str(input_video),
            "--privacy-regions",
            str(privacy_regions),
            "--output-dir",
            str(output_dir),
            "--mode",
            "final",
        ],
    )

    assert result.exit_code == 0
    assert captured["input_video"] == input_video
    assert captured["settings"].mode == rendering.MODE_FINAL
    assert "Final approval status: True" in result.output


def test_full_video_policy_flow_uses_shared_renderer_for_automatic_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_video = tmp_path / "input.mp4"
    synthetic_video(input_video)
    detections = tmp_path / "detections.jsonl"
    detections.write_text("", encoding="utf-8")
    output_dir = tmp_path / "case"
    captured: dict[str, object] = {}

    def fake_tracking(input_video_arg, *, output_dir: Path, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_tracks(output_dir / "tracks.jsonl")
        write_quality(output_dir / "track_quality.json")
        return {
            "auto_confirmed_tracks": 2,
            "schema_version": "tracking_summary.v1",
        }

    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy face renderer called for automatic preview")

    def fake_shared(input_video_arg, **kwargs):
        captured["input_video"] = input_video_arg
        captured.update(kwargs)
        render_dir = Path(kwargs["output_dir"])
        render_dir.mkdir(parents=True, exist_ok=True)
        (render_dir / "redacted_preview.mp4").write_bytes(b"preview")
        (render_dir / "render_manifest.json").write_text(
            json.dumps(
                {
                    "output_video": "redacted_preview.mp4",
                    "renderer_backend": face_rendering.RENDERER_BACKEND,
                    "source_video_hash": media.sha256_file(input_video_arg),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (render_dir / "validation.json").write_text(
            json.dumps({"media_validation": {"passed": True}}, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "final_approval_status": False,
            "output_video": "redacted_preview.mp4",
            "preview_non_final": True,
            "processed_frame_count": 5,
            "rendered_region_count": 4,
            "renderer_backend": face_rendering.RENDERER_BACKEND,
        }

    monkeypatch.setattr(face_policy.tracking, "run_tracking", fake_tracking)
    monkeypatch.setattr(face_policy.rendering, "run_face_render", fail_legacy)
    monkeypatch.setattr(face_policy.face_rendering, "run_shared_face_preview", fake_shared)
    monkeypatch.setattr(
        face_policy,
        "build_policy_review_bundle",
        lambda *args, **kwargs: {"summary": {"review_tasks": 1}},
    )

    audit = face_policy.run_full_video_policy_flow(
        input_video,
        output_dir=output_dir,
        detections_path=detections,
        overwrite=True,
    )

    assert captured["input_video"] == input_video
    assert captured["output_dir"] == output_dir / "automatic_preview"
    assert captured["settings"] == face_policy.DEFAULT_PREVIEW_RENDER_SETTINGS
    assert audit["render_summary"]["renderer_backend"] == face_rendering.RENDERER_BACKEND
    assert audit["final_export_status"] == "blocked_review_required"


def test_face_policy_source_has_no_benchmark_specific_logic() -> None:
    source = Path("src/redactor/face_policy.py").read_text(encoding="utf-8")

    for forbidden in [
        "person_day",
        "person_night",
        "eval_day_face",
        "eval_night_selective",
        "face_benchmark",
        "GroundTruth",
    ]:
        assert forbidden not in source
