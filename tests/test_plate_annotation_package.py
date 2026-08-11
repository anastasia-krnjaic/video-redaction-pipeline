from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest
from click.testing import CliRunner

from redactor import media
from redactor import plate_annotation_package as package
from redactor.cli import main


def write_synthetic_video(path: Path, *, frame_count: int = 8, color_offset: int = 0) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    try:
        for frame_index in range(frame_count):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (25 + frame_index + color_offset, 70, 120)
            cv.rectangle(frame, (18 + frame_index, 22), (34 + frame_index, 30), (230, 230, 230), -1)
            writer.write(frame)
    finally:
        writer.release()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_track_evidence(
    tmp_path: Path,
    night_video: Path,
    records: list[dict[str, Any]],
    *,
    name: str = "track_evidence",
) -> Path:
    evidence_root = tmp_path / name
    evidence_root.mkdir(parents=True, exist_ok=True)
    tracks_path = evidence_root / "tracks.jsonl"
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        frame_index = int(record["frame_index"])
        by_frame.setdefault(frame_index, []).append(
            {
                "bbox_xyxy": record["bbox_xyxy"],
                "category": "license_plate",
                "frame_index": frame_index,
                "provenance": record.get("provenance", "plate_detector_observation"),
                "score": record.get("score", 0.75),
                "source_candidate_ids": record.get(
                    "source_candidate_ids", [f"synthetic:f{frame_index:04d}:d00"]
                ),
                "status": record.get("status", "needs_review"),
                "track_id": record.get("track_id", 1),
                "track_uid": record.get("track_uid", "plate_track_0001"),
            }
        )
    with tracks_path.open("w", encoding="utf-8") as file:
        for frame_index in sorted(by_frame):
            file.write(
                json.dumps({"frame_index": frame_index, "tracks": by_frame[frame_index]}) + "\n"
            )
    write_json(
        evidence_root / "tracking_manifest.json",
        {
            "detector_id": "synthetic_plate_detector",
            "input_height": 64,
            "input_width": 96,
            "nominal_fps": 10.0,
            "schema_version": "plate_tracking.v1",
            "source_sha256": media.sha256_file(night_video),
            "source_video": night_video.as_posix(),
            "tracking_configuration": {
                "max_accepted_center_jump_norm": 4.0,
                "max_area_fraction": 0.08,
                "max_aspect_ratio": 8.0,
                "max_safe_interpolation_gap_seconds": 0.3,
                "min_area_fraction": 4e-05,
                "min_aspect_ratio": 0.7,
            },
            "tracking_configuration_hash": "synthetic_tracking_hash",
            "tracking_summary": {"max_safe_interpolation_gap_frames": 3},
        },
    )
    return tracks_path


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def state_revision(package_root: Path) -> int:
    return int(package.read_json(package_root / "annotations.draft.json")["state_revision"])


def save_annotation(
    package_root: Path,
    frame_id: str,
    annotation: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    return package.save_frame_annotation(
        package_root,
        frame_id,
        annotation,
        expected_state_revision=state_revision(package_root),
        **kwargs,
    )


def mark_draft_shown(package_root: Path, frame_id: str) -> dict[str, Any]:
    return package.mark_draft_shown(
        package_root,
        frame_id,
        expected_state_revision=state_revision(package_root),
    )


def http_error(url: str) -> int:
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url, timeout=5)
    return int(exc_info.value.code)


def make_targets(day_video: Path, night_video: Path) -> tuple[package.PlateTargetConfig, ...]:
    return (
        package.PlateTargetConfig(
            action="redact",
            benchmark_id="day_plate",
            broad_roi_xyxy=(10, 10, 60, 42),
            dense_frame_offset=-1,
            expected_height=64,
            expected_sha256=media.sha256_file(day_video),
            expected_width=96,
            object_type="license_plate",
            parent_frame_end=2,
            parent_frame_start=1,
            source_interval="00:00:00.100-00:00:00.200",
            source_video_label="day_source.mp4",
            target_id="plate_day_01",
            total_video_frames=8,
            video_path=day_video,
        ),
        package.PlateTargetConfig(
            action="redact",
            benchmark_id="night_selective",
            broad_roi_xyxy=(8, 12, 62, 44),
            dense_frame_offset=-3,
            expected_height=64,
            expected_sha256=media.sha256_file(night_video),
            expected_width=96,
            object_type="license_plate",
            parent_frame_end=5,
            parent_frame_start=3,
            source_interval="00:00:00.300-00:00:00.500",
            source_video_label="night_source.mp4",
            target_id="plate_night_01",
            total_video_frames=8,
            video_path=night_video,
        ),
    )


def complete_all_frames(package_root: Path) -> None:
    review = package.read_json(package_root / "review_data.json")
    for frame in review["frames"]:
        save_annotation(
            package_root,
            frame["frame_id"],
            {
                "annotation_status": "reviewed_box",
                "human_bbox": [18, 22, 36, 31],
                "review_note": "reviewed",
            },
        )


@pytest.fixture()
def synthetic_package(tmp_path: Path) -> tuple[Path, tuple[package.PlateTargetConfig, ...]]:
    day_video = tmp_path / "day.mp4"
    night_video = tmp_path / "night.mp4"
    write_synthetic_video(day_video)
    write_synthetic_video(night_video, color_offset=15)
    targets = make_targets(day_video, night_video)
    summary = package.build_plate_annotation_package(
        include_drafts=False,
        output_root=tmp_path / "packages",
        run_id="run_test",
        targets=targets,
        verify_git_ignore=False,
    )
    return Path(summary["package_root"]), targets


def build_assisted_from_source(
    tmp_path: Path,
    source_package_root: Path,
    targets: tuple[package.PlateTargetConfig, ...],
    *,
    proposal_evidence_paths: tuple[Path, ...] | None = None,
    run_id: str = "run_assisted",
) -> tuple[Path, dict[str, Any]]:
    summary = package.build_assisted_plate_annotation_package(
        output_root=tmp_path / "assisted",
        run_id=run_id,
        source_package_root=source_package_root,
        targets=targets,
        proposal_evidence_paths=proposal_evidence_paths,
        verify_git_ignore=False,
    )
    return Path(summary["package_root"]), summary


def test_default_targets_are_frozen_phase_a_ranges() -> None:
    day, night = package.default_plate_targets()
    assert day.video_path == Path("artifacts/eval_day_plate.mp4")
    assert day.parent_frame_start == 240
    assert day.parent_frame_end == 479
    assert day.dense_frame_start == 0
    assert day.mandatory_frame_count == 240
    assert day.broad_roi_xyxy == (650, 380, 1060, 570)
    assert night.video_path == Path("artifacts/eval_night_selective.mp4")
    assert night.parent_frame_start == 420
    assert night.parent_frame_end == 779
    assert night.dense_frame_start == 0
    assert night.mandatory_frame_count == 360
    assert night.broad_roi_xyxy == (60, 560, 500, 900)


def test_package_construction_counts_mapping_assets(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    manifest = package.read_json(package_root / "package_manifest.json")
    identity = package.read_json(package_root / "package_identity.json")
    review = package.read_json(package_root / "review_data.json")
    progress = package.read_json(package_root / "progress.json")
    state = package.read_json(package_root / "annotations.draft.json")
    assert manifest["expected_mandatory_total"] == 5
    assert identity["package_manifest_sha256"] == media.sha256_file(
        package_root / "package_manifest.json"
    )
    assert state["state_revision"] == 0
    assert manifest["targets"]["day_plate"]["expected_mandatory_count"] == 2
    assert manifest["targets"]["night_selective"]["expected_mandatory_count"] == 3
    assert len(review["frames"]) == 5
    day_dense_frames = {
        frame["dense_frame"] for frame in review["frames"] if frame["benchmark_id"] == "day_plate"
    }
    assert day_dense_frames == {
        0,
        1,
    }
    assert progress["total"]["reviewed_box"] == 0
    assert progress["total"]["mandatory"] == 5
    for frame in review["frames"]:
        assert (package_root / frame["crop"]["path"]).exists()
        assert frame["crop"]["roi_provenance"] == "qa_search_only"
        assert frame["crop"]["origin_xy"] in ([10, 10], [8, 12])
    assert package.validate_plate_annotation_package(package_root)["passed"] is True


def test_validation_accepts_legacy_progress_missing_assisted_counters(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    progress_path = package_root / "progress.json"
    progress = package.read_json(progress_path)
    for row in [progress["total"], *progress["by_benchmark"].values()]:
        for key in package.OPTIONAL_PROGRESS_ROW_KEYS:
            row.pop(key, None)
    write_json(progress_path, progress)
    assert package.validate_plate_annotation_package(package_root)["passed"] is True


@pytest.mark.parametrize(
    "bbox",
    [
        [-1, 0, 2, 2],
        [0, 0, 0, 2],
        [0, 0, 2, 0],
        [3, 3, 2, 5],
        [0, 0, 97, 10],
    ],
)
def test_invalid_geometry_is_rejected(bbox: list[int]) -> None:
    with pytest.raises(ValueError):
        package.validate_bbox_xyxy(bbox, width=96, height=64)


def test_geometry_transform_helpers() -> None:
    assert package.validate_bbox_xyxy([1, 2, 8, 9], width=96, height=64) == [1, 2, 8, 9]
    original = package.crop_box_to_original(
        [4, 5, 14, 12],
        [10, 20],
        frame_width=96,
        frame_height=64,
    )
    assert original == [14, 25, 24, 32]
    assert package.original_box_to_crop(
        original,
        [10, 20],
        crop_width=40,
        crop_height=30,
        frame_width=96,
        frame_height=64,
    ) == [4, 5, 14, 12]


def test_draft_selection_falls_back_to_nearest_source_aligned_candidate() -> None:
    candidate = {"bbox_xyxy": [70, 20, 82, 30], "object_id": "proposal"}
    assert (
        package.choose_main_draft_candidate([candidate], roi_xyxy=(10, 10, 60, 42))
        == candidate
    )


def test_save_reload_progress_and_state_validation(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    review = package.read_json(package_root / "review_data.json")
    first = review["frames"][0]["frame_id"]
    second = review["frames"][1]["frame_id"]
    result = save_annotation(
        package_root,
        first,
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [16, 18, 30, 28],
            "review_note": "tight panel",
        },
        now_utc="2026-08-10T00:00:00Z",
    )
    assert result["progress"]["total"]["reviewed_box"] == 1
    state = package.read_json(package_root / "annotations.draft.json")
    assert state["frames"][first]["human_bbox"] == [16, 18, 30, 28]
    assert state["frames"][first]["human_reviewed"] is True
    save_annotation(
        package_root,
        second,
        {"annotation_status": "needs_recheck", "review_note": "temporal uncertainty"},
    )
    reloaded_progress = package.read_json(package_root / "progress.json")
    assert reloaded_progress["total"]["needs_recheck"] == 1
    assert "save_annotation" in (package_root / "change_log.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        save_annotation(
            package_root,
            first,
            {"annotation_status": "detector_ignore", "review_note": "blocked"},
        )


def test_temporal_gt_conflict_requires_note_blocks_export_and_resolves_explicitly(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    review = package.read_json(package_root / "review_data.json")
    first = review["frames"][0]["frame_id"]

    with pytest.raises(ValueError, match="temporal_gt_conflict requires review_note"):
        save_annotation(package_root, first, {"annotation_status": "temporal_gt_conflict"})
    with pytest.raises(ValueError, match="temporal_gt_conflict requires review_note"):
        save_annotation(
            package_root,
            first,
            {"annotation_status": "temporal_gt_conflict", "review_note": "   "},
        )

    before_conflict_revision = state_revision(package_root)
    conflict = save_annotation(
        package_root,
        first,
        {
            "annotation_status": "temporal_gt_conflict",
            "review_note": "not defensibly localizable",
        },
    )
    assert conflict["state_revision"] == before_conflict_revision + 1
    assert conflict["frame"]["annotation_status"] == "temporal_gt_conflict"
    assert conflict["frame"]["human_bbox"] is None
    assert conflict["progress"]["total"]["reviewed_box"] == 0
    assert conflict["progress"]["total"]["temporal_gt_conflict"] == 1
    assert conflict["progress"]["final_export_ready"] is False

    for frame in review["frames"][1:]:
        save_annotation(
            package_root,
            frame["frame_id"],
            {
                "annotation_status": "reviewed_box",
                "human_bbox": [18, 22, 36, 31],
                "review_note": "reviewed",
            },
        )
    with pytest.raises(ValueError, match="unresolved temporal_gt_conflict"):
        package.export_plate_annotation_review(package_root, final=True)

    before_resolution_revision = state_revision(package_root)
    resolved = save_annotation(
        package_root,
        first,
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [18, 22, 36, 31],
            "review_note": "resolved with tight box",
        },
    )
    assert resolved["state_revision"] == before_resolution_revision + 1
    assert resolved["frame"]["annotation_status"] == "reviewed_box"
    assert resolved["frame"]["conflicts"][0]["resolved"] is True
    assert resolved["progress"]["total"]["reviewed_box"] == 5
    assert resolved["progress"]["total"]["temporal_gt_conflict"] == 0
    assert resolved["progress"]["final_export_ready"] is True
    final_export = package.export_plate_annotation_review(package_root, final=True)
    assert final_export["export_kind"] == "final_review"
    change_log = (package_root / "change_log.jsonl").read_text(encoding="utf-8")
    assert "not defensibly localizable" in change_log
    assert "resolved" in change_log
    assert "reviewed_box" in change_log


def test_draft_seed_hidden_by_default_and_excludes_unmatched(tmp_path: Path) -> None:
    day_video = tmp_path / "day.mp4"
    night_video = tmp_path / "night.mp4"
    write_synthetic_video(day_video)
    write_synthetic_video(night_video, color_offset=15)
    targets = make_targets(day_video, night_video)
    draft_path = tmp_path / "drafts.json"
    write_json(
        draft_path,
        {
            "annotations": [
                {
                    "bbox_xyxy": [18, 22, 36, 31],
                    "clip_id": "source_aligned",
                    "frame_index": 1,
                    "object_id": "proposal_1",
                    "reviewer_provenance": "visual_proposal",
                    "schema_version": "draft.v1",
                    "source_sha256": media.sha256_file(day_video),
                    "visibility": "clear",
                },
                {
                    "bbox_xyxy": [1, 1, 4, 4],
                    "clip_id": "close_clip",
                    "frame_index": 1,
                    "object_id": "proposal_2",
                    "reviewer_provenance": "visual_proposal",
                    "schema_version": "draft.v1",
                    "source_sha256": "not-a-target",
                    "visibility": "clear",
                },
            ],
            "finalists": ["model_a"],
            "sample_frame_count_by_clip": {"source_aligned": 1},
            "schema_version": "drafts.v1",
        },
    )
    summary = package.build_plate_annotation_package(
        draft_path=draft_path,
        include_drafts=True,
        output_root=tmp_path / "packages",
        run_id="run_drafts",
        targets=targets,
        verify_git_ignore=False,
    )
    package_root = Path(summary["package_root"])
    state = package.read_json(package_root / "annotations.draft.json")
    progress = package.read_json(package_root / "progress.json")
    draft_audit = package.read_json(package_root / "draft_import_audit.json")
    frame = state["frames"]["day_plate_000001"]
    assert frame["draft_available"] is True
    assert frame["draft_shown"] is False
    assert frame["human_bbox"] is None
    assert progress["total"]["reviewed_box"] == 0
    assert draft_audit["excluded_by_reason"]["source_hash_not_in_phase_a_targets"] == 1
    result = mark_draft_shown(package_root, "day_plate_000001")
    assert result["frame"]["draft_shown"] is True
    assert result["frame"]["proposal_shown"] is True


def test_draft_visibility_does_not_resolve_temporal_conflict(tmp_path: Path) -> None:
    day_video = tmp_path / "day.mp4"
    night_video = tmp_path / "night.mp4"
    write_synthetic_video(day_video)
    write_synthetic_video(night_video, color_offset=15)
    targets = make_targets(day_video, night_video)
    draft_path = tmp_path / "drafts.json"
    write_json(
        draft_path,
        {
            "annotations": [
                {
                    "bbox_xyxy": [18, 22, 36, 31],
                    "clip_id": "source_aligned",
                    "frame_index": 1,
                    "object_id": "proposal_1",
                    "reviewer_provenance": "visual_proposal",
                    "schema_version": "draft.v1",
                    "source_sha256": media.sha256_file(day_video),
                    "visibility": "clear",
                }
            ],
            "finalists": ["model_a"],
            "sample_frame_count_by_clip": {"source_aligned": 1},
            "schema_version": "drafts.v1",
        },
    )
    summary = package.build_plate_annotation_package(
        draft_path=draft_path,
        include_drafts=True,
        output_root=tmp_path / "packages",
        run_id="run_draft_conflict",
        targets=targets,
        verify_git_ignore=False,
    )
    package_root = Path(summary["package_root"])
    frame_id = "day_plate_000001"
    save_annotation(
        package_root,
        frame_id,
        {
            "annotation_status": "temporal_gt_conflict",
            "review_note": "temporal disagreement",
        },
    )
    result = mark_draft_shown(package_root, frame_id)
    assert result["frame"]["draft_shown"] is True
    assert result["frame"]["proposal_shown"] is True
    assert result["frame"]["annotation_status"] == "temporal_gt_conflict"
    assert result["progress"]["total"]["reviewed_box"] == 0
    assert result["progress"]["total"]["temporal_gt_conflict"] == 1


def test_secondary_instances_and_final_export_gates(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    review = package.read_json(package_root / "review_data.json")
    for frame in review["frames"]:
        save_annotation(
            package_root,
            frame["frame_id"],
            {
                "annotation_status": "reviewed_box",
                "human_bbox": [18, 22, 36, 31],
                "review_note": "reviewed",
            },
        )
    first = review["frames"][0]["frame_id"]
    save_annotation(
        package_root,
        first,
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [18, 22, 36, 31],
            "review_note": "reviewed",
            "secondary_instances": [
                {"bbox_xyxy": [40, 24, 52, 31], "kind": "additional_localizable_plate"},
                {"bbox_xyxy": [4, 5, 7, 8], "kind": "ignore_small", "note": "tiny"},
                {
                    "bbox_xyxy": [58, 20, 62, 24],
                    "kind": "ambiguous_review_required",
                    "note": "unclear",
                },
            ],
        },
    )
    progress = package.read_json(package_root / "progress.json")
    assert progress["total"]["reviewed_box"] == 5
    assert progress["total"]["additional_localizable_plate"] == 1
    assert progress["total"]["ignore_small"] == 1
    assert progress["total"]["unresolved_ambiguous_secondary"] == 1
    with pytest.raises(ValueError):
        package.export_plate_annotation_review(package_root, final=True)
    save_annotation(
        package_root,
        first,
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [18, 22, 36, 31],
            "review_note": "reviewed",
            "secondary_instances": [
                {
                    "bbox_xyxy": [58, 20, 62, 24],
                    "kind": "ambiguous_review_required",
                    "note": "resolved",
                    "resolved": True,
                }
            ],
        },
    )
    first_export = package.export_plate_annotation_review(
        package_root,
        final=True,
        now_utc="2026-08-10T00:00:00Z",
    )
    second_export = package.export_plate_annotation_review(
        package_root,
        final=True,
        now_utc="2026-08-10T01:00:00Z",
    )
    assert first_export["semantic_content_hash"] == second_export["semantic_content_hash"]
    assert (package_root / "plate_annotation_review.export.json").exists()


@pytest.mark.parametrize(
    "tamper_case",
    [
        "day_target_id",
        "night_target_id",
        "benchmark_id",
        "object_type",
        "action",
        "manifest_source_sha",
        "identity_manifest_sha",
        "actual_video_bytes",
        "parent_frame",
        "dense_frame",
        "duplicate_frame",
        "missing_frame",
        "extra_mandatory_record",
        "unknown_package_schema",
        "unknown_certified_config",
        "corrupted_package_manifest",
        "changed_certified_interval",
        "changed_expected_mandatory_count",
    ],
)
def test_completed_export_rejects_identity_and_mapping_tampering(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
    tamper_case: str,
) -> None:
    package_root, targets = synthetic_package
    complete_all_frames(package_root)
    progress = package.read_json(package_root / "progress.json")
    assert progress["final_export_ready"] is True

    manifest_path = package_root / "package_manifest.json"
    review_path = package_root / "review_data.json"
    identity_path = package_root / "package_identity.json"
    manifest = package.read_json(manifest_path)
    review = package.read_json(review_path)
    identity = package.read_json(identity_path)
    if tamper_case == "day_target_id":
        review["frames"][0]["target_id"] = "wrong_target"
        write_json(review_path, review)
    elif tamper_case == "night_target_id":
        night_frame = next(
            frame for frame in review["frames"] if frame["benchmark_id"] == "night_selective"
        )
        night_frame["target_id"] = "wrong_target"
        write_json(review_path, review)
    elif tamper_case == "benchmark_id":
        review["frames"][0]["benchmark_id"] = "wrong_benchmark"
        write_json(review_path, review)
    elif tamper_case == "object_type":
        review["frames"][0]["object_type"] = "face"
        write_json(review_path, review)
    elif tamper_case == "action":
        review["frames"][0]["action"] = "keep"
        write_json(review_path, review)
    elif tamper_case == "manifest_source_sha":
        manifest["targets"]["day_plate"]["evaluation_video"]["sha256"] = "bad"
        write_json(manifest_path, manifest)
    elif tamper_case == "identity_manifest_sha":
        identity["package_manifest_sha256"] = "bad"
        write_json(identity_path, identity)
    elif tamper_case == "actual_video_bytes":
        targets[0].video_path.unlink()
        write_synthetic_video(targets[0].video_path, color_offset=99)
    elif tamper_case == "parent_frame":
        review["frames"][0]["parent_frame"] += 1
        write_json(review_path, review)
    elif tamper_case == "dense_frame":
        review["frames"][0]["dense_frame"] += 1
        write_json(review_path, review)
    elif tamper_case == "duplicate_frame":
        review["frames"].append(dict(review["frames"][0]))
        write_json(review_path, review)
    elif tamper_case == "missing_frame":
        review["frames"].pop()
        write_json(review_path, review)
    elif tamper_case == "extra_mandatory_record":
        extra = dict(review["frames"][0])
        extra["frame_id"] = "day_plate_999999"
        extra["parent_frame"] = 999999
        extra["dense_frame"] = 999999
        review["frames"].append(extra)
        write_json(review_path, review)
    elif tamper_case == "unknown_package_schema":
        manifest["schema_version"] = "unknown"
        write_json(manifest_path, manifest)
    elif tamper_case == "unknown_certified_config":
        manifest["certified_config_version"] = "unknown"
        write_json(manifest_path, manifest)
    elif tamper_case == "corrupted_package_manifest":
        manifest_path.write_text("{", encoding="utf-8")
    elif tamper_case == "changed_certified_interval":
        manifest["targets"]["day_plate"]["frozen_parent_frame_interval"] = [1, 3]
        write_json(manifest_path, manifest)
    elif tamper_case == "changed_expected_mandatory_count":
        manifest["targets"]["day_plate"]["expected_mandatory_count"] = 3
        write_json(manifest_path, manifest)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(tamper_case)

    with pytest.raises(ValueError):
        package.export_plate_annotation_review(package_root, final=True)


def test_validation_rejects_duplicate_and_missing_frame(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    review_path = package_root / "review_data.json"
    review = package.read_json(review_path)
    review["frames"].append(dict(review["frames"][0]))
    write_json(review_path, review)
    result = package.validate_plate_annotation_package(package_root)
    assert result["passed"] is False
    assert any("duplicate frame_id" in error for error in result["errors"])

    state_path = package_root / "annotations.draft.json"
    state = package.read_json(state_path)
    state["frames"].pop(review["frames"][0]["frame_id"])
    write_json(state_path, state)
    result = package.validate_plate_annotation_package(package_root)
    assert result["passed"] is False
    assert any("missing annotation state" in error for error in result["errors"])


def test_server_rejects_unsafe_paths_and_invalid_saves(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    handler = package.handler_for_package(package_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        assert http_error(f"{base_url}/%2e%2e/pyproject.toml") == 403
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            post_json(
                f"{base_url}/api/annotations/save",
                {
                    "annotation": {
                        "annotation_status": "reviewed_box",
                        "human_bbox": [1, 1, 3, 3],
                    },
                    "expected_state_revision": state_revision(package_root),
                    "frame_id": "missing",
                },
            )
        assert exc_info.value.code == 400
        review = package.read_json(package_root / "review_data.json")
        frame_id = review["frames"][0]["frame_id"]
        result = post_json(
            f"{base_url}/api/annotations/save",
            {
                "annotation": {
                    "annotation_status": "reviewed_box",
                    "human_bbox": [18, 22, 36, 31],
                    "review_note": "server save",
                },
                "expected_state_revision": state_revision(package_root),
                "frame_id": frame_id,
            },
        )
        assert result["frame"]["human_bbox"] == [18, 22, 36, 31]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_rejects_stale_save_and_preserves_revision_across_restart(
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    package_root, _targets = synthetic_package
    review = package.read_json(package_root / "review_data.json")
    frame_id = review["frames"][0]["frame_id"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), package.handler_for_package(package_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        loaded = get_json(f"{base_url}/api/state")
        assert loaded["annotations"]["state_revision"] == 0
        first = post_json(
            f"{base_url}/api/annotations/save",
            {
                "annotation": {
                    "annotation_status": "reviewed_box",
                    "human_bbox": [18, 22, 36, 31],
                    "review_note": "client a",
                },
                "expected_state_revision": 0,
                "frame_id": frame_id,
            },
        )
        assert first["state_revision"] == 1
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            post_json(
                f"{base_url}/api/annotations/save",
                {
                    "annotation": {
                        "annotation_status": "reviewed_box",
                        "human_bbox": [19, 23, 37, 32],
                        "review_note": "client b stale",
                    },
                    "expected_state_revision": 0,
                    "frame_id": frame_id,
                },
            )
        assert exc_info.value.code == 409
        stale_body = json.loads(exc_info.value.read().decode("utf-8"))
        assert stale_body["current_state_revision"] == 1
        state = package.read_json(package_root / "annotations.draft.json")
        assert state["frames"][frame_id]["human_bbox"] == [18, 22, 36, 31]
        assert state["frames"][frame_id]["review_note"] == "client a"
        change_events = [
            json.loads(line)
            for line in (package_root / "change_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [event["event"] for event in change_events] == ["save_annotation"]
        reloaded = get_json(f"{base_url}/api/state")
        assert reloaded["annotations"]["state_revision"] == 1
        second = post_json(
            f"{base_url}/api/annotations/save",
            {
                "annotation": {
                    "annotation_status": "reviewed_box",
                    "human_bbox": [19, 23, 37, 32],
                    "review_note": "client b reconciled",
                },
                "expected_state_revision": 1,
                "frame_id": frame_id,
            },
        )
        assert second["state_revision"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    restarted = ThreadingHTTPServer(("127.0.0.1", 0), package.handler_for_package(package_root))
    restarted_thread = threading.Thread(target=restarted.serve_forever, daemon=True)
    restarted_thread.start()
    try:
        restarted_base = f"http://127.0.0.1:{restarted.server_address[1]}"
        restarted_state = get_json(f"{restarted_base}/api/state")
        assert restarted_state["annotations"]["state_revision"] == 2
        reset = post_json(
            f"{restarted_base}/api/annotations/save",
            {
                "annotation": {
                    "annotation_status": "unreviewed",
                    "review_note": "",
                },
                "expected_state_revision": 2,
                "frame_id": frame_id,
            },
        )
        assert reset["state_revision"] == 3
        assert reset["frame"]["human_bbox"] is None
    finally:
        restarted.shutdown()
        restarted.server_close()
        restarted_thread.join(timeout=5)


def test_assisted_package_migrates_reviewed_records_and_generates_proposals(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    night_video = targets[1].video_path
    save_annotation(
        source_package_root,
        "day_plate_000001",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [16, 18, 30, 28],
            "review_note": "day reviewed",
        },
        now_utc="2026-08-10T00:00:00Z",
    )
    save_annotation(
        source_package_root,
        "night_selective_000003",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [20, 22, 38, 31],
            "review_note": "night reviewed",
        },
        now_utc="2026-08-10T00:01:00Z",
    )
    evidence_path = write_track_evidence(
        tmp_path,
        night_video,
        [
            {"bbox_xyxy": [20, 22, 38, 31], "frame_index": 3},
            {"bbox_xyxy": [22, 23, 40, 32], "frame_index": 4},
            {"bbox_xyxy": [24, 24, 42, 33], "frame_index": 5},
        ],
    )
    source_hashes = package.package_core_hashes(source_package_root)
    assisted_root, summary = build_assisted_from_source(
        tmp_path,
        source_package_root,
        targets,
        proposal_evidence_paths=(evidence_path,),
    )

    assert package.package_core_hashes(source_package_root) == source_hashes
    assert package.validate_plate_annotation_package(assisted_root)["passed"] is True
    manifest = package.read_json(assisted_root / "package_manifest.json")
    identity = package.read_json(assisted_root / "package_identity.json")
    progress = package.read_json(assisted_root / "progress.json")
    state = package.read_json(assisted_root / "annotations.draft.json")
    review = package.read_json(assisted_root / "review_data.json")
    source_state = package.read_json(source_package_root / "annotations.draft.json")
    source_review = package.read_json(source_package_root / "review_data.json")

    assert manifest["annotation_mode"] == package.ANNOTATION_MODE_ASSISTED
    assert identity["package_manifest_sha256"] == media.sha256_file(
        assisted_root / "package_manifest.json"
    )
    migration = summary["assisted_migration"]
    assert migration["bbox_mismatch_count"] == 0
    assert migration["changed_record_count"] == 0
    assert migration["missing_record_count"] == 0
    assert migration["migrated_record_count"] == 2
    assert progress["total"]["reviewed_box"] == 2
    assert progress["total"]["unreviewed"] == 3
    assert progress["total"]["proposal_available_total"] == 2
    assert progress["total"]["no_proposal_remaining"] == 1
    assert progress["total"]["proposal_source_breakdown"] == {
        package.PROPOSAL_DIRECT_DETECTOR: 2
    }

    source_frames = {frame["frame_id"]: frame for frame in source_review["frames"]}
    assisted_frames = {frame["frame_id"]: frame for frame in review["frames"]}
    for frame_id in ("day_plate_000001", "night_selective_000003"):
        assert package.migration_compare_record(
            assisted_frames[frame_id], state["frames"][frame_id]
        ) == package.migration_compare_record(
            source_frames[frame_id], source_state["frames"][frame_id]
        )
        assert state["frames"][frame_id]["proposal_available"] is False
        assert state["frames"][frame_id]["legacy_draft_available"] is False

    day_record = state["frames"]["day_plate_000002"]
    assert day_record["annotation_status"] == "unreviewed"
    assert day_record["proposal_available"] is False
    assert day_record["proposal_bbox"] is None

    for frame_id in ("night_selective_000004", "night_selective_000005"):
        record = state["frames"][frame_id]
        assert record["annotation_status"] == "unreviewed"
        assert record["proposal_available"] is True
        assert record["proposal_source"] == package.PROPOSAL_DIRECT_DETECTOR
        assert record["proposal_quality"] == package.PROPOSAL_QUALITY_SAFE_TO_REVIEW
        assert record["draft_bbox"] == record["proposal_bbox"]
        assert record["proposal_bbox"] != [20, 22, 38, 31]
        assert record["human_bbox"] is None

    proposal_audit = package.read_json(assisted_root / "proposal_generation_audit.json")
    assert proposal_audit["total_unreviewed"] == 3
    assert proposal_audit["proposal_source_counts"][package.PROPOSAL_DIRECT_DETECTOR] == 2
    assert proposal_audit["proposal_source_counts"][package.PROPOSAL_HUMAN_MOTION] == 0
    assert package.PROPOSAL_HUMAN_SEEDED not in progress["total"]["proposal_source_breakdown"]
    association = proposal_audit["target_association"][str(evidence_path)]
    assert association["night_selective"]["selected_track_uids"] == ["plate_track_0001"]

    for frame in review["frames"]:
        assert frame["full_frame"]["width"] == frame["frame_width"]
        assert frame["full_frame"]["height"] == frame["frame_height"]
        assert (assisted_root / frame["full_frame"]["path"]).exists()


def test_accept_draft_and_edit_records_assisted_provenance(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    night_video = targets[1].video_path
    save_annotation(
        source_package_root,
        "night_selective_000003",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [20, 22, 38, 31],
            "review_note": "night anchor",
        },
    )
    evidence_path = write_track_evidence(
        tmp_path,
        night_video,
        [
            {"bbox_xyxy": [20, 22, 38, 31], "frame_index": 3},
            {"bbox_xyxy": [22, 23, 40, 32], "frame_index": 4},
            {"bbox_xyxy": [24, 24, 42, 33], "frame_index": 5},
        ],
    )
    assisted_root, _summary = build_assisted_from_source(
        tmp_path,
        source_package_root,
        targets,
        proposal_evidence_paths=(evidence_path,),
    )
    state = package.read_json(assisted_root / "annotations.draft.json")
    proposal = state["frames"]["night_selective_000004"]["proposal_bbox"]
    accepted = package.accept_draft_annotation(
        assisted_root,
        "night_selective_000004",
        expected_state_revision=state_revision(assisted_root),
        now_utc="2026-08-10T00:02:00Z",
    )

    assert accepted["frame"]["human_bbox"] == proposal
    assert accepted["frame"]["accepted_without_geometry_change"] is True
    assert accepted["frame"]["human_adjusted_geometry"] is False
    assert accepted["progress"]["total"]["proposal_accepted_unchanged"] == 1
    with pytest.raises(package.StaleStateRevisionError):
        package.accept_draft_annotation(
            assisted_root,
            "night_selective_000005",
            expected_state_revision=0,
        )

    edited_bbox = [22, 23, 41, 33]
    edited = save_annotation(
        assisted_root,
        "night_selective_000005",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": edited_bbox,
            "proposal_shown": True,
            "review_note": "adjusted",
        },
    )
    assert edited["frame"]["human_bbox"] == edited_bbox
    assert edited["frame"]["accepted_without_geometry_change"] is False
    assert edited["frame"]["human_adjusted_geometry"] is True
    assert edited["progress"]["total"]["proposal_human_adjusted"] == 1
    events = [
        json.loads(line)
        for line in (assisted_root / "change_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events] == ["accept_draft", "save_annotation"]
    assert events[0]["accepted_without_geometry_change"] is True
    assert events[1]["human_adjusted_geometry"] is True


def test_assisted_bounded_interpolation_requires_detector_anchors(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    night_video = targets[1].video_path
    save_annotation(
        source_package_root,
        "night_selective_000003",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [20, 22, 38, 31],
            "review_note": "night anchor",
        },
    )
    evidence_path = write_track_evidence(
        tmp_path,
        night_video,
        [
            {"bbox_xyxy": [20, 22, 38, 31], "frame_index": 3},
            {
                "bbox_xyxy": [22, 23, 40, 32],
                "frame_index": 4,
                "provenance": "plate_track_interpolation",
            },
            {"bbox_xyxy": [24, 24, 42, 33], "frame_index": 5},
        ],
        name="bounded_interpolation",
    )
    assisted_root, _summary = build_assisted_from_source(
        tmp_path,
        source_package_root,
        targets,
        proposal_evidence_paths=(evidence_path,),
    )
    state = package.read_json(assisted_root / "annotations.draft.json")
    interpolated = state["frames"]["night_selective_000004"]
    assert interpolated["proposal_source"] == package.PROPOSAL_BOUNDED_INTERPOLATION
    assert interpolated["proposal_quality"] == package.PROPOSAL_QUALITY_NEEDS_ADJUSTMENT
    assert (
        state["frames"]["night_selective_000005"]["proposal_source"]
        == package.PROPOSAL_DIRECT_DETECTOR
    )
    audit = package.read_json(assisted_root / "proposal_generation_audit.json")
    frame4 = next(
        frame for frame in audit["frames"] if frame["frame_id"] == "night_selective_000004"
    )
    assert frame4["left_anchor"]["parent_frame"] == 3
    assert frame4["right_anchor"]["parent_frame"] == 5


def test_assisted_no_unbounded_extrapolation_or_static_human_carry_forward(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    save_annotation(
        source_package_root,
        "night_selective_000003",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [20, 22, 38, 31],
            "review_note": "night anchor",
        },
    )
    assisted_root, _summary = build_assisted_from_source(
        tmp_path,
        source_package_root,
        targets,
        proposal_evidence_paths=(),
    )
    state = package.read_json(assisted_root / "annotations.draft.json")
    for frame_id in ("night_selective_000004", "night_selective_000005"):
        record = state["frames"][frame_id]
        assert record["proposal_available"] is False
        assert record["proposal_bbox"] is None
        assert record["proposal_source"] is None
        assert record["human_bbox"] is None
    assert package.PROPOSAL_HUMAN_SEEDED not in package.read_json(
        assisted_root / "progress.json"
    )["total"]["proposal_source_breakdown"]
    with pytest.raises(ValueError, match="no proposal is available"):
        package.accept_draft_annotation(
            assisted_root,
            "night_selective_000004",
            expected_state_revision=state_revision(assisted_root),
        )


def test_assisted_rejects_candidate_qa_failures(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    night_video = targets[1].video_path
    save_annotation(
        source_package_root,
        "night_selective_000003",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [20, 22, 38, 31],
            "review_note": "night anchor",
        },
    )
    evidence_path = write_track_evidence(
        tmp_path,
        night_video,
        [
            {"bbox_xyxy": [20, 22, 38, 31], "frame_index": 3},
            {"bbox_xyxy": [22, 10, 24, 54], "frame_index": 4},
        ],
        name="qa_reject",
    )
    assisted_root, _summary = build_assisted_from_source(
        tmp_path,
        source_package_root,
        targets,
        proposal_evidence_paths=(evidence_path,),
    )
    state = package.read_json(assisted_root / "annotations.draft.json")
    assert state["frames"]["night_selective_000004"]["proposal_available"] is False
    audit = package.read_json(assisted_root / "proposal_generation_audit.json")
    frame4 = next(
        frame for frame in audit["frames"] if frame["frame_id"] == "night_selective_000004"
    )
    assert frame4["rejection_reason"] == "candidate_rejected_by_qa"
    assert (
        frame4["discarded_candidates"][0]["rejection_reason"]
        == "aspect_ratio_outside_source_bounds"
    )


def test_assisted_accepted_tracker_takes_precedence_over_direct_detector(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    night_video = targets[1].video_path
    save_annotation(
        source_package_root,
        "night_selective_000003",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [20, 22, 38, 31],
            "review_note": "night anchor",
        },
    )
    accepted_path = write_track_evidence(
        tmp_path,
        night_video,
        [
            {"bbox_xyxy": [20, 22, 38, 31], "frame_index": 3, "status": "accepted_plate_track"},
            {"bbox_xyxy": [21, 23, 39, 32], "frame_index": 4, "status": "accepted_plate_track"},
        ],
        name="accepted_tracker",
    )
    direct_path = write_track_evidence(
        tmp_path,
        night_video,
        [
            {"bbox_xyxy": [20, 22, 38, 31], "frame_index": 3},
            {"bbox_xyxy": [25, 24, 43, 33], "frame_index": 4},
        ],
        name="direct_detector",
    )
    assisted_root, _summary = build_assisted_from_source(
        tmp_path,
        source_package_root,
        targets,
        proposal_evidence_paths=(accepted_path, direct_path),
    )
    record = package.read_json(assisted_root / "annotations.draft.json")["frames"][
        "night_selective_000004"
    ]
    assert record["proposal_source"] == package.PROPOSAL_AUTOMATIC_TRACKER
    assert record["proposal_bbox"] == [21, 23, 39, 32]


def test_full_frame_geometry_allows_boxes_outside_static_crop(
    tmp_path: Path,
    synthetic_package: tuple[Path, tuple[package.PlateTargetConfig, ...]],
) -> None:
    source_package_root, targets = synthetic_package
    save_annotation(
        source_package_root,
        "day_plate_000001",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": [16, 18, 30, 28],
            "review_note": "anchor",
        },
    )
    assisted_root, _summary = build_assisted_from_source(tmp_path, source_package_root, targets)
    outside_crop_bbox = [70, 48, 88, 60]
    result = save_annotation(
        assisted_root,
        "day_plate_000002",
        {
            "annotation_status": "reviewed_box",
            "human_bbox": outside_crop_bbox,
            "proposal_shown": True,
            "review_note": "outside old crop but visible in full frame",
        },
    )
    review = package.read_json(assisted_root / "review_data.json")
    frame = next(item for item in review["frames"] if item["frame_id"] == "day_plate_000002")
    assert result["frame"]["human_bbox"] == outside_crop_bbox
    assert outside_crop_bbox[0] > frame["crop"]["origin_xy"][0] + frame["crop"]["width"]
    assert frame["full_frame"]["width"] == 96
    assert frame["full_frame"]["height"] == 64
    assert package.validate_plate_annotation_package(assisted_root)["passed"] is True


def test_assisted_ui_exposes_full_frame_and_keyboard_controls() -> None:
    html = package.index_html()
    js = package.app_js()
    assert "Accept Draft + Next" in html
    assert "Fit Frame" in html
    assert "QA ROI" in html
    assert "A/Enter accept" in html
    assert "S save+next" in html
    assert "proposalBanner" in html
    assert "acceptDraft" in js
    assert "/api/annotations/accept-draft" in js
    assert "frame.full_frame" in js
    assert "fitFrame" in js
    assert "fitTarget" in js
    assert "isTypingTarget" in js
    assert "PROPOSAL - NOT YET REVIEWED" in js
    assert "PROPOSAL NEEDS ADJUSTMENT" in js
    assert "save(true)" in js


def test_cli_help_for_plate_annotation_commands() -> None:
    runner = CliRunner()
    for command in (
        "build-assisted-plate-annotation-package",
        "build-plate-annotation-package",
        "validate-plate-annotation-package",
        "summarize-plate-annotation-package",
        "serve-plate-annotation-package",
    ):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0
