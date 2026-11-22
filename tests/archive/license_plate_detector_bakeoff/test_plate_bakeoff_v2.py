from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from experiments.archive.license_plate_detector_bakeoff import (
    license_plate_detector_bakeoff_v2 as experiment_v2,
)
from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff_v2


def test_oim_yolov9_empty_output_produces_empty_frame() -> None:
    _tensor, mapping = plate_bakeoff_v2.preprocess_letterbox_rgb_01(
        np.zeros((50, 100, 3), dtype=np.uint8),
        64,
    )

    run = plate_bakeoff_v2.parse_oim_yolov9_output(
        np.zeros((0, 7), dtype=np.float32),
        detector_id="oim",
        frame_width=100,
        frame_height=50,
        mapping=mapping,
        score_floor=0.05,
        inference_time_ms=1.0,
    )

    assert run.final_detections == []
    assert run.diagnostics["model_output_shape"] == [0, 7]


def test_oim_yolov9_class_filtering_and_coordinate_restoration() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    _tensor, mapping = plate_bakeoff_v2.preprocess_letterbox_rgb_01(frame, 64)
    raw = np.array(
        [
            [0, 20, 28, 44, 36, 0, 0.9],
            [0, 20, 28, 44, 36, 1, 0.95],
            [0, 20, 28, 44, 36, 0, 0.01],
        ],
        dtype=np.float32,
    )

    run = plate_bakeoff_v2.parse_oim_yolov9_output(
        raw,
        detector_id="oim",
        frame_width=100,
        frame_height=50,
        mapping=mapping,
        score_floor=0.05,
        inference_time_ms=1.0,
    )

    assert len(run.final_detections) == 1
    assert run.final_detections[0].label == 0
    assert run.final_detections[0].bbox_xyxy == (31, 18, 69, 32)


def test_rfdetr_output_decode_and_duplicate_suppression() -> None:
    boxes = np.array(
        [
            [0.5, 0.5, 0.2, 0.1],
            [0.505, 0.5, 0.2, 0.1],
            [0.2, 0.2, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    logits = np.array([[6.0], [5.0], [-6.0]], dtype=np.float32)

    run = plate_bakeoff_v2.parse_rfdetr_outputs(
        boxes,
        logits,
        detector_id="rf",
        frame_width=200,
        frame_height=100,
        score_floor=0.05,
        nms_iou_threshold=0.45,
        inference_time_ms=2.0,
    )

    assert len(run.pre_nms_detections) == 2
    assert len(run.final_detections) == 1
    assert run.final_detections[0].bbox_xyxy == (79, 44, 121, 56)


def test_normalized_cxcywh_clips_to_frame() -> None:
    bbox = plate_bakeoff_v2.normalized_cxcywh_to_xyxy(
        np.array([0.0, 1.0, 0.4, 0.4], dtype=np.float32),
        frame_width=100,
        frame_height=50,
    )

    assert bbox == (0, 39, 21, 50)


def test_uniform_tile_policy_is_shared_and_not_clip_specific() -> None:
    modes = experiment_v2.inference_modes()
    tiled = next(mode for mode in modes if mode.tiled)

    assert tiled.tile_size == plate_bakeoff_v2.TILE_SIZE
    assert tiled.tile_size == 1280
    assert tiled.tile_overlap == 0.25
    assert tiled.include_full_frame is True


def test_correct_primary_day_clip_mapping() -> None:
    clip_map = {clip.clip_id: clip for clip in experiment_v2.clips()}

    assert clip_map["day_plate_primary"].video_path.name == "eval_day_plate_close.mp4"
    assert clip_map["day_plate_close_stress"].video_path.name == "eval_day_plate.mp4"
    assert clip_map["day_plate_primary"].evaluated_ranges == ((0, 239),)
    assert clip_map["day_plate_close_stress"].certified_quantitative is False


def test_candidate_threshold_selection_is_one_threshold_across_clips() -> None:
    metrics_by_clip = {
        "day_plate_primary": {
            "thresholds": [
                {"threshold": threshold, "certified_quantitative": False}
                for threshold in plate_bakeoff_v2.THRESHOLD_GRID
            ]
        },
        "night_plate_primary": {
            "thresholds": [
                {
                    "threshold": threshold,
                    "certified_quantitative": True,
                    "object_frame_recall": 1.0 if threshold == 0.1 else 0.9,
                    "longest_consecutive_miss_run": 0 if threshold == 0.1 else 3,
                    "false_positive_frames": 2 if threshold == 0.1 else 1,
                }
                for threshold in plate_bakeoff_v2.THRESHOLD_GRID
            ]
        },
    }

    threshold, rows = plate_bakeoff_v2.choose_candidate_threshold(metrics_by_clip)

    assert threshold == 0.1
    assert set(rows) == {"day_plate_primary", "night_plate_primary"}


def test_missing_candidate_weight_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise path validation without installing this archived adapter's runtime.
    # The empty module also fails if model compilation is attempted before validation.
    monkeypatch.setitem(sys.modules, "openvino", ModuleType("openvino"))
    candidate = plate_bakeoff_v2.CandidateSpec(
        detector_id="missing",
        display_name="Missing",
        adapter_type="open_image_models_yolov9_openvino",
        model_path=tmp_path / "missing.onnx",
        source_repo="source",
        source_revision="rev",
        source_license="license",
        model_license="license",
        architecture="arch",
        native_input=(608, 608),
        preprocessing="prep",
        output_format="output",
        postprocessing="post",
        runtime="runtime",
    )

    with pytest.raises(FileNotFoundError, match="YOLOv9 ONNX not found"):
        plate_bakeoff_v2.create_adapter(candidate)


def test_unsupported_candidate_adapter_is_rejected(tmp_path: Path) -> None:
    candidate = plate_bakeoff_v2.CandidateSpec(
        detector_id="bad",
        display_name="Bad",
        adapter_type="not_supported",
        model_path=tmp_path / "model.bin",
        source_repo="source",
        source_revision="rev",
        source_license="license",
        model_license="license",
        architecture="arch",
        native_input=(1, 1),
        preprocessing="prep",
        output_format="output",
        postprocessing="post",
        runtime="runtime",
    )

    with pytest.raises(ValueError, match="unsupported v2 plate candidate adapter"):
        plate_bakeoff_v2.create_adapter(candidate)
