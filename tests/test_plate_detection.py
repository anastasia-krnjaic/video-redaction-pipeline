from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from redactor import cli as redactor_cli
from redactor import detection, plate_detection


def test_oim_preprocessing_preserves_letterbox_mapping() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    tensor, mapping = plate_detection.preprocess_letterbox_rgb_01(frame, 64)

    assert tensor.shape == (1, 3, 64, 64)
    assert mapping.original_width == 100
    assert mapping.original_height == 50
    assert mapping.resized_width == 64
    assert mapping.resized_height == 32
    assert mapping.pad_x == 0
    assert mapping.pad_y == 16


def test_oim_output_thresholding_and_coordinate_restoration() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    _tensor, mapping = plate_detection.preprocess_letterbox_rgb_01(frame, 64)
    raw = np.array(
        [
            [0, 20, 28, 44, 36, 0, 0.90],
            [0, 20, 28, 44, 36, 1, 0.95],
            [0, 20, 28, 44, 36, 0, 0.29],
        ],
        dtype=np.float32,
    )

    detections = plate_detection.parse_oim_yolov9_output(
        raw,
        frame_width=100,
        frame_height=50,
        mapping=mapping,
        score_floor=0.30,
        inference_time_ms=1.0,
    )

    assert len(detections) == 1
    assert detections[0].label == 0
    assert detections[0].bbox_xyxy == (31, 18, 69, 32)


def test_oim_empty_outputs_are_supported() -> None:
    _tensor, mapping = plate_detection.preprocess_letterbox_rgb_01(
        np.zeros((50, 100, 3), dtype=np.uint8),
        64,
    )

    detections = plate_detection.parse_oim_yolov9_output(
        np.zeros((0, 7), dtype=np.float32),
        frame_width=100,
        frame_height=50,
        mapping=mapping,
        score_floor=0.30,
        inference_time_ms=1.0,
    )

    assert detections == []


def test_uniform_small_object_tiles_are_global_not_clip_specific() -> None:
    tiles = plate_detection.build_uniform_tiles(
        1920,
        1080,
        tile_size=plate_detection.TILE_SIZE,
        overlap_fraction=plate_detection.TILE_OVERLAP,
        include_full_frame=True,
    )

    assert (0, 0, 1920, 1080) in tiles
    assert (0, 0, 1280, 1080) in tiles
    assert (640, 0, 1920, 1080) in tiles


def test_cross_tile_duplicate_suppression() -> None:
    detections = [
        plate_detection.ParsedPlateDetection(
            score=0.9,
            bbox_xyxy=(10, 10, 50, 25),
            label=0,
            source_detection_index=0,
            inference_time_ms=1.0,
            geometry={},
            preprocessing={},
            postprocessing={},
        ),
        plate_detection.ParsedPlateDetection(
            score=0.8,
            bbox_xyxy=(11, 10, 51, 25),
            label=0,
            source_detection_index=1,
            inference_time_ms=1.0,
            geometry={},
            preprocessing={},
            postprocessing={},
        ),
    ]

    kept = plate_detection.non_max_suppression(detections, iou_threshold=0.45)

    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_configuration_hash_is_deterministic() -> None:
    assert plate_detection.OIMPlateDetectorConfig().configuration_hash() == (
        plate_detection.OIMPlateDetectorConfig().configuration_hash()
    )
    assert plate_detection.OIMPlateDetectorConfig().to_dict()["primary_threshold"] == 0.30


def test_active_configuration_rejects_lower_proposal_threshold() -> None:
    try:
        plate_detection.OIMPlateDetectorConfig(
            proposal_threshold=0.20,
            primary_threshold=0.30,
        )
    except ValueError as exc:
        assert "one selected threshold" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("split OIM plate config was accepted")


def test_default_plate_model_path_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("REDACTOR_OIM_PLATE_MODEL", "models/custom/oim.onnx")

    assert plate_detection.default_model_path() == Path("models/custom/oim.onnx")


def test_active_plate_backend_is_oim_only() -> None:
    assert detection.resolve_detector_backend("license_plate", None) == "oim_yolov9_s_608"
    assert (
        detection.resolve_detector_backend("license_plate", "oim_yolov9_s_608")
        == "oim_yolov9_s_608"
    )

    for rejected in ["license_plate_yunet", "rtdetr", "rfdetr", "intel_yolo", "openvino_0106"]:
        try:
            detection.resolve_detector_backend("license_plate", rejected)
        except ValueError as exc:
            assert "active plate detector is oim_yolov9_s_608" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"rejected backend was accepted: {rejected}")


def test_active_plate_detector_rejects_split_threshold_policy() -> None:
    try:
        detection.resolve_primary_threshold(
            "oim_yolov9_s_608",
            proposal_threshold=0.20,
            primary_threshold=0.30,
        )
    except ValueError as exc:
        assert "one selected threshold" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("split OIM plate thresholds were accepted")


def test_detect_plates_help_is_oim_only() -> None:
    result = CliRunner().invoke(redactor_cli.main, ["detect-plates", "--help"])

    assert result.exit_code == 0
    assert "Open Image Models YOLOv9-S-608" in result.output
    assert "license_plate_yunet" not in result.output
    assert "rtdetr" not in result.output.lower()


def test_generic_detect_rejects_removed_plate_backend(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")

    result = CliRunner().invoke(
        redactor_cli.main,
        [
            "detect",
            str(input_path),
            "--category",
            "license_plate",
            "--detector-backend",
            "license_plate_yunet",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--detector-backend'" in result.output


def test_production_plate_code_does_not_import_archived_modules() -> None:
    active_files = [
        Path("src/redactor/detection.py"),
        Path("src/redactor/plate_detection.py"),
        Path("src/redactor/cli.py"),
    ]

    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert "experiments.archive.license_plate_detector_bakeoff" not in text
