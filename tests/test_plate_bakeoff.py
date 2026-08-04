from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from redactor import plate_bakeoff


def detection(
    score: float,
    bbox: tuple[int, int, int, int],
    *,
    frame_index: int = 0,
    detector_id: str = "detector",
) -> plate_bakeoff.PlateDetection:
    return plate_bakeoff.PlateDetection(
        detector_id=detector_id,
        frame_index=frame_index,
        score=score,
        bbox_xyxy=bbox,
        inference_time_ms=1.0,
    )


def benchmark(tmp_path: Path) -> plate_bakeoff.PlateBenchmark:
    return plate_bakeoff.PlateBenchmark(
        clip_id="synthetic_plate",
        video_path=tmp_path / "video.mp4",
        source_sha256="a" * 64,
        evaluated_ranges=((10, 13),),
        target_id="plate_01",
        target_roi_xyxy=(10, 10, 30, 30),
    )


def test_normalized_plate_record_is_deterministic() -> None:
    item = detection(0.5, (1, 2, 3, 4), frame_index=9)

    assert item.to_dict() == {
        "bbox_xyxy": [1, 2, 3, 4],
        "category": "license_plate",
        "detector_id": "detector",
        "frame_index": 9,
        "inference_time_ms": 1.0,
        "postprocessing": {},
        "preprocessing": {},
        "score": 0.5,
    }


def test_normalized_plate_record_can_preserve_geometry() -> None:
    item = plate_bakeoff.PlateDetection(
        detector_id="detector",
        frame_index=4,
        score=0.8,
        bbox_xyxy=(10, 20, 30, 40),
        inference_time_ms=1.0,
        geometry={
            "derived_redaction_bbox_xyxy": [10, 20, 30, 40],
            "model_geometry_type": "axis_aligned_xywh",
            "oriented_bbox": None,
        },
    )

    assert item.to_dict()["geometry"] == {
        "derived_redaction_bbox_xyxy": [10, 20, 30, 40],
        "model_geometry_type": "axis_aligned_xywh",
        "oriented_bbox": None,
    }


def test_rejects_invalid_plate_record() -> None:
    with pytest.raises(ValueError, match="positive area"):
        detection(0.9, (1, 1, 1, 2))
    with pytest.raises(ValueError, match="license_plate"):
        plate_bakeoff.PlateDetection(
            category="face",
            detector_id="bad",
            frame_index=0,
            score=0.9,
            bbox_xyxy=(1, 1, 2, 2),
            inference_time_ms=1.0,
        )


def test_score_thresholding_and_longest_miss(tmp_path: Path) -> None:
    records = {
        10: [detection(0.8, (12, 12, 18, 18), frame_index=10)],
        11: [detection(0.1, (12, 12, 18, 18), frame_index=11)],
        12: [detection(0.7, (40, 40, 60, 60), frame_index=12)],
        13: [],
    }

    metrics = plate_bakeoff.evaluate_threshold(records, benchmark(tmp_path), threshold=0.5)

    assert metrics["detected_target_frames"] == 1
    assert metrics["missed_frames"] == 3
    assert metrics["longest_consecutive_miss_run"] == 3
    assert metrics["false_positive_frames"] == 1
    assert metrics["object_frame_recall"] == 0.25


def test_metrics_report_duplicates_before_and_after_nms(tmp_path: Path) -> None:
    records = {
        10: [detection(0.9, (12, 12, 18, 18), frame_index=10)],
        11: [],
        12: [],
        13: [],
    }
    pre_nms = {
        10: [
            detection(0.9, (12, 12, 18, 18), frame_index=10),
            detection(0.8, (13, 13, 19, 19), frame_index=10),
        ],
        11: [],
        12: [],
        13: [],
    }

    metrics = plate_bakeoff.evaluate_threshold(
        records,
        benchmark(tmp_path),
        threshold=0.5,
        pre_nms_records_by_frame=pre_nms,
    )

    assert metrics["duplicate_frames_before_nms"] == 1
    assert metrics["duplicate_frames_after_nms"] == 0
    assert metrics["detections_per_frame_mean_before_nms"] == 0.5
    assert metrics["detections_per_frame_mean_after_nms"] == 0.25


def test_detector_ignore_absent_for_unscored_close_clip(tmp_path: Path) -> None:
    spec = plate_bakeoff.PlateBenchmark(
        clip_id="close",
        video_path=tmp_path / "video.mp4",
        source_sha256="b" * 64,
        evaluated_ranges=((0, 2),),
        certified_quantitative=False,
        target_roi_xyxy=None,
    )

    metrics = plate_bakeoff.evaluate_threshold(
        {0: [detection(0.9, (1, 1, 2, 2))]}, spec, threshold=0.5
    )

    assert metrics["certified_quantitative"] is False
    assert metrics["object_frame_recall"] is None
    assert metrics["unscored_reason"]


def test_box_clipping_and_ssd_plate_label_filter() -> None:
    raw = np.array(
        [
            [0, 1, 0.99, 0.1, 0.1, 0.2, 0.2],
            [0, 2, 0.20, -0.1, 0.25, 0.5, 0.75],
            [0, 2, 0.009, 0.1, 0.1, 0.2, 0.2],
            [-1, 2, 1.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    detections = plate_bakeoff.parse_ssd_output(
        raw,
        frame_width=100,
        frame_height=80,
        detector_id="ssd",
        score_floor=0.01,
        inference_time_ms=2.0,
    )

    assert len(detections) == 1
    assert detections[0].label == 2
    assert detections[0].bbox_xyxy == (0, 20, 50, 60)


def test_yolov8_letterbox_mapping_and_thresholding() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    _tensor, mapping = plate_bakeoff.preprocess_yolov8_frame(
        frame,
        input_width=64,
        input_height=64,
    )
    raw = np.zeros((1, 5, 2), dtype=np.float32)
    raw[0, :, 0] = [32, 32, 20, 10, 0.9]
    raw[0, :, 1] = [32, 32, 20, 10, 0.1]

    detections = plate_bakeoff.parse_yolov8_output(
        raw,
        frame_width=100,
        frame_height=50,
        detector_id="yolo",
        score_floor=0.5,
        nms_iou_threshold=0.45,
        letterbox_mapping=mapping,
        inference_time_ms=3.0,
    )

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (34, 17, 66, 33)


def test_yolov8_decode_preserves_pre_nms_candidates() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    _tensor, mapping = plate_bakeoff.preprocess_yolov8_frame(
        frame,
        input_width=64,
        input_height=64,
    )
    raw = np.zeros((1, 5, 3), dtype=np.float32)
    raw[0, :, 0] = [32, 32, 20, 10, 0.9]
    raw[0, :, 1] = [32, 32, 21, 11, 0.8]
    raw[0, :, 2] = [10, 10, 4, 4, 0.1]

    run = plate_bakeoff.parse_yolov8_output_run(
        raw,
        frame_width=100,
        frame_height=50,
        detector_id="yolo",
        score_floor=0.5,
        nms_iou_threshold=0.45,
        letterbox_mapping=mapping,
        inference_time_ms=3.0,
    )

    assert len(run.pre_nms_detections) == 2
    assert len(run.final_detections) == 1
    assert run.diagnostics["model_geometry_type"] == "axis_aligned_xywh"
    assert run.final_detections[0].geometry["oriented_bbox"] is None


def test_yolov8_rejects_unexpected_output_rows() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    _tensor, mapping = plate_bakeoff.preprocess_yolov8_frame(
        frame,
        input_width=64,
        input_height=64,
    )

    with pytest.raises(ValueError, match="5 rows"):
        plate_bakeoff.parse_yolov8_output_run(
            np.zeros((1, 6, 1), dtype=np.float32),
            frame_width=100,
            frame_height=50,
            detector_id="yolo",
            score_floor=0.5,
            nms_iou_threshold=0.45,
            letterbox_mapping=mapping,
            inference_time_ms=3.0,
        )


def test_non_maximum_suppression_removes_duplicate_boxes() -> None:
    kept = plate_bakeoff.non_max_suppression(
        [
            detection(0.9, (10, 10, 30, 30)),
            detection(0.8, (11, 11, 31, 31)),
            detection(0.7, (40, 40, 60, 60)),
        ],
        iou_threshold=0.5,
    )

    assert [item.score for item in kept] == [0.9, 0.7]


def test_oriented_box_geometry_and_nms() -> None:
    bbox = plate_bakeoff.oriented_box_to_axis_aligned_xyxy(
        cx=50,
        cy=50,
        width=40,
        height=20,
        angle_degrees=45,
        frame_width=100,
        frame_height=100,
    )
    assert bbox is not None
    assert bbox[0] < 50 < bbox[2]
    assert bbox[1] < 50 < bbox[3]
    first = detection(0.9, (25, 25, 75, 75)).with_metadata(
        geometry={
            "oriented_bbox": {
                "angle_degrees": 10,
                "cx": 50,
                "cy": 50,
                "height": 20,
                "width": 40,
            }
        }
    )
    second = detection(0.8, (26, 26, 76, 76)).with_metadata(
        geometry={
            "oriented_bbox": {
                "angle_degrees": 10,
                "cx": 51,
                "cy": 50,
                "height": 20,
                "width": 40,
            }
        }
    )

    assert (
        plate_bakeoff.oriented_bbox_iou(
            first.geometry["oriented_bbox"],
            second.geometry["oriented_bbox"],
        )
        > 0.8
    )
    assert plate_bakeoff.oriented_or_axis_aligned_nms(
        [first, second],
        iou_threshold=0.5,
    ) == [first]


def test_uniform_tiles_cover_frame_with_full_frame() -> None:
    tiles = plate_bakeoff.build_uniform_tiles(
        1920,
        1080,
        tile_size=960,
        overlap_fraction=0.25,
        include_full_frame=True,
    )

    assert (0, 0, 1920, 1080) in tiles
    assert all(x2 <= 1920 and y2 <= 1080 for x1, y1, x2, y2 in tiles)
    assert all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in tiles)


def test_tiled_coordinate_restoration_and_cross_tile_nms() -> None:
    first = detection(0.9, (100, 100, 150, 130))
    second = detection(0.8, (102, 101, 152, 131))

    kept = plate_bakeoff.non_max_suppression([first, second], iou_threshold=0.45)

    assert kept == [first]


def test_empty_frames_are_serialized(tmp_path: Path) -> None:
    path = tmp_path / "detections.jsonl"
    path.write_text(
        '{"clip_id":"clip","detections":[],"frame_index":0}\n',
        encoding="utf-8",
    )

    assert plate_bakeoff.load_detection_jsonl(path) == {0: []}


def test_candidate_metadata_records_model_hashes(tmp_path: Path) -> None:
    xml = tmp_path / "model.xml"
    bin_path = tmp_path / "model.bin"
    xml.write_text("<xml />", encoding="utf-8")
    bin_path.write_bytes(b"weights")
    candidate = plate_bakeoff.PlateCandidate(
        detector_id="candidate",
        display_name="Candidate",
        adapter_type="openvino_ssd",
        model_path=xml,
        model_bin_path=bin_path,
    )

    metadata = candidate.metadata()

    assert sorted(metadata["model_files_sha256"]) == ["model.bin", "model.xml"]
    assert metadata["threshold_grid"] == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_unsupported_candidate_adapter_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    candidate = plate_bakeoff.PlateCandidate(
        detector_id="bad",
        display_name="Bad",
        adapter_type="unknown",
        model_path=model,
    )

    with pytest.raises(ValueError, match="unsupported plate bake-off candidate"):
        plate_bakeoff.create_adapter(candidate)
