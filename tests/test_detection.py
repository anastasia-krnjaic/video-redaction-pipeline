from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from redactor import cli as redactor_cli
from redactor import detection
from redactor.models import Detection, MediaInfo, VideoStreamInfo


class SequenceDetector:
    def __init__(self, detections_by_frame: list[list[Detection]]):
        self.detections_by_frame = detections_by_frame
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        assert int(frame[0, 0, 0]) == self.calls
        detections = self.detections_by_frame[self.calls]
        self.calls += 1
        return detections


class FakeCapture:
    instances: list[FakeCapture] = []

    def __init__(self, path: str, frame_count: int = 3):
        self.path = path
        self.frame_count = frame_count
        self.read_calls = 0
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_calls >= self.frame_count:
            return False, None
        frame = np.full((8, 10, 3), self.read_calls, dtype=np.uint8)
        self.read_calls += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeWriter:
    instances: list[FakeWriter] = []

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.write_count = 0
        self.released = False
        FakeWriter.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape[:2] == (8, 10)
        self.write_count += 1

    def release(self) -> None:
        self.released = True


def media_info_for(path: Path) -> MediaInfo:
    return MediaInfo(
        schema_version="1.0",
        created_at="2026-01-01T00:00:00Z",
        source_filename=path.name,
        source_path=path.name,
        file_size_bytes=123,
        sha256="1" * 64,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=1.0,
        video_stream=VideoStreamInfo(
            index=0,
            codec="h264",
            width=10,
            height=8,
            pixel_format="yuv420p",
            nominal_frame_rate="30000/1001",
            average_frame_rate="30000/1001",
            time_base="1/30000",
            frame_count="3",
        ),
        audio_stream=None,
        ffprobe_version="ffprobe version test",
    )


def patch_video_io(monkeypatch: pytest.MonkeyPatch, frame_count: int = 3) -> None:
    FakeCapture.instances.clear()
    FakeWriter.instances.clear()

    def fake_capture(path: str) -> FakeCapture:
        return FakeCapture(path, frame_count=frame_count)

    monkeypatch.setattr(detection.cv, "VideoCapture", fake_capture)
    monkeypatch.setattr(detection.cv, "VideoWriter", FakeWriter)


def test_bbox_conversion_and_clipping() -> None:
    assert detection.xywh_to_clipped_xyxy(1.2, 2.8, 3.1, 4.2, 10, 10) == (1, 2, 5, 7)
    assert detection.xywh_to_clipped_xyxy(-5, -2, 8, 6, 10, 10) == (0, 0, 3, 4)
    assert detection.quadrilateral_to_clipped_xyxy(
        np.array([-1.2, 1.0, 5.7, 0.5, 6.2, 7.9, 0.1, 8.3], dtype=np.float32),
        10,
        8,
    ) == (0, 0, 7, 8)


def test_rejects_invalid_boxes() -> None:
    assert detection.clip_xyxy_to_frame((5, 1, 5, 4), 10, 10) is None
    assert detection.clip_xyxy_to_frame((9, 9, 20, 20), 10, 10) == (9, 9, 10, 10)
    with pytest.raises(ValueError, match="x2"):
        Detection("face", 0.9, (2, 1, 2, 5))


def test_writes_one_jsonl_record_per_frame_including_empty_detections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "video with spaces.mp4"
    model_path = tmp_path / "model with spaces.onnx"
    model_path.write_bytes(b"fake onnx bytes")
    output_dir = tmp_path / "output dir"
    patch_video_io(monkeypatch)
    monkeypatch.setattr(detection.media, "probe_media", lambda path: media_info_for(path))

    detector = SequenceDetector(
        [
            [Detection("face", 0.9, (1, 1, 4, 4))],
            [],
            [Detection("face", 0.8, (2, 2, 5, 5))],
        ]
    )

    summary = detection.run_detection(
        input_path,
        category="face",
        model_path=model_path,
        threshold=0.6,
        output_dir=output_dir,
        overwrite=True,
        detector=detector,
    )

    lines = (output_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines == [
        '{"detections":[{"bbox_xyxy":[1,1,4,4],"category":"face","score":0.9}],"frame_index":0}',
        '{"detections":[],"frame_index":1}',
        '{"detections":[{"bbox_xyxy":[2,2,5,5],"category":"face","score":0.8}],"frame_index":2}',
    ]
    assert summary["frames_processed"] == 3
    assert summary["frames_containing_at_least_one_detection"] == 2
    assert summary["total_detections"] == 2
    assert FakeWriter.instances[0].write_count == 3


def test_json_representation_is_deterministic() -> None:
    detection_obj = Detection("license_plate", 0.75, (1, 2, 3, 4))
    record = {
        "detections": [detection_obj.to_dict()],
        "frame_index": 12,
    }

    assert json.dumps(record, sort_keys=True, separators=(",", ":")) == (
        '{"detections":[{"bbox_xyxy":[1,2,3,4],"category":"license_plate","score":0.75}],'
        '"frame_index":12}'
    )


def test_adas_preprocess_uses_bgr_chw_float_tensor() -> None:
    frame = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )

    tensor = detection.preprocess_adas_frame(frame, input_width=2, input_height=2)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert tensor[0, :, 1, 1].tolist() == [10.0, 11.0, 12.0]


def test_adas_normalized_coordinate_conversion_and_boundary_clipping() -> None:
    assert detection.normalized_to_adas_xyxy(0.25, 0.25, 0.75, 0.5, 1920, 1080) == (
        480,
        270,
        1440,
        540,
    )
    assert detection.normalized_to_adas_xyxy(-0.1, -0.2, 1.2, 1.1, 100, 80) == (
        0,
        0,
        100,
        80,
    )
    assert detection.normalized_to_adas_xyxy(0.5, 0.5, 0.5, 0.8, 100, 80) is None


def test_adas_output_tensor_parsing_thresholding_and_ordering() -> None:
    raw = np.array(
        [
            [0, 1, 0.30, 0.20, 0.20, 0.40, 0.40],
            [0, 1, 0.199, 0.10, 0.10, 0.20, 0.20],
            [0, 1, 0.20, 0.60, 0.60, 0.60, 0.70],
            [0, 1, 0.90, 0.50, 0.50, 0.70, 0.80],
            [-1, 1, 0.99, 0.00, 0.00, 1.00, 1.00],
        ],
        dtype=np.float32,
    )

    detections = detection.parse_adas_detections(
        raw,
        frame_width=100,
        frame_height=80,
        proposal_threshold=0.20,
    )

    assert [item.score for item in detections] == [
        pytest.approx(0.90),
        pytest.approx(0.30),
    ]
    assert [item.bbox_xyxy for item in detections] == [(50, 40, 70, 64), (20, 16, 40, 32)]
    assert all(item.category == "face" for item in detections)
    assert all(item.detector_id == detection.DETECTOR_BACKEND_ADAS for item in detections)
    assert all(item.label == 1 for item in detections)


def test_adas_proposal_classification_threshold_boundaries() -> None:
    classify = detection.classify_observation

    assert classify(0.199999, proposal_threshold=0.20, primary_threshold=0.30) == (
        "below_proposal_floor"
    )
    assert classify(0.20, proposal_threshold=0.20, primary_threshold=0.30) == "weak"
    assert classify(0.299999, proposal_threshold=0.20, primary_threshold=0.30) == "weak"
    assert classify(0.30, proposal_threshold=0.20, primary_threshold=0.30) == "primary"


def test_adas_frame_record_includes_canonical_metadata() -> None:
    output = io.StringIO()

    detection.write_frame_record(
        output,
        12,
        [
            Detection(
                "face",
                0.9,
                (10, 20, 30, 40),
                detector_id=detection.DETECTOR_BACKEND_ADAS,
                label=1,
            )
        ],
        clip_id="day_face",
    )

    assert json.loads(output.getvalue()) == {
        "clip": "day_face",
        "detections": [
            {
                "bbox_xyxy": [10, 20, 30, 40],
                "candidate_id": "openvino_face_adas_0001:day_face:f0012:d00",
                "category": "face",
                "detector_id": "openvino_face_adas_0001",
                "label": 1,
                "score": 0.9,
                "source_detection_index": 0,
            }
        ],
        "frame_index": 12,
    }


def test_adas_empty_frame_record_includes_clip() -> None:
    output = io.StringIO()

    detection.write_frame_record(output, 5, [], clip_id="night_selective")

    assert output.getvalue().strip() == '{"clip":"night_selective","detections":[],"frame_index":5}'


def test_model_file_hashes_include_adas_xml_and_bin(tmp_path: Path) -> None:
    xml = tmp_path / "face-detection-adas-0001.xml"
    bin_path = tmp_path / "face-detection-adas-0001.bin"
    xml.write_bytes(b"xml")
    bin_path.write_bytes(b"bin")

    assert detection.model_file_hashes(xml, bin_path) == {
        "face-detection-adas-0001.bin": hashlib.sha256(b"bin").hexdigest(),
        "face-detection-adas-0001.xml": hashlib.sha256(b"xml").hexdigest(),
    }


def test_missing_adas_model_fails_clearly(tmp_path: Path) -> None:
    missing_xml = tmp_path / "missing.xml"

    with pytest.raises(FileNotFoundError, match="model file not found"):
        detection.OpenVINOFaceADASDetector(missing_xml)


def test_face_backend_rejects_removed_yunet_selection() -> None:
    with pytest.raises(ValueError, match="active face detector is openvino_face_adas_0001"):
        detection.resolve_detector_backend("face", "yunet")


def test_face_pipeline_help_has_no_detector_bakeoff_selector() -> None:
    result = CliRunner().invoke(redactor_cli.main, ["face-pipeline", "--help"])

    assert result.exit_code == 0
    assert "--detector-backend" not in result.output
    assert "yunet" not in result.output.lower()
    assert "face-detection-adas-0001" in result.output


def test_removed_face_yunet_backend_is_rejected_by_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"not a real video")

    detect_result = CliRunner().invoke(
        redactor_cli.main,
        [
            "detect",
            str(input_path),
            "--category",
            "face",
            "--detector-backend",
            "yunet",
            "--output-dir",
            str(tmp_path / "detect"),
        ],
    )
    pipeline_result = CliRunner().invoke(
        redactor_cli.main,
        [
            "face-pipeline",
            str(input_path),
            "--detector-backend",
            "yunet",
            "--output-dir",
            str(tmp_path / "pipeline"),
        ],
    )

    assert detect_result.exit_code != 0
    assert "Invalid value for '--detector-backend'" in detect_result.output
    assert pipeline_result.exit_code != 0
    assert "No such option '--detector-backend'" in pipeline_result.output


def test_backend_selection_defaults_face_to_adas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    xml = tmp_path / "face-detection-adas-0001.xml"
    bin_path = tmp_path / "face-detection-adas-0001.bin"
    xml.write_bytes(b"xml")
    bin_path.write_bytes(b"bin")
    created: dict[str, object] = {}

    class FakeADAS:
        detector_id = detection.DETECTOR_BACKEND_ADAS

        def __init__(
            self,
            model_xml_path: Path,
            *,
            model_bin_path: Path | None,
            proposal_threshold: float,
            primary_threshold: float,
        ):
            created["model_xml_path"] = model_xml_path
            created["model_bin_path"] = model_bin_path
            created["proposal_threshold"] = proposal_threshold
            created["primary_threshold"] = primary_threshold

        def detect(self, frame: np.ndarray) -> list[Detection]:
            return []

    monkeypatch.setattr(detection, "OpenVINOFaceADASDetector", FakeADAS)

    detector = detection.create_detector("face", xml, 0.20, adas_bin_path=bin_path)

    assert isinstance(detector, FakeADAS)
    assert created == {
        "model_xml_path": xml,
        "model_bin_path": bin_path,
        "proposal_threshold": 0.20,
        "primary_threshold": 0.30,
    }


def test_overwrite_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        detection.ensure_output_dir(output_dir, overwrite=False)

    detection.ensure_output_dir(output_dir, overwrite=True)


def test_safe_paths_with_spaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "clip with spaces.mp4"
    model_path = tmp_path / "model with spaces.onnx"
    model_path.write_bytes(b"fake onnx bytes")
    output_dir = tmp_path / "output with spaces"
    patch_video_io(monkeypatch, frame_count=1)
    monkeypatch.setattr(detection.media, "probe_media", lambda path: media_info_for(path))

    detection.run_detection(
        input_path,
        category="face",
        model_path=model_path,
        threshold=0.6,
        output_dir=output_dir,
        overwrite=True,
        max_frames=1,
        detector=SequenceDetector([[]]),
    )

    assert FakeCapture.instances[0].path == str(input_path)
    assert FakeWriter.instances[0].path == str(output_dir / "overlay.mp4")


def test_summary_count_calculations(tmp_path: Path) -> None:
    stats = detection.DetectionStats()
    for count in (0, 2, 1):
        stats.update(count)

    summary = detection.build_summary(
        created_at="2026-01-01T00:00:00Z",
        input_video=tmp_path / "input.mp4",
        source_sha256="1" * 64,
        source_frame_count="3",
        category="face",
        model_path=tmp_path / "model.onnx",
        model_sha256_value="2" * 64,
        threshold=0.6,
        stats=stats,
        elapsed_seconds=2.0,
        width=10,
        height=8,
        nominal_frame_rate="30000/1001",
    )

    assert summary["frames_processed"] == 3
    assert summary["frames_containing_at_least_one_detection"] == 2
    assert summary["total_detections"] == 3
    assert summary["min_detections_in_frame"] == 0
    assert summary["max_detections_in_frame"] == 2
    assert summary["processing_fps"] == 1.5


def test_model_file_hashing_and_lfs_pointer_rejection(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_bytes = b"\x08\x01binary model"
    model_path.write_bytes(model_bytes)

    assert detection.model_sha256(model_path) == hashlib.sha256(model_bytes).hexdigest()

    pointer_path = tmp_path / "pointer.onnx"
    pointer_path.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 456\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        detection.model_sha256(pointer_path)


def test_detection_does_not_use_subprocess() -> None:
    assert not hasattr(detection, "subprocess")


def test_max_frames_stops_incremental_video_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake onnx bytes")
    output_dir = tmp_path / "output"
    patch_video_io(monkeypatch, frame_count=10)
    monkeypatch.setattr(detection.media, "probe_media", lambda path: media_info_for(path))

    detection.run_detection(
        input_path,
        category="face",
        model_path=model_path,
        threshold=0.6,
        output_dir=output_dir,
        overwrite=True,
        max_frames=2,
        detector=SequenceDetector([[], []]),
    )

    assert FakeCapture.instances[0].read_calls == 2
    assert FakeWriter.instances[0].write_count == 2
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_video_frame_count"] == 3
    assert summary["processed_frame_count"] == 2
    assert summary["frames_processed"] == 2
