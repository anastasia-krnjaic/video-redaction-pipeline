from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Protocol

import cv2 as cv
import numpy as np

from redactor import media
from redactor.models import Detection

SCHEMA_VERSION = "1.0"
DETECTOR_BACKEND_ADAS = "openvino_face_adas_0001"
DETECTOR_BACKEND_LICENSE_PLATE_YUNET = "license_plate_yunet"
DEFAULT_FACE_DETECTOR_BACKEND = DETECTOR_BACKEND_ADAS
DEFAULT_ADAS_PROPOSAL_THRESHOLD = 0.20
DEFAULT_ADAS_PRIMARY_THRESHOLD = 0.30
DEFAULT_ADAS_XML_PATH = Path("models/face-detection-adas-0001/face-detection-adas-0001.xml")
DEFAULT_THRESHOLDS = {
    "face": DEFAULT_ADAS_PROPOSAL_THRESHOLD,
    "license_plate": 0.40,
}
NMS_THRESHOLD = 0.30
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass
class DetectionStats:
    frames_processed: int = 0
    frames_with_detections: int = 0
    total_detections: int = 0
    min_detections_in_frame: int | None = None
    max_detections_in_frame: int = 0

    def update(self, detection_count: int) -> None:
        self.frames_processed += 1
        self.total_detections += detection_count
        if detection_count > 0:
            self.frames_with_detections += 1
        if self.min_detections_in_frame is None:
            self.min_detections_in_frame = detection_count
        else:
            self.min_detections_in_frame = min(self.min_detections_in_frame, detection_count)
        self.max_detections_in_frame = max(self.max_detections_in_frame, detection_count)

    @property
    def finalized_min_detections(self) -> int:
        return self.min_detections_in_frame if self.min_detections_in_frame is not None else 0


class OpenVINOFaceADASDetector:
    detector_id = DETECTOR_BACKEND_ADAS
    expected_color_order = "BGR"
    model_source = "OpenVINO Open Model Zoo face-detection-adas-0001 FP32"

    def __init__(
        self,
        model_xml_path: Path,
        *,
        model_bin_path: Path | None = None,
        proposal_threshold: float = DEFAULT_ADAS_PROPOSAL_THRESHOLD,
        primary_threshold: float = DEFAULT_ADAS_PRIMARY_THRESHOLD,
        device: str = "CPU",
    ):
        validate_threshold_pair(proposal_threshold, primary_threshold)
        self.model_xml_path = model_xml_path
        self.model_bin_path = model_bin_path or model_xml_path.with_suffix(".bin")
        self.proposal_threshold = proposal_threshold
        self.primary_threshold = primary_threshold
        self.device = device
        self.model_file_hashes = model_file_hashes(model_xml_path, self.model_bin_path)

        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO is required for openvino_face_adas_0001; install the openvino "
                "Python package and provide the ADAS XML/BIN model files"
            ) from exc

        self.openvino_version = ov.__version__
        try:
            self.core = ov.Core()
            model = self.core.read_model(str(model_xml_path), str(self.model_bin_path))
            self.compiled_model = self.core.compile_model(model, device)
            self.input_port = self.compiled_model.input(0)
            self.output_port = self.compiled_model.output(0)
        except Exception as exc:
            raise ValueError(
                f"failed to load OpenVINO ADAS face model {model_xml_path.name}"
            ) from exc

        self.input_shape = [int(dim) for dim in self.input_port.shape]
        if len(self.input_shape) != 4:
            raise ValueError(f"expected four-dimensional ADAS input, got {self.input_shape}")
        _, channels, self.input_height, self.input_width = self.input_shape
        if channels != 3:
            raise ValueError(f"expected three-channel ADAS input, got {self.input_shape}")
        self.output_shape = [int(dim) for dim in self.output_port.shape]
        self.input_name = self.input_port.get_any_name()
        self.output_name = self.output_port.get_any_name()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor = preprocess_adas_frame(
            frame,
            input_width=self.input_width,
            input_height=self.input_height,
        )
        result = self.compiled_model({self.input_port: tensor})
        return parse_adas_detections(
            result[self.output_port],
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            proposal_threshold=self.proposal_threshold,
        )

    def summary_metadata(self) -> dict[str, object]:
        return {
            "detector_id": self.detector_id,
            "expected_color_order": self.expected_color_order,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "model_files_sha256": self.model_file_hashes,
            "model_precision": "FP32",
            "model_source": self.model_source,
            "openvino_version": self.openvino_version,
            "output_format": (
                "SSD detections: [image_id, label, confidence, x_min, y_min, x_max, y_max] "
                "with normalized box coordinates"
            ),
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "primary_threshold": self.primary_threshold,
            "proposal_threshold": self.proposal_threshold,
            "runtime": f"OpenVINO {self.device}",
        }


class LicensePlateYuNetDetector:
    detector_id = DETECTOR_BACKEND_LICENSE_PLATE_YUNET

    def __init__(
        self,
        model_path: Path,
        threshold: float,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = 5000,
        keep_top_k: int = 750,
    ):
        try:
            self.net = cv.dnn.readNet(str(model_path))
        except cv.error as exc:
            raise ValueError(f"failed to load license-plate model {model_path.name}") from exc
        self.threshold = threshold
        self.model_path = model_path
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.output_names = ["loc", "conf", "iou"]
        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]
        self.input_size: tuple[int, int] | None = None
        self.priors = np.empty((0, 4), dtype=np.float32)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        self.set_input_size(width, height)
        blob = cv.dnn.blobFromImage(frame)
        self.net.setInput(blob)
        output_blob = self.net.forward(self.output_names)
        dets = self._postprocess(output_blob)

        detections: list[Detection] = []
        for det in dets:
            score = float(det[-1])
            bbox = quadrilateral_to_clipped_xyxy(det[:8], width, height)
            if bbox is not None:
                detections.append(
                    Detection("license_plate", score, bbox, detector_id=self.detector_id)
                )
        return sorted(detections, key=detection_sort_key)

    def set_input_size(self, width: int, height: int) -> None:
        input_size = (width, height)
        if input_size == self.input_size:
            return
        self.input_size = input_size
        self.priors = self._generate_priors(width, height)

    def _postprocess(self, blob: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
        dets = self._decode(blob)
        if dets.size == 0:
            return np.empty((0, 9), dtype=np.float32)

        keep_mask = dets[:, -1] >= self.threshold
        candidates = dets[keep_mask]
        if candidates.size == 0:
            return np.empty((0, 9), dtype=np.float32)

        rects = [_quadrilateral_to_xywh(det[:8]) for det in candidates]
        scores = candidates[:, -1].tolist()
        keep_indices = normalize_nms_indices(
            cv.dnn.NMSBoxes(
                bboxes=rects,
                scores=scores,
                score_threshold=self.threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
        )
        if not keep_indices:
            return np.empty((0, 9), dtype=np.float32)
        return candidates[keep_indices[: self.keep_top_k]]

    def _decode(self, blob: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
        loc, conf, iou = (_squeeze_output(output) for output in blob)
        if loc.size == 0:
            return np.empty((0, 9), dtype=np.float32)

        cls_scores = np.maximum(conf[:, 1], 0.0)
        iou_scores = np.clip(iou[:, 0], 0.0, 1.0)
        scores = np.sqrt(cls_scores * iou_scores)[:, np.newaxis]

        scale = np.array(self.input_size, dtype=np.float32)
        bboxes = np.hstack(
            (
                (self.priors[:, 0:2] + loc[:, 4:6] * self.variance[0] * self.priors[:, 2:4])
                * scale,
                (self.priors[:, 0:2] + loc[:, 6:8] * self.variance[0] * self.priors[:, 2:4])
                * scale,
                (self.priors[:, 0:2] + loc[:, 10:12] * self.variance[0] * self.priors[:, 2:4])
                * scale,
                (self.priors[:, 0:2] + loc[:, 12:14] * self.variance[0] * self.priors[:, 2:4])
                * scale,
            )
        )
        return np.hstack((bboxes, scores)).astype(np.float32)

    def _generate_priors(self, width: int, height: int) -> np.ndarray:
        feature_map_2th = [int(int((height + 1) / 2) / 2), int(int((width + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]
        feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]

        priors = []
        for k, feature_map in enumerate(feature_maps):
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in self.min_sizes[k]:
                    priors.append(
                        [
                            (j + 0.5) * self.steps[k] / width,
                            (i + 0.5) * self.steps[k] / height,
                            min_size / width,
                            min_size / height,
                        ]
                    )
        return np.array(priors, dtype=np.float32)


def run_detection(
    input_video: Path,
    *,
    category: str,
    model_path: Path,
    threshold: float,
    output_dir: Path,
    overwrite: bool = False,
    max_frames: int | None = None,
    detector: Detector | None = None,
    detector_backend: str | None = None,
    adas_bin_path: Path | None = None,
    primary_threshold: float | None = None,
    clip_id: str | None = None,
    detections_output_path: Path | None = None,
    write_overlay_video: bool = True,
    write_summary_file: bool = True,
) -> dict[str, object]:
    if category not in DEFAULT_THRESHOLDS:
        raise ValueError(f"unsupported category: {category}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max-frames must be greater than 0")

    ensure_output_dir(output_dir, overwrite=overwrite)
    resolved_backend = (
        resolve_detector_backend(category, detector_backend)
        if detector is None or detector_backend is not None
        else getattr(detector, "detector_id", "injected_detector")
    )
    resolved_primary_threshold = resolve_primary_threshold(
        resolved_backend,
        proposal_threshold=threshold,
        primary_threshold=primary_threshold,
    )
    model_sha = model_sha256(model_path)
    detector = detector or create_detector(
        category,
        model_path,
        threshold,
        detector_backend=resolved_backend,
        adas_bin_path=adas_bin_path,
        primary_threshold=resolved_primary_threshold,
    )
    detector_metadata = detector_summary_metadata(detector)

    source_info = media.probe_media(input_video)
    source_sha = source_info.sha256
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    frame_rate = source_info.video_stream.nominal_frame_rate
    fps = rational_to_float(frame_rate) or 30.0

    capture = cv.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_video}")

    writer = None
    if write_overlay_video:
        overlay_path = output_dir / "overlay.mp4"
        writer = cv.VideoWriter(
            str(overlay_path),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"could not create overlay video: {overlay_path}")

    stats = DetectionStats()
    jsonl_path = detections_output_path or output_dir / "detections.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    detector_elapsed_seconds = 0.0
    created_at = media.utc_timestamp()
    frame_index = 0

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                detector_started_at = time.perf_counter()
                detections = detector.detect(frame)
                detector_elapsed_seconds += time.perf_counter() - detector_started_at
                write_frame_record(jsonl_file, frame_index, detections, clip_id=clip_id)
                if writer is not None:
                    writer.write(draw_overlay(frame, frame_index, detections))
                stats.update(len(detections))
                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_seconds = time.perf_counter() - started_at
    summary = build_summary(
        created_at=created_at,
        input_video=input_video,
        source_sha256=source_sha,
        source_frame_count=source_info.video_stream.frame_count,
        category=category,
        model_path=model_path,
        model_sha256_value=model_sha,
        detector_backend=resolved_backend,
        detector_metadata=detector_metadata,
        threshold=threshold,
        primary_threshold=resolved_primary_threshold,
        stats=stats,
        elapsed_seconds=elapsed_seconds,
        detector_elapsed_seconds=detector_elapsed_seconds,
        width=width,
        height=height,
        nominal_frame_rate=frame_rate,
        detections_path=jsonl_path,
        clip_id=clip_id,
    )
    if write_summary_file:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def create_detector(
    category: str,
    model_path: Path,
    threshold: float,
    *,
    detector_backend: str | None = None,
    adas_bin_path: Path | None = None,
    primary_threshold: float | None = None,
) -> Detector:
    backend = resolve_detector_backend(category, detector_backend)
    if category == "face" and backend == DETECTOR_BACKEND_ADAS:
        return OpenVINOFaceADASDetector(
            model_path,
            model_bin_path=adas_bin_path,
            proposal_threshold=threshold,
            primary_threshold=resolve_primary_threshold(
                backend,
                proposal_threshold=threshold,
                primary_threshold=primary_threshold,
            ),
        )
    if category == "license_plate" and backend == DETECTOR_BACKEND_LICENSE_PLATE_YUNET:
        return LicensePlateYuNetDetector(model_path, threshold)
    if category == "face":
        raise ValueError(f"unsupported face detector backend: {backend}")
    if category == "license_plate":
        raise ValueError(f"unsupported license-plate detector backend: {backend}")
    raise ValueError(f"unsupported category: {category}")


def write_frame_record(
    jsonl_file,
    frame_index: int,
    detections: list[Detection],
    *,
    clip_id: str | None = None,
) -> None:
    serialized = [
        detection_record_with_frame_metadata(
            detection, frame_index, detection_index, clip_id=clip_id
        )
        for detection_index, detection in enumerate(detections)
    ]
    record = {
        **_optional_dict_field("clip", clip_id),
        "detections": serialized,
        "frame_index": frame_index,
    }
    jsonl_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def detection_record_with_frame_metadata(
    detection: Detection,
    frame_index: int,
    detection_index: int,
    *,
    clip_id: str | None,
) -> dict[str, object]:
    data = detection.to_dict()
    if clip_id is None:
        return data
    detector_id = data.get("detector_id")
    if detector_id:
        data.setdefault("source_detection_index", detection_index)
        data.setdefault(
            "candidate_id",
            f"{detector_id}:{clip_id}:f{frame_index:04d}:d{detection_index:02d}",
        )
    return data


def build_summary(
    *,
    created_at: str,
    input_video: Path,
    source_sha256: str,
    source_frame_count: str | None,
    category: str,
    model_path: Path,
    model_sha256_value: str,
    threshold: float,
    stats: DetectionStats,
    elapsed_seconds: float,
    width: int,
    height: int,
    nominal_frame_rate: str | None,
    detector_elapsed_seconds: float | None = None,
    detector_backend: str = "unknown",
    detector_metadata: dict[str, object] | None = None,
    primary_threshold: float | None = None,
    detections_path: Path | None = None,
    clip_id: str | None = None,
) -> dict[str, object]:
    processing_fps = stats.frames_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    detector_inference_fps = (
        stats.frames_processed / detector_elapsed_seconds
        if detector_elapsed_seconds is not None and detector_elapsed_seconds > 0
        else 0.0
    )
    resolved_detector_metadata = detector_metadata or {}
    return {
        "category": category,
        "clip": clip_id,
        "canonical_output": detections_path.name
        if detections_path is not None
        else "detections.jsonl",
        "created_at": created_at,
        "detector_backend": detector_backend,
        "diagnostic_overlay_has_audio": False,
        "detector_inference_fps": round(detector_inference_fps, 6),
        "detector_inference_seconds": (
            round(detector_elapsed_seconds, 6) if detector_elapsed_seconds is not None else None
        ),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "frames_containing_at_least_one_detection": stats.frames_with_detections,
        "frames_processed": stats.frames_processed,
        "input_height": height,
        "input_width": width,
        "max_detections_in_frame": stats.max_detections_in_frame,
        "min_detections_in_frame": stats.finalized_min_detections,
        "model_filename": model_path.name,
        "model_sha256": model_sha256_value,
        "nominal_frame_rate": nominal_frame_rate,
        "opencv_version": cv.__version__,
        "processing_fps": round(processing_fps, 6),
        "schema_version": SCHEMA_VERSION,
        "source_filename": input_video.name,
        "source_frame_count": _optional_int(source_frame_count),
        "source_sha256": source_sha256,
        "primary_threshold": primary_threshold,
        "proposal_threshold": threshold,
        "threshold": threshold,
        "total_detections": stats.total_detections,
        **resolved_detector_metadata,
    }


def draw_overlay(frame: np.ndarray, frame_index: int, detections: list[Detection]) -> np.ndarray:
    output = frame.copy()
    cv.putText(
        output,
        f"frame {frame_index}",
        (12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = (0, 255, 0) if detection.category == "face" else (0, 200, 255)
        cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            output,
            f"{detection.category} {detection.score:.2f}",
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv.LINE_AA,
        )
    return output


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists and is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


def model_sha256(model_path: Path) -> str:
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if is_git_lfs_pointer(model_path):
        raise ValueError(f"{model_path.name} is a Git LFS pointer, not a usable model file")
    return media.sha256_file(model_path)


def model_file_hashes(model_path: Path, *extra_paths: Path | None) -> dict[str, str]:
    paths = [model_path, *(path for path in extra_paths if path is not None)]
    return {
        path.name: model_sha256(path) for path in sorted(set(paths), key=lambda item: item.name)
    }


def is_git_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(256).startswith(GIT_LFS_POINTER_PREFIX)


def xywh_to_clipped_xyxy(
    x: float,
    y: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    bbox = (
        math.floor(x),
        math.floor(y),
        math.ceil(x + width),
        math.ceil(y + height),
    )
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def quadrilateral_to_clipped_xyxy(
    points: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    xs = points[0::2]
    ys = points[1::2]
    bbox = (
        math.floor(float(np.min(xs))),
        math.floor(float(np.min(ys))),
        math.ceil(float(np.max(xs))),
        math.ceil(float(np.max(ys))),
    )
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def clip_xyxy_to_frame(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    clipped = (
        min(max(x1, 0), frame_width),
        min(max(y1, 0), frame_height),
        min(max(x2, 0), frame_width),
        min(max(y2, 0), frame_height),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def preprocess_adas_frame(
    frame: np.ndarray,
    *,
    input_width: int,
    input_height: int,
) -> np.ndarray:
    resized = cv.resize(frame, (input_width, input_height), interpolation=cv.INTER_LINEAR)
    return resized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)


def parse_adas_detections(
    output: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    proposal_threshold: float,
    detector_id: str = DETECTOR_BACKEND_ADAS,
) -> list[Detection]:
    raw_detections = np.asarray(output).reshape(-1, 7)
    detections: list[Detection] = []
    for raw in raw_detections:
        image_id, label, confidence, x_min, y_min, x_max, y_max = raw.tolist()
        if image_id < 0:
            continue
        score = float(confidence)
        if score < proposal_threshold:
            continue
        bbox = normalized_to_adas_xyxy(x_min, y_min, x_max, y_max, frame_width, frame_height)
        if bbox is None:
            continue
        detections.append(
            Detection(
                "face",
                score,
                bbox,
                detector_id=detector_id,
                label=int(label),
            )
        )
    return sorted(detections, key=detection_sort_key)


def normalized_to_adas_xyxy(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    bbox = (
        round(float(x_min) * frame_width),
        round(float(y_min) * frame_height),
        round(float(x_max) * frame_width),
        round(float(y_max) * frame_height),
    )
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def classify_observation(
    score: float, *, proposal_threshold: float, primary_threshold: float
) -> str:
    validate_threshold_pair(proposal_threshold, primary_threshold)
    if score < proposal_threshold:
        return "below_proposal_floor"
    if score < primary_threshold:
        return "weak"
    return "primary"


def validate_threshold_pair(proposal_threshold: float, primary_threshold: float) -> None:
    if not 0 <= proposal_threshold <= 1:
        raise ValueError("proposal threshold must be between 0 and 1")
    if not 0 <= primary_threshold <= 1:
        raise ValueError("primary threshold must be between 0 and 1")
    if proposal_threshold > primary_threshold:
        raise ValueError("proposal threshold must be less than or equal to primary threshold")


def detection_sort_key(detection: Detection) -> tuple[float, tuple[int, int, int, int]]:
    return (-detection.score, detection.bbox_xyxy)


def resolve_detector_backend(category: str, detector_backend: str | None) -> str:
    if category == "face":
        if detector_backend not in {None, DETECTOR_BACKEND_ADAS}:
            raise ValueError(
                "unsupported face detector backend: "
                f"{detector_backend}; active face detector is {DETECTOR_BACKEND_ADAS}"
            )
        return DEFAULT_FACE_DETECTOR_BACKEND
    if category == "license_plate":
        if detector_backend not in {None, DETECTOR_BACKEND_LICENSE_PLATE_YUNET}:
            raise ValueError(f"unsupported license-plate detector backend: {detector_backend}")
        return DETECTOR_BACKEND_LICENSE_PLATE_YUNET
    raise ValueError(f"unsupported category: {category}")


def default_proposal_threshold(category: str, detector_backend: str | None = None) -> float:
    backend = resolve_detector_backend(category, detector_backend)
    if backend == DETECTOR_BACKEND_ADAS:
        return DEFAULT_ADAS_PROPOSAL_THRESHOLD
    return DEFAULT_THRESHOLDS[category]


def resolve_primary_threshold(
    detector_backend: str,
    *,
    proposal_threshold: float,
    primary_threshold: float | None,
) -> float | None:
    if detector_backend == DETECTOR_BACKEND_ADAS:
        resolved = (
            DEFAULT_ADAS_PRIMARY_THRESHOLD if primary_threshold is None else primary_threshold
        )
        validate_threshold_pair(proposal_threshold, resolved)
        return resolved
    if primary_threshold is not None:
        validate_threshold_pair(proposal_threshold, primary_threshold)
    return primary_threshold


def default_adas_xml_path() -> Path:
    if env_path := os.environ.get("REDACTOR_ADAS_XML"):
        return Path(env_path)
    return DEFAULT_ADAS_XML_PATH


def default_adas_bin_path(xml_path: Path) -> Path:
    if env_path := os.environ.get("REDACTOR_ADAS_BIN"):
        return Path(env_path)
    return xml_path.with_suffix(".bin")


def detector_summary_metadata(detector: Detector) -> dict[str, object]:
    summary = getattr(detector, "summary_metadata", None)
    if callable(summary):
        return summary()
    detector_id = getattr(detector, "detector_id", None)
    return _optional_dict_field("detector_id", detector_id)


def _optional_dict_field(key: str, value: Any) -> dict[str, Any]:
    return {} if value is None else {key: value}


def normalize_nms_indices(indices) -> list[int]:
    if indices is None or len(indices) == 0:
        return []
    return [int(index) for index in np.array(indices).reshape(-1)]


def rational_to_float(value: str | None) -> float | None:
    if value is None:
        return None
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    denominator_float = float(denominator)
    if denominator_float == 0:
        return None
    return float(numerator) / denominator_float


def _quadrilateral_to_xywh(points: np.ndarray) -> list[float]:
    xs = points[0::2]
    ys = points[1::2]
    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))
    return [x1, y1, x2 - x1, y2 - y1]


def _squeeze_output(output: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(output)
    if squeezed.ndim == 1:
        return squeezed[np.newaxis, :]
    return squeezed


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
