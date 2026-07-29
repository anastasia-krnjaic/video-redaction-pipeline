from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np

from redactor import media
from redactor.models import Detection

SCHEMA_VERSION = "1.0"
DEFAULT_THRESHOLDS = {
    "face": 0.60,
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


class FaceYuNetDetector:
    def __init__(self, model_path: Path, threshold: float, nms_threshold: float = NMS_THRESHOLD):
        try:
            self.detector = cv.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),
                threshold,
                nms_threshold,
                5000,
            )
        except cv.error as exc:
            raise ValueError(f"failed to load face model {model_path.name}") from exc
        self.threshold = threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        detections: list[Detection] = []
        for face in faces:
            score = float(face[-1])
            bbox = xywh_to_clipped_xyxy(
                float(face[0]),
                float(face[1]),
                float(face[2]),
                float(face[3]),
                width,
                height,
            )
            if bbox is not None:
                detections.append(Detection("face", score, bbox))
        return detections


class LicensePlateYuNetDetector:
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
                detections.append(Detection("license_plate", score, bbox))
        return detections

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
) -> dict[str, object]:
    if category not in DEFAULT_THRESHOLDS:
        raise ValueError(f"unsupported category: {category}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max-frames must be greater than 0")

    ensure_output_dir(output_dir, overwrite=overwrite)
    model_sha = model_sha256(model_path)
    detector = detector or create_detector(category, model_path, threshold)

    source_info = media.probe_media(input_video)
    source_sha = source_info.sha256
    width = source_info.video_stream.width
    height = source_info.video_stream.height
    frame_rate = source_info.video_stream.nominal_frame_rate
    fps = rational_to_float(frame_rate) or 30.0

    capture = cv.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_video}")

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
    jsonl_path = output_dir / "detections.jsonl"
    started_at = time.perf_counter()
    created_at = media.utc_timestamp()
    frame_index = 0

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = detector.detect(frame)
                write_frame_record(jsonl_file, frame_index, detections)
                writer.write(draw_overlay(frame, frame_index, detections))
                stats.update(len(detections))
                frame_index += 1
    finally:
        capture.release()
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
        threshold=threshold,
        stats=stats,
        elapsed_seconds=elapsed_seconds,
        width=width,
        height=height,
        nominal_frame_rate=frame_rate,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def create_detector(category: str, model_path: Path, threshold: float) -> Detector:
    if category == "face":
        return FaceYuNetDetector(model_path, threshold)
    if category == "license_plate":
        return LicensePlateYuNetDetector(model_path, threshold)
    raise ValueError(f"unsupported category: {category}")


def write_frame_record(jsonl_file, frame_index: int, detections: list[Detection]) -> None:
    record = {
        "detections": [detection.to_dict() for detection in detections],
        "frame_index": frame_index,
    }
    jsonl_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


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
) -> dict[str, object]:
    processing_fps = stats.frames_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "category": category,
        "created_at": created_at,
        "diagnostic_overlay_has_audio": False,
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
        "threshold": threshold,
        "total_detections": stats.total_detections,
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
        raise ValueError(f"{model_path.name} is a Git LFS pointer, not an ONNX model binary")
    return media.sha256_file(model_path)


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
