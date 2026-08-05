from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor.detection import clip_xyxy_to_frame, detection_sort_key, model_sha256
from redactor.models import Detection

DETECTOR_ID = "oim_yolov9_s_608"
DETECTOR_DISPLAY_NAME = "Open Image Models YOLOv9-S-608 end-to-end license-plate detector"
MODEL_SOURCE_REPOSITORY = "https://github.com/ankandrew/open-image-models"
MODEL_REVISION = "f22000e02b30642f317cdba7755c0631638b109e"
LICENSE_STATUS = (
    "MIT repository license; no separate weight-specific notice was found in the "
    "audited local model artifacts."
)
MODEL_FILENAME = "yolo-v9-s-608-license-plates-end2end.onnx"
MODEL_ENV_VAR = "REDACTOR_OIM_PLATE_MODEL"

INFERENCE_MODE = "uniform_small_object_tiles"
MODEL_INPUT_SIZE = 608
TILE_SIZE = 1280
TILE_OVERLAP = 0.25
INCLUDE_FULL_FRAME = True
NMS_IOU_THRESHOLD = 0.45
DEFAULT_PRIMARY_THRESHOLD = 0.30
DEFAULT_PROPOSAL_THRESHOLD = DEFAULT_PRIMARY_THRESHOLD
CATEGORY = "license_plate"


@dataclass(frozen=True)
class LetterboxMapping:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int
    scale: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "input_height": self.input_height,
            "input_width": self.input_width,
            "original_height": self.original_height,
            "original_width": self.original_width,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class OIMPlateDetectorConfig:
    detector_id: str = DETECTOR_ID
    model_revision: str = MODEL_REVISION
    inference_mode: str = INFERENCE_MODE
    input_size: int = MODEL_INPUT_SIZE
    proposal_threshold: float = DEFAULT_PROPOSAL_THRESHOLD
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD
    tile_size: int = TILE_SIZE
    tile_overlap: float = TILE_OVERLAP
    include_full_frame: bool = INCLUDE_FULL_FRAME
    nms_iou_threshold: float = NMS_IOU_THRESHOLD
    source_repository: str = MODEL_SOURCE_REPOSITORY
    license_status: str = LICENSE_STATUS

    def __post_init__(self) -> None:
        if self.detector_id != DETECTOR_ID:
            raise ValueError(f"active plate detector id must be {DETECTOR_ID}")
        if self.inference_mode != INFERENCE_MODE:
            raise ValueError(f"active plate inference mode must be {INFERENCE_MODE}")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1)")
        if not 0 <= self.proposal_threshold <= 1:
            raise ValueError("proposal_threshold must be between 0 and 1")
        if not 0 <= self.primary_threshold <= 1:
            raise ValueError("primary_threshold must be between 0 and 1")
        if self.proposal_threshold > self.primary_threshold:
            raise ValueError("proposal_threshold cannot exceed primary_threshold")
        if self.proposal_threshold != self.primary_threshold:
            raise ValueError(
                "OIM YOLOv9-S-608 v1 uses one selected threshold; "
                "no lower proposal threshold is configured"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "include_full_frame": self.include_full_frame,
            "inference_mode": self.inference_mode,
            "input_size": self.input_size,
            "license_status": self.license_status,
            "model_revision": self.model_revision,
            "nms_iou_threshold": self.nms_iou_threshold,
            "primary_threshold": self.primary_threshold,
            "proposal_threshold": self.proposal_threshold,
            "source_repository": self.source_repository,
            "tile_overlap": self.tile_overlap,
            "tile_size": self.tile_size,
        }

    def configuration_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedPlateDetection:
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    label: int
    source_detection_index: int
    geometry: dict[str, Any]
    preprocessing: dict[str, Any]
    postprocessing: dict[str, Any]
    inference_time_ms: float


class OpenImageModelsYOLOv9S608PlateDetector:
    detector_id = DETECTOR_ID
    category = CATEGORY

    def __init__(
        self,
        model_path: Path,
        *,
        config: OIMPlateDetectorConfig | None = None,
        device: str = "CPU",
    ):
        self.config = config or OIMPlateDetectorConfig()
        self.model_path = model_path
        self.device = device
        self.model_files_sha256 = {model_path.name: model_sha256(model_path)}

        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO is required for the active OIM YOLOv9-S-608 plate detector; "
                "install the openvino Python package and provide the ONNX model file"
            ) from exc

        self.openvino_version = ov.__version__
        try:
            self.core = ov.Core()
            model = self.core.read_model(str(model_path))
            self.compiled_model = self.core.compile_model(model, device)
            self.input_port = self.compiled_model.input(0)
            self.output_port = self.compiled_model.output(0)
        except Exception as exc:
            raise ValueError(f"failed to load active OIM plate model {model_path.name}") from exc

        self.input_shape = [int(dim) for dim in self.input_port.shape]
        if (
            len(self.input_shape) != 4
            or self.input_shape[1] != 3
            or self.input_shape[2] != self.input_shape[3]
        ):
            raise ValueError(f"expected square NCHW YOLOv9-S input, got {self.input_shape}")
        if self.input_shape[2] != self.config.input_size:
            raise ValueError(
                "active OIM plate model input size does not match selected configuration: "
                f"{self.input_shape[2]} != {self.config.input_size}"
            )
        self.input_name = self.input_port.get_any_name()
        self.output_name = self.output_port.get_any_name()
        self.output_shape = openvino_shape_to_list(self.output_port.partial_shape)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        parsed = detect_oim_yolov9_tiled(
            frame,
            compiled_model=self.compiled_model,
            input_port=self.input_port,
            output_port=self.output_port,
            config=self.config,
        )
        records: list[Detection] = []
        for index, detection in enumerate(parsed):
            records.append(
                Detection(
                    CATEGORY,
                    detection.score,
                    detection.bbox_xyxy,
                    detector_id=self.detector_id,
                    label=detection.label,
                    source_detection_index=index,
                    metadata={
                        "configuration_hash": self.config.configuration_hash(),
                        "detector_display_name": DETECTOR_DISPLAY_NAME,
                        "geometry": detection.geometry,
                        "inference_mode": self.config.inference_mode,
                        "inference_time_ms": round(detection.inference_time_ms, 6),
                        "model_revision": self.config.model_revision,
                        "observation_status": classify_plate_observation(
                            detection.score,
                            proposal_threshold=self.config.proposal_threshold,
                            primary_threshold=self.config.primary_threshold,
                        ),
                        "postprocessing": detection.postprocessing,
                        "preprocessing": detection.preprocessing,
                        "source_repository": self.config.source_repository,
                    },
                )
            )
        return sorted(records, key=detection_sort_key)

    def summary_metadata(self) -> dict[str, object]:
        return {
            "configuration_hash": self.config.configuration_hash(),
            "detector_display_name": DETECTOR_DISPLAY_NAME,
            "detector_id": self.detector_id,
            "expected_color_order": "BGR",
            "inference_mode": self.config.inference_mode,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "license_status": self.config.license_status,
            "model_files_sha256": self.model_files_sha256,
            "model_revision": self.config.model_revision,
            "model_source": self.config.source_repository,
            "nms_iou_threshold": self.config.nms_iou_threshold,
            "openvino_version": self.openvino_version,
            "output_format": "end-to-end YOLOv9 detections [batch,x1,y1,x2,y2,class,score]",
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "primary_threshold": self.config.primary_threshold,
            "proposal_threshold": self.config.proposal_threshold,
            "runtime": f"OpenVINO {self.device}",
            "tile_overlap": self.config.tile_overlap,
            "tile_size": self.config.tile_size,
        }


def default_config(
    *,
    proposal_threshold: float = DEFAULT_PROPOSAL_THRESHOLD,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
) -> OIMPlateDetectorConfig:
    return OIMPlateDetectorConfig(
        proposal_threshold=proposal_threshold,
        primary_threshold=primary_threshold,
    )


def default_model_path() -> Path:
    if env_path := os.environ.get(MODEL_ENV_VAR):
        return Path(env_path)
    return Path("models/open-image-models") / MODEL_FILENAME


def classify_plate_observation(
    score: float,
    *,
    proposal_threshold: float,
    primary_threshold: float,
) -> str:
    if score < proposal_threshold:
        return "below_proposal_threshold"
    if score < primary_threshold:
        return "proposal"
    return "primary"


def detect_oim_yolov9_tiled(
    frame: np.ndarray,
    *,
    compiled_model: Any,
    input_port: Any,
    output_port: Any,
    config: OIMPlateDetectorConfig,
) -> list[ParsedPlateDetection]:
    height, width = frame.shape[:2]
    tiles = build_uniform_tiles(
        width,
        height,
        tile_size=config.tile_size,
        overlap_fraction=config.tile_overlap,
        include_full_frame=config.include_full_frame,
    )
    pre_nms: list[ParsedPlateDetection] = []
    for tile_index, (x1, y1, x2, y2) in enumerate(tiles):
        tile = frame[y1:y2, x1:x2]
        started = time.perf_counter()
        tensor, mapping = preprocess_letterbox_rgb_01(tile, config.input_size)
        output = compiled_model({input_port: tensor})[output_port]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        tile_detections = parse_oim_yolov9_output(
            output,
            frame_width=tile.shape[1],
            frame_height=tile.shape[0],
            mapping=mapping,
            score_floor=config.proposal_threshold,
            inference_time_ms=elapsed_ms,
        )
        for detection in tile_detections:
            mapped_bbox = clip_xyxy_to_frame(
                (
                    detection.bbox_xyxy[0] + x1,
                    detection.bbox_xyxy[1] + y1,
                    detection.bbox_xyxy[2] + x1,
                    detection.bbox_xyxy[3] + y1,
                ),
                width,
                height,
            )
            if mapped_bbox is None:
                continue
            pre_nms.append(
                ParsedPlateDetection(
                    score=detection.score,
                    bbox_xyxy=mapped_bbox,
                    label=detection.label,
                    source_detection_index=detection.source_detection_index,
                    inference_time_ms=detection.inference_time_ms,
                    geometry={
                        **detection.geometry,
                        "derived_redaction_bbox_xyxy": list(mapped_bbox),
                        "source_tile_bbox_xyxy": list(detection.bbox_xyxy),
                    },
                    preprocessing={
                        **detection.preprocessing,
                        "tile_index": tile_index,
                        "tile_xyxy": [x1, y1, x2, y2],
                        "tiled_inference": True,
                    },
                    postprocessing={
                        **detection.postprocessing,
                        "nms_stage": "pre_cross_tile_nms",
                    },
                )
            )
    final = []
    for detection in non_max_suppression(pre_nms, iou_threshold=config.nms_iou_threshold):
        final.append(
            ParsedPlateDetection(
                score=detection.score,
                bbox_xyxy=detection.bbox_xyxy,
                label=detection.label,
                source_detection_index=detection.source_detection_index,
                inference_time_ms=detection.inference_time_ms,
                geometry=detection.geometry,
                preprocessing=detection.preprocessing,
                postprocessing={
                    **detection.postprocessing,
                    "cross_tile_nms_iou_threshold": config.nms_iou_threshold,
                    "inference_mode": config.inference_mode,
                    "nms_stage": "post_cross_tile_nms",
                    "pre_nms_candidate_count": len(pre_nms),
                    "tile_count": len(tiles),
                },
            )
        )
    return sort_plate_detections(final)


def preprocess_letterbox_rgb_01(
    frame: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, LetterboxMapping]:
    height, width = frame.shape[:2]
    scale = min(input_size / height, input_size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    left = int(round((input_size - resized_width) / 2.0 - 0.1))
    top = int(round((input_size - resized_height) / 2.0 - 0.1))
    resized = cv.resize(frame, (resized_width, resized_height), interpolation=cv.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = canvas.transpose(2, 0, 1)[::-1].astype(np.float32)[np.newaxis, :] / 255.0
    mapping = LetterboxMapping(
        original_width=width,
        original_height=height,
        input_width=input_size,
        input_height=input_size,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_x=left,
        pad_y=top,
        scale=scale,
    )
    return tensor, mapping


def parse_oim_yolov9_output(
    output: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    mapping: LetterboxMapping,
    score_floor: float,
    inference_time_ms: float,
) -> list[ParsedPlateDetection]:
    raw = np.asarray(output)
    if raw.size == 0:
        return []
    predictions = raw.reshape(-1, raw.shape[-1])
    if predictions.shape[1] < 7:
        raise ValueError(f"expected YOLOv9 output with at least 7 columns, got {raw.shape}")
    detections: list[ParsedPlateDetection] = []
    for index, row in enumerate(predictions):
        label = int(row[5])
        score = float(row[6])
        if label != 0 or score < score_floor:
            continue
        bbox = yolo_letterboxed_xyxy_to_original(
            row[1],
            row[2],
            row[3],
            row[4],
            frame_width,
            frame_height,
            mapping,
        )
        if bbox is None:
            continue
        detections.append(
            ParsedPlateDetection(
                score=score,
                bbox_xyxy=bbox,
                label=label,
                source_detection_index=index,
                inference_time_ms=inference_time_ms,
                geometry={
                    "derived_redaction_bbox_xyxy": list(bbox),
                    "model_geometry_type": "axis_aligned_xyxy",
                    "model_xyxy_letterboxed": [round(float(value), 6) for value in row[1:5]],
                    "oriented_bbox": None,
                },
                preprocessing={
                    "color_order": "BGR_to_RGB",
                    "layout": "NCHW",
                    "letterbox": mapping.to_dict(),
                    "pixel_scale": "0..1",
                },
                postprocessing={
                    "class_filter": 0,
                    "end_to_end_nms_model": True,
                    "model_output_shape": list(raw.shape),
                    "nms_stage": "post_model_nms",
                    "score_floor": score_floor,
                },
            )
        )
    return sort_plate_detections(detections)


def yolo_letterboxed_xyxy_to_original(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    frame_width: int,
    frame_height: int,
    mapping: LetterboxMapping,
) -> tuple[int, int, int, int] | None:
    x1 = (float(x_min) - mapping.pad_x) / mapping.scale
    y1 = (float(y_min) - mapping.pad_y) / mapping.scale
    x2 = (float(x_max) - mapping.pad_x) / mapping.scale
    y2 = (float(y_max) - mapping.pad_y) / mapping.scale
    bbox = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def build_uniform_tiles(
    frame_width: int,
    frame_height: int,
    *,
    tile_size: int,
    overlap_fraction: float,
    include_full_frame: bool = True,
) -> list[tuple[int, int, int, int]]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    stride = max(1, int(round(tile_size * (1 - overlap_fraction))))
    max_x = max(0, frame_width - tile_size)
    max_y = max(0, frame_height - tile_size)
    x_starts = list(range(0, max_x + 1, stride)) or [0]
    y_starts = list(range(0, max_y + 1, stride)) or [0]
    if x_starts[-1] != max_x:
        x_starts.append(max_x)
    if y_starts[-1] != max_y:
        y_starts.append(max_y)
    tiles: list[tuple[int, int, int, int]] = []
    if include_full_frame:
        tiles.append((0, 0, frame_width, frame_height))
    for y_start in y_starts:
        for x_start in x_starts:
            tiles.append(
                (
                    x_start,
                    y_start,
                    min(frame_width, x_start + tile_size),
                    min(frame_height, y_start + tile_size),
                )
            )
    return sorted(set(tiles))


def non_max_suppression(
    detections: list[ParsedPlateDetection],
    *,
    iou_threshold: float,
) -> list[ParsedPlateDetection]:
    if not detections:
        return []
    sorted_detections = sort_plate_detections(detections)
    keep: list[ParsedPlateDetection] = []
    for detection in sorted_detections:
        if all(bbox_iou(detection.bbox_xyxy, kept.bbox_xyxy) <= iou_threshold for kept in keep):
            keep.append(detection)
    return keep


def sort_plate_detections(detections: list[ParsedPlateDetection]) -> list[ParsedPlateDetection]:
    return sorted(detections, key=lambda item: (-item.score, item.bbox_xyxy))


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = float((ix2 - ix1) * (iy2 - iy1))
    left_area = float((lx2 - lx1) * (ly2 - ly1))
    right_area = float((rx2 - rx1) * (ry2 - ry1))
    return intersection / (left_area + right_area - intersection)


def model_source_metadata(model_path: Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "detector_display_name": DETECTOR_DISPLAY_NAME,
        "detector_id": DETECTOR_ID,
        "inference_mode": INFERENCE_MODE,
        "license_status": LICENSE_STATUS,
        "model_revision": MODEL_REVISION,
        "primary_threshold": DEFAULT_PRIMARY_THRESHOLD,
        "proposal_threshold": DEFAULT_PROPOSAL_THRESHOLD,
        "source_repository": MODEL_SOURCE_REPOSITORY,
    }
    if model_path is not None:
        metadata["model_files_sha256"] = {model_path.name: model_sha256(model_path)}
    metadata["configuration_hash"] = OIMPlateDetectorConfig().configuration_hash()
    return metadata


def openvino_shape_to_list(shape: Any) -> list[int | str]:
    values: list[int | str] = []
    for dimension in shape:
        is_static_attr = getattr(dimension, "is_static", False)
        is_static = bool(is_static_attr() if callable(is_static_attr) else is_static_attr)
        if is_static:
            values.append(int(dimension.get_length()))
            continue
        values.append(str(dimension))
    return values
