from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2 as cv
import numpy as np

from experiments.archive.license_plate_detector_bakeoff import plate_bakeoff
from redactor import media
from redactor.detection import clip_xyxy_to_frame

SCHEMA_VERSION = "license_plate_detector_bakeoff_v2"
THRESHOLD_GRID: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
SCORE_FLOOR = min(THRESHOLD_GRID)
NMS_IOU_THRESHOLD = 0.45
TILE_SIZE = 1280
TILE_OVERLAP = 0.25


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    role: str
    video_path: Path
    source_sha256: str
    evaluated_ranges: tuple[tuple[int, int], ...]
    scene_description: str
    target_id: str | None = None
    target_roi_xyxy: tuple[int, int, int, int] | None = None
    certified_quantitative: bool = False
    notes: str = ""

    def evaluated_frames(self) -> list[int]:
        frames: list[int] = []
        for start, end in self.evaluated_ranges:
            if end < start:
                raise ValueError("evaluated range end must be >= start")
            frames.extend(range(start, end + 1))
        return frames

    def to_plate_benchmark(self) -> plate_bakeoff.PlateBenchmark:
        return plate_bakeoff.PlateBenchmark(
            clip_id=self.clip_id,
            video_path=self.video_path,
            source_sha256=self.source_sha256,
            evaluated_ranges=self.evaluated_ranges,
            target_id=self.target_id,
            target_roi_xyxy=self.target_roi_xyxy,
            certified_quantitative=self.certified_quantitative,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, Any]:
        info = media.probe_media(self.video_path)
        return {
            "certified_quantitative": self.certified_quantitative,
            "clip_id": self.clip_id,
            "evaluated_frame_count": len(self.evaluated_frames()),
            "evaluated_ranges": [list(item) for item in self.evaluated_ranges],
            "fps": info.video_stream.nominal_frame_rate,
            "frame_count": int(info.video_stream.frame_count or 0),
            "height": info.video_stream.height,
            "notes": self.notes,
            "role": self.role,
            "scene_description": self.scene_description,
            "source_sha256": info.sha256,
            "target_id": self.target_id,
            "target_roi_xyxy": list(self.target_roi_xyxy) if self.target_roi_xyxy else None,
            "video_path": plate_bakeoff.portable_path(self.video_path),
            "width": info.video_stream.width,
        }


@dataclass(frozen=True)
class CandidateSpec:
    detector_id: str
    display_name: str
    adapter_type: str
    model_path: Path
    source_repo: str
    source_revision: str
    source_license: str
    model_license: str
    architecture: str
    native_input: tuple[int, int]
    preprocessing: str
    output_format: str
    postprocessing: str
    runtime: str
    notes: str = ""
    local_source_path: Path | None = None
    model_files: tuple[Path, ...] = field(default_factory=tuple)

    def model_artifacts(self) -> tuple[Path, ...]:
        return self.model_files or (self.model_path,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "architecture": self.architecture,
            "detector_id": self.detector_id,
            "display_name": self.display_name,
            "local_source_path": (
                plate_bakeoff.portable_path(self.local_source_path)
                if self.local_source_path
                else None
            ),
            "model_files": [
                {
                    "path": plate_bakeoff.portable_path(path),
                    "sha256": media.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in self.model_artifacts()
            ],
            "model_license": self.model_license,
            "native_input_resolution": list(self.native_input),
            "notes": self.notes,
            "output_format": self.output_format,
            "postprocessing": self.postprocessing,
            "preprocessing": self.preprocessing,
            "runtime": self.runtime,
            "source_license": self.source_license,
            "source_repository": self.source_repo,
            "source_revision": self.source_revision,
            "threshold_grid": list(THRESHOLD_GRID),
        }


@dataclass(frozen=True)
class InferenceMode:
    mode_id: str
    display_name: str
    tiled: bool = False
    tile_size: int = TILE_SIZE
    tile_overlap: float = TILE_OVERLAP
    include_full_frame: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "include_full_frame": self.include_full_frame,
            "mode_id": self.mode_id,
            "tile_overlap": self.tile_overlap,
            "tile_size": self.tile_size,
            "tiled": self.tiled,
        }


class V2Adapter(Protocol):
    detector_id: str

    def detect_with_diagnostics(self, frame: np.ndarray) -> plate_bakeoff.PlateDetectionRun: ...

    def metadata(self) -> dict[str, Any]: ...


class OpenImageModelsYoloV9Adapter:
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        score_floor: float = SCORE_FLOOR,
        device: str = "CPU",
    ):
        import openvino as ov

        if not model_path.is_file():
            raise FileNotFoundError(f"Open Image Models YOLOv9 ONNX not found: {model_path}")
        self.detector_id = detector_id
        self.model_path = model_path
        self.score_floor = score_floor
        self.device = device
        self.core = ov.Core()
        model = self.core.read_model(str(model_path))
        self.compiled_model = self.core.compile_model(model, device)
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)
        shape = [int(dim) for dim in self.input_port.shape]
        if len(shape) != 4 or shape[1] != 3 or shape[2] != shape[3]:
            raise ValueError(f"expected square NCHW YOLOv9 input, got {shape}")
        self.input_size = shape[2]

    def detect_with_diagnostics(self, frame: np.ndarray) -> plate_bakeoff.PlateDetectionRun:
        started = time.perf_counter()
        tensor, mapping = preprocess_letterbox_rgb_01(frame, self.input_size)
        output = self.compiled_model({self.input_port: tensor})[self.output_port]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return parse_oim_yolov9_output(
            output,
            detector_id=self.detector_id,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            mapping=mapping,
            score_floor=self.score_floor,
            inference_time_ms=elapsed_ms,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "input_shape": [1, 3, self.input_size, self.input_size],
            "model_files_sha256": {self.model_path.name: media.sha256_file(self.model_path)},
            "output_format": "end-to-end YOLOv9 detections [batch,x1,y1,x2,y2,class,score]",
            "runtime": f"OpenVINO {self.device}",
            "score_floor": self.score_floor,
        }


class RFDETRALPRAdapter:
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        score_floor: float = SCORE_FLOOR,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
        device: str = "CPU",
    ):
        import openvino as ov

        if not model_path.is_file():
            raise FileNotFoundError(f"RF-DETR ALPR ONNX not found: {model_path}")
        self.detector_id = detector_id
        self.model_path = model_path
        self.score_floor = score_floor
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.core = ov.Core()
        model = self.core.read_model(str(model_path))
        self.compiled_model = self.core.compile_model(model, device)
        self.input_port = self.compiled_model.input(0)
        shape = [openvino_dimension_to_int(dim) for dim in self.input_port.partial_shape]
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"expected NCHW RF-DETR input, got {shape}")
        self.input_height = shape[2]
        self.input_width = shape[3]
        self.logits_port, self.boxes_port = resolve_rfdetr_output_ports(self.compiled_model)

    def detect_with_diagnostics(self, frame: np.ndarray) -> plate_bakeoff.PlateDetectionRun:
        started = time.perf_counter()
        tensor = preprocess_rfdetr(frame, self.input_height, self.input_width)
        outputs = self.compiled_model({self.input_port: tensor})
        logits = np.asarray(outputs[self.logits_port])
        boxes = np.asarray(outputs[self.boxes_port])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return parse_rfdetr_outputs(
            boxes,
            logits,
            detector_id=self.detector_id,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            score_floor=self.score_floor,
            nms_iou_threshold=self.nms_iou_threshold,
            inference_time_ms=elapsed_ms,
            output_name_note=(
                "Local ONNX output names are misleading: 'boxes' is logits [Q,1], "
                "and 'scores' is normalized boxes [Q,4]."
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "input_shape": [1, 3, self.input_height, self.input_width],
            "model_files_sha256": {self.model_path.name: media.sha256_file(self.model_path)},
            "nms_iou_threshold": self.nms_iou_threshold,
            "output_format": "RF-DETR normalized cxcywh boxes plus raw class logits",
            "runtime": f"OpenVINO {self.device}",
            "score_floor": self.score_floor,
        }


class RTDetrV2Adapter:
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        score_floor: float = SCORE_FLOOR,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
        device: str = "auto",
    ):
        if not model_path.is_dir():
            raise FileNotFoundError(f"RT-DETR v2 model directory not found: {model_path}")
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.detector_id = detector_id
        self.model_path = model_path
        self.score_floor = score_floor
        self.nms_iou_threshold = nms_iou_threshold
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(device)
        self.model.eval()

    def detect_with_diagnostics(self, frame: np.ndarray) -> plate_bakeoff.PlateDetectionRun:
        from PIL import Image

        started = time.perf_counter()
        image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        if self.device == "mps":
            self.torch.mps.synchronize()
        post = self.processor.post_process_object_detection(
            outputs,
            target_sizes=self.torch.tensor([image.size[::-1]]),
            threshold=self.score_floor,
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pre_nms: list[plate_bakeoff.PlateDetection] = []
        boxes = post["boxes"].detach().cpu().numpy()
        scores = post["scores"].detach().cpu().numpy()
        labels = post["labels"].detach().cpu().numpy()
        for index, (box, score, label) in enumerate(zip(boxes, scores, labels, strict=True)):
            bbox = clip_xyxy_to_frame(
                tuple(round(float(value)) for value in box.tolist()),
                frame.shape[1],
                frame.shape[0],
            )
            if bbox is None:
                continue
            pre_nms.append(
                plate_bakeoff.PlateDetection(
                    detector_id=self.detector_id,
                    frame_index=-1,
                    score=float(score),
                    bbox_xyxy=bbox,
                    inference_time_ms=elapsed_ms,
                    label=int(label),
                    source_detection_index=index,
                    geometry={
                        "derived_redaction_bbox_xyxy": list(bbox),
                        "model_geometry_type": "axis_aligned_xyxy",
                        "oriented_bbox": None,
                    },
                    preprocessing={
                        "official_transformers_image_processor": True,
                        "device": self.device,
                    },
                    postprocessing={
                        "class_filter": "license_plate",
                        "nms_stage": "pre_nms",
                        "score_floor": self.score_floor,
                    },
                )
            )
        final = [
            item.with_metadata(
                postprocessing={
                    **item.postprocessing,
                    "nms_iou_threshold": self.nms_iou_threshold,
                    "nms_stage": "post_nms",
                    "pre_nms_candidate_count": len(pre_nms),
                }
            )
            for item in plate_bakeoff.non_max_suppression(
                pre_nms,
                iou_threshold=self.nms_iou_threshold,
            )
        ]
        return plate_bakeoff.PlateDetectionRun(
            final_detections=plate_bakeoff.sort_plate_detections(final),
            pre_nms_detections=plate_bakeoff.sort_plate_detections(pre_nms),
            diagnostics={
                "device": self.device,
                "final_detection_count": len(final),
                "nms_iou_threshold": self.nms_iou_threshold,
                "raw_candidate_count": len(pre_nms),
                "score_floor": self.score_floor,
            },
        )

    def metadata(self) -> dict[str, Any]:
        model_files = sorted(
            path
            for path in self.model_path.iterdir()
            if path.name in {"config.json", "preprocessor_config.json", "model.safetensors"}
        )
        return {
            "detector_id": self.detector_id,
            "model_files_sha256": {path.name: media.sha256_file(path) for path in model_files},
            "nms_iou_threshold": self.nms_iou_threshold,
            "output_format": "Transformers RT-DETR post_process_object_detection boxes",
            "runtime": f"PyTorch {self.device}",
            "score_floor": self.score_floor,
        }


class TiledAdapter:
    def __init__(
        self,
        base: V2Adapter,
        *,
        tile_size: int = TILE_SIZE,
        tile_overlap: float = TILE_OVERLAP,
        include_full_frame: bool = True,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    ):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1)")
        self.base = base
        self.detector_id = base.detector_id
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.include_full_frame = include_full_frame
        self.nms_iou_threshold = nms_iou_threshold

    def detect_with_diagnostics(self, frame: np.ndarray) -> plate_bakeoff.PlateDetectionRun:
        height, width = frame.shape[:2]
        tiles = plate_bakeoff.build_uniform_tiles(
            width,
            height,
            tile_size=self.tile_size,
            overlap_fraction=self.tile_overlap,
            include_full_frame=self.include_full_frame,
        )
        started = time.perf_counter()
        pre_nms: list[plate_bakeoff.PlateDetection] = []
        tile_counts: list[dict[str, Any]] = []
        for tile_index, (x1, y1, x2, y2) in enumerate(tiles):
            run = self.base.detect_with_diagnostics(frame[y1:y2, x1:x2])
            count = 0
            for detection in run.final_detections:
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
                    plate_bakeoff.PlateDetection(
                        detector_id=self.detector_id,
                        frame_index=-1,
                        score=detection.score,
                        bbox_xyxy=mapped_bbox,
                        inference_time_ms=detection.inference_time_ms,
                        label=detection.label,
                        source_detection_index=detection.source_detection_index,
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
                count += 1
            tile_counts.append({"detections": count, "tile_index": tile_index})
        final = [
            item.with_metadata(
                postprocessing={
                    **item.postprocessing,
                    "cross_tile_nms_iou_threshold": self.nms_iou_threshold,
                    "nms_stage": "post_cross_tile_nms",
                    "pre_nms_candidate_count": len(pre_nms),
                    "tile_count": len(tiles),
                }
            )
            for item in plate_bakeoff.non_max_suppression(
                pre_nms,
                iou_threshold=self.nms_iou_threshold,
            )
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return plate_bakeoff.PlateDetectionRun(
            final_detections=plate_bakeoff.sort_plate_detections(final),
            pre_nms_detections=plate_bakeoff.sort_plate_detections(pre_nms),
            diagnostics={
                "cross_tile_nms_iou_threshold": self.nms_iou_threshold,
                "elapsed_ms": round(elapsed_ms, 6),
                "final_detection_count": len(final),
                "include_full_frame": self.include_full_frame,
                "raw_candidate_count": len(pre_nms),
                "tile_detection_counts": tile_counts,
                "tile_overlap": self.tile_overlap,
                "tile_size": self.tile_size,
                "tiles": [list(tile) for tile in tiles],
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **self.base.metadata(),
            "cross_tile_nms_iou_threshold": self.nms_iou_threshold,
            "include_full_frame": self.include_full_frame,
            "small_object_mode": "whole_frame_plus_uniform_tiles",
            "tile_overlap": self.tile_overlap,
            "tile_size": self.tile_size,
        }


def create_adapter(
    candidate: CandidateSpec,
    *,
    mode: InferenceMode | None = None,
    score_floor: float = SCORE_FLOOR,
) -> V2Adapter:
    if candidate.adapter_type == "open_image_models_yolov9_openvino":
        base: V2Adapter = OpenImageModelsYoloV9Adapter(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            score_floor=score_floor,
        )
    elif candidate.adapter_type == "rfdetr_alpr_openvino":
        base = RFDETRALPRAdapter(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            score_floor=score_floor,
        )
    elif candidate.adapter_type == "rtdetr_v2_transformers":
        base = RTDetrV2Adapter(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            score_floor=score_floor,
        )
    else:
        raise ValueError(f"unsupported v2 plate candidate adapter: {candidate.adapter_type}")
    if mode is not None and mode.tiled:
        return TiledAdapter(
            base,
            tile_size=mode.tile_size,
            tile_overlap=mode.tile_overlap,
            include_full_frame=mode.include_full_frame,
        )
    return base


def preprocess_letterbox_rgb_01(
    frame: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    return tensor, {
        "input_height": input_size,
        "input_width": input_size,
        "original_height": height,
        "original_width": width,
        "pad_x": left,
        "pad_y": top,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "scale": scale,
    }


def preprocess_rfdetr(frame: np.ndarray, input_height: int, input_width: int) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (input_width, input_height), interpolation=cv.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)


def parse_oim_yolov9_output(
    output: np.ndarray,
    *,
    detector_id: str,
    frame_width: int,
    frame_height: int,
    mapping: dict[str, Any],
    score_floor: float,
    inference_time_ms: float,
) -> plate_bakeoff.PlateDetectionRun:
    raw = np.asarray(output)
    if raw.size == 0:
        return plate_bakeoff.PlateDetectionRun(
            final_detections=[],
            pre_nms_detections=[],
            diagnostics={
                "final_detection_count": 0,
                "model_output_shape": list(raw.shape),
                "raw_candidate_count": 0,
                "score_floor": score_floor,
            },
        )
    predictions = raw.reshape(-1, raw.shape[-1])
    if predictions.shape[1] < 7:
        raise ValueError(f"expected YOLOv9 output with at least 7 columns, got {raw.shape}")
    detections: list[plate_bakeoff.PlateDetection] = []
    for index, row in enumerate(predictions):
        class_id = int(row[5])
        score = float(row[6])
        if class_id != 0 or score < score_floor:
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
            plate_bakeoff.PlateDetection(
                detector_id=detector_id,
                frame_index=-1,
                score=score,
                bbox_xyxy=bbox,
                inference_time_ms=inference_time_ms,
                label=class_id,
                source_detection_index=index,
                geometry={
                    "derived_redaction_bbox_xyxy": list(bbox),
                    "model_geometry_type": "axis_aligned_xyxy",
                    "model_xyxy_letterboxed": [round(float(value), 6) for value in row[1:5]],
                    "oriented_bbox": None,
                },
                preprocessing={
                    "color_order": "BGR_to_RGB",
                    "layout": "NCHW",
                    "letterbox": mapping,
                    "pixel_scale": "0..1",
                },
                postprocessing={
                    "class_filter": 0,
                    "end_to_end_nms_model": True,
                    "model_output_shape": list(raw.shape),
                    "nms_stage": "post_model_nms",
                },
            )
        )
    detections = plate_bakeoff.sort_plate_detections(detections)
    return plate_bakeoff.PlateDetectionRun(
        final_detections=detections,
        pre_nms_detections=detections,
        diagnostics={
            "end_to_end_nms_model": True,
            "final_detection_count": len(detections),
            "model_output_shape": list(raw.shape),
            "raw_candidate_count": len(detections),
            "score_floor": score_floor,
        },
    )


def parse_rfdetr_outputs(
    boxes: np.ndarray,
    logits: np.ndarray,
    *,
    detector_id: str,
    frame_width: int,
    frame_height: int,
    score_floor: float,
    nms_iou_threshold: float,
    inference_time_ms: float,
    output_name_note: str = "",
) -> plate_bakeoff.PlateDetectionRun:
    boxes_2d = remove_batch_dimension(np.asarray(boxes), "boxes")
    logits_2d = remove_batch_dimension(np.asarray(logits), "logits")
    if boxes_2d.ndim != 2 or boxes_2d.shape[1] != 4:
        raise ValueError(f"expected RF-DETR boxes [Q,4], got {boxes_2d.shape}")
    if logits_2d.ndim != 2:
        raise ValueError(f"expected RF-DETR logits [Q,C], got {logits_2d.shape}")
    if boxes_2d.shape[0] != logits_2d.shape[0]:
        raise ValueError("RF-DETR boxes/logits query counts do not match")

    scores = sigmoid(logits_2d[:, 0])
    pre_nms: list[plate_bakeoff.PlateDetection] = []
    for index, (box, score) in enumerate(zip(boxes_2d, scores, strict=True)):
        if float(score) < score_floor:
            continue
        bbox = normalized_cxcywh_to_xyxy(box, frame_width, frame_height)
        if bbox is None:
            continue
        pre_nms.append(
            plate_bakeoff.PlateDetection(
                detector_id=detector_id,
                frame_index=-1,
                score=float(score),
                bbox_xyxy=bbox,
                inference_time_ms=inference_time_ms,
                label=0,
                source_detection_index=index,
                geometry={
                    "derived_redaction_bbox_xyxy": list(bbox),
                    "model_cxcywh_normalized": [round(float(value), 8) for value in box.tolist()],
                    "model_geometry_type": "normalized_cxcywh",
                    "oriented_bbox": None,
                },
                preprocessing={
                    "color_order": "BGR_to_RGB",
                    "imagenet_normalization": True,
                    "layout": "NCHW",
                    "resize": [frame_width, frame_height],
                },
                postprocessing={
                    "class_filter": 0,
                    "nms_stage": "pre_nms",
                    "output_name_note": output_name_note,
                },
            )
        )
    final = [
        item.with_metadata(
            postprocessing={
                **item.postprocessing,
                "nms_iou_threshold": nms_iou_threshold,
                "nms_stage": "post_nms",
                "pre_nms_candidate_count": len(pre_nms),
            }
        )
        for item in plate_bakeoff.non_max_suppression(
            pre_nms,
            iou_threshold=nms_iou_threshold,
        )
    ]
    return plate_bakeoff.PlateDetectionRun(
        final_detections=plate_bakeoff.sort_plate_detections(final),
        pre_nms_detections=plate_bakeoff.sort_plate_detections(pre_nms),
        diagnostics={
            "final_detection_count": len(final),
            "nms_iou_threshold": nms_iou_threshold,
            "output_name_note": output_name_note,
            "raw_candidate_count": len(pre_nms),
            "score_floor": score_floor,
        },
    )


def resolve_rfdetr_output_ports(compiled_model: Any) -> tuple[Any, Any]:
    ports = list(compiled_model.outputs)
    logits_port = next(
        (port for port in ports if len(port.shape) == 3 and int(port.shape[-1]) == 1),
        None,
    )
    boxes_port = next(
        (port for port in ports if len(port.shape) == 3 and int(port.shape[-1]) == 4),
        None,
    )
    if logits_port is None or boxes_port is None:
        raise ValueError("could not resolve RF-DETR logits/boxes outputs")
    return logits_port, boxes_port


def openvino_dimension_to_int(dimension: Any) -> int:
    if hasattr(dimension, "is_static"):
        is_static = dimension.is_static() if callable(dimension.is_static) else dimension.is_static
        if not is_static:
            return -1
    elif str(dimension) == "?":
        return -1
    if hasattr(dimension, "get_length"):
        return int(dimension.get_length())
    return int(dimension)


def remove_batch_dimension(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim != 3:
        return values
    if values.shape[0] != 1:
        raise ValueError(f"only single-image {name} output is supported")
    return values[0]


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), -88.0, 88.0)
    return np.reciprocal(np.float32(1.0) + np.exp(-clipped))


def yolo_letterboxed_xyxy_to_original(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    mapping: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    scale = float(mapping["scale"])
    pad_x = float(mapping["pad_x"])
    pad_y = float(mapping["pad_y"])
    bbox = (
        math.floor((float(x1) - pad_x) / scale),
        math.floor((float(y1) - pad_y) / scale),
        math.ceil((float(x2) - pad_x) / scale),
        math.ceil((float(y2) - pad_y) / scale),
    )
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def normalized_cxcywh_to_xyxy(
    box: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    cx, cy, width, height = [float(value) for value in box.tolist()]
    bbox = (
        math.floor((cx - width / 2.0) * frame_width),
        math.floor((cy - height / 2.0) * frame_height),
        math.ceil((cx + width / 2.0) * frame_width),
        math.ceil((cy + height / 2.0) * frame_height),
    )
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def run_candidate_on_clip(
    candidate: CandidateSpec,
    clip: ClipSpec,
    mode: InferenceMode,
    output_path: Path,
    *,
    score_floor: float = SCORE_FLOOR,
    max_frames: int | None = None,
) -> dict[str, Any]:
    adapter = create_adapter(candidate, mode=mode, score_floor=score_floor)
    benchmark = clip.to_plate_benchmark()
    info = media.probe_media(clip.video_path)
    plate_bakeoff.validate_clip_hash(benchmark, info.sha256)
    capture = cv.VideoCapture(str(clip.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {clip.video_path}")

    frame_indexes = clip.evaluated_frames()
    if max_frames is not None:
        frame_indexes = frame_indexes[:max_frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inference_ms_total = 0.0
    total_final = 0
    total_pre_nms = 0
    frames_with_final = 0
    frames_with_pre_nms = 0
    diagnostics_sample: dict[str, Any] = {}
    try:
        with output_path.open("w", encoding="utf-8") as file:
            for source_frame_index in frame_indexes:
                capture.set(cv.CAP_PROP_POS_FRAMES, source_frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"could not read {clip.clip_id} frame {source_frame_index}")
                frame_started = time.perf_counter()
                run = adapter.detect_with_diagnostics(frame)
                elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
                inference_ms_total += elapsed_ms
                final = [
                    item.with_frame_and_index(source_frame_index, index)
                    for index, item in enumerate(run.final_detections)
                ]
                pre_nms = [
                    item.with_frame_and_index(source_frame_index, index)
                    for index, item in enumerate(run.pre_nms_detections)
                ]
                record = {
                    "candidate_id": candidate.detector_id,
                    "clip_id": clip.clip_id,
                    "detections": [item.to_dict() for item in final],
                    "diagnostics": run.diagnostics,
                    "frame_index": source_frame_index,
                    "inference_mode": mode.mode_id,
                    "pre_nms_detections": [item.to_dict() for item in pre_nms],
                    "schema_version": SCHEMA_VERSION,
                }
                file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                total_final += len(final)
                total_pre_nms += len(pre_nms)
                frames_with_final += int(bool(final))
                frames_with_pre_nms += int(bool(pre_nms))
                if len(diagnostics_sample) < 5:
                    diagnostics_sample[str(source_frame_index)] = run.diagnostics
    finally:
        capture.release()
    elapsed_seconds = time.perf_counter() - started
    frames_processed = len(frame_indexes)
    return {
        "adapter_metadata": adapter.metadata(),
        "candidate": candidate.to_dict(),
        "clip": clip.to_dict(),
        "created_at": media.utc_timestamp(),
        "detections_path": plate_bakeoff.portable_path(output_path),
        "detector_inference_fps": (
            round(frames_processed / (inference_ms_total / 1000.0), 6)
            if inference_ms_total > 0
            else 0.0
        ),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "evaluated_frames_processed": frames_processed,
        "frames_containing_at_least_one_detection": frames_with_final,
        "frames_containing_at_least_one_pre_nms_detection": frames_with_pre_nms,
        "inference_mode": mode.to_dict(),
        "postprocessing_frame_diagnostics_sample": diagnostics_sample,
        "processing_fps": round(frames_processed / elapsed_seconds, 6)
        if elapsed_seconds > 0
        else 0.0,
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": int(info.video_stream.frame_count or 0),
        "source_sha256": info.sha256,
        "total_detections": total_final,
        "total_pre_nms_detections": total_pre_nms,
    }


def load_records(
    path: Path,
) -> tuple[
    dict[int, list[plate_bakeoff.PlateDetection]], dict[int, list[plate_bakeoff.PlateDetection]]
]:
    return plate_bakeoff.load_detection_jsonl_with_pre_nms(path)


def evaluate_threshold_grid(
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    pre_nms: dict[int, list[plate_bakeoff.PlateDetection]],
    clip: ClipSpec,
) -> dict[str, Any]:
    return plate_bakeoff.evaluate_threshold_grid(
        records,
        clip.to_plate_benchmark(),
        THRESHOLD_GRID,
        pre_nms_records_by_frame=pre_nms,
    )


def selected_metric(metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    for row in metrics["thresholds"]:
        if float(row["threshold"]) == threshold:
            return row
    raise ValueError(f"missing threshold {threshold}")


def choose_candidate_threshold(
    by_clip: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    best_threshold = THRESHOLD_GRID[0]
    best_row: dict[str, Any] = {}
    for threshold in THRESHOLD_GRID:
        rows = {
            clip_id: selected_metric(metrics, threshold) for clip_id, metrics in by_clip.items()
        }
        quantitative = [row for row in rows.values() if row["certified_quantitative"]]
        if not quantitative:
            score = (-1.0, 10**9, 10**9)
        else:
            worst_recall = min(float(row["object_frame_recall"] or 0.0) for row in quantitative)
            max_miss = max(int(row["longest_consecutive_miss_run"] or 0) for row in quantitative)
            false_frames = sum(int(row["false_positive_frames"] or 0) for row in quantitative)
            score = (worst_recall, -max_miss, -false_frames)
        if not best_row or score > best_row["selection_score"]:
            best_threshold = threshold
            best_row = {"rows": rows, "selection_score": score}
    return best_threshold, best_row["rows"]


def write_overlay_video(
    clip: ClipSpec,
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    output_path: Path,
    *,
    threshold: float,
    max_frames: int | None = None,
) -> dict[str, Any]:
    temp_path = output_path.with_suffix(".raw.mp4")
    info = media.probe_media(clip.video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    width = info.video_stream.width
    height = info.video_stream.height
    capture = cv.VideoCapture(str(clip.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {clip.video_path}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(str(temp_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create overlay video: {temp_path}")
    frames = clip.evaluated_frames()
    if max_frames is not None:
        frames = frames[:max_frames]
    try:
        for frame_index in frames:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            writer.write(
                plate_bakeoff.draw_plate_overlay(
                    frame,
                    frame_index,
                    records.get(frame_index, []),
                    threshold=threshold,
                    target_roi=clip.target_roi_xyxy,
                )
            )
    finally:
        capture.release()
        writer.release()
    transcode_to_h264(temp_path, output_path)
    temp_path.unlink(missing_ok=True)
    return {
        "frames": len(frames),
        "output_path": plate_bakeoff.portable_path(output_path),
        "output_sha256": media.sha256_file(output_path),
        "threshold": threshold,
    }


def write_contact_sheet(
    clip: ClipSpec,
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    output_path: Path,
    *,
    frames: list[int],
    threshold: float,
) -> dict[str, Any]:
    return plate_bakeoff.write_contact_sheet_for_frames(
        clip.to_plate_benchmark(),
        records,
        output_path,
        frames=frames,
        threshold=threshold,
    )


def write_comparison_video(
    clip: ClipSpec,
    records_by_candidate: dict[str, dict[int, list[plate_bakeoff.PlateDetection]]],
    output_path: Path,
    *,
    threshold: float,
    max_frames: int = 180,
) -> dict[str, Any]:
    temp_path = output_path.with_suffix(".raw.mp4")
    info = media.probe_media(clip.video_path)
    fps = plate_bakeoff.rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    capture = cv.VideoCapture(str(clip.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {clip.video_path}")
    tile_width = 640
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    candidate_ids = list(records_by_candidate)
    output_size = (tile_width * (len(candidate_ids) + 1), tile_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(str(temp_path), cv.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create comparison video: {temp_path}")
    frames = clip.evaluated_frames()[:max_frames]
    try:
        for frame_index in frames:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            tiles = [labelled_tile(frame, f"Original {frame_index}", tile_width)]
            for candidate_id in candidate_ids:
                overlay = plate_bakeoff.draw_plate_overlay(
                    frame,
                    frame_index,
                    records_by_candidate[candidate_id].get(frame_index, []),
                    threshold=threshold,
                    target_roi=clip.target_roi_xyxy,
                )
                tiles.append(labelled_tile(overlay, candidate_id, tile_width))
            writer.write(np.hstack(tiles))
    finally:
        capture.release()
        writer.release()
    transcode_to_h264(temp_path, output_path)
    temp_path.unlink(missing_ok=True)
    return {
        "frames": len(frames),
        "output_path": plate_bakeoff.portable_path(output_path),
        "output_sha256": media.sha256_file(output_path),
        "threshold": threshold,
    }


def labelled_tile(frame: np.ndarray, label: str, tile_width: int) -> np.ndarray:
    tile_height = round(tile_width * frame.shape[0] / frame.shape[1])
    resized = cv.resize(frame, (tile_width, tile_height), interpolation=cv.INTER_AREA)
    cv.rectangle(resized, (0, 0), (tile_width, 28), (0, 0, 0), -1)
    cv.putText(
        resized,
        label[:64],
        (8, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return resized


def transcode_to_h264(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def frames_missing_at_threshold(
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    clip: ClipSpec,
    *,
    threshold: float,
) -> list[int]:
    return plate_bakeoff.frames_missing_at_threshold(
        records,
        clip.to_plate_benchmark(),
        threshold=threshold,
    )


def frames_with_offtarget_detections(
    records: dict[int, list[plate_bakeoff.PlateDetection]],
    clip: ClipSpec,
    *,
    threshold: float,
) -> list[int]:
    return plate_bakeoff.frames_with_offtarget_detections(
        records,
        clip.to_plate_benchmark(),
        threshold=threshold,
    )
