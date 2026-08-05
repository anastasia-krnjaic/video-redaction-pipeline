from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Protocol

import cv2 as cv
import numpy as np

from redactor import media
from redactor.models import Detection

SCHEMA_VERSION = "license_plate_detector_bakeoff_v1_1"
PLATE_CATEGORY = "license_plate"


class PlateDetectorAdapter(Protocol):
    detector_id: str

    def detect(self, frame: np.ndarray) -> list[PlateDetection]: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PlateDetection:
    detector_id: str
    frame_index: int
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    inference_time_ms: float
    category: str = PLATE_CATEGORY
    label: int | None = None
    source_detection_index: int | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category != PLATE_CATEGORY:
            raise ValueError("plate bake-off detections must use category license_plate")
        if not math.isfinite(self.score):
            raise ValueError("detection score must be finite")
        if self.score < 0:
            raise ValueError("detection score cannot be negative")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detection bbox must have positive area")
        if self.inference_time_ms < 0:
            raise ValueError("inference_time_ms cannot be negative")

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def with_frame_and_index(self, frame_index: int, detection_index: int) -> PlateDetection:
        return PlateDetection(
            detector_id=self.detector_id,
            frame_index=frame_index,
            score=self.score,
            bbox_xyxy=self.bbox_xyxy,
            inference_time_ms=self.inference_time_ms,
            category=self.category,
            label=self.label,
            source_detection_index=detection_index,
            geometry=self.geometry,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing,
        )

    def with_metadata(
        self,
        *,
        geometry: dict[str, Any] | None = None,
        preprocessing: dict[str, Any] | None = None,
        postprocessing: dict[str, Any] | None = None,
    ) -> PlateDetection:
        return PlateDetection(
            detector_id=self.detector_id,
            frame_index=self.frame_index,
            score=self.score,
            bbox_xyxy=self.bbox_xyxy,
            inference_time_ms=self.inference_time_ms,
            category=self.category,
            label=self.label,
            source_detection_index=self.source_detection_index,
            geometry=geometry if geometry is not None else self.geometry,
            preprocessing=preprocessing if preprocessing is not None else self.preprocessing,
            postprocessing=postprocessing if postprocessing is not None else self.postprocessing,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "category": self.category,
            "detector_id": self.detector_id,
            "frame_index": self.frame_index,
            "inference_time_ms": round(self.inference_time_ms, 6),
            "score": self.score,
            **optional_field("label", self.label),
            **optional_field("source_detection_index", self.source_detection_index),
            **optional_field("geometry", self.geometry or None),
            "postprocessing": self.postprocessing,
            "preprocessing": self.preprocessing,
        }


@dataclass(frozen=True)
class PlateDetectionRun:
    final_detections: list[PlateDetection]
    pre_nms_detections: list[PlateDetection] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlateCandidate:
    detector_id: str
    display_name: str
    adapter_type: str
    model_path: Path
    model_bin_path: Path | None = None
    threshold_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
    source_url: str | None = None
    license_name: str | None = None
    notes: str = ""
    adapter_options: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        model_hashes = {self.model_path.name: media.sha256_file(self.model_path)}
        if self.model_bin_path is not None:
            model_hashes[self.model_bin_path.name] = media.sha256_file(self.model_bin_path)
        return {
            "adapter_type": self.adapter_type,
            "detector_id": self.detector_id,
            "display_name": self.display_name,
            "license": self.license_name,
            "model_files_sha256": model_hashes,
            "model_path": portable_path(self.model_path),
            "notes": self.notes,
            "source_url": self.source_url,
            "threshold_grid": list(self.threshold_grid),
            **optional_field("adapter_options", self.adapter_options or None),
        }


@dataclass(frozen=True)
class PlateBenchmark:
    clip_id: str
    video_path: Path
    source_sha256: str
    evaluated_ranges: tuple[tuple[int, int], ...]
    target_id: str | None = None
    target_roi_xyxy: tuple[int, int, int, int] | None = None
    certified_quantitative: bool = True
    notes: str = ""

    def evaluated_frames(self) -> list[int]:
        frames: list[int] = []
        for start, end in self.evaluated_ranges:
            if end < start:
                raise ValueError("evaluated range end must be greater than or equal to start")
            frames.extend(range(start, end + 1))
        return frames

    def metadata(self) -> dict[str, Any]:
        return {
            "certified_quantitative": self.certified_quantitative,
            "clip_id": self.clip_id,
            "evaluated_frame_count": len(self.evaluated_frames()),
            "evaluated_ranges": [list(item) for item in self.evaluated_ranges],
            "notes": self.notes,
            "source_sha256": self.source_sha256,
            "target_id": self.target_id,
            "target_roi_xyxy": list(self.target_roi_xyxy) if self.target_roi_xyxy else None,
            "video_path": portable_path(self.video_path),
        }


class OpenVINOSSDPlateDetector:
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        model_bin_path: Path | None = None,
        score_floor: float = 0.01,
        device: str = "CPU",
    ):
        import openvino as ov

        self.detector_id = detector_id
        self.model_path = model_path
        self.model_bin_path = model_bin_path or model_path.with_suffix(".bin")
        self.score_floor = score_floor
        self.device = device
        self.core = ov.Core()
        model = self.core.read_model(str(model_path), str(self.model_bin_path))
        self.compiled_model = self.core.compile_model(model, device)
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)
        shape = [int(dim) for dim in self.input_port.shape]
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"expected NHWC plate SSD input, got {shape}")
        self.input_height = shape[1]
        self.input_width = shape[2]

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        return self.detect_with_diagnostics(frame).final_detections

    def detect_with_diagnostics(self, frame: np.ndarray) -> PlateDetectionRun:
        started = time.perf_counter()
        resized = cv.resize(
            frame,
            (self.input_width, self.input_height),
            interpolation=cv.INTER_LINEAR,
        ).astype(np.float32)
        output = self.compiled_model({self.input_port: resized[np.newaxis, :]})[self.output_port]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detections = parse_ssd_output(
            output,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            detector_id=self.detector_id,
            score_floor=self.score_floor,
            inference_time_ms=elapsed_ms,
        )
        return PlateDetectionRun(
            final_detections=detections,
            pre_nms_detections=detections,
            diagnostics={
                "nms_applied": False,
                "raw_candidate_count": len(detections),
                "final_detection_count": len(detections),
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "expected_color_order": "BGR",
            "input_shape": [1, self.input_height, self.input_width, 3],
            "model_files_sha256": {
                self.model_path.name: media.sha256_file(self.model_path),
                self.model_bin_path.name: media.sha256_file(self.model_bin_path),
            },
            "output_format": "SSD [image_id, label, confidence, x_min, y_min, x_max, y_max]",
            "plate_label": 2,
            "runtime": f"OpenVINO {self.device}",
            "score_floor": self.score_floor,
        }


class TiledOpenVINOSSDPlateDetector(OpenVINOSSDPlateDetector):
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        model_bin_path: Path | None = None,
        score_floor: float = 0.01,
        tile_size: int = 960,
        tile_overlap: float = 0.25,
        include_full_frame: bool = True,
        cross_tile_nms_iou_threshold: float = 0.45,
        device: str = "CPU",
    ):
        super().__init__(
            detector_id=detector_id,
            model_path=model_path,
            model_bin_path=model_bin_path,
            score_floor=score_floor,
            device=device,
        )
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1)")
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.include_full_frame = include_full_frame
        self.cross_tile_nms_iou_threshold = cross_tile_nms_iou_threshold

    def detect_with_diagnostics(self, frame: np.ndarray) -> PlateDetectionRun:
        frame_height, frame_width = frame.shape[:2]
        tiles = build_uniform_tiles(
            frame_width,
            frame_height,
            tile_size=self.tile_size,
            overlap_fraction=self.tile_overlap,
            include_full_frame=self.include_full_frame,
        )
        started = time.perf_counter()
        pre_nms: list[PlateDetection] = []
        tile_detection_counts: list[dict[str, Any]] = []
        for tile_index, (x1, y1, x2, y2) in enumerate(tiles):
            crop = frame[y1:y2, x1:x2]
            crop_run = super().detect_with_diagnostics(crop)
            tile_count = 0
            for detection in crop_run.final_detections:
                mapped_bbox = clip_xyxy_to_frame(
                    (
                        detection.bbox_xyxy[0] + x1,
                        detection.bbox_xyxy[1] + y1,
                        detection.bbox_xyxy[2] + x1,
                        detection.bbox_xyxy[3] + y1,
                    ),
                    frame_width,
                    frame_height,
                )
                if mapped_bbox is None:
                    continue
                preprocessing = {
                    **detection.preprocessing,
                    "tile_index": tile_index,
                    "tile_xyxy": [x1, y1, x2, y2],
                    "tiled_inference": True,
                }
                geometry = {
                    "model_geometry_type": "axis_aligned_xyxy",
                    "source_tile_bbox_xyxy": list(detection.bbox_xyxy),
                    "redaction_bbox_xyxy": list(mapped_bbox),
                }
                pre_nms.append(
                    PlateDetection(
                        detector_id=self.detector_id,
                        frame_index=-1,
                        score=detection.score,
                        bbox_xyxy=mapped_bbox,
                        inference_time_ms=detection.inference_time_ms,
                        label=detection.label,
                        geometry=geometry,
                        preprocessing=preprocessing,
                        postprocessing={
                            **detection.postprocessing,
                            "nms_stage": "pre_cross_tile_nms",
                        },
                    )
                )
                tile_count += 1
            tile_detection_counts.append({"detections": tile_count, "tile_index": tile_index})
        final = [
            detection.with_metadata(
                postprocessing={
                    **detection.postprocessing,
                    "cross_tile_nms_iou_threshold": self.cross_tile_nms_iou_threshold,
                    "nms_stage": "post_cross_tile_nms",
                    "pre_nms_candidate_count": len(pre_nms),
                    "tile_count": len(tiles),
                }
            )
            for detection in non_max_suppression(
                pre_nms,
                iou_threshold=self.cross_tile_nms_iou_threshold,
            )
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return PlateDetectionRun(
            final_detections=sort_plate_detections(final),
            pre_nms_detections=sort_plate_detections(pre_nms),
            diagnostics={
                "cross_tile_nms_iou_threshold": self.cross_tile_nms_iou_threshold,
                "elapsed_ms": round(elapsed_ms, 6),
                "final_detection_count": len(final),
                "include_full_frame": self.include_full_frame,
                "raw_candidate_count": len(pre_nms),
                "tile_detection_counts": tile_detection_counts,
                "tile_overlap": self.tile_overlap,
                "tile_size": self.tile_size,
                "tiles": [list(tile) for tile in tiles],
            },
        )

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "cross_tile_nms_iou_threshold": self.cross_tile_nms_iou_threshold,
                "include_full_frame": self.include_full_frame,
                "small_object_mode": "whole_frame_plus_uniform_tiles",
                "tile_overlap": self.tile_overlap,
                "tile_size": self.tile_size,
            }
        )
        return metadata


class OpenVINOYOLOv8PlateDetector:
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        model_bin_path: Path | None = None,
        score_floor: float = 0.01,
        nms_iou_threshold: float = 0.45,
        device: str = "CPU",
    ):
        import openvino as ov

        self.detector_id = detector_id
        self.model_path = model_path
        self.model_bin_path = model_bin_path or model_path.with_suffix(".bin")
        self.score_floor = score_floor
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.core = ov.Core()
        model = self.core.read_model(str(model_path), str(self.model_bin_path))
        self.compiled_model = self.core.compile_model(model, device)
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)
        shape = [int(dim) for dim in self.input_port.shape]
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"expected NCHW YOLOv8 input, got {shape}")
        self.input_height = shape[2]
        self.input_width = shape[3]

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        return self.detect_with_diagnostics(frame).final_detections

    def detect_with_diagnostics(self, frame: np.ndarray) -> PlateDetectionRun:
        started = time.perf_counter()
        tensor, mapping = preprocess_yolov8_frame(
            frame,
            input_width=self.input_width,
            input_height=self.input_height,
        )
        output = self.compiled_model({self.input_port: tensor})[self.output_port]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return parse_yolov8_output_run(
            output,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            detector_id=self.detector_id,
            score_floor=self.score_floor,
            nms_iou_threshold=self.nms_iou_threshold,
            letterbox_mapping=mapping,
            inference_time_ms=elapsed_ms,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "expected_color_order": "BGR",
            "input_shape": [1, 3, self.input_height, self.input_width],
            "model_files_sha256": {
                self.model_path.name: media.sha256_file(self.model_path),
                self.model_bin_path.name: media.sha256_file(self.model_bin_path),
            },
            "nms_iou_threshold": self.nms_iou_threshold,
            "output_format": (
                "YOLOv8 detect [1, 5, anchors], rows are decoded "
                "axis-aligned cx, cy, w, h, class score; no angle channel is present"
            ),
            "pixel_scale": "0..255 float32",
            "postprocessing_audit": {
                "class_count": 1,
                "class_filter": 0,
                "model_metadata_task": "detect",
                "nms_in_model": False,
                "oriented_bbox_output": False,
            },
            "runtime": f"OpenVINO {self.device}",
            "score_floor": self.score_floor,
        }


class LicensePlateYuNetDetector:
    detector_id = "license_plate_yunet"

    def __init__(
        self,
        model_path: Path,
        threshold: float,
        nms_threshold: float = 0.30,
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
        return sorted(detections, key=lambda item: (-item.score, item.bbox_xyxy))

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


class OpenCVLPDYuNetPlateDetector:
    def __init__(
        self,
        *,
        detector_id: str,
        model_path: Path,
        score_floor: float = 0.01,
    ):
        self.detector_id = detector_id
        self.model_path = model_path
        self.score_floor = score_floor
        self.detector = LicensePlateYuNetDetector(model_path, score_floor)

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        started = time.perf_counter()
        detections = self.detector.detect(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        records: list[PlateDetection] = []
        for index, detection in enumerate(detections):
            records.append(
                PlateDetection(
                    detector_id=self.detector_id,
                    frame_index=-1,
                    source_detection_index=index,
                    score=detection.score,
                    bbox_xyxy=detection.bbox_xyxy,
                    inference_time_ms=elapsed_ms,
                    preprocessing={"opencv_blob_from_image": True},
                    postprocessing={"nms_iou_threshold": self.detector.nms_threshold},
                )
            )
        return records

    def metadata(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "expected_color_order": "BGR",
            "model_files_sha256": {self.model_path.name: media.sha256_file(self.model_path)},
            "output_format": "YuNet quadrilateral converted to clipped xyxy",
            "runtime": f"OpenCV DNN {cv.__version__}",
            "score_floor": self.score_floor,
        }


def create_adapter(candidate: PlateCandidate, *, score_floor: float = 0.01) -> PlateDetectorAdapter:
    if candidate.adapter_type == "openvino_ssd":
        return OpenVINOSSDPlateDetector(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            model_bin_path=candidate.model_bin_path,
            score_floor=score_floor,
        )
    if candidate.adapter_type == "openvino_ssd_tiled":
        return TiledOpenVINOSSDPlateDetector(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            model_bin_path=candidate.model_bin_path,
            score_floor=score_floor,
            tile_size=int(candidate.adapter_options.get("tile_size", 960)),
            tile_overlap=float(candidate.adapter_options.get("tile_overlap", 0.25)),
            include_full_frame=bool(candidate.adapter_options.get("include_full_frame", True)),
            cross_tile_nms_iou_threshold=float(
                candidate.adapter_options.get("cross_tile_nms_iou_threshold", 0.45)
            ),
        )
    if candidate.adapter_type == "openvino_yolov8":
        return OpenVINOYOLOv8PlateDetector(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            model_bin_path=candidate.model_bin_path,
            score_floor=score_floor,
            nms_iou_threshold=float(candidate.adapter_options.get("nms_iou_threshold", 0.45)),
        )
    if candidate.adapter_type == "opencv_lpd_yunet":
        return OpenCVLPDYuNetPlateDetector(
            detector_id=candidate.detector_id,
            model_path=candidate.model_path,
            score_floor=score_floor,
        )
    raise ValueError(f"unsupported plate bake-off candidate adapter: {candidate.adapter_type}")


def parse_ssd_output(
    output: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    detector_id: str,
    score_floor: float,
    inference_time_ms: float,
    plate_label: int = 2,
) -> list[PlateDetection]:
    raw_detections = np.asarray(output).reshape(-1, 7)
    detections: list[PlateDetection] = []
    for raw in raw_detections:
        image_id, label, confidence, x_min, y_min, x_max, y_max = raw.tolist()
        if image_id < 0 or int(label) != plate_label:
            continue
        score = float(confidence)
        if score < score_floor:
            continue
        bbox = normalized_xyxy_to_frame(x_min, y_min, x_max, y_max, frame_width, frame_height)
        if bbox is None:
            continue
        detections.append(
            PlateDetection(
                detector_id=detector_id,
                frame_index=-1,
                score=score,
                bbox_xyxy=bbox,
                inference_time_ms=inference_time_ms,
                label=int(label),
                preprocessing={"resize": [300, 300], "layout": "NHWC", "color_order": "BGR"},
                postprocessing={"label_filter": plate_label},
            )
        )
    return sort_plate_detections(detections)


def preprocess_yolov8_frame(
    frame: np.ndarray,
    *,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_height, image_width = frame.shape[:2]
    scale = min(input_width / image_width, input_height / image_height)
    resized_width = int(round(image_width * scale))
    resized_height = int(round(image_height * scale))
    pad_x = (input_width - resized_width) / 2.0
    pad_y = (input_height - resized_height) / 2.0
    resized = cv.resize(frame, (resized_width, resized_height), interpolation=cv.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :]
    mapping = {
        "input_height": input_height,
        "input_width": input_width,
        "original_height": image_height,
        "original_width": image_width,
        "pad_x": left,
        "pad_y": top,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "scale": scale,
    }
    return tensor, mapping


def parse_yolov8_output(
    output: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    detector_id: str,
    score_floor: float,
    nms_iou_threshold: float,
    letterbox_mapping: dict[str, Any],
    inference_time_ms: float,
) -> list[PlateDetection]:
    return parse_yolov8_output_run(
        output,
        frame_width=frame_width,
        frame_height=frame_height,
        detector_id=detector_id,
        score_floor=score_floor,
        nms_iou_threshold=nms_iou_threshold,
        letterbox_mapping=letterbox_mapping,
        inference_time_ms=inference_time_ms,
    ).final_detections


def parse_yolov8_output_run(
    output: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    detector_id: str,
    score_floor: float,
    nms_iou_threshold: float,
    letterbox_mapping: dict[str, Any],
    inference_time_ms: float,
) -> PlateDetectionRun:
    raw = np.asarray(output)
    if raw.ndim != 3 or raw.shape[1] < 5:
        raise ValueError(f"expected YOLOv8 output shape [1, rows, anchors], got {raw.shape}")
    if raw.shape[1] != 5:
        raise ValueError(
            "the Intel LPR adapter expects a single-class YOLOv8 detect output with "
            f"5 rows; got {raw.shape[1]} rows"
        )
    candidates = raw[0].T.tolist()
    pre_nms_detections: list[PlateDetection] = []
    for source_index, (cx, cy, width, height, score) in enumerate(candidates):
        score = float(score)
        if score < score_floor:
            continue
        bbox = yolo_xywh_to_original_xyxy(
            cx,
            cy,
            width,
            height,
            frame_width,
            frame_height,
            letterbox_mapping,
        )
        if bbox is None:
            continue
        geometry = {
            "derived_redaction_bbox_xyxy": list(bbox),
            "model_geometry_type": "axis_aligned_xywh",
            "model_xywh_letterboxed": [
                round(float(cx), 6),
                round(float(cy), 6),
                round(float(width), 6),
                round(float(height), 6),
            ],
            "oriented_bbox": None,
        }
        pre_nms_detections.append(
            PlateDetection(
                detector_id=detector_id,
                frame_index=-1,
                score=score,
                bbox_xyxy=bbox,
                inference_time_ms=inference_time_ms,
                label=0,
                source_detection_index=source_index,
                geometry=geometry,
                preprocessing={
                    "layout": "NCHW",
                    "letterbox": letterbox_mapping,
                    "pixel_scale": "0..255",
                    "color_order": "BGR",
                },
                postprocessing={
                    "class_filter": 0,
                    "model_output_shape": list(raw.shape),
                    "nms_stage": "pre_nms",
                    "oriented_bbox_output": False,
                },
            )
        )
    final_detections = [
        detection.with_metadata(
            postprocessing={
                **detection.postprocessing,
                "nms_iou_threshold": nms_iou_threshold,
                "nms_stage": "post_nms",
                "pre_nms_candidate_count": len(pre_nms_detections),
            }
        )
        for detection in non_max_suppression(
            pre_nms_detections,
            iou_threshold=nms_iou_threshold,
        )
    ]
    return PlateDetectionRun(
        final_detections=sort_plate_detections(final_detections),
        pre_nms_detections=sort_plate_detections(pre_nms_detections),
        diagnostics={
            "class_count": 1,
            "final_detection_count": len(final_detections),
            "model_geometry_type": "axis_aligned_xywh",
            "model_output_shape": list(raw.shape),
            "nms_iou_threshold": nms_iou_threshold,
            "oriented_bbox_output": False,
            "raw_candidate_count": len(pre_nms_detections),
            "score_floor": score_floor,
        },
    )


def build_uniform_tiles(
    frame_width: int,
    frame_height: int,
    *,
    tile_size: int,
    overlap_fraction: float,
    include_full_frame: bool,
) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    stride = max(1, int(round(tile_size * (1.0 - overlap_fraction))))
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


def normalized_xyxy_to_frame(
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


def yolo_xywh_to_original_xyxy(
    cx: float,
    cy: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
    mapping: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    scale = float(mapping["scale"])
    pad_x = float(mapping["pad_x"])
    pad_y = float(mapping["pad_y"])
    x1 = (float(cx) - float(width) / 2.0 - pad_x) / scale
    y1 = (float(cy) - float(height) / 2.0 - pad_y) / scale
    x2 = (float(cx) + float(width) / 2.0 - pad_x) / scale
    y2 = (float(cy) + float(height) / 2.0 - pad_y) / scale
    bbox = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    return clip_xyxy_to_frame(bbox, frame_width, frame_height)


def non_max_suppression(
    detections: list[PlateDetection],
    *,
    iou_threshold: float,
) -> list[PlateDetection]:
    if not detections:
        return []
    sorted_detections = sort_plate_detections(detections)
    rects = [xyxy_to_xywh(item.bbox_xyxy) for item in sorted_detections]
    scores = [float(item.score) for item in sorted_detections]
    raw_indices = cv.dnn.NMSBoxes(
        bboxes=rects,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    indices = normalize_nms_indices(raw_indices)
    return [sorted_detections[index] for index in indices]


def oriented_box_to_axis_aligned_xyxy(
    *,
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_degrees: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    points = cv.boxPoints(((float(cx), float(cy)), (float(width), float(height)), angle_degrees))
    x_values = points[:, 0]
    y_values = points[:, 1]
    return clip_xyxy_to_frame(
        (
            math.floor(float(np.min(x_values))),
            math.floor(float(np.min(y_values))),
            math.ceil(float(np.max(x_values))),
            math.ceil(float(np.max(y_values))),
        ),
        frame_width,
        frame_height,
    )


def oriented_bbox_iou(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    left_rect = (
        (float(left["cx"]), float(left["cy"])),
        (float(left["width"]), float(left["height"])),
        float(left["angle_degrees"]),
    )
    right_rect = (
        (float(right["cx"]), float(right["cy"])),
        (float(right["width"]), float(right["height"])),
        float(right["angle_degrees"]),
    )
    left_area = float(left["width"]) * float(left["height"])
    right_area = float(right["width"]) * float(right["height"])
    if left_area <= 0 or right_area <= 0:
        return 0.0
    _status, intersection = cv.rotatedRectangleIntersection(left_rect, right_rect)
    if intersection is None:
        return 0.0
    intersection_area = float(cv.contourArea(intersection))
    if intersection_area <= 0:
        return 0.0
    return intersection_area / (left_area + right_area - intersection_area)


def oriented_or_axis_aligned_nms(
    detections: list[PlateDetection],
    *,
    iou_threshold: float,
) -> list[PlateDetection]:
    kept: list[PlateDetection] = []
    for candidate in sort_plate_detections(detections):
        candidate_oriented = candidate.geometry.get("oriented_bbox") if candidate.geometry else None
        duplicate = False
        for existing in kept:
            existing_oriented = (
                existing.geometry.get("oriented_bbox") if existing.geometry else None
            )
            if candidate_oriented and existing_oriented:
                overlap = oriented_bbox_iou(candidate_oriented, existing_oriented)
            else:
                overlap = bbox_iou(candidate.bbox_xyxy, existing.bbox_xyxy)
            if overlap >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def xyxy_to_xywh(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def normalize_nms_indices(indices: Any) -> list[int]:
    if indices is None:
        return []
    if isinstance(indices, tuple):
        return [int(index) for index in indices]
    array = np.asarray(indices)
    if array.size == 0:
        return []
    return [int(index) for index in array.reshape(-1).tolist()]


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


def sort_plate_detections(detections: list[PlateDetection]) -> list[PlateDetection]:
    return sorted(detections, key=lambda item: (-item.score, item.bbox_xyxy))


def run_candidate_on_clip(
    candidate: PlateCandidate,
    benchmark: PlateBenchmark,
    output_path: Path,
    *,
    score_floor: float = 0.01,
    max_frames: int | None = None,
) -> dict[str, Any]:
    adapter = create_adapter(candidate, score_floor=score_floor)
    info = media.probe_media(benchmark.video_path)
    validate_clip_hash(benchmark, info.sha256)
    capture = cv.VideoCapture(str(benchmark.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {benchmark.video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frames_processed = 0
    total_detections = 0
    total_pre_nms_detections = 0
    frames_with_detections = 0
    frames_with_pre_nms_detections = 0
    inference_ms_total = 0.0
    frame_diagnostics: dict[str, Any] = {}
    try:
        with output_path.open("w", encoding="utf-8") as file:
            while max_frames is None or frames_processed < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                detection_run = detect_with_diagnostics(adapter, frame)
                frame_inference_ms = (time.perf_counter() - frame_started) * 1000.0
                detections = [
                    item.with_frame_and_index(frames_processed, index)
                    for index, item in enumerate(detection_run.final_detections)
                ]
                pre_nms_detections = [
                    item.with_frame_and_index(frames_processed, index)
                    for index, item in enumerate(detection_run.pre_nms_detections)
                ]
                inference_ms_total += frame_inference_ms
                frame_record = {
                    "clip_id": benchmark.clip_id,
                    "detections": [item.to_dict() for item in detections],
                    "diagnostics": detection_run.diagnostics,
                    "frame_index": frames_processed,
                    "pre_nms_detections": [item.to_dict() for item in pre_nms_detections],
                }
                file.write(json.dumps(frame_record, sort_keys=True, separators=(",", ":")) + "\n")
                total_detections += len(detections)
                total_pre_nms_detections += len(pre_nms_detections)
                if detections:
                    frames_with_detections += 1
                if pre_nms_detections:
                    frames_with_pre_nms_detections += 1
                frame_diagnostics[str(frames_processed)] = detection_run.diagnostics
                frames_processed += 1
    finally:
        capture.release()

    elapsed_seconds = time.perf_counter() - started
    return {
        "adapter_metadata": adapter.metadata(),
        "candidate": candidate.metadata(),
        "clip": benchmark.metadata(),
        "created_at": media.utc_timestamp(),
        "detections_path": portable_path(output_path),
        "detector_inference_fps": (
            round(frames_processed / (inference_ms_total / 1000.0), 6)
            if inference_ms_total > 0
            else 0.0
        ),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "frames_containing_at_least_one_detection": frames_with_detections,
        "frames_containing_at_least_one_pre_nms_detection": frames_with_pre_nms_detections,
        "frames_processed": frames_processed,
        "postprocessing_frame_diagnostics_sample": {
            key: frame_diagnostics[key] for key in list(frame_diagnostics)[:5]
        },
        "processing_fps": round(frames_processed / elapsed_seconds, 6)
        if elapsed_seconds > 0
        else 0.0,
        "schema_version": SCHEMA_VERSION,
        "source_frame_count": int(info.video_stream.frame_count or 0),
        "source_sha256": info.sha256,
        "total_detections": total_detections,
        "total_pre_nms_detections": total_pre_nms_detections,
    }


def detect_with_diagnostics(
    adapter: PlateDetectorAdapter,
    frame: np.ndarray,
) -> PlateDetectionRun:
    method = getattr(adapter, "detect_with_diagnostics", None)
    if method is not None:
        return method(frame)
    detections = adapter.detect(frame)
    return PlateDetectionRun(
        final_detections=detections,
        pre_nms_detections=detections,
        diagnostics={
            "final_detection_count": len(detections),
            "raw_candidate_count": len(detections),
        },
    )


def load_detection_jsonl(path: Path) -> dict[int, list[PlateDetection]]:
    return load_detection_jsonl_with_pre_nms(path)[0]


def load_detection_jsonl_with_pre_nms(
    path: Path,
) -> tuple[dict[int, list[PlateDetection]], dict[int, list[PlateDetection]]]:
    records: dict[int, list[PlateDetection]] = {}
    pre_nms_records: dict[int, list[PlateDetection]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            frame_index = int(data["frame_index"])
            detections = []
            for item in data.get("detections", []):
                detections.append(
                    PlateDetection(
                        detector_id=str(item["detector_id"]),
                        frame_index=frame_index,
                        score=float(item["score"]),
                        bbox_xyxy=tuple(int(v) for v in item["bbox_xyxy"]),
                        inference_time_ms=float(item.get("inference_time_ms", 0.0)),
                        label=item.get("label"),
                        source_detection_index=item.get("source_detection_index"),
                        geometry=dict(item.get("geometry") or {}),
                        preprocessing=dict(item.get("preprocessing") or {}),
                        postprocessing=dict(item.get("postprocessing") or {}),
                    )
                )
            records[frame_index] = detections
            pre_detections = []
            for item in data.get("pre_nms_detections", data.get("detections", [])):
                pre_detections.append(
                    PlateDetection(
                        detector_id=str(item["detector_id"]),
                        frame_index=frame_index,
                        score=float(item["score"]),
                        bbox_xyxy=tuple(int(v) for v in item["bbox_xyxy"]),
                        inference_time_ms=float(item.get("inference_time_ms", 0.0)),
                        label=item.get("label"),
                        source_detection_index=item.get("source_detection_index"),
                        geometry=dict(item.get("geometry") or {}),
                        preprocessing=dict(item.get("preprocessing") or {}),
                        postprocessing=dict(item.get("postprocessing") or {}),
                    )
                )
            pre_nms_records[frame_index] = pre_detections
    return records, pre_nms_records


def evaluate_threshold(
    records_by_frame: dict[int, list[PlateDetection]],
    benchmark: PlateBenchmark,
    *,
    threshold: float,
    pre_nms_records_by_frame: dict[int, list[PlateDetection]] | None = None,
) -> dict[str, Any]:
    evaluated_frames = benchmark.evaluated_frames()
    hits: set[int] = set()
    missed: list[int] = []
    false_positive_frames: set[int] = set()
    duplicate_frames = 0
    duplicate_frames_before_nms = 0
    detections_per_frame: list[int] = []
    pre_nms_detections_per_frame: list[int] = []
    score_values: list[float] = []
    for frame_index in evaluated_frames:
        detections = [
            item
            for item in records_by_frame.get(frame_index, [])
            if item.score >= threshold and item.category == PLATE_CATEGORY
        ]
        detections_per_frame.append(len(detections))
        score_values.extend(item.score for item in detections)
        pre_nms_detections = [
            item
            for item in (pre_nms_records_by_frame or records_by_frame).get(frame_index, [])
            if item.score >= threshold and item.category == PLATE_CATEGORY
        ]
        pre_nms_detections_per_frame.append(len(pre_nms_detections))
        target_hits = [
            item
            for item in detections
            if benchmark.target_roi_xyxy is not None
            and center_inside_roi(item.center, benchmark.target_roi_xyxy)
        ]
        off_target = [
            item
            for item in detections
            if benchmark.target_roi_xyxy is not None
            and not center_inside_roi(item.center, benchmark.target_roi_xyxy)
        ]
        if target_hits:
            hits.add(frame_index)
            if len(target_hits) > 1 or has_overlapping_duplicates(target_hits):
                duplicate_frames += 1
        pre_nms_target_hits = [
            item
            for item in pre_nms_detections
            if benchmark.target_roi_xyxy is not None
            and center_inside_roi(item.center, benchmark.target_roi_xyxy)
        ]
        if pre_nms_target_hits and (
            len(pre_nms_target_hits) > 1 or has_overlapping_duplicates(pre_nms_target_hits)
        ):
            duplicate_frames_before_nms += 1
        if (
            not target_hits
            and benchmark.certified_quantitative
            and benchmark.target_roi_xyxy is not None
        ):
            missed.append(frame_index)
        if off_target:
            false_positive_frames.add(frame_index)

    denominator = (
        len(evaluated_frames)
        if benchmark.certified_quantitative and benchmark.target_roi_xyxy is not None
        else None
    )
    return {
        "certified_quantitative": benchmark.certified_quantitative
        and benchmark.target_roi_xyxy is not None,
        "detected_target_frames": len(hits) if denominator is not None else None,
        "detections_per_frame_mean": (
            round(sum(detections_per_frame) / len(detections_per_frame), 6)
            if detections_per_frame
            else 0.0
        ),
        "detections_per_frame_mean_after_nms": (
            round(sum(detections_per_frame) / len(detections_per_frame), 6)
            if detections_per_frame
            else 0.0
        ),
        "detections_per_frame_mean_before_nms": (
            round(sum(pre_nms_detections_per_frame) / len(pre_nms_detections_per_frame), 6)
            if pre_nms_detections_per_frame
            else 0.0
        ),
        "duplicate_frames": duplicate_frames,
        "duplicate_frames_after_nms": duplicate_frames,
        "duplicate_frames_before_nms": duplicate_frames_before_nms,
        "evaluated_frame_count": len(evaluated_frames),
        "false_positive_frames": len(false_positive_frames),
        "longest_consecutive_miss_run": longest_run(missed),
        "missed_frames": len(missed) if denominator is not None else None,
        "object_frame_recall": (
            round(len(hits) / denominator, 6) if denominator not in {None, 0} else None
        ),
        "score_distribution": score_distribution(score_values),
        "threshold": threshold,
        "unscored_reason": None
        if denominator is not None
        else "no certified close-clip target ROI or per-frame localization truth",
    }


def evaluate_threshold_grid(
    records_by_frame: dict[int, list[PlateDetection]],
    benchmark: PlateBenchmark,
    thresholds: tuple[float, ...],
    *,
    pre_nms_records_by_frame: dict[int, list[PlateDetection]] | None = None,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark.metadata(),
        "schema_version": SCHEMA_VERSION,
        "thresholds": [
            evaluate_threshold(
                records_by_frame,
                benchmark,
                threshold=threshold,
                pre_nms_records_by_frame=pre_nms_records_by_frame,
            )
            for threshold in thresholds
        ],
    }


def draw_plate_overlay(
    frame: np.ndarray,
    frame_index: int,
    detections: list[PlateDetection],
    *,
    threshold: float,
    target_roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    output = frame.copy()
    cv.putText(
        output,
        f"frame {frame_index}",
        (12, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    if target_roi is not None:
        x1, y1, x2, y2 = target_roi
        cv.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv.putText(
            output,
            "target search ROI",
            (x1, max(24, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv.LINE_AA,
        )
    for detection in detections:
        if detection.score < threshold:
            continue
        x1, y1, x2, y2 = detection.bbox_xyxy
        in_roi = target_roi is not None and center_inside_roi(detection.center, target_roi)
        color = (0, 255, 0) if in_roi else (0, 180, 255)
        cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            output,
            f"{detection.detector_id} {detection.score:.2f}",
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )
    return output


def write_overlay_video(
    benchmark: PlateBenchmark,
    records_by_frame: dict[int, list[PlateDetection]],
    output_path: Path,
    *,
    threshold: float,
) -> dict[str, Any]:
    info = media.probe_media(benchmark.video_path)
    fps = rational_to_float(info.video_stream.nominal_frame_rate) or 30.0
    width = info.video_stream.width
    height = info.video_stream.height
    capture = cv.VideoCapture(str(benchmark.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {benchmark.video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not create overlay video: {output_path}")
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(
                draw_plate_overlay(
                    frame,
                    frames,
                    records_by_frame.get(frames, []),
                    threshold=threshold,
                    target_roi=benchmark.target_roi_xyxy,
                )
            )
            frames += 1
    finally:
        capture.release()
        writer.release()
    return {
        "output_path": portable_path(output_path),
        "output_sha256": media.sha256_file(output_path),
        "threshold": threshold,
        "frames": frames,
    }


def write_contact_sheet_for_frames(
    benchmark: PlateBenchmark,
    records_by_frame: dict[int, list[PlateDetection]],
    output_path: Path,
    *,
    frames: list[int],
    threshold: float,
    columns: int = 4,
) -> dict[str, Any]:
    info = media.probe_media(benchmark.video_path)
    tile_width = 360
    tile_height = round(tile_width * info.video_stream.height / info.video_stream.width)
    selected_frames = frames[:16]
    if not selected_frames:
        selected_frames = benchmark.evaluated_frames()[
            :: max(1, len(benchmark.evaluated_frames()) // 8)
        ][:8]
    rows = math.ceil(len(selected_frames) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    capture = cv.VideoCapture(str(benchmark.video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {benchmark.video_path}")
    try:
        for index, frame_index in enumerate(selected_frames):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            overlay = draw_plate_overlay(
                frame,
                frame_index,
                records_by_frame.get(frame_index, []),
                threshold=threshold,
                target_roi=benchmark.target_roi_xyxy,
            )
            resized = cv.resize(overlay, (tile_width, tile_height), interpolation=cv.INTER_AREA)
            row = index // columns
            column = index % columns
            sheet[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = resized
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), sheet)
    return {
        "frames": selected_frames,
        "output_path": portable_path(output_path),
        "output_sha256": media.sha256_file(output_path),
    }


def frames_missing_at_threshold(
    records_by_frame: dict[int, list[PlateDetection]],
    benchmark: PlateBenchmark,
    *,
    threshold: float,
) -> list[int]:
    if benchmark.target_roi_xyxy is None:
        return []
    missing: list[int] = []
    for frame_index in benchmark.evaluated_frames():
        detections = [
            item
            for item in records_by_frame.get(frame_index, [])
            if item.score >= threshold and center_inside_roi(item.center, benchmark.target_roi_xyxy)
        ]
        if not detections:
            missing.append(frame_index)
    return missing


def frames_with_offtarget_detections(
    records_by_frame: dict[int, list[PlateDetection]],
    benchmark: PlateBenchmark,
    *,
    threshold: float,
) -> list[int]:
    if benchmark.target_roi_xyxy is None:
        return []
    frames: list[int] = []
    for frame_index in benchmark.evaluated_frames():
        detections = [
            item
            for item in records_by_frame.get(frame_index, [])
            if item.score >= threshold
            and not center_inside_roi(item.center, benchmark.target_roi_xyxy)
        ]
        if detections:
            frames.append(frame_index)
    return frames


def candidate_metrics_summary(
    metrics_by_clip: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for clip_id, metrics in metrics_by_clip.items():
        row = next(
            (item for item in metrics["thresholds"] if float(item["threshold"]) == threshold),
            None,
        )
        if row is not None:
            selected[clip_id] = row
    quantitative = [row for row in selected.values() if row["certified_quantitative"]]
    worst_recall = min(
        (
            row["object_frame_recall"]
            for row in quantitative
            if row["object_frame_recall"] is not None
        ),
        default=None,
    )
    max_miss = max((row["longest_consecutive_miss_run"] for row in quantitative), default=None)
    return {
        "clips": selected,
        "max_longest_miss_run": max_miss,
        "threshold": threshold,
        "worst_quantitative_recall": worst_recall,
    }


def center_inside_roi(
    center: tuple[float, float],
    roi: tuple[int, int, int, int],
) -> bool:
    x, y = center
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def has_overlapping_duplicates(detections: list[PlateDetection], threshold: float = 0.5) -> bool:
    for index, left in enumerate(detections):
        for right in detections[index + 1 :]:
            if bbox_iou(left.bbox_xyxy, right.bbox_xyxy) >= threshold:
                return True
    return False


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
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / float(left_area + right_area - intersection)


def longest_run(frames: list[int]) -> int:
    if not frames:
        return 0
    longest = 1
    current = 1
    previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        previous = frame
    return max(longest, current)


def score_distribution(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "max": None, "mean": None, "min": None, "p50": None, "p90": None}
    values = np.array(scores, dtype=np.float32)
    return {
        "count": int(values.size),
        "max": round(float(np.max(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "min": round(float(np.min(values)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p90": round(float(np.percentile(values, 90)), 6),
    }


def rational_to_float(value: str | None) -> float | None:
    if value is None:
        return None
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    denom = float(denominator)
    if denom == 0:
        return None
    return float(numerator) / denom


def validate_clip_hash(benchmark: PlateBenchmark, actual_sha256: str) -> None:
    if benchmark.source_sha256 != actual_sha256:
        raise ValueError(
            f"{benchmark.clip_id} source hash mismatch: expected "
            f"{benchmark.source_sha256}, got {actual_sha256}"
        )


def optional_field(name: str, value: Any) -> dict[str, Any]:
    return {} if value is None else {name: value}


def portable_path(path: Path) -> str:
    return media.portable_source_path(path, Path.cwd())
