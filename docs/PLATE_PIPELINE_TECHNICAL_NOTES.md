# Plate Pipeline Technical Notes

The active plate subsystem is currently detector-only.

## Selected Detector

- Detector: Open Image Models YOLOv9-S-608 end-to-end license-plate detector.
- Detector ID: `oim_yolov9_s_608`.
- Source: `ankandrew/open-image-models`.
- Source revision: `f22000e02b30642f317cdba7755c0631638b109e`.
- License finding: MIT repository license; no separate weight-specific notice was found.
- Model file: `yolo-v9-s-608-license-plates-end2end.onnx`.
- Model SHA-256: `2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a`.

The selected configuration is tracked in
`data/evaluations/license_plate_detector_oim_yolov9_s_608_v1/selected_configuration.json`.

## Active Inference Policy

- Inference mode: `uniform_small_object_tiles`.
- Tile size: `1280`.
- Tile overlap: `0.25`.
- Native model input: `608 x 608`.
- Cross-tile NMS IoU: `0.45`.
- Primary threshold: `0.30`.
- Proposal threshold: none selected for this detector-only milestone.
- Configuration hash:
  `7d914438255905e2a249e5bbd30fd6e70eace1e91b57e205bd572d004ffb5a55`.

The same configuration is used for the primary day clip, night clip, close/stress
clip, and all-visible adjudication sample. Do not introduce per-video plate
thresholds, frame ranges, ROIs, coordinates, or clip-specific special cases.

## Supported Command

```bash
redactor detect-plates INPUT_VIDEO \
  --model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir artifacts/diagnostics/plate_detection \
  --overwrite
```

The model path may also come from `REDACTOR_OIM_PLATE_MODEL`. Model weights are
local-only and ignored by Git.

## Output Records

Active plate detections use the repository's normalized detection JSONL frame
records. Each detection has:

- `category: license_plate`;
- `detector_id: oim_yolov9_s_608`;
- source-frame index through the containing frame record;
- score;
- clipped `bbox_xyxy`;
- label;
- source detection index;
- metadata for model revision, inference mode, configuration hash, tiling,
  preprocessing, postprocessing, geometry, and per-detection timing.

Plate detector records are detector evidence only. They are not plate tracks,
privacy-region coverage, reviewed corrections, or final rendering proof.

## Historical Experiments

Previous experiments are archived under
`docs/archive/license_plate_detector_bakeoff/` and
`experiments/archive/license_plate_detector_bakeoff/`.

Rejected models are historical evidence only:

- OpenCV LPD-YuNet;
- OpenVINO vehicle-license-plate-detection-barrier-0106 and 0123;
- Intel YOLOv8n license-plate detector;
- RF-DETR Medium ALPR;
- RT-DETR v2 license-plate detector.

Archived commands are not supported production workflow and must not be imported
by active production code.

## Current Limitations

OIM was selected for a prototype plate pipeline because visual adjudication found
it much cleaner than RT-DETR despite lower recall. Known nighttime gaps remain.
The next milestone is plate tracking with bounded interpolation and review
classification. Final plate redaction is not implemented yet.
