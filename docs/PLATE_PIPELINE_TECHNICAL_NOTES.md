# Plate Pipeline Technical Notes

The active plate subsystem currently has one selected detector plus a minimal
temporal stabilization and bounded gap-recovery stage. Plate review integration
and final plate rendering are not implemented yet.

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

## Supported Commands

```bash
redactor detect-plates INPUT_VIDEO \
  --model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir artifacts/diagnostics/plate_detection \
  --overwrite
```

The model path may also come from `REDACTOR_OIM_PLATE_MODEL`. Model weights are
local-only and ignored by Git.

After detection, run the plate tracking prototype:

```bash
redactor track-plates \
  --source-video INPUT_VIDEO \
  --detections artifacts/diagnostics/plate_detection/detections.jsonl \
  --output-dir artifacts/diagnostics/plate_tracking \
  --overwrite
```

The command writes `tracks.jsonl`, `track_summary.json`, `gap_decisions.json`,
`review_intervals.json`, `tracking_manifest.json`, `processing_log.txt`, a local
track gallery, contact sheets, and an H.264/yuv420p tracking overlay.

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

## Plate Tracking v1

The tracker is a deterministic post-detector stage for OIM plate detections. It
uses one global configuration across the primary day clip, night clip,
close/stress clip, and synthetic fixtures.

Selected tracking configuration hash:
`27fc564737bf7bf3d2bc5389c4394fe991b5b9e90f9e76789ed0d8231e3dff9d`.

Core policy:

- maximum association gap: `1.50` seconds;
- maximum safe interpolation gap: `0.30` seconds;
- primary detector threshold remains `0.30`;
- no one-sided prediction before the first observation or after the last
  observation;
- interpolation requires reliable observations on both sides, bounded center
  motion, bounded scale change, and no scene cut;
- long or unsafe gaps become review intervals instead of synthetic boxes.

Track statuses are:

- `accepted_plate_track`: stable, persistent plate evidence;
- `needs_review`: persistent but ambiguous, unstable, sparse, or internally
  unresolved evidence;
- `rejected_noise_track`: isolated or invalid plate-like proposals.

Frame-level provenance is preserved as:

- `plate_detector_observation`;
- `plate_track_interpolation`;
- `plate_track_prediction` (reserved, not emitted by v1);
- `reviewed_plate_correction` (reserved for a later review/correction stage).

Interpolation and future reviewed corrections must never be counted as raw
detector hits. Tracking metrics therefore report detector recall, observed track
coverage, interpolated coverage, unresolved gaps, false track burden, and review
intervals separately.

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
it much cleaner than RT-DETR despite lower recall. Plate tracking now stabilizes
short gaps and rejects isolated noise, but known nighttime gaps remain review
items. The next milestone is experimental plate rendering from accepted tracks
and explicitly reviewed intervals. Final unified plate redaction is not
implemented yet.
