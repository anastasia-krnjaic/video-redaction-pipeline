# Plate Pipeline Technical Notes

The active plate subsystem currently has one selected detector, a minimal
temporal stabilization and bounded gap-recovery stage, and a non-final automatic
plate preview renderer with a local review bundle. Reviewed-final plate
completion and unified face/plate rendering are not implemented yet.

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

`detect-plates` processes every source frame by default. `--max-frames` is a
diagnostic limiter and must not be confused with a supported production span.
Benchmark frame windows are evaluation-only; they do not define which frames get
plate detections, tracks, or preview redaction.

After detection, run the plate tracking prototype:

```bash
redactor track-plates \
  --source-video INPUT_VIDEO \
  --detections artifacts/diagnostics/plate_detection/detections.jsonl \
  --output-dir artifacts/diagnostics/plate_tracking \
  --overwrite
```

The command writes `tracks.jsonl`, `track_summary.json`, `gap_decisions.json`,
`review_intervals.json`, `tracking_manifest.json`, `frame_span_report.json`,
`processing_log.txt`, a local track gallery, contact sheets, and an H.264/yuv420p
tracking overlay.

Render the automatic plate preview from accepted tracks only:

```bash
redactor render-plates \
  --source-video INPUT_VIDEO \
  --tracks artifacts/diagnostics/plate_tracking/tracks.jsonl \
  --review-intervals artifacts/diagnostics/plate_tracking/review_intervals.json \
  --output-dir artifacts/diagnostics/plate_render \
  --overwrite
```

Rendering validates the source span before publishing a normal preview. If the
track JSONL does not contain one record for every source frame, normal preview
and reviewed-final rendering are blocked. A deliberately partial diagnostic
preview may be requested with `--allow-partial-preview`; that output is
watermarked and recorded as `PARTIAL PROCESSING - NOT FINAL`.

Build the local plate review bundle:

```bash
redactor build-plate-review-bundle \
  --source-video INPUT_VIDEO \
  --tracks artifacts/diagnostics/plate_tracking/tracks.jsonl \
  --review-intervals artifacts/diagnostics/plate_tracking/review_intervals.json \
  --automatic-preview artifacts/diagnostics/plate_render/automatic_preview.mp4 \
  --diagnostic-overlay artifacts/diagnostics/plate_tracking/tracking_overlay.mp4 \
  --output-dir artifacts/diagnostics/plate_review \
  --overwrite
```

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

## Frame-Span Safety

Plate processing records keep source-video frame indices authoritative. The
shared span report records:

- `source_frame_count`;
- `processed_frame_count`;
- `first_processed_frame`;
- `last_processed_frame`;
- `missing_processed_ranges`;
- `full_source_processed`.

`track-plates` preserves the detector frame span and writes the same accounting
to `frame_span_report.json`. `render-plates` consumes the track span and refuses
to present partial tracking as a normal full-video preview or reviewed-final
candidate. This is separate from benchmark scoring: a benchmark interval can be
evaluated without implying that the rest of the source video was processed.

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

## Plate Rendering And Review v1

Plate Rendering v1 consumes `tracks.jsonl` and `review_intervals.json`; it does
not change detector or tracker outputs. The clean automatic preview may render
only:

- accepted plate detector observations;
- accepted plate-track observations;
- accepted bounded interpolation.

It must not render rejected tracks, `needs_review` tracks, unresolved gaps, or
speculative one-sided prediction as if they were approved regions.

Selected rendering configuration:

- method: `pixelate`;
- margin fraction: `0.15`;
- margin pixels: `0`;
- pixel size: `16`;
- video codec: H.264;
- pixel format: `yuv420p`;
- MP4 faststart enabled;
- configuration hash:
  `508146780991e009b1d0683084236d5dce00e4547bd21e1ea7286c6f972ec02d`.

The margin was selected from a small global sweep over `0.10`, `0.15`, and
`0.20`. It covers tight accepted OIM boxes in reviewed evidence without changing
detector or tracker coordinates and without using per-video tuning.

Rendered-region validation decodes the output video and checks that expected
plate boxes changed inside the expanded bbox. FFprobe metadata alone is not
treated as proof that redaction was applied.

The plate review bundle creates local, relative-path HTML evidence for
unresolved gaps, uncertain tracks, short accepted tracks that may be false
positives, track-boundary audit records, and deterministic spot checks. A long
unresolved gap does not offer a `Looks correct` action unless a later correction
workflow supplies a valid privacy region. Review decisions do not rewrite plate
tracks or override renderer validation.

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
short gaps and rejects isolated noise, and Plate Rendering v1 creates clean
automatic previews from accepted evidence. Known nighttime gaps remain review
items. Reviewed correction intake and final unified face/plate redaction are not
implemented yet.
