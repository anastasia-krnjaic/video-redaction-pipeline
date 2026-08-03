# Video Redaction Pipeline - System Overview

- **Document version:** v010
- **Last updated:** 2026-08-03
- **System status:** Face Pipeline v1 is integrated for ADAS face detection, ordinary tracking, privacy gap recovery, face redaction rendering from accepted privacy regions, automated evaluation against machine-readable face ground truth, and local automated review bundles. License-plate work is not part of the current completed pipeline.
- **Canonical document:** `docs/SYSTEM_OVERVIEW.md`

## What The System Is For

This repository builds a local video-redaction pipeline. Its current focus is face privacy: finding faces, keeping those detections connected over time, rendering accepted privacy regions, measuring the result against available truth, and creating a compact review package for intervals that deserve human attention.

The system is not a finished automatic redaction product, and it does not make a universal zero-leak claim. It produces structured evidence, diagnostic outputs, non-final previews, and approved face-redacted MP4s when the available privacy-region evidence passes final safety gates.

## Current User Workflow

A user provides a local video clip and local ADAS model files. The supported face workflow is:

1. Run the face pipeline command.
2. Inspect detection, tracking, and privacy-gap artifacts.
3. Review intervals marked `needs_review` or `rejected`.
4. Preserve accepted corrections so they can be reused instead of regenerated.
5. Render a non-final preview or an approved final face-redacted MP4 from accepted privacy regions.
6. Evaluate detector, automatic, or reviewed-final results against machine-readable face ground truth.
7. Build a local review bundle that gathers risk intervals, visual evidence, and deterministic audit samples.

The main command is:

```bash
redactor face-pipeline INPUT_VIDEO \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --output-dir artifacts/face_pipeline/RUN_NAME \
  --overwrite
```

Privacy gap recovery runs when target track IDs are supplied.

Rendering is a separate explicit step:

```bash
redactor render-faces INPUT_VIDEO \
  --privacy-regions artifacts/face_pipeline/RUN_NAME/privacy_gap_recovery/privacy_regions.jsonl \
  --interval-decisions artifacts/face_pipeline/RUN_NAME/privacy_gap_recovery/interval_decisions.json \
  --output-dir artifacts/face_pipeline/RUN_NAME/render_preview \
  --mode preview
```

Evaluation is also explicit:

```bash
redactor evaluate-faces \
  --ground-truth data/ground_truth/redaction_ground_truth_v1.0.json \
  --clip-id day_face \
  --mode reviewed_final \
  --source-video artifacts/eval_day_face.mp4 \
  --detections artifacts/adas_detector_integration_parity_v1/day_integrated.detections.jsonl \
  --adjudication data/evaluations/face_detector_adas_v1.0/adjudication.completed.corrected.json \
  --tracks artifacts/adas_integrated_tracking_validation_v1/runs/current_cli_default/day/tracks.jsonl \
  --privacy-regions artifacts/face_renderer_v1/day_face/recovery/privacy_regions.jsonl \
  --interval-decisions artifacts/face_renderer_v1/day_face/recovery/interval_decisions.json \
  --rendered-video artifacts/face_renderer_v1/day_face/render_final/redacted_final.mp4 \
  --render-manifest artifacts/face_renderer_v1/day_face/render_final/render_manifest.json \
  --render-validation artifacts/face_renderer_v1/day_face/render_final/validation.json \
  --output-dir artifacts/face_pipeline_evaluation_v1/day_reviewed_final
```

The review bundle command turns pipeline artifacts into a local HTML package:

```bash
redactor build-review-bundle \
  --source-video artifacts/eval_night_selective.mp4 \
  --case-id night_automatic_review_v1 \
  --privacy-regions artifacts/face_renderer_v1/night_auto_preview/recovery/privacy_regions.jsonl \
  --interval-decisions artifacts/face_renderer_v1/night_auto_preview/recovery/interval_decisions.json \
  --render-manifest artifacts/face_renderer_v1/night_auto_preview/render_preview/render_manifest.json \
  --automatic-preview-video artifacts/face_renderer_v1/night_auto_preview/render_preview/redacted_preview.mp4 \
  --evaluation-dir artifacts/face_pipeline_evaluation_v1/night_automatic \
  --output-dir artifacts/review/night_automatic
```

## How The Pipeline Works

```mermaid
flowchart LR
  A["Input video"] --> B["ADAS face detection"]
  B --> C["Ordinary tracking"]
  C --> D["Privacy gap recovery"]
  D --> E["Safety checks"]
  E --> F{"Confident?"}
  F -->|Yes| G["Privacy regions and audit data"]
  F -->|No| H["Needs review or rejected interval"]
  H --> I["Preserved correction after review"]
  I --> G
  G --> J["Face rendering"]
  J --> K["Preview or approved final MP4"]
  K --> L["Automated evaluation report"]
  L --> M["Local review bundle"]
```

The stages are:

- **Face detection:** finds face proposals on individual frames.
- **Ordinary tracking:** connects proposals that appear to be the same face over time.
- **Privacy gap recovery:** tries to cover short or recoverable gaps in a redact target's track.
- **Safety checks:** prevent uncertain boxes from being treated as reliable.
- **Review preservation:** keeps accepted corrections so future runs do not have to rediscover them.
- **Face rendering:** applies blur or pixelation to accepted privacy regions on the original-resolution video.
- **Evaluation:** compares detector observations, tracking, privacy regions, preserved corrections, and render manifests against the selected machine-readable face ground truth.
- **Review bundle:** gathers risk items, benchmark-only failures, random audit samples, clips, contact sheets, and reviewer decision export into a local folder.

## Why Detection Is Not Enough

A detector sees one frame at a time. It may temporarily lose a face when the person turns, lighting changes, the camera moves, the face is occluded, or the person is partly outside the frame.

For privacy work, a single missed frame can matter. That is why the pipeline separates raw detector observations from tracking and privacy recovery. A generated privacy box can help maintain coverage, but it is not counted as a detector hit.

## How Uncertain Intervals Are Handled

When the system sees a gap, it chooses the safest available recovery strategy:

- interpolate a short gap between reliable boxes;
- track forward and backward from reliable boxes on both sides;
- track from one side only when a gap touches a clip edge;
- preserve a previously accepted correction;
- mark the interval for review when evidence is not strong enough.

The system records an interval decision:

- `accepted_automatically`
- `accepted_after_refinement`
- `needs_review`
- `rejected`

The safer failure mode is review, not invented coverage. A gap that crosses a scene cut, jumps too far, changes size too much, overlaps a keep-person track, or moves near a competing identity should not be silently accepted.

Preview rendering can still be useful when review-required intervals exist, but it must remain non-final. Final rendering refuses export when relevant intervals need review, are rejected, contain unresolved privacy-required frames, or contain invalid boxes.

## Human Review And Preserved Corrections

Human review is part of the intended workflow. A reviewer can accept or correct privacy regions for difficult intervals. Once accepted, those regions can be supplied back to the pipeline as preserved regions.

Preserved corrections remain distinct from detector output and ordinary tracking output. This matters because each kind of evidence supports a different claim:

- a detector observation says the model saw a face proposal;
- ordinary tracking says proposals were linked over time;
- privacy recovery says a synthetic privacy region was generated or preserved;
- review says a human accepted or corrected an uncertain interval.
- rendering says accepted privacy regions were applied to video frames. It does not change the type of evidence that created each region.

## Automated Evaluation

The selected machine-readable ground truth is `data/ground_truth/redaction_ground_truth_v1.0.json`. It converts the frozen face portions of GroundTruth v1.1 into structured ranges without inventing tight per-frame boxes.

The evaluator has three modes:

- `detector`: raw ADAS detector observations only;
- `automatic`: detector, ordinary tracking, and automatically accepted privacy recovery, excluding preserved or manual reviewed corrections;
- `reviewed_final`: includes explicitly reviewed or preserved corrections and can validate the final rendered video manifest and media checks.

The evaluator keeps these claims separate. A reviewed correction can complete a final privacy-region result, but it does not improve raw detector recall. A renderer approval says the renderer accepted and applied its input regions; it is not, by itself, proof that the regions match ground truth.

## Automated Review Bundles

After processing or evaluation, the system can build a local review bundle. The bundle is meant to save reviewer time: it selects short intervals that look risky, creates contact sheets and short clips, and presents them in `index.html`.

Review items come from three sources:

- **Pipeline-derived risk:** review-required or rejected intervals, unresolved frames, invalid boxes, suspicious provenance changes, tracking fragmentation, render or media validation failures, and similar signals.
- **Ground-Truth-derived benchmark evidence:** known detector misses, uncovered privacy-required frames, or keep-person false redaction when machine-readable Ground Truth and evaluator outputs are available.
- **Random audit samples:** deterministic samples from apparently normal intervals, used to reduce the chance that silent problems are missed.

The review bundle never proves that no face was missed. On a new production video, Ground Truth usually does not exist, so the bundle can only show risks the pipeline can infer plus random audit samples.

The HTML page stores temporary reviewer choices in browser-local state and provides an explicit export button for a decisions JSON file. Exported decisions can be validated against the original queue, but they do not directly rewrite privacy regions or override renderer/evaluator approval. If an interval still lacks valid reviewed privacy regions, final export remains blocked.

## Current Face-Detector Decision

The active face detector is OpenVINO `face-detection-adas-0001`, recorded in outputs as `openvino_face_adas_0001`.

The selected detector policy is:

- proposal/storage floor: `0.20`;
- primary observation threshold: `0.30`;
- weak proposals from `0.20` through below `0.30` may support tracking but must not independently confirm a track.

The decision is backed by the tracked ADAS evidence record in `data/evaluations/face_detector_adas_v1.0/`.

OpenCV YuNet and OpenVINO `face-detection-0205` were historical comparison models. They are not active face detector backends and are not selectable in the supported face pipeline. The comparison history is preserved in `docs/archive/face_detector_bakeoff/`.

## Outputs And Audit Information

A face-pipeline run writes ignored artifacts under the selected output directory:

- `detection/`: frame-level ADAS detections, model hashes, timing, and optional overlay;
- `tracking/`: tracks, track quality, review flags, review windows, contact sheets, and overlay;
- `privacy_gap_recovery/`: privacy regions, interval decisions, metrics, source hashes, and optional overlay.
- render directory chosen by the user: face-redacted MP4, render manifest, summary, validation, processing log, and optional diagnostic comparison clip.
- evaluation directory chosen by the user: metrics, object metrics, interval decisions, failures, render-validation summary, report, summary table, and concise visual evidence for failures.
- review bundle directory chosen by the user: local HTML, review queue and summary JSON, local CSS/JS, short clips, contact sheets, checklist, and review-decision template/export files.

The render manifest records the source video hash, privacy-region input hash, rendering mode, redaction method, provenance counts, unresolved intervals, review status, output video hash, audio handling, and final approval status.

## Current Status

Implemented:

- media inspection, hashing, trimming, and contact sheets;
- ADAS face detection as the only active face detector;
- face tracking with weak and primary observations;
- privacy gap recovery with interval decisions and safety checks;
- preserved privacy-region support;
- face redaction rendering with preview and final approval modes;
- machine-readable face ground truth derived from frozen GroundTruth v1.1;
- automated face evaluation in detector, automatic, and reviewed-final modes;
- automated local review bundles with risk queues, concise visual evidence, deterministic audit samples, and exported-decision validation;
- FFprobe validation of rendered media resolution, frame rate, duration, frame count, and audio presence;
- tracked ADAS detector evaluation evidence;
- historical archive for the detector bake-off.

Not yet implemented:

- correction UI for drawing or editing privacy regions;
- license-plate pipeline work.

## Known Limitations

- The system is not fully automatic.
- It does not guarantee perfect face coverage.
- Raw detector recall is not the same as tracking coverage or privacy-region coverage.
- Machine-readable face ground truth is range-level for the evaluated clips; it does not provide tight per-frame boxes for IoU scoring.
- Automated evaluation does not perform pixel-level rendered privacy verification.
- Automated review queues cannot detect every possible silent missed face, especially without Ground Truth.
- Rendered face video quality is only as good as the accepted privacy-region evidence supplied to the renderer.
- Automatically recovered privacy regions must still be treated according to their interval decision.
- Universal or production-wide zero-leak claims have not been established.
- The evaluated detector decision applies to the current operating clips and evidence record; it is not a universal model ranking.

## Next Planned Stages

1. Add a correction workflow that can supply reviewed privacy regions for review-required intervals.
2. Expand validation of rendered face-only outputs before making broader redaction-quality claims.
3. Freeze the evaluated Face Pipeline v1 state before adding new object classes.
4. Only after the face pipeline is coherent, resume separate license-plate detector work.

## Detailed Technical Notes

Implementation details, JSON output conventions, provenance fields, thresholds, safety gates, and diagnostic command notes live in `docs/FACE_PIPELINE_TECHNICAL_NOTES.md`.
