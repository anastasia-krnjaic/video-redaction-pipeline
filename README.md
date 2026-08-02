# Video Redaction Pipeline

Video Redaction Pipeline is a local Python project for building and auditing a face-redaction workflow on short evaluation clips. The current Face Pipeline v1 uses one active face detector: OpenVINO `face-detection-adas-0001` (`openvino_face_adas_0001`).

The pipeline can inspect media, run ADAS face detection, connect detections into temporal face tracks, recover or flag privacy gaps after tracking, and render face-redacted preview or approved final MP4s from accepted privacy regions. A full review UI, automated benchmark runners, and license-plate work are not part of the current supported workflow.

## Install

Use Python 3.11 or 3.12 and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

FFmpeg and FFprobe must be available on `PATH`:

```bash
redactor doctor
```

OpenVINO is required for the active face detector. The ADAS model XML/BIN files are local model inputs and are not tracked in Git. By default the CLI looks for:

```text
models/face-detection-adas-0001/face-detection-adas-0001.xml
models/face-detection-adas-0001/face-detection-adas-0001.bin
```

You can also pass `--adas-xml` and `--adas-bin`, or set `REDACTOR_ADAS_XML` and `REDACTOR_ADAS_BIN`.

## Supported Face Workflow

Run the current face pipeline:

```bash
redactor face-pipeline artifacts/eval_day_face.mp4 \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --output-dir artifacts/face_pipeline/day_face \
  --overwrite
```

The command runs:

1. ADAS face detection at proposal threshold `0.20`.
2. Ordinary face tracking with primary observation threshold `0.30`.
3. Optional privacy gap recovery when target track IDs are provided.

For privacy gap recovery on a known target track, add target and safety-track IDs:

```bash
redactor face-pipeline artifacts/eval_night_selective.mp4 \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --target-track-id 8 \
  --keep-track-id 10 \
  --competing-track-id 10 \
  --frame-start 420 \
  --frame-end 779 \
  --output-dir artifacts/face_pipeline/night_selective \
  --overwrite
```

Privacy recovery writes machine-readable interval decisions. A gap may be accepted automatically, accepted after configured refinement, marked `needs_review`, or rejected. Synthetic privacy regions never count as raw detector hits.

Render a face-redacted preview or approved final MP4 from the privacy regions:

```bash
redactor render-faces artifacts/eval_night_selective.mp4 \
  --privacy-regions artifacts/face_pipeline/night_selective/privacy_gap_recovery/privacy_regions.jsonl \
  --interval-decisions artifacts/face_pipeline/night_selective/privacy_gap_recovery/interval_decisions.json \
  --output-dir artifacts/face_pipeline/night_selective/render_preview \
  --mode preview \
  --method gaussian_blur \
  --margin-fraction 0.15 \
  --comparison-frames 120 \
  --overwrite
```

Preview mode may render accepted regions even when some intervals still need review, and the output is explicitly marked non-final. Final mode refuses export when relevant intervals are `needs_review` or `rejected`, when privacy-required frames remain unresolved, or when invalid boxes are present. Preserved reviewed corrections can be rendered in final mode while keeping their provenance.

## Output Layout

A face-pipeline run writes subdirectories under the output directory:

| Directory | Contents |
| --- | --- |
| `detection/` | `detections.jsonl`, `summary.json`, optional `overlay.mp4`. |
| `tracking/` | `tracks.jsonl`, `summary.json`, track quality, review flags/windows, contact sheets, and overlay. |
| `privacy_gap_recovery/` | `privacy_regions.jsonl`, `interval_decisions.json`, `metrics.json`, `summary.json`, optional overlay. Written only when target track IDs are provided. |
| `render/` or chosen render directory | Redacted MP4, `render_manifest.json`, `render_summary.json`, `validation.json`, `processing_log.jsonl`, and optional comparison clip. |

Generated artifacts belong under `artifacts/` and are ignored by Git.

## Advanced Diagnostics

Lower-level commands remain available for focused debugging:

```bash
redactor inspect data/raw/bodycam_sample.mp4 --output artifacts/bodycam_sample.metadata.json
redactor contact-sheet data/raw/bodycam_sample.mp4 artifacts/bodycam_sample_contact_sheet.jpg --interval-seconds 30 --columns 5 --max-frames 10
redactor trim data/raw/bodycam_sample.mp4 artifacts/smoke_clip.mp4 --start 0 --duration 5
redactor detect artifacts/eval_day_face.mp4 --category face --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin --output-dir artifacts/diagnostics/day_face --overwrite
redactor track artifacts/eval_day_face.mp4 --detections artifacts/diagnostics/day_face/detections.jsonl --output-dir artifacts/diagnostics/day_face_tracking --overwrite
redactor recover-face-gaps artifacts/eval_night_selective.mp4 --tracks artifacts/diagnostics/night_tracking/tracks.jsonl --target-track-id 8 --keep-track-id 10 --output-dir artifacts/diagnostics/night_gap_recovery --overwrite
redactor render-faces artifacts/eval_night_selective.mp4 --privacy-regions artifacts/diagnostics/night_gap_recovery/privacy_regions.jsonl --interval-decisions artifacts/diagnostics/night_gap_recovery/interval_decisions.json --output-dir artifacts/diagnostics/night_render_preview --mode preview --overwrite
```

These commands are useful during development, but `face-pipeline` is the supported entry point for the current face workflow.

## Current Evidence And History

The selected detector evidence record lives in `data/evaluations/face_detector_adas_v1.0/`. It records the ADAS-only corrected adjudication, raw-detector metrics, selected thresholds, and FPS provenance audit.

The earlier three-model face detector comparison is preserved as history under `docs/archive/face_detector_bakeoff/`. OpenCV YuNet and OpenVINO `face-detection-0205` are not active face detector backends and are not selectable in the supported face pipeline.

Detailed architecture notes live in `docs/FACE_PIPELINE_TECHNICAL_NOTES.md`. The broader system explanation lives in `docs/SYSTEM_OVERVIEW.md`.

## Current Limitations

- Final face rendering is supported only from accepted privacy-region inputs and remains bounded by the quality of the upstream tracking, recovery, and review evidence.
- Face identity grouping and review decisions are still diagnostic inputs, not a complete analyst workflow.
- Privacy gap recovery can reduce temporal gaps, but uncertain intervals must remain review-required.
- Raw detector recall, tracking coverage, privacy recovery, and final redaction quality are separate claims.
- No universal or production-wide zero-leak claim has been established.
- License-plate detection and plate tracking are intentionally outside the current Face Pipeline v1 cleanup.
