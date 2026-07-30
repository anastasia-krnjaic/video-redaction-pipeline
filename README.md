# Video Redaction Pipeline

Video Redaction Pipeline is a small Python project for inspecting local video files,
creating lightweight development artifacts, and running baseline frame-level detections.

Redaction, identity grouping, rendering policies, manifests, and a UI are not implemented
yet.

## Current Scope

- Inspect container, video, and optional audio metadata with FFprobe.
- Calculate SHA-256 hashes with chunked reads.
- Create short MP4 development clips with FFmpeg.
- Create JPEG contact sheets with FFmpeg filters.
- Run diagnostic face or license-plate detection over every decoded video frame.
- Run an offline face-only temporal-safety layer over frame-indexed detections.
- Provide a small `redactor` command-line interface.

## Architecture

- `src/redactor/cli.py` handles commands and user-facing validation.
- `src/redactor/media.py` handles hashing, probing, trimming, contact sheets, and tool checks.
- `src/redactor/detection.py` handles OpenCV-based diagnostic detection runs.
- `src/redactor/tracking.py` handles face-only temporal-safety diagnostics.
- `src/redactor/models.py` defines metadata dataclasses.

## Installation

Use Python 3.11 and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

FFmpeg and FFprobe must be available on `PATH`:

```bash
redactor doctor
```

## CLI Examples

Inspect a local sample:

```bash
redactor inspect data/raw/bodycam_sample.mp4 --output artifacts/bodycam_sample.metadata.json
```

Create a contact sheet:

```bash
redactor contact-sheet data/raw/bodycam_sample.mp4 artifacts/bodycam_sample_contact_sheet.jpg \
  --interval-seconds 30 \
  --columns 5 \
  --max-frames 10
```

Create a short development clip:

```bash
redactor trim data/raw/bodycam_sample.mp4 artifacts/smoke_clip.mp4 \
  --start 0 \
  --duration 5
```

Run a face-detection diagnostic:

```bash
redactor detect artifacts/eval_day_face.mp4 \
  --category face \
  --model models/face_detection_yunet_2023mar.onnx \
  --threshold 0.60 \
  --output-dir artifacts/baseline/day_face \
  --overwrite
```

Detection outputs include `detections.jsonl`, `summary.json`, and a diagnostic
`overlay.mp4`. The overlay does not include audio.

Run face temporal-safety diagnostics:

```bash
redactor track artifacts/eval_day_face.mp4 \
  --detections artifacts/baseline/day_face_proposals_t045/detections.jsonl \
  --output-dir artifacts/tracking/day_face \
  --overwrite
```

Tracking outputs include `tracks.jsonl`, `summary.json`, `track_quality.json`,
`review_flags.json`, `review_windows.json`, per-track diagnostic contact sheets, and a
diagnostic `overlay.mp4` without audio. This layer classifies face tracklets as
`auto_confirmed`, `review_candidate`, or `rejected`, fills only short reviewable gaps,
and does not claim automatic zero-leak performance.

## Local Models

Model binaries are stored locally in `models/` and are excluded from Git.

- `models/face_detection_yunet_2023mar.onnx`: YuNet face detector from the
  official OpenCV Zoo model repository. The model files in that directory are licensed
  under the MIT License.
- `models/license_plate_detection_lpd_yunet_2023mar.onnx`: LPD-YuNet license-plate
  detector from the official OpenCV Zoo model repository. The model files in that
  directory are licensed under the Apache 2.0 License.

The license-plate model was trained with Chinese license plates, so results on other
plate styles may be limited. The detection command records each local model SHA-256 in
`summary.json`.

## Data Policy

Source videos belong in `data/raw/` and are not committed. Generated files belong in
`artifacts/` and are not committed. `data/source.yaml` documents the local sample for this
phase but is not parsed by the application.

## Roadmap

- Add detection modules for faces and license plates.
- Add analyst decision records.
- Add redaction rendering.
- Add processing manifests.
- Add evaluation outputs for quality checks.
