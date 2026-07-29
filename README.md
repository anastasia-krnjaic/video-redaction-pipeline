# Video Redaction Pipeline

Video Redaction Pipeline is a small Python project for inspecting local video files and
creating lightweight development artifacts. Phase 1 focuses on metadata inspection,
streaming hashes, short clips, and contact sheets.

Detection, tracking, and redaction are not implemented yet.

## Current Scope

- Inspect container, video, and optional audio metadata with FFprobe.
- Calculate SHA-256 hashes with chunked reads.
- Create short MP4 development clips with FFmpeg.
- Create JPEG contact sheets with FFmpeg filters.
- Provide a small `redactor` command-line interface.

## Architecture

- `src/redactor/cli.py` handles commands and user-facing validation.
- `src/redactor/media.py` handles hashing, probing, trimming, contact sheets, and tool checks.
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

## Data Policy

Source videos belong in `data/raw/` and are not committed. Generated files belong in
`artifacts/` and are not committed. `data/source.yaml` documents the local sample for this
phase but is not parsed by the application.

## Roadmap

- Add detection modules for faces and license plates.
- Add tracking for detections across frames.
- Add analyst decision records.
- Add redaction rendering.
- Add processing manifests.
- Add evaluation outputs for quality checks.
