# Video Redaction Pipeline

Video Redaction Pipeline is a local, source-bound video privacy workflow for detecting, tracking, reviewing, correcting, validating, and publishing redacted media. It handles faces and license plates, keeps the automatic pipeline separate from human review, and publishes immutable Final media only after the exact current reviewed video has been watched and approved.

The repository is designed for local operation. Source videos, model weights, review sessions, Final media, and delivery packages live in ignored artifact roots unless a release process copies them elsewhere.

## What It Does

The pipeline:

- detects privacy-sensitive faces and license plates;
- tracks candidate objects through source-frame intervals;
- renders Automatic review media from accepted machine regions;
- presents source-bound Guidance, Entities, and Manual correction tools;
- compiles immutable Latest reviewed revisions from the authoritative review session;
- requires a full-video approval bound to the exact current Latest reviewed revision;
- publishes immutable Final media and delivery packages;
- preserves audit, lineage, hash, validation, and review provenance.

## Core Capabilities

- Face detection and tracking with the supported ADAS face model path.
- License-plate detection and tracking with the selected plate detector path.
- Shared privacy rendering for face and plate regions.
- Case-scoped human review with append-only event history.
- Manual correction preview and save for single-frame, fixed-interval, and interpolated regions.
- Entity policy actions for source-backed people and plates.
- Source-bound review state, reviewed-region hashes, and revision-bound media.
- Immutable Latest reviewed and Final revisions.
- Full-video approval as the final human visual gate before Final publication.
- Delivery packages with portable manifests and checksums.
- Frozen detector, automatic, and reviewed-final evaluation contracts.
- Production safeguards for single-writer sessions, atomic JSON/event state, no-op materialization, caching, resource admission, resumable stages, and telemetry.

## Safety / Trust Model

The release workflow is:

```text
Automatic -> Guidance and corrections -> Latest reviewed -> full-video sweep -> Final
```

Automatic media is a non-final review input. Guidance items are visible, auditable hotspots that a reviewer may inspect or adjudicate, but unreviewed Guidance alone does not block Latest reviewed or Finalization. The authoritative full-video sweep covers any Guidance left unreviewed without fabricating individual Correct decisions.

Needs correction is fail-closed. A task explicitly marked Needs correction blocks reviewed-video publication and Finalization until a valid correction is saved and verified, or until the reviewer changes the adjudication to Correct or Not sensitive.

Finalization requires all of the following to match the current case and session:

- current validated Latest reviewed revision;
- current reviewed-region-set identity;
- current full-video sweep acknowledgement bound to that revision and media;
- no active Needs correction or invalid correction state;
- source, lineage, media, region, and pointer integrity gates.

Review sessions are append-only and hash-chained. The workspace enforces a single writable server per workspace, writes JSON state atomically, and keeps historical Latest, sweep, and Final artifacts immutable but non-current when superseded.

## Quick Start

Use Python 3.11 or 3.12, FFmpeg, FFprobe, Node.js, and npm.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
redactor doctor
redactor --help
```

Install and build the frontend:

```bash
cd frontend
npm ci
npm run build
npm test -- --run
cd ..
```

Run the Python test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -rs -p no:cacheprovider --basetemp=/tmp/video_redaction_pytest
```

Run a supported automatic face and plate preview on a local input:

```bash
redactor redact-video path/to/input.mp4 \
  --categories face,license_plate \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --plate-model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir artifacts/cases/example_automatic_preview \
  --overwrite
```

Serve a generated review workspace:

```bash
redactor serve-final-review-workspace artifacts/cases/example_automatic_preview/review_workspace --open
```

Release evidence artifacts are intentionally ignored by Git. A clean source checkout can install, build, test, and generate new workspaces, but it does not include private source media, model weights, review sessions, Finals, or delivery zips.

## Review Workspace

The Final Review Workspace is a local browser app served by the Python backend:

```bash
redactor serve-final-review-workspace artifacts/reviewed_final_combined_v2_1/run_20260813T210250Z/review_workspace --open
```

Use this path only when the local release artifact root exists. For a clean clone, generate a workspace with `redactor redact-video` first.

The workspace presents Day and Night cases with these media roles when available:

- Original
- Automatic
- Latest reviewed
- Final

The reviewer workflow is:

- inspect Guidance and Entities;
- use Manual redaction only for real correction needs;
- save corrections after Preview confirms rendered privacy pixels;
- build or select the current Latest reviewed revision;
- watch the complete current Latest reviewed video;
- approve the full-video review;
- create Final.

There is no Restart review button in the frozen workflow. New review session and explicit Resume review remain available, and Reset this case to Automatic baseline remains the deliberate case reset action.

More detail lives in [docs/FINAL_REVIEW_WORKSPACE.md](docs/FINAL_REVIEW_WORKSPACE.md).

## Validation & Evaluation

Evaluation reports intentionally keep claim layers separate:

- detector recall measures raw selected-detector proposal matching;
- automatic eligible coverage measures machine-generated privacy regions;
- reviewed-final geometric coverage measures authoritative reviewed regions against frozen Ground Truth windows;
- human full-video sweep is evidence that the current Latest reviewed media was watched and approved, not a numerical metric.

Frozen plate detector recall:

- Day: 236 / 240 = 0.983333
- Night: 321 / 360 = 0.891667
- Aggregate: 557 / 600 = 0.928333

Frozen automatic eligible coverage:

- Day: 240 / 240 = 1.0
- Night: 0 / 360 = 0.0
- Aggregate: 240 / 600 = 0.4

Reviewed-final disposition:

- Day reviewed-final plate score is unavailable because the frozen benchmark source is not comparable to the product Day Final source.
- Night reviewed-final strict geometric coverage is 284 / 360 = 0.788889.
- The 76 Night strict geometric misses were visually audited: no true visible plate leaks and no plate-present-undercovered frames were found.
- Face reviewed-final numerical scoring is not supported by the frozen face evaluation contract.

See [docs/EVALUATION.md](docs/EVALUATION.md).

## Production Readiness

The frozen production-readiness pass added or verified:

- single-writer review workspace locking;
- atomic event, pointer, and JSON state handling;
- event projection caching;
- immutable hash and FFprobe metadata caching;
- reviewed-region no-op reuse;
- no-op and coalesced materialization;
- resource-aware heavy-job admission control;
- resumable stage manifests;
- structured telemetry;
- streaming review-bundle validation.

Audited finalization-status latency is about 0.2 seconds on warm paths, with cold Day requests about 1.6-2.7 seconds and cold Night requests about 1.0-1.2 seconds on the release machine. Synthetic review-bundle validation peak RSS dropped from 813.31 MB to 202.06 MB. Detector inference remains the major heavy workload and was not semantically optimized in the final production pass.

See [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

## Repository Structure

- `src/redactor/`: Python CLI, pipeline, review workspace, evaluation, validation, and production hardening code.
- `frontend/`: React/Vite review workspace UI and frontend tests.
- `tests/`: Python unit, integration, and Firefox workspace tests.
- `data/ground_truth/`: frozen machine-readable Ground Truth inputs.
- `data/evaluations/`: frozen evaluation policies and adjudication inputs.
- `docs/`: current product, release, evaluation, and technical documentation.
- `artifacts/`: ignored local outputs, release evidence, media, sessions, and delivery packages.

## Testing

```bash
npm --prefix frontend run build
npm --prefix frontend test -- --run
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -rs -p no:cacheprovider --basetemp=/tmp/video_redaction_pytest
REDACTOR_RUN_FIREFOX_E2E=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -rs -p no:cacheprovider --basetemp=/tmp/video_redaction_firefox tests/test_final_review_workspace_firefox.py
git diff --check
```

The release baseline is frontend build pass, 46 frontend tests passed, Ruff pass, 743 Python tests passed with 1 skipped, and 1 explicit Firefox test passed.

## Release Notes And Evidence

- [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)
- [docs/RELEASE_EVIDENCE.md](docs/RELEASE_EVIDENCE.md)
- [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)

## Deployment Notes

The project is a local processing and review tool. Remote or multi-user deployment requires a separate deployment hardening pass for authentication, authorization, transport security, data retention, backup/restore, access logging, quota handling, and privacy review.
