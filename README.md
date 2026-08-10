# Video Redaction Pipeline

Video Redaction Pipeline is a local Python project for building and auditing video-redaction workflows on short evaluation clips. The current Face Pipeline v1 uses one active face detector: OpenVINO `face-detection-adas-0001` (`openvino_face_adas_0001`). The prototype plate work uses one selected detector, Open Image Models YOLOv9-S-608 (`oim_yolov9_s_608`), followed by bounded temporal plate tracking and automatic preview rendering from accepted tracks.

The face pipeline can inspect media, run ADAS face detection, connect detections into temporal face tracks, recover or flag privacy gaps after tracking, render face-redacted preview or approved final MP4s from accepted privacy regions, evaluate those outputs against machine-readable face ground truth, and build compact local review bundles. Plate support currently covers detection, shared accumulated-evidence track confirmation, short defensible interpolation, non-final automatic plate previews, and review tasks for unresolved or uncertain plate intervals. The Final Review Workspace v2.3.4 backend is frozen for review behavior, and Frontend Foundation v1 adds a React + TypeScript review app with the same Python-owned sessions, entities, corrections, audit trail, media serving, source-frame mapping, and finalization gates.

Faces and plates keep separate detector adapters and thresholds, but the post-detection foundation now shares the record contracts for observations, temporal regions, gap decisions, privacy regions, frame spans, rendering, and review tasks. The architecture decision is documented in `docs/PRIVACY_PIPELINE_ARCHITECTURE.md`.

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

The active plate detector uses an ONNX model from Open Image Models. Model
weights are local inputs and are not tracked in Git. Pass the model path with
`--model` or set `REDACTOR_OIM_PLATE_MODEL`.

## Final Review React Frontend

The frozen backend reference package is:

```text
artifacts/final_pipeline_package_v2_3_4/
```

Generate the current React frontend package without changing backend/privacy artifacts:

```bash
cd frontend
npm ci
npm run build
cd ..
PYTHONPATH=src .venv/bin/python -m redactor.cli build-final-review-workspace \
  --output-dir artifacts/final_pipeline_package_frontend_v1_5 \
  --frontend react \
  --react-reference-package artifacts/final_pipeline_package_frontend_v1_4_1 \
  --skip-frontend-build \
  --overwrite
```

Serve it locally:

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli serve-final-review-workspace \
  artifacts/final_pipeline_package_frontend_v1_5 --open
```

React assets are served from `/app/`; the legacy generated workspace remains available at `/legacy.html`. The stable API contract is checked in at `docs/api/final_review_workspace_v1.schema.json` and served at `/api/v1/schema`.

Frontend v1.4.1 fixes manual-redaction Save persistence so the exact previewed draft becomes active working state and creates a correction-verification task. Canonical case-generated workspaces now publish Latest reviewed revisions through the reviewed-region compiler after required blocking review tasks are resolved. Correction preview media remains diagnostic and is never a reviewed output. Latest reviewed is selected only through an immutable revision manifest referenced by the atomic `latest_reviewed/current.json` pointer; generated correction IDs, revision IDs, timestamps, preview artifacts, sweep acknowledgements, and finalization bookkeeping stay in audit records but are excluded from the current `active_projection.v2` media projection. Pointers created before the schema field existed validate only through explicit `active_projection.v1` compatibility when media-affecting review state is unchanged; unknown schemas fail closed. Finalization is now an explicit canonical approval gate: a revision-bound full-video sweep acknowledgement, source/config lineage checks, media and region validation artifacts, and correction/task gates must all match the current Latest reviewed revision before `final/current.json` is published. Final is the exact approved Latest reviewed MP4 bytes, materialized by hardlink or verified copy, not a rerender. Segment-level reviewed-media reuse is deferred until parity can be proven. Evidence lives under `artifacts/manual_redaction_save_fix_v1_4_1/`, `artifacts/incremental_materialization_v1_5/`, `artifacts/mixed_latest_reviewed_v1/`, `artifacts/mixed_latest_reviewed_v1_final_stabilization/`, and `artifacts/mixed_finalization_v1/`.

Frontend details:

- architecture: `docs/FRONTEND_ARCHITECTURE.md`;
- development and test commands: `docs/FRONTEND_DEVELOPMENT.md`.
- performance and local hardening: `docs/PERFORMANCE_AND_PRODUCTION_HARDENING.md`;
- local runbook: `docs/LOCAL_PRODUCTION_RUNBOOK.md`.

## Supported Face Workflow

Run the current full-video face workflow:

```bash
redactor redact-video input.mp4 \
  --faces \
  --policy redact_all \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --output-dir artifacts/cases/input_face_redaction \
  --overwrite
```

The command runs:

1. ADAS face detection at proposal threshold `0.20`.
2. Ordinary face tracking with primary observation threshold `0.30`.
3. A full-video face track census.
4. A case privacy policy. The supported default is `redact_all`.
5. Privacy-region compilation and gap recovery for redacted tracks.
6. A non-final automatic preview.
7. A local review bundle for unresolved or risky intervals.
8. A legacy face final-status artifact when that path is available. The case-level automatic command itself remains non-final and does not run mixed finalization.

The normal workflow does not require target-track IDs, frame windows, source-video-specific coordinates, or Ground Truth. Ground Truth is benchmark/evaluation input only; it is not the production privacy policy.

## Unified Automatic Case Preview

One canonical command can run face and license-plate automatic processing for a single source video:

```bash
redactor redact-video input.mp4 \
  --categories face,license_plate \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --plate-model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir artifacts/cases/input_automatic_preview \
  --overwrite
```

This writes one authoritative source identity, category run manifests, one effective-region render plan, one fail-closed `case_manifest.json`, and, when multiple selected categories have validated effective regions, one combined non-final automatic preview. It also builds `review_workspace/`, a single review workspace with source-backed Person and Plate entities, one normalized task list, original and automatic-preview media roles, baseline region identity, media inventory, validation, category-run identities, and a not-started session namespace. The workspace runtime authority is `review_workspace/workspace_manifest.json`, which is referenced and hashed by `case_manifest.json`. Latest reviewed publication happens later from the review session by compiling reviewed regions, directly validating the selected category-run/config identities against the automatic baseline, rendering from Original, validating the full source span, and atomically selecting the immutable revision through `latest_reviewed/current.json`; explicit finalization then approves that exact revision through `final/current.json` without re-running privacy rendering. The automatic command still does not run case-level finalization.

One canonical automatic-preview renderer now handles faces, plates, and combined cases while preserving the source-frame span through strict decoded-frame validation. Face and plate detectors, temporal logic, and privacy policy remain category-specific. Face automated evaluation remains available through `evaluate-faces`; plate evaluation is currently a capability report only and does not publish scores.

To keep a person visible, create an explicit case-policy choice:

```bash
redactor redact-video input.mp4 \
  --faces \
  --policy redact_all_except_keep \
  --keep-track-id KEEP_TRACK_ID \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --output-dir artifacts/cases/input_face_redaction_keep_track_10 \
  --overwrite
```

Use the generated `track_census.json` and `track_gallery/index.html` to understand track IDs before making a keep decision. Ambiguous or unsupported fragments remain review blockers.

Privacy recovery writes machine-readable interval decisions. A gap may be accepted automatically, accepted after configured refinement, marked `needs_review`, or rejected. Synthetic privacy regions never count as raw detector hits.

Lower-level rendering remains available for diagnostics or reviewed correction workflows:

```bash
redactor render-faces input.mp4 \
  --privacy-regions artifacts/cases/input_face_redaction/privacy_regions/privacy_regions.jsonl \
  --interval-decisions artifacts/cases/input_face_redaction/privacy_regions/interval_decisions.json \
  --output-dir artifacts/cases/input_face_redaction/render_preview \
  --mode preview \
  --method pixelate \
  --margin-fraction 0.15 \
  --comparison-frames 120 \
  --overwrite
```

Preview mode may render accepted regions even when some intervals still need review, and the output is explicitly marked non-final. Final mode remains a separate reviewed-rendering boundary: it refuses export when relevant intervals are `needs_review` or `rejected`, when privacy-required frames remain unresolved, or when invalid boxes are present. Preserved reviewed corrections can be rendered in final mode while keeping their provenance.

## Face Evaluation

The selected machine-readable face ground truth is:

```text
data/ground_truth/redaction_ground_truth_v1.0.json
```

It derives from frozen GroundTruth v1.1 without adding new boxes or per-frame annotations. Validate it against local source clips:

```bash
redactor validate-ground-truth data/ground_truth/redaction_ground_truth_v1.0.json \
  --source-video day_face=artifacts/eval_day_face.mp4 \
  --source-video night_selective=artifacts/eval_night_selective.mp4
```

Run the automated evaluator in one of three explicit modes:

```bash
redactor evaluate-faces \
  --ground-truth data/ground_truth/redaction_ground_truth_v1.0.json \
  --clip-id night_selective \
  --mode automatic \
  --source-video artifacts/eval_night_selective.mp4 \
  --detections artifacts/adas_detector_integration_parity_v1/night_integrated.detections.jsonl \
  --adjudication data/evaluations/face_detector_adas_v1.0/adjudication.completed.corrected.json \
  --tracks artifacts/adas_integrated_tracking_validation_v1/runs/current_cli_default/night/tracks.jsonl \
  --privacy-regions artifacts/face_renderer_v1/night_auto_preview/recovery/privacy_regions.jsonl \
  --interval-decisions artifacts/face_renderer_v1/night_auto_preview/recovery/interval_decisions.json \
  --output-dir artifacts/face_pipeline_evaluation_v1/night_automatic \
  --overwrite
```

Evaluation modes:

| Mode | Meaning |
| --- | --- |
| `detector` | Raw ADAS observations only. Tracking and privacy regions do not count as detector hits. |
| `automatic` | Detector, ordinary tracking, and automatically accepted privacy recovery. Preserved/manual corrections are excluded. |
| `reviewed_final` | Includes explicitly reviewed or preserved corrections and may validate the rendered video manifest/media. |

The evaluator reports detector recall, tracking coverage, privacy coverage, provenance counts, unresolved frames, interval decisions, render-manifest consistency, and failure-focused visual evidence. It does not claim pixel-level privacy verification unless that measurement is explicitly implemented.

## Supported Plate Detection, Tracking, And Preview Workflow

The supported plate detector command always uses Open Image Models YOLOv9-S-608:

```bash
redactor detect-plates artifacts/eval_day_plate_close.mp4 \
  --model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir artifacts/diagnostics/day_plate_oim \
  --overwrite
```

The command writes normalized `license_plate` detections, `summary.json`, and an
optional diagnostic overlay. The active configuration is global:

- detector ID: `oim_yolov9_s_608`;
- inference mode: `uniform_small_object_tiles`;
- primary threshold: `0.30`;
- no separate lower proposal threshold selected for the active detector.

By default `detect-plates` processes the complete input video. `--max-frames`
is a diagnostic limiter only; benchmark windows are evaluation inputs and do
not control production plate processing.

After detection, run the prototype temporal stabilization pass:

```bash
redactor track-plates \
  --source-video artifacts/eval_day_plate_close.mp4 \
  --detections artifacts/diagnostics/day_plate_oim/detections.jsonl \
  --output-dir artifacts/diagnostics/day_plate_tracks \
  --overwrite
```

The tracker writes `tracks.jsonl`, `track_summary.json`, `gap_decisions.json`,
`review_intervals.json`, `tracking_manifest.json`, a compact track gallery,
contact sheets, and an H.264/yuv420p diagnostic overlay. Detector observations,
short interpolations, review intervals, and rejected noise remain separate
provenance categories. Long or unsafe gaps are not filled speculatively.
Tracking manifests also record the processed source-frame span and any missing
ranges.

Plate tracking now uses the shared temporal evidence engine. New tracks start
tentative, accumulate audited evidence over time, and become accepted only after
enough detector observations, persistence, density, motion consistency, scale
consistency, and confidence support the same object. Hysteresis keeps a
confirmed track stable through short misses, but long ambiguous gaps remain
review intervals. Interpolation can support continuity after confirmation, but
it cannot by itself satisfy the detector-observation count required for
automatic confirmation.

Render a non-final automatic plate preview from accepted tracks only:

```bash
redactor render-plates \
  --source-video artifacts/eval_day_plate_close.mp4 \
  --tracks artifacts/diagnostics/day_plate_tracks/tracks.jsonl \
  --review-intervals artifacts/diagnostics/day_plate_tracks/review_intervals.json \
  --output-dir artifacts/diagnostics/day_plate_render \
  --overwrite
```

Plate rendering v1 uses one global preview style: pixelation, margin fraction
`0.15`, margin pixels `0`, and pixel size `16`. The preview may contain
unresolved intervals and is machine-readably non-final. It renders accepted
detector observations and accepted short interpolation only; `needs_review`
tracks, rejected noise, unresolved gaps, and speculative prediction are excluded.
The renderer refuses a normal full-duration preview or reviewed-final candidate
when the plate tracks do not cover the complete source span. A diagnostic
partial preview requires `--allow-partial-preview` and is visibly/machine-readably
marked `PARTIAL PROCESSING - NOT FINAL`.

Build the local plate review bundle:

```bash
redactor build-plate-review-bundle \
  --source-video artifacts/eval_day_plate_close.mp4 \
  --tracks artifacts/diagnostics/day_plate_tracks/tracks.jsonl \
  --review-intervals artifacts/diagnostics/day_plate_tracks/review_intervals.json \
  --automatic-preview artifacts/diagnostics/day_plate_render/automatic_preview.mp4 \
  --diagnostic-overlay artifacts/diagnostics/day_plate_tracks/tracking_overlay.mp4 \
  --output-dir artifacts/diagnostics/day_plate_review \
  --overwrite
```

The plate review bundle creates concise tasks for unresolved gaps, uncertain
tracks, possible false accepted tracks, track-boundary audit records, and
deterministic spot checks. A long unresolved gap cannot be approved with
`Looks correct` unless a later correction workflow supplies valid plate regions.

The normal plate workflow does not expose rejected detector candidates. Previous
OpenVINO 0106/0123, Intel YOLOv8n, RF-DETR, RT-DETR, and LPD-YuNet experiments
remain in the archive for historical context.

## Review Bundles

After rendering or evaluation, build a compact local bundle so a reviewer can inspect risky intervals without opening the full video or reading JSON by hand:

```bash
redactor build-review-bundle \
  --source-video artifacts/eval_night_selective.mp4 \
  --case-id night_case_review_v1_2_3 \
  --privacy-regions artifacts/face_renderer_v1/night_auto_preview/recovery/privacy_regions.jsonl \
  --interval-decisions artifacts/face_renderer_v1/night_auto_preview/recovery/interval_decisions.json \
  --render-manifest artifacts/face_renderer_v1/night_auto_preview/render_preview/render_manifest.json \
  --render-validation artifacts/face_renderer_v1/night_auto_preview/render_preview/validation.json \
  --automatic-preview-video artifacts/face_renderer_v1/night_auto_preview/render_preview/redacted_preview.mp4 \
  --evaluation-dir artifacts/face_pipeline_evaluation_v1/night_automatic \
  --corrected-privacy-regions artifacts/face_renderer_v1/night_reviewed_final/recovery/privacy_regions.jsonl \
  --corrected-interval-decisions artifacts/face_renderer_v1/night_reviewed_final/recovery/interval_decisions.json \
  --output-dir artifacts/review/night_case_review \
  --review-render-method pixelate \
  --review-margin-fraction 0.15 \
  --review-pixel-size 16 \
  --overwrite
```

The bundle contains `index.html`, a raw review queue, a smaller reviewer-task list, contact sheets, short evidence clips, local CSS/JS, a checklist, and a decision-export template. It separates pipeline-derived risk, Ground-Truth-only benchmark evidence, and deterministic random audit samples. A clean queue does not prove that every missed face was found.

Review Bundle v1.2.3 presents reviewer tasks instead of raw signal counts. The default guided workflow groups the page into required review, corrected-result confirmation, optional spot checks, and a technical audit. Technical-only signals remain in JSON and expandable audit details, but do not appear as separate default tasks.

Review Bundle v1.2.3.1 keeps the same review evidence and fixes the grouped media controls. In Guided review, `Play together` / `Pause together` and `Restart media` affect only the currently visible task's comparison videos. In View-all mode the grouped controls are disabled so the page does not accidentally start hidden clips.

Random audit samples are initially optional. If a reviewer finds a missed privacy target in one, record it as a reviewer finding and pass it back with `--reviewer-findings`. The bundle promotes the finding to a blocking corrected-result task while preserving the original random-audit evidence in the raw queue. This is separate from frozen Ground Truth: benchmark coverage may pass while a whole-video random audit still discovers a real privacy issue outside the selected benchmark window.

For reliable embedded playback, serve the bundle over localhost:

```bash
redactor serve-review-bundle artifacts/review/night_case_review --open
```

The server binds only to `127.0.0.1`, serves only files under the selected bundle directory, and makes no external network calls. Opening `index.html` directly through `file://` remains possible, but browser handling of local MP4 playback can be inconsistent.

Automatic clips are rendered from the automatic privacy-region JSONL. Corrected final clips are rendered from the reviewed/corrected privacy-region JSONL. Both use the same explicit review-render settings so the comparison differs by privacy-region evidence, not by blur or pixelation style.

Review clips are encoded as browser-compatible H.264/yuv420p MP4s and undergo decoded-frame visual-content validation. FFprobe metadata checks are still recorded, but they are not treated as proof that the clip contains visible frames; bundle generation fails if decoded output turns black while the matching source frames contain visible structure.

Task clips also record source-to-local frame mapping. Privacy regions remain keyed by source frame, while local clip frame `0` means the first frame of the short review clip. Bundle generation verifies expected privacy bboxes against decoded Original and rendered clips; a frame is not counted as visually rendered merely because a JSON region exists.

Reviewer choices are stored in browser-local state until explicitly exported. Validate an exported decisions file with:

```bash
redactor validate-review-decisions \
  --queue artifacts/review/night_case_review/review_queue.json \
  --decisions artifacts/review/night_case_review/decisions/review_decisions.json
```

## Output Layout

A full-video `redact-video` run writes a case directory:

| Directory | Contents |
| --- | --- |
| `source_manifest.json` | Source video hash, frame count, FPS, and dimensions. |
| `privacy_policy.json` | Case-level privacy policy and per-track redact/keep/review assignments. |
| `track_census.json`, `track_gallery/` | Full-video face track census and compact local gallery for policy review. |
| `detection/` | `detections.jsonl`, `summary.json`, optional `overlay.mp4`. |
| `tracking/` | `tracks.jsonl`, `summary.json`, track quality, review flags/windows, contact sheets, and overlay. |
| `privacy_regions/` | Full-video compiled privacy regions, interval decisions, and summary. |
| `automatic_preview/` | Non-final preview MP4, render manifest, summary, validation, and processing log. |
| `review/` | Local `index.html`, queue/task/summary/config JSON, contact sheets, short clips, decision template, and checklist. |
| `final/` | Final render artifacts or a blocked final-status JSON. |
| `audit_manifest.json` | Source, detection, tracking, policy, privacy, rendering, review, and final-status summary. |
| `case_source_identity.json`, `case_manifest.json`, `render_plan/` | Automatic case-preview source identity, fail-closed artifact inventory, category lineage, and effective regions. |
| `review_workspace/` | Case review package with dynamic case tabs, source-backed Person/Plate entities, normalized review tasks, original/automatic media inventory, validation, and `sessions/` for local review history. |

Generated artifacts belong under `artifacts/` and are ignored by Git.

## Advanced Diagnostics

Lower-level commands remain available for focused debugging:

```bash
redactor inspect data/raw/bodycam_sample.mp4 --output artifacts/bodycam_sample.metadata.json
redactor contact-sheet data/raw/bodycam_sample.mp4 artifacts/bodycam_sample_contact_sheet.jpg --interval-seconds 30 --columns 5 --max-frames 10
redactor trim data/raw/bodycam_sample.mp4 artifacts/smoke_clip.mp4 --start 0 --duration 5
redactor detect artifacts/eval_day_face.mp4 --category face --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin --output-dir artifacts/diagnostics/day_face --overwrite
redactor track artifacts/eval_day_face.mp4 --detections artifacts/diagnostics/day_face/detections.jsonl --output-dir artifacts/diagnostics/day_face_tracking --overwrite
redactor recover-face-gaps input.mp4 --tracks artifacts/diagnostics/input_tracking/tracks.jsonl --target-track-id TRACK_ID --keep-track-id KEEP_TRACK_ID --output-dir artifacts/diagnostics/input_gap_recovery --overwrite
redactor render-faces input.mp4 --privacy-regions artifacts/diagnostics/input_gap_recovery/privacy_regions.jsonl --interval-decisions artifacts/diagnostics/input_gap_recovery/interval_decisions.json --output-dir artifacts/diagnostics/input_render_preview --mode preview --overwrite
```

These commands are useful during development. `redact-video` is the supported entry point for the current full-video face workflow.

## Current Evidence And History

The selected detector evidence record lives in `data/evaluations/face_detector_adas_v1.0/`. It records the ADAS-only corrected adjudication, raw-detector metrics, selected thresholds, and FPS provenance audit.

The current machine-readable redaction ground truth is `data/ground_truth/redaction_ground_truth_v1.0.json`. Current non-authoritative automated evaluation artifacts are written under `artifacts/face_pipeline_evaluation_v1/`.

The earlier three-model face detector comparison is preserved as history under `docs/archive/face_detector_bakeoff/`. OpenCV YuNet and OpenVINO `face-detection-0205` are not active face detector backends and are not selectable in the supported face pipeline.

Detailed architecture notes live in `docs/FACE_PIPELINE_TECHNICAL_NOTES.md`. The broader system explanation lives in `docs/SYSTEM_OVERVIEW.md`.

The active plate detector decision lives in
`data/evaluations/license_plate_detector_oim_yolov9_s_608_v1/`. Plate technical
notes live in `docs/PLATE_PIPELINE_TECHNICAL_NOTES.md`, shared post-detection
contracts live in `docs/PRIVACY_PIPELINE_ARCHITECTURE.md`, and older detector
bake-offs are archived under `docs/archive/license_plate_detector_bakeoff/`.

The broad-reader project summary is `docs/PROJECT_OVERVIEW.md`, and the short
local demo guide is `docs/DEMO_GUIDE.md`. The behavior-freeze package is
preserved under `artifacts/final_pipeline_package_v1/`. The current case-based
review package is generated under `artifacts/final_pipeline_package_frontend_v1_5/`; it
contains case manifests, reviewer tasks, local session/correction provenance,
stable Person/Plate review objects, representative thumbnails, a single
versioned frame-aware player, timeline-linked review items, temporal correction
records, and finalization gates.
These generated package artifacts are ignored by Git.

Build the current review workspace with explicit frozen model paths, or the
matching environment variables:

```bash
redactor build-final-review-workspace \
  --output-dir artifacts/final_pipeline_package_frontend_v1_5 \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --plate-model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx
```

Serve the current review workspace with:

```bash
redactor serve-final-review-workspace artifacts/final_pipeline_package_frontend_v1_5 --open
```

Validate or summarize an exported/local session with:

```bash
redactor validate-review-session --session-dir SESSION_DIR
redactor summarize-review-session --session-dir SESSION_DIR
```

## Current Limitations

- Final face rendering is supported only from accepted privacy-region inputs and remains bounded by the quality of the upstream tracking, recovery, and review evidence.
- Face identity grouping and review decisions are still diagnostic inputs, not a complete analyst workflow.
- Privacy gap recovery can reduce temporal gaps, but uncertain intervals must remain review-required.
- Raw detector recall, tracking coverage, privacy recovery, and final redaction quality are separate claims.
- Automated evaluation currently uses range-level face truth and reviewed pipeline evidence; it does not invent missing per-frame boxes.
- Automated review bundles highlight known risks and audit samples, but cannot detect every possible silent miss without Ground Truth.
- No universal or production-wide zero-leak claim has been established.
- Plate detection, minimal tracking, non-final previews, and case-based review/correction records are implemented. Reviewed-final plate output still requires human decisions and verified reviewed corrections for blockers.
- Plate tracking bridges only short, defensible two-anchor gaps. Longer gaps and unstable plate-like tracks remain review-required evidence.
- The unified automatic command can produce a combined non-final preview, manifest, and mixed review workspace for face plus plate evidence. Review-session creation is user-started, Latest reviewed publication is available after required review blockers are resolved, and canonical finalization is an explicit approval step for the exact current Latest reviewed revision.
