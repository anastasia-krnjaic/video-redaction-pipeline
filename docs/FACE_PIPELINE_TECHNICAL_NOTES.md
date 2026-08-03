# Face Pipeline Technical Notes

These notes capture implementation-level detail for the current Face Pipeline v1. The broader explanation is in `docs/SYSTEM_OVERVIEW.md`.

## Active Face Detector

The only active face detector backend is OpenVINO `face-detection-adas-0001`, recorded as `openvino_face_adas_0001`.

Default model discovery:

- XML: `models/face-detection-adas-0001/face-detection-adas-0001.xml`
- BIN: same path with `.bin` suffix
- Environment overrides: `REDACTOR_ADAS_XML`, `REDACTOR_ADAS_BIN`

Default thresholds:

- Proposal/storage floor: `0.20`
- Primary observation threshold: `0.30`
- Weak observation range: `[0.20, 0.30)`

Weak observations may support temporal tracking, but they must not independently confirm a track.

## Pipeline Stages

`redactor face-pipeline` runs:

1. `detection.run_detection(...)` for ADAS face detections.
2. `tracking.run_tracking(...)` for ordinary temporal grouping and track quality.
3. `privacy_gap_recovery.run_privacy_gap_recovery(...)` only when target track IDs are supplied.

`redactor render-faces` consumes the privacy-region output and writes a redacted MP4 plus render manifests. It is intentionally separate from `face-pipeline` so preview and final approval can be controlled explicitly.

Lower-level commands (`detect`, `track`, `recover-face-gaps`, and `render-faces`) remain diagnostic and advanced entry points.

## Detection Output

Detection output is frame-indexed JSONL. The writer emits one record for every processed frame, including empty frames.

When a clip ID is supplied, detection records include deterministic candidate IDs:

```text
<detector_id>:<clip_id>:f<frame_index>:d<detection_index>
```

ADAS output stores:

- `category: face`
- detector ID
- score
- clipped `bbox_xyxy`
- source detection index and candidate ID when a clip ID is supplied
- model metadata and model file hashes in the detection summary

## Tracking Semantics

The tracker consumes frame-indexed face detections and distinguishes:

- weak observations
- primary observations
- auto-confirmed tracks
- review-candidate tracks
- rejected tracks
- observed boxes
- interpolated boxes
- predicted boxes

Synthetic boxes may improve temporal continuity, but only real observations and manual inputs count as evidence that a track exists.

## Privacy Gap Recovery

Privacy gap recovery runs after ordinary tracking. It is a privacy-continuity layer, not detector recall and not track confirmation.

Supported recovery strategies:

- short interpolation
- bidirectional privacy bridge
- one-sided bridge
- preserved accepted region
- conservative fallback or review-required record

Bidirectional recovery tracks forward from the last reliable pre-gap anchor and backward from the first reliable post-gap anchor. It records raw and refined trajectories, disagreement, selected direction, fused boxes where compatible, and unresolved frames where evidence is insufficient.

The optional `bottom_component` refinement is disabled by default. It is a strategy-specific visual refinement for cases where a dark visual component connected to a configured edge is useful; it is not universal face logic.

## Safety Gates

Recovery checks are adaptive and scale-normalized. They include:

- invalid boxes
- boxes outside the frame
- center movement relative to box size
- forward/backward disagreement relative to box diagonal
- scale change
- overlap with keep-person tracks
- proximity to competing identities
- scene cuts
- excessive one-sided gap duration
- unexplained trajectory jumps

Each gap receives an interval-level decision:

- `accepted_automatically`
- `accepted_after_refinement`
- `needs_review`
- `rejected`

Machine-readable reasons are written with each decision. Safety gates should favor `needs_review` over unsafe automatic coverage.

## Provenance

Generated privacy regions distinguish:

- detector evidence
- ordinary tracker output
- interpolation
- bidirectional privacy bridge
- forward-only bridge
- backward-only bridge
- conservative fallback
- manual correction
- preserved accepted regions

This provenance separation is required because raw detector recall, tracking continuity, privacy recovery, and final rendering quality are separate claims.

## Face Redaction Rendering

`rendering.run_face_render(...)` renders accepted face privacy regions on original-resolution frames. The renderer supports:

- Gaussian blur;
- pixelation;
- configurable pixel and scale-relative margins;
- optional side-by-side diagnostic comparison clips;
- audio remuxing from the source video when an audio stream is present.

Renderer inputs may be either the frame-grouped privacy-gap recovery JSONL or a flat preserved reviewed privacy-track JSONL. The renderer reads interval decisions when supplied and writes:

- `redacted_preview.mp4` or `redacted_final.mp4`;
- `render_manifest.json`;
- `render_summary.json`;
- `validation.json`;
- `processing_log.jsonl`;
- optional `diagnostic_comparison.mp4`.

Preview mode may render accepted regions even when interval decisions still contain `needs_review` or unresolved frames. Preview output is marked non-final in both the manifest and video overlay.

Final mode blocks export when:

- any relevant interval is `needs_review` or `rejected`;
- privacy-required frames are unresolved;
- invalid boxes are present.

Preserved reviewed corrections are renderable in final mode, but their provenance remains distinct from detector, ordinary tracker, and synthetic bridge regions. Rendering does not convert synthetic or reviewed regions into raw detector evidence.

FFprobe validation records expected and observed resolution, frame rate, frame count, duration, audio presence, and audio stream duration when present. MP4 container frame counts may differ from decoded writer counts by a small documented tolerance; the renderer records both the internal rendered-frame count and FFprobe validation result.

## Machine-Readable Ground Truth And Evaluation

The selected machine-readable redaction ground truth is `data/ground_truth/redaction_ground_truth_v1.0.json`. It derives from frozen GroundTruth v1.1 and the face benchmark v1.1 JSON. It supports future object types such as license plates, but the current evaluator only scores face objects.

The schema records:

- clip ID, source video hash, FPS, source frame count, and evaluated frame range;
- object ID, object type, and policy (`redact`, `keep`, or `ignore`);
- detector-scored ranges;
- detector-ignore ranges;
- privacy-required ranges;
- keep-clear ranges;
- optional per-frame boxes or keyframes only when those annotations actually exist;
- visibility, occlusion, provenance, review status, and notes.

`redactor validate-ground-truth` validates required fields, frame ranges, policies, object IDs, boxes, contradictory ranges, and optional source-video hashes.

`redactor evaluate-faces` writes:

- `metrics.json`;
- `object_metrics.json`;
- `interval_decisions.json`;
- `failures.json`;
- `render_validation.json`;
- `summary_table.csv`;
- `report.md`;
- concise failure visual evidence when a decodable source video is available.

Evaluation modes:

- `detector`: raw ADAS observations only. The evaluator uses the corrected ADAS adjudication as identity truth for detector-scored frames. Tracking, synthetic privacy boxes, and reviewed corrections never count as detector hits.
- `automatic`: detector, ordinary tracking, and automatically accepted privacy recovery. Preserved or manual reviewed corrections are excluded, and unresolved or review-required privacy frames keep the run in `review_required`.
- `reviewed_final`: includes explicitly reviewed or preserved corrections and can validate the rendered-video manifest, output hash, and FFprobe media checks.

Final statuses are:

- `detector_evaluated`;
- `review_required`;
- `passed_automatic`;
- `passed_reviewed_final`;
- `failed_privacy_acceptance`;
- `failed_input_validation`.

Because current face truth is range-level, the evaluator reports object-frame recall and frame coverage. It does not claim IoU accuracy, pixel-level rendered privacy, or universal zero-leak performance.

## Automated Review Bundles

`redactor build-review-bundle` builds a self-contained local review package from pipeline artifacts. The command is intentionally a review layer, not a correction UI and not an approval override.

Inputs may include:

- source video;
- detections and tracks;
- privacy regions and interval decisions;
- render manifest and render validation;
- automatic preview and reviewed-final videos;
- evaluator output directory;
- machine-readable Ground Truth for benchmark context.

The review queue schema is versioned as `face_review_queue_schema_v1.0`. Each item records a stable item ID, clip/case ID, source video hash, object category, optional object or track IDs, frame/timestamp span, severity, blocking status, source stage, reason codes, explanation, provenance, metrics, unresolved-frame count, candidate-region status, review status, allowed actions, visual-evidence paths, and creation metadata.

Reason-code families are deliberately separated:

- pipeline risk, such as `needs_review_interval`, `rejected_interval`, `unresolved_privacy_frames`, `invalid_box`, `long_detection_gap`, `one_sided_recovery`, `excessive_bidirectional_disagreement`, `excessive_center_jump`, `excessive_scale_change`, `possible_identity_competition`, `possible_keep_person_overlap`, `scene_cut_inside_track`, `track_fragmentation`, `render_validation_failure`, and `media_validation_failure`;
- Ground-Truth-only evaluation evidence, such as `ground_truth_detector_miss` and `ground_truth_privacy_miss`;
- non-blocking `random_audit_sample` items.

Nearby or overlapping events are merged only when they describe the same practical review interval. Distinct `needs_review`, `rejected`, or preserved reviewed intervals remain separate cards. Raw detector misses remain detector evidence and are not folded into privacy blockers.

The generated bundle contains local `index.html`, `INDEX.md`, `REVIEW_CHECKLIST.md`, queue/summary/config JSON, local CSS/JS, contact sheets, short clips, and a decisions template. HTML and JSON paths are relative so the bundle can be moved as a folder. No external JavaScript, CSS, fonts, analytics, or network calls are used.

The HTML page stores temporary decisions in browser-local state. Reviewers must explicitly export decisions before closing the page or moving machines. `redactor validate-review-decisions` validates the exported file against the original queue, rejects duplicate or unknown item IDs, checks status/action compatibility, and reports unresolved blocking items. It does not alter privacy regions. A `correction_required` decision preserves the final blocker until valid corrected regions are supplied and normal validation passes.

## Processed-Frame Accounting

Detection and tracking summaries distinguish:

- `source_video_frame_count`: total frames reported by FFprobe for the source video;
- `processed_frame_count`: frames actually read and processed by the stage;
- `input_detection_frame_count`: tracking's input detection-record count.

This matters when `--max-frames` is used. Tracking output is bounded by the detection records it consumes, not by the source video's total frame count.

## Evidence Boundary

The tracked ADAS detector evidence record is `data/evaluations/face_detector_adas_v1.0/`.

Historical detector comparison context lives in `docs/archive/face_detector_bakeoff/`. Generated artifacts, videos, overlays, model binaries, and browser state remain ignored.
