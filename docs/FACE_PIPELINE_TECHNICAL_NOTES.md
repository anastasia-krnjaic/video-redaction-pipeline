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

`redactor redact-video` is the supported full-video workflow. It runs:

1. ADAS face detection, unless a validated ADAS detections JSONL is supplied for diagnostics.
2. `tracking.run_tracking(...)` for ordinary temporal grouping and track quality.
3. A full-video track census over auto-confirmed, review-candidate, weak-only, and rejected tracks.
4. A case privacy policy, defaulting to `redact_all`.
5. Full-video policy compilation into privacy regions and interval decisions.
6. Non-final preview rendering.
7. Local review-bundle generation.
8. Final render or a blocked final-status record.

`redactor face-pipeline` remains a lower-level diagnostic command for detection, tracking, and optional target-track gap recovery. It is not the primary production-oriented command.

`redactor render-faces` consumes the privacy-region output and writes a redacted MP4 plus render manifests. It is intentionally separate from `face-pipeline` so preview and final approval can be controlled explicitly.

Lower-level commands (`detect`, `track`, `recover-face-gaps`, and `render-faces`) remain diagnostic and advanced entry points.

## Shared Post-Detection Contracts

Face detection and face tracking behavior remain frozen, but face render regions
can now be adapted to the shared privacy records in
`src/redactor/privacy_records.py`. The face profile in
`src/redactor/privacy_profiles.py` records the active ADAS detector ID, selected
ADAS threshold policy, existing tracking profile, privacy-gap recovery settings,
and render profile hash.

The current face tracking and privacy-gap recovery engines are intentionally not
rewritten in this milestone. Shared contracts are used for parity tests and for
future unified rendering/review work once exact behavior preservation has been
demonstrated.

Shared Temporal Evidence v1 is active for plate track confirmation only. The
Face Pipeline continues to use the frozen ADAS face tracking, privacy gap
recovery, reviewed-region preservation, and rendering behavior. Face profile
metadata records compatibility values for future migration planning, but those
values are not used to confirm or reject production face tracks.

## Final Case Review Workspace

`redactor build-final-review-workspace` packages the frozen face evidence into a
case-based local review workspace. The workspace can display face review tasks,
record reviewer decisions, and add reviewed correction records as a separate
provenance layer. It does not rerun or retune ADAS detection, ordinary tracking,
privacy-gap recovery, or the accepted reviewed face corrections.

Final Review Workspace v2.2 validates the expected ADAS XML/BIN hashes when it
is built. It also indexes face regions into reviewer-facing entities such as
`Person 01` instead of exposing raw detector track IDs in the default interface.
Reviewed face corrections use explicit keyframes and checkpoint verification;
unverified interpolation checkpoints block finalization.

When finalization is requested, face regions come from frozen automatic face
privacy regions plus existing accepted reviewed face corrections and any newly
saved reviewed-correction events. The event log and final provenance identify
which reviewer supplied each change. The clean final media does not burn frame
numbers or diagnostics into the deliverable video; frame numbers are part of the
local review player only.

## Full-Video Case Privacy Policy

`src/redactor/face_policy.py` contains the production-oriented policy layer. It keeps operational privacy policy separate from benchmark Ground Truth.

Machine-readable policy files use schema ID `face_privacy_policy_v1` and record:

- case ID;
- source video hash and frame count;
- object category;
- policy mode;
- full source frame span;
- per-track assignments;
- assignment provenance;
- unresolved policy assignments;
- review status.

Supported policy modes:

- `redact_all`: default. Every supported face track is assigned redaction.
- `redact_all_except_keep`: all supported face tracks are redacted except tracks explicitly marked `keep`.
- `redact_selected`: only explicitly selected tracks are redacted.
- `review_only`: all assignments require review.

Assignments can be `redact`, `keep`, `ignore`, or `review`. A rejected track is retained in the census and creates a blocking review assignment, but it is not rendered as a defensible automatic privacy region. This avoids treating one-off weak or rejected proposals as accepted privacy evidence while still surfacing them to the reviewer.

`redact_all_except_keep` is a case-policy choice. The system does not infer keep identity from benchmark labels, broad regions of interest, clothing, or role. The current implementation supports explicit keep track IDs and blocks ambiguous or unsupported fragments for review; it does not implement face recognition.

Ground Truth is not imported by the policy compiler. It remains evaluation-only.

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

The raw review queue schema is versioned as `face_review_queue_schema_v1.0`. Each raw item records a stable item ID, clip/case ID, source video hash, object category, optional object or track IDs, frame/timestamp span, severity, blocking status, source stage, reason codes, explanation, provenance, metrics, unresolved-frame count, candidate-region status, review status, allowed actions, visual-evidence paths, and creation metadata.

Review Bundle v1.2 keeps the presentation layer introduced in v1.1 and emits reviewer tasks under `face_reviewer_task_schema_v1.2`. This layer does not remove or rewrite raw signals. It groups related raw items into one concise task for the same object, overlapping or adjacent interval, and practical reviewer action. The generated `review_tasks.json` mirrors the task list and records which raw review item IDs each task represents.

Reason-code families are deliberately separated:

- pipeline risk, such as `needs_review_interval`, `rejected_interval`, `unresolved_privacy_frames`, `invalid_box`, `long_detection_gap`, `one_sided_recovery`, `excessive_bidirectional_disagreement`, `excessive_center_jump`, `excessive_scale_change`, `possible_identity_competition`, `possible_keep_person_overlap`, `scene_cut_inside_track`, `track_fragmentation`, `render_validation_failure`, and `media_validation_failure`;
- Ground-Truth-only evaluation evidence, such as `ground_truth_detector_miss` and `ground_truth_privacy_miss`;
- non-blocking `random_audit_sample` items.

Task categories are:

- `action_required`: automatic evidence is insufficient and corrected regions are still needed;
- `corrected_result_check`: a reviewed or preserved final result is available and should be confirmed;
- `automatic_continuity_check`: a raw detector miss or similar signal appears covered by downstream tracking or rendering;
- `optional_spot_check`: deterministic random audit samples from apparently normal intervals;
- `technical_only`: audit evidence that should not inflate the normal task count.

Nearby or overlapping events are merged only when they describe the same practical review interval. Distinct `needs_review`, `rejected`, or preserved reviewed intervals remain separate tasks. Handled detector misses and provenance transitions can be kept in the Technical audit when they do not require a correction decision.

Media roles are explicit:

- `original`: unredacted source context;
- `automatic`: the automatic privacy-region JSONL rendered over the shared source-frame interval;
- `corrected_final`: the reviewed/corrected privacy-region JSONL rendered over the same source-frame interval;
- `diagnostic`: overlays or contact sheets used as secondary evidence.

The benchmark comparison clips use an explicit review-render configuration, currently `pixelate` with `margin_fraction=0.15`, `margin_pixels=0`, and `pixel_size=16`. This configuration is recorded in `review_config.json`, each role's media audit, and `review_media_audit.json`. It does not change the global `render-faces` defaults.

Review Bundle v1.2.1 encodes task clips as browser-compatible H.264/yuv420p MP4s. Earlier review clips could carry valid FFprobe duration, frame-count, and resolution metadata while still failing local browser playback because the task-level writer emitted OpenCV `mp4v` video streams. FFprobe metadata remains useful, but bundle generation now also decodes first, middle, and final output frames, compares them against the selected source interval, and records luminance, black-pixel, frame-count, codec, pixel-format, and source-relative alignment checks. If visible source content is replaced by black output or repeated frames, the bundle fails before publication.

Review Bundle v1.2.2 makes frame mapping explicit. Privacy-region JSONL records use source-video frame indices; short review clips use local frame indices starting at zero. For a task clip beginning at source frame `S`, local frame `L` maps to source frame `S + L`, and source frame `F` maps to local frame `F - S`. The bundle records this mapping for every displayed media role.

The bundle also verifies rendered-region application. For every expected privacy region in a task clip, it decodes the unredacted Original clip and the rendered Automatic or Corrected final clip, maps the source frame to the local clip frame, expands the bbox with the configured review margin, and checks for a localized pixel change inside that bbox. This distinguishes JSON privacy coverage from visibly verified rendered coverage. Bundle generation fails when a region exists in JSON but does not visibly alter the intended decoded frame, when a region appears shifted to another local frame, or when declared and verified rendered coverage disagree.

Review Bundle v1.2.3 adds reviewer-discovered findings. A deterministic random-audit sample starts as optional, but if a reviewer identifies a missed privacy target, that finding is recorded as human-review evidence with reason codes such as `reviewer_found_privacy_miss` and `silent_miss_found_by_random_audit`. Supplying the finding through `--reviewer-findings` promotes the interval to a blocking corrected-result task and moves the original random-audit item into the Technical audit so the default workflow does not ask the reviewer to decide the same interval twice.

Reviewer findings are not frozen Ground Truth. They can expose whole-video privacy problems outside a selected dense benchmark window. A correction may therefore add reviewed privacy regions outside the benchmark ranges while the benchmark evaluator still reports its range-level metrics separately.

Review Bundle v1.2.3 also adds `redactor serve-review-bundle`. The command serves a bundle from `127.0.0.1` using the Python standard library, restricts paths to the selected bundle directory, sets MP4 MIME types for embedded playback, and makes no external network calls. The HTML keeps direct file links as a fallback and displays a `file://` hint recommending the localhost server.

The guided page saves a decision immediately, updates progress, marks the selected action, and advances to the next pending required task before optional spot checks. If an optional random-audit task is marked `Needs review`, exported-decision validation treats it as a remaining final blocker until corrected regions are supplied and confirmed.

Review Bundle v1.2.3.1 fixes the grouped media controls. The toolbar resolves comparison videos from the currently visible Guided-review task at click time, so `Play together`, `Pause together`, and `Restart media` do not act on the hidden View-all task list. View-all mode disables grouped playback with an explanatory message; direct native video controls and file links remain available for every clip.

The bundle computes hashes for media-role clips and suppresses duplicate display roles. Validation fails if one identical file would be silently shown as distinct primary evidence, such as both Original and Corrected final. Validation also fails when Automatic and Corrected final use different source-frame ranges, rendered frame counts, renderer configuration hashes, or incompatible durations.

Corrected final media must be traceable to a corrected privacy-region input. A pre-rendered final MP4 alone is not sufficient provenance for a benchmark comparison. The corrected role records the privacy-region input path and hash, interval-decision input hash, source-frame range, source-frame sample hashes, output hash, and provenance counts for the interval.

The generated bundle contains local `index.html`, `INDEX.md`, `REVIEW_CHECKLIST.md`, `review_queue.json`, `review_tasks.json`, `review_media_audit.json`, summary/config JSON, local CSS/JS, contact sheets, short clips, and a decisions template. HTML and JSON paths are relative so the bundle can be moved as a folder. No external JavaScript, CSS, fonts, analytics, or network calls are used.

The HTML page stores temporary decisions in browser-local state. Reviewers must explicitly export decisions before closing the page or moving machines. Exported decisions identify the reviewer task ID, underlying raw review item IDs, selected user-facing action, machine-readable status, reviewed media role, and timestamp. `redactor validate-review-decisions` validates the exported file against the original queue, rejects duplicate or unknown task/item IDs, checks status/action compatibility, and reports unresolved blocking tasks. It does not alter privacy regions. A `correction_required` decision preserves the final blocker until valid corrected regions are supplied and normal validation passes.

## Processed-Frame Accounting

Detection and tracking summaries distinguish:

- `source_video_frame_count`: total frames reported by FFprobe for the source video;
- `processed_frame_count`: frames actually read and processed by the stage;
- `input_detection_frame_count`: tracking's input detection-record count.

This matters when `--max-frames` is used. Tracking output is bounded by the detection records it consumes, not by the source video's total frame count.

## Evidence Boundary

The tracked ADAS detector evidence record is `data/evaluations/face_detector_adas_v1.0/`.

Historical detector comparison context lives in `docs/archive/face_detector_bakeoff/`. Generated artifacts, videos, overlays, model binaries, and browser state remain ignored.

The selected plate detector is documented separately in
`docs/PLATE_PIPELINE_TECHNICAL_NOTES.md`. Plate detector and prototype tracking
work remain separate from the frozen Face Pipeline behavior until a dedicated
unified review or rendering milestone explicitly integrates plate regions.
