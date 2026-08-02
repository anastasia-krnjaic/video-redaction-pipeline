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

## Processed-Frame Accounting

Detection and tracking summaries distinguish:

- `source_video_frame_count`: total frames reported by FFprobe for the source video;
- `processed_frame_count`: frames actually read and processed by the stage;
- `input_detection_frame_count`: tracking's input detection-record count.

This matters when `--max-frames` is used. Tracking output is bounded by the detection records it consumes, not by the source video's total frame count.

## Evidence Boundary

The tracked ADAS detector evidence record is `data/evaluations/face_detector_adas_v1.0/`.

Historical detector comparison context lives in `docs/archive/face_detector_bakeoff/`. Generated artifacts, videos, overlays, model binaries, and browser state remain ignored.
