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

Lower-level commands (`detect`, `track`, and `recover-face-gaps`) remain diagnostic and advanced entry points.

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

## Evidence Boundary

The tracked ADAS detector evidence record is `data/evaluations/face_detector_adas_v1.0/`.

Historical detector comparison context lives in `docs/archive/face_detector_bakeoff/`. Generated artifacts, videos, overlays, model binaries, and browser state remain ignored.
