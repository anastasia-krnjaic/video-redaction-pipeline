# Shared Post-Detection Privacy Pipeline Foundation v1

This document records the shared foundation for post-detection privacy processing. It does not change the selected detectors or their thresholds.

## Architecture Decision

Detectors remain category-specific. Faces use OpenVINO `face-detection-adas-0001`; license plates use Open Image Models YOLOv9-S-608. After a detector emits evidence, common records describe the rest of the privacy pipeline:

```text
category-specific detector adapter
-> normalized privacy observations
-> category tracking profile
-> temporal regions and gap decisions
-> approved privacy regions
-> shared renderer
-> shared review and audit contracts
```

The shared layer exists so future privacy-object categories do not need a separate copy of frame-span accounting, provenance naming, region rendering, review-task structure, and manifest/audit conventions.

## Common Records

`src/redactor/privacy_records.py` defines the small versioned record set:

- `PrivacyObservation`: detector evidence from one category-specific model.
- `TemporalRegion`: a source-frame region produced by tracking, interpolation, review, or another temporal stage.
- `GapDecisionRecord`: an accepted, review-required, or rejected decision for a gap.
- `PrivacyRegionRecord`: an approved or unresolved privacy region that can be rendered.
- `ProcessingSpan`: source-frame coverage and missing-range accounting.
- `LifecycleTransition`: common state transitions such as `tentative`, `confirmed`, `temporarily_lost`, `needs_review`, `rejected`, and `terminated`.
- `ReviewTaskRecord`: category-aware review issue, media, action, and blocker metadata.

Existing face and plate serialized artifacts remain valid. Adapters convert current face render regions and plate render regions into the shared records where needed; accepted artifacts are not rewritten.

`src/redactor/privacy_lifecycle.py` defines lightweight protocol contracts for association, evidence accumulation, hysteresis transitions, gap decisions, retrospective promotion, and tracklet stitching. `src/redactor/privacy_evidence.py` implements the shared accumulated-evidence and hysteresis engine for post-detection tracklets. It is active for plate track confirmation and eligibility, but it is not enabled for the production Face Pipeline.

## Category Profiles

`src/redactor/privacy_profiles.py` defines a lightweight profile abstraction. It is not a plugin framework. A profile records the selected detector ID and configuration hash plus category-appropriate geometry, association, confirmation, gap-recovery, rendering, and review settings.

Face and plate profiles intentionally keep different thresholds because the objects are different:

- faces use ADAS weak/primary observation thresholds and face tracking/recovery safety gates;
- plates use the selected OIM detector hash, plate geometry constraints, plate association gates, bounded plate interpolation, and the plate render margin.

Sharing the profile interface does not mean sharing literal threshold values.

## Shared Temporal Evidence

The shared evidence engine consumes normalized `TemporalRegion` and `GapDecisionRecord` data. It records component-level evidence instead of hiding decisions inside one opaque score. Components include persistence, detector-observation count, observation density, mean and lower-tail confidence, aspect-ratio consistency, center-motion consistency, scale consistency, source-provenance support, internal-gap penalty, abrupt-jump penalty, competing-track penalty, scene-cut penalty, and screen-lock risk.

Tracks start as `tentative`. A category profile defines when accumulated evidence is enough for `confirmed`, when hysteresis can keep a confirmed track through short misses, and when a track should become `needs_review`, `temporarily_lost`, `terminated`, or `rejected`. When a track becomes confirmed, earlier valid detector observations and accepted short interpolation from the same consistent tracklet may be retrospectively promoted for rendering. This promotion preserves original provenance: a detector observation remains detector evidence, and interpolation remains interpolation.

The plate profile requires enough true detector observations for automatic confirmation. Interpolated frames can support continuity but cannot satisfy that detector-observation count. This prevents a sparse false track from becoming accepted merely because interpolation filled its span. The engine also keeps persistent screen-fixed evidence as a review risk rather than automatically accepting it just because it lasts for many frames.

Face profiles record compatibility metadata only in this milestone. Production face tracking, recovery, rendering, reviewed corrections, and provenance remain on the frozen Face Pipeline v1 path.

## Shared Renderer

`src/redactor/privacy_rendering.py` contains the category-independent rendering core. It accepts `rendering.RenderRegion` records derived from approved `PrivacyRegionRecord`s and supports:

- pixelation;
- Gaussian blur through the existing render settings;
- category-specific margins supplied by the caller/profile;
- multiple categories on one frame;
- deterministic region ordering;
- provenance counts;
- H.264/yuv420p/faststart MP4 output;
- decoded rendered-region verification.

`render-plates` now uses this shared core through its existing plate renderer wrapper. Face rendering has adapters and parity tests, but the frozen face renderer remains in place until a later migration proves exact full-run parity.

## Frame-Span Safety

`ProcessingSpan` wraps the existing frame-span accounting. Both face and plate renderers can use the same representation for complete spans, partial spans, missing ranges, and final-output blockers.

The current plate partial-span safety behavior is preserved: a normal full-duration preview or reviewed-final candidate is blocked unless detector/tracker inputs cover the declared source span, or the user explicitly requests a visibly non-final partial diagnostic preview.

## Shared Review Contract

Review tasks now have a common category-aware shape: category, object or track ID, interval, issue type, blocking state, available media roles, allowed reviewer actions, and whether corrected privacy regions are required.

The existing face and plate review pages are not redesigned in this milestone. Category-specific labels and wording remain in their current builders while the common task concepts are available for future shared review logic.

## Migration Strategy

The migration is deliberately incremental:

1. Plate rendering uses the shared rendering core because plate rendering already has accepted-track eligibility, span validation, and decoded-region checks.
2. Plate tracking now uses the shared temporal evidence engine for confirmation, hysteresis, retrospective promotion, and bounded stitch decisions while preserving the existing association and gap-recovery parameters.
3. Face artifacts are adapted to shared records and parity-tested, but the current face tracking, privacy gap recovery, and renderer behavior stay frozen.
4. Future tracking lifecycle work can use the common states and transition records without embedding accumulated-evidence behavior inside a category module.
4. A later unified face/plate render can consume `PrivacyRegionRecord`s from both categories once reviewed plate corrections exist.

## Intentionally Not Unified

The following remain category-specific:

- detector models, model loading, preprocessing, postprocessing, and thresholds;
- face ordinary tracking and privacy gap recovery behavior;
- plate association and gap-recovery parameters;
- category-specific geometry validation;
- category-specific review language;
- current CLI entry points such as `face-pipeline`, `detect-plates`, `track-plates`, `render-faces`, and `render-plates`.

The foundation is meant to reduce duplicate contracts, not to erase useful category boundaries.
