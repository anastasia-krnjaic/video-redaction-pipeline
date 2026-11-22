# System Overview

Video Redaction Pipeline is a local media privacy system for faces and license plates. It combines automatic detection/tracking, shared privacy rendering, source-bound human review, manual correction, immutable Final publication, evaluation, and production-readiness safeguards.

## Release Status

The processing and review behavior remains frozen during Portfolio Release Preparation v1. The public repository demonstrates the existing face/plate pipeline, synthetic developer workflow, and compact evaluation evidence. Private submission media, sessions, and delivery packages remain ignored local artifacts; their existence is not a clean-clone prerequisite.

Local submission checkpoints (historical product labels, not a public portfolio release tag):

- production-readiness tag: `production-readiness-v1`
- current submission release label: `submission-release-v1.0.3`
- current review policy: `review_guidance_full_sweep.v1`

## Pipeline Shape

```text
source video
  -> category detectors
  -> category tracking and temporal evidence
  -> automatic privacy regions
  -> Automatic review media
  -> Guidance, Entities, and Manual correction
  -> Latest reviewed
  -> full-video approval
  -> immutable Final
  -> delivery package and evaluation evidence
```

Faces and license plates keep separate detector adapters and category-specific tracking policy. After detection, shared contracts represent observations, temporal regions, privacy regions, reviewed regions, renderer inputs, review tasks, validation, and audit provenance.

## Review And Finalization

The Final Review Workspace is the human review boundary. Guidance is advisory and remains auditable. Needs correction is blocking. Manual corrections are previewed as real rendered privacy pixels, saved as immutable correction revisions, and folded into reviewed-region compilation.

Finalization requires the exact current Latest reviewed revision and a current full-video sweep acknowledgement bound to that revision, media, reviewed-region set, session, source, and active projection. Final media is the approved Latest reviewed media bytes and is selected through a case-scoped current pointer.

## Evaluation

Evaluation is frozen and separated by claim type:

- detector recall;
- automatic eligible coverage;
- reviewed-final geometric coverage;
- full-video human sweep evidence.

Tracked detector evidence reports ADAS face recall of 833 / 852 at the primary 0.30 threshold and OIM plate recall of 557 / 600. These source-bound detector metrics exclude temporal and reviewed regions.

Day reviewed-final plate scoring is unavailable because the frozen benchmark source differs from the product Day Final source. The Night geometric result 284 / 360 = 0.788889 and its 76-frame visual audit belong to historical revision `latest_reviewed_133bd8a1e52cfa36`, not current submission revision `latest_reviewed_41fcfdd5b7e2349d`. No current-Final score is inferred from that result. Face reviewed-final numerical scoring is not supported by the frozen face evaluation contract.

See [EVALUATION.md](EVALUATION.md).

## Production Safeguards

The release includes single-writer locking, atomic state writes, strict JSON parsing, append-only hash-chained events, immutable metadata caching, event projection caching, no-op materialization reuse, heavy-job admission control, resumable stage manifests, structured telemetry, and streaming validation.

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

## Important Boundaries

The release does not claim universal or mathematical zero leaks for complete product videos, does not publish detector precision without complete negative Ground Truth, and does not reinterpret Ground Truth after Final publication. Current Final approval means the exact Latest reviewed media was watched through in full, explicitly approved, and published as immutable Final media. Detector inference remains the primary heavy workload and was not semantically changed during production optimization.

## Public developer entry points

Start with the [README](../README.md), [model setup](MODEL_SETUP.md), and
[synthetic demo guide](DEMO_GUIDE.md). The public suite uses synthetic fixtures;
private release-lineage tests are explicit optional integration checks. Archived
comparison models and historical overviews are research context, not supported
setup alternatives.
