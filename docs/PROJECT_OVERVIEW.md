# Project Overview

Video Redaction Pipeline is a local workflow for preparing privacy-redacted videos with auditable machine and human evidence. It detects faces and license plates, tracks them through time, renders privacy protection, supports manual correction, and publishes immutable Final media only after review gates pass.

## Why Video Redaction Needs Review

Video privacy is harder than still-image privacy because targets move, blur, leave the frame, reappear, overlap, and sometimes produce weak or missing detections. The system therefore treats detector output as evidence, not as a complete safety claim.

The workflow combines:

- automatic detection and temporal tracking;
- conservative privacy rendering over accepted regions;
- human Guidance for uncertain or important source spans;
- entity-level policy for people and plates;
- manual corrections for visible issues;
- full-video approval of the exact current reviewed video;
- immutable Final publication and delivery checksums.

## Current Release Behavior

The release supports two privacy-object categories:

- faces;
- license plates.

Faces and plates use category-specific detector/tracker paths, then converge into shared privacy-region, rendering, review, validation, and Finalization contracts.

The Final Review Workspace is the review product. It presents source-bound cases (including synthetic examples and local Day/Night research cases), Original/Automatic/Latest reviewed/Final media roles, advisory Guidance, blocking Needs correction, Optional QA, Entities, Manual redaction, Final Results, and Audit Trail.

## Safety Model

Automatic output is not Final. A reviewer can inspect Guidance and Entities, save corrections when needed, build Latest reviewed, watch the full reviewed video, approve that exact revision, and then create Final.

Guidance remains visible and auditable but does not require one approval click per card. Needs correction remains fail-closed. The full-video sweep records coverage of any unadjudicated Guidance without creating fake Correct decisions.

## Release Evidence

Public source evidence includes frozen face/plate Ground Truth and scoped detector results. The ignored local submission evidence additionally includes:

- current approved Day and Night Final media;
- Day and Night full final package zips;
- detector and automatic coverage metrics;
- historical Night reviewed-final geometric coverage and visual audit for `latest_reviewed_133bd8a1e52cfa36`; this is not the score of current Final `latest_reviewed_41fcfdd5b7e2349d`;
- documented Day reviewed-final source incompatibility;
- documented face reviewed-final numerical limitation;
- production-readiness and cleanup evidence.

See [RELEASE_EVIDENCE.md](RELEASE_EVIDENCE.md).

## Limits

The project does not claim universal or mathematical zero leaks for complete product videos. The frozen metrics apply only to their source-bound Ground Truth and policy contracts. Current Final approval means the exact Latest reviewed media was watched through in full, explicitly approved, and published as immutable Final media. Remote multi-user deployment requires additional operational and security hardening beyond the local release workflow.

## Public portfolio setup

The [README](../README.md) provides the asset-free quick start. Use
[MODEL_SETUP.md](MODEL_SETUP.md) for optional real-video inference and
[DEMO_GUIDE.md](DEMO_GUIDE.md) for generated, non-identifying media. Models,
private research footage, sessions, and delivery packages are not distributed.
