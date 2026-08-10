# Unified Automatic Case Preview Orchestration

This document records the backend refactor that moves full-video automatic privacy-preview orchestration from separate face and plate command paths toward one case-level checkpoint. It does not change detector selection, detector thresholds, tracker behavior, rendering semantics, review policy, or finalization gates.

This checkpoint is not a completed mixed review or finalization pipeline. The canonical `redact-video` command produces automatic, non-final preview evidence and a fail-closed case artifact manifest.

## Verified Before-State

The refactor was started from commit `0fbbd713ebe77983ec42a31782f0c08202a46b36` on `main`/`origin/main`, whose subject is `Freeze final review workspace v1.5.4`. The workspace had no literal `v1.5.4` tag; `review-workspace-v1.5` was the available v1.5 tag, and `git describe` reported `review-workspace-v1.5-2-g0fbbd71`.

Before the change, face full-video execution lived in `redact-video` and delegated to `face_policy.run_full_video_policy_flow` after optional ADAS detection. License plates used separate `detect-plates`, `track-plates`, `render-plates`, and `build-plate-review-bundle` stages. Final review and finalization were already mostly case-level and category-aware, but automatic orchestration and evaluation entry points were not.

## Automatic Preview After-State

`src/redactor/case_pipeline.py` now owns the automatic-preview lifecycle:

```text
INGEST
-> DETECT
-> TEMPORAL_ASSOCIATION
-> BOUNDED_RECOVERY
-> POLICY
-> PRIVACY_REGIONS
-> REVIEW_TASKS
-> AUTOMATIC_PREVIEW
-> LATEST_REVIEWED
-> FINALIZATION
-> EVALUATION
```

Each selected category runs through an adapter. The adapters wrap the existing face and plate implementations rather than replacing them. The pipeline writes one authoritative case source identity, category run manifests, a unified render plan, effective privacy-region JSONL, a validated combined automatic preview when multiple selected categories have effective regions, and `case_manifest.json`.

Executed by the canonical command:

- ingest and source identity publication;
- category detection;
- category temporal processing;
- face policy and effective privacy-region conversion;
- category review-task evidence;
- automatic preview rendering;
- fail-closed case manifest publication.

Represented but not executed:

- mixed review session generation;
- Latest reviewed publication;
- case-level finalization;
- Ground Truth evaluation.

`redact-video` now accepts `--categories face,license_plate,all`. With no category option it still defaults to the legacy face path. `--no-faces` without categories keeps the previous error behavior.

## Shared Contracts

`PrivacyDisposition`, `EvidenceDisposition`, `ReviewAction`, and `ScoringDisposition` now make the semantic namespaces explicit. Compatibility aliases normalize existing terms, while cross-namespace values raise errors instead of silently becoming policy.

The render plan consumes common `PrivacyRegionRecord` objects. Face records are adapted from frozen face render regions. Plate records are adapted from accepted plate render regions. Default face-only, explicit face-only, plate-only, and combined automatic previews now use the shared privacy renderer after render-plan validation. Face detector, tracking, recovery, policy, and review behavior remain category-specific.

## Face Shared-Renderer Parity Evidence

Face shared-renderer parity evidence is versioned to distinguish raw render parity, exact final-file span parity, and migration safety. Raw render parity compares the pre-encoding rendered frame content. Exact final-file parity uses sequential decode counts and terminal timestamps instead of metadata-only frame counts. Migration safety requires the shared renderer to preserve the complete source processing span, even when a frozen legacy output has a documented file-level defect.

The current known baseline defect is limited to the frozen legacy face mux path. The legacy face renderer final mux can drop the final video frame when `-shortest` terminates against the shorter stream. This was observed on the night evaluation clip: source and shared outputs sequentially decode to `1589` frames, while the frozen legacy output decodes to `1588`. The raw rendered frame content remained identical for the generated frames, and the shared automatic-preview path preserved the complete source span. The defect is not reproduced intentionally; the canonical face automatic-preview path no longer uses the defective mux boundary.

## Adapter Boundaries

The face adapter preserves:

- ADAS detector backend and thresholds;
- existing face policy modes and policy files;
- existing ordinary tracking and privacy gap recovery;
- existing face automatic preview semantics through the shared privacy renderer;
- existing face audit manifest output.

The plate adapter preserves:

- OIM YOLOv9-S-608 detector backend and thresholds;
- existing plate deterministic association, temporal evidence, and bounded interpolation;
- existing plate preview rendering wrapper;
- existing plate review bundle and final-export blocker behavior.

## Deliberate Category Differences

Faces and plates remain different where the object geometry and evidence model require it. Face full-video detection, tracking, privacy recovery, and policy stay on the frozen face path, while automatic preview rendering uses the shared privacy renderer. Plates continue to use plate geometry filters, accepted-track eligibility, temporal evidence, and bounded plate interpolation. The unified pipeline shares orchestration, manifests, and automatic-preview rendering, not detector math or category thresholds.

## Review And Finalization

The canonical `redact-video` command does not create one mixed review session, does not publish Latest reviewed, and does not run case-level finalization. Category review artifacts may be generated as automatic-stage evidence, but mixed review generation remains deferred.

`review_workspace`, `review_working_state`, `review_sessions`, `review_corrections`, and `review_finalization` remain the authority for reviewer decisions, latest reviewed state, correction provenance, full sweep acknowledgement, hash checks, and final reviewed rendering. Face reviewed/final rendering remains on the existing reviewed-rendering boundary.

The case manifest records these later phases as `not_run`, never `succeeded`.

## Evaluation

`src/redactor/evaluation.py` now has category evaluation adapters. The face adapter delegates to the existing face evaluator, preserving detector/automatic/reviewed-final separation and the rule that interpolation or reviewed/manual corrections do not improve raw detector metrics.

License-plate evaluation currently writes a capability report rather than scores. The machine-readable ground truth file is face-only, and the human-readable plate notes do not provide tight per-frame plate boxes suitable for detector or privacy-coverage scoring. Plate metrics stay null until machine-readable plate objects and scoring boxes exist.

## Compatibility

Existing commands remain available, including `detect-plates`, `track-plates`, `render-plates`, `build-plate-review-bundle`, `render-faces`, `evaluate-faces`, `face-pipeline`, and the final review workspace commands. The unified case pipeline is additive around those pieces.

Existing face-only `redact-video` output still includes `audit_manifest.json`; it now also writes `case_manifest.json`, category run manifests, source identity metadata, and render-plan artifacts. Multi-category runs add category-scoped plate outputs and a combined non-final automatic preview when effective regions exist and validation passes.

## Deferred Work

- Migrate reviewed/final face rendering after the automatic-preview migration remains stable.
- Add machine-readable plate ground truth before publishing plate scores.
- Generate one mixed review case, one Latest reviewed publication, and one mixed finalization result once reviewed plate corrections and entity-level UX are fully aligned.
- Add performance instrumentation for multi-category runs after behavior parity remains stable.
