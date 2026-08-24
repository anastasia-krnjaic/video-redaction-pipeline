# Unified Automatic Case Preview Orchestration

This document records the backend refactor that moves full-video automatic privacy-preview orchestration from separate face and plate command paths toward one case-level checkpoint. It does not change detector selection, detector thresholds, tracker behavior, rendering semantics, review policy, or finalization gates.

The canonical `redact-video` command produces automatic, non-final preview evidence, a fail-closed case artifact manifest, and a mixed review workspace package whose review session starts only when a reviewer opens review. Reviewed publication and finalization remain explicit review-session actions, not automatic side effects of the command.

## Verified Before-State

This refactor began from a frozen earlier review-workspace checkpoint. That
checkpoint is retained in Git history, but it is not the current release
workflow contract. Current reviewer behavior is documented in
`docs/FINAL_REVIEW_WORKSPACE.md`.

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
-> REVIEW_WORKSPACE
-> REVIEW_SESSION
-> LATEST_REVIEWED
-> FINALIZATION
-> EVALUATION
```

Each selected category runs through an adapter. The adapters wrap the existing face and plate implementations rather than replacing them. The pipeline writes one authoritative case source identity, category run manifests, a unified render plan, effective privacy-region JSONL, a validated combined automatic preview when multiple selected categories have effective regions, one mixed review workspace, and `case_manifest.json`.

Executed by the canonical command:

- ingest and source identity publication;
- category detection;
- category temporal processing;
- face policy and effective privacy-region conversion;
- category review-task evidence;
- automatic preview rendering;
- mixed review workspace generation;
- fail-closed case manifest publication.

Represented but not executed by the automatic command:

- review session start and human decisions;
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

The canonical `redact-video` command now creates one mixed review workspace under `review_workspace/`. The workspace consumes case-level automatic artifacts, not detector internals: original media, the authoritative automatic preview, category review task files, the effective render plan, and category run lineage. It normalizes face and license-plate review tasks into one deterministic `reviewer_tasks.json` queue, writes one Person/Plate review entity registry, writes media inventory and validation artifacts, and keeps one `sessions/` namespace for future local review history.

`review_workspace/workspace_manifest.json` is the sole runtime authority for new case-generated workspaces. `case_manifest.json` records and hashes that file rather than a separate mixed manifest. The workspace manifest binds the case content identity, source hash, automatic-preview hash, renderer backend, category run manifest identities, entity registry, task index, media inventory, validation report, session namespace, blockers, and lifecycle statuses. The category-run identity map contains exactly one hashed `category_run_manifest.json` identity per selected category, and workspace creation fails closed if a selected run is missing, stale, for a different case/source/category/configuration, has an incompatible processed span, or reports failed run stages. The reviewed-region compiler revalidates that direct category-run/config binding before publication and also binds the automatic baseline artifact hash, semantic baseline region-set hash, selected category set, source hash, processing span, and category-run identity hash. The top-level case manifest mirrors the same identity hash.

Entities are source-backed temporal targets. Face records come from face policy and track census artifacts, including redact, keep-visible, and review-required tracks; plate records come from accepted and review-required plate tracking artifacts. Rejected wrong-class or inactive duplicate proposals are excluded from active reviewer entities. Tasks are normalized only after entity binding, preserve category-specific allowed actions, and reject invalid intervals, out-of-span frames, duplicate task IDs, unsupported actions, or unbound generic intervals.

The primary media pair is the original source and the authoritative automatic preview. Both roles carry required hashes, frame spans, frame count, dimensions, rational FPS, and source-frame mapping. A zero-region case materializes a distinct no-op automatic-preview artifact by hardlink or copy with `noop_source_copy` lineage instead of aliasing the original source path.

Case IDs use the shared conservative grammar `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` before filesystem joins, route lookup, entity-event construction, session mutations, reset/restore, correction handling, finalization, and browser tab generation. Correction saves keep the preview-token flow: public clients preview first, receive a token bound to the draft and current session revision, and save with that token. Preview media is always a `correction_preview` diagnostic and is never surfaced as Latest reviewed. New canonical workspaces advertise `latest_reviewed_publication.status = supported`, so saved Manual corrections and media-affecting Entity actions mark reviewed media as pending until the reviewer explicitly selects **Build reviewed video**. Legacy deferred workspaces retain the no-publication path, and all queued or running build jobs retain the materialization guard so resets or newer media-affecting mutations supersede stale authoritative media.

Review session creation is deliberately not automatic. `REVIEW_WORKSPACE` is recorded as `ready`, while `REVIEW_SESSION` is recorded as `not_started` until a reviewer starts a session. New canonical face-only, plate-only, and mixed workspaces record Latest reviewed, finalization, and evaluation as `not_run` in the generated workspace manifest. Runtime Latest reviewed readiness is derived from the atomic `latest_reviewed/current.json` pointer and its immutable revision manifest, not from mutable manifest caches. If the active render-state projection changes after publication, readers stop treating the old pointer as current until a matching revision publishes. The active projection hash is versioned separately from the session audit head: `active_projection.v2` excludes non-media sweep and finalization bookkeeping, while pre-schema pointers validate only through explicit `active_projection.v1` compatibility and unknown schemas fail closed. Canonical finalization resolves that current pointer, requires a revision-bound full-video sweep acknowledgement, revalidates source/config/category lineage, task and correction gates, media validation, region validation, and pending-publication state, then materializes Final by hardlinking or copying the exact Latest reviewed MP4 bytes. The final approval is published through `final/current.json`; failed, partial, or stale attempts are never exposed as current Final. Finalization and evaluation remain separate explicit lifecycle steps.

Reviewed-region identity is deliberately semantic. The compiler excludes generated correction IDs, revision IDs, timestamps, event IDs, preview paths, and preview-token artifacts from `reviewed_region_set_sha256`, while retaining those values in append-only correction records and session hash chains. The semantic identity includes source identity, selected categories, category-run/config binding, automatic baseline semantics, effective keep/redact policy, merge/rejection state, semantic corrections, final reviewed geometry, and renderer configuration where it affects pixels.

`review_workspace`, `review_working_state`, `review_sessions`, `review_corrections`, and `review_finalization` remain the authority for reviewer decisions, latest reviewed state, correction provenance, full sweep acknowledgement, hash checks, and final reviewed rendering. Face reviewed/final rendering remains on the existing reviewed-rendering boundary.

The generated case manifest starts Latest reviewed, finalization, and evaluation as `not_run`. Runtime Latest reviewed may later resolve to `ready` through the current pointer and immutable revision manifest. Runtime Final may later resolve to `ready` only through `final/current.json`, whose manifest binds the approved Latest reviewed revision, reviewed-region set, source hash, session projection, sweep acknowledgement, validation artifacts, finalization input hash, and byte-identical final video hash.

## Evaluation

`src/redactor/evaluation.py` now has category evaluation adapters. The face adapter delegates to the existing face scoring path, preserving detector/automatic/reviewed-final separation and the rule that interpolation or reviewed/manual corrections do not improve raw detector metrics.

License-plate evaluation has a capability-only default plus explicit detector, automatic, and reviewed-final modes. The plate scoring path is policy-bound and fail-closed; real certified plate metric execution remains a separate explicit phase from code/policy implementation.

## Compatibility

Existing commands remain available, including `detect-plates`, `track-plates`, `render-plates`, `build-plate-review-bundle`, `render-faces`, `evaluate-faces`, `face-pipeline`, and the final review workspace commands. The unified case pipeline is additive around those pieces.

Existing face-only `redact-video` output still includes `audit_manifest.json`; it now also writes `case_manifest.json`, category run manifests, source identity metadata, render-plan artifacts, and `review_workspace/`. Multi-category runs add category-scoped plate outputs and a combined non-final automatic preview when effective regions exist and validation passes.

## Deferred Work

- Migrate reviewed/final face rendering after the automatic-preview migration remains stable.
- Run the frozen plate scoring path on audited real plate artifacts before publishing plate scores.
- Clean up legacy reviewed/final renderer boundaries once canonical finalization remains stable.
- Add performance instrumentation for multi-category runs after behavior parity remains stable.
