# Performance And Production Hardening

Final Review Workspace v1.5 keeps the corrected v1.4.1 review semantics intact while improving manual-redaction persistence and reviewed-media planning diagnostics. v1.4 remains the no-op materialization hardening baseline; v1.4.1 is the corrected manual Save baseline.

## Frozen Boundaries

v1.5 does not change detector selection, detector thresholds, tracking, temporal evidence, gap recovery, Automatic privacy regions, source-frame mappings, render style, encoding policy, source videos, or finalization gates. Python remains the source of truth for working state, effective privacy regions, corrections, sessions, provenance, and media publication.

## Bottleneck

The v1.3.3 state acceptance path could enqueue Latest reviewed materialization for accepted mutations whose effective rendered privacy regions were unchanged. Examples included idempotent object-policy choices and lineage-only object changes. Full media publication is far more expensive than appending a review event, so this made some operations appear blocked by video work that could not change pixels.

## v1.4 Optimization

Every media-aware mutation now computes a deterministic effective privacy-region digest before and after the accepted state change. If the digest is unchanged and the matching media is already published, the backend records a skipped materialization audit row and returns no live job to the frontend. If the digest is unchanged but a compatible publication is already queued or running, the backend reuses that job instead of creating another render target. If the digest changes, the job is queued with a target effective-region hash and only compatible results may publish.

This classification applies to object-policy changes, bulk object policies, manual correction saves, object merges, checkpoint restores, and destructive baseline resets. Review decisions and checkpoint creation remain non-media mutations.

## Job Publication Guard

Materialization jobs carry:

- accepted working-state revision;
- target effective-region hash;
- target effective-region count;
- affected entity IDs;
- published working-state revision when completed.

A job may publish only when the current active state still has the job's target effective-region hash. This permits a render to survive later non-media review events, while preventing stale media-changing revisions from overwriting newer state.

## v1.4.1 Manual Save Correction

Manual redaction Save now validates the exact previewed draft before accepting it. Preview writes a token bound to the case, session, correction ID, source media hash, working-state revision, stable draft hash, and preview hash. Save rejects missing or stale tokens, persists an immutable correction revision, appends `manual_correction_saved`, creates the correction-verification task, adds the correction to the active projection, and schedules Latest reviewed publication when the effective-region hash changes.

Fixed-box interval corrections with one keyframe now repeat that one source-coordinate box across the inclusive source-frame interval. Single-frame and interpolated corrections keep their existing source-frame-keyed behavior.

## v1.5 Planning Result

v1.5 records exact changed source-frame intervals by comparing normalized per-frame effective-region hashes before and after accepted media-aware mutations. Jobs expose changed-frame counts, planned render/encode counts, current stage, and progress metadata to the frontend.

The full media renderer remains the publication path. Segment-level reviewed-media reuse was not enabled because a safe MP4 segment assembly path with complete parity has not been proven in this repository state.

## HTTP And Request Hardening

The local server keeps binding to `127.0.0.1` by default. JSON API bodies are bounded. Package file serving rejects path traversal through resolved package-root checks. MP4 Range serving supports normal ranges and suffix ranges, rejects invalid/multi/out-of-bounds ranges with `416`, and does not serve temporary render files as published media.

## Evidence

v1.4 evidence is generated under:

```text
artifacts/performance_production_hardening_v1_4/
```

v1.4.1 and v1.5 evidence is generated under:

```text
artifacts/manual_redaction_save_fix_v1_4_1/
artifacts/incremental_materialization_v1_5/
```

Key files:

- `baseline_environment.json`
- `baseline_package_hashes.json`
- `baseline_metrics.json`
- `optimized_metrics.json`
- `before_after_summary.md`
- `non_regression_audit.json`
- `non_regression_audit.md`

The benchmark harness is:

```bash
.venv/bin/python scripts/final_review_performance_harness.py \
  --source-root src \
  --iterations 5 \
  --label optimized_v1_4 \
  --output-json artifacts/performance_production_hardening_v1_4/optimized_metrics.json \
  --output-md artifacts/performance_production_hardening_v1_4/optimized_metrics.md
```

## Rejected Optimizations

v1.5 does not implement interval-local rendering, segment assembly, frame skipping, lower-quality encoding, detector/tracker changes, approximate source-frame seeking, or reduced verification coverage. Segment and cache artifacts document the rejected/deferred approach rather than weakening parity.

## Remaining Gaps

The local app is still a localhost review tool, not a remote multi-user service. Remote deployment would require authentication, authorization, user/session isolation beyond local package sessions, CSRF and origin policy, TLS, operational logging policy, capacity planning, backup/restore policy, and a separate privacy/security review.
