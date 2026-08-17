# Production Readiness

This document summarizes the frozen production-readiness architecture and measured release results. It does not describe detector inference as optimized beyond the already frozen behavior.

## Operational Model

The review workspace is a local server-backed workflow. Python remains the source of truth for sessions, events, pointers, validation, rendering, and finalization. React is the presentation layer and consumes authoritative server state.

The release is ready for local production-style operation with one writable server per workspace. Remote or multi-user deployment requires additional authentication, authorization, network, retention, monitoring, and privacy controls.

## Safety Architecture

- Single-writer lock for each writable workspace.
- Server-owned current-session pointer.
- Append-only hash-chained events.
- Strict JSON parsing for durable state.
- Atomic temp-file, flush, fsync, and replace for state snapshots and pointers.
- Case-scoped Latest reviewed and Final pointers.
- Revision-bound full-video sweep before Finalization.
- Fail-closed validation for stale Latest, stale sweep, invalid corrections, missing media, region mismatch, and source/config mismatch.

## Performance Work

The final production pass optimized orchestration and review/post-processing paths without changing detector outputs, tracker outputs, renderer configuration, Ground Truth, evaluation policy, current Final identities, or delivery package bytes.

Implemented production hardening:

- immutable SHA-256 and FFprobe metadata cache;
- event projection cache keyed by session ID and event-chain head;
- reviewed-region no-op reuse when media projection is unchanged;
- no-op and coalesced materialization;
- resource-aware heavy-job semaphore;
- resumable stage manifests;
- structured run telemetry;
- streaming review-bundle/media validation.

## Measured Release Results

Finalization-status requests:

| Case | Cold | Warm |
| --- | --- | --- |
| Day | about 1.6-2.7 s | about 0.21 s |
| Night | about 1.0-1.2 s | about 0.21 s |

Synthetic review-bundle validation memory profile:

| Profile | Peak RSS |
| --- | ---: |
| Before | 813.31 MB |
| After | 202.06 MB |
| Reduction | 611.25 MB |

The measurements are release-machine observations, not portable service-level objectives. Filesystem, CPU, codec, and media size can change absolute timings.

## Resource Admission

Heavy jobs are admitted through an explicit queue/semaphore so status requests, workspace reads, case switching, and media-role lookups remain responsive while rendering or validation work runs. Cross-case jobs remain case-local, and a stale render cannot publish over a newer media projection.

## Restart Behavior

Expensive stages can persist a manifest that binds source/config identity, input hashes, output hashes, and implementation identity. On restart, only complete matching stages are reused. Partial output is not promoted.

If the server stops during a job, human events remain durable and hash-chained. The current pointers remain the last valid committed state until a compatible job completes and publishes atomically.

## Telemetry

Structured telemetry records stage start/end, wall time, CPU time where available, peak RSS where available, input/output counts, cache hits and misses, bytes read/written, queue wait, and result status. It does not log sensitive frame content.

## Storage

The cleanup pass removed obsolete ignored artifacts only:

| Measure | Bytes |
| --- | ---: |
| Before cleanup | 75,281,506,304 |
| After cleanup | 46,294,134,784 |
| Removed | 28,987,371,520 |
| Reduction | 38.5053% |

No tracked source code was removed during cleanup.

## Deferred Work

Detector inference remains the major heavy workload. The final production pass did not change detector thresholds, tracker semantics, renderer semantics, Ground Truth, evaluation policy, current Finals, or delivery packages.
