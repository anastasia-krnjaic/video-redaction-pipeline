# Performance And Production Hardening

This document is retained as a compatibility entry point for older links. The current release summary is [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

## Frozen Boundaries

The production-readiness work preserved:

- detector outputs;
- tracker outputs;
- renderer configuration;
- Ground Truth;
- evaluation policy;
- current Final media identities;
- delivery package bytes.

It optimized local orchestration, status paths, validation memory behavior, materialization scheduling, restart behavior, telemetry, and storage classification.

## Current Highlights

- Warm finalization-status requests are about 0.21 seconds on the release machine.
- Cold finalization-status requests are about 1.6-2.7 seconds for Day and 1.0-1.2 seconds for Night on the release machine.
- Synthetic review-bundle validation peak RSS improved from 813.31 MB to 202.06 MB.
- Single-writer review serving, atomic JSON/event writes, caches, no-op materialization, heavy-job admission control, resumable stages, and telemetry are part of the frozen release.

Detector inference remains the major heavy workload and was not semantically optimized in the final production pass.

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for the current release contract.
