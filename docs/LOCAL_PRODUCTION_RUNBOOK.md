# Local Production Runbook

This project is designed for local review on `127.0.0.1`. It is not hardened for remote or multi-user deployment.

## Build And Serve

```bash
cd /Users/anastasiasokolenko/Projects/video-redaction-pipeline
cd frontend
npm ci
npm run build
cd ..
PYTHONPATH=src .venv/bin/python -m redactor.cli build-final-review-workspace \
  --output-dir artifacts/final_pipeline_package_frontend_v1_5 \
  --frontend react \
  --react-reference-package artifacts/final_pipeline_package_frontend_v1_4_1 \
  --skip-frontend-build \
  --overwrite
PYTHONPATH=src .venv/bin/python -m redactor.cli serve-final-review-workspace \
  artifacts/final_pipeline_package_frontend_v1_5 --open
```

## Operational Checks

Before review:

- confirm the server is bound to `127.0.0.1`;
- confirm `legacy.html` is still available;
- confirm `/api/v1/schema` returns the checked-in schema version;
- confirm Day and Night expose Original, Automatic, and Latest reviewed;
- confirm MP4 Range requests return `206` for valid byte ranges.

During review:

- accepted object-policy, merge, correction, restore, or reset events should appear in History immediately;
- media-changing work should show an operation-specific pending state, not a disabled workspace;
- manual correction Save should show backend acceptance, then Latest reviewed publication progress, while preserving the active case and source frame;
- no-op/idempotent media work should clear pending state without starting a materialization job;
- failed publication should leave the last known-good Latest reviewed available and expose retry.

After review:

- validate the session hash chain;
- validate finalization blockers before export/finalization;
- keep generated package/session artifacts local unless a separate data-handling process approves transfer.

## Job Recovery

Jobs are scoped by session and case. Each queued media job records an accepted revision and target effective-region hash. Stale jobs cannot publish if the current effective-region hash has changed. If the server stops during a render, existing History remains append-only; the last known-good media remains published, and accepted state can be reloaded from the session event chain.

## Cleanup

Do not delete current, checkpoint-referenced, finalization-referenced, or audit-referenced session media. Superseded temporary render files may be cleaned after verifying they are not referenced by a materialization manifest, checkpoint, History event, or final manifest.

## Remote Deployment Requirements

Before any remote or multi-user deployment, add and validate:

- authentication and authorization;
- CSRF/origin controls;
- TLS;
- per-user/session data isolation;
- request and storage quotas;
- operational backup/restore;
- centralized logging policy that avoids raw pixels and sensitive payloads;
- security review for filesystem access and package data handling.
