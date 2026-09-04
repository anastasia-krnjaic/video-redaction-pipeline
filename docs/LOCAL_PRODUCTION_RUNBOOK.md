# Local Production Runbook

This project is designed for local processing and review on `127.0.0.1`. It is not a remote multi-user service.

## Preflight

```bash
source .venv/bin/activate
redactor doctor
npm --prefix frontend run build
```

Before serving a writable review workspace, confirm no other writable server owns it:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
lsof -nP -iTCP:8766 -sTCP:LISTEN
```

The application also enforces a per-workspace single-writer lock. If a previous process was killed, use the normal server startup path and let stale-lock validation decide whether the lock is reclaimable.

## Serve A Workspace

For a locally available release workspace:

```bash
redactor serve-final-review-workspace path/to/release_review_workspace --open
```

For a newly generated workspace:

```bash
redactor serve-final-review-workspace path/to/example_automatic_preview/review_workspace --open
```

Use one writable server per workspace. Open additional browser tabs against the same server rather than starting a second server on the same workspace.

## During Review

Operational checks:

- the header session ID matches the URL session ID;
- Guidance is advisory and auditable;
- Needs correction is visibly blocking;
- Manual Preview shows rendered privacy pixels;
- Save correction shows saved/applying state until authoritative refresh;
- Latest reviewed appears only when its pointer is current;
- full-video approval is current only for the exact Latest reviewed revision;
- Final appears only through the current Final pointer.

## After Review

Validate session state before relying on it:

```bash
redactor validate-review-session --session-dir SESSION_DIR
redactor summarize-review-session --session-dir SESSION_DIR
```

Release delivery packages should be verified by checksum and extraction before transfer.

## Recovery Notes

Review events are append-only and hash-chained. Derived snapshots, Latest pointers, sweep state, and Final pointers are rebuildable or auditable from source state. If a server stops during materialization, the last valid current pointer remains authoritative until a compatible job publishes atomically.

Do not manually edit session events, current pointers, Latest revisions, Final revisions, or delivery manifests.

## Remote Deployment Requirements

Before any remote deployment, add and validate authentication, authorization, TLS, CSRF/origin controls, per-user data isolation, quota handling, operational backup/restore, and logging policy that avoids sensitive pixels and payloads.
