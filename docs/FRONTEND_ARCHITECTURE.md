# Frontend Architecture

The Final Review Workspace frontend is a React/Vite presentation layer backed by the Python review server. The architecture keeps privacy, review, and finalization authority in Python while making the browser workflow responsive and source-frame aware.

## Ownership Boundary

Python owns:

- workspace, source, media, and case manifests;
- session creation and current-session resume;
- append-only event logs and derived snapshots;
- Guidance, Optional QA, Needs correction, and grouping projection;
- entity policy and merges;
- correction Preview and Save validation;
- reviewed-region compilation and materialization;
- Latest reviewed, sweep, and Final currentness;
- media serving and HTTP Range requests;
- finalization readiness and validation.

React owns:

- layout and route presentation;
- visible filters and selected card state;
- local manual-correction draft UI state before submission;
- frame-aware player controls;
- status display from backend-authoritative responses.

React does not own detector, tracker, renderer, task/group semantics, finalization readiness, or persistence truth.

## App Structure

The main UI concepts are:

- case tabs for Day, Night, Final Results, and Audit Trail;
- media role selector for Original, Automatic, Latest reviewed, and Final;
- large source-frame-aware player;
- Guidance queue with optional QA separated;
- Entity panel;
- Manual redaction editor;
- status panels for Latest reviewed, full-video approval, Final, and downloads.

## API Contract

The stable checked-in schema is:

```text
docs/api/final_review_workspace_v1.schema.json
```

The server exposes the runtime schema at `/api/v1/schema`. Frontend types are generated into `frontend/src/types/api.ts`.

## State Refresh Contract

Successful state-changing API calls must update React from authoritative server state. The frontend may show pending UI while a request is in flight, but it must not synthesize durable approval, readiness, Latest, Final, correction, or entity state locally.

Examples:

- correction Save waits for authoritative working-state/materialization updates;
- sweep approval waits for refreshed authoritative state before showing current approval;
- finalization status is read from the backend endpoint;
- media-affecting changes invalidate stale Latest and sweep state through server projection.

## Media

The browser uses normal media URLs served by Python. Range requests are required for reliable seeking. The player maps UI frame state to source-frame records supplied by the backend and should not pause playback merely because current-frame state changes.

Manual Preview displays generated privacy-rendered media. Geometry overlays are outlines and must not masquerade as privacy output.

## Build Identity

Workspaces include frontend build identity metadata. The server and Technical/About UI expose the served build identity so stale embedded assets can be detected. HTML and manifests are revalidated; hashed static assets may be cached immutably.
