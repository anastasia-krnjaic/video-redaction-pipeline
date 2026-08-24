# Final Review Workspace

The Final Review Workspace is the local browser workflow for reviewing Automatic media, saving human corrections, approving the exact current Latest reviewed revision, and publishing immutable Final media.

Python owns the authoritative state: workspace manifests, media roles, source-frame mapping, sessions, Guidance, entity policy, corrections, Latest reviewed pointers, full-video sweep acknowledgements, Final pointers, audit events, and validation. React presents that state and sends explicit actions through `/api/v1/`; it does not approximate finalization readiness or privacy policy locally.

## Launch

Serve a workspace with:

```bash
redactor serve-final-review-workspace artifacts/reviewed_final_combined_v2_1/run_20260813T210250Z/review_workspace --open
```

For a clean checkout, first generate a workspace with `redactor redact-video` and serve that generated `review_workspace/` directory.

The server binds locally, serves workspace files from the package root, supports MP4 Range requests, and enforces one writable server per workspace. Do not run two writable servers against the same workspace.

## Sessions

The server-owned current-session pointer is authoritative for normal resume. Opening a case without `session_id` resolves to the current session and updates the URL. Explicit `?session_id=...` URLs must load that exact session, and the header must match the URL.

Available session actions:

- New review session
- Resume review
- Reset this case to Automatic baseline

There is no Restart review action in the release UI.

Events are append-only and hash-chained. Derived JSON snapshots are rebuildable from the event log and are written atomically.

Validate and summarize a session with:

```bash
redactor validate-review-session --session-dir SESSION_DIR
redactor summarize-review-session --session-dir SESSION_DIR
```

## Case Views

The release workspace supports Day, Night, Final Results, and Audit Trail views when those cases exist in the workspace manifest. Each case can expose:

- Review
- Manual redaction
- Entities

The normal media roles are:

- Original: unredacted source media.
- Automatic: non-final machine-generated review media.
- Latest reviewed: immutable reviewed revision selected by `latest_reviewed/current.json`.
- Final: immutable final revision selected by `final/current.json`.

Historical Latest and Final revisions remain on disk but are not current if their pointers, source identity, session identity, reviewed-region hash, media hash, active projection, or sweep binding no longer match.

## Guidance

Guidance items are visible review hotspots. They remain auditable, source-bound, and expandable to the underlying task evidence.

Unreviewed Guidance alone does not block:

- reviewed-region compilation;
- Latest reviewed publication;
- full-video approval;
- Finalization.

Guidance is not silently converted to Correct. At full-video sweep and Finalization, the system records the count and deterministic identity of unreviewed Guidance covered by the explicit full-video review.

Optional QA remains separate from primary Guidance and is nonblocking unless a future policy explicitly marks it required.

## Needs Correction

Needs correction is blocking. A Guidance item becomes a blocking item only when a reviewer explicitly marks it Needs correction or when correction validation itself is invalid or pending.

The blocker clears only when:

- a valid correction is saved and verified; or
- the reviewer changes the authoritative adjudication to Correct; or
- the reviewer changes the authoritative adjudication to Not sensitive.

An active valid correction is not a blocker by itself. Blocking counts are derived from unresolved correction requirements, not from the mere existence of correction records.

## Entities

Entities are source-backed people and plates with stable case/category identity. Entity actions are authoritative review events and can reduce compatible Guidance:

- Redact preserves privacy-sensitive treatment.
- Keep visible applies the supported source-bound keep semantics.
- Not a person / Not a plate suppresses applicable false-positive regions and derives compatible Not sensitive task state.
- Needs review reopens visible human review work.
- Merge creates one canonical active entity and preserves source evidence.

Entity actions invalidate stale Latest and sweep state when they change reviewed media. They do not create hidden object-policy verification work.

## Manual Redaction

Manual redaction supports:

- Single frame
- Fixed box across interval
- Interpolate between keyframes

The editor works in source-frame coordinates. Preview generates real rendered privacy media for the draft and returns a token bound to the current case, session, source media, working-state revision, geometry, and preview output. The geometry overlay remains an outline; it is not the rendered privacy effect.

Save requires the current Preview token. A valid Save persists an immutable correction revision, updates authoritative working state, and marks reviewed media as pending when pixels change. Save does not normally queue or run a full-video Latest reviewed render by itself.

Reviewers may save one or more Manual corrections and media-affecting Entity actions before selecting **Build reviewed video**. That explicit build materializes the accumulated reviewed state, either by rendering a new Latest reviewed video or by reusing validated Automatic media when the reviewed state is pixel-identical to Automatic. Superseded jobs cannot publish stale media.

Discard clears the draft and preview token. Reset this case to Automatic baseline clears the selected case's active review and correction state while preserving append-only history.

## Latest Reviewed

Latest reviewed is an immutable revision built from:

- the Automatic baseline region set;
- active reviewer decisions;
- active entity policy;
- active manual corrections;
- source/category/config identities;
- renderer identity;
- the media projection hash.

The current pointer is valid only if it binds to the current session and active media projection. Audit-only events do not stale identical media, but media-affecting review changes do.

Open Guidance alone does not block building a new Latest reviewed. Active Needs correction, invalid corrections, stale source/config identity, media validation failure, region validation failure, pending media-affecting changes, or pointer mismatch do block full-video approval and Final creation until a validated current Latest reviewed revision exists.

## Full-Video Approval

Approve full-video review is the final human visual gate. The acknowledgement is current only when it is bound to the exact current Latest reviewed revision, reviewed-region-set hash, media identity, active projection, case, and session.

When a sweep succeeds, the UI refreshes authoritative server state before showing Current full-video review approved. A later media-affecting change leaves the historical approval intact but marks it stale.

## Create Final

Create Final validates the current Latest reviewed revision and current full-video approval. Finalization does not rerun detection, tracking, reviewed-region compilation, or privacy rendering. The Final media is the approved Latest reviewed media bytes, either hardlinked or verified as an exact copy, and is published through a case-scoped `final/current.json` pointer.

Historical Finals remain immutable. They are not reused or promoted unless the current pointer explicitly and validly selects them for the current case, session, Latest, sweep, media, and region identities.

## Downloads And Delivery

The browser exposes downloads for Original, Automatic, Latest reviewed, Final, and Final package zips when available. Delivery packages contain portable manifests and checksums for the Final video, source identity, case ID, Final revision, Latest revision, reviewed-region-set hash, sweep hash, session/event-chain head, validation status, current status, creation time, and the packaged per-redaction reviewed-region records.

Absolute local paths are not part of portable semantic identity.

## Technical Details

The checked-in API contract is [docs/api/final_review_workspace_v1.schema.json](api/final_review_workspace_v1.schema.json). Release evidence and canonical local workspaces live under ignored `artifacts/` roots and are indexed in [docs/RELEASE_EVIDENCE.md](RELEASE_EVIDENCE.md).
