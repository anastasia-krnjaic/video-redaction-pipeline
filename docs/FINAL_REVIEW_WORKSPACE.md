# Final Review Workspace

The Final Review Workspace is the local, case-based review and finalization layer for the frozen face and plate prototype behavior.

It is generated under:

```text
artifacts/final_pipeline_package_v2/
```

The frozen v2.3.4 backend reference package is generated under:

```text
artifacts/final_pipeline_package_v2_3_4/
```

Frontend Foundation v1 generates a React + TypeScript package under:

```text
artifacts/final_pipeline_package_frontend_v1/
```

Frontend v1.5 is generated under:

```text
artifacts/final_pipeline_package_frontend_v1_5/
```

It does not change detectors, thresholds, tracking, evidence accumulation, recovery, or rendering configuration. It gives a reviewer one case at a time, a single versioned source-frame-aware player, timeline-linked tasks, local correction records, stable Person/Plate review objects with representative thumbnails, an append-only audit trail, and finalization gates.

The unified automatic case pipeline also writes a package under each case output:

```text
<case-output>/review_workspace/
```

That generated package reuses the same local session, entity, task, correction, and media-serving contracts. It contains one case tab for the source case, mixed Person/Plate entities when both categories were selected, original plus automatic-preview media roles, one normalized task queue, media inventory, validation, and an empty `sessions/` namespace. Starting review still happens explicitly through the workspace UI or session API.

For new case-generated packages, `workspace_manifest.json` is the only runtime workspace authority. The case manifest points to and hashes that file, and the server binds new sessions to the workspace manifest, source, automatic preview, category-run identity mapping, entity registry, task index, media inventory, validation report, and per-case content hashes. New sessions are rejected if those identities no longer match the package on disk.

New canonical face-only, plate-only, and mixed packages carry `latest_reviewed_publication.status = supported`. Reviewers start a session, record decisions, preview corrections, and save preview-token-bound corrections. Latest reviewed is published only by the backend reviewed-region compiler, which combines the automatic baseline regions, active decisions, object policies, merges, and active corrections into one reviewed region set, then renders from Original and publishes immutable revision artifacts after full-span media and rendered-region validation pass.

The current Latest reviewed revision has one authority: `latest_reviewed/current.json`. That pointer is written by same-directory temporary file plus atomic replace only after the immutable revision directory, media, reviewed regions, validation files, and revision manifest reopen and validate. Workspace and case manifests do not duplicate mutable current revision state; readers resolve Latest reviewed from the pointer and reject it when the pointer, revision manifest, workspace hash, session, source hash, media hash, or active render-state projection no longer match.

The active render-state projection is a media identity, not the session audit head. Its current hash schema, `active_projection.v2`, includes the review state that can change compiled reviewed pixels and excludes sweep acknowledgements, finalization attempts, materialization job IDs, timestamps, paths, and other audit-only bookkeeping. Older pointers without an explicit schema are treated as `active_projection.v1` and validate only through the declared legacy hash contract, with compatibility limited to unchanged media-affecting state. Unknown projection hash schemas fail closed.

The current Final revision also has one authority: `final/current.json`. Canonical finalization consumes the current Latest reviewed pointer and validates the exact immutable revision behind it. It does not run detection, tracking, reviewed-region compilation for output construction, privacy rendering, FFmpeg encoding, or OpenCV video writing to create Final. If a separate final MP4 is materialized, it is a hardlink or verified copy of the Latest reviewed MP4 and must have the same SHA-256.

The reviewed-region hash is semantic, not an audit-record hash. Correction records still keep generated `correction_id`, `revision_id`, timestamps, reviewer metadata, preview tokens, and event lineage for append-only history and session hash chains. The compiler hashes a separate semantic correction projection plus the selected source/category/config identities, effective policy, merge/rejection state, active reviewed geometry, and renderer behavior. Different audit histories can therefore share the same `reviewed_region_set_sha256`, while their publication manifests and session heads remain distinct.

## Launch

Serve the package locally:

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli serve-final-review-workspace \
  artifacts/final_pipeline_package_frontend_v1_5 --open
```

The server binds to `127.0.0.1`, serves only files under the package root, supports MP4 range requests for seeking, and writes review sessions only under the package `sessions/` directory.

Opening `index.html` directly may load static files, but the local server is the supported workflow because it enables reliable video seeking and durable local session writes.

The review app serves MP4 files with `video/mp4` MIME type and HTTP range
requests so Firefox and Chromium can seek reliably. Review videos use inline
playback with Picture-in-Picture disabled. Fullscreen actions open the custom
player wrapper, not the raw MP4, so the source-frame HUD, timeline, and
keyboard frame stepping remain available.

React assets are served from `/app/`. The legacy generated v2.3.4 workspace is
retained as a fallback at `/legacy.html`.

## React Frontend And API v1

Frontend Foundation v1 moves presentation into `frontend/` with React,
TypeScript, Vite, a committed npm lockfile, and Playwright E2E for Chromium and
Firefox. Python remains the backend and source of truth for manifests, media,
source-frame mapping, review tasks, entities, sessions, decisions, corrections,
correction previews, audit events, finalization, and Range serving.

The stable contract prefix is:

```text
/api/v1/
```

The checked-in schema is:

```text
docs/api/final_review_workspace_v1.schema.json
```

Legacy `/api/...` endpoints remain available during this milestone.

## Case Tabs

The workspace builds case tabs from `workspace_manifest.json` and validates every case ID before routing or DOM lookup. The frozen package has:

- Day
- Night
- Final Results
- Audit Trail

A unified automatic case package has one tab for its safe case ID plus Final Results and Audit Trail. The root page redirects to the first validated discovered case rather than hardcoding `day` or `night`. The same dynamic tab contract also supports future multi-case packages.

Each primary case has two modes plus the object registry:

- Review
- Manual redaction
- Entities

Review-card and timeline-marker clicks select the same task and seek the shared player to the task source-frame start. After a persisted Correct, Needs correction, or Unsure decision, the UI selects the next remaining Open task in source-frame order; when no Open tasks remain, it shows the explicit next-step completion state instead of wrapping.

Manual redaction drafts are keyed only by source frame. The interval modes are Single frame, Fixed box across interval, and Interpolate between keyframes. Drawing happens directly over the main video frame and immediately updates the source-coordinate keyframe list and validation state.

Entity merges are backend-owned append-only events. The React modal selects multiple same-case, same-category entities, warns on overlap, and the backend creates one canonical active entity while retiring the selected aliases. Merged entities get stable case/category-scoped labels such as `Merged Person 01` and inherit a shared policy when all source entities agree; conflicting policies become Needs review. Entity thumbnail jumps pause the player, seek to the representative source frame, and show a temporary red source-bbox focus outline for source-backed redacted, kept-visible, needs-review, unresolved, merged, person, and plate entities. The focus outline is not rendered into media.

Object policy changes, merges, correction saves, checkpoint restores, and destructive baseline resets return accepted working-state revisions plus local materialization job IDs only when media publication is required. The workspace computes deterministic reviewed privacy-region hashes before and after each media-aware mutation. Canonical workspaces publish through the reviewed-region compiler; legacy deferred workspaces still return `publication_status: deferred` and queue no job. If a legacy workspace hash is unchanged and current media is already compatible, media rendering is skipped and audited; if a compatible publication is already queued or running, that job is reused. v1.5 also records the exact source frames whose normalized effective region sets changed and exposes that plan through job progress. Publication still uses the proven full-render path unless a future segment renderer proves output parity.

Queued materialization jobs capture the accepted render-state revision, session head, target reviewed-region hash, case ID, workspace manifest hash, category-run baseline identity, renderer identity, and a materialization generation. Reset, restore, newer media-affecting decisions, policies, merges, and corrections supersede queued or running jobs. Durable correction-verification task scaffolding remains in audit/finalization state but does not by itself make the just-published media stale. Explicit materialization retry on a legacy deferred workspace returns `latest_reviewed_publication_deferred` and does not queue a job.

Sweep acknowledgement and canonical finalization events affect approval eligibility and the session hash chain. They do not change active media projection identity unless they also carry a media-affecting review mutation; a stale pointer still stays hidden when decisions, policies, merges, correction geometry, or reset generation change the compiled reviewed regions.

`Restart review` is a narrow case-scoped workflow action. It clears active review decisions, task buckets, task selection, sweep acknowledgement, and transient review progress for the selected case while preserving entity policy overrides, canonical merges, manual corrections, checkpoints, History, and immutable Latest reviewed artifacts; supported-canonical readers still hide any pointer that no longer matches the active render-state projection. `Reset this case to Automatic baseline` is the separate destructive case-scoped action that removes the active case's entity edits, merges, manual corrections, reviewer-created manual issues, review decisions, sweep acknowledgement, transient correction/editor state, and active Latest reviewed projection while preserving append-only History, checkpoints, source media, Automatic baseline, and the other case.

Review mode shows one large video player with explicit media roles:

- Original is the source media.
- Automatic is the non-final automatic preview.
- Correction preview is a transient, task-local diagnostic render returned by Preview. It is not persisted as a normal version, never fills a `latest_reviewed` field, never satisfies finalization, and remains separate after Save.
- Latest reviewed appears only after the compiler publishes a validated immutable revision and `latest_reviewed/current.json` points to it for the current session projection.
- Final appears only after explicit finalization publishes `final/current.json` for the same current Latest reviewed revision and matching sweep acknowledgement.
- Stale or failed revisions remain historical artifacts only; they are not shown as current Latest reviewed.
- A previous Final remains immutable history after a later decision, correction, merge, or reset, but it is not shown as current Final for the changed review state.
- Legacy packages that already publish Latest reviewed retain their existing version selector behavior.

- Original
- Automatic
- Latest reviewed
- Final

Before finalization, the normal selector shows Original, Automatic, and Latest
reviewed when the reviewed pointer is current. After finalization, Final is
added only through the valid final pointer. Switching versions preserves the
active source frame, and a package build fails if the displayed versions do not
share source-frame mapping, frame count, FPS, resolution, and decoded-frame
evidence at the audited sample frames.

Manual redaction Preview is transient. It does not create a persistent
Correction preview version and does not create a verification task until Save.
Preview records a hash-chained preview event and a token bound to the current
case, session, source media, working-state revision, draft hash, and preview
hash. Save validates that token, persists an immutable correction revision,
appends `manual_correction_saved`, creates the verification task, schedules
Latest reviewed materialization, and preserves the current source frame.
The public browser flow calls Preview before Save and sends the returned
`preview_token` with the exact correction draft. Missing, stale, cross-case, or
changed-draft tokens are rejected.

Long raw review intervals are preserved in audit lineage but projected into
non-overlapping reviewer-facing child tasks. The chunking target is about three
seconds and the hard maximum is five seconds, with child task IDs and timeline
markers retaining the parent raw signal, parent source-frame span, child index,
and exact child source-frame span.

Diagnostic face-only, plate-only, provenance, and raw-signal details stay behind technical sections instead of cluttering the default review view.

## Current Day Boundary

The previous day face and day plate outputs came from different extracted clips. Their metadata now maps both clips to a common parent source, `artifacts/day_primary.mp4`, and the v2.3 package extracts one canonical Day original interval from that parent source.

The combined Day current result is not fabricated from incompatible encoded clips. It is rendered from lineage-proven frozen evidence mapped into the canonical Day source-frame interval. The build validates the local ADAS XML/BIN and OIM ONNX hashes before packaging.

## Review Objects

The workspace indexes automatic and reviewed region tracks into simple reviewer-facing objects such as `Person 01` and `Plate 01`. Raw detector track IDs remain in Technical details. Each object has a deterministic representative thumbnail, first-seen frame, status badge, and compact reviewer actions:

- Redact
- Keep visible
- Not a person / Not a plate
- Merge with another object

These actions are separate from technical overlay and geometry state. Keep and
Not-a-person/Not-a-plate append an event, create a verification item, and block
finalization until verified. They do not require a browser prompt or mandatory
reason field. Merge is same-category only and is rejected when entities are
simultaneously visible.

## Review Sessions

Click **Start new review** to create an anonymous server-backed local session. A previous local session is not loaded silently from browser storage; resume by opening an explicit `?session_id=...` URL or using the resume control in the React app. The server assigns a session UUID and UTC start time automatically. Reviewer name is required only when exporting or finalizing; the workspace appends a `reviewer_attribution_set` event and does not rewrite earlier events.

The reviewer identity is locally declared, not enterprise authentication. Changing attribution appends another event instead of editing history.

The server creates:

```text
sessions/<session_id>/review_session.json
sessions/<session_id>/events.jsonl
sessions/<session_id>/decisions_current.json
sessions/<session_id>/corrections.jsonl
sessions/<session_id>/review_summary.json
```

Events are append-only and hash-chained. If a decision changes, a new event is appended and `decisions_current.json` is updated as a derived current-state view.

Validate a session with:

```bash
redactor validate-review-session --session-dir SESSION_DIR
```

Summarize it with:

```bash
redactor summarize-review-session --session-dir SESSION_DIR
```

## Decisions

Review mode uses task-specific action sets instead of one generic button list.
Coverage-risk tasks offer actions such as Looks correct, Needs correction, Not
sensitive, and Unsure. Possible false-redaction tasks offer actions such as
Redaction is correct, Remove redaction, Wrong target, and Unsure. A decision is
saved to the local server before auto-advance.

Reviewer-facing actions are plain-language:

- Looks correct
- Privacy leak
- Partial coverage
- False redaction
- Wrong target
- Not sensitive
- Cosmetic only
- Uncertain / Escalate

Only accepted, corrected-and-verified, not-sensitive, or documented cosmetic-only decisions make a task green. Leak, partial-coverage, false-redaction, wrong-target, and escalation decisions remain blockers until resolved.

## Corrections

Corrections are a separate reviewed layer. They do not change detector, tracker, evidence, recovery, or render thresholds.

Supported correction records include:

- add privacy region;
- replace privacy region;
- suppress existing region;
- suppress track segment;
- mark region not sensitive;
- mark interval not visible or occluded.

Corrections are tied to a case, task, review object, source-video hash, source-frame interval, reviewer identity, reason, keyframe boxes, checkpoint verification states, interpolation method, application version, and configuration freeze hash. Interpolation is bounded to reviewed keyframes inside the task interval; unbounded extrapolation is rejected and unreviewed checkpoints block finalization.

Suggested keyframes are deterministic review aids. Intervals up to 0.5 seconds use start/end, intervals up to 2 seconds add a midpoint, longer intervals use checkpoints no farther than about one second apart, and intervals over 5 seconds are split into 5-second review pages. These suggestions do not alter detector, tracker, or recovery behavior.

## Finalization

Finalization is allowed only when:

- every required task is green;
- every correction has been verified;
- no Uncertain / Escalate task remains;
- the full-video sweep has been explicitly acknowledged for the exact current Latest reviewed revision and video hash;
- the session event hash chain is valid;
- source hashes and configuration-freeze hashes match;
- frame spans are complete;
- output media validation passes;
- rendered-region verification passes.

For new canonical workspaces, finalization also requires an authoritative Latest reviewed publication whose reviewed-region hash still matches the active media-affecting review projection. The sweep acknowledgement record binds the case ID, source SHA-256, session ID, Latest reviewed revision ID, Latest reviewed video SHA-256, reviewed-region set hash, and workspace manifest SHA-256. A sweep for an older revision cannot approve a newer reviewed output. Correction-only preview media is never used as the finalization candidate.

Canonical finalization writes one immutable revision under `final/revisions/<finalization_id>/` and publishes `final/current.json` only after the final video hash equals the Latest reviewed video hash and the publication guard still matches. Repeating finalization for the same approved state reuses the existing finalization input identity instead of overwriting an immutable revision.

Finalize with:

```bash
redactor finalize-reviewed-case \
  --package-root artifacts/final_pipeline_package_v2_3 \
  --case-id CASE_ID \
  --session-dir SESSION_DIR
```

The result may be labeled “Reviewed final for this source case.” It must not be labeled zero-leak, guaranteed private, or universally production-ready.
