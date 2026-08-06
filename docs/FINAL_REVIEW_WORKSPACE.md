# Final Review Workspace

The Final Review Workspace is the local, case-based review and finalization layer for the frozen face and plate prototype behavior.

It is generated under:

```text
artifacts/final_pipeline_package_v2/
```

The current v2.2 package is generated under:

```text
artifacts/final_pipeline_package_v2_2/
```

It does not change detectors, thresholds, tracking, evidence accumulation, recovery, or rendering configuration. It gives a reviewer one case at a time, clear tasks, local correction records, stable Person/Plate review objects with representative thumbnails, an append-only audit trail, and finalization gates.

## Launch

Serve the package locally:

```bash
redactor serve-final-review-workspace artifacts/final_pipeline_package_v2_2 --open
```

The server binds to `127.0.0.1`, serves only files under the package root, supports MP4 range requests for seeking, and writes review sessions only under the package `sessions/` directory.

Opening `index.html` directly may load static files, but the local server is the supported workflow because it enables reliable video seeking and durable local session writes.

The review app serves MP4 files with `video/mp4` MIME type and HTTP range
requests so Firefox and Chromium can seek reliably. Review videos use inline
playback with Picture-in-Picture disabled. Fullscreen actions open the custom
player wrapper, not the raw MP4, so source-frame, case-frame, timestamp, case
name, role, and frame-step controls remain visible.

## Case Tabs

The workspace has four top-level tabs:

- Day case
- Night case
- Final results
- Audit trail

Each primary case shows at most two default video panes:

- Original
- Current result

After successful finalization, Current result becomes Final reviewed.

Diagnostic face-only, plate-only, provenance, and raw-signal details stay behind technical sections instead of cluttering the default review view.

## Current Day Boundary

The previous day face and day plate outputs came from different extracted clips. Their metadata now maps both clips to a common parent source, `artifacts/day_primary.mp4`, and the v2.2 package extracts one canonical Day original interval from that parent source.

The combined Day current result is not fabricated from incompatible encoded clips. It is rendered from lineage-proven frozen evidence mapped into the canonical Day source-frame interval. The build validates the local ADAS XML/BIN and OIM ONNX hashes before packaging.

## Review Objects

The workspace indexes automatic and reviewed region tracks into simple reviewer-facing objects such as `Person 01` and `Plate 01`. Raw detector track IDs remain in Technical details. Each object has a deterministic representative thumbnail, first-seen frame, status badge, and separate controls for editor overlay visibility, privacy action, and geometry lock state.

Overlay visibility only changes what the editor draws. It never changes final rendering. Privacy actions are separate: Redact, Keep, and Needs review. Keep requires a reviewer note, creates an append-only event, and remains blocked until verified.

Entities and review tasks open in drawers instead of a fixed sidebar. Opening a drawer does not resize or restart the active video comparison.

## Review Sessions

Click **Start review** to create an anonymous server-backed local session. The server assigns a session UUID and UTC start time automatically. Reviewer name is required only when exporting or finalizing; the workspace appends a `reviewer_attribution_set` event and does not rewrite earlier events.

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
- the full-video sweep has been acknowledged or overridden with a written reason;
- the session event hash chain is valid;
- source hashes and configuration-freeze hashes match;
- frame spans are complete;
- output media validation passes;
- rendered-region verification passes.

Finalize with:

```bash
redactor finalize-reviewed-case \
  --package-root artifacts/final_pipeline_package_v2_2 \
  --case-id CASE_ID \
  --session-dir SESSION_DIR
```

The result may be labeled “Reviewed final for this source case.” It must not be labeled zero-leak, guaranteed private, or universally production-ready.
