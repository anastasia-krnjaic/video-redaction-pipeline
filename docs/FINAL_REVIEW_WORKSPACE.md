# Final Review Workspace

The Final Review Workspace is the local, case-based review and finalization layer for the frozen face and plate prototype behavior.

It is generated under:

```text
artifacts/final_pipeline_package_v2/
```

The current v2.1 package is generated under:

```text
artifacts/final_pipeline_package_v2_1/
```

It does not change detectors, thresholds, tracking, evidence accumulation, recovery, or rendering configuration. It gives a reviewer one case at a time, clear tasks, local correction records, logical privacy entities, an append-only audit trail, and finalization gates.

## Launch

Serve the package locally:

```bash
redactor serve-final-review-workspace artifacts/final_pipeline_package_v2_1 --open
```

The server binds to `127.0.0.1`, serves only files under the package root, supports MP4 range requests for seeking, and writes review sessions only under the package `sessions/` directory.

Opening `index.html` directly may load static files, but the local server is the supported workflow because it enables reliable video seeking and durable local session writes.

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

The previous day face and day plate outputs came from different extracted clips. Their metadata now maps both clips to a common parent source, `artifacts/day_primary.mp4`, and the v2.1 package extracts one canonical Day original interval from that parent source.

The combined Day current result is not fabricated from incompatible encoded clips. It is rendered from lineage-proven frozen evidence mapped into the canonical Day source-frame interval. The build validates the local ADAS XML/BIN and OIM ONNX hashes before packaging.

## Privacy Entities

The workspace indexes automatic and reviewed region tracks into simple reviewer-facing entities such as `Human 1` and `Plate 1`. Raw detector track IDs remain in Technical details. Each entity has separate controls for editor overlay visibility, privacy policy, and reviewed lock state.

## Review Sessions

On first use, the reviewer enters a display name and reviewer ID or email. This is locally declared identity, not enterprise authentication.

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

Corrections are tied to a case, task, entity, source-video hash, source-frame interval, reviewer identity, reason, keyframe boxes, checkpoint verification states, interpolation method, application version, and configuration freeze hash. Interpolation is bounded to reviewed keyframes inside the task interval; unbounded extrapolation is rejected and unreviewed checkpoints block finalization.

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
  --package-root artifacts/final_pipeline_package_v2_1 \
  --case-id CASE_ID \
  --session-dir SESSION_DIR
```

The result may be labeled “Reviewed final for this source case.” It must not be labeled zero-leak, guaranteed private, or universally production-ready.
