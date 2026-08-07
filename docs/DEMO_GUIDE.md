# Demo Guide

This guide points to the frozen local package and the current case-based review workspace.

## Open First

Start with the broad-reader showcase:

`artifacts/final_pipeline_package_v1/showcase/PROJECT_OVERVIEW.html`

It gives the simplest visual walkthrough: what the system does, what the current outputs show, and which results are still pending human audit.

## Reference Media

For review, serve:

```bash
redactor serve-final-review-workspace artifacts/final_pipeline_package_frontend_v1 --open
```

The workspace shows one case at a time:

- Original video context
- Current combined result where source-compatible
- One large versioned player with Original / Automatic / Latest reviewed
- Timeline-linked guided risk-review tasks
- Full-video sweep controls
- Compact object list with representative Person/Plate thumbnails
- Audit trail and local session export

The React app loads under `/app/day`; the frozen generated v2.3.4 workspace is
available at `/legacy.html` as a fallback. Both use the same Python-owned
backend package data, media Range serving, sessions, corrections, audit trail,
and finalization gates.

The Day and Night cases each show only Original and Current result by default. The Day case uses a canonical interval from `artifacts/day_primary.mp4`; its Current result is rendered from lineage-proven frozen face and plate evidence rather than concatenated old clips.

Click **Start review** to create a local anonymous session. Reviewer name is
requested only when exporting or finalizing. Use the app's fullscreen controls,
not raw MP4 fullscreen, so frame numbers and source-frame mapping remain
visible.

## How To Read The Labels

- **Original:** unredacted source context.
- **Face-only reviewed reference:** current accepted face reference output.
- **Plate-only automatic preview:** accepted plate tracks rendered automatically. This is review-required, not final.
- **Current result:** compatible face and plate regions rendered together for review where available. This remains non-final while blockers exist.
- **Final reviewed:** appears only after every required task is green, corrections are verified, sweep acknowledgement exists, hashes match, and media validation passes.

## Detector, Tracking, Review, And Rendering

Detection finds candidate faces or plates.

Tracking connects observations over time. For plates, temporal evidence accumulates before a track is trusted.

Review handles uncertain intervals, long gaps, false positives, and spot checks.

Rendering applies pixelation or blur only to approved regions and writes validated MP4 output.

## Current Limitations To Mention

- The project does not claim zero leaks.
- Human full-video audit is pending.
- Plate results are automatic previews and still require review.
- Some nighttime plate gaps remain unresolved.
- Some false plate redactions remain visible and should be adjudicated.

## Validating Audit Decisions

After a reviewer creates or exports a session, validate it with:

```bash
redactor validate-review-session --session-dir SESSION_DIR
```

Summarize review progress with:

```bash
redactor summarize-review-session --session-dir SESSION_DIR
```
