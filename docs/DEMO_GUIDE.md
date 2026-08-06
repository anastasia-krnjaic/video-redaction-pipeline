# Demo Guide

This guide points to the frozen local package and the current case-based review workspace.

## Open First

Start with the broad-reader showcase:

`artifacts/final_pipeline_package_v1/showcase/PROJECT_OVERVIEW.html`

It gives the simplest visual walkthrough: what the system does, what the current outputs show, and which results are still pending human audit.

## Reference Media

For review, serve:

```bash
redactor serve-final-review-workspace artifacts/final_pipeline_package_v2 --open
```

The workspace shows one case at a time:

- Original video context
- Current combined result where source-compatible
- Guided risk-review tasks
- Full-video sweep controls
- Audit trail and local session export

The Night case has compatible Original and Current result media. The Day case now has proven parent-source lineage and a canonical original interval, but the combined Day current result remains blocked until the frozen local model files are supplied and the face/plate pipelines are rerun on that parent interval.

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
