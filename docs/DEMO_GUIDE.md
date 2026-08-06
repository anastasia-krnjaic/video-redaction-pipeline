# Demo Guide

This guide points to the frozen local package created for the current prototype state.

## Open First

Start with:

`artifacts/final_pipeline_package_v1/showcase/PROJECT_OVERVIEW.html`

It gives the simplest visual walkthrough: what the system does, what the current outputs show, and which results are still pending human audit.

## Reference Media

Open:

`artifacts/final_pipeline_package_v1/audit/index.html`

The audit page shows:

- Original video context
- Face-only reviewed reference, where available
- Plate-only automatic preview, where available
- Combined face-and-plate demo preview, where source-compatible

Day face and day plate references are shown separately because their source hashes differ. Night references are source-compatible, so the package includes a combined night demo preview.

## How To Read The Labels

- **Original:** unredacted source context.
- **Face-only reviewed reference:** current accepted face reference output.
- **Plate-only automatic preview:** accepted plate tracks rendered automatically. This is review-required, not final.
- **Combined demo preview:** compatible face and plate regions rendered together for demonstration. This remains non-final while plate blockers exist.

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

After a reviewer exports decisions, validate them with:

```bash
redactor validate-final-audit-decisions \
  --tasks artifacts/final_pipeline_package_v1/audit/audit_tasks.json \
  --decisions DECISIONS.json
```

Summarize review progress with:

```bash
redactor summarize-final-audit \
  --tasks artifacts/final_pipeline_package_v1/audit/audit_tasks.json \
  --decisions DECISIONS.json
```
