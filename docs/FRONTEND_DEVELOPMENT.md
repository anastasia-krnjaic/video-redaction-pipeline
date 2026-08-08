# Final Review Frontend Development

## Install

```bash
cd frontend
npm ci
```

Do not commit `node_modules`, Playwright browser binaries, generated packages, raw video, model weights, or sessions.

## Commands

```bash
cd frontend
npm run generate:api-types
npm run typecheck
npm run lint
npm run test
npm run build
```

Playwright E2E uses the package server and the repo virtualenv:

```bash
cd frontend
FINAL_REVIEW_PACKAGE=artifacts/final_pipeline_package_frontend_v1_5 npm run test:e2e -- --project=chromium
FINAL_REVIEW_PACKAGE=artifacts/final_pipeline_package_frontend_v1_5 npm run test:e2e -- --project=firefox
```

Install managed browsers when needed:

```bash
cd frontend
npx playwright install chromium firefox
```

## Package Generation

Generate the React package without changing backend artifacts:

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli build-final-review-workspace \
  --output-dir artifacts/final_pipeline_package_frontend_v1_5 \
  --frontend react \
  --react-reference-package artifacts/final_pipeline_package_frontend_v1_4_1 \
  --skip-frontend-build \
  --overwrite
```

Omit `--skip-frontend-build` when the builder should run `npm ci` and `npm run build` itself.

Serve it with:

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli serve-final-review-workspace \
  artifacts/final_pipeline_package_frontend_v1_5 --open
```

Open URL:

```text
http://127.0.0.1:<port>/app/day
```

## Parity Audit

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli audit-final-review-backend-parity \
  --reference artifacts/final_pipeline_package_frontend_v1_4 \
  --candidate artifacts/final_pipeline_package_frontend_v1_5 \
  --output artifacts/final_pipeline_package_frontend_v1_5/backend_parity_audit.json
```

The audit compares backend/privacy artifacts and ignores frontend DOM/design differences.

## v1.3 State Notes

- Review decisions are persisted before auto-advance; failed saves keep the active card and player frame unchanged.
- Entity merge is confirmed from checkbox selection and persisted by the Python API as a canonical merged entity with retired aliases.
- Entity policy changes and correction saves return materialization job IDs. The frontend polls `/api/v1/materialization-jobs/{job_id}` and switches Latest reviewed only after the job completes.
- Checkpoint deletion is a tombstone event. It removes the checkpoint from the active list without changing the current reviewed state or media.
- Manual redaction drafts use source-frame keys only and support Single frame, Fixed box across interval, and Interpolate between keyframes.

## v1.3.2 Correctness Notes

- `Restart review` is case-scoped and narrow. It clears active review decisions, task buckets, task selection, sweep acknowledgement, and transient review progress for the selected case while preserving entity policies, canonical merges, manual corrections, checkpoints, Latest reviewed, History, and the other case.
- `Reset all edits to Automatic baseline` is the separate destructive action. It resets the selected case's active entity policies, merges, manual corrections, reviewer-created manual issues, review decisions, sweep acknowledgement, and Latest reviewed projection while preserving append-only History, checkpoints, inherited baseline source policies, and the other case.
- Entity/checkpoint operations keep unrelated controls usable. Accepted mutations are projected immediately, and Latest reviewed rendering is debounced/coalesced in the Python materialization queue.
- The normal version selector contains exactly Original, Automatic, and Latest reviewed. Correction previews are transient and never become a persistent selector option.
- Interpolate between keyframes requires exact start and end source-frame rectangles; interval edits prune out-of-range keyframes.
- Saved manual redactions are verified against effective privacy-region records and decoded output frames, not only API success.
- Long raw review intervals remain intact in audit data and are projected into non-overlapping reviewer-facing child tasks with a three-second target and five-second hard maximum.

## v1.3.3 Bug-Fix Notes

- `Restart review` remains workflow-only.
- `Reset all edits to Automatic baseline` now guards against stale materialization events and compares the active case projection against the immutable package baseline in `reset_baseline_state_audit.json`.
- Entity cards reserve a fixed operation-status slot at the right edge, so pending spinners do not shift policy buttons, card width, modal width, or scroll position.

## v1.4 Hardening Notes

- Object-policy changes, bulk policy changes, manual correction saves, merges, checkpoint restores, and destructive resets classify the effective privacy-region hash before scheduling media work.
- No-op/idempotent mutations append History and update working state without starting a media job.
- Compatible queued/running jobs are reused when repeated submissions target the same effective-region hash.
- Materialization jobs include target effective-region hashes and published working-state revisions so stale jobs cannot overwrite newer media-changing state.
- API request bodies are bounded, and MP4 Range handling covers normal, suffix, invalid, multi-range, and out-of-bounds requests.
- Benchmark evidence is generated with `scripts/final_review_performance_harness.py` under `artifacts/performance_production_hardening_v1_4/`.

## v1.4.1 Manual Save Notes

- Manual correction Save requires the current Preview token and validates the case, session, source media hash, working-state revision compatibility, correction ID, and stable draft hash.
- Preview appends a `correction_previewed` event; Save appends `manual_correction_saved`, creates the linked verification task, and keeps correction previews out of the persistent version selector.
- Single-frame, fixed-box interval, and interpolated corrections are verified against active effective privacy regions and decoded Latest reviewed output.

## v1.5 Planning Notes

- Materialization plans now expose exact changed source-frame intervals, changed-frame counts, planned render/encode counts, current stage, and progress metadata.
- The current renderer still uses the full-render publication fallback. Segment reuse and cache cleanup commands are deferred until full-output parity can be proven.

## API Schema

Regenerate the checked-in schema:

```bash
PYTHONPATH=src .venv/bin/python -m redactor.review_api_contract \
  --output docs/api/final_review_workspace_v1.schema.json
```

Then regenerate frontend types:

```bash
cd frontend
npm run generate:api-types
```
