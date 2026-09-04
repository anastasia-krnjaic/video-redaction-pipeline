# Frontend Development

The review workspace frontend is a React/Vite application under `frontend/`. Python remains the source of truth for review state, media, validation, and finalization.

## Install

```bash
cd frontend
npm ci
```

Do not commit `node_modules`, browser binaries, generated media, review sessions, Final media, delivery zips, or ignored artifacts.

## Commands

```bash
npm --prefix frontend run build
npm --prefix frontend test -- --run
```

Additional frontend commands are available from the package scripts:

```bash
cd frontend
npm run generate:api-types
npm run typecheck
npm run lint
```

Regenerate API types after the checked-in API schema changes:

```bash
cd frontend
npm run generate:api-types
```

## Local Workspace Testing

Serve a generated workspace from the repository root:

```bash
redactor serve-final-review-workspace path/to/example_automatic_preview/review_workspace --open
```

When the local release artifacts are present, the frozen review workspace can be served with:

```bash
redactor serve-final-review-workspace path/to/release_review_workspace --open
```

The app loads under `/app/<case_id>`. The server supplies `/api/v1/` data and media Range responses.

## Firefox Smoke

The repository Firefox smoke is run through pytest so it uses the same Python server harness as the backend tests:

```bash
REDACTOR_RUN_FIREFOX_E2E=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -rs -p no:cacheprovider tests/test_final_review_workspace_firefox.py
```

## Frontend Boundary

React may own presentation, local form drafts, player controls, keyboard handling, and route state. It must consume backend-authoritative state for:

- sessions and current-session resume;
- Guidance and Needs correction;
- entity policy;
- correction Preview and Save;
- materialization jobs;
- Latest reviewed currentness;
- full-video sweep currentness;
- finalization status;
- Final media availability.

Do not duplicate finalization readiness, grouping semantics, correction validation, privacy rendering, event-chain validation, or entity/query synchronization in React.
