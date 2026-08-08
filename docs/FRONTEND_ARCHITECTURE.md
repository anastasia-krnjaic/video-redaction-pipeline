# Final Review Frontend Architecture

Final Review Frontend Foundation v1 migrates the review workspace presentation layer to React and TypeScript while keeping Python as the source of truth for privacy behavior.

## Ownership Boundary

Python owns:

- case manifests and media-version mappings;
- source-frame and case-frame mappings;
- review tasks, entities, merges, corrections, sessions, decisions, attribution, audit events, and finalization;
- correction preview rendering and final media rendering;
- MP4 MIME and HTTP Range serving.

React owns:

- presentation, route state, local draft correction UI state, keyboard handling, and view filters;
- frame-aware player controls that call back into Python-owned source-frame records;
- typed API calls to `/api/v1/`.

React does not reimplement detector, tracker, evidence, grouping, correction validation, rendering, audit hash-chain, or privacy-policy logic.

## Package Layout

The frozen reference package remains:

```text
artifacts/final_pipeline_package_v2_3_4/
```

The current React package is generated at:

```text
artifacts/final_pipeline_package_frontend_v1_3/
```

Frontend v1.3.1 is generated at:

```text
artifacts/final_pipeline_package_frontend_v1_3_1/
```

For backend/privacy parity, the v1.3.1 package copies the accepted v1.3 React package as its base, then writes:

```text
app/                         React/Vite SPA
legacy.html                  legacy generated workspace fallback
frontend_build_manifest.json React build metadata
```

The root `index.html` redirects to `/app/day`. The legacy fallback remains available at `/legacy.html`.

## API Contract

The stable API prefix is:

```text
/api/v1/
```

The checked-in schema is:

```text
docs/api/final_review_workspace_v1.schema.json
```

The backend serves the same schema at `/api/v1/schema`. The frontend generated TypeScript contract is:

```text
frontend/src/types/api.ts
```

Regenerate it with:

```bash
cd frontend
npm run generate:api-types
```

Legacy `/api/...` endpoints remain available during this milestone.

## Component Boundaries

The React app is split into:

- `AppShell`, `CaseTabs`, `SessionHeader`, `AttributionDialog`, `ErrorBoundary`;
- `CaseReviewPage`, `FinalResultsPage`, `AuditTrailPage`;
- `VersionSelector`, `ReviewVideoPlayer`, `ReviewTimeline`;
- `ReviewQueue`, `ReviewTaskCard`;
- `EntityPanel`, `EntityCard`, `EntityMergeDialog`;
- `CorrectionEditor`.

Frame logic is centralized in `useVideoController` and `src/utils/frame.ts`.

## Review State And Materialization

Review decisions, object policies, entity merges, corrections, checkpoints, and History remain Python-owned append-only session state. React may optimistically show an object policy only after the backend accepts the event.

Entity policy changes, correction saves, checkpoint restores, and Restart case review enqueue Python materialization jobs. The server serializes jobs per case, debounces rapid accepted mutations, coalesces queued work to the newest active projection revision, caches already materialized state hashes, and rejects stale render results. React polls `/api/v1/materialization-jobs/{job_id}` and swaps in a revisioned Latest reviewed media URL only after the job completes.

Entity merges are not assembled in React. The merge endpoint creates one canonical merged review entity, retires the selected aliases, preserves child/source evidence, and records the hash-chained event.

## Media

The Python server keeps the proven Range implementation. React uses one active video element per current page and only changes `src` when the selected media version changes.

No remote assets, external CSS CDN, analytics, database, auth service, upload workflow, Electron, or Tauri are introduced.

## Known Limitations

- The lightweight correction editor stores draft UI state locally, then submits Python-owned correction records.
- Browser E2E covers the main parity-critical flows; manual deep review of every correction geometry edge case is still useful.
- Final reviewed media appears only after Python finalization gates pass.
