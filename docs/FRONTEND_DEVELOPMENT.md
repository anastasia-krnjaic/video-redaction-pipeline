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
FINAL_REVIEW_PACKAGE=artifacts/final_pipeline_package_frontend_v1 npm run test:e2e -- --project=chromium
FINAL_REVIEW_PACKAGE=artifacts/final_pipeline_package_frontend_v1 npm run test:e2e -- --project=firefox
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
  --output-dir artifacts/final_pipeline_package_frontend_v1 \
  --frontend react \
  --skip-frontend-build \
  --overwrite
```

Omit `--skip-frontend-build` when the builder should run `npm ci` and `npm run build` itself.

Serve it with:

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli serve-final-review-workspace \
  artifacts/final_pipeline_package_frontend_v1 --open
```

Open URL:

```text
http://127.0.0.1:<port>/app/day
```

## Parity Audit

```bash
PYTHONPATH=src .venv/bin/python -m redactor.cli audit-final-review-backend-parity \
  --reference artifacts/final_pipeline_package_v2_3_4 \
  --candidate artifacts/final_pipeline_package_frontend_v1 \
  --output artifacts/final_pipeline_package_frontend_v1/backend_parity_audit.json
```

The audit compares backend/privacy artifacts and ignores frontend DOM/design differences.

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
