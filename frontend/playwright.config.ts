import { defineConfig, devices } from '@playwright/test';

const port = Number(process.env.FRONTEND_E2E_PORT || 8765);
const packageRoot =
  process.env.FINAL_REVIEW_PACKAGE || 'artifacts/final_pipeline_package_frontend_v1';
const baseURL = process.env.FRONTEND_E2E_BASE_URL || `http://127.0.0.1:${port}`;
const useManagedServer = process.env.FRONTEND_E2E_NO_SERVER !== '1';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: {
    timeout: 5_000,
  },
  use: {
    baseURL,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
  ],
  webServer: useManagedServer
    ? {
        command: `cd .. && PYTHONPATH=src .venv/bin/python -m redactor.cli serve-final-review-workspace ${packageRoot} --port ${port}`,
        url: `${baseURL}/app/day`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      }
    : undefined,
});
