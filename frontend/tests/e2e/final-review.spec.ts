import { expect, test } from '@playwright/test';

function caseFrameFromHud(text: string): number {
  const match = text.match(/Case frame (\d+)/);
  if (!match) throw new Error(`Could not parse HUD: ${text}`);
  return Number(match[1]);
}

test.describe('Final Review React workspace', () => {
  test.beforeEach(async ({ page }) => {
    page.on('dialog', (dialog) => {
      throw new Error(`Native dialog was opened: ${dialog.message()}`);
    });
    await page.goto('/app/day', { waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: 'Final Review Workspace' })).toBeVisible();
  });

  test('restores routes and keeps one active video element', async ({ page }) => {
    await expect(page.locator('video')).toHaveCount(1);
    await page.getByRole('button', { name: 'Night' }).click();
    await expect(page).toHaveURL(/\/app\/night$/);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: /Night/i })).toBeVisible();
    await expect(page.locator('video')).toHaveCount(1);
  });

  test('supports frame stepping, version switching, task markers, and decisions', async ({ page }) => {
    const video = page.locator('video');
    await expect(video).toHaveCount(1);
    await video.evaluate((node) => ((node as HTMLVideoElement & { __reviewTestId?: string }).__reviewTestId = 'stable'));
    const hud = page.locator('.frame-hud');
    const before = caseFrameFromHud(await hud.innerText());
    await page.keyboard.press('ArrowRight');
    await expect.poll(async () => caseFrameFromHud(await hud.innerText())).toBe(before + 1);
    await page.keyboard.press('Shift+ArrowRight');
    await expect.poll(async () => caseFrameFromHud(await hud.innerText())).toBe(before + 11);

    const selectedFrame = caseFrameFromHud(await hud.innerText());
    await page.locator('.version-selector select').selectOption('original');
    await expect.poll(async () => caseFrameFromHud(await hud.innerText())).toBe(selectedFrame);

    await page.locator('.timeline-marker').first().click();
    await expect(page.locator('.task-card.selected')).toHaveCount(1);
    const taskId = await page.locator('.task-card.selected').getAttribute('data-task-id');

    await page.getByRole('button', { name: 'Start review' }).click();
    await expect(page.locator('#sessionBanner')).toContainText('Local session');
    await page.locator(`.task-card[data-task-id="${taskId}"] .decision-correct`).click();
    await expect(page.locator(`.timeline-marker[data-task-id="${taskId}"]`)).toHaveClass(/marker-correct/);
    await expect.poll(async () => page.locator('video').evaluate((node) => (node as HTMLVideoElement & { __reviewTestId?: string }).__reviewTestId)).toBe('stable');
  });

  test('persists entity policy and opens visual merge dialog without raw IDs', async ({ page }) => {
    await page.getByRole('button', { name: 'Start review' }).click();
    await page.getByRole('button', { name: 'Entities' }).click();
    await expect(page.getByRole('dialog', { name: 'Review objects' })).toBeVisible();
    const firstEntity = page.locator('.entity-card').first();
    const entityId = await firstEntity.getAttribute('data-entity-id');
    await firstEntity.locator('[data-policy="keep_visible"]').click();
    await expect(page.locator(`.entity-card[data-entity-id="${entityId}"] [data-policy="keep_visible"]`)).toHaveAttribute(
      'aria-pressed',
      'true',
    );

    await page.locator(`.entity-card[data-entity-id="${entityId}"]`).getByRole('button', { name: 'Merge...' }).click();
    await expect(page.getByRole('dialog', { name: 'Merge review objects' })).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Merge review objects' })).not.toContainText(/entity_[a-f0-9]/);
    await page.getByRole('button', { name: 'Cancel' }).first().click();
  });

  test('requires attribution for export and finalization', async ({ page }) => {
    await page.getByRole('button', { name: 'Start review' }).click();
    await page.getByRole('button', { name: 'Export' }).click();
    await expect(page.getByRole('dialog', { name: 'Reviewer attribution' })).toBeVisible();
    await page.getByLabel('Reviewer name').fill('Reviewer One');
    await page.getByRole('button', { name: 'Save attribution' }).click();
    await expect(page.locator('#sessionBanner')).toContainText('Reviewer One');

    await page.getByRole('button', { name: 'Final Results' }).click();
    await page.getByRole('button', { name: 'Check finalization' }).click();
    await expect(page.locator('.final-status')).toContainText(/Final media|Finalization|unavailable/i);
  });

  test('serves direct route refresh and MP4 range requests', async ({ page, request, baseURL }) => {
    await page.goto('/app/audit', { waitUntil: 'domcontentloaded' });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: 'Audit Trail' })).toBeVisible();
    const response = await request.get(`${baseURL}/media/day/original_reference.mp4`, {
      headers: { Range: 'bytes=0-15' },
    });
    expect(response.status()).toBe(206);
    expect(response.headers()['content-type']).toContain('video/mp4');
    expect((await response.body()).length).toBe(16);
  });
});
