import { expect, test } from '@playwright/test';
import type { APIRequestContext, Locator, Page } from '@playwright/test';

interface WorkspaceSummary {
  cases: Record<string, { frame_count: number; source_frame_start: number }>;
  entities: { cases: Record<string, EntityPayload[]> };
  tasks: { tasks: TaskPayload[] };
}

interface TaskPayload {
  case_id: string;
  review_task_id: string;
  source_frame_end: number;
  source_frame_start: number;
  technical_only?: boolean;
  title: string;
}

interface EntityPayload {
  case_id: string;
  category: string;
  current_policy?: string;
  entity_id: string;
  first_visible_source_frame?: number;
  last_visible_source_frame?: number;
  representative_bbox?: number[];
}

function caseFrameFromHud(text: string): number {
  const match = text.match(/Case frame (\d+)/);
  if (!match) throw new Error(`Could not parse HUD: ${text}`);
  return Number(match[1]);
}

function sourceFrameFromHud(text: string): number {
  const match = text.match(/Source frame (\d+)/);
  if (!match) throw new Error(`Could not parse HUD: ${text}`);
  return Number(match[1]);
}

async function loadSummary(request: APIRequestContext): Promise<WorkspaceSummary> {
  const response = await request.get('/api/v1/workspace/summary');
  expect(response.ok()).toBeTruthy();
  return (await response.json()) as WorkspaceSummary;
}

async function currentCaseFrame(page: Page): Promise<number> {
  return caseFrameFromHud(await page.locator('.frame-hud').innerText());
}

async function currentSourceFrame(page: Page): Promise<number> {
  return sourceFrameFromHud(await page.locator('.frame-hud').innerText());
}

async function expectCaseFrameNear(page: Page, expectedFrame: number): Promise<void> {
  await expect.poll(async () => Math.abs((await currentCaseFrame(page)) - expectedFrame) <= 1).toBe(true);
}

async function installTimelineProbe(page: Page, frameCount: number): Promise<void> {
  const slider = page.getByRole('slider', { name: 'Video timeline' });
  await slider.evaluate((element, totalFrames) => {
    const state = window as Window & { __timelineExpectedFrame?: number | null };
    state.__timelineExpectedFrame = null;
    const update: EventListener = (event) => {
      const pointer = event as PointerEvent | MouseEvent;
      const rect = element.getBoundingClientRect();
      const ratio = rect.width > 0 ? Math.min(1, Math.max(0, (pointer.clientX - rect.left) / rect.width)) : 0;
      state.__timelineExpectedFrame = Math.round(ratio * Math.max(0, totalFrames - 1));
    };
    element.addEventListener('pointerdown', update, { capture: true, once: true });
    element.addEventListener('pointermove', update, { capture: true });
    element.addEventListener('mousedown', update, { capture: true, once: true });
    element.addEventListener('mousemove', update, { capture: true });
  }, frameCount);
}

async function expectedTimelineFrame(page: Page): Promise<number> {
  const expected = await page.evaluate(() => {
    const state = window as Window & { __timelineExpectedFrame?: number | null };
    return state.__timelineExpectedFrame;
  });
  if (typeof expected !== 'number') throw new Error('Timeline pointer probe did not record a frame.');
  return expected;
}

async function clickTimeline(page: Page, frameCount: number, ratio: number): Promise<number> {
  const slider = page.getByRole('slider', { name: 'Video timeline' });
  await slider.scrollIntoViewIfNeeded();
  await installTimelineProbe(page, frameCount);
  const box = await slider.boundingBox();
  if (!box) throw new Error('Timeline slider was not visible.');
  await page.mouse.click(box.x + box.width * ratio, box.y + box.height - 6);
  return expectedTimelineFrame(page);
}

async function dragTimeline(page: Page, frameCount: number, startRatio: number, endRatio: number): Promise<number> {
  const slider = page.getByRole('slider', { name: 'Video timeline' });
  await slider.scrollIntoViewIfNeeded();
  await installTimelineProbe(page, frameCount);
  const box = await slider.boundingBox();
  if (!box) throw new Error('Timeline slider was not visible.');
  const y = box.y + box.height - 6;
  await page.mouse.move(box.x + box.width * startRatio, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * endRatio, y, { steps: 8 });
  await page.mouse.up();
  return expectedTimelineFrame(page);
}

async function enterFullscreen(page: Page): Promise<void> {
  await page.locator('button[title="Fullscreen"]').click();
  await expect.poll(async () => page.evaluate(() => Boolean(document.fullscreenElement))).toBe(true);
}

async function exitFullscreen(page: Page): Promise<void> {
  await page.evaluate(() => {
    if (document.fullscreenElement) void document.exitFullscreen();
  });
  await expect.poll(async () => page.evaluate(() => Boolean(document.fullscreenElement))).toBe(false);
}

async function stepToCaseFrame(page: Page, targetFrame: number): Promise<void> {
  await page.locator('body').click({ position: { x: 4, y: 4 } });
  for (let guard = 0; guard < 80; guard += 1) {
    const current = await currentCaseFrame(page);
    const diff = targetFrame - current;
    if (diff === 0) return;
    if (Math.abs(diff) >= 10) {
      await page.keyboard.press(diff > 0 ? 'Shift+ArrowRight' : 'Shift+ArrowLeft');
    } else {
      await page.keyboard.press(diff > 0 ? 'ArrowRight' : 'ArrowLeft');
    }
  }
  throw new Error(`Could not step to case frame ${targetFrame}; currently at ${await currentCaseFrame(page)}`);
}

async function startReview(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Start review' }).click();
  await expect(page.locator('#sessionBanner')).toContainText('Local session');
}

async function drawCorrectionBox(page: Page, x1 = 0.32, y1 = 0.34, x2 = 0.52, y2 = 0.56): Promise<void> {
  const overlay = page.getByTestId('correction-video-overlay');
  const box = await overlay.boundingBox();
  if (!box) throw new Error('Correction overlay was not visible.');
  await page.mouse.move(box.x + box.width * x1, box.y + box.height * y1);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * x2, box.y + box.height * y2, { steps: 4 });
  await page.mouse.up();
}

async function stableBox(locator: Locator): Promise<{ x: number; y: number; width: number; height: number }> {
  const box = await locator.boundingBox();
  if (!box) throw new Error('Expected element to expose a bounding box.');
  return box;
}

function expectBoxStable(
  before: { x: number; y: number; width: number; height: number },
  after: { x: number; y: number; width: number; height: number },
  tolerance = 1,
): void {
  expect(Math.abs(after.x - before.x)).toBeLessThanOrEqual(tolerance);
  expect(Math.abs(after.width - before.width)).toBeLessThanOrEqual(tolerance);
}

test.describe('Final Review React workspace', () => {
  test.beforeEach(async ({ page }) => {
    page.on('dialog', (dialog) => {
      throw new Error(`Native dialog was opened: ${dialog.message()}`);
    });
    await page.goto('/app/day', { waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: 'Final Review Workspace' })).toBeVisible();
  });

  test('restores routes, keeps one active video element, and supports click play/pause', async ({ page }) => {
    await expect(page.locator('video')).toHaveCount(1);
    const video = page.locator('video');
    await video.evaluate((node) => {
      const media = node as HTMLVideoElement & { __reviewPaused?: boolean };
      media.__reviewPaused = true;
      Object.defineProperty(media, 'paused', {
        configurable: true,
        get() {
          return media.__reviewPaused;
        },
      });
      media.play = () => {
        media.__reviewPaused = false;
        media.dispatchEvent(new Event('play'));
        return Promise.resolve();
      };
      media.pause = () => {
        media.__reviewPaused = true;
        media.dispatchEvent(new Event('pause'));
      };
    });
    await video.click();
    await expect.poll(async () => video.evaluate((node) => !(node as HTMLVideoElement).paused)).toBe(true);
    await video.click();
    await expect.poll(async () => video.evaluate((node) => (node as HTMLVideoElement).paused)).toBe(true);
    await page.getByRole('button', { name: 'Night' }).click();
    await expect(page).toHaveURL(/\/app\/night$/);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: /Night/i })).toBeVisible();
    await expect(page.locator('video')).toHaveCount(1);
  });

  test('isolates Day and Night frame mappings, versions, and fullscreen media', async ({ page }) => {
    await stepToCaseFrame(page, 173);
    await expect.poll(async () => currentSourceFrame(page)).toBe(2001);
    await expect(page.locator('video')).toHaveAttribute('data-media-path', /media\/day\//);
    await expect(page.locator('.version-selector select option')).toContainText([
      'Original',
      'Automatic',
      'Latest reviewed',
    ]);
    await enterFullscreen(page);
    await expect(page.locator('video')).toHaveAttribute('data-case', 'day');
    await expect(page.locator('video')).toHaveAttribute('data-media-path', /media\/day\//);
    await exitFullscreen(page);

    await page.getByRole('button', { name: 'Night' }).click();
    await expect(page.locator('.filter-row button.active')).toContainText('Open');
    await expect(page.locator('.version-selector select option')).toContainText([
      'Original',
      'Automatic',
      'Latest reviewed',
    ]);
    await stepToCaseFrame(page, 173);
    await expect.poll(async () => currentSourceFrame(page)).toBe(173);
    await expect(page.locator('video')).toHaveAttribute('data-media-path', /media\/night\//);
    await enterFullscreen(page);
    await expect(page.locator('video')).toHaveAttribute('data-case', 'night');
    await expect(page.locator('video')).toHaveAttribute('data-media-path', /media\/night\//);
    await exitFullscreen(page);

    await page.getByRole('button', { name: 'Day' }).click();
    await expect.poll(async () => currentCaseFrame(page)).toBe(173);
    await expect.poll(async () => currentSourceFrame(page)).toBe(2001);
  });

  test('supports empty-track timeline scrubbing and marker selection', async ({ page, request }) => {
    const summary = await loadSummary(request);
    const maxFrame = summary.cases.day.frame_count - 1;
    for (const ratio of [0.25, 0.5, 0.9]) {
      const expectedFrame = await clickTimeline(page, summary.cases.day.frame_count, ratio);
      expect(Math.abs(expectedFrame - Math.round(maxFrame * ratio))).toBeLessThanOrEqual(3);
      await expectCaseFrameNear(page, expectedFrame);
    }

    const dragFrame = await dragTimeline(page, summary.cases.day.frame_count, 0.1, 0.68);
    expect(Math.abs(dragFrame - Math.round(maxFrame * 0.68))).toBeLessThanOrEqual(3);
    await expectCaseFrameNear(page, dragFrame);

    const task = summary.tasks.tasks.find((item) => item.case_id === 'day' && !item.technical_only);
    if (!task) throw new Error('No day task available for marker test.');
    await page.locator(`.timeline-marker[data-task-id="${task.review_task_id}"]`).click();
    await expectCaseFrameNear(page, task.source_frame_start - summary.cases.day.source_frame_start);
    await expect(page.locator(`.task-card.selected[data-task-id="${task.review_task_id}"]`)).toBeVisible();
  });

  test('keeps timeline scrubbing, markers, HUD, and frame stepping in fullscreen', async ({ page, request }) => {
    const summary = await loadSummary(request);
    const maxFrame = summary.cases.day.frame_count - 1;
    await enterFullscreen(page);
    await expect(page.locator('.frame-hud')).toBeVisible();
    await expect(page.getByRole('slider', { name: 'Video timeline' })).toBeVisible();

    for (const ratio of [0.25, 0.5, 0.9]) {
      const expectedFrame = await clickTimeline(page, summary.cases.day.frame_count, ratio);
      expect(Math.abs(expectedFrame - Math.round(maxFrame * ratio))).toBeLessThanOrEqual(3);
      await expectCaseFrameNear(page, expectedFrame);
    }
    const dragFrame = await dragTimeline(page, summary.cases.day.frame_count, 0.9, 0.42);
    expect(Math.abs(dragFrame - Math.round(maxFrame * 0.42))).toBeLessThanOrEqual(3);
    await expectCaseFrameNear(page, dragFrame);

    const before = await currentCaseFrame(page);
    await page.keyboard.press('ArrowRight');
    await expect.poll(async () => currentCaseFrame(page)).toBe(before + 1);
    await page.keyboard.press('Shift+ArrowLeft');
    await expect.poll(async () => currentCaseFrame(page)).toBe(before - 9);

    const task = summary.tasks.tasks.find((item) => item.case_id === 'day' && !item.technical_only);
    if (!task) throw new Error('No day task available for fullscreen marker test.');
    await page.locator(`.timeline-marker[data-task-id="${task.review_task_id}"]`).click();
    await expectCaseFrameNear(page, task.source_frame_start - summary.cases.day.source_frame_start);
    await expect(page.locator(`.task-card.selected[data-task-id="${task.review_task_id}"]`)).toBeVisible();
    await page.evaluate(() => document.exitFullscreen());
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

    await startReview(page);
    await expect(page.locator('#sessionBanner')).toContainText('Local session');
    await page.locator(`.task-card[data-task-id="${taskId}"] .decision-correct`).click();
    await expect(page.locator(`.timeline-marker[data-task-id="${taskId}"]`)).toHaveClass(/marker-correct/);
    await expect.poll(async () => page.locator('video').evaluate((node) => (node as HTMLVideoElement & { __reviewTestId?: string }).__reviewTestId)).toBe('stable');
  });

  test('uses the simplified reviewer bucket lifecycle', async ({ page }) => {
    await startReview(page);
    await expect(page.locator('.case-panel.active')).not.toContainText('Needs attention');
    await expect(page.getByRole('button', { name: /Resolved/i })).toHaveCount(0);

    const initialTaskIds = await page
      .locator('.task-card')
      .evaluateAll((cards) => cards.slice(0, 3).map((card) => (card as HTMLElement).dataset.taskId || ''));
    if (initialTaskIds.length < 3 || initialTaskIds.some((id) => !id)) throw new Error('Expected at least three open tasks.');

    await page.locator(`.task-card[data-task-id="${initialTaskIds[0]}"] .decision-correct`).click();
    await page.locator('.filter-row').getByRole('button', { name: /^Completed/ }).click();
    await expect(page.locator(`.task-card[data-task-id="${initialTaskIds[0]}"]`)).toBeVisible();

    await page.locator('.filter-row').getByRole('button', { name: /^Open/ }).click();
    await page.locator(`.task-card[data-task-id="${initialTaskIds[1]}"] .decision-needs-correction`).click();
    await page.locator('.filter-row').getByRole('button', { name: /^Needs correction/ }).click();
    await expect(page.locator(`.task-card[data-task-id="${initialTaskIds[1]}"]`)).toBeVisible();
    await page.locator('.filter-row').getByRole('button', { name: /^Open/ }).click();
    await expect(page.locator(`.task-card[data-task-id="${initialTaskIds[1]}"]`)).toHaveCount(0);

    await page.locator(`.task-card[data-task-id="${initialTaskIds[2]}"] .decision-unsure`).click();
    await page.locator('.filter-row').getByRole('button', { name: /^Unsure \/ Escalated/ }).click();
    await expect(page.locator(`.task-card[data-task-id="${initialTaskIds[2]}"]`)).toBeVisible();
    await page.locator('.filter-row').getByRole('button', { name: /^Open/ }).click();
    await expect(page.locator(`.task-card[data-task-id="${initialTaskIds[2]}"]`)).toHaveCount(0);
  });

  test('persists entity policy, merge events, and Night keep-policy people', async ({ page, request }) => {
    test.setTimeout(240_000);
    await startReview(page);
    await page.getByRole('button', { name: 'Entities' }).click();
    await expect(page.getByRole('dialog', { name: 'Review objects' })).toBeVisible();
    const dialog = page.getByRole('dialog', { name: 'Review objects' });
    await dialog.locator('.modal-scroll-body').evaluate((node) => {
      node.scrollTop = node.scrollHeight;
    });
    await expect(dialog.getByRole('button', { name: 'Close' }).first()).toBeVisible();
    await dialog.getByRole('button', { name: 'People' }).click();

    const firstEntity = page.locator('.entity-card').first();
    const entityId = await firstEntity.getAttribute('data-entity-id');
    await firstEntity.locator('[data-policy="keep_visible"]').click();
    await expect(page.locator(`.entity-card[data-entity-id="${entityId}"] [data-policy="keep_visible"]`)).toHaveAttribute(
      'aria-pressed',
      'true',
      { timeout: 120_000 },
    );
    await expect(firstEntity.locator('.entity-operation-status .inline-spinner')).toHaveCount(0, { timeout: 180_000 });

    const entityCountBefore = await page.locator('.entity-card').count();
    await page.locator('.entity-card').nth(0).locator('input[type="checkbox"]').check();
    await page.locator('.entity-card').nth(1).locator('input[type="checkbox"]').check();
    await expect(page.getByRole('button', { name: /Merge selected \(2\)/ })).toBeEnabled();
    await page.getByRole('button', { name: /Merge selected \(2\)/ }).click();
    const mergeDialog = page.getByRole('dialog', { name: 'Merge selected objects' });
    await expect(mergeDialog).toBeVisible();
    await expect(mergeDialog).not.toContainText(/entity_[a-f0-9]/);
    await expect(mergeDialog.getByRole('button', { name: 'Merge selected' })).toBeEnabled();
    await mergeDialog.getByRole('button', { name: 'Merge selected' }).click();
    await expect(mergeDialog).toHaveCount(0);
    await expect.poll(async () => page.locator('.entity-card').count()).toBe(entityCountBefore - 1);

    const summary = await loadSummary(request);
    expect(summary.entities.cases.day.some((entity) => Array.isArray(entity.representative_bbox))).toBeTruthy();

    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    await expect.poll(async () => page.locator('.entity-card').count()).toBe(entityCountBefore - 1);
    const thumb = page.locator('.entity-card .thumbnail-button').first();
    await thumb.click();
    await expect(page.getByTestId('entity-focus-overlay')).toBeVisible();
    await expect(page.getByTestId('entity-focus-overlay')).toHaveCount(0, { timeout: 5000 });

    await page.getByRole('button', { name: 'Night' }).click();
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    await expect(page.locator('.entity-card[data-category="face"] [data-policy="keep_visible"][aria-pressed="true"]').first()).toBeVisible();
  });

  test('reset all edits restores the baseline entity registry after merge and policy changes', async ({ page }) => {
    test.setTimeout(240_000);
    await startReview(page);
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    const baselineIds = await page.locator('.entity-card').evaluateAll((cards) =>
      cards.map((card) => (card as HTMLElement).dataset.entityId || ''),
    );
    const baselineCount = baselineIds.length;
    await page.locator('.entity-card').nth(0).locator('input[type="checkbox"]').check();
    await page.locator('.entity-card').nth(1).locator('input[type="checkbox"]').check();
    await page.getByRole('button', { name: /Merge selected \(2\)/ }).click();
    await page.getByRole('dialog', { name: 'Merge selected objects' }).getByRole('button', { name: 'Merge selected' }).click();
    await expect.poll(async () => page.locator('.entity-card').count(), { timeout: 120_000 }).toBe(baselineCount - 1);
    await expect(page.locator('.entity-card', { hasText: /Merged Person/ }).first()).toBeVisible();
    await page.locator('.entity-card').first().locator('[data-policy="keep_visible"]').click();
    await expect(page.locator('.entity-card').first().locator('[data-policy="keep_visible"]')).toHaveAttribute('aria-pressed', 'true', { timeout: 120_000 });
    await page.getByRole('button', { name: 'Close' }).first().click();

    await page.getByRole('button', { name: 'Night' }).click();
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    const nightCount = await page.locator('.entity-card').count();
    await page.getByRole('button', { name: 'Close' }).first().click();
    await page.getByRole('button', { name: 'Day' }).click();

    await page.getByRole('button', { name: /^Reset all edits to Automatic baseline$/ }).click();
    await page.getByRole('dialog', { name: /Reset all edits to Automatic baseline/ }).getByRole('button', {
      name: /^Reset all edits to Automatic baseline$/,
    }).click();
    await expect(page.getByRole('dialog', { name: /Reset all edits to Automatic baseline/ })).toHaveCount(0, { timeout: 120_000 });
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    await expect.poll(async () => page.locator('.entity-card').count()).toBe(baselineCount);
    await expect(page.locator('.entity-card', { hasText: /Merged Person/ })).toHaveCount(0);
    const resetIds = await page.locator('.entity-card').evaluateAll((cards) =>
      cards.map((card) => (card as HTMLElement).dataset.entityId || ''),
    );
    expect(resetIds).toEqual(baselineIds);
    await expect(page.locator(`.entity-card[data-entity-id="${baselineIds[0]}"] [data-policy="redact"]`)).toHaveAttribute('aria-pressed', 'true');
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    await expect.poll(async () => page.locator('.entity-card').count()).toBe(baselineCount);
    await expect(page.locator('.entity-card', { hasText: /Merged Person/ })).toHaveCount(0);
    await page.getByRole('button', { name: 'Close' }).first().click();
    await page.getByRole('button', { name: 'Night' }).click();
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    await expect.poll(async () => page.locator('.entity-card').count()).toBe(nightCount);
    await expect(page.locator('.entity-card[data-category="face"] [data-policy="keep_visible"][aria-pressed="true"]').first()).toBeVisible();
  });

  test('entity pending status uses a reserved slot without modal or card reflow', async ({ page }) => {
    test.setTimeout(180_000);
    await page.route('**/api/v1/object-policy', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 200));
      await route.continue();
    });
    await startReview(page);
    await page.getByRole('button', { name: 'Night' }).click();
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    const dialog = page.getByRole('dialog', { name: 'Review objects' });
    const scrollBody = dialog.locator('.modal-scroll-body');
    await scrollBody.evaluate((node) => {
      node.scrollTop = 80;
    });
    const first = page
      .locator('.entity-card')
      .filter({ has: page.locator('[data-policy="keep_visible"][aria-pressed="true"]') })
      .first();
    await expect(first).toBeVisible();
    const firstEntityId = await first.getAttribute('data-entity-id');
    if (!firstEntityId) throw new Error('Expected baseline Keep visible entity to expose an id.');
    const second = page.locator(`.entity-card:not([data-entity-id="${firstEntityId}"])`).first();
    const beforeScroll = await scrollBody.evaluate((node) => node.scrollTop);
    const beforeDialog = await stableBox(dialog);
    const beforeFirst = await stableBox(first);
    const beforeSecond = await stableBox(second);
    const beforeActions = await stableBox(first.locator('.entity-actions'));
    const beforeStatus = await stableBox(first.getByTestId('entity-operation-status'));

    await first.locator('[data-policy="keep_visible"]').click();
    await expect(first.locator('.entity-operation-status .inline-spinner')).toBeVisible();
    await expect(second.locator('.entity-operation-status .inline-spinner')).toHaveCount(0);
    const duringDialog = await stableBox(dialog);
    const duringFirst = await stableBox(first);
    const duringSecond = await stableBox(second);
    const duringActions = await stableBox(first.locator('.entity-actions'));
    const duringStatus = await stableBox(first.getByTestId('entity-operation-status'));
    const duringScroll = await scrollBody.evaluate((node) => node.scrollTop);
    expectBoxStable(beforeDialog, duringDialog);
    expectBoxStable(beforeFirst, duringFirst);
    expectBoxStable(beforeSecond, duringSecond);
    expectBoxStable(beforeActions, duringActions);
    expectBoxStable(beforeStatus, duringStatus);
    expect(Math.abs(duringScroll - beforeScroll)).toBeLessThanOrEqual(1);

    await expect(first.locator('.entity-operation-status .inline-spinner')).toHaveCount(0, { timeout: 20_000 });
    const afterFirst = await stableBox(first);
    const afterActions = await stableBox(first.locator('.entity-actions'));
    const afterStatus = await stableBox(first.getByTestId('entity-operation-status'));
    expectBoxStable(beforeFirst, afterFirst);
    expectBoxStable(beforeActions, afterActions);
    expectBoxStable(beforeStatus, afterStatus);
  });

  test('draws correction keyframes on the video overlay and preserves preview/save frame state', async ({ page }) => {
    test.setTimeout(180_000);
    await startReview(page);
    await page.keyboard.press('Shift+ArrowRight');
    await page.getByRole('button', { name: 'Manual redaction' }).click();
    await expect(page.locator('video')).toBeVisible();
    const overlay = page.getByTestId('correction-video-overlay');
    await expect(overlay).toBeVisible();
    await expect(overlay).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)');
    await expect(page.locator('.editor-backdrop')).toHaveCount(0);

    await page.getByRole('button', { name: 'Set start to current frame' }).click();
    await page.getByRole('button', { name: 'Set end to current frame' }).click();
    const sourceBeforePreview = await currentSourceFrame(page);
    await drawCorrectionBox(page);
    await expect(page.locator('.keyframe-row')).toContainText(new RegExp(`frame ${sourceBeforePreview}: \\d+, \\d+, \\d+, \\d+`));
    await expect(page.getByText('Draw at least one rectangle keyframe.')).toHaveCount(0);

    await expect(page.getByRole('button', { name: 'Preview correction' })).toBeEnabled();
    await page.getByRole('button', { name: 'Preview correction' }).click();
    await expect(page.locator('.version-selector select')).toHaveValue('latest_reviewed', { timeout: 120_000 });
    await expect(page.locator('.version-selector option')).toHaveText(['Original', 'Automatic', 'Latest reviewed']);
    await expect.poll(async () => currentSourceFrame(page)).toBe(sourceBeforePreview);
    await expect(page.getByRole('button', { name: 'Save correction' })).toBeEnabled({ timeout: 120_000 });
    await page.getByRole('button', { name: 'Save correction' }).click();
    await expect(page.locator('.case-subtitle')).toContainText(/Saved - updating Latest reviewed|Published/, {
      timeout: 120_000,
    });
    await expect(page.locator('.version-selector select')).toHaveValue('latest_reviewed', { timeout: 120_000 });

    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.locator('.version-selector select')).toHaveValue('latest_reviewed', { timeout: 120_000 });
    await page.getByRole('button', { name: 'History' }).click();
    await expect(page.locator('body')).toContainText('Manual correction saved');
  });

  test('validates and previews a two-keyframe multi-frame correction', async ({ page }) => {
    test.setTimeout(150_000);
    await startReview(page);
    await page.keyboard.press('Shift+ArrowRight');
    await page.getByRole('button', { name: 'Day' }).click();
    await page.getByRole('button', { name: 'Manual redaction' }).click();
    await page.getByRole('button', { name: 'Interpolate between keyframes' }).click();
    await page.getByRole('button', { name: 'Set start to current frame' }).click();
    await drawCorrectionBox(page, 0.26, 0.32, 0.42, 0.48);
    await page.keyboard.press('Shift+ArrowRight');
    await page.getByRole('button', { name: 'Set end to current frame' }).click();
    await expect(page.getByRole('button', { name: 'Preview correction' })).toBeDisabled();
    await drawCorrectionBox(page, 0.44, 0.38, 0.62, 0.6);
    await expect(page.getByRole('button', { name: 'Preview correction' })).toBeEnabled();
    const multiFrameSource = await currentSourceFrame(page);
    await page.getByRole('button', { name: 'Preview correction' }).click();
    await expect(page.locator('.version-selector select')).toHaveValue('latest_reviewed', { timeout: 120_000 });
    await expect(page.locator('.version-selector option')).toHaveText(['Original', 'Automatic', 'Latest reviewed']);
    await expect.poll(async () => currentSourceFrame(page)).toBe(multiFrameSource);
  });

  test('requires attribution for export and finalization', async ({ page }) => {
    await startReview(page);
    await page.getByRole('button', { name: 'Export' }).click();
    await expect(page.getByRole('dialog', { name: 'Reviewer attribution' })).toBeVisible();
    await page.getByLabel('Reviewer name').fill('Reviewer One');
    await page.getByRole('button', { name: 'Save attribution' }).click();
    await expect(page.locator('#sessionBanner')).toContainText('Reviewer One');

    await page.getByRole('button', { name: 'Final Results' }).click();
    await page.getByRole('button', { name: 'Check finalization' }).click();
    await expect(page.locator('.final-status')).toContainText(/Final media|Finalization|unavailable/i);
  });

  test('creates and restores a named checkpoint while preserving history', async ({ page }) => {
    test.setTimeout(120_000);
    await startReview(page);
    await page.getByPlaceholder('Name').fill('Before policy work');
    await page.getByRole('button', { name: 'Create checkpoint' }).click();
    await expect(page.getByRole('button', { name: /Restore Before policy work/ })).toBeVisible();
    const taskId = await page.locator('.task-card').first().getAttribute('data-task-id');
    if (!taskId) throw new Error('No task available for checkpoint test.');
    await page.locator(`.task-card[data-task-id="${taskId}"] .decision-correct`).click();
    await page.getByRole('button', { name: /Restore Before policy work/ }).click();
    await expect(page.getByRole('button', { name: /Restore Before policy work/ })).toBeEnabled({ timeout: 120_000 });
    await page.getByRole('button', { name: /Delete checkpoint Before policy work/ }).click();
    await expect(page.getByRole('dialog', { name: /Delete checkpoint/ })).toBeVisible();
    await page.getByRole('button', { name: 'Cancel' }).click();
    await page.getByRole('button', { name: /Delete checkpoint Before policy work/ }).click();
    await page.getByRole('dialog', { name: /Delete checkpoint/ }).getByRole('button', { name: 'Delete checkpoint', exact: true }).click();
    await page.getByRole('button', { name: 'History' }).click();
    await expect(page.locator('body')).toContainText('Checkpoint "Before policy work" created');
    await expect(page.locator('body')).toContainText('Restored checkpoint "Before policy work"');
    await expect(page.locator('body')).toContainText('Checkpoint "Before policy work" deleted');
    await expect(page.locator('details').first()).toContainText('event_hash');
  });

  test('materializes object policy changes into Latest reviewed and preserves reload state', async ({ page }) => {
    test.setTimeout(240_000);
    await startReview(page);
    await page.getByRole('button', { name: 'Night' }).click();
    await page.getByRole('button', { name: 'Entities' }).click();
    await page.getByRole('button', { name: 'People' }).click();
    const keepEntity = page.locator('.entity-card[data-category="face"]').filter({ has: page.locator('[data-policy="keep_visible"][aria-pressed="true"]') }).first();
    await expect(keepEntity).toBeVisible();
    const keepEntityId = await keepEntity.getAttribute('data-entity-id');
    if (!keepEntityId) throw new Error('Keep-policy entity did not expose an entity id.');
    await keepEntity.locator('[data-policy="redact"]').click();
    await expect(page.locator(`.entity-card[data-entity-id="${keepEntityId}"] [data-policy="redact"]`)).toHaveAttribute('aria-pressed', 'true', { timeout: 120_000 });
    await page.getByRole('button', { name: 'Close' }).first().click();
    await expect(page.locator('.version-selector select')).toHaveValue('latest_reviewed');
    await expect(page.locator('video')).toHaveAttribute(
      'data-media-path',
      /sessions\/session_.*\/media\/night\/latest_reviewed_[a-f0-9]+\.mp4/,
      { timeout: 120_000 },
    );
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.locator('.version-selector select')).toHaveValue('latest_reviewed');
    await expect(page.locator('video')).toHaveAttribute(
      'data-media-path',
      /sessions\/session_.*\/media\/night\/latest_reviewed_[a-f0-9]+\.mp4/,
      { timeout: 120_000 },
    );
  });

  test('serves direct route refresh and MP4 range requests', async ({ page, request, baseURL }) => {
    await page.goto('/app/audit', { waitUntil: 'domcontentloaded' });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: 'History' })).toBeVisible();
    const response = await request.get(`${baseURL}/media/day/original_reference.mp4`, {
      headers: { Range: 'bytes=0-15' },
    });
    expect(response.status()).toBe(206);
    expect(response.headers()['content-type']).toContain('video/mp4');
    expect((await response.body()).length).toBe(16);
  });
});
