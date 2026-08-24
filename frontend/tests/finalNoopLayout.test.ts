import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const source = (relativePath: string) =>
  fs.readFileSync(path.resolve(process.cwd(), relativePath), 'utf8');

describe('final no-op navigation and status UI contracts', () => {
  it('uses a 60/40 review layout on desktop', () => {
    const css = source('src/styles/app.css');

    expect(css).toContain('grid-template-columns: minmax(0, 3fr) minmax(320px, 2fr);');
  });

  it('does not show the old empty Guidance-clear card', () => {
    const queue = source('src/components/ReviewQueue.tsx');

    expect(queue).not.toContain('GUIDANCE CLEAR');
    expect(queue).not.toContain('Ready to build reviewed video.</p>');
  });

  it('keeps case navigation from using the stale global rebuild copy path', () => {
    const page = source('src/pages/CaseReviewPage.tsx');

    expect(page).not.toContain('Reviewed video changed — rebuilding...');
    expect(page).not.toContain('claimLatestReviewedAutoBuildRequest');
    expect(page).not.toContain('latestReviewedBuildRequestKey');
    expect(page).toContain('const requestReviewedVideoBuild = async () =>');
    expect(page).toContain("'Build reviewed video'");
    expect(page).toContain("group.key === 'latest_reviewed'");
    expect(page).toContain("'Full-video review approved'");
    expect(page).toContain("'Full-video approval stale'");
  });

  it('auto-loads finalization status without invoking Final creation', () => {
    const page = source('src/pages/FinalResultsPage.tsx');
    const effectBody = page.match(/caseIds\.forEach\(\(targetCaseId\) => void load\(targetCaseId\)\);/)?.[0] || '';
    const readyIndex = page.indexOf("? 'Ready to create Final'");
    const historicalIndex = page.indexOf("? 'Final historical");

    expect(page).toContain('loadFinalizationStatus({');
    expect(effectBody).toContain('caseIds.forEach');
    expect(page).toContain('Loading finalization status...');
    expect(page).not.toContain('No finalization result has been checked for this local session.');
    expect(page).toContain('onClick={createFinal}');
    expect(page).not.toContain('void createFinal()');
    expect(readyIndex).toBeGreaterThan(-1);
    expect(historicalIndex).toBeGreaterThan(-1);
    expect(readyIndex).toBeLessThan(historicalIndex);
  });
});
