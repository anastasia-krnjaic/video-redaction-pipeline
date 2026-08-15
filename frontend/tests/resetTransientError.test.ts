import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

describe('reset transient correction errors', () => {
  it('clears preview and correction-local state during case reset', () => {
    const source = fs.readFileSync(path.resolve(process.cwd(), 'src/pages/CaseReviewPage.tsx'), 'utf8');
    const resetBody = source.match(/const resetReviewState = async \(\) => \{(?<body>[\s\S]+?)\n  \};/)?.groups?.body || '';
    const resetLocalBody = source.match(/const resetLocalCorrectionState = \(nextTask: TaskWithDecision \| null = null\) => \{(?<body>[\s\S]+?)\n  \};/)?.groups?.body || '';

    expect(resetBody).toContain('resetLocalCorrectionState(null)');
    expect(resetLocalBody).toContain('setPreviewReady(false)');
    expect(resetLocalBody).toContain('setPreviewInvalidated(false)');
    expect(resetLocalBody).toContain('setPreviewToken(undefined)');
    expect(resetLocalBody).toContain('setPreviewRole(undefined)');
    expect(resetLocalBody).toContain('setCorrectionSaveStatus(undefined)');
    expect(resetBody).toContain('setMaterializationError(undefined)');
    expect(resetBody).toContain('setFocusTarget(null)');
    expect(resetBody).toContain('onError(undefined)');
    expect(resetLocalBody).toContain('setDraft(initialDraft(caseId, manifest, caseFrame, nextTask))');
  });

  it('clears correction-editor validation errors when the draft is replaced or discarded', () => {
    const source = fs.readFileSync(path.resolve(process.cwd(), 'src/components/CorrectionEditor.tsx'), 'utf8');

    expect(source).toContain("useEffect(() => {\n    setError('');\n  }, [draft.draft_hash, draft.case_id]);");
    expect(source).toContain("setError('');\n              onDiscard();");
  });

  it('keeps full-video sweep approval non-optimistic until authoritative state is refreshed', () => {
    const source = fs.readFileSync(path.resolve(process.cwd(), 'src/pages/CaseReviewPage.tsx'), 'utf8');
    const sweepBody = source.match(/const acknowledgeSweep = async \(\) => \{(?<body>[\s\S]+?)\n  \};/)?.groups?.body || '';
    const approvalButton = source.match(/<button\s+type="button"\s+className="primary"(?<body>[\s\S]+?)<\/button>/)?.groups?.body || '';

    expect(sweepBody).toContain('setSweepBusy(true)');
    expect(sweepBody).toContain('authoritativeWorkingStateAfterSweepApproval(');
    expect(sweepBody).toContain('loadWorkingState');
    expect(sweepBody).toContain('applyWorkingStateProjection(refreshedWorkingState)');
    expect(sweepBody).toContain('Full-video review was approved, but the refreshed review state could not be loaded');
    expect(sweepBody).toContain('setSweepBusy(false)');
    expect(approvalButton).toContain('disabled={!latestReviewedCurrent || sweepBusy || (sweepAcknowledged && !sweepStale)}');
    expect(source).toContain('workingState?.current_event_chain_head');
    expect(source).not.toContain('setSweepAcknowledged');
  });
});
