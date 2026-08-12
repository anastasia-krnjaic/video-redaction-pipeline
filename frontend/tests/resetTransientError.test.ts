import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

describe('reset transient correction errors', () => {
  it('clears preview and correction-local state during case reset', () => {
    const source = fs.readFileSync(path.resolve(process.cwd(), 'src/pages/CaseReviewPage.tsx'), 'utf8');
    const resetBody = source.match(/const resetReviewState = async \(\) => \{(?<body>[\s\S]+?)\n  \};/)?.groups?.body || '';

    expect(resetBody).toContain('setPreviewReady(false)');
    expect(resetBody).toContain('setPreviewInvalidated(false)');
    expect(resetBody).toContain('setPreviewToken(undefined)');
    expect(resetBody).toContain('setPreviewRole(undefined)');
    expect(resetBody).toContain('setCorrectionSaveStatus(undefined)');
    expect(resetBody).toContain('setMaterializationError(undefined)');
    expect(resetBody).toContain('setFocusTarget(null)');
    expect(resetBody).toContain('onError(undefined)');
    expect(resetBody).toContain('setDraft(initialDraft(caseId, manifest, caseFrame, null))');
  });

  it('clears correction-editor validation errors when the draft is replaced or discarded', () => {
    const source = fs.readFileSync(path.resolve(process.cwd(), 'src/components/CorrectionEditor.tsx'), 'utf8');

    expect(source).toContain("useEffect(() => {\n    setError('');\n  }, [draft.draft_hash, draft.case_id]);");
    expect(source).toContain("setError('');\n              onDiscard();");
  });
});
