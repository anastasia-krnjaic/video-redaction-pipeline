import { describe, expect, it } from 'vitest';
import { caseTabLabel } from '../src/utils/cases';

describe('workspace case labels', () => {
  it('keeps product Day and Night labels for canonical routes', () => {
    expect(caseTabLabel('day', 'Technical day')).toBe('Day');
    expect(caseTabLabel('night', 'Technical night')).toBe('Night');
  });

  it('uses the human display name for arbitrary source-bound internal case ids', () => {
    expect(caseTabLabel('default_face_night', 'Night')).toBe('Night');
    expect(caseTabLabel('night_exact_import_3', 'Night')).toBe('Night');
  });
});
