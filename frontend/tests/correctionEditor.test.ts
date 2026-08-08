import { describe, expect, it } from 'vitest';
import type { CorrectionDraft } from '../src/types/domain';
import { validateCorrectionDraft } from '../src/components/CorrectionEditor';

const baseDraft: CorrectionDraft = {
  caseId: 'day',
  case_id: 'day',
  category: 'face',
  barriers: [],
  checkpoints: [],
  end: 100,
  end_source_frame: 100,
  interpolation_method: 'linear',
  keyframes: [],
  mode: 'single_frame',
  start: 100,
  start_source_frame: 100,
};

describe('correction editor validation', () => {
  it('allows a one-frame correction with one rectangle keyframe', () => {
    const draft = {
      ...baseDraft,
      keyframes: [{ source_frame: 100, bbox_xyxy: [10, 20, 110, 120] }],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({ valid: true, message: '' });
  });

  it('accepts a fixed box interval with one keyframe', () => {
    const draft = {
      ...baseDraft,
      end: 110,
      end_source_frame: 110,
      mode: 'fixed_box',
      keyframes: [{ source_frame: 100, bbox_xyxy: [10, 20, 110, 120] }],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({ valid: true, message: '' });
  });

  it('requires start and end keyframes for interpolation', () => {
    const draft = {
      ...baseDraft,
      end: 110,
      end_source_frame: 110,
      mode: 'interpolate',
      keyframes: [{ source_frame: 100, bbox_xyxy: [10, 20, 110, 120] }],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft).valid).toBe(false);
    expect(validateCorrectionDraft(draft).message).toBe('Add an end-frame rectangle on source frame 110.');
    expect(
      validateCorrectionDraft({
        ...draft,
        keyframes: [
          { source_frame: 100, bbox_xyxy: [10, 20, 110, 120] },
          { source_frame: 110, bbox_xyxy: [20, 30, 120, 130] },
        ],
      }),
    ).toEqual({ valid: true, message: '' });
  });

  it('blocks linear interpolation through scene-cut and not-visible barriers', () => {
    const draft = {
      ...baseDraft,
      end: 110,
      keyframes: [
        { source_frame: 100, bbox_xyxy: [10, 20, 110, 120] },
        { source_frame: 110, bbox_xyxy: [20, 30, 120, 130] },
      ],
      checkpoints: [{ source_frame: 105, state: 'blocked_by_scene_cut' }],
      barriers: [{ source_frame: 105, state: 'blocked_by_scene_cut' }],
      mode: 'interpolate',
      end_source_frame: 110,
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft).valid).toBe(false);
  });
});
