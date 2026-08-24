import { describe, expect, it } from 'vitest';
import { MANUAL_REDACTION_CATEGORY_OPTIONS } from '../src/types/domain';
import type { CorrectionDraft } from '../src/types/domain';
import { applyCorrectionDraftMode, resetSingleFrameDraftForSourceFrame, validateCorrectionDraft } from '../src/components/CorrectionEditor';

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
  it('exposes the supported manual redaction categories', () => {
    expect(MANUAL_REDACTION_CATEGORY_OPTIONS.map((option) => option.value)).toEqual([
      'face',
      'license_plate',
      'other_sensitive',
    ]);
  });

  it('collapses an empty draft to the current source frame when single-frame mode is selected', () => {
    const patch = applyCorrectionDraftMode(
      {
        ...baseDraft,
        end: 125,
        end_source_frame: 125,
        mode: 'fixed_box',
      },
      'single_frame',
      117,
    );

    expect(patch).toMatchObject({
      end: 117,
      end_source_frame: 117,
      interpolation_method: 'none',
      keyframes: [],
      mode: 'single_frame',
      start: 117,
      start_source_frame: 117,
    });
  });

  it('allows a one-frame correction with one rectangle keyframe', () => {
    const draft = {
      ...baseDraft,
      keyframes: [{ source_frame: 100, bbox_xyxy: [10, 20, 110, 120] }],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({ valid: true, message: '' });
  });

  it('rejects a single-frame draft that carries a stale keyframe from another frame', () => {
    const draft = {
      ...baseDraft,
      keyframes: [
        { source_frame: 100, bbox_xyxy: [10, 20, 110, 120] },
        { source_frame: 101, bbox_xyxy: [12, 22, 112, 122] },
      ],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({
      valid: false,
      message: 'A keyframe is outside the selected interval.',
    });
  });

  it('resets a single-frame draft when the reviewer moves to a new source frame', () => {
    const draft = {
      ...baseDraft,
      keyframes: [{ source_frame: 89, bbox_xyxy: [10, 20, 110, 120] }],
      start: 89,
      start_source_frame: 89,
      end: 89,
      end_source_frame: 89,
    } satisfies CorrectionDraft;

    expect(resetSingleFrameDraftForSourceFrame(draft, 90)).toMatchObject({
      end: 90,
      end_source_frame: 90,
      keyframes: [],
      preview_token: undefined,
      start: 90,
      start_source_frame: 90,
    });
  });

  it.each(['face', 'license_plate', 'other_sensitive'] as const)('uses the same mode validation for %s', (category) => {
    const draft = {
      ...baseDraft,
      category,
      end: 110,
      end_source_frame: 110,
      mode: 'fixed_box',
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

  it('rejects fixed-box drafts with more than one authoritative rectangle', () => {
    const draft = {
      ...baseDraft,
      end: 110,
      end_source_frame: 110,
      mode: 'fixed_box',
      keyframes: [
        { source_frame: 100, bbox_xyxy: [10, 20, 110, 120] },
        { source_frame: 110, bbox_xyxy: [20, 30, 120, 130] },
      ],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({
      valid: false,
      message: 'Draw exactly one fixed rectangle for source frames 100-110.',
    });
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
    expect(validateCorrectionDraft(draft).message).toBe('Interpolate mode requires exactly one start and one end keyframe.');
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

  it('rejects interpolation drafts with intermediate stale keyframes', () => {
    const draft = {
      ...baseDraft,
      end: 110,
      end_source_frame: 110,
      mode: 'interpolate',
      keyframes: [
        { source_frame: 100, bbox_xyxy: [10, 20, 110, 120] },
        { source_frame: 105, bbox_xyxy: [15, 25, 115, 125] },
        { source_frame: 110, bbox_xyxy: [20, 30, 120, 130] },
      ],
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({
      valid: false,
      message: 'Interpolate mode requires exactly one start and one end keyframe.',
    });
  });

  it('keeps interpolation preview disabled for the observed 1828-1848 start-only draft', () => {
    const draft = {
      ...baseDraft,
      end: 1848,
      end_source_frame: 1848,
      keyframes: [{ source_frame: 1828, bbox_xyxy: [10, 20, 110, 120] }],
      mode: 'interpolate',
      start: 1828,
      start_source_frame: 1828,
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(draft)).toEqual({
      valid: false,
      message: 'Interpolate mode requires exactly one start and one end keyframe.',
    });
  });

  it('invalidates interpolation when the end frame changes away from an existing endpoint keyframe', () => {
    const validDraft = {
      ...baseDraft,
      end: 1848,
      end_source_frame: 1848,
      keyframes: [
        { source_frame: 1828, bbox_xyxy: [10, 20, 110, 120] },
        { source_frame: 1848, bbox_xyxy: [20, 30, 120, 130] },
      ],
      mode: 'interpolate',
      start: 1828,
      start_source_frame: 1828,
    } satisfies CorrectionDraft;

    expect(validateCorrectionDraft(validDraft)).toEqual({ valid: true, message: '' });
    expect(validateCorrectionDraft({ ...validDraft, end: 1849, end_source_frame: 1849 })).toEqual({
      valid: false,
      message: 'Draw a rectangle on end frame 1849 to preview this interpolation.',
    });
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
