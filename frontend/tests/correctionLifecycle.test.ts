import { describe, expect, it } from 'vitest';
import {
  draftHasRenderableGeometry,
  pendingRegionsForSourceFrame,
  roleAfterCorrectionSave,
  validRoleOrAutomatic,
} from '../src/state/correctionLifecycle';
import type { MediaVersion, Task } from '../src/types/api';
import type { CorrectionDraft } from '../src/types/domain';

const roles: MediaVersion[] = [
  { label: 'Original', path: 'media/day/original.mp4', role: 'original', source_frame_end: 99, source_frame_start: 0 },
  { label: 'Automatic', path: 'media/day/automatic.mp4', role: 'automatic', source_frame_end: 99, source_frame_start: 0 },
  { label: 'Latest reviewed', path: 'latest/day/latest.mp4', role: 'latest_reviewed', source_frame_end: 99, source_frame_start: 0 },
];

const draft: CorrectionDraft = {
  barriers: [],
  caseId: 'day',
  case_id: 'day',
  category: 'license_plate',
  checkpoints: [],
  end: 89,
  end_source_frame: 89,
  interpolation_method: 'none',
  keyframes: [],
  mode: 'single_frame',
  start: 89,
  start_source_frame: 89,
};

const correction = {
  case_id: 'day',
  category: 'license_plate',
  correction_id: 'corr_pending',
  interpolation_method: 'linear',
  keyframe_boxes: [
    { bbox_xyxy: [10, 20, 30, 40], source_frame: 89 },
    { bbox_xyxy: [14, 22, 34, 42], source_frame: 91 },
  ],
  operation: 'add_privacy_region',
  source_frame_end: 91,
  source_frame_start: 89,
};

function verificationTask(patch: Partial<Task> = {}): Task {
  const base: Task = {
    actionable: false,
    case_id: 'day',
    category: 'license_plate',
    correction_id: 'corr_pending',
    normal_task_class: 'correction_verification',
    publication_status: 'pending_explicit_build',
    review_task_id: 'correction_verify_day_corr_pending',
    source_frame_end: 91,
    source_frame_start: 89,
    technical_only: false,
    title: 'Verify saved correction',
    what_to_inspect: 'Confirm the saved correction is rendered.',
  };
  return { ...base, ...patch };
}

describe('correction lifecycle UI helpers', () => {
  it('falls back from stale Latest to Automatic after Save without forcing Original', () => {
    expect(roleAfterCorrectionSave('latest_reviewed', roles, true)).toBe('automatic');
    expect(roleAfterCorrectionSave('correction_preview', roles, true)).toBe('automatic');
    expect(roleAfterCorrectionSave('automatic', roles, true)).toBe('automatic');
    expect(roleAfterCorrectionSave('original', roles, true)).toBe('original');
    expect(validRoleOrAutomatic('missing_role', roles)).toBe('automatic');
  });

  it('shows saved overlays only for backend-pending correction verification tasks', () => {
    expect(pendingRegionsForSourceFrame([correction], [verificationTask()], 'day', 90)).toEqual([
      {
        bbox_xyxy: [12, 21, 32, 41],
        correction_id: 'corr_pending',
        source_frame: 90,
      },
    ]);
    expect(pendingRegionsForSourceFrame([correction], [], 'day', 90)).toEqual([]);
    expect(pendingRegionsForSourceFrame([correction], [verificationTask({ publication_status: 'published' })], 'day', 90)).toEqual([]);
    expect(pendingRegionsForSourceFrame([correction], [verificationTask()], 'night', 90)).toEqual([]);
  });

  it('identifies local draft geometry so pending saved overlays can be suppressed', () => {
    expect(draftHasRenderableGeometry(draft)).toBe(false);
    expect(draftHasRenderableGeometry({
      ...draft,
      keyframes: [{ bbox_xyxy: [10, 20, 10, 40], source_frame: 89 }],
    })).toBe(false);
    expect(draftHasRenderableGeometry({
      ...draft,
      keyframes: [{ bbox_xyxy: [10, 20, 30, 40], source_frame: 89 }],
    })).toBe(true);
  });
});
