import type { MediaVersion, Task } from '../types/api';
import type { CaseId, CorrectionDraft } from '../types/domain';
import type { PendingCorrectionRegion } from '../components/CorrectionEditor';

export function validRoleOrAutomatic(role: string, roles: MediaVersion[]): string {
  if (roles.some((item) => item.role === role)) return role;
  if (roles.some((item) => item.role === 'automatic')) return 'automatic';
  return defaultRole(roles);
}

export function automaticOrDefaultNonLatest(roles: MediaVersion[]): string {
  if (roles.some((item) => item.role === 'automatic')) return 'automatic';
  if (roles.some((item) => item.role === 'original')) return 'original';
  return defaultRole(roles);
}

export function roleAfterCorrectionSave(role: string, roles: MediaVersion[], latestBecomesStale: boolean): string {
  if (role === 'correction_preview') return automaticOrDefaultNonLatest(roles);
  if (latestBecomesStale && role === 'latest_reviewed') return automaticOrDefaultNonLatest(roles);
  return validRoleOrAutomatic(role, roles);
}

export function draftHasRenderableGeometry(draft: CorrectionDraft): boolean {
  return draft.keyframes.some((keyframe) => {
    const [x1, y1, x2, y2] = keyframe.bbox_xyxy;
    return Number.isFinite(x1) && Number.isFinite(y1) && x2 > x1 && y2 > y1;
  });
}

export function pendingRegionsForSourceFrame(
  corrections: Record<string, unknown>[],
  manualTasks: Task[],
  caseId: CaseId,
  sourceFrame: number,
): PendingCorrectionRegion[] {
  const pendingIds = new Set(
    manualTasks
      .filter((task) => task.case_id === caseId)
      .filter((task) => task.normal_task_class === 'correction_verification')
      .filter((task) => task.actionable === false && task.publication_status !== 'published')
      .map((task) => String(task.correction_id || correctionIdFromTask(task)))
      .filter(Boolean),
  );
  if (!pendingIds.size) return [];
  return corrections
    .filter((correction) => String(correction.case_id || '') === caseId)
    .filter((correction) => pendingIds.has(String(correction.correction_id || '')))
    .filter((correction) => correction.operation === 'add_privacy_region' || correction.operation === 'replace_privacy_region')
    .map((correction) => {
      const bbox = correctionBoxAtSourceFrame(correction, sourceFrame);
      if (!bbox) return null;
      return {
        bbox_xyxy: bbox,
        correction_id: String(correction.correction_id || ''),
        source_frame: sourceFrame,
      };
    })
    .filter((region): region is PendingCorrectionRegion => Boolean(region));
}

function defaultRole(roles: MediaVersion[]): string {
  if (roles.some((role) => role.role === 'latest_reviewed')) return 'latest_reviewed';
  if (roles.some((role) => role.role === 'automatic')) return 'automatic';
  return 'original';
}

function correctionIdFromTask(task: Task): string {
  const casePrefix = `correction_verify_${task.case_id}_`;
  if (task.review_task_id.startsWith(casePrefix)) return task.review_task_id.slice(casePrefix.length);
  return '';
}

function correctionBoxAtSourceFrame(
  correction: Record<string, unknown>,
  sourceFrame: number,
): [number, number, number, number] | null {
  const start = Number(correction.source_frame_start);
  const end = Number(correction.source_frame_end);
  if (!Number.isFinite(start) || !Number.isFinite(end) || sourceFrame < start || sourceFrame > end) {
    return null;
  }
  const keyframes = (Array.isArray(correction.keyframe_boxes) ? correction.keyframe_boxes : [])
    .map((item) => item as { source_frame?: unknown; bbox_xyxy?: unknown })
    .map((item) => ({
      source_frame: Number(item.source_frame),
      bbox_xyxy: Array.isArray(item.bbox_xyxy) ? item.bbox_xyxy.map(Number) : [],
    }))
    .filter((item) => Number.isFinite(item.source_frame) && item.bbox_xyxy.length === 4)
    .sort((left, right) => left.source_frame - right.source_frame);
  if (!keyframes.length) return null;
  if (correction.interpolation_method === 'none') {
    if (keyframes.length === 1) return tupleBox(keyframes[0].bbox_xyxy);
    const match = keyframes.find((keyframe) => keyframe.source_frame === sourceFrame);
    return match ? tupleBox(match.bbox_xyxy) : null;
  }
  if (keyframes.length === 1) return tupleBox(keyframes[0].bbox_xyxy);
  for (let index = 0; index < keyframes.length - 1; index += 1) {
    const left = keyframes[index];
    const right = keyframes[index + 1];
    if (sourceFrame < left.source_frame || sourceFrame > right.source_frame) continue;
    const span = Math.max(1, right.source_frame - left.source_frame);
    const t = (sourceFrame - left.source_frame) / span;
    return tupleBox(left.bbox_xyxy.map((value, axis) => Math.round((1 - t) * value + t * right.bbox_xyxy[axis])));
  }
  return null;
}

function tupleBox(values: number[]): [number, number, number, number] {
  return [values[0], values[1], values[2], values[3]];
}
