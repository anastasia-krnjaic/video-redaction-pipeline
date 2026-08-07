import type { Entity, MediaVersion, Task, TaskDecision } from './api';

export type CaseId = 'day' | 'night';
export type TopLevelRoute = CaseId | 'final' | 'audit';
export type ReviewMode = 'review' | 'edit';
export type QueueBucket = 'Open' | 'Needs correction' | 'Unsure / Escalated' | 'Completed' | 'Resolved';
export type EntityFilter = 'all' | 'face' | 'license_plate' | 'needs_review' | 'unresolved';
export type EntityPolicy = 'redact' | 'keep_visible' | 'not_sensitive';

export interface FinalMediaState {
  [caseId: string]: MediaVersion | undefined;
}

export interface CorrectionKeyframeDraft {
  source_frame: number;
  bbox_xyxy: [number, number, number, number];
}

export interface CorrectionCheckpointDraft {
  source_frame: number;
  state: 'unreviewed' | 'verified' | 'needs_adjustment' | 'not_visible' | 'blocked_by_scene_cut';
  bbox_xyxy?: [number, number, number, number] | null;
  reviewer_note?: string;
}

export interface CorrectionDraft {
  caseId: CaseId;
  start: number;
  end: number;
  keyframes: CorrectionKeyframeDraft[];
  checkpoints: CorrectionCheckpointDraft[];
  interpolation_method: 'linear' | 'none';
  selectedTaskId?: string;
  selectedEntityId?: string;
}

export interface EffectiveEntity extends Entity {
  current_status?: string;
}

export interface TaskWithDecision extends Task {
  decision?: TaskDecision;
  bucket: QueueBucket;
}
