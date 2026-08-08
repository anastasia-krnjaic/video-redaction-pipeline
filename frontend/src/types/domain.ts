import type { Entity, MediaVersion, Task, TaskDecision, WorkingState } from './api';

export type CaseId = 'day' | 'night';
export type TopLevelRoute = CaseId | 'final' | 'audit';
export type ReviewMode = 'review' | 'edit';
export type QueueBucket = 'Open' | 'Needs correction' | 'Unsure / Escalated' | 'Completed';
export type EntityFilter = 'all' | 'face' | 'license_plate' | 'needs_review' | 'unresolved';
export type EntityPolicy = 'redact' | 'keep_visible' | 'not_sensitive' | 'review_required';
export type CorrectionIntervalMode = 'single_frame' | 'fixed_box' | 'interpolate';

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
  case_id: CaseId;
  category: string;
  start: number;
  end: number;
  start_source_frame: number;
  end_source_frame: number;
  mode: CorrectionIntervalMode;
  keyframes: CorrectionKeyframeDraft[];
  checkpoints: CorrectionCheckpointDraft[];
  barriers: CorrectionCheckpointDraft[];
  interpolation_method: 'linear' | 'none';
  holdSameBox?: boolean;
  draft_hash?: string;
  preview_token?: string;
  selectedTaskId?: string;
  selectedEntityId?: string;
}

export interface EffectiveEntity extends Entity {
  current_status?: string;
  representative_bbox?: [number, number, number, number] | number[] | null;
  aliases?: string[];
  source_entity_ids?: string[];
}

export interface MaterializationJob {
  accepted_working_state_revision?: string | null;
  affected_entity_ids?: string[];
  case_id: string;
  completed_at_utc?: string | null;
  created_at_utc?: string | null;
  error?: string | null;
  job_id: string;
  latest_reviewed?: MediaVersion | null;
  materialization?: Record<string, unknown> | null;
  requested_policy?: string | null;
  session_id?: string;
  started_at_utc?: string | null;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'superseded';
  superseded_by?: string | null;
  working_state?: WorkingState;
  [key: string]: unknown;
}

export interface TaskWithDecision extends Task {
  decision?: TaskDecision;
  bucket: QueueBucket;
}
