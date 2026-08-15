import type { Entity, ManualPrivacyCategory, MediaVersion, Task, TaskDecision, WorkingState } from './api';

export type CaseId = string;
export type TopLevelRoute = CaseId | 'final' | 'audit';
export type ReviewMode = 'review' | 'edit';
export type QueueBucket = 'Open' | 'Needs correction' | 'Correct' | 'Not sensitive';
export type EntityFilter = 'all' | 'face' | 'license_plate' | 'needs_review' | 'unresolved';
export type EntityPolicy = 'redact' | 'keep_visible' | 'not_sensitive' | 'review_required';
export type CorrectionIntervalMode = 'single_frame' | 'fixed_box' | 'interpolate';
export type ManualRedactionCategory = ManualPrivacyCategory;
export type DecisionsByCase = Record<CaseId, Record<string, TaskDecision>>;
export type TaskResolutionsByCase = Record<CaseId, Record<string, TaskDecision>>;

export const MANUAL_REDACTION_CATEGORIES = ['face', 'license_plate', 'other_sensitive'] as const satisfies readonly ManualRedactionCategory[];
export const MANUAL_REDACTION_CATEGORY_OPTIONS: readonly {
  label: string;
  value: ManualRedactionCategory;
}[] = [
  { value: 'face', label: 'Face / person' },
  { value: 'license_plate', label: 'Plate' },
  { value: 'other_sensitive', label: 'Other sensitive region' },
];

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
  category: ManualRedactionCategory;
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
  changed_segment_count?: number;
  changed_source_frame_count?: number;
  changed_source_frame_intervals?: Record<string, unknown>[];
  changed_source_frame_percentage?: number;
  completed_at_utc?: string | null;
  created_at_utc?: string | null;
  current_stage?: string | null;
  error?: string | null;
  job_id: string;
  latest_reviewed?: MediaVersion | null;
  materialization?: Record<string, unknown> | null;
  materialization_plan?: Record<string, unknown> | null;
  planned_frames_to_encode?: number;
  planned_frames_to_render?: number;
  progress?: number;
  published_working_state_revision?: string | null;
  requested_policy?: string | null;
  reused_segment_count?: number;
  session_id?: string;
  started_at_utc?: string | null;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'superseded' | 'skipped_no_effective_region_change';
  superseded_by?: string | null;
  working_state?: WorkingState;
  [key: string]: unknown;
}

export interface TaskWithDecision extends Task {
  decision?: TaskDecision;
  resolution?: TaskDecision;
  bucket: QueueBucket;
}

export interface ReviewGroup extends TaskWithDecision {
  child_task_ids: string[];
  child_tasks: TaskWithDecision[];
  group_id: string;
  is_review_group: true;
  optional_group: boolean;
  reason_chips: string[];
  segment_count: number;
}
