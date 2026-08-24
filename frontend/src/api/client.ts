import type {
  Attribution,
  AuditEvent,
  Correction,
  CorrectionPreview,
  Entity,
  FinalizationStatus,
  MediaVersion,
  MaterializationJob,
  Session,
  SessionState,
  TaskDecision,
  Task,
  WorkingState,
  WorkspaceSummary,
} from '../types/api';
import { API_PREFIX } from '../types/api';
import type { CaseId, EffectiveEntity, EntityPolicy } from '../types/domain';

interface JsonResponse {
  ok?: boolean;
  error?: string;
  event?: AuditEvent;
  session?: Session;
  attribution?: Attribution;
  preview?: CorrectionPreview;
  result?: FinalizationStatus;
  events?: AuditEvent[];
  entity?: EffectiveEntity;
  entities?: EffectiveEntity[];
  latest_reviewed?: MediaVersion;
  materialization?: Record<string, unknown>;
  materialization_job?: MaterializationJob;
  materialization_jobs?: MaterializationJob[];
  materialization_error?: string | null;
  verification_task?: Task;
  working_state?: WorkingState;
  checkpoint?: Record<string, unknown>;
  checkpoints?: Record<string, unknown>[];
  job?: MaterializationJob;
  merged_entity?: EffectiveEntity;
  retired_entity_uuids?: string[];
  resulting_policy?: string;
  [key: string]: unknown;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: 'no-store' });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || (payload as JsonResponse).ok === false) {
    throw new Error((payload as JsonResponse).error || `${path}: ${response.status}`);
  }
  return payload as T;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || (data as JsonResponse).ok === false) {
    throw new Error((data as JsonResponse).error || `${path}: ${response.status}`);
  }
  return data as T;
}

export function loadWorkspace(): Promise<WorkspaceSummary> {
  return getJson<WorkspaceSummary>(`${API_PREFIX}/workspace/summary`);
}

export function loadSessionState(sessionId: string): Promise<SessionState> {
  return getJson<SessionState>(`${API_PREFIX}/session-state?session_id=${encodeURIComponent(sessionId)}`);
}

export function loadCurrentSessionState(): Promise<SessionState> {
  return getJson<SessionState>(`${API_PREFIX}/session-state`);
}

export function loadWorkingState(sessionId: string): Promise<{ ok: true; working_state: WorkingState }> {
  return getJson<{ ok: true; working_state: WorkingState }>(`${API_PREFIX}/working-state?session_id=${encodeURIComponent(sessionId)}`);
}

export function loadCaseVersions(caseId: CaseId, sessionId?: string): Promise<{ ok: true; versions: MediaVersion[] }> {
  const suffix = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return getJson<{ ok: true; versions: MediaVersion[] }>(`${API_PREFIX}/cases/${caseId}/versions${suffix}`);
}

export async function startSession(sourceRevision?: unknown): Promise<Session> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/sessions`, {
    reviewer: {},
    source_revision: sourceRevision || null,
  });
  if (!response.session) throw new Error('Session response did not include a session.');
  return response.session;
}

export async function saveDecision(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  task_id: string;
  action: string;
  note?: string;
}): Promise<{ decision: TaskDecision; event: AuditEvent; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/decisions`, payload);
  const after = response.event?.after_state as TaskDecision | undefined;
  if (!after) throw new Error('Decision response did not include persisted state.');
  if (!response.event) throw new Error('Decision response did not include an event.');
  return {
    decision: { ...after, event_hash: response.event.event_hash, event_id: response.event.event_id },
    event: response.event,
    working_state: response.working_state,
  };
}

export async function saveDecisionGroup(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  review_group_id: string;
  task_ids: string[];
  action: string;
  note?: string;
}): Promise<{ events: AuditEvent[]; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/decision-groups`, payload);
  return {
    events: response.events || [],
    working_state: response.working_state,
  };
}

export async function saveEntityPolicy(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  entity: Entity;
  after: EffectiveEntity;
  policy: EntityPolicy;
}): Promise<{
  entity: EffectiveEntity;
  event: AuditEvent;
  events: AuditEvent[];
  latest_reviewed?: MediaVersion;
  materialization?: Record<string, unknown>;
  materialization_job?: MaterializationJob;
  verification_task?: Task;
  working_state?: WorkingState;
}> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/object-policy`, {
    session_id: payload.session_id,
    actor: payload.actor,
    case_id: payload.case_id,
    entity_id: payload.entity.entity_id,
    policy: payload.policy,
  });
  if (!response.event) throw new Error('Entity policy response did not include an event.');
  if (!response.entity) throw new Error('Entity policy response did not include persisted entity state.');
  return {
    entity: response.entity,
    event: response.event,
    events: response.events || [response.event],
    latest_reviewed: response.latest_reviewed,
    materialization: response.materialization,
    materialization_job: response.materialization_job,
    verification_task: response.verification_task,
    working_state: response.working_state,
  };
}

export async function saveBulkEntityPolicy(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  entity_ids: string[];
  policy: EntityPolicy;
}): Promise<{
  entities: EffectiveEntity[];
  event: AuditEvent;
  events: AuditEvent[];
  latest_reviewed?: MediaVersion;
  materialization?: Record<string, unknown>;
  materialization_job?: MaterializationJob;
  working_state?: WorkingState;
}> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/bulk-object-policy`, payload);
  if (!response.event) throw new Error('Bulk policy response did not include an event.');
  return {
    entities: response.entities || [],
    event: response.event,
    events: response.events || [response.event],
    latest_reviewed: response.latest_reviewed,
    materialization: response.materialization,
    materialization_job: response.materialization_job,
    working_state: response.working_state,
  };
}

export async function mergeEntities(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  entity_ids: string[];
}): Promise<{
  event: AuditEvent;
  merged_entity: EffectiveEntity;
  retired_entity_uuids: string[];
  resulting_policy?: string;
  working_state?: WorkingState;
}> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/entities/merge`, {
    ...payload,
    reason: 'Reviewer confirmed visual merge.',
  });
  if (!response.event) throw new Error('Merge response did not include an event.');
  if (!response.merged_entity) throw new Error('Merge response did not include the canonical merged entity.');
  return {
    event: response.event,
    merged_entity: response.merged_entity,
    retired_entity_uuids: response.retired_entity_uuids || [],
    resulting_policy: response.resulting_policy,
    working_state: response.working_state,
  };
}

export async function saveCorrection(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  correction: Correction;
  reason: string;
  preview_token?: string;
}): Promise<{
  event: AuditEvent;
  events?: AuditEvent[];
  materialization?: Record<string, unknown>;
  materialization_job?: MaterializationJob;
  materialization_required?: boolean;
  publication_status?: string;
  preview?: CorrectionPreview;
  verification_task?: Task;
  working_state?: WorkingState;
}> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/corrections`, payload);
  if (!response.event) throw new Error('Correction response did not include an event.');
  if (response.materialization_error) throw new Error(response.materialization_error);
  return {
    event: response.event,
    events: response.events,
    materialization: response.materialization,
    materialization_job: response.materialization_job,
    materialization_required: typeof response.materialization_required === 'boolean' ? response.materialization_required : undefined,
    publication_status: typeof response.publication_status === 'string' ? response.publication_status : undefined,
    preview: response.preview,
    verification_task: response.verification_task,
    working_state: response.working_state,
  };
}

export async function previewCorrection(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  correction: Correction;
  reason: string;
}): Promise<{ event?: AuditEvent; preview: CorrectionPreview }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/corrections/preview`, payload);
  if (!response.preview) throw new Error('Correction preview response did not include preview media.');
  return { event: response.event, preview: response.preview };
}

export async function createCheckpoint(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  name: string;
  note?: string;
}): Promise<{ checkpoint: Record<string, unknown>; event: AuditEvent; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/checkpoints`, payload);
  if (!response.event || !response.checkpoint) throw new Error('Checkpoint response was incomplete.');
  return { checkpoint: response.checkpoint, event: response.event, working_state: response.working_state };
}

export async function restoreCheckpoint(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  checkpoint_uuid: string;
  case_id?: CaseId;
}): Promise<{ checkpoint: Record<string, unknown>; event: AuditEvent; materialization_jobs?: MaterializationJob[]; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/checkpoints/restore`, payload);
  if (!response.event || !response.checkpoint) throw new Error('Restore checkpoint response was incomplete.');
  return { checkpoint: response.checkpoint, event: response.event, materialization_jobs: response.materialization_jobs, working_state: response.working_state };
}

export async function resetToAutomaticBaseline(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id?: CaseId;
}): Promise<{ event: AuditEvent; materialization_jobs?: MaterializationJob[]; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/checkpoints/reset`, payload);
  if (!response.event) throw new Error('Reset response did not include an event.');
  return { event: response.event, materialization_jobs: response.materialization_jobs, working_state: response.working_state };
}

export async function restartReviewWorkflow(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
}): Promise<{ event: AuditEvent; materialization_jobs?: MaterializationJob[]; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/checkpoints/restart-review`, payload);
  if (!response.event) throw new Error('Restart response did not include an event.');
  return { event: response.event, materialization_jobs: response.materialization_jobs, working_state: response.working_state };
}

export async function deleteCheckpoint(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  checkpoint_uuid: string;
}): Promise<{ checkpoint: Record<string, unknown>; event: AuditEvent; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/checkpoints/delete`, payload);
  if (!response.event || !response.checkpoint) throw new Error('Delete checkpoint response was incomplete.');
  return { checkpoint: response.checkpoint, event: response.event, working_state: response.working_state };
}

export function loadMaterializationJob(jobId: string): Promise<{ ok: true; job: MaterializationJob }> {
  return getJson<{ ok: true; job: MaterializationJob }>(`${API_PREFIX}/materialization-jobs/${encodeURIComponent(jobId)}`);
}

export async function retryMaterializationJob(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  job_id: string;
}): Promise<{ job: MaterializationJob }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/materialization-jobs/retry`, payload);
  if (!response.job) throw new Error('Retry response did not include a job.');
  return { job: response.job };
}

export async function requestLatestReviewedBuild(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
}): Promise<{ job?: MaterializationJob; latest_reviewed?: MediaVersion; materialization?: Record<string, unknown>; materialization_job?: MaterializationJob; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/materialization-jobs`, payload);
  return {
    job: response.job,
    latest_reviewed: response.latest_reviewed,
    materialization: response.materialization,
    materialization_job: response.materialization_job,
    working_state: response.working_state,
  };
}

export async function acknowledgeFullVideoSweep(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  viewed_ranges?: Record<string, unknown>[];
}): Promise<{ event: AuditEvent; working_state?: WorkingState }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/events/sweep`, {
    session_id: payload.session_id,
    actor: payload.actor,
    case_id: payload.case_id,
    viewed_ranges: payload.viewed_ranges || [],
  });
  if (!response.event) throw new Error('Sweep acknowledgement response did not include an event.');
  return { event: response.event, working_state: response.working_state };
}

export async function loadFinalizationStatus(payload: {
  case_id: CaseId;
  session_id: string;
}): Promise<FinalizationStatus> {
  const response = await getJson<JsonResponse>(
    `${API_PREFIX}/finalization-status?case_id=${encodeURIComponent(payload.case_id)}&session_id=${encodeURIComponent(payload.session_id)}`,
  );
  if (!response.result) throw new Error('Finalization status response did not include a result.');
  return response.result;
}

export async function setReviewerAttribution(payload: {
  session_id: string;
  display_name: string;
  actor: Attribution | Record<string, unknown>;
}): Promise<{ attribution: Attribution; event: AuditEvent }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/reviewer-attribution`, {
    ...payload,
    note: 'Reviewer attribution set.',
  });
  if (!response.event) throw new Error('Attribution response did not include an event.');
  return { attribution: response.attribution || null, event: response.event };
}

export async function finalizeCase(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  finalization_request: Record<string, unknown>;
}): Promise<{ result: FinalizationStatus; event?: AuditEvent }> {
  const response = await fetch(`${API_PREFIX}/finalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = (await response.json().catch(() => ({}))) as JsonResponse;
  if (!response.ok) throw new Error(data.error || `${API_PREFIX}/finalize: ${response.status}`);
  if (!data.result) throw new Error('Finalization response did not include a result.');
  return { result: data.result, event: data.event };
}
