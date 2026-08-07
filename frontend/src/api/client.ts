import type {
  Attribution,
  AuditEvent,
  Correction,
  CorrectionPreview,
  Entity,
  FinalizationStatus,
  Session,
  SessionState,
  TaskDecision,
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
}): Promise<{ decision: TaskDecision; event: AuditEvent }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/decisions`, payload);
  const after = response.event?.after_state as TaskDecision | undefined;
  if (!after) throw new Error('Decision response did not include persisted state.');
  if (!response.event) throw new Error('Decision response did not include an event.');
  return {
    decision: { ...after, event_hash: response.event.event_hash, event_id: response.event.event_id },
    event: response.event,
  };
}

export async function saveEntityPolicy(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  entity: Entity;
  after: EffectiveEntity;
  policy: EntityPolicy;
}): Promise<AuditEvent> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/entity-events`, {
    session_id: payload.session_id,
    actor: payload.actor,
    case_id: payload.case_id,
    event: {
      event_type: 'entity_policy_changed',
      case_id: payload.case_id,
      entity_id: payload.entity.entity_id,
      payload: {
        before: payload.entity,
        after: payload.after,
        child_track_ids: payload.entity.member_detector_tracker_ids || payload.entity.automatic_source_track_ids || [],
      },
    },
    reason: 'Reviewer object policy update.',
  });
  if (!response.event) throw new Error('Entity policy response did not include an event.');
  return response.event;
}

export async function mergeEntities(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  case_id: CaseId;
  before_state: Record<string, unknown>;
  after_state: Record<string, unknown>;
}): Promise<AuditEvent> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/entities/merge`, {
    ...payload,
    reason: 'Reviewer confirmed visual merge.',
  });
  if (!response.event) throw new Error('Merge response did not include an event.');
  return response.event;
}

export async function saveCorrection(payload: {
  session_id: string;
  actor: Attribution | Record<string, unknown>;
  correction: Correction;
  reason: string;
}): Promise<{ event: AuditEvent; preview?: CorrectionPreview }> {
  const response = await postJson<JsonResponse>(`${API_PREFIX}/corrections`, payload);
  if (!response.event) throw new Error('Correction response did not include an event.');
  return { event: response.event, preview: response.preview };
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
