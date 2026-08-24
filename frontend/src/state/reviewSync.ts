import type { AuditEvent, TaskDecision, WorkingState } from '../types/api';

export function decisionFromEvent(event: AuditEvent): TaskDecision | undefined {
  if (event.event_type !== 'decision_recorded') return undefined;
  const after = event.after_state || {};
  const action = String(after.action || '');
  const caseId = String(after.case_id || event.case_id || '');
  const taskId = String(after.task_id || event.task_id || '');
  if (!action || !caseId || !taskId) return undefined;
  return {
    action,
    case_id: caseId,
    decision_status: after.decision_status === 'blocking' ? 'blocking' : 'green',
    event_hash: event.event_hash,
    event_id: event.event_id,
    note: typeof after.note === 'string' ? after.note : undefined,
    reviewed_at_utc: typeof after.reviewed_at_utc === 'string' ? after.reviewed_at_utc : undefined,
    task_id: taskId,
  };
}

export function workingStateIsOlderThanKnownEvents(
  state: WorkingState,
  knownEvents: AuditEvent[],
  incomingEvents: AuditEvent[] = [],
): boolean {
  const stateHead = String(state.current_event_chain_head || '');
  const knownHead = latestKnownEventHash(knownEvents);
  if (!stateHead || !knownHead || stateHead === knownHead) return false;
  return eventHashIsAncestor(stateHead, knownHead, [...knownEvents, ...incomingEvents]);
}

function latestKnownEventHash(events: AuditEvent[]): string | undefined {
  return events.length ? events[events.length - 1]?.event_hash : undefined;
}

function eventHashIsAncestor(candidateHash: string, headHash: string, events: AuditEvent[]): boolean {
  const byHash = new Map(events.map((event) => [event.event_hash, event]));
  let cursor: string | null | undefined = headHash;
  for (let depth = 0; cursor && depth <= events.length; depth += 1) {
    const event = byHash.get(cursor);
    const previous = event?.previous_event_hash;
    if (previous === candidateHash) return true;
    cursor = previous;
  }
  return false;
}
