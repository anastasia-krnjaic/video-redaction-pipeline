import type { AuditEvent, TaskDecision, WorkingState } from '../types/api';
import type { DecisionsByCase, TaskResolutionsByCase } from '../types/domain';

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

export function eventListIsOlderThanKnownEvents(
  incomingEvents: AuditEvent[],
  knownEvents: AuditEvent[],
): boolean {
  const incomingHead = latestKnownEventHash(incomingEvents);
  const knownHead = latestKnownEventHash(knownEvents);
  if (!incomingHead || !knownHead || incomingHead === knownHead) return false;
  return eventHashIsAncestor(incomingHead, knownHead, [...knownEvents, ...incomingEvents]);
}

export function mergeAuditEvents(current: AuditEvent[], incoming: AuditEvent[]): AuditEvent[] {
  if (!incoming.length) return current;
  const merged = [...current];
  const seen = new Set(current.map(eventIdentity).filter(Boolean));
  for (const event of incoming) {
    const identity = eventIdentity(event);
    if (identity && seen.has(identity)) continue;
    if (identity) seen.add(identity);
    merged.push(event);
  }
  return orderEventsByHashChain(merged);
}

export function mergeDecisionsByEventOrder<T extends DecisionsByCase | TaskResolutionsByCase>(
  current: T,
  incoming: T,
  events: AuditEvent[],
): T {
  const order = eventOrderIndex(events);
  const next: DecisionsByCase | TaskResolutionsByCase = { ...current };
  for (const [caseId, incomingCase] of Object.entries(incoming || {})) {
    const currentCase = { ...(next[caseId] || {}) };
    for (const [taskId, incomingDecision] of Object.entries(incomingCase || {})) {
      const existing = currentCase[taskId];
      if (decisionIsNewerOrSame(existing, incomingDecision, order)) {
        currentCase[taskId] = incomingDecision;
      }
    }
    next[caseId] = currentCase;
  }
  return next as T;
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

function eventIdentity(event: AuditEvent): string {
  return String(event.event_hash || event.event_id || '');
}

function orderEventsByHashChain(events: AuditEvent[]): AuditEvent[] {
  const originalIndex = new Map<string, number>();
  const byHash = new Map<string, AuditEvent>();
  events.forEach((event, index) => {
    originalIndex.set(eventIdentity(event) || `index:${index}`, index);
    const hash = String(event.event_hash || '');
    if (hash) byHash.set(hash, event);
  });
  const depthCache = new Map<string, number>();
  const depthFor = (event: AuditEvent, visiting = new Set<string>()): number => {
    const hash = String(event.event_hash || '');
    if (!hash) return originalIndex.get(eventIdentity(event)) ?? 0;
    const cached = depthCache.get(hash);
    if (cached !== undefined) return cached;
    if (visiting.has(hash)) return originalIndex.get(hash) ?? 0;
    visiting.add(hash);
    const previous = String(event.previous_event_hash || '');
    const previousEvent = previous ? byHash.get(previous) : undefined;
    const depth = previousEvent ? depthFor(previousEvent, visiting) + 1 : 0;
    depthCache.set(hash, depth);
    visiting.delete(hash);
    return depth;
  };
  return [...events].sort((left, right) => {
    const depthDelta = depthFor(left) - depthFor(right);
    if (depthDelta !== 0) return depthDelta;
    const leftIdentity = eventIdentity(left);
    const rightIdentity = eventIdentity(right);
    return (originalIndex.get(leftIdentity) ?? 0) - (originalIndex.get(rightIdentity) ?? 0);
  });
}

function eventOrderIndex(events: AuditEvent[]): Map<string, number> {
  return new Map(orderEventsByHashChain(events).map((event, index) => [String(event.event_hash || ''), index]));
}

function decisionIsNewerOrSame(
  existing: TaskDecision | undefined,
  incoming: TaskDecision,
  order: Map<string, number>,
): boolean {
  const incomingHash = String(incoming.event_hash || '');
  const existingHash = String(existing?.event_hash || '');
  if (!existingHash) return true;
  if (!incomingHash) return true;
  const incomingIndex = order.get(incomingHash);
  const existingIndex = order.get(existingHash);
  if (incomingIndex === undefined || existingIndex === undefined) return true;
  return incomingIndex >= existingIndex;
}
