import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadCurrentSessionState, loadSessionState, loadWorkspace } from '../api/client';
import type { AuditEvent, Session, TaskDecision, WorkspaceSummary, Attribution, WorkingState } from '../types/api';
import { normalizeDecisionsByCase, normalizeTaskResolutionsByCase } from '../state/reviewState';
import {
  eventListIsOlderThanKnownEvents,
  mergeAuditEvents,
  mergeDecisionsByEventOrder,
  workingStateIsOlderThanKnownEvents,
} from '../state/reviewSync';
import type { DecisionsByCase, TaskResolutionsByCase } from '../types/domain';
import { explicitSessionRequestFromSearch } from '../utils/sessionRouting';
import { removeStoredValue, writeStoredString } from './usePersistentState';

export interface WorkspaceLoadState {
  workspace?: WorkspaceSummary;
  loading: boolean;
  error?: string;
  session: Session | null;
  attribution: Attribution;
  decisions: DecisionsByCase;
  taskResolutions: TaskResolutionsByCase;
  events: AuditEvent[];
  corrections: Record<string, unknown>[];
  workingState?: WorkingState;
  resumableSessionId?: string;
  sessionLoadKind?: 'new' | 'resumed';
  resumeStoredSession: () => Promise<void>;
  setSession: (session: Session | null) => void;
  setAttribution: (attribution: Attribution) => void;
  setDecisions: (decisions: DecisionsByCase) => void;
  setTaskResolutions: (resolutions: TaskResolutionsByCase) => void;
  setEvents: (events: AuditEvent[]) => void;
  setCorrections: (corrections: Record<string, unknown>[]) => void;
  setWorkingState: (state?: WorkingState, responseEvents?: AuditEvent[]) => boolean;
  refreshSession: (sessionId?: string, loadKind?: 'new' | 'resumed') => Promise<void>;
  setError: (error?: string) => void;
}

export function useWorkspace(): WorkspaceLoadState {
  const [workspace, setWorkspace] = useState<WorkspaceSummary>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [session, setSessionState] = useState<Session | null>(null);
  const [attribution, setAttribution] = useState<Attribution>(null);
  const [decisions, setDecisionsState] = useState<DecisionsByCase>({});
  const [taskResolutions, setTaskResolutionsState] = useState<TaskResolutionsByCase>({});
  const [events, setEventsState] = useState<AuditEvent[]>([]);
  const [corrections, setCorrectionsState] = useState<Record<string, unknown>[]>([]);
  const [workingState, setWorkingStateState] = useState<WorkingState>();
  const [resumableSessionId, setResumableSessionId] = useState<string>();
  const [workspaceStorageKey, setWorkspaceStorageKey] = useState<string>();
  const [sessionLoadKind, setSessionLoadKind] = useState<'new' | 'resumed'>();
  const decisionsRef = useRef(decisions);
  const eventsRef = useRef(events);
  const sessionRef = useRef(session);
  const taskResolutionsRef = useRef(taskResolutions);
  const correctionsRef = useRef(corrections);
  const workingStateRef = useRef(workingState);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    decisionsRef.current = decisions;
  }, [decisions]);

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  useEffect(() => {
    taskResolutionsRef.current = taskResolutions;
  }, [taskResolutions]);

  useEffect(() => {
    correctionsRef.current = corrections;
  }, [corrections]);

  useEffect(() => {
    workingStateRef.current = workingState;
  }, [workingState]);

  const storeLastSessionId = (loadedWorkspace: WorkspaceSummary | undefined, sessionId: string) => {
    const key = loadedWorkspace ? lastSessionStorageKey(loadedWorkspace) : workspaceStorageKey;
    if (!key) return;
    writeStoredString(key, sessionId);
    setResumableSessionId(sessionId);
  };

  const clearLastSessionId = (loadedWorkspace?: WorkspaceSummary) => {
    const key = loadedWorkspace ? lastSessionStorageKey(loadedWorkspace) : workspaceStorageKey;
    if (!key) return;
    removeStoredValue(key);
    setResumableSessionId(undefined);
  };

  const clearLoadedSession = () => {
    setSessionState(null);
    sessionRef.current = null;
    setSessionLoadKind(undefined);
    setAttribution(null);
    decisionsRef.current = {};
    taskResolutionsRef.current = {};
    eventsRef.current = [];
    correctionsRef.current = [];
    workingStateRef.current = undefined;
    setDecisionsState({});
    setTaskResolutionsState({});
    setEventsState([]);
    setCorrectionsState([]);
    setWorkingStateState(undefined);
  };

  const applySessionPayload = (
    payload: Awaited<ReturnType<typeof loadSessionState>>,
    loadKind: 'new' | 'resumed',
    loadedWorkspace?: WorkspaceSummary,
  ) => {
    const incomingEvents = payload.events || [];
    const currentSessionId = sessionRef.current?.session_id;
    if (
      currentSessionId === payload.session.session_id &&
      eventListIsOlderThanKnownEvents(incomingEvents, eventsRef.current)
    ) {
      return;
    }
    setSessionState(payload.session);
    sessionRef.current = payload.session;
    setSessionLoadKind(loadKind);
    setAttribution(payload.attribution);
    const nextDecisions = normalizeDecisionsByCase(payload.decisions_by_case, payload.decisions || {});
    const nextTaskResolutions = normalizeTaskResolutionsByCase(
      payload.working_state?.task_resolutions_by_case as
        | Record<string, Record<string, TaskDecision>>
        | undefined,
      payload.working_state?.task_resolutions as Record<string, TaskDecision> | undefined,
    );
    decisionsRef.current = nextDecisions;
    taskResolutionsRef.current = nextTaskResolutions;
    eventsRef.current = incomingEvents;
    correctionsRef.current = payload.corrections || [];
    workingStateRef.current = payload.working_state;
    setDecisionsState(nextDecisions);
    setTaskResolutionsState(nextTaskResolutions);
    setEventsState(incomingEvents);
    setCorrectionsState(payload.corrections || []);
    setWorkingStateState(payload.working_state);
    storeLastSessionId(loadedWorkspace, payload.session.session_id);
    replaceUrlSessionId(payload.session.session_id);
  };

  const mergeEvents = useCallback((incomingEvents: AuditEvent[]) => {
    const merged = mergeAuditEvents(eventsRef.current, incomingEvents);
    eventsRef.current = merged;
    setEventsState(merged);
  }, []);

  const mergeDecisions = useCallback((incomingDecisions: DecisionsByCase) => {
    const merged = mergeDecisionsByEventOrder(decisionsRef.current, incomingDecisions, eventsRef.current);
    decisionsRef.current = merged;
    setDecisionsState(merged);
  }, []);

  const mergeTaskResolutions = useCallback((incomingResolutions: TaskResolutionsByCase) => {
    const merged = mergeDecisionsByEventOrder(taskResolutionsRef.current, incomingResolutions, eventsRef.current);
    taskResolutionsRef.current = merged;
    setTaskResolutionsState(merged);
  }, []);

  const mergeCorrections = useCallback((incomingCorrections: Record<string, unknown>[]) => {
    const merged = mergeCorrectionsByIdentity(correctionsRef.current, incomingCorrections);
    correctionsRef.current = merged;
    setCorrectionsState(merged);
  }, []);

  const applyWorkingState = useCallback((state?: WorkingState, responseEvents: AuditEvent[] = []) => {
    if (!state) {
      workingStateRef.current = undefined;
      setWorkingStateState(undefined);
      return true;
    }
    if (workingStateIsOlderThanKnownEvents(state, eventsRef.current, responseEvents)) {
      return false;
    }
    const nextDecisions = normalizeDecisionsByCase(
      state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
      state.task_decisions as Record<string, TaskDecision> | undefined,
    );
    const nextTaskResolutions = normalizeTaskResolutionsByCase(
      state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
      state.task_resolutions as Record<string, TaskDecision> | undefined,
    );
    workingStateRef.current = state;
    decisionsRef.current = nextDecisions;
    taskResolutionsRef.current = nextTaskResolutions;
    correctionsRef.current = (Array.isArray(state.manual_corrections) ? state.manual_corrections : []) as Record<string, unknown>[];
    setWorkingStateState(state);
    setDecisionsState(nextDecisions);
    setTaskResolutionsState(nextTaskResolutions);
    setCorrectionsState(correctionsRef.current);
    return true;
  }, []);

  const refreshSession = useMemo(
    () => async (sessionId?: string, loadKind?: 'new' | 'resumed') => {
      const id = sessionId || session?.session_id;
      if (!id) return;
      const payload = await loadSessionState(id);
      applySessionPayload(payload, loadKind || sessionLoadKind || 'resumed', workspace);
    },
    [session?.session_id, sessionLoadKind, workspace],
  );

  const setSession = (nextSession: Session | null) => {
    setSessionState(nextSession);
    sessionRef.current = nextSession;
    setSessionLoadKind(nextSession ? 'new' : undefined);
    if (nextSession) {
      storeLastSessionId(workspace, nextSession.session_id);
      replaceUrlSessionId(nextSession.session_id);
    } else {
      clearLastSessionId(workspace);
      replaceUrlSessionId(null);
    }
  };

  const resumeStoredSession = async () => {
    const id = resumableSessionId;
    if (!id) return;
    const payload = await loadSessionState(id);
    applySessionPayload(payload, 'resumed', workspace);
  };

  useEffect(() => {
    let cancelled = false;
    async function run() {
      setLoading(true);
      setError(undefined);
      try {
        const loaded = await loadWorkspace();
        if (cancelled) return;
        setWorkspace(loaded);
        const storageKey = lastSessionStorageKey(loaded);
        setWorkspaceStorageKey(storageKey);
        const explicitSession = explicitSessionRequestFromSearch(window.location.search);
        if (explicitSession.requested) {
          if (!explicitSession.sessionId) {
            clearLoadedSession();
            setResumableSessionId(undefined);
            setError(explicitSession.error || 'Invalid requested review session.');
            return;
          }
          try {
            const payload = await loadSessionState(explicitSession.sessionId);
            if (cancelled) return;
            applySessionPayload(payload, 'resumed', loaded);
            return;
          } catch (err) {
            if (cancelled) return;
            clearLoadedSession();
            setResumableSessionId(undefined);
            const detail = err instanceof Error ? err.message : String(err);
            setError(`Could not load requested review session ${explicitSession.sessionId}: ${detail}`);
            return;
          }
        }
        const currentSessionId = String((loaded.current_session as Record<string, unknown> | undefined)?.session_id || '');
        setResumableSessionId(currentSessionId || undefined);
        const payload = await loadCurrentSessionState();
        if (cancelled) return;
        applySessionPayload(payload, payload.events.length > 1 ? 'resumed' : 'new', loaded);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void run();
    return () => {
      cancelled = true;
    };
  }, []);

  return {
    workspace,
    loading,
    error,
    session,
    attribution,
    decisions,
    taskResolutions,
    events,
    corrections,
    workingState,
    resumableSessionId,
    sessionLoadKind,
    resumeStoredSession,
    setSession,
    setAttribution,
    setDecisions: mergeDecisions,
    setTaskResolutions: mergeTaskResolutions,
    setEvents: mergeEvents,
    setCorrections: mergeCorrections,
    setWorkingState: applyWorkingState,
    refreshSession,
    setError,
  };
}

function lastSessionStorageKey(workspace: WorkspaceSummary): string {
  return `last_session_id:${workspaceIdentity(workspace)}`;
}

function workspaceIdentity(workspace: WorkspaceSummary): string {
  const identity = workspace.workspace?.workspace_manifest_sha256
    || workspace.workspace?.workspace_identity_hash
    || workspace.workspace?.source_sha256
    || workspace.backend_workspace_schema_version
    || 'unknown-workspace';
  return String(identity);
}

function replaceUrlSessionId(sessionId: string | null): void {
  const url = new URL(window.location.href);
  if (sessionId) url.searchParams.set('session_id', sessionId);
  else url.searchParams.delete('session_id');
  url.searchParams.delete('sessionId');
  window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
}

function mergeCorrectionsByIdentity(
  current: Record<string, unknown>[],
  incoming: Record<string, unknown>[],
): Record<string, unknown>[] {
  if (!incoming.length) return current;
  const byIdentity = new Map<string, Record<string, unknown>>();
  const order: string[] = [];
  const add = (correction: Record<string, unknown>, fallbackIndex: number) => {
    const identity = String(
      correction.correction_id || correction.revision_id || correction.correction_hash || `index:${fallbackIndex}`,
    );
    if (!byIdentity.has(identity)) order.push(identity);
    byIdentity.set(identity, correction);
  };
  current.forEach(add);
  incoming.forEach((correction, index) => add(correction, current.length + index));
  return order.map((identity) => byIdentity.get(identity)).filter((item): item is Record<string, unknown> => Boolean(item));
}
