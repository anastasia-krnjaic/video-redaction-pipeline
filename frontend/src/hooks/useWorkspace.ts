import { useEffect, useMemo, useState } from 'react';
import { loadCurrentSessionState, loadSessionState, loadWorkspace } from '../api/client';
import type { AuditEvent, Session, TaskDecision, WorkspaceSummary, Attribution, WorkingState } from '../types/api';
import { normalizeDecisionsByCase, normalizeTaskResolutionsByCase } from '../state/reviewState';
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
  setWorkingState: (state?: WorkingState) => void;
  refreshSession: (sessionId?: string, loadKind?: 'new' | 'resumed') => Promise<void>;
  setError: (error?: string) => void;
}

export function useWorkspace(): WorkspaceLoadState {
  const [workspace, setWorkspace] = useState<WorkspaceSummary>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [session, setSessionState] = useState<Session | null>(null);
  const [attribution, setAttribution] = useState<Attribution>(null);
  const [decisions, setDecisions] = useState<DecisionsByCase>({});
  const [taskResolutions, setTaskResolutions] = useState<TaskResolutionsByCase>({});
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [corrections, setCorrections] = useState<Record<string, unknown>[]>([]);
  const [workingState, setWorkingState] = useState<WorkingState>();
  const [resumableSessionId, setResumableSessionId] = useState<string>();
  const [workspaceStorageKey, setWorkspaceStorageKey] = useState<string>();
  const [sessionLoadKind, setSessionLoadKind] = useState<'new' | 'resumed'>();

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
    setSessionLoadKind(undefined);
    setAttribution(null);
    setDecisions({});
    setTaskResolutions({});
    setEvents([]);
    setCorrections([]);
    setWorkingState(undefined);
  };

  const applySessionPayload = (
    payload: Awaited<ReturnType<typeof loadSessionState>>,
    loadKind: 'new' | 'resumed',
    loadedWorkspace?: WorkspaceSummary,
  ) => {
    setSessionState(payload.session);
    setSessionLoadKind(loadKind);
    setAttribution(payload.attribution);
    setDecisions(normalizeDecisionsByCase(payload.decisions_by_case, payload.decisions || {}));
    setTaskResolutions(
      normalizeTaskResolutionsByCase(
        payload.working_state?.task_resolutions_by_case as
          | Record<string, Record<string, TaskDecision>>
          | undefined,
        payload.working_state?.task_resolutions as Record<string, TaskDecision> | undefined,
      ),
    );
    setEvents(payload.events || []);
    setCorrections(payload.corrections || []);
    setWorkingState(payload.working_state);
    storeLastSessionId(loadedWorkspace, payload.session.session_id);
    replaceUrlSessionId(payload.session.session_id);
  };

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
    setDecisions,
    setTaskResolutions,
    setEvents,
    setCorrections,
    setWorkingState,
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
