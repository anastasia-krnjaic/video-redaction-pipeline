import { useEffect, useMemo, useState } from 'react';
import { loadSessionState, loadWorkspace } from '../api/client';
import type { AuditEvent, Session, TaskDecision, WorkspaceSummary, Attribution, WorkingState } from '../types/api';
import { normalizeDecisionsByCase, normalizeTaskResolutionsByCase } from '../state/reviewState';
import type { DecisionsByCase, TaskResolutionsByCase } from '../types/domain';
import { readStoredString, removeStoredValue, writeStoredString } from './usePersistentState';

function readUrlSessionId(): string | null {
  const params = new URLSearchParams(window.location.search);
  const raw = params.get('session_id') || params.get('sessionId');
  const sessionId = raw?.trim();
  if (!sessionId) return null;
  return /^session_[A-Za-z0-9]+$/.test(sessionId) ? sessionId : null;
}

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
  setSession: (session: Session | null) => void;
  setAttribution: (attribution: Attribution) => void;
  setDecisions: (decisions: DecisionsByCase) => void;
  setTaskResolutions: (resolutions: TaskResolutionsByCase) => void;
  setEvents: (events: AuditEvent[]) => void;
  setCorrections: (corrections: Record<string, unknown>[]) => void;
  setWorkingState: (state?: WorkingState) => void;
  refreshSession: (sessionId?: string) => Promise<void>;
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

  const refreshSession = useMemo(
    () => async (sessionId?: string) => {
      const id = sessionId || session?.session_id || readStoredString('session_id');
      if (!id) return;
      const payload = await loadSessionState(id);
      setSessionState(payload.session);
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
      writeStoredString('session_id', payload.session.session_id);
    },
    [session?.session_id],
  );

  const setSession = (nextSession: Session | null) => {
    setSessionState(nextSession);
    if (nextSession) writeStoredString('session_id', nextSession.session_id);
    else removeStoredValue('session_id');
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
        const storedSessionId = readUrlSessionId() || readStoredString('session_id');
        if (storedSessionId) {
          try {
            const payload = await loadSessionState(storedSessionId);
            if (cancelled) return;
            setSessionState(payload.session);
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
            writeStoredString('session_id', payload.session.session_id);
          } catch {
            removeStoredValue('session_id');
          }
        }
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
