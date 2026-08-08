import { useEffect, useMemo, useState } from 'react';
import { loadSessionState, loadWorkspace } from '../api/client';
import type { AuditEvent, Session, TaskDecision, WorkspaceSummary, Attribution, WorkingState } from '../types/api';
import { readStoredString, removeStoredValue, writeStoredString } from './usePersistentState';

export interface WorkspaceLoadState {
  workspace?: WorkspaceSummary;
  loading: boolean;
  error?: string;
  session: Session | null;
  attribution: Attribution;
  decisions: Record<string, TaskDecision>;
  events: AuditEvent[];
  corrections: Record<string, unknown>[];
  workingState?: WorkingState;
  setSession: (session: Session | null) => void;
  setAttribution: (attribution: Attribution) => void;
  setDecisions: (decisions: Record<string, TaskDecision>) => void;
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
  const [decisions, setDecisions] = useState<Record<string, TaskDecision>>({});
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
      setDecisions(payload.decisions || {});
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
        const storedSessionId = readStoredString('session_id');
        if (storedSessionId) {
          try {
            const payload = await loadSessionState(storedSessionId);
            if (cancelled) return;
            setSessionState(payload.session);
            setAttribution(payload.attribution);
            setDecisions(payload.decisions || {});
            setEvents(payload.events || []);
            setCorrections(payload.corrections || []);
            setWorkingState(payload.working_state);
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
    events,
    corrections,
    workingState,
    setSession,
    setAttribution,
    setDecisions,
    setEvents,
    setCorrections,
    setWorkingState,
    refreshSession,
    setError,
  };
}
