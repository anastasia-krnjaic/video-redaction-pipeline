import { useEffect, useMemo, useState } from 'react';
import { setReviewerAttribution, startSession as startSessionRequest } from './api/client';
import { AppShell } from './components/AppShell';
import { AttributionDialog } from './components/AttributionDialog';
import { useRouteState } from './hooks/useRouteState';
import { useWorkspace } from './hooks/useWorkspace';
import { policyVerificationTask } from './state/reviewState';
import type { AuditEvent, Task } from './types/api';
import type { EffectiveEntity, TopLevelRoute } from './types/domain';
import { caseTabLabel } from './utils/cases';
import { AuditTrailPage } from './pages/AuditTrailPage';
import { CaseReviewPage } from './pages/CaseReviewPage';
import { FinalResultsPage } from './pages/FinalResultsPage';

export function App() {
  const [route, setRoute] = useRouteState();
  const workspaceState = useWorkspace();
  const [attributionOpen, setAttributionOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [manualTasks, setManualTasks] = useState<Task[]>([]);
  const [entityOverrides, setEntityOverrides] = useState<Record<string, EffectiveEntity>>({});
  const caseTabs = useMemo(() => caseTabsForWorkspace(workspaceState.workspace), [workspaceState.workspace]);

  useEffect(() => {
    if (!workspaceState.workspace || route === 'final' || route === 'audit') return;
    if (workspaceState.workspace.cases[route]) return;
    const firstCase = caseTabs[0]?.caseId;
    if (firstCase) setRoute(firstCase);
  }, [caseTabs, route, setRoute, workspaceState.workspace]);

  useEffect(() => {
    if (workspaceState.workingState) {
      setEntityOverrides(
        (workspaceState.workingState.object_policy_overrides || {}) as Record<string, EffectiveEntity>,
      );
      setManualTasks(
        (Array.isArray(workspaceState.workingState.active_manual_tasks)
          ? workspaceState.workingState.active_manual_tasks
          : []) as Task[],
      );
      return;
    }
    const derived = rebuildFromEvents(workspaceState.events);
    setManualTasks(derived.manualTasks);
    setEntityOverrides(derived.entityOverrides);
  }, [workspaceState.events, workspaceState.workingState]);

  const page = useMemo(() => {
    if (!workspaceState.workspace) return null;
    if (route === 'final') {
      return (
        <FinalResultsPage
          attribution={workspaceState.attribution}
          events={workspaceState.events}
          session={workspaceState.session}
          workingState={workspaceState.workingState}
          workspace={workspaceState.workspace}
          onError={workspaceState.setError}
          onEventsChange={workspaceState.setEvents}
          onRequireAttribution={() => setAttributionOpen(true)}
        />
      );
    }
    if (route === 'audit') {
      return <AuditTrailPage events={workspaceState.events} workspace={workspaceState.workspace} />;
    }
    if (!workspaceState.workspace.cases[route]) return null;
    return (
      <CaseReviewPage
        key={route}
        attribution={workspaceState.attribution}
        caseId={route}
        corrections={workspaceState.corrections}
        decisions={workspaceState.decisions}
        entityOverrides={entityOverrides}
        events={workspaceState.events}
        manualTasks={manualTasks}
        session={workspaceState.session}
        taskResolutions={workspaceState.taskResolutions}
        workingState={workspaceState.workingState}
        workspace={workspaceState.workspace}
        onCorrectionsChange={workspaceState.setCorrections}
        onDecisionsChange={workspaceState.setDecisions}
        onEntityOverridesChange={setEntityOverrides}
        onError={workspaceState.setError}
        onEventsChange={workspaceState.setEvents}
        onManualTasksChange={setManualTasks}
        onTaskResolutionsChange={workspaceState.setTaskResolutions}
        onWorkingStateChange={workspaceState.setWorkingState}
      />
    );
  }, [entityOverrides, manualTasks, route, workspaceState]);

  const startSession = async () => {
    if (!workspaceState.workspace) return;
    setBusy(true);
    workspaceState.setError(undefined);
    try {
      const session = await startSessionRequest(workspaceState.workspace.workspace.source_revision);
      workspaceState.setSession(session);
      await workspaceState.refreshSession(session.session_id, 'new');
    } catch (err) {
      workspaceState.setError(`Could not start local review session: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const resumeSession = async () => {
    setBusy(true);
    workspaceState.setError(undefined);
    try {
      await workspaceState.resumeStoredSession();
    } catch (err) {
      workspaceState.setError(`Could not resume review session: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const exportSession = () => {
    if (!workspaceState.session) {
      workspaceState.setError('Start review before exporting.');
      return;
    }
    if (!workspaceState.attribution?.display_name) {
      setAttributionOpen(true);
      return;
    }
    window.open(`/sessions/${workspaceState.session.session_id}/review_summary.json`, '_blank', 'noopener');
  };

  const saveAttribution = async (displayName: string) => {
    if (!workspaceState.session) {
      workspaceState.setError('Start review before setting attribution.');
      return;
    }
    const { attribution, event } = await setReviewerAttribution({
      session_id: workspaceState.session.session_id,
      display_name: displayName,
      actor: workspaceState.attribution || { display_name: null },
    });
    workspaceState.setAttribution(attribution);
    workspaceState.setEvents([...workspaceState.events, event]);
    setAttributionOpen(false);
  };

  if (workspaceState.loading) {
    return <main className="loading-screen">Loading final review workspace...</main>;
  }

  if (!workspaceState.workspace) {
    return (
      <main className="fatal-error">
        <h1>Workspace unavailable</h1>
        <p>{workspaceState.error || 'The versioned workspace API did not return a workspace summary.'}</p>
      </main>
    );
  }

  return (
    <>
      <AppShell
        attribution={workspaceState.attribution}
        busy={busy}
        cases={caseTabs}
        error={workspaceState.error}
        route={route}
        resumableSessionId={workspaceState.resumableSessionId}
        session={workspaceState.session}
        sessionLoadKind={workspaceState.sessionLoadKind}
        onExport={exportSession}
        onOpenAttribution={() => setAttributionOpen(true)}
        onResumeSession={resumeSession}
        onRouteChange={(nextRoute: TopLevelRoute) => {
          workspaceState.setError(undefined);
          setRoute(nextRoute);
          if (nextRoute === 'audit' && workspaceState.session) {
            void workspaceState.refreshSession(workspaceState.session.session_id);
          }
        }}
        onStartSession={startSession}
      >
        {page}
      </AppShell>
      <AttributionDialog open={attributionOpen} onCancel={() => setAttributionOpen(false)} onSubmit={saveAttribution} />
    </>
  );
}

function rebuildFromEvents(events: AuditEvent[]): {
  entityOverrides: Record<string, EffectiveEntity>;
  manualTasks: Task[];
} {
  const entityOverrides: Record<string, EffectiveEntity> = {};
  const manualTasks: Task[] = [];
  for (const event of events) {
    const after = event.after_state as Record<string, unknown> | undefined;
    if (!after) continue;
    if (event.event_type === 'manual_issue_created' && isTask(after.task)) {
      manualTasks.push(after.task);
    }
    if (event.event_type === 'entity_policy_changed' && typeof after.entity_id === 'string') {
      const entity = after as unknown as EffectiveEntity;
      entityOverrides[entity.entity_id] = entity;
      const taskId = typeof after.verification_task_id === 'string' ? after.verification_task_id : null;
      if (taskId && typeof event.case_id === 'string') {
        manualTasks.push(policyVerificationTask(event.case_id, entity, taskId));
      }
    }
    if (event.event_type === 'entities_merged' && isRecord(after.primary)) {
      const primary = after.primary as unknown as EffectiveEntity;
      entityOverrides[primary.entity_id] = primary;
      const retired = Array.isArray(after.retired_entity_ids) ? after.retired_entity_ids : [];
      for (const id of retired) {
        if (typeof id === 'string') entityOverrides[id] = { entity_id: id, current_status: 'merged' } as EffectiveEntity;
      }
    }
  }
  return { entityOverrides, manualTasks };
}

export function caseTabsForWorkspace(
  workspace: ReturnType<typeof useWorkspace>['workspace'],
): Array<{ caseId: string; label: string }> {
  if (!workspace) return [];
  return Object.values(workspace.cases).map((item) => ({
    caseId: item.case_id,
    label: caseTabLabel(item.case_id, item.display_name),
  }));
}

function isTask(value: unknown): value is Task {
  return isRecord(value) && typeof value.review_task_id === 'string' && typeof value.case_id === 'string';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
