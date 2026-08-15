import { useEffect, useMemo, useRef, useState } from 'react';
import {
  acknowledgeFullVideoSweep,
  createCheckpoint,
  deleteCheckpoint,
  loadFinalizationStatus,
  loadMaterializationJob,
  loadWorkingState,
  mergeEntities,
  previewCorrection,
  resetToAutomaticBaseline,
  requestLatestReviewedBuild,
  retryMaterializationJob,
  restoreCheckpoint,
  saveCorrection as saveCorrectionRequest,
  saveDecision,
  saveDecisionGroup,
  saveBulkEntityPolicy,
  saveEntityPolicy,
} from '../api/client';
import {
  CorrectionEditor,
  CorrectionVideoOverlay,
  PendingCorrectionOverlay,
  type PendingCorrectionRegion,
} from '../components/CorrectionEditor';
import { EntityFocusOverlay, type EntityFocusTarget } from '../components/EntityFocusOverlay';
import { EntityMergeDialog } from '../components/EntityMergeDialog';
import { EntityPanel } from '../components/EntityPanel';
import { ReviewQueue } from '../components/ReviewQueue';
import { ReviewVideoPlayer } from '../components/ReviewVideoPlayer';
import { usePersistentState } from '../hooks/usePersistentState';
import { useVideoController } from '../hooks/useVideoController';
import {
  ACTIONS,
  authoritativeWorkingStateAfterSweepApproval,
  bucketCounts,
  optionalReviewGroups,
  requiredReviewGroups,
  tasksForCase,
  normalizeDecisionsByCase,
  normalizeTaskResolutionsByCase,
  withTaskDecision,
  withoutTaskResolution,
} from '../state/reviewState';
import { materializationJobMatchesCurrentProjection } from '../state/materialization';
import type {
  Attribution,
  AuditEvent,
  Correction,
  Entity,
  FinalizationStatus,
  MaterializationJob,
  MediaVersion,
  Session,
  Task,
  TaskDecision,
  WorkingState,
  WorkspaceSummary,
} from '../types/api';
import type {
  CaseId,
  CorrectionDraft,
  DecisionsByCase,
  EffectiveEntity,
  EntityFilter,
  EntityPolicy,
  ManualRedactionCategory,
  QueueBucket,
  ReviewGroup,
  ReviewMode,
  TaskResolutionsByCase,
  TaskWithDecision,
} from '../types/domain';
import { MANUAL_REDACTION_CATEGORIES } from '../types/domain';
import { caseToSourceFrame } from '../utils/frame';

interface Props {
  attribution: Attribution;
  caseId: CaseId;
  corrections: Record<string, unknown>[];
  decisions: DecisionsByCase;
  entityOverrides: Record<string, EffectiveEntity>;
  events: AuditEvent[];
  manualTasks: Task[];
  session: Session | null;
  taskResolutions: TaskResolutionsByCase;
  workingState?: WorkingState;
  workspace: WorkspaceSummary;
  onCorrectionsChange: (corrections: Record<string, unknown>[]) => void;
  onDecisionsChange: (decisions: DecisionsByCase) => void;
  onEntityOverridesChange: (overrides: Record<string, EffectiveEntity>) => void;
  onError: (message?: string) => void;
  onEventsChange: (events: AuditEvent[]) => void;
  onManualTasksChange: (tasks: Task[]) => void;
  onTaskResolutionsChange: (resolutions: TaskResolutionsByCase) => void;
  onWorkingStateChange: (state?: WorkingState) => void;
}

interface TrackedMaterializationJob {
  entityIds: string[];
  job: MaterializationJob;
  preserveFrame: number;
}

export function CaseReviewPage({
  attribution,
  caseId,
  corrections,
  decisions,
  entityOverrides,
  events,
  manualTasks,
  onCorrectionsChange,
  onDecisionsChange,
  onEntityOverridesChange,
  onError,
  onEventsChange,
  onManualTasksChange,
  onTaskResolutionsChange,
  onWorkingStateChange,
  session,
  taskResolutions,
  workingState,
  workspace,
}: Props) {
  const manifest = workspace.cases[caseId];
  const baseTasks = workspace.tasks.tasks || [];
  const uiScope = `${workspaceIdentityForUi(workspace)}:${session?.session_id || 'no-session'}:${caseId}`;
  const [mode, setMode] = usePersistentState<ReviewMode>(`${uiScope}:mode`, 'review');
  const [caseFrame, setCaseFrame] = usePersistentState<number>(`${uiScope}:frame`, 0);
  const [selectedRole, setSelectedRole] = usePersistentState<string>(`${uiScope}:version`, defaultRole(manifest.video_roles));
  const [selectedTaskId, setSelectedTaskId] = usePersistentState<string | null>(`${uiScope}:task`, null);
  const [bucket, setBucket] = usePersistentState<QueueBucket>(`${uiScope}:bucket`, 'Open');
  const [entityFilter, setEntityFilter] = usePersistentState<EntityFilter>(`${uiScope}:entity_filter`, 'needs_review');
  const [entitiesOpen, setEntitiesOpen] = useState(false);
  const [mergeSelection, setMergeSelection] = useState<EffectiveEntity[]>([]);
  const [pendingTaskId, setPendingTaskId] = useState<string>();
  const [pendingEntityIds, setPendingEntityIds] = useState<Set<string>>(() => new Set());
  const [latestRole, setLatestRole] = useState<MediaVersion | undefined>();
  const [previewRole, setPreviewRole] = useState<MediaVersion | undefined>();
  const [previewInvalidated, setPreviewInvalidated] = useState(false);
  const [previewReady, setPreviewReady] = useState(false);
  const [previewToken, setPreviewToken] = useState<string>();
  const [correctionSaveStatus, setCorrectionSaveStatus] = useState<string>();
  const [sweepBusy, setSweepBusy] = useState(false);
  const [finalizationStatus, setFinalizationStatus] = useState<FinalizationStatus>();
  const [checkpointName, setCheckpointName] = useState('');
  const [checkpointBusy, setCheckpointBusy] = useState<string | null>(null);
  const [checkpointDeleteTarget, setCheckpointDeleteTarget] = useState<Record<string, unknown> | null>(null);
  const [resetEditsConfirmOpen, setResetEditsConfirmOpen] = useState(false);
  const [materializationJobs, setMaterializationJobs] = useState<Record<string, TrackedMaterializationJob>>({});
  const [materializationError, setMaterializationError] = useState<string>();
  const [focusTarget, setFocusTarget] = useState<EntityFocusTarget | null>(null);
  const eventsRef = useRef(events);
  const materializationJobsRef = useRef(materializationJobs);
  const onWorkingStateChangeRef = useRef(onWorkingStateChange);
  const setSelectedRoleRef = useRef(setSelectedRole);
  const seekToFrameRef = useRef<(frame: number) => void>(() => {});
  const workingStateRef = useRef<WorkingState | undefined>(workingState);
  const editDraftContextRef = useRef<string | null>(null);
  const autoBuildRequestRef = useRef<string>('');

  const allTasks = useMemo(
    () => [...baseTasks, ...manualTasks].filter((task) => task.case_id === caseId),
    [baseTasks, caseId, manualTasks],
  );
  const taskRows = useMemo(
    () => tasksForCase(allTasks, decisions, caseId, taskResolutions),
    [allTasks, caseId, decisions, taskResolutions],
  );
  const tasks = useMemo(() => requiredReviewGroups(taskRows), [taskRows]);
  const optionalTasks = useMemo(() => optionalReviewGroups(taskRows), [taskRows]);
  const counts = useMemo(() => bucketCounts(tasks), [tasks]);
  const selectedGroup = tasks.find((task) => task.review_task_id === selectedTaskId) || null;
  const selectedTask =
    selectedGroup?.child_tasks[0] ||
    taskRows.find((task) => task.review_task_id === selectedTaskId) ||
    taskRows[0] ||
    null;
  const materializedLatestRole = workingState?.materialized_latest_reviewed?.[caseId] as MediaVersion | undefined;
  const caseEntityIds = useMemo(() => {
    const ids = new Set((workspace.entities.cases[caseId] || []).map((entity) => entity.entity_id));
    for (const [entityId, entity] of Object.entries(entityOverrides)) {
      if (entity.case_id === caseId) ids.add(entityId);
    }
    return ids;
  }, [caseId, entityOverrides, workspace.entities.cases]);
  const relatedTaskCountByEntity = useMemo(() => {
    const countsByEntity: Record<string, number> = {};
    for (const task of taskRows) {
      const targets = (Array.isArray(task.target_entity_ids) ? task.target_entity_ids : []).map(String);
      if (targets.length !== 1) continue;
      countsByEntity[targets[0]] = (countsByEntity[targets[0]] || 0) + 1;
    }
    return countsByEntity;
  }, [taskRows]);

  const roles = useMemo(() => {
    const sessionLatestRole = previewRole || latestRole || materializedLatestRole;
    return publicVideoRoles(manifest.video_roles, sessionLatestRole);
  }, [latestRole, manifest.video_roles, materializedLatestRole, previewRole]);
  const selectedVersion = roles.find((role) => role.role === selectedRole) || roles.find((role) => role.role === defaultRole(roles));
  const controller = useVideoController(manifest, selectedVersion, caseFrame, setCaseFrame);
  const [draft, setDraft] = useState<CorrectionDraft>(() => initialDraft(caseId, manifest, caseFrame, selectedTask));
  const currentSourceFrame = caseToSourceFrame(manifest, caseFrame);
  const activeMaterializationJobs = Object.values(materializationJobs).filter(({ job }) => job.case_id === caseId && (job.status === 'queued' || job.status === 'running'));
  const savedCorrectionOverlayActive = activeMaterializationJobs.length > 0 || Boolean(correctionSaveStatus?.startsWith('Correction saved'));
  const pendingCorrectionRegions = useMemo(
    () => pendingRegionsForSourceFrame(corrections, manualTasks, caseId, currentSourceFrame, savedCorrectionOverlayActive),
    [caseId, corrections, currentSourceFrame, manualTasks, savedCorrectionOverlayActive],
  );
  const activeMaterializationJobKey = useMemo(
    () => Object.values(materializationJobs)
      .filter(({ job }) => job.status === 'queued' || job.status === 'running')
      .map(({ job }) => job.job_id)
      .sort()
      .join('|'),
    [materializationJobs],
  );
  const materializationStatusText = activeMaterializationJobs[0]
    ? formatMaterializationStatus(activeMaterializationJobs[0].job)
    : undefined;
  const correctionAttentionCount = manualTasks
    .filter((task) => task.case_id === caseId)
    .filter((task) => task.normal_task_class === 'correction_verification')
    .filter((task) => task.actionable !== false && task.decision_state !== 'resolved')
    .length + counts['Needs correction'];
  const reviewReady = correctionAttentionCount === 0;
  const allGuidanceAddressed = reviewReady && counts.Open === 0;
  const latestReviewedStatus = latestReviewedStatusText({
    activeJobs: activeMaterializationJobs,
    latestRole: materializedLatestRole,
    materializationError,
    reviewReady,
    workingState,
  });
  const latestReviewedCurrent = latestReviewedStatus === 'Latest reviewed current';
  const sweepAcknowledgement = (workingState?.sweep_acknowledgements as Record<string, unknown> | undefined)?.[caseId] as Record<string, unknown> | undefined;
  const sweepAcknowledged = Boolean(sweepAcknowledgement);
  const sweepStale = sweepAcknowledged && (
    !latestReviewedCurrent ||
    finalizationBlockerGroups(finalizationStatus).some((group) => group.key === 'full_video_approval')
  );
  const finalNextStep = finalizationStatus?.ready
    ? 'Ready to finalize'
    : latestReviewedCurrent
      ? 'Ready for final full-video review'
      : 'Final full-video review waits for current Latest reviewed';

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  useEffect(() => {
    materializationJobsRef.current = materializationJobs;
  }, [materializationJobs]);

  useEffect(() => {
    onWorkingStateChangeRef.current = onWorkingStateChange;
  }, [onWorkingStateChange]);

  useEffect(() => {
    workingStateRef.current = workingState;
  }, [workingState]);

  useEffect(() => {
    setSelectedRoleRef.current = setSelectedRole;
  }, [setSelectedRole]);

  useEffect(() => {
    seekToFrameRef.current = controller.seekToFrame;
  }, [controller.seekToFrame]);

  useEffect(() => {
    let cancelled = false;
    if (!session?.session_id) {
      setFinalizationStatus(undefined);
      return undefined;
    }
    const load = async () => {
      try {
        const status = await loadFinalizationStatus({
          case_id: caseId,
          session_id: session.session_id,
        });
        if (!cancelled) setFinalizationStatus(status);
      } catch {
        if (!cancelled) setFinalizationStatus(undefined);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [caseId, session?.session_id, workingState?.active_projection_hash, workingState?.current_event_chain_head]);

  useEffect(() => {
    if (mode !== 'review' || !session?.session_id || !reviewReady || activeMaterializationJobKey) return;
    if (latestReviewedStatus !== 'Ready to build Latest reviewed' && latestReviewedStatus !== 'Latest reviewed stale — review changed') return;
    const revision = String(workingState?.active_projection_hash || workingState?.state_hash || '');
    const requestKey = `${caseId}:${revision}:${latestReviewedStatus}`;
    if (!revision || autoBuildRequestRef.current === requestKey) return;
    autoBuildRequestRef.current = requestKey;
    const run = async () => {
      try {
        const result = await requestLatestReviewedBuild({
          session_id: session.session_id,
          actor: attribution || { display_name: null },
          case_id: caseId,
        });
        if (result.working_state) applyWorkingStateProjection(result.working_state);
        trackMaterializationJob(result.materialization_job || result.job);
      } catch (err) {
        setMaterializationError(err instanceof Error ? err.message : String(err));
      }
    };
    void run();
  }, [
    activeMaterializationJobKey,
    attribution,
    caseId,
    latestReviewedStatus,
    mode,
    reviewReady,
    session?.session_id,
    workingState?.active_projection_hash,
    workingState?.state_hash,
  ]);

  useEffect(() => {
    if (!roles.some((role) => role.role === selectedRole)) {
      setSelectedRole(defaultRole(roles));
    }
  }, [roles, selectedRole, setSelectedRole]);

  useEffect(() => {
    if (mode !== 'review') return;
    if (!tasks.length) return;
    if (!selectedTaskId || !tasks.some((task) => task.review_task_id === selectedTaskId)) {
      const firstOpen = tasks.find((task) => task.bucket === 'Open');
      setSelectedTaskId((firstOpen || tasks[0]).review_task_id);
    }
  }, [mode, selectedTaskId, setSelectedTaskId, tasks]);

  useEffect(() => {
    setDraft((current) => withDraftHash({ ...current, selectedTaskId: selectedTask?.review_task_id }));
  }, [selectedTask?.review_task_id]);

  useEffect(() => {
    if (mode !== 'edit') {
      editDraftContextRef.current = null;
      return;
    }
    const contextKey = `${caseId}:${selectedTask?.review_task_id || ''}`;
    const shouldInitialize = editDraftContextRef.current !== contextKey;
    if (shouldInitialize) controller.pause();
    editDraftContextRef.current = contextKey;
    setDraft((current) => {
      if (current.caseId === caseId && !shouldInitialize) return current;
      if (current.caseId === caseId && current.keyframes.length) return current;
      return initialDraft(caseId, manifest, caseFrame, selectedTask);
    });
  }, [caseId, controller.pause, manifest, mode, selectedTask?.review_task_id]);

  useEffect(() => {
    if (!focusTarget) return undefined;
    const timer = window.setTimeout(() => setFocusTarget(null), 3000);
    return () => window.clearTimeout(timer);
  }, [focusTarget]);

  useEffect(() => {
    if (controller.playing) setFocusTarget(null);
  }, [controller.playing]);

  useEffect(() => {
    let cancelled = false;
    if (!activeMaterializationJobKey) return undefined;
    const activeJobIds = activeMaterializationJobKey.split('|').filter(Boolean);
    const poll = async () => {
      const activeJobs = activeJobIds
        .map((jobId) => materializationJobsRef.current[jobId])
        .filter(
          (tracked): tracked is TrackedMaterializationJob =>
            Boolean(tracked && (tracked.job.status === 'queued' || tracked.job.status === 'running')),
        );
      if (!activeJobs.length) return;
      const updates = await Promise.all(
        activeJobs.map(async (tracked) => {
          try {
            const response = await loadMaterializationJob(tracked.job.job_id);
            return { ...tracked, job: response.job };
          } catch (err) {
            return {
              ...tracked,
              job: {
                ...tracked.job,
                error: err instanceof Error ? err.message : String(err),
                status: 'failed' as const,
              },
            };
          }
        }),
      );
      if (cancelled) return;
      setMaterializationJobs((current) => {
        const next = { ...current };
        for (const tracked of updates) next[tracked.job.job_id] = tracked;
        return next;
      });
      for (const tracked of updates) {
        if (tracked.job.status === 'completed') {
          if (tracked.entityIds.length) {
            setPendingEntityIds((current) => {
              const next = new Set(current);
              tracked.entityIds.forEach((id) => next.delete(id));
              return next;
            });
          }
          const jobMatchesCurrentProjection = materializationJobMatchesCurrentProjection(
            tracked.job,
            workingStateRef.current,
            {
              caseId,
              sessionId: session?.session_id,
            },
          );
          if (jobMatchesCurrentProjection && tracked.job.working_state) {
            onWorkingStateChangeRef.current(tracked.job.working_state);
          }
          if (jobMatchesCurrentProjection && tracked.job.latest_reviewed && tracked.job.case_id === caseId) {
            setLatestRole(tracked.job.latest_reviewed);
            setSelectedRoleRef.current('latest_reviewed');
            seekToFrameRef.current(tracked.preserveFrame);
            setCorrectionSaveStatus('Correction saved — Latest reviewed current');
          }
        }
        if (tracked.job.status === 'superseded') {
          if (tracked.entityIds.length) {
            setPendingEntityIds((current) => {
              const next = new Set(current);
              tracked.entityIds.forEach((id) => next.delete(id));
              return next;
            });
          }
        }
        if (tracked.job.status === 'failed') {
          if (tracked.entityIds.length) {
            setPendingEntityIds((current) => {
              const next = new Set(current);
              tracked.entityIds.forEach((id) => next.delete(id));
              return next;
            });
          }
          setMaterializationError(tracked.job.error || 'Latest reviewed media update failed.');
          setCorrectionSaveStatus('Correction saved');
        }
      }
    };
    void poll();
    const interval = window.setInterval(() => void poll(), 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeMaterializationJobKey, caseId, session?.session_id]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTextEntryTarget(document.activeElement)) return;
      if (event.key === 'ArrowRight' || event.key === 'Right') {
        event.preventDefault();
        controller.stepFrame(event.shiftKey ? 10 : 1);
      } else if (event.key === 'ArrowLeft' || event.key === 'Left') {
        event.preventDefault();
        controller.stepFrame(event.shiftKey ? -10 : -1);
      } else if (event.key === ' ') {
        event.preventDefault();
        controller.togglePlay();
      } else if (event.key.toLowerCase() === 'f') {
        controller.toggleFullscreen();
      }
    };
    window.addEventListener('keydown', onKeyDown, { capture: true });
    return () => window.removeEventListener('keydown', onKeyDown, { capture: true });
  }, [controller]);

  const selectTask = (task: ReviewGroup) => {
    setFocusTarget(null);
    setBucket(task.bucket);
    setSelectedTaskId(task.review_task_id);
    controller.seekToSourceFrame(task.source_frame_start);
  };

  const updateDraft = (nextDraft: CorrectionDraft) => {
    setPreviewInvalidated(previewReady || Boolean(previewToken) || Boolean(nextDraft.preview_token));
    setPreviewReady(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setCorrectionSaveStatus(undefined);
    if (selectedRole === 'correction_preview') setSelectedRole('latest_reviewed');
    setDraft(withDraftHash({ ...nextDraft, preview_token: undefined }));
  };

  const resetLocalCorrectionState = (nextTask: TaskWithDecision | null = null) => {
    setPreviewReady(false);
    setPreviewInvalidated(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setCorrectionSaveStatus(undefined);
    setDraft(initialDraft(caseId, manifest, caseFrame, nextTask));
    setSelectedRole(defaultRole(publicVideoRoles(manifest.video_roles)));
    editDraftContextRef.current = null;
  };

  const clearCaseTransientJobs = () => {
    setMaterializationJobs((current) => Object.fromEntries(
      Object.entries(current).filter(([, tracked]) => tracked.job.case_id !== caseId),
    ));
    setPendingEntityIds((current) => {
      const next = new Set(current);
      for (const id of caseEntityIds) next.delete(id);
      return next;
    });
  };

  const appendEvents = (nextEvents: AuditEvent[]) => {
    const combined = [...eventsRef.current, ...nextEvents];
    eventsRef.current = combined;
    onEventsChange(combined);
  };

  const trackMaterializationJob = (job: MaterializationJob | undefined, entityIds: string[] = []) => {
    const trackedEntityIds = job?.affected_entity_ids?.length ? job.affected_entity_ids : entityIds;
    if (!job?.job_id) {
      if (trackedEntityIds.length) {
        setPendingEntityIds((current) => {
          const next = new Set(current);
          trackedEntityIds.forEach((id) => next.delete(id));
          return next;
        });
      }
      return;
    }
    if (!['queued', 'running'].includes(job.status)) {
      if (trackedEntityIds.length) {
        setPendingEntityIds((current) => {
          const next = new Set(current);
          trackedEntityIds.forEach((id) => next.delete(id));
          return next;
        });
      }
      return;
    }
    setMaterializationError(undefined);
    setMaterializationJobs((current) => ({
      ...current,
      [job.job_id]: { entityIds: trackedEntityIds, job, preserveFrame: caseFrame },
    }));
  };

  const applyWorkingStateProjection = (state?: WorkingState) => {
    if (!state) return;
    workingStateRef.current = state;
    onWorkingStateChange(state);
    onDecisionsChange(
      normalizeDecisionsByCase(
        state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
        state.task_decisions as Record<string, TaskDecision> | undefined,
      ),
    );
    onTaskResolutionsChange(
      normalizeTaskResolutionsByCase(
        state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
        state.task_resolutions as Record<string, TaskDecision> | undefined,
      ),
    );
    onManualTasksChange((Array.isArray(state.active_manual_tasks) ? state.active_manual_tasks : []) as Task[]);
    onEntityOverridesChange((state.object_policy_overrides || {}) as Record<string, EffectiveEntity>);
    onCorrectionsChange((Array.isArray(state.manual_corrections) ? state.manual_corrections : []) as Record<string, unknown>[]);
    setLatestRole((state.materialized_latest_reviewed || {})[caseId] as MediaVersion | undefined);
  };

  const persistDecision = async (task: ReviewGroup, action: string) => {
    if (!session) {
      onError('Start review before saving decisions.');
      return;
    }
    if (task.child_tasks.length > 1 && action === ACTIONS.needsCorrection) {
      openCorrectionEditor(task);
      return;
    }
    setPendingTaskId(task.review_task_id);
    onError(undefined);
    try {
      let nextDecisions = decisions;
      let nextResolutions = taskResolutions;
      if (task.child_tasks.length > 1) {
        const result = await saveDecisionGroup({
          session_id: session.session_id,
          actor: attribution || { display_name: null },
          case_id: caseId,
          review_group_id: task.group_id,
          task_ids: task.child_task_ids,
          action,
          note: groupActionNote(task, action),
        });
        appendEvents(result.events);
        if (result.working_state) {
          applyWorkingStateProjection(result.working_state);
          nextDecisions = normalizeDecisionsByCase(
            result.working_state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            result.working_state.task_decisions as Record<string, TaskDecision> | undefined,
          );
          nextResolutions = normalizeTaskResolutionsByCase(
            result.working_state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            result.working_state.task_resolutions as Record<string, TaskDecision> | undefined,
          );
        }
      } else {
        const childTask = task.child_tasks[0] || task;
        const { decision, event, working_state } = await saveDecision({
          session_id: session.session_id,
          actor: attribution || { display_name: null },
          case_id: caseId,
          task_id: childTask.review_task_id,
          action,
          note:
            action === ACTIONS.looksCorrect
              ? 'Correct'
              : action === ACTIONS.notSensitive
                ? 'Not sensitive'
                : 'Needs correction',
        });
        if (working_state) {
          applyWorkingStateProjection(working_state);
          nextDecisions = normalizeDecisionsByCase(
            working_state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            working_state.task_decisions as Record<string, TaskDecision> | undefined,
          );
          nextResolutions = normalizeTaskResolutionsByCase(
            working_state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            working_state.task_resolutions as Record<string, TaskDecision> | undefined,
          );
        } else {
          nextDecisions = withTaskDecision(decisions, caseId, childTask.review_task_id, decision);
          nextResolutions = withoutTaskResolution(taskResolutions, caseId, childTask.review_task_id);
          onDecisionsChange(nextDecisions);
          onTaskResolutionsChange(nextResolutions);
        }
        appendEvents([event]);
      }
      if (action === ACTIONS.needsCorrection) {
        openCorrectionEditor(task);
        return;
      }
      const nextOpenTask = nearestNextOpenGroup(
        tasks,
        requiredReviewGroups(tasksForCase(allTasks, nextDecisions, caseId, nextResolutions)),
        task.review_task_id,
      );
      if (nextOpenTask) {
        setBucket('Open');
        setSelectedTaskId(nextOpenTask.review_task_id);
        controller.seekToSourceFrame(nextOpenTask.source_frame_start);
      } else {
        setBucket(action === ACTIONS.notSensitive ? 'Not sensitive' : 'Correct');
        setSelectedTaskId(task.review_task_id);
      }
    } catch (err) {
      onError(`Decision was not saved: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingTaskId(undefined);
    }
  };

  const openCorrectionEditor = (task: ReviewGroup) => {
    const correctionTask = task.child_tasks[0] || task;
    setBucket('Needs correction');
    setSelectedTaskId(correctionTask.review_task_id);
    setSelectedRole('latest_reviewed');
    setMode('edit');
    resetLocalCorrectionState(correctionTask);
    controller.seekToSourceFrame(correctionTask.source_frame_start);
  };

  const acknowledgeSweep = async () => {
    if (!session) {
      onError('Start review before approving the full-video review.');
      return;
    }
    if (!latestReviewedCurrent) {
      onError('Latest reviewed must be current before approving the full-video review.');
      return;
    }
    setSweepBusy(true);
    onError(undefined);
    try {
      const result = await acknowledgeFullVideoSweep({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        viewed_ranges: [
          {
            source_frame_end: manifest.source_frame_end,
            source_frame_start: manifest.source_frame_start,
          },
        ],
      });
      appendEvents([result.event]);
      try {
        const refreshedWorkingState = await authoritativeWorkingStateAfterSweepApproval(
          result,
          session.session_id,
          loadWorkingState,
        );
        applyWorkingStateProjection(refreshedWorkingState);
      } catch (refreshErr) {
        onError(
          `Full-video review was approved, but the refreshed review state could not be loaded: ${refreshErr instanceof Error ? refreshErr.message : String(refreshErr)}. Reload this review session to synchronize.`,
        );
      }
    } catch (err) {
      onError(`Full-video review was not approved: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSweepBusy(false);
    }
  };

  const updatePolicy = async (entity: EffectiveEntity, policy: EntityPolicy) => {
    if (!session) {
      onError('Start review before changing object policy.');
      return;
    }
    const after: EffectiveEntity = {
      ...entity,
      current_policy: policy,
      default_privacy_policy: policy,
      privacy_action: policy,
    };
    setPendingEntityIds((current) => new Set([...current, entity.entity_id]));
    setFocusTarget(null);
    onError(undefined);
    try {
      const result = await saveEntityPolicy({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        entity,
        after,
        policy,
      });
      appendEvents(result.events || [result.event]);
      if (result.working_state) {
        applyWorkingStateProjection(result.working_state);
      } else {
        onEntityOverridesChange({ ...entityOverrides, [entity.entity_id]: result.entity || after });
      }
      trackMaterializationJob(result.materialization_job, [entity.entity_id]);
    } catch (err) {
      setPendingEntityIds((current) => {
        const next = new Set(current);
        next.delete(entity.entity_id);
        return next;
      });
      onError(`Entity action was not saved: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const updateBulkPolicy = async (entities: EffectiveEntity[], policy: EntityPolicy) => {
    if (!session) {
      onError('Start review before changing object policy.');
      return;
    }
    if (!entities.length) return;
    const entityIds = entities.map((entity) => entity.entity_id);
    setPendingEntityIds((current) => new Set([...current, ...entityIds]));
    setFocusTarget(null);
    onError(undefined);
    try {
      const result = await saveBulkEntityPolicy({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        entity_ids: entityIds,
        policy,
      });
      appendEvents(result.events || [result.event]);
      if (result.working_state) {
        applyWorkingStateProjection(result.working_state);
      } else {
        const nextOverrides = { ...entityOverrides };
        for (const entity of result.entities) nextOverrides[entity.entity_id] = entity;
        onEntityOverridesChange(nextOverrides);
      }
      trackMaterializationJob(result.materialization_job, entityIds);
    } catch (err) {
      setPendingEntityIds((current) => {
        const next = new Set(current);
        entityIds.forEach((id) => next.delete(id));
        return next;
      });
      onError(`Bulk entity action was not saved: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const merge = async (entities: EffectiveEntity[]) => {
    if (!session) {
      onError('Start review before merging objects.');
      return;
    }
    const result = await mergeEntities({
      session_id: session.session_id,
      actor: attribution || { display_name: null },
      case_id: caseId,
      entity_ids: entities.map((entity) => entity.entity_id),
    });
    appendEvents([result.event]);
    if (result.working_state) {
      applyWorkingStateProjection(result.working_state);
    } else {
      const nextOverrides = { ...entityOverrides, [result.merged_entity.entity_id]: result.merged_entity };
      for (const retiredId of result.retired_entity_uuids) {
        nextOverrides[retiredId] = { entity_id: retiredId, current_status: 'merged' } as EffectiveEntity;
      }
      onEntityOverridesChange(nextOverrides);
    }
    setMergeSelection([]);
  };

  const saveCorrection = async (correction: Correction, reason: string) => {
    if (!session) {
      onError('Start review before saving corrections.');
      return;
    }
    const {
      event,
      events: responseEvents,
      materialization_job,
      materialization_required,
      publication_status,
      verification_task,
      working_state,
    } = await saveCorrectionRequest({
      session_id: session.session_id,
      actor: attribution || { display_name: null },
      correction,
      reason,
      preview_token: previewToken,
    });
    appendEvents(responseEvents || [event]);
    if (working_state) {
      applyWorkingStateProjection(working_state);
    } else {
      onCorrectionsChange([...corrections, event.after_state || correction]);
    }
    if (verification_task && !working_state) {
      onManualTasksChange([
        ...manualTasks.filter((task) => task.review_task_id !== verification_task.review_task_id),
        verification_task,
      ]);
    }
    setPreviewReady(false);
    setPreviewInvalidated(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setCorrectionSaveStatus(
      materialization_required
        ? 'Correction saved — applying changes...'
        : publication_status === 'deferred'
          ? 'Correction saved — review still incomplete'
          : 'Correction saved',
    );
    setSelectedRole('latest_reviewed');
    trackMaterializationJob(materialization_job, correction.entity_id ? [correction.entity_id] : []);
  };

  const previewDraftCorrection = async (correction: Correction) => {
    if (!session) {
      onError('Start review before previewing corrections.');
      return;
    }
    const preservedFrame = caseFrame;
    const { event, preview } = await previewCorrection({
      session_id: session.session_id,
      actor: attribution || { display_name: null },
      correction,
      reason: correction.reason,
    });
    if (event) appendEvents([event]);
    const path = preview.latest_reviewed || preview.corrected_preview;
    if (!path) throw new Error('Preview did not produce playable media.');
    setPreviewRole({
      label: 'Latest reviewed',
      path,
      role: 'latest_reviewed',
      sha256: null,
      source_frame_start: preview.frame_start ?? correction.source_frame_start,
      source_frame_end: preview.frame_end ?? correction.source_frame_end,
    });
    setPreviewReady(true);
    setPreviewInvalidated(false);
    setPreviewToken(preview.preview_token || undefined);
    setCorrectionSaveStatus(undefined);
    setDraft((current) => withDraftHash({ ...current, preview_token: preview.preview_token || undefined }));
    setSelectedRole('latest_reviewed');
    controller.seekToFrame(preservedFrame);
  };

  const unresolvedCount = unresolvedEvidenceCount(workspace, caseId);
  const checkpoints = (workingState?.checkpoints || []) as Record<string, unknown>[];

  const createNamedCheckpoint = async () => {
    if (!session) {
      onError('Start review before creating checkpoints.');
      return;
    }
    setCheckpointBusy('create');
    onError(undefined);
    try {
      const result = await createCheckpoint({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        name: checkpointName.trim() || `Checkpoint ${checkpoints.length + 1}`,
      });
      appendEvents([result.event]);
      applyWorkingStateProjection(result.working_state);
      setCheckpointName('');
    } catch (err) {
      onError(`Checkpoint was not created: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCheckpointBusy(null);
    }
  };

  const restoreNamedCheckpoint = async (checkpointUuid: string) => {
    if (!session) {
      onError('Start review before restoring checkpoints.');
      return;
    }
    setCheckpointBusy(`restore:${checkpointUuid}`);
    onError(undefined);
    try {
      const result = await restoreCheckpoint({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        checkpoint_uuid: checkpointUuid,
        case_id: caseId,
      });
      appendEvents([result.event]);
      applyWorkingStateProjection(result.working_state);
      for (const job of result.materialization_jobs || []) trackMaterializationJob(job);
    } catch (err) {
      onError(`Checkpoint was not restored: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCheckpointBusy(null);
    }
  };

  const resetReviewState = async () => {
    if (!session) {
      onError('Start review before resetting edits.');
      return;
    }
    setCheckpointBusy('reset-edits');
    onError(undefined);
    try {
      const result = await resetToAutomaticBaseline({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
      });
      appendEvents([result.event]);
      applyWorkingStateProjection(result.working_state);
      setBucket('Open');
      setSelectedTaskId(null);
      setMode('review');
      resetLocalCorrectionState(null);
      setMaterializationError(undefined);
      clearCaseTransientJobs();
      setFocusTarget(null);
      onError(undefined);
      for (const job of result.materialization_jobs || []) trackMaterializationJob(job);
      setResetEditsConfirmOpen(false);
    } catch (err) {
      onError(`Case edits were not reset: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCheckpointBusy(null);
    }
  };

  const deleteNamedCheckpoint = async () => {
    if (!session || !checkpointDeleteTarget) return;
    setCheckpointBusy(`delete:${String(checkpointDeleteTarget['checkpoint_uuid'] || '')}`);
    onError(undefined);
    try {
      const result = await deleteCheckpoint({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        checkpoint_uuid: String(checkpointDeleteTarget['checkpoint_uuid'] || ''),
      });
      appendEvents([result.event]);
      applyWorkingStateProjection(result.working_state);
      setCheckpointDeleteTarget(null);
    } catch (err) {
      onError(`Checkpoint was not deleted: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCheckpointBusy(null);
    }
  };

  const retryLatestMaterialization = async () => {
    if (!session) return;
    const failed = Object.values(materializationJobs).find(({ job }) => job.status === 'failed');
    if (!failed) return;
    const result = await retryMaterializationJob({
      session_id: session.session_id,
      actor: attribution || { display_name: null },
      job_id: failed.job.job_id,
    });
    trackMaterializationJob(result.job, failed.entityIds);
    setMaterializationError(undefined);
  };

  return (
    <section className="case-panel active" data-case-id={caseId}>
      <div className="case-header">
        <div>
          <h2>{manifest.display_name}</h2>
          <p className="case-subtitle">
            {manifest.frame_count} frames · source {manifest.source_frame_start}-{manifest.source_frame_end}
            {correctionSaveStatus ? ` · ${correctionSaveStatus}` : ''}
            {materializationStatusText ? ` · ${materializationStatusText}` : ''}
            {checkpointBusy ? ' · Review state update in progress...' : ''}
            {previewReady ? ' · Previewing unsaved correction' : ''}
          </p>
          {correctionSaveStatus ? (
            <div className="operation-status" role="status">
              {correctionSaveStatus.includes('applying') ? <span className="inline-spinner" aria-hidden="true" /> : null}
              <span>{correctionSaveStatus}</span>
            </div>
          ) : null}
          {materializationError ? (
            <details className="materialization-blockers">
              <summary>
                <strong>Latest reviewed not ready</strong>
                <span>{counts.Open} guidance items remain optional</span>
                <span>{correctionAttentionCount} corrections need attention</span>
                <span>View technical blockers</span>
              </summary>
              <pre>{materializationError}</pre>
              <button type="button" onClick={() => void retryLatestMaterialization()}>
                Retry
              </button>
            </details>
          ) : null}
        </div>
        <div className="segmented" role="group" aria-label="Review mode">
          <button type="button" className={mode === 'review' ? 'active' : ''} onClick={() => { setFocusTarget(null); setMode('review'); }}>
            Review
          </button>
          <button type="button" className={mode === 'edit' ? 'active' : ''} onClick={() => { setFocusTarget(null); setMode('edit'); }}>
            Manual redaction
          </button>
          <button type="button" className={entitiesOpen ? 'active' : ''} onClick={() => { setFocusTarget(null); setEntitiesOpen(true); }}>
            Entities
          </button>
        </div>
      </div>
      {session ? (
        <div className="checkpoint-strip" aria-label="Review checkpoints">
          <label>
            Checkpoint
            <input
              type="text"
              value={checkpointName}
              placeholder="Name"
              onChange={(event) => setCheckpointName(event.currentTarget.value)}
            />
          </label>
          <button type="button" onClick={() => void createNamedCheckpoint()} disabled={checkpointBusy === 'create'}>
            {checkpointBusy === 'create' ? 'Creating...' : 'Create checkpoint'}
          </button>
          {checkpoints.map((checkpoint) => (
            <span className="checkpoint-item" key={String(checkpoint['checkpoint_uuid'])}>
              <button
                type="button"
                onClick={() => void restoreNamedCheckpoint(String(checkpoint['checkpoint_uuid']))}
                disabled={checkpointBusy === `restore:${String(checkpoint['checkpoint_uuid'])}`}
                title={String(checkpoint['created_at_utc'] || '')}
              >
                {checkpointBusy === `restore:${String(checkpoint['checkpoint_uuid'])}` ? 'Restoring...' : `Restore ${String(checkpoint['name'] || 'Checkpoint')}`}
              </button>
              <button
                type="button"
                className="danger-icon"
                aria-label={`Delete checkpoint ${String(checkpoint['name'] || 'Checkpoint')}`}
                onClick={() => setCheckpointDeleteTarget(checkpoint)}
                disabled={checkpointBusy === `delete:${String(checkpoint['checkpoint_uuid'])}`}
              >
                x
              </button>
            </span>
          ))}
          <button type="button" className="danger" onClick={() => setResetEditsConfirmOpen(true)} disabled={checkpointBusy === 'reset-edits'}>
            {checkpointBusy === 'reset-edits' ? 'Resetting...' : 'Reset this case to Automatic baseline'}
          </button>
        </div>
      ) : null}
      <section className={`next-step-panel ${reviewReady ? 'complete' : 'progress'}`} aria-label={`${manifest.display_name} review progress`}>
        <div>
          <strong>{reviewReady ? (allGuidanceAddressed ? 'GUIDANCE CLEAR' : 'GUIDANCE READY') : 'NEEDS CORRECTION'}</strong>
          <p className="muted">
            Guidance remaining: {counts.Open} — optional · Needs correction: {counts['Needs correction']} — blocking · Correct {counts.Correct} · Not sensitive {counts['Not sensitive']}
          </p>
        </div>
        <div>
          <strong>{latestReviewedStatus}</strong>
          <p className="muted">{finalNextStep}</p>
        </div>
        <div className="button-row">
          <button
            type="button"
            className="primary"
            disabled={!latestReviewedCurrent || sweepBusy || (sweepAcknowledged && !sweepStale)}
            onClick={() => void acknowledgeSweep()}
          >
            {sweepBusy ? (
              <>
                <span className="inline-spinner" aria-hidden="true" /> Approving full-video review...
              </>
            ) : (
              'Approve full-video review'
            )}
          </button>
          {sweepAcknowledged && !sweepStale ? <span className="badge done">Current full-video review approved</span> : null}
          {sweepStale ? <span className="badge warn">Previous full-video approval stale</span> : null}
        </div>
      </section>
      <div className="workspace-layout">
        <div className="review-main">
          <ReviewVideoPlayer
            controller={controller}
            manifest={manifest}
            selectedTaskId={selectedTaskId}
            sessionId={session?.session_id}
            roles={roles}
            selectedRole={selectedVersion?.role || selectedRole}
            tasks={tasks}
            version={selectedVersion}
            onVersionChange={(role) => {
              setFocusTarget(null);
              setSelectedRole(role);
            }}
            onSelectTask={selectTask}
          >
            <EntityFocusOverlay focus={focusTarget} manifest={manifest} />
            {mode === 'edit' ? (
              <CorrectionVideoOverlay
                currentSourceFrame={currentSourceFrame}
                draft={draft}
                manifest={manifest}
                videoRef={controller.videoRef}
                onDraftChange={updateDraft}
              />
            ) : null}
            {pendingCorrectionRegions.length ? (
              <PendingCorrectionOverlay manifest={manifest} regions={pendingCorrectionRegions} />
            ) : null}
          </ReviewVideoPlayer>
        </div>
        {mode === 'review' ? (
          <ReviewQueue
            bucket={bucket}
            manifest={manifest}
            pendingTaskId={pendingTaskId}
            selectedTaskId={selectedTaskId}
            tasks={tasks}
            optionalTasks={optionalTasks}
            reviewComplete={allGuidanceAddressed}
            onBucketChange={setBucket}
            onDecision={persistDecision}
            onFixNow={openCorrectionEditor}
            onSelectTask={selectTask}
          />
        ) : (
          <aside className="side-panel">
            <CorrectionEditor
              actor={attribution || { display_name: null }}
              caseId={caseId}
              configurationFreezeHash={workspace.configuration_freeze_hash}
              currentSourceFrame={currentSourceFrame}
              draft={draft}
              previewInvalidated={previewInvalidated}
              previewReady={previewReady}
              selectedTask={selectedTask}
              sourceVideoHash={manifest.source_video_hash}
              onDiscard={() => {
                setPreviewReady(false);
                setPreviewToken(undefined);
                setPreviewRole(undefined);
                setCorrectionSaveStatus(undefined);
                setSelectedRole('latest_reviewed');
                setDraft(initialDraft(caseId, manifest, caseFrame, selectedTask));
              }}
              onDraftChange={updateDraft}
              onGoToSourceFrame={controller.seekToSourceFrame}
              onPreview={previewDraftCorrection}
              onSave={saveCorrection}
            />
          </aside>
        )}
      </div>
      <EntityPanel
        caseId={caseId}
        entities={(workspace.entities.cases[caseId] || []) as Entity[]}
        entityFilter={entityFilter}
        open={entitiesOpen}
        overrides={entityOverrides}
        pendingEntityIds={pendingEntityIds}
        relatedTaskCountByEntity={relatedTaskCountByEntity}
        unresolvedCount={unresolvedCount}
        onClose={() => setEntitiesOpen(false)}
        onFilterChange={setEntityFilter}
        onJump={(entity) => {
          setEntitiesOpen(false);
          controller.pause();
          const bbox = entityRepresentativeBBox(entity);
          if (typeof entity.representative_source_frame === 'number') {
            controller.seekToSourceFrame(entity.representative_source_frame);
            if (bbox) {
              setFocusTarget({
                bbox,
                entityId: entity.entity_id,
                label: entity.label,
                sourceFrame: entity.representative_source_frame,
              });
            }
          }
        }}
        onMergeSelected={setMergeSelection}
        onBulkPolicy={updateBulkPolicy}
        onPolicy={updatePolicy}
      />
      <EntityMergeDialog
        entities={mergeSelection}
        open={mergeSelection.length >= 2}
        onCancel={() => setMergeSelection([])}
        onMerge={merge}
      />
      {checkpointDeleteTarget ? (
        <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="delete-checkpoint-title">
          <div className="modal-card">
            <h2 id="delete-checkpoint-title">Delete checkpoint "{String(checkpointDeleteTarget['name'] || 'Checkpoint')}"?</h2>
            <p className="muted">The current reviewed state will not change.</p>
            <div className="button-row">
              <button type="button" className="danger" onClick={() => void deleteNamedCheckpoint()} disabled={checkpointBusy === `delete:${String(checkpointDeleteTarget['checkpoint_uuid'] || '')}`}>
                {checkpointBusy === `delete:${String(checkpointDeleteTarget['checkpoint_uuid'] || '')}` ? 'Deleting...' : 'Delete checkpoint'}
              </button>
              <button type="button" onClick={() => setCheckpointDeleteTarget(null)} disabled={checkpointBusy === `delete:${String(checkpointDeleteTarget['checkpoint_uuid'] || '')}`}>
                Cancel
              </button>
            </div>
          </div>
        </section>
      ) : null}
      {resetEditsConfirmOpen ? (
        <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="reset-edits-title">
          <div className="modal-card">
            <h2 id="reset-edits-title">Reset this case to Automatic baseline?</h2>
            <p className="muted">
              This clears only {manifest.display_name}'s active decisions, object policies, merges, manual corrections, correction drafts, preview state, Latest reviewed currentness, and full-video sweep readiness. Source media, Automatic baseline, the other case, checkpoints, and historical audit events are preserved.
            </p>
            <div className="button-row">
              <button type="button" className="danger" onClick={() => void resetReviewState()} disabled={checkpointBusy === 'reset-edits'}>
                {checkpointBusy === 'reset-edits' ? 'Resetting...' : 'Reset this case to Automatic baseline'}
              </button>
              <button type="button" onClick={() => setResetEditsConfirmOpen(false)} disabled={checkpointBusy === 'reset-edits'}>
                Cancel
              </button>
            </div>
          </div>
        </section>
      ) : null}
    </section>
  );
}

function defaultRole(roles: MediaVersion[]): string {
  if (roles.some((role) => role.role === 'latest_reviewed')) return 'latest_reviewed';
  if (roles.some((role) => role.role === 'automatic')) return 'automatic';
  return 'original';
}

function publicVideoRoles(roles: MediaVersion[], latestRole?: MediaVersion): MediaVersion[] {
  const byRole = new Map(roles.map((role) => [role.role, role]));
  const automatic = byRole.get('automatic') || byRole.get('current_result');
  return [
    byRole.get('original'),
    automatic ? { ...automatic, label: 'Automatic', role: 'automatic' as const } : undefined,
    latestRole ? { ...latestRole, label: 'Latest reviewed', role: 'latest_reviewed' as const } : undefined,
  ].filter((role): role is MediaVersion => Boolean(role));
}

function nearestNextOpenGroup(
  previousGroups: ReviewGroup[],
  nextGroups: ReviewGroup[],
  currentGroupId: string,
): ReviewGroup | undefined {
  const orderedNext = nextGroups
    .filter((group) => group.bucket === 'Open')
    .sort((left, right) => {
      if (left.source_frame_start !== right.source_frame_start) return left.source_frame_start - right.source_frame_start;
      return left.review_task_id.localeCompare(right.review_task_id);
    });
  if (!orderedNext.length) return undefined;
  const previousIndex = previousGroups.findIndex((group) => group.review_task_id === currentGroupId);
  const currentGroup = previousGroups[previousIndex];
  if (!currentGroup) return orderedNext[0];
  const next = orderedNext.find((candidate) =>
    candidate.source_frame_start > currentGroup.source_frame_start ||
    (candidate.source_frame_start === currentGroup.source_frame_start && candidate.review_task_id > currentGroup.review_task_id),
  );
  if (next) return next;
  return [...orderedNext].reverse().find((candidate) =>
    candidate.source_frame_start < currentGroup.source_frame_start ||
    (candidate.source_frame_start === currentGroup.source_frame_start && candidate.review_task_id < currentGroup.review_task_id),
  );
}

function groupActionNote(group: ReviewGroup, action: string): string {
  const label =
    action === ACTIONS.looksCorrect
      ? 'Correct'
      : action === ACTIONS.notSensitive
        ? 'Not sensitive'
        : 'Needs correction';
  return `${label} review group ${group.group_id} covering ${group.child_task_ids.length} task${group.child_task_ids.length === 1 ? '' : 's'}.`;
}

interface FinalizationBlockerGroup {
  blockers?: string[];
  count: number;
  key: string;
  label: string;
}

function finalizationBlockerGroups(status?: FinalizationStatus): FinalizationBlockerGroup[] {
  const raw = status?.['blocker_groups'];
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    .map((item) => ({
      blockers: Array.isArray(item.blockers) ? item.blockers.map(String) : undefined,
      count: Number(item.count || 0),
      key: String(item.key || ''),
      label: String(item.label || ''),
    }))
    .filter((item) => item.key && item.label);
}

function workspaceIdentityForUi(workspace: WorkspaceSummary): string {
  const identity = workspace.workspace?.workspace_manifest_sha256
    || workspace.workspace?.workspace_identity_hash
    || workspace.workspace?.source_sha256
    || workspace.backend_workspace_schema_version
    || 'unknown-workspace';
  return String(identity).replace(/[^A-Za-z0-9_-]/g, '_');
}

function latestReviewedStatusText({
  activeJobs,
  latestRole,
  materializationError,
  reviewReady,
  workingState,
}: {
  activeJobs: TrackedMaterializationJob[];
  latestRole?: MediaVersion;
  materializationError?: string;
  reviewReady: boolean;
  workingState?: WorkingState;
}): 'Needs correction blocks Latest reviewed' | 'Ready to build Latest reviewed' | 'Building Latest reviewed' | 'Latest reviewed current' | 'Latest reviewed stale — review changed' | 'Latest reviewed validation failed' {
  if (materializationError) return 'Latest reviewed validation failed';
  if (activeJobs.length) return 'Building Latest reviewed';
  if (!reviewReady) return 'Needs correction blocks Latest reviewed';
  if (!latestRole) return 'Ready to build Latest reviewed';
  if (latestRole.media_validation_passed === false) return 'Latest reviewed validation failed';
  const activeRevision = String(workingState?.active_projection_hash || '');
  const publishedRevision = String(
    latestRole.published_working_state_revision ||
      latestRole.accepted_working_state_revision ||
      latestRole.latest_reviewed_revision_id ||
      '',
  );
  if (activeRevision && publishedRevision && activeRevision !== publishedRevision) {
    return 'Latest reviewed stale — review changed';
  }
  return 'Latest reviewed current';
}

function formatMaterializationStatus(job: MaterializationJob): string {
  const rawStage = String(job.current_stage || job.status || 'updating').replaceAll('_', ' ');
  const changedFrames = typeof job.changed_source_frame_count === 'number'
    ? ` (${job.changed_source_frame_count} changed frames)`
    : '';
  return `Updating Latest reviewed: ${rawStage}${changedFrames}`;
}

function pendingRegionsForSourceFrame(
  corrections: Record<string, unknown>[],
  manualTasks: Task[],
  caseId: CaseId,
  sourceFrame: number,
  includeAllSavedCorrections = false,
): PendingCorrectionRegion[] {
  const pendingIds = new Set(
    manualTasks
      .filter((task) => task.case_id === caseId)
      .filter((task) => task.normal_task_class === 'correction_verification')
      .filter((task) => task.actionable === false && task.publication_status !== 'published')
      .map((task) => String(task.correction_id || correctionIdFromTask(task)))
      .filter(Boolean),
  );
  if (!pendingIds.size && !includeAllSavedCorrections) return [];
  return corrections
    .filter((correction) => String(correction.case_id || '') === caseId)
    .filter((correction) => includeAllSavedCorrections || pendingIds.has(String(correction.correction_id || '')))
    .filter((correction) => correction.operation === 'add_privacy_region' || correction.operation === 'replace_privacy_region')
    .map((correction) => {
      const bbox = correctionBoxAtSourceFrame(correction, sourceFrame);
      if (!bbox) return null;
      return {
        bbox_xyxy: bbox,
        correction_id: String(correction.correction_id || ''),
        source_frame: sourceFrame,
      };
    })
    .filter((region): region is PendingCorrectionRegion => Boolean(region));
}

function correctionIdFromTask(task: Task): string {
  const casePrefix = `correction_verify_${task.case_id}_`;
  if (task.review_task_id.startsWith(casePrefix)) return task.review_task_id.slice(casePrefix.length);
  return '';
}

function correctionBoxAtSourceFrame(
  correction: Record<string, unknown>,
  sourceFrame: number,
): [number, number, number, number] | null {
  const start = Number(correction.source_frame_start);
  const end = Number(correction.source_frame_end);
  if (!Number.isFinite(start) || !Number.isFinite(end) || sourceFrame < start || sourceFrame > end) {
    return null;
  }
  const keyframes = (Array.isArray(correction.keyframe_boxes) ? correction.keyframe_boxes : [])
    .map((item) => item as { source_frame?: unknown; bbox_xyxy?: unknown })
    .map((item) => ({
      source_frame: Number(item.source_frame),
      bbox_xyxy: Array.isArray(item.bbox_xyxy) ? item.bbox_xyxy.map(Number) : [],
    }))
    .filter((item) => Number.isFinite(item.source_frame) && item.bbox_xyxy.length === 4)
    .sort((left, right) => left.source_frame - right.source_frame);
  if (!keyframes.length) return null;
  if (correction.interpolation_method === 'none') {
    if (keyframes.length === 1) return tupleBox(keyframes[0].bbox_xyxy);
    const match = keyframes.find((keyframe) => keyframe.source_frame === sourceFrame);
    return match ? tupleBox(match.bbox_xyxy) : null;
  }
  if (keyframes.length === 1) return tupleBox(keyframes[0].bbox_xyxy);
  for (let index = 0; index < keyframes.length - 1; index += 1) {
    const left = keyframes[index];
    const right = keyframes[index + 1];
    if (sourceFrame < left.source_frame || sourceFrame > right.source_frame) continue;
    const span = Math.max(1, right.source_frame - left.source_frame);
    const t = (sourceFrame - left.source_frame) / span;
    return tupleBox(left.bbox_xyxy.map((value, axis) => Math.round((1 - t) * value + t * right.bbox_xyxy[axis])));
  }
  return null;
}

function tupleBox(values: number[]): [number, number, number, number] {
  return [values[0], values[1], values[2], values[3]];
}

function initialDraft(caseId: CaseId, manifest: WorkspaceSummary['cases'][string], caseFrame: number, task?: Task | null): CorrectionDraft {
  const sourceFrame =
    typeof task?.source_frame_start === 'number'
      ? task.source_frame_start
      : caseToSourceFrame(manifest, caseFrame);
  const endFrame = typeof task?.source_frame_end === 'number' ? task.source_frame_end : sourceFrame;
  const mode = endFrame === sourceFrame ? 'single_frame' : 'fixed_box';
  return withDraftHash({
    caseId,
    case_id: caseId,
    category: manualRedactionCategory(task?.category),
    start: sourceFrame,
    end: endFrame,
    start_source_frame: sourceFrame,
    end_source_frame: endFrame,
    mode,
    keyframes: [],
    barriers: [],
    checkpoints: [],
    interpolation_method: 'none',
    selectedTaskId: task?.review_task_id,
  });
}

function manualRedactionCategory(category: string | undefined): ManualRedactionCategory {
  if (MANUAL_REDACTION_CATEGORIES.includes(category as ManualRedactionCategory)) {
    return category as ManualRedactionCategory;
  }
  return 'license_plate';
}

function isTextEntryTarget(element: Element | null): boolean {
  if (!element) return false;
  const tag = element.tagName.toLowerCase();
  if ((element as HTMLElement).isContentEditable) return true;
  if (tag === 'textarea' || tag === 'select') return true;
  if (tag !== 'input') return false;
  const type = ((element as HTMLInputElement).type || 'text').toLowerCase();
  return !['button', 'checkbox', 'color', 'file', 'hidden', 'image', 'radio', 'range', 'reset', 'submit'].includes(type);
}

function unresolvedEvidenceCount(workspace: WorkspaceSummary, caseId: CaseId): number {
  const byCase = workspace.entities.unresolved_evidence_by_case as Record<string, { fragment_count?: number } | undefined> | undefined;
  return Number(byCase?.[caseId]?.fragment_count || 0);
}

function entityRepresentativeBBox(entity: EffectiveEntity): [number, number, number, number] | null {
  const direct = entity.representative_bbox;
  if (Array.isArray(direct) && direct.length === 4) return direct.map(Number) as [number, number, number, number];
  const quality = entity.representative_quality as { bbox_xyxy?: unknown } | undefined;
  if (Array.isArray(quality?.bbox_xyxy) && quality.bbox_xyxy.length === 4) {
    return quality.bbox_xyxy.map(Number) as [number, number, number, number];
  }
  return null;
}

function withDraftHash(draft: CorrectionDraft): CorrectionDraft {
  const payload = {
    barriers: draft.barriers,
    case_id: draft.case_id,
    category: draft.category,
    end_source_frame: draft.end_source_frame,
    keyframes: draft.keyframes,
    mode: draft.mode,
    start_source_frame: draft.start_source_frame,
  };
  return { ...draft, draft_hash: JSON.stringify(payload) };
}
