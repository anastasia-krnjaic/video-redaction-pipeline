import { useEffect, useMemo, useRef, useState } from 'react';
import {
  acceptMergeCandidate,
  acknowledgeFullVideoSweep,
  createCheckpoint,
  deleteCheckpoint,
  loadFinalizationStatus,
  loadMaterializationJob,
  prepareEntityPreviews,
  loadWorkingState,
  mergeEntities,
  previewCorrection,
  rejectMergeCandidate,
  removeCorrection as removeCorrectionRequest,
  resetToAutomaticBaseline,
  requestLatestReviewedBuild,
  retryMaterializationJob,
  restoreCheckpoint,
  saveCorrection as saveCorrectionRequest,
  saveDecision,
  saveDecisionGroup,
  saveBulkEntityPolicy,
  saveEntityPolicy,
  splitMixedEntity,
  splitTrackletOut,
} from '../api/client';
import {
  CorrectionEditor,
  CorrectionVideoOverlay,
  PendingCorrectionOverlay,
  resetSingleFrameDraftForSourceFrame,
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
  needsCorrectionBlockerCount,
  optionalReviewGroups,
  requiredReviewGroups,
  tasksForCase,
  normalizeDecisionsByCase,
  normalizeTaskResolutionsByCase,
  withTaskDecision,
  withoutTaskResolution,
} from '../state/reviewState';
import {
  latestReviewedLifecycleState,
  materializationJobMatchesCurrentProjection,
  type LatestReviewedLifecycleState,
} from '../state/materialization';
import {
  draftHasRenderableGeometry,
  pendingRegionsForSourceFrame,
  roleAfterCorrectionSave,
  validRoleOrAutomatic,
} from '../state/correctionLifecycle';
import { decisionFromEvent, workingStateIsOlderThanKnownEvents } from '../state/reviewSync';
import { candidateEntityIds } from '../state/entityRefinement';
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
  completionStatus?: string;
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
  const [pendingTaskIds, setPendingTaskIds] = useState<Set<string>>(() => new Set());
  const [pendingEntityIds, setPendingEntityIds] = useState<Set<string>>(() => new Set());
  const [pendingPreviewEntityIds, setPendingPreviewEntityIds] = useState<Set<string>>(() => new Set());
  const [entityPreviewOverrides, setEntityPreviewOverrides] = useState<Record<string, EffectiveEntity>>({});
  const [latestRole, setLatestRole] = useState<MediaVersion | undefined>();
  const [deferredLatestRole, setDeferredLatestRole] = useState<MediaVersion | undefined>();
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
  const [removingCorrectionId, setRemovingCorrectionId] = useState<string | null>(null);
  const [focusTarget, setFocusTarget] = useState<EntityFocusTarget | null>(null);
  const decisionsRef = useRef(decisions);
  const eventsRef = useRef(events);
  const materializationJobsRef = useRef(materializationJobs);
  const modeRef = useRef(mode);
  const onWorkingStateChangeRef = useRef(onWorkingStateChange);
  const selectedRoleValueRef = useRef(selectedRole);
  const setSelectedRoleRef = useRef(setSelectedRole);
  const seekToFrameRef = useRef<(frame: number) => void>(() => {});
  const taskResolutionsRef = useRef(taskResolutions);
  const workingStateRef = useRef<WorkingState | undefined>(workingState);
  const controllerPlayingRef = useRef(false);
  const caseFrameValueRef = useRef(caseFrame);
  const editDraftContextRef = useRef<string | null>(null);

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
  const entityPanelOverrides = useMemo(() => {
    const merged: Record<string, EffectiveEntity> = { ...entityOverrides };
    for (const [entityId, preview] of Object.entries(entityPreviewOverrides)) {
      merged[entityId] = {
        ...(merged[entityId] || {}),
        ...preview,
      } as EffectiveEntity;
    }
    return merged;
  }, [entityOverrides, entityPreviewOverrides]);

  useEffect(() => {
    setEntityPreviewOverrides({});
    setPendingPreviewEntityIds(new Set());
  }, [caseId, session?.session_id]);

  const roles = useMemo(() => {
    const sessionLatestRole = previewRole || latestRole || materializedLatestRole;
    return publicVideoRoles(manifest.video_roles, sessionLatestRole);
  }, [latestRole, manifest.video_roles, materializedLatestRole, previewRole]);
  const selectedVersion = roles.find((role) => role.role === selectedRole) || roles.find((role) => role.role === defaultRole(roles));
  const controller = useVideoController(manifest, selectedVersion, caseFrame, setCaseFrame);
  const [draft, setDraft] = useState<CorrectionDraft>(() => initialDraft(caseId, manifest, caseFrame, selectedTask));
  const currentSourceFrame = caseToSourceFrame(manifest, caseFrame);
  const savedCorrectionsForCase = useMemo(
    () => corrections.filter((correction) => String(correction.case_id || '') === caseId),
    [caseId, corrections],
  );
  const activeMaterializationJobs = Object.values(materializationJobs).filter(({ job }) => job.case_id === caseId && (job.status === 'queued' || job.status === 'running'));
  const caseMaterializationJobs = useMemo(
    () => Object.values(materializationJobs).filter(({ job }) => job.case_id === caseId).map(({ job }) => job),
    [caseId, materializationJobs],
  );
  const pendingMediaChangeCount = useMemo(
    () => pendingMediaChangesForCase(workingState, caseId),
    [caseId, workingState],
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
  const correctionAttentionCount = needsCorrectionBlockerCount(tasks);
  const reviewReady = correctionAttentionCount === 0;
  const allGuidanceAddressed = reviewReady && counts.Open === 0;
  const latestReviewedBackendStale = finalizationStatus
    ? finalizationBlockerGroups(finalizationStatus).some((group) => group.key === 'latest_reviewed')
    : undefined;
  const latestReviewedLifecycle = latestReviewedLifecycleState({
    activeJobs: caseMaterializationJobs,
    caseLatestReviewedStale: latestReviewedBackendStale,
    caseId,
    latestRole: materializedLatestRole,
    materializationError,
    reviewReady,
    sessionId: session?.session_id,
    workingState,
  });
  const localDraftOverlayActive = mode === 'edit' && (
    draftHasRenderableGeometry(draft) ||
    previewReady ||
    Boolean(previewToken) ||
    Boolean(previewRole)
  );
  const savedCorrectionOverlayActive = mode === 'edit'
    && !localDraftOverlayActive
    && (latestReviewedLifecycle === 'BUILD_REQUIRED' || latestReviewedLifecycle === 'BUILD_RUNNING');
  const pendingCorrectionRegions = useMemo(
    () => savedCorrectionOverlayActive
      ? pendingRegionsForSourceFrame(corrections, manualTasks, caseId, currentSourceFrame)
      : [],
    [caseId, corrections, currentSourceFrame, manualTasks, savedCorrectionOverlayActive],
  );
  const latestReviewedStatus = latestReviewedStatusText({
    activeJobs: activeMaterializationJobs,
    finalizationStatus,
    lifecycleState: latestReviewedLifecycle,
    latestRole: materializedLatestRole,
    materializationError,
    reviewReady,
  });
  const latestReviewedCurrent = latestReviewedStatus === 'Reviewed video ready';
  const sweepAcknowledgement = (workingState?.sweep_acknowledgements as Record<string, unknown> | undefined)?.[caseId] as Record<string, unknown> | undefined;
  const sweepAcknowledged = Boolean(sweepAcknowledgement);
  const sweepStale = sweepAcknowledged && (
    !latestReviewedCurrent ||
    finalizationBlockerGroups(finalizationStatus).some((group) => group.key === 'full_video_approval')
  );
  const lifecycleMessage = reviewLifecycleMessage({
    correctionAttentionCount,
    latestReviewedStatus,
    reviewReady,
    sweepAcknowledged,
    sweepStale,
  });

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  useEffect(() => {
    caseFrameValueRef.current = caseFrame;
  }, [caseFrame]);

  useEffect(() => {
    decisionsRef.current = decisions;
  }, [decisions]);

  useEffect(() => {
    materializationJobsRef.current = materializationJobs;
  }, [materializationJobs]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    onWorkingStateChangeRef.current = onWorkingStateChange;
  }, [onWorkingStateChange]);

  useEffect(() => {
    selectedRoleValueRef.current = selectedRole;
  }, [selectedRole]);

  useEffect(() => {
    taskResolutionsRef.current = taskResolutions;
  }, [taskResolutions]);

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
    setPreviewReady(false);
    setPreviewInvalidated(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setCorrectionSaveStatus(undefined);
    setMaterializationError(undefined);
    setRemovingCorrectionId(null);
    setDraft(initialDraft(caseId, manifest, caseFrameValueRef.current, null));
    editDraftContextRef.current = null;
  }, [caseId, manifest, session?.session_id, uiScope]);

  useEffect(() => {
    if (latestReviewedLifecycle === 'CURRENT_REUSABLE' && correctionSaveStatus?.includes('pending')) {
      setCorrectionSaveStatus(undefined);
    }
  }, [correctionSaveStatus, latestReviewedLifecycle]);

  useEffect(() => {
    controllerPlayingRef.current = controller.playing;
    if (!controller.playing && deferredLatestRole) {
      setLatestRole(deferredLatestRole);
      setDeferredLatestRole(undefined);
    }
  }, [controller.playing, deferredLatestRole]);

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
      return initialDraft(caseId, manifest, caseFrame, selectedTask);
    });
  }, [caseId, controller.pause, manifest, mode, selectedTask?.review_task_id]);

  useEffect(() => {
    if (mode !== 'edit' || draft.mode !== 'single_frame') return;
    if (
      draft.start_source_frame === currentSourceFrame &&
      draft.end_source_frame === currentSourceFrame &&
      draft.keyframes.every((keyframe) => keyframe.source_frame === currentSourceFrame)
    ) {
      return;
    }
    setPreviewReady(false);
    setPreviewInvalidated(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setDraft((current) => {
      const reset = resetSingleFrameDraftForSourceFrame(current, currentSourceFrame);
      const resetManualTask =
        !selectedTask?.review_task_id &&
        reset !== current &&
        String(current.selectedTaskId || '').startsWith(`manual_${caseId}_`);
      return withDraftHash({
        ...reset,
        selectedTaskId: resetManualTask
          ? manualDraftTaskId(caseId, currentSourceFrame)
          : reset.selectedTaskId,
      });
    });
  }, [caseId, currentSourceFrame, draft.end_source_frame, draft.keyframes, draft.mode, draft.start_source_frame, mode, selectedTask?.review_task_id]);

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
            applyWorkingStateProjection(tracked.job.working_state);
          }
          if (jobMatchesCurrentProjection && tracked.job.latest_reviewed && tracked.job.case_id === caseId) {
            const manualPlaybackActive = modeRef.current === 'edit' && controllerPlayingRef.current;
            setLatestRoleWithoutPlaybackInterruption(tracked.job.latest_reviewed);
            if (!manualPlaybackActive) {
              const restoreFrame = tracked.preserveFrame;
              setSelectedRoleRef.current('latest_reviewed');
              window.setTimeout(() => seekToFrameRef.current(restoreFrame), 0);
            }
            if (tracked.completionStatus) {
              setCorrectionSaveStatus(tracked.completionStatus);
              window.setTimeout(() => setCorrectionSaveStatus((current) => (
                current === tracked.completionStatus ? undefined : current
              )), 3000);
            }
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
          setMaterializationError(tracked.job.error || 'Reviewed video update failed.');
          if (tracked.completionStatus?.startsWith('Correction saved')) {
            setCorrectionSaveStatus('Correction saved');
          } else {
            setCorrectionSaveStatus(undefined);
          }
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
    const seen = new Set(eventsRef.current.map((event) => event.event_hash));
    const combined = [
      ...eventsRef.current,
      ...nextEvents.filter((event) => !seen.has(event.event_hash)),
    ];
    eventsRef.current = combined;
    onEventsChange(combined);
  };

  const setLatestRoleWithoutPlaybackInterruption = (role?: MediaVersion) => {
    if (
      role &&
      modeRef.current === 'edit' &&
      controllerPlayingRef.current &&
      selectedRoleValueRef.current === 'latest_reviewed'
    ) {
      setDeferredLatestRole(role);
      return;
    }
    setLatestRole(role);
  };

  const trackMaterializationJob = (
    job: MaterializationJob | undefined,
    entityIds: string[] = [],
    completionStatus?: string,
  ) => {
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
      [job.job_id]: {
        completionStatus,
        entityIds: trackedEntityIds,
        job,
        preserveFrame: caseFrameValueRef.current,
      },
    }));
  };

  const applyWorkingStateProjection = (state?: WorkingState, responseEvents: AuditEvent[] = []) => {
    if (!state) return false;
    if (workingStateIsOlderThanKnownEvents(state, eventsRef.current, responseEvents)) {
      return false;
    }
    workingStateRef.current = state;
    onWorkingStateChange(state);
    const nextDecisions = normalizeDecisionsByCase(
      state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
      state.task_decisions as Record<string, TaskDecision> | undefined,
    );
    const nextResolutions = normalizeTaskResolutionsByCase(
      state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
      state.task_resolutions as Record<string, TaskDecision> | undefined,
    );
    decisionsRef.current = nextDecisions;
    taskResolutionsRef.current = nextResolutions;
    onDecisionsChange(nextDecisions);
    onTaskResolutionsChange(nextResolutions);
    onManualTasksChange((Array.isArray(state.active_manual_tasks) ? state.active_manual_tasks : []) as Task[]);
    onEntityOverridesChange((state.object_policy_overrides || {}) as Record<string, EffectiveEntity>);
    onCorrectionsChange((Array.isArray(state.manual_corrections) ? state.manual_corrections : []) as Record<string, unknown>[]);
    setLatestRoleWithoutPlaybackInterruption((state.materialized_latest_reviewed || {})[caseId] as MediaVersion | undefined);
    return true;
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
    setPendingTaskIds((current) => new Set(current).add(task.review_task_id));
    onError(undefined);
    try {
      let nextDecisions = decisionsRef.current;
      let nextResolutions = taskResolutionsRef.current;
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
        const responseEvents = result.events || [];
        if (result.working_state && applyWorkingStateProjection(result.working_state, responseEvents)) {
          nextDecisions = normalizeDecisionsByCase(
            result.working_state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            result.working_state.task_decisions as Record<string, TaskDecision> | undefined,
          );
          nextResolutions = normalizeTaskResolutionsByCase(
            result.working_state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            result.working_state.task_resolutions as Record<string, TaskDecision> | undefined,
          );
        } else {
          for (const responseEvent of responseEvents) {
            const decision = decisionFromEvent(responseEvent);
            if (!decision) continue;
            nextDecisions = withTaskDecision(nextDecisions, caseId, decision.task_id, decision);
            nextResolutions = withoutTaskResolution(nextResolutions, caseId, decision.task_id);
          }
          decisionsRef.current = nextDecisions;
          taskResolutionsRef.current = nextResolutions;
          onDecisionsChange(nextDecisions);
          onTaskResolutionsChange(nextResolutions);
        }
        appendEvents(responseEvents);
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
        const responseEvents = [event].filter(Boolean);
        if (working_state && applyWorkingStateProjection(working_state, responseEvents)) {
          nextDecisions = normalizeDecisionsByCase(
            working_state.task_decisions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            working_state.task_decisions as Record<string, TaskDecision> | undefined,
          );
          nextResolutions = normalizeTaskResolutionsByCase(
            working_state.task_resolutions_by_case as Record<string, Record<string, TaskDecision>> | undefined,
            working_state.task_resolutions as Record<string, TaskDecision> | undefined,
          );
        } else {
          nextDecisions = withTaskDecision(decisionsRef.current, caseId, childTask.review_task_id, decision);
          nextResolutions = withoutTaskResolution(taskResolutionsRef.current, caseId, childTask.review_task_id);
          decisionsRef.current = nextDecisions;
          taskResolutionsRef.current = nextResolutions;
          onDecisionsChange(nextDecisions);
          onTaskResolutionsChange(nextResolutions);
        }
        appendEvents(responseEvents);
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
      setPendingTaskIds((current) => {
        const next = new Set(current);
        next.delete(task.review_task_id);
        return next;
      });
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
      onError('Reviewed video must be current before approving the full-video review.');
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

  const acceptCandidate = async (candidate: Record<string, unknown>) => {
    if (!session) {
      onError('Start review before accepting merge candidates.');
      return;
    }
    const entityIds = candidateEntityIds(candidate);
    if (entityIds.length !== 2) return;
    setPendingEntityIds((current) => new Set([...current, ...entityIds]));
    try {
      const result = await acceptMergeCandidate({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        candidate_id: String(candidate.candidate_id || ''),
        entity_ids: entityIds,
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
      trackMaterializationJob(result.materialization_job, [result.merged_entity.entity_id, ...entityIds]);
    } catch (err) {
      onError(`Merge candidate was not accepted: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingEntityIds((current) => {
        const next = new Set(current);
        entityIds.forEach((id) => next.delete(id));
        return next;
      });
    }
  };

  const rejectCandidate = async (candidate: Record<string, unknown>) => {
    if (!session) {
      onError('Start review before rejecting merge candidates.');
      return;
    }
    const entityIds = candidateEntityIds(candidate);
    try {
      const result = await rejectMergeCandidate({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        candidate_id: String(candidate.candidate_id || ''),
        entity_ids: entityIds,
      });
      appendEvents([result.event]);
      if (result.working_state) applyWorkingStateProjection(result.working_state);
    } catch (err) {
      onError(`Merge candidate rejection was not saved: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const splitMixed = async (entity: EffectiveEntity) => {
    if (!session) {
      onError('Start review before splitting entities.');
      return;
    }
    setPendingEntityIds((current) => new Set([...current, entity.entity_id]));
    try {
      const result = await splitMixedEntity({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        entity_id: entity.entity_id,
      });
      appendEvents([result.event]);
      if (result.working_state) applyWorkingStateProjection(result.working_state);
      trackMaterializationJob(result.materialization_job, [entity.entity_id, ...result.split_entities.map((item) => item.entity_id)]);
    } catch (err) {
      onError(`Mixed entity was not split: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingEntityIds((current) => {
        const next = new Set(current);
        next.delete(entity.entity_id);
        return next;
      });
    }
  };

  const splitTracklet = async (entity: EffectiveEntity, trackletId: string) => {
    if (!session) {
      onError('Start review before splitting tracklets.');
      return;
    }
    setPendingEntityIds((current) => new Set([...current, entity.entity_id]));
    try {
      const result = await splitTrackletOut({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        entity_id: entity.entity_id,
        tracklet_id: trackletId,
      });
      appendEvents([result.event]);
      if (result.working_state) applyWorkingStateProjection(result.working_state);
      trackMaterializationJob(result.materialization_job, [entity.entity_id, ...result.split_entities.map((item) => item.entity_id)]);
    } catch (err) {
      onError(`Tracklet was not split: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingEntityIds((current) => {
        const next = new Set(current);
        next.delete(entity.entity_id);
        return next;
      });
    }
  };

  const loadEntityPreviews = async (entity: EffectiveEntity) => {
    if (!session) {
      onError('Start review before loading entity previews.');
      return;
    }
    setPendingPreviewEntityIds((current) => new Set([...current, entity.entity_id]));
    try {
      const result = await prepareEntityPreviews({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        entity_id: entity.entity_id,
      });
      setEntityPreviewOverrides((current) => ({
        ...current,
        [entity.entity_id]: result.entity,
      }));
    } catch (err) {
      onError(`Entity previews were not loaded: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingPreviewEntityIds((current) => {
        const next = new Set(current);
        next.delete(entity.entity_id);
        return next;
      });
    }
  };

  const saveCorrection = async (correction: Correction, reason: string) => {
    if (!session) {
      onError('Start review before saving corrections.');
      return;
    }
    const preservedFrame = caseFrameValueRef.current;
    const roleBeforeSave = selectedRoleValueRef.current;
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
    const savedStatus = publication_status === 'deferred'
      ? 'Correction saved — review still incomplete'
      : 'Correction saved';
    setCorrectionSaveStatus(savedStatus);
    window.setTimeout(() => setCorrectionSaveStatus((current) => (
      current === savedStatus ? undefined : current
    )), 3000);
    setDraft(initialDraft(caseId, manifest, preservedFrame, null));
    setSelectedRole(roleAfterCorrectionSave(roleBeforeSave, roles, Boolean(materialization_required)));
    window.setTimeout(() => seekToFrameRef.current(preservedFrame), 0);
    if (materialization_job) {
      trackMaterializationJob(
        materialization_job,
        correction.entity_id ? [correction.entity_id] : [],
        'Reviewed video ready',
      );
    }
  };

  const removeSavedCorrection = async (correction: Record<string, unknown>) => {
    if (!session) {
      onError('Start review before removing corrections.');
      return;
    }
    const correctionId = String(correction.correction_id || '');
    if (!correctionId) {
      onError('Saved correction is missing an ID.');
      return;
    }
    const preservedFrame = caseFrameValueRef.current;
    const preservedRole = selectedRoleValueRef.current;
    setRemovingCorrectionId(correctionId);
    setCorrectionSaveStatus(undefined);
    setPreviewReady(false);
    setPreviewInvalidated(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    onError(undefined);
    try {
      const result = await removeCorrectionRequest({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        correction_id: correctionId,
        reason: 'Saved correction removed by reviewer.',
      });
      appendEvents(result.events || [result.event].filter((event): event is AuditEvent => Boolean(event)));
      if (result.working_state) applyWorkingStateProjection(result.working_state);
      setSelectedRole(validRoleOrAutomatic(preservedRole, roles));
      window.setTimeout(() => seekToFrameRef.current(preservedFrame), 0);
      trackMaterializationJob(result.materialization_job, []);
    } catch (err) {
      onError(`Saved correction was not removed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRemovingCorrectionId(null);
    }
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
    trackMaterializationJob(result.job, failed.entityIds, failed.completionStatus);
    setMaterializationError(undefined);
  };

  const requestReviewedVideoBuild = async () => {
    if (!session?.session_id) return;
    const preservedFrame = caseFrameValueRef.current;
    setMaterializationError(undefined);
    setCorrectionSaveStatus('Building reviewed video...');
    try {
      const result = await requestLatestReviewedBuild({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
      });
      if (result.working_state) applyWorkingStateProjection(result.working_state);
      if (result.latest_reviewed) setLatestRoleWithoutPlaybackInterruption(result.latest_reviewed);
      trackMaterializationJob(result.materialization_job || result.job, [], 'Reviewed video ready');
      if (!(result.materialization_job || result.job)) {
        setCorrectionSaveStatus(result.latest_reviewed ? 'Reviewed video ready' : undefined);
        if (result.latest_reviewed) {
          setSelectedRole('latest_reviewed');
          window.setTimeout(() => seekToFrameRef.current(preservedFrame), 0);
        }
        window.setTimeout(() => setCorrectionSaveStatus((current) => (
          current === 'Reviewed video ready' ? undefined : current
        )), 3000);
      }
    } catch (err) {
      setMaterializationError(err instanceof Error ? err.message : String(err));
      setCorrectionSaveStatus(undefined);
    }
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
          <strong>Reviewed video not ready</strong>
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
          <strong>{lifecycleMessage}</strong>
          {latestReviewedLifecycle === 'BUILD_REQUIRED' ? (
            <p className="muted">
              {pendingMediaChangeCount > 0
                ? `${pendingMediaChangeCount} saved media change${pendingMediaChangeCount === 1 ? '' : 's'} pending reviewed-video build`
                : 'No saved media changes; Automatic can be published as Latest'}
            </p>
          ) : null}
        </div>
        <div className="button-row">
          <button
            type="button"
            disabled={!session?.session_id || latestReviewedLifecycle !== 'BUILD_REQUIRED' || Boolean(activeMaterializationJobs.length)}
            onClick={() => void requestReviewedVideoBuild()}
          >
            {activeMaterializationJobs.length ? (
              <>
                <span className="inline-spinner" aria-hidden="true" /> Building reviewed video...
              </>
            ) : (
              'Build reviewed video'
            )}
          </button>
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
              <PendingCorrectionOverlay
                manifest={manifest}
                regions={pendingCorrectionRegions}
                statusLabel={latestReviewedLifecycle === 'BUILD_RUNNING' ? 'Correction saved — building reviewed video' : 'Correction saved — pending reviewed-video build'}
              />
            ) : null}
          </ReviewVideoPlayer>
        </div>
        {mode === 'review' ? (
          <ReviewQueue
            bucket={bucket}
            manifest={manifest}
            pendingTaskIds={pendingTaskIds}
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
              buildRunning={activeMaterializationJobs.length > 0}
              latestReviewedCurrent={latestReviewedCurrent}
              savedCorrections={savedCorrectionsForCase}
              selectedTask={selectedTask}
              sourceVideoHash={manifest.source_video_hash}
              onDiscard={() => {
                setPreviewReady(false);
                setPreviewToken(undefined);
                setPreviewRole(undefined);
                setCorrectionSaveStatus(undefined);
                setSelectedRole(defaultRole(roles));
                setDraft(initialDraft(caseId, manifest, caseFrame, selectedTask));
              }}
              onDraftChange={updateDraft}
              onGoToSourceFrame={controller.seekToSourceFrame}
              onInspectCorrection={controller.seekToSourceFrame}
              onPreview={previewDraftCorrection}
              onRemoveCorrection={removeSavedCorrection}
              onSave={saveCorrection}
              removingCorrectionId={removingCorrectionId}
            />
          </aside>
        )}
      </div>
      <EntityPanel
        caseId={caseId}
        entities={(workspace.entities.cases[caseId] || []) as Entity[]}
        entityFilter={entityFilter}
        events={events}
        open={entitiesOpen}
        overrides={entityPanelOverrides}
        pendingEntityIds={pendingEntityIds}
        pendingPreviewEntityIds={pendingPreviewEntityIds}
        relatedTaskCountByEntity={relatedTaskCountByEntity}
        unresolvedCount={unresolvedCount}
        onAcceptCandidate={acceptCandidate}
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
        onLoadPreviews={loadEntityPreviews}
        onMergeSelected={setMergeSelection}
        onBulkPolicy={updateBulkPolicy}
        onPolicy={updatePolicy}
        onRejectCandidate={rejectCandidate}
        onSplitMixed={splitMixed}
        onSplitTracklet={splitTracklet}
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
              This clears only {manifest.display_name}'s active decisions, object policies, merges, manual corrections, correction drafts, preview state, reviewed-video currentness, and full-video sweep readiness. Source media, Automatic baseline, the other case, checkpoints, and historical audit events are preserved.
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

type ReviewedVideoStatus =
  | 'Needs correction blocks reviewed video'
  | 'Ready to build reviewed video'
  | 'Updating reviewed video...'
  | 'Reviewed video ready'
  | 'Could not update reviewed video — Retry';

function latestReviewedStatusText({
  activeJobs,
  finalizationStatus,
  lifecycleState,
  latestRole,
  materializationError,
  reviewReady,
}: {
  activeJobs: TrackedMaterializationJob[];
  finalizationStatus?: FinalizationStatus;
  lifecycleState: LatestReviewedLifecycleState;
  latestRole?: MediaVersion;
  materializationError?: string;
  reviewReady: boolean;
}): ReviewedVideoStatus {
  if (materializationError) return 'Could not update reviewed video — Retry';
  if (activeJobs.length) return 'Updating reviewed video...';
  if (lifecycleState === 'BUILD_RUNNING') return 'Updating reviewed video...';
  if (lifecycleState === 'BUILD_FAILED_RETRYABLE') return 'Could not update reviewed video — Retry';
  if (!reviewReady) return 'Needs correction blocks reviewed video';
  if (!latestRole) return 'Ready to build reviewed video';
  if (latestRole.media_validation_passed === false) return 'Could not update reviewed video — Retry';
  if (
    lifecycleState !== 'CURRENT_REUSABLE' &&
    finalizationBlockerGroups(finalizationStatus).some((group) => group.key === 'latest_reviewed')
  ) {
    return 'Ready to build reviewed video';
  }
  return 'Reviewed video ready';
}

function reviewLifecycleMessage({
  correctionAttentionCount,
  latestReviewedStatus,
  reviewReady,
  sweepAcknowledged,
  sweepStale,
}: {
  correctionAttentionCount: number;
  latestReviewedStatus: ReviewedVideoStatus;
  reviewReady: boolean;
  sweepAcknowledged: boolean;
  sweepStale: boolean;
}): string {
  if (!reviewReady) return `Needs correction: ${correctionAttentionCount}`;
  if (latestReviewedStatus !== 'Reviewed video ready') return latestReviewedStatus;
  if (sweepAcknowledged && !sweepStale) return 'Full-video review approved';
  if (sweepStale) return 'Full-video approval stale';
  return 'Reviewed video ready';
}

function formatMaterializationStatus(job: MaterializationJob): string {
  const rawStage = String(job.current_stage || job.status || 'updating').replaceAll('_', ' ');
  const changedFrames = typeof job.changed_source_frame_count === 'number'
    ? ` (${job.changed_source_frame_count} changed frames)`
    : '';
  return `Updating reviewed video: ${rawStage}${changedFrames}`;
}

function pendingMediaChangesForCase(workingState: WorkingState | undefined, caseId: CaseId): number {
  const currentRevision = String(workingState?.active_projection_hash || '');
  const latest = (workingState?.materialized_latest_reviewed || {})[caseId] as Record<string, unknown> | undefined;
  const latestRevision = String(
    latest?.published_working_state_revision || latest?.accepted_working_state_revision || '',
  );
  if (currentRevision && latestRevision === currentRevision) return 0;
  const corrections = Array.isArray(workingState?.manual_corrections)
    ? workingState.manual_corrections.filter((correction) => String(correction.case_id || '') === caseId).length
    : 0;
  const entityActions = Object.values(workingState?.object_policy_overrides || {})
    .filter((entity) => String((entity as Record<string, unknown>).case_id || '') === caseId)
    .length;
  return corrections + entityActions;
}

function initialDraft(caseId: CaseId, manifest: WorkspaceSummary['cases'][string], caseFrame: number, task?: Task | null): CorrectionDraft {
  const taskNeedsCorrection =
    (task && 'bucket' in task && task.bucket === 'Needs correction') ||
    task?.normal_task_class === 'correction_verification' ||
    task?.correction_required === true ||
    task?.decision_state === 'needs_correction';
  const sourceFrame =
    taskNeedsCorrection && typeof task?.source_frame_start === 'number'
      ? task.source_frame_start
      : caseToSourceFrame(manifest, caseFrame);
  const endFrame = taskNeedsCorrection && typeof task?.source_frame_end === 'number' ? task.source_frame_end : sourceFrame;
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
    selectedTaskId: task?.review_task_id || manualDraftTaskId(caseId, sourceFrame),
  });
}

function manualDraftTaskId(caseId: CaseId, sourceFrame: number): string {
  return `manual_${caseId}_${sourceFrame}_${Date.now()}`;
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
