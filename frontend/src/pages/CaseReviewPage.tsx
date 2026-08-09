import { useEffect, useMemo, useRef, useState } from 'react';
import {
  createCheckpoint,
  deleteCheckpoint,
  loadMaterializationJob,
  mergeEntities,
  previewCorrection,
  resetToAutomaticBaseline,
  restartReviewWorkflow,
  retryMaterializationJob,
  restoreCheckpoint,
  saveCorrection as saveCorrectionRequest,
  saveDecision,
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
  policyVerificationTask,
  tasksForCase,
  withPolicyTaskId,
} from '../state/reviewState';
import type {
  Attribution,
  AuditEvent,
  Correction,
  Entity,
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
  EffectiveEntity,
  EntityFilter,
  EntityPolicy,
  QueueBucket,
  ReviewMode,
  TaskWithDecision,
} from '../types/domain';
import { caseToSourceFrame } from '../utils/frame';

interface Props {
  attribution: Attribution;
  caseId: CaseId;
  corrections: Record<string, unknown>[];
  decisions: Record<string, TaskDecision>;
  entityOverrides: Record<string, EffectiveEntity>;
  events: AuditEvent[];
  manualTasks: Task[];
  session: Session | null;
  workingState?: WorkingState;
  workspace: WorkspaceSummary;
  onCorrectionsChange: (corrections: Record<string, unknown>[]) => void;
  onDecisionsChange: (decisions: Record<string, TaskDecision>) => void;
  onEntityOverridesChange: (overrides: Record<string, EffectiveEntity>) => void;
  onError: (message?: string) => void;
  onEventsChange: (events: AuditEvent[]) => void;
  onManualTasksChange: (tasks: Task[]) => void;
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
  onWorkingStateChange,
  session,
  workingState,
  workspace,
}: Props) {
  const manifest = workspace.cases[caseId];
  const baseTasks = workspace.tasks.tasks || [];
  const [mode, setMode] = usePersistentState<ReviewMode>(`${caseId}:mode`, 'review');
  const [caseFrame, setCaseFrame] = usePersistentState<number>(`${caseId}:frame`, 0);
  const [selectedRole, setSelectedRole] = usePersistentState<string>(`${caseId}:version`, defaultRole(manifest.video_roles));
  const [selectedTaskId, setSelectedTaskId] = usePersistentState<string | null>(`${caseId}:task`, null);
  const [bucket, setBucket] = usePersistentState<QueueBucket>(`${caseId}:bucket`, 'Open');
  const [entityFilter, setEntityFilter] = usePersistentState<EntityFilter>(`${caseId}:entity_filter`, 'needs_review');
  const [entitiesOpen, setEntitiesOpen] = useState(false);
  const [mergeSelection, setMergeSelection] = useState<EffectiveEntity[]>([]);
  const [pendingTaskId, setPendingTaskId] = useState<string>();
  const [pendingEntityIds, setPendingEntityIds] = useState<Set<string>>(() => new Set());
  const [latestRole, setLatestRole] = useState<MediaVersion | undefined>();
  const [previewRole, setPreviewRole] = useState<MediaVersion | undefined>();
  const [previewReady, setPreviewReady] = useState(false);
  const [previewToken, setPreviewToken] = useState<string>();
  const [correctionSaveStatus, setCorrectionSaveStatus] = useState<string>();
  const [checkpointName, setCheckpointName] = useState('');
  const [checkpointBusy, setCheckpointBusy] = useState<string | null>(null);
  const [checkpointDeleteTarget, setCheckpointDeleteTarget] = useState<Record<string, unknown> | null>(null);
  const [restartConfirmOpen, setRestartConfirmOpen] = useState(false);
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

  const allTasks = useMemo(
    () => [...baseTasks, ...manualTasks].filter((task) => task.case_id === caseId),
    [baseTasks, caseId, manualTasks],
  );
  const tasks = useMemo(() => tasksForCase(allTasks, decisions, caseId), [allTasks, caseId, decisions]);
  const selectedTask = tasks.find((task) => task.review_task_id === selectedTaskId) || tasks[0] || null;
  const materializedLatestRole = workingState?.materialized_latest_reviewed?.[caseId] as MediaVersion | undefined;

  const roles = useMemo(() => {
    const sessionLatestRole = previewRole || latestRole || materializedLatestRole;
    return publicVideoRoles(manifest.video_roles, sessionLatestRole);
  }, [latestRole, manifest.video_roles, materializedLatestRole, previewRole]);
  const selectedVersion = roles.find((role) => role.role === selectedRole) || roles.find((role) => role.role === defaultRole(roles));
  const controller = useVideoController(manifest, selectedVersion, caseFrame, setCaseFrame);
  const [draft, setDraft] = useState<CorrectionDraft>(() => initialDraft(caseId, manifest, caseFrame, selectedTask));
  const currentSourceFrame = caseToSourceFrame(manifest, caseFrame);
  const activeMaterializationJobs = Object.values(materializationJobs).filter(({ job }) => job.case_id === caseId && (job.status === 'queued' || job.status === 'running'));
  const pendingCorrectionRegions = useMemo(
    () => pendingRegionsForSourceFrame(corrections, manualTasks, caseId, currentSourceFrame),
    [caseId, corrections, currentSourceFrame, manualTasks],
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
  const openComplete = tasks.length > 0 && tasks.every((task) => task.bucket !== 'Open');

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
    if (!roles.some((role) => role.role === selectedRole)) {
      setSelectedRole(defaultRole(roles));
    }
  }, [roles, selectedRole, setSelectedRole]);

  useEffect(() => {
    if (!selectedTaskId && tasks[0]) setSelectedTaskId(tasks[0].review_task_id);
  }, [selectedTaskId, setSelectedTaskId, tasks]);

  useEffect(() => {
    setDraft((current) => withDraftHash({ ...current, selectedTaskId: selectedTask?.review_task_id }));
  }, [selectedTask?.review_task_id]);

  useEffect(() => {
    if (mode === 'edit') controller.pause();
  }, [controller, mode]);

  useEffect(() => {
    if (mode !== 'edit') return;
    controller.pause();
    setDraft((current) => {
      if (current.caseId === caseId && current.keyframes.length) return current;
      return initialDraft(caseId, manifest, caseFrame, selectedTask);
    });
  }, [caseFrame, caseId, controller, manifest, mode, selectedTask]);

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
          );
          if (jobMatchesCurrentProjection && tracked.job.working_state) {
            onWorkingStateChangeRef.current(tracked.job.working_state);
          }
          if (jobMatchesCurrentProjection && tracked.job.latest_reviewed && tracked.job.case_id === caseId) {
            setLatestRole(tracked.job.latest_reviewed);
            setSelectedRoleRef.current('latest_reviewed');
            seekToFrameRef.current(tracked.preserveFrame);
            setCorrectionSaveStatus('Published');
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
          setCorrectionSaveStatus('Materialization failed');
        }
      }
    };
    void poll();
    const interval = window.setInterval(() => void poll(), 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeMaterializationJobKey, caseId]);

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

  const selectTask = (task: TaskWithDecision) => {
    setFocusTarget(null);
    setBucket(task.bucket);
    setSelectedTaskId(task.review_task_id);
    controller.seekToSourceFrame(task.source_frame_start);
  };

  const updateDraft = (nextDraft: CorrectionDraft) => {
    setPreviewReady(false);
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setCorrectionSaveStatus(undefined);
    if (selectedRole === 'correction_preview') setSelectedRole('latest_reviewed');
    setDraft(withDraftHash({ ...nextDraft, preview_token: undefined }));
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
    onDecisionsChange((state.task_decisions || {}) as Record<string, TaskDecision>);
    onManualTasksChange((Array.isArray(state.active_manual_tasks) ? state.active_manual_tasks : []) as Task[]);
    onEntityOverridesChange((state.object_policy_overrides || {}) as Record<string, EffectiveEntity>);
    onCorrectionsChange((Array.isArray(state.manual_corrections) ? state.manual_corrections : []) as Record<string, unknown>[]);
    setLatestRole((state.materialized_latest_reviewed || {})[caseId] as MediaVersion | undefined);
  };

  const persistDecision = async (task: TaskWithDecision, action: string) => {
    if (!session) {
      onError('Start review before saving decisions.');
      return;
    }
    setPendingTaskId(task.review_task_id);
    onError(undefined);
    try {
      const { decision, event } = await saveDecision({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        task_id: task.review_task_id,
        action,
        note: action === ACTIONS.looksCorrect ? 'Correct' : action === ACTIONS.unsure ? 'Unsure' : 'Needs correction',
      });
      const nextDecisions = { ...decisions, [task.review_task_id]: decision };
      onDecisionsChange(nextDecisions);
      appendEvents([event]);
      const nextOpenTask = tasksForCase(allTasks, nextDecisions, caseId)
        .filter((candidate) => candidate.bucket === 'Open')
        .sort((left, right) => left.source_frame_start - right.source_frame_start)
        .find((candidate) => candidate.source_frame_start >= task.source_frame_start)
        || tasksForCase(allTasks, nextDecisions, caseId)
          .filter((candidate) => candidate.bucket === 'Open')
          .sort((left, right) => left.source_frame_start - right.source_frame_start)[0];
      if (nextOpenTask) {
        setBucket('Open');
        setSelectedTaskId(nextOpenTask.review_task_id);
        controller.seekToSourceFrame(nextOpenTask.source_frame_start);
      } else {
        setBucket('Open');
        setSelectedTaskId(task.review_task_id);
      }
    } catch (err) {
      onError(`Decision was not saved: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingTaskId(undefined);
    }
  };

  const updatePolicy = async (entity: EffectiveEntity, policy: EntityPolicy) => {
    if (!session) {
      onError('Start review before changing object policy.');
      return;
    }
    const verificationTaskId = withPolicyTaskId(caseId, entity.entity_id, policy);
    const after: EffectiveEntity = {
      ...entity,
      current_policy: policy,
      default_privacy_policy: policy,
      privacy_action: policy,
      verification_task_id: verificationTaskId || undefined,
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
      const nextTask = result.verification_task || (verificationTaskId ? policyVerificationTask(caseId, result.entity || after, verificationTaskId) : null);
      if (nextTask) {
        onManualTasksChange([
          ...manualTasks.filter((task) => task.review_task_id !== nextTask.review_task_id),
          nextTask,
        ]);
      }
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
    setPreviewToken(undefined);
    setPreviewRole(undefined);
    setCorrectionSaveStatus(materialization_required ? 'Saved - updating Latest reviewed' : `Saved - ${publication_status || 'published'}`);
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

  const restartReviewState = async () => {
    if (!session) {
      onError('Start review before restarting review.');
      return;
    }
    setCheckpointBusy('restart');
    onError(undefined);
    try {
      const result = await restartReviewWorkflow({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
      });
      appendEvents([result.event]);
      applyWorkingStateProjection(result.working_state);
      setBucket('Open');
      setSelectedTaskId(null);
      setFocusTarget(null);
      setRestartConfirmOpen(false);
    } catch (err) {
      onError(`Review was not restarted: ${err instanceof Error ? err.message : String(err)}`);
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
      setPreviewReady(false);
      setPreviewToken(undefined);
      setPreviewRole(undefined);
      setDraft(initialDraft(caseId, manifest, caseFrame, null));
      setSelectedRole('latest_reviewed');
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
          {materializationError ? (
            <p className="error-text">
              {materializationError}{' '}
              <button type="button" onClick={() => void retryLatestMaterialization()}>
                Retry
              </button>
            </p>
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
          <button type="button" onClick={() => setRestartConfirmOpen(true)} disabled={checkpointBusy === 'restart'}>
            {checkpointBusy === 'restart' ? 'Restarting...' : 'Restart review'}
          </button>
          <button type="button" className="danger" onClick={() => setResetEditsConfirmOpen(true)} disabled={checkpointBusy === 'reset-edits'}>
            {checkpointBusy === 'reset-edits' ? 'Resetting...' : 'Reset all edits to Automatic baseline'}
          </button>
        </div>
      ) : null}
      <div className="workspace-layout">
        <div className="review-main">
          <ReviewVideoPlayer
            controller={controller}
            manifest={manifest}
            selectedTaskId={selectedTaskId}
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
            {mode !== 'edit' && selectedVersion?.role === 'latest_reviewed' ? (
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
            openComplete={openComplete}
            onBucketChange={setBucket}
            onDecision={persistDecision}
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
      {restartConfirmOpen ? (
        <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="restart-case-title">
          <div className="modal-card">
            <h2 id="restart-case-title">Restart {manifest.display_name} review?</h2>
            <p className="muted">
              Clear this case's task decisions, review progress counters, selected task, and full-video sweep acknowledgement. Entity edits, merges, manual redactions, checkpoints, and Latest reviewed are preserved.
            </p>
            <div className="button-row">
              <button type="button" className="primary" onClick={() => void restartReviewState()} disabled={checkpointBusy === 'restart'}>
                {checkpointBusy === 'restart' ? 'Restarting...' : 'Restart review'}
              </button>
              <button type="button" onClick={() => setRestartConfirmOpen(false)} disabled={checkpointBusy === 'restart'}>
                Cancel
              </button>
            </div>
          </div>
        </section>
      ) : null}
      {resetEditsConfirmOpen ? (
        <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="reset-edits-title">
          <div className="modal-card">
            <h2 id="reset-edits-title">Reset all edits to Automatic baseline?</h2>
            <p className="muted">
              Return this case's active decisions, object policies, merges, corrections, and full-video sweep state to the Automatic baseline. History and checkpoints are preserved.
            </p>
            <div className="button-row">
              <button type="button" className="danger" onClick={() => void resetReviewState()} disabled={checkpointBusy === 'reset-edits'}>
                {checkpointBusy === 'reset-edits' ? 'Resetting...' : 'Reset all edits to Automatic baseline'}
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
  const latest = latestRole || byRole.get('latest_reviewed') || automatic;
  return [
    byRole.get('original'),
    automatic ? { ...automatic, label: 'Automatic', role: 'automatic' as const } : undefined,
    latest ? { ...latest, label: 'Latest reviewed', role: 'latest_reviewed' as const } : undefined,
  ].filter((role): role is MediaVersion => Boolean(role));
}

function materializationJobMatchesCurrentProjection(
  job: MaterializationJob,
  currentState?: WorkingState,
): boolean {
  const jobRevision = job.published_working_state_revision || job.working_state?.active_projection_hash || job.accepted_working_state_revision;
  const currentRevision = currentState?.active_projection_hash;
  if (!jobRevision || !currentRevision) return true;
  return jobRevision === currentRevision;
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
): PendingCorrectionRegion[] {
  const pendingIds = new Set(
    manualTasks
      .filter((task) => task.case_id === caseId)
      .filter((task) => task.normal_task_class === 'correction_verification')
      .filter((task) => task.actionable === false && task.publication_status !== 'published')
      .map((task) => String(task.correction_id || correctionIdFromTask(task)))
      .filter(Boolean),
  );
  if (!pendingIds.size) return [];
  return corrections
    .filter((correction) => String(correction.case_id || '') === caseId)
    .filter((correction) => pendingIds.has(String(correction.correction_id || '')))
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
  const sourceFrame = caseToSourceFrame(manifest, caseFrame);
  return withDraftHash({
    caseId,
    case_id: caseId,
    category: task?.category || 'license_plate',
    start: sourceFrame,
    end: sourceFrame,
    start_source_frame: sourceFrame,
    end_source_frame: sourceFrame,
    mode: 'single_frame',
    keyframes: [],
    barriers: [],
    checkpoints: [],
    interpolation_method: 'linear',
    selectedTaskId: task?.review_task_id,
  });
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
