import { useEffect, useMemo, useState } from 'react';
import {
  mergeEntities,
  saveCorrection as saveCorrectionRequest,
  saveDecision,
  saveEntityPolicy,
} from '../api/client';
import { CorrectionEditor } from '../components/CorrectionEditor';
import { EntityMergeDialog } from '../components/EntityMergeDialog';
import { EntityPanel } from '../components/EntityPanel';
import { ReviewQueue } from '../components/ReviewQueue';
import { ReviewTimeline } from '../components/ReviewTimeline';
import { ReviewVideoPlayer } from '../components/ReviewVideoPlayer';
import { usePersistentState } from '../hooks/usePersistentState';
import { useVideoController } from '../hooks/useVideoController';
import {
  effectiveEntities,
  policyVerificationTask,
  tasksForCase,
  withPolicyTaskId,
} from '../state/reviewState';
import type {
  Attribution,
  AuditEvent,
  Correction,
  Entity,
  MediaVersion,
  Session,
  Task,
  TaskDecision,
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
  workspace: WorkspaceSummary;
  onCorrectionsChange: (corrections: Record<string, unknown>[]) => void;
  onDecisionsChange: (decisions: Record<string, TaskDecision>) => void;
  onEntityOverridesChange: (overrides: Record<string, EffectiveEntity>) => void;
  onError: (message?: string) => void;
  onEventsChange: (events: AuditEvent[]) => void;
  onManualTasksChange: (tasks: Task[]) => void;
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
  session,
  workspace,
}: Props) {
  const manifest = workspace.cases[caseId];
  const baseTasks = workspace.tasks.tasks || [];
  const [mode, setMode] = usePersistentState<ReviewMode>(`${caseId}:mode`, 'review');
  const [caseFrame, setCaseFrame] = usePersistentState<number>(`${caseId}:frame`, 0);
  const [selectedRole, setSelectedRole] = usePersistentState<string>(`${caseId}:version`, defaultRole(manifest.video_roles));
  const [selectedTaskId, setSelectedTaskId] = usePersistentState<string | null>(`${caseId}:task`, null);
  const [bucket, setBucket] = usePersistentState<QueueBucket>(`${caseId}:bucket`, 'Open');
  const [entityFilter, setEntityFilter] = usePersistentState<EntityFilter>(`${caseId}:entity_filter`, 'all');
  const [entitiesOpen, setEntitiesOpen] = useState(false);
  const [mergeSource, setMergeSource] = useState<EffectiveEntity | null>(null);
  const [pendingTaskId, setPendingTaskId] = useState<string>();
  const [pendingEntityId, setPendingEntityId] = useState<string>();
  const [latestRole, setLatestRole] = useState<MediaVersion | undefined>();

  const allTasks = useMemo(
    () => [...baseTasks, ...manualTasks].filter((task) => task.case_id === caseId),
    [baseTasks, caseId, manualTasks],
  );
  const tasks = useMemo(() => tasksForCase(allTasks, decisions, caseId), [allTasks, caseId, decisions]);
  const selectedTask = tasks.find((task) => task.review_task_id === selectedTaskId) || tasks[0] || null;
  const effectiveCaseEntities = useMemo(
    () => effectiveEntities(workspace.entities.cases[caseId] || [], entityOverrides),
    [caseId, entityOverrides, workspace.entities.cases],
  );

  const roles = useMemo(() => {
    const nextRoles = latestRole
      ? manifest.video_roles.filter((role) => role.role !== 'latest_reviewed').concat(latestRole)
      : manifest.video_roles;
    return nextRoles;
  }, [latestRole, manifest.video_roles]);
  const selectedVersion = roles.find((role) => role.role === selectedRole) || roles.find((role) => role.role === defaultRole(roles));
  const controller = useVideoController(manifest, selectedVersion, caseFrame, setCaseFrame);
  const [draft, setDraft] = useState<CorrectionDraft>(() => initialDraft(caseId, manifest, caseFrame, selectedTask));

  useEffect(() => {
    if (!selectedTaskId && tasks[0]) setSelectedTaskId(tasks[0].review_task_id);
  }, [selectedTaskId, setSelectedTaskId, tasks]);

  useEffect(() => {
    setDraft((current) => ({
      ...current,
      start: caseToSourceFrame(manifest, caseFrame),
      end: caseToSourceFrame(manifest, caseFrame),
      selectedTaskId: selectedTask?.review_task_id,
    }));
  }, [caseFrame, manifest, selectedTask?.review_task_id]);

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
    setSelectedTaskId(task.review_task_id);
    controller.seekToSourceFrame(task.source_frame_start);
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
        note: action === 'looks_correct' ? 'Correct' : action === 'uncertain_escalate' ? 'Unsure' : 'Needs correction',
      });
      onDecisionsChange({ ...decisions, [task.review_task_id]: decision });
      onEventsChange([...events, event]);
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
    setPendingEntityId(entity.entity_id);
    onError(undefined);
    try {
      const event = await saveEntityPolicy({
        session_id: session.session_id,
        actor: attribution || { display_name: null },
        case_id: caseId,
        entity,
        after,
        policy,
      });
      onEventsChange([...events, event]);
      onEntityOverridesChange({ ...entityOverrides, [entity.entity_id]: after });
      if (verificationTaskId) {
        onManualTasksChange([
          ...manualTasks.filter((task) => task.review_task_id !== verificationTaskId),
          policyVerificationTask(caseId, after, verificationTaskId),
        ]);
      }
    } catch (err) {
      onError(`Entity action was not saved: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPendingEntityId(undefined);
    }
  };

  const merge = async (source: EffectiveEntity, target: EffectiveEntity) => {
    if (!session) {
      onError('Start review before merging objects.');
      return;
    }
    const merged: EffectiveEntity = {
      ...source,
      member_detector_tracker_ids: unique([
        ...(source.member_detector_tracker_ids || []),
        ...(target.member_detector_tracker_ids || []),
      ]),
      automatic_source_track_ids: unique([
        ...(source.automatic_source_track_ids || []),
        ...(target.automatic_source_track_ids || []),
      ]),
      merge_split_lineage: [
        ...(source.merge_split_lineage || []),
        {
          operation: 'merge',
          merged_entity_id: target.entity_id,
          source_entity_ids: [source.entity_id, target.entity_id],
        },
      ],
    };
    const retired: EffectiveEntity = { ...target, current_status: 'merged' };
    const event = await mergeEntities({
      session_id: session.session_id,
      actor: attribution || { display_name: null },
      case_id: caseId,
      before_state: { source, target },
      after_state: {
        primary: merged,
        retired_entity_ids: [target.entity_id],
        source_entity_ids: [source.entity_id, target.entity_id],
      },
    });
    onEventsChange([...events, event]);
    onEntityOverridesChange({ ...entityOverrides, [source.entity_id]: merged, [target.entity_id]: retired });
    setMergeSource(null);
  };

  const saveCorrection = async (correction: Correction, reason: string) => {
    if (!session) {
      onError('Start review before saving corrections.');
      return;
    }
    const { event, preview } = await saveCorrectionRequest({
      session_id: session.session_id,
      actor: attribution || { display_name: null },
      correction,
      reason,
    });
    onEventsChange([...events, event]);
    onCorrectionsChange([...corrections, event.after_state || correction]);
    if (preview?.latest_reviewed || preview?.corrected_preview) {
      const path = preview.latest_reviewed || preview.corrected_preview;
      if (path) {
        setLatestRole({
          label: 'Latest reviewed',
          path,
          role: 'latest_reviewed',
          sha256: null,
          source_frame_start: manifest.source_frame_start,
          source_frame_end: manifest.source_frame_end,
        });
        setSelectedRole('latest_reviewed');
      }
    }
  };

  const unresolvedCount = unresolvedEvidenceCount(workspace, caseId);

  return (
    <section className="case-panel active" data-case-id={caseId}>
      <div className="case-header">
        <div>
          <h2>{manifest.display_name}</h2>
          <p className="case-subtitle">
            {manifest.frame_count} frames · source {manifest.source_frame_start}-{manifest.source_frame_end}
          </p>
        </div>
        <div className="segmented" role="group" aria-label="Review mode">
          <button type="button" className={mode === 'review' ? 'active' : ''} onClick={() => setMode('review')}>
            Review
          </button>
          <button type="button" className={mode === 'edit' ? 'active' : ''} onClick={() => setMode('edit')}>
            Edit
          </button>
        </div>
      </div>
      <div className="workspace-layout">
        <div className="review-main">
          <ReviewVideoPlayer
            controller={controller}
            manifest={manifest}
            roles={roles}
            selectedRole={selectedRole}
            version={selectedVersion}
            onVersionChange={setSelectedRole}
          />
          <ReviewTimeline
            currentFrame={caseFrame}
            manifest={manifest}
            selectedTaskId={selectedTaskId}
            tasks={tasks}
            onSelectTask={selectTask}
          />
        </div>
        {mode === 'review' ? (
          <ReviewQueue
            bucket={bucket}
            manifest={manifest}
            pendingTaskId={pendingTaskId}
            selectedTaskId={selectedTaskId}
            tasks={tasks}
            onBucketChange={setBucket}
            onDecision={persistDecision}
            onOpenEntities={() => setEntitiesOpen(true)}
            onSelectTask={selectTask}
          />
        ) : (
          <aside className="side-panel">
            <CorrectionEditor
              actor={attribution || { display_name: null }}
              caseId={caseId}
              configurationFreezeHash={workspace.configuration_freeze_hash}
              draft={draft}
              manifest={manifest}
              selectedTask={selectedTask}
              sourceVideoHash={manifest.source_video_hash}
              onDiscard={() => setDraft(initialDraft(caseId, manifest, caseFrame, selectedTask))}
              onDraftChange={setDraft}
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
        pendingEntityId={pendingEntityId}
        unresolvedCount={unresolvedCount}
        onClose={() => setEntitiesOpen(false)}
        onFilterChange={setEntityFilter}
        onJump={(entity) => {
          setEntitiesOpen(false);
          if (typeof entity.representative_source_frame === 'number') {
            controller.seekToSourceFrame(entity.representative_source_frame);
          }
        }}
        onMerge={setMergeSource}
        onPolicy={updatePolicy}
      />
      <EntityMergeDialog
        entities={effectiveCaseEntities}
        open={Boolean(mergeSource)}
        source={mergeSource}
        onCancel={() => setMergeSource(null)}
        onMerge={merge}
      />
    </section>
  );
}

function defaultRole(roles: MediaVersion[]): string {
  if (roles.some((role) => role.role === 'latest_reviewed')) return 'latest_reviewed';
  if (roles.some((role) => role.role === 'automatic')) return 'automatic';
  return 'original';
}

function initialDraft(caseId: CaseId, manifest: WorkspaceSummary['cases'][string], caseFrame: number, task?: Task | null): CorrectionDraft {
  const sourceFrame = caseToSourceFrame(manifest, caseFrame);
  return {
    caseId,
    start: sourceFrame,
    end: sourceFrame,
    keyframes: [],
    checkpoints: [],
    interpolation_method: 'linear',
    selectedTaskId: task?.review_task_id,
  };
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

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function unresolvedEvidenceCount(workspace: WorkspaceSummary, caseId: CaseId): number {
  const byCase = workspace.entities.unresolved_evidence_by_case as Record<string, { fragment_count?: number } | undefined> | undefined;
  return Number(byCase?.[caseId]?.fragment_count || 0);
}
