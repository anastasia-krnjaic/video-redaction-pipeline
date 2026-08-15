import type { Entity, Task, TaskDecision, WorkingState } from '../types/api';
import type { CaseId, DecisionsByCase, EffectiveEntity, EntityFilter, EntityPolicy, QueueBucket, ReviewGroup, TaskResolutionsByCase, TaskWithDecision } from '../types/domain';

export const ACTIONS = {
  looksCorrect: 'looks_correct',
  needsCorrection: 'privacy_leak',
  notSensitive: 'not_sensitive',
} as const;

const NEEDS_CORRECTION_ACTIONS = new Set<string>([ACTIONS.needsCorrection]);

const ACTION_ORDER = [
  ACTIONS.looksCorrect,
  ACTIONS.needsCorrection,
  ACTIONS.notSensitive,
] as const;

export interface DecisionActionOption {
  action: string;
  className: string;
  label: string;
}

export interface TaskGuidance {
  instruction: string;
  queryClass: string;
  title: string;
}

export function taskBucket(task: Task, decision?: TaskDecision, resolution?: TaskDecision): QueueBucket {
  const effective = resolution || decision;
  if (task.decision_state === 'resolved' && !effective) return 'Correct';
  if (!effective) return 'Open';
  if (NEEDS_CORRECTION_ACTIONS.has(effective.action)) return 'Needs correction';
  if (effective.action === ACTIONS.notSensitive) return 'Not sensitive';
  return 'Correct';
}

export function decisionForTask(decisions: DecisionsByCase, caseId: CaseId, taskId: string): TaskDecision | undefined {
  return decisions[caseId]?.[taskId];
}

export function withTaskDecision(
  decisions: DecisionsByCase,
  caseId: CaseId,
  taskId: string,
  decision: TaskDecision,
): DecisionsByCase {
  return {
    ...decisions,
    [caseId]: {
      ...(decisions[caseId] || {}),
      [taskId]: decision,
    },
  };
}

export function withoutTaskResolution(
  resolutions: TaskResolutionsByCase,
  caseId: CaseId,
  taskId: string,
): TaskResolutionsByCase {
  const nextCase = { ...(resolutions[caseId] || {}) };
  delete nextCase[taskId];
  const next = { ...resolutions };
  if (Object.keys(nextCase).length) next[caseId] = nextCase;
  else delete next[caseId];
  return next;
}

export function normalizeDecisionsByCase(
  byCase?: Record<string, Record<string, TaskDecision>>,
  flat?: Record<string, TaskDecision>,
): DecisionsByCase {
  const next: DecisionsByCase = {};
  for (const [caseId, rows] of Object.entries(byCase || {})) {
    next[caseId] = { ...(rows || {}) };
  }
  for (const [taskId, decision] of Object.entries(flat || {})) {
    const caseId = String(decision.case_id || '');
    if (!caseId || next[caseId]?.[taskId]) continue;
    next[caseId] = { ...(next[caseId] || {}), [String(decision.task_id || taskId)]: decision };
  }
  return next;
}

export const normalizeTaskResolutionsByCase = normalizeDecisionsByCase;

export function tasksForCase(
  tasks: Task[],
  decisions: DecisionsByCase,
  caseId: CaseId,
  resolutions: TaskResolutionsByCase = {},
): TaskWithDecision[] {
  return tasks
    .filter(
      (task) =>
        task.case_id === caseId &&
        !task.technical_only &&
        !durableVerificationTask(task),
    )
    .map((task) => {
      const decision = decisionForTask(decisions, caseId, task.review_task_id);
      const resolution = decisionForTask(resolutions, caseId, task.review_task_id);
      return { ...task, decision, resolution, bucket: taskBucket(task, decision, resolution) };
    });
}

function durableVerificationTask(task: Task): boolean {
  return (
    task.normal_task_class === 'correction_verification' ||
    task.review_task_id.startsWith('correction_verify_') ||
    task.review_task_id.startsWith('object_policy_verify_')
  );
}

export function bucketCounts(tasks: TaskWithDecision[]): Record<QueueBucket, number> {
  return {
    Open: tasks.filter((task) => task.bucket === 'Open').length,
    'Needs correction': tasks.filter((task) => task.bucket === 'Needs correction').length,
    Correct: tasks.filter((task) => task.bucket === 'Correct').length,
    'Not sensitive': tasks.filter((task) => task.bucket === 'Not sensitive').length,
  };
}

export function reviewGroupsForCase(tasks: TaskWithDecision[], includeOptional = false): ReviewGroup[] {
  const buckets = new Map<string, TaskWithDecision[]>();
  for (const task of tasks) {
    if (Boolean(task.optional) !== includeOptional) continue;
    const key = reviewGroupKey(task);
    buckets.set(key, [...(buckets.get(key) || []), task]);
  }
  return [...buckets.entries()]
    .map(([groupId, childTasks]) => buildReviewGroup(groupId, childTasks))
    .sort((left, right) => {
      if (left.source_frame_start !== right.source_frame_start) return left.source_frame_start - right.source_frame_start;
      return left.review_task_id.localeCompare(right.review_task_id);
    });
}

export function requiredReviewGroups(tasks: TaskWithDecision[]): ReviewGroup[] {
  return reviewGroupsForCase(tasks, false);
}

export function optionalReviewGroups(tasks: TaskWithDecision[]): ReviewGroup[] {
  return reviewGroupsForCase(tasks, true);
}

function buildReviewGroup(groupId: string, children: TaskWithDecision[]): ReviewGroup {
  const childTasks = [...children].sort((left, right) => {
    if (left.source_frame_start !== right.source_frame_start) return left.source_frame_start - right.source_frame_start;
    return left.review_task_id.localeCompare(right.review_task_id);
  });
  const first = childTasks[0];
  const bucket = reviewGroupBucket(childTasks);
  const childIds = childTasks.map((task) => task.review_task_id);
  const guidance = reviewGroupGuidance(childTasks);
  const decision = sharedDecision(childTasks);
  return {
    ...first,
    allowed_actions: commonAllowedActions(childTasks),
    bucket,
    child_task_ids: childIds,
    child_tasks: childTasks,
    correction_required: childTasks.some((task) => task.correction_required),
    decision,
    group_id: groupId,
    is_review_group: true,
    optional: childTasks.every((task) => task.optional),
    optional_group: childTasks.every((task) => task.optional),
    reason_chips: reviewGroupReasonChips(childTasks),
    review_task_id: groupId,
    segment_count: childTasks.length,
    short_instruction: guidance.instruction,
    short_title: guidance.title,
    source_frame_end: Math.max(...childTasks.map((task) => task.source_frame_end)),
    source_frame_start: Math.min(...childTasks.map((task) => task.source_frame_start)),
    target_entity_ids: commonTargetEntityIds(childTasks),
    title: guidance.title,
    what_to_inspect: guidance.instruction,
  } as ReviewGroup;
}

function reviewGroupKey(task: TaskWithDecision): string {
  const binding = isRecord(task.target_binding) ? task.target_binding : {};
  const allowed = (task.allowed_actions || []).join(',');
  const sourceTaskId = String(binding.source_task_id || '');
  const targetEntityIds = (Array.isArray(task.target_entity_ids) ? task.target_entity_ids : [])
    .map(String)
    .sort();
  const groupClass = groupableClass(task);
  if (sourceTaskId) {
    return [
      'source',
      task.case_id,
      task.category,
      sourceTaskId,
      groupClass,
      allowed,
      Boolean(task.correction_required),
    ].join(':');
  }
  if (targetEntityIds.length === 1 && sameEntityCompositeClass(task)) {
    return [
      'entity',
      task.case_id,
      task.category,
      targetEntityIds[0],
      'entity_review',
      allowed,
      Boolean(task.correction_required),
    ].join(':');
  }
  return ['task', task.case_id, task.review_task_id].join(':');
}

function groupableClass(task: TaskWithDecision): string {
  const queryClass = validQueryPresentationClass(task.query_presentation_class) || guidanceFamily(task);
  if (sameEntityCompositeClass(task)) return 'entity_review';
  return queryClass;
}

function sameEntityCompositeClass(task: TaskWithDecision): boolean {
  const queryClass = validQueryPresentationClass(task.query_presentation_class) || guidanceFamily(task);
  return (
    task.category === 'license_plate' &&
    ['PLATE_COVERAGE_GAP', 'TRACKING_UNCERTAIN', 'POSSIBLE_FALSE_POSITIVE'].includes(queryClass)
  );
}

function reviewGroupBucket(children: TaskWithDecision[]): QueueBucket {
  if (children.some((task) => task.bucket === 'Needs correction')) return 'Needs correction';
  if (children.some((task) => task.bucket === 'Open')) return 'Open';
  const buckets = new Set(children.map((task) => task.bucket));
  if (buckets.size === 1) return children[0].bucket;
  return 'Correct';
}

function sharedDecision(children: TaskWithDecision[]): TaskDecision | undefined {
  const decisions = children.map((task) => task.resolution || task.decision).filter(Boolean) as TaskDecision[];
  if (decisions.length !== children.length) return undefined;
  const actions = new Set(decisions.map((decision) => decision.action));
  return actions.size === 1 ? decisions[0] : undefined;
}

function commonAllowedActions(children: TaskWithDecision[]): string[] {
  const [first, ...rest] = children.map((task) => new Set(task.allowed_actions || []));
  if (!first) return [];
  return [...first].filter((action) => rest.every((allowed) => allowed.has(action)));
}

function commonTargetEntityIds(children: TaskWithDecision[]): string[] {
  const targets = children.map((task) =>
    (Array.isArray(task.target_entity_ids) ? task.target_entity_ids : []).map(String).sort(),
  );
  if (!targets.length) return [];
  const first = targets[0].join('|');
  return targets.every((target) => target.join('|') === first) ? targets[0] : [];
}

function reviewGroupGuidance(children: TaskWithDecision[]): TaskGuidance {
  const first = children[0];
  const chips = reviewGroupReasonChips(children);
  if (children.length > 1 && first.category === 'license_plate' && chips.length > 1) {
    return {
      instruction: 'Check tracking and coverage across the listed segments.',
      queryClass: 'OTHER_REVIEW',
      title: 'Plate needs review',
    };
  }
  const base = taskGuidance(first);
  if (children.length === 1) return base;
  return {
    ...base,
    instruction: `${base.instruction} Review all ${children.length} listed segments.`,
  };
}

function reviewGroupReasonChips(children: TaskWithDecision[]): string[] {
  const labels = new Set<string>();
  for (const child of children) {
    const queryClass = validQueryPresentationClass(child.query_presentation_class) || guidanceFamily(child);
    if (queryClass === 'TRACKING_UNCERTAIN') labels.add('Tracking');
    else if (queryClass === 'PLATE_COVERAGE_GAP' || queryClass === 'FACE_COVERAGE_GAP') labels.add('Coverage');
    else if (queryClass === 'IDENTITY_UNCERTAIN') labels.add('Identity');
    else if (queryClass === 'INTERPOLATION_OR_SCALE_RISK') labels.add('Drift');
    else if (queryClass === 'POSSIBLE_FALSE_POSITIVE') labels.add('False positive');
    else if (queryClass === 'OPTIONAL_SPOT_CHECK') labels.add('Optional QA');
  }
  return [...labels];
}

export function decisionLabel(action?: string): string {
  if (!action) return 'Guidance';
  if (action === ACTIONS.looksCorrect) return 'Correct';
  if (action === ACTIONS.notSensitive) return 'Not sensitive';
  if (NEEDS_CORRECTION_ACTIONS.has(action)) return 'Needs correction';
  return action;
}

export function taskDecisionActionOptions(task: Pick<Task, 'allowed_actions'>): DecisionActionOption[] {
  const allowed = new Set(task.allowed_actions || []);
  return ACTION_ORDER
    .filter((action) => allowed.has(action))
    .map((action) => ({
      action,
      className: decisionActionClass(action),
      label: decisionLabel(action),
    }));
}

export function decisionActionClass(action: string): string {
  if (action === ACTIONS.looksCorrect) {
    return 'decision-correct';
  }
  if (action === ACTIONS.notSensitive) return 'decision-not-sensitive';
  return 'decision-needs-correction';
}

export function taskColorClass(task: TaskWithDecision): string {
  if (task.bucket === 'Correct') return 'marker-correct';
  if (task.bucket === 'Not sensitive') return 'marker-not-sensitive';
  if (task.bucket === 'Needs correction') return 'marker-needs-correction';
  if (task.normal_task_class === 'possible_false_redaction') return 'marker-false';
  if (task.category === 'face') return 'marker-face';
  return 'marker-plate';
}

export function taskGuidance(
  task: Pick<
    Task,
    | 'category'
    | 'issue_type'
    | 'normal_task_class'
    | 'query_presentation_class'
    | 'review_task_id'
    | 'short_instruction'
    | 'short_title'
    | 'title'
    | 'what_happened'
    | 'what_to_inspect'
  > & Record<string, unknown>,
): TaskGuidance {
  const queryClass = validQueryPresentationClass(task.query_presentation_class) || guidanceFamily(task);
  const template = QUERY_PRESENTATION_TEMPLATES[queryClass] || QUERY_PRESENTATION_TEMPLATES.OTHER_REVIEW;
  return {
    instruction: task.short_instruction || template.instruction,
    queryClass,
    title: task.short_title || template.title,
  };
}

export const QUERY_PRESENTATION_TEMPLATES: Record<string, { instruction: string; title: string }> = {
  FACE_COVERAGE_GAP: {
    instruction: 'Check that the face is fully covered in this interval.',
    title: 'Face may be visible',
  },
  PLATE_COVERAGE_GAP: {
    instruction: 'Check that the plate is fully covered in this interval.',
    title: 'Plate may be visible',
  },
  TRACKING_UNCERTAIN: {
    instruction: 'Check that the redaction stays on the intended target.',
    title: 'Tracking needs review',
  },
  IDENTITY_UNCERTAIN: {
    instruction: 'Check that the redaction follows the intended person or plate.',
    title: 'Wrong target possible',
  },
  INTERPOLATION_OR_SCALE_RISK: {
    instruction: 'Check that the redaction follows the target through this interval.',
    title: 'Redaction may drift',
  },
  POSSIBLE_FALSE_POSITIVE: {
    instruction: 'Check whether the marked content actually needs redaction.',
    title: 'Possible false positive',
  },
  VERIFY_SAVED_CORRECTION: {
    instruction: 'Check that the saved correction covers the intended target.',
    title: 'Verify saved correction',
  },
  OPTIONAL_SPOT_CHECK: {
    instruction: 'Confirm the current redaction is acceptable.',
    title: 'Quick spot check',
  },
  OTHER_REVIEW: {
    instruction: 'Compare Original and the current result for this interval.',
    title: 'Review this interval',
  },
};

function validQueryPresentationClass(value?: string): string | undefined {
  return value && QUERY_PRESENTATION_TEMPLATES[value] ? value : undefined;
}

export function guidanceFamily(task: Pick<Task, 'category' | 'issue_type' | 'normal_task_class'> & Record<string, unknown>): string {
  const semanticText = taskSemanticValues(task).join(' ').toLowerCase();
  const reasonCodes = taskReasonCodes(task);
  if (isCorrectionVerificationTask(task, semanticText)) return 'VERIFY_SAVED_CORRECTION';
  if (
    includesAny(semanticText, 'optional_spot_check', 'spot_check', 'spot check') ||
    reasonIncludesAny(reasonCodes, 'spot_check', 'spot check', 'audit_sample')
  ) {
    return 'OPTIONAL_SPOT_CHECK';
  }
  if (
    includesAny(semanticText, 'false_positive', 'false_redaction', 'possible_false_redaction') ||
    reasonIncludesAny(reasonCodes, 'false_positive', 'false_redaction', 'possible_false')
  ) {
    return 'POSSIBLE_FALSE_POSITIVE';
  }
  if (includesAny(semanticText, 'plate_coverage_gap')) return 'PLATE_COVERAGE_GAP';
  if (includesAny(semanticText, 'face_coverage_gap')) return 'FACE_COVERAGE_GAP';
  if (
    includesAny(semanticText, 'competition', 'duplicate', 'identity') ||
    reasonIncludesAny(reasonCodes, 'competition', 'duplicate', 'identity')
  ) {
    return 'IDENTITY_UNCERTAIN';
  }
  if (
    includesAny(semanticText, 'drift', 'interpolation', 'long_detection_gap', 'one-sided', 'one_sided', 'scale change', 'scale-change') ||
    reasonIncludesAny(
      reasonCodes,
      'anchor_center_motion',
      'anchor_scale_change',
      'drift',
      'gap_exceeds',
      'interpolation',
      'long_detection_gap',
      'one_sided_recovery',
      'scale',
      'scene_cut_inside_gap',
    )
  ) {
    return 'INTERPOLATION_OR_SCALE_RISK';
  }
  if (
    includesAny(semanticText, 'plate_track_review_required', 'requires_review', 'track_review') ||
    reasonIncludesAny(
      reasonCodes,
      'insufficient_detector_observations',
      'insufficient_observations',
      'insufficient_observed',
      'insufficient_strong',
      'internal_gap_penalty',
      'low_confidence',
      'low_mean_confidence',
      'low_observation_density',
      'observed_hits',
      'quality_status',
      'weak_only',
    )
  ) {
    return 'TRACKING_UNCERTAIN';
  }
  if (
    includesAny(semanticText, 'coverage gap', 'coverage_gap', 'privacy gap', 'privacy_gap') ||
    reasonIncludesAny(reasonCodes, 'missing_box', 'render_validation_failure', 'unresolved_privacy')
  ) {
    return task.category === 'license_plate' ? 'PLATE_COVERAGE_GAP' : 'FACE_COVERAGE_GAP';
  }
  return 'OTHER_REVIEW';
}

function taskSemanticValues(task: Record<string, unknown>): string[] {
  const sourceTask = isRecord(task.source_task) ? task.source_task : {};
  return [
    task.normal_task_class,
    task.issue_type,
    task.task_type,
    sourceTask.category,
    sourceTask.task_type,
  ].filter((value): value is string => typeof value === 'string' && Boolean(value));
}

function taskReasonCodes(task: Record<string, unknown>): string[] {
  const sourceTask = isRecord(task.source_task) ? sourceTaskRecord(task.source_task) : {};
  const codes = new Set<string>();
  for (const payload of [task, sourceTask]) {
    for (const key of ['raw_reason_codes', 'reason_codes', 'reasons']) {
      const values = payload[key];
      if (Array.isArray(values)) {
        values.forEach((value) => {
          if (value) codes.add(String(value).toLowerCase());
        });
      }
    }
  }
  return [...codes];
}

function sourceTaskRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function isCorrectionVerificationTask(task: Record<string, unknown>, semanticText: string): boolean {
  if (includesAny(semanticText, 'corrected_result_check', 'corrected_result_verification', 'correction_verification', 'saved_correction_verification')) {
    return true;
  }
  const hasCorrectionReference = [
    task.accepted_correction_revision_id,
    task.correction_id,
    task.correction_revision_id,
    task.revision_id,
  ].some(Boolean);
  return hasCorrectionReference && includesAny(semanticText, 'corrected', 'correction');
}

function includesAny(text: string, ...needles: string[]): boolean {
  return needles.some((needle) => text.includes(needle));
}

function reasonIncludesAny(reasonCodes: string[], ...needles: string[]): boolean {
  return reasonCodes.some((code) => needles.some((needle) => code.includes(needle)));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function effectiveEntities(
  entities: Entity[],
  overrides: Record<string, EffectiveEntity>,
): EffectiveEntity[] {
  const seen = new Set<string>();
  const caseIds = new Set(entities.map((entity) => String(entity.case_id || '')).filter(Boolean));
  const effective = entities
    .map((entity) => {
      seen.add(entity.entity_id);
      return { ...entity, ...(overrides[entity.entity_id] || {}) };
    })
    .filter((entity) => entity.current_status !== 'merged');
  for (const [entityId, override] of Object.entries(overrides)) {
    if (seen.has(entityId) || override.current_status === 'merged') continue;
    if (!override.case_id) continue;
    if (caseIds.size && !caseIds.has(String(override.case_id))) continue;
    effective.push(override);
  }
  return effective.sort((left, right) => {
    const leftFrame = typeof left.first_visible_source_frame === 'number' ? left.first_visible_source_frame : Number.MAX_SAFE_INTEGER;
    const rightFrame = typeof right.first_visible_source_frame === 'number' ? right.first_visible_source_frame : Number.MAX_SAFE_INTEGER;
    if (leftFrame !== rightFrame) return leftFrame - rightFrame;
    return left.label.localeCompare(right.label);
  });
}

export function filteredEntities(
  entities: EffectiveEntity[],
  filter: EntityFilter,
): EffectiveEntity[] {
  if (filter === 'all') return entities;
  if (filter === 'face' || filter === 'license_plate') {
    return entities.filter((entity) => entity.category === filter);
  }
  if (filter === 'needs_review') {
    return entities.filter((entity) => {
      const policy = entity.current_policy || entity.privacy_action || entity.default_privacy_policy;
      return policy === 'review_required' || entity.review_state === 'needs_review';
    });
  }
  return entities.filter((entity) => entity.review_entity_state === 'unresolved');
}

export function policyLabel(policy?: string, category?: string): string {
  if (policy === 'redact') return 'Redact';
  if (policy === 'keep_visible') return 'Keep visible';
  if (policy === 'not_sensitive') return category === 'face' ? 'Not a person' : 'Not a plate';
  return 'Needs review';
}

export function withPolicyTaskId(caseId: string, entityId: string, policy: EntityPolicy): string | null {
  void caseId;
  void entityId;
  void policy;
  return null;
}

export async function authoritativeWorkingStateAfterSweepApproval(
  result: { working_state?: WorkingState },
  sessionId: string,
  loadWorkingState: (sessionId: string) => Promise<{ working_state: WorkingState }>,
): Promise<WorkingState> {
  if (result.working_state) return result.working_state;
  const refreshed = await loadWorkingState(sessionId);
  return refreshed.working_state;
}

export function policyVerificationTask(caseId: CaseId, entity: EffectiveEntity, taskId: string): Task {
  const frame =
    typeof entity.first_visible_source_frame === 'number' ? entity.first_visible_source_frame : 0;
  return {
    allowed_actions: [ACTIONS.looksCorrect, ACTIONS.needsCorrection, ACTIONS.notSensitive],
    blocking: true,
    case_id: caseId,
    category: entity.category,
    correction_required: false,
    decision_state: 'pending',
    issue_type: 'entity_policy_verification',
    normal_task_class: 'entity_policy_verification',
    raw_signal_ids: [],
    review_task_id: taskId,
    schema_version: 'case_reviewer_task.v1',
    severity: 'blocker',
    source_frame_start: frame,
    source_frame_end: frame,
    technical_only: false,
    title: `Verify ${policyLabel(entity.current_policy, entity.category)} for ${entity.label}`,
    what_happened: 'Reviewer changed this object policy.',
    what_to_inspect: 'Confirm this object policy is correct before finalization.',
  };
}
