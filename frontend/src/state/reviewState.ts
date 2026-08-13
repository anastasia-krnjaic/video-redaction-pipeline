import type { Entity, Task, TaskDecision } from '../types/api';
import type { CaseId, EffectiveEntity, EntityFilter, EntityPolicy, QueueBucket, TaskWithDecision } from '../types/domain';

export const ACTIONS = {
  looksCorrect: 'looks_correct',
  needsCorrection: 'privacy_leak',
  partialCoverage: 'partial_coverage',
  falseRedaction: 'false_redaction',
  wrongTarget: 'wrong_target',
  notSensitive: 'not_sensitive',
  cosmeticOnly: 'cosmetic_only',
  unsure: 'uncertain_escalate',
} as const;

const NEEDS_CORRECTION_ACTIONS = new Set<string>([
  ACTIONS.needsCorrection,
  ACTIONS.partialCoverage,
  ACTIONS.falseRedaction,
  ACTIONS.wrongTarget,
]);

const ACTION_ORDER = [
  ACTIONS.looksCorrect,
  ACTIONS.needsCorrection,
  ACTIONS.partialCoverage,
  ACTIONS.falseRedaction,
  ACTIONS.wrongTarget,
  ACTIONS.notSensitive,
  ACTIONS.cosmeticOnly,
  ACTIONS.unsure,
] as const;

export interface DecisionActionOption {
  action: string;
  className: string;
  label: string;
}

export function taskBucket(task: Task, decision?: TaskDecision): QueueBucket {
  if (task.decision_state === 'resolved') return 'Completed';
  if (task.normal_task_class === 'correction_verification' && task.actionable === false) return 'Waiting';
  if (!decision) return 'Open';
  if (decision.action === ACTIONS.unsure) return 'Unsure / Escalated';
  if (NEEDS_CORRECTION_ACTIONS.has(decision.action)) return 'Needs correction';
  return 'Completed';
}

export function tasksForCase(tasks: Task[], decisions: Record<string, TaskDecision>, caseId: CaseId): TaskWithDecision[] {
  return tasks
    .filter((task) => task.case_id === caseId && !task.technical_only)
    .map((task) => {
      const decision = decisions[task.review_task_id];
      return { ...task, decision, bucket: taskBucket(task, decision) };
    });
}

export function bucketCounts(tasks: TaskWithDecision[]): Record<QueueBucket, number> {
  return {
    Open: tasks.filter((task) => task.bucket === 'Open').length,
    Waiting: tasks.filter((task) => task.bucket === 'Waiting').length,
    'Needs correction': tasks.filter((task) => task.bucket === 'Needs correction').length,
    'Unsure / Escalated': tasks.filter((task) => task.bucket === 'Unsure / Escalated').length,
    Completed: tasks.filter((task) => task.bucket === 'Completed').length,
  };
}

export function decisionLabel(action?: string): string {
  if (!action) return 'Open';
  if (action === ACTIONS.looksCorrect) return 'Correct';
  if (action === ACTIONS.unsure) return 'Unsure';
  if (action === ACTIONS.notSensitive) return 'Not sensitive';
  if (action === ACTIONS.cosmeticOnly) return 'Cosmetic only';
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
  if (action === ACTIONS.looksCorrect || action === ACTIONS.notSensitive || action === ACTIONS.cosmeticOnly) {
    return 'decision-correct';
  }
  if (action === ACTIONS.unsure) return 'decision-unsure';
  return 'decision-needs-correction';
}

export function taskColorClass(task: TaskWithDecision): string {
  if (task.bucket === 'Completed') return 'marker-correct';
  if (task.bucket === 'Needs correction') return 'marker-needs-correction';
  if (task.bucket === 'Unsure / Escalated') return 'marker-unsure';
  if (task.normal_task_class === 'possible_false_redaction') return 'marker-false';
  if (task.category === 'face') return 'marker-face';
  return 'marker-plate';
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
  return policy === 'redact' ? null : `object_policy_verify_${caseId}_${entityId}_${policy}`;
}

export function policyVerificationTask(caseId: CaseId, entity: EffectiveEntity, taskId: string): Task {
  const frame =
    typeof entity.first_visible_source_frame === 'number' ? entity.first_visible_source_frame : 0;
  return {
    allowed_actions: [ACTIONS.looksCorrect, ACTIONS.needsCorrection, ACTIONS.unsure],
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
