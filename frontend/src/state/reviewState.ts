import type { Entity, Task, TaskDecision } from '../types/api';
import type { CaseId, DecisionsByCase, EffectiveEntity, EntityFilter, EntityPolicy, QueueBucket, TaskResolutionsByCase, TaskWithDecision } from '../types/domain';

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
  correctMeaning: string;
  needsCorrectionMeaning: string;
  notSensitiveMeaning: string;
  whatToCheck: string;
  whyFlagged: string;
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

export function decisionLabel(action?: string): string {
  if (!action) return 'Open';
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

export function taskGuidance(task: Pick<Task, 'category' | 'issue_type' | 'normal_task_class' | 'review_task_id' | 'title' | 'what_happened' | 'what_to_inspect'>): TaskGuidance {
  const family = guidanceFamily(task);
  const subject = task.category === 'license_plate' ? 'plate' : task.category === 'face' ? 'face or person' : 'sensitive region';
  const fallbackWhatToCheck = task.what_to_inspect || `Inspect the ${subject} through the marked source-frame interval.`;
  const shared = {
    correctMeaning: `Choose Correct when the current privacy result protects the ${subject} throughout the authoritative interval.`,
    needsCorrectionMeaning: `Choose Needs correction when the ${subject} is visible, under-covered, on the wrong target, or otherwise needs a manual redaction adjustment.`,
    notSensitiveMeaning: `Choose Not sensitive only when the marked object is not actually sensitive in this context, such as a non-person or non-plate false alarm.`,
    whatToCheck: fallbackWhatToCheck,
  };
  const mappings: Record<string, TaskGuidance> = {
    automatic_uncertain: {
      ...shared,
      whyFlagged: 'The automatic privacy result did not have enough confidence to close this interval without human review.',
      whatToCheck: `Check whether the ${subject} is safely hidden for the whole marked interval.`,
    },
    corrected_result_check: {
      ...shared,
      whyFlagged: 'A saved correction changed the reviewed result and needs confirmation against the authoritative source span.',
      whatToCheck: 'Inspect the corrected interval and confirm the saved redaction covers the intended area without drifting or hiding the wrong target.',
    },
    false_redaction_check: {
      ...shared,
      correctMeaning: `Choose Correct when the visible redaction is appropriate for the marked ${subject}.`,
      needsCorrectionMeaning: 'Choose Needs correction when the redaction covers the wrong place, misses the actual sensitive target, or needs a different manual region.',
      notSensitiveMeaning: 'Choose Not sensitive when the automatic redaction is unnecessary because the marked object is not sensitive.',
      whyFlagged: 'This interval may be an unnecessary or wrong-target redaction.',
      whatToCheck: `Confirm whether the redacted target is really a ${subject} that should be protected.`,
    },
    identity_competition: {
      ...shared,
      whyFlagged: 'Nearby candidate identities overlap or compete, so the automatic assignment may have followed the wrong target.',
      whatToCheck: `Follow the marked ${subject} across the interval and verify the redaction stays on the same real-world target.`,
    },
    insufficient_observations: {
      ...shared,
      whyFlagged: 'The automatic system saw too few reliable observations to close this interval confidently.',
      whatToCheck: `Step through the interval and check whether the ${subject} remains protected in frames with weak or missing observations.`,
    },
    insufficient_strong_hits: {
      ...shared,
      whyFlagged: 'The track had observations, but too few were strong enough for automatic closure.',
      whatToCheck: `Check the clearest and weakest frames in the interval to confirm the ${subject} is protected.`,
    },
    interpolation_risk: {
      ...shared,
      whyFlagged: 'The redaction depends on interpolation, scale changes, or motion that can drift between observed frames.',
      whatToCheck: `Scrub through the interval and verify the redaction shape follows the ${subject} without shrinking, drifting, or lagging.`,
    },
    long_detection_gap: {
      ...shared,
      whyFlagged: 'The track spans a long gap without direct detections.',
      whatToCheck: `Inspect the gap and both recovery ends to confirm the ${subject} is still covered when automatic detections are absent.`,
    },
    one_sided_recovery: {
      ...shared,
      whyFlagged: 'The track recovered from only one side of a gap, which can indicate that the target changed or drifted.',
      whatToCheck: `Check both sides of the recovery and confirm the redaction follows the same ${subject}.`,
    },
    optional_spot_check: {
      ...shared,
      whyFlagged: 'This is a sampled spot check used to keep the reviewed result honest.',
      whatToCheck: `Inspect the marked sample and confirm the ${subject} is handled correctly.`,
    },
    privacy_coverage_gap: {
      ...shared,
      whyFlagged: 'Source-backed privacy geometry indicates a possible coverage gap in the reviewed result.',
      whatToCheck: `Inspect the marked frames and verify the expected ${subject} region is actually hidden.`,
    },
    source_span_review: {
      ...shared,
      whyFlagged: 'This task is tied to an authoritative source-frame span that needs a case-scoped human decision.',
      whatToCheck: `Review exactly the marked source span for this ${subject}; do not apply the decision to other cases or unrelated intervals.`,
    },
    track_requires_review: {
      ...shared,
      whyFlagged: 'The object track still requires human review before the case can be finalized.',
      whatToCheck: `Inspect the marked ${subject} track and confirm whether its current privacy treatment is safe.`,
    },
  };
  return mappings[family] || mappings.automatic_uncertain;
}

function guidanceFamily(task: Pick<Task, 'issue_type' | 'normal_task_class' | 'review_task_id' | 'title' | 'what_happened' | 'what_to_inspect'>): string {
  const text = [
    task.issue_type,
    task.normal_task_class,
    task.review_task_id,
    task.title,
    task.what_happened,
    task.what_to_inspect,
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
  if (text.includes('correction') || text.includes('corrected')) return 'corrected_result_check';
  if (text.includes('false') || text.includes('wrong target') || text.includes('unnecessary')) return 'false_redaction_check';
  if (text.includes('source_span') || text.includes('source span') || text.includes('authoritative')) return 'source_span_review';
  if (text.includes('coverage_gap') || text.includes('coverage gap') || text.includes('privacy_gap') || text.includes('privacy gap')) return 'privacy_coverage_gap';
  if (text.includes('insufficient_strong') || text.includes('strong hits')) return 'insufficient_strong_hits';
  if (text.includes('insufficient_observed') || text.includes('insufficient observation') || text.includes('observed hits')) return 'insufficient_observations';
  if (text.includes('long') && text.includes('gap')) return 'long_detection_gap';
  if (text.includes('one-sided') || text.includes('one_sided')) return 'one_sided_recovery';
  if (text.includes('identity') || text.includes('competition') || text.includes('duplicate')) return 'identity_competition';
  if (text.includes('interpolation') || text.includes('scale-change') || text.includes('scale change') || text.includes('drift')) return 'interpolation_risk';
  if (text.includes('optional') || text.includes('spot_check') || text.includes('spot check')) return 'optional_spot_check';
  if (text.includes('track_review') || text.includes('requires review') || text.includes('needs review')) return 'track_requires_review';
  return 'automatic_uncertain';
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
