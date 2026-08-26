import type { AuditEvent } from '../types/api';
import type { EffectiveEntity, EntityPolicy } from '../types/domain';

export const PURITY_PURE = 'PURE';
export const PURITY_LIKELY_MIXED = 'LIKELY_MIXED';
export const PURITY_LOW_QUALITY = 'LOW_QUALITY';
export const PURITY_UNVERIFIED = 'UNVERIFIED';

export type EntityPurity =
  | typeof PURITY_PURE
  | typeof PURITY_LIKELY_MIXED
  | typeof PURITY_LOW_QUALITY
  | typeof PURITY_UNVERIFIED;

export type EntitySection =
  | 'Verified entities'
  | 'Candidate groups'
  | 'Unverified fragments'
  | 'Low-quality / likely false positives'
  | 'Likely mixed';

const policyRank: Record<string, number> = {
  not_sensitive: 0,
  keep_visible: 0,
  review_required: 1,
  redact: 2,
};

export function entityPurity(entity: EffectiveEntity): EntityPurity {
  const direct = entity.purity_classification;
  if (isPurity(direct)) return direct;
  const refinement = entity.entity_refinement as Record<string, unknown> | undefined;
  const nested = refinement?.purity;
  if (isPurity(nested)) return nested;
  const childCount = childTrackletCount(entity);
  if (childCount <= 1) return PURITY_PURE;
  return PURITY_UNVERIFIED;
}

export function childTrackletCount(entity: EffectiveEntity): number {
  const refinement = entity.entity_refinement as Record<string, unknown> | undefined;
  const count = refinement?.child_tracklet_count;
  if (typeof count === 'number') return count;
  return entity.member_detector_tracker_ids?.length || entity.automatic_source_track_ids?.length || 0;
}

export function entitySection(entity: EffectiveEntity): EntitySection {
  const purity = entityPurity(entity);
  if (purity === PURITY_LIKELY_MIXED) return 'Likely mixed';
  if (purity === PURITY_LOW_QUALITY) return 'Low-quality / likely false positives';
  if ((entity.advisory_candidate_matches || []).length) return 'Candidate groups';
  if (purity === PURITY_UNVERIFIED) return 'Unverified fragments';
  return 'Verified entities';
}

export function sectionEntities(entities: EffectiveEntity[]): Array<{ title: EntitySection; entities: EffectiveEntity[] }> {
  const order: EntitySection[] = [
    'Likely mixed',
    'Candidate groups',
    'Unverified fragments',
    'Low-quality / likely false positives',
    'Verified entities',
  ];
  return order
    .map((title) => ({ title, entities: entities.filter((entity) => entitySection(entity) === title) }))
    .filter((section) => section.entities.length > 0);
}

export function policyAllowed(entity: EffectiveEntity, policy: EntityPolicy): boolean {
  const availability = entity.policy_action_availability as Record<string, unknown> | undefined;
  const actions = availability?.actions as Record<string, { allowed?: boolean }> | undefined;
  const action = actions?.[policy];
  if (action && typeof action.allowed === 'boolean') return action.allowed;
  const purity = entityPurity(entity);
  if ((purity === PURITY_LIKELY_MIXED || purity === PURITY_UNVERIFIED) && isRevealPolicy(policy)) {
    return false;
  }
  if (purity === PURITY_LOW_QUALITY && isRevealPolicy(policy)) return false;
  return true;
}

export function policyDisabledReason(entity: EffectiveEntity, policy: EntityPolicy): string {
  const availability = entity.policy_action_availability as Record<string, unknown> | undefined;
  const actions = availability?.actions as Record<string, { reason?: string }> | undefined;
  const reason = actions?.[policy]?.reason;
  if (reason) return reason.replaceAll('_', ' ');
  if (!policyAllowed(entity, policy)) return 'requires explicit verification';
  return '';
}

export function mostProtectivePolicy(entities: EffectiveEntity[]): EntityPolicy {
  return entities
    .map((entity) => String(entity.current_policy || entity.privacy_action || entity.default_privacy_policy || 'review_required') as EntityPolicy)
    .sort((left, right) => (policyRank[right] ?? -1) - (policyRank[left] ?? -1))[0] || 'review_required';
}

export function mergeSelectedEligibility(entities: EffectiveEntity[]): { enabled: boolean; reason: string } {
  if (entities.length < 2) return { enabled: false, reason: 'Select at least two same-category objects to merge.' };
  const cases = new Set(entities.map((entity) => entity.case_id));
  if (cases.size > 1) return { enabled: false, reason: 'Selected objects belong to different cases.' };
  const categories = new Set(entities.map((entity) => entity.category));
  if (categories.size > 1) return { enabled: false, reason: 'People and plates cannot be merged together.' };
  if (entities.some((entity) => entityPurity(entity) === PURITY_LIKELY_MIXED)) {
    return { enabled: false, reason: 'Split mixed entities before merging.' };
  }
  if (anyOverlap(entities)) {
    return { enabled: false, reason: 'Selected objects have simultaneous source-frame conflict.' };
  }
  return { enabled: true, reason: 'Merge selected review objects.' };
}

export function rejectedCandidateIds(events: AuditEvent[]): Set<string> {
  const rejected = new Set<string>();
  for (const event of events) {
    if (event.event_type !== 'entity_merge_candidate_rejected') continue;
    const after = event.after_state as Record<string, unknown> | undefined;
    if (typeof after?.candidate_id === 'string') rejected.add(after.candidate_id);
  }
  return rejected;
}

export function visibleCandidateMatches(entity: EffectiveEntity, events: AuditEvent[]): Record<string, unknown>[] {
  const rejected = rejectedCandidateIds(events);
  return (entity.advisory_candidate_matches || []).filter((candidate) => {
    const id = String((candidate as Record<string, unknown>).candidate_id || '');
    return id && !rejected.has(id);
  }) as Record<string, unknown>[];
}

export function candidateEntityIds(candidate: Record<string, unknown>): string[] {
  return Array.isArray(candidate.entity_ids) ? candidate.entity_ids.map(String) : [];
}

function isRevealPolicy(policy: EntityPolicy): boolean {
  return policy === 'keep_visible' || policy === 'not_sensitive';
}

function isPurity(value: unknown): value is EntityPurity {
  return (
    value === PURITY_PURE ||
    value === PURITY_LIKELY_MIXED ||
    value === PURITY_LOW_QUALITY ||
    value === PURITY_UNVERIFIED
  );
}

function anyOverlap(entities: EffectiveEntity[]): boolean {
  for (let index = 0; index < entities.length; index += 1) {
    for (let other = index + 1; other < entities.length; other += 1) {
      if (overlap(entities[index], entities[other])) return true;
    }
  }
  return false;
}

function overlap(left: EffectiveEntity, right: EffectiveEntity): boolean {
  const l1 = left.first_visible_source_frame;
  const l2 = left.last_visible_source_frame;
  const r1 = right.first_visible_source_frame;
  const r2 = right.last_visible_source_frame;
  if ([l1, l2, r1, r2].some((value) => typeof value !== 'number')) return false;
  return Math.max(l1 as number, r1 as number) <= Math.min(l2 as number, r2 as number);
}
