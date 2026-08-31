import { describe, expect, it } from 'vitest';
import {
  candidateAcceptEligibility,
  candidateConflictRanges,
  candidateEntityIds,
  candidateExplanation,
  entityPurity,
  firstCandidateConflictFrame,
  mergeSelectedEligibility,
  mostProtectivePolicy,
  policyAllowed,
  sectionEntities,
  temporalConflictRangesForEntities,
  visibleCandidateMatches,
} from '../src/state/entityRefinement';
import type { AuditEvent } from '../src/types/api';
import type { EffectiveEntity, EntityPolicy } from '../src/types/domain';

function entity(patch: Partial<EffectiveEntity>): EffectiveEntity {
  return {
    case_id: 'day',
    category: 'face',
    current_policy: 'redact',
    entity_id: 'entity_a',
    first_visible_source_frame: 10,
    label: 'Person 01',
    last_visible_source_frame: 20,
    representative_source_frame: 12,
    ...patch,
  } as EffectiveEntity;
}

describe('entity refinement UI helpers', () => {
  it('classifies sections and disables reveal actions for unsafe purity states', () => {
    const mixed = entity({
      entity_id: 'mixed',
      policy_action_availability: {
        actions: {
          keep_visible: { allowed: false, reason: 'mixed_or_unverified' },
          not_sensitive: { allowed: false, reason: 'mixed_or_unverified' },
          redact: { allowed: true },
        },
      },
      purity_classification: 'LIKELY_MIXED',
    });
    const unverified = entity({ entity_id: 'fragment', purity_classification: 'UNVERIFIED' });
    const lowQuality = entity({ entity_id: 'tiny', purity_classification: 'LOW_QUALITY' });
    const reviewedLowQuality = entity({
      entity_id: 'tiny_reviewed',
      policy_action_availability: {
        actions: {
          keep_visible: { allowed: true, reason: 'complete_child_review' },
        },
      },
      purity_classification: 'LOW_QUALITY',
    });

    expect(entityPurity(mixed)).toBe('LIKELY_MIXED');
    expect(policyAllowed(mixed, 'redact')).toBe(true);
    expect(policyAllowed(mixed, 'keep_visible')).toBe(false);
    expect(policyAllowed(unverified, 'not_sensitive')).toBe(false);
    expect(policyAllowed(lowQuality, 'keep_visible')).toBe(false);
    expect(policyAllowed(reviewedLowQuality, 'keep_visible')).toBe(true);
    expect(sectionEntities([mixed, unverified, lowQuality]).map((section) => section.title)).toEqual([
      'Likely mixed',
      'Unverified fragments',
      'Low-quality / likely false positives',
    ]);
  });

  it('uses most-protective merge policy and blocks temporal conflicts', () => {
    const policies: EntityPolicy[] = ['keep_visible', 'review_required', 'redact'];
    const entities = policies.map((policy, index) =>
      entity({
        current_policy: policy,
        entity_id: `entity_${index}`,
        first_visible_source_frame: index * 20,
        last_visible_source_frame: index * 20 + 5,
      }),
    );
    const overlapping = [
      entity({ entity_id: 'left', first_visible_source_frame: 10, last_visible_source_frame: 20 }),
      entity({ entity_id: 'right', first_visible_source_frame: 15, last_visible_source_frame: 25 }),
    ];
    const mixedSelection = [
      entity({ entity_id: 'mixed', purity_classification: 'LIKELY_MIXED' }),
      entity({ entity_id: 'plain', first_visible_source_frame: 40, last_visible_source_frame: 45 }),
    ];

    expect(mostProtectivePolicy(entities)).toBe('redact');
    expect(mergeSelectedEligibility(entities.slice(0, 2)).enabled).toBe(true);
    expect(mergeSelectedEligibility(overlapping)).toEqual({
      enabled: false,
      reason: 'Selected objects coexist on source frames 15-20, so they cannot be the same identity.',
    });
    expect(temporalConflictRangesForEntities(overlapping)).toEqual([
      { end: 20, entityIds: ['left', 'right'], labels: ['Person 01', 'Person 01'], start: 15 },
    ]);
    expect(mergeSelectedEligibility(mixedSelection)).toEqual({
      enabled: false,
      reason: 'Split mixed entities before merging.',
    });
  });

  it('keeps advisory candidates visible until append-only rejection and never merges automatically', () => {
    const candidate = {
      candidate_id: 'candidate_1',
      classification: 'POSSIBLE_CANDIDATE',
      entity_ids: ['night_person_04', 'night_person_01'],
    };
    const left = entity({
      advisory_candidate_matches: [candidate],
      case_id: 'night',
      entity_id: 'night_person_04',
      first_visible_source_frame: 0,
      last_visible_source_frame: 1,
    });
    const right = entity({
      case_id: 'night',
      entity_id: 'night_person_01',
      first_visible_source_frame: 4,
      last_visible_source_frame: 5,
    });
    const rejected: AuditEvent = {
      after_state: { candidate_id: 'candidate_1' },
      actor: {},
      case_id: 'night',
      event_hash: 'hash',
      event_id: 'event_1',
      event_type: 'entity_merge_candidate_rejected',
      previous_event_hash: null,
      schema_version: 'review_event.v1',
      timestamp_utc: '2026-08-26T00:00:00Z',
    };

    expect(visibleCandidateMatches(left, [])).toHaveLength(1);
    expect(candidateEntityIds(candidate)).toEqual(['night_person_04', 'night_person_01']);
    expect([left, right].map((item) => item.entity_id)).toEqual([
      'night_person_04',
      'night_person_01',
    ]);
    expect(visibleCandidateMatches(left, [rejected])).toHaveLength(0);
  });

  it('explains disabled candidate actions differently for conflicts and weak evidence', () => {
    const temporal = {
      candidate_id: 'candidate_conflict',
      classification: 'REJECT_TEMPORAL_CONFLICT',
      entity_ids: ['left', 'right'],
      similarity: 0.94,
      source_intervals: {
        left: [10, 20],
        right: [15, 25],
      },
      temporal_conflict_sample_frames: [15, 16, 17, 18, 19],
    };
    const insufficient = {
      candidate_id: 'candidate_weak',
      classification: 'INSUFFICIENT_EVIDENCE',
      entity_ids: ['one', 'two'],
      similarity: 0.72,
    };
    const possible = {
      candidate_id: 'candidate_possible',
      classification: 'POSSIBLE_CANDIDATE',
      entity_ids: ['one', 'two'],
      similarity: 0.94,
    };

    expect(candidateConflictRanges(temporal)).toEqual([{ end: 20, start: 15 }]);
    expect(firstCandidateConflictFrame(temporal)).toBe(15);
    expect(candidateAcceptEligibility(temporal)).toEqual({
      enabled: false,
      reason: 'These objects coexist on source frames 15-20, so they cannot be the same identity.',
    });
    expect(candidateExplanation(temporal)).toContain('Temporal conflict');
    expect(candidateAcceptEligibility(insufficient)).toEqual({
      enabled: false,
      reason: 'Insufficient visual evidence to merge safely; this is not a temporal conflict.',
    });
    expect(candidateExplanation(insufficient)).toContain('Insufficient visual evidence');
    expect(candidateAcceptEligibility(possible)).toEqual({
      enabled: true,
      reason: 'Reviewer may accept this visual merge candidate.',
    });
    expect(candidateExplanation(possible)).toContain('similarity score is only a triage signal');
  });
});
