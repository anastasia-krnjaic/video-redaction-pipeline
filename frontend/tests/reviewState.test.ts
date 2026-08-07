import { describe, expect, it } from 'vitest';
import type { Entity, Task, TaskDecision } from '../src/types/api';
import { ACTIONS, effectiveEntities, filteredEntities, taskBucket } from '../src/state/reviewState';
import { entitiesOverlap, entityGap, mergeCandidates } from '../src/components/EntityMergeDialog';

const task: Task = {
  case_id: 'day',
  category: 'face',
  review_task_id: 'task_1',
  source_frame_start: 10,
  source_frame_end: 20,
  technical_only: false,
  title: 'Task',
  what_to_inspect: 'Inspect',
};

function decision(action: string): TaskDecision {
  return {
    action,
    case_id: 'day',
    decision_status: action === ACTIONS.looksCorrect ? 'green' : 'blocking',
    task_id: 'task_1',
  };
}

const entityA: Entity = {
  case_id: 'day',
  category: 'face',
  current_policy: 'redact',
  entity_id: 'a',
  label: 'Person 01',
  representative_source_frame: 12,
  first_visible_source_frame: 10,
  last_visible_source_frame: 20,
};

const entityB: Entity = {
  case_id: 'day',
  category: 'face',
  current_policy: 'review_required',
  entity_id: 'b',
  label: 'Possible person 01',
  representative_source_frame: 35,
  first_visible_source_frame: 35,
  last_visible_source_frame: 40,
  review_entity_state: 'unresolved',
};

describe('review state helpers', () => {
  it('maps persisted decisions to lifecycle buckets', () => {
    expect(taskBucket(task)).toBe('Open');
    expect(taskBucket(task, decision(ACTIONS.looksCorrect))).toBe('Completed');
    expect(taskBucket(task, decision(ACTIONS.needsCorrection))).toBe('Needs correction');
    expect(taskBucket(task, decision(ACTIONS.unsure))).toBe('Unsure / Escalated');
  });

  it('filters effective entities and hides retired merge targets', () => {
    const effective = effectiveEntities([entityA, entityB], { b: { ...entityB, current_status: 'merged' } });
    expect(effective.map((entity) => entity.entity_id)).toEqual(['a']);
    expect(filteredEntities([entityA, entityB], 'unresolved').map((entity) => entity.entity_id)).toEqual(['b']);
  });

  it('keeps merge candidates same-category and non-overlapping', () => {
    expect(entitiesOverlap(entityA, entityB)).toBe(false);
    expect(entityGap(entityA, entityB)).toBe(15);
    expect(mergeCandidates(entityA, [entityA, entityB]).map((entity) => entity.entity_id)).toEqual(['b']);
  });
});
