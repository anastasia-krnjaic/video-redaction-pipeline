import { describe, expect, it } from 'vitest';
import type { Entity, Task, TaskDecision } from '../src/types/api';
import {
  ACTIONS,
  bucketCounts,
  effectiveEntities,
  filteredEntities,
  taskBucket,
  taskDecisionActionOptions,
  tasksForCase,
} from '../src/state/reviewState';

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
    expect(taskBucket({ ...task, decision_state: 'resolved' })).toBe('Completed');
    expect(taskBucket({ ...task, actionable: false, normal_task_class: 'correction_verification' })).toBe('Waiting');
  });

  it('keeps the normal review buckets to the simplified reviewer lifecycle', () => {
    const tasks = tasksForCase([
      task,
      { ...task, review_task_id: 'task_2', decision_state: 'resolved' },
      { ...task, review_task_id: 'task_3' },
    ], {
      task_1: decision(ACTIONS.needsCorrection),
      task_3: { ...decision(ACTIONS.unsure), task_id: 'task_3' },
    }, 'day');
    const counts = bucketCounts(tasks);
    expect(counts).toEqual({
      Open: 0,
      Waiting: 0,
      'Needs correction': 1,
      'Unsure / Escalated': 1,
      Completed: 1,
    });
    expect(Object.keys(counts)).not.toContain('Resolved');
  });

  it('renders only source-span restricted task actions from the manifest allowlist', () => {
    const actions = taskDecisionActionOptions({
      allowed_actions: [ACTIONS.needsCorrection, ACTIONS.unsure],
    });

    expect(actions.map((action) => action.action)).toEqual([
      ACTIONS.needsCorrection,
      ACTIONS.unsure,
    ]);
    expect(actions.map((action) => action.label)).toEqual(['Needs correction', 'Unsure']);
    expect(actions.some((action) => action.action === ACTIONS.looksCorrect)).toBe(false);
    expect(actions.some((action) => action.action === ACTIONS.notSensitive)).toBe(false);
  });

  it('preserves normal entity task actions when the manifest allows Correct', () => {
    const actions = taskDecisionActionOptions({
      allowed_actions: [
        ACTIONS.looksCorrect,
        ACTIONS.needsCorrection,
        ACTIONS.notSensitive,
        ACTIONS.unsure,
      ],
    });

    expect(actions.map((action) => action.action)).toEqual([
      ACTIONS.looksCorrect,
      ACTIONS.needsCorrection,
      ACTIONS.notSensitive,
      ACTIONS.unsure,
    ]);
    expect(actions.map((action) => action.label)).toContain('Correct');
  });

  it('filters effective entities and hides retired merge targets', () => {
    const effective = effectiveEntities([entityA, entityB], { b: { ...entityB, current_status: 'merged' } });
    expect(effective.map((entity) => entity.entity_id)).toEqual(['a']);
    expect(filteredEntities([entityA, entityB], 'unresolved').map((entity) => entity.entity_id)).toEqual(['b']);
  });

  it('includes event-created canonical merge entities in the active registry', () => {
    const merged = {
      ...entityA,
      entity_id: 'merged',
      label: 'Merged Person',
      source_entity_ids: ['a', 'b'],
    };
    const effective = effectiveEntities([entityA, entityB], {
      a: { entity_id: 'a', current_status: 'merged' } as Entity,
      b: { entity_id: 'b', current_status: 'merged' } as Entity,
      merged,
    });
    expect(effective.map((entity: Entity) => entity.entity_id)).toEqual(['merged']);
  });

  it('does not leak event-created entities between cases', () => {
    const nightEntity = {
      ...entityA,
      case_id: 'night',
      entity_id: 'night-merged',
      label: 'Night merged person',
    };
    const effective = effectiveEntities([entityA, entityB], { 'night-merged': nightEntity });
    expect(effective.map((entity: Entity) => entity.entity_id)).toEqual(['a', 'b']);
  });
});
