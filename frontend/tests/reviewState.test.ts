import { describe, expect, it } from 'vitest';
import type { Entity, Task, TaskDecision } from '../src/types/api';
import {
  ACTIONS,
  bucketCounts,
  effectiveEntities,
  filteredEntities,
  guidanceFamily,
  taskBucket,
  taskDecisionActionOptions,
  taskGuidance,
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

function decision(action: string, caseId = 'day', taskId = 'task_1'): TaskDecision {
  return {
    action,
    case_id: caseId,
    decision_status:
      action === ACTIONS.looksCorrect || action === ACTIONS.notSensitive
        ? 'green'
        : 'blocking',
    task_id: taskId,
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
    expect(taskBucket(task, decision(ACTIONS.looksCorrect))).toBe('Correct');
    expect(taskBucket(task, decision(ACTIONS.needsCorrection))).toBe('Needs correction');
    expect(taskBucket(task, decision(ACTIONS.notSensitive))).toBe('Not sensitive');
    expect(taskBucket({ ...task, decision_state: 'resolved' })).toBe('Correct');
  });

  it('keeps the normal review buckets to the simplified reviewer lifecycle', () => {
    const tasks = tasksForCase([
      task,
      { ...task, review_task_id: 'task_2', decision_state: 'resolved' },
      { ...task, review_task_id: 'task_3' },
    ], {
      day: {
        task_1: decision(ACTIONS.needsCorrection),
        task_3: decision(ACTIONS.notSensitive, 'day', 'task_3'),
      },
    }, 'day');
    const counts = bucketCounts(tasks);
    expect(counts).toEqual({
      Open: 0,
      'Needs correction': 1,
      Correct: 1,
      'Not sensitive': 1,
    });
    expect(Object.keys(counts)).not.toContain('Resolved');
    expect(Object.keys(counts)).not.toContain('Unsure / Escalated');
    expect(Object.keys(counts)).not.toContain('Completed');
  });

  it('keeps duplicate task ids isolated across cases', () => {
    const duplicateTasks: Task[] = [
      { ...task, case_id: 'day', review_task_id: 'task_001' },
      { ...task, case_id: 'night', review_task_id: 'task_001' },
    ];
    const decisions = {
      day: {
        task_001: decision(ACTIONS.looksCorrect, 'day', 'task_001'),
      },
    };

    const dayTasks = tasksForCase(duplicateTasks, decisions, 'day');
    const nightTasks = tasksForCase(duplicateTasks, decisions, 'night');
    const refreshedDayTasks = tasksForCase(duplicateTasks, { ...decisions }, 'day');

    expect(dayTasks[0].bucket).toBe('Correct');
    expect(nightTasks[0].bucket).toBe('Open');
    expect(refreshedDayTasks[0].bucket).toBe('Correct');
  });

  it('keeps duplicate not-sensitive decisions and derived resolutions case-scoped', () => {
    const duplicateTasks: Task[] = [
      { ...task, case_id: 'day', review_task_id: 'task_001' },
      { ...task, case_id: 'night', review_task_id: 'task_001' },
    ];

    expect(
      tasksForCase(
        duplicateTasks,
        { day: { task_001: decision(ACTIONS.notSensitive, 'day', 'task_001') } },
        'night',
      )[0].bucket,
    ).toBe('Open');

    expect(
      tasksForCase(
        duplicateTasks,
        {},
        'day',
        { day: { task_001: decision(ACTIONS.looksCorrect, 'day', 'task_001') } },
      )[0].bucket,
    ).toBe('Correct');
    expect(
      tasksForCase(
        duplicateTasks,
        {},
        'night',
        { day: { task_001: decision(ACTIONS.looksCorrect, 'day', 'task_001') } },
      )[0].bucket,
    ).toBe('Open');
  });

  it('renders the three canonical task actions from the manifest allowlist', () => {
    const actions = taskDecisionActionOptions({
      allowed_actions: [ACTIONS.looksCorrect, ACTIONS.needsCorrection, ACTIONS.notSensitive],
    });

    expect(actions.map((action) => action.action)).toEqual([
      ACTIONS.looksCorrect,
      ACTIONS.needsCorrection,
      ACTIONS.notSensitive,
    ]);
    expect(actions.map((action) => action.label)).toEqual(['Correct', 'Needs correction', 'Not sensitive']);
  });

  it('ignores legacy actions that are not present in the canonical manifest allowlist', () => {
    const actions = taskDecisionActionOptions({
      allowed_actions: [
        ACTIONS.looksCorrect,
        ACTIONS.needsCorrection,
        ACTIONS.notSensitive,
        'uncertain_escalate',
      ],
    });

    expect(actions.map((action) => action.action)).toEqual([
      ACTIONS.looksCorrect,
      ACTIONS.needsCorrection,
      ACTIONS.notSensitive,
    ]);
    expect(actions.map((action) => action.label)).toContain('Correct');
  });

  it('uses compact query presentation copy instead of repeated reviewer instructions', () => {
    expect(taskGuidance({
      ...task,
      category: 'license_plate',
      query_presentation_class: 'PLATE_COVERAGE_GAP',
    })).toEqual({
      instruction: 'Check that the plate is fully covered in this interval.',
      queryClass: 'PLATE_COVERAGE_GAP',
      title: 'Plate may be visible',
    });

    const savedCorrection = taskGuidance({
      ...task,
      normal_task_class: 'correction_verification',
      short_instruction: 'Verify this saved geometry.',
      short_title: 'Saved edit check',
    });
    expect(savedCorrection).toEqual({
      instruction: 'Verify this saved geometry.',
      queryClass: 'VERIFY_SAVED_CORRECTION',
      title: 'Saved edit check',
    });
    expect(savedCorrection.instruction).not.toContain('Correct means');
    expect(savedCorrection.instruction).not.toContain('Needs correction when');
  });

  it('does not infer saved-correction cards from decision copy', () => {
    const plateTrackReview = taskGuidance({
      ...task,
      allowed_actions: [ACTIONS.looksCorrect, ACTIONS.needsCorrection, ACTIONS.notSensitive],
      category: 'license_plate',
      issue_type: 'plate_track_review_required',
      normal_task_class: 'plate_track_review_required',
      title: 'Check a low-confidence plate track',
      what_to_inspect: 'Check the plate-like target and choose Needs correction if any sensitive plate remains exposed.',
      raw_reason_codes: ['insufficient_observations'],
    });

    expect(plateTrackReview).toEqual({
      instruction: 'Check that the redaction stays on the intended target.',
      queryClass: 'TRACKING_UNCERTAIN',
      title: 'Tracking needs review',
    });
  });

  it('keeps fallback query classification semantic and compact', () => {
    expect(guidanceFamily({
      category: 'face',
      issue_type: 'possible_privacy_gap',
      normal_task_class: 'possible_privacy_gap',
      source_task: { raw_reason_codes: ['identity_competition'] },
    })).toBe('IDENTITY_UNCERTAIN');
    expect(guidanceFamily({
      category: 'face',
      issue_type: 'possible_privacy_gap',
      normal_task_class: 'possible_privacy_gap',
      source_task: { raw_reason_codes: ['long_detection_gap'] },
    })).toBe('INTERPOLATION_OR_SCALE_RISK');
    expect(guidanceFamily({
      category: 'license_plate',
      issue_type: 'plate_coverage_gap',
      normal_task_class: 'plate_coverage_gap',
      source_task: { raw_reason_codes: ['gap_exceeds_safe_interpolation_limit'] },
    })).toBe('PLATE_COVERAGE_GAP');
    expect(guidanceFamily({
      category: 'face',
      correction_id: 'corr_1',
      issue_type: 'correction_verification',
      normal_task_class: 'correction_verification',
    })).toBe('VERIFY_SAVED_CORRECTION');
    expect(guidanceFamily({
      category: 'license_plate',
      issue_type: 'possible_false_redaction',
      normal_task_class: 'possible_false_redaction',
    })).toBe('POSSIBLE_FALSE_POSITIVE');
    expect(guidanceFamily({
      category: 'face',
      issue_type: 'optional_spot_check',
      normal_task_class: 'optional_spot_check',
    })).toBe('OPTIONAL_SPOT_CHECK');
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
