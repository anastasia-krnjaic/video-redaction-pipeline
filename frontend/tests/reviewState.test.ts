import { describe, expect, it } from 'vitest';
import type { Entity, Task, TaskDecision, WorkingState } from '../src/types/api';
import {
  ACTIONS,
  authoritativeWorkingStateAfterSweepApproval,
  bucketCounts,
  effectiveEntities,
  filteredEntities,
  guidanceFamily,
  needsCorrectionBlockerCount,
  optionalReviewGroups,
  requiredReviewGroups,
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

function workingState(revision?: string): WorkingState {
  return {
    active_projection_hash: revision,
    baseline: {},
    current_event_chain_head: revision || null,
    manual_corrections: [],
    materialized_latest_reviewed: {},
    object_policy_overrides: {},
    schema_version: 'review_working_state.v1',
    state_hash: revision || 'state_hash',
    task_decisions: {},
    task_decisions_by_case: {},
    task_resolutions: {},
    task_resolutions_by_case: {},
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
  it('uses authoritative sweep working state from the mutation response before reloading', async () => {
    const responseState = workingState('sweep_head');
    const loadedState = workingState('loaded_head');
    let reloadCount = 0;

    const resolved = await authoritativeWorkingStateAfterSweepApproval(
      { working_state: responseState },
      'session_1',
      async () => {
        reloadCount += 1;
        return { working_state: loadedState };
      },
    );

    expect(resolved).toBe(responseState);
    expect(reloadCount).toBe(0);
  });

  it('reloads authoritative sweep working state when a legacy response omits it', async () => {
    const loadedState = workingState('loaded_head');

    const resolved = await authoritativeWorkingStateAfterSweepApproval(
      {},
      'session_1',
      async (sessionId) => {
        expect(sessionId).toBe('session_1');
        return { working_state: loadedState };
      },
    );

    expect(resolved).toBe(loadedState);
  });

  it('propagates sweep working-state reload failures without synthesizing approval', async () => {
    await expect(
      authoritativeWorkingStateAfterSweepApproval(
        {},
        'session_1',
        async () => {
          throw new Error('network down');
        },
      ),
    ).rejects.toThrow('network down');
  });

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

  it('does not count published correction verification evidence as a Needs-correction blocker', () => {
    const verificationTasks: Task[] = [
      {
        ...task,
        review_task_id: 'correction_verify_day_corr_valid_one',
        correction_id: 'corr_valid_one',
        normal_task_class: 'correction_verification',
        publication_status: 'published',
        decision_state: 'pending',
        actionable: true,
      },
      {
        ...task,
        review_task_id: 'correction_verify_day_corr_valid_two',
        correction_id: 'corr_valid_two',
        normal_task_class: 'correction_verification',
        publication_status: 'published',
        decision_state: 'pending',
        actionable: true,
      },
    ];

    const queueRows = tasksForCase(verificationTasks, {}, 'day');
    const required = requiredReviewGroups(queueRows);
    const counts = bucketCounts(required);

    expect(verificationTasks).toHaveLength(2);
    expect(queueRows).toHaveLength(0);
    expect(counts['Needs correction']).toBe(0);
    expect(needsCorrectionBlockerCount(required)).toBe(0);
  });

  it('keeps top-level Needs-correction count aligned with visible queue blockers', () => {
    const tasks = tasksForCase([
      {
        ...task,
        review_task_id: 'correction_verify_day_corr_valid',
        correction_id: 'corr_valid',
        normal_task_class: 'correction_verification',
        publication_status: 'published',
        decision_state: 'pending',
        actionable: true,
      },
      { ...task, review_task_id: 'real_blocker', source_frame_start: 30, source_frame_end: 40 },
      { ...task, review_task_id: 'resolved', source_frame_start: 50, source_frame_end: 60 },
    ], {
      day: {
        real_blocker: decision(ACTIONS.needsCorrection, 'day', 'real_blocker'),
        resolved: decision(ACTIONS.looksCorrect, 'day', 'resolved'),
      },
    }, 'day');
    const required = requiredReviewGroups(tasks);
    const counts = bucketCounts(required);

    expect(counts['Needs correction']).toBe(1);
    expect(needsCorrectionBlockerCount(required)).toBe(counts['Needs correction']);
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

  it('groups chunked required tasks by authoritative source task and separates optional QA', () => {
    const taskRows = tasksForCase([
      {
        ...task,
        review_task_id: 'identity_chunk_1',
        source_frame_start: 10,
        source_frame_end: 20,
        target_binding: { source_task_id: 'identity_root_1' },
        query_presentation_class: 'IDENTITY_UNCERTAIN',
      },
      {
        ...task,
        review_task_id: 'identity_chunk_2',
        source_frame_start: 21,
        source_frame_end: 30,
        target_binding: { source_task_id: 'identity_root_1' },
        query_presentation_class: 'IDENTITY_UNCERTAIN',
      },
      {
        ...task,
        optional: true,
        review_task_id: 'spot_check_1',
        source_frame_start: 31,
        source_frame_end: 31,
        query_presentation_class: 'OPTIONAL_SPOT_CHECK',
      },
    ], {}, 'day');

    const required = requiredReviewGroups(taskRows);
    const optional = optionalReviewGroups(taskRows);

    expect(required).toHaveLength(1);
    expect(required[0].child_task_ids).toEqual(['identity_chunk_1', 'identity_chunk_2']);
    expect(required[0].segment_count).toBe(2);
    expect(required[0].optional_group).toBe(false);
    expect(optional).toHaveLength(1);
    expect(optional[0].child_task_ids).toEqual(['spot_check_1']);
  });

  it('groups compatible same-entity plate checks into one review group', () => {
    const taskRows = tasksForCase([
      {
        ...task,
        category: 'license_plate',
        review_task_id: 'plate_coverage',
        source_frame_start: 40,
        source_frame_end: 50,
        query_presentation_class: 'PLATE_COVERAGE_GAP',
        target_entity_ids: ['plate_03'],
      },
      {
        ...task,
        category: 'license_plate',
        review_task_id: 'plate_tracking',
        source_frame_start: 42,
        source_frame_end: 56,
        query_presentation_class: 'TRACKING_UNCERTAIN',
        target_entity_ids: ['plate_03'],
      },
      {
        ...task,
        category: 'license_plate',
        review_task_id: 'other_plate',
        source_frame_start: 42,
        source_frame_end: 56,
        query_presentation_class: 'TRACKING_UNCERTAIN',
        target_entity_ids: ['plate_04'],
      },
    ], {}, 'day');

    const groups = requiredReviewGroups(taskRows);

    expect(groups).toHaveLength(2);
    const plateGroup = groups.find((group) => group.child_task_ids.includes('plate_coverage'));
    expect(plateGroup?.title).toBe('Plate needs review');
    expect(plateGroup?.child_task_ids).toEqual(['plate_coverage', 'plate_tracking']);
    expect(plateGroup?.reason_chips).toEqual(['Coverage', 'Tracking']);
    expect(groups.find((group) => group.child_task_ids.includes('other_plate'))?.segment_count).toBe(1);
  });
});
