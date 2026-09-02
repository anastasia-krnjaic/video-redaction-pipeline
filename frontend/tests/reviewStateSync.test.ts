import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import {
  decisionFromEvent,
  eventListIsOlderThanKnownEvents,
  mergeAuditEvents,
  mergeDecisionsByEventOrder,
  workingStateIsOlderThanKnownEvents,
} from '../src/state/reviewSync';
import type { AuditEvent, WorkingState } from '../src/types/api';

function event(hash: string, previous: string | null, taskId: string): AuditEvent {
  return {
    after_state: {
      action: 'looks_correct',
      case_id: 'day',
      decision_status: 'green',
      task_id: taskId,
    },
    case_id: 'day',
    event_hash: hash,
    event_id: `event_${hash}`,
    event_type: 'decision_recorded',
    previous_event_hash: previous,
    schema_version: 'review_event.v1',
    task_id: taskId,
    timestamp_utc: '2026-08-18T00:00:00Z',
  };
}

function workingState(head: string): WorkingState {
  return {
    active_projection_hash: 'media_revision',
    baseline: {},
    current_event_chain_head: head,
    manual_corrections: [],
    materialized_latest_reviewed: {},
    object_policy_overrides: {},
    schema_version: 'review_working_state.v1',
    state_hash: `state_${head}`,
    task_decisions: {},
    task_decisions_by_case: {},
    task_resolutions: {},
    task_resolutions_by_case: {},
  };
}

describe('review workspace state synchronization', () => {
  it('rejects an older returned projection once a newer response was applied', () => {
    const first = event('hash_a', 'session_started', 'task_a');
    const second = event('hash_b', 'hash_a', 'task_b');

    expect(workingStateIsOlderThanKnownEvents(workingState('hash_a'), [first, second])).toBe(true);
    expect(workingStateIsOlderThanKnownEvents(workingState('hash_b'), [first, second])).toBe(false);
    expect(workingStateIsOlderThanKnownEvents(workingState('hash_c'), [first, second])).toBe(false);
  });

  it('preserves a successful decision event when its working-state projection is skipped', () => {
    expect(decisionFromEvent(event('hash_a', 'session_started', 'task_a'))).toMatchObject({
      action: 'looks_correct',
      case_id: 'day',
      decision_status: 'green',
      event_hash: 'hash_a',
      task_id: 'task_a',
    });
  });

  it('orders merged audit events by hash chain instead of response arrival', () => {
    const first = event('hash_a', 'session_started', 'task_a');
    const second = event('hash_b', 'hash_a', 'task_b');

    expect(mergeAuditEvents([second], [first]).map((item) => item.event_hash)).toEqual([
      'hash_a',
      'hash_b',
    ]);
  });

  it('detects a delayed session payload older than locally accepted events', () => {
    const sessionStarted = event('session_started_hash', null, 'session');
    const acceptedDecision = event('hash_a', 'session_started_hash', 'task_a');

    expect(eventListIsOlderThanKnownEvents([sessionStarted], [sessionStarted, acceptedDecision])).toBe(true);
    expect(eventListIsOlderThanKnownEvents([sessionStarted, acceptedDecision], [sessionStarted])).toBe(false);
  });

  it('does not let an older overlapping decision response replace a newer decision', () => {
    const first = event('hash_a', 'session_started', 'task_a');
    const second = event('hash_b', 'hash_a', 'task_a');

    const merged = mergeDecisionsByEventOrder(
      {
        day: {
          task_a: {
            action: 'not_sensitive',
            case_id: 'day',
            decision_status: 'green',
            event_hash: 'hash_b',
            task_id: 'task_a',
          },
        },
      },
      {
        day: {
          task_a: {
            action: 'looks_correct',
            case_id: 'day',
            decision_status: 'green',
            event_hash: 'hash_a',
            task_id: 'task_a',
          },
        },
      },
      [first, second],
    );

    expect(merged.day.task_a.action).toBe('not_sensitive');
  });

  it('keeps routine decision posts alive across route unloads', () => {
    const clientSource = fs.readFileSync(path.resolve(process.cwd(), 'src/api/client.ts'), 'utf8');

    expect(clientSource).toContain("`${API_PREFIX}/decisions`, payload, { keepalive: true }");
    expect(clientSource).toContain("`${API_PREFIX}/decision-groups`, payload, { keepalive: true }");
  });
});
