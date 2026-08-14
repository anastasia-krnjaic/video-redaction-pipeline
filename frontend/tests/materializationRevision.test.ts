import { describe, expect, it } from 'vitest';
import { materializationJobMatchesCurrentProjection } from '../src/state/materialization';
import type { MaterializationJob, WorkingState } from '../src/types/api';

const state = (revision?: string): WorkingState => ({
  active_projection_hash: revision,
  baseline: {},
  current_event_chain_head: null,
  manual_corrections: [],
  materialized_latest_reviewed: {},
  object_policy_overrides: {},
  schema_version: 'review_working_state.v1',
  state_hash: 'state_hash',
  task_decisions: {},
  task_decisions_by_case: {},
  task_resolutions: {},
  task_resolutions_by_case: {},
});

const job = (patch: Partial<MaterializationJob>): MaterializationJob => ({
  case_id: 'day',
  job_id: 'job_1',
  published_working_state_revision: 'rev_current',
  session_id: 'session_1',
  status: 'completed',
  ...patch,
});

describe('materialization completion revision guard', () => {
  it('accepts only completed same case/session jobs with an exact current revision', () => {
    const expected = { caseId: 'day', sessionId: 'session_1' };

    expect(materializationJobMatchesCurrentProjection(
      job({}),
      state('rev_current'),
      expected,
    )).toBe(true);
    expect(materializationJobMatchesCurrentProjection(
      job({ case_id: 'night' }),
      state('rev_current'),
      expected,
    )).toBe(false);
    expect(materializationJobMatchesCurrentProjection(
      job({ session_id: 'session_2' }),
      state('rev_current'),
      expected,
    )).toBe(false);
    expect(materializationJobMatchesCurrentProjection(
      job({ published_working_state_revision: 'rev_old' }),
      state('rev_current'),
      expected,
    )).toBe(false);
    expect(materializationJobMatchesCurrentProjection(
      job({ published_working_state_revision: undefined, accepted_working_state_revision: undefined, working_state: undefined }),
      state('rev_current'),
      expected,
    )).toBe(false);
    expect(materializationJobMatchesCurrentProjection(
      job({}),
      state(undefined),
      expected,
    )).toBe(false);
    expect(materializationJobMatchesCurrentProjection(
      job({ status: 'running' }),
      state('rev_current'),
      expected,
    )).toBe(false);
  });
});
