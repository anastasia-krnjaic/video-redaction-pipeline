import { describe, expect, it } from 'vitest';
import { explicitSessionRequestFromSearch } from '../src/utils/sessionRouting';

describe('explicit review session routing', () => {
  it('accepts canonical explicit session ids', () => {
    expect(explicitSessionRequestFromSearch('?session_id=session_c336594eea714ffd95afce371ea67202')).toEqual({
      raw: 'session_c336594eea714ffd95afce371ea67202',
      requested: true,
      sessionId: 'session_c336594eea714ffd95afce371ea67202',
    });
  });

  it('accepts the legacy camel-case query name without changing semantics', () => {
    expect(explicitSessionRequestFromSearch('?sessionId=session_abc123')).toEqual({
      raw: 'session_abc123',
      requested: true,
      sessionId: 'session_abc123',
    });
  });

  it('distinguishes absent session parameters from bad explicit parameters', () => {
    expect(explicitSessionRequestFromSearch('?case=day')).toEqual({ requested: false });
    expect(explicitSessionRequestFromSearch('?session_id=')).toMatchObject({
      requested: true,
      error: 'A review session id was requested, but the value was empty.',
    });
    expect(explicitSessionRequestFromSearch('?session_id=../session_abc123')).toMatchObject({
      requested: true,
      error:
        'Invalid review session id "../session_abc123". Session ids must start with session_ and contain only letters and numbers.',
    });
  });
});
