export interface ExplicitSessionRequest {
  requested: boolean;
  raw?: string;
  sessionId?: string;
  error?: string;
}

export function explicitSessionRequestFromSearch(search: string): ExplicitSessionRequest {
  const params = new URLSearchParams(search);
  const raw = params.get('session_id') ?? params.get('sessionId');
  if (raw === null) return { requested: false };
  const sessionId = raw?.trim();
  if (!sessionId) {
    return {
      error: 'A review session id was requested, but the value was empty.',
      raw,
      requested: true,
    };
  }
  if (!/^session_[A-Za-z0-9]+$/.test(sessionId)) {
    return {
      error: `Invalid review session id "${sessionId}". Session ids must start with session_ and contain only letters and numbers.`,
      raw,
      requested: true,
    };
  }
  return { raw, requested: true, sessionId };
}
