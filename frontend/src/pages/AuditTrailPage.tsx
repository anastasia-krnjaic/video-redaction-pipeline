import type { AuditEvent, WorkspaceSummary } from '../types/api';

interface Props {
  events: AuditEvent[];
  workspace: WorkspaceSummary;
}

export function AuditTrailPage({ events, workspace }: Props) {
  const rawSignals = Array.isArray(workspace.raw_signals.raw_signals)
    ? workspace.raw_signals.raw_signals.length
    : 0;
  return (
    <section className="case-panel active">
      <div className="case-header">
        <div>
          <h2>Audit Trail</h2>
          <p className="case-subtitle">Append-only session events with technical identifiers behind details.</p>
        </div>
      </div>
      <div className="audit-list">
        {events.length ? (
          events.map((event) => (
            <article className="audit-card" key={event.event_id}>
              <strong>{humanEvent(event.event_type)}</strong>
              <p className="task-meta">
                {event.timestamp_utc} · {event.case_id || 'workspace'}
              </p>
              {event.note ? <p>{event.note}</p> : null}
              <details>
                <summary>Technical details</summary>
                <pre>{JSON.stringify(event, null, 2)}</pre>
              </details>
            </article>
          ))
        ) : (
          <p className="muted">No session events yet.</p>
        )}
      </div>
      <details className="technical">
        <summary>Raw machine audit signals ({rawSignals})</summary>
        <pre>{JSON.stringify(workspace.raw_signals, null, 2)}</pre>
      </details>
    </section>
  );
}

function humanEvent(eventType: string): string {
  return eventType
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
