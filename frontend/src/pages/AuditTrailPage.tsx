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
          <h2>History</h2>
          <p className="case-subtitle">Read-only review history with technical identifiers behind details.</p>
        </div>
      </div>
      <div className="audit-list">
        {events.length ? (
          events.map((event) => (
            <article className="audit-card" key={event.event_id}>
              <strong>{humanEvent(event)}</strong>
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

function humanEvent(event: AuditEvent): string {
  const after = event.after_state || {};
  if (event.event_type === 'session_started') return 'Review session started';
  if (event.event_type === 'entity_policy_changed') {
    return `${String(after.label || after.display_name || 'Review object')} set to ${policyText(String(after.current_policy || after.privacy_action || 'review_required'), String(after.category || ''))}`;
  }
  if (event.event_type === 'manual_correction_saved' || event.event_type === 'correction_saved') {
    return `Manual ${categoryText(String(after.category || ''))} correction saved, frames ${String(after.source_frame_start ?? '?')}-${String(after.source_frame_end ?? '?')}`;
  }
  if (event.event_type === 'checkpoint_created') return `Checkpoint "${String(after.name || 'Checkpoint')}" created`;
  if (event.event_type === 'restore_checkpoint') return `Restored checkpoint "${String(after.name || 'Checkpoint')}"`;
  if (event.event_type === 'checkpoint_deleted') return `Checkpoint "${String(after.name || 'Checkpoint')}" deleted`;
  if (event.event_type === 'reviewed_media_materialization_started') return 'Latest reviewed video update started';
  if (event.event_type === 'reviewed_media_materialization_completed') return 'Latest reviewed video update completed';
  if (event.event_type === 'reviewed_media_materialization_failed') return 'Latest reviewed video update failed';
  if (event.event_type === 'entities_merged') return 'Merged review objects';
  if (event.event_type === 'correction_previewed') return 'Manual correction previewed';
  if (event.event_type === 'latest_reviewed_materialized') return 'Latest reviewed video materialized';
  if (event.event_type === 'case_finalized') return 'Final video materialized';
  return event.event_type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function policyText(policy: string, category: string): string {
  if (policy === 'redact') return 'Redact';
  if (policy === 'keep_visible') return 'Keep visible';
  if (policy === 'not_sensitive') return category === 'face' ? 'Not a person' : 'Not a plate';
  return 'Needs review';
}

function categoryText(category: string): string {
  if (category === 'face') return 'Face / person';
  if (category === 'license_plate') return 'Plate';
  if (category === 'other_sensitive') return 'Other sensitive region';
  return 'privacy';
}
