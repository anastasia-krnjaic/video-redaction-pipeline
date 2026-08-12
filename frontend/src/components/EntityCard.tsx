import type { EffectiveEntity, EntityPolicy } from '../types/domain';
import { policyLabel } from '../state/reviewState';

interface Props {
  entity: EffectiveEntity;
  onJump: (entity: EffectiveEntity) => void;
  onPolicy: (entity: EffectiveEntity, policy: EntityPolicy) => void;
  onSelect: (checked: boolean) => void;
  pending?: boolean;
  selected: boolean;
}

const policies: EntityPolicy[] = ['redact', 'keep_visible', 'not_sensitive', 'review_required'];

export function EntityCard({ entity, onJump, onPolicy, onSelect, pending, selected }: Props) {
  const policy = (entity.current_policy || entity.privacy_action || entity.default_privacy_policy || 'review_required') as EntityPolicy;
  const observations = Number(entity.evidence_summary?.review_observation_count || entity.evidence_summary?.region_count || 0);
  const childCount = entity.member_detector_tracker_ids?.length || entity.automatic_source_track_ids?.length || 0;
  const duration = durationText(entity);
  const canFocus = Boolean(representativeBBox(entity) && typeof entity.representative_source_frame === 'number');

  return (
    <article className="entity-card" data-entity-id={entity.entity_id} data-category={entity.category} aria-busy={pending ? 'true' : 'false'}>
      <label className="entity-select">
        <input type="checkbox" checked={selected} onChange={(event) => onSelect(event.currentTarget.checked)} />
        <span className="visually-hidden">Select {entity.label}</span>
      </label>
      <button
        type="button"
        className="thumbnail-button"
        onClick={() => onJump(entity)}
        title={canFocus ? 'Jump to representative frame' : 'No representative geometry is available'}
        disabled={!canFocus}
      >
        {entity.representative_thumbnail_path ? (
          <img src={`/${entity.representative_thumbnail_path}`} alt={`${entity.label} representative thumbnail`} />
        ) : (
          <span className="thumbnail-placeholder">No preview available</span>
        )}
      </button>
      <div className="entity-copy">
        <div className="task-title">{entity.label}</div>
        <div className="task-meta">
          {entity.category === 'face' ? 'Person' : 'Plate'} · source {entity.first_visible_source_frame ?? '?'}-
          {entity.last_visible_source_frame ?? '?'} · {duration}
        </div>
        <p className="muted">
          {observations} observations · {childCount} child tracklets · {entity.review_entity_state || 'review object'}
        </p>
        {!canFocus ? <p className="muted">No representative geometry available.</p> : null}
      </div>
      <div className="entity-actions">
        <div className="policy-segment" role="group" aria-label={`${entity.label} privacy action`}>
          {policies.map((item) => (
            <button
              key={item}
              type="button"
              data-policy={item}
              aria-pressed={policy === item}
              disabled={pending && policy === item}
              onClick={() => onPolicy(entity, item)}
            >
              {policyLabel(item, entity.category)}
            </button>
          ))}
        </div>
      </div>
      <div className="entity-operation-status" data-testid="entity-operation-status" role="status" aria-live="polite">
        {pending ? (
          <>
            <span className="inline-spinner" aria-hidden="true" />
            <span className="visually-hidden">Updating Latest reviewed for {entity.label}</span>
          </>
        ) : null}
      </div>
      <details className="technical">
        <summary>Technical details</summary>
        <pre>
          {JSON.stringify(
            {
              entity_id: entity.entity_id,
              state: entity.review_entity_state,
              child_fragments: entity.attached_fragment_track_ids || [],
              tracks: entity.member_detector_tracker_ids || entity.automatic_source_track_ids || [],
              provenance: entity.created_provenance,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </article>
  );
}

function representativeBBox(entity: EffectiveEntity): number[] | null {
  if (Array.isArray(entity.representative_bbox) && entity.representative_bbox.length === 4) return entity.representative_bbox;
  const quality = entity.representative_quality as Record<string, unknown> | undefined;
  const bbox = quality?.bbox_xyxy;
  return Array.isArray(bbox) && bbox.length === 4 ? bbox : null;
}

function durationText(entity: EffectiveEntity): string {
  const start = entity.first_visible_source_frame;
  const end = entity.last_visible_source_frame;
  if (typeof start !== 'number' || typeof end !== 'number') return 'duration unknown';
  return `${Math.max(1, end - start + 1)} frames`;
}
