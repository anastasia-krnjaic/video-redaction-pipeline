import { useMemo, useState } from 'react';
import type { EffectiveEntity } from '../types/domain';

interface Props {
  source: EffectiveEntity | null;
  entities: EffectiveEntity[];
  open: boolean;
  onCancel: () => void;
  onMerge: (source: EffectiveEntity, target: EffectiveEntity) => Promise<void>;
}

export function EntityMergeDialog({ entities, onCancel, onMerge, open, source }: Props) {
  const candidates = useMemo(() => (source ? mergeCandidates(source, entities) : []), [entities, source]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  if (!open || !source) return null;
  const selected = candidates.find((candidate) => candidate.entity_id === (selectedId || candidates[0]?.entity_id));

  const confirm = async () => {
    if (!selected) {
      setError('Choose one merge candidate first.');
      return;
    }
    setSaving(true);
    setError('');
    try {
      await onMerge(source, selected);
      setSelectedId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="merge-title">
      <div className="modal-card large">
        <div className="case-header">
          <div>
            <h2 id="merge-title">Merge review objects</h2>
            <p className="case-subtitle">Suggested candidates are review-only groupings, not identity recognition.</p>
          </div>
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
        </div>
        <p className="task-title">Merge {source.label} with...</p>
        <div className="merge-candidates">
          {candidates.length ? (
            candidates.map((candidate) => {
              const selectedCandidate = candidate.entity_id === (selectedId || candidates[0]?.entity_id);
              return (
                <button
                  type="button"
                  key={candidate.entity_id}
                  className={`merge-card ${selectedCandidate ? 'selected' : ''}`}
                  onClick={() => setSelectedId(candidate.entity_id)}
                >
                  {candidate.representative_thumbnail_path ? (
                    <img src={`/${candidate.representative_thumbnail_path}`} alt={`${candidate.label} thumbnail`} />
                  ) : (
                    <span className="thumbnail-placeholder" />
                  )}
                  <span>
                    <strong>{candidate.label}</strong>
                    <span className="task-meta">
                      source {candidate.first_visible_source_frame ?? '?'}-{candidate.last_visible_source_frame ?? '?'} · gap{' '}
                      {entityGap(source, candidate)} frames · {similarityLabel(source, candidate)} similarity
                    </span>
                  </span>
                </button>
              );
            })
          ) : (
            <p className="muted">No safe merge candidates are available.</p>
          )}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="button-row">
          <button type="button" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
          <button type="button" className="primary" onClick={confirm} disabled={saving || !selected}>
            Merge selected
          </button>
        </div>
      </div>
    </section>
  );
}

export function mergeCandidates(source: EffectiveEntity, entities: EffectiveEntity[]): EffectiveEntity[] {
  return entities
    .filter((entity) => entity.entity_id !== source.entity_id)
    .filter((entity) => entity.category === source.category)
    .filter((entity) => !entitiesOverlap(source, entity))
    .sort((left, right) => entityGap(source, left) - entityGap(source, right))
    .slice(0, 12);
}

export function entitiesOverlap(left: EffectiveEntity, right: EffectiveEntity): boolean {
  const l1 = left.first_visible_source_frame;
  const l2 = left.last_visible_source_frame;
  const r1 = right.first_visible_source_frame;
  const r2 = right.last_visible_source_frame;
  if ([l1, l2, r1, r2].some((value) => typeof value !== 'number')) return false;
  return Math.max(l1 as number, r1 as number) <= Math.min(l2 as number, r2 as number);
}

export function entityGap(left: EffectiveEntity, right: EffectiveEntity): number {
  if (entitiesOverlap(left, right)) return 0;
  const l1 = left.first_visible_source_frame;
  const l2 = left.last_visible_source_frame;
  const r1 = right.first_visible_source_frame;
  const r2 = right.last_visible_source_frame;
  if ([l1, l2, r1, r2].some((value) => typeof value !== 'number')) return 999999;
  return Math.max(0, Math.max(l1 as number, r1 as number) - Math.min(l2 as number, r2 as number));
}

function similarityLabel(source: EffectiveEntity, candidate: EffectiveEntity): string {
  const suggestion = (source.possible_duplicate_suggestions || []).find((item) => {
    const childTrack = String(item.child_track_id || '');
    const parentTrack = String(item.parent_track_id || '');
    const candidateTracks = candidate.member_detector_tracker_ids || candidate.automatic_source_track_ids || [];
    return candidateTracks.includes(childTrack) || candidateTracks.includes(parentTrack);
  });
  const score = Number(suggestion?.score ?? (entityGap(source, candidate) <= 30 ? 0.65 : 0.45));
  return score >= 0.8 ? 'High' : score >= 0.5 ? 'Medium' : 'Low';
}
