import { useEffect, useMemo, useState } from 'react';
import type { EffectiveEntity } from '../types/domain';
import { policyLabel } from '../state/reviewState';
import { mostProtectivePolicy } from '../state/entityRefinement';

interface Props {
  entities: EffectiveEntity[];
  open: boolean;
  onCancel: () => void;
  onMerge: (entities: EffectiveEntity[]) => Promise<void>;
}

export function EntityMergeDialog({ entities, onCancel, onMerge, open }: Props) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const hasOverlap = useMemo(() => anyOverlap(entities), [entities]);
  const policies = new Set(entities.map((entity) => entity.current_policy || entity.privacy_action || entity.default_privacy_policy || 'review_required'));
  const resultingPolicy = mostProtectivePolicy(entities);

  useEffect(() => {
    if (open) {
      setError('');
      setSaving(false);
    }
  }, [open]);

  if (!open) return null;

  const confirm = async () => {
    setSaving(true);
    setError('');
    try {
      await onMerge(entities);
      onCancel();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="merge-title">
      <div className="modal-card large merge-modal-card">
        <div className="modal-sticky-header">
          <div>
            <h2 id="merge-title">Merge selected objects</h2>
            <p className="case-subtitle">This records a reviewer-confirmed review grouping, not an identity claim.</p>
          </div>
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
        </div>
        <div className="modal-scroll-body">
          {hasOverlap ? (
            <p className="warning-text">Selected objects have overlapping source-frame spans. Resolve the conflict before merging.</p>
          ) : null}
          {policies.size > 1 ? (
            <p className="warning-text">Selected policies conflict. The merged object will inherit the most protective policy.</p>
          ) : null}
          <div className="merge-candidates">
            {entities.map((entity) => (
              <article className="merge-card selected" key={entity.entity_id}>
                {entity.representative_thumbnail_path ? (
                  <img src={`/${entity.representative_thumbnail_path}`} alt={`${entity.label} thumbnail`} />
                ) : (
                  <span className="thumbnail-placeholder">No preview available</span>
                )}
                <span>
                  <strong>{entity.label}</strong>
                  <span className="task-meta">
                    source {entity.first_visible_source_frame ?? '?'}-{entity.last_visible_source_frame ?? '?'} ·{' '}
                    {policyLabel(String(entity.current_policy || entity.privacy_action || entity.default_privacy_policy), entity.category)}
                  </span>
                </span>
              </article>
            ))}
          </div>
          <p className="muted">Resulting policy: {policyLabel(String(resultingPolicy), entities[0]?.category)}</p>
          {error ? <p className="error-text">{error}</p> : null}
        </div>
        <div className="modal-sticky-footer">
          <button type="button" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
          <button type="button" className="primary" onClick={confirm} disabled={saving || entities.length < 2 || hasOverlap}>
            Merge selected
          </button>
        </div>
      </div>
    </section>
  );
}

function anyOverlap(entities: EffectiveEntity[]): boolean {
  for (let index = 0; index < entities.length; index += 1) {
    for (let other = index + 1; other < entities.length; other += 1) {
      if (overlap(entities[index], entities[other])) return true;
    }
  }
  return false;
}

function overlap(left: EffectiveEntity, right: EffectiveEntity): boolean {
  const l1 = left.first_visible_source_frame;
  const l2 = left.last_visible_source_frame;
  const r1 = right.first_visible_source_frame;
  const r2 = right.last_visible_source_frame;
  if ([l1, l2, r1, r2].some((value) => typeof value !== 'number')) return false;
  return Math.max(l1 as number, r1 as number) <= Math.min(l2 as number, r2 as number);
}
