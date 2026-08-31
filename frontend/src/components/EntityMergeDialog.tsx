import { useEffect, useMemo, useState } from 'react';
import type { EffectiveEntity } from '../types/domain';
import { policyLabel } from '../state/reviewState';
import { formatFrameRanges, mostProtectivePolicy, temporalConflictRangesForEntities } from '../state/entityRefinement';

interface Props {
  entities: EffectiveEntity[];
  open: boolean;
  onCancel: () => void;
  onInspectFrame?: (frame: number) => void;
  onMerge: (entities: EffectiveEntity[]) => Promise<void>;
}

export function EntityMergeDialog({ entities, onCancel, onInspectFrame, onMerge, open }: Props) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const conflictRanges = useMemo(() => temporalConflictRangesForEntities(entities), [entities]);
  const conflictFrame = conflictRanges[0]?.start;
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
          {conflictRanges.length ? (
            <div className="merge-disabled-explanation warning-text">
              <span>
                Selected objects coexist on source frames {formatFrameRanges(conflictRanges)}, so they cannot be the same identity.
              </span>
              {conflictFrame !== undefined && onInspectFrame ? (
                <button type="button" onClick={() => onInspectFrame(conflictFrame)}>
                  Inspect frames {formatFrameRanges(conflictRanges)}
                </button>
              ) : null}
            </div>
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
          <button type="button" className="primary" onClick={confirm} disabled={saving || entities.length < 2 || Boolean(conflictRanges.length)}>
            Merge selected
          </button>
        </div>
      </div>
    </section>
  );
}
