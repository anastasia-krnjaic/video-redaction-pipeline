import { useEffect, useMemo, useRef, useState } from 'react';
import type { Entity } from '../types/api';
import type { CaseId, EffectiveEntity, EntityFilter, EntityPolicy } from '../types/domain';
import { effectiveEntities, filteredEntities } from '../state/reviewState';
import { EntityCard } from './EntityCard';

const filters: Array<{ value: EntityFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'face', label: 'People' },
  { value: 'license_plate', label: 'Plates' },
  { value: 'needs_review', label: 'Needs review' },
  { value: 'unresolved', label: 'Unresolved evidence' },
];

interface Props {
  caseId: CaseId;
  entities: Entity[];
  entityFilter: EntityFilter;
  open: boolean;
  overrides: Record<string, EffectiveEntity>;
  pendingEntityIds?: Set<string>;
  relatedTaskCountByEntity?: Record<string, number>;
  unresolvedCount: number;
  onClose: () => void;
  onBulkPolicy: (entities: EffectiveEntity[], policy: EntityPolicy) => void;
  onFilterChange: (filter: EntityFilter) => void;
  onJump: (entity: EffectiveEntity) => void;
  onMergeSelected: (entities: EffectiveEntity[]) => void;
  onPolicy: (entity: EffectiveEntity, policy: EntityPolicy) => void;
}

export function EntityPanel({
  caseId,
  entities,
  entityFilter,
  onClose,
  onBulkPolicy,
  onFilterChange,
  onJump,
  onMergeSelected,
  onPolicy,
  open,
  overrides,
  pendingEntityIds,
  relatedTaskCountByEntity = {},
  unresolvedCount,
}: Props) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, open]);

  const effective = useMemo(() => effectiveEntities(entities, overrides), [entities, overrides]);
  const visible = useMemo(() => filteredEntities(effective, entityFilter), [effective, entityFilter]);
  const visibleIds = useMemo(() => new Set(visible.map((entity) => entity.entity_id)), [visible]);
  const selectedVisible = visible.filter((entity) => selectedIds.has(entity.entity_id));
  const selectedEffective = effective.filter((entity) => selectedIds.has(entity.entity_id));
  const mergeEligibility = mergeSelectedEligibility(selectedEffective);
  const bulk = (policy: EntityPolicy) => {
    if (!confirmEntityPolicyChange(selectedVisible, policy, relatedTaskCountByEntity)) return;
    onBulkPolicy(selectedVisible, policy);
    setSelectedIds(new Set());
  };

  if (!open) return null;

  return (
    <section
      className="modal open"
      aria-modal="true"
      role="dialog"
      aria-labelledby="entities-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal-card large entity-modal-card" ref={cardRef}>
        <div className="modal-sticky-header">
          <div>
            <h2 id="entities-title">Review objects</h2>
            <p className="case-subtitle">
              Review source-backed people and plates. These actions update reviewed media after backend validation.
            </p>
          </div>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="modal-scroll-body">
          <div className="entity-filter-row" role="tablist" aria-label={`${caseId} entity filters`}>
            {filters.map((filter) => (
              <button
                key={filter.value}
                type="button"
                className={filter.value === entityFilter ? 'active' : ''}
                onClick={() => onFilterChange(filter.value)}
              >
                {filter.label}
              </button>
            ))}
          </div>
          {entityFilter === 'unresolved' ? (
            <div className="technical">
              <strong>Unresolved evidence ({unresolvedCount})</strong>
              <p className="muted">Low-evidence unmatched fragments are preserved for audit and are not named as identities.</p>
            </div>
          ) : null}
          <div className="bulk-action-row">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={visible.length > 0 && visible.every((entity) => selectedIds.has(entity.entity_id))}
                onChange={(event) => {
                  if (event.currentTarget.checked) setSelectedIds(new Set([...selectedIds, ...visibleIds]));
                  else setSelectedIds(new Set([...selectedIds].filter((id) => !visibleIds.has(id))));
                }}
              />
              Select all visible
            </label>
            <span className="muted">{selectedVisible.length} selected</span>
            <button
              type="button"
              className="primary"
              onClick={() => onMergeSelected(selectedEffective)}
              disabled={!mergeEligibility.enabled}
              title={mergeEligibility.reason}
            >
              Merge selected ({selectedEffective.length})
            </button>
            <button type="button" onClick={() => bulk('redact')} disabled={!selectedVisible.length}>
              Bulk Redact
            </button>
            <button type="button" onClick={() => bulk('keep_visible')} disabled={!selectedVisible.length}>
              Bulk Keep visible
            </button>
            <button type="button" onClick={() => bulk('not_sensitive')} disabled={!selectedVisible.length}>
              Bulk Not a person/plate
            </button>
            <button type="button" onClick={() => bulk('review_required')} disabled={!selectedVisible.length}>
              Bulk Needs review
            </button>
          </div>
          {!mergeEligibility.enabled && selectedEffective.length >= 2 ? (
            <p className="muted">{mergeEligibility.reason}</p>
          ) : null}
          <div className="object-list">
            {visible.length ? (
              visible.map((entity) => (
                <EntityCard
                  key={entity.entity_id}
                  entity={entity}
                  pending={pendingEntityIds?.has(entity.entity_id)}
                  selected={selectedIds.has(entity.entity_id)}
                  onSelect={(checked) => {
                    const next = new Set(selectedIds);
                    if (checked) next.add(entity.entity_id);
                    else next.delete(entity.entity_id);
                    setSelectedIds(next);
                  }}
                  onJump={onJump}
                  onPolicy={(target, policy) => {
                    if (!confirmEntityPolicyChange([target], policy, relatedTaskCountByEntity)) return;
                    onPolicy(target, policy);
                  }}
                />
              ))
            ) : (
              <p className="muted">No review objects match this filter.</p>
            )}
          </div>
        </div>
        <div className="modal-sticky-footer">
          <span className="muted">
            Showing {visible.length} of {effective.length} review objects
          </span>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </section>
  );
}

function confirmEntityPolicyChange(
  entities: EffectiveEntity[],
  policy: EntityPolicy,
  relatedTaskCountByEntity: Record<string, number>,
): boolean {
  if (policy !== 'not_sensitive') return true;
  const affectedTasks = entities.reduce((sum, entity) => sum + Number(relatedTaskCountByEntity[entity.entity_id] || 0), 0);
  const label = entities.length === 1 ? entities[0].label : `${entities.length} objects`;
  const category = entities.length === 1 && entities[0].category === 'face' ? 'person' : entities.length === 1 ? 'plate' : 'object';
  const start = Math.min(...entities.map((entity) => Number(entity.first_visible_source_frame)).filter(Number.isFinite));
  const end = Math.max(...entities.map((entity) => Number(entity.last_visible_source_frame)).filter(Number.isFinite));
  const range = Number.isFinite(start) && Number.isFinite(end) ? ` across frames ${start}-${end}` : '';
  return window.confirm(
    `Remove redaction for ${label}${range}? This marks the ${category} as not sensitive and resolves ${affectedTasks} related review issue${affectedTasks === 1 ? '' : 's'}.`,
  );
}

function mergeSelectedEligibility(entities: EffectiveEntity[]): { enabled: boolean; reason: string } {
  if (entities.length < 2) return { enabled: false, reason: 'Select at least two same-category objects to merge.' };
  const cases = new Set(entities.map((entity) => entity.case_id));
  if (cases.size > 1) return { enabled: false, reason: 'Selected objects belong to different cases.' };
  const categories = new Set(entities.map((entity) => entity.category));
  if (categories.size > 1) return { enabled: false, reason: 'People and plates cannot be merged together.' };
  return { enabled: true, reason: 'Merge selected review objects.' };
}
