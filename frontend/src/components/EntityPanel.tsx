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
  pendingEntityId?: string;
  unresolvedCount: number;
  onClose: () => void;
  onFilterChange: (filter: EntityFilter) => void;
  onJump: (entity: EffectiveEntity) => void;
  onMerge: (entity: EffectiveEntity) => void;
  onPolicy: (entity: EffectiveEntity, policy: EntityPolicy) => void;
}

export function EntityPanel({
  caseId,
  entities,
  entityFilter,
  onClose,
  onFilterChange,
  onJump,
  onMerge,
  onPolicy,
  open,
  overrides,
  pendingEntityId,
  unresolvedCount,
}: Props) {
  if (!open) return null;
  const effective = effectiveEntities(entities, overrides);
  const visible = filteredEntities(effective, entityFilter);

  return (
    <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="entities-title">
      <div className="modal-card large">
        <div className="case-header">
          <div>
            <h2 id="entities-title">Review objects</h2>
            <p className="case-subtitle">Object actions affect reviewer policy and lineage only.</p>
          </div>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
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
        <div className="object-list">
          {visible.length ? (
            visible.map((entity) => (
              <EntityCard
                key={entity.entity_id}
                entity={entity}
                pending={pendingEntityId === entity.entity_id}
                onJump={onJump}
                onMerge={onMerge}
                onPolicy={onPolicy}
              />
            ))
          ) : (
            <p className="muted">No review objects match this filter.</p>
          )}
        </div>
      </div>
    </section>
  );
}
