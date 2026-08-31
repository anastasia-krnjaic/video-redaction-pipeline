import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { AuditEvent, Entity } from '../types/api';
import type { CaseId, EffectiveEntity, EntityFilter, EntityPolicy } from '../types/domain';
import { effectiveEntities, filteredEntities } from '../state/reviewState';
import {
  candidateAcceptEligibility,
  candidateConflictRanges,
  candidateEntityIds,
  candidateExplanation,
  childTrackletCount,
  CANDIDATE_POSSIBLE,
  CANDIDATE_REJECT_TEMPORAL_CONFLICT,
  firstCandidateConflictFrame,
  formatFrameRanges,
  mergeSelectedEligibility,
  policyAllowed,
  sectionEntities,
  temporalConflictRangesForEntities,
  visibleCandidateMatches,
} from '../state/entityRefinement';
import {
  EntityCard,
  EntityCropButton,
  cropEntityTarget,
  previewStatusValue,
  representativeCrops,
  type EntityPreviewImageStatus,
  type EntityPreviewRequestOptions,
  type RepresentativeCrop,
} from './EntityCard';

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
  events?: AuditEvent[];
  open: boolean;
  overrides: Record<string, EffectiveEntity>;
  pendingEntityIds?: Set<string>;
  previewErrorsByEntity?: Record<string, string>;
  pendingPreviewEntityIds?: Set<string>;
  relatedTaskCountByEntity?: Record<string, number>;
  unresolvedCount: number;
  onClose: () => void;
  onAcceptCandidate?: (candidate: Record<string, unknown>) => void;
  onBulkPolicy: (entities: EffectiveEntity[], policy: EntityPolicy) => void;
  onFilterChange: (filter: EntityFilter) => void;
  onInspectFrame?: (frame: number) => void;
  onJump: (entity: EffectiveEntity) => void;
  onLoadPreviews?: (entity: EffectiveEntity, options?: EntityPreviewRequestOptions) => void;
  onMergeSelected: (entities: EffectiveEntity[]) => void;
  onPolicy: (entity: EffectiveEntity, policy: EntityPolicy) => void;
  onRejectCandidate?: (candidate: Record<string, unknown>) => void;
  onSplitMixed?: (entity: EffectiveEntity) => void;
  onSplitTracklet?: (entity: EffectiveEntity, trackletId: string) => void;
}

interface CandidateRow {
  candidate: Record<string, unknown>;
  entities: EffectiveEntity[];
}

export function EntityPanel({
  caseId,
  entities,
  entityFilter,
  events = [],
  onAcceptCandidate,
  onClose,
  onBulkPolicy,
  onFilterChange,
  onInspectFrame,
  onJump,
  onLoadPreviews,
  onMergeSelected,
  onPolicy,
  onRejectCandidate,
  onSplitMixed,
  onSplitTracklet,
  open,
  overrides,
  pendingEntityIds,
  pendingPreviewEntityIds,
  previewErrorsByEntity = {},
  relatedTaskCountByEntity = {},
  unresolvedCount,
}: Props) {
  const cardRef = useRef<HTMLDivElement>(null);
  const scrollBodyRef = useRef<HTMLDivElement>(null);
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
  const sections = useMemo(() => sectionEntities(visible), [visible]);
  const candidateRows = useMemo(() => candidateRowsForEntities(visible, effective, events), [effective, events, visible]);
  const visibleIds = useMemo(() => new Set(visible.map((entity) => entity.entity_id)), [visible]);
  const selectedVisible = visible.filter((entity) => selectedIds.has(entity.entity_id));
  const selectedEffective = effective.filter((entity) => selectedIds.has(entity.entity_id));
  const mergeEligibility = mergeSelectedEligibility(selectedEffective);
  const selectedConflictRanges = temporalConflictRangesForEntities(selectedEffective);
  const selectedConflictFrame = selectedConflictRanges[0]?.start;
  const selectedPolicyAllowed = (policy: EntityPolicy) =>
    selectedVisible.length > 0 && selectedVisible.every((entity) => policyAllowed(entity, policy));
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
        <div className="modal-scroll-body" ref={scrollBodyRef}>
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
            <button type="button" onClick={() => bulk('keep_visible')} disabled={!selectedPolicyAllowed('keep_visible')}>
              Bulk Keep visible
            </button>
            <button type="button" onClick={() => bulk('not_sensitive')} disabled={!selectedPolicyAllowed('not_sensitive')}>
              Bulk Not a person/plate
            </button>
            <button type="button" onClick={() => bulk('review_required')} disabled={!selectedVisible.length}>
              Bulk Needs review
            </button>
          </div>
          {!mergeEligibility.enabled && selectedEffective.length >= 2 ? (
            <div className={selectedConflictRanges.length ? 'merge-disabled-explanation warning-text' : 'merge-disabled-explanation muted'}>
              <span>{mergeEligibility.reason}</span>
              {selectedConflictFrame !== undefined && onInspectFrame ? (
                <button type="button" onClick={() => onInspectFrame(selectedConflictFrame)}>
                  Inspect frames {formatFrameRanges(selectedConflictRanges)}
                </button>
              ) : null}
            </div>
          ) : null}
          {candidateRows.length ? (
            <section className="entity-section candidate-group-section">
              <h3>Candidate groups</h3>
              <div className="candidate-group-list">
                {candidateRows.map((row) => (
                  <CandidateComparisonCard
                    key={String(row.candidate.candidate_id)}
                    row={row}
                    onAcceptCandidate={onAcceptCandidate}
                    onInspectFrame={onInspectFrame}
                    onJump={onJump}
                    onLoadPreviews={onLoadPreviews}
                    onRejectCandidate={onRejectCandidate}
                    pendingPreviewEntityIds={pendingPreviewEntityIds}
                    previewErrorsByEntity={previewErrorsByEntity}
                    previewRoot={scrollBodyRef}
                  />
                ))}
              </div>
            </section>
          ) : null}
          <div className="object-list">
            {sections.length ? (
              sections.map((section) => (
                <section className="entity-section" key={section.title}>
                  <h3>
                    <span>{section.title}</span>
                    <span className="section-count">{section.entities.length}</span>
                  </h3>
                  {section.entities.map((entity) => (
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
                      onLoadPreviews={onLoadPreviews}
                      onPolicy={(target, policy) => {
                        if (!confirmEntityPolicyChange([target], policy, relatedTaskCountByEntity)) return;
                        onPolicy(target, policy);
                      }}
                      onSplitMixed={onSplitMixed}
                      onSplitTracklet={onSplitTracklet}
                      previewError={previewErrorsByEntity[entity.entity_id]}
                      previewPending={pendingPreviewEntityIds?.has(entity.entity_id)}
                      previewRoot={scrollBodyRef}
                    />
                  ))}
                </section>
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

function candidateRowsForEntities(
  visible: EffectiveEntity[],
  effective: EffectiveEntity[],
  events: AuditEvent[],
): CandidateRow[] {
  const byId = new Map(effective.map((entity) => [entity.entity_id, entity]));
  const seen = new Set<string>();
  const rows: CandidateRow[] = [];
  for (const entity of visible) {
    for (const candidate of visibleCandidateMatches(entity, events)) {
      const candidateId = String(candidate.candidate_id || '');
      if (!candidateId || seen.has(candidateId)) continue;
      const candidateEntities = candidateEntityIds(candidate)
        .map((id) => byId.get(id))
        .filter((item): item is EffectiveEntity => Boolean(item));
      if (candidateEntities.length < 2) continue;
      seen.add(candidateId);
      rows.push({ candidate, entities: candidateEntities });
    }
  }
  return rows.sort((left, right) => candidateSortRank(left.candidate) - candidateSortRank(right.candidate));
}

interface CandidateComparisonCardProps {
  onAcceptCandidate?: (candidate: Record<string, unknown>) => void;
  onInspectFrame?: (frame: number) => void;
  onJump: (entity: EffectiveEntity) => void;
  onLoadPreviews?: (entity: EffectiveEntity, options?: EntityPreviewRequestOptions) => void;
  onRejectCandidate?: (candidate: Record<string, unknown>) => void;
  pendingPreviewEntityIds?: Set<string>;
  previewErrorsByEntity: Record<string, string>;
  previewRoot: RefObject<Element>;
  row: CandidateRow;
}

function CandidateComparisonCard({
  onAcceptCandidate,
  onInspectFrame,
  onJump,
  onLoadPreviews,
  onRejectCandidate,
  pendingPreviewEntityIds,
  previewErrorsByEntity,
  previewRoot,
  row,
}: CandidateComparisonCardProps) {
  const cardRef = useRef<HTMLElement>(null);
  const [imageStatuses, setImageStatuses] = useState<Record<string, EntityPreviewImageStatus>>({});
  const classification = String(row.candidate.classification || 'INSUFFICIENT_EVIDENCE');
  const acceptEligibility = candidateAcceptEligibility(row.candidate);
  const conflictRanges = candidateConflictRanges(row.candidate);
  const conflictFrame = firstCandidateConflictFrame(row.candidate);
  const evidenceReady = classification === CANDIDATE_POSSIBLE ? candidateEvidenceReady(row.entities, imageStatuses) : true;
  const visualEvidenceReason = 'Compare loaded visual evidence for both candidates before accepting. Similarity is only a triage signal.';
  const acceptDisabledReason = !acceptEligibility.enabled
    ? acceptEligibility.reason
    : !evidenceReady
      ? visualEvidenceReason
      : '';
  const acceptDisabled = !onAcceptCandidate || !acceptEligibility.enabled || !evidenceReady;
  const comparisonKey = row.entities
    .map((entity) => {
      const cropKey = representativeCrops(entity)
        .map((crop, index) => cropImageKey(entity, crop, index))
        .join(',');
      return [
        entity.entity_id,
        previewStatusValue(entity),
        cropKey,
        previewErrorsByEntity[entity.entity_id] || '',
        pendingPreviewEntityIds?.has(entity.entity_id) ? 'pending' : 'idle',
      ].join(':');
    })
    .join('|');

  const updateImageStatus = useCallback((imageKey: string, status: EntityPreviewImageStatus) => {
    setImageStatuses((current) => {
      if (current[imageKey] === status) return current;
      return { ...current, [imageKey]: status };
    });
  }, []);

  useEffect(() => {
    if (!onLoadPreviews) return undefined;
    const needsEvidence = row.entities.some((entity) => candidateNeedsPreviewRequest(
      entity,
      pendingPreviewEntityIds?.has(entity.entity_id),
      previewErrorsByEntity[entity.entity_id],
    ));
    if (!needsEvidence) return undefined;
    const card = cardRef.current;
    const requestEvidence = () => row.entities.forEach((entity) => onLoadPreviews(entity));
    if (!card || typeof IntersectionObserver === 'undefined') {
      requestEvidence();
      return undefined;
    }
    let requested = false;
    const observer = new IntersectionObserver(
      (entries) => {
        if (requested || !entries.some((entry) => entry.isIntersecting || entry.intersectionRatio > 0)) return;
        requested = true;
        requestEvidence();
        observer.disconnect();
      },
      { root: previewRoot.current || null, rootMargin: '140px 0px', threshold: 0.01 },
    );
    observer.observe(card);
    return () => observer.disconnect();
  }, [comparisonKey, onLoadPreviews, pendingPreviewEntityIds, previewErrorsByEntity, previewRoot, row.entities]);

  return (
    <article
      ref={cardRef}
      className={`candidate-group-card candidate-${classification.toLowerCase()}`}
      data-candidate-id={String(row.candidate.candidate_id || '')}
      data-evidence-ready={evidenceReady ? 'true' : 'false'}
    >
      <div className="candidate-group-copy">
        <div className="candidate-title-row">
          <strong>{row.entities.map((entity) => entity.label).join(' + ')}</strong>
          <span className={`badge candidate-status-badge candidate-status-${classification.toLowerCase()}`}>
            {candidateStatusLabel(classification)}
          </span>
        </div>
        <p className="task-meta">
          similarity {String(row.candidate.similarity ?? 'n/a')}
          {conflictRanges.length ? ` · conflict frames ${formatFrameRanges(conflictRanges)}` : ''}
        </p>
        <p className={classification === CANDIDATE_REJECT_TEMPORAL_CONFLICT ? 'warning-text' : 'muted'}>
          {candidateExplanation(row.candidate)}
        </p>
        {classification === CANDIDATE_POSSIBLE ? (
          <p className={evidenceReady ? 'muted candidate-score-note' : 'warning-text candidate-score-note'}>
            {evidenceReady
              ? 'Visual evidence is loaded for both sides; compare the target crops and source frames before accepting.'
              : visualEvidenceReason}
          </p>
        ) : null}
      </div>
      <div className="candidate-comparison-grid" aria-label={`${row.entities.map((entity) => entity.label).join(' and ')} visual evidence`}>
        {row.entities.slice(0, 2).map((entity) => (
          <CandidateEvidenceSide
            key={entity.entity_id}
            candidate={row.candidate}
            entity={entity}
            imageStatuses={imageStatuses}
            onImageStatusChange={updateImageStatus}
            onJump={onJump}
            onLoadPreviews={onLoadPreviews}
            pending={pendingPreviewEntityIds?.has(entity.entity_id)}
            previewError={previewErrorsByEntity[entity.entity_id]}
          />
        ))}
      </div>
      <div className="entity-actions candidate-actions">
        {conflictFrame !== null && onInspectFrame ? (
          <button type="button" onClick={() => onInspectFrame(conflictFrame)}>
            Inspect conflict frames {formatFrameRanges(conflictRanges)}
          </button>
        ) : null}
        <button
          type="button"
          className="primary"
          onClick={() => onAcceptCandidate?.(row.candidate)}
          disabled={acceptDisabled}
          title={acceptDisabledReason || undefined}
        >
          Accept merge candidate
        </button>
        <button type="button" onClick={() => onRejectCandidate?.(row.candidate)} disabled={!onRejectCandidate}>
          Reject candidate
        </button>
      </div>
      {acceptDisabledReason ? (
        <p className={classification === CANDIDATE_REJECT_TEMPORAL_CONFLICT ? 'warning-text candidate-disabled-reason' : 'muted candidate-disabled-reason'}>
          {acceptDisabledReason}
        </p>
      ) : null}
    </article>
  );
}

interface CandidateEvidenceSideProps {
  candidate: Record<string, unknown>;
  entity: EffectiveEntity;
  imageStatuses: Record<string, EntityPreviewImageStatus>;
  onImageStatusChange: (imageKey: string, status: EntityPreviewImageStatus) => void;
  onJump: (entity: EffectiveEntity) => void;
  onLoadPreviews?: (entity: EffectiveEntity, options?: EntityPreviewRequestOptions) => void;
  pending?: boolean;
  previewError?: string;
}

function CandidateEvidenceSide({
  candidate,
  entity,
  imageStatuses,
  onImageStatusChange,
  onJump,
  onLoadPreviews,
  pending,
  previewError,
}: CandidateEvidenceSideProps) {
  const crops = representativeCrops(entity);
  const sourceRange = candidateSourceRange(candidate, entity);
  const firstCrop = crops.find((crop, index) => imageStatuses[cropImageKey(entity, crop, index)] === 'loaded') || crops[0];
  const previewStatus = previewStatusValue(entity);
  const hasLoadedCrop = crops.some((crop, index) => imageStatuses[cropImageKey(entity, crop, index)] === 'loaded');
  const inspectTarget = firstCrop ? cropEntityTarget(entity, firstCrop) : entity;
  const observations = Number(entity.evidence_summary?.review_observation_count || entity.evidence_summary?.region_count || 0);
  const childCount = childTrackletCount(entity);

  return (
    <section className="candidate-evidence-side" aria-label={`${entity.label} candidate evidence`}>
      <div className="candidate-evidence-header">
        <div>
          <strong>{entity.label}</strong>
          <p className="task-meta">
            source {formatSingleFrameRange(sourceRange)} · {observations} observations · {childCount} child tracklets
          </p>
        </div>
        <button
          type="button"
          onClick={() => onJump(inspectTarget)}
          disabled={typeof inspectTarget.representative_source_frame !== 'number'}
        >
          Inspect {entity.label}
        </button>
      </div>
      <div className="candidate-evidence-crops" aria-label={`${entity.label} representative crops`}>
        {crops.length ? (
          crops.map((crop, index) => (
            <EntityCropButton
              compact
              key={`${crop.path || 'crop'}-${crop.source_frame || index}`}
              crop={crop}
              entity={entity}
              imageKey={cropImageKey(entity, crop, index)}
              onImageStatusChange={onImageStatusChange}
              onJump={onJump}
              onRetryPreview={onLoadPreviews ? () => onLoadPreviews(entity, { force: true, manual: true }) : undefined}
            />
          ))
        ) : previewError && onLoadPreviews ? (
          <button
            type="button"
            className="candidate-evidence-placeholder preview-retry-button"
            onClick={() => onLoadPreviews(entity, { force: true, manual: true })}
            disabled={pending}
            title={previewError}
          >
            {pending ? 'Retrying evidence' : 'Retry evidence'}
          </button>
        ) : (
          <span className="candidate-evidence-placeholder">
            {pending
              ? 'Preparing evidence'
              : previewStatus === 'NOT_PREPARED'
                ? 'Evidence will load automatically'
                : 'No reliable visual evidence'}
          </span>
        )}
      </div>
      <p className={previewError ? 'error-text candidate-evidence-state' : hasLoadedCrop ? 'success-text candidate-evidence-state' : 'muted candidate-evidence-state'}>
        {previewError ? `Preview failed: ${previewError}` : hasLoadedCrop ? 'Target crop loaded' : pending ? 'Loading target crop' : 'Awaiting visual evidence'}
      </p>
    </section>
  );
}

function candidateNeedsPreviewRequest(entity: EffectiveEntity, pending?: boolean, previewError?: string): boolean {
  return previewStatusValue(entity) === 'NOT_PREPARED'
    && !representativeCrops(entity).length
    && !pending
    && !previewError;
}

function candidateEvidenceReady(
  entities: EffectiveEntity[],
  imageStatuses: Record<string, EntityPreviewImageStatus>,
): boolean {
  return entities.slice(0, 2).every((entity) => {
    const crops = representativeCrops(entity);
    if (!crops.length) return false;
    return crops.some((crop, index) => imageStatuses[cropImageKey(entity, crop, index)] === 'loaded');
  });
}

function cropImageKey(entity: EffectiveEntity, crop: RepresentativeCrop, index: number): string {
  return `${entity.entity_id}:${crop.path || 'missing'}:${crop.source_frame ?? 'unknown'}:${index}`;
}

function candidateSourceRange(candidate: Record<string, unknown>, entity: EffectiveEntity): { start: number; end: number } | null {
  const intervals = candidate.source_intervals as Record<string, unknown> | undefined;
  const interval = intervals?.[entity.entity_id];
  if (Array.isArray(interval) && interval.length >= 2) {
    const start = Number(interval[0]);
    const end = Number(interval[1]);
    if (Number.isFinite(start) && Number.isFinite(end)) return { start: Math.trunc(start), end: Math.trunc(end) };
  }
  const start = Number(entity.first_visible_source_frame);
  const end = Number(entity.last_visible_source_frame);
  return Number.isFinite(start) && Number.isFinite(end) ? { start: Math.trunc(start), end: Math.trunc(end) } : null;
}

function formatSingleFrameRange(range: { start: number; end: number } | null): string {
  if (!range) return 'unknown';
  return range.start === range.end ? String(range.start) : `${range.start}-${range.end}`;
}

function candidateStatusLabel(classification: string): string {
  if (classification === 'POSSIBLE_CANDIDATE') return 'Possible merge';
  if (classification === 'REJECT_TEMPORAL_CONFLICT') return 'Temporal conflict';
  if (classification === 'REJECT_VISUAL_MISMATCH') return 'Visual mismatch';
  return 'Insufficient visual evidence';
}

function candidateSortRank(candidate: Record<string, unknown>): number {
  const classification = String(candidate.classification || 'INSUFFICIENT_EVIDENCE');
  if (classification === CANDIDATE_POSSIBLE) return 0;
  if (classification === CANDIDATE_REJECT_TEMPORAL_CONFLICT) return 1;
  if (classification === 'REJECT_VISUAL_MISMATCH') return 2;
  return 3;
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
