import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { EffectiveEntity, EntityPolicy } from '../types/domain';
import { policyLabel } from '../state/reviewState';
import {
  childTrackletCount,
  entityPurity,
  policyAllowed,
  policyDisabledReason,
  PURITY_LIKELY_MIXED,
} from '../state/entityRefinement';

export interface EntityPreviewRequestOptions {
  force?: boolean;
  manual?: boolean;
}

export type EntityPreviewImageStatus = 'loading' | 'loaded' | 'error';

export interface RepresentativeCrop {
  bbox_xyxy?: number[];
  crop_role?: string;
  path?: string;
  raw_tracklet_id?: string;
  source_frame?: number;
}

interface EntityCropButtonProps {
  compact?: boolean;
  crop: RepresentativeCrop;
  entity: EffectiveEntity;
  imageKey?: string;
  onImageStatusChange?: (imageKey: string, status: EntityPreviewImageStatus) => void;
  onJump: (entity: EffectiveEntity) => void;
  onRetryPreview?: () => void;
}

interface Props {
  entity: EffectiveEntity;
  onJump: (entity: EffectiveEntity) => void;
  onPolicy: (entity: EffectiveEntity, policy: EntityPolicy) => void;
  onLoadPreviews?: (entity: EffectiveEntity, options?: EntityPreviewRequestOptions) => void;
  onSelect: (checked: boolean) => void;
  onSplitMixed?: (entity: EffectiveEntity) => void;
  onSplitTracklet?: (entity: EffectiveEntity, trackletId: string) => void;
  pending?: boolean;
  previewError?: string;
  previewPending?: boolean;
  previewRoot?: RefObject<Element>;
  selected: boolean;
}

const policies: EntityPolicy[] = ['redact', 'keep_visible', 'not_sensitive', 'review_required'];

export function EntityCard({
  entity,
  onJump,
  onLoadPreviews,
  onPolicy,
  onSelect,
  onSplitMixed,
  onSplitTracklet,
  pending,
  previewError,
  previewPending,
  previewRoot,
  selected,
}: Props) {
  const cardRef = useRef<HTMLElement>(null);
  const policy = (entity.current_policy || entity.privacy_action || entity.default_privacy_policy || 'review_required') as EntityPolicy;
  const observations = Number(entity.evidence_summary?.review_observation_count || entity.evidence_summary?.region_count || 0);
  const childCount = childTrackletCount(entity);
  const duration = durationText(entity);
  const purity = entityPurity(entity);
  const crops = representativeCrops(entity);
  const canFocus = crops.length > 0 || Boolean(representativeBBox(entity) && typeof entity.representative_source_frame === 'number');
  const rawTracklets = Array.isArray(entity.raw_tracklets) ? entity.raw_tracklets : [];
  const previewStatus = previewStatusValue(entity);
  const shouldAutoLoadPreview = previewStatus === 'NOT_PREPARED' && !crops.length && !previewPending && !previewError && Boolean(onLoadPreviews);

  useEffect(() => {
    if (!shouldAutoLoadPreview || !onLoadPreviews) return undefined;
    const card = cardRef.current;
    if (!card || typeof IntersectionObserver === 'undefined') {
      onLoadPreviews(entity);
      return undefined;
    }

    let requested = false;
    const observer = new IntersectionObserver(
      (entries) => {
        if (requested || !entries.some((entry) => entry.isIntersecting || entry.intersectionRatio > 0)) return;
        requested = true;
        onLoadPreviews(entity);
        observer.disconnect();
      },
      { root: previewRoot?.current || null, rootMargin: '120px 0px', threshold: 0.01 },
    );
    observer.observe(card);
    return () => observer.disconnect();
  }, [entity, onLoadPreviews, previewRoot, shouldAutoLoadPreview]);

  const requestPreviewAfterExpansion = () => {
    if (shouldAutoLoadPreview) onLoadPreviews?.(entity);
  };

  return (
    <article
      ref={cardRef}
      className={`entity-card purity-${purity.toLowerCase()}`}
      data-entity-id={entity.entity_id}
      data-category={entity.category}
      data-purity={purity}
      aria-busy={pending ? 'true' : 'false'}
    >
      <label className="entity-select">
        <input type="checkbox" checked={selected} onChange={(event) => onSelect(event.currentTarget.checked)} />
        <span className="visually-hidden">Select {entity.label}</span>
      </label>
      <div className="thumbnail-strip" aria-label={`${entity.label} representative crops`}>
        {crops.length ? (
          crops.map((crop, index) => (
            <EntityCropButton
              key={`${crop.path || 'crop'}-${crop.source_frame || index}`}
              crop={crop}
              entity={entity}
              onJump={onJump}
              onRetryPreview={onLoadPreviews ? () => onLoadPreviews(entity, { force: true, manual: true }) : undefined}
            />
          ))
        ) : previewError && onLoadPreviews ? (
          <button
            type="button"
            className="thumbnail-button preview-load-button preview-retry-button"
            onClick={() => onLoadPreviews(entity, { force: true, manual: true })}
            disabled={previewPending}
            title={previewError}
          >
            <span className="thumbnail-placeholder">{previewPending ? 'Retrying preview' : 'Retry preview'}</span>
          </button>
        ) : previewStatus === 'NOT_PREPARED' ? (
          <span className="thumbnail-placeholder">
            {previewPending ? 'Loading previews' : 'Preview will load when visible'}
          </span>
        ) : previewPending ? (
          <span className="thumbnail-placeholder">Loading previews</span>
        ) : (
          <span className="thumbnail-placeholder">No reliable representative crop</span>
        )}
      </div>
      <div className="entity-copy">
        <div className="task-title">
          {entity.label}
          <span className="badge entity-purity-badge">{purityLabel(purity)}</span>
        </div>
        <div className="task-meta">
          {entity.category === 'face' ? 'Person' : 'Plate'} · source {entity.first_visible_source_frame ?? '?'}-
          {entity.last_visible_source_frame ?? '?'} · {duration}
        </div>
        <p className="muted">
          {observations} observations · {childCount} child tracklets · {entity.review_entity_state || 'review object'}
        </p>
        {!canFocus ? (
          <p className={previewError ? 'error-text' : 'muted'}>
            {previewError
              ? `Preview failed: ${previewError}`
              : previewStatus === 'NOT_PREPARED'
                ? previewPending
                  ? 'Preparing representative crops'
                  : 'Representative crops load automatically as this card enters view'
                : 'No reliable representative crop'}
          </p>
        ) : null}
      </div>
      <div className="entity-actions">
        <div className="policy-segment" role="group" aria-label={`${entity.label} privacy action`}>
          {policies.map((item) => {
            const allowed = policyAllowed(entity, item);
            return (
              <button
                key={item}
                type="button"
                data-policy={item}
                aria-pressed={policy === item}
                disabled={Boolean(pending) || !allowed}
                title={!allowed ? policyDisabledReason(entity, item) : undefined}
                onClick={() => onPolicy(entity, item)}
              >
                {policyLabel(item, entity.category)}
              </button>
            );
          })}
        </div>
        {purity === PURITY_LIKELY_MIXED && onSplitMixed ? (
          <button type="button" className="primary split-mixed-action" onClick={() => onSplitMixed(entity)} disabled={pending}>
            Split mixed entity
          </button>
        ) : null}
      </div>
      {purity === PURITY_LIKELY_MIXED ? (
        <div className="entity-card-notices">
          <p className="warning-text">MIXED - contains conflicting simultaneous tracklets</p>
        </div>
      ) : null}
      <div className="entity-operation-status" data-testid="entity-operation-status" role="status" aria-live="polite">
        {pending ? (
          <>
            <span className="inline-spinner" aria-hidden="true" />
            <span className="visually-hidden">Updating reviewed media for {entity.label}</span>
          </>
        ) : null}
      </div>
      <details
        className="technical"
        onToggle={(event) => {
          if (event.currentTarget.open) requestPreviewAfterExpansion();
        }}
      >
        <summary>Technical details</summary>
        <pre>
          {JSON.stringify(
            {
              entity_id: entity.entity_id,
              state: entity.review_entity_state,
              child_fragments: entity.attached_fragment_track_ids || [],
              tracks: entity.member_detector_tracker_ids || entity.automatic_source_track_ids || [],
              raw_tracklets: rawTracklets,
              provenance: entity.created_provenance,
            },
            null,
            2,
          )}
        </pre>
        {rawTracklets.length && onSplitTracklet ? (
          <div className="tracklet-list">
            {rawTracklets.map((tracklet) => {
              const trackletId = String(tracklet.tracklet_id || '');
              return (
                <button type="button" key={trackletId} onClick={() => onSplitTracklet(entity, trackletId)}>
                  Split out {trackletId}
                </button>
              );
            })}
          </div>
        ) : null}
      </details>
    </article>
  );
}

export function EntityCropButton({
  compact,
  crop,
  entity,
  imageKey,
  onImageStatusChange,
  onJump,
  onRetryPreview,
}: EntityCropButtonProps) {
  const path = typeof crop.path === 'string' ? crop.path : '';
  const statusKey = imageKey || `${entity.entity_id}:${path}:${crop.source_frame ?? ''}`;
  const [imageStatus, setImageStatus] = useState<EntityPreviewImageStatus>(path ? 'loading' : 'error');
  const [retryVersion, setRetryVersion] = useState(0);
  const sourceFrame = crop.source_frame ?? entity.representative_source_frame ?? '?';
  const canRetry = Boolean(path && onRetryPreview && imageStatus === 'error');
  const canJump = Boolean(path && imageStatus === 'loaded');
  const imageSrc = path ? `/${path}${retryVersion ? `${path.includes('?') ? '&' : '?'}retry=${retryVersion}` : ''}` : '';

  useEffect(() => {
    setImageStatus(path ? 'loading' : 'error');
  }, [path, crop.source_frame]);

  useEffect(() => {
    onImageStatusChange?.(statusKey, imageStatus);
  }, [imageStatus, onImageStatusChange, statusKey]);

  const handleClick = () => {
    if (canRetry) {
      setRetryVersion((version) => version + 1);
      setImageStatus('loading');
      onRetryPreview?.();
      return;
    }
    if (canJump) onJump(cropEntityTarget(entity, crop));
  };

  const placeholderText = !path
    ? 'No reliable representative crop'
    : imageStatus === 'error'
      ? canRetry
        ? 'Preview image unavailable. Retry preview'
        : 'Preview image unavailable'
      : 'Loading preview';
  const className = [
    'thumbnail-button',
    compact ? 'thumbnail-button-compact' : '',
    `image-${imageStatus}`,
    canRetry ? 'preview-retry-button' : '',
  ].filter(Boolean).join(' ');

  return (
    <button
      type="button"
      className={className}
      aria-busy={imageStatus === 'loading' ? 'true' : 'false'}
      aria-disabled={!canJump && !canRetry ? 'true' : undefined}
      aria-label={
        canRetry
          ? `Retry preview for ${entity.label}`
          : canJump
            ? `Jump to ${entity.label} source frame ${sourceFrame}`
            : `${entity.label} preview loading`
      }
      onClick={handleClick}
      title={
        canRetry
          ? `Retry preview for ${entity.label}`
          : canJump
            ? `Jump to source frame ${sourceFrame}`
            : placeholderText
      }
    >
      {path ? (
        <img
          src={imageSrc}
          alt=""
          aria-hidden="true"
          draggable={false}
          onLoad={() => setImageStatus('loaded')}
          onError={() => setImageStatus('error')}
        />
      ) : null}
      {imageStatus !== 'loaded' ? (
        <span className="thumbnail-placeholder thumbnail-image-state">
          {imageStatus === 'loading' ? <span className="inline-spinner" aria-hidden="true" /> : null}
          <span>{placeholderText}</span>
        </span>
      ) : null}
      <span className="crop-frame">F{sourceFrame}</span>
    </button>
  );
}

export function representativeBBox(entity: EffectiveEntity): number[] | null {
  if (Array.isArray(entity.representative_bbox) && entity.representative_bbox.length === 4) return entity.representative_bbox;
  const quality = entity.representative_quality as Record<string, unknown> | undefined;
  const bbox = quality?.bbox_xyxy;
  return Array.isArray(bbox) && bbox.length === 4 ? bbox : null;
}

export function representativeCrops(entity: EffectiveEntity): RepresentativeCrop[] {
  const status = previewStatusValue(entity);
  const crops = Array.isArray(entity.representative_crops)
    ? entity.representative_crops
      .map((crop) => crop as RepresentativeCrop)
      .filter((crop) => crop.path && typeof crop.source_frame === 'number')
    : [];
  if (crops.length) return crops.slice(0, 3);
  if (entity.preview_status && status !== 'READY') return [];
  if (entity.representative_thumbnail_path && typeof entity.representative_source_frame === 'number') {
    return [{
      bbox_xyxy: representativeBBox(entity) || undefined,
      crop_role: 'best_middle',
      path: entity.representative_thumbnail_path,
      source_frame: entity.representative_source_frame,
    }];
  }
  return [];
}

export function previewStatusValue(entity: EffectiveEntity): string {
  const status = entity.preview_status?.status;
  return typeof status === 'string' ? status : 'NOT_PREPARED';
}

export function cropEntityTarget(entity: EffectiveEntity, crop: RepresentativeCrop): EffectiveEntity {
  return {
    ...entity,
    representative_bbox: crop.bbox_xyxy || entity.representative_bbox,
    representative_source_frame: typeof crop.source_frame === 'number' ? crop.source_frame : entity.representative_source_frame,
  };
}

function purityLabel(purity: string): string {
  if (purity === 'LIKELY_MIXED') return 'Likely mixed';
  if (purity === 'LOW_QUALITY') return 'Low quality';
  if (purity === 'UNVERIFIED') return 'Unverified';
  return 'Pure';
}

function durationText(entity: EffectiveEntity): string {
  const start = entity.first_visible_source_frame;
  const end = entity.last_visible_source_frame;
  if (typeof start !== 'number' || typeof end !== 'number') return 'duration unknown';
  return `${Math.max(1, end - start + 1)} frames`;
}
