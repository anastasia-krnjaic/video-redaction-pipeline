import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent, RefObject } from 'react';
import type { Attribution, Case, Correction, Task } from '../types/api';
import type { CaseId, CorrectionCheckpointDraft, CorrectionDraft, CorrectionKeyframeDraft, ManualRedactionCategory } from '../types/domain';
import { MANUAL_REDACTION_CATEGORY_OPTIONS } from '../types/domain';
import { clamp } from '../utils/frame';

interface Props {
  actor: Attribution | Record<string, unknown>;
  caseId: CaseId;
  configurationFreezeHash?: string | null;
  currentSourceFrame: number;
  draft: CorrectionDraft;
  previewInvalidated: boolean;
  previewReady: boolean;
  selectedTask?: Task | null;
  sourceVideoHash?: string | null;
  onDiscard: () => void;
  onDraftChange: (draft: CorrectionDraft) => void;
  onGoToSourceFrame: (sourceFrame: number) => void;
  onPreview: (correction: Correction) => Promise<void>;
  onSave: (correction: Correction, reason: string) => Promise<void>;
}

export function CorrectionEditor({
  actor,
  caseId,
  configurationFreezeHash,
  currentSourceFrame,
  draft,
  onDiscard,
  onDraftChange,
  onGoToSourceFrame,
  onPreview,
  onSave,
  previewInvalidated,
  previewReady,
  selectedTask,
  sourceVideoHash,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const validation = deriveCorrectionDraftValidation(draft);
  const canPreview = validation.valid && !busy;
  const canSave = previewReady && !busy;
  const currentKeyframe = draft.keyframes.find((keyframe) => keyframe.source_frame === currentSourceFrame);

  useEffect(() => {
    setError('');
  }, [draft.draft_hash, draft.case_id]);

  const startFrame = draft.start_source_frame;
  const endFrame = draft.end_source_frame;
  const updateDraft = (patch: Partial<CorrectionDraft>) => onDraftChange({ ...draft, ...patch });
  const updateInterval = (nextStart: number, nextEnd: number) => {
    const intervalStart = Math.min(nextStart, nextEnd);
    const intervalEnd = Math.max(nextStart, nextEnd);
    updateDraft({
      start: nextStart,
      end: nextEnd,
      start_source_frame: nextStart,
      end_source_frame: nextEnd,
      keyframes: draft.keyframes.filter((keyframe) => keyframe.source_frame >= intervalStart && keyframe.source_frame <= intervalEnd),
    });
  };
  const setIntervalMode = (mode: CorrectionDraft['mode']) => updateDraft(applyCorrectionDraftMode(draft, mode, currentSourceFrame));
  const deleteCurrentKeyframe = () => {
    updateDraft({ keyframes: draft.keyframes.filter((keyframe) => keyframe.source_frame !== currentSourceFrame) });
  };
  const addCheckpoint = (state: CorrectionCheckpointDraft['state']) => {
    updateDraft({
      barriers: [
        ...draft.barriers.filter((checkpoint) => !(checkpoint.source_frame === currentSourceFrame && checkpoint.state === state)),
        { source_frame: currentSourceFrame, state },
      ],
      checkpoints: [
        ...draft.barriers.filter((checkpoint) => !(checkpoint.source_frame === currentSourceFrame && checkpoint.state === state)),
        { source_frame: currentSourceFrame, state },
      ],
    });
  };
  const deleteCheckpoint = (checkpoint: CorrectionCheckpointDraft) => {
    const barriers = draft.barriers.filter((item) => item !== checkpoint);
    updateDraft({ barriers, checkpoints: barriers });
  };
  const correctionOrError = () => {
    if (!configurationFreezeHash) return 'Correction cannot be saved because the package freeze hash is unavailable.';
    if (!sourceVideoHash) return 'Correction cannot be saved because the source video hash is unavailable.';
    if (!validation.valid) return validation.message;
    const taskId = selectedTask?.review_task_id || draft.selectedTaskId || `manual_${caseId}_${Date.now()}`;
    const normalized = normalizedKeyframes(draft);
    return {
      case_id: caseId,
      task_id: taskId,
      category: draft.category || selectedTask?.category || 'license_plate',
      operation: 'add_privacy_region',
      source_frame_start: Math.min(startFrame, endFrame),
      source_frame_end: Math.max(startFrame, endFrame),
      keyframe_boxes: normalized,
      checkpoint_verifications: draft.barriers.map((checkpoint) => ({ ...checkpoint })),
      interpolation_method: draft.mode === 'interpolate' ? 'linear' : 'none',
      reviewer_identity: actor || {},
      reason: 'Reviewed correction revision',
      source_video_hash: sourceVideoHash,
      configuration_freeze_hash: configurationFreezeHash,
    } satisfies Correction;
  };
  const run = async (mode: 'preview' | 'save') => {
    const currentValidation = deriveCorrectionDraftValidation(draft);
    if (mode === 'preview' && !currentValidation.valid) {
      setError(currentValidation.message);
      return;
    }
    if (mode === 'save' && !previewReady) {
      setError('The correction changed. Preview again before saving.');
      return;
    }
    const correction = correctionOrError();
    if (typeof correction === 'string') {
      setError(correction);
      return;
    }
    setBusy(true);
    setError('');
    try {
      if (mode === 'preview') {
        await onPreview({ ...correction, reason: 'Preview correction draft' });
      } else {
        await onSave(correction, correction.reason);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="edit-section" aria-label={`${caseId} correction editor`}>
      <article className="edit-card">
        <h3>Manual redaction</h3>
        <p className="muted">Current source frame {currentSourceFrame}</p>
        <p className="muted">Drag directly on the video to add a redaction box.</p>
        <label>
          Category
          <select
            value={draft.category}
            onChange={(event) => updateDraft({ category: event.currentTarget.value as ManualRedactionCategory })}
          >
            {MANUAL_REDACTION_CATEGORY_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="mode-options" role="radiogroup" aria-label="Manual redaction interval mode">
          {(['single_frame', 'fixed_box', 'interpolate'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={draft.mode === mode ? 'active-command' : ''}
              aria-pressed={draft.mode === mode}
              onClick={() => setIntervalMode(mode)}
            >
              {modeLabel(mode)}
            </button>
          ))}
        </div>
        <p className="muted">{modeHelp(draft.mode)}</p>
        <div className="button-row">
          <button type="button" onClick={() => updateInterval(currentSourceFrame, draft.mode === 'single_frame' ? currentSourceFrame : endFrame)}>
            Set start to current frame
          </button>
          <button type="button" onClick={() => updateInterval(draft.mode === 'single_frame' ? currentSourceFrame : startFrame, currentSourceFrame)}>
            Set end to current frame
          </button>
        </div>
        <div className="field-grid">
          <label>
            Start source frame
            <input type="number" value={startFrame} onChange={(event) => updateInterval(Number(event.currentTarget.value), endFrame)} />
          </label>
          <label>
            End source frame
            <input type="number" value={endFrame} onChange={(event) => updateInterval(startFrame, Number(event.currentTarget.value))} />
          </label>
        </div>
        {draft.mode === 'interpolate' ? (
          <div className="interpolation-steps" aria-label="Interpolation keyframe workflow">
            <div className="step-card">
              <strong>Step 1 - Start frame</strong>
              <p>Source frame {startFrame}</p>
              <p className={draft.keyframes.some((keyframe) => keyframe.source_frame === startFrame) ? 'success-text' : 'muted'}>
                {draft.keyframes.some((keyframe) => keyframe.source_frame === startFrame) ? 'Start rectangle ready' : `Draw start keyframe on source frame ${startFrame}.`}
              </p>
              <div className="button-row">
                <button type="button" onClick={() => onGoToSourceFrame(startFrame)}>Go to start frame</button>
                <button type="button" onClick={() => updateInterval(currentSourceFrame, endFrame)}>Set start to current frame</button>
              </div>
            </div>
            <div className="step-card">
              <strong>Step 2 - End frame</strong>
              <p>Source frame {endFrame}</p>
              <p className={draft.keyframes.some((keyframe) => keyframe.source_frame === endFrame) ? 'success-text' : 'muted'}>
                {draft.keyframes.some((keyframe) => keyframe.source_frame === endFrame) ? 'End rectangle ready' : `Draw end keyframe on source frame ${endFrame}.`}
              </p>
              <div className="button-row">
                <button type="button" onClick={() => onGoToSourceFrame(endFrame)}>Go to end frame</button>
                <button type="button" onClick={() => updateInterval(startFrame, currentSourceFrame)}>Set end to current frame</button>
              </div>
            </div>
          </div>
        ) : null}
      </article>
      <article className="edit-card">
        <h3>Rectangle keyframes</h3>
        <div className="button-row">
          <button type="button" onClick={deleteCurrentKeyframe} disabled={!currentKeyframe}>
            Delete current keyframe
          </button>
        </div>
        <p className="muted">Drawing is active. Drawing again on the same source frame replaces that frame's rectangle.</p>
        <div className="keyframe-list">
          {draft.keyframes.length ? (
            draft.keyframes.map((keyframe) => (
              <div
                className="keyframe-row"
                key={keyframe.source_frame}
                onClick={() => onGoToSourceFrame(keyframe.source_frame)}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  onGoToSourceFrame(keyframe.source_frame);
                }}
                role="button"
                tabIndex={0}
              >
                <span>
                  {keyframeLabel(keyframe, draft)} - frame {keyframe.source_frame}: {keyframe.bbox_xyxy.join(', ')}
                </span>
                <button type="button" onClick={() => onGoToSourceFrame(keyframe.source_frame)}>
                  Edit
                </button>
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    updateDraft({ keyframes: draft.keyframes.filter((item) => item !== keyframe) });
                  }}
                >
                  Delete
                </button>
              </div>
            ))
          ) : (
            <p className="muted">No keyframes yet.</p>
          )}
        </div>
      </article>
      <article className="edit-card">
        <h3>Preview and save</h3>
        <details className="advanced-controls">
          <summary>Advanced safety controls (optional)</summary>
          <div className="button-row">
            <button type="button" onClick={() => addCheckpoint('verified')}>
              Verified checkpoint
            </button>
            <button type="button" onClick={() => addCheckpoint('blocked_by_scene_cut')}>
              Scene-cut barrier
            </button>
            <button type="button" onClick={() => addCheckpoint('not_visible')}>
              Not-visible barrier
            </button>
          </div>
          <p className="muted">Verified checkpoint confirms the generated box at this frame. Scene-cut barriers stop interpolation across discontinuities. Not-visible barriers mark spans where the object should not be redacted.</p>
          <div className="chip-list">
            {draft.barriers.map((checkpoint) => (
              <span className="chip" key={`${checkpoint.source_frame}:${checkpoint.state}`}>
                Frame {checkpoint.source_frame}: {checkpointLabel(checkpoint.state)}
                <button type="button" onClick={() => deleteCheckpoint(checkpoint)} aria-label={`Delete ${checkpoint.state} checkpoint`}>
                  x
                </button>
              </span>
            ))}
          </div>
        </details>
        {error || !validation.valid ? <p className="error-text">{error || validation.message}</p> : null}
        <div className="button-row">
          <button type="button" className="primary" onClick={() => void run('preview')} disabled={!canPreview}>
            Preview correction
          </button>
          <button type="button" onClick={() => void run('save')} disabled={!canSave}>
            Save correction
          </button>
          <button
            type="button"
            onClick={() => {
              setError('');
              onDiscard();
            }}
            disabled={busy}
          >
            Discard
          </button>
        </div>
        {!previewReady && validation.valid ? (
          <p className="muted">
            {previewInvalidated ? 'The correction changed. Preview again before saving.' : 'Preview the correction before saving.'}
          </p>
        ) : null}
      </article>
      <p className="muted">Correction geometry is stored in source-frame and source-video coordinates.</p>
    </section>
  );
}

interface OverlayProps {
  currentSourceFrame: number;
  draft: CorrectionDraft;
  manifest: Case;
  videoRef: RefObject<HTMLVideoElement>;
  onDraftChange: (draft: CorrectionDraft) => void;
}

export interface PendingCorrectionRegion {
  bbox_xyxy: [number, number, number, number];
  correction_id: string;
  source_frame: number;
}

export function PendingCorrectionOverlay({
  manifest,
  regions,
}: {
  manifest: Case;
  regions: PendingCorrectionRegion[];
}) {
  const dimensions = useMemo(() => ({ width: manifest.width || 1920, height: manifest.height || 1080 }), [manifest.height, manifest.width]);
  if (!regions.length) return null;
  return (
    <div className="pending-correction-overlay" aria-label="Saved pending correction overlay">
      <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} preserveAspectRatio="xMidYMid meet">
        {regions.map((region) => {
          const [x1, y1, x2, y2] = region.bbox_xyxy;
          return (
            <rect
              key={`${region.correction_id}:${region.source_frame}`}
              className="pending-correction-rect"
              x={x1}
              y={y1}
              width={Math.max(1, x2 - x1)}
              height={Math.max(1, y2 - y1)}
            />
          );
        })}
      </svg>
      <span className="pending-correction-label">Saved - updating Latest reviewed</span>
    </div>
  );
}

export function CorrectionVideoOverlay({
  currentSourceFrame,
  draft,
  manifest,
  onDraftChange,
  videoRef,
}: OverlayProps) {
  const [origin, setOrigin] = useState<{ x: number; y: number } | null>(null);
  const originRef = useRef<{ x: number; y: number } | null>(null);
  const dimensions = useMemo(() => ({ width: manifest.width || 1920, height: manifest.height || 1080 }), [manifest.height, manifest.width]);
  const rect = draft.keyframes.find((keyframe) => keyframe.source_frame === currentSourceFrame)?.bbox_xyxy;
  const updateDraft = (keyframe: CorrectionKeyframeDraft) => {
    onDraftChange({
      ...draft,
      keyframes: [
        ...draft.keyframes.filter((item) => item.source_frame !== keyframe.source_frame),
        keyframe,
      ].sort((left, right) => left.source_frame - right.source_frame),
    });
  };
  const point = (event: PointerEvent<SVGSVGElement>) => {
    const video = videoRef.current;
    const videoBox = video?.getBoundingClientRect() || event.currentTarget.getBoundingClientRect();
    const width = video?.videoWidth || dimensions.width;
    const height = video?.videoHeight || dimensions.height;
    const scale = Math.min(videoBox.width / width, videoBox.height / height);
    const renderedWidth = width * scale;
    const renderedHeight = height * scale;
    const offsetX = (videoBox.width - renderedWidth) / 2;
    const offsetY = (videoBox.height - renderedHeight) / 2;
    return {
      x: clamp(Math.round(((event.clientX - videoBox.left - offsetX) / renderedWidth) * width), 0, width),
      y: clamp(Math.round(((event.clientY - videoBox.top - offsetY) / renderedHeight) * height), 0, height),
    };
  };

  return (
    <div className="correction-video-overlay draw-active" data-testid="correction-video-overlay">
      <svg
        viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
        preserveAspectRatio="xMidYMid meet"
        role="application"
        aria-label="Correction rectangle overlay"
        onPointerDown={(event) => {
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          const start = point(event);
          originRef.current = start;
          setOrigin(start);
          updateDraft({ source_frame: currentSourceFrame, bbox_xyxy: [start.x, start.y, start.x, start.y] });
        }}
        onPointerMove={(event) => {
          const dragOrigin = originRef.current || origin;
          if (!dragOrigin) return;
          event.preventDefault();
          const next = point(event);
          updateDraft({
            source_frame: currentSourceFrame,
            bbox_xyxy: [
              Math.min(dragOrigin.x, next.x),
              Math.min(dragOrigin.y, next.y),
              Math.max(dragOrigin.x, next.x),
              Math.max(dragOrigin.y, next.y),
            ],
          });
        }}
        onPointerUp={(event) => {
          originRef.current = null;
          setOrigin(null);
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
      >
        {rect ? (
          <rect
            className={`correction-rect ${currentSourceFrame === draft.start_source_frame ? 'start-rect' : ''} ${currentSourceFrame === draft.end_source_frame ? 'end-rect' : ''}`}
            x={rect[0]}
            y={rect[1]}
            width={Math.max(1, rect[2] - rect[0])}
            height={Math.max(1, rect[3] - rect[1])}
          />
        ) : null}
      </svg>
    </div>
  );
}

export interface CorrectionDraftValidation {
  valid: boolean;
  message: string;
}

export function deriveCorrectionDraftValidation(draft: CorrectionDraft): CorrectionDraftValidation {
  const start = Math.min(draft.start_source_frame, draft.end_source_frame);
  const end = Math.max(draft.start_source_frame, draft.end_source_frame);
  const keyframes = draft.keyframes.filter((keyframe) => keyframe.source_frame >= start && keyframe.source_frame <= end);
  const blockingCheckpoints = draft.barriers.filter(
    (checkpoint) =>
      checkpoint.source_frame > start &&
      checkpoint.source_frame < end &&
      (checkpoint.state === 'not_visible' || checkpoint.state === 'blocked_by_scene_cut'),
  );
  if (draft.keyframes.length && !keyframes.length) {
    return { valid: false, message: 'The current keyframe is outside the selected interval.' };
  }
  if (draft.mode === 'single_frame') {
    if (draft.start_source_frame !== draft.end_source_frame) return { valid: false, message: 'Single frame mode requires start and end to match.' };
    if (!keyframes.some((keyframe) => keyframe.source_frame === start)) return { valid: false, message: `Draw a rectangle on source frame ${start}.` };
    return { valid: true, message: '' };
  }
  if (draft.mode === 'fixed_box') {
    if (draft.start_source_frame > draft.end_source_frame) return { valid: false, message: 'Choose a valid source-frame interval.' };
    if (!keyframes.length) return { valid: false, message: `Draw one rectangle inside source frames ${start}-${end}.` };
    return { valid: true, message: '' };
  }
  if (draft.start_source_frame >= draft.end_source_frame) return { valid: false, message: 'Interpolate mode requires a multi-frame interval.' };
  if (blockingCheckpoints.length) {
    return { valid: false, message: 'Linear correction cannot cross scene-cut or not-visible barriers.' };
  }
  if (!keyframes.some((keyframe) => keyframe.source_frame === start)) {
    return { valid: false, message: `Draw a rectangle on start frame ${start} to preview this interpolation.` };
  }
  if (!keyframes.some((keyframe) => keyframe.source_frame === end)) {
    return { valid: false, message: `Draw a rectangle on end frame ${end} to preview this interpolation.` };
  }
  return { valid: true, message: '' };
}

export const validateCorrectionDraft = deriveCorrectionDraftValidation;

function normalizedKeyframes(draft: CorrectionDraft): CorrectionKeyframeDraft[] {
  const start = Math.min(draft.start_source_frame, draft.end_source_frame);
  const end = Math.max(draft.start_source_frame, draft.end_source_frame);
  const inInterval = draft.keyframes
    .filter((keyframe) => keyframe.source_frame >= start && keyframe.source_frame <= end)
    .sort((left, right) => left.source_frame - right.source_frame);
  if (draft.mode === 'single_frame') return inInterval.filter((keyframe) => keyframe.source_frame === start).slice(0, 1);
  if (draft.mode === 'fixed_box') {
    const first = inInterval[0];
    if (!first) return [];
    return [{ source_frame: start, bbox_xyxy: first.bbox_xyxy }];
  }
  return inInterval;
}

function modeLabel(mode: CorrectionDraft['mode']): string {
  if (mode === 'fixed_box') return 'Fixed box across interval';
  if (mode === 'interpolate') return 'Interpolate between keyframes';
  return 'Single frame';
}

function modeHelp(mode: CorrectionDraft['mode']): string {
  if (mode === 'fixed_box') {
    return 'Use the same rectangle at the same screen position on every frame in the interval. This does not follow a moving object.';
  }
  if (mode === 'interpolate') {
    return 'Draw a rectangle at the first and last frame. The box moves and resizes linearly between them. This is not automatic object tracking.';
  }
  return 'The rectangle applies only to the selected source frame.';
}

function keyframeLabel(keyframe: CorrectionKeyframeDraft, draft: CorrectionDraft): string {
  if (draft.mode === 'interpolate' && keyframe.source_frame === draft.start_source_frame) return 'Start';
  if (draft.mode === 'interpolate' && keyframe.source_frame === draft.end_source_frame) return 'End';
  if (draft.mode === 'interpolate') return 'Intermediate';
  return 'Keyframe';
}

function checkpointLabel(state: CorrectionCheckpointDraft['state']): string {
  if (state === 'blocked_by_scene_cut') return 'Scene cut';
  if (state === 'not_visible') return 'Not visible';
  if (state === 'verified') return 'Verified';
  if (state === 'needs_adjustment') return 'Needs adjustment';
  return 'Unreviewed';
}

export function applyCorrectionDraftMode(
  draft: CorrectionDraft,
  mode: CorrectionDraft['mode'],
  currentSourceFrame: number,
): Partial<CorrectionDraft> {
  const patch: Partial<CorrectionDraft> = { mode };
  if (mode === 'single_frame') {
    patch.start_source_frame = currentSourceFrame;
    patch.end_source_frame = currentSourceFrame;
    patch.start = currentSourceFrame;
    patch.end = currentSourceFrame;
    patch.keyframes = draft.keyframes.filter((keyframe) => keyframe.source_frame === currentSourceFrame);
    patch.holdSameBox = false;
    patch.interpolation_method = 'linear';
  }
  if (mode === 'fixed_box') {
    patch.holdSameBox = true;
    patch.interpolation_method = 'none';
  }
  if (mode === 'interpolate') {
    patch.holdSameBox = false;
    patch.interpolation_method = 'linear';
  }
  return patch;
}
