import { useMemo, useState } from 'react';
import type { PointerEvent } from 'react';
import type { Attribution, Case, Correction, Task } from '../types/api';
import type { CaseId, CorrectionDraft, CorrectionKeyframeDraft } from '../types/domain';
import { clamp } from '../utils/frame';

interface Props {
  actor: Attribution | Record<string, unknown>;
  caseId: CaseId;
  configurationFreezeHash?: string | null;
  draft: CorrectionDraft;
  manifest: Case;
  selectedTask?: Task | null;
  sourceVideoHash?: string | null;
  onDiscard: () => void;
  onDraftChange: (draft: CorrectionDraft) => void;
  onSave: (correction: Correction, reason: string) => Promise<void>;
}

export function CorrectionEditor({
  actor,
  caseId,
  configurationFreezeHash,
  draft,
  manifest,
  onDiscard,
  onDraftChange,
  onSave,
  selectedTask,
  sourceVideoHash,
}: Props) {
  const [drawing, setDrawing] = useState<{ x: number; y: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const currentSourceFrame = draft.start;
  const currentKeyframe = draft.keyframes.find((keyframe) => keyframe.source_frame === currentSourceFrame);
  const rect = currentKeyframe?.bbox_xyxy;
  const viewBox = useMemo(() => `0 0 ${manifest.width || 1920} ${manifest.height || 1080}`, [manifest.height, manifest.width]);

  const updateDraft = (patch: Partial<CorrectionDraft>) => onDraftChange({ ...draft, ...patch });

  const addCheckpoint = (state: CorrectionDraft['checkpoints'][number]['state']) => {
    updateDraft({
      checkpoints: [
        ...draft.checkpoints.filter((checkpoint) => checkpoint.source_frame !== currentSourceFrame),
        { source_frame: currentSourceFrame, state },
      ],
    });
  };

  const pointerPoint = (event: PointerEvent<SVGSVGElement>) => {
    const svg = event.currentTarget;
    const box = svg.getBoundingClientRect();
    const width = manifest.width || 1920;
    const height = manifest.height || 1080;
    return {
      x: clamp(Math.round(((event.clientX - box.left) / box.width) * width), 0, width),
      y: clamp(Math.round(((event.clientY - box.top) / box.height) * height), 0, height),
    };
  };

  const upsertKeyframe = (keyframe: CorrectionKeyframeDraft) => {
    updateDraft({
      keyframes: [
        ...draft.keyframes.filter((item) => item.source_frame !== keyframe.source_frame),
        keyframe,
      ].sort((left, right) => left.source_frame - right.source_frame),
    });
  };

  const save = async (previewOnly: boolean) => {
    if (!configurationFreezeHash) {
      setError('Correction cannot be saved because the package freeze hash is unavailable.');
      return;
    }
    if (!sourceVideoHash) {
      setError('Correction cannot be saved because the source video hash is unavailable.');
      return;
    }
    if (!draft.keyframes.length) {
      setError('Add at least one rectangle keyframe before saving.');
      return;
    }
    if (draft.start !== draft.end && draft.keyframes.length < 2) {
      setError('Intervals longer than one frame require start and end keyframes.');
      return;
    }
    const taskId = selectedTask?.review_task_id || draft.selectedTaskId || `manual_${caseId}_${Date.now()}`;
    const correction: Correction = {
      case_id: caseId,
      task_id: taskId,
      category: selectedTask?.category || 'license_plate',
      operation: 'add_privacy_region',
      source_frame_start: Math.min(draft.start, draft.end),
      source_frame_end: Math.max(draft.start, draft.end),
      keyframe_boxes: draft.keyframes,
      checkpoint_verifications: draft.checkpoints.map((checkpoint) => ({ ...checkpoint })),
      interpolation_method: draft.interpolation_method,
      reviewer_identity: actor || {},
      reason: previewOnly ? 'Preview correction draft' : 'Reviewed correction revision',
      source_video_hash: sourceVideoHash,
      configuration_freeze_hash: configurationFreezeHash,
    };
    setSaving(true);
    setError('');
    try {
      await onSave(correction, correction.reason);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="edit-section" aria-label={`${caseId} correction editor`}>
      <article className="edit-card">
        <h3>Correction interval</h3>
        <div className="field-grid">
          <label>
            Start source frame
            <input
              type="number"
              value={draft.start}
              onChange={(event) => updateDraft({ start: Number(event.currentTarget.value) })}
            />
          </label>
          <label>
            End source frame
            <input
              type="number"
              value={draft.end}
              onChange={(event) => updateDraft({ end: Number(event.currentTarget.value) })}
            />
          </label>
          <label>
            Interpolation
            <select
              value={draft.interpolation_method}
              onChange={(event) => updateDraft({ interpolation_method: event.currentTarget.value as 'linear' | 'none' })}
            >
              <option value="linear">Linear</option>
              <option value="none">No interpolation barrier</option>
            </select>
          </label>
        </div>
        <div className="button-row">
          <button type="button" onClick={() => addCheckpoint('verified')}>
            Add verified checkpoint
          </button>
          <button type="button" onClick={() => addCheckpoint('blocked_by_scene_cut')}>
            Add scene-cut barrier
          </button>
          <button type="button" onClick={() => addCheckpoint('not_visible')}>
            Add not-visible barrier
          </button>
        </div>
      </article>
      <article className="edit-card">
        <h3>Rectangle keyframes</h3>
        <p className="muted">Draw a rectangle for source frame {currentSourceFrame}.</p>
        <div className="editor-canvas-wrap">
          <svg
            viewBox={viewBox}
            role="application"
            aria-label="Correction rectangle editor"
            onPointerDown={(event) => {
              const point = pointerPoint(event);
              setDrawing(point);
              upsertKeyframe({ source_frame: currentSourceFrame, bbox_xyxy: [point.x, point.y, point.x, point.y] });
            }}
            onPointerMove={(event) => {
              if (!drawing) return;
              const point = pointerPoint(event);
              const bbox: [number, number, number, number] = [
                Math.min(drawing.x, point.x),
                Math.min(drawing.y, point.y),
                Math.max(drawing.x, point.x),
                Math.max(drawing.y, point.y),
              ];
              upsertKeyframe({ source_frame: currentSourceFrame, bbox_xyxy: bbox });
            }}
            onPointerUp={() => setDrawing(null)}
          >
            <rect x="0" y="0" width="100%" height="100%" className="editor-backdrop" />
            {rect ? (
              <rect
                className="correction-rect"
                x={rect[0]}
                y={rect[1]}
                width={Math.max(1, rect[2] - rect[0])}
                height={Math.max(1, rect[3] - rect[1])}
              />
            ) : null}
          </svg>
        </div>
        <div className="keyframe-list">
          {draft.keyframes.length ? (
            draft.keyframes.map((keyframe) => (
              <div className="keyframe-row" key={keyframe.source_frame}>
                <span>
                  Frame {keyframe.source_frame}: {keyframe.bbox_xyxy.join(', ')}
                </span>
                <button
                  type="button"
                  onClick={() => updateDraft({ keyframes: draft.keyframes.filter((item) => item !== keyframe) })}
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
        {error ? <p className="error-text">{error}</p> : null}
        <div className="button-row">
          <button type="button" className="primary" onClick={() => save(true)} disabled={saving}>
            Preview correction
          </button>
          <button type="button" onClick={() => save(false)} disabled={saving}>
            Save correction
          </button>
          <button type="button" onClick={() => addCheckpoint('verified')} disabled={saving}>
            Verification
          </button>
          <button type="button" onClick={onDiscard} disabled={saving}>
            Discard
          </button>
        </div>
      </article>
      <p className="muted">Correction geometry is stored in source-frame coordinates.</p>
    </section>
  );
}
