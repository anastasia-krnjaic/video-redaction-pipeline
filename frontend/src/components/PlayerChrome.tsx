import { Maximize2, Pause, Play, StepBack, StepForward } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent, PointerEvent } from 'react';
import type { Case } from '../types/api';
import type { VideoController } from '../hooks/useVideoController';
import type { ReviewGroup } from '../types/domain';
import { taskColorClass } from '../state/reviewState';
import { sourceToCaseFrame, timelinePercent, timelineRatioToFrame, timelineTooltip } from '../utils/frame';

interface Props {
  controller: VideoController;
  manifest: Case;
  selectedTaskId?: string | null;
  tasks?: ReviewGroup[];
  onSelectTask?: (task: ReviewGroup) => void;
}

export function PlayerChrome({ controller, manifest, onSelectTask, selectedTaskId, tasks = [] }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [scrubbing, setScrubbing] = useState(false);
  const maxFrame = Math.max(0, manifest.frame_count - 1);
  const playheadPercent = timelinePercent(manifest, controller.caseFrame);
  const markers = useMemo(
    () =>
      tasks.map((task, index) => {
        const start = sourceToCaseFrame(manifest, task.source_frame_start);
        return {
          lane: index % 3,
          start,
          task,
        };
      }),
    [manifest, maxFrame, tasks],
  );

  const seekFromClientX = (clientX: number) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const ratio = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
    controller.seekToFrame(timelineRatioToFrame(manifest, ratio));
  };

  const seekFromPointer = (event: PointerEvent<HTMLDivElement>) => seekFromClientX(event.clientX);

  const scrubStart = (event: PointerEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest('.player-marker')) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setScrubbing(true);
    seekFromPointer(event);
  };

  const scrubMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!scrubbing) return;
    event.preventDefault();
    seekFromPointer(event);
  };

  const scrubEnd = (event: PointerEvent<HTMLDivElement>) => {
    if (!scrubbing) return;
    setScrubbing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const mouseScrubStart = (event: ReactMouseEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest('.player-marker')) return;
    event.preventDefault();
    setScrubbing(true);
    seekFromClientX(event.clientX);
  };

  const mouseScrubMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!scrubbing) return;
    event.preventDefault();
    seekFromClientX(event.clientX);
  };

  const mouseScrubEnd = () => {
    if (scrubbing) setScrubbing(false);
  };

  return (
    <div className="player-controls" data-testid="player-chrome">
      <div className="controls-top">
        <button type="button" onClick={controller.togglePlay} title="Play or pause">
          {controller.playing ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
          <span>{controller.playing ? 'Pause' : 'Play'}</span>
        </button>
        <button type="button" onClick={() => controller.stepFrame(-1)} title="Step back one frame">
          <StepBack size={16} aria-hidden="true" />
        </button>
        <button type="button" onClick={() => controller.stepFrame(1)} title="Step forward one frame">
          <StepForward size={16} aria-hidden="true" />
        </button>
        <button type="button" onClick={controller.toggleFullscreen} title="Fullscreen">
          <Maximize2 size={16} aria-hidden="true" />
        </button>
        <div className="time-readout" aria-label="Current time">
          {controller.timeText}
        </div>
      </div>
      <div
        ref={trackRef}
        className="scrub-surface"
        role="slider"
        aria-label="Video timeline"
        aria-valuemin={0}
        aria-valuemax={maxFrame}
        aria-valuenow={controller.caseFrame}
        tabIndex={0}
        onPointerDown={scrubStart}
        onPointerMove={scrubMove}
        onPointerUp={scrubEnd}
        onPointerCancel={scrubEnd}
        onMouseDown={mouseScrubStart}
        onMouseMove={mouseScrubMove}
        onMouseUp={mouseScrubEnd}
        onMouseLeave={mouseScrubEnd}
        onKeyDown={(event) => {
          if (event.key === 'ArrowRight') {
            event.preventDefault();
            controller.stepFrame(event.shiftKey ? 10 : 1);
          } else if (event.key === 'ArrowLeft') {
            event.preventDefault();
            controller.stepFrame(event.shiftKey ? -10 : -1);
          } else if (event.key === 'Home') {
            event.preventDefault();
            controller.seekToFrame(0);
          } else if (event.key === 'End') {
            event.preventDefault();
            controller.seekToFrame(maxFrame);
          }
        }}
      >
        <div className="scrub-base" />
        {markers.map(({ lane, start, task }) => (
          <button
            key={task.review_task_id}
            type="button"
            className={[
              'timeline-marker',
              'player-marker',
              taskColorClass(task),
              selectedTaskId === task.review_task_id ? 'selected' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            data-task-id={task.review_task_id}
            style={{
              left: `${timelinePercent(manifest, start)}%`,
              top: `${6 + lane * 14}px`,
            }}
            title={timelineTooltip(manifest, task)}
            aria-label={timelineTooltip(manifest, task)}
            onClick={(event) => {
              event.stopPropagation();
              onSelectTask?.(task);
            }}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <span className="visually-hidden">
              {task.title} source {task.source_frame_start}-{task.source_frame_end}
            </span>
          </button>
        ))}
        <div className="scrub-playhead" style={{ left: `${playheadPercent}%` }} />
      </div>
      <div className="legend">
        <span className="legend-plate">Open plate</span>
        <span className="legend-face">Open face</span>
        <span className="legend-correct">Correct</span>
        <span className="legend-needs">Needs correction</span>
        <span className="legend-not-sensitive">Not sensitive</span>
      </div>
    </div>
  );
}
