import type { Case } from '../types/api';
import type { ReviewGroup } from '../types/domain';
import { sourceToCaseFrame } from '../utils/frame';
import { taskColorClass } from '../state/reviewState';

interface Props {
  manifest: Case;
  tasks: ReviewGroup[];
  selectedTaskId?: string | null;
  currentFrame: number;
  onSelectTask: (task: ReviewGroup) => void;
}

export function ReviewTimeline({ currentFrame, manifest, onSelectTask, selectedTaskId, tasks }: Props) {
  const width = 1000;
  const height = 54;
  const maxFrame = Math.max(1, manifest.frame_count - 1);
  const playheadX = (currentFrame / maxFrame) * width;

  return (
    <section className="timeline-panel" aria-label="Review timeline">
      <svg viewBox={`0 0 ${width} ${height}`} className="review-timeline" role="img" aria-label="Full case review timeline">
        <line x1="0" x2={width} y1="26" y2="26" className="timeline-base" />
        {tasks.map((task, index) => {
          const start = sourceToCaseFrame(manifest, task.source_frame_start);
          const end = sourceToCaseFrame(manifest, task.source_frame_end);
          const x = (start / maxFrame) * width;
          const markerWidth = Math.max(8, ((Math.max(start, end) - start + 1) / maxFrame) * width);
          const y = 12 + (index % 3) * 10;
          const className = [
            'timeline-marker',
            taskColorClass(task),
            selectedTaskId === task.review_task_id ? 'selected' : '',
          ]
            .filter(Boolean)
            .join(' ');
          return (
            <g key={task.review_task_id}>
              <title>
                {task.title} · source {task.source_frame_start}-{task.source_frame_end}
              </title>
              <rect
                className={className}
                data-task-id={task.review_task_id}
                x={x}
                y={y}
                width={markerWidth}
                height="8"
                rx="4"
                role="button"
                tabIndex={0}
                onClick={() => onSelectTask(task)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') onSelectTask(task);
                }}
              />
            </g>
          );
        })}
        <line x1={playheadX} x2={playheadX} y1="4" y2="48" className="timeline-playhead" />
      </svg>
      <div className="legend">
        <span className="legend-plate">Plate guidance</span>
        <span className="legend-face">Face guidance</span>
        <span className="legend-correct">Correct</span>
        <span className="legend-needs">Needs correction</span>
        <span className="legend-not-sensitive">Not sensitive</span>
      </div>
    </section>
  );
}
