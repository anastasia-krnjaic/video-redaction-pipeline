import { useEffect, useMemo, useRef } from 'react';
import type { Case } from '../types/api';
import type { QueueBucket, TaskWithDecision } from '../types/domain';
import { bucketCounts } from '../state/reviewState';
import { ReviewTaskCard } from './ReviewTaskCard';

const buckets: QueueBucket[] = ['Open', 'Needs correction', 'Unsure / Escalated', 'Completed'];

interface Props {
  bucket: QueueBucket;
  manifest: Case;
  pendingTaskId?: string;
  selectedTaskId?: string | null;
  tasks: TaskWithDecision[];
  openComplete?: boolean;
  onBucketChange: (bucket: QueueBucket) => void;
  onDecision: (task: TaskWithDecision, action: string) => void;
  onSelectTask: (task: TaskWithDecision) => void;
}

export function ReviewQueue({
  bucket,
  manifest,
  onBucketChange,
  onDecision,
  onSelectTask,
  openComplete,
  pendingTaskId,
  selectedTaskId,
  tasks,
}: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const counts = useMemo(() => bucketCounts(tasks), [tasks]);
  const visibleTasks = tasks.filter((task) => task.bucket === bucket);

  useEffect(() => {
    if (!selectedTaskId) return;
    const selectedCard = Array.from(listRef.current?.querySelectorAll<HTMLElement>('.task-card') || []).find(
      (card) => card.dataset.taskId === selectedTaskId,
    );
    selectedCard?.scrollIntoView({ block: 'nearest' });
  }, [bucket, selectedTaskId]);

  return (
    <aside className="side-panel" aria-label={`${manifest.display_name} review queue`}>
      <div className="queue-toolbar">
        <div className="task-summary">
          <div className="stat">
            <strong>{counts.Open}</strong>
            <span>Open</span>
          </div>
          <div className="stat">
            <strong>{counts.Completed}</strong>
            <span>Completed</span>
          </div>
        </div>
      </div>
      <div className="filter-row" role="tablist" aria-label="Review queue buckets">
        {buckets.map((item) => (
          <button
            key={item}
            type="button"
            className={item === bucket ? 'active' : ''}
            onClick={() => onBucketChange(item)}
          >
            {item} <span className="count">{counts[item]}</span>
          </button>
        ))}
      </div>
      <div className="task-list" ref={listRef}>
        {visibleTasks.length ? (
          visibleTasks.map((task) => (
            <ReviewTaskCard
              key={task.review_task_id}
              manifest={manifest}
              pending={pendingTaskId === task.review_task_id}
              selected={selectedTaskId === task.review_task_id}
              task={task}
              onDecision={onDecision}
              onSelect={onSelectTask}
            />
          ))
        ) : openComplete && bucket === 'Open' ? (
          <div className="completion-state" role="status">
            <strong>All open review items are complete.</strong>
            <p className="muted">Next: review Needs correction, review Unsure / Escalated, complete the full-video sweep, or check finalization.</p>
          </div>
        ) : (
          <p className="muted empty-state">No tasks in this bucket.</p>
        )}
      </div>
    </aside>
  );
}
