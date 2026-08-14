import { useEffect, useMemo, useRef } from 'react';
import type { Case } from '../types/api';
import type { QueueBucket, TaskWithDecision } from '../types/domain';
import { bucketCounts } from '../state/reviewState';
import { ReviewTaskCard } from './ReviewTaskCard';

const buckets: QueueBucket[] = ['Open', 'Needs correction', 'Correct', 'Not sensitive'];

interface Props {
  bucket: QueueBucket;
  manifest: Case;
  pendingTaskId?: string;
  selectedTaskId?: string | null;
  tasks: TaskWithDecision[];
  reviewComplete?: boolean;
  onBucketChange: (bucket: QueueBucket) => void;
  onDecision: (task: TaskWithDecision, action: string) => void;
  onFixNow: (task: TaskWithDecision) => void;
  onSelectTask: (task: TaskWithDecision) => void;
}

export function ReviewQueue({
  bucket,
  manifest,
  onBucketChange,
  onDecision,
  onFixNow,
  onSelectTask,
  reviewComplete,
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
            <strong>{counts['Needs correction']}</strong>
            <span>Needs correction</span>
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
      <div className="action-legend" aria-label="Review action meanings">
        <span><strong>Correct</strong> current redaction is acceptable.</span>
        <span><strong>Needs correction</strong> redaction geometry must be changed.</span>
        <span><strong>Not sensitive</strong> this content does not require redaction.</span>
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
              onFixNow={onFixNow}
              onSelect={onSelectTask}
            />
          ))
        ) : reviewComplete && bucket === 'Open' ? (
          <div className="completion-state" role="status">
            <strong>REVIEW COMPLETE</strong>
            <p className="muted">Ready to build Latest reviewed.</p>
          </div>
        ) : (
          <p className="muted empty-state">No tasks in this bucket.</p>
        )}
      </div>
    </aside>
  );
}
