import { useMemo, useRef } from 'react';
import type { Case } from '../types/api';
import type { QueueBucket, TaskWithDecision } from '../types/domain';
import { bucketCounts } from '../state/reviewState';
import { ReviewTaskCard } from './ReviewTaskCard';

const buckets: QueueBucket[] = ['Open', 'Needs correction', 'Unsure / Escalated', 'Completed', 'Resolved'];

interface Props {
  bucket: QueueBucket;
  manifest: Case;
  pendingTaskId?: string;
  selectedTaskId?: string | null;
  tasks: TaskWithDecision[];
  onBucketChange: (bucket: QueueBucket) => void;
  onDecision: (task: TaskWithDecision, action: string) => void;
  onSelectTask: (task: TaskWithDecision) => void;
  onOpenEntities: () => void;
}

export function ReviewQueue({
  bucket,
  manifest,
  onBucketChange,
  onDecision,
  onOpenEntities,
  onSelectTask,
  pendingTaskId,
  selectedTaskId,
  tasks,
}: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const counts = useMemo(() => bucketCounts(tasks), [tasks]);
  const visibleTasks = tasks.filter((task) => task.bucket === bucket);

  return (
    <aside className="side-panel" aria-label={`${manifest.display_name} review queue`}>
      <div className="queue-toolbar">
        <div className="task-summary">
          <div className="stat">
            <strong>{counts.Open}</strong>
            <span>Open</span>
          </div>
          <div className="stat">
            <strong>{counts['Needs correction'] + counts['Unsure / Escalated']}</strong>
            <span>Needs attention</span>
          </div>
        </div>
        <button type="button" onClick={onOpenEntities}>
          Entities
        </button>
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
        ) : (
          <p className="muted empty-state">No tasks in this bucket.</p>
        )}
      </div>
    </aside>
  );
}
