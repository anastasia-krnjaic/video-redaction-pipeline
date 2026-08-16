import { useEffect, useMemo, useRef } from 'react';
import type { Case } from '../types/api';
import type { QueueBucket, ReviewGroup } from '../types/domain';
import { bucketCounts } from '../state/reviewState';
import { ReviewTaskCard } from './ReviewTaskCard';

const buckets: QueueBucket[] = ['Open', 'Needs correction', 'Correct', 'Not sensitive'];
const bucketLabels: Record<QueueBucket, string> = {
  Open: 'Guidance',
  'Needs correction': 'Needs correction',
  Correct: 'Correct',
  'Not sensitive': 'Not sensitive',
};

interface Props {
  bucket: QueueBucket;
  manifest: Case;
  optionalTasks?: ReviewGroup[];
  pendingTaskId?: string;
  selectedTaskId?: string | null;
  tasks: ReviewGroup[];
  reviewComplete?: boolean;
  onBucketChange: (bucket: QueueBucket) => void;
  onDecision: (task: ReviewGroup, action: string) => void;
  onFixNow: (task: ReviewGroup) => void;
  onSelectTask: (task: ReviewGroup) => void;
}

export function ReviewQueue({
  bucket,
  manifest,
  onBucketChange,
  onDecision,
  onFixNow,
  onSelectTask,
  optionalTasks = [],
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
    if (selectedCard && listRef.current) {
      listRef.current.scrollTo({
        top: Math.max(0, selectedCard.offsetTop - listRef.current.offsetTop - 12),
        behavior: 'smooth',
      });
    }
  }, [bucket, selectedTaskId]);

  return (
    <aside className="side-panel" aria-label={`${manifest.display_name} review queue`}>
      <div className="queue-toolbar">
        <div className="task-summary">
          <div className="stat">
            <strong>{counts.Open}</strong>
            <span>Guidance</span>
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
            {bucketLabels[item]} <span className="count">{counts[item]}</span>
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
              onFixNow={onFixNow}
              onSelect={onSelectTask}
            />
          ))
        ) : reviewComplete && bucket === 'Open' ? (
          <div className="completion-state" role="status">
            <strong>GUIDANCE CLEAR</strong>
            <p className="muted">Ready to build reviewed video.</p>
          </div>
        ) : (
          <p className="muted empty-state">No guidance in this bucket.</p>
        )}
      </div>
      {optionalTasks.length ? (
        <details className="optional-qa">
          <summary>Optional QA checks ({optionalTasks.length})</summary>
          <div className="task-list optional">
            {optionalTasks.map((task) => (
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
            ))}
          </div>
        </details>
      ) : null}
    </aside>
  );
}
