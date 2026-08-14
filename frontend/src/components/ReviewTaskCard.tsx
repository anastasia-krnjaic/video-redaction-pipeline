import type { Case } from '../types/api';
import type { TaskWithDecision } from '../types/domain';
import { decisionLabel, taskDecisionActionOptions, taskGuidance } from '../state/reviewState';
import { sourceSpan, taskTimestamp } from '../utils/frame';

interface Props {
  manifest: Case;
  pending: boolean;
  selected: boolean;
  task: TaskWithDecision;
  onDecision: (task: TaskWithDecision, action: string) => void;
  onFixNow: (task: TaskWithDecision) => void;
  onSelect: (task: TaskWithDecision) => void;
}

export function ReviewTaskCard({ manifest, onDecision, onFixNow, onSelect, pending, selected, task }: Props) {
  const taskActionable = task.actionable !== false;
  const actionsDisabled = pending || !taskActionable;
  const actionOptions = taskDecisionActionOptions(task);
  const guidance = taskGuidance(task);
  return (
    <article
      className={`task-card ${selected ? 'selected' : ''}`}
      data-task-id={task.review_task_id}
      onClick={() => onSelect(task)}
    >
      <div className="task-title">{guidance.title}</div>
      <p className="task-instruction">{guidance.instruction}</p>
      <div className="task-meta">
        {categoryLabel(task.category)} · source {sourceSpan(task)} · {taskTimestamp(manifest, task)}
      </div>
      <div className="task-state-row">
        <span className={`badge ${badgeClass(task)}`}>
          {task.decision ? decisionLabel(task.decision.action) : task.bucket}
        </span>
        {task.decision_state === 'resolved' ? <span className="badge done">Corrected</span> : null}
        {task.blocking ? <span className="badge warn">Blocking</span> : null}
      </div>
      <div className="task-actions" onClick={(event) => event.stopPropagation()}>
        {actionOptions.map((option) => (
          <button
            key={option.action}
            type="button"
            className={`decision-choice ${option.className} ${task.decision?.action === option.action ? 'selected' : ''}`}
            disabled={actionsDisabled}
            onClick={() => onDecision(task, option.action)}
          >
            {option.label}
          </button>
        ))}
        {task.bucket === 'Needs correction' ? (
          <button type="button" className="primary fix-now" disabled={pending} onClick={() => onFixNow(task)}>
            Fix this now
          </button>
        ) : null}
      </div>
      {pending ? <p className="saved-note">Saving...</p> : null}
      <details className="technical task-technical" onClick={(event) => event.stopPropagation()}>
        <summary>Technical details</summary>
        <pre>{JSON.stringify(technicalDetails(task), null, 2)}</pre>
      </details>
    </article>
  );
}

function badgeClass(task: TaskWithDecision): string {
  if (task.bucket === 'Open') return 'open';
  if (task.bucket === 'Correct') return 'done';
  if (task.bucket === 'Not sensitive') return 'neutral';
  return 'needs-correction';
}

function technicalDetails(task: TaskWithDecision): Record<string, unknown> {
  return {
    allowed_actions: task.allowed_actions || [],
    blocking: task.blocking,
    decision_state: task.decision_state,
    issue_type: task.issue_type,
    normal_task_class: task.normal_task_class,
    query_presentation_class: task.query_presentation_class,
    raw_signal_ids: task.raw_signal_ids || [],
    review_task_id: task.review_task_id,
    title: task.title,
    what_happened: task.what_happened,
    what_to_inspect: task.what_to_inspect,
  };
}

function categoryLabel(category: string): string {
  if (category === 'face') return 'Person';
  if (category === 'license_plate') return 'Plate';
  if (category === 'other_sensitive') return 'Other sensitive region';
  return category;
}
