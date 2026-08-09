import type { Case } from '../types/api';
import type { TaskWithDecision } from '../types/domain';
import { ACTIONS, decisionLabel } from '../state/reviewState';
import { sourceSpan, taskTimestamp } from '../utils/frame';

interface Props {
  manifest: Case;
  pending: boolean;
  selected: boolean;
  task: TaskWithDecision;
  onDecision: (task: TaskWithDecision, action: string) => void;
  onSelect: (task: TaskWithDecision) => void;
}

export function ReviewTaskCard({ manifest, onDecision, onSelect, pending, selected, task }: Props) {
  const taskActionable = task.actionable !== false;
  const waitingForMedia = task.normal_task_class === 'correction_verification' && !taskActionable;
  const actionsDisabled = pending || !taskActionable;
  return (
    <article
      className={`task-card ${selected ? 'selected' : ''}`}
      data-task-id={task.review_task_id}
      onClick={() => onSelect(task)}
    >
      <div className="task-title">{task.title}</div>
      <div className="task-meta">
        {categoryLabel(task.category)} · source {sourceSpan(task)} · {taskTimestamp(manifest, task)}
      </div>
      <p>{task.what_to_inspect}</p>
      <div className="task-state-row">
        <span className={`badge ${task.bucket === 'Open' ? 'open' : task.bucket === 'Completed' ? 'done' : 'warn'}`}>
          {task.decision ? decisionLabel(task.decision.action) : task.bucket}
        </span>
        {waitingForMedia ? <span className="badge warn">Waiting for Latest reviewed media</span> : null}
        {task.decision_state === 'resolved' ? <span className="badge done">Corrected</span> : null}
        {task.blocking ? <span className="badge warn">Blocking</span> : null}
      </div>
      <div className="task-actions" onClick={(event) => event.stopPropagation()}>
        <button
          type="button"
          className={`decision-choice decision-correct ${task.decision?.action === ACTIONS.looksCorrect ? 'selected' : ''}`}
          disabled={actionsDisabled}
          onClick={() => onDecision(task, ACTIONS.looksCorrect)}
        >
          Correct
        </button>
        <button
          type="button"
          className={`decision-choice decision-needs-correction ${task.decision?.action === ACTIONS.needsCorrection ? 'selected' : ''}`}
          disabled={actionsDisabled}
          onClick={() => onDecision(task, ACTIONS.needsCorrection)}
        >
          Needs correction
        </button>
        <button
          type="button"
          className={`decision-choice decision-unsure ${task.decision?.action === ACTIONS.unsure ? 'selected' : ''}`}
          disabled={actionsDisabled}
          onClick={() => onDecision(task, ACTIONS.unsure)}
        >
          Unsure
        </button>
      </div>
      {pending ? <p className="saved-note">Saving...</p> : null}
      {waitingForMedia ? <p className="saved-note">Saved - updating Latest reviewed</p> : null}
    </article>
  );
}

function categoryLabel(category: string): string {
  if (category === 'face') return 'Person';
  if (category === 'license_plate') return 'Plate';
  return category;
}
