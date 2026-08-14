import type { MaterializationJob, WorkingState } from '../types/api';
import type { CaseId } from '../types/domain';

export function materializationJobMatchesCurrentProjection(
  job: MaterializationJob,
  currentState?: WorkingState,
  expected?: { caseId?: CaseId; sessionId?: string },
): boolean {
  if (job.status !== 'completed') return false;
  if (!expected?.caseId || job.case_id !== expected.caseId) return false;
  if (!expected.sessionId || job.session_id !== expected.sessionId) return false;
  const jobRevision = job.published_working_state_revision || job.working_state?.active_projection_hash || job.accepted_working_state_revision;
  const currentRevision = currentState?.active_projection_hash;
  if (!jobRevision || !currentRevision) return false;
  return jobRevision === currentRevision;
}
