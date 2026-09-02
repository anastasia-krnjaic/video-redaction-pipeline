import type { MaterializationJob, WorkingState } from '../types/api';
import type { CaseId } from '../types/domain';

export type LatestReviewedLifecycleState =
  | 'CURRENT_REUSABLE'
  | 'BUILD_REQUIRED'
  | 'BUILD_RUNNING'
  | 'BUILD_FAILED_RETRYABLE'
  | 'BLOCKED';

export function materializationJobMatchesCurrentProjection(
  job: MaterializationJob,
  currentState?: WorkingState,
  expected?: { caseId?: CaseId; sessionId?: string },
): boolean {
  if (job.status !== 'completed') return false;
  return materializationJobRepresentsCurrentProjection(job, currentState, expected);
}

export function materializationJobRepresentsCurrentProjection(
  job: MaterializationJob,
  currentState?: WorkingState,
  expected?: { caseId?: CaseId; sessionId?: string },
): boolean {
  if (!expected?.caseId || job.case_id !== expected.caseId) return false;
  if (!expected.sessionId || job.session_id !== expected.sessionId) return false;
  const jobRevision = job.published_working_state_revision || job.working_state?.active_projection_hash || job.accepted_working_state_revision;
  const currentRevision = currentState?.active_projection_hash;
  if (!jobRevision || !currentRevision) return false;
  return jobRevision === currentRevision;
}

export function latestReviewedLifecycleState({
  activeJobs,
  caseLatestReviewedStale,
  caseId,
  latestRole,
  materializationError,
  reviewReady,
  sessionId,
  workingState,
}: {
  activeJobs: MaterializationJob[];
  caseLatestReviewedStale?: boolean;
  caseId: CaseId;
  latestRole?: Record<string, unknown>;
  materializationError?: string;
  reviewReady: boolean;
  sessionId?: string;
  workingState?: WorkingState;
}): LatestReviewedLifecycleState {
  if (!reviewReady || !sessionId) return 'BLOCKED';
  if (materializationError) return 'BUILD_FAILED_RETRYABLE';
  if (activeJobs.some((job) => job.status === 'failed')) return 'BUILD_FAILED_RETRYABLE';
  if (
    activeJobs.some(
      (job) =>
        (job.status === 'queued' || job.status === 'running') &&
        job.case_id === caseId &&
        job.session_id === sessionId,
    )
  ) {
    return 'BUILD_RUNNING';
  }
  if (latestRole && latestRole.media_validation_passed !== false) {
    if (caseLatestReviewedStale === false) return 'CURRENT_REUSABLE';
    if (caseLatestReviewedStale === true) return 'BUILD_REQUIRED';
    const currentRevision = String(workingState?.active_projection_hash || '');
    const latestRevision = String(
      latestRole.published_working_state_revision || latestRole.accepted_working_state_revision || '',
    );
    if (!currentRevision || !latestRevision || currentRevision === latestRevision) {
      return 'CURRENT_REUSABLE';
    }
  }
  return 'BUILD_REQUIRED';
}
