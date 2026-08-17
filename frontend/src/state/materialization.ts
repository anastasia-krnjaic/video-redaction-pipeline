import type { MaterializationJob, WorkingState } from '../types/api';
import type { CaseId } from '../types/domain';

export type LatestReviewedLifecycleState =
  | 'CURRENT_REUSABLE'
  | 'BUILD_REQUIRED'
  | 'BUILD_RUNNING'
  | 'BUILD_FAILED_RETRYABLE'
  | 'BLOCKED';

const requestedAutoBuilds = new Set<string>();

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

export function latestReviewedBuildRequestKey({
  caseId,
  sessionId,
  workingState,
}: {
  caseId?: CaseId;
  sessionId?: string;
  workingState?: WorkingState;
}): string {
  const revision = String(workingState?.active_projection_hash || workingState?.state_hash || '');
  if (!caseId || !sessionId || !revision) return '';
  return `${sessionId}:${caseId}:${revision}`;
}

export function latestReviewedLifecycleState({
  activeJobs,
  caseId,
  latestRole,
  materializationError,
  reviewReady,
  sessionId,
  workingState,
}: {
  activeJobs: MaterializationJob[];
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
        materializationJobRepresentsCurrentProjection(job, workingState, { caseId, sessionId }),
    )
  ) {
    return 'BUILD_RUNNING';
  }
  if (latestRole && latestRole.media_validation_passed !== false) {
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

export function claimLatestReviewedAutoBuildRequest(key: string): boolean {
  if (!key || requestedAutoBuilds.has(key)) return false;
  requestedAutoBuilds.add(key);
  return true;
}

export function releaseLatestReviewedAutoBuildRequest(key: string): void {
  if (key) requestedAutoBuilds.delete(key);
}

export function resetLatestReviewedAutoBuildRequestsForTests(): void {
  requestedAutoBuilds.clear();
}
