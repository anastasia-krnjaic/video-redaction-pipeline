import { useEffect, useState } from 'react';
import { finalizeCase as finalizeCaseRequest, loadFinalizationStatus } from '../api/client';
import { ReviewVideoPlayer } from '../components/ReviewVideoPlayer';
import { usePersistentState } from '../hooks/usePersistentState';
import { useVideoController } from '../hooks/useVideoController';
import type { Attribution, AuditEvent, FinalizationStatus, MediaVersion, Session, WorkingState, WorkspaceSummary } from '../types/api';
import type { CaseId } from '../types/domain';

interface Props {
  attribution: Attribution;
  events: AuditEvent[];
  session: Session | null;
  workingState?: WorkingState;
  workspace: WorkspaceSummary;
  onError: (message?: string) => void;
  onEventsChange: (events: AuditEvent[]) => void;
  onRequireAttribution: () => void;
}

export function FinalResultsPage({
  attribution,
  events,
  onError,
  onEventsChange,
  onRequireAttribution,
  session,
  workingState,
  workspace,
}: Props) {
  const caseIds = Object.keys(workspace.cases);
  const defaultCaseId = caseIds.includes('day') ? 'day' : caseIds[0] || '';
  const finalUiScope = `${workspaceIdentityForUi(workspace)}:${session?.session_id || 'no-session'}:final`;
  const [persistedCaseId, setCaseId] = usePersistentState<CaseId>(`${finalUiScope}:case`, defaultCaseId);
  const caseId = workspace.cases[persistedCaseId] ? persistedCaseId : defaultCaseId;
  const manifest = workspace.cases[caseId];
  const [caseFrame, setCaseFrame] = usePersistentState<number>(`${finalUiScope}:${caseId}:frame`, 0);
  const [selectedRole, setSelectedRole] = usePersistentState<string>(`${finalUiScope}:${caseId}:version`, 'latest_reviewed');
  const [finalStatus, setFinalStatus] = useState<Record<string, FinalizationStatus>>({});
  const [busy, setBusy] = useState(false);
  const [confirmingFinal, setConfirmingFinal] = useState(false);
  const status = finalStatus[caseId];

  if (!manifest) {
    return (
      <section className="case-panel active">
        <div className="fatal-error">No review cases are available in this workspace.</div>
      </section>
    );
  }

  const materializedLatestRole = workingState?.materialized_latest_reviewed?.[caseId] as MediaVersion | undefined;
  const finalRole = status?.finalized && status.final_video
    ? ({
        label: 'Final',
        path: status.final_video,
        role: 'final_reviewed',
        sha256: status.final_video_sha256 as string | null | undefined,
        source_frame_end: manifest.source_frame_end,
        source_frame_start: manifest.source_frame_start,
      } satisfies MediaVersion)
    : undefined;
  const roles = publicVideoRoles(manifest.video_roles, materializedLatestRole, finalRole);
  const selectedVersion =
    roles.find((role) => role.role === selectedRole) ||
    roles.find((role) => role.role === 'latest_reviewed') ||
    roles[0];
  const controller = useVideoController(manifest, selectedVersion, caseFrame, setCaseFrame);
  const finalizationReady = Boolean(status?.ready);
  const finalizationRequest = recordOrUndefined(status?.finalization_request);
  const historicalFinal = recordOrUndefined(status?.historical_final);

  useEffect(() => {
    let cancelled = false;
    if (!session?.session_id || !caseId) return undefined;
    setConfirmingFinal(false);
    const load = async () => {
      try {
        const result = await loadFinalizationStatus({
          case_id: caseId,
          session_id: session.session_id,
        });
        if (!cancelled) setFinalStatus((current) => ({ ...current, [caseId]: result }));
      } catch {
        if (!cancelled) setFinalStatus((current) => {
          const next = { ...current };
          delete next[caseId];
          return next;
        });
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [caseId, session?.session_id, workingState?.active_projection_hash, workingState?.current_event_chain_head]);

  const createFinal = async () => {
    if (!session) {
      onError('Start review before creating Final.');
      return;
    }
    if (!attribution?.display_name) {
      onRequireAttribution();
      return;
    }
    if (!finalizationRequest) {
      onError('Refresh finalization status before creating Final.');
      return;
    }
    setBusy(true);
    onError(undefined);
    try {
      const { event, result } = await finalizeCaseRequest({
        session_id: session.session_id,
        actor: attribution,
        case_id: caseId,
        finalization_request: finalizationRequest,
      });
      setFinalStatus((current) => ({ ...current, [caseId]: result }));
      if (event) onEventsChange([...events, event]);
    } catch (err) {
      onError(`Could not create Final: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
      setConfirmingFinal(false);
    }
  };

  const lifecycleLine = finalRole
    ? 'Final current'
    : historicalFinal
      ? 'Final historical — case reopened'
      : finalizationReady
        ? 'Ready to create Final'
        : status
          ? 'Full-video review required'
          : 'Reviewed video not ready';

  return (
    <section className="case-panel active">
      <div className="case-header">
        <div>
          <h2>Final Results</h2>
          <p className="case-subtitle">Final media appears only after backend gates pass.</p>
        </div>
        <div className="segmented">
          {caseIds.map((item) => (
            <button
              key={item}
              type="button"
              className={caseId === item ? 'active' : ''}
              onClick={() => setCaseId(item)}
            >
              {caseLabel(item, workspace.cases[item]?.display_name)}
            </button>
          ))}
        </div>
      </div>
      <div className="final-layout">
        <ReviewVideoPlayer
          controller={controller}
          manifest={manifest}
          roles={roles}
          finalRole={finalRole}
          sessionId={session?.session_id}
          selectedRole={selectedVersion?.role || selectedRole}
          version={selectedVersion}
          onVersionChange={setSelectedRole}
        />
        <aside className="side-panel final-status">
          <h3>{manifest.display_name}</h3>
          <div className={`status-box ${finalizationReady ? 'success' : 'warning'}`}>
            <strong>{lifecycleLine}</strong>
          </div>
          {finalizationReady && !finalRole ? (
            confirmingFinal ? (
              <div className="status-box warning">
                <strong>Create the immutable Final for {caseLabel(caseId, manifest.display_name)} from the current reviewed video and full-video approval?</strong>
                <dl className="compact-facts">
                  <dt>Latest revision</dt>
                  <dd>{String(finalizationRequest?.['latest_reviewed_revision_id'] || '')}</dd>
                  <dt>Reviewed regions</dt>
                  <dd>{String(finalizationRequest?.['reviewed_region_set_sha256'] || '')}</dd>
                  <dt>Full-video approval</dt>
                  <dd>{String(finalizationRequest?.['sweep_acknowledgement_sha256'] || '')}</dd>
                </dl>
                <div className="button-row">
                  <button type="button" onClick={() => setConfirmingFinal(false)} disabled={busy}>
                    Cancel
                  </button>
                  <button type="button" className="primary" onClick={createFinal} disabled={busy}>
                    {busy ? 'Creating Final…' : 'Create Final'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="button-row">
                <button
                  type="button"
                  className="primary"
                  onClick={() => setConfirmingFinal(true)}
                  disabled={busy}
                >
                  Create Final
                </button>
              </div>
            )
          ) : null}
          {status ? (
            status.finalized ? (
              <div className="status-box success">Final media is available.</div>
            ) : historicalFinal ? (
              <div className="status-box warning">
                Historical Final {String(historicalFinal['finalization_id'] || '')} is no longer current.
              </div>
            ) : (
              <div className="status-box warning">
                <strong>Final media unavailable</strong>
                {finalizationBlockerGroups(status).length ? (
                  <ul>
                    {finalizationBlockerGroups(status).map((group) => (
                      <li key={group.key}>{group.label} ({group.count})</li>
                    ))}
                  </ul>
                ) : null}
                <ul>
                  {(status.blockers || []).map((blocker) => (
                    <li key={blocker}>{blocker}</li>
                  ))}
                </ul>
              </div>
            )
          ) : (
            <p className="muted">No finalization result has been checked for this local session.</p>
          )}
        </aside>
      </div>
    </section>
  );
}

function caseLabel(caseId: string, displayName?: string): string {
  if (caseId === 'day') return 'Day';
  if (caseId === 'night') return 'Night';
  return displayName || caseId;
}

function recordOrUndefined(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
}

function publicVideoRoles(roles: MediaVersion[], latestRole?: MediaVersion, finalRole?: MediaVersion): MediaVersion[] {
  const byRole = new Map(roles.map((role) => [role.role, role]));
  const automatic = byRole.get('automatic') || byRole.get('current_result');
  return [
    byRole.get('original'),
    automatic ? { ...automatic, label: 'Automatic', role: 'automatic' as const } : undefined,
    latestRole ? { ...latestRole, label: 'Latest reviewed', role: 'latest_reviewed' as const } : undefined,
    finalRole ? { ...finalRole, label: 'Final', role: 'final_reviewed' as const } : undefined,
  ].filter((role): role is MediaVersion => Boolean(role));
}

function workspaceIdentityForUi(workspace: WorkspaceSummary): string {
  const identity = workspace.workspace?.workspace_manifest_sha256
    || workspace.workspace?.workspace_identity_hash
    || workspace.workspace?.source_sha256
    || workspace.backend_workspace_schema_version
    || 'unknown-workspace';
  return String(identity).replace(/[^A-Za-z0-9_-]/g, '_');
}

interface FinalizationBlockerGroup {
  count: number;
  key: string;
  label: string;
}

function finalizationBlockerGroups(status?: FinalizationStatus): FinalizationBlockerGroup[] {
  const raw = status?.['blocker_groups'];
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    .map((item) => ({
      count: Number(item.count || 0),
      key: String(item.key || ''),
      label: String(item.label || ''),
    }))
    .filter((item) => item.key && item.label);
}
