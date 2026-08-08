import { useState } from 'react';
import { finalizeCase as finalizeCaseRequest } from '../api/client';
import { ReviewVideoPlayer } from '../components/ReviewVideoPlayer';
import { usePersistentState } from '../hooks/usePersistentState';
import { useVideoController } from '../hooks/useVideoController';
import type { Attribution, AuditEvent, FinalizationStatus, MediaVersion, Session, WorkspaceSummary } from '../types/api';
import type { CaseId } from '../types/domain';

interface Props {
  attribution: Attribution;
  events: AuditEvent[];
  session: Session | null;
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
  workspace,
}: Props) {
  const [caseId, setCaseId] = usePersistentState<CaseId>('final:case', 'day');
  const manifest = workspace.cases[caseId];
  const [caseFrame, setCaseFrame] = usePersistentState<number>(`final:${caseId}:frame`, 0);
  const [selectedRole, setSelectedRole] = usePersistentState<string>(`final:${caseId}:version`, 'latest_reviewed');
  const [finalStatus, setFinalStatus] = useState<Record<string, FinalizationStatus>>({});
  const [busy, setBusy] = useState(false);

  const roles = publicVideoRoles(manifest.video_roles);
  const selectedVersion =
    roles.find((role) => role.role === selectedRole) ||
    roles.find((role) => role.role === 'latest_reviewed') ||
    roles[0];
  const controller = useVideoController(manifest, selectedVersion, caseFrame, setCaseFrame);

  const finalize = async () => {
    if (!session) {
      onError('Start review before checking finalization.');
      return;
    }
    if (!attribution?.display_name) {
      onRequireAttribution();
      return;
    }
    setBusy(true);
    onError(undefined);
    try {
      const { event, result } = await finalizeCaseRequest({
        session_id: session.session_id,
        actor: attribution,
        case_id: caseId,
      });
      setFinalStatus((current) => ({ ...current, [caseId]: result }));
      if (event) onEventsChange([...events, event]);
    } catch (err) {
      onError(`Finalization check failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const status = finalStatus[caseId];

  return (
    <section className="case-panel active">
      <div className="case-header">
        <div>
          <h2>Final Results</h2>
          <p className="case-subtitle">Final media appears only after backend gates pass.</p>
        </div>
        <div className="segmented">
          <button type="button" className={caseId === 'day' ? 'active' : ''} onClick={() => setCaseId('day')}>
            Day
          </button>
          <button type="button" className={caseId === 'night' ? 'active' : ''} onClick={() => setCaseId('night')}>
            Night
          </button>
        </div>
      </div>
      <div className="final-layout">
        <ReviewVideoPlayer
          controller={controller}
          manifest={manifest}
          roles={roles}
          selectedRole={selectedVersion?.role || selectedRole}
          version={selectedVersion}
          onVersionChange={setSelectedRole}
        />
        <aside className="side-panel final-status">
          <h3>{manifest.display_name}</h3>
          <div className="button-row">
            <button type="button" className="primary" onClick={finalize} disabled={busy}>
              Check finalization
            </button>
          </div>
          {status ? (
            status.finalized ? (
              <div className="status-box success">Final media is available.</div>
            ) : (
              <div className="status-box warning">
                <strong>Final media unavailable</strong>
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

function publicVideoRoles(roles: MediaVersion[]): MediaVersion[] {
  const byRole = new Map(roles.map((role) => [role.role, role]));
  const automatic = byRole.get('automatic') || byRole.get('current_result');
  const latest = byRole.get('latest_reviewed') || automatic;
  return [
    byRole.get('original'),
    automatic ? { ...automatic, label: 'Automatic', role: 'automatic' as const } : undefined,
    latest ? { ...latest, label: 'Latest reviewed', role: 'latest_reviewed' as const } : undefined,
  ].filter((role): role is MediaVersion => Boolean(role));
}
