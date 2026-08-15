import { Download, UserRound } from 'lucide-react';
import type { Attribution, Session } from '../types/api';

interface Props {
  attribution: Attribution;
  error?: string;
  frontend?: Record<string, unknown>;
  session: Session | null;
  sessionLoadKind?: 'new' | 'resumed';
  resumableSessionId?: string;
  onStartSession: () => void;
  onResumeSession: () => void;
  onOpenAttribution: () => void;
  onExport: () => void;
  busy?: boolean;
}

export function SessionHeader({
  attribution,
  busy,
  error,
  frontend,
  onExport,
  onOpenAttribution,
  onResumeSession,
  onStartSession,
  resumableSessionId,
  session,
  sessionLoadKind,
}: Props) {
  const bannerClass = error ? 'session-banner error' : session ? 'session-banner ready' : 'session-banner';
  const sessionPrefix = sessionLoadKind === 'resumed' ? 'Resumed review session' : 'Active review session';
  const frontendBuildId = String(frontend?.served_build_id || frontend?.build_id || '');
  const frontendStatus = String(frontend?.served_asset_status || '');
  const text = error
    ? error
    : session
      ? `${sessionPrefix} ${session.session_id}${attribution?.display_name ? ` · Reviewer ${attribution.display_name}` : ' · anonymous until export/finalization'}`
      : resumableSessionId
        ? `No session loaded. Resume ${resumableSessionId} explicitly or start a new clean review session.`
        : 'No local review session loaded. Start a new review session to enable decisions and corrections.';

  return (
    <>
      <header className="app-header">
        <div>
          <h1>Final Review Workspace</h1>
          <p>Source-frame accurate redaction review.</p>
        </div>
        <div className="header-actions">
          {frontendBuildId ? (
            <details className={`build-identity ${frontendStatus === 'stale' ? 'stale' : ''}`}>
              <summary>Technical</summary>
              <span>UI build {frontendBuildId.slice(0, 12)}</span>
              {frontendStatus ? <span>{frontendStatus}</span> : null}
            </details>
          ) : null}
          {resumableSessionId && !session ? (
            <button type="button" onClick={onResumeSession} disabled={busy}>
              Resume review
            </button>
          ) : null}
          <button type="button" className="primary" onClick={onStartSession} disabled={busy}>
            {session ? 'New review session' : 'Start new review'}
          </button>
          <button type="button" onClick={onOpenAttribution} title="Reviewer attribution">
            <UserRound size={16} aria-hidden="true" />
            <span>Attribution</span>
          </button>
          <button type="button" onClick={onExport} title="Export session">
            <Download size={16} aria-hidden="true" />
            <span>Export</span>
          </button>
        </div>
      </header>
      <div id="sessionBanner" className={bannerClass}>
        {text}
      </div>
    </>
  );
}
