import { Download, UserRound } from 'lucide-react';
import type { Attribution, Session } from '../types/api';

interface Props {
  attribution: Attribution;
  error?: string;
  session: Session | null;
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
  onExport,
  onOpenAttribution,
  onResumeSession,
  onStartSession,
  resumableSessionId,
  session,
}: Props) {
  const bannerClass = error ? 'session-banner error' : busy ? 'session-banner' : session ? 'session-banner ready' : 'session-banner';
  const text = error
    ? error
    : busy
      ? 'Preparing review session...'
      : session
      ? `Review session active · ${session.session_id}${attribution?.display_name ? ` · ${attribution.display_name}` : ''}`
      : resumableSessionId
        ? `Ready to resume ${resumableSessionId}.`
        : 'Preparing review session...';

  return (
    <>
      <header className="app-header">
        <div>
          <h1>Final Review Workspace</h1>
          <p>Source-frame accurate redaction review.</p>
        </div>
        <div className="header-actions">
          {resumableSessionId && !session ? (
            <button type="button" onClick={onResumeSession} disabled={busy}>
              Resume review
            </button>
          ) : null}
          <button type="button" className="primary" onClick={onStartSession} disabled={busy}>
            New review session
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
