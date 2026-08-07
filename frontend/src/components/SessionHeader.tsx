import { Download, UserRound } from 'lucide-react';
import type { Attribution, Session } from '../types/api';

interface Props {
  attribution: Attribution;
  error?: string;
  session: Session | null;
  onStartSession: () => void;
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
  onStartSession,
  session,
}: Props) {
  const bannerClass = error ? 'session-banner error' : session ? 'session-banner ready' : 'session-banner';
  const text = error
    ? error
    : session
      ? `Local session ${session.session_id}${attribution?.display_name ? ` · Reviewer ${attribution.display_name}` : ' · anonymous until export/finalization'}`
      : 'No local review session yet. Start review to enable decisions and corrections.';

  return (
    <>
      <header className="app-header">
        <div>
          <h1>Final Review Workspace</h1>
          <p>Source-frame accurate redaction review.</p>
        </div>
        <div className="header-actions">
          <button type="button" className="primary" onClick={onStartSession} disabled={Boolean(session) || busy}>
            Start review
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
