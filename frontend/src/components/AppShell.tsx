import type { ReactNode } from 'react';
import type { Attribution, Session } from '../types/api';
import type { TopLevelRoute } from '../types/domain';
import { CaseTabs } from './CaseTabs';
import { SessionHeader } from './SessionHeader';

interface Props {
  attribution: Attribution;
  busy?: boolean;
  cases: Array<{ caseId: string; label: string }>;
  children: ReactNode;
  error?: string;
  route: TopLevelRoute;
  resumableSessionId?: string;
  session: Session | null;
  sessionLoadKind?: 'new' | 'resumed';
  onExport: () => void;
  onOpenAttribution: () => void;
  onResumeSession: () => void;
  onRouteChange: (route: TopLevelRoute) => void;
  onStartSession: () => void;
}

export function AppShell({
  attribution,
  busy,
  cases,
  children,
  error,
  onExport,
  onOpenAttribution,
  onResumeSession,
  onRouteChange,
  onStartSession,
  resumableSessionId,
  route,
  session,
  sessionLoadKind,
}: Props) {
  return (
    <>
      <SessionHeader
        attribution={attribution}
        busy={busy}
        error={error}
        onExport={onExport}
        onOpenAttribution={onOpenAttribution}
        onResumeSession={onResumeSession}
        onStartSession={onStartSession}
        resumableSessionId={resumableSessionId}
        session={session}
        sessionLoadKind={sessionLoadKind}
      />
      <CaseTabs active={route} cases={cases} onChange={onRouteChange} />
      <main>{children}</main>
    </>
  );
}
