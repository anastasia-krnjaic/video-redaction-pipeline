import type { ReactNode } from 'react';
import type { Attribution, Session } from '../types/api';
import type { TopLevelRoute } from '../types/domain';
import { CaseTabs } from './CaseTabs';
import { SessionHeader } from './SessionHeader';

interface Props {
  attribution: Attribution;
  busy?: boolean;
  children: ReactNode;
  error?: string;
  route: TopLevelRoute;
  session: Session | null;
  onExport: () => void;
  onOpenAttribution: () => void;
  onRouteChange: (route: TopLevelRoute) => void;
  onStartSession: () => void;
}

export function AppShell({
  attribution,
  busy,
  children,
  error,
  onExport,
  onOpenAttribution,
  onRouteChange,
  onStartSession,
  route,
  session,
}: Props) {
  return (
    <>
      <SessionHeader
        attribution={attribution}
        busy={busy}
        error={error}
        onExport={onExport}
        onOpenAttribution={onOpenAttribution}
        onStartSession={onStartSession}
        session={session}
      />
      <CaseTabs active={route} onChange={onRouteChange} />
      <main>{children}</main>
    </>
  );
}
