import type { TopLevelRoute } from '../types/domain';

interface Props {
  active: TopLevelRoute;
  cases: Array<{ caseId: string; label: string }>;
  onChange: (route: TopLevelRoute) => void;
}

export function CaseTabs({ active, cases, onChange }: Props) {
  const tabs: Array<{ route: TopLevelRoute; label: string }> = [
    ...cases.map((item) => ({ route: item.caseId, label: item.label })),
    { route: 'final', label: 'Final Results' },
    { route: 'audit', label: 'History' },
  ];
  return (
    <nav className="case-tabs" aria-label="Workspace pages">
      {tabs.map((tab) => (
        <button
          key={tab.route}
          type="button"
          className={active === tab.route ? 'active' : ''}
          onClick={() => onChange(tab.route)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
