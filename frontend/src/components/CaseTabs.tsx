import type { TopLevelRoute } from '../types/domain';

const tabs: Array<{ route: TopLevelRoute; label: string }> = [
  { route: 'day', label: 'Day' },
  { route: 'night', label: 'Night' },
  { route: 'final', label: 'Final Results' },
  { route: 'audit', label: 'Audit Trail' },
];

interface Props {
  active: TopLevelRoute;
  onChange: (route: TopLevelRoute) => void;
}

export function CaseTabs({ active, onChange }: Props) {
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
