import type { MediaVersion } from '../types/api';

interface Props {
  roles: MediaVersion[];
  selectedRole: string;
  finalRole?: MediaVersion;
  onChange: (role: string) => void;
}

const preferredOrder = ['original', 'automatic', 'latest_reviewed', 'final_reviewed'];

export function VersionSelector({ finalRole, onChange, roles, selectedRole }: Props) {
  const available = roles
    .filter((role) => preferredOrder.includes(role.role))
    .concat(finalRole && !roles.some((role) => role.role === 'final_reviewed') ? [finalRole] : [])
    .sort((left, right) => preferredOrder.indexOf(left.role) - preferredOrder.indexOf(right.role));

  return (
    <label className="version-selector">
      Version
      <select value={selectedRole} onChange={(event) => onChange(event.currentTarget.value)}>
        {available.map((role) => (
          <option key={`${role.role}:${role.path}`} value={role.role}>
            {roleLabel(role)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function roleLabel(role: MediaVersion): string {
  if (role.role === 'original') return 'Original';
  if (role.role === 'automatic') return 'Automatic';
  if (role.role === 'latest_reviewed') return 'Latest reviewed';
  if (role.role === 'final_reviewed') return 'Final';
  return role.label;
}
