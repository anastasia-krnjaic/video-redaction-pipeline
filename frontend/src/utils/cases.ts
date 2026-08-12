export function caseTabLabel(caseId: string, displayName?: string): string {
  if (caseId === 'day') return 'Day';
  if (caseId === 'night') return 'Night';
  return displayName || caseId;
}
