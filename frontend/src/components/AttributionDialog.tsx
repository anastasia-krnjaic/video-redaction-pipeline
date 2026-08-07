import { useEffect, useRef, useState } from 'react';

interface Props {
  open: boolean;
  onCancel: () => void;
  onSubmit: (displayName: string) => Promise<void>;
}

export function AttributionDialog({ onCancel, onSubmit, open }: Props) {
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setError('');
      setSaving(false);
      window.setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  if (!open) return null;

  const submit = async () => {
    const name = displayName.trim();
    if (!name) {
      setError('Reviewer name is required.');
      return;
    }
    setSaving(true);
    setError('');
    try {
      await onSubmit(name);
      setDisplayName('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="modal open" aria-modal="true" role="dialog" aria-labelledby="attribution-title">
      <div className="modal-card">
        <h2 id="attribution-title">Reviewer attribution</h2>
        <p className="muted">This local review records a declared reviewer name for export and finalization events.</p>
        <label>
          Reviewer name
          <input
            ref={inputRef}
            autoComplete="name"
            value={displayName}
            onChange={(event) => setDisplayName(event.currentTarget.value)}
          />
        </label>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="button-row">
          <button type="button" className="primary" onClick={submit} disabled={saving}>
            Save attribution
          </button>
          <button type="button" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
        </div>
      </div>
    </section>
  );
}
