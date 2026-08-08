import { useEffect, useState } from 'react';

const namespace = `final-review-react-v1:${window.location.origin}${window.location.pathname.split('/app/')[0]}`;

export function storageKey(key: string): string {
  return `${namespace}:${key}`;
}

export function usePersistentState<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const loadValue = (nextKey: string, fallback: T): T => {
    const raw = window.localStorage.getItem(storageKey(nextKey));
    if (raw === null) return fallback;
    try {
      return JSON.parse(raw) as T;
    } catch {
      return raw as T;
    }
  };
  const [state, setState] = useState<{ key: string; value: T }>(() => ({
    key,
    value: loadValue(key, initialValue),
  }));

  useEffect(() => {
    setState({ key, value: loadValue(key, initialValue) });
  }, [key, initialValue]);

  useEffect(() => {
    if (state.key === key) {
      window.localStorage.setItem(storageKey(key), JSON.stringify(state.value));
    }
  }, [key, state]);

  return [state.value, (value: T) => setState({ key, value })];
}

export function readStoredString(key: string): string | null {
  const raw = window.localStorage.getItem(storageKey(key));
  if (raw === null) return null;
  try {
    return JSON.parse(raw) as string;
  } catch {
    return raw;
  }
}

export function writeStoredString(key: string, value: string): void {
  window.localStorage.setItem(storageKey(key), JSON.stringify(value));
}

export function removeStoredValue(key: string): void {
  window.localStorage.removeItem(storageKey(key));
}
