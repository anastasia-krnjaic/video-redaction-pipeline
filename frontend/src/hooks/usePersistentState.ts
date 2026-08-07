import { useEffect, useState } from 'react';

const namespace = `final-review-react-v1:${window.location.origin}${window.location.pathname.split('/app/')[0]}`;

export function storageKey(key: string): string {
  return `${namespace}:${key}`;
}

export function usePersistentState<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    const raw = window.localStorage.getItem(storageKey(key));
    if (raw === null) return initialValue;
    try {
      return JSON.parse(raw) as T;
    } catch {
      return raw as T;
    }
  });

  useEffect(() => {
    window.localStorage.setItem(storageKey(key), JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
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
