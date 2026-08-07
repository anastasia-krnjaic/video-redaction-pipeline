import type { Case, Task } from '../types/api';

export function fpsValue(fps: string | undefined): number {
  if (!fps) return 30;
  const [num, den] = String(fps).split('/').map(Number);
  if (Number.isFinite(num) && Number.isFinite(den) && den) return num / den;
  return Number.isFinite(num) && num > 0 ? num : 30;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function sourceToCaseFrame(manifest: Case, sourceFrame: number): number {
  return clamp(sourceFrame - manifest.source_frame_start, 0, Math.max(0, manifest.frame_count - 1));
}

export function caseToSourceFrame(manifest: Case, caseFrame: number): number {
  return manifest.source_frame_start + clamp(caseFrame, 0, Math.max(0, manifest.frame_count - 1));
}

export function frameToSeconds(manifest: Case, caseFrame: number): number {
  return clamp(caseFrame, 0, Math.max(0, manifest.frame_count - 1)) / fpsValue(manifest.fps);
}

export function secondsToFrame(manifest: Case, seconds: number): number {
  return clamp(Math.round(seconds * fpsValue(manifest.fps)), 0, Math.max(0, manifest.frame_count - 1));
}

export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return [hours, minutes, secs].map((value) => String(value).padStart(2, '0')).join(':');
}

export function taskTimestamp(manifest: Case, task: Task): string {
  return formatTime(frameToSeconds(manifest, sourceToCaseFrame(manifest, task.source_frame_start)));
}

export function sourceSpan(task: Pick<Task, 'source_frame_start' | 'source_frame_end'>): string {
  return `${task.source_frame_start}-${task.source_frame_end}`;
}
