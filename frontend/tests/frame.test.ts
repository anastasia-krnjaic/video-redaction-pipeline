import { describe, expect, it } from 'vitest';
import type { Case, Task } from '../src/types/api';
import { caseToSourceFrame, formatTime, fpsValue, frameToSeconds, sourceToCaseFrame, taskTimestamp } from '../src/utils/frame';

const manifest: Case = {
  case_id: 'day',
  display_name: 'Day',
  fps: '30000/1001',
  frame_count: 300,
  source_frame_start: 1828,
  source_frame_end: 2127,
  source_path: 'media/day/original_reference.mp4',
  status: 'ready',
  video_roles: [],
};

describe('frame utilities', () => {
  it('maps case and source frames without drift', () => {
    expect(sourceToCaseFrame(manifest, 1828)).toBe(0);
    expect(sourceToCaseFrame(manifest, 1838)).toBe(10);
    expect(caseToSourceFrame(manifest, 10)).toBe(1838);
  });

  it('steps on rational frame rates and displays HH:MM:SS', () => {
    expect(fpsValue('30000/1001')).toBeCloseTo(29.97003);
    expect(frameToSeconds(manifest, 300)).toBeCloseTo(9.9766);
    expect(formatTime(3723.9)).toBe('01:02:03');
  });

  it('formats task timestamps from exact source-frame starts', () => {
    const task = {
      case_id: 'day',
      category: 'license_plate',
      review_task_id: 'task',
      source_frame_start: 1858,
      source_frame_end: 1858,
      technical_only: false,
      title: 'Task',
      what_to_inspect: 'Review',
    } satisfies Task;
    expect(taskTimestamp(manifest, task)).toBe('00:00:01');
  });
});
