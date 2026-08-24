import { describe, expect, it } from 'vitest';
import { displayPointToSourcePoint } from '../src/state/manualGeometry';

describe('manual correction display-to-source geometry', () => {
  it('maps a non-letterboxed drag to exact source pixels using round-to-nearest', () => {
    const display = { left: 100, top: 50, width: 960, height: 540 };
    const source = { width: 1920, height: 1080 };

    const start = displayPointToSourcePoint({ clientX: 164.2, clientY: 111.6 }, display, source);
    const end = displayPointToSourcePoint({ clientX: 319.8, clientY: 220.1 }, display, source);

    expect(start).toMatchObject({ x: 128, y: 123, scale: 0.5, offsetX: 0, offsetY: 0 });
    expect(end).toMatchObject({ x: 440, y: 340, scale: 0.5, offsetX: 0, offsetY: 0 });
  });

  it('removes object-fit letterbox offsets before source-pixel rounding', () => {
    const display = { left: 10, top: 20, width: 1000, height: 700 };
    const source = { width: 1920, height: 1080 };

    const point = displayPointToSourcePoint(
      { clientX: 10 + 260.4166667, clientY: 20 + 68.75 + 104.1666667 },
      display,
      source,
    );

    expect(point.x).toBe(500);
    expect(point.y).toBe(200);
    expect(point.scale).toBeCloseTo(0.5208333333);
    expect(point.offsetX).toBeCloseTo(0);
    expect(point.offsetY).toBeCloseTo(68.75);
  });
});
