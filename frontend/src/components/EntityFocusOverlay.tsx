import { useMemo } from 'react';
import type { Case } from '../types/api';

export interface EntityFocusTarget {
  bbox: [number, number, number, number];
  entityId: string;
  label: string;
  sourceFrame: number;
}

interface Props {
  focus: EntityFocusTarget | null;
  manifest: Case;
}

export function EntityFocusOverlay({ focus, manifest }: Props) {
  const dimensions = useMemo(() => ({ width: manifest.width || 1920, height: manifest.height || 1080 }), [manifest.height, manifest.width]);
  if (!focus) return null;
  const [x1, y1, x2, y2] = focus.bbox;
  return (
    <div className="entity-focus-overlay" data-testid="entity-focus-overlay">
      <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <rect
          className="entity-focus-rect"
          x={x1}
          y={y1}
          width={Math.max(1, x2 - x1)}
          height={Math.max(1, y2 - y1)}
        />
        <text className="entity-focus-label" x={x1} y={Math.max(24, y1 - 10)}>
          {focus.label}
        </text>
      </svg>
    </div>
  );
}
