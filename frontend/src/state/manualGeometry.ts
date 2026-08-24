export interface VideoDisplayBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface SourceDimensions {
  width: number;
  height: number;
}

export interface DisplayPoint {
  clientX: number;
  clientY: number;
}

export interface SourcePoint {
  x: number;
  y: number;
  scale: number;
  offsetX: number;
  offsetY: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function displayPointToSourcePoint(
  point: DisplayPoint,
  displayBox: VideoDisplayBox,
  source: SourceDimensions,
): SourcePoint {
  const scale = Math.min(displayBox.width / source.width, displayBox.height / source.height);
  const renderedWidth = source.width * scale;
  const renderedHeight = source.height * scale;
  const offsetX = (displayBox.width - renderedWidth) / 2;
  const offsetY = (displayBox.height - renderedHeight) / 2;
  return {
    x: clamp(Math.round(((point.clientX - displayBox.left - offsetX) / renderedWidth) * source.width), 0, source.width),
    y: clamp(Math.round(((point.clientY - displayBox.top - offsetY) / renderedHeight) * source.height), 0, source.height),
    scale,
    offsetX,
    offsetY,
  };
}
