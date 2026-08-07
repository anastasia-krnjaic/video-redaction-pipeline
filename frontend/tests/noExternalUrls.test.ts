import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

describe('runtime URL policy', () => {
  it('does not include remote runtime URLs in source', () => {
    const srcRoot = path.resolve(process.cwd(), 'src');
    const files = walk(srcRoot).filter((file) => /\.(ts|tsx|css|html)$/.test(file));
    const offenders = files.filter((file) => /https?:\/\//.test(fs.readFileSync(file, 'utf8')));
    expect(offenders).toEqual([]);
  });
});

function walk(root: string): string[] {
  const entries = fs.readdirSync(root, { withFileTypes: true });
  return entries.flatMap((entry) => {
    const fullPath = path.join(root, entry.name);
    return entry.isDirectory() ? walk(fullPath) : [fullPath];
  });
}
