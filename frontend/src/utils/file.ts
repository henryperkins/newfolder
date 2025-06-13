/**
 * Human-readable file-size formatter.
 *
 * Converts a byte count into an abbreviated string with appropriate unit.
 * Examples:
 *   1024        → "1 KB"
 *   1_572_864   → "1.5 MB"
 *   3_221_225_472 → "3 GB"
 */

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let idx = 0;
  let value = bytes;

  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }

  const rounded = value < 10 ? value.toFixed(1) : Math.round(value).toString();
  return `${rounded.replace(/\.0$/, '')} ${units[idx]}`;
}
