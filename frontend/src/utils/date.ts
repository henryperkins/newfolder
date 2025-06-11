/**
 * Date / time formatting helpers shared across the front-end.
 */

export function formatRelativeTime(iso: string | number | Date): string {
  const now = new Date();
  const date = new Date(iso);
  const diff = now.getTime() - date.getTime();

  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return 'just now';

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;

  return date.toLocaleDateString();
}

export function groupByDate<T extends { created_at: string | Date }>(items: T[]) {
  return items.reduce<Record<string, T[]>>((acc, item) => {
    const key = new Date(item.created_at).toDateString();
    acc[key] = acc[key] || [];
    acc[key].push(item);
    return acc;
  }, {});
}
