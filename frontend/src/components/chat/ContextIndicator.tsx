import React from 'react';

export const ContextIndicator: React.FC<{ percent: number }> = ({ percent }) => {
  const colour = percent > 95 ? 'bg-red-500' : percent > 80 ? 'bg-yellow-400' : 'bg-emerald-500';
  return (
    <div className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
      <div className="w-20 h-1 bg-neutral-200 dark:bg-neutral-700 rounded">
        <div className={`h-full rounded ${colour}`} style={{ width: `${percent}%` }} />
      </div>
      <span>{percent}%</span>
    </div>
  );
};
