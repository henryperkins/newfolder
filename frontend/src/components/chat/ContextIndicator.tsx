import React from 'react';

interface ContextIndicatorProps {
  usedTokens: number;
  maxTokens: number;
}

/**
 * Displays a horizontal bar that turns yellow when 80 % of the context window
 * is used and red when 95 % is exceeded, as outlined in Phase-3 spec.
 */
export const ContextIndicator: React.FC<ContextIndicatorProps> = ({
  usedTokens,
  maxTokens,
}) => {
  const ratio = Math.min(usedTokens / maxTokens, 1);
  const percent = Math.round(ratio * 100);

  let barColor = 'bg-green-500';
  if (ratio >= 0.95) barColor = 'bg-red-500';
  else if (ratio >= 0.8) barColor = 'bg-yellow-400';

  return (
    <div className="my-2">
      <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
        <div
          className={`${barColor} h-full`} // width via inline style
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="text-right text-xs text-slate-500 mt-1">
        {usedTokens}/{maxTokens} tokens
      </div>
    </div>
  );
};

export default ContextIndicator;
