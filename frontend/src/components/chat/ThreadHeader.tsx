import React from 'react';
import type { ChatThread } from '../../types/chat';

interface ThreadHeaderProps {
  thread?: ChatThread | null;
}

export const ThreadHeader: React.FC<ThreadHeaderProps> = ({ thread }) => {
  return (
    <header className="flex items-center justify-between px-4 py-2 border-b border-neutral-200 dark:border-neutral-700">
      <h2 className="text-sm font-medium truncate text-neutral-800 dark:text-neutral-100">
        {thread ? thread.title : 'Chat'}
      </h2>
      {/* Placeholder for action buttons */}
    </header>
  );
};
