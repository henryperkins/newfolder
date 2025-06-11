import React from 'react';
import type { ChatThread } from '../../types/chat';

export interface RecentChatsSidebarProps {
  threads: ChatThread[];
  activeThreadId?: string | null;
  onSelect: (threadId: string) => void;
  onClose?: () => void;
}

export const RecentChatsSidebar: React.FC<RecentChatsSidebarProps> = ({ threads, activeThreadId, onSelect }) => {
  return (
    <aside className="w-64 shrink-0 border-r border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 overflow-y-auto">
      <h3 className="px-4 py-3 text-sm font-semibold text-neutral-700 dark:text-neutral-300">Recent Chats</h3>
      <ul>
        {threads.map((t) => (
          <li key={t.id}>
            <button
              className={`block w-full text-left px-4 py-2 text-sm truncate hover:bg-neutral-100 dark:hover:bg-neutral-800 ${
                t.id === activeThreadId ? 'font-medium bg-neutral-100 dark:bg-neutral-800' : ''
              }`}
              onClick={() => onSelect(t.id)}
            >
              {t.title || 'Untitled'}
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
};
