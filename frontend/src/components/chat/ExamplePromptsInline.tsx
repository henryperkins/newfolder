import React from 'react';
import { ExamplePrompt } from '@/types';
import { cn } from '@/utils/cn';

export interface ExamplePromptsInlineProps {
  prompts: ExamplePrompt[];
  onPromptSelect: (prompt: string) => void;
  maxPrompts?: number;
  className?: string;
}

// Render a pill-style inline prompt suggestion list. Appears underneath the
// ChatInputBar when the thread is empty or during long idle periods.

export const ExamplePromptsInline: React.FC<ExamplePromptsInlineProps> = ({
  prompts,
  onPromptSelect,
  maxPrompts = 4,
  className,
}) => {
  if (!prompts.length) return null;

  return (
    <div className={cn('flex flex-wrap gap-2 mt-3', className)}>
      {prompts.slice(0, maxPrompts).map((p) => (
        <button
          key={p.id}
          type="button"
          onClick={() => onPromptSelect(p.prompt)}
          className="px-3 py-1 rounded-full bg-slate-100 hover:bg-slate-200 text-sm text-slate-700 transition-colors"
        >
          {p.title}
        </button>
      ))}
    </div>
  );
};

export default ExamplePromptsInline;
