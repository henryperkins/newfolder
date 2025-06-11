import React from 'react';
import { cn } from '@/utils';
import { ExamplePrompt } from '@/types';
import { Card } from '@/components/common';

interface ExamplePromptsProps {
  prompts: ExamplePrompt[];
  onPromptSelect: (prompt: string) => void;
  variant?: 'grid' | 'inline';
  loading?: boolean;
}

const iconMap: Record<string, React.FC<{ className?: string }>> = {
  FileText: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
  Code2: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
  ),
  Calendar: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  TrendingUp: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  ),
};

const categoryColors = {
  productivity: 'bg-blue-100 text-blue-700',
  creative: 'bg-purple-100 text-purple-700',
  analysis: 'bg-green-100 text-green-700',
  code: 'bg-orange-100 text-orange-700',
};

export const ExamplePrompts: React.FC<ExamplePromptsProps> = ({
  prompts,
  onPromptSelect,
  variant = 'grid',
  loading = false
}) => {
  const handlePromptClick = (prompt: ExamplePrompt) => {
    onPromptSelect(prompt.prompt);
  };

  if (loading) {
    return (
      <div className={cn(
        variant === 'grid' 
          ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'
          : 'flex flex-wrap gap-3'
      )}>
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-4 animate-pulse">
            <div className="w-8 h-8 bg-gray-200 rounded-lg mb-3"></div>
            <div className="h-4 bg-gray-200 rounded mb-2"></div>
            <div className="h-3 bg-gray-200 rounded w-3/4"></div>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className={cn(
      variant === 'grid' 
        ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'
        : 'flex flex-wrap gap-3'
    )}>
      {prompts.map((prompt) => {
        const IconComponent = iconMap[prompt.icon];
        
        return (
          <Card
            key={prompt.id}
            className={cn(
              'p-4 cursor-pointer transition-all duration-200 hover:scale-102 hover:shadow-lg border-2 border-transparent hover:border-blue-200',
              variant === 'inline' && 'flex-shrink-0'
            )}
            onClick={() => handlePromptClick(prompt)}
          >
            <div className="flex items-start gap-3">
              <div className={cn(
                'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
                categoryColors[prompt.category]
              )}>
                {IconComponent && <IconComponent className="w-5 h-5" />}
              </div>
              
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 mb-1 text-sm">
                  {prompt.title}
                </h3>
                <p className="text-xs text-gray-600 line-clamp-2">
                  {prompt.prompt}
                </p>
              </div>
            </div>
            
            {variant === 'grid' && (
              <div className="mt-3 flex items-center justify-between">
                <span className={cn(
                  'inline-block px-2 py-1 text-xs font-medium rounded-full',
                  categoryColors[prompt.category]
                )}>
                  {prompt.category}
                </span>
                <svg 
                  className="w-4 h-4 text-gray-400 group-hover:text-blue-500" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={2} 
                    d="M9 5l7 7-7 7" 
                  />
                </svg>
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
};