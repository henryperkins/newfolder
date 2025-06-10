import React from 'react';
import { cn } from '@/utils';

interface ContentAreaProps {
  children: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  padding?: boolean;
  className?: string;
}

export const ContentArea: React.FC<ContentAreaProps> = ({
  children,
  maxWidth = 'xl',
  padding = true,
  className,
}) => {
  const maxWidthClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-4xl',
    xl: 'max-w-6xl',
    full: 'max-w-none',
  };

  return (
    <main
      className={cn(
        'flex-1 overflow-auto',
        padding && 'p-8',
        'bg-gray-50',
        className
      )}
    >
      <div className={cn('mx-auto', maxWidthClasses[maxWidth])}>
        {children}
      </div>
    </main>
  );
};