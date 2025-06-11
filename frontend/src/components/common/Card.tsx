import React from 'react';
import { cn } from '@/utils';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  padding?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  className,
  padding = true,
  ...props
}) => {
  return (
    <div className={cn('card', padding && 'p-6', className)} {...props}>
      {children}
    </div>
  );
};