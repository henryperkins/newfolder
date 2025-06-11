import React, { useState } from 'react';
import { cn } from '@/utils';
import { ProjectTemplate } from '@/types';
import { Card, Button } from '@/components/common';

interface ProjectTemplatesProps {
  templates: ProjectTemplate[];
  onTemplateSelect: (template: ProjectTemplate) => void;
  loading?: boolean;
  variant?: 'carousel' | 'grid';
}

const iconMap: Record<string, React.FC<{ className?: string }>> = {
  Microscope: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  ),
  Rocket: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  PenTool: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
    </svg>
  ),
  Code2: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
  ),
  Target: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
    </svg>
  ),
  FileText: ({ className }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
};

export const ProjectTemplates: React.FC<ProjectTemplatesProps> = ({
  templates,
  onTemplateSelect,
  loading = false,
  variant = 'carousel'
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [hoveredTemplate, setHoveredTemplate] = useState<string | null>(null);

  const itemsPerView = {
    mobile: 1,
    tablet: 2,
    desktop: 3
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    const maxIndex = Math.max(0, templates.length - itemsPerView.desktop);
    setCurrentIndex((prev) => Math.min(maxIndex, prev + 1));
  };

  if (loading) {
    return (
      <div className={cn(
        variant === 'grid' 
          ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'
          : 'flex gap-6'
      )}>
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="p-6 animate-pulse">
            <div className="w-12 h-12 bg-gray-200 rounded-xl mb-4"></div>
            <div className="h-4 bg-gray-200 rounded mb-2"></div>
            <div className="h-3 bg-gray-200 rounded mb-4"></div>
            <div className="flex gap-2 mb-4">
              <div className="h-6 w-16 bg-gray-200 rounded-full"></div>
              <div className="h-6 w-20 bg-gray-200 rounded-full"></div>
            </div>
            <div className="h-8 bg-gray-200 rounded"></div>
          </Card>
        ))}
      </div>
    );
  }

  if (variant === 'grid') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {templates.map((template) => (
          <TemplateCard
            key={template.id}
            template={template}
            onSelect={onTemplateSelect}
            isHovered={hoveredTemplate === template.id}
            onHover={setHoveredTemplate}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Carousel Controls */}
      <div className="hidden lg:flex absolute top-1/2 -translate-y-1/2 left-0 right-0 justify-between pointer-events-none z-10">
        <Button
          variant="secondary"
          size="sm"
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          className="pointer-events-auto bg-white shadow-lg border"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </Button>
        
        <Button
          variant="secondary"
          size="sm"
          onClick={handleNext}
          disabled={currentIndex >= templates.length - itemsPerView.desktop}
          className="pointer-events-auto bg-white shadow-lg border"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Button>
      </div>

      {/* Carousel Track */}
      <div className="overflow-hidden">
        <div 
          className="flex transition-transform duration-300 ease-out gap-6"
          style={{
            transform: `translateX(-${currentIndex * (100 / itemsPerView.desktop)}%)`
          }}
        >
          {templates.map((template) => (
            <div
              key={template.id}
              className="flex-shrink-0 w-full md:w-1/2 lg:w-1/3"
            >
              <TemplateCard
                template={template}
                onSelect={onTemplateSelect}
                isHovered={hoveredTemplate === template.id}
                onHover={setHoveredTemplate}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Dots Indicator */}
      <div className="flex justify-center mt-6 gap-2">
        {Array.from({ length: Math.ceil(templates.length / itemsPerView.desktop) }).map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrentIndex(i)}
            className={cn(
              'w-3 h-3 rounded-full transition-colors',
              i === Math.floor(currentIndex / itemsPerView.desktop)
                ? 'bg-blue-500'
                : 'bg-gray-300 hover:bg-gray-400'
            )}
          />
        ))}
      </div>
    </div>
  );
};

interface TemplateCardProps {
  template: ProjectTemplate;
  onSelect: (template: ProjectTemplate) => void;
  isHovered: boolean;
  onHover: (id: string | null) => void;
}

const TemplateCard: React.FC<TemplateCardProps> = ({
  template,
  onSelect,
  isHovered,
  onHover
}) => {
  const IconComponent = iconMap[template.icon];

  return (
    <Card
      className="p-6 cursor-pointer transition-all duration-200 hover:scale-105 hover:shadow-xl border-2 border-transparent hover:border-blue-200 h-full"
      onMouseEnter={() => onHover(template.id)}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect(template)}
      data-testid={`template-${template.id}`}
    >
      <div className="flex flex-col h-full">
        {/* Icon */}
        <div 
          className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 text-white"
          style={{ backgroundColor: template.color }}
        >
          {IconComponent && <IconComponent className="w-6 h-6" />}
        </div>

        {/* Content */}
        <h3 className="font-semibold text-gray-900 mb-2 text-lg">
          {template.name}
        </h3>
        
        <p className="text-gray-600 text-sm mb-4 flex-1 line-clamp-2">
          {template.description}
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-4">
          {template.suggested_tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="inline-block px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded-full"
            >
              {tag}
            </span>
          ))}
          {template.suggested_tags.length > 3 && (
            <span className="inline-block px-2 py-1 text-xs font-medium text-gray-500 bg-gray-50 rounded-full">
              +{template.suggested_tags.length - 3}
            </span>
          )}
        </div>

        {/* Use Template Button */}
        <Button
          variant={isHovered ? 'primary' : 'secondary'}
          size="sm"
          className={cn(
            'w-full transition-all duration-200',
            isHovered ? 'opacity-100' : 'opacity-0 lg:opacity-100'
          )}
        >
          Use Template
        </Button>
      </div>
    </Card>
  );
};