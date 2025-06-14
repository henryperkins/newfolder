import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { X } from 'lucide-react';
import { Button, Input } from '@/components/common';
import { useProjectStore } from '@/stores';
import { ProjectTemplate, CreateProjectData } from '@/types';
import { cn } from '@/utils';

const projectSchema = z.object({
  name: z.string().min(3, 'Name must be at least 3 characters').max(100, 'Name must be less than 100 characters'),
  description: z.string().max(500, 'Description must be less than 500 characters').optional(),
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/, 'Invalid color format'),
  tags: z.array(z.string()).max(10, 'Maximum 10 tags allowed')
});

type ProjectFormData = z.infer<typeof projectSchema>;

interface ProjectCreationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (projectId: string) => void;
  template?: ProjectTemplate;
  initialName?: string;
  editProject?: {
    id: string;
    name: string;
    description?: string;
    color: string;
    tags: string[];
  };
}

const colorOptions = [
  '#6366f1', // Indigo
  '#10b981', // Green
  '#f59e0b', // Amber
  '#8b5cf6', // Purple
  '#3b82f6', // Blue
  '#ec4899', // Pink
  '#ef4444', // Red
  '#6b7280', // Gray
];

export const ProjectCreationModal: React.FC<ProjectCreationModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  template,
  initialName,
  editProject
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tagInput, setTagInput] = useState('');
  const [currentTags, setCurrentTags] = useState<string[]>([]);

  const { createProject, updateProject, projects } = useProjectStore();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    setValue,
    watch,
    setError: setFieldError
  } = useForm<ProjectFormData>({
    resolver: zodResolver(projectSchema),
    defaultValues: {
      color: template?.color || '#6366f1',
      tags: []
    }
  });

  const watchedColor = watch('color');

  // Initialize form with template data or edit data
  useEffect(() => {
    if (editProject) {
      setValue('name', editProject.name);
      setValue('description', editProject.description || '');
      setValue('color', editProject.color);
      setCurrentTags(editProject.tags);
      setValue('tags', editProject.tags);
    } else if (template) {
      setValue('name', template.name);
      setValue('description', template.description);
      setValue('color', template.color);
      setCurrentTags(template.suggested_tags);
      setValue('tags', template.suggested_tags);
    } else if (initialName) {
      setValue('name', initialName);
    }
  }, [editProject, template, initialName, setValue]);

  // Update form tags when currentTags changes
  useEffect(() => {
    setValue('tags', currentTags);
  }, [currentTags, setValue]);

  const checkNameUniqueness = (name: string) => {
    const existingProject = projects.find(
      p => p.name.toLowerCase() === name.toLowerCase()
    );
    if (existingProject) {
      setFieldError('name', { message: 'Project name already exists' });
      return false;
    }
    return true;
  };

  const handleAddTag = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const tag = tagInput.trim();
      if (tag && !currentTags.includes(tag) && currentTags.length < 10) {
        setCurrentTags([...currentTags, tag]);
        setTagInput('');
      }
    }
  };

  const handleRemoveTag = (index: number) => {
    setCurrentTags(currentTags.filter((_, i) => i !== index));
  };

  const onSubmit = async (data: ProjectFormData) => {
    if (!editProject && !checkNameUniqueness(data.name)) {
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      if (editProject) {
        // Update existing project
        await updateProject(editProject.id, {
          name: data.name,
          description: data.description || undefined,
          color: data.color,
          tags: currentTags
        });
        onSuccess(editProject.id);
      } else {
        // Create new project
        const projectData: CreateProjectData = {
          name: data.name,
          description: data.description || undefined,
          color: data.color,
          template_id: template?.id,
          tags: currentTags
        };

        const project = await createProject(projectData);
        onSuccess(project.id);
      }
      handleClose();
    } catch (err: unknown) {
      const message = err instanceof Error && 'response' in err &&
        typeof err.response === 'object' && err.response &&
        'data' in err.response &&
        typeof err.response.data === 'object' && err.response.data &&
        'detail' in err.response.data
        ? String(err.response.data.detail)
        : `Failed to ${editProject ? 'update' : 'create'} project. Please try again.`;
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setError(null);
    setCurrentTags([]);
    setTagInput('');
    reset();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">
              {editProject ? 'Edit Project' : template ? `Create ${template.name}` : 'Create New Project'}
            </h2>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Template Preview */}
          {template && (
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-3 mb-2">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-semibold"
                  style={{ backgroundColor: template.color }}
                >
                  {template.name[0]}
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">{template.name}</h3>
                  <p className="text-sm text-gray-600">{template.description}</p>
                </div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {error && (
              <div className="p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-md">
                {error}
              </div>
            )}

            {/* Project Name */}
            <div>
              <Input
                {...register('name')}
                label="Project Name"
                placeholder="Enter project name"
                error={errors.name?.message}
                required
                onBlur={(e) => checkNameUniqueness(e.target.value)}
              />
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description
              </label>
              <textarea
                {...register('description')}
                placeholder="Describe your project (optional)"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows={3}
              />
              {errors.description && (
                <p className="mt-1 text-sm text-red-600">{errors.description.message}</p>
              )}
            </div>

            {/* Color Picker */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Project Color
              </label>
              <div className="flex gap-2">
                {colorOptions.map((color) => (
                  <button
                    key={color}
                    type="button"
                    onClick={() => setValue('color', color)}
                    className={cn(
                      'w-8 h-8 rounded-lg border-2 transition-all',
                      watchedColor === color
                        ? 'border-gray-900 scale-110'
                        : 'border-gray-300 hover:border-gray-400'
                    )}
                    style={{ backgroundColor: color }}
                    data-testid={`color-picker-${color.slice(1)}`}
                  />
                ))}
              </div>
            </div>

            {/* Tags */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tags
              </label>

              {/* Current Tags */}
              {currentTags.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-3">
                  {currentTags.map((tag, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-gray-700 bg-gray-100 rounded-full"
                    >
                      {tag}
                      <button
                        type="button"
                        onClick={() => handleRemoveTag(index)}
                        className="text-gray-500 hover:text-gray-700"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}

              {/* Tag Input */}
              <input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleAddTag}
                placeholder="Type and press Enter to add tags"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={currentTags.length >= 10}
              />
              <p className="mt-1 text-xs text-gray-500">
                Press Enter or comma to add tags. Max 10 tags.
              </p>
            </div>

            {/* Actions */}
            <div className="flex gap-3 pt-4">
              <Button
                type="button"
                variant="secondary"
                onClick={handleClose}
                className="flex-1"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                isLoading={isSubmitting}
                disabled={isSubmitting}
                className="flex-1"
              >
                {isSubmitting 
                  ? editProject ? 'Updating...' : 'Creating...' 
                  : editProject ? 'Update Project' : 'Create Project'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
