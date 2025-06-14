import React, { useState, useEffect } from 'react';
import { X, Search, Plus } from 'lucide-react';
import { Button } from '@/components/common';
import { useProjectStore } from '@/stores';
import { Project } from '@/types';
import { cn } from '@/utils';

interface SelectProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onProjectSelect: (projectId: string) => void;
  onCreateNew: () => void;
  initialMessage?: string;
}

export const SelectProjectModal: React.FC<SelectProjectModalProps> = ({
  isOpen,
  onClose,
  onProjectSelect,
  onCreateNew,
  initialMessage
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  const { projects, isLoadingProjects, fetchProjects } = useProjectStore();

  useEffect(() => {
    if (isOpen) {
      fetchProjects();
    }
  }, [isOpen, fetchProjects]);

  const filteredProjects = projects.filter(project =>
    !project.is_archived &&
    project.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleProjectSelect = (projectId: string) => {
    setSelectedProjectId(projectId);
  };

  const handleConfirm = () => {
    if (selectedProjectId) {
      onProjectSelect(selectedProjectId);
      handleClose();
    }
  };

  const handleCreateNew = () => {
    onCreateNew();
    handleClose();
  };

  const handleClose = () => {
    setSearchQuery('');
    setSelectedProjectId(null);
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && selectedProjectId) {
      handleConfirm();
    } else if (e.key === 'Escape') {
      handleClose();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const currentIndex = filteredProjects.findIndex(p => p.id === selectedProjectId);
      let newIndex;

      if (e.key === 'ArrowDown') {
        newIndex = currentIndex < filteredProjects.length - 1 ? currentIndex + 1 : 0;
      } else {
        newIndex = currentIndex > 0 ? currentIndex - 1 : filteredProjects.length - 1;
      }

      if (filteredProjects[newIndex]) {
        setSelectedProjectId(filteredProjects[newIndex].id);
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-md w-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              Select a Project for Your Chat
            </h2>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {initialMessage && (
            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-blue-800">
                Starting chat: &quot;{initialMessage}&quot;
              </p>
            </div>
          )}

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search projects..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              autoFocus
            />
          </div>
        </div>

        {/* Project List */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoadingProjects ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="flex items-center gap-3 p-3 animate-pulse">
                  <div className="w-3 h-3 bg-gray-200 rounded-full"></div>
                  <div className="flex-1">
                    <div className="h-4 bg-gray-200 rounded w-3/4 mb-1"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : filteredProjects.length > 0 ? (
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {filteredProjects.map((project) => (
                <ProjectItem
                  key={project.id}
                  project={project}
                  isSelected={selectedProjectId === project.id}
                  onClick={() => handleProjectSelect(project.id)}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              {searchQuery ? 'No projects found matching your search' : 'No projects found'}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="p-4 border-t border-gray-200 space-y-3">
          {/* Create New Project */}
          <button
            onClick={handleCreateNew}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Create New Project
          </button>

          {/* Confirm Selection */}
          <div className="flex gap-3">
            <Button
              variant="secondary"
              onClick={handleClose}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={!selectedProjectId}
              className="flex-1"
            >
              Start Chat
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

interface ProjectItemProps {
  project: Project;
  isSelected: boolean;
  onClick: () => void;
}

const ProjectItem: React.FC<ProjectItemProps> = ({
  project,
  isSelected,
  onClick
}) => {
  const formatDate = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return 'Today';
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return `${days} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        'flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all',
        isSelected
          ? 'bg-blue-50 border-2 border-blue-200'
          : 'hover:bg-gray-50 border-2 border-transparent'
      )}
    >
      {/* Radio Button */}
      <div className={cn(
        'w-4 h-4 rounded-full border-2 flex items-center justify-center',
        isSelected
          ? 'border-blue-500 bg-blue-500'
          : 'border-gray-300'
      )}>
        {isSelected && (
          <div className="w-2 h-2 bg-white rounded-full" />
        )}
      </div>

      {/* Project Color */}
      <div
        className="w-3 h-3 rounded-full flex-shrink-0"
        style={{ backgroundColor: project.color }}
      />

      {/* Project Info */}
      <div className="flex-1 min-w-0">
        <h3 className="font-medium text-gray-900 truncate">
          {project.name}
        </h3>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>Last activity: {formatDate(project.last_activity_at)}</span>
          {project.stats && project.stats.chat_count > 0 && (
            <>
              <span>•</span>
              <span>{project.stats.chat_count} chats</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
