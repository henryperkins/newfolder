// SelectProjectModal.tsx
// ---------------------------------------------------------------------------
// Modal allowing the user to pick an existing project (or create a new one)
// before starting a chat.
//
// • Strictly typed (no implicit any)
// • No side-effects on import
// • Uses DI-friendly utilities (cn, useProjectStore)
// ---------------------------------------------------------------------------

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  KeyboardEvent
} from 'react';
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
  /** If present, shown as “Starting chat: "…"" at top of modal */
  initialMessage?: string;
}

export const SelectProjectModal: React.FC<SelectProjectModalProps> = ({
  isOpen,
  onClose,
  onProjectSelect,
  onCreateNew,
  initialMessage
}) => {
  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(
    null
  );

  // -------------------------------------------------------------------------
  // Data
  // -------------------------------------------------------------------------
  const { projects, isLoadingProjects, fetchProjects } = useProjectStore();

  // Fetch (non-archived) projects each time the modal opens
  useEffect(() => {
    if (isOpen) {
      void fetchProjects({ include_archived: false });
    }
  }, [isOpen, fetchProjects]);

  // Memoise filtering to avoid re-computing on every render
  const filteredProjects = useMemo(
    () =>
      projects.filter(
        (p) =>
          !p.is_archived &&
          p.name.toLowerCase().includes(searchQuery.toLowerCase())
      ),
    [projects, searchQuery]
  );

  // -------------------------------------------------------------------------
  // Event Handlers
  // -------------------------------------------------------------------------
  const handleProjectSelect = useCallback((projectId: string) => {
    setSelectedProjectId(projectId);
  }, []);

  const handleClose = useCallback(() => {
    setSearchQuery('');
    setSelectedProjectId(null);
    onClose();
  }, [onClose]);

  const handleConfirm = useCallback(() => {
    if (selectedProjectId) {
      onProjectSelect(selectedProjectId);
      handleClose();
    }
  }, [onProjectSelect, selectedProjectId, handleClose]);

  const handleCreateNew = useCallback(() => {
    onCreateNew();
    handleClose();
  }, [onCreateNew, handleClose]);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter' && selectedProjectId) {
      handleConfirm();
      return;
    }
    if (e.key === 'Escape') {
      handleClose();
      return;
    }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;

    e.preventDefault();
    const currentIndex = filteredProjects.findIndex(
      (p) => p.id === selectedProjectId
    );
    let newIndex: number;

    if (e.key === 'ArrowDown') {
      newIndex =
        currentIndex < filteredProjects.length - 1 ? currentIndex + 1 : 0;
    } else {
      newIndex =
        currentIndex > 0 ? currentIndex - 1 : filteredProjects.length - 1;
    }

    if (filteredProjects[newIndex]) {
      setSelectedProjectId(filteredProjects[newIndex].id);
    }
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="flex max-h-[80vh] w-full max-w-md flex-col rounded-lg bg-white">
        {/* Header ----------------------------------------------------------- */}
        <div className="border-b border-gray-200 p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">
              Select a Project for Your Chat
            </h2>
            <button
              onClick={handleClose}
              className="text-gray-400 transition-colors hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {initialMessage && (
            <div className="mb-4 rounded-md border border-blue-200 bg-blue-50 p-3">
              <p className="text-sm text-blue-800">
                Starting chat: &quot;{initialMessage}&quot;
              </p>
            </div>
          )}

          {/* Search -------------------------------------------------------- */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search projects…"
              autoFocus
              className="w-full rounded-md border border-gray-300 py-2 pl-10 pr-4 focus:border-transparent focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Project list ----------------------------------------------------- */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoadingProjects ? (
            // Skeletons
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div
                  key={i}
                  className="flex animate-pulse items-center gap-3 p-3"
                >
                  <div className="h-3 w-3 rounded-full bg-gray-200" />
                  <div className="flex-1">
                    <div className="mb-1 h-4 w-3/4 rounded bg-gray-200" />
                    <div className="h-3 w-1/2 rounded bg-gray-200" />
                  </div>
                </div>
              ))}
            </div>
          ) : filteredProjects.length ? (
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
            <div className="py-8 text-center text-gray-500">
              {searchQuery
                ? 'No projects match your search'
                : 'No projects found'}
            </div>
          )}
        </div>

        {/* Footer ----------------------------------------------------------- */}
        <div className="space-y-3 border-t border-gray-200 p-4">
          <button
            onClick={handleCreateNew}
            className="flex w-full items-center justify-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-4 py-3 text-sm font-medium text-blue-600 transition-colors hover:bg-blue-100"
          >
            <Plus className="h-4 w-4" />
            Create New Project
          </button>

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

// ===========================================================================
// Internal Components
// ===========================================================================

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
  const formatDate = (iso: string): string => {
    const now = new Date();
    const date = new Date(iso);
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / 86_400_000); // ms per day

    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        'flex cursor-pointer items-center gap-3 rounded-lg p-3 transition-all',
        isSelected
          ? 'border-2 border-blue-200 bg-blue-50'
          : 'border-2 border-transparent hover:bg-gray-50'
      )}
    >
      {/* Radio indicator */}
      <div
        className={cn(
          'flex h-4 w-4 items-center justify-center rounded-full border-2',
          isSelected ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
        )}
      >
        {isSelected && <div className="h-2 w-2 rounded-full bg-white" />}
      </div>

      {/* Project colour dot */}
      <div
        className="h-3 w-3 flex-shrink-0 rounded-full"
        style={{ backgroundColor: project.color }}
      />

      {/* Name + metadata */}
      <div className="min-w-0 flex-1">
        <h3 className="truncate font-medium text-gray-900">
          {project.name}
        </h3>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>Last activity: {formatDate(project.last_activity_at)}</span>
          {project.stats?.chat_count ? (
            <>
              <span>•</span>
              <span>{project.stats.chat_count} chats</span>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
};
