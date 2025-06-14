import React, { useState } from 'react';
import { MoreVertical, Edit3, Copy, Archive, Trash2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Project } from '@/types';
import { Card } from '@/components/common';
import { cn } from '@/utils';

interface ProjectCardProps {
  project: Project;
  onEdit: (project: Project) => void;
  onDelete: (projectId: string) => void;
  onDuplicate?: (project: Project) => void;
  onArchive?: (projectId: string) => void;
  onNavigate?: (projectId: string) => void;
}

export const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  onEdit,
  onDelete,
  onDuplicate,
  onArchive,
  onNavigate
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const navigate = useNavigate();

  const handleCardClick = (e: React.MouseEvent) => {
    // Don't navigate if clicking on the menu button or menu items
    if ((e.target as HTMLElement).closest('[data-menu]')) {
      return;
    }
    
    if (onNavigate) {
      onNavigate(project.id);
    } else {
      navigate(`/projects/${project.id}`);
    }
  };

  const handleMenuToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowMenu(!showMenu);
  };

  const handleMenuAction = (action: string) => {
    setShowMenu(false);
    
    switch (action) {
      case 'edit':
        onEdit(project);
        break;
      case 'duplicate':
        onDuplicate?.(project);
        break;
      case 'archive':
        onArchive?.(project.id);
        break;
      case 'delete':
        onDelete(project.id);
        break;
    }
  };

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
    <Card
      className={cn(
        'cursor-pointer transition-all duration-200 hover:shadow-lg border-2 border-transparent',
        isHovered && 'border-blue-200 shadow-md scale-102'
      )}
      onClick={handleCardClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setShowMenu(false);
      }}
      data-testid="project-card"
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center text-white font-semibold text-lg flex-shrink-0"
              style={{ backgroundColor: project.color }}
            >
              {project.name[0].toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-gray-900 truncate text-lg">
                {project.name}
              </h3>
              <p className="text-sm text-gray-500">
                Last active {formatDate(project.last_activity_at)}
              </p>
            </div>
          </div>

          {/* Menu Button */}
          <div className="relative" data-menu>
            <button
              onClick={handleMenuToggle}
              className={cn(
                'p-2 rounded-lg transition-all',
                showMenu || isHovered
                  ? 'text-gray-600 bg-gray-100'
                  : 'text-gray-400 hover:text-gray-600 hover:bg-gray-50'
              )}
              data-testid="project-menu"
            >
              <MoreVertical className="w-4 h-4" />
            </button>

            {/* Dropdown Menu */}
            {showMenu && (
              <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10">
                <button
                  onClick={() => handleMenuAction('edit')}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Edit3 className="w-4 h-4" />
                  Edit Details
                </button>
                <button
                  onClick={() => handleMenuAction('duplicate')}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Copy className="w-4 h-4" />
                  Duplicate Project
                </button>
                <button
                  onClick={() => handleMenuAction('archive')}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Archive className="w-4 h-4" />
                  Archive
                </button>
                <div className="border-t border-gray-100 my-1" />
                <button
                  onClick={() => handleMenuAction('delete')}
                  className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        {project.description && (
          <p className="text-gray-600 text-sm mb-4 line-clamp-2">
            {project.description}
          </p>
        )}

        {/* Tags */}
        {project.tags && project.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {project.tags.slice(0, 3).map((tag) => (
              <span
                key={tag.id}
                className="inline-block px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded-full"
              >
                {tag.name}
              </span>
            ))}
            {project.tags.length > 3 && (
              <span className="inline-block px-2 py-1 text-xs font-medium text-gray-500 bg-gray-50 rounded-full">
                +{project.tags.length - 3}
              </span>
            )}
          </div>
        )}

        {/* Stats */}
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center gap-4">
            <span>{project.stats?.chat_count || 0} chats</span>
            <span>{project.stats?.document_count || 0} documents</span>
          </div>
          
          {project.is_archived && (
            <span className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded-full">
              <Archive className="w-3 h-3" />
              Archived
            </span>
          )}
        </div>
      </div>
    </Card>
  );
};