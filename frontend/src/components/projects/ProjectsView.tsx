import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  List,
  Plus,
  Search,
  Filter,
  SortAsc,
  SortDesc
} from 'lucide-react';
import cn from 'clsx';
import { useProjectStore } from '@/stores';
import { Project, ProjectsQueryParams } from '@/types';
import { Button } from '@/components/common';
import { ProjectCard } from './ProjectCard';
import { ProjectCreationModal } from './ProjectCreationModal';
import { EmptyDashboard } from '@/components/dashboard';

export const ProjectsView: React.FC = () => {
  const navigate = useNavigate();

  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);

  const {
    projects,
    isLoadingProjects,
    projectsError,
    projectListView,
    projectFilters,
    setProjectListView,
    setProjectFilters,
    fetchProjects,
    deleteProject
  } = useProjectStore();

  useEffect(() => {
    fetchProjects();
  }, []); // Zustand functions are stable

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    const params: ProjectsQueryParams = {
      ...projectFilters,
      search: query || undefined
    };
    setProjectFilters(params);
    fetchProjects(params);
  };

  const handleSortChange = (sortBy: 'created' | 'updated' | 'name') => {
    const newOrder = projectFilters.sort_by === sortBy && projectFilters.order === 'desc' ? 'asc' : 'desc';
    const params: ProjectsQueryParams = {
      ...projectFilters,
      sort_by: sortBy,
      order: newOrder
    };
    setProjectFilters(params);
    fetchProjects(params);
  };

  const handleArchiveToggle = () => {
    const params: ProjectsQueryParams = {
      ...projectFilters,
      include_archived: !projectFilters.include_archived
    };
    setProjectFilters(params);
    fetchProjects(params);
  };

  const handleProjectEdit = (project: Project) => {
    setSelectedProject(project);
    setShowEditModal(true);
  };

  const handleProjectDelete = async (projectId: string) => {
    if (window.confirm('Are you sure you want to delete this project? This action cannot be undone.')) {
      try {
        await deleteProject(projectId);
      } catch (error) {
        console.error('Failed to delete project:', error);
      }
    }
  };

  const handleProjectCreated = (projectId: string) => {
    navigate(`/projects/${projectId}`);
  };

  const filteredProjects = projects;

  // Show empty state if no projects
  if (!isLoadingProjects && projects.length === 0 && !searchQuery) {
    return <EmptyDashboard onProjectCreated={handleProjectCreated} />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Projects</h1>
          <p className="text-gray-600">
            Manage your workspaces and organize your content
          </p>
        </div>

        <Button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Project
        </Button>
      </div>

      {/* Controls */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search projects..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-2 p-1 bg-gray-100 rounded-lg">
          <button
            onClick={() => setProjectListView('grid')}
            className={cn(
              'p-2 rounded-md transition-colors',
              projectListView === 'grid'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            <Grid className="w-4 h-4" />
          </button>
          <button
            onClick={() => setProjectListView('list')}
            className={cn(
              'p-2 rounded-md transition-colors',
              projectListView === 'list'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            <List className="w-4 h-4" />
          </button>
        </div>

        {/* Sort Options */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleSortChange('name')}
            className={cn(
              'flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-md transition-colors',
              projectFilters.sort_by === 'name'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            )}
          >
            Name
            {projectFilters.sort_by === 'name' && (
              projectFilters.order === 'desc' ?
              <SortDesc className="w-3 h-3" /> :
              <SortAsc className="w-3 h-3" />
            )}
          </button>

          <button
            onClick={() => handleSortChange('updated')}
            className={cn(
              'flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-md transition-colors',
              projectFilters.sort_by === 'updated'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            )}
          >
            Updated
            {projectFilters.sort_by === 'updated' && (
              projectFilters.order === 'desc' ?
              <SortDesc className="w-3 h-3" /> :
              <SortAsc className="w-3 h-3" />
            )}
          </button>

          <button
            onClick={() => handleSortChange('created')}
            className={cn(
              'flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-md transition-colors',
              projectFilters.sort_by === 'created'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            )}
          >
            Created
            {projectFilters.sort_by === 'created' && (
              projectFilters.order === 'desc' ?
              <SortDesc className="w-3 h-3" /> :
              <SortAsc className="w-3 h-3" />
            )}
          </button>
        </div>

        {/* Archive Toggle */}
        <button
          onClick={handleArchiveToggle}
          className={cn(
            'flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors',
            projectFilters.include_archived
              ? 'bg-gray-200 text-gray-900'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
          )}
        >
          <Filter className="w-4 h-4" />
          {projectFilters.include_archived ? 'Hide Archived' : 'Show Archived'}
        </button>
      </div>

      {/* Error State */}
      {projectsError && (
        <div className="p-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-md">
          {projectsError}
        </div>
      )}

      {/* Loading State */}
      {isLoadingProjects ? (
        <div className={cn(
          projectListView === 'grid'
            ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'
            : 'space-y-4'
        )}>
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="animate-pulse">
              {projectListView === 'grid' ? (
                <div className="p-6 bg-white rounded-lg border">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gray-200 rounded-xl"></div>
                    <div className="flex-1">
                      <div className="h-4 bg-gray-200 rounded w-3/4 mb-1"></div>
                      <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                    </div>
                  </div>
                  <div className="h-3 bg-gray-200 rounded mb-2"></div>
                  <div className="h-3 bg-gray-200 rounded w-2/3"></div>
                </div>
              ) : (
                <div className="p-4 bg-white rounded-lg border">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-gray-200 rounded-full"></div>
                    <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                    <div className="h-3 bg-gray-200 rounded w-16"></div>
                    <div className="h-3 bg-gray-200 rounded w-20"></div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <>
          {/* Projects Grid/List */}
          {filteredProjects.length > 0 ? (
            <div className={cn(
              projectListView === 'grid'
                ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'
                : 'space-y-4'
            )}>
              {projectListView === 'grid' ? (
                filteredProjects.map((project) => (
                  <ProjectCard
                    key={project.id}
                    project={project}
                    onEdit={handleProjectEdit}
                    onDelete={handleProjectDelete}
                  />
                ))
              ) : (
                filteredProjects.map((project) => (
                  <ProjectListItem
                    key={project.id}
                    project={project}
                    onEdit={handleProjectEdit}
                    onDelete={handleProjectDelete}
                  />
                ))
              )}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                <Search className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No projects found
              </h3>
              <p className="text-gray-600 mb-6">
                {searchQuery
                  ? `No projects match "${searchQuery}"`
                  : 'Get started by creating your first project'
                }
              </p>
              {!searchQuery && (
                <Button onClick={() => setShowCreateModal(true)}>
                  Create Project
                </Button>
              )}
            </div>
          )}
        </>
      )}

      {/* Modals */}
      <ProjectCreationModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={handleProjectCreated}
      />

      {selectedProject && (
        <ProjectCreationModal
          isOpen={showEditModal}
          onClose={() => {
            setShowEditModal(false);
            setSelectedProject(null);
          }}
          onSuccess={() => {
            setShowEditModal(false);
            setSelectedProject(null);
          }}
          // TODO: Add edit mode props
        />
      )}
    </div>
  );
};

interface ProjectListItemProps {
  project: Project;
  onEdit: (_project: Project) => void;
  onDelete: (_projectId: string) => void;
}

const ProjectListItem: React.FC<ProjectListItemProps> = ({
  project,
  onEdit: _onEdit,
  onDelete: _onDelete
}) => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(`/projects/${project.id}`);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div
      onClick={handleClick}
      className="p-4 bg-white rounded-lg border hover:shadow-md transition-shadow cursor-pointer"
    >
      <div className="flex items-center gap-4">
        <div
          className="w-3 h-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: project.color }}
        />

        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-gray-900 truncate">
            {project.name}
          </h3>
        </div>

        <div className="flex items-center gap-6 text-sm text-gray-500">
          {project.tags.length > 0 && (
            <div className="flex gap-1">
              {project.tags.slice(0, 2).map(tag => (
                <span key={tag.id} className="px-2 py-1 bg-gray-100 rounded text-xs">
                  {tag.name}
                </span>
              ))}
              {project.tags.length > 2 && (
                <span className="px-2 py-1 bg-gray-100 rounded text-xs">
                  +{project.tags.length - 2}
                </span>
              )}
            </div>
          )}

          <span>{formatDate(project.last_activity_at)}</span>

          <div className="flex gap-4">
            <span>{project.stats?.chat_count || 0} chats</span>
            <span>{project.stats?.document_count || 0} docs</span>
          </div>
        </div>
      </div>
    </div>
  );
};
