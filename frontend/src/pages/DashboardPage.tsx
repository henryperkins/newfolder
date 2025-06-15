import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '@/stores';
import { EmptyDashboard } from '@/components/dashboard';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const projects = useProjectStore((state) => state.projects);
  const isLoadingProjects = useProjectStore((state) => state.isLoadingProjects);
  const projectsError = useProjectStore((state) => state.projectsError);
  const fetchProjects = useProjectStore((state) => state.fetchProjects);

  useEffect(() => {
    // Always fetch projects when component mounts to ensure fresh data
    fetchProjects();
  }, []); // Empty dependency array - fetchProjects is stable in Zustand

  const handleProjectCreated = async (projectId: string) => {
    // Store automatically adds new project to the list, no need to refetch
    // Just navigate to the new project
    navigate(`/projects/${projectId}`);
  };

  // Show loading state while fetching projects
  if (isLoadingProjects) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading projects...</p>
        </div>
      </div>
    );
  }

  // Show error state if there was an error fetching projects
  if (projectsError) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center max-w-md">
          <div className="text-red-600 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Error loading projects</h3>
          <p className="text-gray-600 mb-4">{projectsError}</p>
          <button
            onClick={() => fetchProjects()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // Show empty dashboard if no projects exist
  if (projects.length === 0) {
    return <EmptyDashboard onProjectCreated={handleProjectCreated} />;
  }

  // Show regular dashboard with project overview
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Welcome back to your workspace</p>
      </div>

      {/* Project overview cards, recent activity, etc. will be added here */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.slice(0, 6).map((project) => (
          <div
            key={project.id}
            className="p-6 bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => navigate(`/projects/${project.id}`)}
          >
            <div className="flex items-center gap-3 mb-3">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: project.color }}
              />
              <h3 className="font-semibold text-gray-900 truncate">
                {project.name}
              </h3>
            </div>
            {project.description && (
              <p className="text-gray-600 text-sm mb-3 line-clamp-2">
                {project.description}
              </p>
            )}
            <div className="flex items-center justify-between text-sm text-gray-500">
              <span>{project.stats?.chat_count || 0} chats</span>
              <span>{new Date(project.last_activity_at).toLocaleDateString()}</span>
            </div>
          </div>
        ))}
      </div>

      {projects.length > 6 && (
        <div className="text-center">
          <button
            onClick={() => navigate('/projects')}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            View all projects →
          </button>
        </div>
      )}
    </div>
  );
};
