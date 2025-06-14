import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '@/stores';
import { EmptyDashboard } from '@/components/dashboard';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const projects = useProjectStore((state) => state.projects);
  const isLoadingProjects = useProjectStore((state) => state.isLoadingProjects);
  const fetchProjects = useProjectStore((state) => state.fetchProjects);
  const hasFetched = useRef(false);

  useEffect(() => {
    if (projects.length === 0 && !isLoadingProjects && !hasFetched.current) {
      hasFetched.current = true;
      fetchProjects();
    }
  }, [projects.length, isLoadingProjects]);

  const handleProjectCreated = (projectId: string) => {
    navigate(`/projects/${projectId}`);
  };

  // Show empty dashboard if no projects exist
  if (!isLoadingProjects && projects.length === 0) {
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
