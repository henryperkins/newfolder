import React from 'react';
import { Card } from '@/components/common';

export const ProjectsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Projects</h1>
        <p className="text-gray-600">Manage your AI productivity projects</p>
      </div>

      <Card>
        <div className="text-center py-12">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Projects Coming Soon
          </h2>
          <p className="text-gray-600">
            Project management functionality will be implemented in Phase 2.
          </p>
        </div>
      </Card>
    </div>
  );
};