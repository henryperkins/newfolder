import React from 'react';
import { Card } from '@/components/common';

export const DashboardPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Welcome to your AI Productivity App</p>
      </div>

      <Card>
        <div className="text-center py-12">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Welcome to Phase 1!
          </h2>
          <p className="text-gray-600 mb-6">
            The foundation of your AI Productivity App is now complete. This includes:
          </p>
          <ul className="text-left max-w-md mx-auto space-y-2 text-gray-600">
            <li>✅ User authentication and security</li>
            <li>✅ Responsive layout with collapsible sidebar</li>
            <li>✅ Settings and profile management</li>
            <li>✅ Password reset functionality</li>
          </ul>
          <p className="text-gray-600 mt-6">
            Phase 2 will add project management and the empty state dashboard experience.
          </p>
        </div>
      </Card>
    </div>
  );
};