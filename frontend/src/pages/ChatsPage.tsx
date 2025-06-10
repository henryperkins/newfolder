import React from 'react';
import { Card } from '@/components/common';

export const ChatsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Recent Chats</h1>
        <p className="text-gray-600">View your recent chat conversations</p>
      </div>

      <Card>
        <div className="text-center py-12">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Chat History Coming Soon
          </h2>
          <p className="text-gray-600">
            Chat functionality will be implemented in Phase 3.
          </p>
        </div>
      </Card>
    </div>
  );
};