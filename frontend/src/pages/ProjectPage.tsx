import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, Folder, MessageSquare, Clock } from 'lucide-react';

import { useProjectStore } from '@/stores/projectStore';
import { projectApi } from '@/utils/api';
import { useDocumentStore } from '@/stores/documentStore';
import { Button, Card } from '@/components/common';
import { DocumentManager } from '@/components/documents/DocumentManager';
import { NotificationBell } from '@/components/documents/NotificationBell';
import { cn, formatRelativeTime } from '@/utils';

type TabKey = 'documents' | 'chats' | 'activity' | 'settings';

const TABS: { key: TabKey; label: string; icon: React.ElementType }[] = [
  { key: 'documents', label: 'Documents', icon: Folder },
  { key: 'chats', label: 'Chats', icon: MessageSquare },
  { key: 'activity', label: 'Activity', icon: Clock },
  { key: 'settings', label: 'Settings', icon: Clock }, // reuse Clock icon for now
];

export const ProjectPage: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();

  const {
    projects,
    setSelectedProject,
    selectedProject,
    updateProject,
  } = useProjectStore();

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = (searchParams.get('tab') || 'documents') as TabKey;

  const setTab = (tab: TabKey) => {
    setSearchParams({ tab });
  };

  // Fetch project if not present
  useEffect(() => {
    const loadProject = async () => {
      if (!projectId) return;
      const existing = projects.find((p) => p.id === projectId);
      if (existing) {
        setSelectedProject(existing);
        setIsLoading(false);
        return;
      }
      try {
        const project = await projectApi.getProject(projectId);
        setSelectedProject(project);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load project');
      } finally {
        setIsLoading(false);
      }
    };
    loadProject();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // -------------------------------------------------------------------
  // WebSocket – listen for document status updates                      
  // -------------------------------------------------------------------

  const { handleDocumentStatusUpdate } = useDocumentStore();

  useEffect(() => {
    if (!projectId) return;

    const loc = window.location;
    const wsProtocol = loc.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${wsProtocol}://${loc.host}/api/projects/${projectId}/documents/ws`);

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === 'document_status') {
          handleDocumentStatusUpdate(data.document_id, data.status, data.error);
        }
      } catch {
        /* ignore malformed */
      }
    };

    ws.onerror = () => {
      // network issue – auto-close; UI can rely on polling fallback if needed
      ws.close();
    };

    return () => ws.close();
  }, [projectId, handleDocumentStatusUpdate]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (error || !selectedProject) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">{error || 'Project not found'}</p>
        <Button onClick={() => navigate('/projects')}>Back to Projects</Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            className="p-2 hover:bg-gray-100 rounded"
            onClick={() => navigate('/projects')}
            title="Back to Projects"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <h1 className="text-2xl font-bold" style={{ color: selectedProject.color }}>
            {selectedProject.name}
          </h1>
          <span className="text-sm text-gray-500">• Last updated {formatRelativeTime(selectedProject.updated_at)}</span>
        </div>
        <NotificationBell />
      </div>

      {/* Tab Nav */}
      <div className="flex gap-4 border-b">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={cn(
              'flex items-center gap-2 py-2 px-3 -mb-px border-b-2',
              activeTab === key
                ? 'border-blue-600 text-blue-600 font-medium'
                : 'border-transparent text-gray-600 hover:text-gray-900',
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'documents' && <DocumentManager />}

      {activeTab === 'chats' && (
        <Card className="p-8 text-center text-gray-500">Chats coming soon…</Card>
      )}

      {activeTab === 'activity' && (
        <Card className="p-8 text-center text-gray-500">Activity feed coming soon…</Card>
      )}

      {activeTab === 'settings' && (
        <Card className="p-8 text-center text-gray-500">Settings coming soon…</Card>
      )}
    </div>
  );
};
