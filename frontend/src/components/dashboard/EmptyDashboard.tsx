import React, { useState, useEffect } from 'react';
import { useProjectStore } from '@/stores';
import { ExamplePrompt, ProjectTemplate } from '@/types';
import { Button, Card } from '@/components/common';
import { ProjectTemplates } from './ProjectTemplates';
import { ExamplePrompts } from './ExamplePrompts';
import { ProjectCreationModal, SelectProjectModal } from '@/components/projects';

interface EmptyDashboardProps {
  onProjectCreated?: (projectId: string) => void;
}

const examplePrompts: ExamplePrompt[] = [
  {
    id: '1',
    icon: 'FileText',
    title: 'Summarize key points',
    prompt: 'Summarize key points from my meeting notes',
    category: 'productivity'
  },
  {
    id: '2',
    icon: 'Code2',
    title: 'Generate unit tests',
    prompt: 'Generate unit tests for this Python function',
    category: 'code'
  },
  {
    id: '3',
    icon: 'Calendar',
    title: 'Create timeline',
    prompt: 'Create a project timeline for Q2 launch',
    category: 'productivity'
  },
  {
    id: '4',
    icon: 'TrendingUp',
    title: 'Draft executive summary',
    prompt: 'Draft an executive summary of market research',
    category: 'analysis'
  }
];

export const EmptyDashboard: React.FC<EmptyDashboardProps> = ({
  onProjectCreated
}) => {
  const [chatInput, setChatInput] = useState('');
  const [showSelectProjectModal, setShowSelectProjectModal] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ProjectTemplate | null>(null);
  const [showCreateProjectModal, setShowCreateProjectModal] = useState(false);

  const {
    templates,
    isLoadingTemplates,
    fetchTemplates
  } = useProjectStore();

  useEffect(() => {
    fetchTemplates();
  }, []); // Zustand functions are stable

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (chatInput.trim()) {
      setShowSelectProjectModal(true);
    }
  };

  const handlePromptSelect = (prompt: string) => {
    setChatInput(prompt);
  };

  const handleTemplateSelect = (template: ProjectTemplate) => {
    setSelectedTemplate(template);
    setShowCreateProjectModal(true);
  };

  const handleCreateFirstProject = () => {
    setSelectedTemplate(null);
    setShowCreateProjectModal(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Hero Section */}
        <div className="text-center mb-12 animate-fade-in">
          <div className="mb-8">
            <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center">
              <svg
                className="w-10 h-10 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                />
              </svg>
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Organize Your Work with Projects
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Create focused workspaces for your ideas, research, and tasks.
              Keep everything organized and easily searchable.
            </p>
          </div>

          {/* Primary CTA */}
          <Button
            size="lg"
            onClick={handleCreateFirstProject}
            className="mb-8 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold px-8 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
          >
            Create Your First Project
          </Button>
        </div>

        {/* Quick Start Chat */}
        <Card className="mb-12 p-6 bg-white/70 backdrop-blur-sm border-2 border-dashed border-gray-200 hover:border-blue-300 transition-colors">
          <div className="text-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Or start with a quick chat
            </h3>
            <p className="text-gray-600">
              Begin a conversation and we&apos;ll help you organize it into a project
            </p>
          </div>

          <form onSubmit={handleChatSubmit} className="max-w-xl mx-auto">
            <div className="flex gap-3">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Start a new chat..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autoFocus
              />
              <Button type="submit" disabled={!chatInput.trim()}>
                Start Chat
              </Button>
            </div>
          </form>
        </Card>

        {/* Example Prompts */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
            Get started with these examples
          </h2>
          <ExamplePrompts
            prompts={examplePrompts}
            onPromptSelect={handlePromptSelect}
            variant="grid"
          />
        </div>

        {/* Project Templates */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
            Choose a project template
          </h2>
          <ProjectTemplates
            templates={templates}
            loading={isLoadingTemplates}
            onTemplateSelect={handleTemplateSelect}
            variant="carousel"
          />
        </div>
      </div>

      {/* Modals */}
      <SelectProjectModal
        isOpen={showSelectProjectModal}
        onClose={() => setShowSelectProjectModal(false)}
        onProjectSelect={(projectId) => {
          // Navigate to new chat in the selected project with the initial message
          const params = new URLSearchParams();
          if (chatInput.trim()) {
            params.set('message', chatInput.trim());
          }
          const queryString = params.toString();
          const url = `/projects/${projectId}/chat/new${queryString ? `?${queryString}` : ''}`;
          window.location.href = url;
        }}
        onCreateNew={handleCreateFirstProject}
        initialMessage={chatInput}
      />

      <ProjectCreationModal
        isOpen={showCreateProjectModal}
        onClose={() => {
          setShowCreateProjectModal(false);
          setSelectedTemplate(null);
        }}
        onSuccess={(projectId) => {
          if (onProjectCreated) {
            onProjectCreated(projectId);
          }
        }}
        template={selectedTemplate || undefined}
      />
    </div>
  );
};
