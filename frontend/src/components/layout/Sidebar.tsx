import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Plus, FolderOpen, Settings, MessageSquare, Menu } from 'lucide-react';
import { useUiStore } from '@/stores';
import { cn } from '@/utils';
import { UserProfile } from './UserProfile';

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

const navItems = [
  { icon: FolderOpen, label: 'Projects', path: '/projects' },
  { icon: MessageSquare, label: 'Recent Chats', path: '/chats' },
  { icon: Settings, label: 'Settings', path: '/settings' },
];

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
}) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNewChat = () => {
    navigate('/dashboard');
  };

  return (
    <div
      className={cn(
        'sidebar-transition h-full bg-white border-r border-gray-200 flex flex-col',
        isCollapsed ? 'w-16' : 'w-72'
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <h1 className="text-lg font-semibold text-gray-900">
              AI Productivity
            </h1>
          )}
          <button
            onClick={onToggleCollapse}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={handleNewChat}
          className={cn(
            'btn-primary w-full',
            isCollapsed ? 'p-3' : 'px-4 py-3'
          )}
        >
          <Plus className="w-5 h-5" />
          {!isCollapsed && <span className="ml-2">New Chat</span>}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 pb-4">
        <div className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={cn(
                  'w-full flex items-center rounded-lg px-3 py-2 transition-colors',
                  isActive
                    ? 'bg-primary-50 text-primary-700 border-l-4 border-primary-600'
                    : 'text-gray-700 hover:bg-gray-100',
                  isCollapsed && 'justify-center px-2'
                )}
                title={isCollapsed ? item.label : undefined}
              >
                <Icon className="w-5 h-5" />
                {!isCollapsed && <span className="ml-3">{item.label}</span>}
              </button>
            );
          })}
        </div>
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-gray-200">
        <UserProfile isCollapsed={isCollapsed} />
      </div>
    </div>
  );
};