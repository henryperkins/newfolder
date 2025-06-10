import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronDown, Settings, LogOut } from 'lucide-react';
import { useAuthStore } from '@/stores';
import { authApi } from '@/utils';
import { cn } from '@/utils';

interface UserProfileProps {
  isCollapsed: boolean;
}

export const UserProfile: React.FC<UserProfileProps> = ({ isCollapsed }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  if (!user) return null;

  const getInitials = (username: string) => {
    return username
      .split(' ')
      .map((name) => name[0])
      .join('')
      .toUpperCase()
      .substring(0, 2);
  };

  const handleLogout = async () => {
    try {
      await authApi.logout();
      logout();
      navigate('/login');
    } catch (error) {
      console.error('Logout failed:', error);
      logout();
      navigate('/login');
    }
  };

  const handleSettings = () => {
    navigate('/settings');
    setIsDropdownOpen(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        className={cn(
          'w-full flex items-center p-3 hover:bg-gray-100 rounded-lg transition-colors',
          isCollapsed && 'justify-center'
        )}
      >
        <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
          {getInitials(user.username)}
        </div>
        {!isCollapsed && (
          <>
            <div className="ml-3 flex-1 text-left">
              <div className="text-sm font-medium text-gray-900 truncate">
                {user.username}
              </div>
              <div className="text-xs text-gray-500 truncate">
                {user.email}
              </div>
            </div>
            <ChevronDown
              className={cn(
                'w-4 h-4 text-gray-400 transition-transform',
                isDropdownOpen && 'transform rotate-180'
              )}
            />
          </>
        )}
      </button>

      {isDropdownOpen && (
        <div className="absolute bottom-full left-0 right-0 mb-2 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
          <div className="py-1">
            <button
              onClick={handleSettings}
              className="w-full flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            >
              <Settings className="w-4 h-4 mr-3" />
              Account Settings
            </button>
            <button
              onClick={handleLogout}
              className="w-full flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            >
              <LogOut className="w-4 h-4 mr-3" />
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
};