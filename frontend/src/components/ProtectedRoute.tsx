import React, { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '@/stores';
import { authApi } from '@/utils';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [authChecked, setAuthChecked] = useState(false);
  const { isAuthenticated, setUser, user, logout } = useAuthStore();
  const location = useLocation();

  useEffect(() => {
    const checkAuth = async () => {
      // If already checked and not authenticated, skip
      if (authChecked && !isAuthenticated) {
        setIsLoading(false);
        return;
      }

      // If already authenticated with user data, we're good
      if (isAuthenticated && user) {
        setIsLoading(false);
        setAuthChecked(true);
        return;
      }

      try {
        console.log('Checking authentication...');
        const currentUser = await authApi.getCurrentUser();
        console.log('Authentication successful:', currentUser);
        setUser(currentUser);
        setAuthChecked(true);
        setIsLoading(false);
      } catch {
        console.log('Authentication failed, redirecting to login');
        logout(); // Clear any stale auth state
        setAuthChecked(true);
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [isAuthenticated, user, authChecked, setUser, logout]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary-600 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="mt-2 text-gray-600">Checking authentication...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    console.log('User not authenticated, redirecting to login');
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};
