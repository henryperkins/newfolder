/**
 * Minimal *useAuth* hook that exposes the current auth store in a
 * React-friendly fashion.  Phase-3 components only require read-access to the
 * user object and a helper to trigger logout, so we forward those straight
 * from the Zustand store.
 */

import { useAuthStore } from '@/stores/authStore';

export function useAuth() {
  const { user, isAuthenticated, logout } = useAuthStore();
  return { user, isAuthenticated, logout } as const;
}

export default useAuth;
