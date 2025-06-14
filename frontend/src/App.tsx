import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout, LoginPage, RegistrationPage, ResetPasswordPage, ProtectedRoute } from '@/components';
import {
  DashboardPage,
  SettingsPage,
  ProjectsPage,
  ProjectPage,
  ChatsPage,
  ComponentsShowcasePage,
} from '@/pages';
import { ToastProvider } from '@/components/ui';

function App() {
  return (
    <ToastProvider>
      <Router>
        <Routes>
        {/* Public routes */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegistrationPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <AppLayout>
                <DashboardPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <AppLayout>
                <SettingsPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/projects"
          element={
            <ProtectedRoute>
              <AppLayout>
                <ProjectsPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/projects/:projectId"
          element={
            <ProtectedRoute>
              <AppLayout>
                <ProjectPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/projects/:projectId/chat/:threadId"
          element={
            <ProtectedRoute>
              <AppLayout>
                <ProjectPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/projects/:projectId/chat/new"
          element={
            <ProtectedRoute>
              <AppLayout>
                <ProjectPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/chats"
          element={
            <ProtectedRoute>
              <AppLayout>
                <ChatsPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/components"
          element={
            <ProtectedRoute>
              <AppLayout>
                <ComponentsShowcasePage />
              </AppLayout>
            </ProtectedRoute>
          }
        />

        {/* Default redirects */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Router>
    </ToastProvider>
  );
}

export default App;