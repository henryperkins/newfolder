import axios from 'axios';
import {
  Project,
  ProjectTemplate,
  CreateProjectData,
  UpdateProjectData,
  Tag,
  ActivityItem,
  ActivitySummary,
  ProjectsQueryParams,
  ActivitiesQueryParams
} from '../types/project';

export const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for handling authentication errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.log('401 received for:', error.config?.url, 'Current path:', window.location.pathname);
      // Only redirect if we're not already on a public route
      const currentPath = window.location.pathname;
      const publicRoutes = ['/login', '/register', '/reset-password', '/forgot-password'];
      
      if (!publicRoutes.includes(currentPath)) {
        console.log('Redirecting to login due to 401');
        // Clear auth state and redirect to login
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Project API
export const projectApi = {
  // Projects
  async getProjects(params?: ProjectsQueryParams): Promise<{ projects: Project[]; total: number }> {
    const response = await api.get('/projects', { params });
    return response.data;
  },

  async getProject(id: string): Promise<Project> {
    const response = await api.get(`/projects/${id}`);
    return response.data;
  },

  async createProject(data: CreateProjectData): Promise<Project> {
    const response = await api.post('/projects', data);
    return response.data;
  },

  async updateProject(id: string, data: UpdateProjectData): Promise<Project> {
    const response = await api.patch(`/projects/${id}`, data);
    return response.data;
  },

  async deleteProject(id: string): Promise<void> {
    await api.delete(`/projects/${id}?confirm=true`);
  },

  // Templates
  async getTemplates(category?: string): Promise<{ templates: ProjectTemplate[] }> {
    const response = await api.get('/project-templates', { params: { category } });
    return response.data;
  },

  async getTemplate(id: string): Promise<ProjectTemplate> {
    const response = await api.get(`/project-templates/${id}`);
    return response.data;
  },

  // Tags
  async getTags(search?: string): Promise<{ tags: Tag[] }> {
    const response = await api.get('/tags', { params: { search } });
    return response.data;
  },

  async createTag(data: { name: string; color?: string }): Promise<Tag> {
    const response = await api.post('/tags', data);
    return response.data;
  },

  // Activities
  async getActivities(params?: ActivitiesQueryParams): Promise<{
    activities: ActivityItem[];
    total: number;
    has_more: boolean
  }> {
    const response = await api.get('/activities', { params });
    return response.data;
  },

  async getActivitySummary(days = 7): Promise<ActivitySummary> {
    const response = await api.get('/activities/summary', { params: { days } });
    return response.data;
  }
};
// -----------------------------------------------------------------------------
// Chat-specific API helpers (Phase-3)
// -----------------------------------------------------------------------------
import type {
  ChatThread,
  ChatMessage,
  MessagesResponse,
} from '../types/chat';

export const chatApi = {
  // Fetch paginated messages for a thread
  async getThreadMessages(
    threadId: string,
    params?: {
      limit?: number;
      before?: string;
      after?: string;
      include_deleted?: boolean;
    },
  ): Promise<MessagesResponse> {
    const response = await api.get(`/threads/${threadId}/messages`, { params });
    return response.data;
  },

  // Create a new thread (optionally with initial user message)
  async createThread(
    projectId: string,
    initialMessage?: string,
    title = 'New Chat',
  ): Promise<ChatThread> {
    const payload: Record<string, unknown> = {
      project_id: projectId,
      title,
    };
    if (initialMessage) {
      payload.initial_message = initialMessage;
    }
    const response = await api.post('/threads', payload);
    return response.data;
  },

  // Send user message (attachments upload handled elsewhere)
  async sendMessage(
    threadId: string,
    content: string,
    attachments?: File[],
  ): Promise<ChatMessage> {
    // NOTE: The real Phase-3 spec uses WebSocket; this REST call is a fallback
    // so that optimistic UI updates can reconcile with the server in tests.
    const response = await api.post(`/threads/${threadId}/messages`, {
      content,
      attachments,
    });
    return response.data;
  },

  async editMessage(messageId: string, content: string): Promise<ChatMessage> {
    const response = await api.patch(`/threads/messages/${messageId}`, { content });
    return response.data;
  },

  async deleteMessage(messageId: string): Promise<void> {
    await api.delete(`/threads/messages/${messageId}`);
  },

  async regenerateResponse(
    messageId: string,
    options?: Record<string, unknown>,
  ): Promise<void> {
    await api.post('/regenerate', {
      message_id: messageId,
      options,
    });
  },
};

// -----------------------------------------------------------------------------
// Search API (Phase-5)
// -----------------------------------------------------------------------------
export const searchApi = {
  search: async (query: string, projectId?: string): Promise<any> => {
    const params = new URLSearchParams({ q: query });
    if (projectId) params.append('project_id', projectId);
    const response = await api.get(`/search?${params}`);
    return response.data;
  }
};
