import { create } from 'zustand';
import { persist } from 'zustand/middleware';
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
import { projectApi } from '../utils/api';

interface ProjectState {
  // Projects
  projects: Project[];
  isLoadingProjects: boolean;
  projectsError: string | null;
  
  // Templates
  templates: ProjectTemplate[];
  isLoadingTemplates: boolean;
  templatesError: string | null;
  
  // Tags
  tags: Tag[];
  isLoadingTags: boolean;
  tagsError: string | null;
  
  // Activities
  activities: ActivityItem[];
  activitySummary: ActivitySummary | null;
  isLoadingActivities: boolean;
  activitiesError: string | null;
  hasMoreActivities: boolean;
  
  // UI State
  selectedProject: Project | null;
  projectListView: 'grid' | 'list';
  projectFilters: ProjectsQueryParams;
}

interface ProjectActions {
  // Project actions
  fetchProjects: (params?: ProjectsQueryParams) => Promise<void>;
  createProject: (data: CreateProjectData) => Promise<Project>;
  updateProject: (id: string, data: UpdateProjectData) => Promise<void>;
  deleteProject: (id: string) => Promise<void>;
  setSelectedProject: (project: Project | null) => void;
  
  // Template actions
  fetchTemplates: (category?: string) => Promise<void>;
  getTemplateById: (id: string) => ProjectTemplate | undefined;
  
  // Tag actions
  fetchTags: (search?: string) => Promise<void>;
  createTag: (name: string, color?: string) => Promise<Tag>;
  
  // Activity actions
  fetchActivities: (params?: ActivitiesQueryParams) => Promise<void>;
  loadMoreActivities: () => Promise<void>;
  fetchActivitySummary: (days?: number) => Promise<void>;
  
  // UI actions
  setProjectListView: (view: 'grid' | 'list') => void;
  setProjectFilters: (filters: Partial<ProjectsQueryParams>) => void;
  
  // Utility actions
  clearErrors: () => void;
  reset: () => void;
}

type ProjectStore = ProjectState & ProjectActions;

const initialState: ProjectState = {
  projects: [],
  isLoadingProjects: false,
  projectsError: null,
  
  templates: [],
  isLoadingTemplates: false,
  templatesError: null,
  
  tags: [],
  isLoadingTags: false,
  tagsError: null,
  
  activities: [],
  activitySummary: null,
  isLoadingActivities: false,
  activitiesError: null,
  hasMoreActivities: false,
  
  selectedProject: null,
  projectListView: 'grid',
  projectFilters: {
    include_archived: false,
    sort_by: 'updated',
    order: 'desc'
  }
};

export const useProjectStore = create<ProjectStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Project actions
      fetchProjects: async (params?: ProjectsQueryParams) => {
        set({ isLoadingProjects: true, projectsError: null });
        try {
          const finalParams = { ...get().projectFilters, ...params };
          const response = await projectApi.getProjects(finalParams);
          set({ 
            projects: response.projects, 
            isLoadingProjects: false
          });
        } catch (error) {
          console.error('fetchProjects error:', error);
          set({ 
            projectsError: error instanceof Error ? error.message : 'Failed to fetch projects',
            isLoadingProjects: false 
          });
        }
      },

      createProject: async (data: CreateProjectData) => {
        set({ isLoadingProjects: true, projectsError: null });
        try {
          const project = await projectApi.createProject(data);
          set(state => ({ 
            projects: [project, ...state.projects],
            isLoadingProjects: false 
          }));
          return project;
        } catch (error) {
          set({ 
            projectsError: error instanceof Error ? error.message : 'Failed to create project',
            isLoadingProjects: false 
          });
          throw error;
        }
      },

      updateProject: async (id: string, data: UpdateProjectData) => {
        try {
          const updatedProject = await projectApi.updateProject(id, data);
          set(state => ({
            projects: state.projects.map(p => 
              p.id === id ? updatedProject : p
            ),
            selectedProject: state.selectedProject?.id === id ? updatedProject : state.selectedProject
          }));
        } catch (error) {
          set({ projectsError: error instanceof Error ? error.message : 'Failed to update project' });
          throw error;
        }
      },

      deleteProject: async (id: string) => {
        try {
          await projectApi.deleteProject(id);
          set(state => ({
            projects: state.projects.filter(p => p.id !== id),
            selectedProject: state.selectedProject?.id === id ? null : state.selectedProject
          }));
        } catch (error) {
          set({ projectsError: error instanceof Error ? error.message : 'Failed to delete project' });
          throw error;
        }
      },

      setSelectedProject: (project: Project | null) => {
        set({ selectedProject: project });
      },

      // Template actions
      fetchTemplates: async (category?: string) => {
        set({ isLoadingTemplates: true, templatesError: null });
        try {
          const response = await projectApi.getTemplates(category);
          set({ templates: response.templates, isLoadingTemplates: false });
        } catch (error) {
          set({ 
            templatesError: error instanceof Error ? error.message : 'Failed to fetch templates',
            isLoadingTemplates: false 
          });
        }
      },

      getTemplateById: (id: string) => {
        return get().templates.find(t => t.id === id);
      },

      // Tag actions
      fetchTags: async (search?: string) => {
        set({ isLoadingTags: true, tagsError: null });
        try {
          const response = await projectApi.getTags(search);
          set({ tags: response.tags, isLoadingTags: false });
        } catch (error) {
          set({ 
            tagsError: error instanceof Error ? error.message : 'Failed to fetch tags',
            isLoadingTags: false 
          });
        }
      },

      createTag: async (name: string, color?: string) => {
        try {
          const tag = await projectApi.createTag({ name, color });
          set(state => ({ tags: [...state.tags, tag] }));
          return tag;
        } catch (error) {
          set({ tagsError: error instanceof Error ? error.message : 'Failed to create tag' });
          throw error;
        }
      },

      // Activity actions
      fetchActivities: async (params?: ActivitiesQueryParams) => {
        set({ isLoadingActivities: true, activitiesError: null });
        try {
          const response = await projectApi.getActivities(params);
          set({ 
            activities: response.activities,
            hasMoreActivities: response.has_more,
            isLoadingActivities: false 
          });
        } catch (error) {
          set({ 
            activitiesError: error instanceof Error ? error.message : 'Failed to fetch activities',
            isLoadingActivities: false 
          });
        }
      },

      loadMoreActivities: async () => {
        const { activities, isLoadingActivities, hasMoreActivities } = get();
        if (isLoadingActivities || !hasMoreActivities) return;

        set({ isLoadingActivities: true });
        try {
          const response = await projectApi.getActivities({ 
            offset: activities.length 
          });
          set({ 
            activities: [...activities, ...response.activities],
            hasMoreActivities: response.has_more,
            isLoadingActivities: false 
          });
        } catch (error) {
          set({ 
            activitiesError: error instanceof Error ? error.message : 'Failed to load more activities',
            isLoadingActivities: false 
          });
        }
      },

      fetchActivitySummary: async (days = 7) => {
        try {
          const summary = await projectApi.getActivitySummary(days);
          set({ activitySummary: summary });
        } catch (error) {
          set({ activitiesError: error instanceof Error ? error.message : 'Failed to fetch activity summary' });
        }
      },

      // UI actions
      setProjectListView: (view: 'grid' | 'list') => {
        set({ projectListView: view });
      },

      setProjectFilters: (filters: Partial<ProjectsQueryParams>) => {
        set(state => ({ 
          projectFilters: { ...state.projectFilters, ...filters }
        }));
      },

      // Utility actions
      clearErrors: () => {
        set({ 
          projectsError: null,
          templatesError: null,
          tagsError: null,
          activitiesError: null 
        });
      },

      reset: () => {
        set(initialState);
      }
    }),
    {
      name: 'project-store',
      partialize: (state) => ({
        projectListView: state.projectListView,
        projectFilters: state.projectFilters
      })
    }
  )
);