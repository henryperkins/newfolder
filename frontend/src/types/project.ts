export interface Tag {
  id: string;
  name: string;
  color?: string;
  usage_count?: number;
  created_at: string;
}

export interface ProjectStats {
  chat_count: number;
  document_count: number;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  color: string;
  template_id?: string;
  is_archived: boolean;
  tags: Tag[];
  created_at: string;
  updated_at: string;
  last_activity_at: string;
  stats?: ProjectStats;
}

export interface ProjectTemplate {
  id: string;
  name: string;
  description: string;
  icon: string;
  suggested_tags: string[];
  starter_prompts: string[];
  color: string;
  category: string;
}

export interface CreateProjectData {
  name: string;
  description?: string;
  color: string;
  template_id?: string;
  tags: string[];
}

export interface UpdateProjectData {
  name?: string;
  description?: string;
  color?: string;
  tags?: string[];
  is_archived?: boolean;
}

export interface ActivityItem {
  id: string;
  activity_type: string;
  project_id?: string;
  project_name?: string;
  metadata: Record<string, any>;
  created_at: string;
}

export interface ActivitySummary {
  total_activities: number;
  projects_active: number;
  most_active_project?: {
    id: string;
    name: string;
    activity_count: number;
  };
  activity_by_type: Record<string, number>;
  daily_breakdown: Record<string, number>;
}

export interface ExamplePrompt {
  id: string;
  icon: string;
  title: string;
  prompt: string;
  category: 'productivity' | 'creative' | 'analysis' | 'code';
}

export interface ProjectsQueryParams {
  include_archived?: boolean;
  sort_by?: 'created' | 'updated' | 'name';
  order?: 'asc' | 'desc';
  tag?: string;
  search?: string;
}

export interface ActivitiesQueryParams {
  project_id?: string;
  limit?: number;
  offset?: number;
  since?: string;
  activity_type?: string[];
}