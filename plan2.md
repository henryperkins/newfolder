# Phase 2: Projects & The "Empty State" Dashboard - Detailed Implementation Specification

## Phase Summary
Implement the project management lifecycle and the visually guided "first-run" experience. This phase transforms the empty authenticated shell from Phase 1 into an engaging, intuitive workspace that guides new users toward productive action.

## 1. User Interface Components

### 1.1 Dashboard Components

#### **EmptyDashboard Component**
- **Purpose**: First-run experience for new users with no projects
- **Props**:
  ```typescript
  interface EmptyDashboardProps {
    onProjectCreated: (projectId: string) => void;
  }
  ```
- **State Management**:
  ```typescript
  interface EmptyDashboardState {
    isCreatingProject: boolean;
    selectedTemplate: ProjectTemplate | null;
    chatInput: string;
    showSelectProjectModal: boolean;
  }
  ```
- **Zustand Store Interactions**:
  - Reads: `useProjectStore().projects` to determine if empty state should show
  - Writes: None directly (uses callbacks)
- **Key Elements**:
  1. Hero Section: "Organize Your Work with Projects" card
  2. Primary CTA: "Create Your First Project" button (prominent, centered)
  3. Quick Start Chat: Input bar with placeholder "Start a new chat..."
  4. Example Prompts Grid: 4-6 clickable prompt cards
  5. Project Templates Section: Horizontal scrollable template cards
- **Behaviors**:
  - Chat input triggers `SelectProjectModal` if no projects exist
  - Template cards open `ProjectCreationModal` with pre-filled data
  - Example prompts populate chat input when clicked
  - Smooth fade-in animation on mount (300ms)
- **Accessibility**:
  - Focus management: Auto-focus on chat input after page load
  - Screen reader: Announce "Welcome to your workspace" on first visit
  - Keyboard navigation through template cards (arrow keys)
- **Responsive Design**:
  - Mobile (<768px): Stack elements vertically, full-width CTAs
  - Tablet (768-1024px): 2-column template grid
  - Desktop (>1024px): 3-4 column template grid

#### **ExamplePrompts Component**
- **Purpose**: Inspire users with common use cases
- **Props**:
  ```typescript
  interface ExamplePromptsProps {
    onPromptSelect: (prompt: string) => void;
    variant?: 'grid' | 'inline';
  }
  ```
- **Prompt Data Structure**:
  ```typescript
  interface ExamplePrompt {
    id: string;
    icon: LucideIcon;
    title: string;
    prompt: string;
    category: 'productivity' | 'creative' | 'analysis' | 'code';
  }
  ```
- **Default Prompts**:
  1. "Summarize key points from my meeting notes"
  2. "Generate unit tests for this Python function"
  3. "Create a project timeline for Q2 launch"
  4. "Draft an executive summary of market research"
- **Behaviors**:
  - Hover: Slight scale (1.02) and shadow elevation
  - Click: Ripple effect before triggering callback
  - Loading: Disable all cards during chat initialization

#### **ProjectTemplates Component**
- **Purpose**: Display available project templates
- **Props**:
  ```typescript
  interface ProjectTemplatesProps {
    onTemplateSelect: (template: ProjectTemplate) => void;
    loading?: boolean;
    variant?: 'carousel' | 'grid';
  }
  ```
- **State**:
  ```typescript
  interface ProjectTemplatesState {
    templates: ProjectTemplate[];
    isLoading: boolean;
    error: string | null;
  }
  ```
- **Template Card Design**:
  - Icon (48x48px)
  - Title (font-weight: 600)
  - Description (2 lines max, truncated)
  - Tag pills (2-3 suggested tags)
  - "Use Template" button on hover/focus
- **Carousel Behavior** (default):
  - Show 3 cards on desktop, 2 on tablet, 1 on mobile
  - Smooth scroll with momentum
  - Dots indicator below
  - Arrow buttons on desktop only

### 1.2 Project Management Components

#### **ProjectCreationModal Component**
- **Purpose**: Create new projects with or without templates
- **Props**:
  ```typescript
  interface ProjectCreationModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess: (project: Project) => void;
    template?: ProjectTemplate;
    initialName?: string;
  }
  ```
- **Form State**:
  ```typescript
  interface ProjectFormState {
    name: string;
    description: string;
    tags: string[];
    color: string; // hex color for project avatar
    isSubmitting: boolean;
    errors: Record<string, string>;
  }
  ```
- **Modal Layout**:
  1. Header: "Create New Project" with close button
  2. Form Fields:
     - Project Name (required, max 100 chars)
     - Description (optional, max 500 chars)
     - Tags (multi-select with autocomplete)
     - Color Picker (8 preset colors + custom)
  3. Template Preview (if applicable)
  4. Action Buttons: Cancel, Create Project
- **Validation Rules**:
  - Name: Required, 3-100 characters, unique among user's projects
  - Description: Optional, max 500 characters
  - Tags: Max 10 tags, each 2-30 characters
- **Behaviors**:
  - Form autopopulates if template selected
  - Real-time validation on blur
  - Prevent duplicate project names (case-insensitive)
  - Success: Close modal, show toast, navigate to project
- **Accessibility**:
  - Trap focus within modal
  - Escape key closes modal
  - Announce validation errors to screen readers

#### **SelectProjectModal Component**
- **Purpose**: Project selection when starting chat from dashboard
- **Props**:
  ```typescript
  interface SelectProjectModalProps {
    isOpen: boolean;
    onClose: () => void;
    onProjectSelect: (projectId: string) => void;
    onCreateNew: () => void;
    initialMessage?: string;
  }
  ```
- **State**:
  ```typescript
  interface SelectProjectModalState {
    projects: Project[];
    searchQuery: string;
    selectedProjectId: string | null;
    isLoading: boolean;
  }
  ```
- **Layout**:
  1. Header: "Select a Project for Your Chat"
  2. Search bar (filter projects by name)
  3. Project list (scrollable, max-height: 400px)
  4. "Create New Project" option (sticky at bottom)
- **Project Item Display**:
  - Color dot + Project name
  - Last activity timestamp
  - Message count badge
  - Radio button selection indicator
- **Behaviors**:
  - Auto-focus search on open
  - Keyboard navigation (up/down arrows)
  - Enter key confirms selection
  - Click outside dismisses (with confirmation if message entered)

#### **ProjectsView Component**
- **Purpose**: List and manage all user projects
- **Route**: `/projects`
- **State**:
  ```typescript
  interface ProjectsViewState {
    projects: Project[];
    isLoading: boolean;
    error: string | null;
    viewMode: 'grid' | 'list';
    sortBy: 'created' | 'updated' | 'name';
    filterTags: string[];
  }
  ```
- **Zustand Store**:
  ```typescript
  interface ProjectStore {
    projects: Project[];
    isLoading: boolean;
    fetchProjects: () => Promise<void>;
    createProject: (data: CreateProjectDTO) => Promise<Project>;
    updateProject: (id: string, data: UpdateProjectDTO) => Promise<void>;
    deleteProject: (id: string) => Promise<void>;
  }
  ```
- **View Modes**:
  1. **Grid View**: Cards with visual emphasis
     - Project color/icon
     - Name and description
     - Tag pills
     - Last activity
     - Quick actions (edit, archive, delete)
  2. **List View**: Compact table format
     - Columns: Name, Tags, Last Activity, Actions
     - Sortable headers
     - Inline editing for name
- **Empty State**: Show `EmptyDashboard` component
- **Loaded State Features**:
  - View mode toggle (grid/list)
  - Sort dropdown
  - Tag filter multi-select
  - Search bar (client-side filtering)
  - "New Project" button (top-right)

#### **ProjectCard Component**
- **Purpose**: Individual project representation in grid view
- **Props**:
  ```typescript
  interface ProjectCardProps {
    project: Project;
    onEdit: (project: Project) => void;
    onDelete: (projectId: string) => void;
    onNavigate: (projectId: string) => void;
  }
  ```
- **Hover State**:
  - Elevate shadow
  - Show action buttons
  - Darken background slightly
- **Click Behavior**: Navigate to project chat view
- **Action Menu** (three dots):
  - Edit Details
  - Duplicate Project
  - Archive (Phase 3)
  - Delete (with confirmation)

### 1.3 Activity Components

#### **ActivityTimeline Component**
- **Purpose**: Visual timeline of project activities
- **Props**:
  ```typescript
  interface ActivityTimelineProps {
    projectId?: string; // If provided, filter by project
    limit?: number;
    onLoadMore?: () => void;
  }
  ```
- **Activity Item Structure**:
  ```typescript
  interface ActivityItem {
    id: string;
    type: 'project_created' | 'chat_started' | 'document_uploaded' | 'project_updated';
    projectId: string;
    projectName: string;
    timestamp: Date;
    metadata: Record<string, any>;
  }
  ```
- **Visual Design**:
  - Vertical line connecting items
  - Icon for activity type
  - Relative timestamps ("2 hours ago")
  - Expandable details on click
- **Behaviors**:
  - Auto-refresh every 30 seconds
  - Smooth scroll to new items
  - Group by day for older items
  - Load more on scroll (infinite scroll)

## 2. Backend Services

### 2.1 Activity Logging Service
```python
# services/activity_logger.py
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.activity import ActivityLog, ActivityType

class ActivityLogger:
    """Service for logging and retrieving user activity events"""

    def __init__(self, db: Session):
        self.db = db

    async def log_activity(
        self,
        user_id: str,
        activity_type: ActivityType,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ActivityLog:
        """Log a user activity event"""
        activity = ActivityLog(
            user_id=user_id,
            activity_type=activity_type,
            project_id=project_id,
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )

        self.db.add(activity)
        await self.db.commit()
        await self.db.refresh(activity)

        # Trigger real-time notification (Phase 5)
        await self._notify_activity(activity)

        return activity

    async def get_user_activities(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        since: Optional[datetime] = None
    ) -> List[ActivityLog]:
        """Retrieve user activities with filtering"""
        query = self.db.query(ActivityLog).filter(
            ActivityLog.user_id == user_id
        )

        if project_id:
            query = query.filter(ActivityLog.project_id == project_id)

        if since:
            query = query.filter(ActivityLog.created_at >= since)

        activities = query.order_by(
            ActivityLog.created_at.desc()
        ).limit(limit).offset(offset).all()

        return activities

    async def get_activity_summary(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get summarized activity stats for dashboard"""
        since = datetime.utcnow() - timedelta(days=days)

        activities = await self.get_user_activities(
            user_id=user_id,
            since=since
        )

        summary = {
            "total_activities": len(activities),
            "projects_active": len(set(a.project_id for a in activities if a.project_id)),
            "most_active_project": self._get_most_active_project(activities),
            "activity_by_type": self._group_by_type(activities),
            "daily_breakdown": self._daily_breakdown(activities)
        }

        return summary

    def _get_most_active_project(self, activities: List[ActivityLog]) -> Optional[str]:
        """Identify project with most activities"""
        project_counts = {}
        for activity in activities:
            if activity.project_id:
                project_counts[activity.project_id] = project_counts.get(activity.project_id, 0) + 1

        if not project_counts:
            return None

        return max(project_counts, key=project_counts.get)

    def _group_by_type(self, activities: List[ActivityLog]) -> Dict[str, int]:
        """Group activities by type"""
        type_counts = {}
        for activity in activities:
            type_counts[activity.activity_type.value] = type_counts.get(activity.activity_type.value, 0) + 1
        return type_counts

    def _daily_breakdown(self, activities: List[ActivityLog]) -> Dict[str, int]:
        """Break down activities by day"""
        daily_counts = {}
        for activity in activities:
            day_key = activity.created_at.strftime("%Y-%m-%d")
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        return daily_counts

    async def _notify_activity(self, activity: ActivityLog):
        """Placeholder for real-time notifications (Phase 5)"""
        pass
```

### 2.2 Template Service
```python
# services/template_service.py
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ProjectTemplate:
    """Project template data structure"""
    id: str
    name: str
    description: str
    icon: str  # Lucide icon name
    suggested_tags: List[str] = field(default_factory=list)
    starter_prompts: List[str] = field(default_factory=list)
    color: str = "#6366f1"  # Default indigo
    category: str = "general"

class TemplateService:
    """Service for managing project templates"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> List[ProjectTemplate]:
        """Initialize hardcoded templates"""
        return [
            ProjectTemplate(
                id="research-project",
                name="Research Project",
                description="Organize research papers, notes, and findings",
                icon="Microscope",
                suggested_tags=["research", "academic", "literature-review"],
                starter_prompts=[
                    "Summarize the key findings from these papers",
                    "What are the main research gaps identified?",
                    "Create a literature review outline"
                ],
                color="#10b981",  # Green
                category="academic"
            ),
            ProjectTemplate(
                id="product-launch",
                name="Product Launch",
                description="Plan and track your product launch activities",
                icon="Rocket",
                suggested_tags=["product", "launch", "marketing", "timeline"],
                starter_prompts=[
                    "Create a launch timeline for the next 3 months",
                    "Draft a press release for our product",
                    "What are the key metrics to track post-launch?"
                ],
                color="#f59e0b",  # Amber
                category="business"
            ),
            ProjectTemplate(
                id="content-creation",
                name="Content Creation",
                description="Manage blog posts, videos, and creative content",
                icon="PenTool",
                suggested_tags=["content", "writing", "creative", "publishing"],
                starter_prompts=[
                    "Generate blog post ideas about AI productivity",
                    "Create an editorial calendar for next month",
                    "Write an engaging introduction for this topic"
                ],
                color="#8b5cf6",  # Purple
                category="creative"
            ),
            ProjectTemplate(
                id="code-development",
                name="Software Development",
                description="Track features, bugs, and code documentation",
                icon="Code2",
                suggested_tags=["development", "coding", "software", "bugs"],
                starter_prompts=[
                    "Review this code for potential improvements",
                    "Generate unit tests for this function",
                    "Explain this error message and suggest fixes"
                ],
                color="#3b82f6",  # Blue
                category="technical"
            ),
            ProjectTemplate(
                id="personal-goals",
                name="Personal Goals",
                description="Track personal development and life goals",
                icon="Target",
                suggested_tags=["personal", "goals", "habits", "growth"],
                starter_prompts=[
                    "Create a 30-day habit tracking plan",
                    "Break down this goal into actionable steps",
                    "Suggest books for professional development"
                ],
                color="#ec4899",  # Pink
                category="personal"
            ),
            ProjectTemplate(
                id="blank-project",
                name="Blank Project",
                description="Start with a clean slate",
                icon="FileText",
                suggested_tags=["general"],
                starter_prompts=[],
                color="#6b7280",  # Gray
                category="general"
            )
        ]

    async def get_all_templates(self) -> List[ProjectTemplate]:
        """Retrieve all available templates"""
        return self.templates

    async def get_template_by_id(self, template_id: str) -> Optional[ProjectTemplate]:
        """Get specific template by ID"""
        return next((t for t in self.templates if t.id == template_id), None)

    async def get_templates_by_category(self, category: str) -> List[ProjectTemplate]:
        """Filter templates by category"""
        return [t for t in self.templates if t.category == category]

    def get_starter_prompts(self, template_id: str) -> List[str]:
        """Get starter prompts for a specific template"""
        template = next((t for t in self.templates if t.id == template_id), None)
        return template.starter_prompts if template else []
```

## 3. Database Models

### 3.1 Project Model
```python
# models/project.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    color = Column(String(7), nullable=False, default="#6366f1")  # Hex color
    template_id = Column(String(50), nullable=True)  # Reference to template used
    is_archived = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="projects")
    tags = relationship("Tag", secondary="project_tags", back_populates="projects")
    chat_threads = relationship("ChatThread", back_populates="project", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    activities = relationship("ActivityLog", back_populates="project")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_project_name_per_user'),
        CheckConstraint("char_length(name) >= 3", name="project_name_min_length"),
        CheckConstraint("color ~* '^#[0-9A-Fa-f]{6}$'", name="valid_hex_color"),
    )
```

### 3.2 Tag Model
```python
# models/tag.py
class Tag(Base):
    __tablename__ = "tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(30), nullable=False)
    color = Column(String(7), nullable=True)  # Optional hex color
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="tags")
    projects = relationship("Project", secondary="project_tags", back_populates="tags")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_tag_name_per_user'),
        CheckConstraint("char_length(name) >= 2", name="tag_name_min_length"),
    )
```

### 3.3 Project-Tag Junction Table
```python
# models/project_tag.py
class ProjectTag(Base):
    __tablename__ = "project_tags"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Indexes for performance
    __table_args__ = (
        Index('idx_project_tags_project_id', 'project_id'),
        Index('idx_project_tags_tag_id', 'tag_id'),
    )
```

### 3.4 Activity Log Model
```python
# models/activity.py
from enum import Enum

class ActivityType(str, Enum):
    PROJECT_CREATED = "project_created"
    PROJECT_UPDATED = "project_updated"
    PROJECT_ARCHIVED = "project_archived"
    PROJECT_DELETED = "project_deleted"
    CHAT_STARTED = "chat_started"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    activity_type = Column(Enum(ActivityType), nullable=False)
    metadata = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="activities")
    project = relationship("Project", back_populates="activities")

    # Indexes for performance
    __table_args__ = (
        Index('idx_activity_user_created', 'user_id', 'created_at'),
        Index('idx_activity_project_created', 'project_id', 'created_at'),
        Index('idx_activity_type', 'activity_type'),
    )
```

### 3.5 Database Migration
```python
# alembic/versions/002_add_projects.py
"""Add projects and related tables

Revision ID: 002
Revises: 001
Create Date: 2024-01-20 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create projects table
    op.create_table('projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('color', sa.String(7), nullable=False),
        sa.Column('template_id', sa.String(50), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'name', name='unique_project_name_per_user'),
        sa.CheckConstraint("char_length(name) >= 3", name='project_name_min_length'),
        sa.CheckConstraint("color ~* '^#[0-9A-Fa-f]{6}$'", name='valid_hex_color')
    )

    # Create tags table
    op.create_table('tags',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(30), nullable=False),
        sa.Column('color', sa.String(7), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'name', name='unique_tag_name_per_user'),
        sa.CheckConstraint("char_length(name) >= 2", name='tag_name_min_length')
    )

    # Create project_tags junction table
    op.create_table('project_tags',
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tag_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('project_id', 'tag_id')
    )

    # Create activity_logs table
    op.create_table('activity_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('activity_type', sa.Enum('PROJECT_CREATED', 'PROJECT_UPDATED', 'PROJECT_ARCHIVED',
                                         'PROJECT_DELETED', 'CHAT_STARTED', 'DOCUMENT_UPLOADED',
                                         'DOCUMENT_DELETED', name='activitytype'), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=False, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_projects_user_id', 'projects', ['user_id'])
    op.create_index('idx_projects_last_activity', 'projects', ['last_activity_at'])
    op.create_index('idx_project_tags_project_id', 'project_tags', ['project_id'])
    op.create_index('idx_project_tags_tag_id', 'project_tags', ['tag_id'])
    op.create_index('idx_activity_user_created', 'activity_logs', ['user_id', 'created_at'])
    op.create_index('idx_activity_project_created', 'activity_logs', ['project_id', 'created_at'])
    op.create_index('idx_activity_type', 'activity_logs', ['activity_type'])

def downgrade():
    op.drop_index('idx_activity_type', 'activity_logs')
    op.drop_index('idx_activity_project_created', 'activity_logs')
    op.drop_index('idx_activity_user_created', 'activity_logs')
    op.drop_index('idx_project_tags_tag_id', 'project_tags')
    op.drop_index('idx_project_tags_project_id', 'project_tags')
    op.drop_index('idx_projects_last_activity', 'projects')
    op.drop_index('idx_projects_user_id', 'projects')

    op.drop_table('activity_logs')
    op.drop_table('project_tags')
    op.drop_table('tags')
    op.drop_table('projects')

    op.execute('DROP TYPE activitytype')
```

## 4. API Route Specifications

### 4.1 Project Routes

#### **GET /projects**
- **Purpose**: List all projects for authenticated user
- **Authentication**: Required (Bearer token)
- **Query Parameters**:
  ```typescript
  interface ProjectsQueryParams {
    include_archived?: boolean;  // default: false
    sort_by?: 'created' | 'updated' | 'name';  // default: 'updated'
    order?: 'asc' | 'desc';  // default: 'desc'
    tag?: string;  // filter by tag name
    search?: string;  // search in name/description
  }
  ```
- **Success Response** (200):
  ```json
  {
    "projects": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Q1 Product Launch",
        "description": "Planning and tracking Q1 product release",
        "color": "#f59e0b",
        "template_id": "product-launch",
        "is_archived": false,
        "tags": [
          {"id": "tag-id-1", "name": "product", "color": "#10b981"},
          {"id": "tag-id-2", "name": "q1-2024", "color": null}
        ],
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-18T14:22:00Z",
        "last_activity_at": "2024-01-18T14:22:00Z",
        "stats": {
          "chat_count": 15,
          "document_count": 8
        }
      }
    ],
    "total": 12
  }
  ```

#### **POST /projects**
- **Purpose**: Create a new project
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "name": "Research Project 2024",
    "description": "Literature review and analysis",
    "color": "#10b981",
    "template_id": "research-project",
    "tags": ["research", "2024", "ai"]
  }
  ```
- **Validation Rules**:
  - Name: Required, 3-100 chars, unique for user
  - Description: Optional, max 500 chars
  - Color: Valid hex color format
  - Tags: Array of strings, max 10, each 2-30 chars
- **Success Response** (201):
  ```json
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Research Project 2024",
    "description": "Literature review and analysis",
    "color": "#10b981",
    "template_id": "research-project",
    "is_archived": false,
    "tags": [
      {"id": "new-tag-id", "name": "research", "color": null}
    ],
    "created_at": "2024-01-20T10:30:00Z",
    "updated_at": "2024-01-20T10:30:00Z",
    "last_activity_at": "2024-01-20T10:30:00Z"
  }
  ```
- **Error Responses**:
  - 400: `{"detail": "Project name already exists"}`
  - 422: `{"detail": [{"loc": ["body", "name"], "msg": "String too short"}]}`
- **Side Effects**:
  - Creates tags if they don't exist
  - Logs PROJECT_CREATED activity
  - Updates user's last_activity timestamp

#### **GET /projects/{project_id}**
- **Purpose**: Get single project details
- **Authentication**: Required
- **Path Parameters**: `project_id` (UUID)
- **Success Response** (200): Project object with full details
- **Error Response** (404): `{"detail": "Project not found"}`

#### **PATCH /projects/{project_id}**
- **Purpose**: Update project details
- **Authentication**: Required
- **Request Body** (all fields optional):
  ```json
  {
    "name": "Updated Project Name",
    "description": "New description",
    "color": "#3b82f6",
    "tags": ["updated", "tags"],
    "is_archived": false
  }
  ```
- **Success Response** (200): Updated project object
- **Side Effects**: Logs PROJECT_UPDATED activity

#### **DELETE /projects/{project_id}**
- **Purpose**: Delete a project and all associated data
- **Authentication**: Required
- **Query Parameters**:
  - `confirm`: boolean (must be true to proceed)
- **Success Response** (204): No content
- **Error Response** (400): `{"detail": "Confirmation required"}`
- **Side Effects**:
  - Cascades delete to all chat threads, documents
  - Logs PROJECT_DELETED activity

### 4.2 Template Routes

#### **GET /project-templates**
- **Purpose**: Get available project templates
- **Authentication**: Required
- **Query Parameters**:
  - `category`: Filter by category (academic, business, creative, technical, personal, general)
- **Success Response** (200):
  ```json
  {
    "templates": [
      {
        "id": "research-project",
        "name": "Research Project",
        "description": "Organize research papers, notes, and findings",
        "icon": "Microscope",
        "suggested_tags": ["research", "academic", "literature-review"],
        "starter_prompts": [
          "Summarize the key findings from these papers",
          "What are the main research gaps identified?"
        ],
        "color": "#10b981",
        "category": "academic"
      }
    ]
  }
  ```

#### **GET /project-templates/{template_id}**
- **Purpose**: Get specific template details
- **Authentication**: Required
- **Success Response** (200): Single template object
- **Error Response** (404): `{"detail": "Template not found"}`

### 4.3 Tag Routes

#### **GET /tags**
- **Purpose**: List all user's tags
- **Authentication**: Required
- **Query Parameters**:
  - `search`: Filter tags by name
- **Success Response** (200):
  ```json
  {
    "tags": [
      {
        "id": "tag-uuid",
        "name": "research",
        "color": "#10b981",
        "usage_count": 5,
        "created_at": "2024-01-15T10:30:00Z"
      }
    ]
  }
  ```

#### **POST /tags**
- **Purpose**: Create a new tag
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "name": "important",
    "color": "#ef4444"
  }
  ```
- **Success Response** (201): Created tag object
- **Error Response** (400): `{"detail": "Tag already exists"}`

### 4.4 Activity Routes

#### **GET /activities**
- **Purpose**: Get user activity timeline
- **Authentication**: Required
- **Query Parameters**:
  ```typescript
  interface ActivitiesQueryParams {
    project_id?: string;  // Filter by project
    limit?: number;  // default: 50, max: 100
    offset?: number;  // default: 0
    since?: string;  // ISO timestamp
    activity_type?: ActivityType[];  // Filter by types
  }
  ```
- **Success Response** (200):
  ```json
  {
    "activities": [
      {
        "id": "activity-uuid",
        "activity_type": "project_created",
        "project_id": "project-uuid",
        "project_name": "New Research Project",
        "metadata": {
          "template_used": "research-project"
        },
        "created_at": "2024-01-20T10:30:00Z"
      }
    ],
    "total": 150,
    "has_more": true
  }
  ```

#### **GET /activities/summary**
- **Purpose**: Get activity summary statistics
- **Authentication**: Required
- **Query Parameters**:
  - `days`: Number of days to summarize (default: 7, max: 30)
- **Success Response** (200):
  ```json
  {
    "total_activities": 45,
    "projects_active": 3,
    "most_active_project": {
      "id": "project-uuid",
      "name": "Q1 Launch",
      "activity_count": 18
    },
    "activity_by_type": {
      "chat_started": 20,
      "document_uploaded": 15,
      "project_updated": 10
    },
    "daily_breakdown": {
      "2024-01-20": 8,
      "2024-01-19": 12,
      "2024-01-18": 5
    }
  }
  ```

## 5. UX Edge Cases & State Management

### 5.1 Empty States

#### **No Projects State**
- Show `EmptyDashboard` component
- Prominent "Create Your First Project" CTA
- Template carousel visible
- Example prompts to inspire action
- Animated illustration (subtle fade-in)

#### **Empty Project State**
- When navigating to a project with no chats
- Show "Start your first conversation" prompt
- Display template's starter prompts if applicable
- Quick actions: Upload document, Start chat

#### **No Activities State**
- "No recent activity" message
- Suggest actions: Create project, Start chat
- Link to help documentation

### 5.2 Loading States

#### **Project List Loading**
- Show 3-4 skeleton cards in grid view
- Skeleton rows in list view
- Maintain layout structure during load

#### **Template Loading**
- Skeleton cards in carousel
- Disable interaction during load
- Smooth transition when data arrives

#### **Modal Loading**
- Disable form inputs
- Show spinner in submit button
- Prevent modal dismissal during submission

### 5.3 Error States

#### **Project Creation Failure**
- Keep modal open with form data intact
- Show inline error message
- Specific errors for duplicate names
- Generic fallback for unexpected errors

#### **Network Error**
- Toast notification with retry option
- Preserve user input in forms
- Offline indicator in header
- Queue actions for retry when online

### 5.4 Optimistic UI Updates

#### **Project Creation**
- Add project to list immediately
- Show loading indicator on new card
- Rollback on failure with error toast

#### **Tag Addition**
- Show tag immediately in UI
- Sync in background
- Remove if server rejects

#### **Project Deletion**
- Remove from UI immediately
- Show undo toast (5 seconds)
- Actually delete after timeout

### 5.5 Keyboard Navigation

#### **Project Grid/List**
- Tab through project cards
- Enter to open project
- Space to select for bulk actions
- Delete key with confirmation

#### **Modals**
- Trap focus within modal
- Tab cycles through inputs
- Escape closes (with unsaved changes warning)
- Enter submits form (when valid)

#### **Template Carousel**
- Left/Right arrows navigate
- Enter selects template
- Tab moves to next section

### 5.6 Mobile Adaptations

#### **Touch Gestures**
- Swipe to delete projects (with confirmation)
- Pull-to-refresh on project list
- Long press for context menu
- Pinch to zoom on activity timeline

#### **Responsive Modals**
- Full-screen on mobile
- Slide-up animation
- Larger touch targets (min 44px)
- Virtual keyboard aware positioning

#### **Mobile Navigation**
- Bottom sheet for project selection
- Collapsed sidebar by default
- Floating action button for new project

## 6. Technology & Library Choices

### 6.1 Frontend Libraries

#### **Form Management**
- **React Hook Form** + **Zod**
  - Type-safe schema validation
  - Minimal re-renders
  - Built-in touched/dirty tracking
  ```typescript
  const projectSchema = z.object({
    name: z.string().min(3).max(100),
    description: z.string().max(500).optional(),
    color: z.string().regex(/^#[0-9A-Fa-f]{6}$/),
    tags: z.array(z.string()).max(10)
  });
  ```

#### **Modal Management**
- **Radix UI Dialog**
  - Accessible by default
  - Portal rendering
  - Focus management built-in
  - Composable primitives

#### **Drag and Drop** (Phase 2.5 enhancement)
- **react-dropzone**
  - File validation
  - Preview generation
  - Progress callbacks
  - Accessible

#### **Data Fetching**
- **TanStack Query (React Query)**
  - Intelligent caching
  - Optimistic updates
  - Background refetching
  - Pagination helpers
  ```typescript
  const { data, isLoading } = useQuery({
    queryKey: ['projects', filters],
    queryFn: () => projectApi.getProjects(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
  ```

#### **Date Handling**
- **date-fns**
  - Tree-shakeable functions
  - Timezone support
  - Relative time formatting
  ```typescript
  formatRelative(activity.created_at, new Date())
  // "2 hours ago"
  ```

#### **Animation**
- **Framer Motion**
  - Declarative animations
  - Gesture support
  - Layout animations
  - Performance optimized

### 6.2 Backend Libraries

#### **Background Tasks**
- **FastAPI BackgroundTasks**
  - Built-in, no extra deps
  - Good for quick tasks
  - Activity logging
  - Email notifications

#### **Data Validation**
- **Pydantic V2**
  - FastAPI integration
  - Custom validators
  - Serialization control
  ```python
  class ProjectCreate(BaseModel):
      name: str = Field(..., min_length=3, max_length=100)
      description: Optional[str] = Field(None, max_length=500)
      color: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
      tags: List[str] = Field(default_factory=list, max_items=10)

      @validator('name')
      def name_not_empty(cls, v):
          if not v.strip():
              raise ValueError('Name cannot be empty')
          return v.strip()
  ```

#### **Caching** (Phase 2.5)
- **Redis** (via aioredis)
  - Cache templates
  - Activity summaries
  - Tag suggestions

## 7. Testing Strategy

### 7.1 Frontend Unit Tests

#### **Component Tests**
```typescript
// __tests__/components/ProjectCreationModal.test.tsx
describe('ProjectCreationModal', () => {
  const mockOnSuccess = jest.fn();
  const mockOnClose = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('validates project name uniqueness', async () => {
    // Mock API to return existing projects
    mockGetProjects.mockResolvedValue({
      projects: [{ name: 'Existing Project' }]
    });

    render(
      <ProjectCreationModal
        isOpen={true}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />
    );

    await userEvent.type(
      screen.getByLabelText(/project name/i),
      'Existing Project'
    );

    await userEvent.click(screen.getByRole('button', { name: /create/i }));

    expect(
      await screen.findByText(/project name already exists/i)
    ).toBeInTheDocument();
    expect(mockOnSuccess).not.toHaveBeenCalled();
  });

  it('pre-fills form when template provided', () => {
    const template = {
      id: 'research-project',
      name: 'Research Project',
      description: 'Research template description',
      suggested_tags: ['research', 'academic'],
      color: '#10b981'
    };

    render(
      <ProjectCreationModal
        isOpen={true}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
        template={template}
      />
    );

    expect(screen.getByDisplayValue('Research Project')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Research template description')).toBeInTheDocument();
    expect(screen.getByText('research')).toBeInTheDocument();
    expect(screen.getByText('academic')).toBeInTheDocument();
  });
});
```

#### **Store Tests**
```typescript
// __tests__/stores/projectStore.test.ts
describe('ProjectStore', () => {
  beforeEach(() => {
    useProjectStore.setState({ projects: [], isLoading: false });
  });

  it('fetches and stores projects', async () => {
    const mockProjects = [
      { id: '1', name: 'Project 1' },
      { id: '2', name: 'Project 2' }
    ];

    mockApi.getProjects.mockResolvedValue({ projects: mockProjects });

    await act(async () => {
      await useProjectStore.getState().fetchProjects();
    });

    expect(useProjectStore.getState().projects).toEqual(mockProjects);
    expect(useProjectStore.getState().isLoading).toBe(false);
  });

  it('handles project creation optimistically', async () => {
    const newProject = { name: 'New Project', color: '#6366f1' };
    const createdProject = { id: '123', ...newProject };

    mockApi.createProject.mockResolvedValue(createdProject);

    const { createProject } = useProjectStore.getState();
    const result = await createProject(newProject);

    expect(result).toEqual(createdProject);
    expect(useProjectStore.getState().projects).toContainEqual(createdProject);
  });
});
```

### 7.2 Backend Unit Tests

#### **Service Tests**
```python
# tests/services/test_activity_logger.py
@pytest.mark.asyncio
async def test_activity_logger_logs_events(db_session, test_user):
    """Test activity logging service"""
    logger = ActivityLogger(db_session)

    activity = await logger.log_activity(
        user_id=test_user.id,
        activity_type=ActivityType.PROJECT_CREATED,
        project_id="test-project-id",
        metadata={"template_used": "research-project"}
    )

    assert activity.id is not None
    assert activity.user_id == test_user.id
    assert activity.activity_type == ActivityType.PROJECT_CREATED
    assert activity.metadata["template_used"] == "research-project"

@pytest.mark.asyncio
async def test_activity_summary_calculation(db_session, test_user):
    """Test activity summary statistics"""
    logger = ActivityLogger(db_session)

    # Create test activities
    for i in range(5):
        await logger.log_activity(
            user_id=test_user.id,
            activity_type=ActivityType.CHAT_STARTED,
            project_id="project-1"
        )

    for i in range(3):
        await logger.log_activity(
            user_id=test_user.id,
            activity_type=ActivityType.DOCUMENT_UPLOADED,
            project_id="project-2"
        )

    summary = await logger.get_activity_summary(test_user.id, days=7)

    assert summary["total_activities"] == 8
    assert summary["projects_active"] == 2
    assert summary["most_active_project"] == "project-1"
    assert summary["activity_by_type"]["chat_started"] == 5
    assert summary["activity_by_type"]["document_uploaded"] == 3
```

#### **API Route Tests**
```python
# tests/api/test_projects.py
def test_create_project_success(authorized_client, test_user):
    """Test successful project creation"""
    project_data = {
        "name": "Test Project",
        "description": "Test description",
        "color": "#6366f1",
        "tags": ["test", "demo"]
    }

    response = authorized_client.post("/projects", json=project_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Project"
    assert len(data["tags"]) == 2
    assert data["tags"][0]["name"] == "test"

def test_create_duplicate_project_fails(authorized_client, test_project):
    """Test duplicate project name validation"""
    project_data = {
        "name": test_project.name,  # Duplicate name
        "color": "#6366f1"
    }

    response = authorized_client.post("/projects", json=project_data)

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

def test_list_projects_with_filters(authorized_client, test_projects):
    """Test project listing with query filters"""
    # Create projects with different tags
    response = authorized_client.get("/projects?tag=research&sort_by=name&order=asc")

    assert response.status_code == 200
    projects = response.json()["projects"]
    assert all("research" in [t["name"] for t in p["tags"]] for p in projects)
    assert projects == sorted(projects, key=lambda p: p["name"])
```

### 7.3 Integration Tests

#### **Project Creation Flow**
```python
# tests/integration/test_project_flow.py
@pytest.mark.asyncio
async def test_complete_project_creation_flow(async_client, db_session):
    """Test complete project creation with template and activity logging"""
    # Get templates
    templates_response = await async_client.get("/project-templates")
    templates = templates_response.json()["templates"]
    research_template = next(t for t in templates if t["id"] == "research-project")

    # Create project using template
    project_data = {
        "name": "My Research 2024",
        "description": research_template["description"],
        "color": research_template["color"],
        "template_id": research_template["id"],
        "tags": research_template["suggested_tags"]
    }

    create_response = await async_client.post("/projects", json=project_data)
    assert create_response.status_code == 201
    project = create_response.json()

    # Verify activity was logged
    activities_response = await async_client.get(f"/activities?project_id={project['id']}")
    activities = activities_response.json()["activities"]

    assert len(activities) == 1
    assert activities[0]["activity_type"] == "project_created"
    assert activities[0]["metadata"]["template_used"] == "research-project"

    # Verify tags were created
    tags_response = await async_client.get("/tags")
    tags = tags_response.json()["tags"]
    tag_names = [tag["name"] for tag in tags]

    assert "research" in tag_names
    assert "academic" in tag_names
```

### 7.4 E2E Tests

#### **First-Run Experience**
```typescript
// e2e/projects/first-run.spec.ts
test.describe('First-Run Experience', () => {
  test.beforeEach(async ({ page }) => {
    // Login as new user with no projects
    await loginAsNewUser(page);
  });

  test('shows empty dashboard and creates first project', async ({ page }) => {
    // Verify empty state
    await expect(page.locator('h1')).toContainText('Organize Your Work with Projects');
    await expect(page.locator('[data-testid="create-first-project"]')).toBeVisible();

    // Click template card
    await page.click('[data-testid="template-research-project"]');

    // Fill project form
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    await expect(page.locator('input[name="name"]')).toHaveValue('Research Project');

    // Customize name
    await page.fill('input[name="name"]', 'My AI Research 2024');

    // Create project
    await page.click('button:has-text("Create Project")');

    // Verify redirect to project
    await page.waitForURL(/\/projects\/.+/);
    await expect(page.locator('h1')).toContainText('My AI Research 2024');
  });

  test('starts chat from empty dashboard', async ({ page }) => {
    // Type in quick-start chat input
    await page.fill('[data-testid="quick-chat-input"]', 'Help me understand transformers');
    await page.press('[data-testid="quick-chat-input"]', 'Enter');

    // Project selection modal appears
    await expect(page.locator('[role="dialog"]')).toContainText('Select a Project');

    // Create new project
    await page.click('button:has-text("Create New Project")');

    // Fill minimal project form
    await page.fill('input[name="name"]', 'AI Learning');
    await page.click('button:has-text("Create Project")');

    // Verify chat view with message
    await expect(page.locator('[data-testid="chat-input"]')).toHaveValue('Help me understand transformers');
  });
});
```

#### **Project Management**
```typescript
// e2e/projects/management.spec.ts
test.describe('Project Management', () => {
  test('creates, edits, and deletes projects', async ({ page }) => {
    await page.goto('/projects');

    // Create project
    await page.click('button:has-text("New Project")');
    await page.fill('input[name="name"]', 'Test Project');
    await page.fill('textarea[name="description"]', 'E2E test project');
    await page.click('[data-testid="color-picker-blue"]');
    await page.click('button:has-text("Create")');

    // Verify creation
    await expect(page.locator('[data-testid="project-card"]')).toContainText('Test Project');

    // Edit project
    await page.hover('[data-testid="project-card"]');
    await page.click('[data-testid="project-menu"]');
    await page.click('button:has-text("Edit Details")');

    await page.fill('input[name="name"]', 'Updated Test Project');
    await page.click('button:has-text("Save")');

    // Verify update
    await expect(page.locator('[data-testid="project-card"]')).toContainText('Updated Test Project');

    // Delete project
    await page.click('[data-testid="project-menu"]');
    await page.click('button:has-text("Delete")');
    await page.click('button:has-text("Confirm Delete")');

    // Verify deletion
    await expect(page.locator('[data-testid="project-card"]')).not.toBeVisible();
  });
});
```

## 8. Milestones & Acceptance Criteria

### Milestone 1: Data Models & Backend Foundation
**Acceptance Criteria:**
- [x] All project-related database models created with constraints
- [x] Migrations run successfully
- [x] Activity logging service implemented with tests
- [x] Template service returns hardcoded templates
- [x] 90% test coverage on new services

### Milestone 2: Project CRUD API
**Acceptance Criteria:**
- [x] All /projects endpoints functional with validation
- [x] Tag creation and association working
- [x] Activity logging integrated with project operations
- [x] Unique project name constraint enforced per user
- [x] API documentation updated in Swagger

### Milestone 3: Empty Dashboard UI
**Acceptance Criteria:**
- [x] EmptyDashboard renders for users with no projects
- [x] Template carousel displays all templates
- [x] Example prompts clickable and populate input
- [x] Responsive design works on all breakpoints
- [x] Animations smooth and performant

### Milestone 4: Project Creation Flow
**Acceptance Criteria:**
- [x] ProjectCreationModal opens from multiple entry points
- [x] Form validation shows inline errors
- [x] Templates pre-populate form fields
- [x] Successful creation navigates to project
- [x] Duplicate name prevention working
- [x] Tags created automatically if new

### Milestone 5: Project Management UI
**Acceptance Criteria:**
- [x] Projects list view shows all projects
- [x] Grid/list view toggle functional
- [x] Sort and filter options working
- [x] Project cards show accurate information
- [x] Edit/delete operations work with confirmation
- [x] Empty state shows when no projects

### Milestone 6: Activity Timeline
**Acceptance Criteria:**
- [x] Activity timeline displays recent events
- [x] Relative timestamps update correctly
- [x] Filter by project working
- [x] Infinite scroll loads more activities
- [x] Visual design matches mockup

### Milestone 7: Integration & Polish
**Acceptance Criteria:**
- [x] Chat input from dashboard triggers project selection
- [x] All modals have proper focus management
- [x] Loading states show during async operations
- [x] Error states handled gracefully
- [x] All E2E tests passing
- [x] Performance: <100ms response time for project list

## 9. Development Execution Order

### Week 1: Backend Foundation
**Day 1-2: Database & Models**
- Create all SQLAlchemy models
- Write and test migrations
- Set up model relationships
- Create database indexes

**Day 3-4: Core Services**
- Implement ActivityLogger service
- Implement TemplateService
- Write comprehensive unit tests
- Add service dependency injection

**Day 5: Project API Routes**
- Implement CRUD endpoints
- Add validation with Pydantic
- Integrate activity logging
- Test with Swagger UI

### Week 2: Frontend Foundation
**Day 6-7: State Management**
- Set up Zustand project store
- Implement API client methods
- Add React Query for caching
- Create custom hooks

**Day 8-9: Empty Dashboard**
- Build EmptyDashboard component
- Create ExamplePrompts grid
- Implement template carousel
- Add responsive styles

**Day 10: Modals**
- Build ProjectCreationModal
- Implement SelectProjectModal
- Add form validation
- Test keyboard navigation

### Week 3: Project Management
**Day 11-12: Projects View**
- Create ProjectsView page
- Implement grid/list layouts
- Add sort/filter controls
- Build ProjectCard component

**Day 13: Activity Timeline**
- Build ActivityTimeline component
- Implement infinite scroll
- Add relative timestamps
- Create activity type icons

**Day 14-15: Integration**
- Connect all components
- Implement navigation flows
- Add loading/error states
- Polish animations

### Week 4: Testing & Polish
**Day 16-17: Testing**
- Complete unit test coverage
- Write integration tests
- Create E2E test suite
- Fix discovered bugs

**Day 18-19: Performance**
- Optimize bundle size
- Add lazy loading
- Implement caching
- Profile and optimize

**Day 20: Documentation**
- Update API documentation
- Create user guide
- Document deployment
- Prepare for Phase 3

### Parallelization Opportunities
- **Backend team** can work on services/API while **Frontend team** builds UI components
- **Designer** can refine icons/animations while development proceeds
- **QA** can write E2E tests based on specifications before implementation
- **DevOps** can prepare Docker configs and CI/CD pipeline early

This comprehensive specification for Phase 2 provides everything needed to transform the authenticated shell into a fully functional project management system with an engaging first-run experience.
