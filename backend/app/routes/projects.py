from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, asc, func, or_
from typing import List, Optional
import uuid

from ..core.database import get_db
from ..dependencies.auth import get_current_user
from ..models import (
    User,
    Project,
    Tag,
    ProjectTag,
    ActivityLog,
    ChatThread,
    Document,
)
from ..models.activity import ActivityType
from ..core.schemas import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListResponse,
    ProjectStats, TagResponse
)
from ..services.activity_logger import ActivityLogger


router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    include_archived: bool = Query(False),
    sort_by: str = Query("updated", regex="^(created|updated|name)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all projects for authenticated user"""
    query = db.query(Project).options(
        joinedload(Project.tags)
    ).filter(Project.user_id == current_user.id)

    if not include_archived:
        query = query.filter(Project.is_archived == False)

    if tag:
        query = query.join(ProjectTag).join(Tag).filter(Tag.name == tag)

    if search:
        query = query.filter(
            or_(
                Project.name.ilike(f"%{search}%"),
                Project.description.ilike(f"%{search}%")
            )
        )

    # Apply sorting
    sort_column = getattr(Project, sort_by)
    if order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(asc(sort_column))

    projects = query.all()

    # Add stats to each project
    project_responses = []
    for project in projects:
        # Calculate basic stats
        chat_count = db.query(ChatThread).filter(ChatThread.project_id == project.id).count()
        document_count = db.query(Document).filter(Document.project_id == project.id).count()

        stats = ProjectStats(chat_count=chat_count, document_count=document_count)
        
        project_response = ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            color=project.color,
            template_id=project.template_id,
            is_archived=project.is_archived,
            tags=[TagResponse(
                id=tag.id,
                name=tag.name,
                color=tag.color,
                usage_count=0,  # TODO: Calculate usage count
                created_at=tag.created_at
            ) for tag in project.tags],
            created_at=project.created_at,
            updated_at=project.updated_at,
            last_activity_at=project.last_activity_at,
            stats=stats
        )
        project_responses.append(project_response)

    return ProjectListResponse(projects=project_responses, total=len(project_responses))


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new project"""
    # Check for duplicate name
    existing = db.query(Project).filter(
        Project.user_id == current_user.id,
        func.lower(Project.name) == func.lower(project_data.name)
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Project name already exists")

    # Create project
    project = Project(
        user_id=current_user.id,
        name=project_data.name,
        description=project_data.description,
        color=project_data.color,
        template_id=project_data.template_id
    )
    
    db.add(project)
    db.flush()  # Get the project ID

    # Handle tags
    tag_objects = []
    for tag_name in project_data.tags:
        # Get or create tag
        tag = db.query(Tag).filter(
            Tag.user_id == current_user.id,
            func.lower(Tag.name) == func.lower(tag_name.strip())
        ).first()
        
        if not tag:
            tag = Tag(
                user_id=current_user.id,
                name=tag_name.strip()
            )
            db.add(tag)
            db.flush()
        
        tag_objects.append(tag)
        
        # Create project-tag association
        project_tag = ProjectTag(
            project_id=project.id,
            tag_id=tag.id
        )
        db.add(project_tag)

    db.commit()
    db.refresh(project)

    # Log activity in background
    def log_activity():
        activity_logger = ActivityLogger(db)
        metadata = {}
        if project_data.template_id:
            metadata["template_used"] = project_data.template_id
        
        # Note: This is not async in background task
        activity = ActivityLog(
            user_id=current_user.id,
            activity_type=ActivityType.PROJECT_CREATED,
            project_id=project.id,
            metadata=metadata
        )
        db.add(activity)
        db.commit()

    background_tasks.add_task(log_activity)

    # Reload with tags
    project = db.query(Project).options(joinedload(Project.tags)).filter(
        Project.id == project.id
    ).first()

    # Compute stats
    chat_count = db.query(ChatThread).filter(ChatThread.project_id == project.id).count()
    document_count = db.query(Document).filter(Document.project_id == project.id).count()

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        color=project.color,
        template_id=project.template_id,
        is_archived=project.is_archived,
        tags=[TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            usage_count=0,
            created_at=tag.created_at
        ) for tag in project.tags],
        created_at=project.created_at,
        updated_at=project.updated_at,
        last_activity_at=project.last_activity_at,
        stats=ProjectStats(chat_count=chat_count, document_count=document_count)
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get single project details"""
    project = db.query(Project).options(
        joinedload(Project.tags)
    ).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        color=project.color,
        template_id=project.template_id,
        is_archived=project.is_archived,
        tags=[TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            usage_count=0,
            created_at=tag.created_at
        ) for tag in project.tags],
        created_at=project.created_at,
        updated_at=project.updated_at,
        last_activity_at=project.last_activity_at,
        stats=ProjectStats(chat_count=0, document_count=0)
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    project_data: ProjectUpdate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update project details"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check for duplicate name if name is being updated
    if project_data.name and project_data.name != project.name:
        existing = db.query(Project).filter(
            Project.user_id == current_user.id,
            func.lower(Project.name) == func.lower(project_data.name),
            Project.id != project_id
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Project name already exists")

    # Update fields
    update_data = project_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field != "tags":
            setattr(project, field, value)

    # Handle tags if provided
    if project_data.tags is not None:
        # Remove existing associations
        db.query(ProjectTag).filter(ProjectTag.project_id == project_id).delete()
        
        # Add new tags
        for tag_name in project_data.tags:
            tag = db.query(Tag).filter(
                Tag.user_id == current_user.id,
                func.lower(Tag.name) == func.lower(tag_name.strip())
            ).first()
            
            if not tag:
                tag = Tag(
                    user_id=current_user.id,
                    name=tag_name.strip()
                )
                db.add(tag)
                db.flush()
            
            project_tag = ProjectTag(
                project_id=project_id,
                tag_id=tag.id
            )
            db.add(project_tag)

    project.updated_at = func.now()
    project.last_activity_at = func.now()
    db.commit()

    # Log activity in background
    def log_activity():
        activity = ActivityLog(
            user_id=current_user.id,
            activity_type=ActivityType.PROJECT_UPDATED,
            project_id=project_id,
            metadata={"updated_fields": list(update_data.keys())}
        )
        db.add(activity)
        db.commit()

    background_tasks.add_task(log_activity)

    # Reload with tags
    project = db.query(Project).options(joinedload(Project.tags)).filter(
        Project.id == project_id
    ).first()

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        color=project.color,
        template_id=project.template_id,
        is_archived=project.is_archived,
        tags=[TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            usage_count=0,
            created_at=tag.created_at
        ) for tag in project.tags],
        created_at=project.created_at,
        updated_at=project.updated_at,
        last_activity_at=project.last_activity_at,
        stats=ProjectStats(chat_count=0, document_count=0)
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    confirm: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a project and all associated data"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")

    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_name = project.name

    # Delete project (cascades to associated data)
    db.delete(project)
    db.commit()

    # Log activity in background
    def log_activity():
        activity = ActivityLog(
            user_id=current_user.id,
            activity_type=ActivityType.PROJECT_DELETED,
            project_id=None,  # Project no longer exists
            metadata={"project_name": project_name}
        )
        db.add(activity)
        db.commit()

    background_tasks.add_task(log_activity)