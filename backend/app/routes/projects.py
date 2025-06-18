"""Project CRUD routes converted to fully async SQLAlchemy 2.0 usage.

The new implementation keeps behaviour identical to the previous sync version
while adopting the select()/await pattern and *AsyncSession* throughout.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy import select, func, asc, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from ..core.database import get_db
from ..dependencies.auth import get_current_user
from ..models import (
    Project,
    Tag,
    ProjectTag,
    ActivityLog,
    ChatThread,
    Document,
    User,
)
from ..models.activity import ActivityType
from ..core.schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    ProjectStats,
    TagResponse,
)
from ..services.activity_logger import ActivityLogger


router = APIRouter(prefix="/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# Helper functions                                                           
# ---------------------------------------------------------------------------


async def _get_tag_usage(db: AsyncSession, tag_id: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(ProjectTag).where(ProjectTag.tag_id == tag_id)
    count: int = await db.scalar(stmt)
    return count or 0


async def _project_stats(db: AsyncSession, project_id: uuid.UUID) -> ProjectStats:
    stmt_chat = (
        select(func.count())
        .select_from(ChatThread)
        .where(ChatThread.project_id == project_id)
    )
    stmt_doc = (
        select(func.count())
        .select_from(Document)
        .where(Document.project_id == project_id)
    )

    chat_count, document_count = await db.scalar(stmt_chat), await db.scalar(stmt_doc)
    return ProjectStats(chat_count=chat_count or 0, document_count=document_count or 0)


async def _project_response(db: AsyncSession, project: Project) -> ProjectResponse:
    tag_responses: List[TagResponse] = []
    for tag in project.tags:
        tag_responses.append(
            TagResponse(
                id=tag.id,
                name=tag.name,
                color=tag.color,
                usage_count=await _get_tag_usage(db, tag.id),
                created_at=tag.created_at,
            )
        )

    stats = await _project_stats(db, project.id)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        color=project.color,
        template_id=project.template_id,
        is_archived=project.is_archived,
        tags=tag_responses,
        created_at=project.created_at,
        updated_at=project.updated_at,
        last_activity_at=project.last_activity_at,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# List                                                                      
# ---------------------------------------------------------------------------


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    include_archived: bool = Query(False),
    sort_by: str = Query("updated", pattern="^(created|updated|name)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all projects owned by *current_user*."""

    stmt = select(Project).options(joinedload(Project.tags)).where(Project.user_id == current_user.id)

    if not include_archived:
        stmt = stmt.where(Project.is_archived.is_(False))

    if tag:
        stmt = (
            stmt.join(ProjectTag).join(Tag).where(Tag.name == tag)
        )

    if search:
        stmt = stmt.where(
            or_(
                Project.name.ilike(f"%{search}%"),
                Project.description.ilike(f"%{search}%"),
            )
        )

    sort_mapping = {"created": Project.created_at, "updated": Project.updated_at, "name": Project.name}
    sort_column = sort_mapping.get(sort_by, Project.updated_at)
    stmt = stmt.order_by(desc(sort_column) if order == "desc" else asc(sort_column))

    projects = (await db.scalars(stmt)).all()

    project_responses = [await _project_response(db, p) for p in projects]

    return ProjectListResponse(projects=project_responses, total=len(project_responses))


# ---------------------------------------------------------------------------
# Create                                                                    
# ---------------------------------------------------------------------------


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a project and optional tags."""

    # Duplicate name check --------------------------------------------------
    stmt_dup = (
        select(Project.id)
        .where(Project.user_id == current_user.id, func.lower(Project.name) == func.lower(project_data.name))
    )
    if await db.scalar(stmt_dup):
        raise HTTPException(status_code=400, detail="Project name already exists")

    project = Project(
        user_id=current_user.id,
        name=project_data.name,
        description=project_data.description,
        color=project_data.color,
        template_id=project_data.template_id,
    )

    db.add(project)
    await db.flush()  # populate PK

    # Tags ------------------------------------------------------------------
    for tag_name in project_data.tags:
        tag_name_clean = tag_name.strip()
        stmt_tag = select(Tag).where(
            Tag.user_id == current_user.id, func.lower(Tag.name) == func.lower(tag_name_clean)
        )
        tag = await db.scalar(stmt_tag)
        if not tag:
            tag = Tag(user_id=current_user.id, name=tag_name_clean)
            db.add(tag)
            await db.flush()

        db.add(ProjectTag(project_id=project.id, tag_id=tag.id))

    await db.commit()
    await db.refresh(project)

    # Activity log -----------------------------------------------------------
    async def _log_activity():
        logger = ActivityLogger(db)
        metadata: Dict[str, str] = {}
        if project_data.template_id:
            metadata["template_used"] = project_data.template_id
        await logger.log_activity(
            user_id=str(current_user.id),
            activity_type=ActivityType.PROJECT_CREATED,
            project_id=str(project.id),
            metadata=metadata,
        )

    background_tasks.add_task(_log_activity)

    # Reload with tags for response ----------------------------------------
    stmt_full = select(Project).options(joinedload(Project.tags)).where(Project.id == project.id)
    project = await db.scalar(stmt_full)

    return await _project_response(db, project)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Retrieve                                                                  
# ---------------------------------------------------------------------------


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Project)
        .options(joinedload(Project.tags))
        .where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = await db.scalar(stmt)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return await _project_response(db, project)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Update                                                                    
# ---------------------------------------------------------------------------


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    project_data: ProjectUpdate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt_proj = select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    project = await db.scalar(stmt_proj)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Name duplicate check ---------------------------------------------------
    if project_data.name and project_data.name != project.name:
        stmt_dup = select(Project.id).where(
            Project.user_id == current_user.id,
            func.lower(Project.name) == func.lower(project_data.name),
            Project.id != project_id,
        )
        if await db.scalar(stmt_dup):
            raise HTTPException(status_code=400, detail="Project name already exists")

    update_fields = project_data.model_dump(exclude_unset=True)
    for field, value in update_fields.items():
        if field != "tags":
            setattr(project, field, value)

    # Tags ------------------------------------------------------------------
    if project_data.tags is not None:
        # Remove existing links
        from sqlalchemy import delete as sqla_delete

        await db.execute(
            sqla_delete(ProjectTag).where(ProjectTag.project_id == project_id)
        )

        for tag_name in project_data.tags:
            tag_name_clean = tag_name.strip()
            stmt_tag = select(Tag).where(
                Tag.user_id == current_user.id,
                func.lower(Tag.name) == func.lower(tag_name_clean),
            )
            tag = await db.scalar(stmt_tag)
            if not tag:
                tag = Tag(user_id=current_user.id, name=tag_name_clean)
                db.add(tag)
                await db.flush()
            db.add(ProjectTag(project_id=project_id, tag_id=tag.id))

    project.updated_at = func.now()
    project.last_activity_at = func.now()

    await db.commit()

    # Activity log -----------------------------------------------------------
    async def _log_activity():
        logger = ActivityLogger(db)
        await logger.log_activity(
            user_id=str(current_user.id),
            activity_type=ActivityType.PROJECT_UPDATED,
            project_id=str(project_id),
            metadata={"updated_fields": list(update_fields.keys())},
        )

    background_tasks.add_task(_log_activity)

    stmt_full = select(Project).options(joinedload(Project.tags)).where(Project.id == project_id)
    project = await db.scalar(stmt_full)

    return await _project_response(db, project)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Delete                                                                    
# ---------------------------------------------------------------------------


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    confirm: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")

    stmt_proj = select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    project = await db.scalar(stmt_proj)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_name = project.name

    await db.delete(project)
    await db.commit()

    async def _log_activity():
        logger = ActivityLogger(db)
        await logger.log_activity(
            user_id=str(current_user.id),
            activity_type=ActivityType.PROJECT_DELETED,
            metadata={"project_name": project_name},
        )

    background_tasks.add_task(_log_activity)
