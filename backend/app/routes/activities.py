"""User activity timeline routes – fully async."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List
import uuid

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..dependencies.auth import get_current_user
from ..models import User, ActivityLog, Project
from ..models.activity import ActivityType
from ..core.schemas import (
    ActivityResponse,
    ActivityListResponse,
    ActivitySummaryResponse,
)
from ..services.activity_logger import ActivityLogger


router = APIRouter(prefix="/activities", tags=["activities"])


# ---------------------------------------------------------------------------
# Routes                                                                     
# ---------------------------------------------------------------------------


@router.get("", response_model=ActivityListResponse)
async def list_activities(
    project_id: Optional[uuid.UUID] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    since: Optional[datetime] = Query(None),
    activity_type: Optional[List[ActivityType]] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's activity timeline."""

    logger = ActivityLogger(db)

    activities = await logger.get_user_activities(
        user_id=str(current_user.id),
        project_id=str(project_id) if project_id else None,
        limit=limit + 1,  # Sentinel for has_more
        offset=offset,
        since=since,
    )

    has_more = len(activities) > limit
    activities = activities[:limit]

    # Fetch referenced project names in a single query ----------------------
    project_ids = {a.project_id for a in activities if a.project_id}
    project_name_map = {}
    if project_ids:
        stmt_projects = select(Project.id, Project.name).where(Project.id.in_(project_ids))
        result_projects = await db.execute(stmt_projects)
        project_name_map = {str(pid): name for pid, name in result_projects}

    activity_responses = [
        ActivityResponse(
            id=a.id,
            activity_type=a.activity_type.value,
            project_id=a.project_id,
            project_name=project_name_map.get(str(a.project_id)) if a.project_id else None,
            metadata=a.metadata,
            created_at=a.created_at,
        )
        for a in activities
    ]

    # Total activities -------------------------------------------------------
    stmt_total = select(func.count()).select_from(ActivityLog).where(ActivityLog.user_id == current_user.id)
    if project_id:
        stmt_total = stmt_total.where(ActivityLog.project_id == project_id)
    if since:
        stmt_total = stmt_total.where(ActivityLog.created_at >= since)

    total: int = await db.scalar(stmt_total)

    return ActivityListResponse(activities=activity_responses, total=total or 0, has_more=has_more)


@router.get("/summary", response_model=ActivitySummaryResponse)
async def get_activity_summary(
    days: int = Query(7, le=30),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregate activity stats for the past *days* days."""

    logger = ActivityLogger(db)
    summary = await logger.get_activity_summary(user_id=str(current_user.id), days=days)

    most_active_project = None
    if summary["most_active_project"]:
        stmt_project = select(Project).where(Project.id == summary["most_active_project"])
        project = await db.scalar(stmt_project)
        if project:
            stmt_count = (
                select(func.count())
                .select_from(ActivityLog)
                .where(ActivityLog.project_id == project.id, ActivityLog.user_id == current_user.id)
            )
            count: int = await db.scalar(stmt_count)
            most_active_project = {
                "id": str(project.id),
                "name": project.name,
                "activity_count": count or 0,
            }

    return ActivitySummaryResponse(
        total_activities=summary["total_activities"],
        projects_active=summary["projects_active"],
        most_active_project=most_active_project,
        activity_by_type=summary["activity_by_type"],
        daily_breakdown=summary["daily_breakdown"],
    )
