from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload
from typing import Optional, List
import uuid
from datetime import datetime

from ..core.database import get_db
from ..dependencies.auth import get_current_user
from ..models import User, ActivityLog, Project
from ..models.activity import ActivityType
from ..core.schemas import ActivityResponse, ActivityListResponse, ActivitySummaryResponse
from ..services.activity_logger import ActivityLogger


router = APIRouter(prefix="/activities", tags=["activities"])


@router.get("", response_model=ActivityListResponse)
async def list_activities(
    project_id: Optional[uuid.UUID] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    since: Optional[datetime] = Query(None),
    activity_type: Optional[List[ActivityType]] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user activity timeline"""
    activity_logger = ActivityLogger(db)
    
    activities = await activity_logger.get_user_activities(
        user_id=str(current_user.id),
        project_id=str(project_id) if project_id else None,
        limit=limit + 1,  # Get one extra to check for more
        offset=offset,
        since=since
    )

    has_more = len(activities) > limit
    if has_more:
        activities = activities[:limit]

    # Get project names for activities
    activity_responses = []
    for activity in activities:
        project_name = None
        if activity.project_id:
            project = db.query(Project).filter(Project.id == activity.project_id).first()
            if project:
                project_name = project.name

        activity_responses.append(ActivityResponse(
            id=activity.id,
            activity_type=activity.activity_type.value,
            project_id=activity.project_id,
            project_name=project_name,
            metadata=activity.metadata,
            created_at=activity.created_at
        ))

    # Count total activities for this user
    total_query = db.query(ActivityLog).filter(ActivityLog.user_id == current_user.id)
    if project_id:
        total_query = total_query.filter(ActivityLog.project_id == project_id)
    if since:
        total_query = total_query.filter(ActivityLog.created_at >= since)
    
    total = total_query.count()

    return ActivityListResponse(
        activities=activity_responses,
        total=total,
        has_more=has_more
    )


@router.get("/summary", response_model=ActivitySummaryResponse)
async def get_activity_summary(
    days: int = Query(7, le=30),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get activity summary statistics"""
    activity_logger = ActivityLogger(db)
    
    summary = await activity_logger.get_activity_summary(
        user_id=str(current_user.id),
        days=days
    )

    # Get most active project details if it exists
    most_active_project = None
    if summary["most_active_project"]:
        project = db.query(Project).filter(
            Project.id == summary["most_active_project"]
        ).first()
        if project:
            # Count activities for this project
            activity_count = db.query(ActivityLog).filter(
                ActivityLog.project_id == project.id,
                ActivityLog.user_id == current_user.id
            ).count()
            
            most_active_project = {
                "id": str(project.id),
                "name": project.name,
                "activity_count": activity_count
            }

    return ActivitySummaryResponse(
        total_activities=summary["total_activities"],
        projects_active=summary["projects_active"],
        most_active_project=most_active_project,
        activity_by_type=summary["activity_by_type"],
        daily_breakdown=summary["daily_breakdown"]
    )