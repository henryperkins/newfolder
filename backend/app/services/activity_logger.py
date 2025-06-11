from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..models.activity import ActivityLog, ActivityType


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
        self.db.commit()
        self.db.refresh(activity)

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