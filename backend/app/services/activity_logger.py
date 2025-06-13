from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.activity import ActivityLog, ActivityType


class ActivityLogger:  # noqa: D401 – async helper
    """Persist and fetch user activity events using AsyncSession."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # Write helpers                                                     
    # ------------------------------------------------------------------

    async def log_activity(
        self,
        *,
        user_id: str,
        activity_type: ActivityType,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActivityLog:
        activity = ActivityLog(
            user_id=user_id,
            activity_type=activity_type,
            project_id=project_id,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        self.db.add(activity)
        await self.db.commit()
        await self.db.refresh(activity)
        await self._notify_activity(activity)
        return activity

    # ------------------------------------------------------------------
    # Read helpers                                                      
    # ------------------------------------------------------------------

    async def get_user_activities(
        self,
        *,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        since: Optional[datetime] = None,
    ) -> List[ActivityLog]:
        stmt = select(ActivityLog).where(ActivityLog.user_id == user_id)

        if project_id:
            stmt = stmt.where(ActivityLog.project_id == project_id)

        if since:
            stmt = stmt.where(ActivityLog.created_at >= since)

        stmt = stmt.order_by(desc(ActivityLog.created_at)).limit(limit).offset(offset)
        items = await self.db.scalars(stmt)
        return items.all()

    async def get_activity_summary(self, *, user_id: str, days: int = 7) -> Dict[str, Any]:  # noqa: D401
        since = datetime.utcnow() - timedelta(days=days)
        activities = await self.get_user_activities(user_id=user_id, since=since)

        summary = {
            "total_activities": len(activities),
            "projects_active": len({a.project_id for a in activities if a.project_id}),
            "most_active_project": self._get_most_active_project(activities),
            "activity_by_type": self._group_by_type(activities),
            "daily_breakdown": self._daily_breakdown(activities),
        }
        return summary

    # ------------------------------------------------------------------
    # Internal helpers                                                  
    # ------------------------------------------------------------------

    def _get_most_active_project(self, activities: List[ActivityLog]) -> Optional[str]:
        counter: Dict[str, int] = {}
        for act in activities:
            if act.project_id:
                counter[act.project_id] = counter.get(act.project_id, 0) + 1
        return max(counter, key=counter.get) if counter else None

    def _group_by_type(self, acts: List[ActivityLog]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for a in acts:
            d[a.activity_type.value] = d.get(a.activity_type.value, 0) + 1
        return d

    def _daily_breakdown(self, acts: List[ActivityLog]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for a in acts:
            key = a.created_at.strftime("%Y-%m-%d")
            d[key] = d.get(key, 0) + 1
        return d

    async def _notify_activity(self, activity: ActivityLog):  # noqa: D401 – placeholder
        # Real-time push not implemented yet.
        return None
