"""Tag management routes – fully async SQLAlchemy 2.0 style."""

from __future__ import annotations

import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, asc
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..dependencies.auth import get_current_user
from ..models import Tag, ProjectTag, User
from ..core.schemas import TagCreate, TagResponse, TagListResponse


router = APIRouter(prefix="/tags", tags=["tags"])


# ---------------------------------------------------------------------------
# Helpers                                                                    
# ---------------------------------------------------------------------------


async def _usage_count(db: AsyncSession, tag_id: uuid.UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(ProjectTag)
        .where(ProjectTag.tag_id == tag_id)
    )
    count: int = await db.scalar(stmt)
    return count or 0


# ---------------------------------------------------------------------------
# Routes                                                                     
# ---------------------------------------------------------------------------


@router.get("", response_model=TagListResponse)
async def list_tags(
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's tags."""

    stmt = select(Tag).where(Tag.user_id == current_user.id)

    if search:
        stmt = stmt.where(Tag.name.ilike(f"%{search}%"))

    stmt = stmt.order_by(asc(Tag.name))

    result = await db.scalars(stmt)
    tags: List[Tag] = result.all()

    tag_responses = [
        TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            usage_count=await _usage_count(db, tag.id),
            created_at=tag.created_at,
        )
        for tag in tags
    ]

    return TagListResponse(tags=tag_responses)


@router.post("", response_model=TagResponse, status_code=201)
async def create_tag(
    tag_data: TagCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new tag for the current user."""

    # Duplicate check -------------------------------------------------------
    stmt_check = (
        select(Tag)
        .where(
            Tag.user_id == current_user.id,
            func.lower(Tag.name) == func.lower(tag_data.name),
        )
    )
    existing = await db.scalar(stmt_check)
    if existing:
        raise HTTPException(status_code=400, detail="Tag already exists")

    tag = Tag(
        user_id=current_user.id,
        name=tag_data.name.strip(),
        color=tag_data.color,
    )

    db.add(tag)
    await db.commit()
    await db.refresh(tag)

    return TagResponse(
        id=tag.id,
        name=tag.name,
        color=tag.color,
        usage_count=0,
        created_at=tag.created_at,
    )
