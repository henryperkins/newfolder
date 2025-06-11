from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from ..core.database import get_db
from ..dependencies.auth import get_current_user
from ..models import User, Tag, ProjectTag
from ..core.schemas import TagCreate, TagResponse, TagListResponse


router = APIRouter(prefix="/tags", tags=["tags"])


@router.get("", response_model=TagListResponse)
async def list_tags(
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all user's tags"""
    query = db.query(Tag).filter(Tag.user_id == current_user.id)

    if search:
        query = query.filter(Tag.name.ilike(f"%{search}%"))

    tags = query.order_by(Tag.name).all()

    # Calculate usage count for each tag
    tag_responses = []
    for tag in tags:
        usage_count = db.query(ProjectTag).filter(ProjectTag.tag_id == tag.id).count()
        tag_responses.append(TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            usage_count=usage_count,
            created_at=tag.created_at
        ))

    return TagListResponse(tags=tag_responses)


@router.post("", response_model=TagResponse, status_code=201)
async def create_tag(
    tag_data: TagCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new tag"""
    # Check for duplicate name
    existing = db.query(Tag).filter(
        Tag.user_id == current_user.id,
        func.lower(Tag.name) == func.lower(tag_data.name)
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Tag already exists")

    tag = Tag(
        user_id=current_user.id,
        name=tag_data.name.strip(),
        color=tag_data.color
    )
    
    db.add(tag)
    db.commit()
    db.refresh(tag)

    return TagResponse(
        id=tag.id,
        name=tag.name,
        color=tag.color,
        usage_count=0,
        created_at=tag.created_at
    )