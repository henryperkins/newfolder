import uuid
from enum import Enum
from sqlalchemy import Column, DateTime, ForeignKey, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import Enum as SQLEnum
from ..core.database import Base


class ActivityType(str, Enum):
    PROJECT_CREATED = "project_created"
    PROJECT_UPDATED = "project_updated"
    PROJECT_ARCHIVED = "project_archived"
    PROJECT_DELETED = "project_deleted"
    CHAT_STARTED = "chat_started"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    SEARCH_PERFORMED = "search_performed"


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    activity_type = Column(SQLEnum(ActivityType), nullable=False)
    activity_metadata = Column(JSON, nullable=False, default={})
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
