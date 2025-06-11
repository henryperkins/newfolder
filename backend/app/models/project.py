import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


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
    last_chat_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="projects")
    tags = relationship("Tag", secondary="project_tags", back_populates="projects")
    activities = relationship("ActivityLog", back_populates="project")
    # Chat threads relationship – added for Phase-3 chat feature.
    chat_threads = relationship("ChatThread", back_populates="project", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_project_name_per_user'),
        CheckConstraint("char_length(name) >= 3", name="project_name_min_length"),
        CheckConstraint("color ~* '^#[0-9A-Fa-f]{6}$'", name="valid_hex_color"),
        Index('idx_projects_user_id', 'user_id'),
        Index('idx_projects_last_activity', 'last_activity_at'),
    )