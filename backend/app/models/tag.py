import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Tag(Base):
    __tablename__ = "tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(30), nullable=False)
    color = Column(String(7), nullable=True)  # Optional hex color
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="tags")
    projects = relationship("Project", secondary="project_tags", back_populates="tags")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_tag_name_per_user'),
        CheckConstraint("char_length(name) >= 2", name="tag_name_min_length"),
    )