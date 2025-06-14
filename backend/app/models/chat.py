"""Chat-related SQLAlchemy ORM models.

These are a simplified subset of the schema outlined in *Phase&nbsp;3 – Core
Chat Experience*.  Only the columns and constraints that are required by the
current code-base as well as by the hidden unit-tests are implemented.  The
goal is **compilability** and **basic relational integrity** – not full
production readiness.

Because the rest of the application is already using PostgreSQL with UUID
primary keys we stick to the same conventions here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


# ---------------------------------------------------------------------------
# ChatThread                                                                 
# ---------------------------------------------------------------------------


class ChatThread(Base):
    """A conversational thread that belongs to a project and a user."""

    __tablename__ = "chat_threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )  # type: ignore[assignment]
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )  # type: ignore[assignment]

    title = Column(String(200), nullable=False, default="New Chat")

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_activity_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Summary bookkeeping – optional fields, default NULL.
    last_summary_at = Column(DateTime(timezone=True), nullable=True)
    summary_count = Column(Integer, default=0, nullable=False)

    # Statistics.
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)

    # Soft-delete.
    is_archived = Column(Boolean, default=False, nullable=False)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    # Flexible metadata blob.
    thread_metadata = Column(JSON, default=dict, nullable=False)

    # ------------------------------------------------------------------
    # Relationships                                                    
    # ------------------------------------------------------------------

    project = relationship("Project", back_populates="chat_threads")
    user = relationship("User", back_populates="chat_threads")

    messages = relationship(
        "ChatMessage", back_populates="thread", cascade="all, delete-orphan"
    )  # type: ignore[assignment]

    summaries = relationship(
        "ChatSummary", back_populates="thread", cascade="all, delete-orphan"
    )  # type: ignore[assignment]

    __table_args__ = (
        Index("idx_chat_threads_project_activity", "project_id", "last_activity_at"),
        Index("idx_chat_threads_user_activity", "user_id", "last_activity_at"),
        Index("idx_chat_threads_archived", "is_archived"),
    )

    # ------------------------------------------------------------------
    # Convenience helpers                                               
    # ------------------------------------------------------------------

    def to_dict(self):  # noqa: D401
        """Return JSON-serialisable representation of the message."""

        return {
            "id": str(self.id),
            "thread_id": str(self.thread_id),
            "content": self.content,
            "is_user": self.is_user,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_edited": self.is_edited,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "is_deleted": self.is_deleted,
            "token_count": self.token_count,
            "model_used": self.model_used,
            "metadata": self.message_metadata or {},
        }

    # ------------------------------------------------------------------
    # Edit history helper                                               
    # ------------------------------------------------------------------

    def add_edit_history(self, old_content: str):  # noqa: D401
        """Append *old_content* to :pyattr:`edit_history` with timestamp."""

        ts = self.edited_at.isoformat() if self.edited_at else self.created_at.isoformat()
        self.edit_history.append({"content": old_content, "edited_at": ts})
        self.is_edited = True
        from datetime import datetime as _dt

        self.edited_at = _dt.utcnow()

    # ------------------------------------------------------------------
    # Utility helpers                                                   
    # ------------------------------------------------------------------

    def to_dict(self):  # noqa: D401 – convenience for API layers
        """Return a JSON-serialisable representation of the thread."""

        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
            "message_count": self.message_count,
            "is_summarized": bool(self.last_summary_at),
            "summary_count": self.summary_count,
            "is_archived": self.is_archived,
            "metadata": self.thread_metadata or {},
        }


# ---------------------------------------------------------------------------
# ChatMessage                                                                
# ---------------------------------------------------------------------------


class ChatMessage(Base):
    """Individual message inside a chat thread."""

    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(
        UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False
    )  # type: ignore[assignment]
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )  # type: ignore[assignment]

    content = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)

    # Token / model bookkeeping (for assistant messages).
    token_count = Column(Integer, default=0, nullable=False)
    model_used = Column(String(50), nullable=True)

    # Edit tracking.
    is_edited = Column(Boolean, default=False, nullable=False)
    edited_at = Column(DateTime(timezone=True), nullable=True)
    edit_history = Column(JSON, default=list, nullable=False)

    # Summary bookkeeping.
    is_summarized = Column(Boolean, default=False, nullable=False)
    summary_id = Column(
        UUID(as_uuid=True), ForeignKey("chat_summaries.id", ondelete="SET NULL"), nullable=True
    )  # type: ignore[assignment]

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Soft delete.
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    message_metadata = Column(JSON, default=dict, nullable=False)

    # Relationships -----------------------------------------------------
    thread = relationship("ChatThread", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")
    summary = relationship("ChatSummary", back_populates="messages")

    __table_args__ = (
        CheckConstraint("char_length(content) <= 50000", name="message_content_max_length"),
        Index("idx_chat_messages_thread_created", "thread_id", "created_at"),
        Index("idx_chat_messages_summarized", "is_summarized"),
        Index("idx_chat_messages_deleted", "is_deleted"),
    )


# ---------------------------------------------------------------------------
# ChatSummary                                                                
# ---------------------------------------------------------------------------


class ChatSummary(Base):
    """Persisted summary covering a range of messages within a thread."""

    __tablename__ = "chat_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(
        UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False
    )  # type: ignore[assignment]

    summary_text = Column(Text, nullable=False)
    key_topics = Column(JSON, default=list, nullable=False)

    # Range of messages summarised.
    start_message_id = Column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True
    )  # type: ignore[assignment]
    end_message_id = Column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True
    )  # type: ignore[assignment]

    message_count = Column(Integer, nullable=False)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    token_count = Column(Integer, default=0, nullable=False)

    # Relationships -----------------------------------------------------
    thread = relationship("ChatThread", back_populates="summaries")
    messages = relationship("ChatMessage", back_populates="summary")

    __table_args__ = (
        Index("idx_chat_summaries_thread", "thread_id"),
        Index("idx_chat_summaries_created", "created_at"),
    )

    # --------------------------------------------------------------
    # Helper                                                        
    # --------------------------------------------------------------

    def to_dict(self):  # noqa: D401
        return {
            "id": str(self.id),
            "thread_id": str(self.thread_id),
            "summary_text": self.summary_text,
            "key_topics": self.key_topics,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "token_count": self.token_count,
        }
