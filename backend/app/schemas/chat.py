from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict, field_validator


class _BaseModel(BaseModel):
    """Base model with ``from_attributes=True`` for ORM compatibility."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


# ---------------------------------------------------------------------------
# Thread-related schemas
# ---------------------------------------------------------------------------

class ThreadBase(_BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThreadCreate(ThreadBase):
    project_id: uuid.UUID
    initial_message: Optional[str] = None


class ThreadUpdate(_BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    is_archived: Optional[bool] = None


class ThreadResponse(ThreadBase):
    id: uuid.UUID
    project_id: uuid.UUID
    message_count: int
    total_tokens: int
    created_at: datetime
    last_activity_at: datetime
    summary_count: int
    is_archived: bool


class ThreadListResponse(_BaseModel):
    threads: List[ThreadResponse]
    total: int


# ---------------------------------------------------------------------------
# Message-related schemas
# ---------------------------------------------------------------------------

class MessageBase(_BaseModel):
    content: str = Field(..., min_length=1, max_length=50_000)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class MessageCreate(MessageBase):
    thread_id: uuid.UUID
    is_user: bool = True  # False denotes assistant messages


class MessageUpdate(_BaseModel):
    content: str = Field(..., min_length=1, max_length=50_000)


class MessageResponse(MessageBase):
    id: uuid.UUID
    thread_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    is_user: bool
    is_edited: bool
    is_deleted: bool
    token_count: int
    model_used: Optional[str] = None
    created_at: datetime
    edited_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None


class MessageListResponse(_BaseModel):
    messages: List[MessageResponse]
    total: int
