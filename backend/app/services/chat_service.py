"""Async ChatService using SQLAlchemy 2.0 style queries.

Only the subset of behaviour that is required by *chat_routes.py* and the
WebSocket message-handler has been migrated so far.  Additional helpers can be
ported later using the same patterns shown below.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import select, update, delete, desc, asc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

# Model imports -------------------------------------------------------------

from ..models.chat import ChatThread, ChatMessage, ChatSummary
from ..models.project import Project
from ..models.user import User

# AI provider abstraction ---------------------------------------------------

from .ai_provider import AIProvider


class ChatService:  # noqa: D401 – service container
    """Business-logic layer for chat threads and messages (async version)."""

    def __init__(self, db: AsyncSession, ai_provider: Optional[AIProvider] = None) -> None:  # noqa: D401
        self.db = db
        self.ai_provider = ai_provider

    # ---------------------------------------------------------------------
    # Thread management                                                    
    # ---------------------------------------------------------------------

    async def create_thread(
        self,
        *,
        project_id: str,
        user_id: str,
        title: str = "New Chat",
        initial_message: Optional[str] = None,
    ) -> ChatThread:
        """Create a new thread and optional first user message."""

        # Verify ownership -------------------------------------------------
        stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
        project = (await self.db.scalar(stmt))
        if project is None:
            raise ValueError("Project not found or access denied")

        thread = ChatThread(
            project_id=project_id,
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow(),
        )

        self.db.add(thread)
        await self.db.flush()  # populate PK

        # optional initial message ---------------------------------------
        if initial_message:
            await self.create_message(
                thread_id=str(thread.id),
                user_id=user_id,
                content=initial_message,
                is_user=True,
            )

        project.last_chat_at = datetime.utcnow()
        await self.db.commit()

        return thread

    async def get_thread(self, thread_id: str, user_id: str) -> Optional[ChatThread]:
        stmt = (
            select(ChatThread)
            .where(
                ChatThread.id == thread_id,
                ChatThread.user_id == user_id,
                ChatThread.is_archived.is_(False),
            )
        )
        return await self.db.scalar(stmt)

    async def get_project_threads(
        self,
        project_id: str,
        user_id: str,
        *,
        include_archived: bool = False,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> List[ChatThread]:
        stmt = select(ChatThread).where(
            ChatThread.project_id == project_id,
            ChatThread.user_id == user_id,
        )

        if not include_archived:
            stmt = stmt.where(ChatThread.is_archived.is_(False))

        if search:
            # Subquery for message search
            sub = (
                select(ChatMessage.thread_id)
                .where(
                    ChatMessage.content.ilike(f"%{search}%"),
                    ChatMessage.is_deleted.is_(False),
                )
            )
            stmt = stmt.where(
                or_(ChatThread.title.ilike(f"%{search}%"), ChatThread.id.in_(sub))
            )

        stmt = (
            stmt.order_by(desc(ChatThread.last_activity_at)).offset(offset).limit(limit)
        )

        result = await self.db.scalars(stmt)
        return result.all()

    async def update_thread_title(
        self, thread_id: str, user_id: str, title: str
    ) -> Optional[ChatThread]:
        thread = await self.get_thread(thread_id, user_id)
        if thread is None:
            return None

        thread.title = title
        thread.updated_at = datetime.utcnow()
        await self.db.commit()
        return thread

    async def archive_thread(self, thread_id: str, user_id: str) -> bool:
        thread = await self.get_thread(thread_id, user_id)
        if thread is None:
            return False

        thread.is_archived = True
        thread.archived_at = datetime.utcnow()
        await self.db.commit()
        return True

    async def unarchive_thread(self, thread_id: str, user_id: str) -> bool:
        """Unarchive a previously archived thread."""
        stmt = (
            select(ChatThread)
            .where(
                ChatThread.id == thread_id,
                ChatThread.user_id == user_id,
                ChatThread.is_archived.is_(True),
            )
        )
        thread = await self.db.scalar(stmt)
        if thread is None:
            return False

        thread.is_archived = False
        thread.archived_at = None
        thread.updated_at = datetime.utcnow()
        await self.db.commit()
        return True

    # ------------------------------------------------------------------
    # Message helpers                                                   
    # ------------------------------------------------------------------

    async def create_message(
        self,
        *,
        thread_id: str,
        user_id: Optional[str],
        content: str,
        is_user: bool,
        model_used: Optional[str] = None,
        token_count: int = 0,
    ) -> ChatMessage:
        stmt = select(ChatThread).where(ChatThread.id == thread_id)
        thread = await self.db.scalar(stmt)
        if thread is None:
            raise ValueError("Thread not found")

        # token counting ---------------------------------------------------
        if self.ai_provider and token_count == 0:
            token_count = self.ai_provider.count_tokens(content)

        msg = ChatMessage(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            is_user=is_user,
            token_count=token_count,
            model_used=model_used,
            created_at=datetime.utcnow(),
        )

        self.db.add(msg)

        # update thread stats
        thread.message_count += 1
        thread.total_tokens += token_count
        thread.last_activity_at = datetime.utcnow()
        thread.updated_at = datetime.utcnow()

        if thread.project:
            thread.project.last_chat_at = datetime.utcnow()

        await self.db.commit()
        return msg

    async def get_thread_messages(
        self,
        thread_id: str,
        *,
        limit: int = 50,
        include_deleted: bool = False,
    ) -> List[ChatMessage]:
        stmt = select(ChatMessage).where(ChatMessage.thread_id == thread_id)

        if not include_deleted:
            stmt = stmt.where(ChatMessage.is_deleted.is_(False))

        stmt = stmt.order_by(asc(ChatMessage.created_at)).limit(limit)
        result = await self.db.scalars(stmt)
        return result.all()

    # ------------------------------------------------------------------
    # Convenience wrappers used by WebSocket & REST                      
    # ------------------------------------------------------------------

    async def update_message(self, message_id: str, content: str) -> ChatMessage:  # noqa: D401
        stmt = select(ChatMessage).where(ChatMessage.id == message_id)
        msg = await self.db.scalar(stmt)
        if msg is None:
            raise ValueError("Message not found")

        if not msg.is_user:
            raise ValueError("Can only edit user messages")

        old_tokens = msg.token_count
        msg.content = content
        msg.is_edited = True
        msg.edited_at = datetime.utcnow()

        if self.ai_provider:
            new_tokens = self.ai_provider.count_tokens(content)
            msg.token_count = new_tokens

            if msg.thread:
                msg.thread.total_tokens += new_tokens - old_tokens

        await self.db.commit()
        return msg

    async def delete_message(self, message_id: str) -> ChatMessage:
        stmt = select(ChatMessage).where(ChatMessage.id == message_id)
        msg = await self.db.scalar(stmt)
        if msg is None:
            raise ValueError("Message not found")

        msg.is_deleted = True
        msg.deleted_at = datetime.utcnow()

        if msg.thread:
            msg.thread.message_count -= 1
            msg.thread.total_tokens -= msg.token_count

        await self.db.commit()
        return msg

    async def regenerate_response(self, message_id: str, user_id: str) -> Optional[ChatMessage]:
        """Get the user message that triggered a response for regeneration."""
        stmt = select(ChatMessage).where(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user_id,
            ChatMessage.is_user.is_(True),
            ChatMessage.is_deleted.is_(False),
        )
        user_msg = await self.db.scalar(stmt)
        
        if user_msg is None:
            return None
            
        # Find and soft-delete the subsequent assistant response if it exists
        next_msg_stmt = select(ChatMessage).where(
            ChatMessage.thread_id == user_msg.thread_id,
            ChatMessage.created_at > user_msg.created_at,
            ChatMessage.is_user.is_(False),
            ChatMessage.is_deleted.is_(False),
        ).order_by(asc(ChatMessage.created_at)).limit(1)
        
        assistant_msg = await self.db.scalar(next_msg_stmt)
        if assistant_msg:
            # Mark old assistant response as deleted
            assistant_msg.is_deleted = True
            assistant_msg.deleted_at = datetime.utcnow()
            
            # Update thread statistics
            if assistant_msg.thread:
                assistant_msg.thread.message_count -= 1
                assistant_msg.thread.total_tokens -= assistant_msg.token_count
            
            await self.db.commit()
        
        return user_msg
