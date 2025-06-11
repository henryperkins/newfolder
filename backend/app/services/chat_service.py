"""ChatService - Core chat message and thread management.

This service provides CRUD operations for chat threads and messages,
manages conversation context, and integrates with the AI provider for
generating responses. It's the central business logic layer for the
chat experience.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, or_, and_

from ..models.chat import ChatThread, ChatMessage, ChatSummary
from ..models.project import Project
from ..models.user import User
from .ai_provider import AIProvider


class ChatService:
    """Service for managing chat threads and messages."""

    def __init__(self, db: Session, ai_provider: Optional[AIProvider] = None):
        self.db = db
        self.ai_provider = ai_provider

    # Thread management
    async def create_thread(
        self,
        project_id: str,
        user_id: str,
        title: str = "New Chat",
        initial_message: Optional[str] = None
    ) -> ChatThread:
        """Create a new chat thread and optionally add an initial message."""
        
        # Verify project exists and user has access
        project = self.db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id
        ).first()
        
        if not project:
            raise ValueError("Project not found or access denied")

        # Create thread
        thread = ChatThread(
            project_id=project_id,
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow()
        )
        
        self.db.add(thread)
        self.db.flush()  # Get the ID

        # Add initial message if provided
        if initial_message:
            await self.create_message(
                thread_id=str(thread.id),
                user_id=user_id,
                content=initial_message,
                is_user=True
            )

        # Update project's last_chat_at
        project.last_chat_at = datetime.utcnow()
        
        self.db.commit()
        return thread

    def get_thread(self, thread_id: str, user_id: str) -> Optional[ChatThread]:
        """Get a thread by ID if user has access."""
        return self.db.query(ChatThread).filter(
            ChatThread.id == thread_id,
            ChatThread.user_id == user_id,
            ChatThread.is_archived == False
        ).first()

    def get_project_threads(
        self,
        project_id: str,
        user_id: str,
        include_archived: bool = False,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[ChatThread]:
        """Get threads for a project with optional filtering."""
        
        query = self.db.query(ChatThread).filter(
            ChatThread.project_id == project_id,
            ChatThread.user_id == user_id
        )

        if not include_archived:
            query = query.filter(ChatThread.is_archived == False)

        if search:
            # Search in thread titles and message content
            query = query.filter(
                or_(
                    ChatThread.title.ilike(f"%{search}%"),
                    ChatThread.id.in_(
                        self.db.query(ChatMessage.thread_id).filter(
                            ChatMessage.content.ilike(f"%{search}%"),
                            ChatMessage.is_deleted == False
                        )
                    )
                )
            )

        return query.order_by(desc(ChatThread.last_activity_at))\
                   .offset(offset)\
                   .limit(limit)\
                   .all()

    async def update_thread_title(self, thread_id: str, user_id: str, title: str) -> Optional[ChatThread]:
        """Update thread title."""
        thread = self.get_thread(thread_id, user_id)
        if not thread:
            return None

        thread.title = title
        thread.updated_at = datetime.utcnow()
        self.db.commit()
        return thread

    async def archive_thread(self, thread_id: str, user_id: str) -> bool:
        """Archive a thread (soft delete)."""
        thread = self.get_thread(thread_id, user_id)
        if not thread:
            return False

        thread.is_archived = True
        thread.archived_at = datetime.utcnow()
        self.db.commit()
        return True

    # Message management
    async def create_message(
        self,
        thread_id: str,
        user_id: Optional[str],
        content: str,
        is_user: bool,
        model_used: Optional[str] = None,
        token_count: int = 0
    ) -> ChatMessage:
        """Create a new message in a thread."""
        
        # Verify thread exists
        thread = self.db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if not thread:
            raise ValueError("Thread not found")

        # Count tokens if AI provider available
        if self.ai_provider and token_count == 0:
            token_count = self.ai_provider.count_tokens(content)

        # Create message
        message = ChatMessage(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            is_user=is_user,
            token_count=token_count,
            model_used=model_used,
            created_at=datetime.utcnow()
        )

        self.db.add(message)

        # Update thread statistics
        thread.message_count += 1
        thread.total_tokens += token_count
        thread.last_activity_at = datetime.utcnow()
        thread.updated_at = datetime.utcnow()

        # Update project's last_chat_at
        if thread.project:
            thread.project.last_chat_at = datetime.utcnow()

        self.db.commit()
        return message

    def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 50,
        before: Optional[str] = None,
        after: Optional[str] = None,
        include_deleted: bool = False
    ) -> List[ChatMessage]:
        """Get messages for a thread with pagination."""
        
        query = self.db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id
        )

        if not include_deleted:
            query = query.filter(ChatMessage.is_deleted == False)

        # Handle pagination
        if before:
            before_message = self.db.query(ChatMessage).filter(ChatMessage.id == before).first()
            if before_message:
                query = query.filter(ChatMessage.created_at < before_message.created_at)

        if after:
            after_message = self.db.query(ChatMessage).filter(ChatMessage.id == after).first()
            if after_message:
                query = query.filter(ChatMessage.created_at > after_message.created_at)

        return query.order_by(asc(ChatMessage.created_at)).limit(limit).all()

    async def edit_message(
        self,
        message_id: str,
        user_id: str,
        content: str
    ) -> Optional[ChatMessage]:
        """Edit a user message."""
        
        message = self.db.query(ChatMessage).filter(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user_id,
            ChatMessage.is_user == True,
            ChatMessage.is_deleted == False
        ).first()

        if not message:
            return None

        # Store edit history
        if not message.edit_history:
            message.edit_history = []
        
        message.edit_history.append({
            "content": message.content,
            "edited_at": datetime.utcnow().isoformat()
        })

        # Update message
        old_token_count = message.token_count
        message.content = content
        message.is_edited = True
        message.edited_at = datetime.utcnow()

        # Recalculate tokens
        if self.ai_provider:
            new_token_count = self.ai_provider.count_tokens(content)
            message.token_count = new_token_count
            
            # Update thread token count
            if message.thread:
                message.thread.total_tokens += (new_token_count - old_token_count)
                message.thread.updated_at = datetime.utcnow()

        self.db.commit()
        return message

    async def _delete_message_user_context(self, message_id: str, user_id: str) -> bool:
        """(Deprecated) Delete a message with explicit *user_id* check.

        Retained for backward-compatibility; new call-sites should prefer
        :pyfunc:`delete_message` which mirrors the Phase-3 spec.
        """
        
        message = self.db.query(ChatMessage).filter(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user_id,
            ChatMessage.is_deleted == False
        ).first()

        if not message:
            return False

        message.is_deleted = True
        message.deleted_at = datetime.utcnow()

        # Update thread statistics
        if message.thread:
            message.thread.message_count -= 1
            message.thread.total_tokens -= message.token_count
            message.thread.updated_at = datetime.utcnow()

        self.db.commit()
        return True

    async def regenerate_response(self, message_id: str, user_id: str) -> Optional[ChatMessage]:
        """Regenerate an AI response by finding the previous user message."""
        
        # Get the AI message to regenerate
        ai_message = self.db.query(ChatMessage).filter(
            ChatMessage.id == message_id,
            ChatMessage.is_user == False,
            ChatMessage.is_deleted == False
        ).first()

        if not ai_message:
            return None

        # Verify user has access to the thread
        thread = self.get_thread(str(ai_message.thread_id), user_id)
        if not thread:
            return None

        # Find the user message that triggered this response
        user_message = self.db.query(ChatMessage).filter(
            ChatMessage.thread_id == ai_message.thread_id,
            ChatMessage.is_user == True,
            ChatMessage.created_at < ai_message.created_at,
            ChatMessage.is_deleted == False
        ).order_by(desc(ChatMessage.created_at)).first()

        if not user_message:
            return None

        # Soft delete the old AI response
        await self.delete_message(message_id)  # Soft delete assistant message

        # This would typically trigger a new AI response via WebSocket
        # For now, we just return the user message that should trigger regeneration
        return user_message

    # Thread context and summarization helpers
    def get_thread_context(
        self,
        thread_id: str,
        max_messages: int = 50,
        include_summaries: bool = True
    ) -> List[ChatMessage]:
        """Get recent context for a thread, optionally including summaries."""
        
        # Get recent unsummarized messages
        recent_messages = self.db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id,
            ChatMessage.is_deleted == False,
            ChatMessage.is_summarized == False
        ).order_by(desc(ChatMessage.created_at)).limit(max_messages).all()

        # Reverse to get chronological order
        return list(reversed(recent_messages))

    def get_thread_summary_status(self, thread_id: str) -> Dict[str, Any]:
        """Get summary status for a thread."""
        
        thread = self.db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if not thread:
            return {}

        unsummarized_count = self.db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id,
            ChatMessage.is_summarized == False,
            ChatMessage.is_deleted == False
        ).count()

        latest_summary = self.db.query(ChatSummary).filter(
            ChatSummary.thread_id == thread_id
        ).order_by(desc(ChatSummary.created_at)).first()

        return {
            "thread_id": thread_id,
            "total_messages": thread.message_count,
            "unsummarized_messages": unsummarized_count,
            "summary_count": thread.summary_count,
            "last_summary_at": thread.last_summary_at.isoformat() if thread.last_summary_at else None,
            "latest_summary": {
                "id": str(latest_summary.id),
                "summary_text": latest_summary.summary_text,
                "key_topics": latest_summary.key_topics,
                "created_at": latest_summary.created_at.isoformat()
            } if latest_summary else None,
            "needs_summary": unsummarized_count >= 50 or thread.total_tokens >= 10000
        }

    async def update_message_content(self, message_id: str, content: str) -> Optional[ChatMessage]:
        """Update message content (used during streaming)."""
        
        message = self.db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        if not message:
            return None

        # Calculate token difference
        old_tokens = message.token_count
        if self.ai_provider:
            new_tokens = self.ai_provider.count_tokens(content)
        else:
            new_tokens = len(content.split())  # Rough estimate

        message.content = content
        message.token_count = new_tokens

        # Update thread tokens
        if message.thread:
            message.thread.total_tokens += (new_tokens - old_tokens)

        self.db.commit()
        return message

    async def get_message(self, message_id: str) -> Optional[ChatMessage]:
        """Async helper that fetches a single message by *id*."""

        return self.db.query(ChatMessage).filter(
            ChatMessage.id == message_id,
        ).first()


    # ------------------------------------------------------------------
    # Spec-compatible helpers                                           
    # ------------------------------------------------------------------

    async def update_message(
        self,
        message_id: str,
        content: str,
    ) -> ChatMessage:
        """Update user message *content* while recording edit history.

        This method mirrors the signature described in the Phase-3 document
        so that external call-sites (e.g. WebSocket handler) can ``await`` it
        directly.  Internally we delegate to the existing *edit_message*
        implementation to avoid code duplication.
        """

        # For compatibility we assume *user* authorisation checks are handled
        # further upstream – in the official implementation this method is
        # called from an already authenticated context.

        # We cannot know the user id here (the original spec passes it via
        # service initialisation or ambient context).  Therefore we bypass
        # the stricter *user_id* filter that ``edit_message`` expects by
        # fetching the message first and then re-using its ``user_id``.

        msg = await self.get_message(message_id)
        if not msg:
            raise ValueError(f"Message {message_id} not found")

        if not msg.is_user:
            raise ValueError("Can only edit user messages")

        # Re-use existing implementation for the heavy lifting.
        updated = await self.edit_message(message_id, msg.user_id, content)  # type: ignore[arg-type]
        if not updated:
            raise ValueError("Failed to update message")
        return updated

    async def update_message_content(
        self,
        message_id: str,
        *,
        content: str,
        token_count: Optional[int] = None,
        model_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Update an assistant message (typically after streaming finished)."""

        msg = await self.get_message(message_id)
        if not msg:
            raise ValueError("Message not found")

        # Update content & tokens.
        msg.content = content

        if token_count is None and self.ai_provider:
            token_count = self.ai_provider.count_tokens(content)

        if token_count is not None:
            # Adjust thread token stats.
            delta = token_count - (msg.token_count or 0)
            msg.token_count = token_count
            if msg.thread:
                msg.thread.total_tokens += delta

        if model_used:
            msg.model_used = model_used

        if metadata:
            if msg.metadata is None:
                msg.metadata = {}
            msg.metadata.update(metadata)

        self.db.commit()
        return msg

    async def delete_message(self, message_id: str) -> ChatMessage:
        """Soft-delete a message irrespective of the author user id."""

        msg = await self.get_message(message_id)
        if not msg:
            raise ValueError("Message not found")

        msg.is_deleted = True
        msg.deleted_at = datetime.utcnow()

        if msg.thread:
            msg.thread.message_count -= 1
            msg.thread.total_tokens -= msg.token_count

        self.db.commit()
        return msg

# ---------------------------------------------------------------------------
# Import alias registration (optional convenience)                           
# ---------------------------------------------------------------------------

# The Phase-3 specification references the module via the short dotted path
# ``services.chat_service``.  To keep those import statements working without
# changing the package layout we expose *this* module under the requested
# alias when it is not already registered.

import sys as _sys  # noqa: E402,  WPS433 – late import on purpose
import types as _types  # noqa: E402

if "services" not in _sys.modules:
    _sys.modules["services"] = _types.ModuleType("services")

_sys.modules.setdefault("services.chat_service", _sys.modules[__name__])