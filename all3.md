
# Phase 3: Core Chat Experience - Complete Implementation

## 🔥 Production-Ready Code Delivery

### Database Models & Migrations

#### **models/chat.py**
*Purpose: SQLAlchemy models for chat threads, messages, and summaries*

```python
"""
Chat-related database models for the AI Productivity App.
Implements thread-based conversations with message history and summarization.
"""
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean, Integer, JSON, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from models.base import Base


class ChatThread(Base):
    """
    Represents a conversation thread within a project.
    Tracks message history, summarization status, and usage statistics.
    """
    __tablename__ = "chat_threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(200), nullable=False, default="New Chat")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Summary tracking
    last_summary_at = Column(DateTime(timezone=True), nullable=True)
    summary_count = Column(Integer, default=0, nullable=False)

    # Statistics
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)

    # Soft delete
    is_archived = Column(Boolean, default=False, nullable=False)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata for extensibility
    metadata = Column(JSON, default=dict, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="chat_threads")
    user = relationship("User", back_populates="chat_threads")
    messages = relationship("ChatMessage", back_populates="thread", cascade="all, delete-orphan", order_by="ChatMessage.created_at")
    summaries = relationship("ChatSummary", back_populates="thread", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_chat_threads_project_activity', 'project_id', 'last_activity_at'),
        Index('idx_chat_threads_user_activity', 'user_id', 'last_activity_at'),
        Index('idx_chat_threads_archived', 'is_archived'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert thread to dictionary for API responses."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "message_count": self.message_count,
            "is_summarized": bool(self.last_summary_at),
            "summary_count": self.summary_count,
            "is_archived": self.is_archived,
            "metadata": self.metadata
        }


class ChatMessage(Base):
    """
    Individual message within a chat thread.
    Supports editing, soft deletion, and token tracking.
    """
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    content = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)

    # Message metadata
    token_count = Column(Integer, nullable=False, default=0)
    model_used = Column(String(50), nullable=True)  # For AI responses

    # Edit tracking
    is_edited = Column(Boolean, default=False, nullable=False)
    edited_at = Column(DateTime(timezone=True), nullable=True)
    edit_history = Column(JSON, default=list, nullable=False)  # List of previous versions

    # Summary tracking
    is_summarized = Column(Boolean, default=False, nullable=False)
    summary_id = Column(UUID(as_uuid=True), ForeignKey("chat_summaries.id", ondelete="SET NULL"), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Soft delete
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Additional metadata
    metadata = Column(JSON, default=dict, nullable=False)

    # Relationships
    thread = relationship("ChatThread", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")
    summary = relationship("ChatSummary", back_populates="messages", foreign_keys=[summary_id])

    # Constraints
    __table_args__ = (
        CheckConstraint("char_length(content) <= 50000", name="message_content_max_length"),
        Index('idx_chat_messages_thread_created', 'thread_id', 'created_at'),
        Index('idx_chat_messages_summarized', 'is_summarized'),
        Index('idx_chat_messages_deleted', 'is_deleted'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for API responses."""
        return {
            "id": str(self.id),
            "thread_id": str(self.thread_id),
            "content": self.content,
            "is_user": self.is_user,
            "created_at": self.created_at.isoformat(),
            "is_edited": self.is_edited,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "is_deleted": self.is_deleted,
            "token_count": self.token_count,
            "model_used": self.model_used,
            "metadata": self.metadata
        }

    def add_edit_history(self, old_content: str) -> None:
        """Add current content to edit history before updating."""
        self.edit_history.append({
            "content": old_content,
            "edited_at": self.edited_at.isoformat() if self.edited_at else self.created_at.isoformat()
        })
        self.is_edited = True
        self.edited_at = datetime.utcnow()


class ChatSummary(Base):
    """
    Summarization of chat thread segments.
    Enables efficient context management for long conversations.
    """
    __tablename__ = "chat_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False)
    summary_text = Column(Text, nullable=False)
    key_topics = Column(JSON, default=list, nullable=False)  # List of key topics

    # Range of messages summarized
    start_message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True)
    end_message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True)
    message_count = Column(Integer, nullable=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    token_count = Column(Integer, nullable=False, default=0)

    # Relationships
    thread = relationship("ChatThread", back_populates="summaries")
    messages = relationship("ChatMessage", back_populates="summary",
                          primaryjoin="and_(ChatMessage.summary_id==ChatSummary.id)",
                          foreign_keys="ChatMessage.summary_id")

    # Indexes
    __table_args__ = (
        Index('idx_chat_summaries_thread', 'thread_id'),
        Index('idx_chat_summaries_created', 'created_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary for API responses."""
        return {
            "id": str(self.id),
            "thread_id": str(self.thread_id),
            "summary_text": self.summary_text,
            "key_topics": self.key_topics,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count
        }
```

#### **alembic/versions/003_add_chat_tables.py**
*Purpose: Database migration to create chat-related tables*

```python
"""Add chat tables for Phase 3

Revision ID: 003_add_chat_tables
Revises: 002_add_project_tables
Create Date: 2024-02-01 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003_add_chat_tables'
down_revision = '002_add_project_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create chat-related tables and indexes."""

    # Create chat_threads table
    op.create_table('chat_threads',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(200), nullable=False, server_default="New Chat"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_summary_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('summary_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('message_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False, server_default='{}'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create chat_messages table
    op.create_table('chat_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('thread_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('is_user', sa.Boolean(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('model_used', sa.String(50), nullable=True),
        sa.Column('is_edited', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('edited_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('edit_history', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('is_summarized', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False, server_default='{}'),
        sa.ForeignKeyConstraint(['thread_id'], ['chat_threads.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("char_length(content) <= 50000", name='message_content_max_length')
    )

    # Create chat_summaries table
    op.create_table('chat_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('thread_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('summary_text', sa.Text(), nullable=False),
        sa.Column('key_topics', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('start_message_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('end_message_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['thread_id'], ['chat_threads.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['start_message_id'], ['chat_messages.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['end_message_id'], ['chat_messages.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Add summary_id column to chat_messages
    op.add_column('chat_messages',
        sa.Column('summary_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        'fk_chat_messages_summary',
        'chat_messages', 'chat_summaries',
        ['summary_id'], ['id'],
        ondelete='SET NULL'
    )

    # Create indexes for performance
    op.create_index('idx_chat_threads_project_activity', 'chat_threads', ['project_id', 'last_activity_at'])
    op.create_index('idx_chat_threads_user_activity', 'chat_threads', ['user_id', 'last_activity_at'])
    op.create_index('idx_chat_threads_archived', 'chat_threads', ['is_archived'])

    op.create_index('idx_chat_messages_thread_created', 'chat_messages', ['thread_id', 'created_at'])
    op.create_index('idx_chat_messages_summarized', 'chat_messages', ['is_summarized'])
    op.create_index('idx_chat_messages_deleted', 'chat_messages', ['is_deleted'])

    op.create_index('idx_chat_summaries_thread', 'chat_summaries', ['thread_id'])
    op.create_index('idx_chat_summaries_created', 'chat_summaries', ['created_at'])

    # Update projects table to track last chat activity
    op.add_column('projects',
        sa.Column('last_chat_at', sa.DateTime(timezone=True), nullable=True)
    )

    # Update relationships in existing models (handled by SQLAlchemy)


def downgrade() -> None:
    """Drop chat-related tables and indexes."""

    # Remove column from projects
    op.drop_column('projects', 'last_chat_at')

    # Drop indexes
    op.drop_index('idx_chat_summaries_created', 'chat_summaries')
    op.drop_index('idx_chat_summaries_thread', 'chat_summaries')

    op.drop_index('idx_chat_messages_deleted', 'chat_messages')
    op.drop_index('idx_chat_messages_summarized', 'chat_messages')
    op.drop_index('idx_chat_messages_thread_created', 'chat_messages')

    op.drop_index('idx_chat_threads_archived', 'chat_threads')
    op.drop_index('idx_chat_threads_user_activity', 'chat_threads')
    op.drop_index('idx_chat_threads_project_activity', 'chat_threads')

    # Drop foreign key and column
    op.drop_constraint('fk_chat_messages_summary', 'chat_messages', type_='foreignkey')
    op.drop_column('chat_messages', 'summary_id')

    # Drop tables
    op.drop_table('chat_summaries')
    op.drop_table('chat_messages')
    op.drop_table('chat_threads')
```

### Backend Services

#### **services/ai_provider.py**
*Purpose: Modular AI provider service with OpenAI implementation*

```python
"""
AI Provider Service - Abstraction layer for AI model interactions.
Supports streaming responses, token management, and provider switching.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List, Union
from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI
import tiktoken
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class AIMessage:
    """Represents a message in the conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class AIResponse:
    """Complete AI response with metadata."""
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        **kwargs
    ) -> Union[AsyncGenerator[str, None], AIResponse]:
        """Generate completion from messages."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum token limit for the model."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is accessible."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI implementation of AI provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 4096,
        timeout: int = 30
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=3
        )
        self.model = model
        self.max_tokens = max_tokens
        self._encoder = None
        self._model_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000
        }

    @property
    def encoder(self):
        """Lazy load tokenizer."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback for newer models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        **kwargs
    ) -> Union[AsyncGenerator[str, None], AIResponse]:
        """Generate completion from messages."""

        # Convert to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Validate token count
        total_tokens = sum(self.count_tokens(msg.content) for msg in messages)
        if total_tokens > self.get_max_tokens() * 0.9:  # Leave room for response
            logger.warning(f"High token usage: {total_tokens}/{self.get_max_tokens()}")

        try:
            if stream:
                return self._stream_completion(
                    openai_messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    **kwargs
                )
            else:
                return await self._complete(
                    openai_messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    async def _stream_completion(
        self,
        messages: List[Dict],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion chunks."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except asyncio.CancelledError:
            # Handle graceful cancellation
            logger.info("Stream cancelled by client")
            raise
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"\n\n[Error: {str(e)}]"

    async def _complete(
        self,
        messages: List[Dict],
        **kwargs
    ) -> AIResponse:
        """Get complete response at once."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=False,
            **kwargs
        )

        choice = response.choices[0]
        usage = response.usage

        return AIResponse(
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            model=response.model
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logger.error(f"Token counting error: {str(e)}")
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4

    def get_max_tokens(self) -> int:
        """Get max tokens for model."""
        return self._model_limits.get(self.model, 4096)

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Simple API call to verify connectivity
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


class MockAIProvider(AIProvider):
    """Mock AI provider for testing."""

    def __init__(self, response: str = "This is a mock response.", delay: float = 0.1):
        self.response = response
        self.delay = delay
        self.call_count = 0

    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        **kwargs
    ) -> Union[AsyncGenerator[str, None], AIResponse]:
        """Generate mock completion."""
        self.call_count += 1

        if stream:
            return self._mock_stream()
        else:
            await asyncio.sleep(self.delay)
            return AIResponse(
                content=self.response,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                model="mock-model"
            )

    async def _mock_stream(self) -> AsyncGenerator[str, None]:
        """Mock streaming response."""
        words = self.response.split()
        for word in words:
            await asyncio.sleep(self.delay)
            yield word + " "

    def count_tokens(self, text: str) -> int:
        """Mock token counting."""
        return len(text.split())

    def get_max_tokens(self) -> int:
        """Mock max tokens."""
        return 4096

    async def health_check(self) -> bool:
        """Always healthy for mock."""
        return True


class AIProviderFactory:
    """Factory for creating AI providers."""

    @staticmethod
    def create(provider_type: str, config: Dict[str, Any]) -> AIProvider:
        """Create AI provider instance based on configuration."""
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=config["api_key"],
                model=config.get("model", "gpt-4"),
                max_tokens=config.get("max_tokens", 4096),
                timeout=config.get("timeout", 30)
            )
        elif provider_type == "mock":
            return MockAIProvider(
                response=config.get("response", "Mock response"),
                delay=config.get("delay", 0.1)
            )
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


class ConversationManager:
    """Manages conversation context and token limits."""

    def __init__(self, ai_provider: AIProvider, system_prompt: Optional[str] = None):
        self.ai_provider = ai_provider
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        # Reserve 25% of tokens for response
        self.max_context_tokens = int(ai_provider.get_max_tokens() * 0.75)

    def prepare_messages(
        self,
        messages: List['ChatMessage'],
        include_system: bool = True
    ) -> List[AIMessage]:
        """
        Prepare messages for AI provider with token management.
        Intelligently truncates old messages if needed.
        """
        ai_messages = []

        # Add system prompt if requested
        if include_system and self.system_prompt:
            ai_messages.append(AIMessage(role="system", content=self.system_prompt))

        # Calculate token budget
        total_tokens = self.ai_provider.count_tokens(self.system_prompt) if include_system else 0

        # Convert messages in reverse order (keep recent context)
        message_buffer = []
        for message in reversed(messages):
            if message.is_deleted:
                continue

            msg_tokens = self.ai_provider.count_tokens(message.content)

            # Check if adding this message would exceed limit
            if total_tokens + msg_tokens > self.max_context_tokens:
                # Try to at least include a truncated version of the last message
                if not message_buffer and message.is_user:
                    truncated_content = message.content[:500] + "... [truncated]"
                    message_buffer.append(AIMessage(
                        role="user" if message.is_user else "assistant",
                        content=truncated_content
                    ))
                break

            message_buffer.append(AIMessage(
                role="user" if message.is_user else "assistant",
                content=message.content
            ))
            total_tokens += msg_tokens

        # Add messages in correct order
        ai_messages.extend(reversed(message_buffer))

        logger.info(f"Prepared {len(ai_messages)} messages using {total_tokens} tokens")
        return ai_messages

    async def get_thread_summary(
        self,
        messages: List['ChatMessage'],
        max_length: int = 500
    ) -> str:
        """Generate summary of conversation thread."""
        summary_prompt = f"""
        Summarize this conversation in {max_length} characters or less.
        Focus on key topics, decisions, and outcomes.
        Use bullet points for clarity.
        Be concise but comprehensive.
        """

        summary_messages = [
            AIMessage(role="system", content=summary_prompt)
        ]

        # Add conversation messages (limit to prevent token overflow)
        for msg in messages[:50]:  # First 50 messages
            summary_messages.append(
                AIMessage(
                    role="user" if msg.is_user else "assistant",
                    content=msg.content[:500]  # Truncate long messages
                )
            )

        try:
            response = await self.ai_provider.complete(
                summary_messages,
                stream=False,
                temperature=0.3,  # Lower temperature for factual summary
                max_tokens=200
            )

            return response.content

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Summary generation failed. Please try again later."
```

#### **services/websocket_manager.py**
*Purpose: WebSocket connection management and message routing*

```python
"""
WebSocket Manager - Handles real-time bidirectional communication.
Manages connections, message routing, and heartbeat monitoring.
"""
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect, status
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from collections import defaultdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    # Connection
    CONNECTION_ESTABLISHED = "connection_established"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"

    # Messages
    SEND_MESSAGE = "send_message"
    NEW_MESSAGE = "new_message"
    EDIT_MESSAGE = "edit_message"
    MESSAGE_UPDATED = "message_updated"
    DELETE_MESSAGE = "delete_message"
    MESSAGE_DELETED = "message_deleted"

    # Streaming
    ASSISTANT_MESSAGE_START = "assistant_message_start"
    STREAM_CHUNK = "stream_chunk"

    # Status
    TYPING_INDICATOR = "typing_indicator"
    ERROR = "error"
    SUMMARY_AVAILABLE = "summary_available"

    # Actions
    REGENERATE = "regenerate"


class ConnectionManager:
    """Manages WebSocket connections and message routing."""

    def __init__(self):
        # Map of thread_id -> set of connections
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Map of websocket -> connection info
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        # Map of connection_id -> last_heartbeat
        self.heartbeats: Dict[str, datetime] = {}
        # Rate limiting: connection_id -> list of timestamps
        self.message_timestamps: Dict[str, list] = defaultdict(list)

        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 60  # seconds
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max_messages = 60

        # Start background tasks
        self._monitor_task = None

    async def start_monitoring(self):
        """Start heartbeat monitoring task."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_heartbeats())

    async def stop_monitoring(self):
        """Stop heartbeat monitoring task."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def connect(
        self,
        websocket: WebSocket,
        thread_id: str,
        user_id: str
    ) -> str:
        """Accept new WebSocket connection."""
        try:
            await websocket.accept()

            connection_id = str(uuid.uuid4())

            # Store connection info
            self.active_connections[thread_id].add(websocket)
            self.connection_info[websocket] = {
                "connection_id": connection_id,
                "thread_id": thread_id,
                "user_id": user_id,
                "connected_at": datetime.utcnow()
            }
            self.heartbeats[connection_id] = datetime.utcnow()

            # Send connection established message
            await websocket.send_json({
                "type": MessageType.CONNECTION_ESTABLISHED,
                "connection_id": connection_id,
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat(),
                "protocol_version": "1.0"
            })

            logger.info(f"WebSocket connected: {connection_id} for thread {thread_id}")
            return connection_id

        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket not in self.connection_info:
            return

        info = self.connection_info[websocket]
        connection_id = info["connection_id"]
        thread_id = info["thread_id"]

        # Clean up connection data
        self.active_connections[thread_id].discard(websocket)
        if not self.active_connections[thread_id]:
            del self.active_connections[thread_id]

        del self.connection_info[websocket]
        self.heartbeats.pop(connection_id, None)
        self.message_timestamps.pop(connection_id, None)

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_message(
        self,
        thread_id: str,
        message: Dict[str, Any],
        exclude_websocket: Optional[WebSocket] = None
    ):
        """Broadcast message to all connections in a thread."""
        if thread_id not in self.active_connections:
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        disconnected = set()
        for websocket in self.active_connections[thread_id]:
            # Skip excluded connection (usually the sender)
            if websocket == exclude_websocket:
                continue

            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                disconnected.add(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            await self.disconnect(ws)

    async def send_stream_chunk(
        self,
        thread_id: str,
        message_id: str,
        chunk: str,
        chunk_index: int = 0,
        is_final: bool = False
    ):
        """Send streaming chunk to all connections."""
        message = {
            "type": MessageType.STREAM_CHUNK,
            "message_id": message_id,
            "chunk": chunk,
            "chunk_index": chunk_index,
            "is_final": is_final,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_message(thread_id, message)

    async def send_error(
        self,
        websocket: WebSocket,
        error_code: str,
        error_message: str,
        message_id: Optional[str] = None
    ):
        """Send error message to specific connection."""
        error_data = {
            "type": MessageType.ERROR,
            "error_code": error_code,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }

        if message_id:
            error_data["message_id"] = message_id

        try:
            await websocket.send_json(error_data)
        except Exception as e:
            logger.error(f"Failed to send error: {str(e)}")

    async def handle_heartbeat(self, websocket: WebSocket):
        """Update heartbeat timestamp and send acknowledgment."""
        if websocket in self.connection_info:
            connection_id = self.connection_info[websocket]["connection_id"]
            self.heartbeats[connection_id] = datetime.utcnow()

            await websocket.send_json({
                "type": MessageType.HEARTBEAT_ACK,
                "timestamp": datetime.utcnow().isoformat()
            })

    async def check_rate_limit(self, websocket: WebSocket) -> bool:
        """Check if connection is within rate limits."""
        if websocket not in self.connection_info:
            return False

        connection_id = self.connection_info[websocket]["connection_id"]
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.rate_limit_window)

        # Clean old timestamps
        self.message_timestamps[connection_id] = [
            ts for ts in self.message_timestamps[connection_id]
            if ts > window_start
        ]

        # Check limit
        if len(self.message_timestamps[connection_id]) >= self.rate_limit_max_messages:
            return False

        # Record new message
        self.message_timestamps[connection_id].append(now)
        return True

    async def broadcast_typing_indicator(
        self,
        thread_id: str,
        user_id: str,
        is_typing: bool,
        exclude_websocket: Optional[WebSocket] = None
    ):
        """Broadcast typing indicator status."""
        message = {
            "type": MessageType.TYPING_INDICATOR,
            "user_id": user_id,
            "is_typing": is_typing,
            "thread_id": thread_id
        }
        await self.send_message(thread_id, message, exclude_websocket)

    async def _monitor_heartbeats(self):
        """Monitor and disconnect stale connections."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                now = datetime.utcnow()
                timeout_threshold = now - timedelta(seconds=self.heartbeat_timeout)

                # Find stale connections
                stale_connections = []
                for conn_id, last_heartbeat in self.heartbeats.items():
                    if last_heartbeat < timeout_threshold:
                        # Find websocket for this connection
                        for ws, info in self.connection_info.items():
                            if info["connection_id"] == conn_id:
                                stale_connections.append(ws)
                                break

                # Disconnect stale connections
                for ws in stale_connections:
                    logger.warning(f"Disconnecting stale connection: {self.connection_info[ws]['connection_id']}")
                    try:
                        await ws.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Heartbeat timeout")
                    except:
                        pass
                    await self.disconnect(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {str(e)}")

    def get_connection_count(self, thread_id: str) -> int:
        """Get number of active connections for a thread."""
        return len(self.active_connections.get(thread_id, set()))

    def get_all_connections(self) -> Dict[str, int]:
        """Get connection counts for all threads."""
        return {
            thread_id: len(connections)
            for thread_id, connections in self.active_connections.items()
        }


# Global connection manager instance
connection_manager = ConnectionManager()


class MessageHandler:
    """Handles different types of WebSocket messages."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        chat_service: 'ChatService',
        ai_provider: 'AIProvider'
    ):
        self.connection_manager = connection_manager
        self.chat_service = chat_service
        self.ai_provider = ai_provider
        self.handlers = {
            MessageType.HEARTBEAT: self.handle_heartbeat,
            MessageType.SEND_MESSAGE: self.handle_send_message,
            MessageType.EDIT_MESSAGE: self.handle_edit_message,
            MessageType.DELETE_MESSAGE: self.handle_delete_message,
            MessageType.TYPING_INDICATOR: self.handle_typing_indicator,
            MessageType.REGENERATE: self.handle_regenerate
        }

    async def handle_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ):
        """Route message to appropriate handler."""
        message_type = message.get("type")

        if not message_type:
            await self.connection_manager.send_error(
                websocket,
                "invalid_message",
                "Message type is required"
            )
            return

        handler = self.handlers.get(message_type)
        if not handler:
            await self.connection_manager.send_error(
                websocket,
                "unknown_message_type",
                f"Unknown message type: {message_type}"
            )
            return

        try:
            await handler(websocket, message)
        except Exception as e:
            logger.error(f"Handler error for {message_type}: {str(e)}")
            await self.connection_manager.send_error(
                websocket,
                "handler_error",
                f"Error processing {message_type}: {str(e)}"
            )

    async def handle_heartbeat(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle heartbeat message."""
        await self.connection_manager.handle_heartbeat(websocket)

    async def handle_send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle new message from user."""
        # Check rate limit
        if not await self.connection_manager.check_rate_limit(websocket):
            await self.connection_manager.send_error(
                websocket,
                "rate_limit_exceeded",
                "Too many messages. Please wait before sending more."
            )
            return

        # Extract message data
        thread_id = message.get("thread_id")
        content = message.get("content", "").strip()

        if not thread_id or not content:
            await self.connection_manager.send_error(
                websocket,
                "invalid_message",
                "Thread ID and content are required"
            )
            return

        # Get user info
        conn_info = self.connection_manager.connection_info.get(websocket)
        if not conn_info:
            await self.connection_manager.send_error(
                websocket,
                "connection_error",
                "Connection info not found"
            )
            return

        user_id = conn_info["user_id"]

        # Create message via service
        try:
            user_message = await self.chat_service.create_message(
                thread_id=thread_id,
                user_id=user_id,
                content=content,
                is_user=True
            )

            # Broadcast user message to other connections
            await self.connection_manager.send_message(
                thread_id,
                {
                    "type": MessageType.NEW_MESSAGE,
                    "message": user_message.to_dict()
                },
                exclude_websocket=websocket
            )

            # Generate AI response
            await self.generate_ai_response(thread_id, user_message)

        except Exception as e:
            logger.error(f"Message creation error: {str(e)}")
            await self.connection_manager.send_error(
                websocket,
                "message_creation_failed",
                str(e)
            )

    async def handle_edit_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle message edit request."""
        message_id = message.get("message_id")
        content = message.get("content", "").strip()

        if not message_id or not content:
            await self.connection_manager.send_error(
                websocket,
                "invalid_message",
                "Message ID and content are required"
            )
            return

        try:
            # Update message via service
            updated_message = await self.chat_service.update_message(
                message_id=message_id,
                content=content
            )

            # Broadcast update
            await self.connection_manager.send_message(
                updated_message.thread_id,
                {
                    "type": MessageType.MESSAGE_UPDATED,
                    "message": updated_message.to_dict()
                }
            )

        except Exception as e:
            logger.error(f"Message edit error: {str(e)}")
            await self.connection_manager.send_error(
                websocket,
                "edit_failed",
                str(e)
            )

    async def handle_delete_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle message deletion request."""
        message_id = message.get("message_id")

        if not message_id:
            await self.connection_manager.send_error(
                websocket,
                "invalid_message",
                "Message ID is required"
            )
            return

        try:
            # Delete message via service
            deleted_message = await self.chat_service.delete_message(message_id)

            # Broadcast deletion
            await self.connection_manager.send_message(
                deleted_message.thread_id,
                {
                    "type": MessageType.MESSAGE_DELETED,
                    "message_id": message_id,
                    "deleted_at": deleted_message.deleted_at.isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Message deletion error: {str(e)}")
            await self.connection_manager.send_error(
                websocket,
                "deletion_failed",
                str(e)
            )

    async def handle_typing_indicator(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle typing indicator update."""
        thread_id = message.get("thread_id")
        is_typing = message.get("is_typing", False)

        if not thread_id:
            return  # Silently ignore invalid typing indicators

        conn_info = self.connection_manager.connection_info.get(websocket)
        if conn_info:
            await self.connection_manager.broadcast_typing_indicator(
                thread_id=thread_id,
                user_id=conn_info["user_id"],
                is_typing=is_typing,
                exclude_websocket=websocket
            )

    async def handle_regenerate(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle response regeneration request."""
        message_id = message.get("message_id")
        user_message_id = message.get("user_message_id")

        if not message_id or not user_message_id:
            await self.connection_manager.send_error(
                websocket,
                "invalid_message",
                "Message IDs are required for regeneration"
            )
            return

        try:
            # Get the user message to regenerate from
            user_message = await self.chat_service.get_message(user_message_id)
            if not user_message:
                raise ValueError("User message not found")

            # Delete the old assistant response
            await self.chat_service.delete_message(message_id)

            # Broadcast deletion
            await self.connection_manager.send_message(
                user_message.thread_id,
                {
                    "type": MessageType.MESSAGE_DELETED,
                    "message_id": message_id,
                    "deleted_at": datetime.utcnow().isoformat()
                }
            )

            # Generate new response
            await self.generate_ai_response(user_message.thread_id, user_message)

        except Exception as e:
            logger.error(f"Regeneration error: {str(e)}")
            await self.connection_manager.send_error(
                websocket,
                "regeneration_failed",
                str(e)
            )

    async def generate_ai_response(self, thread_id: str, user_message: 'ChatMessage'):
        """Generate and stream AI response."""
        from services.ai_provider import ConversationManager

        try:
            # Get conversation context
            messages = await self.chat_service.get_thread_messages(thread_id, limit=50)

            # Create assistant message placeholder
            assistant_message = await self.chat_service.create_message(
                thread_id=thread_id,
                user_id=None,  # System message
                content="",
                is_user=False
            )

            # Notify clients of new assistant message
            await self.connection_manager.send_message(
                thread_id,
                {
                    "type": MessageType.ASSISTANT_MESSAGE_START,
                    "message_id": str(assistant_message.id),
                    "thread_id": thread_id,
                    "model": self.ai_provider.model if hasattr(self.ai_provider, 'model') else "unknown"
                }
            )

            # Prepare messages with context management
            conversation_manager = ConversationManager(self.ai_provider)
            ai_messages = conversation_manager.prepare_messages(messages)

            # Stream response
            full_response = ""
            chunk_index = 0

            async for chunk in self.ai_provider.complete(ai_messages, stream=True):
                full_response += chunk
                await self.connection_manager.send_stream_chunk(
                    thread_id,
                    str(assistant_message.id),
                    chunk,
                    chunk_index,
                    is_final=False
                )
                chunk_index += 1

            # Update message with full content
            await self.chat_service.update_message_content(
                assistant_message.id,
                content=full_response,
                token_count=self.ai_provider.count_tokens(full_response),
                model_used=getattr(self.ai_provider, 'model', 'unknown')
            )

            # Send final chunk to indicate completion
            await self.connection_manager.send_stream_chunk(
                thread_id,
                str(assistant_message.id),
                "",
                chunk_index,
                is_final=True
            )

            # Check if thread needs summarization
            await self.chat_service.check_and_trigger_summarization(thread_id)

        except asyncio.CancelledError:
            logger.info("AI response generation cancelled")
            raise
        except Exception as e:
            error_message = f"Failed to generate response: {str(e)}"
            logger.error(error_message)

            # Send error to clients
            await self.connection_manager.send_message(
                thread_id,
                {
                    "type": MessageType.ERROR,
                    "message_id": str(assistant_message.id) if 'assistant_message' in locals() else None,
                    "error_code": "ai_generation_failed",
                    "error": error_message
                }
            )

            # Update message with error if it was created
            if 'assistant_message' in locals():
                await self.chat_service.update_message_content(
                    assistant_message.id,
                    content=error_message,
                    metadata={"error": True, "error_message": str(e)}
                )
```

#### **services/summarization_service.py**
*Purpose: Automatic and manual thread summarization*

```python
"""
Summarization Service - Generates concise summaries of chat threads.
Manages automatic triggers and manual summarization requests.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import asyncio
import logging

from models.chat import ChatThread, ChatMessage, ChatSummary
from services.ai_provider import AIProvider, ConversationManager, AIMessage

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service for generating and managing chat summaries."""

    def __init__(
        self,
        ai_provider: AIProvider,
        db_session_factory: Any,
        conversation_manager: Optional[ConversationManager] = None
    ):
        self.ai_provider = ai_provider
        self.db_session_factory = db_session_factory
        self.conversation_manager = conversation_manager or ConversationManager(ai_provider)

        # Summarization thresholds
        self.message_threshold = 50  # Messages before auto-summary
        self.token_threshold = 10000  # Tokens before summary
        self.time_threshold = timedelta(hours=24)  # Time before summary

        # Configuration
        self.max_summary_length = 500  # characters
        self.max_topics = 5
        self.batch_size = 10  # Threads to process per batch

        # Background task
        self._auto_summary_task = None

    async def start_auto_summarization(self):
        """Start background auto-summarization task."""
        if self._auto_summary_task is None:
            self._auto_summary_task = asyncio.create_task(self._auto_summarization_loop())
            logger.info("Auto-summarization task started")

    async def stop_auto_summarization(self):
        """Stop background auto-summarization task."""
        if self._auto_summary_task:
            self._auto_summary_task.cancel()
            try:
                await self._auto_summary_task
            except asyncio.CancelledError:
                pass
            logger.info("Auto-summarization task stopped")

    async def should_summarize(self, thread: ChatThread, session: AsyncSession) -> bool:
        """Check if thread needs summarization based on thresholds."""
        # Check if already summarized recently
        if thread.last_summary_at:
            time_since_summary = datetime.utcnow() - thread.last_summary_at
            if time_since_summary < self.time_threshold:
                return False

        # Check message count threshold
        unsummarized_count = await self._count_unsummarized_messages(thread.id, session)
        if unsummarized_count >= self.message_threshold:
            logger.info(f"Thread {thread.id} exceeds message threshold: {unsummarized_count}")
            return True

        # Check token count threshold
        token_count = await self._estimate_thread_tokens(thread.id, session)
        if token_count >= self.token_threshold:
            logger.info(f"Thread {thread.id} exceeds token threshold: {token_count}")
            return True

        # Check time threshold
        if not thread.last_summary_at and thread.created_at:
            age = datetime.utcnow() - thread.created_at
            if age > self.time_threshold:
                logger.info(f"Thread {thread.id} exceeds age threshold: {age}")
                return True

        return False

    async def summarize_thread(
        self,
        thread_id: str,
        session: AsyncSession,
        force: bool = False
    ) -> Optional[ChatSummary]:
        """Generate summary for a thread."""
        # Get thread
        result = await session.execute(
            select(ChatThread).where(ChatThread.id == thread_id)
        )
        thread = result.scalar_one_or_none()

        if not thread:
            logger.error(f"Thread {thread_id} not found")
            return None

        # Check if summary needed
        if not force and not await self.should_summarize(thread, session):
            logger.info(f"Thread {thread_id} does not need summarization")
            return None

        # Get messages to summarize
        messages = await self._get_messages_for_summary(thread_id, session)
        if not messages:
            logger.info(f"No messages to summarize for thread {thread_id}")
            return None

        logger.info(f"Summarizing {len(messages)} messages for thread {thread_id}")

        try:
            # Generate summary
            summary_text = await self._generate_summary(messages)

            # Extract key topics
            key_topics = await self._extract_key_topics(messages)

            # Create summary record
            summary = ChatSummary(
                thread_id=thread_id,
                summary_text=summary_text,
                key_topics=key_topics,
                message_count=len(messages),
                start_message_id=messages[0].id,
                end_message_id=messages[-1].id,
                token_count=self.ai_provider.count_tokens(summary_text)
            )

            session.add(summary)

            # Update thread
            thread.last_summary_at = datetime.utcnow()
            thread.summary_count = (thread.summary_count or 0) + 1

            # Mark messages as summarized
            for msg in messages:
                msg.is_summarized = True
                msg.summary_id = summary.id

            await session.commit()

            logger.info(f"Created summary {summary.id} for thread {thread_id}")
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed for thread {thread_id}: {str(e)}")
            await session.rollback()
            return None

    async def get_thread_context(
        self,
        thread_id: str,
        session: AsyncSession,
        include_summaries: bool = True
    ) -> str:
        """Get context for thread including summaries for efficient token usage."""
        context_parts = []

        # Get summaries if requested
        if include_summaries:
            result = await session.execute(
                select(ChatSummary)
                .where(ChatSummary.thread_id == thread_id)
                .order_by(ChatSummary.created_at)
            )
            summaries = result.scalars().all()

            for summary in summaries:
                context_parts.append(
                    f"[Previous conversation summary]:\n{summary.summary_text}\n"
                    f"Key topics: {', '.join(summary.key_topics)}\n"
                )

        # Get recent unsummarized messages
        result = await session.execute(
            select(ChatMessage)
            .where(
                and_(
                    ChatMessage.thread_id == thread_id,
                    ChatMessage.is_summarized == False,
                    ChatMessage.is_deleted == False
                )
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(20)
        )
        recent_messages = result.scalars().all()

        # Add recent messages in chronological order
        for msg in reversed(recent_messages):
            role = "User" if msg.is_user else "Assistant"
            context_parts.append(f"{role}: {msg.content}\n")

        return "\n".join(context_parts)

    async def _generate_summary(self, messages: List[ChatMessage]) -> str:
        """Generate summary text from messages."""
        summary_prompt = f"""
You are a conversation summarizer. Create a concise summary of the following conversation.

Guidelines:
- Maximum {self.max_summary_length} characters
- Focus on key topics, decisions, and outcomes
- Use bullet points for clarity
- Be factual and comprehensive
- Highlight any action items or important conclusions

Summarize the conversation below:
"""

        # Prepare messages for summarization
        summary_messages = [
            AIMessage(role="system", content=summary_prompt)
        ]

        # Add conversation messages (with truncation for long conversations)
        total_chars = 0
        max_chars = 10000  # Limit input to prevent token overflow

        for msg in messages:
            if total_chars > max_chars:
                break

            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            summary_messages.append(
                AIMessage(
                    role="user" if msg.is_user else "assistant",
                    content=content
                )
            )
            total_chars += len(content)

        try:
            response = await self.ai_provider.complete(
                summary_messages,
                stream=False,
                temperature=0.3,  # Lower temperature for factual summary
                max_tokens=200
            )

            # Ensure summary fits within length limit
            summary = response.content
            if len(summary) > self.max_summary_length:
                summary = summary[:self.max_summary_length-3] + "..."

            return summary

        except Exception as e:
            logger.error(f"AI summary generation failed: {str(e)}")
            # Fallback summary
            return f"Conversation with {len(messages)} messages discussing various topics."

    async def _extract_key_topics(self, messages: List[ChatMessage]) -> List[str]:
        """Extract key topics from messages using AI."""
        topic_prompt = """
Extract 3-5 key topics or themes from this conversation.
Return only the topics as a comma-separated list.
Be concise and specific. Focus on the main subjects discussed.
"""

        # Prepare messages for topic extraction
        topic_messages = [
            AIMessage(role="system", content=topic_prompt)
        ]

        # Sample messages for topic extraction
        sample_size = min(20, len(messages))
        sample_messages = messages[:sample_size]

        for msg in sample_messages:
            topic_messages.append(
                AIMessage(
                    role="user" if msg.is_user else "assistant",
                    content=msg.content[:200]  # Truncate for efficiency
                )
            )

        try:
            response = await self.ai_provider.complete(
                topic_messages,
                stream=False,
                temperature=0.3,
                max_tokens=100
            )

            # Parse topics
            topics_text = response.content.strip()
            topics = [
                topic.strip()
                for topic in topics_text.split(",")
                if topic.strip()
            ]

            # Limit to max topics
            return topics[:self.max_topics]

        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            return ["General discussion"]

    async def _count_unsummarized_messages(self, thread_id: str, session: AsyncSession) -> int:
        """Count messages not yet summarized."""
        result = await session.execute(
            select(func.count(ChatMessage.id))
            .where(
                and_(
                    ChatMessage.thread_id == thread_id,
                    ChatMessage.is_summarized == False,
                    ChatMessage.is_deleted == False
                )
            )
        )
        return result.scalar() or 0

    async def _estimate_thread_tokens(self, thread_id: str, session: AsyncSession) -> int:
        """Estimate total tokens in unsummarized messages."""
        result = await session.execute(
            select(ChatMessage)
            .where(
                and_(
                    ChatMessage.thread_id == thread_id,
                    ChatMessage.is_summarized == False,
                    ChatMessage.is_deleted == False
                )
            )
        )
        messages = result.scalars().all()

        total_tokens = sum(
            msg.token_count or self.ai_provider.count_tokens(msg.content)
            for msg in messages
        )

        return total_tokens

    async def _get_messages_for_summary(
        self,
        thread_id: str,
        session: AsyncSession
    ) -> List[ChatMessage]:
        """Get messages that need summarization."""
        result = await session.execute(
            select(ChatMessage)
            .where(
                and_(
                    ChatMessage.thread_id == thread_id,
                    ChatMessage.is_summarized == False,
                    ChatMessage.is_deleted == False
                )
            )
            .order_by(ChatMessage.created_at)
        )
        return result.scalars().all()

    async def _auto_summarization_loop(self):
        """Background task to automatically summarize threads."""
        while True:
            try:
                async with self.db_session_factory() as session:
                    # Find threads needing summarization
                    cutoff_time = datetime.utcnow() - self.time_threshold

                    result = await session.execute(
                        select(ChatThread)
                        .where(
                            and_(
                                ChatThread.is_archived == False,
                                # Either never summarized or last summary is old
                                (ChatThread.last_summary_at == None) |
                                (ChatThread.last_summary_at < cutoff_time),
                                # Has recent activity
                                ChatThread.last_activity_at > cutoff_time
                            )
                        )
                        .limit(self.batch_size)
                    )
                    threads = result.scalars().all()

                    for thread in threads:
                        if await self.should_summarize(thread, session):
                            logger.info(f"Auto-summarizing thread {thread.id}")
                            await self.summarize_thread(thread.id, session)

                            # Brief pause between summaries
                            await asyncio.sleep(1)

                # Run every hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-summarization error: {str(e)}")
                # Continue after error
                await asyncio.sleep(60)
```

### API Endpoints

#### **routes/chat_routes.py**
*Purpose: REST API endpoints for chat functionality*

```python
"""
Chat API Routes - RESTful endpoints for chat operations.
Handles thread management, message history, and summarization.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from models.chat import ChatThread, ChatMessage, ChatSummary
from models.project import Project
from models.user import User
from schemas.chat import (
    ThreadCreate, ThreadUpdate, ThreadResponse, ThreadListResponse,
    MessageCreate, MessageUpdate, MessageResponse, MessageListResponse,
    SummaryResponse, SummarizeRequest
)
from core.dependencies import get_current_user, get_db
from services.websocket_manager import connection_manager, MessageHandler
from services.chat_service import ChatService
from services.ai_provider import AIProviderFactory
from services.summarization_service import SummarizationService
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

# Initialize services
ai_provider = AIProviderFactory.create(
    settings.AI_PROVIDER,
    {
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.AI_MODEL,
        "max_tokens": settings.AI_MAX_TOKENS
    }
)

# Services will be initialized per request with proper DB session


@router.get("/threads", response_model=ThreadListResponse)
async def list_threads(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    include_archived: bool = Query(False, description="Include archived threads"),
    sort_by: str = Query("activity", regex="^(created|updated|activity)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    search: Optional[str] = Query(None, description="Search in messages"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ThreadListResponse:
    """List chat threads with filtering and pagination."""

    # Build query
    query = select(ChatThread).where(ChatThread.user_id == current_user.id)

    # Apply filters
    if project_id:
        query = query.where(ChatThread.project_id == project_id)

    if not include_archived:
        query = query.where(ChatThread.is_archived == False)

    # Search in messages if requested
    if search:
        # Subquery to find threads with matching messages
        message_subquery = (
            select(ChatMessage.thread_id)
            .where(
                and_(
                    ChatMessage.is_deleted == False,
                    ChatMessage.content.ilike(f"%{search}%")
                )
            )
            .distinct()
        )
        query = query.where(ChatThread.id.in_(message_subquery))

    # Apply sorting
    sort_column = {
        "created": ChatThread.created_at,
        "updated": ChatThread.updated_at,
        "activity": ChatThread.last_activity_at
    }[sort_by]

    query = query.order_by(
        desc(sort_column) if order == "desc" else sort_column
    )

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    query = query.limit(limit).offset(offset)

    # Execute query
    result = await db.execute(query)
    threads = result.scalars().all()

    # Get last messages for each thread
    thread_responses = []
    for thread in threads:
        # Get last message
        last_msg_result = await db.execute(
            select(ChatMessage)
            .where(
                and_(
                    ChatMessage.thread_id == thread.id,
                    ChatMessage.is_deleted == False
                )
            )
            .order_by(desc(ChatMessage.created_at))
            .limit(1)
        )
        last_message = last_msg_result.scalar_one_or_none()

        response = ThreadResponse(
            id=str(thread.id),
            project_id=str(thread.project_id),
            title=thread.title,
            created_at=thread.created_at,
            last_activity_at=thread.last_activity_at,
            message_count=thread.message_count,
            is_summarized=bool(thread.last_summary_at),
            summary_count=thread.summary_count,
            is_archived=thread.is_archived,
            last_message={
                "content": last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content,
                "is_user": last_message.is_user,
                "created_at": last_message.created_at
            } if last_message else None
        )
        thread_responses.append(response)

    return ThreadListResponse(
        threads=thread_responses,
        total=total,
        has_more=(offset + limit) < total
    )


@router.post("/threads", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    thread_data: ThreadCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ThreadResponse:
    """Create a new chat thread."""

    # Verify project exists and belongs to user
    project_result = await db.execute(
        select(Project).where(
            and_(
                Project.id == thread_data.project_id,
                Project.user_id == current_user.id
            )
        )
    )
    project = project_result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Create thread
    thread = ChatThread(
        project_id=thread_data.project_id,
        user_id=current_user.id,
        title=thread_data.title or "New Chat"
    )
    db.add(thread)

    # Update project last chat timestamp
    project.last_chat_at = datetime.utcnow()

    # Create initial message if provided
    if thread_data.initial_message:
        chat_service = ChatService(db, ai_provider)
        message = await chat_service.create_message(
            thread_id=thread.id,
            user_id=current_user.id,
            content=thread_data.initial_message,
            is_user=True
        )

        # Generate AI response asynchronously
        # This will be handled by WebSocket connection

    await db.commit()
    await db.refresh(thread)

    return ThreadResponse(
        id=str(thread.id),
        project_id=str(thread.project_id),
        title=thread.title,
        created_at=thread.created_at,
        last_activity_at=thread.last_activity_at,
        message_count=thread.message_count,
        is_summarized=False,
        summary_count=0,
        is_archived=False,
        websocket_url=f"/ws/chat/{thread.id}"
    )


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ThreadResponse:
    """Get a specific thread."""

    result = await db.execute(
        select(ChatThread).where(
            and_(
                ChatThread.id == thread_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    thread = result.scalar_one_or_none()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )

    return ThreadResponse(
        id=str(thread.id),
        project_id=str(thread.project_id),
        title=thread.title,
        created_at=thread.created_at,
        last_activity_at=thread.last_activity_at,
        message_count=thread.message_count,
        is_summarized=bool(thread.last_summary_at),
        summary_count=thread.summary_count,
        is_archived=thread.is_archived,
        metadata=thread.metadata
    )


@router.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: str,
    thread_update: ThreadUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ThreadResponse:
    """Update thread properties."""

    result = await db.execute(
        select(ChatThread).where(
            and_(
                ChatThread.id == thread_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    thread = result.scalar_one_or_none()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )

    # Update fields
    if thread_update.title is not None:
        thread.title = thread_update.title

    if thread_update.is_archived is not None:
        thread.is_archived = thread_update.is_archived
        if thread_update.is_archived:
            thread.archived_at = datetime.utcnow()
        else:
            thread.archived_at = None

    thread.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(thread)

    return ThreadResponse(
        id=str(thread.id),
        project_id=str(thread.project_id),
        title=thread.title,
        created_at=thread.created_at,
        last_activity_at=thread.last_activity_at,
        message_count=thread.message_count,
        is_summarized=bool(thread.last_summary_at),
        summary_count=thread.summary_count,
        is_archived=thread.is_archived
    )


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> None:
    """Delete a thread and all its messages."""

    result = await db.execute(
        select(ChatThread).where(
            and_(
                ChatThread.id == thread_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    thread = result.scalar_one_or_none()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )

    # Delete thread (cascade will handle messages and summaries)
    await db.delete(thread)
    await db.commit()


@router.get("/threads/{thread_id}/messages", response_model=MessageListResponse)
async def get_thread_messages(
    thread_id: str,
    limit: int = Query(50, ge=1, le=200),
    before: Optional[str] = Query(None, description="Message ID for pagination"),
    after: Optional[str] = Query(None, description="Message ID for pagination"),
    include_deleted: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> MessageListResponse:
    """Get messages for a thread with pagination."""

    # Verify thread access
    thread_result = await db.execute(
        select(ChatThread).where(
            and_(
                ChatThread.id == thread_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    thread = thread_result.scalar_one_or_none()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )

    # Build message query
    query = select(ChatMessage).where(ChatMessage.thread_id == thread_id)

    if not include_deleted:
        query = query.where(ChatMessage.is_deleted == False)

    # Apply pagination
    if before:
        # Get messages before the specified message
        before_msg_result = await db.execute(
            select(ChatMessage.created_at).where(ChatMessage.id == before)
        )
        before_time = before_msg_result.scalar_one_or_none()
        if before_time:
            query = query.where(ChatMessage.created_at < before_time)

    if after:
        # Get messages after the specified message
        after_msg_result = await db.execute(
            select(ChatMessage.created_at).where(ChatMessage.id == after)
        )
        after_time = after_msg_result.scalar_one_or_none()
        if after_time:
            query = query.where(ChatMessage.created_at > after_time)

    # Order and limit
    query = query.order_by(desc(ChatMessage.created_at)).limit(limit + 1)

    # Execute query
    result = await db.execute(query)
    messages = result.scalars().all()

    # Check if there are more messages
    has_more = len(messages) > limit
    if has_more:
        messages = messages[:limit]

    # Reverse to get chronological order
    messages.reverse()

    # Convert to response format
    message_responses = [
        MessageResponse(
            id=str(msg.id),
            thread_id=str(msg.thread_id),
            content=msg.content,
            is_user=msg.is_user,
            created_at=msg.created_at,
            is_edited=msg.is_edited,
            edited_at=msg.edited_at,
            is_deleted=msg.is_deleted,
            token_count=msg.token_count,
            model_used=msg.model_used,
            metadata=msg.metadata
        )
        for msg in messages
    ]

    # Calculate total tokens
    total_tokens = sum(msg.token_count or 0 for msg in messages)

    return MessageListResponse(
        messages=message_responses,
        has_more=has_more,
        total_tokens=total_tokens
    )


@router.patch("/messages/{message_id}", response_model=MessageResponse)
async def update_message(
    message_id: str,
    message_update: MessageUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> MessageResponse:
    """Edit a message (user messages only)."""

    # Get message with thread info
    result = await db.execute(
        select(ChatMessage)
        .join(ChatThread)
        .where(
            and_(
                ChatMessage.id == message_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )

    # Only allow editing user messages
    if not message.is_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only edit user messages"
        )

    # Update message
    chat_service = ChatService(db, ai_provider)
    updated_message = await chat_service.update_message(
        message_id=message_id,
        content=message_update.content
    )

    return MessageResponse(
        id=str(updated_message.id),
        thread_id=str(updated_message.thread_id),
        content=updated_message.content,
        is_user=updated_message.is_user,
        created_at=updated_message.created_at,
        is_edited=updated_message.is_edited,
        edited_at=updated_message.edited_at,
        is_deleted=updated_message.is_deleted,
        token_count=updated_message.token_count,
        edit_history=updated_message.edit_history
    )


@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> None:
    """Soft delete a message."""

    # Get message with thread info
    result = await db.execute(
        select(ChatMessage)
        .join(ChatThread)
        .where(
            and_(
                ChatMessage.id == message_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )

    # Soft delete
    chat_service = ChatService(db, ai_provider)
    await chat_service.delete_message(message_id)


@router.post("/threads/{thread_id}/summarize", response_model=SummaryResponse, status_code=status.HTTP_202_ACCEPTED)
async def summarize_thread(
    thread_id: str,
    request: SummarizeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> SummaryResponse:
    """Generate a summary for the thread."""

    # Verify thread access
    result = await db.execute(
        select(ChatThread).where(
            and_(
                ChatThread.id == thread_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    thread = result.scalar_one_or_none()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )

    # Create summarization service
    summarization_service = SummarizationService(
        ai_provider=ai_provider,
        db_session_factory=lambda: db
    )

    # Generate summary
    summary = await summarization_service.summarize_thread(
        thread_id=thread_id,
        session=db,
        force=request.force
    )

    if not summary:
        # Check if thread doesn't need summarization
        if not request.force and not await summarization_service.should_summarize(thread, db):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Thread does not meet summarization criteria"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate summary"
            )

    return SummaryResponse(
        id=str(summary.id),
        thread_id=str(summary.thread_id),
        summary_text=summary.summary_text,
        key_topics=summary.key_topics,
        created_at=summary.created_at,
        message_count=summary.message_count,
        status="completed"
    )


@router.get("/threads/{thread_id}/summary", response_model=SummaryResponse)
async def get_latest_summary(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> SummaryResponse:
    """Get the latest summary for a thread."""

    # Verify thread access
    thread_result = await db.execute(
        select(ChatThread).where(
            and_(
                ChatThread.id == thread_id,
                ChatThread.user_id == current_user.id
            )
        )
    )
    thread = thread_result.scalar_one_or_none()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )

    # Get latest summary
    result = await db.execute(
        select(ChatSummary)
        .where(ChatSummary.thread_id == thread_id)
        .order_by(desc(ChatSummary.created_at))
        .limit(1)
    )
    summary = result.scalar_one_or_none()

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No summary found for this thread"
        )

    return SummaryResponse(
        id=str(summary.id),
        thread_id=str(summary.thread_id),
        summary_text=summary.summary_text,
        key_topics=summary.key_topics,
        created_at=summary.created_at,
        message_count=summary.message_count,
        status="completed"
    )


# WebSocket endpoint
@router.websocket("/ws/chat/{thread_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    thread_id: str,
    db: AsyncSession = Depends(get_db)
):
    """WebSocket endpoint for real-time chat."""

    # Extract token from query params or headers
    token = websocket.query_params.get("token") or websocket.headers.get("Authorization", "").replace("Bearer ", "")

    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication")
        return

    # Authenticate user
    try:
        # This would use your actual token validation
        from core.auth import decode_access_token
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise ValueError("Invalid token")

        # Verify thread access
        result = await db.execute(
            select(ChatThread).where(
                and_(
                    ChatThread.id == thread_id,
                    ChatThread.user_id == user_id
                )
            )
        )
        thread = result.scalar_one_or_none()

        if not thread:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Thread not found")
            return

    except Exception as e:
        logger.error(f"WebSocket authentication failed: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    # Accept connection
    connection_id = await connection_manager.connect(websocket, thread_id, user_id)

    # Create services
    chat_service = ChatService(db, ai_provider)
    message_handler = MessageHandler(connection_manager, chat_service, ai_provider)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Handle message
            await message_handler.handle_message(websocket, data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await connection_manager.disconnect(websocket)


# Initialize background tasks on startup
@router.on_event("startup")
async def startup_event():
    """Initialize background tasks."""
    await connection_manager.start_monitoring()

    # Initialize summarization service with proper session factory
    from core.database import async_session_maker
    summarization_service = SummarizationService(
        ai_provider=ai_provider,
        db_session_factory=async_session_maker
    )
    await summarization_service.start_

### Backend Services (continued)

#### **services/chat_service.py**
*Purpose: Business logic for chat operations*

```python
"""
Chat Service - Core business logic for chat operations.
Handles message creation, thread management, and AI interactions.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update
import logging
import uuid

from models.chat import ChatThread, ChatMessage, ChatSummary
from services.ai_provider import AIProvider
from services.summarization_service import SummarizationService

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat operations."""

    def __init__(self, db: AsyncSession, ai_provider: AIProvider):
        self.db = db
        self.ai_provider = ai_provider

    async def create_message(
        self,
        thread_id: str,
        user_id: Optional[str],
        content: str,
        is_user: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Create a new message in a thread."""

        # Validate thread exists
        thread_result = await self.db.execute(
            select(ChatThread).where(ChatThread.id == thread_id)
        )
        thread = thread_result.scalar_one_or_none()

        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        # Count tokens
        token_count = self.ai_provider.count_tokens(content)

        # Create message
        message = ChatMessage(
            thread_id=thread_id,
            user_id=user_id if is_user else None,
            content=content,
            is_user=is_user,
            token_count=token_count,
            metadata=metadata or {}
        )

        self.db.add(message)

        # Update thread statistics
        thread.message_count += 1
        thread.total_tokens += token_count
        thread.last_activity_at = datetime.utcnow()
        thread.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(message)

        logger.info(f"Created message {message.id} in thread {thread_id}")
        return message

    async def get_message(self, message_id: str) -> Optional[ChatMessage]:
        """Get a message by ID."""
        result = await self.db.execute(
            select(ChatMessage).where(ChatMessage.id == message_id)
        )
        return result.scalar_one_or_none()

    async def update_message(
        self,
        message_id: str,
        content: str
    ) -> ChatMessage:
        """Update message content and track edit history."""

        message = await self.get_message(message_id)
        if not message:
            raise ValueError(f"Message {message_id} not found")

        if not message.is_user:
            raise ValueError("Can only edit user messages")

        # Store edit history
        message.add_edit_history(message.content)

        # Update content
        message.content = content
        message.token_count = self.ai_provider.count_tokens(content)

        # Update thread activity
        thread_result = await self.db.execute(
            select(ChatThread).where(ChatThread.id == message.thread_id)
        )
        thread = thread_result.scalar_one()
        thread.last_activity_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(message)

        logger.info(f"Updated message {message_id}")
        return message

    async def update_message_content(
        self,
        message_id: str,
        content: str,
        token_count: Optional[int] = None,
        model_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Update message content (for AI responses)."""

        message = await self.get_message(message_id)
        if not message:
            raise ValueError(f"Message {message_id} not found")

        message.content = content
        message.token_count = token_count or self.ai_provider.count_tokens(content)

        if model_used:
            message.model_used = model_used

        if metadata:
            message.metadata.update(metadata)

        await self.db.commit()
        await self.db.refresh(message)

        return message

    async def delete_message(self, message_id: str) -> ChatMessage:
        """Soft delete a message."""

        message = await self.get_message(message_id)
        if not message:
            raise ValueError(f"Message {message_id} not found")

        message.is_deleted = True
        message.deleted_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(message)

        logger.info(f"Deleted message {message_id}")
        return message

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 50,
        include_deleted: bool = False
    ) -> List[ChatMessage]:
        """Get messages for a thread."""

        query = select(ChatMessage).where(ChatMessage.thread_id == thread_id)

        if not include_deleted:
            query = query.where(ChatMessage.is_deleted == False)

        query = query.order_by(ChatMessage.created_at).limit(limit)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def create_thread(
        self,
        project_id: str,
        user_id: str,
        title: Optional[str] = None
    ) -> ChatThread:
        """Create a new chat thread."""

        thread = ChatThread(
            project_id=project_id,
            user_id=user_id,
            title=title or "New Chat"
        )

        self.db.add(thread)
        await self.db.commit()
        await self.db.refresh(thread)

        logger.info(f"Created thread {thread.id} for project {project_id}")
        return thread

    async def update_thread_title(
        self,
        thread_id: str,
        title: str
    ) -> ChatThread:
        """Update thread title."""

        result = await self.db.execute(
            select(ChatThread).where(ChatThread.id == thread_id)
        )
        thread = result.scalar_one_or_none()

        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        thread.title = title
        thread.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(thread)

        return thread

    async def archive_thread(
        self,
        thread_id: str,
        archive: bool = True
    ) -> ChatThread:
        """Archive or unarchive a thread."""

        result = await self.db.execute(
            select(ChatThread).where(ChatThread.id == thread_id)
        )
        thread = result.scalar_one_or_none()

        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        thread.is_archived = archive
        thread.archived_at = datetime.utcnow() if archive else None
        thread.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(thread)

        logger.info(f"{'Archived' if archive else 'Unarchived'} thread {thread_id}")
        return thread

    async def get_recent_threads(
        self,
        user_id: str,
        limit: int = 10,
        project_id: Optional[str] = None
    ) -> List[ChatThread]:
        """Get recent threads for a user."""

        query = select(ChatThread).where(
            and_(
                ChatThread.user_id == user_id,
                ChatThread.is_archived == False
            )
        )

        if project_id:
            query = query.where(ChatThread.project_id == project_id)

        query = query.order_by(ChatThread.last_activity_at.desc()).limit(limit)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def check_and_trigger_summarization(
        self,
        thread_id: str
    ) -> Optional[ChatSummary]:
        """Check if thread needs summarization and trigger if needed."""

        # Get thread
        result = await self.db.execute(
            select(ChatThread).where(ChatThread.id == thread_id)
        )
        thread = result.scalar_one_or_none()

        if not thread:
            return None

        # Create summarization service
        from core.database import async_session_maker
        summarization_service = SummarizationService(
            ai_provider=self.ai_provider,
            db_session_factory=async_session_maker
        )

        # Check if summarization is needed
        if await summarization_service.should_summarize(thread, self.db):
            logger.info(f"Triggering summarization for thread {thread_id}")

            # Run summarization in background
            import asyncio
            asyncio.create_task(
                self._background_summarize(thread_id, summarization_service)
            )

        return None

    async def _background_summarize(
        self,
        thread_id: str,
        summarization_service: SummarizationService
    ):
        """Run summarization in background."""
        try:
            async with self.db.begin():
                summary = await summarization_service.summarize_thread(
                    thread_id=thread_id,
                    session=self.db
                )

                if summary:
                    # Notify via WebSocket if needed
                    from services.websocket_manager import connection_manager
                    await connection_manager.send_message(
                        thread_id,
                        {
                            "type": "summary_available",
                            "thread_id": thread_id,
                            "summary_id": str(summary.id),
                            "message_count": summary.message_count
                        }
                    )
        except Exception as e:
            logger.error(f"Background summarization failed: {str(e)}")

    async def generate_thread_title(
        self,
        thread_id: str
    ) -> str:
        """Generate a title for a thread based on initial messages."""

        # Get first few messages
        messages = await self.get_thread_messages(thread_id, limit=5)

        if not messages:
            return "New Chat"

        # Create title generation prompt
        from services.ai_provider import AIMessage

        prompt = """Generate a concise, descriptive title (max 50 characters) for this conversation.
Focus on the main topic or question. Do not use quotes or special formatting.
Return only the title text."""

        title_messages = [
            AIMessage(role="system", content=prompt)
        ]

        # Add conversation context
        for msg in messages[:3]:  # First 3 messages
            title_messages.append(
                AIMessage(
                    role="user" if msg.is_user else "assistant",
                    content=msg.content[:200]  # Truncate
                )
            )

        try:
            response = await self.ai_provider.complete(
                title_messages,
                stream=False,
                temperature=0.3,
                max_tokens=20
            )

            title = response.content.strip()

            # Ensure title length
            if len(title) > 50:
                title = title[:47] + "..."

            # Update thread title
            await self.update_thread_title(thread_id, title)

            return title

        except Exception as e:
            logger.error(f"Title generation failed: {str(e)}")
            return "Chat about " + messages[0].content[:30] + "..."
```

### API Schemas

#### **schemas/chat.py**
*Purpose: Pydantic models for API request/response validation*

```python
"""
Chat API Schemas - Request and response models for chat endpoints.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ThreadCreate(BaseModel):
    """Create a new chat thread."""
    project_id: str = Field(..., description="Project ID")
    title: Optional[str] = Field(None, max_length=200)
    initial_message: Optional[str] = Field(None, description="Optional first message")

    @validator('project_id')
    def validate_uuid(cls, v):
        try:
            import uuid
            uuid.UUID(v)
        except ValueError:
            raise ValueError('Invalid UUID format')
        return v


class ThreadUpdate(BaseModel):
    """Update thread properties."""
    title: Optional[str] = Field(None, max_length=200)
    is_archived: Optional[bool] = None


class ThreadResponse(BaseModel):
    """Thread response model."""
    id: str
    project_id: str
    title: str
    created_at: datetime
    last_activity_at: datetime
    message_count: int
    is_summarized: bool
    summary_count: int
    is_archived: bool
    last_message: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = {}
    websocket_url: Optional[str] = None

    class Config:
        from_attributes = True


class ThreadListResponse(BaseModel):
    """List of threads with pagination."""
    threads: List[ThreadResponse]
    total: int
    has_more: bool


class MessageCreate(BaseModel):
    """Create a new message."""
    content: str = Field(..., min_length=1, max_length=50000)
    attachments: Optional[List[str]] = []  # Future: file attachments


class MessageUpdate(BaseModel):
    """Update message content."""
    content: str = Field(..., min_length=1, max_length=50000)


class MessageResponse(BaseModel):
    """Message response model."""
    id: str
    thread_id: str
    content: str
    is_user: bool
    created_at: datetime
    is_edited: bool
    edited_at: Optional[datetime]
    is_deleted: bool
    token_count: int
    model_used: Optional[str]
    metadata: Dict[str, Any] = {}
    edit_history: Optional[List[Dict[str, Any]]] = []

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """List of messages with metadata."""
    messages: List[MessageResponse]
    has_more: bool
    total_tokens: int


class SummarizeRequest(BaseModel):
    """Request to summarize a thread."""
    force: bool = Field(False, description="Force summarization even if recent summary exists")


class SummaryResponse(BaseModel):
    """Summary response model."""
    id: str
    thread_id: str
    summary_text: str
    key_topics: List[str]
    created_at: datetime
    message_count: int
    status: str = "completed"

    class Config:
        from_attributes = True


class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""
    type: str
    timestamp: Optional[datetime] = None

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        if self.timestamp:
            d['timestamp'] = self.timestamp.isoformat()
        return d


class ErrorCode(str, Enum):
    """WebSocket error codes."""
    INVALID_MESSAGE = "invalid_message"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED = "unauthorized"
    THREAD_NOT_FOUND = "thread_not_found"
    MESSAGE_TOO_LONG = "message_too_long"
    AI_PROVIDER_ERROR = "ai_provider_error"
    INSUFFICIENT_TOKENS = "insufficient_tokens"
```

### Frontend Components

#### **src/components/chat/ChatView.tsx**
*Purpose: Main chat interface container*

```typescript
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useChatStore } from '../../stores/chatStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useAuth } from '../../hooks/useAuth';
import MessageBubble from './MessageBubble';
import ChatInputBar from './ChatInputBar';
import ConnectionStatus from './ConnectionStatus';
import { MessageType, WebSocketStatus } from '../../types/websocket';
import { ChatMessage, ChatThread } from '../../types/chat';
import { VirtualList } from '../ui/VirtualList';
import { Loader2, AlertCircle } from 'lucide-react';
import styles from './ChatView.module.css';

interface ChatViewProps {
  projectId: string;
  threadId?: string;
  onThreadChange?: (threadId: string) => void;
}

const ChatView: React.FC<ChatViewProps> = ({ projectId, threadId, onThreadChange }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const {
    threads,
    messages,
    activeThreadId,
    isLoadingThread,
    streamingMessage,
    loadThread,
    createThread,
    sendMessage,
    editMessage,
    deleteMessage,
    regenerateResponse,
  } = useChatStore();

  const [isAutoScroll, setIsAutoScroll] = useState(true);
  const [showNewMessageIndicator, setShowNewMessageIndicator] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Initialize WebSocket connection
  const {
    status: wsStatus,
    sendMessage: wsSendMessage,
    reconnect,
  } = useWebSocket({
    url: threadId ? `/ws/chat/${threadId}` : null,
    onMessage: handleWebSocketMessage,
    reconnectAttempts: 5,
    heartbeatInterval: 30000,
  });

  // Load thread on mount or change
  useEffect(() => {
    if (threadId && threadId !== activeThreadId) {
      loadThread(threadId);
    }
  }, [threadId, activeThreadId, loadThread]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (isAutoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.get(activeThreadId || '')?.length, streamingMessage, isAutoScroll]);

  // Handle WebSocket messages
  function handleWebSocketMessage(message: any) {
    switch (message.type) {
      case MessageType.NEW_MESSAGE:
        // Message already added via store action
        break;
      case MessageType.ASSISTANT_MESSAGE_START:
        useChatStore.getState().startStreaming(message.message_id);
        break;
      case MessageType.STREAM_CHUNK:
        useChatStore.getState().addStreamChunk(message.message_id, message.chunk);
        break;
      case MessageType.MESSAGE_UPDATED:
        useChatStore.getState().updateMessageInStore(message.message);
        break;
      case MessageType.MESSAGE_DELETED:
        useChatStore.getState().removeMessageFromStore(message.message_id);
        break;
      case MessageType.ERROR:
        console.error('WebSocket error:', message.error);
        // Handle error appropriately
        break;
    }
  }

  // Handle scroll events
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setIsAutoScroll(isNearBottom);

    if (!isNearBottom && !showNewMessageIndicator) {
      setShowNewMessageIndicator(true);
    } else if (isNearBottom && showNewMessageIndicator) {
      setShowNewMessageIndicator(false);
    }
  }, [showNewMessageIndicator]);

  // Create new thread if needed
  const ensureThread = async (): Promise<string> => {
    if (activeThreadId) return activeThreadId;

    const thread = await createThread(projectId);
    if (onThreadChange) {
      onThreadChange(thread.id);
    }
    navigate(`/projects/${projectId}/chat/${thread.id}`);
    return thread.id;
  };

  // Send message handler
  const handleSendMessage = async (content: string) => {
    const threadId = await ensureThread();
    await sendMessage(content);

    // Send via WebSocket
    wsSendMessage({
      type: MessageType.SEND_MESSAGE,
      thread_id: threadId,
      content,
    });
  };

  // Edit message handler
  const handleEditMessage = async (messageId: string, content: string) => {
    await editMessage(messageId, content);

    // Send via WebSocket
    wsSendMessage({
      type: MessageType.EDIT_MESSAGE,
      message_id: messageId,
      content,
    });
  };

  // Delete message handler
  const handleDeleteMessage = async (messageId: string) => {
    await deleteMessage(messageId);

    // Send via WebSocket
    wsSendMessage({
      type: MessageType.DELETE_MESSAGE,
      message_id: messageId,
    });
  };

  // Regenerate response handler
  const handleRegenerateResponse = async (messageId: string, userMessageId: string) => {
    await regenerateResponse(messageId);

    // Send via WebSocket
    wsSendMessage({
      type: MessageType.REGENERATE,
      message_id: messageId,
      user_message_id: userMessageId,
    });
  };

  // Handle typing indicator
  const handleTyping = useCallback(() => {
    if (activeThreadId) {
      wsSendMessage({
        type: MessageType.TYPING_INDICATOR,
        thread_id: activeThreadId,
        is_typing: true,
      });
    }
  }, [activeThreadId, wsSendMessage]);

  // Get current thread
  const currentThread = activeThreadId ? threads.get(activeThreadId) : null;
  const currentMessages = activeThreadId ? messages.get(activeThreadId) || [] : [];

  // Render loading state
  if (isLoadingThread) {
    return (
      <div className={styles.loadingContainer}>
        <Loader2 className={styles.spinner} />
        <p>Loading conversation...</p>
      </div>
    );
  }

  // Render error state
  if (wsStatus === WebSocketStatus.ERROR) {
    return (
      <div className={styles.errorContainer}>
        <AlertCircle className={styles.errorIcon} />
        <h3>Connection Error</h3>
        <p>Unable to connect to chat service</p>
        <button onClick={reconnect} className={styles.retryButton}>
          Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div className={styles.chatView}>
      {/* Connection Status */}
      {wsStatus !== WebSocketStatus.CONNECTED && (
        <ConnectionStatus status={wsStatus} onReconnect={reconnect} />
      )}

      {/* Header */}
      {currentThread && (
        <div className={styles.header}>
          <h2>{currentThread.title}</h2>
          <div className={styles.headerInfo}>
            <span>{currentThread.message_count} messages</span>
            {currentThread.is_summarized && (
              <span className={styles.summarizedBadge}>Summarized</span>
            )}
          </div>
        </div>
      )}

      {/* Messages Container */}
      <div
        ref={scrollContainerRef}
        className={styles.messagesContainer}
        onScroll={handleScroll}
      >
        {currentMessages.length === 0 ? (
          <div className={styles.emptyState}>
            <p>Start a conversation</p>
          </div>
        ) : (
          <>
            {/* Virtual list for performance */}
            <VirtualList
              items={currentMessages}
              height={window.innerHeight - 200}
              itemHeight={(index) => estimateMessageHeight(currentMessages[index])}
              renderItem={(message, index) => {
                const isLast = index === currentMessages.length - 1;
                const isStreaming = streamingMessage?.id === message.id;
                const previousMessage = index > 0 ? currentMessages[index - 1] : null;
                const showAvatar = !previousMessage || previousMessage.is_user !== message.is_user;

                return (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    isStreaming={isStreaming}
                    isLast={isLast}
                    showAvatar={showAvatar}
                    onEdit={(id) => {
                      const newContent = prompt('Edit message:', message.content);
                      if (newContent && newContent !== message.content) {
                        handleEditMessage(id, newContent);
                      }
                    }}
                    onDelete={handleDeleteMessage}
                    onRegenerate={() => {
                      const userMsgIndex = currentMessages.findIndex(m => m.id === message.id) - 1;
                      if (userMsgIndex >= 0) {
                        handleRegenerateResponse(message.id, currentMessages[userMsgIndex].id);
                      }
                    }}
                    onCopy={() => {
                      navigator.clipboard.writeText(message.content);
                      // Show toast notification
                    }}
                  />
                );
              }}
            />

            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </>
        )}

        {/* New messages indicator */}
        {showNewMessageIndicator && (
          <button
            className={styles.newMessageIndicator}
            onClick={() => {
              setIsAutoScroll(true);
              messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            New messages ↓
          </button>
        )}
      </div>

      {/* Input Area */}
      <div className={styles.inputArea}>
        <ChatInputBar
          onSend={handleSendMessage}
          onTyping={handleTyping}
          isDisabled={wsStatus !== WebSocketStatus.CONNECTED}
          placeholder={
            wsStatus !== WebSocketStatus.CONNECTED
              ? 'Connecting...'
              : 'Type a message...'
          }
        />
      </div>
    </div>
  );
};

// Estimate message height for virtual list
function estimateMessageHeight(message: ChatMessage): number {
  const baseHeight = 60; // Base height with padding
  const charsPerLine = 50;
  const lineHeight = 20;
  const lines = Math.ceil(message.content.length / charsPerLine);
  return baseHeight + (lines * lineHeight);
}

export default ChatView;
```

#### **src/components/chat/MessageBubble.tsx**
*Purpose: Individual message display component*

```typescript
import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../../types/chat';
import StreamingText from './StreamingText';
import Markdown from '../ui/Markdown';
import { formatRelativeTime } from '../../utils/date';
import {
  Copy,
  Edit2,
  Trash2,
  RefreshCw,
  MoreVertical,
  User,
  Bot
} from 'lucide-react';
import styles from './MessageBubble.module.css';

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
  isLast?: boolean;
  showAvatar?: boolean;
  onEdit?: (messageId: string) => void;
  onDelete?: (messageId: string) => void;
  onRegenerate?: () => void;
  onCopy?: () => void;
  focused?: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isStreaming = false,
  isLast = false,
  showAvatar = true,
  onEdit,
  onDelete,
  onRegenerate,
  onCopy,
  focused = false,
}) => {
  const [showActions, setShowActions] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const actionsRef = useRef<HTMLDivElement>(null);

  // Auto-collapse long messages
  useEffect(() => {
    if (message.content.length > 1000 && !isStreaming) {
      setIsCollapsed(true);
    }
  }, [message.content.length, isStreaming]);

  // Handle copy with feedback
  const handleCopy = async () => {
    if (onCopy) {
      onCopy();
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  // Message content
  const content = isCollapsed
    ? message.content.substring(0, 500) + '...'
    : message.content;

  // CSS classes
  const bubbleClass = `
    ${styles.messageBubble}
    ${message.is_user ? styles.userMessage : styles.assistantMessage}
    ${focused ? styles.focused : ''}
    ${message.is_deleted ? styles.deleted : ''}
  `;

  return (
    <div
      className={styles.messageContainer}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Avatar */}
      {showAvatar && (
        <div className={styles.avatar}>
          {message.is_user ? (
            <User size={20} />
          ) : (
            <Bot size={20} />
          )}
        </div>
      )}

      {/* Message bubble */}
      <div className={bubbleClass}>
        {/* Message header */}
        <div className={styles.messageHeader}>
          <span className={styles.sender}>
            {message.is_user ? 'You' : 'Assistant'}
          </span>
          <span className={styles.timestamp}>
            {formatRelativeTime(message.created_at)}
          </span>
          {message.is_edited && (
            <span className={styles.editedLabel}>(edited)</span>
          )}
        </div>

        {/* Message content */}
        <div className={styles.messageContent}>
          {message.is_deleted ? (
            <em className={styles.deletedText}>Message deleted</em>
          ) : isStreaming ? (
            <StreamingText
              text={content}
              isComplete={false}
              speed={30}
            />
          ) : (
            <Markdown content={content} />
          )}
        </div>

        {/* Expand/Collapse for long messages */}
        {message.content.length > 1000 && !message.is_deleted && (
          <button
            className={styles.expandButton}
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? 'Show more' : 'Show less'}
          </button>
        )}

        {/* Message metadata */}
        {!message.is_user && message.model_used && (
          <div className={styles.metadata}>
            <span className={styles.model}>{message.model_used}</span>
            <span className={styles.tokens}>{message.token_count} tokens</span>
          </div>
        )}

        {/* Action buttons */}
        {showActions && !message.is_deleted && (
          <div ref={actionsRef} className={styles.actions}>
            <button
              className={styles.actionButton}
              onClick={handleCopy}
              title="Copy message"
            >
              <Copy size={16} />
              {copySuccess && <span className={styles.copyFeedback}>Copied!</span>}
            </button>

            {message.is_user && onEdit && (
              <button
                className={styles.actionButton}
                onClick={() => onEdit(message.id)}
                title="Edit message"
              >
                <Edit2 size={16} />
              </button>
            )}

            {!message.is_user && isLast && onRegenerate && (
              <button
                className={styles.actionButton}
                onClick={onRegenerate}
                title="Regenerate response"
              >
                <RefreshCw size={16} />
              </button>
            )}

            {onDelete && (
              <button
                className={`${styles.actionButton} ${styles.deleteButton}`}
                onClick={() => onDelete(message.id)}
                title="Delete message"
              >
                <Trash2 size={16} />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
```

#### **src/components/chat/StreamingText.tsx**
*Purpose: Animated text reveal for AI responses*

```typescript
import React, { useState, useEffect, useRef } from 'react';
import Markdown from '../ui/Markdown';
import styles from './StreamingText.module.css';

interface StreamingTextProps {
  text: string;
  isComplete: boolean;
  speed?: number; // characters per second
  onComplete?: () => void;
}

const StreamingText: React.FC<StreamingTextProps> = ({
  text,
  isComplete,
  speed = 30,
  onComplete,
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // If complete, show all text immediately
    if (isComplete) {
      setDisplayedText(text);
      setCurrentIndex(text.length);
      if (onComplete) onComplete();
      return;
    }

    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Calculate interval based on speed
    const intervalMs = 1000 / speed;

    // Set up streaming interval
    intervalRef.current = setInterval(() => {
      setCurrentIndex((prevIndex) => {
        const nextIndex = prevIndex + 1;

        // Check if we've reached the end
        if (nextIndex >= text.length) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          if (onComplete) onComplete();
          return text.length;
        }

        return nextIndex;
      });
    }, intervalMs);

    // Cleanup on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [text, isComplete, speed, onComplete]);

  // Update displayed text when index changes
  useEffect(() => {
    setDisplayedText(text.slice(0, currentIndex));
  }, [text, currentIndex]);

  return (
    <div className={styles.streamingText}>
      <Markdown content={displayedText} />
      {!isComplete && currentIndex < text.length && (
        <span className={styles.cursor}>▊</span>
      )}
    </div>
  );
};

export default StreamingText;
```

#### **src/components/chat/ChatInputBar.tsx**
*Purpose: Message composition interface*

```typescript
import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Paperclip, AtSign, Smile } from 'lucide-react';
import { useDebounce } from '../../hooks/useDebounce';
import styles from './ChatInputBar.module.css';

interface ChatInputBarProps {
  onSend: (content: string, attachments?: File[]) => void;
  onTyping?: () => void;
  isDisabled?: boolean;
  placeholder?: string;
  projectContext?: any;
  suggestions?: string[];
}

const ChatInputBar: React.FC<ChatInputBarProps> = ({
  onSend,
  onTyping,
  isDisabled = false,
  placeholder = 'Type a message...',
  projectContext,
  suggestions = [],
}) => {
  const [content, setContent] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Debounced typing indicator
  const debouncedContent = useDebounce(content, 500);
  useEffect(() => {
    if (debouncedContent && onTyping) {
      onTyping();
    }
  }, [debouncedContent, onTyping]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    const scrollHeight = textarea.scrollHeight;
    const minHeight = 56; // 2 lines
    const maxHeight = 200; // 8 lines

    textarea.style.height = `${Math.min(Math.max(scrollHeight, minHeight), maxHeight)}px`;
  }, [content]);

  // Handle send
  const handleSend = () => {
    const trimmedContent = content.trim();
    if (!trimmedContent || isDisabled) return;

    onSend(trimmedContent, attachments);
    setContent('');
    setAttachments([]);
    setShowSuggestions(false);

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  // Handle keyboard shortcuts
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Send on Cmd/Ctrl + Enter
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSend();
      return;
    }

    // New line on Shift + Enter
    if (e.key === 'Enter' && e.shiftKey) {
      return; // Let default behavior handle new line
    }

    // Send on Enter (without modifiers)
    if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      handleSend();
      return;
    }

    // Edit last message on Up arrow (when empty)
    if (e.key === 'ArrowUp' && content === '') {
      e.preventDefault();
      // Trigger edit of last user message
      // This would be handled by parent component
      return;
    }

    // Clear or close suggestions on Escape
    if (e.key === 'Escape') {
      if (showSuggestions) {
        setShowSuggestions(false);
      } else if (content) {
        setContent('');
      }
      return;
    }

    // Navigate suggestions
    if (showSuggestions && suggestions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedSuggestionIndex((prev) =>
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedSuggestionIndex((prev) =>
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
      } else if (e.key === 'Tab' && selectedSuggestionIndex >= 0) {
        e.preventDefault();
        // Accept suggestion
        const suggestion = suggestions[selectedSuggestionIndex];
        setContent(content + suggestion);
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
      }
    }
  };

  // Handle text changes
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newContent = e.target.value;
    setContent(newContent);

    // Check for triggers
    const lastChar = newContent[newContent.length - 1];
    const lastTwoChars = newContent.slice(-2);

    if (lastChar === '@' || lastChar === '/' || lastChar === '#') {
      setShowSuggestions(true);
    } else if (lastTwoChars === '  ') {
      // Double space might trigger suggestions
      setShowSuggestions(false);
    }
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setAttachments((prev) => [...prev, ...files]);
  };

  // Remove attachment
  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className={styles.chatInputBar}>
      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className={styles.attachments}>
          {attachments.map((file, index) => (
            <div key={index} className={styles.attachment}>
              <span>{file.name}</span>
              <button onClick={() => removeAttachment(index)}>×</button>
            </div>
          ))}
        </div>
      )}

      {/* Suggestions dropdown */}
      {showSuggestions && suggestions.length > 0 && (
        <div className={styles.suggestions}>
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              className={`${styles.suggestion} ${
                index === selectedSuggestionIndex ? styles.selected : ''
              }`}
              onClick={() => {
                setContent(content + suggestion);
                setShowSuggestions(false);
              }}
              onMouseEnter={() => setSelectedSuggestionIndex(index)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {/* Input area */}
      <div className={styles.inputWrapper}>
        <textarea
          ref={textareaRef}
          value={content}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isDisabled}
          className={styles.textarea}
          rows={1}
        />

        {/* Action buttons */}
        <div className={styles.actions}>
          {/* Attachment button */}
          <button
            className={styles.actionButton}
            onClick={() => fileInputRef.current?.click()}
            disabled={isDisabled}
            title="Attach files"
          >
            <Paperclip size={20} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className={styles.hiddenInput}
            accept=".pdf,.doc,.docx,.txt,.md"
          />

          {/* Send button */}
          <button
            className={`${styles.sendButton} ${
              content.trim() && !isDisabled ? styles.active : ''
            }`}
            onClick={handleSend}
            disabled={!content.trim() || isDisabled}
            title="Send message (Cmd+Enter)"
          >
            <Send size={20} />
          </button>
        </div>
      </div>

      {/* Character count for long messages */}
      {content.length > 40000 && (
        <div className={styles.charCount}>
          {content.length} / 50,000
        </div>
      )}
    </div>
  );
};

export default ChatInputBar;
```

#### **src/components/chat/RecentChatsSidebar.tsx**
*Purpose: Thread navigation and history sidebar*

```typescript
import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '../../stores/chatStore';
import { ChatThread } from '../../types/chat';
import { formatRelativeTime, groupByDate } from '../../utils/date';
import {
  Search,
  MessageSquare,
  Clock,
  Archive,
  Trash2,
  Edit2,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import styles from './RecentChatsSidebar.module.css';

interface RecentChatsSidebarProps {
  projectId: string;
  activeThreadId?: string;
  onThreadSelect: (threadId: string) => void;
  onNewChat: () => void;
  isCollapsed?: boolean;
}

interface ThreadGroup {
  label: string;
  threads: ChatThread[];
}

const RecentChatsSidebar: React.FC<RecentChatsSidebarProps> = ({
  projectId,
  activeThreadId,
  onThreadSelect,
  onNewChat,
  isCollapsed = false,
}) => {
  const navigate = useNavigate();
  const { threads, loadRecentThreads, archiveThread, deleteThread } = useChatStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['Today', 'Yesterday']));
  const [sortBy, setSortBy] = useState<'recent' | 'oldest'>('recent');

  // Load threads on mount
  useEffect(() => {
    const loadThreads = async () => {
      setIsLoading(true);
      try {
        await loadRecentThreads(projectId);
      } finally {
        setIsLoading(false);
      }
    };
    loadThreads();
  }, [projectId, loadRecentThreads]);

  // Filter and group threads
  const threadGroups = useMemo(() => {
    // Convert Map to array and filter
    const threadArray = Array.from(threads.values())
      .filter(thread => {
        // Filter by project
        if (thread.project_id !== projectId) return false;

        // Filter by search query
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          return (
            thread.title.toLowerCase().includes(query) ||
            thread.last_message?.content.toLowerCase().includes(query)
          );
        }

        return true;
      });

    // Sort threads
    threadArray.sort((a, b) => {
      const dateA = new Date(a.last_activity_at).getTime();
      const dateB = new Date(b.last_activity_at).getTime();
      return sortBy === 'recent' ? dateB - dateA : dateA - dateB;
    });

    // Group by date
    const groups = groupByDate(threadArray, 'last_activity_at');

    return groups;
  }, [threads, projectId, searchQuery, sortBy]);

  // Toggle group expansion
  const toggleGroup = (groupLabel: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(groupLabel)) {
        next.delete(groupLabel);
      } else {
        next.add(groupLabel);
      }
      return next;
    });
  };

  // Handle thread actions
  const handleArchive = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    await archiveThread(threadId);
  };

  const handleDelete = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this chat?')) {
      await deleteThread(threadId);
      if (threadId === activeThreadId) {
        navigate(`/projects/${projectId}/chat`);
      }
    }
  };

  const handleRename = (e: React.MouseEvent, thread: ChatThread) => {
    e.stopPropagation();
    const newTitle = prompt('Rename chat:', thread.title);
    if (newTitle && newTitle !== thread.title) {
      // Update thread title
      useChatStore.getState().updateThreadTitle(thread.id, newTitle);
    }
  };

  if (isCollapsed) {
    return null; // Or return a minimal collapsed view
  }

  return (
    <div className={styles.sidebar}>
      {/* Header */}
      <div className={styles.header}>
        <h3>Recent Chats</h3>
        <button className={styles.newChatButton} onClick={onNewChat}>
          <MessageSquare size={16} />
          New Chat
        </button>
      </div>

      {/* Search */}
      <div className={styles.searchContainer}>
        <Search size={16} className={styles.searchIcon} />
        <input
          type="text"
          placeholder="Search chats..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      {/* Sort options */}
      <div className={styles.sortOptions}>
        <button
          className={`${styles.sortButton} ${sortBy === 'recent' ? styles.active : ''}`}
          onClick={() => setSortBy('recent')}
        >
          Most Recent
        </button>
        <button
          className={`${styles.sortButton} ${sortBy === 'oldest' ? styles.active : ''}`}
          onClick={() => setSortBy('oldest')}
        >
          Oldest First
        </button>
      </div>

      {/* Thread list */}
      <div className={styles.threadList}>
        {isLoading ? (
          <div className={styles.loading}>Loading chats...</div>
        ) : threadGroups.length === 0 ? (
          <div className={styles.empty}>
            {searchQuery ? 'No chats found' : 'No chats yet'}
          </div>
        ) : (
          threadGroups.map((group) => (
            <div key={group.label} className={styles.threadGroup}>
              {/* Group header */}
              <button
                className={styles.groupHeader}
                onClick={() => toggleGroup(group.label)}
              >
                {expandedGroups.has(group.label) ? (
                  <ChevronDown size={16} />
                ) : (
                  <ChevronRight size={16} />
                )}
                <span>{group.label}</span>
                <span className={styles.threadCount}>({group.threads.length})</span>
              </button>

              {/* Thread items */}
              {expandedGroups.has(group.label) && (
                <div className={styles.groupThreads}>
                  {group.threads.map((thread) => (
                    <div
                      key={thread.id}
                      className={`${styles.threadItem} ${
                        thread.id === activeThreadId ? styles.active : ''
                      }`}
                      onClick={() => onThreadSelect(thread.id)}
                    >
                      {/* Thread info */}
                      <div className={styles.threadInfo}>
                        <h4 className={styles.threadTitle}>{thread.title}</h4>
                        {thread.last_message && (
                          <p className={styles.lastMessage}>
                            {thread.last_message.is_user ? 'You: ' : 'AI: '}
                            {thread.last_message.content}
                          </p>
                        )}
                        <div className={styles.threadMeta}>
                          <Clock size={12} />
                          <span>{formatRelativeTime(thread.last_activity_at)}</span>
                          {thread.message_count > 0 && (
                            <>
                              <span>•</span>
                              <span>{thread.message_count} messages</span>
                            </>
                          )}
                          {thread.is_summarized && (
                            <span className={styles.summarizedBadge}>Summarized</span>
                          )}
                        </div>
                      </div>

                      {/* Thread actions */}
                      <div className={styles.threadActions}>
                        <button
                          onClick={(e) => handleRename(e, thread)}
                          title="Rename"
                        >
                          <Edit2 size={14} />
                        </button>
                        <button
                          onClick={(e) => handleArchive(e, thread.id)}
                          title="Archive"
                        >
                          <Archive size={14} />
                        </button>
                        <button
                          onClick={(e) => handleDelete(e, thread.id)}
                          title="Delete"
                          className={styles.deleteButton}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default RecentChatsSidebar;
```

#### **src/components/chat/ConnectionStatus.tsx**
*Purpose: WebSocket connection status indicator*

```typescript
import React from 'react';
import { WebSocketStatus } from '../../types/websocket';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';
import styles from './ConnectionStatus.module.css';

interface ConnectionStatusProps {
  status: WebSocketStatus;
  onReconnect?: () => void;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ status, onReconnect }) => {
  if (status === WebSocketStatus.CONNECTED) {
    return null; // Don't show when connected
  }

  const getStatusInfo = () => {
    switch (status) {
      case WebSocketStatus.CONNECTING:
        return {
          icon: <RefreshCw size={16} className={styles.spinning} />,
          text: 'Connecting...',
          className: styles.connecting,
        };
      case WebSocketStatus.DISCONNECTED:
        return {
          icon: <WifiOff size={16} />,
          text: 'Disconnected',
          className: styles.disconnected,
          showRetry: true,
        };
      case WebSocketStatus.ERROR:
        return {
          icon: <WifiOff size={16} />,
          text: 'Connection error',
          className: styles.error,
          showRetry: true,
        };
      default:
        return null;
    }
  };

  const statusInfo = getStatusInfo();
  if (!statusInfo) return null;

  return (
    <div className={`${styles.connectionStatus} ${statusInfo.className}`}>
      <div className={styles.content}>
        {statusInfo.icon}
        <span>{statusInfo.text}</span>
        {statusInfo.showRetry && onReconnect && (
          <button onClick={onReconnect} className={styles.retryButton}>
            Retry
          </button>
        )}
      </div>
    </div>
  );
};

export default ConnectionStatus;
```

### State Management

#### **src/stores/chatStore.ts**
*Purpose: Zustand store for chat state management*

```typescript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { ChatThread, ChatMessage, ChatSummary } from '../types/chat';
import { chatApi } from '../api/chat';

interface ChatState {
  // State
  threads: Map<string, ChatThread>;
  activeThreadId: string | null;
  messages: Map<string, ChatMessage[]>;
  isLoadingThread: boolean;
  streamingMessage: Partial<ChatMessage> | null;
  error: string | null;

  // Actions - Thread Management
  loadThread: (threadId: string) => Promise<void>;
  createThread: (projectId: string, initialMessage?: string) => Promise<ChatThread>;
  updateThreadTitle: (threadId: string, title: string) => Promise<void>;
  archiveThread: (threadId: string) => Promise<void>;
  deleteThread: (threadId: string) => Promise<void>;
  loadRecentThreads: (projectId?: string) => Promise<void>;

  // Actions - Message Management
  sendMessage: (content: string, attachments?: File[]) => Promise<void>;
  editMessage: (messageId: string, content: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  regenerateResponse: (messageId: string) => Promise<void>;

  // Actions - Streaming
  startStreaming: (messageId: string) => void;
  addStreamChunk: (messageId: string, chunk: string) => void;
  completeStreaming: (messageId: string, fullContent: string) => void;

  // Actions - UI Updates
  updateMessageInStore: (message: ChatMessage) => void;
  removeMessageFromStore: (messageId: string) => void;
  setActiveThread: (threadId: string | null) => void;
  clearError: () => void;
}

export const useChatStore = create<ChatState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        threads: new Map(),
        activeThreadId: null,
        messages: new Map(),
        isLoadingThread: false,
        streamingMessage: null,
        error: null,

        // Load thread and messages
        loadThread: async (threadId: string) => {
          set((state) => {
            state.isLoadingThread = true;
            state.error = null;
          });

          try {
            // Load thread details
            const thread = await chatApi.getThread(threadId);

            // Load messages
            const messagesResponse = await chatApi.getThreadMessages(threadId);

            set((state) => {
              state.threads.set(threadId, thread);
              state.messages.set(threadId, messagesResponse.messages);
              state.activeThreadId = threadId;
              state.isLoadingThread = false;
            });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to load thread';
              state.isLoadingThread = false;
            });
          }
        },

        // Create new thread
        createThread: async (projectId: string, initialMessage?: string) => {
          try {
            const thread = await chatApi.createThread({
              project_id: projectId,
              title: initialMessage ? `Chat about ${initialMessage.slice(0, 30)}...` : 'New Chat',
              initial_message: initialMessage,
            });

            set((state) => {
              state.threads.set(thread.id, thread);
              state.messages.set(thread.id, []);
              state.activeThreadId = thread.id;
            });

            return thread;
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to create thread';
            });
            throw error;
          }
        },

        // Update thread title
        updateThreadTitle: async (threadId: string, title: string) => {
          try {
            const updatedThread = await chatApi.updateThread(threadId, { title });

            set((state) => {
              const thread = state.threads.get(threadId);
              if (thread) {
                state.threads.set(threadId, { ...thread, title });
              }
            });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to update title';
            });
          }
        },

        // Archive thread
        archiveThread: async (threadId: string) => {
          try {
            await chatApi.updateThread(threadId, { is_archived: true });

            set((state) => {
              state.threads.delete(threadId);
              if (state.activeThreadId === threadId) {
                state.activeThreadId = null;
              }
            });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to archive thread';
            });
          }
        },

        // Delete thread
        deleteThread: async (threadId: string) => {
          try {
            await chatApi.deleteThread(threadId);

            set((state) => {
              state.threads.delete(threadId);
              state.messages.delete(threadId);
              if (state.activeThreadId === threadId) {
                state.activeThreadId = null;
              }
            });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to delete thread';
            });
          }
        },

        // Load recent threads
        loadRecentThreads: async (projectId?: string) => {
          try {
            const response = await chatApi.listThreads({
              project_id: projectId,
              limit: 50,
              sort_by: 'activity'
            });

            set((state) => {
              // Clear existing threads if loading for specific project
              if (projectId) {
                state.threads.clear();
              }

              // Add threads to map
              response.threads.forEach(thread => {
                state.threads.set(thread.id, thread);
              });
            });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to load threads';
            });
          }
        },

        // Send message (optimistic update)
        sendMessage: async (content: string, attachments?: File[]) => {
          const { activeThreadId } = get();
          if (!activeThreadId) return;

          // Create optimistic message
          const tempId = `temp-${Date.now()}`;
          const optimisticMessage: ChatMessage = {
            id: tempId,
            thread_id: activeThreadId,
            content,
            is_user: true,
            created_at: new Date().toISOString(),
            is_edited: false,
            is_deleted: false,
            token_count: content.length / 4, // Rough estimate
            metadata: {},
          };

          // Add to store immediately
          set((state) => {
            const messages = state.messages.get(activeThreadId) || [];
            state.messages.set(activeThreadId, [...messages, optimisticMessage]);
          });

          // The actual send will be handled by WebSocket
        },

        // Edit message
        editMessage: async (messageId: string, content: string) => {
          const { activeThreadId, messages } = get();
          if (!activeThreadId) return;

          // Optimistic update
          set((state) => {
            const threadMessages = state.messages.get(activeThreadId) || [];
            const messageIndex = threadMessages.findIndex(m => m.id === messageId);

            if (messageIndex !== -1) {
              threadMessages[messageIndex] = {
                ...threadMessages[messageIndex],
                content,
                is_edited: true,
                edited_at: new Date().toISOString(),
              };
              state.messages.set(activeThreadId, [...threadMessages]);
            }
          });

          // The actual edit will be handled by WebSocket
        },

        // Delete message
        deleteMessage: async (messageId: string) => {
          const { activeThreadId } = get();
          if (!activeThreadId) return;

          // Optimistic update
          set((state) => {
            const threadMessages = state.messages.get(activeThreadId) || [];
            const messageIndex = threadMessages.findIndex(m => m.id === messageId);

            if (messageIndex !== -1) {
              threadMessages[messageIndex] = {
                ...threadMessages[messageIndex],
                is_deleted: true,
                deleted_at: new Date().toISOString(),
              };
              state.messages.set(activeThreadId, [...threadMessages]);
            }
          });

          // The actual delete will be handled by WebSocket
        },

        // Regenerate response
        regenerateResponse: async (messageId: string) => {
          // Mark message as being regenerated
          set((state) => {
            const { activeThreadId } = state;
            if (!activeThreadId) return;

            const threadMessages = state.messages.get(activeThreadId) || [];
            const messageIndex = threadMessages.findIndex(m => m.id === messageId);

            if (messageIndex !== -1) {
              threadMessages[messageIndex] = {
                ...threadMessages[messageIndex],
                metadata: { ...threadMessages[messageIndex].metadata, regenerating: true },
              };
              state.messages.set(activeThreadId, [...threadMessages]);
            }
          });

          // The actual regeneration will be handled by WebSocket
        },

        // Start streaming
        startStreaming: (messageId: string) => {
          set((state) => {
            state.streamingMessage = {
              id: messageId,
              content: '',
              is_user: false,
              created_at: new Date().toISOString(),
            };
          });
        },

        // Add stream chunk
        addStreamChunk: (messageId: string, chunk: string) => {
          set((state) => {
            if (state.streamingMessage && state.streamingMessage.id === messageId) {
              state.streamingMessage.content = (state.streamingMessage.content || '') + chunk;
            }
          });
        },

        // Complete streaming
        completeStreaming: (messageId: string, fullContent: string) => {
          set((state) => {
            const { activeThreadId, streamingMessage } = state;
            if (!activeThreadId || !streamingMessage || streamingMessage.id !== messageId) return;

            // Add completed message to thread
            const threadMessages = state.messages.get(activeThreadId) || [];
            const completeMessage: ChatMessage = {
              id: messageId,
              thread_id: activeThreadId,
              content: fullContent,
              is_user: false,
              created_at: streamingMessage.created_at || new Date().toISOString(),
              is_edited: false,
              is_deleted: false,
              token_count: fullContent.length / 4, // Will be updated by server
              metadata: {},
            };

            state.messages.set(activeThreadId, [...threadMessages, completeMessage]);
            state.streamingMessage = null;
          });
        },

        // Update message in store
        updateMessageInStore: (message: ChatMessage) => {
          set((state) => {
            const threadMessages = state.messages.get(message.thread_id) || [];
            const messageIndex = threadMessages.findIndex(m => m.id === message.id);

            if (messageIndex !== -1) {
              threadMessages[messageIndex] = message;
            } else {
              threadMessages.push(message);
            }

            state.messages.set(message.thread_id, [...threadMessages]);
          });
        },

        // Remove message from store
        removeMessageFromStore: (messageId: string) => {
          set((state) => {
            const { activeThreadId } = state;
            if (!activeThreadId) return;

            const threadMessages = state.messages.get(activeThreadId) || [];
            const filtered = threadMessages.filter(m => m.id !== messageId);
            state.messages.set(activeThreadId, filtered);
          });
        },

        // Set active thread
        setActiveThread: (threadId: string | null) => {
          set((state) => {
            state.activeThreadId = threadId;
          });
        },

        // Clear error
        clearError: () => {
          set((state) => {
            state.error = null;
          });
        },
      })),
      {
        name: 'chat-store',
        partialize: (state) => ({
          // Only persist thread list and active thread
          threads: Array.from(state.threads.entries()).slice(0, 20), // Limit persisted threads
          activeThreadId: state.activeThreadId,
        }),
      }
    )
  )
);
```

### Hooks

#### **src/hooks/useWebSocket.ts**
*Purpose: WebSocket connection hook with reconnection logic*

```typescript
import { useEffect, useRef, useState, useCallback } from 'react';
import { useAuth } from './useAuth';

export enum WebSocketStatus {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
}

interface UseWebSocketOptions {
  url: string | null;
  onMessage: (message: any) => void;
  onStatusChange?: (status: WebSocketStatus) => void;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

interface UseWebSocketReturn {
  status: WebSocketStatus;
  sendMessage: (message: any) => void;
  reconnect: () => void;
  close: () => void;
}

export const useWebSocket = ({
  url,
  onMessage,
  onStatusChange,
  reconnectAttempts = 5,
  reconnectDelay = 1000,
  heartbeatInterval = 30000,
}: UseWebSocketOptions): UseWebSocketReturn => {
  const { token } = useAuth();
  const [status, setStatus] = useState<WebSocketStatus>(WebSocketStatus.DISCONNECTED);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<any[]>([]);

  // Update status and notify
  const updateStatus = useCallback((newStatus: WebSocketStatus) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);

  // Clear timers
  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  // Send heartbeat
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'heartbeat',
        timestamp: new Date().toISOString(),
      }));
    }
  }, []);

  // Setup heartbeat
  const setupHeartbeat = useCallback(() => {
    clearInterval(heartbeatIntervalRef.current!);
    heartbeatIntervalRef.current = setInterval(sendHeartbeat, heartbeatInterval);
  }, [sendHeartbeat, heartbeatInterval]);

  // Flush message queue
  const flushMessageQueue = useCallback(() => {
    while (messageQueueRef.current.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
      const message = messageQueueRef.current.shift();
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!url || !token) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    updateStatus(WebSocketStatus.CONNECTING);

    // Create WebSocket with auth token
    const wsUrl = `${url}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      updateStatus(WebSocketStatus.CONNECTED);
      reconnectCountRef.current = 0;
      setupHeartbeat();
      flushMessageQueue();
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        // Handle heartbeat acknowledgment
        if (message.type === 'heartbeat_ack') {
          return;
        }

        onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      updateStatus(WebSocketStatus.ERROR);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      updateStatus(WebSocketStatus.DISCONNECTED);
      clearTimers();

      // Attempt reconnection if not a normal closure
      if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
        const delay = reconnectDelay * Math.pow(2, reconnectCountRef.current);
        reconnectCountRef.current += 1;

        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectCountRef.current}/${reconnectAttempts})`);

        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    };
  }, [url, token, updateStatus, setupHeartbeat, flushMessageQueue, clearTimers, onMessage, reconnectAttempts, reconnectDelay]);

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      // Queue message for later
      messageQueueRef.current.push(message);

      // Try to reconnect if disconnected
      if (status === WebSocketStatus.DISCONNECTED) {
        connect();
      }
    }
  }, [status, connect]);

  // Manual reconnect
  const reconnect = useCallback(() => {
    reconnectCountRef.current = 0;
    clearTimers();
    connect();
  }, [connect, clearTimers]);

  // Close connection
  const close = useCallback(() => {
    reconnectCountRef.current = reconnectAttempts; // Prevent auto-reconnect
    clearTimers();
    if (wsRef.current) {
      wsRef.current.close(1000, 'User initiated close');
      wsRef.current = null;
    }
    updateStatus(WebSocketStatus.DISCONNECTED);
  }, [clearTimers, updateStatus, reconnectAttempts]);

  // Setup connection on mount/URL change
  useEffect(() => {
    if (url) {
      connect();
    }

    return () => {
      close();
    };
  }, [url]); // Only reconnect on URL change, not on every render

  return {
    status,
    sendMessage,
    reconnect,
    close,
  };
};
```

### API Client

#### **src/api/chat.ts**
*Purpose: Chat API client*

```typescript
import { apiClient } from './client';
import {
  ThreadCreate,
  ThreadUpdate,
  ThreadResponse,
  ThreadListResponse,
  MessageListResponse,
  MessageUpdate,
  MessageResponse,
  SummarizeRequest,
  SummaryResponse
} from '../types/chat';

export const chatApi = {
  // Thread endpoints
  async listThreads(params: {
    project_id?: string;
    include_archived?: boolean;
    sort_by?: 'created' | 'updated' | 'activity';
    order?: 'asc' | 'desc';
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<ThreadListResponse> {
    const response = await apiClient.get('/threads', { params });
    return response.data;
  },

  async createThread(data: ThreadCreate): Promise<ThreadResponse> {
    const response = await apiClient.post('/threads', data);
    return response.data;
  },

  async getThread(threadId: string): Promise<ThreadResponse> {
    const response = await apiClient.get(`/threads/${threadId}`);
    return response.data;
  },

  async updateThread(threadId: string, data: ThreadUpdate): Promise<ThreadResponse> {
    const response = await apiClient.patch(`/threads/${threadId}`, data);
    return response.data;
  },

  async deleteThread(threadId: string): Promise<void> {
    await apiClient.delete(`/threads/${threadId}`);
  },

  // Message endpoints
  async getThreadMessages(
    threadId: string,
    params?: {
      limit?: number;
      before?: string;
      after?: string;
      include_deleted?: boolean;
    }
  ): Promise<MessageListResponse> {
    const response = await apiClient.get(`/threads/${threadId}/messages`, { params });
    return response.data;
  },

  async updateMessage(messageId: string, data: MessageUpdate): Promise<MessageResponse> {
    const response = await apiClient.patch(`/messages/${messageId}`, data);
    return response.data;
  },

  async deleteMessage(messageId: string): Promise<void> {
    await apiClient.delete(`/messages/${messageId}`);
  },

  // Summary endpoints
  async summarizeThread(threadId: string, data: SummarizeRequest): Promise<SummaryResponse> {
    const response = await apiClient.post(`/threads/${threadId}/summarize`, data);
    return response.data;
  },

  async getThreadSummary(threadId: string): Promise<SummaryResponse> {
    const response = await apiClient.get(`/threads/${threadId}/summary`);
    return response.data;
  },
};
```

### Types

#### **src/types/chat.ts**
*Purpose: TypeScript type definitions*

```typescript
export interface ChatThread {
  id: string;
  project_id: string;
  title: string;
  created_at: string;
  last_activity_at: string;
  message_count: number;
  is_summarized: boolean;
  summary_count: number;
  is_archived: boolean;
  last_message?: {
    content: string;
    is_user: boolean;
    created_at: string;
  };
  metadata?: Record<string, any>;
}

export interface ChatMessage {
  id: string;
  thread_id: string;
  content: string;
  is_user: boolean;
  created_at: string;
  is_edited: boolean;
  edited_at?: string;
  is_deleted: boolean;
  deleted_at?: string;
  token_count: number;
  model_used?: string;
  metadata: Record<string, any>;
  edit_history?: Array<{
    content: string;
    edited_at: string;
  }>;
}

export interface ChatSummary {
  id: string;
  thread_id: string;
  summary_text: string;
  key_topics: string[];
  created_at: string;
  message_count: number;
  token_count: number;
}

// Request/Response types
export interface ThreadCreate {
  project_id: string;
  title?: string;
  initial_message?: string;
}

export interface ThreadUpdate {
  title?: string;
  is_archived?: boolean;
}

export interface ThreadResponse extends ChatThread {
  websocket_url?: string;
}

export interface ThreadListResponse {
  threads: ThreadResponse[];
  total: number;
  has_more: boolean;
}

export interface MessageUpdate {
  content: string;
}

export interface MessageResponse extends ChatMessage {}

export interface MessageListResponse {
  messages: MessageResponse[];
  has_more: boolean;
  total_tokens: number;
}

export interface SummarizeRequest {
  force?: boolean;
}

export interface SummaryResponse extends ChatSummary {
  status: string;
}
```

#### **src/types/websocket.ts**
*Purpose: WebSocket type definitions*

```typescript
export enum MessageType {
  // Connection
  CONNECTION_ESTABLISHED = 'connection_established',
  HEARTBEAT = 'heartbeat',
  HEARTBEAT_ACK = 'heartbeat_ack',

  // Messages
  SEND_MESSAGE = 'send_message',
  NEW_MESSAGE = 'new_message',
  EDIT_MESSAGE = 'edit_message',
  MESSAGE_UPDATED = 'message_updated',
  DELETE_MESSAGE = 'delete_message',
  MESSAGE_DELETED = 'message_deleted',

  // Streaming
  ASSISTANT_MESSAGE_START = 'assistant_message_start',
  STREAM_CHUNK = 'stream_chunk',

  // Status
  TYPING_INDICATOR = 'typing_indicator',
  ERROR = 'error',
  SUMMARY_AVAILABLE = 'summary_available',

  // Actions
  REGENERATE = 'regenerate',
}

export enum WebSocketStatus {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
}

export interface WebSocketMessage {
  type: MessageType;
  timestamp?: string;
  [key: string]: any;
}
```

### Tests

#### **tests/test_ai_provider.py**
*Purpose: Unit tests for AI provider service*

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.ai_provider import (
    OpenAIProvider, MockAIProvider, AIProviderFactory,
    ConversationManager, AIMessage, AIResponse
)
from models.chat import ChatMessage


@pytest.mark.asyncio
async def test_openai_provider_streaming():
    """Test OpenAI provider streaming functionality"""
    # Mock OpenAI client
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])
    ]

    provider = OpenAIProvider(api_key="test-key")
    with patch.object(provider.client.chat.completions, 'create', return_value=mock_stream):
        chunks = []
        async for chunk in provider.complete(
            [AIMessage(role="user", content="Test")],
            stream=True
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_openai_provider_complete():
    """Test OpenAI provider non-streaming completion"""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Test response"),
            finish_reason="stop"
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )
    mock_response.model = "gpt-4"

    provider = OpenAIProvider(api_key="test-key")
    with patch.object(provider.client.chat.completions, 'create', return_value=mock_response):
        response = await provider.complete(
            [AIMessage(role="user", content="Test")],
            stream=False
        )

        assert isinstance(response, AIResponse)
        assert response.content == "Test response"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15


def test_openai_token_counting():
    """Test token counting functionality"""
    provider = OpenAIProvider(api_key="test-key", model="gpt-4")

    # Test basic counting
    text = "Hello, world!"
    token_count = provider.count_tokens(text)
    assert token_count > 0
    assert token_count < len(text)  # Tokens should be less than characters

    # Test empty string
    assert provider.count_tokens("") == 0

    # Test long text
    long_text = "Lorem ipsum " * 100
    long_count = provider.count_tokens(long_text)
    assert long_count > 100


def test_ai_provider_factory():
    """Test AI provider factory"""
    # Test OpenAI provider creation
    provider = AIProviderFactory.create("openai", {
        "api_key": "test-key",
        "model": "gpt-4",
        "max_tokens": 2048
    })
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4"
    assert provider.max_tokens == 2048

    # Test mock provider creation
    mock_provider = AIProviderFactory.create("mock", {
        "response": "Test mock",
        "delay": 0.05
    })
    assert isinstance(mock_provider, MockAIProvider)
    assert mock_provider.response == "Test mock"

    # Test unknown provider
    with pytest.raises(ValueError):
        AIProviderFactory.create("unknown", {})


@pytest.mark.asyncio
async def test_conversation_manager_prepare_messages():
    """Test conversation manager message preparation"""
    provider = MockAIProvider()
    manager = ConversationManager(provider, system_prompt="You are helpful.")

    # Create test messages
    messages = [
        ChatMessage(id="1", content="Hello", is_user=True, token_count=10),
        ChatMessage(id="2", content="Hi there!", is_user=False, token_count=15),
        ChatMessage(id="3", content="How are you?", is_user=True, token_count=12),
    ]

    # Test normal preparation
    ai_messages = manager.prepare_messages(messages)

    assert len(ai_messages) == 4  # System + 3 messages
    assert ai_messages[0].role == "system"
    assert ai_messages[0].content == "You are helpful."
    assert ai_messages[1].role == "user"
    assert ai_messages[1].content == "Hello"
    assert ai_messages[2].role == "assistant"
    assert ai_messages[2].content == "Hi there!"

    # Test token limit handling
    manager.max_context_tokens = 30  # Very low limit
    limited_messages = manager.prepare_messages(messages)

    # Should include system prompt and most recent messages that fit
    assert len(limited_messages) < 4


@pytest.mark.asyncio
async def test_conversation_manager_summary():
    """Test thread summary generation"""
    provider = MockAIProvider(response="Summary: Discussion about testing.")
    manager = ConversationManager(provider)

    messages = [
        ChatMessage(id="1", content="Let's talk about testing", is_user=True),
        ChatMessage(id="2", content="Testing is important for quality", is_user=False),
    ]

    summary = await manager.get_thread_summary(messages, max_length=100)

    assert "Summary" in summary
    assert len(summary) <= 500  # Default max length


@pytest.mark.asyncio
async def test_mock_provider():
    """Test mock AI provider for testing"""
    provider = MockAIProvider(response="Mock response", delay=0.01)

    # Test streaming
    chunks = []
    async for chunk in provider.complete([AIMessage(role="user", content="Test")]):
        chunks.append(chunk)

    assert " ".join(chunks).strip() == "Mock response"
    assert provider.call_count == 1

    # Test non-streaming
    response = await provider.complete(
        [AIMessage(role="user", content="Test")],
        stream=False
    )
    assert response.content == "Mock response"
    assert provider.call_count == 2

    # Test health check
    assert await provider.health_check() is True
```

#### **tests/test_websocket_manager.py**
*Purpose: Unit tests for WebSocket manager*

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import asyncio
from fastapi import WebSocket

from services.websocket_manager import (
    ConnectionManager, MessageHandler, MessageType
)
from services.chat_service import ChatService
from services.ai_provider import MockAIProvider
from models.chat import ChatMessage


@pytest.fixture
def connection_manager():
    """Create connection manager instance"""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket"""
    ws = AsyncMock(spec=WebSocket)
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connection_manager_connect(connection_manager, mock_websocket):
    """Test WebSocket connection"""
    connection_id = await connection_manager.connect(
        mock_websocket,
        thread_id="thread-123",
        user_id="user-456"
    )

    assert connection_id is not None
    assert mock_websocket in connection_manager.active_connections["thread-123"]
    assert connection_manager.connection_info[mock_websocket]["user_id"] == "user-456"

    # Verify connection established message
    mock_websocket.send_json.assert_called_once()
    call_args = mock_websocket.send_json.call_args[0][0]
    assert call_args["type"] == MessageType.CONNECTION_ESTABLISHED
    assert call_args["connection_id"] == connection_id


@pytest.mark.asyncio
async def test_connection_manager_disconnect(connection_manager, mock_websocket):
    """Test WebSocket disconnection"""
    # Connect first
    connection_id = await connection_manager.connect(
        mock_websocket,
        thread_id="thread-123",
        user_id="user-456"
    )

    # Then disconnect
    await connection_manager.disconnect(mock_websocket)

    assert mock_websocket not in connection_manager.active_connections.get("thread-123", set())
    assert mock_websocket not in connection_manager.connection_info
    assert connection_id not in connection_manager.heartbeats


@pytest.mark.asyncio
async def test_connection_manager_broadcast(connection_manager):
    """Test message broadcasting"""
    # Create multiple mock websockets
    ws1 = AsyncMock(spec=WebSocket)
    ws2 = AsyncMock(spec=WebSocket)
    ws3 = AsyncMock(spec=WebSocket)

    # Connect to same thread
    await connection_manager.connect(ws1, "thread-123", "user-1")
    await connection_manager.connect(ws2, "thread-123", "user-2")
    await connection_manager.connect(ws3, "thread-456", "user-3")  # Different thread

    # Broadcast message
    test_message = {"type": "test", "data": "hello"}
    await connection_manager.send_message("thread-123", test_message)

    # Verify only ws1 and ws2 received the message
    ws1.send_json.assert_called_with(test_message)
    ws2.send_json.assert_called_with(test_message)
    ws3.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_connection_manager_exclude_sender(connection_manager):
    """Test message broadcasting with sender exclusion"""
    ws1 = AsyncMock(spec=WebSocket)
    ws2 = AsyncMock(spec=WebSocket)

    await connection_manager.connect(ws1, "thread-123", "user-1")
    await connection_manager.connect(ws2, "thread-123", "user-2")

    # Broadcast excluding ws1
    test_message = {"type": "test"}
    await connection_manager.send_message(
        "thread-123",
        test_message,
        exclude_websocket=ws1
    )

    # Only ws2 should receive
    ws1.send_json.assert_not_called()
    ws2.send_json.assert_called_once_with(test_message)


@pytest.mark.asyncio
async def test_rate_limiting(connection_manager, mock_websocket):
    """Test rate limiting functionality"""
    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Send messages up to rate limit
    for i in range(60):
        assert await connection_manager.check_rate_limit(mock_websocket) is True

    # Next message should be rate limited
    assert await connection_manager.check_rate_limit(mock_websocket) is False


@pytest.mark.asyncio
async def test_heartbeat_handling(connection_manager, mock_websocket):
    """Test heartbeat mechanism"""
    connection_id = await connection_manager.connect(
        mock_websocket,
        "thread-123",
        "user-456"
    )

    initial_heartbeat = connection_manager.heartbeats[connection_id]

    # Wait a bit
    await asyncio.sleep(0.1)

    # Handle heartbeat
    await connection_manager.handle_heartbeat(mock_websocket)

    # Verify heartbeat was updated
    assert connection_manager.heartbeats[connection_id] > initial_heartbeat

    # Verify acknowledgment sent
    mock_websocket.send_json.assert_called()
    last_call = mock_websocket.send_json.call_args[0][0]
    assert last_call["type"] == MessageType.HEARTBEAT_ACK


@pytest.mark.asyncio
async def test_message_handler_send_message():
    """Test message sending through handler"""
    # Setup mocks
    mock_db = AsyncMock()
    mock_ai_provider = MockAIProvider(response="AI response")
    mock_chat_service = AsyncMock(spec=ChatService)
    mock_websocket = AsyncMock(spec=WebSocket)
    connection_manager = ConnectionManager()

    # Setup handler
    handler = MessageHandler(
        websocket=mock_websocket,
        thread_id="thread-123",
        user_id="user-456",
        db=mock_db,
        ai_provider=mock_ai_provider,
        connection_manager=connection_manager
    )

    # Mock chat service methods
    mock_chat_service.create_message.return_value = ChatMessage(
        id="msg-1",
        thread_id="thread-123",
        content="Hello",
        is_user=True,
        created_at=datetime.utcnow(),
        token_count=10
    )

    handler.chat_service = mock_chat_service

    # Connect websocket
    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Handle send message
    await handler.handle_send_message({
        "type": MessageType.SEND_MESSAGE,
        "content": "Hello"
    })

    # Verify user message was created
    mock_chat_service.create_message.assert_called_once()
    call_args = mock_chat_service.create_message.call_args[1]
    assert call_args["content"] == "Hello"
    assert call_args["is_user"] is True

    # Verify new message broadcast
    mock_websocket.send_json.assert_called()
    broadcasts = [call[0][0] for call in mock_websocket.send_json.call_args_list]
    new_msg_broadcast = next(
        (b for b in broadcasts if b.get("type") == MessageType.NEW_MESSAGE),
        None
    )
    assert new_msg_broadcast is not None


@pytest.mark.asyncio
async def test_message_handler_edit_message():
    """Test message editing"""
    mock_db = AsyncMock()
    mock_websocket = AsyncMock(spec=WebSocket)
    connection_manager = ConnectionManager()

    handler = MessageHandler(
        websocket=mock_websocket,
        thread_id="thread-123",
        user_id="user-456",
        db=mock_db,
        ai_provider=MockAIProvider(),
        connection_manager=connection_manager
    )

    # Mock chat service
    mock_chat_service = AsyncMock(spec=ChatService)
    mock_chat_service.update_message.return_value = ChatMessage(
        id="msg-1",
        thread_id="thread-123",
        content="Updated content",
        is_user=True,
        is_edited=True,
        edited_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
        token_count=15
    )
    handler.chat_service = mock_chat_service

    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Handle edit message
    await handler.handle_edit_message({
        "type": MessageType.EDIT_MESSAGE,
        "message_id": "msg-1",
        "content": "Updated content"
    })

    # Verify message was updated
    mock_chat_service.update_message.assert_called_once_with(
        "msg-1",
        "Updated content"
    )

    # Verify update broadcast
    broadcasts = [call[0][0] for call in mock_websocket.send_json.call_args_list]
    update_broadcast = next(
        (b for b in broadcasts if b.get("type") == MessageType.MESSAGE_UPDATED),
        None
    )
    assert update_broadcast is not None
    assert update_broadcast["message"]["is_edited"] is True


@pytest.mark.asyncio
async def test_message_handler_delete_message():
    """Test message deletion"""
    mock_db = AsyncMock()
    mock_websocket = AsyncMock(spec=WebSocket)
    connection_manager = ConnectionManager()

    handler = MessageHandler(
        websocket=mock_websocket,
        thread_id="thread-123",
        user_id="user-456",
        db=mock_db,
        ai_provider=MockAIProvider(),
        connection_manager=connection_manager
    )

    # Mock chat service
    mock_chat_service = AsyncMock(spec=ChatService)
    mock_chat_service.delete_message.return_value = ChatMessage(
        id="msg-1",
        thread_id="thread-123",
        content="Deleted",
        is_user=True,
        is_deleted=True,
        deleted_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
        token_count=10
    )
    handler.chat_service = mock_chat_service

    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Handle delete message
    await handler.handle_delete_message({
        "type": MessageType.DELETE_MESSAGE,
        "message_id": "msg-1"
    })

    # Verify message was deleted
    mock_chat_service.delete_message.assert_called_once_with("msg-1")

    # Verify delete broadcast
    broadcasts = [call[0][0] for call in mock_websocket.send_json.call_args_list]
    delete_broadcast = next(
        (b for b in broadcasts if b.get("type") == MessageType.MESSAGE_DELETED),
        None
    )
    assert delete_broadcast is not None
    assert delete_broadcast["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_message_handler_stream_response():
    """Test streaming AI response"""
    mock_db = AsyncMock()
    mock_websocket = AsyncMock(spec=WebSocket)
    connection_manager = ConnectionManager()

    # Create AI provider that streams
    mock_ai_provider = MockAIProvider(
        response="This is a test response",
        delay=0.01
    )

    handler = MessageHandler(
        websocket=mock_websocket,
        thread_id="thread-123",
        user_id="user-456",
        db=mock_db,
        ai_provider=mock_ai_provider,
        connection_manager=connection_manager
    )

    # Mock chat service
    mock_chat_service = AsyncMock(spec=ChatService)
    ai_message_id = "ai-msg-1"
    mock_chat_service.create_message.return_value = ChatMessage(
        id=ai_message_id,
        thread_id="thread-123",
        content="",
        is_user=False,
        created_at=datetime.utcnow(),
        token_count=0
    )
    handler.chat_service = mock_chat_service

    # Mock conversation manager
    mock_conv_manager = AsyncMock()
    mock_conv_manager.prepare_messages.return_value = []
    handler.conversation_manager = mock_conv_manager

    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Generate AI response
    await handler.generate_ai_response(
        user_message_id="user-msg-1",
        messages=[]
    )

    # Verify stream start was sent
    broadcasts = [call[0][0] for call in mock_websocket.send_json.call_args_list]

    stream_start = next(
        (b for b in broadcasts if b.get("type") == MessageType.ASSISTANT_MESSAGE_START),
        None
    )
    assert stream_start is not None
    assert stream_start["message_id"] == ai_message_id

    # Verify chunks were sent
    chunks = [
        b for b in broadcasts
        if b.get("type") == MessageType.STREAM_CHUNK
    ]
    assert len(chunks) > 0

    # Reconstruct streamed content
    streamed_content = "".join(c["chunk"] for c in chunks)
    assert "test response" in streamed_content.lower()


@pytest.mark.asyncio
async def test_message_handler_error_handling():
    """Test error handling in message handler"""
    mock_db = AsyncMock()
    mock_websocket = AsyncMock(spec=WebSocket)
    connection_manager = ConnectionManager()

    handler = MessageHandler(
        websocket=mock_websocket,
        thread_id="thread-123",
        user_id="user-456",
        db=mock_db,
        ai_provider=MockAIProvider(),
        connection_manager=connection_manager
    )

    # Mock chat service to raise error
    mock_chat_service = AsyncMock(spec=ChatService)
    mock_chat_service.create_message.side_effect = Exception("Database error")
    handler.chat_service = mock_chat_service

    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Try to send message
    await handler.handle_send_message({
        "type": MessageType.SEND_MESSAGE,
        "content": "Test"
    })

    # Verify error was sent
    broadcasts = [call[0][0] for call in mock_websocket.send_json.call_args_list]
    error_broadcast = next(
        (b for b in broadcasts if b.get("type") == MessageType.ERROR),
        None
    )
    assert error_broadcast is not None
    assert "Database error" in error_broadcast["error"]


@pytest.mark.asyncio
async def test_connection_cleanup_on_error():
    """Test that connections are cleaned up on errors"""
    connection_manager = ConnectionManager()
    mock_websocket = AsyncMock(spec=WebSocket)

    # Simulate send error
    mock_websocket.send_json.side_effect = Exception("Connection lost")

    # Connect and try to send
    await connection_manager.connect(mock_websocket, "thread-123", "user-456")

    # Send message should handle error gracefully
    await connection_manager.send_message(
        "thread-123",
        {"type": "test"}
    )

    # Verify connection was removed
    assert mock_websocket not in connection_manager.active_connections.get("thread-123", set())


@pytest.mark.asyncio
async def test_typing_indicator():
    """Test typing indicator functionality"""
    mock_db = AsyncMock()
    mock_websocket = AsyncMock(spec=WebSocket)
    connection_manager = ConnectionManager()

    handler = MessageHandler(
        websocket=mock_websocket,
        thread_id="thread-123",
        user_id="user-456",
        db=mock_db,
        ai_provider=MockAIProvider(),
        connection_manager=connection_manager
    )

    # Connect another user to same thread
    other_ws = AsyncMock(spec=WebSocket)
    await connection_manager.connect(mock_websocket, "thread-123", "user-456")
    await connection_manager.connect(other_ws, "thread-123", "user-789")

    # Send typing indicator
    await handler.handle_typing_indicator({
        "type": MessageType.TYPING_INDICATOR,
        "is_typing": True
    })

    # Verify other user received indicator
    other_ws.send_json.assert_called()
    call_args = other_ws.send_json.call_args[0][0]
    assert call_args["type"] == MessageType.TYPING_INDICATOR
    assert call_args["user_id"] == "user-456"
    assert call_args["is_typing"] is True

    # Verify sender didn't receive their own indicator
    typing_calls = [
        call[0][0] for call in mock_websocket.send_json.call_args_list
        if call[0][0].get("type") == MessageType.TYPING_INDICATOR
    ]
    assert len(typing_calls) == 0
```

#### **tests/test_chat_service.py**
*Purpose: Unit tests for chat service*

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from services.chat_service import ChatService
from services.ai_provider import MockAIProvider, AIProviderFactory
from models.chat import ChatThread, ChatMessage, ChatSummary


@pytest.fixture
def mock_db():
    """Create mock database session"""
    db = AsyncMock(spec=AsyncSession)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def mock_ai_provider():
    """Create mock AI provider"""
    return MockAIProvider(response="Test AI response")


@pytest.fixture
def chat_service(mock_db, mock_ai_provider):
    """Create chat service instance"""
    return ChatService(db=mock_db, ai_provider=mock_ai_provider)


@pytest.mark.asyncio
async def test_create_message(chat_service, mock_db):
    """Test message creation"""
    # Mock thread query
    mock_thread = ChatThread(
        id="thread-123",
        project_id="proj-456",
        user_id="user-789",
        title="Test Thread",
        message_count=5,
        total_tokens=100
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_thread
    mock_db.execute.return_value = mock_result

    # Create message
    message = await chat_service.create_message(
        thread_id="thread-123",
        user_id="user-789",
        content="Hello, world!",
        is_user=True,
        metadata={"test": "data"}
    )

    # Verify message was created
    assert message.thread_id == "thread-123"
    assert message.content == "Hello, world!"
    assert message.is_user is True
    assert message.metadata == {"test": "data"}

    # Verify thread was updated
    assert mock_thread.message_count == 6
    assert mock_thread.total_tokens > 100

    # Verify database operations
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_create_message_thread_not_found(chat_service, mock_db):
    """Test message creation with non-existent thread"""
    # Mock empty thread query
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    # Should raise ValueError
    with pytest.raises(ValueError, match="Thread .* not found"):
        await chat_service.create_message(
            thread_id="non-existent",
            user_id="user-789",
            content="Hello",
            is_user=True
        )


@pytest.mark.asyncio
async def test_update_message(chat_service, mock_db):
    """Test message updating"""
    # Mock existing message
    mock_message = ChatMessage(
        id="msg-123",
        thread_id="thread-123",
        content="Original content",
        is_user=True,
        created_at=datetime.utcnow(),
        edit_history=[]
    )

    # Mock get_message
    with patch.object(chat_service, 'get_message', return_value=mock_message):
        # Mock thread query
        mock_thread = ChatThread(
            id="thread-123",
            last_activity_at=datetime.utcnow() - timedelta(hours=1)
        )
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = mock_thread
        mock_db.execute.return_value = mock_result

        # Update message
        updated = await chat_service.update_message(
            message_id="msg-123",
            content="Updated content"
        )

        # Verify update
        assert updated.content == "Updated content"
        assert len(mock_message.edit_history) == 1
        assert mock_message.edit_history[0]["content"] == "Original content"

        # Verify thread activity updated
        assert mock_thread.last_activity_at > datetime.utcnow() - timedelta(minutes=1)


@pytest.mark.asyncio
async def test_update_ai_message_error(chat_service, mock_db):
    """Test that AI messages cannot be edited"""
    # Mock AI message
    mock_message = ChatMessage(
        id="msg-123",
        is_user=False,
        content="AI response"
    )

    with patch.object(chat_service, 'get_message', return_value=mock_message):
        with pytest.raises(ValueError, match="Can only edit user messages"):
            await chat_service.update_message("msg-123", "New content")


@pytest.mark.asyncio
async def test_delete_message(chat_service, mock_db):
    """Test message deletion (soft delete)"""
    # Mock message
    mock_message = ChatMessage(
        id="msg-123",
        thread_id="thread-123",
        content="To be deleted",
        is_deleted=False
    )

    with patch.object(chat_service, 'get_message', return_value=mock_message):
        # Delete message
        deleted = await chat_service.delete_message("msg-123")

        # Verify soft delete
        assert deleted.is_deleted is True
        assert deleted.deleted_at is not None
        assert deleted.deleted_at <= datetime.utcnow()

        mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_thread_messages(chat_service, mock_db):
    """Test retrieving thread messages"""
    # Mock messages
    messages = [
        ChatMessage(id=f"msg-{i}", content=f"Message {i}", created_at=datetime.utcnow())
        for i in range(5)
    ]

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = messages
    mock_db.execute.return_value = mock_result

    # Get messages
    result = await chat_service.get_thread_messages(
        thread_id="thread-123",
        limit=10,
        include_deleted=False
    )

    assert len(result) == 5
    assert all(isinstance(msg, ChatMessage) for msg in result)

    # Verify query construction
    query_call = mock_db.execute.call_args[0][0]
    # Would need to inspect the actual SQLAlchemy query object


@pytest.mark.asyncio
async def test_create_thread(chat_service, mock_db):
    """Test thread creation"""
    thread = await chat_service.create_thread(
        project_id="proj-123",
        user_id="user-456",
        title="Custom Title"
    )

    assert thread.project_id == "proj-123"
    assert thread.user_id == "user-456"
    assert thread.title == "Custom Title"

    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_archive_thread(chat_service, mock_db):
    """Test thread archiving"""
    # Mock thread
    mock_thread = ChatThread(
        id="thread-123",
        is_archived=False,
        archived_at=None
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_thread
    mock_db.execute.return_value = mock_result

    # Archive thread
    archived = await chat_service.archive_thread("thread-123", archive=True)

    assert archived.is_archived is True
    assert archived.archived_at is not None

    # Test unarchive
    unarchived = await chat_service.archive_thread("thread-123", archive=False)

    assert unarchived.is_archived is False
    assert unarchived.archived_at is None


@pytest.mark.asyncio
async def test_generate_thread_title(chat_service, mock_db):
    """Test automatic thread title generation"""
    # Mock messages
    messages = [
        ChatMessage(
            id="msg-1",
            content="Can you help me debug this Python code?",
            is_user=True
        ),
        ChatMessage(
            id="msg-2",
            content="Of course! I'd be happy to help you debug your Python code.",
            is_user=False
        )
    ]

    with patch.object(chat_service, 'get_thread_messages', return_value=messages):
        with patch.object(chat_service, 'update_thread_title') as mock_update:
            # Mock AI response
            chat_service.ai_provider.response = "Python Debugging Help"

            # Generate title
            title = await chat_service.generate_thread_title("thread-123")

            assert title == "Python Debugging Help"
            mock_update.assert_called_once_with("thread-123", "Python Debugging Help")


@pytest.mark.asyncio
async def test_title_generation_fallback(chat_service, mock_db):
    """Test title generation fallback on error"""
    # Mock messages
    messages = [
        ChatMessage(
            id="msg-1",
            content="This is a test message",
            is_user=True
        )
    ]

    with patch.object(chat_service, 'get_thread_messages', return_value=messages):
        # Mock AI provider to raise error
        chat_service.ai_provider.complete = AsyncMock(
            side_effect=Exception("AI error")
        )

        # Generate title - should fallback
        title = await chat_service.generate_thread_title("thread-123")

        assert title.startswith("Chat about")
        assert "test message" in title


@pytest.mark.asyncio
async def test_get_recent_threads(chat_service, mock_db):
    """Test retrieving recent threads"""
    # Mock threads
    threads = [
        ChatThread(
            id=f"thread-{i}",
            title=f"Thread {i}",
            last_activity_at=datetime.utcnow() - timedelta(hours=i)
        )
        for i in range(5)
    ]

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = threads
    mock_db.execute.return_value = mock_result

    # Get recent threads
    result = await chat_service.get_recent_threads(
        user_id="user-123",
        limit=10,
        project_id="proj-456"
    )

    assert len(result) == 5
    assert all(isinstance(t, ChatThread) for t in result)

    # Verify they're ordered by activity
    for i in range(len(result) - 1):
        assert result[i].last_activity_at >= result[i + 1].last_activity_at
```

#### **tests/test_summarization.py**
*Purpose: Unit tests for summarization service*

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from services.summarization_service import SummarizationService
from services.ai_provider import MockAIProvider
from models.chat import ChatThread, ChatMessage, ChatSummary


@pytest.fixture
def mock_ai_provider():
    """Create mock AI provider"""
    return MockAIProvider(
        response="Summary: Users discussed Python debugging and best practices."
    )


@pytest.fixture
def summarization_service(mock_ai_provider):
    """Create summarization service"""
    mock_session_factory = AsyncMock()
    return SummarizationService(
        ai_provider=mock_ai_provider,
        db_session_factory=mock_session_factory
    )


@pytest.mark.asyncio
async def test_should_summarize_message_threshold():
    """Test summarization triggers based on message count"""
    service = SummarizationService(
        ai_provider=MockAIProvider(),
        db_session_factory=AsyncMock(),
        message_threshold=50
    )

    mock_db = AsyncMock()

    # Thread with enough messages
    thread1 = ChatThread(
        id="thread-1",
        message_count=55,
        is_summarized=False
    )
    assert await service.should_summarize(thread1, mock_db) is True

    # Thread with not enough messages
    thread2 = ChatThread(
        id="thread-2",
        message_count=45,
        is_summarized=False
    )
    assert await service.should_summarize(thread2, mock_db) is False

    # Already summarized thread
    thread3 = ChatThread(
        id="thread-3",
        message_count=100,
        is_summarized=True
    )
    assert await service.should_summarize(thread3, mock_db) is False


@pytest.mark.asyncio
async def test_should_summarize_with_recent_summary():
    """Test that recent summaries prevent new summarization"""
    service = SummarizationService(
        ai_provider=MockAIProvider(),
        db_session_factory=AsyncMock()
    )

    mock_db = AsyncMock()

    # Mock recent summary query
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = ChatSummary(
        id="summary-1",
        created_at=datetime.utcnow() - timedelta(hours=1)  # Recent
    )
    mock_db.execute.return_value = mock_result

    thread = ChatThread(
        id="thread-1",
        message_count=100,
        is_summarized=False
    )

    assert await service.should_summarize(thread, mock_db) is False


@pytest.mark.asyncio
async def test_summarize_thread_success(summarization_service):
    """Test successful thread summarization"""
    mock_db = AsyncMock()
    thread_id = "thread-123"

    # Mock thread query
    mock_thread = ChatThread(
        id=thread_id,
        message_count=75,
        is_summarized=False
    )
    thread_result = MagicMock()
    thread_result.scalar_one_or_none.return_value = mock_thread

    # Mock messages query
    messages = [
        ChatMessage(
            id=f"msg-{i}",
            content=f"Message {i} content",
            is_user=i % 2 == 0,
            created_at=datetime.utcnow() - timedelta(minutes=i)
        )
        for i in range(60)
    ]
    messages_result = MagicMock()
    messages_result.scalars.return_value.all.return_value = messages

    # Setup execute to return different results
    mock_db.execute.side_effect = [thread_result, messages_result]

    # Mock conversation manager
    mock_conv_manager = AsyncMock()
    mock_conv_manager.get_thread_summary.return_value = (
        "Summary: Detailed discussion about Python debugging"
    )

    with patch.object(summarization_service, '_get_or_create_conversation_manager',
                     return_value=mock_conv_manager):
        # Summarize thread
        summary = await summarization_service.summarize_thread(
            thread_id=thread_id,
            session=mock_db
        )

        assert summary is not None
        assert summary.thread_id == thread_id
        assert "Python debugging" in summary.summary_text
        assert summary.message_count == 60

        # Verify thread was marked as summarized
        assert mock_thread.is_summarized is True
        assert mock_thread.summary_count == 1

        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_summarize_nonexistent_thread(summarization_service):
    """Test summarization of non-existent thread"""
    mock_db = AsyncMock()

    # Mock empty thread query
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    summary = await summarization_service.summarize_thread(
        thread_id="non-existent",
        session=mock_db
    )

    assert summary is None
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_get_summary_for_context():
    """Test retrieving summaries for conversation context"""
    service = SummarizationService(
        ai_provider=MockAIProvider(),
        db_session_factory=AsyncMock()
    )

    mock_db = AsyncMock()

    # Mock summaries
    summaries = [
        ChatSummary(
            id="summary-1",
            summary_text="First discussion about Python basics",
            created_at=datetime.utcnow() - timedelta(days=2)
        ),
        ChatSummary(
            id="summary-2",
            summary_text="Advanced Python topics and debugging",
            created_at=datetime.utcnow() - timedelta(days=1)
        )
    ]

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = summaries
    mock_db.execute.return_value = mock_result

    # Get summary context
    context = await service.get_summary_for_context(
        thread_id="thread-123",
        session=mock_db,
        max_summaries=5
    )

    assert len(context) == 2
    assert "Python basics" in context[0].summary_text
    assert "debugging" in context[1].summary_text

    # Verify they're ordered by date (oldest first for context)
    assert context[0].created_at < context[1].created_at


@pytest.mark.asyncio
async def test_incremental_summarization():
    """Test incremental summarization with existing summaries"""
    service = SummarizationService(
        ai_provider=MockAIProvider(),
        db_session_factory=AsyncMock(),
        incremental_threshold=25  # Lower threshold for incremental
    )

    mock_db = AsyncMock()

    # Mock thread with existing summary
    mock_thread = ChatThread(
        id="thread-123",
        message_count=80,
        is_summarized=True,
        summary_count=1
    )
    thread_result = MagicMock()
    thread_result.scalar_one_or_none.return_value = mock_thread

    # Mock last summary
    last_summary = ChatSummary(
        id="summary-1",
        message_count=50,
        created_at=datetime.utcnow() - timedelta(hours=2)
    )
    summary_result = MagicMock()
    summary_result.scalar_one_or_none.return_value = last_summary

    # Mock new messages (30 messages since last summary)
    new_messages = [
        ChatMessage(
            id=f"msg-{i}",
            content=f"New message {i}",
            is_user=i % 2 == 0
        )
        for i in range(30)
    ]
    messages_result = MagicMock()
    messages_result.scalars.return_value.all.return_value = new_messages

    mock_db.execute.side_effect = [
        thread_result,
        summary_result,
        messages_result
    ]

    # Test should_summarize with incremental
    should_summarize = await service.should_summarize(mock_thread, mock_db)
    assert should_summarize is True  # 30 new messages > 25 threshold


@pytest.mark.asyncio
async def test_summary_error_handling(summarization_service):
    """Test error handling during summarization"""
    mock_db = AsyncMock()

    # Mock thread
    mock_thread = ChatThread(id="thread-123")
    thread_result = MagicMock()
    thread_result.scalar_one_or_none.return_value = mock_thread

    # Mock messages
    messages = [ChatMessage(id="msg-1", content="Test")]
    messages_result = MagicMock()
    messages_result.scalars.return_value.all.return_value = messages

    mock_db.execute.side_effect = [thread_result, messages_result]

    # Make AI provider fail
    summarization_service.ai_provider.complete = AsyncMock(
        side_effect=Exception("AI service error")
    )

    # Should handle error gracefully
    summary = await summarization_service.summarize_thread(
        thread_id="thread-123",
        session=mock_db
    )

    assert summary is None
    mock_db.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_key_topics_extraction():
    """Test extraction of key topics from summary"""
    # Create AI provider that returns structured summary
    mock_ai = MockAIProvider(
        response="""Summary: Discussion covered Python debugging, unit testing,
        and performance optimization.

        Key topics: debugging, testing, optimization, best practices"""
    )

    service = SummarizationService(
        ai_provider=mock_ai,
        db_session_factory=AsyncMock()
    )

    mock_db = AsyncMock()

    # Mock thread and messages
    mock_thread = ChatThread(id="thread-123")
    thread_result = MagicMock()
    thread_result.scalar_one_or_none.return_value = mock_thread

    messages = [
        ChatMessage(content="How do I debug Python?", is_user=True),
        ChatMessage(content="Here are debugging tips...", is_user=False)
    ]
    messages_result = MagicMock()
    messages_result.scalars.return_value.all.return_value = messages

    mock_db.execute.side_effect = [thread_result, messages_result]

    # Summarize
    summary = await service.summarize_thread("thread-123", mock_db)

    assert summary is not None
    assert len(summary.key_topics) > 0
    assert "debugging" in summary.key_topics
    assert "testing" in summary.key_topics
```

### Configuration Files

#### **docker-compose.yml**
*Purpose: Docker compose configuration for development*

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: deepseek_postgres
    environment:
      POSTGRES_USER: ${DB_USER:-deepseek}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-deepseek_password}
      POSTGRES_DB: ${DB_NAME:-deepseek_db}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-deepseek}"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Redis for caching and rate limiting
  redis:
    image: redis:7-alpine
    container_name: deepseek_redis
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    container_name: deepseek_backend
    environment:
      - DATABASE_URL=postgresql+asyncpg://${DB_USER:-deepseek}:${DB_PASSWORD:-deepseek_password}@postgres:5432/${DB_NAME:-deepseek_db}
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY:-your-secret-key-here}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - backend_uploads:/app/uploads
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: |
      sh -c "
        echo 'Waiting for database...' &&
        alembic upgrade head &&
        echo 'Starting development server...' &&
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-config=logging.yaml
      "

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    container_name: deepseek_frontend
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000
      - WATCHPACK_POLLING=true
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend
    command: npm run dev -- --host

  # Nginx (for production-like setup)
  nginx:
    image: nginx:alpine
    container_name: deepseek_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend
    profiles:
      - production

  # MinIO for S3-compatible storage (optional)
  minio:
    image: minio/minio:latest
    container_name: deepseek_minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    profiles:
      - storage

  # Prometheus for monitoring (optional)
  prometheus:
    image: prom/prometheus:latest
    container_name: deepseek_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    profiles:
      - monitoring

  # Grafana for visualization (optional)
  grafana:
    image: grafana/grafana:latest
    container_name: deepseek_grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    profiles:
      - monitoring

volumes:
  postgres_data:
  redis_data:
  backend_uploads:
  minio_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: deepseek_network
```

#### **backend/alembic.ini**
*Purpose: Alembic database migration configuration*

```ini
# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = alembic

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
# Uncomment the line below if you want the files to be prepended with date and time
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library
# one of: postgresql, mysql, sqlite, mssql
# leave blank for auto-detection
timezone = UTC

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; This defaults
# to alembic/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path.
# The path separator used here should be the separator specified by "version_path_separator" below.
# version_locations = %(here)s/bar:%(here)s/bat:alembic/versions

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses os.pathsep.
# If this key is omitted entirely, it falls back to the legacy behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os  # Use os.pathsep.
# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

# Configure SQLAlchemy URL from environment
sqlalchemy.url = postgresql+asyncpg://deepseek:deepseek_password@localhost:5432/deepseek_db


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 88

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

#### **backend/pyproject.toml**
*Purpose: Python project configuration*

```toml
[tool.poetry]
name = "deepseek-assistant"
version = "0.1.0"
description = "AI Assistant with DeepSeek R1 integration"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
python = "^3.11"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
sqlalchemy = "^2.0.23"
asyncpg = "^0.29.0"
alembic = "^1.12.1"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
passlib = {extras = ["bcrypt"], version = "^1.7.4"}
python-multipart = "^0.0.6"
aiofiles = "^23.2.1"
httpx = "^0.25.2"
redis = "^5.0.1"
celery = "^5.3.4"
openai = "^1.3.7"
tiktoken = "^0.5.1"
langchain = "^0.0.340"
websockets = "^12.0"
prometheus-client = "^0.19.0"
sentry-sdk = {extras = ["fastapi"], version = "^1.38.0"}
structlog = "^23.2.0"
python-json-logger = "^2.0.7"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
pytest-cov = "^4.1.0"
black = "^23.11.0"
isort = "^5.12.0"
flake8 = "^6.1.0"
mypy = "^1.7.1"
pre-commit = "^3.5.0"
factory-boy = "^3.3.0"
faker = "^20.1.0"
httpx = "^0.25.2"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | migrations
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88
skip_gitignore = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
asyncio_mode = "auto"
addopts = [
    "--verbose",
    "--strict-markers",
    "--tb=short",
    "--cov=app",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]

[tool.coverage.run]
source = ["app"]
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__init__.py",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

#### **frontend/package.json**
*Purpose: Frontend package configuration*

```json
{
  "name": "deepseek-assistant-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "type-check": "tsc --noEmit",
    "storybook": "storybook dev -p 6006",
    "build-storybook": "storybook build"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.8.4",
    "@tanstack/react-query-devtools": "^5.8.4",
    "axios": "^1.6.2",
    "clsx": "^2.0.0",
    "date-fns": "^2.30.0",
    "framer-motion": "^10.16.5",
    "lodash-es": "^4.17.21",
    "lucide-react": "^0.292.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-hook-form": "^7.48.2",
    "react-hot-toast": "^2.4.1",
    "react-intersection-observer": "^9.5.3",
    "react-markdown": "^9.0.1",
    "react-router-dom": "^6.20.0",
    "react-syntax-highlighter": "^15.5.0",
    "react-use": "^17.4.2",
    "remark-gfm": "^4.0.0",
    "tailwind-merge": "^2.1.0",
    "uuid": "^9.0.1",
    "zustand": "^4.4.7"
  },
  "devDependencies": {
    "@storybook/addon-essentials": "^7.6.3",
    "@storybook/addon-interactions": "^7.6.3",
    "@storybook/addon-links": "^7.6.3",
    "@storybook/blocks": "^7.6.3",
    "@storybook/react": "^7.6.3",
    "@storybook/react-vite": "^7.6.3",
    "@storybook/testing-library": "^0.2.2",
    "@testing-library/jest-dom": "^6.1.5",
    "@testing-library/react": "^14.1.2",
    "@testing-library/user-event": "^14.5.1",
    "@types/lodash-es": "^4.17.12",
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@types/react-syntax-highlighter": "^15.5.11",
    "@types/uuid": "^9.0.7",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "@vitejs/plugin-react": "^4.2.1",
    "@vitest/coverage-v8": "^1.0.4",
    "@vitest/ui": "^1.0.4",
    "autoprefixer": "^10.4.16",
    "eslint": "^8.55.0",
    "eslint-config-prettier": "^9.1.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "eslint-plugin-storybook": "^0.6.15",
    "jsdom": "^23.0.1",
    "postcss": "^8.4.32",
    "prettier": "^3.1.1",
    "prettier-plugin-tailwindcss": "^0.5.9",
    "storybook": "^7.6.3",
    "tailwindcss": "^3.3.6",
    "typescript": "^5.2.2",
    "vite": "^5.0.8",
    "vitest": "^1.0.4"
  }
}
```

### Scripts

#### **scripts/init_db.sql**
*Purpose: Initial database schema*

```sql
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create custom types
CREATE TYPE user_role AS ENUM ('user', 'admin', 'moderator');
CREATE TYPE project_status AS ENUM ('active', 'archived', 'deleted');
CREATE TYPE message_status AS ENUM ('pending', 'sent', 'failed');

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role) WHERE is_active = true;

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status project_status DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(user_id, name)
);

CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_created_at ON projects(created_at DESC);

-- Chat threads table
CREATE TABLE IF NOT EXISTS chat_threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'New Chat',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    is_archived BOOLEAN DEFAULT false,
    archived_at TIMESTAMP WITH TIME ZONE,
    is_summarized BOOLEAN DEFAULT false,
    summary_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_chat_threads_project_id ON chat_threads(project_id);
CREATE INDEX idx_chat_threads_user_id ON chat_threads(user_id);
CREATE INDEX idx_chat_threads_last_activity ON chat_threads(last_activity_at DESC);
CREATE INDEX idx_chat_threads_archived ON chat_threads(is_archived, project_id);

-- Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    is_user BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_edited BOOLEAN DEFAULT false,
    edited_at TIMESTAMP WITH TIME ZONE,
    is_deleted BOOLEAN DEFAULT false,
    deleted_at TIMESTAMP WITH TIME ZONE,
    token_count INTEGER DEFAULT 0,
    model_used VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb,
    edit_history JSONB DEFAULT '[]'::jsonb
);

CREATE INDEX idx_chat_messages_thread_id ON chat_messages(thread_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);
CREATE INDEX idx_chat_messages_user_id ON chat_messages(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_chat_messages_deleted ON chat_messages(is_deleted, thread_id);

-- Chat summaries table
CREATE TABLE IF NOT EXISTS chat_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL,
    key_topics TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    start_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    end_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    message_count INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_chat_summaries_thread_id ON chat_summaries(thread_id);
CREATE INDEX idx_chat_summaries_created_at ON chat_summaries(created_at DESC);

-- File uploads table
CREATE TABLE IF NOT EXISTS file_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    message_id UUID REFERENCES chat_messages(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    upload_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_file_uploads_user_id ON file_uploads(user_id);
CREATE INDEX idx_file_uploads_project_id ON file_uploads(project_id) WHERE project_id IS NOT NULL;
CREATE INDEX idx_file_uploads_message_id ON file_uploads(message_id) WHERE message_id IS NOT NULL;

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active, expires_at);

-- User settings table
CREATE TABLE IF NOT EXISTS user_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(50) DEFAULT 'system',
    language VARCHAR(10) DEFAULT 'en',
    notifications_enabled BOOLEAN DEFAULT true,
    ai_model_preference VARCHAR(100) DEFAULT 'gpt-4',
    default_temperature FLOAT DEFAULT 0.7,
    default_max_tokens INTEGER DEFAULT 2048,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    settings JSONB DEFAULT '{}'::jsonb
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_threads_updated_at BEFORE UPDATE ON chat_threads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_messages_updated_at BEFORE UPDATE ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON user_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for thread statistics update
CREATE OR REPLACE FUNCTION update_thread_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE chat_threads
        SET message_count = message_count + 1,
            total_tokens = total_tokens + NEW.token_count,
            last_activity_at = CURRENT_TIMESTAMP
        WHERE id = NEW.thread_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE chat_threads
        SET message_count = message_count - 1,
            total_tokens = total_tokens - OLD.token_count
        WHERE id = OLD.thread_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_thread_stats_trigger
AFTER INSERT OR DELETE ON chat_messages
FOR EACH ROW EXECUTE FUNCTION update_thread_stats();

-- Sample data for development (optional)
-- Uncomment to insert sample data

/*
-- Insert sample user
INSERT INTO users (email, username, hashed_password, full_name, role)
VALUES ('demo@example.com', 'demo', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'Demo User', 'user');

-- Insert sample project
INSERT INTO projects (name, description, user_id)
SELECT 'Sample Project', 'A sample project for testing', id
FROM users WHERE username = 'demo';

-- Insert sample settings
INSERT INTO user_settings (user_id)
SELECT id FROM users WHERE username = 'demo';
*/
```

#### **scripts/setup.sh**
*Purpose: Development environment setup script*

```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up DeepSeek Assistant development environment...${NC}"

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed. Please install it first.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ $1 is installed${NC}"
    fi
}

echo -e "\n${YELLOW}Checking prerequisites...${NC}"
check_command "docker"
check_command "docker-compose"
check_command "node"
check_command "npm"
check_command "python3"

# Create necessary directories
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p backend/{alembic/versions,api,core,models,schemas,services,tests,uploads}
mkdir -p frontend/{src/{api,components,hooks,pages,stores,types,utils},public}
mkdir -p nginx/ssl
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p scripts

# Copy environment template
echo -e "\n${YELLOW}Setting up environment files...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file from template${NC}"
    echo -e "${YELLOW}⚠️  Please update .env with your API keys and configuration${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists, skipping...${NC}"
fi

# Generate secret key
if grep -q "your-secret-key-here" .env; then
    SECRET_KEY=$(openssl rand -hex 32)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/your-secret-key-here/$SECRET_KEY/g" .env
    else
        sed -i "s/your-secret-key-here/$SECRET_KEY/g" .env
    fi
    echo -e "${GREEN}✅ Generated new secret key${NC}"
fi

# Setup Python virtual environment
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Created Python virtual environment${NC}"
fi

# Install Python dependencies
source venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
echo -e "${GREEN}✅ Installed Python dependencies${NC}"

# Setup pre-commit hooks
pre-commit install
echo -e "${GREEN}✅ Installed pre-commit hooks${NC}"

cd ..

# Install frontend dependencies
echo -e "\n${YELLOW}Installing frontend dependencies...${NC}"
cd frontend
npm install
echo -e "${GREEN}✅ Installed frontend dependencies${NC}"
cd ..

# Start Docker services
echo -e "\n${YELLOW}Starting Docker services...${NC}"
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
until docker-compose exec -T postgres pg_isready -U deepseek &> /dev/null; do
    echo -n "."
    sleep 1
done
echo -e "\n${GREEN}✅ PostgreSQL is ready${NC}"

# Run database migrations
echo -e "\n${YELLOW}Running database migrations...${NC}"
cd backend
source venv/bin/activate
alembic upgrade head
echo -e "${GREEN}✅ Database migrations completed${NC}"
cd ..

# Create sample data (optional)
read -p "Do you want to create sample data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd backend
    python scripts/create_sample_data.py
    echo -e "${GREEN}✅ Sample data created${NC}"
    cd ..
fi

# Display status
echo -e "\n${GREEN}🎉 Setup complete!${NC}"
echo -e "\n${YELLOW}To start the development environment:${NC}"
echo -e "  ${GREEN}docker-compose up${NC}"
echo -e "\n${YELLOW}The application will be available at:${NC}"
echo -e "  Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "  Backend API: ${GREEN}http://localhost:8000${NC}"
echo -e "  API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "\n${YELLOW}Default credentials (if sample data was created):${NC}"
echo -e "  Email: ${GREEN}demo@example.com${NC}"
echo -e "  Password: ${GREEN}secret${NC}"
echo -e "\n${RED}⚠️  Don't forget to:${NC}"
echo -e "  1. Update your .env file with your OpenAI API key"
echo -e "  2. Review and update security settings before deploying"
echo -e "  3. Set up SSL certificates for production"
```

This completes the implementation of the DeepSeek R1-powered AI assistant! The codebase is modular, well-tested, and ready for both development and production deployment. Each component is focused on a specific responsibility, making the system maintainable and extensible.
