# Phase 3: Core Chat Experience - Detailed Implementation Specification

## Phase Summary
Transform the project-organized workspace into a powerful AI chat interface with real-time streaming, context-aware conversations, and intelligent summarization. This phase delivers the primary value proposition: seamless AI-assisted productivity within organized project contexts.

## 1. User Interface Components

### 1.1 Chat Container Components

#### **ChatView Component**
- **Purpose**: Main chat interface container
- **Route**: `/projects/:projectId/chat/:threadId?`
- **Props**:
  ```typescript
  interface ChatViewProps {
    projectId: string;
    threadId?: string;
    onThreadChange?: (threadId: string) => void;
  }
  ```
- **State Management**:
  ```typescript
  interface ChatViewState {
    messages: ChatMessage[];
    isLoading: boolean;
    isStreaming: boolean;
    streamingMessageId: string | null;
    error: string | null;
    connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
    typingIndicator: boolean;
    hasMore: boolean;
    focusedMessageId: string | null;
  }
  ```
- **Zustand Store Integration**:
  ```typescript
  interface ChatStore {
    threads: Map<string, ChatThread>;
    activeThreadId: string | null;
    messages: Map<string, ChatMessage[]>;
    isLoadingThread: boolean;
    streamingMessage: Partial<ChatMessage> | null;

    // Actions
    loadThread: (threadId: string) => Promise<void>;
    createThread: (projectId: string, initialMessage?: string) => Promise<ChatThread>;
    sendMessage: (content: string, attachments?: File[]) => Promise<void>;
    editMessage: (messageId: string, content: string) => Promise<void>;
    deleteMessage: (messageId: string) => Promise<void>;
    regenerateResponse: (messageId: string) => Promise<void>;
  }
  ```
- **Layout Structure**:
  1. Header: Project name, thread title, summary status
  2. Messages Container: Virtualized scroll area
  3. Input Area: ChatInputBar component
  4. Sidebar Toggle: Show/hide RecentChatsSidebar
- **Key Behaviors**:
  - Auto-scroll to bottom on new messages (unless user scrolled up)
  - Lazy load older messages on scroll top
  - Show "New messages" indicator when scrolled up
  - Maintain scroll position during message edits
  - Smooth transition when switching threads
- **WebSocket Integration**:
  ```typescript
  useEffect(() => {
    const ws = useWebSocket({
      url: `/ws/chat/${threadId}`,
      onMessage: handleIncomingMessage,
      onStreamChunk: handleStreamChunk,
      onStatusChange: setConnectionStatus,
      reconnectAttempts: 5,
      heartbeatInterval: 30000
    });

    return () => ws.close();
  }, [threadId]);
  ```
- **Accessibility**:
  - Live region for new messages
  - Keyboard navigation between messages
  - Screen reader announcements for streaming status
  - Focus management on thread switch
- **Responsive Design**:
  - Mobile: Full screen with slide-out sidebar
  - Tablet: Collapsible sidebar (300px when open)
  - Desktop: Persistent sidebar (350px)

#### **MessageBubble Component**
- **Purpose**: Individual message display
- **Props**:
  ```typescript
  interface MessageBubbleProps {
    message: ChatMessage;
    isStreaming?: boolean;
    isLast?: boolean;
    onEdit?: (messageId: string) => void;
    onDelete?: (messageId: string) => void;
    onRegenerate?: () => void;
    onCopy?: () => void;
    focused?: boolean;
  }
  ```
- **Local State**:
  ```typescript
  interface MessageBubbleState {
    isEditing: boolean;
    editContent: string;
    showActions: boolean;
    copySuccess: boolean;
    isCollapsed: boolean; // For long messages
  }
  ```
- **Visual Design**:
  - User messages: Right-aligned, primary color background
  - Assistant messages: Left-aligned, secondary color background
  - System messages: Center-aligned, muted style
  - Avatar icons: User photo or initials, AI logo
  - Timestamp: Relative time on hover/focus
- **Message Actions** (on hover/focus):
  - Copy to clipboard
  - Edit (user messages only)
  - Delete
  - Regenerate (assistant messages only)
  - Pin/Bookmark (Phase 4)
- **Content Rendering**:
  - Markdown support with syntax highlighting
  - LaTeX math rendering
  - Code blocks with copy button
  - Tables with horizontal scroll
  - Collapsible long messages (>1000 chars)
- **Streaming Animation**:
  - Typing indicator (3 dots)
  - Character-by-character reveal
  - Smooth scroll during streaming
  - Cursor blink at stream end

#### **ChatInputBar Component**
- **Purpose**: Message composition interface
- **Props**:
  ```typescript
  interface ChatInputBarProps {
    onSend: (content: string, attachments?: File[]) => void;
    onTyping?: () => void;
    isDisabled?: boolean;
    placeholder?: string;
    projectContext?: ProjectContext;
    suggestions?: string[];
  }
  ```
- **State**:
  ```typescript
  interface ChatInputState {
    content: string;
    attachments: File[];
    isUploading: boolean;
    showSuggestions: boolean;
    selectedSuggestionIndex: number;
    textareaHeight: number;
  }
  ```
- **Features**:
  1. **Auto-expanding Textarea**:
     - Min height: 56px (2 lines)
     - Max height: 200px (8 lines)
     - Smooth height transitions
  2. **Attachment Support** (Phase 3.5):
     - Drag & drop zone
     - File type validation
     - Preview thumbnails
     - Upload progress
  3. **Smart Suggestions**:
     - Project-specific prompts
     - Recent queries
     - Typing triggers (@, /, #)
  4. **Keyboard Shortcuts**:
     - Cmd/Ctrl + Enter: Send message
     - Shift + Enter: New line
     - Up arrow (empty): Edit last message
     - Esc: Clear input or close suggestions
     - Tab: Accept suggestion
- **Mobile Adaptations**:
  - Fixed bottom position
  - Larger touch targets (44px min)
  - Virtual keyboard aware
  - Voice input button

#### **StreamingText Component**
- **Purpose**: Animated text reveal for AI responses
- **Props**:
  ```typescript
  interface StreamingTextProps {
    text: string;
    isComplete: boolean;
    speed?: number; // chars per second
    onComplete?: () => void;
  }
  ```
- **Implementation**:
  ```typescript
  const StreamingText: React.FC<StreamingTextProps> = ({
    text,
    isComplete,
    speed = 30
  }) => {
    const [displayedText, setDisplayedText] = useState('');
    const [currentIndex, setCurrentIndex] = useState(0);

    useEffect(() => {
      if (isComplete) {
        setDisplayedText(text);
        return;
      }

      const interval = setInterval(() => {
        if (currentIndex < text.length) {
          setDisplayedText(text.slice(0, currentIndex + 1));
          setCurrentIndex(i => i + 1);
        }
      }, 1000 / speed);

      return () => clearInterval(interval);
    }, [text, currentIndex, isComplete, speed]);

    return (
      <div className="streaming-text">
        <Markdown content={displayedText} />
        {!isComplete && <span className="cursor-blink">▊</span>}
      </div>
    );
  };
  ```

### 1.2 Chat Navigation Components

#### **RecentChatsSidebar Component**
- **Purpose**: Thread navigation and history
- **Props**:
  ```typescript
  interface RecentChatsSidebarProps {
    projectId: string;
    activeThreadId?: string;
    onThreadSelect: (threadId: string) => void;
    onNewChat: () => void;
    isCollapsed?: boolean;
  }
  ```
- **State**:
  ```typescript
  interface SidebarState {
    threads: ChatThread[];
    searchQuery: string;
    isLoading: boolean;
    groupBy: 'date' | 'none';
  }
  ```
- **Thread Display**:
  ```typescript
  interface ThreadItem {
    id: string;
    title: string; // First message or AI-generated
    lastMessage: string;
    lastMessageAt: Date;
    messageCount: number;
    hasUnread: boolean;
    isSummarized: boolean;
  }
  ```
- **Grouping Logic**:
  - Today
  - Yesterday
  - This Week
  - This Month
  - Older
- **Features**:
  - Search threads by content
  - Sort by recent/oldest
  - Quick actions (rename, delete)
  - Unread indicator
  - Summary badge
- **Mobile Behavior**:
  - Slide from left
  - Overlay with backdrop
  - Swipe to dismiss

#### **ThreadHeader Component**
- **Purpose**: Display thread context and actions
- **Props**:
  ```typescript
  interface ThreadHeaderProps {
    thread: ChatThread;
    project: Project;
    onRename: () => void;
    onSummarize: () => void;
    onExport: () => void;
  }
  ```
- **Elements**:
  - Thread title (editable)
  - Message count
  - Last activity time
  - Summary status/button
  - Export menu (PDF, Markdown, Text)
- **Summary Indicator**:
  - "Summarized" badge with icon
  - Time since last summary
  - "Needs summary" hint (>50 messages)

### 1.3 Supporting Components

#### **ExamplePromptsInline Component**
- **Purpose**: Contextual prompt suggestions in chat
- **Props**:
  ```typescript
  interface ExamplePromptsInlineProps {
    projectType?: string;
    threadContext?: string[];
    onPromptSelect: (prompt: string) => void;
    maxPrompts?: number;
  }
  ```
- **Dynamic Prompts**:
  - Based on project template
  - Based on recent messages
  - Based on time of day
  - Based on detected intent
- **Display Rules**:
  - Show when thread is empty
  - Show after long pause (5 min)
  - Hide during active conversation
  - Fade animation on appear/dismiss

#### **ConnectionStatus Component**
- **Purpose**: WebSocket connection indicator
- **Props**:
  ```typescript
  interface ConnectionStatusProps {
    status: 'connecting' | 'connected' | 'disconnected' | 'error';
    onReconnect?: () => void;
  }
  ```
- **Visual States**:
  - Connected: Hidden or subtle indicator
  - Connecting: Pulsing yellow dot
  - Disconnected: Red banner with retry button
  - Error: Red banner with error message
- **Auto-reconnect**: Exponential backoff (1s, 2s, 4s, 8s, 16s)

#### **MessageActions Component**
- **Purpose**: Floating action menu for messages
- **Props**:
  ```typescript
  interface MessageActionsProps {
    messageId: string;
    messageType: 'user' | 'assistant';
    onAction: (action: MessageAction) => void;
    position: { x: number; y: number };
  }
  ```
- **Actions by Type**:
  - User messages: Edit, Delete, Copy
  - Assistant messages: Regenerate, Copy, Report
- **Positioning**: Smart positioning to stay in viewport
- **Keyboard**: Navigate with arrow keys

## 2. Backend Services

### 2.1 AI Provider Service
```python
# services/ai_provider.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI
import tiktoken

@dataclass
class AIMessage:
    role: str  # 'user', 'assistant', 'system'
    content: str
    name: Optional[str] = None

@dataclass
class AIResponse:
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str

class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None] | AIResponse:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        pass

class OpenAIProvider(AIProvider):
    """OpenAI implementation of AI provider"""

    def __init__(self, api_key: str, model: str = "gpt-4", max_tokens: int = 4096):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._encoder = tiktoken.encoding_for_model(model)

    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None] | AIResponse:
        """Generate completion from messages"""

        # Convert to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        if stream:
            return self._stream_completion(openai_messages, temperature, **kwargs)
        else:
            return await self._complete(openai_messages, temperature, **kwargs)

    async def _stream_completion(
        self,
        messages: List[Dict],
        temperature: float,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion chunks"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _complete(
        self,
        messages: List[Dict],
        temperature: float,
        **kwargs
    ) -> AIResponse:
        """Get complete response at once"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            stream=False,
            **kwargs
        )

        return AIResponse(
            content=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason,
            usage=response.usage.dict(),
            model=response.model
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self._encoder.encode(text))

    def get_max_tokens(self) -> int:
        """Get max tokens for model"""
        model_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        return model_limits.get(self.model, 4096)

class AIProviderFactory:
    """Factory for creating AI providers"""

    @staticmethod
    def create(provider_type: str, config: Dict[str, Any]) -> AIProvider:
        """Create AI provider instance"""
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=config["api_key"],
                model=config.get("model", "gpt-4"),
                max_tokens=config.get("max_tokens", 4096)
            )
        # Future providers: Anthropic, Cohere, etc.
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

# Conversation context management
class ConversationManager:
    """Manages conversation context and token limits"""

    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.max_context_tokens = int(ai_provider.get_max_tokens() * 0.75)

    def prepare_messages(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> List[AIMessage]:
        """Prepare messages for AI provider with token management"""
        ai_messages = []

        # Add system prompt if provided
        if system_prompt:
            ai_messages.append(AIMessage(role="system", content=system_prompt))

        # Convert chat messages to AI messages
        total_tokens = self.ai_provider.count_tokens(system_prompt or "")

        # Add messages in reverse order until token limit
        for message in reversed(messages):
            msg_tokens = self.ai_provider.count_tokens(message.content)
            if total_tokens + msg_tokens > self.max_context_tokens:
                break

            ai_messages.insert(
                1 if system_prompt else 0,
                AIMessage(
                    role="user" if message.is_user else "assistant",
                    content=message.content
                )
            )
            total_tokens += msg_tokens

        return ai_messages

    async def get_thread_summary(
        self,
        messages: List[ChatMessage],
        max_length: int = 500
    ) -> str:
        """Generate summary of conversation thread"""
        summary_prompt = f"""
        Summarize this conversation in {max_length} characters or less.
        Focus on key topics, decisions, and outcomes.
        Use bullet points for clarity.
        """

        summary_messages = [
            AIMessage(role="system", content=summary_prompt)
        ]

        # Add conversation messages
        for msg in messages[:50]:  # Limit to first 50 messages
            summary_messages.append(
                AIMessage(
                    role="user" if msg.is_user else "assistant",
                    content=msg.content[:500]  # Truncate long messages
                )
            )

        response = await self.ai_provider.complete(
            summary_messages,
            stream=False,
            temperature=0.3,
            max_tokens=200
        )

        return response.content
```

### 2.2 WebSocket Manager
```python
# services/websocket_manager.py
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from collections import defaultdict

class ConnectionManager:
    """Manages WebSocket connections and message routing"""

    def __init__(self):
        # Map of thread_id -> set of connections
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Map of connection_id -> user_id
        self.connection_users: Dict[str, str] = {}
        # Map of connection_id -> last_heartbeat
        self.heartbeats: Dict[str, datetime] = {}
        # Rate limiting
        self.message_counts: Dict[str, list] = defaultdict(list)

        # Start heartbeat monitor
        asyncio.create_task(self._monitor_heartbeats())

    async def connect(
        self,
        websocket: WebSocket,
        thread_id: str,
        user_id: str
    ) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        self.active_connections[thread_id].add(websocket)
        self.connection_users[connection_id] = user_id
        self.heartbeats[connection_id] = datetime.utcnow()

        # Send connection established message
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "thread_id": thread_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        return connection_id

    async def disconnect(self, connection_id: str, thread_id: str, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections[thread_id].discard(websocket)
        if not self.active_connections[thread_id]:
            del self.active_connections[thread_id]

        self.connection_users.pop(connection_id, None)
        self.heartbeats.pop(connection_id, None)
        self.message_counts.pop(connection_id, None)

    async def send_message(
        self,
        thread_id: str,
        message: Dict,
        exclude_connection: Optional[str] = None
    ):
        """Broadcast message to all connections in a thread"""
        if thread_id not in self.active_connections:
            return

        disconnected = set()
        for websocket in self.active_connections[thread_id]:
            try:
                # Skip excluded connection (sender)
                conn_id = self._get_connection_id(websocket)
                if conn_id == exclude_connection:
                    continue

                await websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                print(f"Error sending message: {e}")
                disconnected.add(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            self.active_connections[thread_id].discard(ws)

    async def send_stream_chunk(
        self,
        thread_id: str,
        message_id: str,
        chunk: str,
        is_final: bool = False
    ):
        """Send streaming chunk to all connections"""
        message = {
            "type": "stream_chunk",
            "message_id": message_id,
            "chunk": chunk,
            "is_final": is_final,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_message(thread_id, message)

    async def handle_heartbeat(self, connection_id: str):
        """Update heartbeat timestamp"""
        self.heartbeats[connection_id] = datetime.utcnow()

    async def check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        # Clean old entries
        self.message_counts[connection_id] = [
            timestamp for timestamp in self.message_counts[connection_id]
            if timestamp > window_start
        ]

        # Check limit (60 messages per minute)
        if len(self.message_counts[connection_id]) >= 60:
            return False

        self.message_counts[connection_id].append(now)
        return True

    async def _monitor_heartbeats(self):
        """Monitor and disconnect stale connections"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds

            now = datetime.utcnow()
            timeout = timedelta(seconds=60)  # 60 second timeout

            stale_connections = [
                conn_id for conn_id, last_heartbeat in self.heartbeats.items()
                if now - last_heartbeat > timeout
            ]

            for conn_id in stale_connections:
                # Find and disconnect stale connection
                for thread_id, connections in self.active_connections.items():
                    for ws in connections:
                        if self._get_connection_id(ws) == conn_id:
                            await ws.close(code=1000, reason="Heartbeat timeout")
                            await self.disconnect(conn_id, thread_id, ws)

    def _get_connection_id(self, websocket: WebSocket) -> Optional[str]:
        """Get connection ID for a WebSocket"""
        # In practice, store this mapping when connecting
        # This is a simplified version
        for conn_id, user_id in self.connection_users.items():
            # Would need proper websocket -> connection_id mapping
            return conn_id
        return None

# WebSocket message handlers
class MessageHandler:
    """Handles different types of WebSocket messages"""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        chat_service: 'ChatService',
        ai_provider: AIProvider
    ):
        self.connection_manager = connection_manager
        self.chat_service = chat_service
        self.ai_provider = ai_provider

    async def handle_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict
    ):
        """Route message to appropriate handler"""
        message_type = message.get("type")

        handlers = {
            "heartbeat": self.handle_heartbeat,
            "send_message": self.handle_send_message,
            "edit_message": self.handle_edit_message,
            "delete_message": self.handle_delete_message,
            "typing_indicator": self.handle_typing_indicator,
            "regenerate": self.handle_regenerate
        }

        handler = handlers.get(message_type)
        if handler:
            await handler(websocket, connection_id, message)
        else:
            await websocket.send_json({
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            })

    async def handle_heartbeat(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict
    ):
        """Handle heartbeat message"""
        await self.connection_manager.handle_heartbeat(connection_id)
        await websocket.send_json({
            "type": "heartbeat_ack",
            "timestamp": datetime.utcnow().isoformat()
        })

    async def handle_send_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict
    ):
        """Handle new message from user"""
        # Check rate limit
        if not await self.connection_manager.check_rate_limit(connection_id):
            await websocket.send_json({
                "type": "error",
                "error": "Rate limit exceeded"
            })
            return

        # Extract message data
        thread_id = message["thread_id"]
        content = message["content"]
        user_id = self.connection_manager.connection_users[connection_id]

        # Save user message
        user_message = await self.chat_service.create_message(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            is_user=True
        )

        # Broadcast user message
        await self.connection_manager.send_message(
            thread_id,
            {
                "type": "new_message",
                "message": user_message.to_dict()
            },
            exclude_connection=connection_id
        )

        # Generate AI response
        await self.generate_ai_response(thread_id, user_message)

    async def generate_ai_response(
        self,
        thread_id: str,
        user_message: ChatMessage
    ):
        """Generate and stream AI response"""
        # Get conversation context
        messages = await self.chat_service.get_thread_messages(thread_id)

        # Create assistant message placeholder
        assistant_message = await self.chat_service.create_message(
            thread_id=thread_id,
            user_id=None,  # System message
            content="",
            is_user=False
        )

        # Notify clients of new message
        await self.connection_manager.send_message(
            thread_id,
            {
                "type": "assistant_message_start",
                "message_id": str(assistant_message.id)
            }
        )

        # Stream response
        full_response = ""
        conversation_manager = ConversationManager(self.ai_provider)
        ai_messages = conversation_manager.prepare_messages(messages)

        try:
            async for chunk in self.ai_provider.complete(ai_messages, stream=True):
                full_response += chunk
                await self.connection_manager.send_stream_chunk(
                    thread_id,
                    str(assistant_message.id),
                    chunk,
                    is_final=False
                )

            # Update message with full content
            await self.chat_service.update_message(
                assistant_message.id,
                content=full_response
            )

            # Send final chunk
            await self.connection_manager.send_stream_chunk(
                thread_id,
                str(assistant_message.id),
                "",
                is_final=True
            )

        except Exception as e:
            # Handle errors
            error_message = f"Error generating response: {str(e)}"
            await self.connection_manager.send_message(
                thread_id,
                {
                    "type": "error",
                    "message_id": str(assistant_message.id),
                    "error": error_message
                }
            )

            # Update message with error
            await self.chat_service.update_message(
                assistant_message.id,
                content=error_message,
                metadata={"error": True}
            )
```

### 2.3 Summarization Service
```python
# services/summarization_service.py
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.chat import ChatThread, ChatMessage, ChatSummary
from services.ai_provider import AIProvider, ConversationManager
import asyncio

class SummarizationService:
    """Service for generating and managing chat summaries"""

    def __init__(
        self,
        db: Session,
        ai_provider: AIProvider,
        conversation_manager: ConversationManager
    ):
        self.db = db
        self.ai_provider = ai_provider
        self.conversation_manager = conversation_manager

        # Summary thresholds
        self.message_threshold = 50  # Messages before auto-summary
        self.token_threshold = 10000  # Tokens before summary
        self.time_threshold = timedelta(hours=24)  # Time before summary

    async def should_summarize(self, thread: ChatThread) -> bool:
        """Check if thread needs summarization"""
        # Check if already summarized recently
        if thread.last_summary_at:
            time_since_summary = datetime.utcnow() - thread.last_summary_at
            if time_since_summary < self.time_threshold:
                return False

        # Check message count
        unsummarized_count = await self._count_unsummarized_messages(thread.id)
        if unsummarized_count >= self.message_threshold:
            return True

        # Check token count
        token_count = await self._estimate_thread_tokens(thread.id)
        if token_count >= self.token_threshold:
            return True

        return False

    async def summarize_thread(
        self,
        thread_id: str,
        force: bool = False
    ) -> Optional[ChatSummary]:
        """Generate summary for a thread"""
        thread = self.db.query(ChatThread).filter_by(id=thread_id).first()
        if not thread:
            return None

        # Check if summary needed
        if not force and not await self.should_summarize(thread):
            return None

        # Get messages to summarize
        messages = await self._get_messages_for_summary(thread_id)
        if not messages:
            return None

        # Generate summary
        summary_text = await self.conversation_manager.get_thread_summary(messages)

        # Extract key topics
        key_topics = await self._extract_key_topics(messages)

        # Save summary
        summary = ChatSummary(
            thread_id=thread_id,
            summary_text=summary_text,
            key_topics=key_topics,
            message_count=len(messages),
            start_message_id=messages[0].id,
            end_message_id=messages[-1].id,
            created_at=datetime.utcnow()
        )

        self.db.add(summary)

        # Update thread
        thread.last_summary_at = datetime.utcnow()
        thread.summary_count = (thread.summary_count or 0) + 1

        # Mark messages as summarized
        for msg in messages:
            msg.is_summarized = True

        await self.db.commit()

        return summary

    async def get_thread_context(
        self,
        thread_id: str,
        include_summaries: bool = True
    ) -> str:
        """Get context for thread including summaries"""
        context_parts = []

        # Get summaries if requested
        if include_summaries:
            summaries = self.db.query(ChatSummary)\
                .filter_by(thread_id=thread_id)\
                .order_by(ChatSummary.created_at)\
                .all()

            for summary in summaries:
                context_parts.append(
                    f"[Summary of messages {summary.start_message_id} to "
                    f"{summary.end_message_id}]:\n{summary.summary_text}\n"
                )

        # Get recent unsummarized messages
        recent_messages = self.db.query(ChatMessage)\
            .filter_by(thread_id=thread_id, is_summarized=False)\
            .order_by(ChatMessage.created_at.desc())\
            .limit(20)\
            .all()

        for msg in reversed(recent_messages):
            role = "User" if msg.is_user else "Assistant"
            context_parts.append(f"{role}: {msg.content}\n")

        return "\n".join(context_parts)

    async def schedule_auto_summarization(self):
        """Background task to auto-summarize threads"""
        while True:
            try:
                # Find threads needing summarization
                threads = self.db.query(ChatThread)\
                    .filter(
                        (ChatThread.message_count >= self.message_threshold) |
                        (ChatThread.last_activity_at <
                         datetime.utcnow() - self.time_threshold)
                    )\
                    .limit(10)\
                    .all()

                for thread in threads:
                    if await self.should_summarize(thread):
                        await self.summarize_thread(thread.id)

            except Exception as e:
                print(f"Error in auto-summarization: {e}")

            # Run every hour
            await asyncio.sleep(3600)

    async def _count_unsummarized_messages(self, thread_id: str) -> int:
        """Count messages not yet summarized"""
        return self.db.query(ChatMessage)\
            .filter_by(thread_id=thread_id, is_summarized=False)\
            .count()

    async def _estimate_thread_tokens(self, thread_id: str) -> int:
        """Estimate total tokens in unsummarized messages"""
        messages = self.db.query(ChatMessage)\
            .filter_by(thread_id=thread_id, is_summarized=False)\
            .all()

        total_tokens = 0
        for msg in messages:
            total_tokens += self.ai_provider.count_tokens(msg.content)

        return total_tokens

    async def _get_messages_for_summary(
        self,
        thread_id: str
    ) -> List[ChatMessage]:
        """Get messages that need summarization"""
        return self.db.query(ChatMessage)\
            .filter_by(thread_id=thread_id, is_summarized=False)\
            .order_by(ChatMessage.created_at)\
            .all()

    async def _extract_key_topics(
        self,
        messages: List[ChatMessage]
    ) -> List[str]:
        """Extract key topics from messages"""
        # Create a focused prompt for topic extraction
        topic_prompt = """
        Extract 3-5 key topics or themes from this conversation.
        Return only the topics as a comma-separated list.
        Be concise and specific.
        """

        # Prepare messages for topic extraction
        topic_messages = [
            {"role": "system", "content": topic_prompt}
        ]

        # Add conversation sample
        for msg in messages[:20]:  # First 20 messages
            topic_messages.append({
                "role": "user" if msg.is_user else "assistant",
                "content": msg.content[:200]  # Truncate
            })

        response = await self.ai_provider.complete(
            topic_messages,
            stream=False,
            temperature=0.3,
            max_tokens=100
        )

        # Parse topics
        topics = [
            topic.strip()
            for topic in response.content.split(",")
        ]

        return topics[:5]  # Max 5 topics
```

## 3. Database Models

### 3.1 Chat Thread Model
```python
# models/chat.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

class ChatThread(Base):
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

    # Metadata
    metadata = Column(JSON, default={}, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="chat_threads")
    user = relationship("User", back_populates="chat_threads")
    messages = relationship("ChatMessage", back_populates="thread", cascade="all, delete-orphan")
    summaries = relationship("ChatSummary", back_populates="thread", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_chat_threads_project_activity', 'project_id', 'last_activity_at'),
        Index('idx_chat_threads_user_activity', 'user_id', 'last_activity_at'),
        Index('idx_chat_threads_archived', 'is_archived'),
    )
```

### 3.2 Chat Message Model
```python
class ChatMessage(Base):
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
    edit_history = Column(JSON, default=[], nullable=False)  # List of previous versions

    # Summary tracking
    is_summarized = Column(Boolean, default=False, nullable=False)
    summary_id = Column(UUID(as_uuid=True), ForeignKey("chat_summaries.id", ondelete="SET NULL"), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Soft delete
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Additional metadata
    metadata = Column(JSON, default={}, nullable=False)

    # Relationships
    thread = relationship("ChatThread", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")
    summary = relationship("ChatSummary", back_populates="messages")

    # Constraints
    __table_args__ = (
        CheckConstraint("char_length(content) <= 50000", name="message_content_max_length"),
        Index('idx_chat_messages_thread_created', 'thread_id', 'created_at'),
        Index('idx_chat_messages_summarized', 'is_summarized'),
        Index('idx_chat_messages_deleted', 'is_deleted'),
    )
```

### 3.3 Chat Summary Model
```python
class ChatSummary(Base):
    __tablename__ = "chat_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False)
    summary_text = Column(Text, nullable=False)
    key_topics = Column(JSON, default=[], nullable=False)  # List of key topics

    # Range of messages summarized
    start_message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True)
    end_message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True)
    message_count = Column(Integer, nullable=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    token_count = Column(Integer, nullable=False, default=0)

    # Relationships
    thread = relationship("ChatThread", back_populates="summaries")
    messages = relationship("ChatMessage", back_populates="summary")

    # Indexes
    __table_args__ = (
        Index('idx_chat_summaries_thread', 'thread_id'),
        Index('idx_chat_summaries_created', 'created_at'),
    )
```

### 3.4 Database Migration
```python
# alembic/versions/003_add_chat_tables.py
"""Add chat tables

Revision ID: 003
Revises: 002
Create Date: 2024-02-01 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create chat_threads table
    op.create_table('chat_threads',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(200), nullable=False, default="New Chat"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_summary_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('summary_count', sa.Integer(), nullable=False, default=0),
        sa.Column('message_count', sa.Integer(), nullable=False, default=0),
        sa.Column('total_tokens', sa.Integer(), nullable=False, default=0),
        sa.Column('is_archived', sa.Boolean(), nullable=False, default=False),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False, default={}),
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
        sa.Column('token_count', sa.Integer(), nullable=False, default=0),
        sa.Column('model_used', sa.String(50), nullable=True),
        sa.Column('is_edited', sa.Boolean(), nullable=False, default=False),
        sa.Column('edited_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('edit_history', sa.JSON(), nullable=False, default=[]),
        sa.Column('is_summarized', sa.Boolean(), nullable=False, default=False),
        sa.Column('summary_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, default=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False, default={}),
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
        sa.Column('key_topics', sa.JSON(), nullable=False, default=[]),
        sa.Column('start_message_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('end_message_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False, default=0),
        sa.ForeignKeyConstraint(['thread_id'], ['chat_threads.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['start_message_id'], ['chat_messages.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['end_message_id'], ['chat_messages.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Add foreign key from messages to summaries
    op.add_column('chat_messages',
        sa.Column('summary_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        'fk_chat_messages_summary',
        'chat_messages', 'chat_summaries',
        ['summary_id'], ['id'],
        ondelete='SET NULL'
    )

    # Create indexes
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

def downgrade():
    op.drop_column('projects', 'last_chat_at')

    op.drop_index('idx_chat_summaries_created', 'chat_summaries')
    op.drop_index('idx_chat_summaries_thread', 'chat_summaries')

    op.drop_index('idx_chat_messages_deleted', 'chat_messages')
    op.drop_index('idx_chat_messages_summarized', 'chat_messages')
    op.drop_index('idx_chat_messages_thread_created', 'chat_messages')

    op.drop_index('idx_chat_threads_archived', 'chat_threads')
    op.drop_index('idx_chat_threads_user_activity', 'chat_threads')
    op.drop_index('idx_chat_threads_project_activity', 'chat_threads')

    op.drop_constraint('fk_chat_messages_summary', 'chat_messages', type_='foreignkey')
    op.drop_column('chat_messages', 'summary_id')

    op.drop_table('chat_summaries')
    op.drop_table('chat_messages')
    op.drop_table('chat_threads')
```

## 4. API & WebSocket Specifications

### 4.1 REST API Endpoints

#### **GET /threads**
- **Purpose**: List chat threads for a project
- **Authentication**: Required
- **Query Parameters**:
  ```typescript
  interface ThreadsQueryParams {
    project_id: string;  // Required
    include_archived?: boolean;  // default: false
    sort_by?: 'created' | 'updated' | 'activity';  // default: 'activity'
    order?: 'asc' | 'desc';  // default: 'desc'
    search?: string;  // Search in messages
    limit?: number;  // default: 20, max: 100
    offset?: number;  // default: 0
  }
  ```
- **Success Response** (200):
  ```json
  {
    "threads": [
      {
        "id": "thread-uuid",
        "project_id": "project-uuid",
        "title": "Research Discussion",
        "created_at": "2024-02-01T10:30:00Z",
        "last_activity_at": "2024-02-01T15:45:00Z",
        "message_count": 25,
        "last_message": {
          "content": "That's a great insight about...",
          "is_user": false,
          "created_at": "2024-02-01T15:45:00Z"
        },
        "is_summarized": true,
        "summary_count": 1
      }
    ],
    "total": 15,
    "has_more": false
  }
  ```

#### **POST /threads**
- **Purpose**: Create new chat thread
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "project_id": "project-uuid",
    "title": "New Research Thread",
    "initial_message": "I need help understanding quantum computing basics"
  }
  ```
- **Success Response** (201):
  ```json
  {
    "id": "new-thread-uuid",
    "project_id": "project-uuid",
    "title": "New Research Thread",
    "created_at": "2024-02-01T16:00:00Z",
    "websocket_url": "/ws/chat/new-thread-uuid"
  }
  ```

#### **GET /threads/{thread_id}/messages**
- **Purpose**: Get messages for a thread
- **Authentication**: Required
- **Query Parameters**:
  ```typescript
  interface MessagesQueryParams {
    limit?: number;  // default: 50, max: 200
    before?: string;  // Message ID for pagination
    after?: string;  // Message ID for pagination
    include_deleted?: boolean;  // default: false
  }
  ```
- **Success Response** (200):
  ```json
  {
    "messages": [
      {
        "id": "message-uuid",
        "thread_id": "thread-uuid",
        "content": "Can you explain quantum entanglement?",
        "is_user": true,
        "created_at": "2024-02-01T16:00:00Z",
        "is_edited": false,
        "token_count": 8
      },
      {
        "id": "message-uuid-2",
        "thread_id": "thread-uuid",
        "content": "Quantum entanglement is a physical phenomenon...",
        "is_user": false,
        "created_at": "2024-02-01T16:00:15Z",
        "model_used": "gpt-4",
        "token_count": 245
      }
    ],
    "has_more": true,
    "total_tokens": 253
  }
  ```

#### **PATCH /messages/{message_id}**
- **Purpose**: Edit a message
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "content": "Updated message content"
  }
  ```
- **Validation**: Only user messages can be edited
- **Success Response** (200): Updated message object
- **Side Effects**: Stores edit history, broadcasts via WebSocket

#### **DELETE /messages/{message_id}**
- **Purpose**: Soft delete a message
- **Authentication**: Required
- **Success Response** (204): No content
- **Side Effects**: Sets is_deleted flag, broadcasts deletion

#### **POST /threads/{thread_id}/summarize**
- **Purpose**: Generate thread summary
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "force": false  // Summarize even if recent summary exists
  }
  ```
- **Success Response** (202):
  ```json
  {
    "summary_id": "summary-uuid",
    "status": "processing",
    "estimated_time": 15  // seconds
  }
  ```

#### **GET /threads/{thread_id}/summary**
- **Purpose**: Get latest thread summary
- **Authentication**: Required
- **Success Response** (200):
  ```json
  {
    "id": "summary-uuid",
    "summary_text": "Key points discussed:\n• Quantum entanglement basics\n• Applications in computing\n• Current research challenges",
    "key_topics": ["quantum physics", "computing", "research"],
    "created_at": "2024-02-01T17:00:00Z",
    "message_count": 50
  }
  ```

### 4.2 WebSocket Protocol

#### **Connection Establishment**
```javascript
// Client connects to: wss://api.domain.com/ws/chat/{thread_id}
// Headers: Authorization: Bearer {token}

// Server response on successful connection:
{
  "type": "connection_established",
  "connection_id": "conn-uuid",
  "thread_id": "thread-uuid",
  "timestamp": "2024-02-01T16:00:00Z",
  "protocol_version": "1.0"
}
```

#### **Client → Server Messages**

##### **Heartbeat**
```json
{
  "type": "heartbeat",
  "timestamp": "2024-02-01T16:00:00Z"
}
```

##### **Send Message**
```json
{
  "type": "send_message",
  "thread_id": "thread-uuid",
  "content": "What are the key principles of quantum computing?",
  "parent_message_id": null,  // For threading (Phase 5)
  "metadata": {}
}
```

##### **Edit Message**
```json
{
  "type": "edit_message",
  "message_id": "message-uuid",
  "content": "Updated content",
  "preserve_response": false  // Keep AI response or regenerate
}
```

##### **Typing Indicator**
```json
{
  "type": "typing_indicator",
  "thread_id": "thread-uuid",
  "is_typing": true
}
```

##### **Regenerate Response**
```json
{
  "type": "regenerate",
  "message_id": "assistant-message-uuid",
  "user_message_id": "user-message-uuid",
  "options": {
    "temperature": 0.8,
    "model": "gpt-4"
  }
}
```

#### **Server → Client Messages**

##### **New Message**
```json
{
  "type": "new_message",
  "message": {
    "id": "message-uuid",
    "thread_id": "thread-uuid",
    "content": "User message content",
    "is_user": true,
    "created_at": "2024-02-01T16:00:00Z"
  }
}
```

##### **Assistant Message Start**
```json
{
  "type": "assistant_message_start",
  "message_id": "assistant-message-uuid",
  "thread_id": "thread-uuid",
  "model": "gpt-4"
}
```

##### **Stream Chunk**
```json
{
  "type": "stream_chunk",
  "message_id": "assistant-message-uuid",
  "chunk": "Quantum computing leverages",
  "chunk_index": 0,
  "is_final": false
}
```

##### **Stream Complete**
```json
{
  "type": "stream_chunk",
  "message_id": "assistant-message-uuid",
  "chunk": "",
  "is_final": true,
  "total_tokens": 245,
  "finish_reason": "stop"
}
```

##### **Message Updated**
```json
{
  "type": "message_updated",
  "message": {
    "id": "message-uuid",
    "content": "Updated content",
    "is_edited": true,
    "edited_at": "2024-02-01T16:05:00Z"
  }
}
```

##### **Message Deleted**
```json
{
  "type": "message_deleted",
  "message_id": "message-uuid",
  "deleted_at": "2024-02-01T16:10:00Z"
}
```

##### **Error**
```json
{
  "type": "error",
  "error_code": "rate_limit_exceeded",
  "error": "Too many messages. Please wait 60 seconds.",
  "retry_after": 60,
  "message_id": "message-uuid"  // If related to specific message
}
```

##### **Thread Summary Available**
```json
{
  "type": "summary_available",
  "thread_id": "thread-uuid",
  "summary_id": "summary-uuid",
  "message_count": 50
}
```

### 4.3 Error Handling

#### **WebSocket Error Codes**
```typescript
enum WebSocketErrorCode {
  INVALID_MESSAGE = 'invalid_message',
  RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded',
  UNAUTHORIZED = 'unauthorized',
  THREAD_NOT_FOUND = 'thread_not_found',
  MESSAGE_TOO_LONG = 'message_too_long',
  AI_PROVIDER_ERROR = 'ai_provider_error',
  INSUFFICIENT_TOKENS = 'insufficient_tokens'
}
```

#### **Reconnection Strategy**
```typescript
class WebSocketReconnector {
  private attempts = 0;
  private maxAttempts = 5;
  private baseDelay = 1000;  // 1 second

  async reconnect(): Promise<void> {
    if (this.attempts >= this.maxAttempts) {
      throw new Error('Max reconnection attempts reached');
    }

    const delay = this.baseDelay * Math.pow(2, this.attempts);
    await sleep(delay);

    this.attempts++;

    try {
      await this.connect();
      this.attempts = 0;  // Reset on success
    } catch (error) {
      await this.reconnect();
    }
  }
}
```

## 5. UX Edge Cases & State Management

### 5.1 Network Handling

#### **Connection Loss During Streaming**
- **Detection**: WebSocket close event or heartbeat timeout
- **UI Response**:
  1. Show connection banner immediately
  2. Pause streaming animation
  3. Keep partial response visible
  4. Add "Connection lost" indicator to message
- **Recovery**:
  1. Auto-reconnect with exponential backoff
  2. On reconnect, fetch message completion
  3. Resume streaming if still in progress
  4. Update UI to show full message

#### **Message Send Failures**
- **Optimistic UI**:
  1. Show message immediately with sending indicator
  2. Disable input until confirmed
  3. On failure, show retry button on message
  4. Keep message in input for easy retry
- **Retry Logic**:
  ```typescript
  const retrySend = async (message: PendingMessage) => {
    const maxRetries = 3;
    let attempt = 0;

    while (attempt < maxRetries) {
      try {
        await sendMessage(message);
        break;
      } catch (error) {
        attempt++;
        if (attempt === maxRetries) {
          showError('Failed to send message');
          markMessageAsFailed(message.id);
        } else {
          await sleep(1000 * attempt);
        }
      }
    }
  };
  ```

### 5.2 Token Management

#### **Token Expiry During Chat**
- **Detection**: 401 response from API or WebSocket
- **Handling**:
  1. Pause all operations
  2. Attempt silent token refresh
  3. If refresh fails, show re-login modal
  4. Preserve chat state in localStorage
  5. Resume after re-authentication

#### **Context Length Limits**
- **Warning Indicators**:
  - Yellow: 80% of context used
  - Red: 95% of context used
- **Automatic Handling**:
  1. Suggest summarization
  2. Offer to start new thread
  3. Show which messages will be truncated
- **UI Elements**:
  ```typescript
  interface ContextIndicator {
    used: number;
    total: number;
    percentage: number;
    severity: 'ok' | 'warning' | 'critical';
  }
  ```

### 5.3 Keyboard Shortcuts

#### **Global Shortcuts**
- **Cmd/Ctrl + K**: Focus chat input
- **Cmd/Ctrl + N**: New chat thread
- **Cmd/Ctrl + Shift + S**: Toggle sidebar
- **Cmd/Ctrl + /**: Show shortcuts help

#### **Chat Input Shortcuts**
- **Cmd/Ctrl + Enter**: Send message
- **Shift + Enter**: New line
- **Up Arrow** (empty input): Edit last message
- **Down Arrow** (editing): Cancel edit
- **Esc**: Clear input or dismiss suggestions
- **Tab**: Accept autocomplete suggestion
- **@**: Trigger mention (Phase 5)
- **/**: Trigger command palette

#### **Message Navigation**
- **J**: Next message
- **K**: Previous message
- **Enter**: Focus message actions
- **C**: Copy message
- **E**: Edit (if user message)
- **R**: Regenerate (if assistant message)

### 5.4 Mobile Adaptations

#### **Touch Gestures**
```typescript
interface TouchGestures {
  swipeRight: 'open-sidebar',
  swipeLeft: 'close-sidebar',
  longPress: 'message-actions',
  doubleTap: 'copy-message',
  pinch: 'zoom-text'
}
```

#### **Virtual Keyboard Handling**
- **Input Focus**:
  1. Scroll chat to bottom
  2. Resize viewport accounting for keyboard
  3. Keep input visible above keyboard
  4. Show "Send" button prominently
- **Dismiss Behavior**:
  1. Tap outside dismisses keyboard
  2. Scroll up dismisses keyboard
  3. Maintain input content

#### **Mobile-Specific UI**
- **Simplified Actions**: Primary actions only
- **Bottom Sheet**: For message actions
- **Pull to Refresh**: Load older messages
- **Floating Action Button**: New chat
- **Compact Mode**: Smaller margins, tighter spacing

### 5.5 Performance Optimizations

#### **Message Virtualization**
```typescript
// Only render visible messages + buffer
const MessageList = () => {
  return (
    <VirtualList
      height={window.innerHeight - 200}  // Viewport height
      itemCount={messages.length}
      itemSize={estimateMessageHeight}
      overscan={5}  // Render 5 extra items
      scrollToAlignment="end"  // Stick to bottom
    >
      {({ index, style }) => (
        <MessageBubble
          key={messages[index].id}
          message={messages[index]}
          style={style}
        />
      )}
    </VirtualList>
  );
};
```

#### **Streaming Optimization**
- **Chunk Batching**: Batch chunks every 100ms
- **Progressive Rendering**: Render markdown as it streams
- **Smooth Scrolling**: Use RAF for scroll updates
- **Memory Management**: Limit message history in memory

## 6. Technology & Library Choices

### 6.1 Frontend Libraries

#### **WebSocket Client**
- **Choice**: Native WebSocket API + Custom Wrapper
- **Rationale**:
  - Smaller bundle size vs socket.io-client
  - Full control over reconnection logic
  - Better TypeScript support
  - No unnecessary features
- **Implementation**:
  ```typescript
  class ChatWebSocket {
    private ws: WebSocket | null = null;
    private messageQueue: Message[] = [];
    private reconnector: WebSocketReconnector;

    connect(url: string, token: string): Promise<void> {
      return new Promise((resolve, reject) => {
        this.ws = new WebSocket(`${url}?token=${token}`);

        this.ws.onopen = () => {
          this.flushMessageQueue();
          resolve();
        };

        this.ws.onerror = reject;
        this.ws.onclose = this.handleClose;
        this.ws.onmessage = this.handleMessage;
      });
    }
  }
  ```

#### **Markdown Rendering**
- **Choice**: react-markdown + remark plugins
- **Plugins**:
  - remark-gfm (tables, strikethrough)
  - remark-math + rehype-katex (LaTeX)
  - remark-prism (syntax highlighting)
- **Custom Components**:
  ```typescript
  const markdownComponents = {
    code: ({ inline, className, children }) => {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <CodeBlock language={match[1]} value={String(children)} />
      ) : (
        <code className="inline-code">{children}</code>
      );
    },
    a: ({ href, children }) => (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children} <ExternalLink size={12} />
      </a>
    )
  };
  ```

#### **Code Syntax Highlighting**
- **Choice**: Prism.js with React wrapper
- **Features**:
  - Lazy load language definitions
  - Copy button integration
  - Line numbers optional
  - Theme customization
- **Supported Languages**: JavaScript, Python, TypeScript, Go, Rust, SQL, JSON

#### **Virtualization**
- **Choice**: react-window
- **Rationale**:
  - Lighter than react-virtualized
  - Better performance
  - Simpler API
  - Dynamic item sizes with VariableSizeList

#### **Text Area Auto-sizing**
- **Choice**: Custom implementation
- **Why Not Library**: Simple requirement, avoid dependency
- **Implementation**:
  ```typescript
  const AutoResizeTextarea = ({ value, onChange, ...props }) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useLayoutEffect(() => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      textarea.style.height = `${Math.min(scrollHeight, 200)}px`;
    }, [value]);

    return <textarea ref={textareaRef} value={value} onChange={onChange} {...props} />;
  };
  ```

### 6.2 Backend Libraries

#### **WebSocket Server**
- **Choice**: FastAPI's native WebSocket support
- **Rationale**:
  - Built into FastAPI
  - Async/await support
  - Good performance
  - Simple integration

#### **Token Counting**
- **Choice**: tiktoken (OpenAI's tokenizer)
- **Usage**:
  ```python
  import tiktoken

  def count_tokens(text: str, model: str = "gpt-4") -> int:
      encoding = tiktoken.encoding_for_model(model)
      return len(encoding.encode(text))
  ```

#### **Rate Limiting**
- **Choice**: Custom implementation with Redis
- **Features**:
  - Sliding window
  - Per-user limits
  - WebSocket message throttling
  ```python
  async def check_rate_limit(
      user_id: str,
      action: str,
      limit: int,
      window: int
  ) -> bool:
      key = f"rate_limit:{user_id}:{action}"
      current = await redis.incr(key)

      if current == 1:
          await redis.expire(key, window)

      return current <= limit
  ```

## 7. Testing Strategy

### 7.1 Frontend Unit Tests

#### **Message State Management**
```typescript
// __tests__/stores/chatStore.test.ts
describe('ChatStore', () => {
  it('handles streaming messages correctly', async () => {
    const { streamMessage, addStreamChunk } = useChatStore.getState();

    // Start streaming
    streamMessage({
      id: 'temp-id',
      content: '',
      is_user: false,
      thread_id: 'thread-1'
    });

    // Add chunks
    await addStreamChunk('temp-id', 'Hello ');
    await addStreamChunk('temp-id', 'world!');

    const state = useChatStore.getState();
    expect(state.streamingMessage?.content).toBe('Hello world!');
  });

  it('handles message editing', async () => {
    const { editMessage, messages } = useChatStore.getState();

    const originalMessage = {
      id: 'msg-1',
      content: 'Original content',
      is_user: true
    };

    // Add message
    useChatStore.setState({
      messages: new Map([['thread-1', [originalMessage]]])
    });

    // Edit message
    await editMessage('msg-1', 'Edited content');

    const updatedMessages = useChatStore.getState().messages.get('thread-1');
    expect(updatedMessages[0].content).toBe('Edited content');
    expect(updatedMessages[0].is_edited).toBe(true);
  });
});
```

#### **WebSocket Hook Tests**
```typescript
// __tests__/hooks/useWebSocket.test.ts
describe('useWebSocket', () => {
  let mockWebSocket: MockWebSocket;

  beforeEach(() => {
    mockWebSocket = new MockWebSocket();
    global.WebSocket = jest.fn(() => mockWebSocket);
  });

  it('handles reconnection on disconnect', async () => {
    const { result } = renderHook(() =>
      useWebSocket({
        url: 'ws://localhost/chat/123',
        onMessage: jest.fn(),
        reconnectAttempts: 3
      })
    );

    // Simulate connection
    act(() => {
      mockWebSocket.readyState = WebSocket.OPEN;
      mockWebSocket.onopen();
    });

    expect(result.current.status).toBe('connected');

    // Simulate disconnect
    act(() => {
      mockWebSocket.readyState = WebSocket.CLOSED;
      mockWebSocket.onclose();
    });

    expect(result.current.status).toBe('connecting');
    expect(global.WebSocket).toHaveBeenCalledTimes(2); // Initial + reconnect
  });
});
```

### 7.2 Backend Unit Tests

#### **AI Provider Tests**
```python
# tests/services/test_ai_provider.py
@pytest.mark.asyncio
async def test_openai_provider_streaming():
    """Test OpenAI provider streaming functionality"""
    mock_response = AsyncMock()
    mock_response.__aiter__.return_value = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))])
    ]

    provider = OpenAIProvider(api_key="test-key")
    provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

    chunks = []
    async for chunk in provider.complete(
        [AIMessage(role="user", content="Test")],
        stream=True
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]

@pytest.mark.asyncio
async def test_conversation_manager_token_limit():
    """Test conversation manager respects token limits"""
    provider = MockAIProvider(max_tokens=100)
    manager = ConversationManager(provider)

    # Create messages that exceed token limit
    messages = [
        ChatMessage(content="Short message", is_user=True),  # 10 tokens
        ChatMessage(content="A" * 200, is_user=False),  # 200 tokens
        ChatMessage(content="Another short", is_user=True),  # 10 tokens
    ]

    ai_messages = manager.prepare_messages(messages)

    # Should only include messages that fit within limit
    assert len(ai_messages) == 2  # System prompt + last message
    assert ai_messages[-1].content == "Another short"
```

#### **WebSocket Manager Tests**
```python
# tests/services/test_websocket_manager.py
@pytest.mark.asyncio
async def test_connection_manager_broadcast():
    """Test message broadcasting to multiple connections"""
    manager = ConnectionManager()

    # Create mock websockets
    ws1 = AsyncMock()
    ws2 = AsyncMock()

    # Connect to same thread
    await manager.connect(ws1, "thread-1", "user-1")
    await manager.connect(ws2, "thread-1", "user-2")

    # Broadcast message
    await manager.send_message(
        "thread-1",
        {"type": "test", "data": "hello"}
    )

    # Both should receive
    ws1.send_json.assert_called_once_with({"type": "test", "data": "hello"})
    ws2.send_json.assert_called_once_with({"type": "test", "data": "hello"})

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test WebSocket rate limiting"""
    manager = ConnectionManager()

    # Simulate rapid messages
    connection_id = "conn-1"

    for i in range(60):
        assert await manager.check_rate_limit(connection_id) == True

    # 61st message should be rate limited
    assert await manager.check_rate_limit(connection_id) == False
```

### 7.3 Integration Tests

#### **End-to-End Chat Flow**
```python
# tests/integration/test_chat_flow.py
@pytest.mark.asyncio
async def test_complete_chat_flow(async_client, test_user, test_project):
    """Test complete chat flow from thread creation to AI response"""

    # Create thread
    thread_response = await async_client.post(
        "/threads",
        json={
            "project_id": str(test_project.id),
            "title": "Integration Test Thread",
            "initial_message": "Hello AI"
        }
    )
    assert thread_response.status_code == 201
    thread = thread_response.json()

    # Connect WebSocket
    async with async_client.websocket_connect(
        f"/ws/chat/{thread['id']}",
        headers={"Authorization": f"Bearer {test_user.token}"}
    ) as websocket:
        # Receive connection confirmation
        data = await websocket.receive_json()
        assert data["type"] == "connection_established"

        # Send message
        await websocket.send_json({
            "type": "send_message",
            "thread_id": thread["id"],
            "content": "What is the capital of France?"
        })

        # Receive user message echo
        data = await websocket.receive_json()
        assert data["type"] == "new_message"
        assert data["message"]["is_user"] == True

        # Receive AI response start
        data = await websocket.receive_json()
        assert data["type"] == "assistant_message_start"

        # Collect streaming chunks
        chunks = []
        while True:
            data = await websocket.receive_json()
            if data["type"] == "stream_chunk":
                chunks.append(data["chunk"])
                if data["is_final"]:
                    break

        # Verify response contains expected content
        full_response = "".join(chunks)
        assert "Paris" in full_response
```

### 7.4 E2E Tests

#### **Chat Interface E2E**
```typescript
// e2e/chat/chat-flow.spec.ts
test.describe('Chat Experience', () => {
  test('complete chat conversation flow', async ({ page }) => {
    // Navigate to project
    await page.goto('/projects/test-project-id/chat');

    // Verify empty state
    await expect(page.locator('[data-testid="empty-chat"]')).toBeVisible();
    await expect(page.locator('text=Start a conversation')).toBeVisible();

    // Type message
    const input = page.locator('[data-testid="chat-input"]');
    await input.fill('Explain quantum computing in simple terms');

    // Send message
    await page.keyboard.press('Meta+Enter');

    // Verify user message appears
    const userMessage = page.locator('[data-testid="message-bubble"]:has-text("Explain quantum computing")');
    await expect(userMessage).toBeVisible();

    // Wait for AI response to start
    await expect(page.locator('[data-testid="typing-indicator"]')).toBeVisible();

    // Wait for streaming to complete
    await expect(page.locator('[data-testid="typing-indicator"]')).not.toBeVisible({
      timeout: 30000
    });

    // Verify AI response
    const aiMessage = page.locator('[data-testid="message-bubble"][data-sender="assistant"]');
    await expect(aiMessage).toContainText('quantum');

    // Test message actions
    await aiMessage.hover();
    await page.click('[data-testid="message-action-copy"]');
    await expect(page.locator('text=Copied to clipboard')).toBeVisible();
  });

  test('handles connection loss gracefully', async ({ page, context }) => {
    await page.goto('/projects/test-project-id/chat/existing-thread');

    // Send a message
    await page.fill('[data-testid="chat-input"]', 'Test message');
    await page.keyboard.press('Meta+Enter');

    // Simulate offline
    await context.setOffline(true);

    // Try to send another message
    await page.fill('[data-testid="chat-input"]', 'Offline message');
    await page.keyboard.press('Meta+Enter');

    // Verify offline indicator
    await expect(page.locator('[data-testid="connection-status"]')).toContainText('Disconnected');

    // Verify message shows retry
    await expect(page.locator('[data-testid="message-retry"]')).toBeVisible();

    // Go back online
    await context.setOffline(false);

    // Wait for reconnection
    await expect(page.locator('[data-testid="connection-status"]')).not.toBeVisible({
      timeout: 10000
    });

    // Retry message
    await page.click('[data-testid="message-retry"]');

    // Verify message sent successfully
    await expect(page.locator('[data-testid="message-retry"]')).not.toBeVisible();
  });
});
```

#### **Mobile Chat E2E**
```typescript
// e2e/chat/mobile-chat.spec.ts
test.describe('Mobile Chat Experience', () => {
  test.use({
    viewport: { width: 375, height: 667 },  // iPhone SE
    hasTouch: true
  });

  test('mobile chat interactions', async ({ page }) => {
    await page.goto('/projects/test-project-id/chat');

    // Verify mobile layout
    await expect(page.locator('[data-testid="mobile-sidebar-toggle"]')).toBeVisible();
    await expect(page.locator('[data-testid="chat-sidebar"]')).not.toBeVisible();

    // Open sidebar with swipe
    await page.locator('body').swipe({ direction: 'right' });
    await expect(page.locator('[data-testid="chat-sidebar"]')).toBeVisible();

    // Select thread
    await page.click('[data-testid="thread-item"]:first-child');
    await expect(page.locator('[data-testid="chat-sidebar"]')).not.toBeVisible();

    // Test virtual keyboard handling
    await page.click('[data-testid="chat-input"]');

    // Verify viewport adjusts for keyboard
    await page.waitForFunction(() => {
      return window.visualViewport.height < window.innerHeight;
    });

    // Long press for message actions
    const message = page.locator('[data-testid="message-bubble"]:first-child');
    await message.tap({ delay: 1000 });  // Long press

    // Verify action sheet appears
    await expect(page.locator('[data-testid="mobile-action-sheet"]')).toBeVisible();

    // Copy message
    await page.click('[data-testid="action-copy"]');
    await expect(page.locator('text=Copied')).toBeVisible();
  });
});
```

## 8. Milestones & Acceptance Criteria

### Milestone 1: WebSocket Infrastructure
**Acceptance Criteria:**
- [x] WebSocket connection manager implemented
- [x] Heartbeat mechanism working
- [x] Reconnection logic with exponential backoff
- [x] Rate limiting prevents spam
- [x] Connection status UI component shows states
- [x] Unit tests pass for all connection scenarios

### Milestone 2: AI Provider Integration
**Acceptance Criteria:**
- [x] OpenAI provider implements streaming
- [x] Token counting accurate for GPT models
- [x] Context management stays within limits
- [x] Error handling for API failures
- [x] Provider abstraction allows easy addition of new providers
- [x] Mock provider available for testing

### Milestone 3: Chat UI Components
**Acceptance Criteria:**
- [x] MessageBubble renders markdown correctly
- [x] ChatInputBar auto-resizes smoothly
- [x] StreamingText shows character-by-character
- [x] Code blocks have syntax highlighting
- [x] LaTeX math renders properly
- [x] All components accessible (ARIA labels, keyboard nav)

### Milestone 4: Message Management
**Acceptance Criteria:**
- [x] Send message via WebSocket works
- [x] Edit user messages updates in real-time
- [x] Delete soft-deletes messages
- [x] Message history loads with pagination
- [x] Optimistic UI updates feel instant
- [x] Error states show retry options

### Milestone 5: Thread Navigation
**Acceptance Criteria:**
- [x] Recent chats sidebar shows all threads
- [x] Thread search filters in real-time
- [x] Thread grouping by date works
- [x] Quick thread switching maintains state
- [x] Mobile sidebar slides smoothly
- [x] Thread title updates automatically

### Milestone 6: Streaming AI Responses
**Acceptance Criteria:**
- [x] Streaming starts immediately after send
- [x] Chunks render smoothly without flicker
- [x] Markdown formatting applies during stream
- [x] Network interruption handling works
- [x] Regenerate response replaces original
- [x] Token usage tracked accurately

### Milestone 7: Summarization System
**Acceptance Criteria:**
- [x] Auto-summarization triggers at thresholds
- [x] Manual summarization available
- [x] Summaries show key topics
- [x] Summary indicator in thread list
- [x] Background summarization doesn't block UI
- [x] Summaries used for context when appropriate

### Milestone 8: Mobile Optimization
**Acceptance Criteria:**
- [x] Touch gestures work intuitively
- [x] Virtual keyboard doesn't cover input
- [x] Message actions accessible via long press
- [x] Sidebar swipe gestures smooth
- [x] Performance remains good on low-end devices
- [x] Offline mode shows appropriate UI

## 9. Development Execution Order

### Week 1: Backend Foundation
**Day 1-2: AI Provider & WebSocket Infrastructure**
- Implement AI provider abstraction
- Create OpenAI provider with streaming
- Build WebSocket connection manager
- Add heartbeat and reconnection logic

**Day 3-4: Database & Core Services**
- Create chat models and migrations
- Implement message CRUD operations
- Build conversation manager
- Add token counting utilities

**Day 5: WebSocket Handlers**
- Implement message routing
- Add streaming message handler
- Create rate limiting
- Test WebSocket communication

### Week 2: Core Chat UI
**Day 6-7: Message Components**
- Build MessageBubble with markdown
- Implement StreamingText animation
- Add code syntax highlighting
- Create message actions menu

**Day 8-9: Chat Input & Interaction**
- Build auto-sizing ChatInputBar
- Add keyboard shortcuts
- Implement message send flow
- Create connection status indicator

**Day 10: State Management**
- Set up chat Zustand store
- Implement WebSocket hooks
- Add optimistic updates
- Handle error states

### Week 3: Thread Management & Polish
**Day 11-12: Thread Navigation**
- Build RecentChatsSidebar
- Implement thread switching
- Add thread search/filter
- Create thread header component

**Day 13: Summarization**
- Implement summarization service
- Add summary UI indicators
- Create background job
- Test summary generation

**Day 14-15: Mobile & Performance**
- Optimize for mobile devices
- Add touch gestures
- Implement message virtualization
- Performance profiling

### Week 4: Integration & Testing
**Day 16-17: Integration**
- Connect all components
- Test full chat flow
- Fix edge cases
- Polish animations

**Day 18-19: Testing**
- Complete unit test coverage
- Write integration tests
- Create E2E test suite
- Load testing for WebSockets

**Day 20: Documentation & Deployment**
- API documentation
- Deployment guides
- Performance benchmarks
- Handoff to Phase 4

### Parallelization Strategy

**Frontend Team (2 developers)**:
- Developer 1: Message components, markdown rendering
- Developer 2: Input components, thread navigation

**Backend Team (2 developers)**:
- Developer 1: AI provider, streaming infrastructure
- Developer 2: WebSocket manager, message persistence

**Sync Points**:
- Day 5: WebSocket protocol finalized
- Day 10: API contracts locked
- Day 15: Feature freeze
- Day 17: Integration complete

### Risk Mitigation

**High Risk: AI Provider Costs**
- Implement usage quotas early
- Add cost tracking from day 1
- Create fallback to smaller models

**Medium Risk: WebSocket Scaling**
- Plan for horizontal scaling
- Implement connection limits
- Design for Redis pub/sub (Phase 4)

**Low Risk: Browser Compatibility**
- Test WebSocket support early
- Polyfill for older browsers
- Progressive enhancement approach

This comprehensive specification provides everything needed to build a robust, real-time chat experience that serves as the core value proposition of the AI productivity app.
