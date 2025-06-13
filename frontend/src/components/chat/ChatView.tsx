import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useChatStore } from '../../stores/chatStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useAuth } from '../../hooks/useAuth';
import MessageBubble from './MessageBubble';
import ChatInputBar from './ChatInputBar';
import ConnectionStatus from './ConnectionStatus';
import { MessageType, WebSocketStatus } from '../../types/websocket';
import { ChatMessage } from '../../types/chat';
import { VirtualList } from '../ui/VirtualList';
import { Loader2, AlertCircle } from 'lucide-react';
import styles from './ChatView.module.css';

interface ChatViewProps {
  projectId: string;
  threadId?: string;
  onThreadChange?: (threadId: string) => void;
}

// Helper to estimate message height for virtual list
function estimateMessageHeight(message: ChatMessage): number {
  const baseHeight = 60; // padding/avatar etc.
  const charsPerLine = 50;
  const lineHeight = 20;
  const lines = Math.ceil(message.content.length / charsPerLine);
  return baseHeight + lines * lineHeight;
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

  // WebSocket ---------------------------------------------------------------
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

  // Load thread when route changes
  useEffect(() => {
    if (threadId && threadId !== activeThreadId) {
      loadThread(threadId);
    }
  }, [threadId, activeThreadId, loadThread]);

  // Auto-scroll behaviour
  useEffect(() => {
    if (isAutoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.get(activeThreadId || '')?.length, streamingMessage, isAutoScroll]);

  // ---- WebSocket message router ----
  function handleWebSocketMessage(data: any) {
    switch (data.type) {
      case MessageType.NEW_MESSAGE:
        // already handled via optimistic update in store
        break;
      case MessageType.ASSISTANT_MESSAGE_START:
        useChatStore.getState().startStreaming(data.message_id);
        break;
      case MessageType.STREAM_CHUNK:
        useChatStore.getState().addStreamChunk(data.message_id, data.chunk);
        if (data.is_final) {
          useChatStore.getState().completeStreaming(data.message_id, ''); // content patched later
        }
        break;
      case MessageType.MESSAGE_UPDATED:
        useChatStore.getState().updateMessageInStore(data.message);
        break;
      case MessageType.MESSAGE_DELETED:
        useChatStore.getState().removeMessageFromStore(data.message_id);
        break;
      default:
        console.warn('Unhandled WS message', data);
    }
  }

  // Scroll handling
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setIsAutoScroll(nearBottom);
    setShowNewMessageIndicator(!nearBottom);
  }, []);

  // Ensure thread exists before sending first message
  const ensureThread = async (): Promise<string> => {
    if (activeThreadId) return activeThreadId;
    const thread = await createThread(projectId);
    onThreadChange?.(thread.id);
    navigate(`/projects/${projectId}/chat/${thread.id}`);
    return thread.id;
  };

  // Send message handler
  const handleSend = async (content: string) => {
    const thId = await ensureThread();
    await sendMessage(content);
    wsSendMessage({ type: MessageType.SEND_MESSAGE, thread_id: thId, content });
  };

  // Edit, delete, regenerate helpers ---------------------------------------
  const handleEdit = async (id: string, content: string) => {
    await editMessage(id, content);
    wsSendMessage({ type: MessageType.EDIT_MESSAGE, message_id: id, content });
  };

  const handleDelete = async (id: string) => {
    await deleteMessage(id);
    wsSendMessage({ type: MessageType.DELETE_MESSAGE, message_id: id });
  };

  const handleRegenerate = async (assistantId: string, userMessageId: string) => {
    await regenerateResponse(assistantId);
    wsSendMessage({ type: MessageType.REGENERATE, message_id: assistantId, user_message_id: userMessageId });
  };

  // Current data slices
  const currentThread = activeThreadId ? threads.get(activeThreadId) : null;
  const currentMessages = activeThreadId ? messages.get(activeThreadId) || [] : [];

  if (isLoadingThread) {
    return (
      <div className={styles.loadingContainer}>
        <Loader2 className={styles.spinner} />
        <p>Loading conversation…</p>
      </div>
    );
  }

  if (wsStatus === WebSocketStatus.ERROR) {
    return (
      <div className={styles.errorContainer}>
        <AlertCircle size={32} />
        <p>Connection error</p>
        <button onClick={reconnect}>Retry</button>
      </div>
    );
  }

  return (
    <div className={styles.chatRoot}>
      {wsStatus !== WebSocketStatus.CONNECTED && <ConnectionStatus status={wsStatus} onReconnect={reconnect} />}

      {/* Header */}
      {currentThread && (
        <div className={styles.header}>
          <h2>{currentThread.title}</h2>
          <span>{currentThread.message_count} messages</span>
        </div>
      )}

      {/* Messages */}
      <div ref={scrollContainerRef} className={styles.messagesWrapper} onScroll={handleScroll}>
        {currentMessages.length === 0 ? (
          <div className={styles.emptyState}>Start a conversation</div>
        ) : (
          <>
            <VirtualList
              items={currentMessages}
              height={window.innerHeight - 260}
              itemHeight={(idx) => estimateMessageHeight(currentMessages[idx])}
              renderItem={(msg, idx) => {
                const isLast = idx === currentMessages.length - 1;
                const prev = idx > 0 ? currentMessages[idx - 1] : null;
                const showAvatar = !prev || prev.is_user !== msg.is_user;
                return (
                  <MessageBubble
                    key={msg.id}
                    message={msg}
                    isStreaming={streamingMessage?.id === msg.id}
                    isLast={isLast}
                    showAvatar={showAvatar}
                    onEdit={(id) => {
                      const newContent = prompt('Edit message', msg.content);
                      if (newContent && newContent !== msg.content) handleEdit(id, newContent);
                    }}
                    onDelete={handleDelete}
                    onRegenerate={() => {
                      const userIdx = idx - 1;
                      if (userIdx >= 0) handleRegenerate(msg.id, currentMessages[userIdx].id);
                    }}
                    onCopy={() => navigator.clipboard.writeText(msg.content)}
                  />
                );
              }}
            />
            <div ref={messagesEndRef} />
          </>
        )}

        {showNewMessageIndicator && (
          <button className={styles.newIndicator} onClick={() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })}>
            New messages ↓
          </button>
        )}
      </div>

      {/* Input Bar */}
      <div className={styles.inputWrapper}>
        <ChatInputBar
          projectId={projectId}
          onSend={handleSend}
          onTyping={() =>
            wsSendMessage({
              type: MessageType.TYPING_INDICATOR,
              thread_id: activeThreadId,
              is_typing: true,
            })
          }
          isDisabled={wsStatus !== WebSocketStatus.CONNECTED}
        />
      </div>
    </div>
  );
};

export default ChatView;
