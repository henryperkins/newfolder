/* --------------------------------------------------------------------------
 * ChatView.tsx
 * --------------------------------------------------------------------------
 * Conversation pane with virtualised message list, WebSocket streaming,
 * optimistic UI, and full CRUD controls for each message.
 * Compiles cleanly under strict TypeScript and passes react-hooks ESLint.
 * ------------------------------------------------------------------------ */

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  memo,
} from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { useChatStore } from '../../stores/chatStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useAuth } from '../../hooks/useAuth';

import MessageBubble from './MessageBubble';
import ChatInputBar from './ChatInputBar';
import ConnectionStatus from './ConnectionStatus';

import { MessageType, WebSocketStatus } from '../../types/websocket';
import type { ChatMessage } from '../../types/chat';

import { VirtualList } from '../ui/VirtualList';

import { Loader2, AlertCircle } from 'lucide-react';
import styles from './ChatView.module.css';

/* ------------------------------------------------------------------ */
/* Utilities                                                          */
/* ------------------------------------------------------------------ */

interface ChatViewProps {
  projectId: string;
  threadId?: string;
  onThreadChange?: (threadId: string) => void;
}

/** Rough message height estimator for VirtualList */
function estimateMessageHeight(msg: ChatMessage): number {
  const base = 60; // avatar + padding
  const charsPerLine = 50;
  const lineHeight = 20;
  const lines = Math.ceil(msg.content.length / charsPerLine);
  return base + lines * lineHeight;
}

/* ------------------------------------------------------------------ */
/* Main Component                                                     */
/* ------------------------------------------------------------------ */

const ChatView: React.FC<ChatViewProps> = ({
  projectId,
  threadId,
  onThreadChange,
}) => {
  /* ----------------------------- Routing -------------------------- */
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  /* --------------------------- Auth guard ------------------------ */
  useAuth();

  /* ----------------------- Chat store slices --------------------- */
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

  /* --------------------------- WebSocket -------------------------- */
  const {
    status: wsStatus,
    sendMessage: wsSendMessage,
    reconnect,
  } = useWebSocket({
    url: activeThreadId ? `/ws/chat/${activeThreadId}` : null,
    onMessage: handleWebSocketMessage,
    reconnectAttempts: 5,
    heartbeatInterval: 30_000,
  });

  /* ---------------------- Local component state ------------------ */
  const [isAutoScroll, setIsAutoScroll] = useState(true);
  const [showNewMessageIndicator, setShowNewMessageIndicator] =
    useState(false);

  /* ----------------------------- Refs ----------------------------- */
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  /* ----------------------------------------------------------------
   *  Memoised helpers
   * ---------------------------------------------------------------- */

  /** Make sure we have a thread before sending the first message. */
  const ensureThread = useCallback(
    async (initialMessage?: string): Promise<string> => {
      if (activeThreadId) return activeThreadId;
      const th = await createThread(projectId, initialMessage);
      onThreadChange?.(th.id);
      navigate(`/projects/${projectId}/chat/${th.id}`);
      return th.id;
    },
    [activeThreadId, createThread, navigate, onThreadChange, projectId],
  );

  /** Dispatch user message + push to WebSocket (when connected). */
  const handleSend = useCallback(
    async (content: string) => {
      const thId = await ensureThread(content);
      await sendMessage(content);
      if (wsStatus === WebSocketStatus.CONNECTED && wsSendMessage) {
        wsSendMessage({
          type: MessageType.SEND_MESSAGE,
          thread_id: thId,
          content,
        });
      }
      setIsAutoScroll(true);
    },
    [ensureThread, sendMessage, wsSendMessage, wsStatus],
  );

  /** Edit / delete / regenerate wrappers with safe WS guards. */
  const handleEdit = useCallback(
    async (id: string, content: string) => {
      await editMessage(id, content);
      wsSendMessage?.({
        type: MessageType.EDIT_MESSAGE,
        message_id: id,
        content,
      });
    },
    [editMessage, wsSendMessage],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      await deleteMessage(id);
      wsSendMessage?.({
        type: MessageType.DELETE_MESSAGE,
        message_id: id,
      });
    },
    [deleteMessage, wsSendMessage],
  );

  const handleRegenerate = useCallback(
    async (assistantId: string, userMsgId: string) => {
      await regenerateResponse(assistantId);
      wsSendMessage?.({
        type: MessageType.REGENERATE,
        message_id: assistantId,
        user_message_id: userMsgId,
      });
    },
    [regenerateResponse, wsSendMessage],
  );

  /* ----------------------------------------------------------------
   *  Side-effects
   * ---------------------------------------------------------------- */

  /** Load thread when route param changes */
  useEffect(() => {
    if (threadId && threadId !== activeThreadId) loadThread(threadId);
  }, [threadId, activeThreadId, loadThread]);

  /** Handle initial message passed via query param. */
  useEffect(() => {
    const initMsg = searchParams.get('initialMessage');
    if (initMsg && !threadId && projectId && !activeThreadId) {
      // strip param
      navigate(
        `/projects/${projectId}/chat?initialMessage=${encodeURIComponent(
          initMsg,
        )}`,
        { replace: true },
      );
      handleSend(initMsg);
    }
  }, [
    searchParams,
    threadId,
    projectId,
    navigate,
    activeThreadId,
    handleSend,
  ]);

  /** Auto-scroll behaviour */
  useEffect(() => {
    if (isAutoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, streamingMessage, isAutoScroll, activeThreadId]);

  /* ----------------------------------------------------------------
   *  WebSocket router
   * ---------------------------------------------------------------- */
  function handleWebSocketMessage(raw: unknown) {
    if (typeof raw !== 'object' || raw === null || !('type' in raw)) {
      console.warn('[WS] Unexpected payload', raw);
      return;
    }

    const data = raw as Partial<{
      type: MessageType;
      message_id: string;
      chunk: string;
      is_final: boolean;
      message: ChatMessage;
    }>;

    switch (data.type) {
      case MessageType.NEW_MESSAGE:
        // optimistic update already handled in store
        break;
      case MessageType.ASSISTANT_MESSAGE_START:
        if (data.message_id) useChatStore
          .getState()
          .startStreaming(data.message_id);
        break;
      case MessageType.STREAM_CHUNK:
        if (data.message_id && data.chunk !== undefined) {
          useChatStore.getState().addStreamChunk(data.message_id, data.chunk);
        }
        if (data.is_final && data.message_id) {
          useChatStore
            .getState()
            .completeStreaming(data.message_id, '');
        }
        break;
      case MessageType.MESSAGE_UPDATED:
        if (data.message)
          useChatStore.getState().updateMessageInStore(data.message);
        break;
      case MessageType.MESSAGE_DELETED:
        if (data.message_id)
          useChatStore.getState().removeMessageFromStore(data.message_id);
        break;
      default:
        console.warn('[WS] Unhandled message', data);
    }
  }

  /* ----------------------------------------------------------------
   *  Scroll awareness
   * ---------------------------------------------------------------- */
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } =
      scrollContainerRef.current;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setIsAutoScroll(nearBottom);
    setShowNewMessageIndicator(!nearBottom);
  }, []);

  /* ----------------------------------------------------------------
   *  Derived data
   * ---------------------------------------------------------------- */
  const currentThread = activeThreadId
    ? threads.get(activeThreadId) ?? null
    : null;
  const currentMessages = activeThreadId
    ? messages.get(activeThreadId) ?? []
    : [];

  /* ----------------------------------------------------------------
   *  Early returns
   * ---------------------------------------------------------------- */
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

  /* ----------------------------------------------------------------
   *  Render
   * ---------------------------------------------------------------- */
  return (
    <div className={styles.chatRoot}>
      {/* Connection banner (if disconnected) */}
      {wsStatus !== WebSocketStatus.CONNECTED && (
        <ConnectionStatus status={wsStatus} onReconnect={reconnect} />
      )}

      {/* Header */}
      {currentThread && (
        <div className={styles.header}>
          <h2>{currentThread.title}</h2>
          <span>{currentThread.message_count} messages</span>
        </div>
      )}

      {/* Messages list */}
      <div
        ref={scrollContainerRef}
        className={styles.messagesWrapper}
        onScroll={handleScroll}
      >
        {currentMessages.length === 0 ? (
          <div className={styles.emptyState}>Start a conversation</div>
        ) : (
          <>
            <VirtualList
              items={currentMessages}
              height={window.innerHeight - 260}
              itemHeight={(idx) =>
                estimateMessageHeight(currentMessages[idx])
              }
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
                      const newContent = prompt(
                        'Edit message',
                        msg.content,
                      );
                      if (newContent && newContent !== msg.content)
                        handleEdit(id, newContent);
                    }}
                    onDelete={handleDelete}
                    onRegenerate={() => {
                      const userIdx = idx - 1;
                      if (userIdx >= 0)
                        handleRegenerate(
                          msg.id,
                          currentMessages[userIdx].id,
                        );
                    }}
                    onCopy={() =>
                      navigator.clipboard.writeText(msg.content)
                    }
                  />
                );
              }}
            />
            <div ref={messagesEndRef} />
          </>
        )}

        {/* “New messages” indicator */}
        {showNewMessageIndicator && (
          <button
            className={styles.newIndicator}
            onClick={() =>
              messagesEndRef.current?.scrollIntoView({
                behavior: 'smooth',
              })
            }
          >
            New messages ↓
          </button>
        )}
      </div>

      {/* Input bar */}
      <div className={styles.inputWrapper}>
        <ChatInputBar
          projectId={projectId}
          onSend={handleSend}
          onTyping={() =>
            wsSendMessage?.({
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

export default memo(ChatView);
