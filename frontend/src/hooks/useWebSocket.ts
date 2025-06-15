import { useEffect, useRef, useState, useCallback } from 'react';
import { useChatStore } from '../stores/chatStore';
import type { ChatMessage } from '../types/chat';

/**
 * Hook that wraps the native WebSocket API with:
 * • JSON send helpers
 * • Automatic heartbeat / pong handling
 * • Exponential-back-off reconnection (maxAttempts)
 *
 * The semantics follow plan3.md §1 “useWebSocket” description.
 */
import { WebSocketStatus } from '@/types/websocket';

export interface UseWebSocketOptions {
  url: string | null;                       // full ws://… or relative URL; null → disabled
  token?: string;                           // JWT (optional)
  reconnectAttempts?: number;               // default 5
  heartbeatInterval?: number;               // ms, default 30 000
  onMessage?: (msg: unknown) => void;       // custom handler invoked after internal processing
}

export interface WebSocketHandle {
  status: WebSocketStatus;
  sendMessage: (data: Record<string, unknown>) => void;
  reconnect: () => void;
  close: () => void;
}

export const useWebSocket = ({
  url,
  token,
  reconnectAttempts = 5,
  heartbeatInterval = 30_000,
  onMessage,
}: UseWebSocketOptions): WebSocketHandle => {
  const [status, setStatus] = useState<WebSocketStatus>(WebSocketStatus.CONNECTING);
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const heartbeatTimer = useRef<number | null>(null);
  const store = useChatStore();

  // -------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimer.current) clearInterval(heartbeatTimer.current);

    heartbeatTimer.current = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'heartbeat', timestamp: Date.now() }));
      }
    }, heartbeatInterval);
  }, [heartbeatInterval]);

  const cleanup = () => {
    if (heartbeatTimer.current) clearInterval(heartbeatTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
  };

  const handleIncomingMessage = useCallback((msg: unknown) => {
    // Consider defining a more specific type for msg if its structure is known
    if (typeof msg === 'object' && msg !== null && 'type' in msg) {
      const message = msg as { type: string; message_id?: unknown; chunk?: unknown; is_final?: unknown; message?: unknown };
      switch (message.type) {
        case 'new_message':
          // Assuming message.message is of type ChatMessage or compatible
          store.streamMessage(message.message as ChatMessage);
          break;
        case 'assistant_message_start':
          // placeholder assistant message already created in ChatService – nothing to do
          break;
        case 'stream_chunk':
          if (message.message_id && typeof message.chunk === 'string') {
            store.addStreamChunk(
              message.message_id as string,
              message.chunk,
              message.is_final as boolean | undefined
            );
          }
          break;
        case 'message_updated':
          if (
            message.message && typeof message.message === 'object' &&
            'id' in message.message && 'content' in message.message
          ) {
            store.editMessage(
              (message.message as { id: string }).id,
              (message.message as { content: string }).content
            ).catch(() => {});
          }
          break;
        case 'message_deleted':
          store.deleteMessage(message.message_id as string).catch(() => {});
          break;
        default:
          // ignore
          break;
      }
    }

    if (onMessage) {
      try {
        onMessage(msg);
      } catch {
        // swallow errors from user handler
      }
    }
  }, [store, onMessage]);

  const connect = useCallback(() => {
    if (!url) return; // nothing to do when url is null

    cleanup();
    setStatus(WebSocketStatus.CONNECTING);

    // If the caller passes a relative URL we auto-upgrade to ws(s) scheme.
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const fullUrl = import.meta.env.VITE_WS_URL || (url.startsWith('ws://') || url.startsWith('wss://') ? url : `${scheme}://${window.location.host}${url}`);
    const wsWithToken = token ? `${fullUrl}?token=${token}` : fullUrl;

    const ws = new WebSocket(wsWithToken);
    wsRef.current = ws;

    ws.onopen = () => {
      attemptsRef.current = 0;
      setStatus(WebSocketStatus.CONNECTED);
      startHeartbeat();
    };

    ws.onerror = () => {
      setStatus(WebSocketStatus.ERROR);
    };

    ws.onclose = () => {
      setStatus(WebSocketStatus.DISCONNECTED);
      cleanup();
      // schedule reconnect if attempts left
      if (url && attemptsRef.current < reconnectAttempts) {
        const delay = 1000 * Math.pow(2, attemptsRef.current); // exponential
        attemptsRef.current += 1;
        setTimeout(connect, delay);
      }
    };

    ws.onmessage = (evt) => {
      try {
        const parsedMsg = JSON.parse(evt.data as string);
        handleIncomingMessage(parsedMsg);
      } catch {
        // ignore malformed
      }
    };
  }, [url, token, reconnectAttempts, startHeartbeat, handleIncomingMessage]);


  // -------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------
  useEffect(() => {
    if (url) {
      connect();
    }
    return () => {
      cleanup();
    };
  }, [connect, url]);

  // -------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------
  const sendMessage = (data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  };

  const close = () => {
    cleanup();
    setStatus(WebSocketStatus.DISCONNECTED);
  };

  const reconnect = () => {
    attemptsRef.current = 0;
    connect();
  };

  return { status, sendMessage, reconnect, close };
};
