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
export interface UseWebSocketOptions {
  url: string;                              // full ws://… or relative URL
  token?: string;                           // JWT (optional)
  reconnectAttempts?: number;               // default 5
  heartbeatInterval?: number;               // ms, default 30 000
}

type Status = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketHandle {
  status: Status;
  sendJson: (data: Record<string, unknown>) => void;
  close: () => void;
}

export const useWebSocket = ({
  url,
  token,
  reconnectAttempts = 5,
  heartbeatInterval = 30_000,
}: UseWebSocketOptions): WebSocketHandle => {
  const [status, setStatus] = useState<Status>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const heartbeatTimer = useRef<number | null>(null);
  const store = useChatStore();

  // -------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------
  const startHeartbeat = () => {
    if (heartbeatTimer.current) clearInterval(heartbeatTimer.current);
    heartbeatTimer.current = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'heartbeat', timestamp: Date.now() }));
      }
    }, heartbeatInterval);
  };

  const cleanup = () => {
    if (heartbeatTimer.current) clearInterval(heartbeatTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
  };

  const connect = useCallback(() => {
    cleanup();
    setStatus('connecting');

    const fullUrl = token ? `${url}?token=${token}` : url;
    const ws = new WebSocket(fullUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      attemptsRef.current = 0;
      setStatus('connected');
      startHeartbeat();
    };

    ws.onerror = () => {
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus('disconnected');
      cleanup();
      // schedule reconnect if attempts left
      if (attemptsRef.current < reconnectAttempts) {
        const delay = 1000 * Math.pow(2, attemptsRef.current); // exponential
        attemptsRef.current += 1;
        setTimeout(connect, delay);
      }
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        handleIncomingMessage(msg);
      } catch {
        // ignore malformed
      }
    };
  }, [url, token, reconnectAttempts, heartbeatInterval]);

  const handleIncomingMessage = (msg: any) => {
    switch (msg.type) {
      case 'new_message':
        store.streamMessage(msg.message as ChatMessage);
        break;
      case 'assistant_message_start':
        // placeholder assistant message already created in ChatService – nothing to do
        break;
      case 'stream_chunk':
        store.addStreamChunk(msg.message_id, msg.chunk, msg.is_final);
        break;
      case 'message_updated':
        store.editMessage(msg.message.id, msg.message.content).catch(() => {});
        break;
      case 'message_deleted':
        store.deleteMessage(msg.message_id).catch(() => {});
        break;
      default:
        // ignore
        break;
    }
  };

  // -------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------
  useEffect(() => {
    connect();
    return () => {
      cleanup();
    };
  }, [connect]);

  // -------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------
  const sendJson = (data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  };

  const close = () => {
    cleanup();
    setStatus('disconnected');
  };

  return { status, sendJson, close };
};
