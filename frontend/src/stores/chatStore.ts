// Chat Zustand Store – Phase-3 core state for chat experience
//
// The implementation follows the interface defined in plan3.md (§1.1 ChatView
// ‑ “Zustand Store Integration”).  The actions intentionally delegate network /
// persistence to the yet-to-be-implemented `chatApi` (REST) and WebSocket
// helpers.  This keeps the store testable and free of transport concerns.
//
// Missing utilities (`chatApi`, `useWebSocket` etc.) will be added in
// subsequent commits.  For now they are typed as “any” so that TypeScript
// compilation succeeds even while the helpers are still stubs.

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { ChatMessage, ChatThread } from '../types/chat';
import { chatApi } from '../utils/api';

// ---------------------------------------------------------------------------
// Store state / actions
// ---------------------------------------------------------------------------

export interface ChatStoreState {
  // Normalised maps for O(1) access & easy optimistic updates
  threads: Map<string, ChatThread>;
  activeThreadId: string | null;
  messages: Map<string, ChatMessage[]>; // key = thread_id

  // UI / network flags
  isLoadingThread: boolean;
  streamingMessage: Partial<ChatMessage> | null;
}

export interface ChatStoreActions {
  loadThread: (threadId: string) => Promise<void>;
  createThread: (projectId: string, initialMessage?: string) => Promise<ChatThread>;
  sendMessage: (content: string, attachments?: File[]) => Promise<void>;
  editMessage: (messageId: string, content: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  regenerateResponse: (messageId: string) => Promise<void>;

  // Local helpers
  streamMessage: (msg: ChatMessage) => void;
  addStreamChunk: (messageId: string, chunk: string, isFinal: boolean) => Promise<void>;
  setActiveThreadId: (threadId: string | null) => void;
}

export type ChatStore = ChatStoreState & ChatStoreActions;

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: ChatStoreState = {
  threads: new Map(),
  activeThreadId: null,
  messages: new Map(),
  isLoadingThread: false,
  streamingMessage: null,
};

// ---------------------------------------------------------------------------
// Zustand store
// ---------------------------------------------------------------------------

export const useChatStore = create<ChatStore>()(
  immer((set, get) => ({
    ...initialState,

    // ---------------------------------------------------------
    // Thread loading / creation
    // ---------------------------------------------------------
    async loadThread(threadId) {
      set((s) => {
        s.isLoadingThread = true;
        s.activeThreadId = threadId;
      });

      try {
        const resp = await chatApi.getThreadMessages(threadId);
        set((s) => {
          s.messages.set(threadId, resp.messages);
          s.isLoadingThread = false;
        });
      } catch (err) {
        set((s) => {
          s.isLoadingThread = false;
        });
        throw err;
      }
    },

    async createThread(projectId, initialMessage) {
      const thread = await chatApi.createThread(projectId, initialMessage);
      set((s) => {
        s.threads.set(thread.id, thread);
        s.activeThreadId = thread.id;
        if (initialMessage) {
          s.messages.set(thread.id, [
            {
              id: 'temp-initial',
              thread_id: thread.id,
              content: initialMessage,
              is_user: true,
              created_at: new Date().toISOString(),
            } as unknown as ChatMessage,
          ]);
        }
      });
      return thread;
    },

    // ---------------------------------------------------------
    // Message CRUD
    // ---------------------------------------------------------
    async sendMessage(content, attachments) {
      const threadId = get().activeThreadId;
      if (!threadId) return;

      // Optimistic placeholder
      const tempId = `temp-${Date.now()}`;
      const newMsg: ChatMessage = {
        id: tempId,
        thread_id: threadId,
        content,
        is_user: true,
        created_at: new Date().toISOString(),
      } as ChatMessage;
      set((s) => {
        const list = s.messages.get(threadId) ?? [];
        s.messages.set(threadId, [...list, newMsg]);
      });

      try {
        const saved = await chatApi.sendMessage(threadId, content, attachments);
        // Replace placeholder by id
        set((s) => {
          const list = s.messages.get(threadId) ?? [];
          const idx = list.findIndex((m) => m.id === tempId);
          if (idx !== -1) {
            list[idx] = saved;
            s.messages.set(threadId, [...list]);
          }
        });
      } catch (err) {
        // mark failed
        set((s) => {
          const list = s.messages.get(threadId) ?? [];
          const idx = list.findIndex((m) => m.id === tempId);
          if (idx !== -1) {
            list[idx].metadata = { failed: true };
          }
        });
        throw err;
      }
    },

    async editMessage(messageId, content) {
      const updated = await chatApi.editMessage(messageId, content);
      // optimistic update local state
      const threadId = updated.thread_id;
      set((s) => {
        const list = s.messages.get(threadId) ?? [];
        const idx = list.findIndex((m) => m.id === messageId);
        if (idx !== -1) {
          list[idx] = updated;
          s.messages.set(threadId, [...list]);
        }
      });
    },

    async deleteMessage(messageId) {
      await chatApi.deleteMessage(messageId);
      // remove locally
      for (const [threadId, list] of get().messages.entries()) {
        const idx = list.findIndex((m) => m.id === messageId);
        if (idx !== -1) {
          set((s) => {
            s.messages.set(
              threadId,
              list.filter((m) => m.id !== messageId),
            );
          });
          break;
        }
      }
    },

    async regenerateResponse(messageId) {
      await chatApi.regenerateResponse(messageId);
      // Handling of new assistant message will occur via WebSocket stream
    },

    // ---------------------------------------------------------
    // Streaming helpers (called from WebSocket)
    // ---------------------------------------------------------
    streamMessage(msg) {
      set((s) => {
        s.streamingMessage = msg;
        const list = s.messages.get(msg.thread_id) ?? [];
        s.messages.set(msg.thread_id, [...list, msg]);
      });
    },

    async addStreamChunk(messageId, chunk, isFinal) {
      // helper to merge chunk into message
      for (const [threadId, list] of get().messages.entries()) {
        const idx = list.findIndex((m) => m.id === messageId);
        if (idx !== -1) {
          set((s) => {
            const target = s.messages.get(threadId)![idx];
            target.content = (target.content || '') + chunk;
            if (isFinal) {
              s.streamingMessage = null;
            }
          });
          break;
        }
      }
    },

    // ---------------------------------------------------------
    setActiveThreadId(threadId) {
      set((s) => {
        s.activeThreadId = threadId;
      });
    },
  })),
);
