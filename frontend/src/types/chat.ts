/**
 * Chat-specific type definitions shared across the front-end.
 *
 * These interfaces are trimmed versions of the full JSON contracts listed
 * in plan3.md §4 (API & WebSocket).  Fields that the UI does not currently
 * access are marked optional so that incoming server payloads with extra data
 * remain type-compatible.
 */

export interface ChatThread {
  id: string;
  project_id: string;
  title: string;
  created_at: string;        // ISO timestamp
  last_activity_at: string;
  message_count: number;

  /* summarisation / housekeeping */
  last_summary_at?: string | null;
  summary_count?: number;

  /* flags */
  is_archived?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  thread_id: string;
  content: string;
  is_user: boolean;
  created_at: string;

  /* assistant-only bookkeeping */
  model_used?: string;
  token_count?: number;

  /* edits & deletion */
  is_edited?: boolean;
  edited_at?: string;
  is_deleted?: boolean;
  deleted_at?: string;

  /* summarisation */
  is_summarized?: boolean;
  summary_id?: string | null;

  /* arbitrary extra fields (copy status, failure, etc.) */
  metadata?: Record<string, unknown>;
}

/**
 * Helper shapes for paginated responses.
 */
export interface ThreadsResponse {
  threads: ChatThread[];
  total: number;
  has_more: boolean;
}

export interface MessagesResponse {
  messages: ChatMessage[];
  total_tokens: number;
  has_more: boolean;
}
