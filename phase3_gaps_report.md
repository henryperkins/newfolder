# Phase 3 Implementation Gaps & Open Questions

This document captures the outstanding work, hidden dependencies and ambiguous
areas that must be solved before **Phase 3 – Core Chat Experience** can be
declared production-ready.

---

## 1  Missing Code Artefacts

### Frontend

* React components described in the spec are **absent**:
  ChatView, MessageBubble, ChatInputBar, StreamingText, RecentChatsSidebar,
  ThreadHeader, ExamplePromptsInline, ConnectionStatus, MessageActions.
* No Zustand _chat_ store, WebSocket hook, markdown/Prism glue, auto-resize
  textarea or virtualised list.

### Backend

* `ChatService` (message CRUD, thread loading, token bookkeeping) is
  referenced by `websocket_manager` but not implemented.
* WebSocket route (`/ws/chat/{thread_id}`) is missing in `backend/app/main.py`.
* REST endpoints `/threads`, `/messages`, `/summaries`, … are missing.
* `SummarizationService.schedule_auto_summarization` is not wired into
  FastAPI start-up; no background runner framework chosen.
* Auth for WebSockets (token extraction / validation) undefined.
* `AIProvider` configuration (API key, model) not integrated with settings.
* Redis / queue system for background tasks & pub-sub not set up.

### Database

* Alembic migration **003** from the specification is not present; models
  `ChatThread`, `ChatMessage`, `ChatSummary` are not imported in
  `backend/app/models/__init__.py`.
* `projects.last_chat_at` column referenced by migration is missing in model.

---

## 2  Underspecified / Ambiguous Areas

* Pagination semantics: behaviour when both `before` _and_ `after` supplied. 
* How summaries influence token budgeting beyond the 75 % heuristic.
* Rate-limit constants should be tenant / env configurable, not hard-coded.
* Streaming back-pressure (slow clients, tab in background) not addressed.
* Attachment pipeline (Phase 3.5) lacks endpoints, storage, virus-scanning.
* Edit/delete permission matrix (admin, project collaborator, etc.).

---

## 3  Testing & Tooling Gaps

* Mock helpers (`MockWebSocket`, `MockAIProvider`) referenced in tests are
  absent.
* Playwright E2E tests need data-test-ids that real components don’t yet
  expose.
* CI pipeline not configured to run new unit / e2e suites.
* Pre-commit may fail without optional deps (`tiktoken`, `openai`) – needs
  extras / stubs.

---

## 4  UX / Accessibility

* Live-region announcements, keyboard focus management, screen-reader
  behaviour not implemented.
* Responsive sidebar break-points, touch gestures, virtual-keyboard handling
  require concrete CSS/JS.
* Context-length “indicator” (yellow 80 %, red 95 %) not part of component
  tree.

---

## 5  Performance / Operations

* ConnectionManager is in-memory; horizontal scaling needs Redis pub/sub.
* OpenAI cost / usage tracking & quota enforcement absent.
* No log-sanitisation / PII redaction guidelines for streamed content.

---

## 6  Documentation & Configuration

* New ENV variables (OPENAI_API_KEY, REDIS_URL, WEBSOCKET_RATE_LIMIT, …) not
  listed in README or `docker-compose.yml`.
* OpenAPI docs not updated; client SDK generation will break.

---

## 7  Recommended Next Steps

1.  **Database** – add models & Alembic migration 003; update model exports.
2.  **Backend** – implement `ChatService`, mount WebSocket route, create REST
    endpoints, wire `ConversationManager` / `AIProvider` via FastAPI
    dependency.
3.  **Frontend** – build minimal ChatView + Zustand store + WS hook for manual
    E2E testing.
4.  **Background tasks** – decide on scheduler/worker (Celery, APScheduler) and
    wire summarisation.
5.  **CI** – add test harness that spins up stub OpenAI provider and performs
    chat round-trip.

Addressing these gaps early will unblock UI work, automated tests and later
Phase-4/5 features.
