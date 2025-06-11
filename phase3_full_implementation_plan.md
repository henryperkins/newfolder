# Phase 3 – Full Implementation Plan

This document converts every requirement in **plan3.md** into a concrete,
time-boxed engineering plan that leads to a production-ready Core Chat
Experience.

---

## 0  Preparation  (½ day)

| Step | Action |
|------|--------|
|0.1| Create `phase-3` Git branch, enable feature flag in `.env`|
|0.2| Add new ENV keys – `OPENAI_API_KEY`, `CHAT_RATE_LIMIT`, `REDIS_URL`, `SUMMARISE_CRON`, … to *backend/.env.example*, *docker-compose* and README|
|0.3| Open epics / labels in issue tracker|

---

## 1  Database & Core Backend  (3 days)

### 1.1 Models & Migrations

* Import `ChatThread`, `ChatMessage`, `ChatSummary` in `models/__init__.py`.
* Add `projects.last_chat_at` column + relationships in `Project`, `User`.
* Generate Alembic migration **003_add_chat_tables.py** exactly as spec.

### 1.2 ChatService (`backend/app/services/chat_service.py`)

Provides:

```python
create_thread(project_id, user_id, title, initial_message?)
create_message(thread_id, user_id, content, is_user)
get_thread_messages(thread_id, limit, before, after)
edit_message(...)
delete_message(...)
regenerate_response(...)
```

* Updates `message_count`, `total_tokens`, `last_activity_at`,
  `projects.last_chat_at`.
* Unit-tested with transactional Sqlite & Postgres fixtures.

### 1.3 FastAPI Dependencies

`get_chat_service`, `get_ai_provider`, `get_connection_manager`, `get_redis`
located in `backend/app/dependencies`.

---

## 2  Backend Routes & WebSocket  (3 days)

### 2.1 REST Routers

* `threads.py`, `messages.py`, `summaries.py` mounted in `main.py` implementing every endpoint in §4.1 of plan3.md.

### 2.2 WebSocket Endpoint

* `@app.websocket("/ws/chat/{thread_id}")`
* Auth via JWT cookie → `get_current_user`.
* Join `ConnectionManager`, fire up `MessageHandler`.

### 2.3 Rate-Limiting (Redis)

* Sliding-window helper (`check_rate_limit`).
* Configurable via `CHAT_RATE_LIMIT` ENV.

### 2.4 Summarisation Scheduler

* APScheduler job added at FastAPI startup that calls
  `SummarizationService.schedule_auto_summarization`.

---

## 3  Frontend State & Utilities  (2 days)

| Day | Tasks |
|-----|-------|
|3.1| **Zustand** `chatStore` with state/actions from spec |
|3.2| `useWebSocket` hook (auto-reconnect, heartbeat) feeding `chatStore` |
|3.3| Utilities – `api/chat.ts`, `markdown.tsx` (react-markdown + plugins), `AutoResizeTextarea`. |

---

## 4  Frontend Components  (5 days)

1. ChatView skeleton, routing `/projects/:projectId/chat/:threadId?`
2. MessageBubble + StreamingText + action menu
3. ChatInputBar (drag-drop, shortcuts)
4. RecentChatsSidebar & ThreadHeader
5. ExamplePromptsInline, ConnectionStatus, ContextIndicator

Accessibility: aria-live regions, focus-rings; Styling via Tailwind.

---

## 5  Mobile & Performance  (1 day)

* `react-window` VariableSizeList for message virtualisation.
* Touch gestures (`useSwipe`) to open/close sidebar.
* `visualViewport` keyboard handling.
* Chunk batching (100 ms) & RAF scrolling.

---

## 6  Testing Suite  (2 days)

### 6.1 Backend – pytest-asyncio

* MockAIProvider, MockWebSocket fixtures.
* Integration flow: register → thread → ws → summary.

### 6.2 Frontend – Vitest & Playwright

* Unit tests for chatStore, hooks, components.
* E2E desktop & mobile scripts from spec.

CI: GitHub Actions matrix (backend, frontend, playwright).

---

## 7  DevOps / Operations  (1 day)

* Add Redis & websocket proxy config to docker-compose.prod.
* OpenAI usage quota guard cron.
* Security-headers middleware (CSP, HSTS).

---

## 8  Documentation & Handoff  (½ day)

* README updates, new ENV vars table.
* Regenerate OpenAPI / Swagger docs.
* Architecture diagram in `docs/`.
* Operations run-book (Redis down, OpenAI quota exceeded, …).

---

## 9  Milestone Acceptance Checklist

* All Milestones 1-8 in plan3.md ✅ in staging.
* ≥90 % unit-test coverage new code, ≥80 % overall.
* Lighthouse PWA score ≥90 mobile.
* E2E demo recorded, UX sign-off.
* Tag `v0.3.0`, merge to main, deploy.

---

### Estimated Duration

* **4 engineers (parallel)** : 4 weeks (20 working days)
* **Single developer (serial)** : ≈ 7-8 weeks

This plan exhaustively covers every specified feature, edge-case, test and
operational concern needed to finish Phase 3.
