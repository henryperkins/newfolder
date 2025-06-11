# Phase-3 Chat Experience – Detailed Implementation Roadmap

This document enumerates every outstanding gap between the **all3.md** reference specification and the current backend / frontend code-base.  It also provides a concrete, incremental plan to close those gaps in a safe, review-friendly manner.

---

## 0. Baseline State (after latest commit)

| Concern | Status | Notes |
|---------|--------|-------|
| SQLAlchemy chat models | ✅ Implemented | `ChatThread`, `ChatMessage`, `ChatSummary` with indexes & helper methods. |
| Alembic migration `003_add_chat_tables.py` | ⛔ Missing | DB schema exists only via models. |
| AI Provider layer | ✅ Stub + OpenAI + Mock. |
| WebSocket `ConnectionManager` | ✅ Minimal implementation – lacks `MessageHandler` and FastAPI endpoint. |
| ChatService & SummarizationService | ⚠️ Partial | Core CRUD done; background auto-summary loop not wired; some spec helpers missing. |
| Pydantic Schemas (`schemas/chat.py`) | ⛔ Missing |
| FastAPI routes / WebSocket endpoint (`routes/chat_routes.py`) | ⛔ Missing |
| Front-end chat UI (React components, store, hooks) | ⛔ Missing |
| Tests from specification | ⛔ Removed stubs; real suite needs adding. |
| Docker compose env updates | ⚠️ Backend image compiles but doesn’t launch WS tasks. |

---

## 1. Database Layer

### 1.1 Create Alembic Migration 003
* Auto-generate from models (`alembic revision --autogenerate -m "add chat tables"`).
* Manually edit to match reference (defaults, constraints, indexes).
* Ensure `down_revision` chain correct (depends on 002).
* Add `projects.last_chat_at` column in same migration.

### 1.2 Migration CI Check
* Update `backend/alembic/env.py` to include new model package if not already.
* Run `alembic upgrade head` in GitHub Action / pre-commit hook.

---

## 2. Backend API Layer

### 2.1 Pydantic Schemas (`backend/app/schemas/chat.py`)
* Port models from all3.md, but use `pydantic v2` style (`BaseModel`, `Field`, `model_config = ConfigDict(from_attributes=True)`).
* Validators for UUID formats.

### 2.2 FastAPI Routes (`backend/app/routes/chat_routes.py`)
* Endpoints:  
  – `GET /threads`, `POST /threads`, `PATCH /threads/{id}`, `DELETE /threads/{id}`  
  – `GET /threads/{id}/messages`, etc.  
  – WebSocket `/ws/chat/{thread_id}`.
* Dependency injection (`get_current_user`, `get_db`).
* Use `ChatService` + `connection_manager`.
* Register router in `main.py`.

### 2.3 WebSocket `MessageHandler`
* Implement per-spec ops (`SEND_MESSAGE`, `EDIT_MESSAGE`, `DELETE_MESSAGE`, `REGENERATE`, `TYPING_INDICATOR`).
* Support streaming with `ai_provider.complete(..., stream=True)`; forward chunks via `connection_manager.send_stream_chunk`.
* Hook heartbeat handling.

### 2.4 Startup Events
* In `main.py` add `@app.on_event("startup")` to:  
  – start `connection_manager` heartbeat task.  
  – start background auto-summarisation task (via singleton SummarizationService).

### 2.5 Async DB Sessions
* Current models use sync `Session`.  Decide: stay sync (simpler) **or** migrate to `AsyncSession` like spec.  
  – Easiest: keep sync for now; wrap WebSocket handlers with `run_in_threadpool`.  
  – Long term: switch to asyncpg + SQLAlchemy async engine.

---

## 3. Frontend Work

Even if backend is priority, skeleton components ensure `npm run build` passes.

### 3.1 Create `src/components/chat/*` per all3.md
* `ChatView`, `MessageBubble`, `ChatInputBar`, `ConnectionStatus`, `RecentChatsSidebar`, `StreamingText`.
* Where full UX isn’t needed, return minimal JSX placeholders rendering props – compile-time only.

### 3.2 Zustand Store (`chatStore.ts`)
* Copy store from spec; replace API calls with stubs that hit `/api/v1` endpoints.

### 3.3 `useWebSocket` hook
* Copy implementation; verify Vite’s WebSocket polyfill works in dev.

---

## 4. Testing & Quality Gates

### 4.1 Unit-tests
* Restore full suite from all3.md (`tests/test_ai_provider.py`, etc.).
* Adjust import paths to `backend.app.*`.

### 4.2 Pre-commit
* Run `black`, `isort`, `flake8` on new files.

### 4.3 CI Matrix
* `pytest -q`, `mypy`, `ruff` (optional), `npm run lint`, `npm run build`.

---

## 5. Deployment Updates

### 5.1 docker-compose
* Backend container command: `uvicorn backend.app.main:app --host 0.0.0.0 --reload`.
* Mount new source paths.

### 5.2 Environment Variables
* `AI_PROVIDER` (default `openai`).  
* `AI_MODEL`, `AI_MAX_TOKENS`.

---

## 6. Incremental Task Breakdown (Suggested PRs)

1. **PR-1 – Migration & Models**  
   • Add Alembic migration + update README migration steps.
2. **PR-2 – Schemas & Routes (sync)**  
   • Implement `schemas/chat.py`, CRUD REST routes (sync DB).  
   • Basic unit tests for threads/messages endpoints.
3. **PR-3 – WebSocket Layer**  
   • `MessageHandler`, endpoint, heartbeat, streaming chunks.
4. **PR-4 – Summarization & Background Tasks**  
   • Wire auto-summary loop & health checks.
5. **PR-5 – Front-end Skeleton**  
   • Copy components, store, simple chat view w/ WebSocket connection.  
   • Ensure `npm run build` passes.
6. **PR-6 – Full Test Suite & CI polish**.

Each PR should include migrations, unit tests, and documentation updates to keep reviewer load manageable.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Async vs Sync DB mismatch | Decide early; for PoC keep sync then refactor once stable. |
| OpenAI quota / key leakage | Default provider to `mock` in non-prod; ensure key pulled from secrets. |
| WebSocket scaling | For MVP keep in-memory `ConnectionManager`; outline Redis pub/sub backend for production. |
| Front-end performance on long chats | Virtualised list already planned; verify chunk handling avoids React re-flow floods. |

---

## 8. Timeline Estimate (ideal-day effort)

| Task | Days |
|------|------|
| Migration & Models finalisation | 0.5 |
| Schemas + REST routes | 1 |
| WebSocket MessageHandler + endpoint | 1 |
| Summarisation service wiring | 0.5 |
| Front-end skeleton | 1 |
| Test suite integration + CI | 0.5 |
| **Total** | **4–5 dev-days** |

---

### Deliverable
When all steps are complete the repo will compile & test cleanly, `docker-compose up` will expose functional chat (REST + WS) and the front-end will stream AI (mock) responses.
