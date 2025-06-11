# Phase-3 Deep-Dive Gap & Risk Analysis

This document expands on the high-level *gaps report* and captures **latent
risks, architectural bottlenecks and long-term maintenance concerns** that
were uncovered during a line-by-line review of the current code base
(backend + frontend).

---

## 1  Architecture & Domain Model

| ID | Observation | Impact | Recommendation |
|----|-------------|--------|----------------|
| A-1 | Service boundaries blur – `ChatService` mutates `Project` data. | Violates SRP; future modules couple unintentionally. | Introduce a `ProjectService` and let services emit **DomainEvents** processed by a central dispatcher. |
| A-2 | Domain invariants enforced in Python only (e.g. *single-user install*, thread counters). | Easy to bypass via direct SQL; silent data corruption. | Add DB‐level `CHECK`, `UNIQUE` and triggers; schedule periodic reconciliation job. |
| A-3 | Edit history stored as JSON array in `chat_messages`. | Queries & analytics inefficient; no audit guarantees. | Promote to first-class `chat_message_edits` table or at least GIN-indexed JSONB. |

---

## 2  Database Layer

* **Alembic drift** – model ⇄ migration mismatch (`last_chat_at`, index names) and missing downgrades.
* **Transaction granularity** – multiple `commit()` calls inside service loops → non-atomic, high I/O.
* **Optimistic locking absent** – concurrent edits override silently.

★ Create migration 004 to realign schema & add reversible downgrades.

---

## 3  API Surface & Protocol Semantics

* REST vs WS duplicate business logic – inconsistent validation.
* Error envelopes differ (HTTP exceptions vs WS `{type:"error"}` strings).
* Pagination missing on `/threads`; cursor semantics undefined on messages.

★ Introduce internal *command bus* consumed by both transports; harmonise error schema.

---

## 4  Security

| Risk | Details | Mitigation |
|------|---------|-----------|
| CSRF | JWT cookie lacks `SameSite` or CSRF token. | Add `SameSite=Strict` + double-submit CSRF token. |
| Token leakage | WS auth allows token in query string. | Require cookie or `Authorization: Bearer`. |
| Rate-limit gaps | Only WebSocket path limited. | Apply per-IP+user throttling on REST endpoints. |
| Secrets in image | `.env` may be copied into Docker layers. | Multi-stage build & runtime env-vars. |

---

## 5  WebSocket Layer

* **Back-pressure** – broadcasting awaits each `send_json`; one slow client blocks all.
* **Memory leak** – `_message_timestamps` list never purged when connection closes.
* **Heartbeat clean-up** – removes heartbeats but leaves sockets in `active_connections`.

★ Use `asyncio.wait` with timeout; prune dicts on disconnect; store only **counts** per minute instead of entire timestamp list.

---

## 6  AI Provider & Cost Control

* No retry/back-off; timeouts fall back to client defaults (600 s).
* Cost & usage not persisted; summarizer can’t enforce budget.
* Prompt content not sanitised (prompt injection risk).

★ Wrap calls with Circuit-breaker; log `usage` to DB; add content filter.

---

## 7  Summarization Service

* Token estimation under-counts when `tiktoken` absent → summarisation skipped.
* Background loop `sleep(3600)` blocks graceful shutdown.
* No concurrency guard (two workers double-summarise same window).

★ Run via APScheduler/Celery; use advisory locks; UNIQUE `(thread_id,start_message_id)`.

---

## 8  Frontend Insights

* **Optimistic updates** lack error reconciliation – UI diverges on failure.
* **React StrictMode** double-invokes effects; `useWebSocket` registers listeners twice.
* **Accessibility** – no live-regions for streamed text; focus management missing.
* **VirtualList** custom implementation brittle on variable fonts / mobile.

---

## 9  Dev-Ops & Observability

* No `/health` or `/ready` endpoints for orchestrators.
* Logging is unstructured; no request/trace ids.
* No Prometheus metrics for tokens, WS connections, rate limits.

---

## 10  Testing Strategy

* **Python** – pytest collects 0 tests; no fixtures; no coverage gate.
* **TS / React** – no unit or E2E; commit passes with type errors.

---

## High-Impact Fix Roadmap

1. Harden security (CSRF, WS token, HTTP rate-limit).
2. Schema migration 004 + `to_dict` bug-fix.
3. Unit Of Work pattern + optimistic locking.
4. WebSocket back-pressure & memory cleanup.
5. Observability: structured logs, health, metrics.
6. Audit trail table for edits / deletes.
7. Async job framework for summarisation & future tasks.
8. E2E test harness (pytest-asyncio + Playwright).
9. Refactor service boundaries with DomainEvents.

---

### ✨ Outcome

Addressing the above will close hidden reliability holes, unblock Phase-4/5
features (attachments, collaboration, multi-tenant) and lower long-term
maintenance cost.
