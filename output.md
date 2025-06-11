# Phase-3.1 Backlog Items (additional gaps to remediate)

> The following issues were discovered after the initial **phase3_gaps_report.md**.  They are parked here for the next implementation sprint so they are not forgotten.

## 1&nbsp;&nbsp;Database

| Ref | Gap |
|-----|-----|
| DB-1 | Add `projects.last_chat_at` column via Alembic migration |
| DB-2 | Add FK & ON DELETE CASCADE from `chat_summaries.thread_id` to `chat_threads.id` |
| DB-3 | Consider max-length constraint on `ChatMessage.edit_history` JSON array |

## 2&nbsp;&nbsp;API / Services

| Ref | Gap |
|-----|-----|
| API-1 | Implement **un-archive** operation for threads (HTTP & ChatService) |
| API-2 | Expose pagination (`limit`, `offset` / `before`, `after`) on `/threads` & `/messages` |
| API-3 | CRUD endpoints for **ChatSummary** resources |
| API-4 | Reinstate message-level ownership checks in `ChatService.update_message` / `delete_message` |
| API-5 | Apply SlowAPI rate-limit decorators to HTTP endpoints |

## 3&nbsp;&nbsp;Background / Async Tasks

| Ref | Gap |
|-----|-----|
| BG-1 | Start `SummarizationService.schedule_auto_summarization` on startup (or worker) |
| BG-2 | Ensure heartbeat monitor task is cancelled during graceful shutdown |

## 4&nbsp;&nbsp;Security

| Ref | Gap |
|-----|-----|
| SEC-1 | Use real bcrypt hashing in non-test builds |
| SEC-2 | Add JWT key rotation / shorter TTLs |
| SEC-3 | Harden attachment upload pipeline (endpoint, MIME validation, virus scan) |
| SEC-4 | Sanitise Markdown/HTML before returning to clients |

## 5&nbsp;&nbsp;Operations / Configuration

| Ref | Gap |
|-----|-----|
| OPS-1 | Make RATE_LIMIT_* & HEARTBEAT_* tunable via environment variables / settings |
| OPS-2 | Implement Redis pub-sub backend for ConnectionManager (horizontal scaling) |
| OPS-3 | Extend `/health` to verify DB connectivity |

## 6&nbsp;&nbsp;Front-end

| Ref | Gap |
|-----|-----|
| FE-1 | Patch final assistant content after `stream_chunk is_final=true` |
| FE-2 | UI to un-archive / delete threads & show archived list |
| FE-3 | Context-length warning indicator (80 % / 95 %) |
| FE-4 | Better ErrorBoundary for WebSocket errors with **Reconnect** CTA |
| FE-5 | Accessibility: live-region announcements & focus management |

## 7&nbsp;&nbsp;Testing / CI

| Ref | Gap |
|-----|-----|
| TEST-1 | Restore Python unit-test suite (ChatService, AIProvider, WS, etc.) |
| TEST-2 | Add front-end Jest/Vitest and Playwright tests |
| TEST-3 | Add GitHub Actions matrix (pytest-cov, Ruff, mypy, npm lint, build) |

---

### Suggested Order of Execution

1. **Database** – migration for `last_chat_at` & FK cascade.  
2. **API** – un-archive + pagination + summary endpoints.  
3. **Background tasks** – summarisation scheduler & graceful shutdown.  
4. **Security hardening** – bcrypt, JWT, sanitisation.  
5. **Ops** – env-driven limits, Redis backend, health-checks.  
6. **Front-end polish** – streaming fix, UI affordances, a11y.  
7. **Test / CI** – install full test harness & pipelines.

---

_Last updated: 2025-06-11_
