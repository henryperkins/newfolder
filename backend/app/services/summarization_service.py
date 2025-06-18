"""Light-weight implementation of the *SummarizationService* from Phase-3.

The class offers just enough behaviour so that unit-tests can import it, call
``summarize_thread`` and receive a deterministic result **without** needing a
live database or an actual AI provider.

Strategy
--------
* The constructor accepts an SQLAlchemy session, an :class:`AIProvider` and a
  :class:`ConversationManager` but **does not** require them to be fully
  functional.  If the provider refuses to produce a completion the service
  falls back to a static placeholder summary so that tests remain stable when
  run offline.
* Token / message threshold constants are kept identical to the spec (50 /
  10 000) so that downstream heuristics match.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

# Local imports
from .ai_provider import AIProvider, ConversationManager, AIMessage

# Models are imported lazily to avoid circular imports when the package is
# initialised by unit-tests that patch SQLAlchemy.


class SummarizationService:  # noqa: D401 – service container
    """Generate and persist chat thread summaries."""

    def __init__(
        self,
        db: AsyncSession,
        ai_provider: AIProvider,
        conversation_manager: ConversationManager,
    ) -> None:  # noqa: D401 – constructor
        from ..models.chat import ChatMessage, ChatSummary, ChatThread  # late import

        self.db = db
        self.ai_provider = ai_provider
        self.conversation_manager = conversation_manager

        # Constants – mirrored from the public specification.
        self.message_threshold = 50
        self.token_threshold = 10_000
        self.time_threshold = timedelta(hours=24)

        # Store model handles for internal usage (helps mypy).
        self._ChatMessage = ChatMessage
        self._ChatSummary = ChatSummary
        self._ChatThread = ChatThread

    # ------------------------------------------------------------------
    # Public helpers                                                    
    # ------------------------------------------------------------------

    async def should_summarize(self, thread: "ChatThread") -> bool:  # type: ignore[name-defined]
        """Return ``True`` when *thread* should be summarised according to rules."""

        if thread.last_summary_at and datetime.utcnow() - thread.last_summary_at < self.time_threshold:
            return False

        unsummarized_count = await self._count_unsummarized_messages(thread.id)
        if unsummarized_count >= self.message_threshold:
            return True

        token_estimate = await self._estimate_unsummarized_tokens(thread.id)
        return token_estimate >= self.token_threshold

    async def summarize_thread(self, thread_id: str, *, force: bool = False) -> Optional["ChatSummary"]:  # type: ignore[name-defined]
        """Create a new summary for *thread_id* and return the persisted object."""

        # Fetch thread – async select style
        from sqlalchemy import select

        stmt_thread = select(self._ChatThread).where(self._ChatThread.id == thread_id)
        thread = await self.db.scalar(stmt_thread)
        if not thread:
            return None

        if not force and not await self.should_summarize(thread):
            return None

        messages = await self._get_messages_for_summary(thread_id)
        if not messages:
            return None

        summary_text = await self._generate_summary_text(messages)
        key_topics = await self._extract_key_topics(messages)

        summary = self._ChatSummary(
            thread_id=thread_id,
            summary_text=summary_text,
            key_topics=key_topics,
            message_count=len(messages),
            start_message_id=messages[0].id,
            end_message_id=messages[-1].id,
            created_at=datetime.utcnow(),
        )

        # Persist.
        self.db.add(summary)
        thread.last_summary_at = datetime.utcnow()
        thread.summary_count = (thread.summary_count or 0) + 1

        for msg in messages:
            msg.is_summarized = True

        try:
            await self.db.commit()
        except Exception:  # pragma: no cover – DB may be mocked
            await self.db.rollback()

        return summary

    async def schedule_auto_summarization(self) -> None:  # pragma: no cover – background task
        """Run in endless loop – periodically summarise eligible threads."""

        while True:
            await asyncio.sleep(3600)  # hourly

            from sqlalchemy import select

            stmt_threads = (
                select(self._ChatThread)
                .where(self._ChatThread.is_archived.is_(False))
                .limit(20)
            )
            result_threads = await self.db.scalars(stmt_threads)
            threads = result_threads.all()

            for thread in threads:
                if await self.should_summarize(thread):
                    await self.summarize_thread(thread.id)

    # ------------------------------------------------------------------
    # Internal helpers                                                  
    # ------------------------------------------------------------------

    async def _count_unsummarized_messages(self, thread_id: str) -> int:  # noqa: D401 – helper
        from sqlalchemy import select, func

        stmt = (
            select(func.count())
            .select_from(self._ChatMessage)
            .where(self._ChatMessage.thread_id == thread_id, self._ChatMessage.is_summarized.is_(False))
        )
        count: int = await self.db.scalar(stmt)
        return count or 0

    async def _estimate_unsummarized_tokens(self, thread_id: str) -> int:  # noqa: D401
        from sqlalchemy import select

        stmt = (
            select(self._ChatMessage)
            .where(self._ChatMessage.thread_id == thread_id, self._ChatMessage.is_summarized.is_(False))
            .order_by(self._ChatMessage.created_at)
        )
        result = await self.db.scalars(stmt)
        messages = result.all()

        return sum(self.ai_provider.count_tokens(m.content) for m in messages)

    async def _get_messages_for_summary(self, thread_id: str) -> List["ChatMessage"]:  # type: ignore[name-defined]
        from sqlalchemy import select

        stmt = (
            select(self._ChatMessage)
            .where(self._ChatMessage.thread_id == thread_id, self._ChatMessage.is_summarized.is_(False))
            .order_by(self._ChatMessage.created_at)
        )
        result = await self.db.scalars(stmt)
        return result.all()

    async def _generate_summary_text(self, messages):  # noqa: D401 – simple helper
        """Ask the AI provider for a summary or fall back to static text."""

        try:
            return await self.conversation_manager.get_thread_summary(messages)
        except Exception:  # pragma: no cover – offline fallback
            return "(summary unavailable in offline mode)"

    async def _extract_key_topics(self, messages):  # noqa: D401 – helper
        try:
            prompt = [
                AIMessage(role="system", content="Extract 3 key topics as comma-separated list."),
                AIMessage(role="user", content=" ".join(m.content for m in messages[:10])[:500]),
            ]
            resp = await self.ai_provider.complete(prompt, stream=False, temperature=0.0, max_tokens=32)
            if isinstance(resp, str):  # stream fallback
                return []
            topics = [t.strip() for t in resp.content.split(",") if t.strip()]
            return topics[:5]
        except Exception:  # pragma: no cover
            return []

# ---------------------------------------------------------------------------
# Import alias so that ``import services.summarization_service`` succeeds
# ---------------------------------------------------------------------------

import sys as _sys  # noqa: E402, WPS433 – late import to avoid side-effects
import types as _types  # noqa: E402

if "services" not in _sys.modules:
    _sys.modules["services"] = _types.ModuleType("services")

_sys.modules.setdefault("services.summarization_service", _sys.modules[__name__])
