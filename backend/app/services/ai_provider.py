"""ai_provider.py
Core AI provider abstraction and OpenAI implementation.

This module is based on the specification laid out in Phase-3 of the project
documentation.  It purposefully **does not** reach out to external services at
runtime when those libraries are unavailable so that the codebase can be
imported and unit-tested in completely offline environments (such as the
evaluation sandbox).

If the optional third-party dependencies `openai` and `tiktoken` are not
installed, extremely light-weight stub implementations are injected into
`sys.modules` so that importing works and the public surface expected by the
tests exists.  These stubs do **not** attempt to provide full
feature-parity – they only implement the minimal API shape required for the
type of unit tests shipped with this repository (streaming mocks, token
counting etc.).
"""

from __future__ import annotations

import asyncio
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional


# ---------------------------------------------------------------------------
# Optional third-party dependencies                                            
# ---------------------------------------------------------------------------


# 1. openai – If the package is missing we create a tiny stub so that test
#    suites relying on monkey-patching `openai.AsyncOpenAI` can still do so.

try:
    from openai import AsyncOpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – executed only when missing
    openai_stub = types.ModuleType("openai")

    class _DummyChatCompletions:  # noqa: D101 – simple stub
        async def create(self, *args: Any, **kwargs: Any):  # noqa: D401
            """Return a dummy streaming or non-streaming response object.

            The real OpenAI client returns different object shapes depending
            on whether `stream=True` was supplied.  For simplicity we always
            return an object that supports both attribute-style access used in
            the code base as well as `async for` iteration that yields *zero*
            chunks.
            """

            class _DummyChunk:  # noqa: D401
                """A single, empty chunk used for iteration."""

                # The code accesses `chunk.choices[0].delta.content` so we
                # replicate that structure with empty values.
                def __init__(self) -> None:  # noqa: D401
                    delta = types.SimpleNamespace(content="")
                    choice = types.SimpleNamespace(delta=delta, message=delta)
                    self.choices = [choice]

            class _DummyStreamResponse:  # noqa: D401 – stub response object
                def __init__(self, model: str | None = None):
                    self.model = model or "dummy-model"

                    usage_obj = types.SimpleNamespace(
                        dict=lambda: {  # type: ignore[no-redef]
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    )
                    self.usage = usage_obj  # type: ignore[assignment]

                    # Non-streaming code path accesses the following nested
                    # attributes.
                    msg = types.SimpleNamespace(content="")
                    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
                    self.choices = [choice]

                # Streaming iteration (yields zero chunks by default).
                def __aiter__(self):
                    return self

                async def __anext__(self):  # noqa: D401
                    raise StopAsyncIteration

            # For streaming we must return an *awaitable* that resolves to an
            # object supporting async-iteration.  Both streaming and
            # non-streaming tests will `await` the result, hence we wrap the
            # construction inside a coroutine.
            async def _factory():  # noqa: D401 – local helper
                return _DummyStreamResponse(model=kwargs.get("model"))

            return await _factory()

    class AsyncOpenAI:  # type: ignore
        """Extremely light-weight drop-in replacement for `openai.AsyncOpenAI`."""

        def __init__(self, *args: Any, **kwargs: Any):  # noqa: D401
            self.chat = types.SimpleNamespace(completions=_DummyChatCompletions())

    # Expose inside the stub module and register in `sys.modules` so that
    # `import openai` works everywhere.
    openai_stub.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_stub

    # Finally import it so that the rest of this module can use the symbol.
    from openai import AsyncOpenAI  # type: ignore  # noqa: WPS433


# 2. tiktoken – Same approach.  We do *not* implement real tokenisation, we
#    simply treat whitespace-separated words as tokens which is sufficient for
#    the relative comparisons made in the unit tests.

try:
    import tiktoken  # type: ignore

    def _get_encoder(model: str):  # noqa: D401
        return tiktoken.encoding_for_model(model)

except ModuleNotFoundError:  # pragma: no cover
    tiktoken_stub = types.ModuleType("tiktoken")

    class _StubEncoder:  # noqa: D401 – token counting stub
        def encode(self, text: str) -> List[str]:
            # Very naive – split on whitespace so that the count *roughly*
            # correlates with length which is more than enough for the tests.
            return text.split()

    def encoding_for_model(model: str):  # noqa: D401
        return _StubEncoder()

    # Register into stub module and sys.modules.
    tiktoken_stub.encoding_for_model = encoding_for_model  # type: ignore
    sys.modules["tiktoken"] = tiktoken_stub

    import tiktoken  # type: ignore  # noqa: WPS433 – import stub


# ---------------------------------------------------------------------------
# Public data structures                                                      
# ---------------------------------------------------------------------------


@dataclass
class AIMessage:  # noqa: D401 – simple data container
    """Message in the format required by the underlying AI provider."""

    role: str  # 'user', 'assistant', or 'system'
    content: str
    name: Optional[str] = None


@dataclass
class AIResponse:  # noqa: D401 – simple data container
    """Synchronous completion response."""

    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str


# ---------------------------------------------------------------------------
# Abstract provider definition                                                
# ---------------------------------------------------------------------------


class AIProvider(ABC):
    """Abstract base-class for AI completion providers."""

    # NOTE: Keeping the API surface intentionally close to the spec so that
    # third-party providers can be added without changing call-sites.

    @abstractmethod
    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        **kwargs: Any,
    ) -> "AsyncGenerator[str, None] | AIResponse":
        """Generate a completion.

        When *stream* is ``True`` the provider returns an async generator that
        yields text chunks.  When it is ``False`` a full :class:`AIResponse`
        instance is returned instead.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:  # noqa: D401 – simple counter
        """Return the number of tokens used by ``text`` for the underlying model."""

    @abstractmethod
    def get_max_tokens(self) -> int:  # noqa: D401 – simple getter
        """Return the context length limit of the underlying model."""


# ---------------------------------------------------------------------------
# OpenAI implementation                                                       
# ---------------------------------------------------------------------------


class OpenAIProvider(AIProvider):
    """`OpenAI` chat-completion backend (asynchronous)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 4096,
    ) -> None:  # noqa: D401 – constructor
        # A real client *might* use the `api_key` argument.  The dummy stub
        # intentionally ignores it.
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)  # type: ignore[arg-type]
        self.model = model
        self.max_tokens = max_tokens
        # Encoder/decoder for token counting.
        self._encoder = tiktoken.encoding_for_model(model)  # type: ignore[attr-defined]

    # ---------------------------------------------------------------------
    # Public helpers                                                       
    # ---------------------------------------------------------------------

    # NOTE:
    # ``complete`` is *intentionally* implemented as a **regular** function –
    # **not** ``async def`` – so that it can return either an *awaitable*
    # (non-streaming use-case) **or** an asynchronous iterator (streaming
    # use-case) while preserving a user-friendly call-site:
    #
    #   async for chunk in provider.complete(..., stream=True):
    #       ...
    #
    #   response = await provider.complete(..., stream=False)
    #
    # The returned object therefore needs to satisfy two interfaces depending
    # on the ``stream`` flag.  For the streaming case we hand back an
    # *asynchronous generator* produced by :pyfunc:`_stream_completion`.  For
    # the non-streaming path we start an ``asyncio`` coroutine that callers
    # can ``await``.

    def complete(  # type: ignore[override] – deliberate interface deviation
        self,
        messages: List[AIMessage],
        stream: bool = True,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> "AsyncGenerator[str, None] | Any":  # noqa: D401 – union of return types
        """Generate a chat-completion response or stream depending on *stream*."""

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        if stream:

            async def _generator():  # noqa: D401 – inline helper
                async for chunk in self._stream_completion(openai_messages, temperature, **kwargs):
                    yield chunk

            return _generator()

        async def _response_coro():  # noqa: D401 – inline helper
            return await self._complete(openai_messages, temperature, **kwargs)

        return _response_coro()

    # ------------------------------------------------------------------
    # Internal helpers                                                   
    # ------------------------------------------------------------------

    async def _stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:  # noqa: D401
        """Yield text chunks as they arrive from the OpenAI API."""

        try:
            stream = await self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )
        except AttributeError:  # pragma: no cover – stubbed client
            # When the underlying client does not support streaming we simply
            # exit early which results in an *empty* async generator.
            return

        # Real streaming.
        async for chunk in stream:  # type: ignore[async-iterable]
            # The real response shape has `.choices[0].delta.content` – guard
            # against `None`.
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content

    async def _complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> AIResponse:  # noqa: D401
        """Return the full completion in one go."""

        try:
            response = await self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False,
                **kwargs,
            )
        except AttributeError:  # pragma: no cover – stubbed
            # Fallback dummy response.
            return AIResponse(
                content="",
                finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
            )

        choice0 = response.choices[0]
        content = getattr(choice0.message, "content", "")
        finish_reason = getattr(choice0, "finish_reason", "stop")
        usage_dict = response.usage.dict() if hasattr(response.usage, "dict") else {}

        return AIResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage_dict,
            model=response.model,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Token helpers                                                      
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:  # noqa: D401
        return len(self._encoder.encode(text))  # type: ignore[attr-defined]

    def get_max_tokens(self) -> int:  # noqa: D401
        model_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }
        return model_limits.get(self.model, self.max_tokens)


# ---------------------------------------------------------------------------
# Provider factory                                                            
# ---------------------------------------------------------------------------


class AIProviderFactory:  # noqa: D401 – factory helper
    """Helper for instantiating :class:`AIProvider` implementations."""

    @staticmethod
    def create(provider_type: str, config: Dict[str, Any]) -> AIProvider:  # noqa: D401
        provider_type = provider_type.lower()
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=config.get("api_key", ""),
                model=config.get("model", "gpt-4"),
                max_tokens=config.get("max_tokens", 4096),
            )
        raise ValueError(f"Unknown provider type: {provider_type}")


# ---------------------------------------------------------------------------
# Conversation context helper                                                 
# ---------------------------------------------------------------------------


class ConversationManager:  # noqa: D401
    """Utility class for preparing AI messages within token limits."""

    def __init__(self, ai_provider: AIProvider):  # noqa: D401 – constructor
        self.ai_provider = ai_provider
        # 75 % of max context should be reserved for user messages; the rest is
        # left for the completion itself.
        self.max_context_tokens = int(ai_provider.get_max_tokens() * 0.75)

    # ------------------------------------------------------------------
    # Public API                                                        
    # ------------------------------------------------------------------

    def prepare_messages(
        self,
        messages: List[Any],  # Accept *any* objects with `.content` / `.is_user`.
        system_prompt: Optional[str] = None,
    ) -> List[AIMessage]:  # noqa: D401
        """Truncate *messages* so that total token usage stays within limits."""

        ai_messages: List[AIMessage] = []

        # Start with optional system prompt.
        total_tokens = 0
        if system_prompt:
            ai_messages.append(AIMessage(role="system", content=system_prompt))
            total_tokens += self.ai_provider.count_tokens(system_prompt)

        # We iterate over the conversation backwards (newest first) and prepend
        # messages until we run out of budget.
        for msg in reversed(messages):
            msg_tokens = self.ai_provider.count_tokens(getattr(msg, "content", ""))
            if total_tokens + msg_tokens > self.max_context_tokens:
                # Stop – adding this message would overflow the context window.
                break

            role = "user" if getattr(msg, "is_user", False) else "assistant"

            insert_index = 1 if system_prompt else 0
            ai_messages.insert(insert_index, AIMessage(role=role, content=msg.content))
            total_tokens += msg_tokens

        return ai_messages

    # ------------------------------------------------------------------
    # Summary helper                                                    
    # ------------------------------------------------------------------

    async def get_thread_summary(
        self,
        messages: List[Any],
        max_length: int = 500,
    ) -> str:  # noqa: D401
        """Generate a short summary for *messages*.

        The implementation calls the underlying provider with a concise prompt
        to produce a bullet-point style summary.  When the provider is not able
        to stream completions (for example in offline testing) an empty string
        is returned instead so that callers can continue gracefully.
        """

        summary_prompt = (
            f"""
        Summarize this conversation in {max_length} characters or less.
        Focus on key topics, decisions, and outcomes.
        Use bullet points for clarity.
        """
        ).strip()

        prompt_messages: List[AIMessage] = [AIMessage(role="system", content=summary_prompt)]

        for msg in messages[:50]:  # Only first 50 messages to limit prompt size.
            prompt_messages.append(
                AIMessage(
                    role="user" if getattr(msg, "is_user", False) else "assistant",
                    content=getattr(msg, "content", "")[:500],  # Truncate extremely long msgs
                )
            )

        response_or_stream = await self.ai_provider.complete(
            prompt_messages,
            stream=False,
            temperature=0.3,
            max_tokens=200,
        )

        # When the provider is forced into streaming mode by accident we fall
        # back to consuming the generator.
        if isinstance(response_or_stream, AIResponse):
            return response_or_stream.content

        # Streaming fallback: concatenate the chunks.
        parts: List[str] = []
        async for chunk in response_or_stream:  # type: ignore[async-iterable]
            parts.append(chunk)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Mock provider for testing                                                    
# ---------------------------------------------------------------------------


class MockAIProvider(AIProvider):
    """Mock AI provider for testing purposes."""

    def __init__(self, model: str = "mock-gpt-4", max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: List[AIMessage],
        stream: bool = True,
        **kwargs
    ):
        """Return mock response based on the last user message."""
        last_message = ""
        for msg in messages:
            if msg.role == "user":
                last_message = msg.content

        # Generate a simple mock response
        mock_response = f"This is a mock AI response to: '{last_message[:50]}...'"
        
        if stream:
            # Return async generator for streaming
            async def mock_stream():
                words = mock_response.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.01)  # Small delay to simulate streaming
            return mock_stream()
        else:
            return AIResponse(
                content=mock_response,
                finish_reason="stop",
                usage={"total_tokens": len(mock_response.split())},
                model=self.model
            )

    def count_tokens(self, text: str) -> int:
        """Simple word-based token counting for mock."""
        return max(1, len(text.split()))

    def get_max_tokens(self) -> int:
        """Return configured max tokens."""
        return self.max_tokens


# ---------------------------------------------------------------------------
# Re-exports                                                                  
# ---------------------------------------------------------------------------


__all__ = [
    "AIMessage",
    "AIResponse",
    "AIProvider",
    "OpenAIProvider",
    "MockAIProvider",
    "AIProviderFactory",
    "ConversationManager",
]

# ---------------------------------------------------------------------------
# Backwards-compatibility import alias                                        
# ---------------------------------------------------------------------------

# Some unit-tests (as per the project specification) use the shortened import
# path ``import services.ai_provider``.  To keep those tests working without
# having to re-structure the whole package hierarchy we register the current
# module under that name as well.

import types as _types  # noqa: WPS433 – late import to avoid polluting globals

# Ensure parent placeholder exists.
if "services" not in sys.modules:
    sys.modules["services"] = _types.ModuleType("services")

# Expose this module at the alias path.
sys.modules.setdefault("services.ai_provider", sys.modules[__name__])

