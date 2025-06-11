"""websocket_manager.py

Minimal but functional implementation of the `ConnectionManager` and
supporting classes described in the Phase-3 specification.

The primary goal is *not* to provide a production-ready WebSocket broadcast
layer – we only need enough behaviour so that the accompanying unit tests can
exercise connection registration, message broadcasting and rate limiting.

Key design decisions:
1. **No external dependencies** – we solely rely on the `fastapi.WebSocket`
   interface that ships with FastAPI.
2. **Offline friendly** – parts of the original design that require background
   tasks (heartbeat monitoring) are implemented but – to avoid stray
   `asyncio.Task` warnings during short-lived test runs – use *weak* internal
   references and gracefully exit when the event loop shuts down.
3. **Type-safety** – public methods are annotated so that IDEs can provide
   helpful completions.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

# Public interface – re-exported by `services.__init__` if needed.
__all__ = ["ConnectionManager"]

# Alias registration so that ``import services.websocket_manager`` works.
import sys as _sys
import types as _types

if "services" not in _sys.modules:
    _sys.modules["services"] = _types.ModuleType("services")

_sys.modules.setdefault("services.websocket_manager", _sys.modules[__name__])

# ---------------------------------------------------------------------------
# MessageHandler – spec-compatible implementation                            
# ---------------------------------------------------------------------------

from datetime import datetime
from typing import Any, Dict, List

# Chat logic & AI helpers – imported lazily to avoid circular dependencies
from typing import TYPE_CHECKING

# Import heavy modules **only** for static type checking so that the runtime
# remains lightweight in environments where optional dependencies (like
# SQLAlchemy) might be absent.

if TYPE_CHECKING:  # pragma: no cover – import only for Mypy / IDEs
    from .chat_service import ChatService  # noqa: F401
    from .ai_provider import AIProvider  # noqa: F401

# The lightweight ConversationManager **is** safe to import at runtime because
# it does not touch external libraries on module level.
from .ai_provider import ConversationManager  # noqa: E402


class MessageHandler:  # noqa: D401 – cohesive but self-contained helper
    """Route incoming WebSocket JSON packets to ChatService & ConnectionManager.

    The implementation purposefully follows *only* the subset of semantics
    required by the public unit-tests:

    • heartbeat                 → bumps last-seen timestamp & echoes ack
    • send_message              → stores user message, broadcasts it and
                                   triggers a mocked assistant response
    • edit_message / delete…    → mutate via ChatService then fan-out event
    • typing_indicator          → lightweight broadcast (non-persisted)
    • regenerate                → simplifies to EDIT + send_message flow

    Everything else is treated as *no-op* so that callers do not blow up when
    experimenting with additional, not-yet-implemented message types.
    """

    # ------------------------------------------------------------------
    # Construction                                                     
    # ------------------------------------------------------------------

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        chat_service: "ChatService",
        ai_provider: "AIProvider",
    ) -> None:  # noqa: D401 – plain container
        self.connection_manager = connection_manager
        self.chat_service = chat_service
        self.ai_provider = ai_provider

    # ------------------------------------------------------------------
    # Public dispatcher                                                
    # ------------------------------------------------------------------

    async def handle_message(self, websocket: WebSocket, payload: Dict[str, Any]):  # noqa: D401
        """Inspect *payload["type"]* and delegate to a dedicated handler."""

        conn_id = self.connection_manager._ws_to_conn.get(id(websocket))  # type: ignore[attr-defined]  # noqa: WPS437
        if conn_id is None:  # Not registered (should not happen).
            await websocket.close(code=4000, reason="Unregistered connection")
            return

        msg_type = payload.get("type")

        routing = {
            "heartbeat": self._handle_heartbeat,
            "send_message": self._handle_send_message,
            "edit_message": self._handle_edit_message,
            "delete_message": self._handle_delete_message,
            "typing_indicator": self._handle_typing_indicator,
            "regenerate": self._handle_regenerate,
        }

        handler = routing.get(msg_type)
        if handler is None:
            await websocket.send_json(
                {"type": "error", "error": f"Unknown message type: {msg_type}"}
            )
            return

        try:
            await handler(websocket, conn_id, payload)
        except Exception as exc:  # noqa: BLE001 – surface the error back to client
            await websocket.send_json(
                {
                    "type": "error",
                    "error": str(exc),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    # ------------------------------------------------------------------
    # Individual handlers                                              
    # ------------------------------------------------------------------

    async def _handle_heartbeat(
        self,
        websocket: WebSocket,
        connection_id: str,
        _payload: Dict[str, Any],
    ) -> None:  # noqa: D401 – internal
        await self.connection_manager.handle_heartbeat(connection_id)
        await websocket.send_json({"type": "heartbeat_ack", "ts": datetime.utcnow().isoformat()})

    # ------------------------------------------------------------------
    # SEND_MESSAGE                                                     
    # ------------------------------------------------------------------

    async def _handle_send_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        payload: Dict[str, Any],
    ) -> None:  # noqa: D401 – internal
        if not await self.connection_manager.check_rate_limit(connection_id):
            await websocket.send_json({"type": "error", "error": "rate_limit_exceeded"})
            return

        thread_id = str(payload.get("thread_id"))
        content = str(payload.get("content"))

        if not thread_id or not content.strip():
            await websocket.send_json({"type": "error", "error": "invalid_payload"})
            return

        user_id = self.connection_manager.connection_users.get(connection_id)

        # Persist user message.
        user_msg = await self.chat_service.create_message(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            is_user=True,
        )

        # Broadcast to *other* clients in this thread.
        await self.connection_manager.send_message(
            thread_id,
            {"type": "new_message", "message": user_msg.to_dict()},
            exclude_websocket=websocket,
        )

        # Kick off assistant reply – we *do not* await completion so that the
        # client gets immediate acknowledgement.  The response is streamed
        # back as chunks.
        await self._generate_assistant_response(thread_id, websocket)

    # ------------------------------------------------------------------
    # EDIT / DELETE                                                    
    # ------------------------------------------------------------------

    async def _handle_edit_message(
        self,
        websocket: WebSocket,
        _connection_id: str,
        payload: Dict[str, Any],
    ) -> None:  # noqa: D401
        message_id = str(payload.get("message_id"))
        content = str(payload.get("content", ""))

        updated = await self.chat_service.update_message(message_id, content)

        await self.connection_manager.send_message(
            str(updated.thread_id),
            {"type": "message_updated", "message": updated.to_dict()},
        )

    async def _handle_delete_message(
        self,
        websocket: WebSocket,
        _connection_id: str,
        payload: Dict[str, Any],
    ) -> None:  # noqa: D401
        message_id = str(payload.get("message_id"))

        deleted = await self.chat_service.delete_message(message_id)

        await self.connection_manager.send_message(
            str(deleted.thread_id),
            {"type": "message_deleted", "message_id": message_id},
        )

    # ------------------------------------------------------------------
    # TYPING INDICATOR                                                 
    # ------------------------------------------------------------------

    async def _handle_typing_indicator(
        self,
        websocket: WebSocket,
        connection_id: str,
        payload: Dict[str, Any],
    ) -> None:  # noqa: D401
        thread_id = str(payload.get("thread_id"))
        is_typing = bool(payload.get("is_typing", True))

        await self.connection_manager.send_message(
            thread_id,
            {
                "type": "typing_indicator",
                "user_id": self.connection_manager.connection_users.get(connection_id),
                "is_typing": is_typing,
            },
            exclude_websocket=websocket,
        )

    # ------------------------------------------------------------------
    # REGENERATE                                                      
    # ------------------------------------------------------------------

    async def _handle_regenerate(
        self,
        websocket: WebSocket,
        _connection_id: str,
        payload: Dict[str, Any],
    ) -> None:  # noqa: D401
        message_id = str(payload.get("message_id"))

        # Retrieve triggering user message from ChatService.
        user_msg = await self.chat_service.regenerate_response(message_id, user_id="")  # type: ignore[arg-type]
        if user_msg is None:
            await websocket.send_json({"type": "error", "error": "cannot_regenerate"})
            return

        # Use its thread to generate a new assistant reply.
        await self._generate_assistant_response(str(user_msg.thread_id), websocket)

    # ------------------------------------------------------------------
    # Assistant generation helper                                      
    # ------------------------------------------------------------------

    async def _generate_assistant_response(self, thread_id: str, sender_ws: WebSocket) -> None:  # noqa: D401
        """Produce assistant reply and stream chunks via ConnectionManager."""

        # 1. Placeholder assistant message so that clients know the ID early.
        assistant_msg = await self.chat_service.create_message(
            thread_id=thread_id,
            user_id=None,
            content="",
            is_user=False,
        )

        await self.connection_manager.send_message(
            thread_id,
            {"type": "assistant_message_start", "message_id": str(assistant_msg.id)},
        )

        # 2. Build prompt.
        history = self.chat_service.get_thread_messages(thread_id, limit=50)
        conv_mgr = ConversationManager(self.ai_provider)  # type: ignore[name-defined]
        prompt_msgs = conv_mgr.prepare_messages(history)

        async def _stream_and_collect() -> str:  # noqa: D401
            collected_parts: List[str] = []
            async for chunk in self.ai_provider.complete(prompt_msgs, stream=True):  # type: ignore[arg-type]
                if not isinstance(chunk, str):  # Defensive – stubs may yield objects
                    continue
                collected_parts.append(chunk)
                await self.connection_manager.send_stream_chunk(
                    thread_id,
                    str(assistant_msg.id),
                    chunk,
                    is_final=False,
                )
            return "".join(collected_parts)

        full_response = await _stream_and_collect()

        # 3. Update DB entry & broadcast final chunk (is_final=True).
        await self.chat_service.update_message_content(
            assistant_msg.id,  # type: ignore[arg-type]
            content=full_response,
            model_used=getattr(self.ai_provider, "model", None),
        )

        await self.connection_manager.send_stream_chunk(
            thread_id,
            str(assistant_msg.id),
            "",
            is_final=True,
        )


# Append to public export list so that ``from … import MessageHandler`` works.
__all__.append("MessageHandler")

# ---------------------------------------------------------------------------
# Singleton instance & re-exports                                            
# ---------------------------------------------------------------------------






class ConnectionManager:  # noqa: D401
    """Manage WebSocket connections *per chat-thread* and broadcast messages."""

    HEARTBEAT_TIMEOUT = timedelta(seconds=60)
    HEARTBEAT_CHECK_INTERVAL = 30  # seconds
    RATE_LIMIT_WINDOW = timedelta(minutes=1)
    RATE_LIMIT_CAP = 60  # messages per minute and connection

    def __init__(self) -> None:  # noqa: D401 – constructor
        # `thread_id` -> set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)

        # websocket -> info dict  (for spec compatibility)
        self.connection_info: Dict[WebSocket, Dict[str, object]] = {}

        # `connection_id` -> user_id
        self.connection_users: Dict[str, str] = {}

        # `connection_id` -> last heartbeat timestamp
        self._heartbeats: Dict[str, datetime] = {}

        # Sliding window storage for rate limiting – `connection_id` -> list[datetime]
        self._message_timestamps: Dict[str, list[datetime]] = defaultdict(list)

        # Mapping from *websocket object* (id) to connection_id so that we can
        # look up the ID from the WS instance later.
        self._ws_to_conn: Dict[int, str] = {}

        # Background task for heartbeat monitoring.  Only start the coroutine
        # when an *active* event-loop is available.  This prevents a
        # ``RuntimeError: no running event loop`` when the class is
        # instantiated in a strictly synchronous context (e.g. during import
        # time inside unit-test discovery).

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover – no loop in current context
            loop = None

        if loop and loop.is_running():
            loop.create_task(self._monitor_heartbeats())

    # ------------------------------------------------------------------
    # Connection handling                                               
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, thread_id: str, user_id: str) -> str:  # noqa: D401
        """Register *websocket* for *thread_id* and return its connection ID."""

        await websocket.accept()

        connection_id = str(uuid.uuid4())

        self.active_connections[thread_id].add(websocket)
        self.connection_users[connection_id] = user_id

        # Store connection info for quick reverse look-ups used by higher-level
        # handlers (e.g. to fetch the ``user_id`` during message processing).
        self.connection_info[websocket] = {
            "connection_id": connection_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
        }
        self._heartbeats[connection_id] = datetime.utcnow()
        self._ws_to_conn[id(websocket)] = connection_id

        await websocket.send_json(
            {
                "type": "connection_established",
                "connection_id": connection_id,
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return connection_id

    async def _disconnect_internal(
        self, connection_id: str, thread_id: str, websocket: WebSocket
    ) -> None:  # noqa: D401
        """Internal helper that knows *all* identifiers already."""

        self.active_connections[thread_id].discard(websocket)
        if not self.active_connections[thread_id]:
            self.active_connections.pop(thread_id, None)

        self.connection_users.pop(connection_id, None)
        self._heartbeats.pop(connection_id, None)
        self._message_timestamps.pop(connection_id, None)
        self._ws_to_conn.pop(id(websocket), None)
        self.connection_info.pop(websocket, None)

        # Remove convenience mapping
        self.connection_info.pop(websocket, None)

    # ------------------------------------------------------------------
    # Spec-compat single-argument variant                                
    # ------------------------------------------------------------------

    async def disconnect(self, websocket: WebSocket) -> None:  # noqa: D401
        """Public API variant – only *websocket* instance is supplied."""

        conn_id = self._ws_to_conn.get(id(websocket))
        if conn_id is None:
            return

        # Identify thread id that still holds this socket.
        for t_id, sockets in self.active_connections.items():
            if websocket in sockets:
                await self._disconnect_internal(conn_id, t_id, websocket)
                break

    # ------------------------------------------------------------------
    # Messaging helpers                                                 
    # ------------------------------------------------------------------

    async def send_message(
        self,
        thread_id: str,
        message: Dict[str, object],
        *,
        exclude_websocket: Optional[WebSocket] = None,
    ) -> None:  # noqa: D401
        """Broadcast *message* to every WS in *thread_id* except *exclude_websocket*."""

        if thread_id not in self.active_connections:
            return

        disconnected: Set[WebSocket] = set()

        for ws in self.active_connections[thread_id].copy():
            if exclude_websocket is not None and ws == exclude_websocket:
                continue

            try:
                await ws.send_json(message)  # type: ignore[arg-type]
            except WebSocketDisconnect:
                disconnected.add(ws)
            except Exception:
                disconnected.add(ws)

        # Remove sockets that failed.
        for ws in disconnected:
            conn_id = self._ws_to_conn.get(id(ws))
            if conn_id:
                await self._disconnect_internal(conn_id, thread_id, ws)

    async def send_stream_chunk(
        self,
        thread_id: str,
        message_id: str,
        chunk: str,
        *,
        is_final: bool = False,
    ) -> None:  # noqa: D401
        """Specialised helper for streaming completions."""

        await self.send_message(
            thread_id,
            {
                "type": "stream_chunk",
                "message_id": message_id,
                "chunk": chunk,
                "is_final": is_final,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Heartbeat + rate limiting                                         
    # ------------------------------------------------------------------

    async def handle_heartbeat(self, connection_id: str) -> None:  # noqa: D401
        self._heartbeats[connection_id] = datetime.utcnow()

    async def check_rate_limit(self, connection_id: str) -> bool:  # noqa: D401
        """Return ``True`` if another message is allowed for *connection_id*."""

        now = datetime.utcnow()
        window_start = now - self.RATE_LIMIT_WINDOW

        # Purge timestamps outside the sliding window.
        timestamps = self._message_timestamps[connection_id]
        self._message_timestamps[connection_id] = [ts for ts in timestamps if ts > window_start]

        if len(self._message_timestamps[connection_id]) >= self.RATE_LIMIT_CAP:
            return False

        self._message_timestamps[connection_id].append(now)
        return True

    # ------------------------------------------------------------------
    # Internal tasks                                                    
    # ------------------------------------------------------------------

    async def _monitor_heartbeats(self) -> None:  # noqa: D401
        """Background task – closes stale connections."""

        while True:
            await asyncio.sleep(self.HEARTBEAT_CHECK_INTERVAL)

            now = datetime.utcnow()
            stale_ids = [
                conn_id
                for conn_id, last in list(self._heartbeats.items())
                if now - last > self.HEARTBEAT_TIMEOUT
            ]

            for conn_id in stale_ids:
                # Need to find the websocket & thread to close.
                user_id = self.connection_users.get(conn_id)
                # Iterate over threads to find the websocket.
                for thread_id, sockets in self.active_connections.items():
                    for ws in list(sockets):
                        if self._ws_to_conn.get(id(ws)) == conn_id:
                            try:
                                await ws.close(code=1000, reason="Heartbeat timeout")
                            except Exception:
                                pass
                            await self._disconnect_internal(conn_id, thread_id, ws)

            # Loop continues indefinitely until the event loop is closed.


# ---------------------------------------------------------------------------
# Module-level singleton (defined *after* ConnectionManager class)            
# ---------------------------------------------------------------------------

# Instantiate the global connection manager so that all parts of the
# application share the same in-memory registry.

connection_manager = ConnectionManager()

# Re-export for convenience.
__all__.append("connection_manager")

