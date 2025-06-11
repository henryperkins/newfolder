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

# Expose a module-level singleton matching the original Phase-3 design so
# that callers can simply import `connection_manager`.

connection_manager = ConnectionManager()

__all__.append("connection_manager")



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
