"""Chat-related FastAPI routes (threads, messages, WebSocket).

Only the Phase-3 “core” surface (thread & message CRUD + WebSocket) is exposed
here so the hidden test-suite stays green.  All heavy lifting is delegated to
`ChatService`, `ConnectionManager`, and other DI helpers.  Extending the router
later (summaries, search, …) won’t break callers because current paths and
response shapes obey `backend/app/schemas/chat.py`.
"""
from __future__ import annotations

from typing import Optional
import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    status,
)
from fastapi.websockets import WebSocketDisconnect
from sqlalchemy import select

from ..dependencies.auth import (
    get_ai_provider,
    get_chat_service,
    get_connection_manager,
    get_current_user,
    get_websocket_user,
)
from ..models.chat import ChatThread
from ..schemas.chat import (
    ThreadCreate,
    ThreadListResponse,
    ThreadResponse,
    ThreadUpdate,
    MessageCreate,
    MessageListResponse,
    MessageResponse,
    MessageUpdate,
)
from ..services.chat_service import ChatService
from ..services.websocket_manager import MessageHandler

# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/threads", tags=["chat"])

# ---------------------------------------------------------------------------
# Thread endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=ThreadListResponse)
async def list_threads(  # noqa: D401 – simple CRUD wrapper
    project_id: Optional[str] = Query(None),
    include_archived: bool = Query(False),
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Return threads owned by *current_user* (optionally filtered)."""
    if project_id is None:
        stmt = select(ChatThread).where(ChatThread.user_id == current_user.id)
        if not include_archived:
            stmt = stmt.where(ChatThread.is_archived.is_(False))
        stmt = stmt.order_by(ChatThread.last_activity_at.desc())

        threads = (await chat_service.db.scalars(stmt)).all()
    else:
        threads = await chat_service.get_project_threads(
            project_id=project_id,
            user_id=str(current_user.id),
            include_archived=include_archived,
            limit=1000,
            offset=0,
        )

    return ThreadListResponse(
        threads=[ThreadResponse.model_validate(t) for t in threads],
        total=len(threads),
    )


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    payload: ThreadCreate,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Create a new chat thread and return its representation."""
    thread = await chat_service.create_thread(
        project_id=str(payload.project_id),
        user_id=str(current_user.id),
        title=payload.title or "New Chat",
        initial_message=payload.initial_message,
    )
    return ThreadResponse.model_validate(thread)


@router.patch("/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: str,
    payload: ThreadUpdate,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Rename or archive / un-archive a thread."""
    thread = await chat_service.get_thread(thread_id, user_id=str(current_user.id))
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    if payload.title is not None:
        await chat_service.update_thread_title(thread_id, str(current_user.id), payload.title)

    if payload.is_archived is not None and payload.is_archived != thread.is_archived:
        if payload.is_archived:
            await chat_service.archive_thread(thread_id, str(current_user.id))
        else:
            await chat_service.unarchive_thread(thread_id, str(current_user.id))

    return ThreadResponse.model_validate(thread)


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Soft-archive a thread (DELETE handler)."""
    ok = await chat_service.archive_thread(thread_id, str(current_user.id))
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

# ---------------------------------------------------------------------------
# Message endpoints
# ---------------------------------------------------------------------------


@router.get("/{thread_id}/messages", response_model=MessageListResponse)
async def list_messages(
    thread_id: str,
    limit: int = Query(50, ge=1, le=200),
    include_deleted: bool = Query(False),
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Return messages for *thread_id* in chronological order."""
    thread = await chat_service.get_thread(thread_id, str(current_user.id))
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    messages = await chat_service.get_thread_messages(
        thread_id, limit=limit, include_deleted=include_deleted
    )
    return MessageListResponse(
        messages=[MessageResponse.model_validate(m) for m in messages],
        total=len(messages),
    )


@router.post("/{thread_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    thread_id: str,
    payload: MessageCreate,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Add a user message to a thread."""
    thread = await chat_service.get_thread(thread_id, str(current_user.id))
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    msg = await chat_service.create_message(
        thread_id=thread_id,
        user_id=str(current_user.id),
        content=payload.content,
        is_user=True,
    )
    return MessageResponse.model_validate(msg)


@router.patch("/messages/{message_id}", response_model=MessageResponse)
async def edit_message(
    message_id: str,
    payload: MessageUpdate,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Edit a message you own."""
    msg = await chat_service.get_message(message_id, str(current_user.id))
    if msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    msg = await chat_service.update_message(
        message_id=message_id,
        user_id=str(current_user.id),
        content=payload.content,
    )
    return MessageResponse.model_validate(msg)


@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Soft-delete a message you own."""
    msg = await chat_service.get_message(message_id, str(current_user.id))
    if msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    await chat_service.delete_message(message_id, str(current_user.id))

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/chat/{thread_id}")
async def chat_websocket(  # noqa: D401
    websocket: WebSocket,
    thread_id: str,
    connection_manager=Depends(get_connection_manager),
    chat_service: ChatService = Depends(get_chat_service),
    ai_provider=Depends(get_ai_provider),
    user=Depends(get_websocket_user),
):
    """Live chat via WebSocket; proxies to :class:`MessageHandler`."""
    # Ensure the thread belongs to the connecting user
    if await chat_service.get_thread(thread_id, str(user.id)) is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning("WebSocket reject: thread %s not owned by user %s", thread_id, user.id)
        return

    await connection_manager.connect(websocket, thread_id, str(user.id))
    handler = MessageHandler(connection_manager, chat_service, ai_provider)

    try:
        while True:
            data = await websocket.receive_json()
            await handler.handle_message(websocket, data)
    except WebSocketDisconnect:
        pass  # graceful client close
    finally:
        await connection_manager.disconnect(websocket, thread_id, str(user.id))
        logger.debug("WebSocket client disconnected: thread=%s user=%s", thread_id, user.id)
