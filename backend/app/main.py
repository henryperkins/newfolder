from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json
import asyncio
from datetime import datetime
from .core.config import settings
# Routers including Phase-4 documents
from .routes import (
    auth_router,
    users_router,
    projects_router,
    templates_router,
    tags_router,
    activities_router,
    documents_router,
)
# Chat routes (Phase-3)
from .routes.chat_routes import router as chat_router

from .dependencies.auth import (
    get_websocket_user, 
    get_connection_manager, 
    get_chat_service,
    get_ai_provider
)
from .services.websocket_manager import ConnectionManager
from .services.chat_service import ChatService
from .services.ai_provider import AIProvider, ConversationManager
from .models.user import User

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(projects_router)
app.include_router(templates_router)
app.include_router(tags_router)
app.include_router(activities_router)
# Phase-4 documents router
app.include_router(documents_router)
# Newly added chat endpoints
app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.app_name}"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Background tasks (heartbeat monitor etc.)
# ---------------------------------------------------------------------------

from .services.websocket_manager import connection_manager


@app.on_event("startup")
async def _startup_chat_tasks():  # noqa: D401 – internal helper
    # Ensure heartbeat coroutine is running.
    await connection_manager.start_monitoring()