from fastapi import Depends, HTTPException, Request, WebSocket
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.config import settings
from ..models.user import User
from ..services.security import SecurityService
from ..services.email import EmailService
from ..services.chat_service import ChatService
from ..services.websocket_manager import ConnectionManager
from ..services.ai_provider import OpenAIProvider, AIProviderFactory


def get_security_service() -> SecurityService:
    return SecurityService(settings.secret_key, settings.algorithm)


def get_email_service() -> EmailService:
    return EmailService(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_username,
        settings.smtp_password
    )


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service)
) -> User:
    """Extract and validate user from JWT token"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = security_service.decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


# Chat-related dependencies
# ---------------------------------------------------------------------------
# Singleton services                                                         
# ---------------------------------------------------------------------------

from ..services.websocket_manager import connection_manager as _global_cm

_ai_provider = None


def get_connection_manager() -> ConnectionManager:  # noqa: D401
    """Return process-wide :class:`ConnectionManager` (Redis-aware if enabled)."""
    return _global_cm


def get_ai_provider() -> OpenAIProvider:
    """Get AI provider instance"""
    global _ai_provider
    if _ai_provider is None:
        # In a real app, this would come from settings
        # For now, create a mock provider
        try:
            api_key = getattr(settings, 'openai_api_key', 'test-key')
            _ai_provider = OpenAIProvider(api_key=api_key)
        except Exception:
            # Fallback for testing
            from ..services.ai_provider import MockAIProvider
            _ai_provider = MockAIProvider()
    return _ai_provider


def get_chat_service(
    db: Session = Depends(get_db),
    ai_provider: OpenAIProvider = Depends(get_ai_provider)
) -> ChatService:
    """Get chat service instance"""
    return ChatService(db, ai_provider)


async def get_websocket_user(
    websocket: WebSocket,
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db)
) -> User:
    """Extract and validate user from WebSocket connection (via cookie or query param)"""
    # Try to get token from cookie first (sent by browser)
    token = None
    if hasattr(websocket, 'cookies'):
        token = websocket.cookies.get("access_token")
    
    # Fallback to query parameter for programmatic access
    if not token:
        token = websocket.query_params.get("token")
    
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = security_service.decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=4001, reason="Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        await websocket.close(code=4001, reason="Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        await websocket.close(code=4001, reason="User not found or inactive")
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user