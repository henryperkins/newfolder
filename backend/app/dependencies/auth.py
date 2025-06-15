# app/dependencies/auth.py

"""
Authentication and user-loading dependency providers for FastAPI.
"""

from fastapi import Depends, HTTPException, Request, WebSocket  # type: ignore[import-error]
from sqlalchemy.orm import Session  # type: ignore[import-error]
from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore[import-error]
from sqlalchemy.future import select  # type: ignore[import-error]

from ..core.config import settings
from ..core.database import get_db, get_async_db
from ..models.user import User
from ..services.security import SecurityService
from ..services.email import EmailService

# Imports for new service dependencies
from ..services.websocket_manager import connection_manager, ConnectionManager
from ..services.chat_service import ChatService
from ..services.ai_provider import AIProvider
from ..services.document_service import DocumentService  # Assuming DocumentService is in this location
# Added imports for RAGService and VectorDBService
from ..services.rag_service import RAGService
from ..services.vector_db_service import VectorDBService


def get_security_service() -> SecurityService:
    """Return a configured SecurityService for signing and verifying JWTs."""
    return SecurityService(settings.secret_key, settings.algorithm)


def get_email_service() -> EmailService:
    """Return a configured EmailService."""
    return EmailService(
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        username=settings.smtp_username,
        password=settings.smtp_password,
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service),
) -> User:
    """Extract and validate the current user from the JWT stored in cookies."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = security_service.decode_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


async def get_websocket_user(
    websocket: WebSocket,
    security_service: SecurityService = Depends(get_security_service),
    db: AsyncSession = Depends(get_async_db),
) -> User:
    """Extract and validate the current user from JWT provided via WebSocket."""
    token = (
        websocket.cookies.get("access_token")
        or websocket.query_params.get("token")
    )
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = security_service.decode_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError as exc:
        await websocket.close(code=4001, reason="Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if user is None or not user.is_active:
        await websocket.close(code=4001, reason="User not found or inactive")
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


# New service provider functions

def get_connection_manager() -> ConnectionManager:
    """Return the global ConnectionManager instance."""
    return connection_manager


def get_chat_service(db: AsyncSession = Depends(get_async_db)) -> ChatService:
    """Return a ChatService instance."""
    return ChatService(db)


def get_ai_provider() -> AIProvider:
    """Return an AIProvider instance."""
    # This might need configuration if AIProvider has specific init args
    return AIProvider()


def get_document_service(db: AsyncSession = Depends(get_async_db)) -> DocumentService:
    """Return a DocumentService instance."""
    return DocumentService(db)


def get_rag_service(
    ai_provider: AIProvider = Depends(get_ai_provider)
) -> RAGService:
    """Return a RAGService instance."""
    # Ensure VectorDBService uses an embedding model consistent with RAGService query model
    # RAGService defaults to "text-embedding-3-small" for queries.
    # VectorDBService should have used a compatible model for storing document embeddings.
    vdb_embedding_model = getattr(settings, "vector_db_embedding_model", "text-embedding-3-small")

    vector_db_service = VectorDBService(
        db_path=settings.chroma_db_path,  # This setting must be configured
        embedding_model_name=vdb_embedding_model
    )

    rag_service_instance = RAGService(
        vector_db_service=vector_db_service,
        ai_provider=ai_provider,
        embedding_model_name=getattr(settings, "rag_query_embedding_model", "text-embedding-3-small"),
        reranker_model_name=getattr(settings, "rag_reranker_model", 'cross-encoder/ms-marco-MiniLM-L-2-v2')
    )
    return rag_service_instance
