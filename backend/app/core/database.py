"""Database initialisation – sync *and* async engines side-by-side.

During the incremental migration to SQLAlchemy 2.0 async we need both the
classic synchronous *Session* (for yet-to-be-ported modules) *and* the modern
*:class:`AsyncSession`* for new code.  Once the codebase is fully migrated we
can delete the sync section.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base


from .config import settings


# ---------------------------------------------------------------------------
# Async engine (new)                                                        
# ---------------------------------------------------------------------------


def _to_async_driver(url: str) -> str:
    if "+" in url:
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite://") and url != "sqlite:///:memory:":
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


async_engine = create_async_engine(_to_async_driver(settings.database_url), future=True, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False, class_=AsyncSession)


# ---------------------------------------------------------------------------
# Declarative base                                                          
# ---------------------------------------------------------------------------


Base = declarative_base()


# ---------------------------------------------------------------------------
# Dependency helper – async only                                           
# ---------------------------------------------------------------------------


# NOTE:  During the incremental migration the project exposed two separate
# providers (get_db for *sync* and get_async_db for *async* usage).  Now that
# every consumer has been ported we collapse the API back to a single
# ``get_db`` helper that yields an *AsyncSession*.


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides an AsyncSession per-request."""

    async with AsyncSessionLocal() as session:
        yield session


# Temporary alias so that any yet-to-be-updated import continues to work.
get_async_db = get_db  # type: ignore[assignment]


