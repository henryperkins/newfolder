"""Database initialisation – sync *and* async engines side-by-side.

During the incremental migration to SQLAlchemy 2.0 async we need both the
classic synchronous *Session* (for yet-to-be-ported modules) *and* the modern
*:class:`AsyncSession`* for new code.  Once the codebase is fully migrated we
can delete the sync section.
"""

from __future__ import annotations

from typing import Generator, AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from .config import settings


# ---------------------------------------------------------------------------
# Synchronous engine (legacy)                                               
# ---------------------------------------------------------------------------


sync_engine = create_engine(settings.database_url)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


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
# Dependency helpers                                                        
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:  # legacy sync dependency
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session