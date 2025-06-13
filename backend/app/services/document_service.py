"""Async service that encapsulates document CRUD & versioning logic."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.document import Document, DocumentVersion
from ..models.project import Project


class DocumentService:  # noqa: D401 – cohesive service helper
    """All heavy document interactions via async SQLAlchemy 2.x."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # Access helpers                                                    
    # ------------------------------------------------------------------

    async def _check_project_access(self, project_id: UUID, user_id: UUID) -> None:  # noqa: D401
        stmt = select(Project.id).where(Project.id == project_id, Project.user_id == user_id)
        if await self.db.scalar(stmt) is None:
            raise ValueError("Project not found")

    async def _get_document(self, document_id: UUID) -> Optional[Document]:  # noqa: D401
        stmt = select(Document).where(Document.id == document_id)
        return await self.db.scalar(stmt)

    # ------------------------------------------------------------------
    # Public API                                                        
    # ------------------------------------------------------------------

    async def get_document(self, *, project_id: UUID, document_id: UUID, user_id: UUID) -> Document:
        await self._check_project_access(project_id, user_id)

        stmt = (
            select(Document)
            .options(selectinload(Document.versions))
            .where(Document.id == document_id, Document.project_id == project_id)
        )
        doc = await self.db.scalar(stmt)
        if doc is None:
            raise ValueError("Document not found")
        return doc

    async def update_document(
        self,
        *,
        project_id: UUID,
        document_id: UUID,
        user_id: UUID,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        doc = await self.get_document(project_id=project_id, document_id=document_id, user_id=user_id)

        if name:
            doc.name = name

        if tags is not None and doc.current_version_id:
            ver_stmt = select(DocumentVersion).where(DocumentVersion.id == doc.current_version_id)
            version = await self.db.scalar(ver_stmt)
            if version:
                version.tags = tags

        doc.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def delete_document(self, *, project_id: UUID, document_id: UUID, user_id: UUID) -> Document:
        doc = await self.get_document(project_id=project_id, document_id=document_id, user_id=user_id)
        await self.db.delete(doc)
        await self.db.commit()
        return doc

    async def list_versions(
        self,
        *,
        project_id: UUID,
        document_id: UUID,
        user_id: UUID,
    ) -> List[DocumentVersion]:
        # ensure access
        await self.get_document(project_id=project_id, document_id=document_id, user_id=user_id)

        stmt = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(desc(DocumentVersion.version_number))
        )
        vers = await self.db.scalars(stmt)
        return vers.all()

    async def new_version_number(self, document_id: UUID) -> int:  # helper
        stmt = select(func.count()).select_from(DocumentVersion).where(DocumentVersion.document_id == document_id)
        count = await self.db.scalar(stmt)
        return (count or 0) + 1

    async def revert_to_version(
        self,
        *,
        project_id: UUID,
        document_id: UUID,
        target_version_id: UUID,
        user_id: UUID,
        new_file_path: str,
        new_version_number: int,
    ) -> DocumentVersion:
        await self._check_project_access(project_id, user_id)

        # load target version
        stmt = select(DocumentVersion).where(
            DocumentVersion.id == target_version_id,
            DocumentVersion.document_id == document_id,
        )
        target = await self.db.scalar(stmt)
        if target is None:
            raise ValueError("Target version not found")

        new_ver = DocumentVersion(
            document_id=document_id,
            user_id=user_id,
            version_number=new_version_number,
            file_path=new_file_path,
            file_hash=target.file_hash,
            size_bytes=target.size_bytes,
            page_count=target.page_count,
            word_count=target.word_count,
            tags=target.tags,
            suggested_tags=target.suggested_tags,
            extracted_text=target.extracted_text,
            chunk_count=target.chunk_count,
            embedding_model=target.embedding_model,
            created_at=datetime.utcnow(),
        )

        async with self.db.begin():
            self.db.add(new_ver)

            doc_stmt = select(Document).where(Document.id == document_id)
            document = await self.db.scalar(doc_stmt)
            if document:
                document.current_version_id = new_ver.id
                document.status = "indexed"
                document.updated_at = datetime.utcnow()

        return new_ver
