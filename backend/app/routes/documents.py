from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, or_, select, func, desc
from typing import List, Optional
import uuid
import os
import aiofiles
from datetime import datetime

from ..dependencies.auth import (
    get_current_user,
    get_websocket_user,
    get_connection_manager,
    get_document_service,
)
# async db provider
from ..core.database import get_async_db

# Both sync and async sessions are still in use while the migration is
# ongoing.  Upload/list endpoints rely on the legacy dependency, the rest has
# been migrated to the async service.
# LEGACY get_db references removed – use async counterpart everywhere in this module.
from ..models import User, Project
from ..models.document import Document, DocumentVersion
from ..models.activity import ActivityLog, ActivityType
from ..schemas.document import (
    DocumentResponse, DocumentListResponse, DocumentUploadResponse,
    DocumentVersionResponse, DocumentRevertRequest, DocumentRevertResponse,
    DocumentUpdate, DocumentProcessingStatus
)
from ..services.vector_db_service import VectorDBService
from ..services.file_processor_service import FileProcessorService
from ..services.activity_logger import ActivityLogger

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])

# Initialize services
# Singleton services (heavy – keep global)
vector_db_service = VectorDBService()
file_processor_service = FileProcessorService()


@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    overwrite_existing: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db=Depends(get_async_db)
):
    """Upload a document to a project"""
    # Verify project access
    stmt = select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    project = await db.scalar(stmt)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type
    if file.content_type not in file_processor_service.get_supported_mime_types():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    # Validate file size
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    await file.seek(0)  # Reset file pointer

    if file_size > file_processor_service.get_file_size_limit():
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 50MB limit"
        )

    # Check if document with same name exists
    stmt = select(Document).where(Document.project_id == project_id, Document.name == file.filename)
    existing_doc = await db.scalar(stmt)

    if existing_doc and not overwrite_existing:
        raise HTTPException(
            status_code=409,
            detail="Document with this name already exists. Set overwrite_existing=true to create a new version."
        )

    # Create or get document
    if existing_doc:
        document = existing_doc
        cnt_stmt = select(func.count()).select_from(DocumentVersion).where(DocumentVersion.document_id == document.id)
        version_number = (await db.scalar(cnt_stmt) or 0) + 1
    else:
        document = Document(
            project_id=project_id,
            user_id=current_user.id,
            name=file.filename,
            mime_type=file.content_type,
            size_bytes=file_size,
            status="processing"
        )
        db.add(document)
        await db.flush()
        version_number = 1

    # Create storage directory
    storage_dir = f"document_storage/{project_id}/{document.id}"
    os.makedirs(storage_dir, exist_ok=True)

    # Save file
    file_path = f"{storage_dir}/v{version_number}_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file_content)

    # Create document version
    version = DocumentVersion(
        document_id=document.id,
        user_id=current_user.id,
        version_number=version_number,
        file_path=file_path,
        file_hash="pending",  # Will be updated during processing
        size_bytes=file_size
    )
    db.add(version)
    await db.flush()

    # Update document's current version
    document.current_version_id = version.id
    document.updated_at = datetime.utcnow()

    await db.commit()

    # Schedule background processing
    background_tasks.add_task(
        process_document_async,
        document.id,
        version.id,
        file_path,
        file.filename,
        file.content_type,
        db
    )

    # Log activity
    activity_logger = ActivityLogger(db)
    await activity_logger.log_activity(
        user_id=str(current_user.id),
        activity_type=ActivityType.DOCUMENT_UPLOADED,
        project_id=str(project_id),
        metadata={
            "document_name": file.filename,
            "document_id": str(document.id),
            "version": version_number
        }
    )

    return DocumentUploadResponse(
        document_id=document.id,
        status="processing",
        message="Document uploaded successfully and queued for processing"
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    project_id: uuid.UUID,
    status: Optional[str] = Query(None, pattern="^(processing|indexed|error)$"),
    include_versions: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db=Depends(get_async_db)
):
    """List documents for a project (async)."""

    # Verify access ----------------------------------------------------
    proj_stmt = select(Project.id).where(Project.id == project_id, Project.user_id == current_user.id)
    if await db.scalar(proj_stmt) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(Document).where(Document.project_id == project_id)

    if status:
        stmt = stmt.where(Document.status == status)

    if include_versions:
        stmt = stmt.options(selectinload(Document.versions))

    stmt = stmt.order_by(desc(Document.created_at))
    docs = (await db.scalars(stmt)).all()

    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    doc_service=Depends(get_document_service),
):
    """Return document metadata + versions."""

    try:
        doc = await doc_service.get_document(
            project_id=project_id,
            document_id=document_id,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return doc


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    update_data: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    doc_service=Depends(get_document_service),
):
    """Update document name or tags."""

    try:
        updated = await doc_service.update_document(
            project_id=project_id,
            document_id=document_id,
            user_id=current_user.id,
            name=update_data.name,
            tags=update_data.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return updated


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    doc_service=Depends(get_document_service),
):
    """Delete document and its embeddings."""

    try:
        document = await doc_service.delete_document(
            project_id=project_id,
            document_id=document_id,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Schedule vector cleanup.
    background_tasks.add_task(cleanup_document_embeddings, str(document_id))

    # Activity logging (async)
    activity_logger = ActivityLogger(doc_service.db)
    await activity_logger.log_activity(
        user_id=str(current_user.id),
        activity_type=ActivityType.DOCUMENT_DELETED,
        project_id=str(project_id),
        metadata={"document_name": document.name, "document_id": str(document_id)},
    )


@router.get("/{document_id}/versions", response_model=List[DocumentVersionResponse])
async def get_document_versions(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    doc_service=Depends(get_document_service),
):
    try:
        versions = await doc_service.list_versions(
            project_id=project_id,
            document_id=document_id,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return versions


@router.post("/{document_id}/revert", response_model=DocumentRevertResponse)
async def revert_document_version(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    revert_data: DocumentRevertRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    doc_service=Depends(get_document_service),
):
    """Revert to a previous version (async implementation)."""

    # Determine new version number and create file copy ---------------
    try:
        new_ver_num = await doc_service.new_version_number(document_id)
    except Exception as exc:  # any ValueError propagated up
        raise HTTPException(status_code=404, detail="Document not found") from exc

    storage_dir = f"document_storage/{project_id}/{document_id}"
    os.makedirs(storage_dir, exist_ok=True)
    new_file_path = f"{storage_dir}/v{new_ver_num}_{document_id}.bin"

    # Copy file content (async I/O)
    target_version_id = revert_data.target_version_id
    try:
        # Fetch target version to read file path
        versions = await doc_service.list_versions(
            project_id=project_id,
            document_id=document_id,
            user_id=current_user.id,
        )
        target_version = next((v for v in versions if v.id == target_version_id), None)
        if target_version is None:
            raise ValueError("Target version not found")

        async with aiofiles.open(target_version.file_path, "rb") as src:
            content = await src.read()
        async with aiofiles.open(new_file_path, "wb") as dst:
            await dst.write(content)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Persist new version via service ---------------------------------
    new_version = await doc_service.revert_to_version(
        project_id=project_id,
        document_id=document_id,
        target_version_id=target_version_id,
        user_id=current_user.id,
        new_file_path=new_file_path,
        new_version_number=new_ver_num,
    )

    # Schedule re-index if chunks exist --------------------------------
    if target_version.chunk_count > 0:  # type: ignore[attr-defined]
        background_tasks.add_task(
            reindex_document_version,
            str(document_id),
            str(new_version.id),
            str(target_version.id),
        )

    return DocumentRevertResponse(
        document_id=document_id,
        new_version_id=new_version.id,
        message=f"Successfully reverted to version {target_version.version_number}",
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint – document status / notifications                      
# ---------------------------------------------------------------------------

from fastapi import WebSocket, WebSocketDisconnect  # noqa: E402 – local import


@router.websocket("/ws")  #  /projects/{project_id}/documents/ws
async def documents_ws_endpoint(
    project_id: uuid.UUID,
    websocket: WebSocket,
    current_user: User = Depends(get_websocket_user),
    connection_manager=Depends(get_connection_manager),
):  # noqa: D401 – FastAPI handler
    """Push-only WebSocket; server emits document status events for a project."""

    thread_id = f"project-doc-{project_id}"
    await connection_manager.connect(websocket, thread_id, user_id=str(current_user.id))

    try:
        while True:
            # Client messages are ignored; we just keep the socket open.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)


# Background task functions
async def process_document_async(
    document_id: str,
    version_id: str,
    file_path: str,
    file_name: str,
    mime_type: str,
    db: Session
):
    """Process document in background"""
    try:
        # Get fresh DB session
        from ..core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            # Process file
            with open(file_path, 'rb') as f:
                from fastapi import UploadFile
                import io
                file_content = f.read()
                file_like = io.BytesIO(file_content)
                upload_file = UploadFile(
                    filename=file_name,
                    file=file_like,
                    content_type=mime_type
                )

                result = await file_processor_service.process_file(
                    file=upload_file,
                    file_path=file_path
                )

            if result['success']:
                # Pre-fetch project id for metadata creation.
                project_id_val = await db.scalar(
                    select(Document.project_id).where(Document.id == document_id)
                )

            # Update version with processing results
            version_stmt = select(DocumentVersion).where(DocumentVersion.id == version_id)
            version = await db.scalar(version_stmt)

            if version:
                version.file_hash = result['file_hash']
                version.page_count = result.get('page_count', 0)
                version.word_count = result.get('word_count', 0)
                version.suggested_tags = result.get('suggested_tags', [])
                version.extracted_text = result.get('extracted_text', '')
                version.chunk_count = len(result.get('chunks', []))
                version.embedding_model = result.get('embedding_model', '')

                # Store embeddings in vector DB
                embeddings = result['embeddings']
                chunk_metadata = []

                for i, (chunk, metadata) in enumerate(zip(result['chunks'], result['chunk_metadata'])):
                    meta = {
                        'document_id': document_id,
                        'version_id': version_id,
                        'project_id': str(project_id_val),
                        'document_name': file_name,
                        'chunk_index': i,
                        'text': chunk,
                        **metadata
                    }
                    chunk_metadata.append(meta)

                # Add to vector DB
                success = await vector_db_service.add_embeddings(
                    embeddings=embeddings,
                    metadata_list=chunk_metadata
                )

                # Update document status
                document = await db.scalar(select(Document).where(Document.id == document_id))

                if document:
                    if success:
                        document.status = "indexed"
                        document.indexed_at = datetime.utcnow()
                        await _emit_status_update(document, db)
                    else:
                        document.status = "error"
                        document.error_message = "Failed to store embeddings"
                        await _emit_status_update(document, db)

                await db.commit()
        else:
            # Update document with error
            document = await db.scalar(select(Document).where(Document.id == document_id))

            if document:
                document.status = "error"
                document.error_message = result.get('error', 'Unknown error')
                await _emit_status_update(document, db)
                await db.commit()

    except Exception as e:
        # Update document with error
        try:
            document = db.query(Document).filter(
                Document.id == document_id
            ).first()

            if document:
                document.status = "error"
                document.error_message = str(e)
                db.commit()
                await _emit_status_update(document, db)
        except:
            pass
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper – broadcast document status                                         
# ---------------------------------------------------------------------------

async def _emit_status_update(document: Document, db_session: Session):  # noqa: D401
    """Send JSON event over project-specific WS channel via ConnectionManager."""

    try:
        from ..services.websocket_manager import connection_manager  # local import to avoid cycles

        payload = {
            "type": "document_status",
            "document_id": str(document.id),
            "status": document.status,
            "error": document.error_message,
        }

        thread_id = f"project-doc-{document.project_id}"
        await connection_manager.send_message(thread_id, payload)
    except Exception:  # pragma: no cover – never crash background task
        pass


async def cleanup_document_embeddings(document_id: str):
    """Remove document embeddings from vector DB"""
    try:
        await vector_db_service.delete_document_chunks(document_id)
    except Exception as e:
        logger.error(f"Error cleaning up embeddings for document {document_id}: {e}")


async def reindex_document_version(
    document_id: str,
    new_version_id: str,
    source_version_id: str
):
    """Copy embeddings from source version to new version"""
    # In a real implementation, you might copy the actual embeddings
    # For now, we'll just update the metadata
    pass
