from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_
from typing import List, Optional
import uuid
import os
import aiofiles
from datetime import datetime

from ..core.database import get_db
from ..dependencies.auth import get_current_user
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
vector_db_service = VectorDBService()
file_processor_service = FileProcessorService()


@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    overwrite_existing: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a document to a project"""
    # Verify project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

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
    existing_doc = db.query(Document).filter(
        Document.project_id == project_id,
        Document.name == file.filename
    ).first()

    if existing_doc and not overwrite_existing:
        raise HTTPException(
            status_code=409,
            detail="Document with this name already exists. Set overwrite_existing=true to create a new version."
        )

    # Create or get document
    if existing_doc:
        document = existing_doc
        version_number = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document.id
        ).count() + 1
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
        db.flush()
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
    db.flush()

    # Update document's current version
    document.current_version_id = version.id
    document.updated_at = datetime.utcnow()

    db.commit()

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
    status: Optional[str] = Query(None, regex="^(processing|indexed|error)$"),
    include_versions: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List documents in a project"""
    # Verify project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Build query
    query = db.query(Document).filter(Document.project_id == project_id)

    if status:
        query = query.filter(Document.status == status)

    if include_versions:
        query = query.options(joinedload(Document.versions))

    documents = query.order_by(Document.created_at.desc()).all()

    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document details"""
    document = db.query(Document).options(
        joinedload(Document.versions)
    ).filter(
        Document.id == document_id,
        Document.project_id == project_id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Verify user has access through project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return document


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    update_data: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update document metadata"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.project_id == project_id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Verify user has access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update fields
    if update_data.name is not None:
        document.name = update_data.name

    if update_data.tags is not None and document.current_version_id:
        # Update tags on current version
        current_version = db.query(DocumentVersion).filter(
            DocumentVersion.id == document.current_version_id
        ).first()
        if current_version:
            current_version.tags = update_data.tags

    document.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(document)

    return document


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document and all its versions"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.project_id == project_id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Verify user has access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    document_name = document.name

    # Schedule cleanup of vector embeddings
    background_tasks.add_task(
        cleanup_document_embeddings,
        str(document_id)
    )

    # Delete document (cascades to versions)
    db.delete(document)
    db.commit()

    # Log activity
    activity_logger = ActivityLogger(db)
    await activity_logger.log_activity(
        user_id=str(current_user.id),
        activity_type=ActivityType.DOCUMENT_DELETED,
        project_id=str(project_id),
        metadata={
            "document_name": document_name,
            "document_id": str(document_id)
        }
    )


@router.get("/{document_id}/versions", response_model=List[DocumentVersionResponse])
async def get_document_versions(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get version history for a document"""
    # Verify document exists and user has access
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.project_id == project_id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    versions = db.query(DocumentVersion).filter(
        DocumentVersion.document_id == document_id
    ).order_by(DocumentVersion.version_number.desc()).all()

    return versions


@router.post("/{document_id}/revert", response_model=DocumentRevertResponse)
async def revert_document_version(
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    revert_data: DocumentRevertRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revert document to a previous version"""
    # Verify document and access
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.project_id == project_id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get target version
    target_version = db.query(DocumentVersion).filter(
        DocumentVersion.id == revert_data.target_version_id,
        DocumentVersion.document_id == document_id
    ).first()

    if not target_version:
        raise HTTPException(status_code=404, detail="Target version not found")

    # Create new version as copy of target
    new_version_number = db.query(DocumentVersion).filter(
        DocumentVersion.document_id == document_id
    ).count() + 1

    # Copy file to new version
    storage_dir = f"document_storage/{project_id}/{document_id}"
    new_file_path = f"{storage_dir}/v{new_version_number}_{document.name}"

    # Copy file content
    async with aiofiles.open(target_version.file_path, 'rb') as src:
        content = await src.read()
    async with aiofiles.open(new_file_path, 'wb') as dst:
        await dst.write(content)

    # Create new version record
    new_version = DocumentVersion(
        document_id=document_id,
        user_id=current_user.id,
        version_number=new_version_number,
        file_path=new_file_path,
        file_hash=target_version.file_hash,
        size_bytes=target_version.size_bytes,
        page_count=target_version.page_count,
        word_count=target_version.word_count,
        tags=target_version.tags,
        suggested_tags=target_version.suggested_tags,
        extracted_text=target_version.extracted_text,
        chunk_count=target_version.chunk_count,
        embedding_model=target_version.embedding_model
    )
    db.add(new_version)
    db.flush()

    # Update document
    document.current_version_id = new_version.id
    document.status = "indexed"  # Assuming target version was indexed
    document.updated_at = datetime.utcnow()

    db.commit()

    # Re-index if needed
    if target_version.chunk_count > 0:
        background_tasks.add_task(
            reindex_document_version,
            str(document_id),
            str(new_version.id),
            str(target_version.id)
        )

    return DocumentRevertResponse(
        document_id=document_id,
        new_version_id=new_version.id,
        message=f"Successfully reverted to version {target_version.version_number}"
    )


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
        from ..core.database import SessionLocal
        db = SessionLocal()

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
            # Update version with processing results
            version = db.query(DocumentVersion).filter(
                DocumentVersion.id == version_id
            ).first()

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
                        'project_id': str(db.query(Document).filter(Document.id == document_id).first().project_id),
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
                document = db.query(Document).filter(
                    Document.id == document_id
                ).first()

                if document:
                    if success:
                        document.status = "indexed"
                        document.indexed_at = datetime.utcnow()
                    else:
                        document.status = "error"
                        document.error_message = "Failed to store embeddings"

                db.commit()
        else:
            # Update document with error
            document = db.query(Document).filter(
                Document.id == document_id
            ).first()

            if document:
                document.status = "error"
                document.error_message = result.get('error', 'Unknown error')
                db.commit()

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
        except:
            pass
    finally:
        db.close()


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
