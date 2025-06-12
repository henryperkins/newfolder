import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean, Integer, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(255), nullable=False)
    mime_type = Column(String(100), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    status = Column(String(50), default="processing", nullable=False)  # processing, indexed, error
    error_message = Column(Text, nullable=True)
    current_version_id = Column(UUID(as_uuid=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="documents")
    user = relationship("User", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    chat_references = relationship("ChatDocumentReference", back_populates="document", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        Index("idx_documents_project_status", "project_id", "status"),
        Index("idx_documents_user", "user_id"),
        CheckConstraint("status IN ('processing', 'indexed', 'error')", name="valid_document_status"),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    version_number = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA256 hash
    size_bytes = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    tags = Column(ARRAY(String), default=[], nullable=False)
    suggested_tags = Column(ARRAY(String), default=[], nullable=False)

    # Metadata
    extracted_text = Column(Text, nullable=True)  # For search purposes
    chunk_count = Column(Integer, default=0, nullable=False)
    embedding_model = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    document = relationship("Document", back_populates="versions")
    user = relationship("User")

    __table_args__ = (
        Index("idx_doc_versions_document_vernum", "document_id", "version_number", unique=True),
        Index("idx_doc_versions_hash", "file_hash"),
    )


class ChatDocumentReference(Base):
    """Track which documents are referenced in chat messages"""
    __tablename__ = "chat_document_references"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    chat_message = relationship("ChatMessage")
    document = relationship("Document", back_populates="chat_references")
    version = relationship("DocumentVersion")

    __table_args__ = (
        Index("idx_chat_doc_refs_message", "chat_message_id"),
        Index("idx_chat_doc_refs_document", "document_id"),
    )
