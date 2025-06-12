from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid


class DocumentVersionBase(BaseModel):
    tags: List[str] = Field(default_factory=list, max_items=20)


class DocumentVersionCreate(DocumentVersionBase):
    pass


class DocumentVersionResponse(DocumentVersionBase):
    id: uuid.UUID
    document_id: uuid.UUID
    version_number: int
    file_path: str
    file_hash: str
    size_bytes: int
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    suggested_tags: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator('name')
    @classmethod
    def validate_filename(cls, v):
        # Basic filename validation
        invalid_chars = ['/', '\\', '\0', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Filename cannot contain {char}")
        return v.strip()


class DocumentCreate(DocumentBase):
    project_id: uuid.UUID


class DocumentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    tags: Optional[List[str]] = Field(None, max_items=20)


class DocumentResponse(DocumentBase):
    id: uuid.UUID
    project_id: uuid.UUID
    mime_type: str
    size_bytes: int
    status: str
    error_message: Optional[str] = None
    current_version_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime
    indexed_at: Optional[datetime] = None
    versions: List[DocumentVersionResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    message: str


class DocumentRevertRequest(BaseModel):
    target_version_id: uuid.UUID


class DocumentRevertResponse(BaseModel):
    document_id: uuid.UUID
    new_version_id: uuid.UUID
    message: str


class DocumentProcessingStatus(BaseModel):
    document_id: uuid.UUID
    status: str
    progress: int  # 0-100
    message: Optional[str] = None
    error: Optional[str] = None


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    project_id: uuid.UUID
    top_k: int = Field(5, ge=1, le=20)
    include_sources: bool = True


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
