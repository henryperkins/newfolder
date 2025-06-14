"""RAG API endpoints for testing and direct usage."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..dependencies.auth import get_current_user, get_rag_service
from ..models.user import User
from ..services.rag_service import RAGService
from ..core.config import settings

router = APIRouter(prefix="/rag", tags=["rag"])


# Request/Response Models
class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    project_id: str = Field(..., description="Project ID to search documents in")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")


class RAGQueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Document sources used")
    context_used: bool = Field(..., description="Whether document context was used")


class RAGContextRequest(BaseModel):
    query: str = Field(..., description="The question to get context for")
    project_id: str = Field(..., description="Project ID to search documents in")
    top_k: int = Field(default=3, description="Number of relevant chunks to return")


class RAGContextResponse(BaseModel):
    context: str = Field(..., description="The retrieved context")
    sources: List[Dict[str, Any]] = Field(..., description="Document sources")
    chunks_found: int = Field(..., description="Number of relevant chunks found")


class RAGStatusResponse(BaseModel):
    rag_enabled: bool = Field(..., description="Whether RAG is enabled")
    openai_configured: bool = Field(..., description="Whether OpenAI API is configured")
    chroma_db_accessible: bool = Field(..., description="Whether ChromaDB is accessible")
    total_embeddings: int = Field(..., description="Total number of embeddings stored")


# API Endpoints
@router.post("/query", response_model=RAGQueryResponse)
async def query_documents(
    request: RAGQueryRequest,
    user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
    db: AsyncSession = Depends(get_db)
):
    """Query documents using RAG to get an AI-generated answer with sources."""
    
    if not settings.rag_enabled:
        raise HTTPException(status_code=400, detail="RAG is not enabled")
    
    try:
        # Verify user has access to the project
        from ..services.chat_service import ChatService
        chat_service = ChatService(db)
        
        # Create a temporary thread to verify project access
        from sqlalchemy import select
        from ..models.project import Project
        
        stmt = select(Project).where(Project.id == request.project_id, Project.user_id == user.id)
        project = await db.scalar(stmt)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Get RAG-enhanced answer
        answer = await rag_service.answer_query(
            query=request.query,
            project_id=request.project_id,
            stream=False
        )
        
        # Get sources used for the answer
        sources = await rag_service.get_relevant_sources(
            query=request.query,
            project_id=request.project_id,
            top_k=request.top_k
        )
        
        return RAGQueryResponse(
            answer=str(answer),
            sources=sources,
            context_used=len(sources) > 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/context", response_model=RAGContextResponse)
async def get_context(
    request: RAGContextRequest,
    user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
    db: AsyncSession = Depends(get_db)
):
    """Get relevant document context for a query without generating an answer."""
    
    if not settings.rag_enabled:
        raise HTTPException(status_code=400, detail="RAG is not enabled")
    
    try:
        # Verify user has access to the project
        from sqlalchemy import select
        from ..models.project import Project
        
        stmt = select(Project).where(Project.id == request.project_id, Project.user_id == user.id)
        project = await db.scalar(stmt)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Get context-enhanced prompt
        enhanced_prompt = await rag_service.get_context_enhanced_prompt(
            query=request.query,
            project_id=request.project_id,
            top_k_final_context=request.top_k
        )
        
        # Get sources
        sources = await rag_service.get_relevant_sources(
            query=request.query,
            project_id=request.project_id,
            top_k=request.top_k
        )
        
        return RAGContextResponse(
            context=enhanced_prompt,
            sources=sources,
            chunks_found=len(sources)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


@router.get("/status", response_model=RAGStatusResponse)
async def get_rag_status(
    user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get RAG system status and health information."""
    
    try:
        # Check OpenAI configuration
        import os
        openai_configured = bool(os.environ.get("OPENAI_API_KEY"))
        
        # Check ChromaDB accessibility
        chroma_accessible = False
        total_embeddings = 0
        
        try:
            stats = rag_service.vector_db.get_stats()
            chroma_accessible = True
            total_embeddings = stats.get("total_embeddings", 0)
        except Exception:
            pass
        
        return RAGStatusResponse(
            rag_enabled=settings.rag_enabled,
            openai_configured=openai_configured,
            chroma_db_accessible=chroma_accessible,
            total_embeddings=total_embeddings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/test")
async def test_rag_pipeline(
    user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Test the RAG pipeline with a simple query."""
    
    if not settings.rag_enabled:
        raise HTTPException(status_code=400, detail="RAG is not enabled")
    
    try:
        # Test embedding generation
        test_query = "test query"
        embedding = await rag_service._generate_query_embedding(test_query)
        
        if embedding is None:
            return {"status": "error", "message": "Failed to generate embedding"}
        
        # Test vector database query
        try:
            results = await rag_service.vector_db.query(
                query_embedding=embedding,
                top_k=1
            )
            vector_db_works = True
        except Exception as e:
            vector_db_works = False
            results = str(e)
        
        return {
            "status": "success",
            "embedding_generation": "working",
            "vector_db_query": "working" if vector_db_works else "error",
            "vector_db_results": len(results) if vector_db_works else results,
            "openai_api": "working"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"RAG pipeline test failed: {str(e)}"
        }