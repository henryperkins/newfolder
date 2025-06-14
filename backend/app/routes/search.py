from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
import uuid

from ..dependencies.auth import get_current_user, get_async_db
from ..services.search_service import SearchService
from ..models import User
from ..schemas.search import SearchResponse

router = APIRouter(prefix="/search", tags=["search"])

@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    project_id: Optional[uuid.UUID] = Query(None, description="Filter by project"),
    limit: int = Query(20, ge=1, le=100, description="Max results per category"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_async_db)
):
    """
    Unified search across documents, chats, and semantic content.
    """
    search_service = SearchService(db)

    results = await search_service.unified_search(
        query=q,
        user_id=str(current_user.id),
        project_id=str(project_id) if project_id else None,
        limit=limit
    )

    # Convert dict results to SearchResults schema
    from ..schemas.search import SearchResults
    search_results = SearchResults(
        documents=results["documents"],
        chats=results["chats"],
        semantic_matches=results["semantic_matches"]
    )

    return SearchResponse(
        query=q,
        results=search_results,
        total_results=sum(len(v) for v in results.values())
    )
