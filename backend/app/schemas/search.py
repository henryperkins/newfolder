from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SearchResultItem(BaseModel):
    id: str
    type: str
    preview: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    # Type-specific fields
    title: Optional[str] = None
    project_id: Optional[str] = None
    thread_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

class SearchResults(BaseModel):
    documents: List[SearchResultItem] = Field(default_factory=list)
    chats: List[SearchResultItem] = Field(default_factory=list)
    semantic_matches: List[SearchResultItem] = Field(default_factory=list)

class SearchResponse(BaseModel):
    query: str
    results: SearchResults
    total_results: int
