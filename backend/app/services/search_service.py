from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
import logging

from ..models import Document, Project, ChatThread, ChatMessage
from .vector_db_service import VectorDBService
from .activity_logger import ActivityLogger
from ..core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vector_service = VectorDBService()

    async def unified_search(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform unified search across documents, chats, and vector embeddings.
        """
        results = {
            "documents": [],
            "chats": [],
            "semantic_matches": []
        }

        # 1. Document search
        doc_stmt = select(Document).join(Project).where(
            Project.user_id == user_id
        )

        if project_id:
            doc_stmt = doc_stmt.where(Document.project_id == project_id)

        doc_stmt = doc_stmt.where(
            or_(
                Document.title.ilike(f"%{query}%"),
                Document.content.ilike(f"%{query}%")
            )
        ).limit(limit)

        docs = (await self.db.scalars(doc_stmt)).all()

        results["documents"] = [
            {
                "id": str(doc.id),
                "title": doc.title,
                "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "project_id": str(doc.project_id),
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "type": "document"
            }
            for doc in docs
        ]

        # 2. Chat message search
        msg_stmt = select(ChatMessage).join(ChatThread).where(
            and_(
                ChatThread.user_id == user_id,
                ChatMessage.content.ilike(f"%{query}%")
            )
        )

        if project_id:
            msg_stmt = msg_stmt.where(ChatThread.project_id == project_id)

        msg_stmt = msg_stmt.limit(limit)
        messages = (await self.db.scalars(msg_stmt)).all()

        results["chats"] = [
            {
                "id": str(msg.id),
                "thread_id": str(msg.thread_id),
                "preview": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                "created_at": msg.created_at.isoformat(),
                "type": "chat"
            }
            for msg in messages
        ]

        # 3. Vector search (disabled due to ChromaDB compatibility issues)
        try:
            collection_name = f"project_{project_id}" if project_id else f"user_{user_id}"

            semantic_results = await self.vector_service.query(
                query_embedding=None,  # Would need to generate embedding from query
                top_k=limit
            )

            if semantic_results:
                results["semantic_matches"] = [
                    {
                        "id": res.get("id"),
                        "content": res.get("content", "")[:200] + "...",
                        "score": res.get("score", 0),
                        "metadata": res.get("metadata", {}),
                        "type": "semantic"
                    }
                    for res in semantic_results
                ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # Log search activity
        activity_logger = ActivityLogger(self.db)

        await activity_logger.log_activity(
            user_id=user_id,
            activity_type="search_performed",
            project_id=project_id,
            metadata={
                "query": query,
                "project_id": project_id,
                "result_counts": {
                    "documents": len(results["documents"]),
                    "chats": len(results["chats"]),
                    "semantic": len(results["semantic_matches"])
                }
            }
        )

        return results
