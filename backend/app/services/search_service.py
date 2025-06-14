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

        # 3. Vector search - now enabled with proper ChromaDB integration
        try:
            # Generate query embedding using OpenAI
            from openai import AsyncOpenAI
            import numpy as np
            import os
            
            openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    embedding_response = await openai_client.embeddings.create(
                        input=query,
                        model="text-embedding-3-small"
                    )
                    query_embedding = np.array(embedding_response.data[0].embedding)
                    
                    # Set up filters for project-specific search
                    filters = {"project_id": project_id} if project_id else {"user_id": user_id}
                    
                    semantic_results = await self.vector_service.query(
                        query_embedding=query_embedding,
                        top_k=limit,
                        filters=filters
                    )

                    if semantic_results:
                        results["semantic_matches"] = [
                            {
                                "id": res.get("id", ""),
                                "content": res.get("text", "")[:200] + ("..." if len(res.get("text", "")) > 200 else ""),
                                "score": 1.0 - res.get("distance", 0.5),  # Convert distance to similarity score
                                "metadata": res.get("metadata", {}),
                                "type": "semantic"
                            }
                            for res in semantic_results
                            if res.get("text")  # Only include results with text content
                        ]
                except Exception as embedding_error:
                    logger.warning(f"Failed to generate embedding for query: {embedding_error}")
            else:
                logger.warning("OPENAI_API_KEY not set, skipping vector search")
                
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
