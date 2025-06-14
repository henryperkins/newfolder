import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing document embeddings in ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def _ensure_collection(self):
        """Ensure the collection exists"""
        if not self.collection:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

    async def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add embeddings with metadata to ChromaDB"""
        self._ensure_collection()
        if not ids:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        self.collection.add(
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadata_list,
            ids=ids
        )
        return True

    async def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query for similar documents"""
        self._ensure_collection()
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )
        # Process and return results, potentially joining with other data sources
        return results['documents'][0]

    async def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        self._ensure_collection()
        self.collection.delete(ids=ids)
        return True

    async def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a document"""
        self._ensure_collection()
        self.collection.update(ids=[id], metadatas=[metadata])
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        self._ensure_collection()
        count = self.collection.count()
        return {"total_embeddings": count}
