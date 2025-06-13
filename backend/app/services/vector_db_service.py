import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing document embeddings in ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection_name = "documents"
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    async def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add embeddings with metadata to ChromaDB"""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

            # Convert numpy arrays to lists for ChromaDB
            embedding_lists = [emb.tolist() for emb in embeddings]

            # Extract texts from metadata for ChromaDB
            documents = [meta.get("text", "") for meta in metadata_list]

            # Off-load potentially blocking ChromaDB call so that the event
            # loop remains free for other connections.
            from backend.app.utils.concurrency import run_in_thread  # local import to avoid cyclic at start-up

            await run_in_thread(
                self.collection.add,
                embeddings=embedding_lists,
                documents=documents,
                metadatas=metadata_list,
                ids=ids,
            )

            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False

    async def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query for similar documents"""
        try:
            query_list = query_embedding.tolist()

            from backend.app.utils.concurrency import run_in_thread

            results = await run_in_thread(
                self.collection.query,
                query_embeddings=[query_list],
                n_results=top_k,
                where=filters,
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i] if results['documents'] else "",
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error querying embeddings: {e}")
            return []

    async def delete_document_chunks(
        self,
        document_id: str,
        version_id: Optional[str] = None
    ) -> bool:
        """Delete all chunks for a document or specific version"""
        try:
            where_clause = {"document_id": document_id}
            if version_id:
                where_clause["version_id"] = version_id

            # Get IDs to delete
            results = self.collection.get(where=where_clause)
            if results['ids']:
                from backend.app.utils.concurrency import run_in_thread

                await run_in_thread(self.collection.delete, ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")

            return True

        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            return False

    async def update_chunk_metadata(
        self,
        chunk_id: str,
        metadata_update: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific chunk"""
        try:
            # Get existing chunk
            from backend.app.utils.concurrency import run_in_thread

            result = await run_in_thread(self.collection.get, ids=[chunk_id])
            if not result['ids']:
                return False

            # Merge metadata
            existing_metadata = result['metadatas'][0] if result['metadatas'] else {}
            existing_metadata.update(metadata_update)

            # Update in collection
            await run_in_thread(
                self.collection.update,
                ids=[chunk_id],
                metadatas=[existing_metadata],
            )

            return True

        except Exception as e:
            logger.error(f"Error updating chunk metadata: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            from backend.app.utils.concurrency import run_in_thread

            count = await run_in_thread(self.collection.count)
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "error": str(e)}
