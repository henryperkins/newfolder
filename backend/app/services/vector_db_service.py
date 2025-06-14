# import chromadb
# from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing document embeddings in ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        # self.client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=persist_directory,
        #     anonymized_telemetry=False
        # ))
        self.client = None
        self.collection_name = "documents"
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists"""
        # TEMPORARILY DISABLED DUE TO CHROMADB COMPATIBILITY ISSUES
        self.collection = None

    async def add_embeddings(
        self,
        embeddings: List[np.ndarray],  # Expecting list of numpy arrays from FileProcessorService
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add embeddings with metadata to ChromaDB - TEMPORARILY DISABLED"""
        # TEMPORARILY DISABLED DUE TO CHROMADB COMPATIBILITY ISSUES
        logger.warning("VectorDB service is temporarily disabled")
        return False

    async def query(
        self,
        query_embedding: np.ndarray,  # Expecting a single numpy array
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query for similar documents (semantic search)"""
        try:
            query_list = query_embedding.tolist()  # Convert numpy array to list for ChromaDB

            from backend.app.utils.concurrency import run_in_thread

            results = await run_in_thread(
                self.collection.query,
                query_embeddings=[query_list],  # Expects a list of embeddings
                n_results=top_k,
                where=filters,
                include=["metadatas", "documents", "distances"]  # Ensure all are included
            )

            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # Ensure all lists are accessed safely
                    doc_id = results['ids'][0][i]
                    text = results['documents'][0][i] if results.get('documents') and results['documents'][0] else ""
                    metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else float('inf')
                    
                    formatted_results.append({
                        'id': doc_id,
                        'text': text,  # This is the 'document' stored alongside embedding, usually the chunk text
                        'metadata': metadata,
                        'distance': distance
                    })
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
            
            from backend.app.utils.concurrency import run_in_thread
            
            # ChromaDB's delete can take a where clause directly
            await run_in_thread(self.collection.delete, where=where_clause)
            # To log count, would need a get first, or rely on Chroma's potential return value if it provides count
            logger.info(f"Attempted deletion of chunks for document {document_id} (version: {version_id}).")
            return True

        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            return False

    async def update_chunk_metadata(
        self,
        chunk_id: str,
        metadata_update: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific chunk. Note: ChromaDB's update replaces metadata."""
        try:
            from backend.app.utils.concurrency import run_in_thread
            
            # Get existing chunk to merge metadata, as Chroma's update replaces it.
            # This is inefficient. Ideally, structure metadata so full replacement is okay,
            # or ChromaDB offers partial updates in the future.
            get_result = await run_in_thread(self.collection.get, ids=[chunk_id], include=["metadatas"])
            
            if not get_result or not get_result['ids'] or not get_result['ids'][0]:
                logger.warning(f"Chunk ID {chunk_id} not found for metadata update.")
                return False

            existing_metadata = get_result['metadatas'][0] if get_result['metadatas'] and get_result['metadatas'][0] else {}
            existing_metadata.update(metadata_update)  # Merge changes

            await run_in_thread(
                self.collection.update,
                ids=[chunk_id],
                metadatas=[existing_metadata]  # Pass the fully merged metadata
            )
            logger.info(f"Updated metadata for chunk ID {chunk_id}.")
            return True

        except Exception as e:
            logger.error(f"Error updating chunk metadata for {chunk_id}: {e}")
            return False

    # --- Placeholder for Hybrid Search ---
    async def keyword_search_placeholder(
        self, 
        query_text: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Placeholder for keyword search (e.g., PostgreSQL FTS).
        This would require a separate backend/database.
        """
        logger.warning(f"Keyword search for '{query_text}' is a placeholder and not implemented. Filters: {filters}")
        # In a real implementation, this would query your keyword search engine.
        # Example structure of a result item:
        # { 'id': 'doc_kw_1', 'text': 'Full text of keyword match...', 'metadata': {...}, 'keyword_score': 0.85 }
        return []

    async def hybrid_query(
        self,
        query_text: str,  # For keyword search part
        query_embedding: np.ndarray,  # For semantic search part
        top_k_semantic: int = 3,
        top_k_keyword: int = 3,
        final_top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Combines semantic search and (placeholder) keyword search results.
        The RAGService will then re-rank this combined list.
        """
        semantic_results = await self.query(query_embedding, top_k=top_k_semantic, filters=filters)
        keyword_results = await self.keyword_search_placeholder(query_text, top_k=top_k_keyword, filters=filters)

        # Combine and de-duplicate results
        # This is a simple merge. More sophisticated fusion (e.g., RRF) might be needed.
        all_results_dict: Dict[str, Dict[str, Any]] = {}

        for res in semantic_results:
            res['search_type'] = 'semantic'
            # Normalize distance to a score (0-1, higher is better)
            res['semantic_score'] = 1.0 - res.get('distance', 1.0) 
            all_results_dict[res['id']] = res
        
        for res in keyword_results:
            res['search_type'] = 'keyword'
            # Assuming keyword_results have a 'keyword_score'
            if res['id'] in all_results_dict:
                # If already present from semantic, merge scores or add keyword score
                all_results_dict[res['id']]['keyword_score'] = res.get('keyword_score', 0.0)
                all_results_dict[res['id']]['search_type'] += '+keyword'
            else:
                all_results_dict[res['id']] = res
        
        # For now, just combine and take top N. A proper scoring fusion is needed.
        # Example: sort by semantic_score primarily, then keyword_score if semantic is missing
        combined_results = sorted(
            list(all_results_dict.values()), 
            key=lambda x: (x.get('semantic_score', 0.0) + x.get('keyword_score', 0.0))/ (2 if 'semantic_score' in x and 'keyword_score' in x else 1), # simple avg if both present
            reverse=True
        )
        
        return combined_results[:final_top_k]
    # --- End Placeholder for Hybrid Search ---

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
