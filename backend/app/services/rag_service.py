import os
from typing import List, Dict, Any, Optional
# Removed: from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
import logging
from openai import AsyncOpenAI
from ..services.vector_db_service import VectorDBService
from ..services.ai_provider import AIProvider, AIMessage

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service"""

    def __init__(
        self,
        vector_db_service: VectorDBService,
        ai_provider: AIProvider,
        embedding_model_name: str = "text-embedding-3-small",
        reranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-2-v2'
    ):
        self.vector_db = vector_db_service
        self.ai_provider = ai_provider
        
        # Embedding model (OpenAI)
        self.embedding_model_name = embedding_model_name
        self.openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. Query embedding will fail.")

        # Re-ranker model
        try:
            self.reranker = CrossEncoder(reranker_model_name)
            logger.info(f"CrossEncoder model '{reranker_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{reranker_model_name}': {e}. Re-ranking will be skipped.")
            self.reranker = None
            
        self.max_context_length = 3000  # characters (remains character-based for now)
                                        # Consider token-based limit if using tiktoken for prompt assembly

    async def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        try:
            response = await self.openai_client.embeddings.create(
                input=query,
                model=self.embedding_model_name
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding with OpenAI: {e}")
            return None

    async def _rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.reranker or not chunks:
            return chunks
        
        from backend.app.utils.concurrency import run_in_thread

        pairs = [(query, chunk.get('text', '')) for chunk in chunks]
        try:
            scores = await run_in_thread(self.reranker.predict, pairs)
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)
            return sorted(chunks, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            return chunks

    async def get_context_enhanced_prompt(
        self,
        query: str,
        project_id: str,
        top_k_initial_retrieval: int = 10,
        top_k_final_context: int = 3
    ) -> str:
        """Generate a context-enhanced prompt with relevant document chunks"""
        try:
            query_embedding = await self._generate_query_embedding(query)
            if query_embedding is None:
                # Fallback to regular prompt without context if embedding fails
                return f"User Question: {query}\n\nPlease provide a helpful answer."

            filters = {"project_id": project_id}
            # Retrieve more chunks initially for the re-ranker to work with
            relevant_chunks = await self.vector_db.query(
                query_embedding=query_embedding,
                top_k=top_k_initial_retrieval, 
                filters=filters
            )

            # Re-rank the retrieved chunks
            if self.reranker:
                relevant_chunks = await self._rerank_chunks(query, relevant_chunks)
            
            # Select top N chunks after re-ranking for the context
            final_chunks_for_context = relevant_chunks[:top_k_final_context]

            # Build context from chunks
            context_parts = []
            total_length = 0

            for chunk in final_chunks_for_context:
                chunk_text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})

                source_info = f"[Document: {metadata.get('document_name', 'Unknown')}"
                if metadata.get('page_number'):
                    source_info += f", Page {metadata.get('page_number')}"
                elif metadata.get('chunk_index') is not None:
                     source_info += f", Chunk {metadata.get('chunk_index')}"
                source_info += "]"
                
                # Add relevance score if available (either original or rerank_score)
                score = chunk.get('rerank_score', 1.0 - chunk.get('distance', 0.0))
                source_info += f" (Relevance: {score:.2f})]"

                chunk_formatted = f"{source_info}\n{chunk_text}\n"

                if total_length + len(chunk_formatted) > self.max_context_length and context_parts:
                    break 

                context_parts.append(chunk_formatted)
                total_length += len(chunk_formatted)
            
            if not context_parts and relevant_chunks:
                first_chunk = relevant_chunks[0]
                chunk_text = first_chunk.get('text', '')
                metadata = first_chunk.get('metadata', {})
                source_info = f"[Document: {metadata.get('document_name', 'Unknown')}, Chunk {metadata.get('chunk_index',0)} (Truncated)]"
                chunk_formatted = f"{source_info}\n{chunk_text[:self.max_context_length - len(source_info) - 50]}\n"
                context_parts.append(chunk_formatted)

            context = "\n---\n".join(context_parts) if context_parts else "No relevant context found in documents."

            prompt = f"""You are a helpful AI assistant with access to the user's documents. Use the following context from relevant documents to answer the question. If the answer cannot be found in the provided context, state that clearly. Do not make up information.

Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If you reference specific information from the documents, mention which document it came from."""
            return prompt

        except Exception as e:
            logger.error(f"Error creating context-enhanced prompt: {e}")
            return f"User Question: {query}\n\nAn error occurred while retrieving context. Please provide a general answer."

    async def answer_query(
        self,
        query: str,
        project_id: str,
        thread_id: Optional[str] = None,
        stream: bool = True
    ) -> Any:
        """Answer a query using RAG"""
        try:
            # Get context-enhanced prompt
            prompt = await self.get_context_enhanced_prompt(query, project_id)

            # Prepare messages for AI provider
            messages = [
                AIMessage(role="system", content="You are a helpful AI assistant."),
                AIMessage(role="user", content=prompt)
            ]

            # Get response from AI provider
            if stream:
                # Return async generator for streaming
                return self.ai_provider.complete(messages, stream=True)
            else:
                # Return complete response
                response = await self.ai_provider.complete(messages, stream=False)
                return response.content

        except Exception as e:
            logger.error(f"Error answering query with RAG: {e}")
            # Provide a more informative error or fallback response
            if stream:
                async def error_stream():
                    yield "Sorry, an error occurred while processing your query."
                return error_stream()
            return "Sorry, an error occurred while processing your query."

    async def get_relevant_sources(
        self,
        query: str,
        project_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant document sources for a query, re-ranked"""
        try:
            query_embedding = await self._generate_query_embedding(query)
            if query_embedding is None:
                return []

            filters = {"project_id": project_id}
            # Retrieve a bit more for re-ranking if top_k is small
            initial_retrieve_k = max(top_k, 10) if self.reranker else top_k
            relevant_chunks = await self.vector_db.query(
                query_embedding=query_embedding,
                top_k=initial_retrieve_k,
                filters=filters
            )

            if self.reranker:
                relevant_chunks = await self._rerank_chunks(query, relevant_chunks)

            sources = []
            seen_documents = set()

            for chunk in relevant_chunks[:top_k]:
                metadata = chunk.get('metadata', {})
                doc_id = metadata.get('document_id')

                if doc_id and doc_id not in seen_documents:
                    seen_documents.add(doc_id)
                    source = {
                        'document_id': doc_id,
                        'document_name': metadata.get('document_name', 'Unknown'),
                        'version_id': metadata.get('version_id'),
                        'relevance_score': chunk.get('rerank_score', 1.0 - chunk.get('distance', 0.0)),
                        'preview': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                    }
                    sources.append(source)
                if len(sources) >= top_k:
                    break
            return sources

        except Exception as e:
            logger.error(f"Error getting relevant sources: {e}")
            return []

    async def check_project_has_documents(self, project_id: str) -> bool:
        """Check if a project has any indexed documents"""
        try:
            # Generate a generic embedding for testing
            test_embedding = await self._generate_query_embedding("test")
            if test_embedding is None:
                logger.warning("Failed to generate test embedding for checking project documents.")
                return False 
                
            filters = {"project_id": project_id}
            results = await self.vector_db.query(
                query_embedding=test_embedding,
                top_k=1,
                filters=filters
            )
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking project documents: {e}")
            return False
