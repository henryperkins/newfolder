import os
import asyncio
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

    async def _generate_query_embedding(self, query: str, retry_count: int = 2) -> Optional[np.ndarray]:
        """Generate query embedding with retry logic and error handling."""
        for attempt in range(retry_count + 1):
            try:
                response = await self.openai_client.embeddings.create(
                    input=query,
                    model=self.embedding_model_name
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if attempt < retry_count:
                    logger.warning(f"OpenAI embedding attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Error generating query embedding with OpenAI after {retry_count + 1} attempts: {e}")
                    return None

    async def _rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.reranker or not chunks:
            return chunks
        
        try:
            from ..utils.concurrency import run_in_thread
            
            pairs = [(query, chunk.get('text', '')) for chunk in chunks]
            scores = await run_in_thread(self.reranker.predict, pairs)
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)
            return sorted(chunks, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        except ImportError:
            # Fallback if concurrency utils not available - run synchronously
            logger.warning("Concurrency utils not available, running reranker synchronously")
            pairs = [(query, chunk.get('text', '')) for chunk in chunks]
            try:
                scores = self.reranker.predict(pairs)
                for chunk, score in zip(chunks, scores):
                    chunk['rerank_score'] = float(score)
                return sorted(chunks, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            except Exception as e:
                logger.error(f"Error during re-ranking: {e}")
                return chunks
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
                logger.warning(f"Embedding generation failed for query, using fallback prompt: {query[:50]}...")
                return f"User Question: {query}\n\nPlease provide a helpful answer based on your general knowledge."

            filters = {"project_id": project_id}
            # Retrieve more chunks initially for the re-ranker to work with
            try:
                relevant_chunks = await self.vector_db.query(
                    query_embedding=query_embedding,
                    top_k=top_k_initial_retrieval, 
                    filters=filters
                )
            except Exception as e:
                logger.error(f"Vector database query failed: {e}")
                return f"User Question: {query}\n\nPlease provide a helpful answer. (Note: Document context unavailable due to search error)"

            if not relevant_chunks:
                logger.info(f"No relevant chunks found for project {project_id}")
                return f"User Question: {query}\n\nPlease provide a helpful answer. (Note: No relevant documents found in your project)"

            # Re-rank the retrieved chunks with error handling
            try:
                if self.reranker:
                    relevant_chunks = await self._rerank_chunks(query, relevant_chunks)
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original chunk order: {e}")
            
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
            logger.error(f"Unexpected error creating context-enhanced prompt: {e}")
            return f"User Question: {query}\n\nPlease provide a helpful answer. (Note: An error occurred while retrieving document context)"

    async def answer_query(
        self,
        query: str,
        project_id: str,
        thread_id: Optional[str] = None,
        stream: bool = True
    ) -> Any:
        """Answer a query using RAG with comprehensive error handling and fallbacks"""
        try:
            # Get context-enhanced prompt with built-in fallbacks
            prompt = await self.get_context_enhanced_prompt(query, project_id)

            # Prepare messages for AI provider
            messages = [
                AIMessage(role="system", content="You are a helpful AI assistant."),
                AIMessage(role="user", content=prompt)
            ]

            # Get response from AI provider with retry logic
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    if stream:
                        # Return async generator for streaming
                        return self.ai_provider.complete(messages, stream=True)
                    else:
                        # Return complete response
                        response = await self.ai_provider.complete(messages, stream=False)
                        return response.content
                except Exception as ai_error:
                    if attempt < max_retries:
                        logger.warning(f"AI provider attempt {attempt + 1} failed, retrying: {ai_error}")
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    else:
                        raise ai_error

        except Exception as e:
            logger.error(f"Error answering query with RAG: {e}")
            
            # Fallback to basic AI response without RAG context
            try:
                logger.info("Attempting fallback to basic AI response without RAG")
                fallback_messages = [
                    AIMessage(role="system", content="You are a helpful AI assistant."),
                    AIMessage(role="user", content=f"User Question: {query}\n\nPlease provide a helpful answer based on your general knowledge.")
                ]
                
                if stream:
                    return self.ai_provider.complete(fallback_messages, stream=True)
                else:
                    response = await self.ai_provider.complete(fallback_messages, stream=False)
                    return response.content
                    
            except Exception as fallback_error:
                logger.error(f"Fallback AI response also failed: {fallback_error}")
                
                # Final fallback - static error message
                error_message = "I apologize, but I'm currently experiencing technical difficulties and cannot process your query. Please try again later."
                
                if stream:
                    async def error_stream():
                        yield error_message
                    return error_stream()
                return error_message

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
