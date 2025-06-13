from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from ..services.vector_db_service import VectorDBService
from ..services.ai_provider import AIProvider, AIMessage

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service"""

    def __init__(
        self,
        vector_db_service: VectorDBService,
        ai_provider: AIProvider,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.vector_db = vector_db_service
        self.ai_provider = ai_provider
        self.embedder = SentenceTransformer(embedding_model)
        self.max_context_length = 3000  # characters

    async def get_context_enhanced_prompt(
        self,
        query: str,
        project_id: str,
        top_k: int = 5
    ) -> str:
        """Generate a context-enhanced prompt with relevant document chunks"""
        try:
            # Generate query embedding off-thread (SentenceTransformer is blocking)
            from backend.app.utils.concurrency import run_in_thread

            query_embedding = await run_in_thread(self.embedder.encode, query)

            # Search for relevant chunks in the project
            filters = {"project_id": project_id}
            relevant_chunks = await self.vector_db.query(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Build context from chunks
            context_parts = []
            total_length = 0

            for i, chunk in enumerate(relevant_chunks):
                chunk_text = chunk['text']
                metadata = chunk['metadata']

                # Format chunk with source info
                source_info = f"[Document: {metadata.get('document_name', 'Unknown')}"
                if metadata.get('page_number'):
                    source_info += f", Page {metadata['page_number']}"
                source_info += "]"

                chunk_formatted = f"{source_info}\n{chunk_text}\n"

                # Check if adding this chunk would exceed limit
                if total_length + len(chunk_formatted) > self.max_context_length:
                    break

                context_parts.append(chunk_formatted)
                total_length += len(chunk_formatted)

            # Combine context
            context = "\n---\n".join(context_parts)

            # Create enhanced prompt
            prompt = f"""You are a helpful AI assistant with access to the user's documents. Use the following context from relevant documents to answer the question. If the answer cannot be found in the context, say so clearly.

Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If you reference specific information from the documents, mention which document it came from."""

            return prompt

        except Exception as e:
            logger.error(f"Error creating context-enhanced prompt: {e}")
            # Fallback to regular prompt without context
            return f"User Question: {query}\n\nPlease provide a helpful answer."

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
            raise

    async def get_relevant_sources(
        self,
        query: str,
        project_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant document sources for a query"""
        try:
            from backend.app.utils.concurrency import run_in_thread

            query_embedding = await run_in_thread(self.embedder.encode, query)

            # Search for relevant chunks
            filters = {"project_id": project_id}
            relevant_chunks = await self.vector_db.query(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Format sources
            sources = []
            seen_documents = set()

            for chunk in relevant_chunks:
                metadata = chunk['metadata']
                doc_id = metadata.get('document_id')

                # Avoid duplicate documents
                if doc_id and doc_id not in seen_documents:
                    seen_documents.add(doc_id)

                    source = {
                        'document_id': doc_id,
                        'document_name': metadata.get('document_name', 'Unknown'),
                        'version_id': metadata.get('version_id'),
                        'relevance_score': 1.0 - chunk['distance'],  # Convert distance to score
                        'preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                    }
                    sources.append(source)

            return sources

        except Exception as e:
            logger.error(f"Error getting relevant sources: {e}")
            return []

    async def check_project_has_documents(self, project_id: str) -> bool:
        """Check if a project has any indexed documents"""
        try:
            filters = {"project_id": project_id}
            # Do a minimal query to check if any documents exist
            results = await self.vector_db.query(
                query_embedding=self.embedder.encode("test"),
                top_k=1,
                filters=filters
            )
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking project documents: {e}")
            return False
