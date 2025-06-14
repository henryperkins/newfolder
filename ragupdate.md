Okay, I will apply the high-priority RAG improvements:
1.  **Upgrade to OpenAI embeddings (`text-embedding-3-small`)** in `file_processor_service.py` and `rag_service.py`.
2.  **Add re-ranking with `CrossEncoder`** in `rag_service.py`.

I'll also provide a structural suggestion for hybrid search in `vector_db_service.py` as a starting point, keeping in mind its dependency on a keyword search backend that isn't currently integrated with ChromaDB in your provided code.

Here are the modified files:

```python name=backend/app/services/file_processor_service.py
import hashlib
import io
import os
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import tiktoken
# Removed: from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import UploadFile
import logging
import re
from collections import Counter
from openai import AsyncOpenAI # Added for OpenAI

logger = logging.getLogger(__name__)


class FileProcessorService:
    """Service for processing uploaded documents"""

    def __init__(self, embedding_model_name: str = "text-embedding-3-small"): # Updated model
        # self.embedder = SentenceTransformer(embedding_model) # Removed
        self.embedding_model_name = embedding_model_name
        self.openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # Added
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. Embedding generation will fail.")
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 800  # tokens
        self.chunk_overlap = 200  # tokens

    async def process_file(
        self,
        file: UploadFile,
        file_path: str # file_path is unused in the original, keeping for consistency
    ) -> Dict[str, Any]:
        """Process uploaded file and return processing results"""
        try:
            # Calculate file hash
            file_content = await file.read()
            file_hash = hashlib.sha256(file_content).hexdigest()
            await file.seek(0)  # Reset file pointer

            # Extract text based on file type
            if file.content_type == "application/pdf":
                text_content, page_count = await self._extract_pdf_text(file_content)
            elif file.content_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]:
                text_content, page_count = await self._extract_docx_text(file_content)
            elif file.content_type.startswith("text/"):
                text_content = file_content.decode('utf-8', errors='replace')
                page_count = 1
            else:
                raise ValueError(f"Unsupported file type: {file.content_type}")

            # Calculate word count
            word_count = len(text_content.split())

            # Generate chunks
            chunks_texts = self._create_chunks(text_content)

            # Generate embeddings for each chunk using OpenAI
            embeddings_vectors = []
            if chunks_texts:
                try:
                    # OpenAI can process a list of texts in one call
                    response = await self.openai_client.embeddings.create(
                        input=chunks_texts,
                        model=self.embedding_model_name
                    )
                    embeddings_vectors = [np.array(item.embedding) for item in response.data]
                except Exception as e:
                    logger.error(f"Error generating embeddings with OpenAI: {e}")
                    # Fallback or error handling: return success: False or empty embeddings
                    return {
                        "success": False,
                        "error": f"Failed to generate embeddings: {str(e)}"
                    }
            
            chunk_metadata_list = []
            for i, chunk_text_item in enumerate(chunks_texts):
                metadata = {
                    "chunk_index": i,
                    "chunk_text": chunk_text_item, # Storing full chunk text in metadata for now
                    "char_start": chunk_text_item[:50],  # First 50 chars for preview
                    "token_count": len(self.tokenizer.encode(chunk_text_item))
                }
                chunk_metadata_list.append(metadata)

            # Generate suggested tags
            suggested_tags = self._generate_tags(text_content)

            return {
                "success": True,
                "file_hash": file_hash,
                "page_count": page_count,
                "word_count": word_count,
                "extracted_text": text_content[:10000],  # First 10k chars for search
                "chunks": chunks_texts, # List of chunk texts
                "embeddings": embeddings_vectors, # List of numpy arrays (embeddings)
                "chunk_metadata": chunk_metadata_list,
                "suggested_tags": suggested_tags,
                "embedding_model": self.embedding_model_name
            }

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _extract_pdf_text(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text from PDF file"""
        from backend.app.utils.concurrency import run_in_thread

        def _extract() -> Tuple[str, int]:  # heavy synchronous helper
            pdf_stream = io.BytesIO(file_content)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

            text_content = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content += page.get_text() + "\n\n"

            page_count = pdf_document.page_count
            pdf_document.close()

            return text_content.strip(), page_count

        try:
            return await run_in_thread(_extract)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

    async def _extract_docx_text(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text from DOCX file"""
        from backend.app.utils.concurrency import run_in_thread

        def _extract() -> Tuple[str, int]:
            docx_stream = io.BytesIO(file_content)
            doc = DocxDocument(docx_stream)

            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"

            # Approximate page count (assuming ~500 words per page)
            word_count = len(text_content.split())
            page_count = max(1, word_count // 500)

            return text_content.strip(), page_count

        try:
            return await run_in_thread(_extract)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text)
        chunks = []

        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start index forward (with overlap)
            start_idx += self.chunk_size - self.chunk_overlap

            # Break if we've processed all tokens
            if end_idx >= len(tokens):
                break
        
        # Filter out very small or empty chunks that might result from aggressive overlap
        return [chunk for chunk in chunks if len(self.tokenizer.encode(chunk)) > self.chunk_overlap / 2 and chunk.strip()]


    def _generate_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """Generate suggested tags from text content"""
        text_lower = text.lower()
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'that',
            'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'some', 'any', 'few', 'many', 'much', 'most',
            'other', 'another', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'now'
        }
        words = re.findall(r'\b[a-z]+\b', text_lower)
        meaningful_words = [
            word for word in words
            if word not in stop_words and len(word) > 3
        ]
        word_freq = Counter(meaningful_words)
        tags = []
        for word, _ in word_freq.most_common(max_tags * 2):
            if word.endswith('ies') and len(word) > 3 :
                tag = word[:-3] + 'y'
            elif word.endswith('es') and len(word) > 2:
                tag = word[:-2]
            elif word.endswith('s') and len(word) > 4 : # Avoid short words like 'as', 'is'
                tag = word[:-1]
            else:
                tag = word
            if tag not in tags and len(tag) > 3: # Ensure tag itself is meaningful
                tags.append(tag)
            if len(tags) >= max_tags:
                break
        return tags

    def get_file_size_limit(self) -> int:
        """Get maximum file size in bytes (50MB)"""
        return 50 * 1024 * 1024

    def get_supported_mime_types(self) -> List[str]:
        """Get list of supported MIME types"""
        return [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain",
            "text/markdown",
            "text/csv"
        ]

```

```python name=backend/app/services/rag_service.py
import os
from typing import List, Dict, Any, Optional
# Removed: from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder # Added for re-ranking
import numpy as np # Added for numpy array conversion
import logging
from openai import AsyncOpenAI # Added for OpenAI

from ..services.vector_db_service import VectorDBService # Assuming this path is correct
from ..services.ai_provider import AIProvider, AIMessage # Assuming this path is correct

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service"""

    def __init__(
        self,
        vector_db_service: VectorDBService,
        ai_provider: AIProvider,
        embedding_model_name: str = "text-embedding-3-small", # Updated model
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
        
        from backend.app.utils.concurrency import run_in_thread # For blocking reranker

        pairs = [(query, chunk.get('text', '')) for chunk in chunks]
        try:
            scores = await run_in_thread(self.reranker.predict, pairs)
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score) # Ensure score is float
            return sorted(chunks, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            return chunks # Return original chunks on error

    async def get_context_enhanced_prompt(
        self,
        query: str,
        project_id: str,
        top_k_initial_retrieval: int = 10, # Retrieve more initially for re-ranking
        top_k_final_context: int = 3      # Select top N after re-ranking for context
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

            for chunk in final_chunks_for_context: # Use re-ranked and sliced chunks
                chunk_text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})

                source_info = f"[Document: {metadata.get('document_name', 'Unknown')}"
                if metadata.get('page_number'): # Assuming 'page_number' might be in metadata
                    source_info += f", Page {metadata.get('page_number')}"
                elif metadata.get('chunk_index') is not None: # Fallback to chunk_index
                     source_info += f", Chunk {metadata.get('chunk_index')}"
                source_info += "]"
                
                # Add relevance score if available (either original or rerank_score)
                score = chunk.get('rerank_score', 1.0 - chunk.get('distance', 0.0)) # prefer rerank_score
                source_info += f" (Relevance: {score:.2f})]"


                chunk_formatted = f"{source_info}\n{chunk_text}\n"

                if total_length + len(chunk_formatted) > self.max_context_length and context_parts: # ensure at least one part if possible
                    break 

                context_parts.append(chunk_formatted)
                total_length += len(chunk_formatted)
            
            if not context_parts and relevant_chunks: # If all chunks were too long, take the first one truncated
                first_chunk = relevant_chunks[0]
                chunk_text = first_chunk.get('text', '')
                metadata = first_chunk.get('metadata', {})
                source_info = f"[Document: {metadata.get('document_name', 'Unknown')}, Chunk {metadata.get('chunk_index',0)} (Truncated)]"
                chunk_formatted = f"{source_info}\n{chunk_text[:self.max_context_length - len(source_info) - 50]}\n" # Truncate
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
        thread_id: Optional[str] = None, # thread_id is unused, keeping for consistency
        stream: bool = True
    ) -> Any:
        """Answer a query using RAG"""
        try:
            prompt = await self.get_context_enhanced_prompt(query, project_id)
            messages = [
                AIMessage(role="system", content="You are a helpful AI assistant."),
                AIMessage(role="user", content=prompt)
            ]
            if stream:
                return self.ai_provider.complete(messages, stream=True)
            else:
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

            for chunk in relevant_chunks[:top_k]: # Slice to final top_k after re-ranking
                metadata = chunk.get('metadata', {})
                doc_id = metadata.get('document_id') # Assuming document_id is in metadata

                if doc_id and doc_id not in seen_documents: # Ensure we have unique document sources
                    seen_documents.add(doc_id)
                    source = {
                        'document_id': doc_id,
                        'document_name': metadata.get('document_name', 'Unknown'),
                        'version_id': metadata.get('version_id'),
                        'relevance_score': chunk.get('rerank_score', 1.0 - chunk.get('distance', 0.0)),
                        'preview': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                    }
                    sources.append(source)
                if len(sources) >= top_k: # Stop if we have enough unique document sources
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
```

```python name=backend/app/services/vector_db_service.py
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
            # Check if collection exists, if not, create it.
            # Modifying to get_or_create_collection for robustness
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Cosine for OpenAI embeddings
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ensured.")
        except Exception as e:
            logger.error(f"Error ensuring ChromaDB collection: {e}")
            # This is critical, so re-raise or handle appropriately
            raise

    async def add_embeddings(
        self,
        embeddings: List[np.ndarray], # Expecting list of numpy arrays from FileProcessorService
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add embeddings with metadata to ChromaDB"""
        try:
            if not embeddings:
                logger.warning("No embeddings provided to add_embeddings.")
                return False
                
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            elif len(ids) != len(embeddings):
                logger.error("Mismatch between number of embeddings and IDs.")
                return False


            # Convert numpy arrays to lists of floats for ChromaDB
            embedding_lists = [emb.tolist() for emb in embeddings]

            # Extract texts from metadata for ChromaDB 'documents' field
            # ChromaDB requires a 'documents' field (list of strings) even if not strictly used for search by user.
            # It can be the chunk_text itself.
            documents_texts = [meta.get("chunk_text", "") for meta in metadata_list]
            # Ensure all metadata items have 'chunk_text' or handle missing ones.
            if len(documents_texts) != len(embeddings):
                 logger.error("Mismatch between number of embeddings and document texts derived from metadata.")
                 # Fill missing document_texts if necessary, or return error
                 documents_texts = [meta.get("chunk_text", f"Chunk text missing for ID {ids[i]}") for i, meta in enumerate(metadata_list)]


            from backend.app.utils.concurrency import run_in_thread

            await run_in_thread(
                self.collection.add,
                embeddings=embedding_lists,
                documents=documents_texts, # Pass the text of the chunks
                metadatas=metadata_list,
                ids=ids,
            )

            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB.")
            return True

        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False

    async def query(
        self,
        query_embedding: np.ndarray, # Expecting a single numpy array
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query for similar documents (semantic search)"""
        try:
            query_list = query_embedding.tolist() # Convert numpy array to list for ChromaDB

            from backend.app.utils.concurrency import run_in_thread

            results = await run_in_thread(
                self.collection.query,
                query_embeddings=[query_list], # Expects a list of embeddings
                n_results=top_k,
                where=filters,
                include=["metadatas", "documents", "distances"] # Ensure all are included
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
                        'text': text, # This is the 'document' stored alongside embedding, usually the chunk text
                        'metadata': metadata,
                        'distance': distance
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying embeddings: {e}")
            return []

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
        query_text: str, # For keyword search part
        query_embedding: np.ndarray, # For semantic search part
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
            existing_metadata.update(metadata_update) # Merge changes

            await run_in_thread(
                self.collection.update,
                ids=[chunk_id],
                metadatas=[existing_metadata] # Pass the fully merged metadata
            )
            logger.info(f"Updated metadata for chunk ID {chunk_id}.")
            return True
        except Exception as e:
            logger.error(f"Error updating chunk metadata for {chunk_id}: {e}")
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

```
**Key changes made:**

*   **`file_processor_service.py`**:
    *   Replaced `SentenceTransformer` with `AsyncOpenAI` client.
    *   Updated `embedding_model_name` to `"text-embedding-3-small"` by default.
    *   Modified embedding generation in `process_file` to use `self.openai_client.embeddings.create`, processing chunks in a batch.
    *   Ensured `OPENAI_API_KEY` is expected from environment variables.
*   **`rag_service.py`**:
    *   Replaced `SentenceTransformer` for query embedding with `AsyncOpenAI`.
    *   Added `CrossEncoder` for re-ranking, loaded during initialization.
    *   `_generate_query_embedding` now uses OpenAI.
    *   New `_rerank_chunks` method uses the `CrossEncoder` (run in a thread as it's blocking).
    *   `get_context_enhanced_prompt` now:
        *   Retrieves more initial chunks (`top_k_initial_retrieval`).
        *   Re-ranks these chunks.
        *   Selects `top_k_final_context` from re-ranked chunks to build the prompt.
        *   Includes relevance scores in the context.
    *   `get_relevant_sources` also incorporates re-ranking.
    *   `check_project_has_documents` updated to use the new embedding method.
*   **`vector_db_service.py`**:
    *   Updated `_ensure_collection` to use `get_or_create_collection`.
    *   Clarified that `hnsw:space` should be `cosine` for OpenAI embeddings.
    *   Ensured `documents` (chunk texts) are passed to `collection.add` as ChromaDB requires it.
    *   Added placeholders for `keyword_search_placeholder` and `hybrid_query` to illustrate where hybrid search logic would go. This part is **not fully implemented** and requires a separate keyword search backend.
    *   Minor improvements to logging and safety checks (e.g., for empty embeddings list).

**Next Steps & Considerations:**

1.  **Environment Variables**: Ensure `OPENAI_API_KEY` is set in your environment where the backend runs.
2.  **Dependencies**: Add `openai` and `sentence-transformers` (if not already there or if you need a specific version for `CrossEncoder`) to your `requirements.txt`.
    ```
    openai>=1.0.0
    sentence-transformers>=2.2.0 
    ```
3.  **Testing**: Thoroughly test the file processing, embedding generation, and RAG query flow.
4.  **Hybrid Search Implementation**: If you proceed with hybrid search, you'll need to:
    *   Set up a keyword search backend (e.g., PostgreSQL with FTS, Elasticsearch).
    *   Implement the `keyword_search_placeholder` method in `VectorDBService` to query this backend.
    *   Develop a robust strategy for fusing semantic and keyword search results in `hybrid_query` (e.g., Reciprocal Rank Fusion - RRF).
    *   Update `RAGService` to call `hybrid_query` instead of `query` if hybrid search is enabled.
5.  **Configuration**: Make model names (embedding, reranker) and other parameters (top\_k values, etc.) configurable if needed, rather than hardcoding them.
6.  **Error Handling**: Review and enhance error handling, especially around API calls and model interactions.
7.  **`backend.app.models`**: No direct changes were requested or made to the models in this step, but ensure your `DocumentVersion` and related models correctly store/link to the chunk metadata, document ID, version ID, etc., as these are used in filters and metadata. The `chunk_metadata` in `FileProcessorService` and `metadata_list` in `VectorDBService` should align with what you need to filter on.

These changes should provide a significant uplift in your RAG system's accuracy by leveraging better embeddings and re-ranking.

I’ve transformed [[Untitled 20]] into a more structured “RAG Enhancements” planning note with headings, summaries of the code changes, and actionable checklists. You can copy this into your vault or rename the note accordingly.

---

# RAG Enhancements

## 1. Overview  
High-priority improvements:  
- Upgrade to OpenAI embeddings (`text-embedding-3-small`)  
- Add re-ranking with `CrossEncoder`  
- Sketch out hybrid (semantic + keyword) search placeholders  

---

## 2. Code Changes Summary

### file_processor_service.py  
- Swapped out `SentenceTransformer` for OpenAI’s `AsyncOpenAI` client  
- Default embedding model → `"text-embedding-3-small"`  
- Batch-call `openai.embeddings.create` on all chunks  
- Tokenization via `tiktoken.cl100k_base`  

### rag_service.py  
- Query embeddings via OpenAI (`AsyncOpenAI`)  
- New `CrossEncoder` reranker (`ms-marco-MiniLM-L-2-v2`)  
- `_rerank_chunks()` to re-sort retrieved chunks  
- Context prompt builds from top-N reranked chunks with relevance scores  

### vector_db_service.py  
- Use `get_or_create_collection` with `hnsw:space=cosine`  
- Ensure `documents` (chunk text) passed into `collection.add`  
- Placeholder methods for keyword search & hybrid fusion  

---

## 3. Next Steps & To-Dos

### Environment & Dependencies  
- [ ] Set `OPENAI_API_KEY` in your runtime environment  
- [ ] Add to `requirements.txt`:
  - `openai>=1.0.0`
  - `sentence-transformers>=2.2.0`

### Testing  
- [ ] Validate file ingestion → chunking → embedding flow  
- [ ] End-to-end RAG query & streaming responses  

### Hybrid Search (Optional)  
- [ ] Stand up keyword-search backend (PostgreSQL FTS or Elasticsearch)  
- [ ] Implement `keyword_search_placeholder()`  
- [ ] Design result-fusion (e.g. Reciprocal Rank Fusion)  

### Config & Error Handling  
- [ ] Expose model names & top-k parameters via config  
- [ ] Harden API-call error handling & fallbacks  

### Data Models Alignment  
- [ ] Verify `DocumentVersion` model stores chunk metadata (`document_id`, `version_id`, etc.)  
- [ ] Map metadata fields for filtering in vector queries  

---
