import hashlib
import io
import os
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import tiktoken
# Removed: from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import numpy as np
from fastapi import UploadFile
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


class FileProcessorService:
    """Service for processing uploaded documents"""

    def __init__(self, embedding_model_name: str = "text-embedding-3-small"):
        # self.embedder = SentenceTransformer(embedding_model) # Removed
        self.embedding_model_name = embedding_model_name
        self.openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. Embedding generation will fail.")
        self.embedding_model = embedding_model_name  # For backward compatibility
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 800  # tokens
        self.chunk_overlap = 200  # tokens

    async def process_file(
        self,
        file: UploadFile,
        file_path: str
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
                    "chunk_text": chunk_text_item,
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
                "chunks": chunks_texts,  # List of chunk texts
                "embeddings": embeddings_vectors,  # List of numpy arrays (embeddings)
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
        # Simple keyword extraction based on frequency
        # In production, you might use more sophisticated NLP methods

        # Normalize text
        text_lower = text.lower()

        # Remove common stop words
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

        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-z]+\b', text_lower)

        # Filter out stop words and short words
        meaningful_words = [
            word for word in words
            if word not in stop_words and len(word) > 3
        ]

        # Count frequencies
        word_freq = Counter(meaningful_words)

        # Get most common words as tags
        tags = []
        for word, _ in word_freq.most_common(max_tags * 2):
            # Basic singularization (very simple)
            if word.endswith('ies') and len(word) > 3:
                tag = word[:-3] + 'y'
            elif word.endswith('es') and len(word) > 2:
                tag = word[:-2]
            elif word.endswith('s') and len(word) > 4:  # Avoid short words like 'as', 'is'
                tag = word[:-1]
            else:
                tag = word
            if tag not in tags and len(tag) > 3:  # Ensure tag itself is meaningful
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
