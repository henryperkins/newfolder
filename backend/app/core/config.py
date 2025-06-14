from pydantic_settings import BaseSettings
from typing import Optional
import os
import logging


class Settings(BaseSettings):
    # App settings
    app_name: str = "AI Productivity App"
    debug: bool = False

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_hours: int = 24

    # Database
    database_url: str

    # Email settings
    smtp_host: str
    smtp_port: int = 587
    smtp_username: str
    smtp_password: str

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # RAG Configuration
    rag_enabled: bool = False
    openai_api_key: Optional[str] = None
    chroma_db_path: str = "./chroma_db"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    max_context_length: int = 3000
    
    # Frontend
    frontend_base_url: str = "http://localhost:3000"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_rag_settings()

    def _validate_rag_settings(self):
        """Validate RAG-related environment variables and configuration."""
        logger = logging.getLogger(__name__)
        
        # Get OpenAI API key from environment if not provided
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Validate RAG dependencies if RAG is enabled
        if self.rag_enabled:
            validation_errors = []
            
            # Check OpenAI API key
            if not self.openai_api_key:
                validation_errors.append("OPENAI_API_KEY is required when RAG is enabled")
            
            # Check ChromaDB path accessibility
            try:
                import chromadb
                os.makedirs(self.chroma_db_path, exist_ok=True)
                # Test ChromaDB initialization
                client = chromadb.PersistentClient(path=self.chroma_db_path)
                logger.info(f"ChromaDB accessible at {self.chroma_db_path}")
            except ImportError:
                validation_errors.append("chromadb package is required when RAG is enabled")
            except Exception as e:
                validation_errors.append(f"ChromaDB path {self.chroma_db_path} is not accessible: {e}")
            
            # Check required ML packages
            try:
                import sentence_transformers
            except ImportError:
                validation_errors.append("sentence-transformers package is required when RAG is enabled")
            
            try:
                import openai
            except ImportError:
                validation_errors.append("openai package is required when RAG is enabled")
            
            # Log validation results
            if validation_errors:
                logger.error("RAG validation failed:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                logger.warning("RAG functionality will be disabled due to validation errors")
                self.rag_enabled = False
            else:
                logger.info("RAG validation successful - RAG is enabled")
        else:
            logger.info("RAG is disabled in configuration")

    @property
    def is_rag_configured(self) -> bool:
        """Check if RAG is properly configured and available."""
        return (
            self.rag_enabled and
            bool(self.openai_api_key) and
            os.path.exists(self.chroma_db_path)
        )

    class Config:
        env_file = ".env"


settings = Settings()
