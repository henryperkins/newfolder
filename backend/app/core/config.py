from pydantic_settings import BaseSettings
from typing import Optional


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

    # RAG
    rag_enabled: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
