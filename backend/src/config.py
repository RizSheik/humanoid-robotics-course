from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Database settings
    database_url: str = os.getenv("NEON_DB_URL", "postgresql+asyncpg://user:password@localhost/dbname")
    
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Security settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Application settings
    app_name: str = "Educational Book RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS settings
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # RAG settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    
    # Vector search settings
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # Content settings
    content_chunk_size: int = int(os.getenv("CONTENT_CHUNK_SIZE", "1000"))
    content_overlap: int = int(os.getenv("CONTENT_OVERLAP", "100"))

    class Config:
        env_file = ".env"


# Create a global settings instance
settings = Settings()