# Environment configuration for the RAG Chatbot
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# Vector Database Configuration (Qdrant)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_SSL = os.getenv("QDRANT_SSL", "false").lower() == "true"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "robotics_textbook")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

# Database Configuration (Neon)
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")

# Textbook Content Path
TEXTBOOK_CONTENT_PATH = PROJECT_ROOT / "docs"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# RAG Parameters
MAX_CONTEXTS = int(os.getenv("MAX_CONTEXTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))