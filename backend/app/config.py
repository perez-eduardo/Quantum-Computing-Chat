"""Configuration for Quantum Computing LLM API."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of backend/)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Required
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional (for custom model)
MODAL_URL = os.getenv("MODAL_URL", "")

# Groq settings
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 300


def validate_config():
    """Check required environment variables."""
    missing = []
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    
    return True
