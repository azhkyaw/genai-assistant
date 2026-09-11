from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"

MODELS = {
    "openai": "openai:gpt-5.4-mini",
    "anthropic": "anthropic:claude-haiku-4-5",
    "gemini": "google_genai:gemini-3.5-flash-lite"
}

MIN_RELEVANCE_SCORE = 0.0