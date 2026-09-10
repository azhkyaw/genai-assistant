import logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODELS = {
    "openai": "openai:gpt-5.4-mini",
    "anthropic": "anthropic:claude-haiku-4-5",
    "gemini": "google_genai:gemini-3.5-flash-lite"
}

def get_llm(provider: str = "openai"):
    """Return a chat model for the given provider."""
    if provider not in MODELS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(MODELS)}")
    return init_chat_model(MODELS[provider])