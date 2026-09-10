import logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from langchain.chat_models import init_chat_model

from app.config import MODELS

def get_llm(provider: str = "openai"):
    """Return a chat model for the given provider."""
    if provider not in MODELS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(MODELS)}")
    return init_chat_model(MODELS[provider])