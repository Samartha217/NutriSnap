"""
Groq client — replaces Gemini for vision and text inference.

Groq is free tier (1000 req/day), works in India, no billing required.
Uses OpenAI-compatible API via the official groq Python SDK.

Vision model : Llama 4 Scout (native multimodal)
Text model   : Llama 3.3 70B Versatile
"""
from groq import Groq
from config import GROQ_API_KEY
from observability.logger import get_logger

logger = get_logger(__name__)

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
        logger.info("groq_client_initialized")
    return _client
