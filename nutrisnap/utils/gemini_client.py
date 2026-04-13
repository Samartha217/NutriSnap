"""
Singleton Gemini client.

Initialised once at startup. Both vision (Agent 1) and text (Agent 3)
use the same model string — Gemini 2.5 Flash-Lite gives 1000 req/day free.
"""
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from observability.logger import get_logger

logger = get_logger(__name__)

_vision_model = None
_text_model = None


def _init():
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("gemini_client_initialized", extra={"model": GEMINI_MODEL})


def get_vision_model() -> genai.GenerativeModel:
    global _vision_model
    if _vision_model is None:
        _init()
        _vision_model = genai.GenerativeModel(GEMINI_MODEL)
    return _vision_model


def get_text_model() -> genai.GenerativeModel:
    global _text_model
    if _text_model is None:
        _init()
        _text_model = genai.GenerativeModel(GEMINI_MODEL)
    return _text_model
