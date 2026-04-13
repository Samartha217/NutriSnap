# LabelLens — Implementation Plan
> Telegram bot that analyses food ingredient labels using a multi-agent AI pipeline.
> This document is the complete handoff to Claude Code. Follow every section in order.

---

## 1. Project Overview

**Name:** LabelLens  
**What it does:** User sends a photo of the ingredients section of a packaged food item to a Telegram bot. The system analyses it and replies with a plain-language health breakdown — ingredient-by-ingredient explanation, health score, red flags, and healthier alternatives.  
**Stack:** Python, FastAPI, python-telegram-bot v21, LangGraph, Gemini 2.5 Flash-Lite (free tier), Open Food Facts API (free, no auth), ChromaDB (optional, for caching).  
**Cost target:** $0/month — all free tiers only.  
**Deployment target:** Render free tier (or Railway free tier).

---

## 2. Architecture Summary

Three-agent pipeline orchestrated by LangGraph StateGraph:

```
User (Telegram)
    │  photo
    ▼
Telegram layer (python-telegram-bot v21 + FastAPI webhook)
    │  image bytes + chat_id
    ▼
LangGraph Orchestrator (StateGraph, manages pipeline state)
    │
    ├─► Agent 1 — Extraction
    │       Gemini 2.5 Flash-Lite (vision)
    │       Input: image bytes
    │       Output: structured JSON list of ingredients
    │
    ├─► Agent 2 — Grounding
    │       Lookup loop (async, per ingredient)
    │       Open Food Facts API v2
    │       Output: enriched ingredient data (NOVA group, additives, allergens, Nutri-Score)
    │       Fallback: if ingredient not found → LLM knowledge from Agent 3
    │
    └─► Agent 3 — Scoring + Formatting
            Gemini 2.5 Flash-Lite (text)
            Input: enriched ingredient data from Agent 2
            Output: formatted Telegram message (health score, red flags, alternatives)
    │
    ▼
Error handler + input validator
    │
    ▼
User (Telegram) ← formatted reply
```

---

## 3. Key Technical Decisions

### 3.1 Why Gemini 2.5 Flash-Lite (not Flash)
After December 2025 Google quota cuts, Gemini 2.5 Flash is capped at ~20 requests/day on the free tier. Flash-Lite gives 1,000 requests/day. For a portfolio bot, Flash-Lite is the only viable zero-cost option. Model string to use: `gemini-2.5-flash-lite` via the Google AI Python SDK (google-generativeai).

### 3.2 Why Open Food Facts API (not just LLM knowledge)
LLMs hallucinate safety ratings for obscure additives (E471, E551, etc.). Open Food Facts has 4M+ products, returns NOVA group, additives list, allergens, and Nutri-Score, is completely free with no API key, and has no hard rate limits for reasonable use. This gives cited, verifiable data instead of model guesses.

### 3.3 Why LangGraph (not plain Python functions)
LangGraph StateGraph gives us: clean state management between agents, built-in retry logic, easy extensibility (add a 4th agent later without rewriting), and a portfolio-worthy architecture story. It is not overkill here — it is the right tool because the pipeline has conditional branching (cache hit vs miss in Agent 2).

### 3.4 Why FastAPI + webhook (not polling)
Polling works fine locally. For deployment on Render/Railway, webhook is cleaner — one persistent ASGI server, no long-polling threads. python-telegram-bot v21 is fully async and pairs natively with FastAPI via uvicorn.

### 3.5 Open Food Facts API — important limitations
- The API is barcode-based (`/api/v2/product/{barcode}`). We are NOT using barcodes — we are doing ingredient name lookups.
- The correct endpoint for our use case is the **ingredients analysis endpoint**: `POST https://world.openfoodfacts.org/api/v2/product` with ingredient text, OR the **search endpoint** `GET /api/v2/search?ingredients_text={name}`.
- A simpler and more reliable approach: use the `off` Python package (`pip install openfoodfacts`) which wraps the API cleanly.
- For each extracted ingredient name, search Open Food Facts for additive/safety data. If not found (common for generic ingredients like "salt", "water"), fall through to Agent 3's LLM knowledge.

---

## 4. Folder Structure

```
labellens/
├── main.py                  # FastAPI app + Telegram webhook setup entry point
├── bot/
│   ├── __init__.py
│   ├── handlers.py          # Telegram message handlers (photo, /start, /help, errors)
│   └── formatter.py         # Format final agent output into Telegram-friendly message
├── pipeline/
│   ├── __init__.py
│   ├── state.py             # LangGraph PipelineState TypedDict definition
│   ├── orchestrator.py      # LangGraph StateGraph definition + compilation
│   ├── agent1_extraction.py # Agent 1: Gemini vision → ingredient JSON
│   ├── agent2_grounding.py  # Agent 2: Open Food Facts lookup loop
│   └── agent3_scoring.py    # Agent 3: Gemini text → health score + format
├── utils/
│   ├── __init__.py
│   ├── gemini_client.py     # Singleton Gemini client initialisation
│   ├── off_client.py        # Open Food Facts API wrapper
│   ├── validators.py        # Image validation (size, format, is it a food label?)
│   └── rate_limiter.py      # Simple in-memory rate limiter per user
├── config.py                # All config loaded from environment variables
├── .env.example             # Template for required env vars
├── requirements.txt
├── Dockerfile               # For Render deployment
├── render.yaml              # Render service config
└── README.md
```

---

## 5. Environment Variables

Create `.env` (never commit this) and `.env.example` (commit this):

```env
# .env.example
TELEGRAM_BOT_TOKEN=your_bot_token_here        # from @BotFather
GEMINI_API_KEY=your_gemini_api_key_here       # from Google AI Studio (free)
WEBHOOK_URL=https://your-app.onrender.com     # your deployed URL
PORT=8000
LOG_LEVEL=INFO

# Optional
MAX_IMAGE_SIZE_MB=5
RATE_LIMIT_REQUESTS_PER_MINUTE=3             # per user, to protect free quota
```

---

## 6. Dependencies

**requirements.txt:**
```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-telegram-bot==21.9
google-generativeai>=0.8.0
langgraph>=0.2.0
langchain-core>=0.3.0
openfoodfacts>=1.0.0
httpx==0.27.0
pydantic>=2.0.0
python-dotenv==1.0.0
Pillow>=10.0.0
```

Install with: `pip install -r requirements.txt`

---

## 7. Implementation — File by File

### 7.1 `config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
WEBHOOK_URL: str = os.environ.get("WEBHOOK_URL", "")
PORT: int = int(os.environ.get("PORT", 8000))
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
MAX_IMAGE_SIZE_MB: int = int(os.environ.get("MAX_IMAGE_SIZE_MB", 5))
RATE_LIMIT_RPM: int = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", 3))

GEMINI_MODEL = "gemini-2.5-flash-lite"
OFF_API_BASE = "https://world.openfoodfacts.org"
```

---

### 7.2 `pipeline/state.py`

Define the shared state that flows through all three agents:

```python
from typing import TypedDict, Optional
from dataclasses import dataclass, field

@dataclass
class IngredientInfo:
    name: str
    quantity: Optional[str] = None           # e.g. "2g" if label shows it
    nova_group: Optional[int] = None         # 1-4, from OFF API
    additives: list[str] = field(default_factory=list)
    allergens: list[str] = field(default_factory=list)
    nutriscore: Optional[str] = None         # A-E
    is_red_flag: bool = False
    explanation: Optional[str] = None        # Agent 3 fills this
    source: str = "llm"                      # "off_api" or "llm"

class PipelineState(TypedDict):
    # Inputs
    image_bytes: bytes
    chat_id: int

    # Agent 1 output
    raw_ingredients_json: Optional[list[dict]]
    extraction_error: Optional[str]

    # Agent 2 output
    enriched_ingredients: Optional[list[IngredientInfo]]
    grounding_errors: list[str]

    # Agent 3 output
    health_score: Optional[int]              # 1-10
    red_flags: Optional[list[str]]
    good_ingredients: Optional[list[str]]
    alternatives: Optional[list[str]]
    formatted_message: Optional[str]

    # Meta
    pipeline_failed: bool
    failure_reason: Optional[str]
```

---

### 7.3 `utils/gemini_client.py`

```python
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)

# Singleton model instances
_vision_model = None
_text_model = None

def get_vision_model():
    global _vision_model
    if _vision_model is None:
        _vision_model = genai.GenerativeModel(GEMINI_MODEL)
    return _vision_model

def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = genai.GenerativeModel(GEMINI_MODEL)
    return _text_model
```

---

### 7.4 `pipeline/agent1_extraction.py`

Agent 1 takes the image bytes, sends it to Gemini vision, and returns a structured list of ingredients as JSON.

**Prompt design (critical — tune this carefully):**

```python
EXTRACTION_PROMPT = """
You are an ingredient extraction specialist. The user has sent a photo of the ingredients section 
from a packaged food product.

Your task:
1. Read the ingredients list carefully from the image
2. Return ONLY a JSON array. No explanation, no markdown, no backticks.
3. Each element in the array must be an object with:
   - "name": the ingredient name (string, cleaned and normalised, e.g. "Monosodium glutamate" not "MSG")
   - "quantity": the quantity if visible on the label (string or null)
   - "e_number": the E-number if visible, e.g. "E621" (string or null)
4. List ingredients in the order they appear on the label
5. If the image does not show an ingredients section, return: {"error": "no_ingredients_visible"}
6. If the image is too blurry to read, return: {"error": "image_too_blurry"}

Example output:
[
  {"name": "Wheat flour", "quantity": null, "e_number": null},
  {"name": "Sugar", "quantity": "15g", "e_number": null},
  {"name": "Monosodium glutamate", "quantity": null, "e_number": "E621"}
]
"""

import json
import google.generativeai as genai
from PIL import Image
import io
from utils.gemini_client import get_vision_model
from pipeline.state import PipelineState

async def run_agent1(state: PipelineState) -> PipelineState:
    """Extract ingredients from image using Gemini vision."""
    try:
        image = Image.open(io.BytesIO(state["image_bytes"]))
        model = get_vision_model()
        
        response = model.generate_content([EXTRACTION_PROMPT, image])
        raw_text = response.text.strip()
        
        # Strip markdown fences if model returns them despite prompt
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        
        parsed = json.loads(raw_text)
        
        # Check for error object
        if isinstance(parsed, dict) and "error" in parsed:
            state["pipeline_failed"] = True
            state["failure_reason"] = parsed["error"]
            return state
        
        state["raw_ingredients_json"] = parsed
        state["extraction_error"] = None
        
    except json.JSONDecodeError as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = f"Could not parse ingredients from image. Please try with a clearer photo."
        state["extraction_error"] = str(e)
    except Exception as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Extraction failed. Please try again."
        state["extraction_error"] = str(e)
    
    return state
```

---

### 7.5 `utils/off_client.py`

Wrapper around the Open Food Facts API for ingredient lookups:

```python
import httpx
import asyncio
from typing import Optional

OFF_SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
OFF_ADDITIVE_URL = "https://world.openfoodfacts.org/additive/{e_number}.json"

async def lookup_ingredient_by_name(name: str) -> Optional[dict]:
    """
    Search Open Food Facts for additive/safety data by ingredient name.
    Returns first relevant result or None.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OFF_SEARCH_URL, params={
                "ingredients_text": name,
                "fields": "product_name,nova_group,nutriscore_grade,additives_tags,allergens_tags",
                "page_size": 1,
            })
            resp.raise_for_status()
            data = resp.json()
            products = data.get("products", [])
            if products:
                return products[0]
    except Exception:
        pass
    return None

async def lookup_additive_by_enumber(e_number: str) -> Optional[dict]:
    """
    Lookup a specific E-number additive directly.
    More reliable than name search for additives.
    """
    try:
        clean = e_number.lower().replace(" ", "")
        url = f"https://world.openfoodfacts.org/additive/{clean}.json"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        pass
    return None
```

---

### 7.6 `pipeline/agent2_grounding.py`

Agent 2 loops through extracted ingredients and enriches each with Open Food Facts data. Run lookups concurrently with `asyncio.gather` but respect a small semaphore to avoid hammering the API.

```python
import asyncio
from pipeline.state import PipelineState, IngredientInfo
from utils.off_client import lookup_ingredient_by_name, lookup_additive_by_enumber

# Max concurrent OFF API requests
SEMAPHORE = asyncio.Semaphore(3)

async def _enrich_one(raw: dict) -> IngredientInfo:
    """Enrich a single ingredient with OFF API data."""
    ingredient = IngredientInfo(
        name=raw.get("name", "Unknown"),
        quantity=raw.get("quantity"),
    )
    
    async with SEMAPHORE:
        # Try E-number lookup first (most precise for additives)
        if raw.get("e_number"):
            off_data = await lookup_additive_by_enumber(raw["e_number"])
            if off_data:
                ingredient.source = "off_api"
                # Parse additive safety info from off_data here
                # off_data structure varies — extract what's available
                tags = off_data.get("tag", {})
                ingredient.additives = [raw["e_number"]]
                return ingredient
        
        # Fall back to name search
        off_data = await lookup_ingredient_by_name(ingredient.name)
        if off_data:
            ingredient.source = "off_api"
            ingredient.nova_group = off_data.get("nova_group")
            ingredient.nutriscore = off_data.get("nutriscore_grade")
            ingredient.additives = off_data.get("additives_tags", [])
            ingredient.allergens = off_data.get("allergens_tags", [])
    
    return ingredient

async def run_agent2(state: PipelineState) -> PipelineState:
    """Enrich all ingredients with Open Food Facts data concurrently."""
    if state.get("pipeline_failed"):
        return state
    
    raw_list = state.get("raw_ingredients_json", [])
    
    tasks = [_enrich_one(raw) for raw in raw_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    enriched = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"Could not enrich {raw_list[i].get('name', 'unknown')}: {str(result)}")
            # Still add a basic IngredientInfo so Agent 3 can work with LLM knowledge
            enriched.append(IngredientInfo(name=raw_list[i].get("name", "Unknown")))
        else:
            enriched.append(result)
    
    state["enriched_ingredients"] = enriched
    state["grounding_errors"] = errors
    return state
```

---

### 7.7 `pipeline/agent3_scoring.py`

Agent 3 takes the enriched ingredient list and produces the final health analysis using Gemini text:

```python
import json
from utils.gemini_client import get_text_model
from pipeline.state import PipelineState

SCORING_PROMPT_TEMPLATE = """
You are a food safety and nutrition expert. You have been given a list of ingredients from a packaged food product, 
enriched with data from Open Food Facts where available.

Ingredient data:
{ingredient_data}

Your task — respond ONLY with a valid JSON object, no markdown, no explanation:
{{
  "health_score": <integer 1-10, where 10 is very healthy and 1 is very unhealthy>,
  "score_reasoning": "<one sentence explaining the score>",
  "red_flags": [
    {{"ingredient": "<name>", "reason": "<why it is concerning, plain language>"}}
  ],
  "good_ingredients": ["<ingredient name>", ...],
  "alternatives": ["<healthier alternative product or ingredient suggestion>", ...],
  "ingredient_explanations": [
    {{"ingredient": "<name>", "explanation": "<what it is in plain language, 1 sentence>"}}
  ]
}}

Rules:
- health_score must account for NOVA group if available (NOVA 4 = ultra-processed = lower score)
- red_flags should cover: E-numbers with known concerns, high sugar/sodium, allergens, artificial additives
- good_ingredients are naturally occurring, minimally processed items
- alternatives should be practical and specific (e.g. "Britannia NutriChoice" not just "healthier biscuits")
- All explanations in plain English, readable by a non-scientist
- If enriched OFF data is present, cite it in explanations (e.g. "classified as NOVA group 4")
"""

async def run_agent3(state: PipelineState) -> PipelineState:
    """Score, explain, and format the final health analysis."""
    if state.get("pipeline_failed"):
        return state
    
    enriched = state.get("enriched_ingredients", [])
    
    # Build ingredient data summary for prompt
    ingredient_data = []
    for ing in enriched:
        entry = {
            "name": ing.name,
            "quantity": ing.quantity,
            "source": ing.source,
            "nova_group": ing.nova_group,
            "nutriscore": ing.nutriscore,
            "additives": ing.additives,
            "allergens": ing.allergens,
        }
        ingredient_data.append(entry)
    
    prompt = SCORING_PROMPT_TEMPLATE.format(
        ingredient_data=json.dumps(ingredient_data, indent=2)
    )
    
    try:
        model = get_text_model()
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Strip markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        
        result = json.loads(raw_text)
        
        state["health_score"] = result.get("health_score")
        state["red_flags"] = result.get("red_flags", [])
        state["good_ingredients"] = result.get("good_ingredients", [])
        state["alternatives"] = result.get("alternatives", [])
        
        # Store full result for formatter
        state["agent3_result"] = result
        
    except Exception as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Could not generate analysis. Please try again."
    
    return state
```

---

### 7.8 `pipeline/orchestrator.py`

Assemble the LangGraph StateGraph:

```python
from langgraph.graph import StateGraph, END
from pipeline.state import PipelineState
from pipeline.agent1_extraction import run_agent1
from pipeline.agent2_grounding import run_agent2
from pipeline.agent3_scoring import run_agent3

def should_continue(state: PipelineState) -> str:
    """Conditional edge — stop early if pipeline failed."""
    if state.get("pipeline_failed"):
        return "end"
    return "continue"

def build_pipeline():
    graph = StateGraph(PipelineState)
    
    graph.add_node("agent1", run_agent1)
    graph.add_node("agent2", run_agent2)
    graph.add_node("agent3", run_agent3)
    
    graph.set_entry_point("agent1")
    
    graph.add_conditional_edges(
        "agent1",
        should_continue,
        {"continue": "agent2", "end": END}
    )
    graph.add_conditional_edges(
        "agent2",
        should_continue,
        {"continue": "agent3", "end": END}
    )
    graph.add_edge("agent3", END)
    
    return graph.compile()

# Singleton — compile once at startup
PIPELINE = build_pipeline()
```

---

### 7.9 `bot/formatter.py`

Convert the pipeline state into a clean Telegram message:

```python
from pipeline.state import PipelineState

SCORE_EMOJI = {
    (1, 3): "🔴",
    (4, 5): "🟠",
    (6, 7): "🟡",
    (8, 10): "🟢",
}

def _score_emoji(score: int) -> str:
    for (low, high), emoji in SCORE_EMOJI.items():
        if low <= score <= high:
            return emoji
    return "⚪"

def format_response(state: PipelineState) -> str:
    """Format pipeline output as a Telegram-ready message."""
    
    if state.get("pipeline_failed"):
        reason = state.get("failure_reason", "Unknown error")
        error_messages = {
            "no_ingredients_visible": "❌ Couldn't find an ingredients section in your photo. Try cropping to just the ingredients list.",
            "image_too_blurry": "❌ Image is too blurry to read. Please try again with better lighting.",
        }
        return error_messages.get(reason, f"❌ Something went wrong: {reason}\n\nPlease try again.")
    
    result = state.get("agent3_result", {})
    score = state.get("health_score", 0)
    emoji = _score_emoji(score)
    
    lines = []
    lines.append(f"🧪 *Ingredient Analysis*")
    lines.append("")
    lines.append(f"*Health Score:* {emoji} {score}/10")
    lines.append(f"_{result.get('score_reasoning', '')}_")
    lines.append("")
    
    red_flags = state.get("red_flags", [])
    if red_flags:
        lines.append("🚨 *Red Flags:*")
        for flag in red_flags:
            lines.append(f"• *{flag['ingredient']}* — {flag['reason']}")
        lines.append("")
    
    good = state.get("good_ingredients", [])
    if good:
        lines.append("✅ *Okay Ingredients:*")
        for g in good[:5]:  # cap at 5
            lines.append(f"• {g}")
        lines.append("")
    
    alternatives = state.get("alternatives", [])
    if alternatives:
        lines.append("💡 *Healthier Alternatives:*")
        for alt in alternatives[:3]:
            lines.append(f"• {alt}")
        lines.append("")
    
    explanations = result.get("ingredient_explanations", [])
    if explanations:
        lines.append("📖 *Full Breakdown:*")
        for exp in explanations:
            lines.append(f"• *{exp['ingredient']}:* {exp['explanation']}")
    
    lines.append("")
    lines.append("_Data sourced from Open Food Facts + Gemini AI_")
    
    return "\n".join(lines)
```

---

### 7.10 `bot/handlers.py`

Telegram message handlers:

```python
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
import io

from pipeline.orchestrator import PIPELINE
from pipeline.state import PipelineState
from bot.formatter import format_response
from utils.validators import validate_image
from utils.rate_limiter import check_rate_limit

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *LabelLens*!\n\n"
        "📸 Send me a photo of the *ingredients section* on any packaged food item "
        "and I'll tell you exactly what's in it — what's healthy, what's not, and what to watch out for.\n\n"
        "Just send the photo directly in this chat.",
        parse_mode=ParseMode.MARKDOWN
    )

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 *How to use LabelLens:*\n\n"
        "1. Find the ingredients list on the back of any packaged food\n"
        "2. Take a clear, well-lit photo of just that section\n"
        "3. Send it here\n\n"
        "💡 *Tips for best results:*\n"
        "• Make sure the text is in focus\n"
        "• Avoid shadows and glare\n"
        "• Crop to just the ingredients section if possible",
        parse_mode=ParseMode.MARKDOWN
    )

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Rate limiting
    if not check_rate_limit(chat_id):
        await update.message.reply_text(
            "⏳ You're sending too many requests. Please wait a minute and try again."
        )
        return
    
    # Acknowledge receipt immediately
    processing_msg = await update.message.reply_text(
        "🔍 Analysing your label... this takes about 10-15 seconds."
    )
    
    try:
        # Download image
        photo = update.message.photo[-1]  # highest resolution
        file = await context.bot.get_file(photo.file_id)
        
        image_bytes_io = io.BytesIO()
        await file.download_to_memory(image_bytes_io)
        image_bytes = image_bytes_io.getvalue()
        
        # Validate
        validation_error = validate_image(image_bytes)
        if validation_error:
            await processing_msg.edit_text(validation_error)
            return
        
        # Run pipeline
        initial_state: PipelineState = {
            "image_bytes": image_bytes,
            "chat_id": chat_id,
            "raw_ingredients_json": None,
            "extraction_error": None,
            "enriched_ingredients": None,
            "grounding_errors": [],
            "health_score": None,
            "red_flags": None,
            "good_ingredients": None,
            "alternatives": None,
            "formatted_message": None,
            "pipeline_failed": False,
            "failure_reason": None,
        }
        
        final_state = await PIPELINE.ainvoke(initial_state)
        
        response_text = format_response(final_state)
        await processing_msg.edit_text(response_text, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        await processing_msg.edit_text(
            "❌ Something went wrong on our end. Please try again in a moment."
        )
        raise  # Re-raise for logging

async def unknown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Please send me a photo of a food label's ingredients section.\n"
        "Type /help for instructions."
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors and notify user."""
    import logging
    logger = logging.getLogger(__name__)
    logger.error("Exception while handling update:", exc_info=context.error)
```

---

### 7.11 `utils/validators.py`

```python
from config import MAX_IMAGE_SIZE_MB
from PIL import Image
import io

def validate_image(image_bytes: bytes) -> str | None:
    """
    Validate image before sending to pipeline.
    Returns error message string if invalid, None if valid.
    """
    # Size check
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return f"❌ Image too large ({size_mb:.1f}MB). Please send an image under {MAX_IMAGE_SIZE_MB}MB."
    
    # Format check
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        return "❌ Could not read image. Please try a different photo."
    
    return None
```

---

### 7.12 `utils/rate_limiter.py`

Simple in-memory rate limiter (no Redis needed for portfolio scale):

```python
from collections import defaultdict
from time import time
from config import RATE_LIMIT_RPM

# {chat_id: [timestamp, timestamp, ...]}
_request_log: dict[int, list[float]] = defaultdict(list)

def check_rate_limit(chat_id: int) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time()
    window = 60.0  # 1 minute window
    
    # Clean old entries
    _request_log[chat_id] = [t for t in _request_log[chat_id] if now - t < window]
    
    if len(_request_log[chat_id]) >= RATE_LIMIT_RPM:
        return False
    
    _request_log[chat_id].append(now)
    return True
```

---

### 7.13 `main.py`

FastAPI app with Telegram webhook:

```python
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN, WEBHOOK_URL, PORT, LOG_LEVEL
from bot.handlers import (
    start_handler, help_handler, photo_handler,
    unknown_handler, error_handler
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Build PTB application
ptb_app = (
    Application.builder()
    .token(TELEGRAM_BOT_TOKEN)
    .updater(None)  # Webhook mode — no updater needed
    .build()
)

# Register handlers
ptb_app.add_handler(CommandHandler("start", start_handler))
ptb_app.add_handler(CommandHandler("help", help_handler))
ptb_app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_handler))
ptb_app.add_error_handler(error_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up webhook on startup, clean up on shutdown."""
    await ptb_app.initialize()
    if WEBHOOK_URL:
        await ptb_app.bot.set_webhook(
            url=f"{WEBHOOK_URL}/webhook",
            allowed_updates=["message"]
        )
        logger.info(f"Webhook set to {WEBHOOK_URL}/webhook")
    else:
        logger.warning("WEBHOOK_URL not set — webhook not registered. Use polling for local dev.")
    await ptb_app.start()
    yield
    await ptb_app.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Receive Telegram updates."""
    data = await request.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return Response(status_code=200)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
```

---

### 7.14 Local development (polling mode)

For local testing without a webhook, create `run_local.py`:

```python
"""Use this for local development. Do NOT deploy this file."""
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from config import TELEGRAM_BOT_TOKEN
from bot.handlers import start_handler, help_handler, photo_handler, unknown_handler, error_handler

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_handler))
    app.add_error_handler(error_handler)
    
    print("Bot running in polling mode...")
    app.run_polling()

if __name__ == "__main__":
    main()
```

Run with: `python run_local.py`

---

## 8. Deployment

### 8.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 render.yaml

```yaml
services:
  - type: web
    name: labellens
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: TELEGRAM_BOT_TOKEN
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: WEBHOOK_URL
        sync: false
```

Set `WEBHOOK_URL` to your Render app URL after first deploy: `https://labellens.onrender.com`

---

## 9. Implementation Order (for Claude Code)

Build in this exact sequence to avoid dependency issues:

1. `config.py` + `.env`
2. `pipeline/state.py`
3. `utils/gemini_client.py`
4. `utils/off_client.py`
5. `pipeline/agent1_extraction.py` → test with a sample image immediately
6. `pipeline/agent2_grounding.py` → test with a known ingredient list
7. `pipeline/agent3_scoring.py` → test with enriched mock data
8. `pipeline/orchestrator.py` → wire agents together + test full pipeline
9. `bot/formatter.py`
10. `utils/validators.py` + `utils/rate_limiter.py`
11. `bot/handlers.py`
12. `main.py` + `run_local.py`
13. Test end-to-end locally with `run_local.py`
14. `Dockerfile` + `render.yaml` + deploy

---

## 10. Testing Checklist

Before deploying, manually test each scenario:

- [ ] Clear photo of a known product (e.g. Maggi noodles) → correct ingredient extraction
- [ ] Blurry photo → graceful error message
- [ ] Non-food photo → graceful error message  
- [ ] Text message (not a photo) → prompted to send a photo
- [ ] `/start` command → welcome message
- [ ] `/help` command → instructions
- [ ] Ingredient with a known E-number (e.g. E621 MSG) → flagged as red flag
- [ ] Sending 4+ photos quickly from same user → rate limit triggered
- [ ] Large image (>5MB) → size validation error

---

## 11. Known Limitations to Document in README

- Free tier Gemini 2.5 Flash-Lite: 1,000 requests/day per project. Sufficient for portfolio demo.
- Open Food Facts ingredient name search is imprecise. E-number lookups are more reliable.
- Currently stateless — no conversation history, no user preferences. Personalization is a v2 feature.
- English-language labels work best. Multi-language support is a future enhancement.
- Health score is AI-generated and should not be used as medical/nutritional advice.

---

## 12. Portfolio Talking Points

When explaining this project in interviews:

- **Multimodal AI:** Vision model (Gemini) reading real-world photos, not just text inputs.
- **Grounded AI:** Open Food Facts API prevents hallucination on additive safety — distinguishes this from a naive single-LLM approach.
- **Agent orchestration:** LangGraph StateGraph with conditional edges — not linear function chaining.
- **Production patterns:** Rate limiting, input validation, graceful error handling, webhook vs polling tradeoff.
- **Zero cost architecture:** Deliberate model selection (Flash-Lite over Flash) based on actual free tier research.
- **Separation of concerns:** Three distinct agents with clear responsibilities, independently testable.

---

*End of implementation plan. Hand this file to Claude Code and start with Step 1 of Section 9.*
