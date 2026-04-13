"""
Agent 1 — Ingredient Extraction

Input:  image bytes (from photo handler) OR ingredients_text (from barcode path)
Output: raw_ingredients_json — a list of dicts, one per ingredient

If the barcode path already gave us ingredients_text, we skip the vision call
and parse the text directly via a lightweight Gemini text call (faster + cheaper).
"""
import json
import time
from PIL import Image
import io

from pipeline.state import PipelineState
from utils.gemini_client import get_vision_model, get_text_model
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)

VISION_PROMPT = """
You are an ingredient extraction specialist. The user has sent a photo of the
ingredients section from a packaged food product.

Your task:
1. Read the ingredients list carefully from the image.
2. Return ONLY a JSON array. No explanation, no markdown, no backticks.
3. Each element must be an object with:
   - "name": ingredient name, cleaned and normalised (e.g. "Monosodium glutamate" not "MSG")
   - "quantity": quantity if visible on label (string or null)
   - "e_number": E-number if visible, e.g. "E621" (string or null)
4. List ingredients in the order they appear on the label.
5. If no ingredients section is visible: return {"error": "no_ingredients_visible"}
6. If the image is too blurry to read: return {"error": "image_too_blurry"}

Example output:
[
  {"name": "Wheat flour", "quantity": "70%", "e_number": null},
  {"name": "Monosodium glutamate", "quantity": null, "e_number": "E621"},
  {"name": "Tartrazine", "quantity": null, "e_number": "E102"}
]
"""

TEXT_PARSE_PROMPT = """
You are an ingredient extraction specialist. You have been given raw ingredients text
from a food product (extracted from a barcode scan).

Parse it and return ONLY a JSON array. No explanation, no markdown, no backticks.
Each element must be an object with:
  - "name": ingredient name, cleaned and normalised
  - "quantity": quantity if present (string or null)
  - "e_number": E-number if present, e.g. "E621" (string or null)

Ingredients text:
{ingredients_text}
"""


def _strip_fences(text: str) -> str:
    """Strip markdown code fences if the model returns them despite the prompt."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


async def run_agent1(state: PipelineState) -> PipelineState:
    start = time.monotonic()

    # Barcode path — parse text instead of running vision
    if state.get("barcode_detected") and state.get("ingredients_text_from_barcode"):
        return await _parse_from_text(state, start)

    return await _extract_from_image(state, start)


async def _extract_from_image(state: PipelineState, start: float) -> PipelineState:
    trace_id = state.get("trace_id", "")
    try:
        image = Image.open(io.BytesIO(state["image_bytes"]))
        model = get_vision_model()
        response = model.generate_content([VISION_PROMPT, image])
        raw_text = _strip_fences(response.text)
        parsed = json.loads(raw_text)

        if isinstance(parsed, dict) and "error" in parsed:
            error_code = parsed["error"]
            state["pipeline_failed"] = True
            state["failure_reason"] = error_code
            logger.info("agent1_image_error", extra={
                "trace_id": trace_id, "error_code": error_code
            })
            return state

        state["raw_ingredients_json"] = parsed
        state["extraction_error"] = None

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("agent1_completed", extra={
            "trace_id": trace_id,
            "source": "vision",
            "ingredients_extracted": len(parsed),
            "duration_ms": round(duration_ms),
        })

    except json.JSONDecodeError as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Could not parse ingredients from image. Please try with a clearer photo."
        state["extraction_error"] = str(e)
        metrics.increment("agent1_errors")
        logger.error("agent1_json_parse_error", extra={"trace_id": trace_id, "error": str(e)})

    except Exception as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Extraction failed. Please try again."
        state["extraction_error"] = str(e)
        metrics.increment("agent1_errors")
        logger.error("agent1_unexpected_error", extra={"trace_id": trace_id, "error": str(e)})

    return state


async def _parse_from_text(state: PipelineState, start: float) -> PipelineState:
    trace_id = state.get("trace_id", "")
    try:
        prompt = TEXT_PARSE_PROMPT.format(
            ingredients_text=state["ingredients_text_from_barcode"]
        )
        model = get_text_model()
        response = model.generate_content(prompt)
        raw_text = _strip_fences(response.text)
        parsed = json.loads(raw_text)

        state["raw_ingredients_json"] = parsed
        state["extraction_error"] = None

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("agent1_completed", extra={
            "trace_id": trace_id,
            "source": "barcode_text",
            "ingredients_extracted": len(parsed),
            "duration_ms": round(duration_ms),
        })

    except Exception as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Could not parse ingredients. Please try again."
        state["extraction_error"] = str(e)
        metrics.increment("agent1_errors")
        logger.error("agent1_text_parse_error", extra={"trace_id": trace_id, "error": str(e)})

    return state
