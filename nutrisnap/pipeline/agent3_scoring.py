"""
Agent 3 — Scoring + Formatting

Takes enriched ingredient data from Agent 2 and produces a full
health analysis using Gemini text. Reasons from facts (EFSA data,
USDA nutritional values) — not from guesses.
"""
import json
import time
from dataclasses import asdict

from pipeline.state import PipelineState
from utils.gemini_client import get_client
from config import GROQ_TEXT_MODEL
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)

SCORING_PROMPT = """
You are a food safety and nutrition expert. You have been given a list of ingredients
from a packaged food product, enriched with factual data from Open Food Facts
(EFSA evaluations) and USDA FoodData Central where available.

Ingredient data (JSON):
{ingredient_data}

Respond ONLY with a valid JSON object. No markdown, no explanation, no backticks.

{{
  "health_score": <integer 1-10, where 10 = very healthy, 1 = very unhealthy>,
  "score_reasoning": "<one sentence explaining the score>",
  "red_flags": [
    {{"ingredient": "<name>", "reason": "<why it is concerning, plain language>"}}
  ],
  "good_ingredients": ["<ingredient name>", ...],
  "alternatives": ["<practical healthier alternative>", ...],
  "ingredient_explanations": [
    {{"ingredient": "<name>", "explanation": "<what it is, 1 plain-English sentence>"}}
  ]
}}

Scoring rules:
- If efsa_risk is "high" for an ingredient → strong negative weight on health_score
- If usda_nutrients shows saturated_fat_g > 20 per 100g → flag it
- If usda_nutrients shows sodium_mg > 400 per 100g → flag it
- If usda_nutrients shows sugar_g > 20 per 100g → flag it
- source="llm_fallback" means no external data found — use your own knowledge carefully
- good_ingredients: naturally occurring, minimally processed items only
- alternatives: specific products or ingredient swaps (e.g. "cold-pressed coconut oil" not "healthier oil")
- All text in plain English, readable by a non-scientist
- If EFSA data is present, cite it: e.g. "EFSA flagged high overexposure risk"
- Maximum 5 red_flags, 5 good_ingredients, 3 alternatives
"""


def _serialize_enriched(enriched_ingredients) -> list[dict]:
    result = []
    for ing in enriched_ingredients:
        entry = {
            "name": ing.name,
            "quantity": ing.quantity,
            "e_number": ing.e_number,
            "source": ing.source,
            "additive_class": ing.additive_class,
            "efsa_risk": ing.efsa_risk,
            "efsa_adi": ing.efsa_adi,
            "vegan": ing.vegan,
        }
        if ing.usda_nutrients:
            entry["usda_nutrients"] = ing.usda_nutrients
        result.append(entry)
    return result


async def run_agent3(state: PipelineState) -> PipelineState:
    if state.get("pipeline_failed"):
        return state

    start = time.monotonic()
    trace_id = state.get("trace_id", "")
    enriched = state.get("enriched_ingredients") or []

    ingredient_data = _serialize_enriched(enriched)
    prompt = SCORING_PROMPT.format(ingredient_data=json.dumps(ingredient_data, indent=2))

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw_text = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw_text.startswith("```"):
            parts = raw_text.split("```")
            raw_text = parts[1] if len(parts) > 1 else raw_text
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        result = json.loads(raw_text)

        state["health_score"] = result.get("health_score")
        state["red_flags"] = result.get("red_flags", [])
        state["good_ingredients"] = result.get("good_ingredients", [])
        state["alternatives"] = result.get("alternatives", [])
        state["agent3_result"] = result

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("agent3_completed", extra={
            "trace_id": trace_id,
            "health_score": state["health_score"],
            "red_flags_count": len(state["red_flags"]),
            "duration_ms": round(duration_ms),
        })

    except json.JSONDecodeError as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Could not generate analysis. Please try again."
        metrics.increment("agent3_errors")
        logger.error("agent3_json_parse_error", extra={"trace_id": trace_id, "error": str(e)})

    except Exception as e:
        state["pipeline_failed"] = True
        state["failure_reason"] = "Could not generate analysis. Please try again."
        metrics.increment("agent3_errors")
        logger.error("agent3_unexpected_error", extra={"trace_id": trace_id, "error": str(e)})

    return state
