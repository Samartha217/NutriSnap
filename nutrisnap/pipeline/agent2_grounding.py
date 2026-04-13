"""
Agent 2 — Ingredient Grounding

No LLM calls. Pure data lookups in this order:
  1. OFF additives.json (local, O(1)) — for E-numbers
  2. USDA FoodData Central API — for non-additive ingredients
  3. LLM fallback — if both return nothing, pass through for Agent 3 to handle

Runs all lookups concurrently (asyncio.gather) with a semaphore
to avoid hammering USDA.
"""
import asyncio
import time
from typing import Optional

from pipeline.state import PipelineState, IngredientInfo
from utils.off_client import lookup_additive, extract_enumber_from_name
from utils.usda_client import lookup_ingredient_nutrition
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)

_USDA_SEMAPHORE = asyncio.Semaphore(3)  # Max 3 concurrent USDA requests


async def _enrich_one(raw: dict, trace_id: str) -> IngredientInfo:
    ingredient = IngredientInfo(
        name=raw.get("name", "Unknown"),
        quantity=raw.get("quantity"),
        e_number=raw.get("e_number"),
    )

    # Step 1: Try OFF taxonomy for E-numbers (local, always fast)
    e_num = ingredient.e_number or extract_enumber_from_name(ingredient.name)
    if e_num:
        off_data = lookup_additive(e_num)
        if off_data:
            ingredient.e_number = e_num
            ingredient.additive_class = off_data.get("additive_class")
            ingredient.efsa_risk = off_data.get("efsa_risk")
            ingredient.efsa_adi = off_data.get("efsa_adi")
            ingredient.vegan = off_data.get("vegan")
            ingredient.source = "off_taxonomy"
            metrics.increment("agent2_off_hits")
            return ingredient

    # Step 2: Try USDA for nutritional facts (non-additive ingredients)
    async with _USDA_SEMAPHORE:
        usda_data = await lookup_ingredient_nutrition(ingredient.name)

    if usda_data:
        ingredient.usda_nutrients = usda_data
        ingredient.source = "usda_api"
        metrics.increment("agent2_usda_hits")
        return ingredient

    # Step 3: Fallback — Agent 3 will use its own LLM knowledge
    ingredient.source = "llm_fallback"
    metrics.increment("agent2_llm_fallbacks")
    return ingredient


async def run_agent2(state: PipelineState) -> PipelineState:
    if state.get("pipeline_failed"):
        return state

    start = time.monotonic()
    trace_id = state.get("trace_id", "")
    raw_list = state.get("raw_ingredients_json") or []

    tasks = [_enrich_one(raw, trace_id) for raw in raw_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    enriched: list[IngredientInfo] = []
    errors: list[str] = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            name = raw_list[i].get("name", "unknown") if i < len(raw_list) else "unknown"
            errors.append(f"{name}: {str(result)}")
            # Still add a basic entry so Agent 3 can work with LLM knowledge
            enriched.append(IngredientInfo(
                name=name,
                source="llm_fallback",
            ))
        else:
            enriched.append(result)

    state["enriched_ingredients"] = enriched
    state["grounding_errors"] = errors

    off_hits = sum(1 for e in enriched if e.source == "off_taxonomy")
    usda_hits = sum(1 for e in enriched if e.source == "usda_api")
    fallbacks = sum(1 for e in enriched if e.source == "llm_fallback")
    duration_ms = (time.monotonic() - start) * 1000

    logger.info("agent2_completed", extra={
        "trace_id": trace_id,
        "total": len(enriched),
        "off_hits": off_hits,
        "usda_hits": usda_hits,
        "llm_fallbacks": fallbacks,
        "errors": len(errors),
        "duration_ms": round(duration_ms),
    })

    return state
