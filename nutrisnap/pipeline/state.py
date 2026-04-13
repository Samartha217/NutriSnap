"""
Shared state that flows through all three agents in the LangGraph pipeline.

Every agent receives the full state and returns the full state
with its fields populated. Agents must not mutate fields they don't own.
"""
from typing import TypedDict, Optional
from dataclasses import dataclass, field


@dataclass
class IngredientInfo:
    name: str
    quantity: Optional[str] = None

    # E-number if present on label (e.g. "E621")
    e_number: Optional[str] = None

    # Populated by Agent 2 from OFF additives taxonomy (local)
    additive_class: Optional[str] = None          # e.g. "flavour enhancer"
    efsa_risk: Optional[str] = None               # "high" | "moderate" | "low" | None
    efsa_adi: Optional[str] = None                # e.g. "7.5 mg/kg bw/day"
    vegan: Optional[str] = None                   # "yes" | "no" | "maybe"

    # Populated by Agent 2 from USDA FoodData Central (API)
    usda_nutrients: Optional[dict] = None         # {"fat_g": 49.3, "saturated_fat_g": ...}

    # Source tracking — important for Agent 3 to know how reliable the data is
    source: str = "llm_fallback"                  # "off_taxonomy" | "usda_api" | "llm_fallback"

    # Populated by Agent 3
    explanation: Optional[str] = None
    is_red_flag: bool = False


class PipelineState(TypedDict):
    # Set at request entry point
    trace_id: str
    chat_id: int

    # Barcode path — set by router before Agent 1 if barcode detected
    barcode_detected: bool
    ingredients_text_from_barcode: Optional[str]   # raw text from OFF product API
    off_product_data: Optional[dict]               # full OFF product record

    # Image bytes — set by handler, used by Agent 1
    image_bytes: Optional[bytes]

    # Agent 1 outputs
    raw_ingredients_json: Optional[list[dict]]
    extraction_error: Optional[str]

    # Agent 2 outputs
    enriched_ingredients: Optional[list[IngredientInfo]]
    grounding_errors: list[str]

    # Agent 3 outputs
    health_score: Optional[int]                    # 1-10
    red_flags: Optional[list[dict]]               # [{"ingredient": ..., "reason": ...}]
    good_ingredients: Optional[list[str]]
    alternatives: Optional[list[str]]
    agent3_result: Optional[dict]                  # full raw JSON from Agent 3

    # Pipeline control
    pipeline_failed: bool
    failure_reason: Optional[str]
