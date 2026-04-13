"""
Integration tests for Agent 3 — makes real Gemini text API calls.
Run with: pytest tests/integration/ --run-integration
"""
import pytest
import os

pytestmark = pytest.mark.integration


def _make_state(enriched_ingredients) -> dict:
    return {
        "trace_id": "test-agent3",
        "chat_id": 0,
        "barcode_detected": False,
        "ingredients_text_from_barcode": None,
        "off_product_data": None,
        "image_bytes": None,
        "raw_ingredients_json": None,
        "extraction_error": None,
        "enriched_ingredients": enriched_ingredients,
        "grounding_errors": [],
        "health_score": None,
        "red_flags": None,
        "good_ingredients": None,
        "alternatives": None,
        "agent3_result": None,
        "pipeline_failed": False,
        "failure_reason": None,
    }


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_agent3_scores_high_risk_ingredients():
    """Ingredients with known EFSA high risk should get a low health score."""
    from pipeline.agent3_scoring import run_agent3
    from pipeline.state import IngredientInfo

    enriched = [
        IngredientInfo(
            name="Tartrazine",
            e_number="E102",
            additive_class="colour",
            efsa_risk="high",
            efsa_adi="7.5 mg/kg bw/day",
            source="off_taxonomy",
        ),
        IngredientInfo(
            name="Palm oil",
            usda_nutrients={"fat_g": 100.0, "saturated_fat_g": 49.3, "energy_kcal": 884.0},
            source="usda_api",
        ),
        IngredientInfo(name="Wheat flour", source="llm_fallback"),
    ]

    state = _make_state(enriched)
    result = await run_agent3(state)

    assert not result["pipeline_failed"], f"Pipeline failed: {result.get('failure_reason')}"
    assert result["health_score"] is not None
    assert 1 <= result["health_score"] <= 10
    assert result["red_flags"] is not None
    assert result["agent3_result"] is not None

    # High risk E102 should be flagged
    red_flag_names = [rf["ingredient"].lower() for rf in result["red_flags"]]
    assert any("tartrazine" in n or "e102" in n for n in red_flag_names)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_agent3_skips_on_failed_pipeline():
    """Agent 3 should be a no-op if pipeline already failed."""
    from pipeline.agent3_scoring import run_agent3

    state = _make_state([])
    state["pipeline_failed"] = True
    state["failure_reason"] = "test_failure"

    result = await run_agent3(state)

    assert result["pipeline_failed"] is True
    assert result["health_score"] is None  # Should not have run
