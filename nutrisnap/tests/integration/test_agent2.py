"""
Integration tests for Agent 2 — tests real USDA API + mocked OFF taxonomy.
Run with: pytest tests/integration/ --run-integration
"""
import pytest
import os
import json
from unittest.mock import patch
from pathlib import Path

pytestmark = pytest.mark.integration

MOCK_ADDITIVES = {
    "en:e621": {
        "name": {"en": "Monosodium glutamate"},
        "additives_classes": {"en": "en:flavour-enhancer"},
        "efsa_evaluation_overexposure_risk": {"en": "en:low"},
        "efsa_evaluation_adi": {"en": "30 mg/kg bw/day"},
        "vegan": {"en": "en:yes"},
        "efsa_evaluation_url": {},
    }
}


@pytest.fixture(autouse=True)
def mock_off_taxonomy(tmp_path):
    additives_path = tmp_path / "additives.json"
    ingredients_path = tmp_path / "ingredients.json"
    additives_path.write_text(json.dumps(MOCK_ADDITIVES))
    ingredients_path.write_text(json.dumps({}))

    import utils.off_client as off_client
    off_client._load_additives.cache_clear()
    off_client._load_ingredients.cache_clear()
    with patch.object(off_client, "ADDITIVES_JSON_PATH", additives_path), \
         patch.object(off_client, "INGREDIENTS_JSON_PATH", ingredients_path):
        yield
    off_client._load_additives.cache_clear()
    off_client._load_ingredients.cache_clear()


def _make_state(raw_ingredients: list[dict]) -> dict:
    return {
        "trace_id": "test-agent2",
        "chat_id": 0,
        "barcode_detected": False,
        "ingredients_text_from_barcode": None,
        "off_product_data": None,
        "image_bytes": None,
        "raw_ingredients_json": raw_ingredients,
        "extraction_error": None,
        "enriched_ingredients": None,
        "grounding_errors": [],
        "health_score": None,
        "red_flags": None,
        "good_ingredients": None,
        "alternatives": None,
        "agent3_result": None,
        "pipeline_failed": False,
        "failure_reason": None,
    }


@pytest.mark.asyncio
async def test_enumber_ingredient_uses_off_taxonomy():
    """E621 should resolve via local OFF taxonomy, no USDA call."""
    from pipeline.agent2_grounding import run_agent2
    state = _make_state([
        {"name": "Monosodium glutamate", "quantity": None, "e_number": "E621"}
    ])
    result = await run_agent2(state)

    enriched = result["enriched_ingredients"]
    assert len(enriched) == 1
    assert enriched[0].source == "off_taxonomy"
    assert enriched[0].efsa_risk == "low"
    assert enriched[0].additive_class == "flavour enhancer"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("USDA_API_KEY"), reason="USDA_API_KEY not set")
async def test_common_ingredient_uses_usda():
    """Palm oil should resolve via USDA FoodData Central."""
    from pipeline.agent2_grounding import run_agent2
    state = _make_state([
        {"name": "Palm oil", "quantity": None, "e_number": None}
    ])
    result = await run_agent2(state)

    enriched = result["enriched_ingredients"]
    assert len(enriched) == 1
    assert enriched[0].source == "usda_api"
    assert enriched[0].usda_nutrients is not None
    assert "fat_g" in enriched[0].usda_nutrients


@pytest.mark.asyncio
async def test_unknown_ingredient_falls_back_to_llm():
    """Completely unknown ingredient should fall through to llm_fallback."""
    from pipeline.agent2_grounding import run_agent2
    with patch("pipeline.agent2_grounding.lookup_ingredient_nutrition", return_value=None):
        state = _make_state([
            {"name": "Xyzunknownstuff123", "quantity": None, "e_number": None}
        ])
        result = await run_agent2(state)

    enriched = result["enriched_ingredients"]
    assert enriched[0].source == "llm_fallback"


@pytest.mark.asyncio
async def test_mixed_ingredients_processed_correctly():
    """Multiple ingredients with different sources."""
    from pipeline.agent2_grounding import run_agent2

    async def mock_usda(name: str):
        if name == "Sugar":
            return {"food_name": "Sugars", "energy_kcal": 387.0, "sugar_g": 100.0}
        return None

    with patch("pipeline.agent2_grounding.lookup_ingredient_nutrition", side_effect=mock_usda):
        state = _make_state([
            {"name": "Monosodium glutamate", "quantity": None, "e_number": "E621"},
            {"name": "Sugar", "quantity": "15g", "e_number": None},
            {"name": "Xyzunknown", "quantity": None, "e_number": None},
        ])
        result = await run_agent2(state)

    enriched = result["enriched_ingredients"]
    assert enriched[0].source == "off_taxonomy"
    assert enriched[1].source == "usda_api"
    assert enriched[2].source == "llm_fallback"
    assert len(result["grounding_errors"]) == 0
