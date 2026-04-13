"""
Full end-to-end pipeline integration test.
Mocks Gemini + USDA so it runs without API keys, but exercises
the complete LangGraph flow including conditional edges.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

pytestmark = pytest.mark.integration

MOCK_AGENT1_JSON = [
    {"name": "Monosodium glutamate", "quantity": None, "e_number": "E621"},
    {"name": "Palm oil", "quantity": None, "e_number": None},
]

MOCK_AGENT3_JSON = {
    "health_score": 4,
    "score_reasoning": "Contains ultra-processed additives and high saturated fat.",
    "red_flags": [{"ingredient": "Palm oil", "reason": "High in saturated fat."}],
    "good_ingredients": [],
    "alternatives": ["Cold-pressed coconut oil"],
    "ingredient_explanations": [
        {"ingredient": "Monosodium glutamate", "explanation": "A flavour enhancer."},
        {"ingredient": "Palm oil", "explanation": "A vegetable oil high in saturated fat."},
    ],
}


def _make_initial_state(image_bytes: bytes = b"fake_image") -> dict:
    return {
        "trace_id": "test-full-pipeline",
        "chat_id": 12345,
        "barcode_detected": False,
        "ingredients_text_from_barcode": None,
        "off_product_data": None,
        "image_bytes": image_bytes,
        "raw_ingredients_json": None,
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


@pytest.fixture(autouse=True)
def mock_off_taxonomy(tmp_path):
    import utils.off_client as off_client
    additives_path = tmp_path / "additives.json"
    ingredients_path = tmp_path / "ingredients.json"
    additives_path.write_text(json.dumps({
        "en:e621": {
            "name": {"en": "Monosodium glutamate"},
            "additives_classes": {"en": "en:flavour-enhancer"},
            "efsa_evaluation_overexposure_risk": {"en": "en:low"},
            "efsa_evaluation_adi": {"en": "30 mg/kg bw/day"},
            "vegan": {"en": "en:yes"},
            "efsa_evaluation_url": {},
        }
    }))
    ingredients_path.write_text(json.dumps({}))
    off_client._load_additives.cache_clear()
    off_client._load_ingredients.cache_clear()
    with patch.object(off_client, "ADDITIVES_JSON_PATH", additives_path), \
         patch.object(off_client, "INGREDIENTS_JSON_PATH", ingredients_path):
        yield
    off_client._load_additives.cache_clear()
    off_client._load_ingredients.cache_clear()


@pytest.mark.asyncio
async def test_full_pipeline_happy_path():
    """Full pipeline with mocked Gemini and USDA — verifies state flows correctly."""
    mock_vision_response = MagicMock()
    mock_vision_response.text = json.dumps(MOCK_AGENT1_JSON)

    mock_text_response = MagicMock()
    mock_text_response.text = json.dumps(MOCK_AGENT3_JSON)

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = [mock_vision_response, mock_text_response]

    usda_result = {"food_name": "Oil, palm", "fat_g": 100.0, "saturated_fat_g": 49.3}

    with patch("utils.gemini_client.get_vision_model", return_value=mock_model), \
         patch("utils.gemini_client.get_text_model", return_value=mock_model), \
         patch("pipeline.agent2_grounding.lookup_ingredient_nutrition", return_value=usda_result):

        from pipeline.orchestrator import build_pipeline
        pipeline = build_pipeline()
        final_state = await pipeline.ainvoke(_make_initial_state())

    assert not final_state["pipeline_failed"]
    assert final_state["raw_ingredients_json"] is not None
    assert final_state["enriched_ingredients"] is not None
    assert len(final_state["enriched_ingredients"]) == 2
    assert final_state["health_score"] == 4
    assert len(final_state["red_flags"]) == 1


@pytest.mark.asyncio
async def test_pipeline_short_circuits_on_agent1_failure():
    """If Agent 1 fails, Agent 2 and 3 should NOT run."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Gemini quota exceeded")

    agent2_called = False
    agent3_called = False

    async def fake_agent2(state):
        nonlocal agent2_called
        agent2_called = True
        return state

    async def fake_agent3(state):
        nonlocal agent3_called
        agent3_called = True
        return state

    with patch("utils.gemini_client.get_vision_model", return_value=mock_model), \
         patch("pipeline.orchestrator.run_agent2", fake_agent2), \
         patch("pipeline.orchestrator.run_agent3", fake_agent3):

        from pipeline.orchestrator import build_pipeline
        pipeline = build_pipeline()
        final_state = await pipeline.ainvoke(_make_initial_state())

    assert final_state["pipeline_failed"] is True
    assert not agent2_called
    assert not agent3_called
