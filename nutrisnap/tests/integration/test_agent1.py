"""
Integration test for Agent 1 — makes a real Gemini API call.
Requires GEMINI_API_KEY in .env and a fixture image.

Run with: pytest tests/integration/ --run-integration
"""
import pytest
import os
from pathlib import Path

pytestmark = pytest.mark.integration

FIXTURE_IMAGE = Path(__file__).parent.parent / "fixtures" / "maggi_label.jpg"


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_agent1_extracts_ingredients_from_image():
    """Real Gemini vision call — verifies we get a valid ingredient list back."""
    if not FIXTURE_IMAGE.exists():
        pytest.skip(f"Fixture image not found: {FIXTURE_IMAGE}")

    from pipeline.agent1_extraction import run_agent1
    from pipeline.state import PipelineState

    image_bytes = FIXTURE_IMAGE.read_bytes()
    state: PipelineState = {
        "trace_id": "test-agent1",
        "chat_id": 0,
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

    result = await run_agent1(state)

    assert not result["pipeline_failed"], f"Pipeline failed: {result.get('failure_reason')}"
    assert result["raw_ingredients_json"] is not None
    assert isinstance(result["raw_ingredients_json"], list)
    assert len(result["raw_ingredients_json"]) > 0

    # Each entry must have a name
    for item in result["raw_ingredients_json"]:
        assert "name" in item
        assert isinstance(item["name"], str)
        assert len(item["name"]) > 0
