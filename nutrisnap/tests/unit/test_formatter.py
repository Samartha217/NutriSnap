"""Unit tests for bot/formatter.py"""
import pytest


def _make_state(**overrides) -> dict:
    base = {
        "trace_id": "test",
        "chat_id": 1,
        "barcode_detected": False,
        "ingredients_text_from_barcode": None,
        "off_product_data": None,
        "image_bytes": None,
        "raw_ingredients_json": None,
        "extraction_error": None,
        "enriched_ingredients": None,
        "grounding_errors": [],
        "health_score": 7,
        "red_flags": [],
        "good_ingredients": ["Wheat flour", "Water"],
        "alternatives": ["Brand X"],
        "agent3_result": {
            "health_score": 7,
            "score_reasoning": "Mostly natural ingredients.",
            "red_flags": [],
            "good_ingredients": ["Wheat flour", "Water"],
            "alternatives": ["Brand X"],
            "ingredient_explanations": [
                {"ingredient": "Wheat flour", "explanation": "A refined grain flour."}
            ],
        },
        "pipeline_failed": False,
        "failure_reason": None,
    }
    base.update(overrides)
    return base


class TestFormatResponse:
    def test_success_contains_score(self):
        from bot.formatter import format_response
        result = format_response(_make_state())
        assert "7/10" in result
        assert "NutriSnap Analysis" in result

    def test_score_8_to_10_is_green(self):
        from bot.formatter import format_response
        state = _make_state(health_score=9)
        state["agent3_result"]["health_score"] = 9
        assert "🟢" in format_response(state)

    def test_score_4_to_5_is_orange(self):
        from bot.formatter import format_response
        state = _make_state(health_score=4)
        assert "🟠" in format_response(state)

    def test_score_1_to_3_is_red(self):
        from bot.formatter import format_response
        state = _make_state(health_score=2)
        assert "🔴" in format_response(state)

    def test_red_flags_shown(self):
        from bot.formatter import format_response
        state = _make_state(red_flags=[
            {"ingredient": "Tartrazine", "reason": "EFSA high overexposure risk."}
        ])
        result = format_response(state)
        assert "Tartrazine" in result
        assert "Red Flags" in result

    def test_no_red_flags_section_hidden(self):
        from bot.formatter import format_response
        state = _make_state(red_flags=[])
        result = format_response(state)
        assert "Red Flags" not in result

    def test_pipeline_failed_no_ingredients(self):
        from bot.formatter import format_response
        state = _make_state(
            pipeline_failed=True,
            failure_reason="no_ingredients_visible",
        )
        result = format_response(state)
        assert "❌" in result
        assert "ingredients section" in result.lower()

    def test_pipeline_failed_blurry(self):
        from bot.formatter import format_response
        state = _make_state(
            pipeline_failed=True,
            failure_reason="image_too_blurry",
        )
        result = format_response(state)
        assert "blurry" in result.lower()

    def test_pipeline_failed_unknown_reason(self):
        from bot.formatter import format_response
        state = _make_state(
            pipeline_failed=True,
            failure_reason="some_unexpected_error",
        )
        result = format_response(state)
        assert "❌" in result
        assert "some_unexpected_error" in result

    def test_alternatives_capped_at_three(self):
        from bot.formatter import format_response
        state = _make_state()
        state["alternatives"] = ["A", "B", "C", "D", "E"]
        result = format_response(state)
        assert result.count("•") <= 15  # rough sanity check
        # Only first 3 alternatives shown
        assert "D" not in result
        assert "E" not in result

    def test_disclaimer_always_present(self):
        from bot.formatter import format_response
        assert "Not medical advice" in format_response(_make_state())
