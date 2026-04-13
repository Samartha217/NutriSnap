"""
Unit tests for utils/usda_client.py — all HTTP calls are mocked.
"""
import pytest
import httpx
import respx
from unittest.mock import patch


MOCK_USDA_RESPONSE = {
    "foods": [
        {
            "description": "Oil, palm",
            "foodNutrients": [
                {"nutrient": {"id": 1004}, "amount": 100.0},   # fat
                {"nutrient": {"id": 1258}, "amount": 49.3},    # saturated fat
                {"nutrient": {"id": 1008}, "amount": 884.0},   # energy
                {"nutrient": {"id": 1093}, "amount": 0.0},     # sodium
            ],
        }
    ]
}

EMPTY_USDA_RESPONSE = {"foods": []}


@pytest.mark.asyncio
class TestLookupIngredientNutrition:

    @respx.mock
    async def test_successful_lookup(self):
        from utils.usda_client import lookup_ingredient_nutrition
        respx.get("https://api.nal.usda.gov/fdc/v1/foods/search").mock(
            return_value=httpx.Response(200, json=MOCK_USDA_RESPONSE)
        )
        with patch("utils.usda_client.USDA_API_KEY", "test_key"):
            result = await lookup_ingredient_nutrition("palm oil")

        assert result is not None
        assert result["food_name"] == "Oil, palm"
        assert result["fat_g"] == 100.0
        assert result["saturated_fat_g"] == 49.3
        assert result["energy_kcal"] == 884.0

    @respx.mock
    async def test_not_found_returns_none(self):
        from utils.usda_client import lookup_ingredient_nutrition
        respx.get("https://api.nal.usda.gov/fdc/v1/foods/search").mock(
            return_value=httpx.Response(200, json=EMPTY_USDA_RESPONSE)
        )
        with patch("utils.usda_client.USDA_API_KEY", "test_key"):
            result = await lookup_ingredient_nutrition("xyznonexistent")

        assert result is None

    @respx.mock
    async def test_timeout_returns_none(self):
        from utils.usda_client import lookup_ingredient_nutrition
        respx.get("https://api.nal.usda.gov/fdc/v1/foods/search").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        with patch("utils.usda_client.USDA_API_KEY", "test_key"):
            result = await lookup_ingredient_nutrition("sugar")

        assert result is None  # graceful degradation

    @respx.mock
    async def test_api_error_returns_none(self):
        from utils.usda_client import lookup_ingredient_nutrition
        respx.get("https://api.nal.usda.gov/fdc/v1/foods/search").mock(
            return_value=httpx.Response(403, json={"error": "forbidden"})
        )
        with patch("utils.usda_client.USDA_API_KEY", "test_key"):
            result = await lookup_ingredient_nutrition("sugar")

        assert result is None

    async def test_missing_api_key_returns_none(self):
        from utils.usda_client import lookup_ingredient_nutrition
        with patch("utils.usda_client.USDA_API_KEY", ""):
            result = await lookup_ingredient_nutrition("palm oil")
        assert result is None
