"""
USDA FoodData Central API client.

Used by Agent 2 to fetch nutritional facts for non-additive ingredients
(palm oil, sugar, wheat flour, etc.) where OFF taxonomy has no data.

Free tier: 1000 requests/hour. API key required (free signup).
Docs: https://fdc.nal.usda.gov/api-guide/
"""
import httpx
from typing import Optional

from config import USDA_API_KEY, USDA_API_BASE
from observability.logger import get_logger

logger = get_logger(__name__)

# Nutrients we care about — keyed by USDA nutrient number
NUTRIENT_IDS = {
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1008: "energy_kcal",
    1079: "fiber_g",
    1093: "sodium_mg",
    1258: "saturated_fat_g",
    2000: "sugar_g",
}


def _parse_nutrients(food_nutrients: list[dict]) -> dict:
    """Extract the nutrients we care about from USDA response."""
    result: dict[str, float] = {}
    for nutrient in food_nutrients:
        nutrient_id = nutrient.get("nutrient", {}).get("id")
        if nutrient_id in NUTRIENT_IDS:
            key = NUTRIENT_IDS[nutrient_id]
            result[key] = round(nutrient.get("amount", 0.0), 2)
    return result


async def lookup_ingredient_nutrition(name: str) -> Optional[dict]:
    """
    Search USDA FoodData Central for nutritional data on an ingredient.

    Returns a dict of nutrient values per 100g, or None if not found / API error.

    Example return:
    {
        "food_name": "Oil, palm",
        "fat_g": 100.0,
        "saturated_fat_g": 49.3,
        "energy_kcal": 884.0,
        ...
    }
    """
    if not USDA_API_KEY:
        logger.warning("usda_api_key_missing")
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{USDA_API_BASE}/foods/search",
                params={
                    "query": name,
                    "api_key": USDA_API_KEY,
                    "dataType": "Foundation,SR Legacy",  # Most reliable for raw ingredients
                    "pageSize": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        foods = data.get("foods", [])
        if not foods:
            logger.info("usda_not_found", extra={"ingredient": name})
            return None

        food = foods[0]
        nutrients = _parse_nutrients(food.get("foodNutrients", []))

        if not nutrients:
            return None

        return {
            "food_name": food.get("description", name),
            **nutrients,
        }

    except httpx.TimeoutException:
        logger.warning("usda_timeout", extra={"ingredient": name})
        return None
    except httpx.HTTPStatusError as e:
        logger.error("usda_http_error", extra={
            "ingredient": name,
            "status_code": e.response.status_code,
        })
        return None
    except Exception as e:
        logger.error("usda_unexpected_error", extra={
            "ingredient": name,
            "error": str(e),
        })
        return None
