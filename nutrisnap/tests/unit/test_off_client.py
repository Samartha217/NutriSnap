"""
Unit tests for utils/off_client.py

These tests mock the filesystem so they run without
the actual data files present (no download needed for CI).
"""
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path


# Sample minimal additives.json entries for testing
MOCK_ADDITIVES = {
    "en:e621": {
        "name": {"en": "Monosodium glutamate"},
        "additives_classes": {"en": "en:flavour-enhancer"},
        "efsa_evaluation_overexposure_risk": {"en": "en:low"},
        "efsa_evaluation_adi": {"en": "30 mg/kg bw/day"},
        "vegan": {"en": "en:yes"},
        "efsa_evaluation_url": {"en": "https://efsa.example.com/e621"},
    },
    "en:e102": {
        "name": {"en": "Tartrazine"},
        "additives_classes": {"en": "en:colour"},
        "efsa_evaluation_overexposure_risk": {"en": "en:high"},
        "efsa_evaluation_adi": {"en": "7.5 mg/kg bw/day"},
        "vegan": {"en": "en:yes"},
        "efsa_evaluation_url": {"en": "https://efsa.example.com/e102"},
    },
    "en:e211": {
        "name": {"en": "Sodium benzoate"},
        "additives_classes": {"en": "en:preservative"},
        "efsa_evaluation_overexposure_risk": {},
        "efsa_evaluation_adi": {},
        "vegan": {},
        "efsa_evaluation_url": {},
    },
}


@pytest.fixture(autouse=True)
def mock_taxonomy_files(tmp_path):
    """Redirect taxonomy paths to temp files for all tests in this module."""
    additives_path = tmp_path / "additives.json"
    ingredients_path = tmp_path / "ingredients.json"
    additives_path.write_text(json.dumps(MOCK_ADDITIVES), encoding="utf-8")
    ingredients_path.write_text(json.dumps({}), encoding="utf-8")

    import utils.off_client as off_client
    # Clear lru_cache so each test gets a fresh load
    off_client._load_additives.cache_clear()
    off_client._load_ingredients.cache_clear()

    with patch.object(off_client, "ADDITIVES_JSON_PATH", additives_path), \
         patch.object(off_client, "INGREDIENTS_JSON_PATH", ingredients_path):
        yield

    off_client._load_additives.cache_clear()
    off_client._load_ingredients.cache_clear()


class TestLookupAdditive:
    def test_known_enumber_returns_data(self):
        from utils.off_client import lookup_additive
        result = lookup_additive("E621")
        assert result is not None
        assert result["name"] == "Monosodium glutamate"
        assert result["e_number"] == "E621"
        assert result["additive_class"] == "flavour enhancer"
        assert result["efsa_risk"] == "low"
        assert result["efsa_adi"] == "30 mg/kg bw/day"
        assert result["vegan"] == "yes"

    def test_high_risk_additive(self):
        from utils.off_client import lookup_additive
        result = lookup_additive("E102")
        assert result is not None
        assert result["name"] == "Tartrazine"
        assert result["efsa_risk"] == "high"

    def test_unknown_enumber_returns_none(self):
        from utils.off_client import lookup_additive
        result = lookup_additive("E999")
        assert result is None

    def test_case_insensitive_lookup(self):
        from utils.off_client import lookup_additive
        assert lookup_additive("e621") is not None
        assert lookup_additive("E621") is not None

    def test_missing_optional_fields_return_none(self):
        from utils.off_client import lookup_additive
        # E211 has empty dicts for several fields
        result = lookup_additive("E211")
        assert result is not None
        assert result["efsa_risk"] is None
        assert result["efsa_adi"] is None
        assert result["vegan"] is None

    def test_enumber_with_spaces_normalizes(self):
        from utils.off_client import lookup_additive
        result = lookup_additive("E 621")
        assert result is not None


class TestExtractEnumberFromName:
    def test_enumber_in_parentheses(self):
        from utils.off_client import extract_enumber_from_name
        assert extract_enumber_from_name("Monosodium glutamate (E621)") == "E621"

    def test_enumber_standalone(self):
        from utils.off_client import extract_enumber_from_name
        assert extract_enumber_from_name("E102") == "E102"

    def test_no_enumber_returns_none(self):
        from utils.off_client import extract_enumber_from_name
        assert extract_enumber_from_name("Palm oil") is None
        assert extract_enumber_from_name("Sugar") is None

    def test_four_digit_enumber(self):
        from utils.off_client import extract_enumber_from_name
        assert extract_enumber_from_name("E1422") == "E1422"


class TestGetTaxonomyStats:
    def test_returns_counts(self):
        from utils.off_client import get_taxonomy_stats
        stats = get_taxonomy_stats()
        assert stats["additives_count"] == len(MOCK_ADDITIVES)
        assert "ingredients_count" in stats
