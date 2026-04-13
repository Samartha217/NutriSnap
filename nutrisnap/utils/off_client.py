"""
Open Food Facts taxonomy client — local lookups only, zero API calls at runtime.

Data is loaded once at startup from data/additives.json and data/ingredients.json
which are downloaded by scripts/download_data.py.

Lookup priority:
  1. If ingredient has an E-number → look up in additives taxonomy
  2. Try to find a matching E-number via ingredients taxonomy name normalization
  3. Return None if not found (Agent 2 will fall through to USDA)
"""
import json
import re
from typing import Optional
from functools import lru_cache

from config import ADDITIVES_JSON_PATH, INGREDIENTS_JSON_PATH
from observability.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_additives() -> dict:
    if not ADDITIVES_JSON_PATH.exists():
        logger.warning("off_taxonomy_missing", extra={
            "file": str(ADDITIVES_JSON_PATH),
            "hint": "Run: python scripts/download_data.py",
        })
        return {}
    with open(ADDITIVES_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("off_additives_loaded", extra={"entries": len(data)})
    return data


@lru_cache(maxsize=1)
def _load_ingredients() -> dict:
    if not INGREDIENTS_JSON_PATH.exists():
        logger.warning("off_ingredients_taxonomy_missing", extra={
            "file": str(INGREDIENTS_JSON_PATH),
        })
        return {}
    with open(INGREDIENTS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("off_ingredients_loaded", extra={"entries": len(data)})
    return data


def _normalize_enumber(raw: str) -> str:
    """Normalize E-number to the key format used in additives.json: 'en:e621'"""
    clean = raw.strip().lower().replace(" ", "")
    if not clean.startswith("e"):
        clean = "e" + clean
    return f"en:{clean}"


def lookup_additive(e_number: str) -> Optional[dict]:
    """
    Look up an additive by E-number.
    Returns a parsed dict with normalized fields, or None if not found.

    Returned dict fields (all optional, None if not present):
        name (str), additive_class (str), efsa_risk (str),
        efsa_adi (str), vegan (str), efsa_evaluation_url (str)
    """
    key = _normalize_enumber(e_number)
    additives = _load_additives()
    entry = additives.get(key)
    if not entry:
        return None

    # Extract English name
    name = None
    names = entry.get("name", {})
    name = names.get("en") or next(iter(names.values()), None) if names else None

    # Additive class — e.g. "en:flavour-enhancer" → "flavour enhancer"
    classes_raw = entry.get("additives_classes", {})
    additive_class = None
    if classes_raw:
        first = next(iter(classes_raw.values()), None) if isinstance(classes_raw, dict) else None
        if first:
            additive_class = first.replace("en:", "").replace("-", " ")

    # EFSA overexposure risk
    efsa_risk_raw = entry.get("efsa_evaluation_overexposure_risk", {})
    efsa_risk = None
    if efsa_risk_raw:
        val = next(iter(efsa_risk_raw.values()), None) if isinstance(efsa_risk_raw, dict) else None
        if val:
            efsa_risk = val.replace("en:", "")

    # ADI
    efsa_adi_raw = entry.get("efsa_evaluation_adi", {})
    efsa_adi = None
    if efsa_adi_raw:
        efsa_adi = next(iter(efsa_adi_raw.values()), None) if isinstance(efsa_adi_raw, dict) else efsa_adi_raw

    # Vegan status
    vegan_raw = entry.get("vegan", {})
    vegan = None
    if vegan_raw:
        val = next(iter(vegan_raw.values()), None) if isinstance(vegan_raw, dict) else None
        if val:
            vegan = val.replace("en:", "")

    # EFSA evaluation URL
    efsa_url_raw = entry.get("efsa_evaluation_url", {})
    efsa_url = None
    if efsa_url_raw:
        efsa_url = next(iter(efsa_url_raw.values()), None) if isinstance(efsa_url_raw, dict) else None

    return {
        "name": name,
        "e_number": e_number.upper(),
        "additive_class": additive_class,
        "efsa_risk": efsa_risk,
        "efsa_adi": efsa_adi,
        "vegan": vegan,
        "efsa_evaluation_url": efsa_url,
    }


def extract_enumber_from_name(name: str) -> Optional[str]:
    """
    Try to detect an E-number embedded in an ingredient name.
    e.g. "Monosodium glutamate (E621)" → "E621"
         "tartrazine" → None (would need ingredients taxonomy)
    """
    match = re.search(r"\b(E\d{3,4}[a-z]?)\b", name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def get_taxonomy_stats() -> dict:
    """Return counts for health check endpoint."""
    return {
        "additives_count": len(_load_additives()),
        "ingredients_count": len(_load_ingredients()),
    }
