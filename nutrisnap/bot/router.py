"""
Barcode detection router.

Runs before Agent 1. If a barcode is detected in the image:
  1. Decode it (pyzbar, local — <100ms)
  2. Fetch product data from Open Food Facts product API
  3. Extract ingredients_text and store in state
  4. Agent 1 will parse text instead of running vision (faster + cheaper)

If no barcode found → normal vision path.
"""
import io
import httpx
from typing import Optional
from PIL import Image

from config import OFF_PRODUCT_API
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)


def detect_barcode(image_bytes: bytes) -> Optional[str]:
    """
    Attempt to decode a barcode from image bytes.
    Returns the barcode string if found, None otherwise.
    Requires: pip install pyzbar
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        image = Image.open(io.BytesIO(image_bytes))
        barcodes = pyzbar_decode(image)
        if barcodes:
            barcode_value = barcodes[0].data.decode("utf-8")
            logger.info("barcode_detected", extra={"barcode": barcode_value})
            metrics.increment("barcode_detections")
            return barcode_value
    except ImportError:
        logger.warning("pyzbar_not_installed", extra={
            "hint": "pip install pyzbar to enable barcode support"
        })
    except Exception as e:
        logger.info("barcode_detection_failed", extra={"error": str(e)})
    return None


async def fetch_off_product(barcode: str) -> Optional[dict]:
    """
    Fetch product data from Open Food Facts by barcode.
    Returns the product dict, or None if not found.
    """
    try:
        url = f"{OFF_PRODUCT_API}/{barcode}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"fields": "product_name,ingredients_text,nova_group,nutriscore_grade,additives_tags"})
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != 1:
            logger.info("off_product_not_found", extra={"barcode": barcode})
            return None

        product = data.get("product", {})
        logger.info("off_product_fetched", extra={
            "barcode": barcode,
            "product_name": product.get("product_name", "unknown"),
        })
        return product

    except httpx.TimeoutException:
        logger.warning("off_product_timeout", extra={"barcode": barcode})
        return None
    except Exception as e:
        logger.error("off_product_error", extra={"barcode": barcode, "error": str(e)})
        return None


async def route_image(image_bytes: bytes, state: dict) -> dict:
    """
    Detect barcode and enrich state before Agent 1 runs.
    Mutates and returns state.
    """
    barcode = detect_barcode(image_bytes)

    if not barcode:
        state["barcode_detected"] = False
        return state

    product = await fetch_off_product(barcode)
    if not product:
        # Barcode found but not in OFF — fall back to vision path
        state["barcode_detected"] = False
        return state

    ingredients_text = product.get("ingredients_text", "").strip()
    if not ingredients_text:
        state["barcode_detected"] = False
        return state

    state["barcode_detected"] = True
    state["ingredients_text_from_barcode"] = ingredients_text
    state["off_product_data"] = product
    return state
