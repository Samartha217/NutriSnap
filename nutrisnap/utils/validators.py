"""
Input validation for images before they enter the pipeline.
Returns an error message string if invalid, None if valid.
"""
from PIL import Image
import io
from config import MAX_IMAGE_SIZE_MB
from observability.logger import get_logger

logger = get_logger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}


def validate_image(image_bytes: bytes) -> str | None:
    """
    Validate image bytes before sending to pipeline.
    Returns a user-facing error string if invalid, None if valid.
    """
    # Size check
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        logger.info("validation_failed_size", extra={"size_mb": round(size_mb, 2)})
        return (
            f"❌ Image too large ({size_mb:.1f}MB). "
            f"Please send an image under {MAX_IMAGE_SIZE_MB}MB."
        )

    # Format + integrity check
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Raises on corrupt files
    except Exception as e:
        logger.info("validation_failed_format", extra={"error": str(e)})
        return "❌ Could not read image. Please try a different photo."

    # Re-open after verify (verify() closes the file)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ALLOWED_FORMATS:
            logger.info("validation_failed_format_type", extra={"format": img.format})
            return f"❌ Unsupported image format ({img.format}). Please send a JPEG or PNG."
    except Exception:
        return "❌ Could not read image. Please try a different photo."

    return None
