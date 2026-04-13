"""Unit tests for utils/validators.py"""
import io
import pytest
from unittest.mock import patch
from PIL import Image


def _make_image_bytes(format: str = "JPEG", size: tuple = (100, 100)) -> bytes:
    """Create a valid in-memory image."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(100, 149, 237))
    img.save(buf, format=format)
    return buf.getvalue()


class TestValidateImage:
    def test_valid_jpeg_passes(self):
        from utils.validators import validate_image
        result = validate_image(_make_image_bytes("JPEG"))
        assert result is None

    def test_valid_png_passes(self):
        from utils.validators import validate_image
        result = validate_image(_make_image_bytes("PNG"))
        assert result is None

    def test_oversized_image_rejected(self):
        from utils.validators import validate_image
        with patch("utils.validators.MAX_IMAGE_SIZE_MB", 1):
            big_bytes = b"x" * (2 * 1024 * 1024)  # 2MB
            result = validate_image(big_bytes)
        assert result is not None
        assert "too large" in result.lower()

    def test_corrupt_bytes_rejected(self):
        from utils.validators import validate_image
        result = validate_image(b"not_an_image_at_all")
        assert result is not None
        assert "could not read" in result.lower()

    def test_empty_bytes_rejected(self):
        from utils.validators import validate_image
        result = validate_image(b"")
        assert result is not None

    def test_exact_size_limit_passes(self):
        from utils.validators import validate_image
        # Just under the limit should pass
        with patch("utils.validators.MAX_IMAGE_SIZE_MB", 5):
            result = validate_image(_make_image_bytes("JPEG"))
        assert result is None
