import os
from dotenv import load_dotenv

load_dotenv()

# Required
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
USDA_API_KEY: str = os.environ["USDA_API_KEY"]

# Deployment
WEBHOOK_URL: str = os.environ.get("WEBHOOK_URL", "")
PORT: int = int(os.environ.get("PORT", 8000))

# Logging
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# Tuning
MAX_IMAGE_SIZE_MB: int = int(os.environ.get("MAX_IMAGE_SIZE_MB", 5))
RATE_LIMIT_RPM: int = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", 3))

# Model
GEMINI_MODEL: str = "gemini-2.5-flash-lite-preview-06-17"

# External URLs
USDA_API_BASE: str = "https://api.nal.usda.gov/fdc/v1"
OFF_PRODUCT_API: str = "https://world.openfoodfacts.org/api/v2/product"
OFF_ADDITIVES_TAXONOMY_URL: str = "https://static.openfoodfacts.org/data/taxonomies/additives.json"
OFF_INGREDIENTS_TAXONOMY_URL: str = "https://static.openfoodfacts.org/data/taxonomies/ingredients.json"

# Local data paths
import pathlib
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ADDITIVES_JSON_PATH = DATA_DIR / "additives.json"
INGREDIENTS_JSON_PATH = DATA_DIR / "ingredients.json"
