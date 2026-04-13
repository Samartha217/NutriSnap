"""
Local development runner — polling mode.
Do NOT deploy this file.

Usage:
    cd nutrisnap
    python run_local.py
"""
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN, LOG_LEVEL
from bot.handlers import (
    start_handler, help_handler, photo_handler,
    unknown_handler, error_handler,
)
from observability.logger import configure_root_logger
from observability.metrics import metrics
from utils.off_client import get_taxonomy_stats

configure_root_logger(LOG_LEVEL)

from observability.logger import get_logger
logger = get_logger(__name__)


def main() -> None:
    # Warm up OFF taxonomy
    stats = get_taxonomy_stats()
    logger.info("taxonomy_loaded", extra=stats)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_handler))
    app.add_error_handler(error_handler)

    logger.info("nutrisnap_polling_started")
    print("\nNutriSnap running in polling mode. Send a photo to your bot.\nCtrl+C to stop.\n")
    app.run_polling()


if __name__ == "__main__":
    main()
