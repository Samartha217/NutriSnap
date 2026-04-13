"""
NutriSnap — FastAPI app + Telegram webhook entry point.

Startup sequence:
  1. Configure structured logging
  2. Build Telegram PTB application
  3. On lifespan start: register webhook, load OFF taxonomy
  4. /webhook receives Telegram updates
  5. /health and /metrics for observability
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN, WEBHOOK_URL, PORT, LOG_LEVEL
from bot.handlers import (
    start_handler, help_handler, photo_handler,
    unknown_handler, error_handler,
)
from observability.logger import get_logger, configure_root_logger
from observability.metrics import metrics
from utils.off_client import get_taxonomy_stats

configure_root_logger(LOG_LEVEL)
logger = get_logger(__name__)

# Build PTB application (webhook mode — no updater)
ptb_app = (
    Application.builder()
    .token(TELEGRAM_BOT_TOKEN)
    .updater(None)
    .build()
)

ptb_app.add_handler(CommandHandler("start", start_handler))
ptb_app.add_handler(CommandHandler("help", help_handler))
ptb_app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_handler))
ptb_app.add_error_handler(error_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ptb_app.initialize()

    if WEBHOOK_URL:
        webhook_url = f"{WEBHOOK_URL}/webhook"
        await ptb_app.bot.set_webhook(
            url=webhook_url,
            allowed_updates=["message"],
        )
        logger.info("webhook_registered", extra={"url": webhook_url})
    else:
        logger.warning("webhook_url_not_set", extra={
            "hint": "Set WEBHOOK_URL in .env for production. Use run_local.py for polling."
        })

    # Warm up OFF taxonomy (loads + caches JSON files)
    stats = get_taxonomy_stats()
    logger.info("taxonomy_loaded", extra=stats)

    await ptb_app.start()
    logger.info("nutrisnap_ready")
    yield

    await ptb_app.stop()
    logger.info("nutrisnap_shutdown")


app = FastAPI(title="NutriSnap", lifespan=lifespan)


@app.post("/webhook")
async def telegram_webhook(request: Request) -> Response:
    data = await request.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return Response(status_code=200)


@app.get("/health")
async def health_check() -> dict:
    taxonomy_stats = get_taxonomy_stats()
    return {
        "status": "ok",
        "off_taxonomy_loaded": taxonomy_stats["additives_count"] > 0,
        **taxonomy_stats,
    }


@app.get("/metrics")
async def get_metrics() -> dict:
    return metrics.snapshot()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
