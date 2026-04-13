"""
Telegram message handlers.

Photo flow:
  1. Rate limit check
  2. Download image from Telegram
  3. Validate (size, format)
  4. Route (barcode detection)
  5. Run LangGraph pipeline
  6. Format and send reply
"""
import io
import time
import uuid

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from pipeline.orchestrator import PIPELINE
from pipeline.state import PipelineState
from bot.formatter import format_response
from bot.router import route_image
from utils.validators import validate_image
from utils.rate_limiter import check_rate_limit
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Welcome to *NutriSnap*\\!\n\n"
        "📸 Send me a photo of the *ingredients section* on any packaged food item\\.\n"
        "I'll tell you exactly what's in it — what's healthy, what's not, and what to watch out for\\.\n\n"
        "You can also send a *barcode photo* and I'll look up the product automatically\\.\n\n"
        "Just send the photo directly in this chat\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📌 *How to use NutriSnap:*\n\n"
        "1\\. Find the ingredients list on the back of any packaged food\n"
        "2\\. Take a clear, well\\-lit photo of just that section\n"
        "3\\. Send it here\n\n"
        "Or send a *barcode photo* — I'll look up the product automatically\\.\n\n"
        "💡 *Tips for best results:*\n"
        "• Make sure the text is in focus\n"
        "• Avoid shadows and glare\n"
        "• Crop to just the ingredients section if possible",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    trace_id = str(uuid.uuid4())[:8]
    start = time.monotonic()

    logger.info("pipeline_started", extra={"trace_id": trace_id, "chat_id": chat_id})
    metrics.increment("requests_total")

    # Rate limit
    if not check_rate_limit(chat_id):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a minute and try again."
        )
        return

    # Acknowledge immediately — pipeline takes 8-12s
    processing_msg = await update.message.reply_text(
        "🔍 Analysing your label… this takes about 10-15 seconds."
    )

    try:
        # Download highest-resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        image_bytes = buf.getvalue()

        # Validate
        validation_error = validate_image(image_bytes)
        if validation_error:
            metrics.increment("validation_errors")
            await processing_msg.edit_text(validation_error)
            return

        # Build initial state
        initial_state: PipelineState = {
            "trace_id": trace_id,
            "chat_id": chat_id,
            "barcode_detected": False,
            "ingredients_text_from_barcode": None,
            "off_product_data": None,
            "image_bytes": image_bytes,
            "raw_ingredients_json": None,
            "extraction_error": None,
            "enriched_ingredients": None,
            "grounding_errors": [],
            "health_score": None,
            "red_flags": None,
            "good_ingredients": None,
            "alternatives": None,
            "agent3_result": None,
            "pipeline_failed": False,
            "failure_reason": None,
        }

        # Router — barcode detection (mutates state in-place)
        initial_state = await route_image(image_bytes, initial_state)

        # Run pipeline
        final_state = await PIPELINE.ainvoke(initial_state)

        response_text = format_response(final_state)
        await processing_msg.edit_text(response_text, parse_mode=ParseMode.MARKDOWN)

        duration_ms = (time.monotonic() - start) * 1000
        metrics.increment("requests_success")
        metrics.record_latency(duration_ms)
        logger.info("pipeline_completed", extra={
            "trace_id": trace_id,
            "chat_id": chat_id,
            "health_score": final_state.get("health_score"),
            "duration_ms": round(duration_ms),
        })

    except Exception as e:
        metrics.increment("requests_failed")
        logger.error("pipeline_unhandled_error", extra={
            "trace_id": trace_id,
            "chat_id": chat_id,
            "error": str(e),
        }, exc_info=True)
        await processing_msg.edit_text(
            "❌ Something went wrong on our end. Please try again in a moment."
        )


async def unknown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📸 Please send me a photo of a food label's ingredients section.\n"
        "Type /help for instructions."
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("telegram_error", extra={"error": str(context.error)}, exc_info=context.error)
