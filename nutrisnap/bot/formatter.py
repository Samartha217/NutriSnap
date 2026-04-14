"""
Converts the final pipeline state into a Telegram-ready message.
Uses Markdown (Telegram MarkdownV1 via ParseMode.MARKDOWN).
"""
from pipeline.state import PipelineState

_SCORE_BANDS = [
    (8, 10, "🟢"),
    (6, 7,  "🟡"),
    (4, 5,  "🟠"),
    (1, 3,  "🔴"),
]

_ERROR_MESSAGES = {
    "no_ingredients_visible": (
        "❌ Couldn't find an ingredients section in your photo.\n"
        "Try cropping to just the ingredients list and send again."
    ),
    "image_too_blurry": (
        "❌ Image is too blurry to read.\n"
        "Please try again with better lighting and a steady hand."
    ),
}


def _score_emoji(score: int) -> str:
    for low, high, emoji in _SCORE_BANDS:
        if low <= score <= high:
            return emoji
    return "⚪"


def format_response(state: PipelineState) -> str:
    if state.get("pipeline_failed"):
        reason = state.get("failure_reason", "unknown")
        return _ERROR_MESSAGES.get(
            reason,
            f"❌ Something went wrong: {reason}\n\nPlease try again."
        )

    result = state.get("agent3_result") or {}
    score = state.get("health_score") or 0
    emoji = _score_emoji(score)

    lines = []
    lines.append("🧪 *NutriSnap Analysis*")
    lines.append("")
    lines.append(f"*Health Score:* {emoji} {score}/10")

    reasoning = result.get("score_reasoning", "")
    if reasoning:
        lines.append(f"_{reasoning}_")
    lines.append("")

    # Red flags
    red_flags = state.get("red_flags") or []
    if red_flags:
        lines.append("🚨 *Red Flags:*")
        for flag in red_flags:
            lines.append(f"• *{flag['ingredient']}* — {flag['reason']}")
        lines.append("")

    # Good ingredients
    good = state.get("good_ingredients") or []
    if good:
        lines.append("✅ *Okay Ingredients:*")
        for g in good[:5]:
            lines.append(f"• {g}")
        lines.append("")

    # Alternatives
    alternatives = state.get("alternatives") or []
    if alternatives:
        lines.append("💡 *Healthier Alternatives:*")
        for alt in alternatives[:3]:
            lines.append(f"• {alt}")
        lines.append("")

    # Full breakdown
    explanations = result.get("ingredient_explanations") or []
    if explanations:
        lines.append("📖 *Ingredient Breakdown:*")
        for exp in explanations:
            lines.append(f"• *{exp['ingredient']}:* {exp['explanation']}")
        lines.append("")

    lines.append("_Data: Open Food Facts (EFSA) · USDA FoodData Central · Groq / Llama 4_")
    lines.append("_⚠️ Not medical advice._")

    return "\n".join(lines)
