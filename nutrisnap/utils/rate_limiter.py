"""
Simple in-memory per-user rate limiter.
No Redis needed — this is portfolio scale.

Window: sliding 60-second window per chat_id.
Limit: RATE_LIMIT_RPM requests per minute (default: 3).
"""
from collections import defaultdict
from time import time
from config import RATE_LIMIT_RPM
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)

_request_log: dict[int, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 60.0


def check_rate_limit(chat_id: int) -> bool:
    """
    Returns True if the request is allowed, False if rate limited.
    Side effect: records the current request timestamp if allowed.
    """
    now = time()

    # Evict timestamps outside the sliding window
    _request_log[chat_id] = [
        t for t in _request_log[chat_id] if now - t < _WINDOW_SECONDS
    ]

    if len(_request_log[chat_id]) >= RATE_LIMIT_RPM:
        logger.info("rate_limit_hit", extra={"chat_id": chat_id, "limit": RATE_LIMIT_RPM})
        metrics.increment("rate_limit_hits")
        return False

    _request_log[chat_id].append(now)
    return True


def get_remaining(chat_id: int) -> int:
    """How many requests remain in the current window for this user."""
    now = time()
    recent = [t for t in _request_log[chat_id] if now - t < _WINDOW_SECONDS]
    return max(0, RATE_LIMIT_RPM - len(recent))
