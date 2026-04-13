"""
Structured JSON logger for NutriSnap.

Every log line is a JSON object — works with Render logs,
Datadog, or any log aggregator. Grep by trace_id to see
the full lifecycle of any single request.

Usage:
    from observability.logger import get_logger
    logger = get_logger(__name__)
    logger.info("agent1_completed", extra={
        "trace_id": "abc123",
        "chat_id": 456,
        "duration_ms": 3200,
        "ingredients_extracted": 8,
    })
"""
import logging
import json
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # Merge any extra fields passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                log[key] = value

        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        return json.dumps(log, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def configure_root_logger(level: str = "INFO") -> None:
    """Call once at app startup in main.py."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
