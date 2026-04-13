"""
In-process metrics for NutriSnap.

No external dependency (no Prometheus, no StatsD).
Exposed at GET /metrics for quick inspection on Render.

Usage:
    from observability.metrics import metrics
    metrics.increment("requests_total")
    metrics.increment("agent2_off_hits")
    metrics.record_latency(duration_ms=3200)
"""
import time
import threading
from typing import Any


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._counters: dict[str, int] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "agent1_errors": 0,
            "agent2_off_hits": 0,
            "agent2_usda_hits": 0,
            "agent2_llm_fallbacks": 0,
            "agent3_errors": 0,
            "rate_limit_hits": 0,
            "barcode_detections": 0,
            "validation_errors": 0,
        }
        self._latencies: list[float] = []

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            if key not in self._counters:
                self._counters[key] = 0
            self._counters[key] += amount

    def record_latency(self, duration_ms: float) -> None:
        with self._lock:
            self._latencies.append(duration_ms)
            # Keep only last 100 to avoid unbounded growth
            if len(self._latencies) > 100:
                self._latencies = self._latencies[-100:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies else 0.0
            )
            return {
                **self._counters,
                "avg_pipeline_latency_ms": round(avg_latency, 1),
                "uptime_seconds": round(time.time() - self._start_time),
            }


# Singleton — import and use anywhere
metrics = Metrics()
