"""
TimeStone AI — Observability

Production-grade observability: structured logging, Prometheus metrics,
and distributed tracing via OpenTelemetry.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional

import structlog


# ---- Structured Logging ----

def setup_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured logging for the application."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}.get(log_level.upper(), 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "timestone") -> structlog.BoundLogger:
    return structlog.get_logger(name)


# ---- Metrics ----

class MetricsCollector:
    """
    Application metrics collector.

    Uses Prometheus client if available, falls back to in-memory counters.
    """

    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._gauges: Dict[str, float] = {}
        self._prom_available = False

        try:
            from prometheus_client import Counter, Histogram, Gauge, REGISTRY
            self._prom_available = True

            def _counter(name, desc, labels=None):
                try:
                    return Counter(name, desc, labels or [])
                except ValueError:
                    return REGISTRY._names_to_collectors.get(name)

            def _histogram(name, desc, labels=None, buckets=None):
                try:
                    return Histogram(name, desc, labels or [], buckets=buckets or Histogram.DEFAULT_BUCKETS)
                except ValueError:
                    return REGISTRY._names_to_collectors.get(name)

            def _gauge(name, desc):
                try:
                    return Gauge(name, desc)
                except ValueError:
                    return REGISTRY._names_to_collectors.get(name)

            self.prom_requests = _counter("timestone_api_requests_total", "Total API requests", ["method", "endpoint", "status"])
            self.prom_simulation_duration = _histogram("timestone_simulation_duration_seconds", "Simulation duration", ["method", "scenarios"], [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0])
            self.prom_active_simulations = _gauge("timestone_active_simulations", "Currently running simulations")
            self.prom_companies = _gauge("timestone_companies_total", "Total registered companies")
            self.prom_predictions = _counter("timestone_predictions_total", "Total predictions made", ["industry"])
            self.prom_calibration_score = _gauge("timestone_calibration_score", "Bayesian calibration score")
        except ImportError:
            pass

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict] = None) -> None:
        key = f"{name}:{labels}" if labels else name
        self._counters[key] = self._counters.get(key, 0) + value

        if self._prom_available and name == "api_requests":
            self.prom_requests.labels(**labels).inc(value)
        elif self._prom_available and name == "predictions":
            self.prom_predictions.labels(**labels).inc(value)

    def observe(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        key = f"{name}:{labels}" if labels else name
        self._histograms.setdefault(key, []).append(value)

        if self._prom_available and name == "simulation_duration":
            self.prom_simulation_duration.labels(**labels).observe(value)

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

        if self._prom_available:
            if name == "active_simulations":
                self.prom_active_simulations.set(value)
            elif name == "companies_total":
                self.prom_companies.set(value)
            elif name == "calibration_score":
                self.prom_calibration_score.set(value)

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe(name, elapsed, labels)

    def get_summary(self) -> Dict[str, Any]:
        import numpy as np

        summary = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
        }
        for key, values in self._histograms.items():
            if values:
                arr = np.array(values)
                summary["histograms"][key] = {
                    "count": len(arr),
                    "mean": float(np.mean(arr)),
                    "p50": float(np.median(arr)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(np.max(arr)),
                }
        return summary


# Global singleton
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# ---- Tracing ----

class SpanTracer:
    """
    Simplified distributed tracing.

    Uses OpenTelemetry if available, falls back to structured log spans.
    """

    def __init__(self):
        self._otel_available = False
        self._tracer = None

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter

            provider = TracerProvider()
            self._exporter = InMemorySpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(self._exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("timestone-ai")
            self._otel_available = True
        except ImportError:
            pass

        self._logger = get_logger("tracing")

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict] = None) -> Iterator[None]:
        if self._otel_available and self._tracer:
            with self._tracer.start_as_current_span(name, attributes=attributes):
                yield
        else:
            start = time.perf_counter()
            self._logger.info("span_start", span_name=name, **(attributes or {}))
            try:
                yield
            finally:
                elapsed = time.perf_counter() - start
                self._logger.info("span_end", span_name=name, duration_ms=int(elapsed * 1000))

    def get_recent_spans(self) -> list:
        if self._otel_available:
            return [
                {
                    "name": s.name,
                    "duration_ns": s.end_time - s.start_time if s.end_time and s.start_time else 0,
                    "attributes": dict(s.attributes) if s.attributes else {},
                }
                for s in self._exporter.get_finished_spans()
            ]
        return []


_tracer: Optional[SpanTracer] = None


def get_tracer() -> SpanTracer:
    global _tracer
    if _tracer is None:
        _tracer = SpanTracer()
    return _tracer


# ---- Decorators ----

def traced(name: Optional[str] = None):
    """Decorator to trace a function execution."""
    def decorator(fn: Callable) -> Callable:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.span(span_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def metered(metric_name: str, labels: Optional[Dict] = None):
    """Decorator to measure function execution time."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.timer(metric_name, labels):
                return fn(*args, **kwargs)
        return wrapper
    return decorator
