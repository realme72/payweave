"""Wire the OpenTelemetry SDK: where spans and metrics go. Needs ``payweave[otel]``.

Call once at process start (an app, a script), never from library code::

    shutdown = configure_telemetry()
    ...
    shutdown()  # flushes pending spans

Export goes over OTLP/HTTP, so the backend is a config choice, not a code change:
Jaeger locally (``make jaeger``), or Grafana Cloud / any OTLP backend through
the standard variables ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_HEADERS``.
Metrics are exported only when ``OTEL_METRICS_EXPORTER`` is ``otlp`` or
``console`` (Jaeger stores traces only).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib.metadata import version

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter


def _metric_reader_from_env() -> MetricReader | None:
    match os.environ.get("OTEL_METRICS_EXPORTER", "none"):
        case "otlp":
            return PeriodicExportingMetricReader(OTLPMetricExporter())
        case "console":
            return PeriodicExportingMetricReader(ConsoleMetricExporter())
        case _:
            return None


def configure_telemetry(
    service_name: str = "payweave",
    *,
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
    instrument_http: bool = True,
) -> Callable[[], None]:
    """Install global tracer/meter providers. Returns a shutdown function.

    ``instrument_http`` adds a child span per outgoing HTTP request (method,
    URL, status code). Headers are not captured, so credentials stay out.
    """
    resource = Resource.create(
        {"service.name": service_name, "service.version": version("payweave")}
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter or OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    reader = metric_reader or _metric_reader_from_env()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader] if reader else [])
    metrics.set_meter_provider(meter_provider)

    if instrument_http:
        HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)

    def shutdown() -> None:
        if instrument_http:
            HTTPXClientInstrumentor().uninstrument()
        tracer_provider.shutdown()
        meter_provider.shutdown()

    return shutdown
