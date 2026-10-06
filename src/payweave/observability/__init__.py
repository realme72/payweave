"""Observability: OpenTelemetry traces and metrics for every gateway call.

``traced`` (instrumentation) only needs ``opentelemetry-api``. The SDK wiring
lives in ``payweave.observability.setup`` and needs the ``otel`` extra.
"""

from payweave.observability.tracing import OPERATIONS, TracedGateway, traced

__all__ = ["OPERATIONS", "TracedGateway", "traced"]
