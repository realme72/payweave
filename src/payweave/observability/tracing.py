"""OpenTelemetry instrumentation for gateways (Decorator pattern).

``traced(gateway)`` returns an object with the same operations as ``gateway``,
where every call becomes a span plus two metrics:

* ``payweave.gateway.operations``         counter, one per call
* ``payweave.gateway.operation.duration`` histogram, seconds

Gateways stay unaware of telemetry, so a new (or AI-generated) gateway is
instrumented for free. Without an OpenTelemetry SDK configured every call here
is a no-op, so wrapping is always safe.

**Allowlist, not denylist.** Only the attributes built below are recorded:
gateway/operation names, ids, amounts, statuses and error classification.
Never request bodies, headers, signatures, metadata, merchant references,
idempotency keys or exception messages. ``tests/observability`` enforces this.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import Span, Status, StatusCode, TracerProvider

from payweave.core.capabilities import INTERFACE_FOR, Capability, Gateway
from payweave.core.errors import GatewayError
from payweave.core.models import (
    CaptureRequest,
    CreatePaymentRequest,
    PaymentIntent,
    Refund,
    RefundRequest,
    VoidRequest,
    WebhookEvent,
)

SCOPE = "payweave"

# Every method declared by an operation interface (create_payment, refund, ...).
OPERATIONS: tuple[str, ...] = tuple(
    sorted(
        {
            name
            for interface in INTERFACE_FOR.values()
            for name, value in vars(interface).items()
            if inspect.isfunction(value) and not name.startswith("_")
        }
    )
)

type Attributes = dict[str, str | int | bool]


class _Instruments:
    def __init__(
        self, tracer_provider: TracerProvider | None, meter_provider: MeterProvider | None
    ) -> None:
        self.tracer = trace.get_tracer(SCOPE, tracer_provider=tracer_provider)
        meter = metrics.get_meter(SCOPE, meter_provider=meter_provider)
        self.calls = meter.create_counter(
            "payweave.gateway.operations", unit="{operation}", description="Gateway calls"
        )
        self.duration = meter.create_histogram(
            "payweave.gateway.operation.duration", unit="s", description="Gateway call latency"
        )


# --- allowlisted attributes ---------------------------------------------------


def _request_attributes(operation: str, args: tuple[Any, ...]) -> Attributes:
    attrs: Attributes = {}
    for arg in args:
        match arg:
            case CreatePaymentRequest():
                attrs["payweave.amount.minor"] = arg.amount.amount_minor
                attrs["payweave.currency"] = arg.amount.currency
                attrs["payweave.capture_mode"] = str(arg.capture_mode)
            case CaptureRequest() | RefundRequest() | VoidRequest():
                attrs["payweave.gateway_intent_id"] = arg.gateway_intent_id
                amount = getattr(arg, "amount", None)
                if amount is not None:
                    attrs["payweave.amount.minor"] = amount.amount_minor
                    attrs["payweave.currency"] = amount.currency
            case str() if operation == "get_payment":
                attrs["payweave.gateway_intent_id"] = arg
            case str() if operation == "get_refund":
                attrs["payweave.gateway_refund_id"] = arg
            case _:
                pass  # raw webhook bodies, headers, checkout params: never recorded
    return attrs


def _result_attributes(result: object) -> Attributes:
    match result:
        case PaymentIntent():
            return {
                "payweave.gateway_intent_id": result.gateway_intent_id,
                "payweave.intent.status": str(result.status),
                "payweave.intent.attempts": len(result.attempts),
            }
        case Refund():
            return {
                "payweave.gateway_refund_id": result.gateway_refund_id,
                "payweave.refund.status": str(result.status),
            }
        case WebhookEvent():
            attrs: Attributes = {
                "payweave.webhook.event_id": result.event_id,
                "payweave.webhook.event_type": str(result.event_type),
                "payweave.webhook.gateway_event_type": result.gateway_event_type,
            }
            if result.gateway_intent_id:
                attrs["payweave.gateway_intent_id"] = result.gateway_intent_id
            if result.gateway_refund_id:
                attrs["payweave.gateway_refund_id"] = result.gateway_refund_id
            return attrs
        case str():  # verify_checkout_return -> intent id
            return {"payweave.gateway_intent_id": result}
        case _:
            return {}


def _error_attributes(exc: Exception) -> Attributes:
    attrs: Attributes = {"error.type": type(exc).__name__}
    if isinstance(exc, GatewayError):
        attrs["payweave.error.retryable"] = exc.retryable
        attrs["payweave.error.outcome_unknown"] = exc.outcome_unknown
        if exc.gateway_code:
            attrs["payweave.error.gateway_code"] = exc.gateway_code
        if exc.decline_reason:
            attrs["payweave.error.decline_reason"] = exc.decline_reason
    return attrs


# --- the decorator --------------------------------------------------------------


class _Call:
    """Span + metrics bookkeeping for one operation call."""

    def __init__(self, gateway: TracedGateway, operation: str, args: tuple[Any, ...]) -> None:
        self.metric_attrs: Attributes = {
            "payweave.gateway": gateway.name,
            "payweave.operation": operation,
        }
        self.span_attrs = {**self.metric_attrs, **_request_attributes(operation, args)}
        self.instruments = gateway._instruments
        self.name = f"payweave.{operation}"
        self.started = time.perf_counter()

    def succeeded(self, span: Span, result: object) -> None:
        span.set_attributes(_result_attributes(result))
        self._record({"payweave.outcome": "ok"})

    def failed(self, span: Span, exc: Exception) -> None:
        attrs = _error_attributes(exc)
        span.set_attributes(attrs)
        # Status carries the error class only: messages can echo gateway input.
        span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        self._record({"payweave.outcome": "error", "error.type": attrs["error.type"]})

    def _record(self, outcome: Attributes) -> None:
        attrs = {**self.metric_attrs, **outcome}
        self.instruments.calls.add(1, attrs)
        self.instruments.duration.record(time.perf_counter() - self.started, attrs)


def _wrap(operation: str, is_async: bool) -> Callable[..., Any]:
    # record_exception/set_status_on_exception are off: both would export the
    # exception message and stack trace, which is outside the allowlist.
    def span_for(call: _Call) -> Any:
        return call.instruments.tracer.start_as_current_span(
            call.name,
            attributes=call.span_attrs,
            record_exception=False,
            set_status_on_exception=False,
        )

    if is_async:

        async def async_method(self: TracedGateway, *args: Any, **kwargs: Any) -> Any:
            call = _Call(self, operation, args)
            with span_for(call) as span:
                try:
                    result = await getattr(self.inner, operation)(*args, **kwargs)
                except Exception as exc:
                    call.failed(span, exc)
                    raise
                call.succeeded(span, result)
                return result

        method: Callable[..., Any] = async_method
    else:

        def sync_method(self: TracedGateway, *args: Any, **kwargs: Any) -> Any:
            call = _Call(self, operation, args)
            with span_for(call) as span:
                try:
                    result = getattr(self.inner, operation)(*args, **kwargs)
                except Exception as exc:
                    call.failed(span, exc)
                    raise
                call.succeeded(span, result)
                return result

        method = sync_method
    method.__name__ = method.__qualname__ = operation
    return method


class TracedGateway:
    """Base of the generated wrappers. ``inner`` is the undecorated gateway."""

    def __init__(self, inner: Gateway, instruments: _Instruments) -> None:
        self._inner = inner
        self._instruments = instruments

    @property
    def inner(self) -> Gateway:
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._inner.capabilities

    def __getattr__(self, attr: str) -> Any:  # non-operation extras, e.g. aclose()
        return getattr(self._inner, attr)

    def __repr__(self) -> str:
        return f"traced({self._inner!r})"


def traced(
    gateway: Gateway,
    *,
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
) -> Gateway:
    """Wrap ``gateway`` so each operation it implements is traced and measured.

    The wrapper defines exactly the operations the gateway has (as real class
    attributes), so ``isinstance(wrapped, Refunder)`` answers like the original
    and the registry's capability checks keep working.
    """
    if isinstance(gateway, TracedGateway):
        return gateway
    namespace: Mapping[str, object] = {
        op: _wrap(op, inspect.iscoroutinefunction(impl))
        for op in OPERATIONS
        if callable(impl := inspect.getattr_static(gateway, op, None))
    }
    cls = type(f"Traced{type(gateway).__name__}", (TracedGateway,), dict(namespace))
    wrapper: Gateway = cls(gateway, _Instruments(tracer_provider, meter_provider))
    return wrapper
