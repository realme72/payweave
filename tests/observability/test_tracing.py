import base64
import hashlib
import hmac
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
import respx
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from payweave.core.capabilities import INTERFACE_FOR, PaymentCreator, PaymentReader, Refunder
from payweave.core.errors import AuthenticationFailedError, InvalidSignatureError
from payweave.core.factory import GatewayFactory, GatewaySpec
from payweave.core.models import CaptureMode, CreatePaymentRequest, RefundRequest
from payweave.core.money import Money
from payweave.gateways.razorpay import RazorpayGateway, provider
from payweave.observability import OPERATIONS, TracedGateway, traced
from tests.gateways.razorpay import fixtures as fx

API = "https://api.razorpay.com/v1"
KEY_SECRET = "test_key_secret_DO_NOT_EXPORT"
WEBHOOK_SECRET = "whsec_DO_NOT_EXPORT"
IDEM = "idem_secretish_0123456789"
REFERENCE = "customer-jane@example.com"
METADATA_VALUE = "pii-value-DO-NOT-EXPORT"
CONFIG = {"key_id": fx.KEY_ID, "key_secret": KEY_SECRET, "webhook_secrets": [WEBHOOK_SECRET]}


@dataclass
class Telemetry:
    spans: InMemorySpanExporter
    metrics: InMemoryMetricReader
    tracer_provider: TracerProvider
    meter_provider: MeterProvider

    def finished(self) -> dict[str, ReadableSpan]:
        return {s.name: s for s in self.spans.get_finished_spans()}

    def counts(self) -> dict[tuple[str, str], int]:
        data = self.metrics.get_metrics_data()
        assert data is not None
        out: dict[tuple[str, str], int] = {}
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name != "payweave.gateway.operations":
                        continue
                    for point in metric.data.data_points:
                        attrs = point.attributes or {}
                        key = (str(attrs["payweave.operation"]), str(attrs["payweave.outcome"]))
                        out[key] = int(point.value)  # type: ignore[union-attr]
        return out


@pytest.fixture
def telemetry() -> Telemetry:
    spans = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    reader = InMemoryMetricReader()
    return Telemetry(spans, reader, tracer_provider, MeterProvider(metric_readers=[reader]))


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def raw() -> AsyncIterator[RazorpayGateway]:
    gw = GatewayFactory([provider]).create("razorpay", CONFIG)
    assert isinstance(gw, RazorpayGateway)
    yield gw
    await gw.aclose()


@pytest.fixture
def gateway(raw: RazorpayGateway, telemetry: Telemetry) -> TracedGateway:
    wrapped = traced(
        raw,
        tracer_provider=telemetry.tracer_provider,
        meter_provider=telemetry.meter_provider,
    )
    assert isinstance(wrapped, TracedGateway)
    return wrapped


def create_request() -> CreatePaymentRequest:
    return CreatePaymentRequest(
        amount=Money.of(50000, "INR"),
        merchant_reference=REFERENCE,
        idempotency_key=IDEM,
        capture_mode=CaptureMode.MANUAL,
        metadata={"customer": METADATA_VALUE},
    )


def signed_webhook() -> tuple[bytes, dict[str, str]]:
    body = json.dumps(fx.webhook("payment.captured", payment=fx.payment("captured"))).encode()
    sig = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return body, {"X-Razorpay-Signature": sig, "X-Razorpay-Event-Id": "evt_1"}


def checkout_params() -> dict[str, str]:
    msg = f"{fx.ORDER_ID}|pay_L0nSsccovt6zyp".encode()
    return {
        "razorpay_order_id": fx.ORDER_ID,
        "razorpay_payment_id": "pay_L0nSsccovt6zyp",
        "razorpay_signature": hmac.new(KEY_SECRET.encode(), msg, hashlib.sha256).hexdigest(),
    }


# --- the wrapper is a faithful stand-in -----------------------------------------


def test_wrapper_implements_exactly_what_the_gateway_does(
    raw: RazorpayGateway, gateway: TracedGateway
) -> None:
    for interface in INTERFACE_FOR.values():
        assert isinstance(gateway, interface) == isinstance(raw, interface), interface
    assert (gateway.name, gateway.capabilities, gateway.inner) == (raw.name, raw.capabilities, raw)
    assert traced(gateway) is gateway  # never double-wrapped
    assert set(OPERATIONS) == {
        "create_payment", "capture", "void", "refund", "get_payment", "get_refund",
        "verify_and_parse", "verify_checkout_return",
    }  # fmt: skip
    assert not hasattr(type(gateway), "void")  # Razorpay has no void API


def test_build_registry_traces_by_default() -> None:
    spec = GatewaySpec(name="razorpay", type="razorpay", config=CONFIG)
    factory = GatewayFactory([provider])
    traced_gw = factory.build_registry([spec]).get("razorpay", Refunder)
    assert isinstance(traced_gw, TracedGateway)
    plain_gw = factory.build_registry([spec], decorate=None).get("razorpay", Refunder)
    assert isinstance(plain_gw, RazorpayGateway)


# --- spans and metrics -------------------------------------------------------------


async def test_success_span_and_metrics(
    router: respx.MockRouter, gateway: TracedGateway, telemetry: Telemetry
) -> None:
    router.post("/orders").respond(json=fx.order())
    assert isinstance(gateway, PaymentCreator)
    await gateway.create_payment(create_request())

    span = telemetry.finished()["payweave.create_payment"]
    assert span.status.status_code is StatusCode.UNSET
    assert dict(span.attributes or {}) == {
        "payweave.gateway": "razorpay",
        "payweave.operation": "create_payment",
        "payweave.amount.minor": 50000,
        "payweave.currency": "INR",
        "payweave.capture_mode": "manual",
        "payweave.gateway_intent_id": fx.ORDER_ID,
        "payweave.intent.status": "requires_action",
        "payweave.intent.attempts": 0,
    }
    assert telemetry.counts() == {("create_payment", "ok"): 1}


async def test_error_span_records_classification_not_message(
    router: respx.MockRouter, gateway: TracedGateway, telemetry: Telemetry
) -> None:
    router.get(f"/orders/{fx.ORDER_ID}").respond(
        401, json={"error": {"code": "BAD_REQUEST_ERROR", "description": "Authentication failed"}}
    )
    router.get(f"/orders/{fx.ORDER_ID}/payments").respond(json=fx.payments())
    assert isinstance(gateway, PaymentReader)
    with pytest.raises(AuthenticationFailedError):
        await gateway.get_payment(fx.ORDER_ID)

    span = telemetry.finished()["payweave.get_payment"]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "AuthenticationFailedError"
    assert span.events == ()  # no exception event: message and stack stay out
    attrs = dict(span.attributes or {})
    assert attrs["error.type"] == "AuthenticationFailedError"
    assert attrs["payweave.error.retryable"] is False
    assert telemetry.counts() == {("get_payment", "error"): 1}


def test_sync_operations_are_traced(gateway: TracedGateway, telemetry: Telemetry) -> None:
    body, headers = signed_webhook()
    event = gateway.verify_and_parse(body, headers)
    assert event.event_id == "evt_1"
    with pytest.raises(InvalidSignatureError):
        gateway.verify_checkout_return({})

    spans = telemetry.finished()
    webhook = dict(spans["payweave.verify_and_parse"].attributes or {})
    assert webhook["payweave.webhook.event_id"] == "evt_1"
    assert webhook["payweave.webhook.gateway_event_type"] == "payment.captured"
    assert spans["payweave.verify_checkout_return"].status.status_code is StatusCode.ERROR


# --- the security property: nothing sensitive is exported ---------------------------


async def test_no_secret_or_pii_reaches_telemetry(
    router: respx.MockRouter, raw: RazorpayGateway, telemetry: Telemetry
) -> None:
    """Every operation, success and failure, with HTTP spans on: dump everything
    exported and look for anything that must never leave the process."""
    HTTPXClientInstrumentor().instrument(tracer_provider=telemetry.tracer_provider)
    try:
        gw = traced(
            raw,
            tracer_provider=telemetry.tracer_provider,
            meter_provider=telemetry.meter_provider,
        )
        router.post("/orders").respond(json=fx.order())
        router.get(f"/orders/{fx.ORDER_ID}").respond(json=fx.order("paid"))
        router.get(f"/orders/{fx.ORDER_ID}/payments").respond(
            json=fx.payments(fx.payment("captured"))
        )
        router.post("/payments/pay_L0nSsccovt6zyp/refund").respond(
            400, json={"error": {"code": "BAD_REQUEST_ERROR", "description": REFERENCE}}
        )
        assert isinstance(gw, PaymentCreator)
        assert isinstance(gw, Refunder)
        await gw.create_payment(create_request())
        await gw.get_payment(fx.ORDER_ID)  # type: ignore[attr-defined]
        with pytest.raises(Exception):  # noqa: B017, PT011 - any error; we inspect telemetry
            await gw.refund(
                RefundRequest(
                    gateway_intent_id=fx.ORDER_ID,
                    idempotency_key=IDEM,
                    amount=Money.of(100, "INR"),
                )
            )
        body, headers = signed_webhook()
        gw.verify_and_parse(body, headers)  # type: ignore[attr-defined]
        params = checkout_params()
        gw.verify_checkout_return(params)  # type: ignore[attr-defined]
    finally:
        HTTPXClientInstrumentor().uninstrument()

    spans = telemetry.spans.get_finished_spans()
    assert any(s.name.startswith("POST") for s in spans), "HTTP spans should be present"
    metrics = telemetry.metrics.get_metrics_data()
    assert metrics is not None
    exported = "\n".join(s.to_json() for s in spans) + metrics.to_json()

    basic_auth = base64.b64encode(f"{fx.KEY_ID}:{KEY_SECRET}".encode()).decode()
    forbidden = {
        "key secret": KEY_SECRET,
        "webhook secret": WEBHOOK_SECRET,
        "basic auth header": basic_auth,
        "webhook signature": headers["X-Razorpay-Signature"],
        "checkout signature": params["razorpay_signature"],
        "idempotency key": IDEM,
        "merchant reference / gateway error message": REFERENCE,
        "metadata value": METADATA_VALUE,
    }
    leaked = [label for label, value in forbidden.items() if value in exported]
    assert leaked == []
