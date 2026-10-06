import base64
import hashlib
import hmac
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import respx

from payweave.core.capabilities import Capability
from payweave.core.errors import (
    AuthenticationFailedError,
    DuplicateRequestError,
    GatewayTimeoutError,
    GatewayUnavailableError,
    InvalidAmountError,
    InvalidRequestError,
    InvalidSignatureError,
    RateLimitedError,
    UnknownGatewayError,
    UnsupportedCapabilityError,
)
from payweave.core.factory import GatewayFactory, InvalidGatewayConfigError
from payweave.core.models import CaptureMode, CaptureRequest, CreatePaymentRequest, RefundRequest
from payweave.core.money import Money
from payweave.core.status import IntentStatus, RefundStatus
from payweave.gateways.razorpay import RazorpayGateway, provider
from tests.gateways.razorpay import fixtures as fx

API = "https://api.razorpay.com/v1"
KEY_SECRET = "test_key_secret"
WEBHOOK_SECRET = "whsec_current"
OLD_WEBHOOK_SECRET = "whsec_old"
IDEM = "idem_0123456789abcdef"


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def gateway() -> AsyncIterator[RazorpayGateway]:
    factory = GatewayFactory([provider])
    gw = factory.create(
        "razorpay",
        {
            "key_id": fx.KEY_ID,
            "key_secret": KEY_SECRET,
            "webhook_secrets": [WEBHOOK_SECRET, OLD_WEBHOOK_SECRET],
        },
    )
    assert isinstance(gw, RazorpayGateway)
    yield gw
    await gw.aclose()


def inr(n: int) -> Money:
    return Money.of(n, "INR")


def stub_intent(router: respx.MockRouter, order: dict, *items: dict) -> None:  # type: ignore[type-arg]
    router.get(f"/orders/{fx.ORDER_ID}").respond(json=order)
    router.get(f"/orders/{fx.ORDER_ID}/payments").respond(json=fx.payments(*items))


# --- create -------------------------------------------------------------------


async def test_create_payment_sends_order(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    route = router.post("/orders").respond(json=fx.order())
    req = CreatePaymentRequest(
        amount=inr(50000), merchant_reference="Receipt #20", idempotency_key=IDEM,
        capture_mode=CaptureMode.MANUAL, metadata={"cart": "42"},
    )  # fmt: skip
    result = await gateway.create_payment(req)

    assert result.status is IntentStatus.REQUIRES_ACTION
    sent = json.loads(route.calls.last.request.content)
    assert sent["amount"] == 50000
    assert sent["currency"] == "INR"
    assert sent["receipt"] == "Receipt #20"
    assert sent["notes"] == {"cart": "42", "pw_idempotency_key": IDEM, "pw_capture_mode": "manual"}
    assert sent["payment"]["capture"] == "manual"
    auth = route.calls.last.request.headers["authorization"]
    assert auth == "Basic " + base64.b64encode(f"{fx.KEY_ID}:{KEY_SECRET}".encode()).decode()


async def test_three_decimal_currency_must_end_in_zero(gateway: RazorpayGateway) -> None:
    req = CreatePaymentRequest(
        amount=Money.of(99991, "KWD"), merchant_reference="r", idempotency_key=IDEM
    )
    with pytest.raises(InvalidAmountError):
        await gateway.create_payment(req)


async def test_reserved_metadata_keys_rejected(gateway: RazorpayGateway) -> None:
    req = CreatePaymentRequest(
        amount=inr(100),
        merchant_reference="r",
        idempotency_key=IDEM,
        metadata={"pw_capture_mode": "x"},
    )
    with pytest.raises(InvalidRequestError):
        await gateway.create_payment(req)


# --- capture ------------------------------------------------------------------


async def test_capture_authorized_attempt(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    capture = router.post("/payments/pay_L0nSsccovt6zyp/capture").respond(
        json=fx.payment("captured")
    )
    router.get(f"/orders/{fx.ORDER_ID}").mock(
        side_effect=[
            httpx.Response(200, json=fx.order("attempted")),
            httpx.Response(200, json=fx.order("paid")),
        ]
    )
    router.get(f"/orders/{fx.ORDER_ID}/payments").mock(
        side_effect=[
            httpx.Response(200, json=fx.payments(fx.payment("authorized"))),
            httpx.Response(200, json=fx.payments(fx.payment("captured"))),
        ]
    )
    result = await gateway.capture(
        CaptureRequest(gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM)
    )
    assert result.status is IntentStatus.CAPTURED
    assert json.loads(capture.calls.last.request.content) == {"amount": 50000, "currency": "INR"}


async def test_capture_is_idempotent_when_already_captured(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    stub_intent(router, fx.order("paid"), fx.payment("captured"))
    capture = router.post("/payments/pay_L0nSsccovt6zyp/capture")
    result = await gateway.capture(
        CaptureRequest(gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM)
    )
    assert result.status is IntentStatus.CAPTURED
    assert not capture.called


async def test_partial_capture_unsupported(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    stub_intent(router, fx.order("attempted"), fx.payment("authorized"))
    req = CaptureRequest(gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM, amount=inr(100))
    with pytest.raises(UnsupportedCapabilityError):
        await gateway.capture(req)


async def test_capture_without_authorized_attempt(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    stub_intent(router, fx.order("attempted"), fx.payment("failed"))
    with pytest.raises(InvalidRequestError):
        await gateway.capture(CaptureRequest(gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM))


# --- refund -------------------------------------------------------------------


async def test_partial_refund_uses_native_idempotency(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    stub_intent(router, fx.order("paid"), fx.payment("captured"))
    route = router.post("/payments/pay_L0nSsccovt6zyp/refund").respond(json=fx.refund("pending"))
    result = await gateway.refund(
        RefundRequest(
            gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM, amount=inr(10000), reason="damaged"
        )
    )
    assert result.status is RefundStatus.PENDING
    request = route.calls.last.request
    assert request.headers["x-refund-idempotency"] == IDEM
    assert json.loads(request.content) == {"amount": 10000, "notes": {"reason": "damaged"}}


async def test_full_refund_defaults_to_remaining(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    stub_intent(router, fx.order("paid"), fx.payment("captured", refunded=10000))
    route = router.post("/payments/pay_L0nSsccovt6zyp/refund").respond(json=fx.refund(amount=40000))
    await gateway.refund(RefundRequest(gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM))
    assert json.loads(route.calls.last.request.content) == {"amount": 40000}


@pytest.mark.parametrize("amount", [Money.of(40001, "INR"), Money.of(100, "USD")])
async def test_refund_over_refundable_or_wrong_currency(
    router: respx.MockRouter, gateway: RazorpayGateway, amount: Money
) -> None:
    stub_intent(router, fx.order("paid"), fx.payment("captured", refunded=10000))
    with pytest.raises(InvalidAmountError):
        await gateway.refund(
            RefundRequest(gateway_intent_id=fx.ORDER_ID, idempotency_key=IDEM, amount=amount)
        )


# --- HTTP error mapping -------------------------------------------------------


def razorpay_error(description: str, reason: str = "input_validation_failed") -> dict:  # type: ignore[type-arg]
    return {"error": {"code": "BAD_REQUEST_ERROR", "description": description, "reason": reason}}


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            httpx.Response(401, json=razorpay_error("Authentication failed")),
            AuthenticationFailedError,
        ),
        (httpx.Response(429, json=razorpay_error("Too many requests")), RateLimitedError),
        (
            httpx.Response(400, json=razorpay_error("The id provided does not exist")),
            InvalidRequestError,
        ),
        (
            httpx.Response(
                400,
                json=razorpay_error(
                    "Different request with the same idempotency key has already been processed."
                ),
            ),
            DuplicateRequestError,
        ),
        (httpx.Response(500, json={}), UnknownGatewayError),  # POST: outcome unknown
        (httpx.Response(200, text="<html>"), UnknownGatewayError),
    ],
)
async def test_post_error_mapping(
    router: respx.MockRouter,
    gateway: RazorpayGateway,
    response: httpx.Response,
    error: type[Exception],
) -> None:
    router.post("/orders").mock(return_value=response)
    req = CreatePaymentRequest(amount=inr(100), merchant_reference="r", idempotency_key=IDEM)
    with pytest.raises(error):
        await gateway.create_payment(req)


async def test_post_timeout_is_outcome_unknown(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    router.post("/orders").mock(side_effect=httpx.ReadTimeout("slow"))
    req = CreatePaymentRequest(amount=inr(100), merchant_reference="r", idempotency_key=IDEM)
    with pytest.raises(GatewayTimeoutError) as exc:
        await gateway.create_payment(req)
    assert exc.value.outcome_unknown


async def test_connect_failure_and_get_5xx_are_safe_to_retry(
    router: respx.MockRouter, gateway: RazorpayGateway
) -> None:
    router.post("/orders").mock(side_effect=httpx.ConnectError("down"))
    req = CreatePaymentRequest(amount=inr(100), merchant_reference="r", idempotency_key=IDEM)
    with pytest.raises(GatewayUnavailableError):
        await gateway.create_payment(req)
    router.get(f"/orders/{fx.ORDER_ID}").respond(503)
    router.get(f"/orders/{fx.ORDER_ID}/payments").respond(503)
    with pytest.raises(GatewayUnavailableError) as exc:
        await gateway.get_payment(fx.ORDER_ID)
    assert not exc.value.outcome_unknown


@pytest.mark.parametrize(
    "bad_id", ["order_../../payments", "pay_123", "order_", "order_abc/x", "order_ąbc"]
)
async def test_ids_are_validated_before_use_in_urls(gateway: RazorpayGateway, bad_id: str) -> None:
    with pytest.raises(InvalidRequestError):
        await gateway.get_payment(bad_id)


# --- webhooks -----------------------------------------------------------------


def signed(body: bytes, secret: str = WEBHOOK_SECRET, **headers: str) -> dict[str, str]:
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Razorpay-Signature": sig, "X-Razorpay-Event-Id": "evt_1", **headers}


def test_webhook_verified_then_parsed(gateway: RazorpayGateway) -> None:
    body = json.dumps(fx.webhook("payment.captured", payment=fx.payment("captured"))).encode()
    event = gateway.verify_and_parse(body, signed(body))
    assert (event.event_id, event.gateway_intent_id) == ("evt_1", fx.ORDER_ID)


def test_webhook_headers_are_case_insensitive(gateway: RazorpayGateway) -> None:
    body = json.dumps(fx.webhook("payment.captured", payment=fx.payment("captured"))).encode()
    lowered = {k.lower(): v for k, v in signed(body).items()}
    assert gateway.verify_and_parse(body, lowered).gateway_intent_id == fx.ORDER_ID


def test_webhook_old_secret_accepted_during_rotation(gateway: RazorpayGateway) -> None:
    body = json.dumps(fx.webhook("payment.failed", payment=fx.payment("failed"))).encode()
    assert gateway.verify_and_parse(body, signed(body, OLD_WEBHOOK_SECRET)).event_id == "evt_1"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda b, h: (b + b" ", h),  # body changed after signing
        lambda b, h: (b, {**h, "X-Razorpay-Signature": "0" * 64}),
        lambda b, h: (b, {k: v for k, v in h.items() if k != "X-Razorpay-Signature"}),
        lambda b, h: (b, signed(b, "attacker_secret")),
    ],
)
def test_webhook_tampering_rejected(gateway: RazorpayGateway, tamper) -> None:  # type: ignore[no-untyped-def]
    body = json.dumps(fx.webhook("payment.captured", payment=fx.payment("captured"))).encode()
    bad_body, bad_headers = tamper(body, signed(body))
    with pytest.raises(InvalidSignatureError):
        gateway.verify_and_parse(bad_body, bad_headers)


def test_webhook_signature_checked_before_parsing(gateway: RazorpayGateway) -> None:
    with pytest.raises(InvalidSignatureError):  # not a JSON error: we never parsed it
        gateway.verify_and_parse(
            b"not json", {"X-Razorpay-Signature": "00", "X-Razorpay-Event-Id": "e"}
        )


def test_webhooks_disabled_without_secret() -> None:
    gw = GatewayFactory([provider]).create(
        "razorpay", {"key_id": fx.KEY_ID, "key_secret": KEY_SECRET}
    )
    assert Capability.WEBHOOKS not in gw.capabilities


# --- checkout return ----------------------------------------------------------


def test_checkout_return_signature(gateway: RazorpayGateway) -> None:
    msg = f"{fx.ORDER_ID}|pay_L0nSsccovt6zyp".encode()
    sig = hmac.new(KEY_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    params = {
        "razorpay_order_id": fx.ORDER_ID,
        "razorpay_payment_id": "pay_L0nSsccovt6zyp",
        "razorpay_signature": sig,
    }
    assert gateway.verify_checkout_return(params) == fx.ORDER_ID
    with pytest.raises(InvalidSignatureError):
        gateway.verify_checkout_return({**params, "razorpay_payment_id": "pay_OTHER"})
    with pytest.raises(InvalidSignatureError):
        gateway.verify_checkout_return({})


# --- config safety --------------------------------------------------------------


def test_live_keys_refused_unless_allowed() -> None:
    factory = GatewayFactory([provider])
    with pytest.raises(InvalidGatewayConfigError):
        factory.create("razorpay", {"key_id": "rzp_live_abc", "key_secret": "s"})
    gw = factory.create(
        "razorpay", {"key_id": "rzp_live_abc", "key_secret": "s", "allow_live": True}
    )
    assert gw.name == "razorpay"


def test_secrets_not_in_repr(gateway: RazorpayGateway) -> None:
    assert KEY_SECRET not in repr(gateway._config)
