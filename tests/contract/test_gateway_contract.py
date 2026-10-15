"""The gateway contract: behaviour every gateway must show through PayWeave's API.

Parameterised by harness. A new gateway (hand-written or AI-generated) joins by
adding a harness to ``HARNESSES``; it has to pass this file unchanged. Tests for
optional features are skipped when the gateway doesn't declare the capability,
and *declaring* a capability without honouring it fails here.

Gates the AI pipeline relies on (PLAN.md §7): this suite is never written by
the AI and is never relaxed to make a gateway pass.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pytest
import respx

from payweave.core.capabilities import (
    Capability,
    Capturer,
    CheckoutReturnVerifier,
    PaymentCreator,
    PaymentReader,
    Refunder,
    RefundReader,
    Voider,
    WebhookHandler,
)
from payweave.core.errors import (
    DuplicateRequestError,
    GatewayError,
    InvalidAmountError,
    InvalidRequestError,
    InvalidSignatureError,
    UnsupportedCapabilityError,
)
from payweave.core.models import (
    CaptureMode,
    CaptureRequest,
    CreatePaymentRequest,
    PaymentIntent,
    Refund,
    RefundRequest,
    VoidRequest,
)
from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus, RefundStatus, can_advance
from payweave.packages.runtime import PackageProvider, load_package
from payweave.testing import GatewayHarness
from payweave.testing.razorpay import RazorpayHarness
from payweave.verify.references import REFERENCES
from payweave.verify.report import UNDER_TEST_ENV
from tests.contract.harnesses import RAZORPAY_PACKAGE, FakeHarness

AMOUNT = Money.of(50000, "INR")


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock:
        yield mock


HARNESSES: dict[str, Callable[[respx.MockRouter], GatewayHarness]] = {
    "fake": lambda _router: FakeHarness(),
    "razorpay": RazorpayHarness,
    # The same simulator, through the Razorpay Gateway Package instead.
    "razorpay_package": lambda router: RazorpayHarness(
        router, PackageProvider(load_package(RAZORPAY_PACKAGE), allow_unsigned=True)
    ),
}

# Gate G5 for any package: `python -m payweave.verify <dir>` sets this and runs `-k under_test`.
if _under_test := os.environ.get(UNDER_TEST_ENV):
    _package = load_package(Path(_under_test))
    _reference = REFERENCES[_package.manifest.gateway]
    HARNESSES["under_test"] = lambda router: _reference.harness(
        router, PackageProvider(_package, allow_unsigned=True), None
    )


@pytest.fixture(params=list(HARNESSES))
async def h(
    request: pytest.FixtureRequest, router: respx.MockRouter
) -> AsyncIterator[GatewayHarness]:
    harness = HARNESSES[request.param](router)
    yield harness
    if isinstance(harness, RazorpayHarness):
        await harness.aclose()


# --- helpers ------------------------------------------------------------------------


def key() -> str:
    return f"contract_{secrets.token_hex(8)}"


def need(h: GatewayHarness, *caps: Capability) -> None:
    missing = [c for c in caps if c not in h.gateway.capabilities]
    if missing:
        pytest.skip(f"{h.gateway.name} does not declare {', '.join(missing)}")


def op[T](h: GatewayHarness, interface: type[T]) -> T:
    assert isinstance(h.gateway, interface), f"declared but not implemented: {interface}"
    return h.gateway


async def create(h: GatewayHarness, mode: CaptureMode = CaptureMode.MANUAL) -> PaymentIntent:
    return await op(h, PaymentCreator).create_payment(
        CreatePaymentRequest(
            amount=AMOUNT,
            merchant_reference="contract-order-1",
            idempotency_key=key(),
            capture_mode=mode,
            metadata={"cart": "42"},
        )
    )


async def fetch(h: GatewayHarness, intent_id: str) -> PaymentIntent:
    return await op(h, PaymentReader).get_payment(intent_id)


async def captured_intent(h: GatewayHarness) -> PaymentIntent:
    intent = await create(h, CaptureMode.AUTOMATIC)
    await h.customer_pays(intent.gateway_intent_id)
    return await fetch(h, intent.gateway_intent_id)


async def refund(
    h: GatewayHarness, intent_id: str, paise: int | None, k: str | None = None
) -> Refund:
    return await op(h, Refunder).refund(
        RefundRequest(
            gateway_intent_id=intent_id,
            idempotency_key=k or key(),
            amount=None if paise is None else Money.of(paise, "INR"),
        )
    )


# --- creating and reading -------------------------------------------------------------


async def test_capabilities_are_implemented(h: GatewayHarness) -> None:
    from payweave.core.registry import validate_gateway  # noqa: PLC0415

    validate_gateway(h.gateway)


async def test_create_returns_an_intent_awaiting_the_customer(h: GatewayHarness) -> None:
    intent = await create(h)
    assert intent.gateway == h.gateway.name
    assert intent.status is IntentStatus.REQUIRES_ACTION
    assert intent.next_action is not None
    assert intent.amount == AMOUNT
    assert intent.amount_captured.is_zero
    assert intent.amount_refunded.is_zero
    assert intent.attempts == ()
    assert intent.merchant_reference == "contract-order-1"
    assert intent.metadata == {"cart": "42"}
    assert intent.capture_mode in (CaptureMode.MANUAL, None)


async def test_get_payment_round_trips(h: GatewayHarness) -> None:
    created = await create(h)
    fetched = await fetch(h, created.gateway_intent_id)
    assert fetched.gateway_intent_id == created.gateway_intent_id
    assert (fetched.status, fetched.amount, fetched.metadata) == (
        created.status,
        created.amount,
        created.metadata,
    )


async def test_unknown_intent_is_an_invalid_request(h: GatewayHarness) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        await fetch(h, h.unknown_intent_id)
    assert exc.value.gateway == h.gateway.name
    assert not exc.value.retryable


# --- customer outcomes ---------------------------------------------------------------


async def test_automatic_capture(h: GatewayHarness) -> None:
    intent = await captured_intent(h)
    assert intent.status is IntentStatus.CAPTURED
    assert intent.amount_captured == AMOUNT
    assert intent.refundable == AMOUNT
    assert [a.status for a in intent.attempts] == [AttemptStatus.CAPTURED]


async def test_failed_attempt_keeps_the_intent_payable(h: GatewayHarness) -> None:
    intent = await create(h)
    await h.customer_pays(intent.gateway_intent_id, succeed=False)
    intent = await fetch(h, intent.gateway_intent_id)
    assert intent.status is IntentStatus.REQUIRES_ACTION
    assert intent.next_action is not None
    [attempt] = intent.attempts
    assert attempt.status is AttemptStatus.FAILED
    assert attempt.failure is not None


async def test_retry_after_failure_on_the_same_intent(h: GatewayHarness) -> None:
    need(h, Capability.MANUAL_CAPTURE)
    intent = await create(h)
    await h.customer_pays(intent.gateway_intent_id, succeed=False)
    await h.customer_pays(intent.gateway_intent_id)
    intent = await fetch(h, intent.gateway_intent_id)
    assert intent.status is IntentStatus.AUTHORIZED
    assert [a.status for a in intent.attempts] == [AttemptStatus.FAILED, AttemptStatus.AUTHORIZED]


# --- capture and void ---------------------------------------------------------------


async def test_manual_capture_flow_is_idempotent(h: GatewayHarness) -> None:
    need(h, Capability.MANUAL_CAPTURE, Capability.CAPTURE)
    intent = await create(h)
    await h.customer_pays(intent.gateway_intent_id)
    authorized = await fetch(h, intent.gateway_intent_id)
    assert authorized.status is IntentStatus.AUTHORIZED
    assert authorized.amount_captured.is_zero

    capture = CaptureRequest(gateway_intent_id=intent.gateway_intent_id, idempotency_key=key())
    captured = await op(h, Capturer).capture(capture)
    assert captured.status is IntentStatus.CAPTURED
    assert captured.amount_captured == AMOUNT
    assert can_advance(authorized.status, captured.status)

    again = await op(h, Capturer).capture(capture)  # a retry after a timeout, say
    assert (again.status, again.amount_captured) == (captured.status, captured.amount_captured)


async def test_capture_without_authorization_is_rejected(h: GatewayHarness) -> None:
    need(h, Capability.CAPTURE)
    intent = await create(h)
    with pytest.raises(InvalidRequestError):
        await op(h, Capturer).capture(
            CaptureRequest(gateway_intent_id=intent.gateway_intent_id, idempotency_key=key())
        )
    assert (await fetch(h, intent.gateway_intent_id)).status is IntentStatus.REQUIRES_ACTION


async def test_partial_capture_only_if_declared(h: GatewayHarness) -> None:
    need(h, Capability.MANUAL_CAPTURE, Capability.CAPTURE)
    if Capability.PARTIAL_CAPTURE in h.gateway.capabilities:
        pytest.skip("partial capture supported; covered by gateway-specific tests")
    intent = await create(h)
    await h.customer_pays(intent.gateway_intent_id)
    with pytest.raises(UnsupportedCapabilityError):
        await op(h, Capturer).capture(
            CaptureRequest(
                gateway_intent_id=intent.gateway_intent_id,
                idempotency_key=key(),
                amount=Money.of(100, "INR"),
            )
        )
    assert (await fetch(h, intent.gateway_intent_id)).status is IntentStatus.AUTHORIZED


async def test_void_releases_an_authorization(h: GatewayHarness) -> None:
    need(h, Capability.MANUAL_CAPTURE, Capability.VOID)
    intent = await create(h)
    await h.customer_pays(intent.gateway_intent_id)
    voided = await op(h, Voider).void(
        VoidRequest(gateway_intent_id=intent.gateway_intent_id, idempotency_key=key())
    )
    assert voided.status is IntentStatus.VOIDED
    assert voided.amount_captured.is_zero


# --- refunds -------------------------------------------------------------------------


async def test_partial_then_full_refund(h: GatewayHarness) -> None:
    need(h, Capability.REFUND, Capability.PARTIAL_REFUND, Capability.MULTIPLE_REFUNDS)
    intent = await captured_intent(h)
    first = await refund(h, intent.gateway_intent_id, 10000)
    assert first.amount == Money.of(10000, "INR")
    assert first.status in (RefundStatus.PENDING, RefundStatus.SUCCEEDED)

    after_first = await fetch(h, intent.gateway_intent_id)
    assert after_first.status is IntentStatus.PARTIALLY_REFUNDED
    assert after_first.refundable == Money.of(40000, "INR")

    rest = await refund(h, intent.gateway_intent_id, None)  # None = everything left
    assert rest.amount == Money.of(40000, "INR")
    done = await fetch(h, intent.gateway_intent_id)
    assert done.status is IntentStatus.REFUNDED
    assert done.refundable.is_zero
    assert can_advance(after_first.status, done.status)


async def test_refund_more_than_refundable_is_blocked(h: GatewayHarness) -> None:
    need(h, Capability.REFUND)
    intent = await captured_intent(h)
    with pytest.raises(InvalidAmountError):
        await refund(h, intent.gateway_intent_id, AMOUNT.amount_minor + 100)
    with pytest.raises(InvalidAmountError):
        await op(h, Refunder).refund(
            RefundRequest(
                gateway_intent_id=intent.gateway_intent_id,
                idempotency_key=key(),
                amount=Money.of(100, "USD"),
            )
        )
    assert (await fetch(h, intent.gateway_intent_id)).amount_refunded.is_zero


async def test_refund_before_capture_is_rejected(h: GatewayHarness) -> None:
    need(h, Capability.REFUND, Capability.MANUAL_CAPTURE)
    intent = await create(h)
    await h.customer_pays(intent.gateway_intent_id)  # authorized, not captured
    with pytest.raises(InvalidRequestError):
        await refund(h, intent.gateway_intent_id, 10000)


async def test_refund_idempotency(h: GatewayHarness) -> None:
    need(h, Capability.REFUND, Capability.PARTIAL_REFUND, Capability.NATIVE_IDEMPOTENT_REFUND)
    intent = await captured_intent(h)
    k = key()
    first = await refund(h, intent.gateway_intent_id, 10000, k)
    replay = await refund(h, intent.gateway_intent_id, 10000, k)
    assert replay.gateway_refund_id == first.gateway_refund_id
    with pytest.raises(DuplicateRequestError):
        await refund(h, intent.gateway_intent_id, 20000, k)
    assert (await fetch(h, intent.gateway_intent_id)).amount_refunded == Money.of(10000, "INR")


async def test_get_refund_round_trips(h: GatewayHarness) -> None:
    need(h, Capability.REFUND, Capability.GET_REFUND, Capability.PARTIAL_REFUND)
    intent = await captured_intent(h)
    created = await refund(h, intent.gateway_intent_id, 10000)
    reader = op(h, RefundReader)
    fetched = await reader.get_refund(
        created.gateway_refund_id, gateway_intent_id=intent.gateway_intent_id
    )
    assert (fetched.gateway_refund_id, fetched.amount, fetched.gateway_attempt_id) == (
        created.gateway_refund_id,
        created.amount,
        created.gateway_attempt_id,
    )
    with pytest.raises(InvalidRequestError):
        await reader.get_refund(h.unknown_refund_id, gateway_intent_id=intent.gateway_intent_id)


# --- signed inputs -----------------------------------------------------------------------


async def test_webhook_verified_and_tampering_rejected(h: GatewayHarness) -> None:
    need(h, Capability.WEBHOOKS)
    intent = await captured_intent(h)
    body, headers = h.signed_webhook(intent.gateway_intent_id)
    event = op(h, WebhookHandler).verify_and_parse(body, headers)
    assert event.gateway == h.gateway.name
    assert event.gateway_intent_id == intent.gateway_intent_id

    lowered = {k.lower(): v for k, v in headers.items()}
    assert op(h, WebhookHandler).verify_and_parse(body, lowered).event_id == event.event_id
    for bad_body, bad_headers in [
        (body + b" ", headers),
        (body[:10] + bytes([body[10] ^ 1]) + body[11:], headers),  # one bit flipped
        (body, {k: v for k, v in headers.items() if "signature" not in k.lower()}),
    ]:
        with pytest.raises(InvalidSignatureError):
            op(h, WebhookHandler).verify_and_parse(bad_body, bad_headers)


async def test_checkout_return_verified_and_tampering_rejected(h: GatewayHarness) -> None:
    need(h, Capability.CHECKOUT_RETURN)
    intent = await create(h)
    attempt_id = await h.customer_pays(intent.gateway_intent_id)
    params = h.checkout_return(intent.gateway_intent_id, attempt_id)
    verifier = op(h, CheckoutReturnVerifier)
    assert verifier.verify_checkout_return(params) == intent.gateway_intent_id

    other = await create(h)
    for name in params:  # change any one field -> rejected
        tampered = {**params, name: other.gateway_intent_id}
        with pytest.raises(InvalidSignatureError):
            verifier.verify_checkout_return(tampered)
    with pytest.raises(InvalidSignatureError):
        verifier.verify_checkout_return({})


# --- errors -----------------------------------------------------------------------------


async def test_gateway_errors_name_their_gateway(h: GatewayHarness) -> None:
    calls: list[Callable[[], Awaitable[object]]] = [lambda: fetch(h, h.unknown_intent_id)]
    if Capability.CAPTURE in h.gateway.capabilities:
        calls.append(
            lambda: op(h, Capturer).capture(
                CaptureRequest(gateway_intent_id=h.unknown_intent_id, idempotency_key=key())
            )
        )
    for call in calls:
        with pytest.raises(GatewayError) as exc:
            await call()
        assert exc.value.gateway == h.gateway.name
