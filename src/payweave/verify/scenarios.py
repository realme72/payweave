"""G7 scenarios: gateway-neutral flows, driven only through canonical interfaces.

A step that needs a capability the gateway doesn't declare is recorded as
skipped, so a capability difference between reference and candidate shows up
as a difference instead of hiding.
"""

from __future__ import annotations

from typing import Any

from payweave.core.capabilities import (
    Capability,
    Capturer,
    CheckoutReturnVerifier,
    PaymentCreator,
    PaymentReader,
    Refunder,
    RefundReader,
    WebhookHandler,
)
from payweave.core.models import CaptureMode, CaptureRequest, CreatePaymentRequest, RefundRequest
from payweave.core.money import Money
from payweave.testing import GatewayHarness
from payweave.verify.differential import Recorder, Scenario

AMOUNT = Money.of(50000, "INR")


def key(label: str) -> str:
    return f"g7-{label}".replace(" ", "-").ljust(16, "0")


def has(h: GatewayHarness, *caps: Capability) -> bool:
    return all(c in h.gateway.capabilities for c in caps)


async def create(
    r: Recorder,
    h: GatewayHarness,
    label: str,
    mode: CaptureMode = CaptureMode.AUTOMATIC,
    amount: Money = AMOUNT,
) -> Any:
    if not has(h, Capability.CREATE_PAYMENT):
        r.note(f"create {label}", "skipped: no create_payment")
        return None
    gw = h.gateway
    if not isinstance(gw, PaymentCreator):
        r.note(f"create {label}", "declared but not implemented: create_payment")
        return None
    return await r.step(
        f"create {label}",
        lambda: gw.create_payment(
            CreatePaymentRequest(
                amount=amount,
                merchant_reference=f"g7-{label}",
                idempotency_key=key(label),
                capture_mode=mode,
                metadata={"cart": "42"},
            )
        ),
    )


async def get(r: Recorder, h: GatewayHarness, intent_id: str, label: str = "get") -> Any:
    gw = h.gateway
    if not isinstance(gw, PaymentReader):
        r.note(label, "skipped: no get_payment")
        return None
    return await r.step(label, lambda: gw.get_payment(intent_id))


async def pay(r: Recorder, h: GatewayHarness, intent_id: str, *, succeed: bool = True) -> Any:
    label = "customer pays" if succeed else "customer fails"
    return await r.step(label, lambda: h.customer_pays(intent_id, succeed=succeed))


async def refund(
    r: Recorder, h: GatewayHarness, intent_id: str, label: str, *, paise: int | None, k: str
) -> Any:
    gw = h.gateway
    if not isinstance(gw, Refunder) or not has(h, Capability.REFUND):
        r.note(label, "skipped: no refund")
        return None
    amount = None if paise is None else Money.of(paise, AMOUNT.currency)
    return await r.step(
        label,
        lambda: gw.refund(
            RefundRequest(gateway_intent_id=intent_id, idempotency_key=key(k), amount=amount)
        ),
    )


async def capture(r: Recorder, h: GatewayHarness, intent_id: str, label: str) -> Any:
    gw = h.gateway
    if not isinstance(gw, Capturer) or not has(h, Capability.CAPTURE):
        r.note(label, "skipped: no capture")
        return None
    return await r.step(
        label,
        lambda: gw.capture(CaptureRequest(gateway_intent_id=intent_id, idempotency_key=key(label))),
    )


# --- scenarios ------------------------------------------------------------------------------


async def capabilities(r: Recorder, h: GatewayHarness) -> None:
    r.note("capabilities", h.gateway.capabilities)


async def create_and_read(r: Recorder, h: GatewayHarness) -> None:
    intent = await create(r, h, "read", CaptureMode.MANUAL)
    if intent:
        await get(r, h, intent.gateway_intent_id)


async def automatic_capture(r: Recorder, h: GatewayHarness) -> None:
    intent = await create(r, h, "auto")
    if intent:
        await pay(r, h, intent.gateway_intent_id)
        await get(r, h, intent.gateway_intent_id)


async def manual_capture(r: Recorder, h: GatewayHarness) -> None:
    intent = await create(r, h, "manual", CaptureMode.MANUAL)
    if intent:
        await pay(r, h, intent.gateway_intent_id)
        await get(r, h, intent.gateway_intent_id, "get authorized")
        await capture(r, h, intent.gateway_intent_id, "capture")
        await capture(r, h, intent.gateway_intent_id, "capture again")  # idempotent
        await get(r, h, intent.gateway_intent_id, "get captured")


async def failed_then_paid(r: Recorder, h: GatewayHarness) -> None:
    intent = await create(r, h, "retry")
    if intent:
        await pay(r, h, intent.gateway_intent_id, succeed=False)
        await get(r, h, intent.gateway_intent_id, "get after failure")
        await pay(r, h, intent.gateway_intent_id)
        await get(r, h, intent.gateway_intent_id, "get after success")


async def refunds(r: Recorder, h: GatewayHarness) -> None:
    intent = await create(r, h, "refunds")
    if not intent:
        return
    pid = intent.gateway_intent_id
    await pay(r, h, pid)
    first = await refund(r, h, pid, "partial refund", paise=10000, k="rf-a")
    await refund(r, h, pid, "same key, same body (replay)", paise=10000, k="rf-a")
    await refund(r, h, pid, "same key, different amount", paise=5000, k="rf-a")
    await refund(r, h, pid, "more than refundable", paise=10_000_000, k="rf-b")
    await refund(r, h, pid, "below gateway minimum", paise=50, k="rf-c")
    await refund(r, h, pid, "the rest", paise=None, k="rf-d")
    await refund(r, h, pid, "nothing left", paise=None, k="rf-e")
    await get(r, h, pid, "get refunded")
    gw = h.gateway
    if first is not None and isinstance(gw, RefundReader):
        await r.step("get refund", lambda: gw.get_refund(first.gateway_refund_id))


async def refund_before_capture(r: Recorder, h: GatewayHarness) -> None:
    intent = await create(r, h, "uncaptured", CaptureMode.MANUAL)
    if intent:
        await refund(r, h, intent.gateway_intent_id, "refund before paying", paise=None, k="rf-u1")
        await pay(r, h, intent.gateway_intent_id)
        await refund(
            r, h, intent.gateway_intent_id, "refund authorized only", paise=None, k="rf-u2"
        )
        await capture(r, h, intent.gateway_intent_id, "capture nothing twice")


async def bad_inputs(r: Recorder, h: GatewayHarness) -> None:
    gw = h.gateway
    await create(r, h, "tiny", amount=Money.of(50, "INR"))
    await create(r, h, "dollars", amount=Money.of(500, "USD"))
    await get(r, h, h.unknown_intent_id, "unknown intent")
    await get(r, h, "order_../../refunds", "path traversal id")
    await get(r, h, "not an id", "malformed id")
    if isinstance(gw, RefundReader):
        await r.step("unknown refund", lambda: gw.get_refund(h.unknown_refund_id))
        await r.step("malformed refund id", lambda: gw.get_refund("rfnd_x/../../y"))
    await capture(r, h, h.unknown_intent_id, "capture unknown")
    await refund(r, h, h.unknown_intent_id, "refund unknown", paise=None, k="rf-x")


async def webhooks(r: Recorder, h: GatewayHarness) -> None:
    gw = h.gateway
    intent = await create(r, h, "webhook")
    if not intent or not isinstance(gw, WebhookHandler):
        r.note("webhooks", "skipped: no webhooks")
        return
    await pay(r, h, intent.gateway_intent_id)
    body, headers = h.signed_webhook(intent.gateway_intent_id)
    lower = {k.lower(): v for k, v in headers.items()}
    sig_header = next(k for k in lower if "signature" in k)
    without_id = {k: v for k, v in lower.items() if "signature" in k}
    tampered = body.replace(b'"captured"', b'"refunded"', 1)
    await r.step("valid webhook", lambda: gw.verify_and_parse(body, headers))
    await r.step(
        "upper-case headers",
        lambda: gw.verify_and_parse(body, {k.upper(): v for k, v in headers.items()}),
    )
    await r.step("tampered body", lambda: gw.verify_and_parse(tampered, headers))
    await r.step(
        "wrong signature", lambda: gw.verify_and_parse(body, lower | {sig_header: "0" * 64})
    )
    await r.step("no signature", lambda: gw.verify_and_parse(body, {}))
    await r.step("no event id", lambda: gw.verify_and_parse(body, without_id))
    await r.step("signed non-JSON", lambda: gw.verify_and_parse(b"[]", headers))


async def checkout_return(r: Recorder, h: GatewayHarness) -> None:
    gw = h.gateway
    intent = await create(r, h, "checkout", CaptureMode.MANUAL)
    if not intent or not isinstance(gw, CheckoutReturnVerifier):
        r.note("checkout return", "skipped: no checkout_return")
        return
    attempt = await pay(r, h, intent.gateway_intent_id)
    params = h.checkout_return(intent.gateway_intent_id, attempt)
    sig = next(k for k in params if "signature" in k)
    other = next(k for k in params if k != sig)
    await r.step("valid return", lambda: gw.verify_checkout_return(params))
    await r.step("tampered signature", lambda: gw.verify_checkout_return(params | {sig: "0" * 64}))
    await r.step(
        "swapped id", lambda: gw.verify_checkout_return(params | {other: params[other] + "x"})
    )
    await r.step("missing param", lambda: gw.verify_checkout_return({sig: params[sig]}))


SCENARIOS: tuple[Scenario, ...] = tuple(
    Scenario(fn.__name__, fn)
    for fn in (
        capabilities,
        create_and_read,
        automatic_capture,
        manual_capture,
        failed_then_paid,
        refunds,
        refund_before_capture,
        bad_inputs,
        webhooks,
        checkout_return,
    )
)
