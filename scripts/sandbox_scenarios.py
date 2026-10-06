"""Server-side Razorpay scenarios against the TEST-mode API (no customer needed).

    set -a; . ./.env; set +a
    uv run python scripts/sandbox_scenarios.py [--paid-order order_xxx]

Each scenario states what we expect and records what actually happened, so
surprises show up as MISMATCH rather than being hidden. ``--paid-order`` runs
the extra scenarios that need an already-captured payment.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from payweave.core.errors import PayWeaveError
from payweave.core.factory import GatewayFactory
from payweave.core.models import CaptureMode, CaptureRequest, CreatePaymentRequest, RefundRequest
from payweave.core.money import Money
from payweave.gateways.razorpay import RazorpayGateway

API = "https://api.razorpay.com/v1/"
PACE_SECONDS = 2.0  # test mode rate-limits bursts (~9 quick order creations -> 429)


@dataclass
class Result:
    name: str
    expected: str
    actual: str

    @property
    def ok(self) -> bool:
        return self.actual.startswith(self.expected)


def key(prefix: str) -> str:
    return f"pw_{prefix}_{secrets.token_hex(8)}"


def order_req(amount: Money, **kw: Any) -> CreatePaymentRequest:
    return CreatePaymentRequest(
        amount=amount,
        merchant_reference=kw.pop("ref", f"pw-scn-{secrets.token_hex(4)}"),
        idempotency_key=key("ord"),
        **kw,
    )


async def outcome(call: Callable[[], Awaitable[Any]]) -> str:
    """'ok: <summary>' or '<ErrorClass>: <message>'."""
    try:
        value = await call()
    except PayWeaveError as exc:
        return f"{type(exc).__name__}: {exc}"
    status = getattr(value, "status", None)
    return f"ok: {status}" if status is not None else f"ok: {value!r}"[:120]


async def raw(client: httpx.AsyncClient, method: str, path: str, **kw: Any) -> str:
    r = await client.request(method, path, **kw)
    body = r.json()
    if r.is_success:
        return f"ok: {body.get('id')} {body.get('status')}"
    err = body.get("error", {})
    return f"{r.status_code}: {err.get('description')} (reason={err.get('reason')})"


async def main(paid_order: str | None) -> int:  # noqa: PLR0915 - a flat scenario list
    factory = GatewayFactory.with_installed_providers()
    config = factory.config_from_env("razorpay")
    config.pop("webhook_secrets", None)
    if not str(config.get("key_id", "")).startswith("rzp_test_"):
        sys.exit("Refusing to run: needs a TEST key (rzp_test_...)")
    gw = factory.create("razorpay", config)
    bad = factory.create("razorpay", {**config, "key_secret": "wrong_secret_value"}, name="bad")
    if not (isinstance(gw, RazorpayGateway) and isinstance(bad, RazorpayGateway)):
        sys.exit("expected RazorpayGateway instances")
    auth = (str(config["key_id"]), str(config["key_secret"]))
    client = httpx.AsyncClient(base_url=API, auth=auth, timeout=30)
    inr = lambda paise: Money.of(paise, "INR")  # noqa: E731
    results: list[Result] = []

    async def check(name: str, expected: str, call: Callable[[], Awaitable[Any]]) -> None:
        await asyncio.sleep(PACE_SECONDS)
        results.append(Result(name, expected, await outcome(call)))

    async def check_raw(name: str, expected: str, method: str, path: str, **kw: Any) -> None:
        await asyncio.sleep(PACE_SECONDS)
        results.append(Result(name, expected, await raw(client, method, path, **kw)))

    try:
        # --- order creation -------------------------------------------------
        await check(
            "order: ₹1.00 (documented minimum)",
            "ok",
            lambda: gw.create_payment(order_req(inr(100))),
        )
        await check(
            "order: ₹0.99 (below minimum)",
            "InvalidRequestError",
            lambda: gw.create_payment(order_req(inr(99))),
        )
        await check(
            "order: automatic capture",
            "ok",
            lambda: gw.create_payment(order_req(inr(500), capture_mode=CaptureMode.AUTOMATIC)),
        )
        await check(
            "order: USD (international)",
            "ok",
            lambda: gw.create_payment(order_req(Money.of(1000, "USD"))),
        )
        await check(
            "order: JPY (zero-decimal)",
            "ok",
            lambda: gw.create_payment(order_req(Money.of(500, "JPY"))),
        )
        await check(
            "order: KWD 1.230 (3-decimal, ends in 0)",
            "ok",
            lambda: gw.create_payment(order_req(Money.of(1230, "KWD"))),
        )
        await check(
            "order: KWD 1.231 (3-decimal, not 0) blocked locally",
            "InvalidAmountError",
            lambda: gw.create_payment(order_req(Money.of(1231, "KWD"))),
        )
        ref = f"pw-dup-{secrets.token_hex(4)}"
        await check(
            "order: receipt first use",
            "ok",
            lambda: gw.create_payment(order_req(inr(100), ref=ref)),
        )
        # Docs say receipts "have to be unique", but Razorpay accepts duplicates (finding F3).
        await check(
            "order: same receipt again (accepted despite docs)",
            "ok",
            lambda: gw.create_payment(order_req(inr(100), ref=ref)),
        )
        await check(
            "order: 10 metadata keys",
            "ok",
            lambda: gw.create_payment(
                order_req(inr(100), metadata={f"k{i}": "v" for i in range(10)})
            ),
        )

        # --- lookups & auth ---------------------------------------------------
        await check(
            "fetch: unknown order",
            "InvalidRequestError",
            lambda: gw.get_payment("order_DoesNotExist01"),
        )
        await check(
            "fetch: malformed id blocked locally",
            "InvalidRequestError",
            lambda: gw.get_payment("order_../payments"),
        )
        await check(
            "fetch: unknown refund",
            "InvalidRequestError",
            lambda: gw.get_refund("rfnd_DoesNotExist01"),
        )
        await check(
            "auth: wrong secret",
            "AuthenticationFailedError",
            lambda: bad.create_payment(order_req(inr(100))),
        )

        # --- operations in the wrong state -------------------------------------
        fresh = await gw.create_payment(order_req(inr(100), capture_mode=CaptureMode.MANUAL))
        oid = fresh.gateway_intent_id
        await check(
            "capture: order with no payment",
            "InvalidRequestError",
            lambda: gw.capture(CaptureRequest(gateway_intent_id=oid, idempotency_key=key("cap"))),
        )
        await check(
            "refund: order with no payment",
            "InvalidRequestError",
            lambda: gw.refund(RefundRequest(gateway_intent_id=oid, idempotency_key=key("rf"))),
        )

        # --- scenarios needing a captured payment -------------------------------
        if paid_order:
            intent = await gw.get_payment(paid_order)
            pid = intent.attempts[-1].gateway_attempt_id
            await check("paid: fetch maps cleanly", "ok", lambda: gw.get_payment(paid_order))
            await check(
                "paid: capture again is idempotent (no API call)",
                "ok",
                lambda: gw.capture(
                    CaptureRequest(gateway_intent_id=paid_order, idempotency_key=key("cap"))
                ),
            )
            await check_raw(
                "paid: raw capture of captured payment",
                "400",
                "POST",
                f"payments/{pid}/capture",
                json={"amount": intent.amount.amount_minor, "currency": "INR"},
            )
            await check_raw(
                "paid: raw capture wrong amount",
                "400",
                "POST",
                f"payments/{pid}/capture",
                json={"amount": 1, "currency": "INR"},
            )
            await check(
                "paid: refund more than refundable blocked locally",
                "InvalidAmountError",
                lambda: gw.refund(
                    RefundRequest(
                        gateway_intent_id=paid_order,
                        idempotency_key=key("rf"),
                        amount=inr(intent.refundable.amount_minor + 1),
                    )
                ),
            )
            await check(
                "paid: refund in USD blocked locally",
                "InvalidAmountError",
                lambda: gw.refund(
                    RefundRequest(
                        gateway_intent_id=paid_order,
                        idempotency_key=key("rf"),
                        amount=Money.of(100, "USD"),
                    )
                ),
            )
            k = str(uuid.uuid4())
            await check_raw(
                "paid: refund ₹0.50 (below ₹1 minimum)",
                "400",
                "POST",
                f"payments/{pid}/refund",
                json={"amount": 50},
                headers={"X-Refund-Idempotency": str(uuid.uuid4())},
            )
            await check_raw(
                "paid: refund ₹1.00 with key K",
                "",
                "POST",
                f"payments/{pid}/refund",
                json={"amount": 100},
                headers={"X-Refund-Idempotency": k},
            )
            await check_raw(
                "paid: same key K, different amount",
                "400",
                "POST",
                f"payments/{pid}/refund",
                json={"amount": 110},
                headers={"X-Refund-Idempotency": k},
            )
            await check_raw(
                "paid: idempotency key too short (9 chars)",
                "400",
                "POST",
                f"payments/{pid}/refund",
                json={"amount": 50},
                headers={"X-Refund-Idempotency": "short_key"},
            )
    finally:
        await client.aclose()
        await gw.aclose()
        await bad.aclose()

    width = max(len(r.name) for r in results)
    for r in results:
        mark = "PASS    " if r.ok else "MISMATCH"
        print(
            f"{mark}  {r.name:<{width}}  "
            f"expected~{r.expected or '(record)':<26} got: {r.actual[:110]}"
        )
    mismatches = sum(not r.ok for r in results)
    print(f"\n{len(results) - mismatches}/{len(results)} as expected, {mismatches} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paid-order", help="an order with a captured payment (test mode)")
    sys.exit(asyncio.run(main(parser.parse_args().paid_order)))
