"""Razorpay test doubles: payload builders and an offline API simulator.

Payload shapes follow the samples in docs/sources/razorpay/ (api__orders__entity.md,
api__payments__entity.md, api__refunds__entity.md, webhooks__payments.md).
Used by the test suite and by the verification gates (G5 contract, G7
differential), which is why they live in the package rather than under tests/.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import re
import secrets
from collections.abc import Callable
from typing import Any

import httpx
import respx

from payweave.core.capabilities import Gateway
from payweave.core.factory import GatewayContext, GatewayFactory, GatewayProvider
from payweave.gateways.razorpay import provider as hand_written_provider

# --- payloads -------------------------------------------------------------------------------

ORDER_ID = "order_DaZlswtdcn9UNV"
KEY_ID = "rzp_test_abcdefghijklmn"


def order(status: str = "created", amount: int = 50000, **kw: Any) -> dict[str, Any]:
    return {
        "id": ORDER_ID,
        "entity": "order",
        "amount": amount,
        "amount_paid": 0,
        "amount_due": amount,
        "currency": "INR",
        "receipt": "Receipt #20",
        "status": status,
        "attempts": 0,
        "notes": {"pw_capture_mode": "manual", "pw_idempotency_key": "k" * 16, "cart": "42"},
        "created_at": 1572502745,
        **kw,
    }


def payment(
    status: str,
    *,
    pid: str = "pay_L0nSsccovt6zyp",
    amount: int = 50000,
    refunded: int = 0,
    captured: bool | None = None,
    created_at: int = 1572502800,
    **kw: Any,
) -> dict[str, Any]:
    failed = status == "failed"
    return {
        "id": pid,
        "entity": "payment",
        "amount": amount,
        "currency": "INR",
        "status": status,
        "order_id": ORDER_ID,
        "method": "card",
        "amount_refunded": refunded,
        "refund_status": None if refunded == 0 else ("full" if refunded == amount else "partial"),
        "captured": captured if captured is not None else status in ("captured", "refunded"),
        "notes": [],  # Razorpay sends [] for empty notes in its samples
        "error_code": "BAD_REQUEST_ERROR" if failed else None,
        "error_description": "Your payment has been cancelled." if failed else None,
        "error_source": "customer" if failed else None,
        "error_step": "payment_authentication" if failed else None,
        "error_reason": "payment_cancelled" if failed else None,
        "created_at": created_at,
        **kw,
    }


def payments(*items: dict[str, Any]) -> dict[str, Any]:
    return {"entity": "collection", "count": len(items), "items": list(items)}


def refund(status: str = "processed", amount: int = 10000, **kw: Any) -> dict[str, Any]:
    return {
        "id": "rfnd_FgRAHdNOM4ZVbO",
        "entity": "refund",
        "amount": amount,
        "currency": "INR",
        "payment_id": "pay_L0nSsccovt6zyp",
        "notes": {},
        "status": status,
        "speed_requested": "normal",
        "created_at": 1600856650,
        **kw,
    }


def webhook(event: str, **payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": "event",
        "account_id": "acc_BFQ7uQEaa7j2z7",
        "event": event,
        "contains": list(payload),
        "payload": {name: {"entity": entity} for name, entity in payload.items()},
        "created_at": 1567674606,
    }


# --- Razorpay (offline simulator) ----------------------------------------------------

API = "https://api.razorpay.com/v1"
KEY_SECRET = "sim_key_secret"  # noqa: S105 - simulator-only
WEBHOOK_SECRET = "sim_webhook_secret"  # noqa: S105 - simulator-only
ID = r"[A-Za-z0-9]+"


def random_ids(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(7)}"


def sequential_ids() -> Callable[[str], str]:
    """Deterministic ids, so two runs of the same scenario can be compared."""
    counter = itertools.count(1)
    return lambda prefix: f"{prefix}_{next(counter):014d}"


def _bad_request(description: str) -> httpx.Response:
    error = {"code": "BAD_REQUEST_ERROR", "description": description, "reason": "NA"}
    return httpx.Response(400, json={"error": error})


class RazorpaySimulator:
    """Stateful stand-in for the Razorpay endpoints PayWeave uses.

    Behaviour comes from the docs *and* the sandbox findings in
    docs/gateways/razorpay/NATIVE_MODEL.md: unknown ids -> 400, full capture
    only, refunds limited to the remaining amount with a ₹1 minimum, refund
    idempotency by header (same body -> same refund, different body -> 400).
    """

    def __init__(self, router: respx.MockRouter, ids: Callable[[str], str] = random_ids) -> None:
        self._ids = ids
        self.orders: dict[str, dict[str, Any]] = {}
        self.payments: dict[str, dict[str, Any]] = {}
        self.refunds: dict[str, dict[str, Any]] = {}
        self.refund_keys: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._clock = itertools.count(1_700_000_000)
        routes: list[tuple[str, str, Callable[..., httpx.Response]]] = [
            ("POST", r"/orders$", self._create_order),
            ("GET", rf"/orders/(?P<oid>order_{ID})$", self._get_order),
            ("GET", rf"/orders/(?P<oid>order_{ID})/payments$", self._order_payments),
            ("POST", rf"/payments/(?P<pid>pay_{ID})/capture$", self._capture),
            ("POST", rf"/payments/(?P<pid>pay_{ID})/refund$", self._refund),
            ("GET", rf"/refunds/(?P<rid>rfnd_{ID})$", self._get_refund),
        ]
        for method, path, handler in routes:
            router.route(method=method, url__regex=re.escape(API) + path).mock(side_effect=handler)

    # --- endpoints ---

    def _create_order(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["amount"] < 100:
            return _bad_request("Order amount less than minimum amount allowed")
        created = order(
            id=self._ids("order"),
            amount=body["amount"],
            amount_due=body["amount"],
            currency=body["currency"],
            receipt=body["receipt"],
            notes=body["notes"],
            created_at=next(self._clock),
        )
        created["_capture"] = body["payment"]["capture"]  # simulator-private
        self.orders[created["id"]] = created
        return httpx.Response(200, json=self._public(created))

    def _get_order(self, request: httpx.Request, oid: str) -> httpx.Response:
        if oid not in self.orders:
            return _bad_request("The id provided does not exist")
        return httpx.Response(200, json=self._public(self.orders[oid]))

    def _order_payments(self, request: httpx.Request, oid: str) -> httpx.Response:
        if oid not in self.orders:
            return _bad_request("The id provided does not exist")
        items = [p for p in self.payments.values() if p["order_id"] == oid]
        return httpx.Response(200, json=payments(*items))

    def _capture(self, request: httpx.Request, pid: str) -> httpx.Response:
        payment = self.payments.get(pid)
        if payment is None:
            return _bad_request("The id provided does not exist")
        body = json.loads(request.content)
        if payment["status"] != "authorized":
            return _bad_request("This payment has already been captured")
        if body.get("amount") != payment["amount"]:
            return _bad_request("Capture amount must be equal to the amount authorized")
        payment.update(status="captured", captured=True)
        self._order_paid(payment)
        return httpx.Response(200, json=payment)

    def _refund(self, request: httpx.Request, pid: str) -> httpx.Response:  # noqa: PLR0911
        payment = self.payments.get(pid)
        if payment is None:
            return _bad_request("The id provided does not exist")
        body = json.loads(request.content)
        key = request.headers.get("x-refund-idempotency", "")
        if key in self.refund_keys:
            seen_body, seen = self.refund_keys[key]
            if seen_body != body:
                return _bad_request(
                    "Different request with the same idempotency key has already been processed."
                )
            return httpx.Response(200, json=seen)
        if not payment["captured"]:
            return _bad_request("Payment has not been captured")
        amount = body.get("amount", payment["amount"] - payment["amount_refunded"])
        if amount < 100:
            return _bad_request("The amount must be atleast INR 1.00")
        if amount > payment["amount"] - payment["amount_refunded"]:
            return _bad_request("invalid request sent")  # Razorpay's catch-all (F4)
        created = refund(
            id=self._ids("rfnd"), amount=amount, payment_id=pid, created_at=next(self._clock)
        )
        refunded = payment["amount_refunded"] + amount
        full = refunded == payment["amount"]
        payment.update(
            amount_refunded=refunded,
            refund_status="full" if full else "partial",
            status="refunded" if full else payment["status"],
        )
        self.refunds[created["id"]] = created
        if key:
            self.refund_keys[key] = (body, created)
        return httpx.Response(200, json=created)

    def _get_refund(self, request: httpx.Request, rid: str) -> httpx.Response:
        if rid not in self.refunds:
            return _bad_request("The id provided does not exist")
        return httpx.Response(200, json=self.refunds[rid])

    # --- customer side ---

    def pay(self, order_id: str, *, succeed: bool) -> str:
        order = self.orders[order_id]
        automatic = order["_capture"] == "automatic"
        status = ("captured" if automatic else "authorized") if succeed else "failed"
        attempt = payment(
            status,
            pid=self._ids("pay"),
            amount=order["amount"],
            currency=order["currency"],
            order_id=order_id,
            created_at=next(self._clock),
        )
        self.payments[attempt["id"]] = attempt
        order["status"] = "attempted"
        order["attempts"] += 1
        if status == "captured":
            self._order_paid(attempt)
        return str(attempt["id"])

    def _order_paid(self, payment: dict[str, Any]) -> None:
        order = self.orders[payment["order_id"]]
        order.update(status="paid", amount_paid=payment["amount"], amount_due=0)

    @staticmethod
    def _public(order: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in order.items() if not k.startswith("_")}


class RazorpayHarness:
    def __init__(
        self,
        router: respx.MockRouter,
        rz_provider: GatewayProvider[Any] = hand_written_provider,
        *,
        ids: Callable[[str], str] = random_ids,
    ) -> None:
        self.sim = RazorpaySimulator(router, ids)
        self._gateway = GatewayFactory([rz_provider]).create(
            "razorpay",
            {"key_id": KEY_ID, "key_secret": KEY_SECRET, "webhook_secrets": [WEBHOOK_SECRET]},
            context=GatewayContext(name="razorpay"),
        )
        self._events = itertools.count(1)

    @property
    def gateway(self) -> Gateway:
        return self._gateway

    unknown_intent_id = "order_DoesNotExist01"
    unknown_refund_id = "rfnd_DoesNotExist01"

    async def customer_pays(self, gateway_intent_id: str, *, succeed: bool = True) -> str:
        return self.sim.pay(gateway_intent_id, succeed=succeed)

    def signed_webhook(self, gateway_intent_id: str) -> tuple[bytes, dict[str, str]]:
        payment = next(p for p in self.sim.payments.values() if p["order_id"] == gateway_intent_id)
        body = json.dumps(webhook("payment.captured", payment=payment)).encode()
        signature = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        return body, {
            "X-Razorpay-Signature": signature,
            "X-Razorpay-Event-Id": f"evt_{next(self._events)}",
        }

    def checkout_return(self, gateway_intent_id: str, attempt_id: str) -> dict[str, str]:
        message = f"{gateway_intent_id}|{attempt_id}".encode()
        return {
            "razorpay_order_id": gateway_intent_id,
            "razorpay_payment_id": attempt_id,
            "razorpay_signature": hmac.new(
                KEY_SECRET.encode(), message, hashlib.sha256
            ).hexdigest(),
        }

    async def aclose(self) -> None:
        await self._gateway.aclose()  # type: ignore[attr-defined]
