"""Harnesses: one per gateway the contract suite runs against."""

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
from payweave.core.factory import GatewayContext, GatewayFactory
from payweave.gateways.razorpay import provider
from payweave.testing.fake import EVENT_ID_HEADER, SIGNATURE_HEADER, FakeGateway
from tests.gateways.razorpay import fixtures as fx

# --- fake ---------------------------------------------------------------------------


class FakeHarness:
    def __init__(self) -> None:
        self._fake = FakeGateway("fake")
        self._events = itertools.count(1)

    @property
    def gateway(self) -> Gateway:
        return self._fake

    unknown_intent_id = "fake_pi_99999999"
    unknown_refund_id = "fake_rf_99999999"

    async def customer_pays(self, gateway_intent_id: str, *, succeed: bool = True) -> str:
        return self._fake.customer_pays(gateway_intent_id, succeed=succeed)

    def signed_webhook(self, gateway_intent_id: str) -> tuple[bytes, dict[str, str]]:
        event_id = f"evt_{next(self._events)}"
        body = json.dumps(
            {"id": event_id, "type": "payment.updated", "intent_id": gateway_intent_id}
        ).encode()
        return body, {
            SIGNATURE_HEADER: self._fake.webhook_signature(body),
            EVENT_ID_HEADER: event_id,
        }

    def checkout_return(self, gateway_intent_id: str, attempt_id: str) -> dict[str, str]:
        return {
            "intent_id": gateway_intent_id,
            "attempt_id": attempt_id,
            "signature": self._fake.checkout_signature(gateway_intent_id, attempt_id),
        }


# --- Razorpay (offline simulator) ----------------------------------------------------

API = "https://api.razorpay.com/v1"
KEY_SECRET = "sim_key_secret"
WEBHOOK_SECRET = "sim_webhook_secret"
ID = r"[A-Za-z0-9]+"


def _rid(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(7)}"


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

    def __init__(self, router: respx.MockRouter) -> None:
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
        order = fx.order(
            id=_rid("order"),
            amount=body["amount"],
            amount_due=body["amount"],
            currency=body["currency"],
            receipt=body["receipt"],
            notes=body["notes"],
            created_at=next(self._clock),
        )
        order["_capture"] = body["payment"]["capture"]  # simulator-private
        self.orders[order["id"]] = order
        return httpx.Response(200, json=self._public(order))

    def _get_order(self, request: httpx.Request, oid: str) -> httpx.Response:
        if oid not in self.orders:
            return _bad_request("The id provided does not exist")
        return httpx.Response(200, json=self._public(self.orders[oid]))

    def _order_payments(self, request: httpx.Request, oid: str) -> httpx.Response:
        if oid not in self.orders:
            return _bad_request("The id provided does not exist")
        items = [p for p in self.payments.values() if p["order_id"] == oid]
        return httpx.Response(200, json=fx.payments(*items))

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
            seen_body, refund = self.refund_keys[key]
            if seen_body != body:
                return _bad_request(
                    "Different request with the same idempotency key has already been processed."
                )
            return httpx.Response(200, json=refund)
        if not payment["captured"]:
            return _bad_request("Payment has not been captured")
        amount = body.get("amount", payment["amount"] - payment["amount_refunded"])
        if amount < 100:
            return _bad_request("The amount must be atleast INR 1.00")
        if amount > payment["amount"] - payment["amount_refunded"]:
            return _bad_request("invalid request sent")  # Razorpay's catch-all (F4)
        refund = fx.refund(
            id=_rid("rfnd"), amount=amount, payment_id=pid, created_at=next(self._clock)
        )
        refunded = payment["amount_refunded"] + amount
        full = refunded == payment["amount"]
        payment.update(
            amount_refunded=refunded,
            refund_status="full" if full else "partial",
            status="refunded" if full else payment["status"],
        )
        self.refunds[refund["id"]] = refund
        if key:
            self.refund_keys[key] = (body, refund)
        return httpx.Response(200, json=refund)

    def _get_refund(self, request: httpx.Request, rid: str) -> httpx.Response:
        if rid not in self.refunds:
            return _bad_request("The id provided does not exist")
        return httpx.Response(200, json=self.refunds[rid])

    # --- customer side ---

    def pay(self, order_id: str, *, succeed: bool) -> str:
        order = self.orders[order_id]
        automatic = order["_capture"] == "automatic"
        status = ("captured" if automatic else "authorized") if succeed else "failed"
        payment = fx.payment(
            status,
            pid=_rid("pay"),
            amount=order["amount"],
            currency=order["currency"],
            order_id=order_id,
            created_at=next(self._clock),
        )
        self.payments[payment["id"]] = payment
        order["status"] = "attempted"
        order["attempts"] += 1
        if status == "captured":
            self._order_paid(payment)
        return str(payment["id"])

    def _order_paid(self, payment: dict[str, Any]) -> None:
        order = self.orders[payment["order_id"]]
        order.update(status="paid", amount_paid=payment["amount"], amount_due=0)

    @staticmethod
    def _public(order: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in order.items() if not k.startswith("_")}


class RazorpayHarness:
    def __init__(self, router: respx.MockRouter) -> None:
        self.sim = RazorpaySimulator(router)
        self._gateway = GatewayFactory([provider]).create(
            "razorpay",
            {"key_id": fx.KEY_ID, "key_secret": KEY_SECRET, "webhook_secrets": [WEBHOOK_SECRET]},
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
        body = json.dumps(fx.webhook("payment.captured", payment=payment)).encode()
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
