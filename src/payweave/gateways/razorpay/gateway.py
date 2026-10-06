"""Razorpay gateway: implements the canonical capability interfaces.

Flow: our server creates an **order** (the intent); the customer pays it in
Razorpay Checkout (``ClientCheckout`` next action); we capture/refund the
resulting **payment** (the attempt). See docs/gateways/razorpay/NATIVE_MODEL.md.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from payweave.core.capabilities import Capability
from payweave.core.errors import (
    InvalidAmountError,
    InvalidRequestError,
    InvalidSignatureError,
    UnsupportedCapabilityError,
)
from payweave.core.factory import GatewayContext
from payweave.core.models import (
    CaptureMode,
    CaptureRequest,
    CreatePaymentRequest,
    PaymentAttempt,
    PaymentIntent,
    Refund,
    RefundRequest,
    WebhookEvent,
)
from payweave.core.money import Money, currency_exponent
from payweave.core.status import AttemptStatus, IntentStatus
from payweave.gateways.razorpay import mapping
from payweave.gateways.razorpay.client import RazorpayClient
from payweave.gateways.razorpay.config import RazorpayConfig
from payweave.gateways.razorpay.signatures import (
    verify_checkout_signature,
    verify_webhook_signature,
)

_BASE_CAPABILITIES = frozenset(
    {
        Capability.CREATE_PAYMENT,
        Capability.GET_PAYMENT,
        Capability.CAPTURE,
        Capability.MANUAL_CAPTURE,
        Capability.REFUND,
        Capability.PARTIAL_REFUND,
        Capability.MULTIPLE_REFUNDS,
        Capability.NATIVE_IDEMPOTENT_REFUND,
        Capability.GET_REFUND,
        Capability.CHECKOUT_RETURN,
    }
)  # Not supported by Razorpay: VOID, PARTIAL_CAPTURE.

_CAPTURED_STATES = frozenset(
    {IntentStatus.CAPTURED, IntentStatus.PARTIALLY_REFUNDED, IntentStatus.REFUNDED}
)


class RazorpayGateway:
    def __init__(
        self, config: RazorpayConfig, context: GatewayContext, client: RazorpayClient | None = None
    ) -> None:
        self._config = config
        self._name = context.name
        self._client = client or RazorpayClient(
            config, gateway_name=context.name, transport=context.http_transport
        )
        caps = set(_BASE_CAPABILITIES)
        if config.webhook_secrets:
            caps.add(Capability.WEBHOOKS)
        self._capabilities = frozenset(caps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- intents --------------------------------------------------------------

    async def create_payment(self, request: CreatePaymentRequest) -> PaymentIntent:
        _check_amount(request.amount)
        if any(k.startswith(mapping.NOTE_PREFIX) for k in request.metadata):
            raise InvalidRequestError(
                f"metadata keys starting with {mapping.NOTE_PREFIX!r} are reserved",
                gateway=self._name,
            )
        cfg = self._config
        capture: dict[str, Any]
        if request.capture_mode is CaptureMode.AUTOMATIC:
            capture = {
                "capture": "automatic",
                "capture_options": {
                    "automatic_expiry_period": cfg.automatic_capture_expiry_minutes,
                    "manual_expiry_period": cfg.manual_capture_expiry_minutes,
                    "refund_speed": cfg.uncaptured_refund_speed,
                },
            }
        else:
            capture = {
                "capture": "manual",
                "capture_options": {
                    "manual_expiry_period": cfg.manual_capture_expiry_minutes,
                    "refund_speed": cfg.uncaptured_refund_speed,
                },
            }
        order = await self._client.post(
            "orders",
            {
                "amount": request.amount.amount_minor,
                "currency": request.amount.currency,
                "receipt": request.merchant_reference,
                "notes": {
                    **request.metadata,
                    mapping.NOTE_IDEMPOTENCY_KEY: request.idempotency_key,
                    mapping.NOTE_CAPTURE_MODE: request.capture_mode.value,
                },
                "payment": capture,
            },
        )
        return self._intent(order, [])

    async def get_payment(self, gateway_intent_id: str) -> PaymentIntent:
        _check_id(gateway_intent_id, "order_", self._name)
        order, payments = await asyncio.gather(
            self._client.get(f"orders/{gateway_intent_id}"),
            self._client.get(f"orders/{gateway_intent_id}/payments"),
        )
        items = payments.get("items")
        if not isinstance(items, list):
            raise mapping.RazorpayMappingError("payments list has no items", gateway=self._name)
        return self._intent(order, items)

    async def capture(self, request: CaptureRequest) -> PaymentIntent:
        """Capture the authorised attempt. Razorpay supports full capture only.

        Idempotent: if the intent is already captured (e.g. a retry after a
        timeout), the current intent is returned without calling Razorpay.
        """
        intent = await self.get_payment(request.gateway_intent_id)
        authorized = [a for a in intent.attempts if a.status is AttemptStatus.AUTHORIZED]
        if request.attempt_id is not None:
            authorized = [a for a in authorized if a.gateway_attempt_id == request.attempt_id]
        if not authorized and intent.status in _CAPTURED_STATES:
            return intent
        attempt = self._single(authorized, "authorized attempt", request.gateway_intent_id)
        if request.amount is not None and request.amount != attempt.amount:
            raise UnsupportedCapabilityError(self._name, Capability.PARTIAL_CAPTURE)
        await self._client.post(
            f"payments/{attempt.gateway_attempt_id}/capture",
            {"amount": attempt.amount.amount_minor, "currency": attempt.amount.currency},
        )
        return await self.get_payment(request.gateway_intent_id)

    # --- refunds --------------------------------------------------------------

    async def refund(self, request: RefundRequest) -> Refund:
        intent = await self.get_payment(request.gateway_intent_id)
        captured = [a for a in intent.attempts if a.status is AttemptStatus.CAPTURED]
        attempt = self._single(captured, "captured attempt", request.gateway_intent_id)
        refundable = attempt.amount - attempt.amount_refunded
        amount = request.amount or refundable
        if amount.currency != refundable.currency:
            raise InvalidAmountError(f"refund currency {amount.currency} != {refundable.currency}")
        if amount.is_zero or amount > refundable:
            raise InvalidAmountError(f"refund {amount} must be > 0 and <= refundable {refundable}")
        _check_amount(amount)
        body: dict[str, Any] = {"amount": amount.amount_minor}
        if request.reason:
            body["notes"] = {"reason": request.reason}
        refund = await self._client.post(
            f"payments/{attempt.gateway_attempt_id}/refund",
            body,
            headers={"X-Refund-Idempotency": request.idempotency_key},
        )
        return mapping.refund_from_entity(refund, gateway=self._name)

    async def get_refund(self, gateway_refund_id: str) -> Refund:
        _check_id(gateway_refund_id, "rfnd_", self._name)
        refund = await self._client.get(f"refunds/{gateway_refund_id}")
        return mapping.refund_from_entity(refund, gateway=self._name)

    # --- signed inputs --------------------------------------------------------

    def verify_and_parse(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookEvent:
        if not self._config.webhook_secrets:
            raise UnsupportedCapabilityError(self._name, Capability.WEBHOOKS)
        lower = {k.lower(): v for k, v in headers.items()}
        signature = lower.get("x-razorpay-signature")
        secrets = [s.get_secret_value() for s in self._config.webhook_secrets]
        if not signature or not verify_webhook_signature(raw_body, signature, secrets):
            raise InvalidSignatureError("webhook signature did not verify", gateway=self._name)
        # Only parse after the signature check passed.
        event_id = lower.get("x-razorpay-event-id")
        if not event_id:
            raise InvalidRequestError("missing x-razorpay-event-id header", gateway=self._name)
        try:
            body = json.loads(raw_body)
        except ValueError as exc:
            raise InvalidRequestError("webhook body is not JSON", gateway=self._name) from exc
        if not isinstance(body, dict):
            raise InvalidRequestError("webhook body is not a JSON object", gateway=self._name)
        return mapping.event_from_webhook(body, event_id=event_id, gateway=self._name)

    def verify_checkout_return(self, params: Mapping[str, str]) -> str:
        order_id = params.get("razorpay_order_id", "")
        payment_id = params.get("razorpay_payment_id", "")
        signature = params.get("razorpay_signature", "")
        if not (order_id and payment_id and signature) or not verify_checkout_signature(
            order_id, payment_id, signature, self._config.key_secret.get_secret_value()
        ):
            raise InvalidSignatureError("checkout signature did not verify", gateway=self._name)
        return order_id

    # --- helpers --------------------------------------------------------------

    def _intent(self, order: Mapping[str, Any], payments: list[Any]) -> PaymentIntent:
        return mapping.intent_from_order(
            order, payments, gateway=self._name, key_id=self._config.key_id
        )

    def _single(self, attempts: list[PaymentAttempt], what: str, intent_id: str) -> PaymentAttempt:
        if len(attempts) != 1:
            raise InvalidRequestError(
                f"intent {intent_id} has {len(attempts)} {what}s; expected exactly one"
                + (" (pass attempt_id)" if len(attempts) > 1 else ""),
                gateway=self._name,
            )
        return attempts[0]


def _check_amount(amount: Money) -> None:
    """orders/payments/refunds entity docs: 3-decimal currencies must end in 0."""
    if currency_exponent(amount.currency) == 3 and amount.amount_minor % 10 != 0:
        raise InvalidAmountError(
            f"Razorpay requires the last decimal of {amount.currency} amounts to be 0"
        )


def _check_id(value: str, prefix: str, gateway: str) -> None:
    # IDs are interpolated into URL paths: allow only Razorpay's own format.
    rest = value.removeprefix(prefix)
    if rest == value or not rest.isascii() or not rest.isalnum():
        raise InvalidRequestError(f"not a Razorpay id: {value!r}", gateway=gateway)
