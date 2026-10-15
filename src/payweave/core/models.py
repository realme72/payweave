"""Canonical payment models: the one shape every gateway is mapped onto.

* ``PaymentIntent``: what the merchant wants to collect (one per merchant
  order). Merchants work with intents.
* ``PaymentAttempt``: one try by the customer against an intent. An intent may
  have several (a declined card, then a successful UPI payment).
* ``Refund``: money returned against a captured attempt.

All models are immutable and reject unknown fields. Invariants that would let
a wrong amount or status slip through are enforced at construction, so an
inconsistent intent cannot exist.

PayWeave never handles raw card data (PAN/CVV): customers pay on the gateway's
own checkout (see ``ClientCheckout``) or with gateway-issued tokens.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus, RefundStatus

NonEmptyStr = Annotated[str, Field(min_length=1, max_length=255)]
# Safe for every gateway seen so far (Razorpay refunds need >= 10 of [A-Za-z0-9_-]).
IdempotencyKey = Annotated[str, Field(min_length=16, max_length=255, pattern=r"^[A-Za-z0-9_-]+$")]
Metadata = Annotated[
    dict[Annotated[str, Field(min_length=1, max_length=40)], Annotated[str, Field(max_length=255)]],
    Field(max_length=10),
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# --- next actions -------------------------------------------------------------


class ClientCheckout(_Model):
    """Open the gateway's own checkout in the customer's browser/app.

    ``params`` holds only **public** values the client SDK needs (e.g. Razorpay
    ``key_id`` + ``order_id``). Secrets must never appear here.
    """

    type: Literal["client_checkout"] = "client_checkout"
    gateway: NonEmptyStr
    params: dict[NonEmptyStr, NonEmptyStr]


class RedirectToUrl(_Model):
    type: Literal["redirect_to_url"] = "redirect_to_url"
    url: HttpUrl
    return_url: HttpUrl | None = None


class UpiIntent(_Model):
    type: Literal["upi_intent"] = "upi_intent"
    intent_uri: Annotated[str, Field(pattern=r"^upi://")]


class DisplayQr(_Model):
    type: Literal["display_qr"] = "display_qr"
    qr_data: NonEmptyStr
    expires_at: AwareDatetime | None = None


NextAction = Annotated[
    ClientCheckout | RedirectToUrl | UpiIntent | DisplayQr, Field(discriminator="type")
]


# --- requests -----------------------------------------------------------------


class CaptureMode(StrEnum):
    AUTOMATIC = "automatic"  # authorise and capture in one go
    MANUAL = "manual"  # authorise only; merchant captures later


class Customer(_Model):
    """Who pays, for gateways that require it on every order (Cashfree).

    Always supplied by the merchant; a mapper never makes these up.
    """

    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,50}$")]
    phone: Annotated[str, Field(pattern=r"^\+?[0-9]{8,15}$")] | None = None
    email: Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=100)] | None = (
        None
    )
    name: Annotated[str, Field(min_length=1, max_length=100)] | None = None


class CreatePaymentRequest(_Model):
    amount: Money
    merchant_reference: Annotated[str, Field(min_length=1, max_length=40)]
    idempotency_key: IdempotencyKey
    capture_mode: CaptureMode = CaptureMode.AUTOMATIC
    metadata: Metadata = Field(default_factory=dict)
    customer: Customer | None = None

    @model_validator(mode="after")
    def _positive_amount(self) -> Self:
        if self.amount.is_zero:
            raise ValueError("payment amount must be greater than zero")
        return self


class CaptureRequest(_Model):
    gateway_intent_id: NonEmptyStr
    idempotency_key: IdempotencyKey
    amount: Money | None = None  # None = the full authorised amount
    attempt_id: NonEmptyStr | None = None  # only needed if several attempts are authorised


class VoidRequest(_Model):
    gateway_intent_id: NonEmptyStr
    idempotency_key: IdempotencyKey


class RefundRequest(_Model):
    gateway_intent_id: NonEmptyStr
    idempotency_key: IdempotencyKey
    amount: Money | None = None  # None = everything still refundable
    reason: Annotated[str, Field(max_length=255)] | None = None


# --- results ------------------------------------------------------------------


class FailureCategory(StrEnum):
    """Why an attempt failed, in gateway-neutral terms."""

    CUSTOMER_CANCELLED = "customer_cancelled"
    AUTHENTICATION_FAILED = "authentication_failed"  # OTP / PIN / 3DS
    INSUFFICIENT_FUNDS = "insufficient_funds"
    LIMIT_EXCEEDED = "limit_exceeded"
    INVALID_PAYMENT_DETAILS = "invalid_payment_details"  # card number, expiry, CVV, VPA
    DECLINED = "declined"  # issuer/PSP declined without a more specific reason
    RISK_DECLINED = "risk_declined"
    TIMED_OUT = "timed_out"
    TECHNICAL_ERROR = "technical_error"  # bank / gateway / PSP outage
    MERCHANT_CONFIGURATION = "merchant_configuration"  # method not enabled, etc.
    UNKNOWN = "unknown"


class FailureInfo(_Model):
    category: FailureCategory
    gateway_code: str | None = None  # gateway's machine-readable reason
    message: str | None = None


class PaymentAttempt(_Model):
    gateway_attempt_id: NonEmptyStr
    status: AttemptStatus
    amount: Money
    amount_refunded: Money
    method: str | None = None  # gateway's method name, e.g. "card", "upi"
    failure: FailureInfo | None = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.amount_refunded.currency != self.amount.currency:
            raise ValueError("attempt amounts must share one currency")
        if self.amount_refunded > self.amount:
            raise ValueError("attempt refunded more than its amount")
        if not self.amount_refunded.is_zero and self.status is not AttemptStatus.CAPTURED:
            raise ValueError("only a captured attempt can have refunds")
        if (self.status is AttemptStatus.FAILED) != (self.failure is not None):
            raise ValueError("failure must be set exactly when the attempt failed")
        return self


class PaymentIntent(_Model):
    gateway: NonEmptyStr
    gateway_intent_id: NonEmptyStr
    merchant_reference: NonEmptyStr | None  # None if created outside PayWeave
    status: IntentStatus
    capture_mode: CaptureMode | None  # None if the gateway didn't report it
    amount: Money
    amount_captured: Money
    amount_refunded: Money
    attempts: tuple[PaymentAttempt, ...] = ()
    next_action: NextAction | None = None
    created_at: AwareDatetime
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        currency = self.amount.currency
        amounts = [self.amount_captured, self.amount_refunded]
        amounts += [m for a in self.attempts for m in (a.amount, a.amount_refunded)]
        if any(m.currency != currency for m in amounts):
            raise ValueError("all amounts on an intent must share one currency")

        captured = [a for a in self.attempts if a.status is AttemptStatus.CAPTURED]
        if self.attempts:
            # Totals must agree with the attempts exactly.
            if sum(a.amount.amount_minor for a in captured) != self.amount_captured.amount_minor:
                raise ValueError("amount_captured doesn't match captured attempts")
            refunded = sum(a.amount_refunded.amount_minor for a in self.attempts)
            if refunded != self.amount_refunded.amount_minor:
                raise ValueError("amount_refunded doesn't match attempts")
        if self.amount_captured > self.amount:
            raise ValueError("captured more than the intent amount")
        if self.amount_refunded > self.amount_captured:
            raise ValueError("refunded more than was captured")

        s = IntentStatus
        if (self.status is s.REQUIRES_ACTION) != (self.next_action is not None):
            raise ValueError("next_action must be set exactly when status is requires_action")
        captured_states = {s.CAPTURED, s.PARTIALLY_REFUNDED, s.REFUNDED}
        if (self.status in captured_states) != (not self.amount_captured.is_zero):
            raise ValueError(f"status {self.status} is inconsistent with the captured amount")
        if self.status is s.CAPTURED and not self.amount_refunded.is_zero:
            raise ValueError("an intent with refunds must be partially_refunded or refunded")
        if self.status is s.PARTIALLY_REFUNDED and not (
            Money.zero(currency) < self.amount_refunded < self.amount_captured
        ):
            raise ValueError("partially_refunded requires 0 < refunded < captured")
        if self.status is s.REFUNDED and self.amount_refunded != self.amount_captured:
            raise ValueError("refunded requires refunded == captured")
        if self.status is s.AUTHORIZED and not any(
            a.status is AttemptStatus.AUTHORIZED for a in self.attempts
        ):
            raise ValueError("an authorized intent needs an authorized attempt")
        return self

    @property
    def refundable(self) -> Money:
        return self.amount_captured - self.amount_refunded


class Refund(_Model):
    gateway: NonEmptyStr
    gateway_refund_id: NonEmptyStr
    gateway_attempt_id: NonEmptyStr
    status: RefundStatus
    amount: Money
    failure: FailureInfo | None = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.amount.is_zero:
            raise ValueError("refund amount must be greater than zero")
        if (self.status is RefundStatus.FAILED) != (self.failure is not None):
            raise ValueError("failure must be set exactly when the refund failed")
        return self


# --- webhooks -----------------------------------------------------------------


class WebhookEventType(StrEnum):
    INTENT_UPDATED = "intent.updated"
    REFUND_UPDATED = "refund.updated"
    IGNORED = "ignored"  # a verified gateway event we deliberately don't act on


class WebhookEvent(_Model):
    """A *verified* notification that something changed.

    Deliberately carries only identifiers, not state: webhook payloads can be
    stale snapshots and arrive out of order, so the receiver re-fetches the
    intent/refund from the gateway before updating anything.
    """

    gateway: NonEmptyStr
    event_id: NonEmptyStr  # unique per event, used for dedupe
    event_type: WebhookEventType
    gateway_event_type: NonEmptyStr  # original name, e.g. "payment.captured"
    occurred_at: AwareDatetime
    gateway_intent_id: NonEmptyStr | None = None
    gateway_refund_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _ids_match_type(self) -> Self:
        t = WebhookEventType
        if self.event_type is t.INTENT_UPDATED and self.gateway_intent_id is None:
            raise ValueError("intent.updated needs gateway_intent_id")
        if self.event_type is t.REFUND_UPDATED and self.gateway_refund_id is None:
            raise ValueError("refund.updated needs gateway_refund_id")
        return self
