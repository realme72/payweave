"""Razorpay mappers: native JSON <-> canonical models. Pure functions only.

Ported by hand from src/payweave/gateways/razorpay/mapping.py; the runtime
(payweave.packages.runtime) supplies the flows, HTTP, auth and signatures.
Every mapping is total: an unknown native status raises instead of guessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from payweave.packages.sdk import (
    AttemptStatus,
    CaptureMode,
    ClientCheckout,
    CreatePaymentRequest,
    FailureCategory,
    FailureInfo,
    IntentStatus,
    Json,
    MapperContext,
    MappingError,
    Money,
    PaymentAttempt,
    PaymentIntent,
    Refund,
    RefundStatus,
    WebhookEvent,
    WebhookEventType,
    currency_exponent,
    from_unix,
    money,
)

# Notes keys PayWeave writes on every order it creates.
NOTE_PREFIX = "pw_"
NOTE_IDEMPOTENCY_KEY = "pw_idempotency_key"
NOTE_CAPTURE_MODE = "pw_capture_mode"

# errors__payments__list.md `reason` -> category. Unlisted reasons map to UNKNOWN.
FAILURE_REASONS: dict[str, FailureCategory] = {
    "payment_cancelled": FailureCategory.CUSTOMER_CANCELLED,
    "authentication_failed": FailureCategory.AUTHENTICATION_FAILED,
    "incorrect_otp": FailureCategory.AUTHENTICATION_FAILED,
    "incorrect_pin": FailureCategory.AUTHENTICATION_FAILED,
    "incorrect_atm_pin": FailureCategory.AUTHENTICATION_FAILED,
    "otp_expired": FailureCategory.AUTHENTICATION_FAILED,
    "otp_attempts_exceeded": FailureCategory.AUTHENTICATION_FAILED,
    "pin_attempts_exceeded": FailureCategory.AUTHENTICATION_FAILED,
    "pin_not_set": FailureCategory.AUTHENTICATION_FAILED,
    "insufficient_funds": FailureCategory.INSUFFICIENT_FUNDS,
    "transaction_limit_exceeded": FailureCategory.LIMIT_EXCEEDED,
    "transaction_daily_limit_exceeded": FailureCategory.LIMIT_EXCEEDED,
    "transaction_frequency_limit_exceeded": FailureCategory.LIMIT_EXCEEDED,
    "emi_greater_than_max_amount": FailureCategory.LIMIT_EXCEEDED,
    "card_expired": FailureCategory.INVALID_PAYMENT_DETAILS,
    "card_number_invalid": FailureCategory.INVALID_PAYMENT_DETAILS,
    "card_type_invalid": FailureCategory.INVALID_PAYMENT_DETAILS,
    "incorrect_card_details": FailureCategory.INVALID_PAYMENT_DETAILS,
    "incorrect_card_expiry_date": FailureCategory.INVALID_PAYMENT_DETAILS,
    "incorrect_cardholder_name": FailureCategory.INVALID_PAYMENT_DETAILS,
    "incorrect_cvv": FailureCategory.INVALID_PAYMENT_DETAILS,
    "invalid_vpa": FailureCategory.INVALID_PAYMENT_DETAILS,
    "bank_account_invalid": FailureCategory.INVALID_PAYMENT_DETAILS,
    "card_declined": FailureCategory.DECLINED,
    "debit_instrument_blocked": FailureCategory.DECLINED,
    "card_not_enrolled": FailureCategory.DECLINED,
    "international_transaction_not_allowed": FailureCategory.DECLINED,
    "transaction_on_vpa_restricted": FailureCategory.DECLINED,
    "authorisation_declined_by_psp": FailureCategory.DECLINED,
    "user_not_eligible": FailureCategory.DECLINED,
    "payment_risk_check_failed": FailureCategory.RISK_DECLINED,
    "payment_timed_out": FailureCategory.TIMED_OUT,
    "bank_technical_error": FailureCategory.TECHNICAL_ERROR,
    "bank_not_available": FailureCategory.TECHNICAL_ERROR,
    "bank_cutoff_in_progress": FailureCategory.TECHNICAL_ERROR,
    "upi_app_technical_error": FailureCategory.TECHNICAL_ERROR,
    "payment_failed": FailureCategory.TECHNICAL_ERROR,
    "server_error": FailureCategory.TECHNICAL_ERROR,
    "payment_method_not_enabled": FailureCategory.MERCHANT_CONFIGURATION,
    "bank_not_enabled": FailureCategory.MERCHANT_CONFIGURATION,
    "card_network_not_enabled": FailureCategory.MERCHANT_CONFIGURATION,
    "upi_collect_not_enabled": FailureCategory.MERCHANT_CONFIGURATION,
    "upi_intent_not_enabled": FailureCategory.MERCHANT_CONFIGURATION,
    "merchant_not_activated": FailureCategory.MERCHANT_CONFIGURATION,
}

# Webhook events (webhooks__payments.md, webhooks__orders.md, webhooks__refunds.md).
INTENT_EVENTS = frozenset(
    {"payment.authorized", "payment.captured", "payment.failed", "order.paid"}
)
REFUND_EVENTS = frozenset({"refund.created", "refund.processed", "refund.failed"})


# --- requests -------------------------------------------------------------------------------


def check_amount(amount: Money) -> str | None:
    """orders/payments/refunds entity docs: 3-decimal currencies must end in 0."""
    if currency_exponent(amount.currency) == 3 and amount.amount_minor % 10 != 0:
        return f"Razorpay requires the last decimal of {amount.currency} amounts to be 0"
    return None


def check_create(request: CreatePaymentRequest) -> str | None:
    if any(k.startswith(NOTE_PREFIX) for k in request.metadata):
        return f"metadata keys starting with {NOTE_PREFIX!r} are reserved"
    return None


def create_payment_body(
    request: CreatePaymentRequest, settings: Mapping[str, Any]
) -> dict[str, Any]:
    capture: dict[str, Any]
    if request.capture_mode is CaptureMode.AUTOMATIC:
        capture = {
            "capture": "automatic",
            "capture_options": {
                "automatic_expiry_period": settings["automatic_capture_expiry_minutes"],
                "manual_expiry_period": settings["manual_capture_expiry_minutes"],
                "refund_speed": settings["uncaptured_refund_speed"],
            },
        }
    else:
        capture = {
            "capture": "manual",
            "capture_options": {
                "manual_expiry_period": settings["manual_capture_expiry_minutes"],
                "refund_speed": settings["uncaptured_refund_speed"],
            },
        }
    return {
        "amount": request.amount.amount_minor,
        "currency": request.amount.currency,
        "receipt": request.merchant_reference,
        "notes": {
            **request.metadata,
            NOTE_IDEMPOTENCY_KEY: request.idempotency_key,
            NOTE_CAPTURE_MODE: request.capture_mode.value,
        },
        "payment": capture,
    }


def capture_body(attempt: PaymentAttempt, amount: Money) -> dict[str, Any]:
    # api__payments__capture.md: amount and currency both required.
    return {"amount": amount.amount_minor, "currency": amount.currency}


def refund_body(amount: Money, reason: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"amount": amount.amount_minor}
    if reason:
        body["notes"] = {"reason": reason}
    return body


# --- responses ------------------------------------------------------------------------------


def _notes(entity: Json) -> dict[str, str]:
    # Razorpay returns `notes` as {} or, when empty, sometimes as [] (sample payloads).
    notes = entity.get("notes") or {}
    if not isinstance(notes, Mapping):
        return {}
    return {str(k): str(v) for k, v in notes.items()}


def _attempt_status(payment: Json) -> AttemptStatus:
    """Native ``refunded`` is either a refunded capture or an auto-refunded,
    never-captured hold (decision D3 -> VOIDED); ``captured`` tells them apart."""
    native = payment.get("status")
    match native:
        case "created":
            return AttemptStatus.PENDING
        case "authorized":
            return AttemptStatus.AUTHORIZED
        case "captured":
            return AttemptStatus.CAPTURED
        case "failed":
            return AttemptStatus.FAILED
        case "refunded":
            if payment.get("captured") is True:
                return AttemptStatus.CAPTURED
            return AttemptStatus.VOIDED
        case _:
            raise MappingError(f"unmapped Razorpay payment status {native!r}")


def _attempt(payment: Json) -> PaymentAttempt:
    status = _attempt_status(payment)
    amount = money(payment.get("amount"), payment.get("currency"))
    refunded = money(payment.get("amount_refunded", 0), payment.get("currency"))
    if status is AttemptStatus.VOIDED:
        refunded = Money.zero(amount.currency)  # an auto-refunded hold isn't a merchant refund
    failure = None
    if status is AttemptStatus.FAILED:
        reason = payment.get("error_reason") or None
        failure = FailureInfo(
            category=FAILURE_REASONS.get(reason or "", FailureCategory.UNKNOWN),
            gateway_code=reason,
            message=payment.get("error_description") or None,
        )
    return PaymentAttempt(
        gateway_attempt_id=str(payment["id"]),
        status=status,
        amount=amount,
        amount_refunded=refunded,
        method=payment.get("method") or None,
        failure=failure,
        created_at=from_unix(payment.get("created_at")),
    )


def intent(primary: Json, related: Mapping[str, Json], ctx: MapperContext) -> PaymentIntent:
    """An order (``primary``) and all its payments (``related["payments"]``, absent
    right after creation) -> the intent (decision D1). At most one payment per
    order is captured (``order_already_paid``)."""
    order = primary
    if "payments" in related:
        items = related["payments"].get("items")
        if not isinstance(items, list):
            raise MappingError("payments list has no items")
        payments: list[Json] = items
    else:
        payments = []
    amount = money(order.get("amount"), order.get("currency"))
    attempts = tuple(sorted((_attempt(p) for p in payments), key=lambda a: a.created_at))
    if any(p.get("order_id") not in (None, order.get("id")) for p in payments):
        raise MappingError("payment belongs to a different order")

    captured = [a for a in attempts if a.status is AttemptStatus.CAPTURED]
    # api__orders__entity.md: `paid` only once a payment is captured; stays `paid` after refunds.
    if (order.get("status") == "paid") != bool(captured):
        raise MappingError(f"order status {order.get('status')!r} disagrees with its payments")
    zero = Money.zero(amount.currency)
    amount_captured = sum((a.amount for a in captured), zero)
    amount_refunded = sum((a.amount_refunded for a in attempts), zero)

    next_action = None
    if captured:
        if amount_refunded.is_zero:
            status = IntentStatus.CAPTURED
        elif amount_refunded < amount_captured:
            status = IntentStatus.PARTIALLY_REFUNDED
        else:
            status = IntentStatus.REFUNDED
    elif any(a.status is AttemptStatus.AUTHORIZED for a in attempts):
        status = IntentStatus.AUTHORIZED
    elif any(a.status is AttemptStatus.VOIDED for a in attempts):
        status = IntentStatus.VOIDED
    else:
        status = IntentStatus.REQUIRES_ACTION
        next_action = ClientCheckout(
            gateway=ctx.gateway,
            params={"key_id": ctx.public["key_id"], "order_id": str(order["id"])},
        )

    notes = _notes(order)
    mode = notes.get(NOTE_CAPTURE_MODE)
    return PaymentIntent(
        gateway=ctx.gateway,
        gateway_intent_id=str(order["id"]),
        merchant_reference=order.get("receipt") or None,
        status=status,
        capture_mode=CaptureMode(mode) if mode in {m.value for m in CaptureMode} else None,
        amount=amount,
        amount_captured=amount_captured,
        amount_refunded=amount_refunded,
        attempts=attempts,
        next_action=next_action,
        created_at=from_unix(order.get("created_at")),
        metadata={k: v for k, v in notes.items() if not k.startswith(NOTE_PREFIX)},
    )


def refund(entity: Json, ctx: MapperContext) -> Refund:
    """api__refunds__entity.md: pending | processed | failed."""
    native = entity.get("status")
    match native:
        case "pending":
            status = RefundStatus.PENDING
        case "processed":
            status = RefundStatus.SUCCEEDED
        case "failed":
            status = RefundStatus.FAILED
        case _:
            raise MappingError(f"unmapped Razorpay refund status {native!r}")
    return Refund(
        gateway=ctx.gateway,
        gateway_refund_id=str(entity["id"]),
        gateway_attempt_id=str(entity["payment_id"]),
        status=status,
        amount=money(entity.get("amount"), entity.get("currency")),
        failure=FailureInfo(category=FailureCategory.UNKNOWN)
        if status is RefundStatus.FAILED
        else None,
        created_at=from_unix(entity.get("created_at")),
    )


def _entity(body: Json, name: str) -> Json:
    payload = body.get("payload")
    section = payload.get(name) if isinstance(payload, Mapping) else None
    found = section.get("entity") if isinstance(section, Mapping) else None
    if found is None:
        raise MappingError(f"webhook payload has no {name} entity")
    if not isinstance(found, Mapping):
        raise MappingError(f"webhook {name} entity is not an object")
    return found


def webhook_event(body: Json, event_id: str, ctx: MapperContext) -> WebhookEvent:
    """A *verified* webhook body -> a canonical event carrying ids only."""
    name = body.get("event")
    if not isinstance(name, str) or not name:
        raise MappingError("webhook has no event name")
    occurred_at = from_unix(body.get("created_at"))
    if name in INTENT_EVENTS:
        if name == "order.paid":
            order_id = _entity(body, "order").get("id")
        else:
            order_id = _entity(body, "payment").get("order_id")
        if not order_id:  # a payment not created through an order: not ours
            return WebhookEvent(
                gateway=ctx.gateway, event_id=event_id, event_type=WebhookEventType.IGNORED,
                gateway_event_type=name, occurred_at=occurred_at,
            )  # fmt: skip
        return WebhookEvent(
            gateway=ctx.gateway, event_id=event_id, event_type=WebhookEventType.INTENT_UPDATED,
            gateway_event_type=name, occurred_at=occurred_at, gateway_intent_id=str(order_id),
        )  # fmt: skip
    if name in REFUND_EVENTS:
        entity = _entity(body, "refund")
        order_id = None
        payload = body.get("payload")
        payment = payload.get("payment") if isinstance(payload, Mapping) else None
        if isinstance(payment, Mapping) and isinstance(payment.get("entity"), Mapping):
            order_id = payment["entity"].get("order_id")
        return WebhookEvent(
            gateway=ctx.gateway, event_id=event_id, event_type=WebhookEventType.REFUND_UPDATED,
            gateway_event_type=name, occurred_at=occurred_at,
            gateway_refund_id=str(entity["id"]),
            gateway_intent_id=str(order_id) if order_id else None,
        )  # fmt: skip
    return WebhookEvent(
        gateway=ctx.gateway, event_id=event_id, event_type=WebhookEventType.IGNORED,
        gateway_event_type=name, occurred_at=occurred_at,
    )  # fmt: skip
