"""Razorpay native data -> canonical models. Pure functions, no I/O.

See docs/gateways/razorpay/NATIVE_MODEL.md for the native model and decisions.
Every mapping is total: an unknown native status or event raises instead of
being guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from payweave.core.errors import UnknownGatewayError
from payweave.core.models import (
    CaptureMode,
    ClientCheckout,
    FailureCategory,
    FailureInfo,
    PaymentAttempt,
    PaymentIntent,
    Refund,
    WebhookEvent,
    WebhookEventType,
)
from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus, RefundStatus

type Json = Mapping[str, Any]

# Notes keys PayWeave writes on every order it creates.
NOTE_PREFIX = "pw_"
NOTE_IDEMPOTENCY_KEY = "pw_idempotency_key"
NOTE_CAPTURE_MODE = "pw_capture_mode"

# errors__payments__list.md `reason` -> category. Unlisted reasons map to UNKNOWN.
_F = FailureCategory
FAILURE_REASONS: Mapping[str, FailureCategory] = MappingProxyType(
    {
        "payment_cancelled": _F.CUSTOMER_CANCELLED,
        "authentication_failed": _F.AUTHENTICATION_FAILED,
        "incorrect_otp": _F.AUTHENTICATION_FAILED,
        "incorrect_pin": _F.AUTHENTICATION_FAILED,
        "incorrect_atm_pin": _F.AUTHENTICATION_FAILED,
        "otp_expired": _F.AUTHENTICATION_FAILED,
        "otp_attempts_exceeded": _F.AUTHENTICATION_FAILED,
        "pin_attempts_exceeded": _F.AUTHENTICATION_FAILED,
        "pin_not_set": _F.AUTHENTICATION_FAILED,
        "insufficient_funds": _F.INSUFFICIENT_FUNDS,
        "transaction_limit_exceeded": _F.LIMIT_EXCEEDED,
        "transaction_daily_limit_exceeded": _F.LIMIT_EXCEEDED,
        "transaction_frequency_limit_exceeded": _F.LIMIT_EXCEEDED,
        "emi_greater_than_max_amount": _F.LIMIT_EXCEEDED,
        "card_expired": _F.INVALID_PAYMENT_DETAILS,
        "card_number_invalid": _F.INVALID_PAYMENT_DETAILS,
        "card_type_invalid": _F.INVALID_PAYMENT_DETAILS,
        "incorrect_card_details": _F.INVALID_PAYMENT_DETAILS,
        "incorrect_card_expiry_date": _F.INVALID_PAYMENT_DETAILS,
        "incorrect_cardholder_name": _F.INVALID_PAYMENT_DETAILS,
        "incorrect_cvv": _F.INVALID_PAYMENT_DETAILS,
        "invalid_vpa": _F.INVALID_PAYMENT_DETAILS,
        "bank_account_invalid": _F.INVALID_PAYMENT_DETAILS,
        "card_declined": _F.DECLINED,
        "debit_instrument_blocked": _F.DECLINED,
        "card_not_enrolled": _F.DECLINED,
        "international_transaction_not_allowed": _F.DECLINED,
        "transaction_on_vpa_restricted": _F.DECLINED,
        "authorisation_declined_by_psp": _F.DECLINED,
        "user_not_eligible": _F.DECLINED,
        "payment_risk_check_failed": _F.RISK_DECLINED,
        "payment_timed_out": _F.TIMED_OUT,
        "bank_technical_error": _F.TECHNICAL_ERROR,
        "bank_not_available": _F.TECHNICAL_ERROR,
        "bank_cutoff_in_progress": _F.TECHNICAL_ERROR,
        "upi_app_technical_error": _F.TECHNICAL_ERROR,
        "payment_failed": _F.TECHNICAL_ERROR,
        "server_error": _F.TECHNICAL_ERROR,
        "payment_method_not_enabled": _F.MERCHANT_CONFIGURATION,
        "bank_not_enabled": _F.MERCHANT_CONFIGURATION,
        "card_network_not_enabled": _F.MERCHANT_CONFIGURATION,
        "upi_collect_not_enabled": _F.MERCHANT_CONFIGURATION,
        "upi_intent_not_enabled": _F.MERCHANT_CONFIGURATION,
        "merchant_not_activated": _F.MERCHANT_CONFIGURATION,
    }
)

# Webhook events (webhooks__payments.md, webhooks__orders.md, webhooks__refunds.md).
_INTENT_EVENTS = frozenset(
    {"payment.authorized", "payment.captured", "payment.failed", "order.paid"}
)
_REFUND_EVENTS = frozenset({"refund.created", "refund.processed", "refund.failed"})


class RazorpayMappingError(UnknownGatewayError):
    """Razorpay returned something our mapping doesn't cover. Fail closed."""


def _err(message: str) -> RazorpayMappingError:
    return RazorpayMappingError(message, gateway="razorpay")


def from_unix(ts: object) -> datetime:
    if not isinstance(ts, int) or isinstance(ts, bool):
        raise _err(f"expected a unix timestamp, got {ts!r}")
    return datetime.fromtimestamp(ts, tz=UTC)


def _money(amount: object, currency: object) -> Money:
    if not isinstance(amount, int) or isinstance(amount, bool) or not isinstance(currency, str):
        raise _err(f"bad amount/currency: {amount!r} {currency!r}")
    return Money.of(amount, currency)


def _notes(entity: Json) -> dict[str, str]:
    # Razorpay returns `notes` as {} or, when empty, sometimes as [] (sample payloads).
    notes = entity.get("notes") or {}
    if not isinstance(notes, Mapping):
        return {}
    return {str(k): str(v) for k, v in notes.items()}


def failure_category(reason: str | None) -> FailureCategory:
    return FAILURE_REASONS.get(reason or "", FailureCategory.UNKNOWN)


def attempt_status(payment: Json) -> AttemptStatus:
    """payments__payments.md states -> attempt status.

    Native ``refunded`` covers two different things: a refunded *captured*
    payment, or an authorisation auto-refunded because it was never captured
    (decision D3 -> VOIDED). The ``captured`` flag tells them apart.
    """
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
            return (
                AttemptStatus.CAPTURED if payment.get("captured") is True else AttemptStatus.VOIDED
            )
        case _:
            raise _err(f"unmapped Razorpay payment status {native!r}")


def attempt_from_payment(payment: Json) -> PaymentAttempt:
    status = attempt_status(payment)
    amount = _money(payment.get("amount"), payment.get("currency"))
    refunded = _money(payment.get("amount_refunded", 0), payment.get("currency"))
    if status is AttemptStatus.VOIDED:
        refunded = Money.zero(amount.currency)  # an auto-refunded hold isn't a merchant refund
    failure = None
    if status is AttemptStatus.FAILED:
        reason = payment.get("error_reason") or None
        failure = FailureInfo(
            category=failure_category(reason),
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


def intent_from_order(
    order: Json, payments: list[Json], *, gateway: str, key_id: str
) -> PaymentIntent:
    """Derive the intent from an order and all its payments (decision D1).

    Razorpay allows one successful payment per order (errors__payments__list.md
    ``order_already_paid``), so at most one attempt is captured.
    """
    amount = _money(order.get("amount"), order.get("currency"))
    attempts = tuple(
        sorted((attempt_from_payment(p) for p in payments), key=lambda a: a.created_at)
    )
    if any(p.get("order_id") not in (None, order.get("id")) for p in payments):
        raise _err("payment belongs to a different order")

    captured = [a for a in attempts if a.status is AttemptStatus.CAPTURED]
    # api__orders__entity.md: an order is `paid` only once a payment is captured,
    # and stays `paid` after refunds. Any disagreement means our view is wrong.
    if (order.get("status") == "paid") != bool(captured):
        raise _err(f"order status {order.get('status')!r} disagrees with its payments")
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
            gateway=gateway, params={"key_id": key_id, "order_id": str(order["id"])}
        )

    notes = _notes(order)
    mode = notes.get(NOTE_CAPTURE_MODE)
    return PaymentIntent(
        gateway=gateway,
        gateway_intent_id=str(order["id"]),
        merchant_reference=order.get("receipt") or None,
        status=status,
        capture_mode=CaptureMode(mode) if mode in set(CaptureMode) else None,
        amount=amount,
        amount_captured=amount_captured,
        amount_refunded=amount_refunded,
        attempts=attempts,
        next_action=next_action,
        created_at=from_unix(order.get("created_at")),
        metadata={k: v for k, v in notes.items() if not k.startswith(NOTE_PREFIX)},
    )


def refund_from_entity(refund: Json, *, gateway: str) -> Refund:
    """api__refunds__entity.md: pending | processed | failed."""
    native = refund.get("status")
    match native:
        case "pending":
            status = RefundStatus.PENDING
        case "processed":
            status = RefundStatus.SUCCEEDED
        case "failed":
            status = RefundStatus.FAILED
        case _:
            raise _err(f"unmapped Razorpay refund status {native!r}")
    return Refund(
        gateway=gateway,
        gateway_refund_id=str(refund["id"]),
        gateway_attempt_id=str(refund["payment_id"]),
        status=status,
        amount=_money(refund.get("amount"), refund.get("currency")),
        failure=FailureInfo(category=FailureCategory.UNKNOWN)
        if status is RefundStatus.FAILED
        else None,
        created_at=from_unix(refund.get("created_at")),
    )


def _entity(body: Json, name: str) -> Json:
    try:
        entity = body["payload"][name]["entity"]
    except (KeyError, TypeError):
        raise _err(f"webhook payload has no {name} entity") from None
    if not isinstance(entity, Mapping):
        raise _err(f"webhook {name} entity is not an object")
    return entity


def event_from_webhook(body: Json, *, event_id: str, gateway: str) -> WebhookEvent:
    """Turn a *verified* webhook body into a canonical event carrying ids only."""
    name = body.get("event")
    if not isinstance(name, str) or not name:
        raise _err("webhook has no event name")
    base: dict[str, Any] = {
        "gateway": gateway,
        "event_id": event_id,
        "gateway_event_type": name,
        "occurred_at": from_unix(body.get("created_at")),
    }
    if name in _INTENT_EVENTS:
        if name == "order.paid":
            order_id = _entity(body, "order").get("id")
        else:
            order_id = _entity(body, "payment").get("order_id")
        if not order_id:  # a payment not created through an order: not ours
            return WebhookEvent(event_type=WebhookEventType.IGNORED, **base)
        return WebhookEvent(
            event_type=WebhookEventType.INTENT_UPDATED, gateway_intent_id=str(order_id), **base
        )
    if name in _REFUND_EVENTS:
        refund = _entity(body, "refund")
        order_id = None
        payment = body.get("payload", {}).get("payment")
        if isinstance(payment, Mapping) and isinstance(payment.get("entity"), Mapping):
            order_id = payment["entity"].get("order_id")
        return WebhookEvent(
            event_type=WebhookEventType.REFUND_UPDATED,
            gateway_refund_id=str(refund["id"]),
            gateway_intent_id=str(order_id) if order_id else None,
            **base,
        )
    return WebhookEvent(event_type=WebhookEventType.IGNORED, **base)
