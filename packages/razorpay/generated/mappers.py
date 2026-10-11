"""Gateway mappers, written by the PayWeave integration agent. Pure functions only."""

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


def create_payment_body(request: CreatePaymentRequest, settings: Mapping[str, Any]) -> dict[str, Any]:
    # Determine capture mode string
    if request.capture_mode is CaptureMode.AUTOMATIC:
        capture_mode_str: str = "automatic"
        automatic_expiry: int = int(settings.get("automatic_capture_expiry_minutes", 12))
        manual_expiry: int = int(settings.get("manual_capture_expiry_minutes", 7200))
        capture_options: dict[str, Any] = {
            "automatic_expiry_period": automatic_expiry,
            "manual_expiry_period": manual_expiry,
            "refund_speed": str(settings.get("uncaptured_refund_speed", "normal")),
        }
    elif request.capture_mode is CaptureMode.MANUAL:
        capture_mode_str = "manual"
        manual_expiry = int(settings.get("manual_capture_expiry_minutes", 7200))
        capture_options = {
            "manual_expiry_period": manual_expiry,
            "refund_speed": str(settings.get("uncaptured_refund_speed", "normal")),
        }
    else:
        raise MappingError(f"unsupported capture mode: {request.capture_mode!r}")

    # Build notes/metadata with reserved keys
    notes: dict[str, str] = dict(request.metadata)
    notes["pw_idempotency_key"] = request.idempotency_key
    notes["pw_capture_mode"] = request.capture_mode.value

    body: dict[str, Any] = {
        "amount": request.amount.amount_minor,
        "currency": request.amount.currency,
        "receipt": request.merchant_reference,
        "notes": notes,
        "payment": {
            "capture": capture_mode_str,
            "capture_options": capture_options,
        },
    }
    return body


_INTENT_FAILURE_REASONS: dict[str, FailureCategory] = {
    "payment_cancelled": FailureCategory.CUSTOMER_CANCELLED,
    "customer_cancelled": FailureCategory.CUSTOMER_CANCELLED,
    "user_cancelled": FailureCategory.CUSTOMER_CANCELLED,
    "payment_authentication_failed": FailureCategory.AUTHENTICATION_FAILED,
    "authentication_failed": FailureCategory.AUTHENTICATION_FAILED,
    "payment_verification_failed": FailureCategory.AUTHENTICATION_FAILED,
    "invalid_otp": FailureCategory.AUTHENTICATION_FAILED,
    "invalid_pin": FailureCategory.AUTHENTICATION_FAILED,
    "incorrect_pin": FailureCategory.AUTHENTICATION_FAILED,
    "otp_expired": FailureCategory.AUTHENTICATION_FAILED,
    "insufficient_funds": FailureCategory.INSUFFICIENT_FUNDS,
    "limit_exceeded": FailureCategory.LIMIT_EXCEEDED,
    "card_limit_exceeded": FailureCategory.LIMIT_EXCEEDED,
    "transaction_limit_exceeded": FailureCategory.LIMIT_EXCEEDED,
    "invalid_card_details": FailureCategory.INVALID_PAYMENT_DETAILS,
    "invalid_card": FailureCategory.INVALID_PAYMENT_DETAILS,
    "invalid_card_number": FailureCategory.INVALID_PAYMENT_DETAILS,
    "invalid_cvv": FailureCategory.INVALID_PAYMENT_DETAILS,
    "invalid_expiry_date": FailureCategory.INVALID_PAYMENT_DETAILS,
    "invalid_expiry": FailureCategory.INVALID_PAYMENT_DETAILS,
    "expired_card": FailureCategory.INVALID_PAYMENT_DETAILS,
    "card_expired": FailureCategory.INVALID_PAYMENT_DETAILS,
    "card_declined": FailureCategory.DECLINED,
    "payment_declined": FailureCategory.DECLINED,
    "card_inactive": FailureCategory.DECLINED,
    "issuer_declined": FailureCategory.DECLINED,
    "card_issuer_declined": FailureCategory.DECLINED,
    "card_not_supported": FailureCategory.DECLINED,
    "declined": FailureCategory.DECLINED,
    "risk_check_failed": FailureCategory.RISK_DECLINED,
    "risk_declined": FailureCategory.RISK_DECLINED,
    "suspected_fraud": FailureCategory.RISK_DECLINED,
    "timed_out": FailureCategory.TIMED_OUT,
    "payment_timed_out": FailureCategory.TIMED_OUT,
    "gateway_timed_out": FailureCategory.TIMED_OUT,
    "session_timed_out": FailureCategory.TIMED_OUT,
    "payment_failed": FailureCategory.TECHNICAL_ERROR,
    "gateway_error": FailureCategory.TECHNICAL_ERROR,
    "technical_error": FailureCategory.TECHNICAL_ERROR,
    "bank_technical_error": FailureCategory.TECHNICAL_ERROR,
    "network_error": FailureCategory.TECHNICAL_ERROR,
    "network_failure": FailureCategory.TECHNICAL_ERROR,
    "server_error": FailureCategory.TECHNICAL_ERROR,
    "merchant_configuration": FailureCategory.MERCHANT_CONFIGURATION,
    "account_inactive": FailureCategory.MERCHANT_CONFIGURATION,
    "currency_not_supported": FailureCategory.MERCHANT_CONFIGURATION,
}


def _intent_attempt_status(p: Json) -> AttemptStatus:
    native_status: str = str(p["status"])
    if native_status == "refunded":
        if not p.get("captured", False):
            return AttemptStatus.VOIDED
        return AttemptStatus.CAPTURED
    if native_status == "captured":
        return AttemptStatus.CAPTURED
    if native_status == "authorized":
        return AttemptStatus.AUTHORIZED
    if native_status == "created":
        return AttemptStatus.PENDING
    if native_status == "failed":
        return AttemptStatus.FAILED
    raise MappingError(f"unknown payment status: {native_status!r}")


def intent(primary: Json, related: Mapping[str, Json], ctx: MapperContext) -> PaymentIntent:
    intent_id = str(primary["id"])
    created_at = from_unix(primary["created_at"])
    currency = str(primary["currency"])
    amount = money(primary["amount"], currency)

    notes = primary.get("notes")
    raw_notes: Mapping[str, Any] = notes if isinstance(notes, Mapping) else {}
    metadata: dict[str, str] = {
        str(k): str(v)
        for k, v in raw_notes.items()
        if not str(k).startswith("pw_")
    }
    capture_mode: CaptureMode | None = None
    if "pw_capture_mode" in raw_notes:
        capture_mode = CaptureMode(str(raw_notes["pw_capture_mode"]))

    receipt = primary.get("receipt")
    merchant_reference: str | None = str(receipt) if receipt is not None else None

    payments_json = related.get("payments")
    payment_items: list[Json] = []
    if payments_json is not None:
        if isinstance(payments_json, list):
            payment_items = payments_json
        elif isinstance(payments_json, Mapping) and "items" in payments_json:
            raw_items = payments_json["items"]
            if isinstance(raw_items, list):
                payment_items = raw_items
            else:
                raise MappingError("unexpected shape for related['payments']['items']")
        else:
            raise MappingError("unexpected shape for related['payments']")

    attempts_list: list[PaymentAttempt] = []
    for p in payment_items:
        attempt_id = str(p["id"])
        attempt_status = _intent_attempt_status(p)
        payment_currency = str(p["currency"])
        attempt_amount = money(p["amount"], payment_currency)
        if attempt_status is AttemptStatus.CAPTURED:
            attempt_refunded = money(p.get("amount_refunded", 0), payment_currency)
        else:
            attempt_refunded = Money.zero(payment_currency)

        method = str(p["method"]) if p.get("method") is not None else None
        failure: FailureInfo | None = None
        if attempt_status is AttemptStatus.FAILED:
            gateway_code: str | None = None
            if p.get("error_reason") is not None:
                gateway_code = str(p["error_reason"])
            elif p.get("error_code") is not None:
                gateway_code = str(p["error_code"])

            category = (
                _INTENT_FAILURE_REASONS.get(gateway_code.lower(), FailureCategory.UNKNOWN)
                if gateway_code is not None
                else FailureCategory.UNKNOWN
            )
            message = str(p["error_description"]) if p.get("error_description") is not None else None
            failure = FailureInfo(
                category=category,
                gateway_code=gateway_code,
                message=message,
            )

        attempts_list.append(
            PaymentAttempt(
                gateway_attempt_id=attempt_id,
                status=attempt_status,
                amount=attempt_amount,
                amount_refunded=attempt_refunded,
                method=method,
                failure=failure,
                created_at=from_unix(p["created_at"]),
            )
        )

    attempts_list.sort(key=lambda a: a.created_at)
    attempts = tuple(attempts_list)

    amount_captured = Money.zero(currency)
    amount_refunded_total = Money.zero(currency)
    has_captured = False
    has_authorized = False
    has_voided = False

    for att in attempts:
        amount_refunded_total += att.amount_refunded
        if att.status is AttemptStatus.CAPTURED:
            amount_captured += att.amount
            has_captured = True
        elif att.status is AttemptStatus.AUTHORIZED:
            has_authorized = True
        elif att.status is AttemptStatus.VOIDED:
            has_voided = True

    if has_captured:
        if amount_refunded_total.is_zero:
            intent_status = IntentStatus.CAPTURED
        elif amount_refunded_total == amount_captured:
            intent_status = IntentStatus.REFUNDED
        else:
            intent_status = IntentStatus.PARTIALLY_REFUNDED
        next_action = None
    elif has_authorized:
        intent_status = IntentStatus.AUTHORIZED
        next_action = None
    elif has_voided:
        intent_status = IntentStatus.VOIDED
        next_action = None
    else:
        intent_status = IntentStatus.REQUIRES_ACTION
        next_action = ClientCheckout(
            gateway=ctx.gateway,
            params={
                "key_id": ctx.public["key_id"],
                "order_id": intent_id,
            },
        )

    return PaymentIntent(
        gateway=ctx.gateway,
        gateway_intent_id=intent_id,
        merchant_reference=merchant_reference,
        status=intent_status,
        capture_mode=capture_mode,
        amount=amount,
        amount_captured=amount_captured,
        amount_refunded=amount_refunded_total,
        attempts=attempts,
        next_action=next_action,
        created_at=created_at,
        metadata=metadata,
    )


def capture_body(attempt: PaymentAttempt, amount: Money) -> dict[str, Any]:
    return {
        "amount": amount.amount_minor,
        "currency": amount.currency,
    }


def refund_body(amount: Money, reason: str | None) -> dict[str, Any]:
    """Create the JSON body for a Razorpay refund request.

    Razorpay expects the amount in minor currency units under the ``amount`` key.
    An optional ``notes`` object can be supplied for free‑form key/value pairs.
    If a ``reason`` is provided we store it under the ``reason`` key inside ``notes``.
    If ``reason`` is ``None`` the ``notes`` field is omitted entirely.
    """
    body: dict[str, Any] = {"amount": amount.amount_minor}
    if reason is not None:
        body["notes"] = {"reason": reason}
    return body


def refund(entity: Json, ctx: MapperContext) -> Refund:
    return Refund(
        gateway=ctx.gateway,
        gateway_refund_id=_refund_str(entity, "id"),
        gateway_attempt_id=_refund_str(entity, "payment_id"),
        status=_refund_status(entity.get("status")),
        amount=money(entity.get("amount"), entity.get("currency")),
        failure=_refund_failure(entity.get("status")),
        created_at=from_unix(entity.get("created_at")),
    )


def _refund_str(entity: Json, key: str) -> str:
    value: Any = entity.get(key)
    if not isinstance(value, str):
        raise MappingError(f"refund: missing or invalid string field {key!r}")
    return value


def _refund_status(value: Any) -> RefundStatus:
    if value == "pending":
        return RefundStatus.PENDING
    if value == "processed":
        return RefundStatus.SUCCEEDED
    if value == "failed":
        return RefundStatus.FAILED
    raise MappingError(f"refund: unrecognised status {value!r}")


def _refund_failure(value: Any) -> FailureInfo | None:
    if value == "failed":
        return FailureInfo(category=FailureCategory.UNKNOWN)
    return None


def _webhook_event_extract_entity(obj: Any) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise MappingError("Expected a mapping for entity container")
    entity = obj.get("entity")
    if not isinstance(entity, Mapping):
        raise MappingError("Missing 'entity' object in webhook payload")
    return entity


def _webhook_event_order_id_from_payment(payment_entity: Mapping[str, Any]) -> str | None:
    oid = payment_entity.get("order_id")
    return oid if isinstance(oid, str) and oid else None


def _webhook_event_order_id_from_order(order_entity: Mapping[str, Any]) -> str | None:
    oid = order_entity.get("id")
    return oid if isinstance(oid, str) and oid else None


def webhook_event(body: Json, event_id: str, ctx: MapperContext) -> WebhookEvent:
    if not isinstance(body, Mapping):
        raise MappingError("Webhook body must be a mapping")

    native_event = body.get("event")
    if not isinstance(native_event, str) or not native_event:
        raise MappingError("Missing or invalid 'event' field in webhook payload")

    created_ts = body.get("created_at")
    if not isinstance(created_ts, int) or isinstance(created_ts, bool):
        raise MappingError("Missing or invalid 'created_at' field in webhook payload")
    try:
        occurred_at = from_unix(created_ts)
    except Exception as exc:
        raise MappingError(f"Invalid 'created_at' timestamp: {exc}") from exc

    payload = body.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}

    event_type: WebhookEventType = WebhookEventType.IGNORED
    intent_id: str | None = None
    refund_id: str | None = None

    if native_event.startswith("refund."):
        refund_obj = payload.get("refund")
        if refund_obj is None:
            raise MappingError("Refund webhook missing 'refund' object in payload")
        refund_entity = _webhook_event_extract_entity(refund_obj)
        rid = refund_entity.get("id")
        if not isinstance(rid, str) or not rid:
            raise MappingError("Refund entity missing 'id' field")
        refund_id = rid
        event_type = WebhookEventType.REFUND_UPDATED

        payment_obj = payload.get("payment")
        if isinstance(payment_obj, Mapping):
            payment_entity = _webhook_event_extract_entity(payment_obj)
            intent_id = _webhook_event_order_id_from_payment(payment_entity)
        if intent_id is None:
            order_obj = payload.get("order")
            if isinstance(order_obj, Mapping):
                order_entity = _webhook_event_extract_entity(order_obj)
                intent_id = _webhook_event_order_id_from_order(order_entity)

    elif native_event.startswith("payment.") or native_event.startswith("order."):
        if native_event.startswith("payment."):
            payment_obj = payload.get("payment")
            if payment_obj is None:
                raise MappingError("Payment webhook missing 'payment' object in payload")
            payment_entity = _webhook_event_extract_entity(payment_obj)
            intent_id = _webhook_event_order_id_from_payment(payment_entity)
        else:
            order_obj = payload.get("order")
            if order_obj is None:
                raise MappingError("Order webhook missing 'order' object in payload")
            order_entity = _webhook_event_extract_entity(order_obj)
            intent_id = _webhook_event_order_id_from_order(order_entity)

        if isinstance(intent_id, str):
            event_type = WebhookEventType.INTENT_UPDATED
        else:
            event_type = WebhookEventType.IGNORED
            intent_id = None
    else:
        event_type = WebhookEventType.IGNORED

    return WebhookEvent(
        gateway=ctx.gateway,
        event_id=event_id,
        event_type=event_type,
        gateway_event_type=native_event,
        occurred_at=occurred_at,
        gateway_intent_id=intent_id,
        gateway_refund_id=refund_id,
    )
