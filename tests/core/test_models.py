from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from payweave.core.models import (
    ClientCheckout,
    CreatePaymentRequest,
    FailureCategory,
    FailureInfo,
    PaymentAttempt,
    PaymentIntent,
)
from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus

NOW = datetime(2026, 10, 7, tzinfo=UTC)


def inr(n: int) -> Money:
    return Money.of(n, "INR")


def attempt(
    status: AttemptStatus, amount: int = 1000, refunded: int = 0, **kw: Any
) -> PaymentAttempt:
    if status is AttemptStatus.FAILED:
        kw.setdefault("failure", FailureInfo(category=FailureCategory.DECLINED))
    return PaymentAttempt(
        gateway_attempt_id=kw.pop("id", f"pay_{status}"),
        status=status,
        amount=inr(amount),
        amount_refunded=inr(refunded),
        created_at=NOW,
        **kw,
    )


def intent(
    status: IntentStatus, attempts: tuple[PaymentAttempt, ...] = (), **kw: Any
) -> PaymentIntent:
    captured = sum(a.amount.amount_minor for a in attempts if a.status is AttemptStatus.CAPTURED)
    refunded = sum(a.amount_refunded.amount_minor for a in attempts)
    fields: dict[str, Any] = {
        "gateway": "test",
        "gateway_intent_id": "order_1",
        "merchant_reference": "ref",
        "status": status,
        "capture_mode": None,
        "amount": inr(1000),
        "amount_captured": inr(captured),
        "amount_refunded": inr(refunded),
        "attempts": attempts,
        "created_at": NOW,
    }
    if status is IntentStatus.REQUIRES_ACTION:
        fields["next_action"] = ClientCheckout(gateway="test", params={"order_id": "order_1"})
    fields.update(kw)
    return PaymentIntent(**fields)


def test_valid_lifecycle_shapes() -> None:
    intent(IntentStatus.REQUIRES_ACTION)
    intent(IntentStatus.REQUIRES_ACTION, (attempt(AttemptStatus.FAILED),))
    intent(
        IntentStatus.AUTHORIZED,
        (attempt(AttemptStatus.FAILED, id="a"), attempt(AttemptStatus.AUTHORIZED)),
    )
    intent(IntentStatus.CAPTURED, (attempt(AttemptStatus.CAPTURED),))
    intent(IntentStatus.PARTIALLY_REFUNDED, (attempt(AttemptStatus.CAPTURED, refunded=400),))
    intent(IntentStatus.REFUNDED, (attempt(AttemptStatus.CAPTURED, refunded=1000),))


@pytest.mark.parametrize(
    "build",
    [
        # status says captured but nothing is
        lambda: intent(IntentStatus.CAPTURED),
        # refunds present but status still captured
        lambda: intent(IntentStatus.CAPTURED, (attempt(AttemptStatus.CAPTURED, refunded=1),)),
        # fully refunded but status says partial
        lambda: intent(
            IntentStatus.PARTIALLY_REFUNDED, (attempt(AttemptStatus.CAPTURED, refunded=1000),)
        ),
        # authorized intent without an authorized attempt
        lambda: intent(IntentStatus.AUTHORIZED, (attempt(AttemptStatus.FAILED),)),
        # requires_action without a next action
        lambda: intent(IntentStatus.REQUIRES_ACTION, next_action=None),
        # totals don't match attempts
        lambda: intent(
            IntentStatus.CAPTURED, (attempt(AttemptStatus.CAPTURED),), amount_captured=inr(999)
        ),
        # captured more than the intent amount
        lambda: intent(IntentStatus.CAPTURED, (attempt(AttemptStatus.CAPTURED, amount=2000),)),
        # mixed currencies
        lambda: intent(IntentStatus.REQUIRES_ACTION, amount_captured=Money.zero("USD")),
    ],
)
def test_inconsistent_intents_cannot_exist(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: attempt(AttemptStatus.FAILED, failure=None),
        lambda: attempt(
            AttemptStatus.CAPTURED, failure=FailureInfo(category=FailureCategory.UNKNOWN)
        ),
        lambda: attempt(AttemptStatus.AUTHORIZED, refunded=1),
        lambda: attempt(AttemptStatus.CAPTURED, refunded=1001),
    ],
)
def test_inconsistent_attempts_cannot_exist(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    "key", ["short", "has space in it!!", "a" * 15, "x" * 256, "key:with:colons:1"]
)
def test_idempotency_key_format(key: str) -> None:
    with pytest.raises(ValidationError):
        CreatePaymentRequest(amount=inr(100), merchant_reference="r", idempotency_key=key)


def test_zero_amount_payment_rejected() -> None:
    with pytest.raises(ValidationError):
        CreatePaymentRequest(amount=inr(0), merchant_reference="r", idempotency_key="k" * 16)


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        CreatePaymentRequest(
            amount=inr(1),
            merchant_reference="r",
            idempotency_key="k" * 16,
            surprise=1,  # type: ignore[call-arg]
        )
