import pytest

from payweave.core.models import FailureCategory, WebhookEventType
from payweave.core.status import AttemptStatus, IntentStatus, RefundStatus
from payweave.gateways.razorpay import mapping
from tests.gateways.razorpay import fixtures as fx


def intent(order: dict, *items: dict):  # type: ignore[no-untyped-def,type-arg]
    return mapping.intent_from_order(order, list(items), gateway="razorpay", key_id=fx.KEY_ID)


# --- status mapping is total and follows the docs ----------------------------


@pytest.mark.parametrize(
    ("native", "captured", "expected"),
    [
        ("created", False, AttemptStatus.PENDING),
        ("authorized", False, AttemptStatus.AUTHORIZED),
        ("captured", True, AttemptStatus.CAPTURED),
        ("failed", False, AttemptStatus.FAILED),
        ("refunded", True, AttemptStatus.CAPTURED),  # refunded after capture
        ("refunded", False, AttemptStatus.VOIDED),  # D3: uncaptured auto-refund
    ],
)
def test_attempt_status(native: str, captured: bool, expected: AttemptStatus) -> None:
    assert mapping.attempt_status(fx.payment(native, captured=captured)) is expected


def test_unknown_native_status_fails_closed() -> None:
    with pytest.raises(mapping.RazorpayMappingError):
        mapping.attempt_status(fx.payment("disputed"))


def test_new_order_requires_checkout_with_public_params_only() -> None:
    result = intent(fx.order())
    assert result.status is IntentStatus.REQUIRES_ACTION
    assert result.next_action is not None
    assert result.next_action.type == "client_checkout"
    assert result.next_action.params == {"key_id": fx.KEY_ID, "order_id": fx.ORDER_ID}
    assert result.metadata == {"cart": "42"}  # pw_* notes are ours, hidden from metadata
    assert result.capture_mode == "manual"


def test_failed_attempt_does_not_fail_intent() -> None:
    result = intent(fx.order("attempted"), fx.payment("failed"))
    assert result.status is IntentStatus.REQUIRES_ACTION  # customer may retry
    failure = result.attempts[0].failure
    assert failure is not None
    assert failure.category is FailureCategory.CUSTOMER_CANCELLED
    assert failure.gateway_code == "payment_cancelled"


def test_retry_after_failure_then_success() -> None:
    result = intent(
        fx.order("paid"),
        fx.payment("captured", pid="pay_B", created_at=2),
        fx.payment("failed", pid="pay_A", created_at=1),
    )
    assert result.status is IntentStatus.CAPTURED
    assert [a.gateway_attempt_id for a in result.attempts] == ["pay_A", "pay_B"]  # sorted by time
    assert result.amount_captured.amount_minor == 50000


def test_late_authorisation_revives_intent() -> None:
    # Same payment previously reported failed, now authorized (late-authorisation.md).
    result = intent(fx.order("attempted"), fx.payment("authorized"))
    assert result.status is IntentStatus.AUTHORIZED
    assert result.attempts[0].failure is None  # stale error fields are not carried over


@pytest.mark.parametrize(
    ("refunded", "expected"),
    [
        (0, IntentStatus.CAPTURED),
        (10000, IntentStatus.PARTIALLY_REFUNDED),
        (50000, IntentStatus.REFUNDED),
    ],
)
def test_refund_progression(refunded: int, expected: IntentStatus) -> None:
    native = "refunded" if refunded == 50000 else "captured"
    result = intent(fx.order("paid"), fx.payment(native, refunded=refunded))
    assert result.status is expected
    assert result.amount_refunded.amount_minor == refunded


def test_uncaptured_auto_refund_is_voided() -> None:
    result = intent(fx.order("attempted"), fx.payment("refunded", captured=False, refunded=50000))
    assert result.status is IntentStatus.VOIDED
    assert result.amount_refunded.is_zero  # not a merchant refund


def test_paid_order_without_captured_payment_is_inconsistent() -> None:
    with pytest.raises(mapping.RazorpayMappingError):
        intent(fx.order("paid"), fx.payment("authorized"))


def test_payment_from_other_order_rejected() -> None:
    with pytest.raises(mapping.RazorpayMappingError):
        intent(fx.order(), fx.payment("captured", order_id="order_OTHER"))


def test_empty_notes_list_handled() -> None:
    assert intent(fx.order(notes=[])).metadata == {}


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("insufficient_funds", FailureCategory.INSUFFICIENT_FUNDS),
        ("incorrect_otp", FailureCategory.AUTHENTICATION_FAILED),
        ("payment_risk_check_failed", FailureCategory.RISK_DECLINED),
        ("something_new", FailureCategory.UNKNOWN),
        (None, FailureCategory.UNKNOWN),
    ],
)
def test_failure_reasons(reason: str | None, category: FailureCategory) -> None:
    assert mapping.failure_category(reason) is category


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("pending", RefundStatus.PENDING),
        ("processed", RefundStatus.SUCCEEDED),
        ("failed", RefundStatus.FAILED),
    ],
)
def test_refund_status(native: str, expected: RefundStatus) -> None:
    assert mapping.refund_from_entity(fx.refund(native), gateway="razorpay").status is expected


# --- webhooks -----------------------------------------------------------------


def event(body: dict):  # type: ignore[no-untyped-def,type-arg]
    return mapping.event_from_webhook(body, event_id="evt_1", gateway="razorpay")


@pytest.mark.parametrize("name", ["payment.authorized", "payment.captured", "payment.failed"])
def test_payment_events_point_at_intent(name: str) -> None:
    e = event(fx.webhook(name, payment=fx.payment("authorized")))
    assert (e.event_type, e.gateway_intent_id) == (WebhookEventType.INTENT_UPDATED, fx.ORDER_ID)


def test_order_paid_event() -> None:
    e = event(fx.webhook("order.paid", order=fx.order("paid"), payment=fx.payment("captured")))
    assert e.gateway_intent_id == fx.ORDER_ID


def test_refund_event() -> None:
    e = event(fx.webhook("refund.processed", refund=fx.refund(), payment=fx.payment("captured")))
    assert e.event_type is WebhookEventType.REFUND_UPDATED
    assert (e.gateway_refund_id, e.gateway_intent_id) == ("rfnd_FgRAHdNOM4ZVbO", fx.ORDER_ID)


def test_payment_without_order_and_unknown_events_ignored() -> None:
    assert event(
        fx.webhook("payment.captured", payment=fx.payment("captured", order_id=None))
    ).event_type is (WebhookEventType.IGNORED)
    assert (
        event(fx.webhook("refund.speed_changed", refund=fx.refund())).event_type
        is WebhookEventType.IGNORED
    )
    assert event(fx.webhook("payment.downtime.started")).event_type is WebhookEventType.IGNORED
