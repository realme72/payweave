# ruff: noqa: E501  (one scenario per line reads best as a table)
"""Customer-side Razorpay test scenarios (TEST MODE ONLY).

Cards and UPI IDs come from Razorpay's own test docs:
docs/sources/razorpay/payments__payments__test-card-details.md and
payments__payments__test-upi-details.md. Any CVV, any future expiry.
On the contact step choose "skip OTP" and never enter a real phone number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from payweave.core.models import FailureCategory
from payweave.core.status import IntentStatus


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    steps: str  # what the human does in Checkout
    expect_status: IntentStatus  # intent status after the payment (before any capture/refund)
    mode: str = "manual"  # capture mode
    expiry: int | None = None  # manual capture window in minutes (12 = Razorpay minimum)
    expect_failures: tuple[FailureCategory, ...] = field(default=())  # failed attempts, in order
    note: str = ""


SUCCESS_CARD = "4100 2800 0000 1007"
_FAIL = "pay, then choose FAILURE on the mock bank page"

SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in [
        Scenario("c1", "Card success, manual capture", f"Card {SUCCESS_CARD} → Success",
                 IntentStatus.AUTHORIZED, note="I then capture it and test refund idempotency"),
        Scenario("c2", "Card success, automatic capture", f"Card {SUCCESS_CARD} → Success",
                 IntentStatus.CAPTURED, mode="automatic", note="captured by Razorpay, no capture call"),
        Scenario("c3", "Authorisation left uncaptured (12-min window)", f"Card {SUCCESS_CARD} → Success",
                 IntentStatus.AUTHORIZED, expiry=12,
                 note="after ~12 min Razorpay auto-refunds it → expect VOIDED (re-run verify later)"),
        Scenario("f1", "Insufficient funds", f"Card 4100 2800 0008 0001, {_FAIL}",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.INSUFFICIENT_FUNDS,),
                 note="docs disagree: insufficient_fund vs insufficient_funds"),
        Scenario("f2", "Payment cancelled", f"Card 4100 2800 0007 0002, {_FAIL}",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.CUSTOMER_CANCELLED,)),
        Scenario("f3", "Card declined", f"Card 4100 2800 0006 0003, {_FAIL}",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.DECLINED,)),
        Scenario("f4", "Payment timed out", f"Card 4100 2800 0009 0000, {_FAIL}",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.TIMED_OUT,)),
        Scenario("f5", "Authentication failed", f"Card 4100 2800 0000 0009, {_FAIL}",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.AUTHENTICATION_FAILED,)),
        Scenario("f6", "Gateway technical error", f"Card 4100 2800 0002 0007, {_FAIL}",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.TECHNICAL_ERROR,)),
        Scenario("r1", "Fail, then retry and succeed on the SAME order",
                 f"Card 4100 2800 0008 0001 → Failure; then in the same Checkout retry with {SUCCESS_CARD} → Success",
                 IntentStatus.AUTHORIZED, expect_failures=(FailureCategory.INSUFFICIENT_FUNDS,),
                 note="one intent, two attempts"),
        Scenario("n1", "Netbanking success", "Netbanking → any bank → Success", IntentStatus.AUTHORIZED),
        Scenario("n2", "Netbanking failure", "Netbanking → any bank → Failure",
                 IntentStatus.REQUIRES_ACTION, expect_failures=(FailureCategory.UNKNOWN,),
                 note="category recorded as observed"),
        Scenario("u1", "UPI success", "UPI → ID success@razorpay", IntentStatus.AUTHORIZED,
                 note="UPI collect was deprecated by NPCI in Feb 2026; may be unavailable"),
        Scenario("u2", "UPI failure", "UPI → ID failure@razorpay", IntentStatus.REQUIRES_ACTION,
                 expect_failures=(FailureCategory.UNKNOWN,), note="category recorded as observed"),
        Scenario("d1", "Customer closes Checkout without paying", "Open Checkout, then close it (x)",
                 IntentStatus.REQUIRES_ACTION),
    ]
}  # fmt: skip
