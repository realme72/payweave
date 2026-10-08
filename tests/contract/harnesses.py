"""Harnesses: one per gateway the contract suite runs against."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from payweave.core.capabilities import Gateway
from payweave.testing.fake import EVENT_ID_HEADER, SIGNATURE_HEADER, FakeGateway

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


# --- Razorpay ------------------------------------------------------------------------
# The simulator lives in payweave.testing.razorpay so the verification gates can use it.

RAZORPAY_PACKAGE = Path(__file__).parents[2] / "packages" / "razorpay" / "v1"
