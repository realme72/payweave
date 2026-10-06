"""Real Razorpay test-mode API. Run with: ``set -a; . ./.env; set +a; uv run pytest -m sandbox``."""

import os
import secrets

import pytest

from payweave.core.capabilities import PaymentCreator, PaymentReader
from payweave.core.errors import InvalidRequestError
from payweave.core.factory import GatewayFactory, GatewaySpec
from payweave.core.models import CaptureMode, CreatePaymentRequest
from payweave.core.money import Money
from payweave.core.status import IntentStatus

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(
        not os.environ.get("RAZORPAY_KEY_ID", "").startswith("rzp_test_"),
        reason="needs Razorpay TEST keys in the environment",
    ),
]


@pytest.fixture
def registry():  # type: ignore[no-untyped-def]
    factory = GatewayFactory.with_installed_providers()
    config = factory.config_from_env("razorpay")
    config.pop("webhook_secrets", None)
    return factory.build_registry([GatewaySpec(name="razorpay", type="razorpay", config=config)])


@pytest.mark.parametrize("mode", list(CaptureMode))
async def test_create_and_fetch_order(registry, mode: CaptureMode) -> None:  # type: ignore[no-untyped-def]
    creator = registry.get("razorpay", PaymentCreator)
    req = CreatePaymentRequest(
        amount=Money.from_major("10.00", "INR"),
        merchant_reference=f"pw-sandbox-{secrets.token_hex(4)}",
        idempotency_key=f"pw_sandbox_{secrets.token_hex(12)}",
        capture_mode=mode,
        metadata={"suite": "sandbox"},
    )
    created = await creator.create_payment(req)
    try:
        assert created.status is IntentStatus.REQUIRES_ACTION
        fetched = await registry.get("razorpay", PaymentReader).get_payment(
            created.gateway_intent_id
        )
        assert fetched == created
        assert fetched.capture_mode is mode
        assert fetched.metadata == {"suite": "sandbox"}
    finally:
        await creator.aclose()


async def test_unknown_order_is_invalid_request(registry) -> None:  # type: ignore[no-untyped-def]
    reader = registry.get("razorpay", PaymentReader)
    try:
        with pytest.raises(InvalidRequestError):
            await reader.get_payment("order_DoesNotExist1")
    finally:
        await reader.aclose()
