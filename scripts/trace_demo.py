"""Send a few real TEST-mode gateway calls to Jaeger so there is something to look at.

    make jaeger        # terminal 1
    make trace-demo    # terminal 2, then open http://localhost:16686 (service: payweave)

Creates test orders only: no payment, no charge. Includes failures on purpose
so error spans show up too.
"""

from __future__ import annotations

import asyncio
import secrets
import sys

from opentelemetry import trace

from payweave.core.capabilities import Capturer, PaymentCreator, PaymentReader, RefundReader
from payweave.core.errors import PayWeaveError
from payweave.core.factory import GatewayFactory, GatewaySpec
from payweave.core.models import CaptureMode, CaptureRequest, CreatePaymentRequest
from payweave.core.money import Money
from payweave.observability.setup import configure_telemetry


async def main() -> None:
    factory = GatewayFactory.with_installed_providers()
    config = factory.config_from_env("razorpay")
    config.pop("webhook_secrets", None)
    if not str(config.get("key_id", "")).startswith("rzp_test_"):
        sys.exit("Refusing to run: needs a TEST key (rzp_test_...)")
    registry = factory.build_registry([GatewaySpec("razorpay", "razorpay", config)])
    tracer = trace.get_tracer("payweave.demo")

    for mode in CaptureMode:
        # One parent span per "checkout", so Jaeger shows the calls as one trace.
        with tracer.start_as_current_span(f"demo checkout ({mode} capture)"):
            intent = await registry.get("razorpay", PaymentCreator).create_payment(
                CreatePaymentRequest(
                    amount=Money.of(49900, "INR"),
                    merchant_reference=f"pw-demo-{secrets.token_hex(4)}",
                    idempotency_key=f"pw_demo_{secrets.token_hex(12)}",
                    capture_mode=mode,
                )
            )
            await registry.get("razorpay", PaymentReader).get_payment(intent.gateway_intent_id)
            try:  # nobody paid, so this fails: shows an error span
                await registry.get("razorpay", Capturer).capture(
                    CaptureRequest(
                        gateway_intent_id=intent.gateway_intent_id,
                        idempotency_key=f"pw_demo_{secrets.token_hex(12)}",
                    )
                )
            except PayWeaveError as exc:
                print(f"expected failure: {type(exc).__name__}")
        await asyncio.sleep(1)  # test mode rate-limits bursts

    with tracer.start_as_current_span("demo unknown refund lookup"):
        try:
            await registry.get("razorpay", RefundReader).get_refund("rfnd_DoesNotExist01")
        except PayWeaveError as exc:
            print(f"expected failure: {type(exc).__name__}")

    await registry.get("razorpay", PaymentReader).aclose()  # type: ignore[attr-defined]


if __name__ == "__main__":
    shutdown = configure_telemetry(service_name="payweave")
    try:
        asyncio.run(main())
    finally:
        shutdown()  # flush spans to Jaeger before exiting
    print("Done: open http://localhost:16686 and pick service 'payweave'.")
