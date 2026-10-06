# ruff: noqa: E501  (report lines read best unwrapped)
"""Check every scenario recorded by checkout_harness.py against the real test-mode API.

    set -a; . ./.env; set +a
    uv run python scripts/verify_scenarios.py

For each scenario order: fetch it through PayWeave, compare the intent status and
the failure categories with what the scenario expects, and show Razorpay's raw
``error_reason`` so mapping gaps are visible. For c1 it also captures and runs the
refund idempotency checks. Safe to re-run (c3 turns VOIDED ~12 min after paying).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from razorpay_scenarios import SCENARIOS, Scenario

from payweave.core.errors import PayWeaveError
from payweave.core.factory import GatewayFactory
from payweave.core.models import CaptureRequest, PaymentIntent, RefundRequest
from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus
from payweave.gateways.razorpay import RazorpayGateway

RESULTS_FILE = Path(__file__).resolve().parent.parent / ".sandbox" / "scenario_results.json"


def key(prefix: str) -> str:
    return f"pw_verify_{prefix}_{secrets.token_hex(8)}"


def expected_status(sc: Scenario, intent: PaymentIntent) -> IntentStatus:
    # c3: once the 12-minute capture window has passed, Razorpay auto-refunds (D3 -> VOIDED).
    if sc.expiry and datetime.now(UTC) - intent.created_at > timedelta(minutes=sc.expiry + 3):
        return IntentStatus.VOIDED
    return sc.expect_status


async def c1_lifecycle(gw: RazorpayGateway, order_id: str) -> list[str]:
    """Capture, then refund idempotency: replay must not duplicate, conflict must be rejected."""
    lines = []
    intent = await gw.capture(
        CaptureRequest(gateway_intent_id=order_id, idempotency_key=key("cap"))
    )
    lines.append(f"capture -> {intent.status}")
    k = key("rf")
    req = RefundRequest(gateway_intent_id=order_id, idempotency_key=k, amount=Money.of(5000, "INR"))
    first = await gw.refund(req)
    again = await gw.refund(req)
    lines.append(
        f"refund ₹50 twice with one key -> same refund: {first.gateway_refund_id == again.gateway_refund_id}"
    )
    try:
        await gw.refund(req.model_copy(update={"amount": Money.of(6000, "INR")}))
        lines.append("same key, different amount -> ACCEPTED (unexpected)")
    except PayWeaveError as exc:
        lines.append(f"same key, different amount -> {type(exc).__name__}: {exc}")
    intent = await gw.get_payment(order_id)
    lines.append(f"intent now -> {intent.status}, refunded {intent.amount_refunded}")
    return lines


async def main() -> int:
    if not RESULTS_FILE.exists():
        sys.exit("No results yet: run scripts/checkout_harness.py and pay some scenarios first.")
    log: list[dict[str, Any]] = json.loads(RESULTS_FILE.read_text())
    factory = GatewayFactory.with_installed_providers()
    config = factory.config_from_env("razorpay")
    config.pop("webhook_secrets", None)
    if not str(config.get("key_id", "")).startswith("rzp_test_"):
        sys.exit("Refusing to run: needs a TEST key (rzp_test_...)")
    gw = factory.create("razorpay", config)
    if not isinstance(gw, RazorpayGateway):
        sys.exit("expected a RazorpayGateway")

    outcomes: dict[str, list[str]] = {}
    for e in log:
        if e.get("order_id") and e["kind"] != "created":
            outcomes.setdefault(e["order_id"], []).append(e["kind"])
    passed = failed = skipped = 0
    try:
        for entry in (e for e in log if e["kind"] == "created"):
            sc, order_id = SCENARIOS[entry["scenario"]], entry["order_id"]
            seen = outcomes.get(order_id, [])
            if not seen:
                skipped += 1
                continue
            await asyncio.sleep(1)  # test mode rate-limits bursts
            intent = await gw.get_payment(order_id)
            raw = (await gw._client.get(f"orders/{order_id}/payments"))["items"]
            want = expected_status(sc, intent)
            got_failures = tuple(a.failure.category for a in intent.attempts if a.failure)
            ok = intent.status is want and (
                not sc.expect_failures
                or got_failures[: len(sc.expect_failures)] == sc.expect_failures
            )
            passed, failed = passed + ok, failed + (not ok)
            print(f"{'PASS' if ok else 'MISMATCH':8} {sc.id:3} {sc.title}  [{order_id}]")
            print(f"         checkout events: {seen}")
            print(f"         intent: {intent.status} (expected {want})")
            raw_by_id = {p["id"]: p for p in raw}
            for a in intent.attempts:
                r = raw_by_id[a.gateway_attempt_id]
                failure = (
                    f" failure={a.failure.category} raw_reason={r.get('error_reason')!r}"
                    if a.failure
                    else ""
                )
                print(
                    f"         attempt {a.gateway_attempt_id}: {r['status']}/{r.get('method')} -> {a.status}{failure}"
                )
            if sc.expect_failures and got_failures != sc.expect_failures:
                print(
                    f"         failures: got {[str(f) for f in got_failures]}, expected {[str(f) for f in sc.expect_failures]}"
                )
            if sc.id == "c1" and any(a.status is AttemptStatus.AUTHORIZED for a in intent.attempts):
                for line in await c1_lifecycle(gw, order_id):
                    print(f"         c1: {line}")
    finally:
        await gw.aclose()
    print(f"\n{passed} passed, {failed} mismatched, {skipped} created but not paid yet")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
