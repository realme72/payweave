"""Gate G6 (automatic part): the package against the gateway's real sandbox, no person needed.

For gateways with no offline simulator (G5) or hand-written reference (G7), this is
the only check that the mappers speak the real API. It never pays and never uses live
keys (the package's live guard runs first), and only runs when the sandbox credentials
are configured:

    <GATEWAY>_<CREDENTIAL>         e.g. CASHFREE_KEY_ID, CASHFREE_KEY_SECRET
    PAYWEAVE_TEST_CUSTOMER_ID      plus _PHONE / _EMAIL if the gateway needs them

Steps: create an intent, fetch it back, then look up an intent and a refund that don't
exist (they must fail as InvalidRequestError, not crash). Each finding names its step
(``step create: ...``) so the mapper loop can route it to the mapper responsible.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from typing import Any

from payweave.core.errors import GatewayError, InvalidRequestError
from payweave.core.factory import GatewayContext, GatewayFactory
from payweave.core.models import CreatePaymentRequest, Customer
from payweave.core.money import Money
from payweave.core.status import IntentStatus
from payweave.packages.runtime import GatewayPackage, PackageProvider

UNKNOWN_ID = "payweavesmoke00000000"


def sandbox_config(
    package: GatewayPackage, env: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """Instance config from ``<GATEWAY>_<CREDENTIAL>`` variables; None if any is missing."""
    env = os.environ if env is None else env
    m = package.manifest
    config: dict[str, Any] = {}
    for name, cred in m.credentials.items():
        value = env.get(f"{m.gateway}_{name}".upper(), "")
        if cred.multiple:
            config[name] = [v for v in value.split(",") if v]
        elif value:
            config[name] = value
        elif cred.required:
            return None
    return config


def sandbox_customer(env: Mapping[str, str] | None = None) -> Customer | None:
    env = os.environ if env is None else env
    if not env.get("PAYWEAVE_TEST_CUSTOMER_ID"):
        return None
    return Customer(
        id=env["PAYWEAVE_TEST_CUSTOMER_ID"],
        phone=env.get("PAYWEAVE_TEST_CUSTOMER_PHONE") or None,
        email=env.get("PAYWEAVE_TEST_CUSTOMER_EMAIL") or None,
    )


async def smoke(
    package: GatewayPackage, config: Mapping[str, Any], customer: Customer | None
) -> tuple[list[str], list[str]]:
    """Returns (steps that passed, findings)."""
    factory = GatewayFactory([PackageProvider(package, allow_unsigned=True)])
    gw: Any = factory.create(
        package.manifest.gateway,
        dict(config),
        context=GatewayContext(name=package.manifest.gateway),
    )
    passed: list[str] = []
    findings: list[str] = []

    async def step(label: str, call: Any) -> Any:
        try:
            value = await call
        except GatewayError as exc:
            findings.append(f"step {label}: {type(exc).__name__}: {exc}")
            return None
        passed.append(label)
        return value

    async def must_fail(label: str, call: Any) -> None:
        try:
            await call
        except InvalidRequestError:
            passed.append(label)
            return
        except GatewayError as exc:
            findings.append(
                f"step {label}: expected InvalidRequestError, got {type(exc).__name__}: {exc}"
            )
            return
        findings.append(f"step {label}: expected InvalidRequestError, but it succeeded")

    try:
        amount = Money.of(10000, "INR")
        request = CreatePaymentRequest(
            amount=amount,
            merchant_reference="payweave-g6",
            idempotency_key=f"pwg6{uuid.uuid4().hex[:20]}",
            customer=customer,
        )
        created = await step("create", gw.create_payment(request))
        if created is not None:
            if created.amount != amount:
                findings.append(f"step create: amount {created.amount} != requested {amount}")
            fetched = await step("get", gw.get_payment(created.gateway_intent_id))
            if fetched is not None:
                if fetched.gateway_intent_id != created.gateway_intent_id:
                    findings.append(
                        f"step get: id {fetched.gateway_intent_id!r} != created "
                        f"{created.gateway_intent_id!r}"
                    )
                if (
                    fetched.status is not IntentStatus.REQUIRES_ACTION
                    or fetched.next_action is None
                ):
                    findings.append(
                        f"step get: an unpaid intent should be REQUIRES_ACTION with a "
                        f"next_action, got {fetched.status} / {fetched.next_action}"
                    )
        await must_fail("get unknown", gw.get_payment(UNKNOWN_ID))
        if hasattr(gw, "get_refund") and package.manifest.operations.get_refund is not None:
            intent_id = created.gateway_intent_id if created is not None else UNKNOWN_ID
            await must_fail(
                "get unknown refund", gw.get_refund(UNKNOWN_ID, gateway_intent_id=intent_id)
            )
    finally:
        await gw.aclose()
    return passed, findings
