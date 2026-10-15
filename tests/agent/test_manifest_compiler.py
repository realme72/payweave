import tomllib
from pathlib import Path
from typing import Any

import pytest

from payweave.agent.manifest_compiler import compare, compile_manifest
from payweave.agent.spec_card import SpecCard
from payweave.packages.runtime import PackageProvider, load_package_from_sources
from payweave.testing.razorpay import sequential_ids
from payweave.verify.differential import run_differential
from payweave.verify.references import REFERENCES
from payweave.verify.scenarios import SCENARIOS

RAZORPAY = Path(__file__).parents[2] / "packages" / "razorpay" / "v1"
REFERENCE = tomllib.loads((RAZORPAY / "manifest.toml").read_text())


def c(value: Any) -> dict[str, Any]:
    return {"value": value, "citations": [{"ref": "razorpay:x.md#0", "quote": f"quote {value}"}]}


def none() -> dict[str, Any]:
    return {"value": None, "citations": []}


def razorpay_card(**overrides: dict[str, Any] | None) -> dict[str, Any]:
    """The Razorpay facts as a person would confirm them (matches eval/spec_cards)."""
    card: dict[str, Any] = {
        "gateway": "razorpay",
        "basics": {
            "api_base_url": c("https://api.razorpay.com/v1"),
            "auth_scheme": c("basic"),
            "amount_unit": c("minor_unit"),
        },
        "payment_flow": {"intent_object": c("order"), "customer_completion": c("hosted_checkout")},
        "capture": {
            "manual_capture": c(True),
            "partial_capture": c(False),
            "void_supported": none(),
        },
        "refunds": {
            "partial_refunds": c(True),
            "multiple_refunds": c(True),
            "idempotency_header": c("X-Refund-Idempotency"),
        },
        "webhooks": {
            "signature_header": c("X-Razorpay-Signature"),
            "signature_algorithm": c("hmac_sha256_hex"),
            "signing_secret": c("webhook_secret"),
            "event_id_header": c("X-Razorpay-Event-Id"),
            "ordered_delivery": c(False),
            "duplicate_deliveries": c(True),
        },
        "errors": {"error_code_field": c("code"), "reason_field": c("reason")},
        "endpoints": {
            "create_intent": c("POST /v1/orders"),
            "get_intent": c("GET /v1/orders/:id"),
            "list_attempts": c("GET /v1/orders/:id/payments"),
            "capture": c("POST /v1/payments/:id/capture"),
            "refund": c("POST /v1/payments/:id/refund"),
            "get_refund": c("GET /v1/refunds/:id"),
        },
        "ids": {
            "intent_id_example": c("order_DaZlswtdcn9UNV"),
            "attempt_id_example": c("pay_29QQoUBi66xm2f"),
            "refund_id_example": c("rfnd_FgRAHdNOM4ZVbO"),
        },
        "api_keys": {
            "test_key_prefix": c("rzp_test_"),
            "live_key_prefix": c("rzp_live_"),
            "prefixed_key": c("key_id"),
        },
        "error_shape": {
            "message_path": c("error.description"),
            "code_path": c("error.code"),
            "reason_path": c("error.reason"),
            "idempotency_conflict_message": c(
                "Different request with the same idempotency key has already been processed."
            ),
        },
        "checkout_return": {
            "intent_param": c("razorpay_order_id"),
            "attempt_param": c("razorpay_payment_id"),
            "signature_param": c("razorpay_signature"),
            "signed_message": c("intent_id|payment_id"),
            "signature_scheme": c("hmac_sha256_hex"),
            "signing_key": c("api_key_secret"),
        },
    }
    for section, fields in overrides.items():
        card[section] = None if fields is None else {**card.get(section, {}), **fields}
    return card


def test_compiles_a_valid_manifest_with_provenance() -> None:
    out = compile_manifest(SpecCard.model_validate(razorpay_card()))
    assert out.ok, (out.gaps, out.problems)
    m = out.manifest
    assert m["ids"]["intent"] == "^order_[A-Za-z0-9]+$"
    assert (
        m["operations"]["get_payment"]["related"]["payments"]["path"]
        == "orders/{intent_id}/payments"
    )
    assert m["operations"]["capture"]["path"] == "payments/{attempt_id}/capture"
    assert m["live_guard"] == {"credential": "key_id", "test_pattern": "^rzp_test_"}
    assert m["checkout_return"]["message_template"] == "{razorpay_order_id}|{razorpay_payment_id}"
    assert out.provenance["operations.capture"][0]["quote"] == "quote POST /v1/payments/:id/capture"
    assert tomllib.loads(out.toml()) == m


def test_matches_the_hand_written_manifest_field_by_field() -> None:
    out = compile_manifest(SpecCard.model_validate(razorpay_card()))
    diff = compare(out.manifest, REFERENCE)
    assert diff["missing"] == ["live_guard.live_pattern = '^rzp_live_'"]
    assert diff["extra"] == ["live_guard.test_pattern = '^rzp_test_'"]
    # Written differently, same behaviour: the key pattern accepts the same keys, and the
    # full documented message contains the hand-picked substring (G7 below confirms both).
    assert [d.split(":")[0] for d in diff["different"]] == [
        "credentials.key_id.pattern",
        "errors.rules[0].message_contains",
    ]
    assert len(diff["same"]) > 30


async def test_compiled_manifest_behaves_like_the_reference() -> None:
    # Hand-written mappers + compiled manifest (+ the person-added settings) must pass G7.
    manifest = compile_manifest(SpecCard.model_validate(razorpay_card())).manifest
    manifest["settings"] = REFERENCE["settings"]
    import tomli_w  # noqa: PLC0415

    package = load_package_from_sources(
        tomli_w.dumps(manifest), (RAZORPAY / "mappers.py").read_text()
    )
    ref = REFERENCES["razorpay"]
    candidate = PackageProvider(package, allow_unsigned=True)
    result = await run_differential(
        SCENARIOS,
        reference=lambda router: ref.harness(router, ref.provider, sequential_ids()),
        candidate=lambda router: ref.harness(router, candidate, sequential_ids()),
    )
    assert result.passed, [s.differences + s.crashes for s in result.scenarios if not s.passed]


@pytest.mark.parametrize(
    ("overrides", "gap"),
    [
        ({"basics": {"auth_scheme": c("oauth2_client_credentials")}}, "hand-written auth"),
        ({"endpoints": {"capture": c("POST /v2/payments/:id/capture")}}, "can't place"),
        ({"endpoints": {"refund": c("POST /v1/payments/refund")}}, "with its ids"),
        ({"ids": {"refund_id_example": none()}}, "ids.refund: needs ids.refund_id_example"),
        (
            {"webhooks": {"signature_algorithm": c("timestamped_hmac_sha256")}},
            "needs webhooks.signature_encoding",
        ),
        ({"endpoints": None}, "endpoints missing"),
    ],
)
def test_open_facts_become_gaps_not_guesses(overrides: dict[str, Any], gap: str) -> None:
    out = compile_manifest(SpecCard.model_validate(razorpay_card(**overrides)))
    assert not out.ok
    assert any(gap in g for g in out.gaps), out.gaps


def test_optional_facts_are_left_out() -> None:
    out = compile_manifest(
        SpecCard.model_validate(
            razorpay_card(
                capture={"manual_capture": c(False)},
                endpoints={"list_attempts": none()},
                checkout_return=None,
            )
        )
    )
    assert out.ok, (out.gaps, out.problems)
    assert "capture" not in out.manifest["operations"]
    assert "manual_capture" not in out.manifest["flags"]
    assert "related" not in out.manifest["operations"]["get_payment"]
    assert "checkout_return" not in out.manifest


def test_bearer_scheme_guards_the_secret_key() -> None:
    out = compile_manifest(
        SpecCard.model_validate(
            razorpay_card(
                basics={"auth_scheme": c("bearer")},
                api_keys={
                    "test_key_prefix": c("sk_test_"),
                    "live_key_prefix": c("sk_live_"),
                    "prefixed_key": c("secret_key"),
                },
                checkout_return=None,
            )
        )
    )
    assert out.ok, (out.gaps, out.problems)
    assert out.manifest["live_guard"] == {"credential": "secret_key", "test_pattern": "^sk_test_"}
    assert out.manifest["auth"] == {"scheme": "bearer", "password": "secret_key"}


def test_live_guard_from_the_configured_test_key_when_docs_are_silent() -> None:
    from payweave.agent.manifest_compiler import test_key_prefix  # noqa: PLC0415

    silent = {"test_key_prefix": none(), "live_key_prefix": none(), "prefixed_key": none()}
    out = compile_manifest(SpecCard.model_validate(razorpay_card(api_keys=silent)))
    assert not out.ok
    assert any(g.startswith("live_guard:") for g in out.gaps)

    prefix = test_key_prefix("rzp_test_AbCdEf123456")
    assert prefix == "rzp_test_"
    out = compile_manifest(
        SpecCard.model_validate(razorpay_card(api_keys=silent)), observed_test_key_prefix=prefix
    )
    assert out.ok, (out.gaps, out.problems)
    assert out.manifest["live_guard"] == {"credential": "key_id", "test_pattern": "^rzp_test_"}
    assert out.provenance["live_guard"][0]["ref"] == "observed:configured-test-key"
    assert test_key_prefix("app_1234") is None  # no test marker: don't guess
    # Nothing after the test marker may leak, even when the secret has underscores.
    assert test_key_prefix("cfsk_ma_test_0123abcd_ef45") == "cfsk_ma_test_"
    assert test_key_prefix("sk_test") is None  # the marker must be followed by the key


def test_test_pattern_guard_fails_closed() -> None:
    from payweave.core.factory import GatewayFactory, InvalidGatewayConfigError  # noqa: PLC0415

    silent = {"test_key_prefix": none(), "live_key_prefix": none(), "prefixed_key": none()}
    card = SpecCard.model_validate(razorpay_card(api_keys=silent))
    manifest = compile_manifest(card, observed_test_key_prefix="rzp_test_").manifest
    manifest["settings"] = REFERENCE["settings"]
    import tomli_w  # noqa: PLC0415

    package = load_package_from_sources(
        tomli_w.dumps(manifest), (RAZORPAY / "mappers.py").read_text()
    )
    factory = GatewayFactory([PackageProvider(package, allow_unsigned=True)])
    with pytest.raises(InvalidGatewayConfigError, match="not a test-mode credential"):
        factory.create("razorpay", {"key_id": "rzp_prod_Unexpected1", "key_secret": "s"})
    factory.create("razorpay", {"key_id": "rzp_test_Abc123", "key_secret": "s"})
    factory.create(
        "razorpay", {"key_id": "rzp_prod_Unexpected1", "key_secret": "s", "allow_live": True}
    )


def test_header_auth_relative_paths_and_intent_scoped_operations() -> None:
    """Cashfree's shape: x-client-id/x-client-secret, a version header, paths written
    relative to the base URL (.../pg), and capture/refund through the order id."""
    card = razorpay_card(
        basics={
            "api_base_url": c("https://api.cashfree.com/pg"),
            "auth_scheme": c("api_key_headers"),
            "sandbox_base_url": c("https://sandbox.cashfree.com/pg"),
        },
        auth_headers={
            "key_id_header": c("x-client-id"),
            "secret_header": c("x-client-secret"),
            "version_header": c("x-api-version"),
            "api_version": c("2026-01-01"),
        },
        endpoints={
            "create_intent": c("POST /orders"),
            "get_intent": c("GET /orders/{order_id}"),
            "list_attempts": c("GET /orders/{order_id}/payments"),
            "capture": c("POST /orders/{order_id}/authorization"),
            "refund": c("POST /orders/{order_id}/refunds"),
            "get_refund": c("GET /orders/{order_id}/refunds/{refund_id}"),
        },
        api_keys={"test_key_prefix": none(), "live_key_prefix": none(), "prefixed_key": none()},
        checkout_return=None,
    )
    out = compile_manifest(
        SpecCard.model_validate(card),
        observed_test_key_prefix="cfsk_ma_test_",
        observed_on="secret_key",
    )
    assert out.ok, (out.gaps, out.problems)
    m = out.manifest
    assert m["api"]["headers"] == {"x-api-version": "2026-01-01"}
    assert m["api"]["sandbox_base_url"] == "https://sandbox.cashfree.com/pg/"
    assert m["api"]["allowed_hosts"] == ["api.cashfree.com", "sandbox.cashfree.com"]
    assert m["auth"] == {
        "scheme": "api_key_headers",
        "username": "key_id",
        "password": "key_secret",
        "username_header": "x-client-id",
        "password_header": "x-client-secret",
    }
    assert m["live_guard"] == {"credential": "key_secret", "test_pattern": "^cfsk_ma_test_"}
    paths = {
        "create": m["operations"]["create_payment"]["path"],
        "get": m["operations"]["get_payment"]["primary"]["path"],
        "attempts": m["operations"]["get_payment"]["related"]["payments"]["path"],
        "capture": m["operations"]["capture"]["path"],
        "refund": m["operations"]["refund"]["path"],
        "get_refund": m["operations"]["get_refund"]["path"],
    }
    assert paths == {
        "create": "orders",
        "get": "orders/{intent_id}",
        "attempts": "orders/{intent_id}/payments",
        "capture": "orders/{intent_id}/authorization",
        "refund": "orders/{intent_id}/refunds",
        "get_refund": "orders/{intent_id}/refunds/{refund_id}",
    }
    assert any("relative to the base URL" in n for n in out.notes)


def test_mixed_path_styles_stay_gaps() -> None:
    card = razorpay_card(endpoints={"capture": c("POST /payments/:id/capture")})
    out = compile_manifest(SpecCard.model_validate(card))
    assert any(g.startswith("operations.capture: can't place") for g in out.gaps), out.gaps
