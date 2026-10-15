import json
from typing import Any

import httpx
import pytest

from payweave.agent.manifest_compiler import compile_manifest
from payweave.agent.spec_card import SpecCard
from payweave.packages.manifest import Manifest
from payweave.verify.error_probe import infer_error_shape, probe_error_shape
from tests.agent.test_manifest_compiler import c, none, razorpay_card


def cashfree_error(code: str, message: str) -> dict[str, str]:
    help_ = "Check latest errors in the dashboard: https://bit.ly/x"  # a sentence, with a link
    return {"code": code, "message": message, "help": help_, "type": "invalid_request_error"}


CASHFREE = [
    cashfree_error("order_not_found", "Order Reference Id does not exist"),
    cashfree_error("order_amount_invalid", "order_amount : Invalid amount entered"),
    cashfree_error("refund_id_invalid", "refund_id : should be at least 3"),
]


def test_broad_code_is_constant_and_reason_varies() -> None:
    seen = infer_error_shape(CASHFREE)
    assert (seen.message_path, seen.code_path, seen.reason_path) == ("message", "type", "code")
    assert seen.bodies == 3


def test_nested_bodies() -> None:
    def razorpay(reason: str) -> dict[str, Any]:
        error = {
            "code": "BAD_REQUEST_ERROR",
            "description": f"The {reason} is bad",
            "reason": reason,
        }
        return {"error": error}

    seen = infer_error_shape([razorpay("input_validation_failed"), razorpay("not_found")])
    assert (seen.message_path, seen.code_path, seen.reason_path) == (
        "error.description",
        "error.code",
        "error.reason",
    )


def test_one_failure_settles_nothing_about_codes() -> None:
    seen = infer_error_shape(CASHFREE[:1])
    assert seen.message_path == "message"
    assert (seen.code_path, seen.reason_path) == (None, None)


def cashfree_manifest() -> dict[str, Any]:
    card = razorpay_card(
        basics={
            "api_base_url": c("https://api.cashfree.com/pg"),
            "auth_scheme": c("api_key_headers"),
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
        error_shape={"code_path": c("code"), "reason_path": c("type")},  # the docs misread
        api_keys={"test_key_prefix": none(), "live_key_prefix": none(), "prefixed_key": none()},
        checkout_return=None,
    )
    out = compile_manifest(
        SpecCard.model_validate(card),
        observed_test_key_prefix="cfsk_ma_test_",
        observed_on="secret_key",
    )
    return out.manifest


async def test_probe_sends_only_failing_requests_and_reads_the_bodies() -> None:
    sent: list[httpx.Request] = []

    def sandbox(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = CASHFREE[len(sent) - 1]
        return httpx.Response(404 if request.method == "GET" else 400, json=body)

    manifest = Manifest.model_validate(cashfree_manifest())
    seen, log = await probe_error_shape(
        manifest,
        {"key_id": "TEST123", "key_secret": "cfsk_ma_test_abc"},
        base_url="https://sandbox.cashfree.com/pg/",
        transport=httpx.MockTransport(sandbox),
    )
    assert [(r.method, r.url.path) for r in sent] == [
        ("GET", "/pg/orders/payweaveprobe0000000000"),
        ("POST", "/pg/orders"),
        ("GET", "/pg/orders/payweaveprobe0000000000/refunds/payweaveprobe0000000000"),
    ]
    assert json.loads(sent[1].content) == {}  # nothing that could create an order
    assert sent[0].headers["x-client-secret"] == "cfsk_ma_test_abc"
    assert sent[0].headers["x-api-version"] == "2026-01-01"
    assert (seen.code_path, seen.reason_path) == ("type", "code")
    assert len(log) == 3


async def test_probe_refuses_live_keys_and_plain_http() -> None:
    manifest = Manifest.model_validate(cashfree_manifest())
    with pytest.raises(ValueError, match="isn't a test key"):
        await probe_error_shape(
            manifest, {"key_secret": "cfsk_ma_prod_abc"}, base_url="https://api.cashfree.com/pg/"
        )
    with pytest.raises(ValueError, match="https"):
        await probe_error_shape(
            manifest, {"key_secret": "cfsk_ma_test_abc"}, base_url="http://sandbox.cashfree.com/pg/"
        )


def test_what_the_sandbox_shows_wins_over_the_card() -> None:
    card = SpecCard.model_validate(
        razorpay_card(error_shape={"code_path": c("code"), "reason_path": c("type")})
    )
    out = compile_manifest(card, observed_errors=infer_error_shape(CASHFREE))
    assert out.manifest["errors"]["code_paths"] == ["code", "type"]  # reason first, then broad
    assert out.manifest["errors"]["message_paths"] == ["message"]
    assert out.provenance["errors.code_paths"][0]["ref"] == "observed:sandbox"
    assert any("card said" in n for n in out.notes)


def test_observations_become_cited_card_facts() -> None:
    from payweave.agent.citations import check_section  # noqa: PLC0415
    from payweave.agent.spec_card import Errors, ErrorShape  # noqa: PLC0415
    from payweave.verify.error_probe import observed_card  # noqa: PLC0415

    card = observed_card(infer_error_shape(CASHFREE))
    assert card["error_shape"]["code_path"]["value"] == "type"
    assert card["errors"] == {
        "error_code_field": {
            "value": "type",
            "citations": [
                {"ref": "observed:sandbox", "quote": '{"type": "invalid_request_error"}'}
            ],
        },
        "reason_field": {
            "value": "code",
            "citations": [{"ref": "observed:sandbox", "quote": '{"code": "order_not_found"}'}],
        },
    }
    unknown: dict[str, Any] = {"value": None, "citations": []}
    shape = ErrorShape.model_validate(
        card["error_shape"] | {"idempotency_conflict_message": unknown}
    )
    assert check_section(shape, {}) == []  # observed evidence needs no doc page
    assert check_section(Errors.model_validate(card["errors"]), {}) == []
