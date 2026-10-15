import base64
import hashlib
import hmac
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import tomli_w

from payweave.core.errors import InvalidRequestError, InvalidSignatureError, UnknownGatewayError
from payweave.core.factory import GatewayContext, GatewayFactory
from payweave.gateways.razorpay import mapping
from payweave.packages.manifest import Manifest
from payweave.packages.runtime import (
    PackageLoadError,
    PackageProvider,
    load_package,
    load_package_from_sources,
)
from payweave.packages.sandbox import MapperRejectedError, check_source, load_mappers
from payweave.packages.sdk import MapperContext
from tests.gateways.razorpay import fixtures as fx

RAZORPAY = Path(__file__).parents[2] / "packages" / "razorpay" / "v1"
MANIFEST = (RAZORPAY / "manifest.toml").read_text()
MAPPERS = (RAZORPAY / "mappers.py").read_text()


def manifest_with(**changes: Any) -> dict[str, Any]:
    data = tomllib.loads(MANIFEST)
    for dotted, value in changes.items():
        *parents, last = dotted.split("__")
        node = data
        for p in parents:
            node = node[p]
        if value is None:
            node.pop(last)
        else:
            node[last] = value
    return data


# --- sandbox: static check ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        ("import os", "whole-module imports"),
        ("from os import system", "import from 'os'"),
        ("from typing import get_type_hints", "'get_type_hints' from 'typing'"),
        ("from payweave.core import money", "import from 'payweave.core'"),
        ("def f(x):\n    return x.__class__", "private attribute '__class__'"),
        ("def f():\n    return ().__class__.__subclasses__()", "private attribute"),
        ("def f(x):\n    return x._secret", "private attribute '_secret'"),
        ("def f():\n    return eval('1')", "'eval'"),
        ("def f(x):\n    return getattr(x, 'y')", "'getattr'"),
        ("def f():\n    return open('/etc/passwd')", "'open'"),
        ("def f():\n    return __builtins__", "'__builtins__'"),
        ("def f(x):\n    return '{0.__class__}'.format(x)", "'format'"),
        ("from datetime import datetime\ndef f():\n    return datetime.now()", "clock"),
        ("class A:\n    pass", "classes"),
        ("async def f():\n    pass", "async functions"),
        ("def f():\n    while True:\n        pass", "while loops"),
        ("def f():\n    yield 1", "generators"),
        ("X = 1\nprint(X)", "module level"),
        ("def f():\n    global X", "global"),
        ("def deco(f):\n    return f\n@deco\ndef g():\n    pass", "decorators"),
        ("def f(:", "syntax error"),
    ],
)
def test_static_check_rejects(source: str, problem: str) -> None:
    problems = check_source(source)
    assert any(problem in p for p in problems), problems


def test_razorpay_mappers_pass_the_static_check() -> None:
    assert check_source(MAPPERS) == []


def test_restricted_builtins_back_up_the_static_check() -> None:
    # Even code the AST check allows has no dangerous builtins at runtime.
    loaded = load_mappers("def f():\n    return len('ab')\n", module_name="t")
    assert loaded.functions["f"]() == 2
    with pytest.raises(MapperRejectedError):
        load_mappers("def f():\n    return open\n", module_name="t")


# --- manifest -------------------------------------------------------------------------------


def test_razorpay_manifest_is_valid() -> None:
    manifest = Manifest.model_validate(tomllib.loads(MANIFEST))
    assert manifest.api.allowed_hosts == ("api.razorpay.com",)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"api__base_url": "https://evil.example/v1/"}, "not in allowed_hosts"),
        ({"api__base_url": "http://api.razorpay.com/v1/"}, "https"),
        ({"auth__password": "key_id"}, "must be a secret"),
        ({"auth__username": "nope"}, "unknown credential"),
        ({"ids__intent": "^order_.+$"}, "unsafe id"),
        ({"ids__intent": "order_[a-z]+"}, "anchored"),
        ({"operations__capture__path": "payments/{refund_id}/capture"}, "may only use"),
        ({"operations__capture__path": "/v2/payments/{attempt_id}"}, "relative"),
        ({"operations__get_payment": None}, "need get_payment"),
        ({"checkout_return__message_template": "{x.__class__}"}, "plain parameter names"),
        ({"credentials__key_id__secret": True}, "both secret and public"),
        ({"unexpected": 1}, "Extra inputs"),
        ({"api__headers": {"Authorization": "x"}}, "may not set"),
        ({"api__headers": {"x-api-version": "a b"}}, "pattern"),
        ({"auth__scheme": "api_key_headers"}, "needs both header names"),
        ({"auth__username_header": "x-client-id"}, "take none"),
        ({"operations__get_refund__path": "refunds/{attempt_id}"}, "may only use"),
    ],
)
def test_manifest_rejects(changes: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        Manifest.model_validate(manifest_with(**changes))


def header_auth_gateway(seen: list[httpx.Request]) -> Any:
    """The Razorpay package switched to api_key_headers and an intent-scoped get_refund."""
    data = manifest_with(
        auth__scheme="api_key_headers",
        auth__username_header="x-client-id",
        auth__password_header="x-client-secret",  # noqa: S106 - a header name
        api__headers={"x-api-version": "2026-01-01"},
        operations__get_refund__path="orders/{intent_id}/refunds/{refund_id}",
    )

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=fx.refund())

    package = load_package_from_sources(tomli_w.dumps(data), MAPPERS)
    return GatewayFactory([PackageProvider(package, allow_unsigned=True)]).create(
        "razorpay",
        {"key_id": fx.KEY_ID, "key_secret": "s3cret"},
        context=GatewayContext(name="razorpay", http_transport=httpx.MockTransport(respond)),
    )


async def test_api_key_headers_auth_and_intent_scoped_refund_lookup() -> None:
    seen: list[httpx.Request] = []
    gw = header_auth_gateway(seen)
    refund = await gw.get_refund("rfnd_00000000000001", gateway_intent_id=fx.ORDER_ID)
    assert refund.gateway_refund_id == fx.refund()["id"]
    [request] = seen
    assert request.url.path == f"/v1/orders/{fx.ORDER_ID}/refunds/rfnd_00000000000001"
    assert request.headers["x-client-id"] == fx.KEY_ID
    assert request.headers["x-client-secret"] == "s3cret"
    assert request.headers["x-api-version"] == "2026-01-01"
    assert "authorization" not in request.headers
    with pytest.raises(InvalidRequestError, match="pass gateway_intent_id"):
        await gw.get_refund("rfnd_00000000000001")
    await gw.aclose()


# --- loading --------------------------------------------------------------------------------


def test_missing_or_misnamed_mappers_are_rejected() -> None:
    without_refund = MAPPERS.replace("def refund(entity", "def refund_entity(entity")
    with pytest.raises(PackageLoadError, match="missing mapper refund") as exc:
        load_package_from_sources(MANIFEST, without_refund)
    assert "unexpected public function 'refund_entity'" in str(exc.value)
    renamed = MAPPERS.replace("def refund_body(amount, reason)", "def refund_body(amt, reason)")
    renamed = renamed.replace(
        "def refund_body(amount: Money, reason: str | None)",
        "def refund_body(amt: Money, reason: str | None)",
    )
    with pytest.raises(PackageLoadError, match=r"refund_body must take \(amount, reason\)"):
        load_package_from_sources(MANIFEST, renamed)


def test_sandbox_rejection_is_a_load_error() -> None:
    with pytest.raises(PackageLoadError, match="whole-module imports"):
        load_package_from_sources(MANIFEST, "import os\n" + MAPPERS)


def test_hash_covers_manifest_and_mappers() -> None:
    base = load_package(RAZORPAY).sha256
    assert load_package_from_sources(MANIFEST, MAPPERS).sha256 == base
    assert load_package_from_sources(MANIFEST + "\n", MAPPERS).sha256 != base
    assert load_package_from_sources(MANIFEST, MAPPERS + "\n").sha256 != base


async def test_mapper_crash_fails_closed() -> None:
    broken = MAPPERS.replace('native = entity.get("status")', 'native = entity["no_such_field"]')
    gw: Any = GatewayFactory(
        [PackageProvider(load_package_from_sources(MANIFEST, broken), allow_unsigned=True)]
    ).create("razorpay", {"key_id": fx.KEY_ID, "key_secret": "s"})
    with pytest.raises(UnknownGatewayError, match="mapper refund failed: KeyError"):
        gw._call("refund", fx.refund(), gw._ctx)
    await gw.aclose()


def test_mappers_get_public_credentials_only() -> None:
    gw: Any = GatewayFactory([PackageProvider(load_package(RAZORPAY), allow_unsigned=True)]).create(
        "razorpay", {"key_id": fx.KEY_ID, "key_secret": "very_secret", "webhook_secrets": ["w"]}
    )
    assert gw._ctx == MapperContext(gateway="razorpay", public={"key_id": fx.KEY_ID})
    assert "very_secret" not in repr(gw._config)


# --- differential: the package's mappers vs the hand-written mapping ------------------------


PAYMENT_SETS = [
    [],
    [fx.payment("created")],
    [fx.payment("authorized")],
    [fx.payment("failed", error_reason="incorrect_otp")],
    [fx.payment("failed", error_reason="something_new"), fx.payment("authorized", pid="pay_B2")],
    [fx.payment("refunded", captured=False)],
]


@pytest.mark.parametrize("payments", PAYMENT_SETS)
def test_intent_mapping_matches_hand_written(payments: list[Any]) -> None:
    package = load_package(RAZORPAY)
    order = fx.order("attempted" if payments else "created")
    ctx = MapperContext(gateway="razorpay", public={"key_id": fx.KEY_ID})
    expected = mapping.intent_from_order(order, payments, gateway="razorpay", key_id=fx.KEY_ID)
    got = package.mappers["intent"](order, {"payments": fx.payments(*payments)}, ctx)
    assert got == expected


@pytest.mark.parametrize("status", ["pending", "processed", "failed"])
def test_refund_mapping_matches_hand_written(status: str) -> None:
    package = load_package(RAZORPAY)
    entity = fx.refund(status)
    ctx = MapperContext(gateway="razorpay")
    assert package.mappers["refund"](entity, ctx) == mapping.refund_from_entity(
        entity, gateway="razorpay"
    )


def test_builtins_missing_from_the_sandbox_are_caught_statically() -> None:
    problems = check_source("def f(x: object) -> bool:\n    return isinstance(x, (int, float))\n")
    assert any("builtin 'float' isn't available" in p for p in problems)
    # A local that happens to share a builtin's name is fine.
    assert check_source("def f(iter: int) -> int:\n    return iter + 1\n") == []


# --- timestamped webhooks (Cashfree's scheme) --------------------------------------------

NOW = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)


def timestamped_gateway() -> Any:
    data = manifest_with(
        webhooks__scheme="timestamped_hmac_sha256_base64",
        webhooks__timestamp_header="x-webhook-timestamp",
        webhooks__secrets_credential="key_secret",  # signed with the API secret
    )
    package = load_package_from_sources(tomli_w.dumps(data), MAPPERS)
    return GatewayFactory([PackageProvider(package, allow_unsigned=True)]).create(
        "razorpay",
        {"key_id": fx.KEY_ID, "key_secret": "s3cret"},
        context=GatewayContext(name="razorpay", clock=lambda: NOW),
    )


def signed(body: bytes, stamp: str, secret: str = "s3cret") -> dict[str, str]:  # noqa: S107
    # The same computation as Cashfree's SDK: Base64(HMAC-SHA256(timestamp + rawBody)).
    digest = hmac.new(secret.encode(), stamp.encode() + body, hashlib.sha256).digest()
    return {
        "X-Webhook-Signature": base64.b64encode(digest).decode(),
        "x-webhook-timestamp": stamp,
        "X-Razorpay-Event-Id": "evt_1",
    }


async def test_timestamped_webhook_signature() -> None:
    gw = timestamped_gateway()
    body = json.dumps(fx.webhook("refund.processed", refund=fx.refund())).encode()
    stamp = str(int(NOW.timestamp() * 1000))  # Cashfree sends milliseconds
    headers = signed(body, stamp)
    headers["X-Razorpay-Signature"] = headers.pop("X-Webhook-Signature")
    assert gw.verify_and_parse(body, headers).event_id == "evt_1"

    stale = str(int((NOW.timestamp() - 3600) * 1000))
    old = signed(body, stale)
    old["X-Razorpay-Signature"] = old.pop("X-Webhook-Signature")
    with pytest.raises(InvalidSignatureError, match="too old"):
        gw.verify_and_parse(body, old)

    moved = dict(headers, **{"x-webhook-timestamp": str(int(stamp) + 1)})  # not what was signed
    with pytest.raises(InvalidSignatureError, match="did not verify"):
        gw.verify_and_parse(body, moved)
    await gw.aclose()


def test_timestamped_scheme_needs_its_header() -> None:
    with pytest.raises(ValueError, match="need timestamp_header"):
        Manifest.model_validate(manifest_with(webhooks__scheme="timestamped_hmac_sha256_base64"))
    with pytest.raises(ValueError, match="or the auth secret"):
        Manifest.model_validate(manifest_with(webhooks__secrets_credential="key_id"))


async def test_test_keys_go_to_the_sandbox_host() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=fx.refund())

    data = manifest_with(
        api__sandbox_base_url="https://sandbox.razorpay.example/v1/",
        api__allowed_hosts=["api.razorpay.com", "sandbox.razorpay.example"],
    )
    package = load_package_from_sources(tomli_w.dumps(data), MAPPERS)
    gw: Any = GatewayFactory([PackageProvider(package, allow_unsigned=True)]).create(
        "razorpay",
        {"key_id": fx.KEY_ID, "key_secret": "s"},
        context=GatewayContext(name="razorpay", http_transport=httpx.MockTransport(respond)),
    )
    await gw.get_refund("rfnd_00000000000001")
    assert seen[0].url.host == "sandbox.razorpay.example"
    await gw.aclose()
    with pytest.raises(ValueError, match="sandbox_base_url host"):
        Manifest.model_validate(manifest_with(api__sandbox_base_url="https://evil.example/v1/"))


def test_sdk_reads_iso_times_and_decimal_amounts_exactly() -> None:
    from payweave.packages import sdk  # noqa: PLC0415

    assert sdk.from_iso("2026-10-15T13:05:20+05:30").utcoffset() is not None
    assert sdk.money_major(10.01, "INR") == sdk.Money.of(1001, "INR")
    assert sdk.money_major(1, "INR") == sdk.Money.of(100, "INR")
    assert sdk.major_amount(sdk.Money.of(1001, "INR")) == 10.01
    for bad in (10.001, True, None, "abc"):
        with pytest.raises(sdk.MappingError):
            sdk.money_major(bad, "INR")
    for bad_ts in ("2026-10-15T13:05:20", 1700000000, "yesterday"):
        with pytest.raises(sdk.MappingError):
            sdk.from_iso(bad_ts)
