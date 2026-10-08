import tomllib
from pathlib import Path
from typing import Any

import pytest

from payweave.core.errors import UnknownGatewayError
from payweave.core.factory import GatewayFactory
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
    ],
)
def test_manifest_rejects(changes: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        Manifest.model_validate(manifest_with(**changes))


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
