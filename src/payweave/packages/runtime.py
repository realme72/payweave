"""Run a Gateway Package as a gateway: hand-written flows, package-supplied data and mappers.

The flows (validate ids, fetch the intent, pick the attempt, send the request,
map the result) are the same for every gateway and live here. A package only
supplies *what* to send and *how to read* the answers:

* ``manifest.toml``: endpoints, auth choice, id formats, error rules, signature schemes.
* ``mappers.py``: pure functions with fixed names and parameters (``MAPPERS``).

Anything a mapper raises that isn't a ``GatewayError`` becomes an
``UnknownGatewayError``: a mapping bug fails closed, it never guesses.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, SecretStr, create_model, model_validator

from payweave.core.capabilities import Capability
from payweave.core.errors import (
    GatewayError,
    InvalidAmountError,
    InvalidRequestError,
    InvalidSignatureError,
    UnknownGatewayError,
    UnsupportedCapabilityError,
)
from payweave.core.factory import GatewayContext
from payweave.core.models import (
    CaptureRequest,
    CreatePaymentRequest,
    PaymentAttempt,
    PaymentIntent,
    Refund,
    RefundRequest,
    WebhookEvent,
)
from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus
from payweave.packages.http import Json, PackageHttpClient
from payweave.packages.manifest import Manifest, Request
from payweave.packages.sandbox import MapperRejectedError, load_mappers
from payweave.packages.schemes import http_auth, verify_signature
from payweave.packages.sdk import MapperContext

if TYPE_CHECKING:
    from payweave.packages.signing import Approval

# Mapper name -> (parameter names, when it is required). Fixed by us, not the package.
MAPPERS: Mapping[str, tuple[tuple[str, ...], str]] = {
    "create_payment_body": (("request", "settings"), "create_payment"),
    "intent": (("primary", "related", "ctx"), "create_payment|get_payment"),
    "capture_body": (("attempt", "amount"), "capture"),
    "refund_body": (("amount", "reason"), "refund"),
    "refund": (("entity", "ctx"), "refund|get_refund"),
    "webhook_event": (("body", "event_id", "ctx"), "webhooks"),
    "check_amount": (("amount",), "optional"),  # -> error message or None
    "check_create": (("request",), "optional"),  # -> error message or None
}

_CAPTURED_STATES = frozenset(
    {IntentStatus.CAPTURED, IntentStatus.PARTIALLY_REFUNDED, IntentStatus.REFUNDED}
)


class PackageLoadError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class GatewayPackage:
    manifest: Manifest
    mappers: Mapping[str, Callable[..., Any]]
    sha256: str  # over manifest + mappers bytes: what G8 signs
    path: Path | None = None
    approval: Approval | None = None  # set only by signing.load_signed_package


def package_hash(manifest_bytes: bytes, mappers_bytes: bytes) -> str:
    digest = hashlib.sha256()
    for part in (b"manifest.toml", manifest_bytes, b"mappers.py", mappers_bytes):
        digest.update(len(part).to_bytes(8, "big") + part)  # length-prefixed: unambiguous
    return digest.hexdigest()


def load_package_from_sources(
    manifest_text: str, mappers_text: str, *, path: Path | None = None
) -> GatewayPackage:
    try:
        manifest = Manifest.model_validate(tomllib.loads(manifest_text))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise PackageLoadError(f"invalid manifest: {exc}") from exc
    try:
        loaded = load_mappers(
            mappers_text, module_name=f"payweave_package.{manifest.gateway}.v{manifest.version}"
        )
    except MapperRejectedError as exc:
        raise PackageLoadError(str(exc)) from exc
    _check_mappers(manifest, loaded.functions)
    return GatewayPackage(
        manifest,
        loaded.functions,
        package_hash(manifest_text.encode(), mappers_text.encode()),
        path,
    )


def load_package(directory: Path) -> GatewayPackage:
    return load_package_from_sources(
        (directory / "manifest.toml").read_text(),
        (directory / "mappers.py").read_text(),
        path=directory,
    )


def _check_mappers(manifest: Manifest, functions: Mapping[str, Callable[..., Any]]) -> None:
    present = {
        "create_payment": manifest.operations.create_payment is not None,
        "get_payment": manifest.operations.get_payment is not None,
        "capture": manifest.operations.capture is not None,
        "refund": manifest.operations.refund is not None,
        "get_refund": manifest.operations.get_refund is not None,
        "webhooks": manifest.webhooks is not None,
    }
    problems = [f"unexpected public function {n!r}" for n in functions if n not in MAPPERS]
    for name, (params, needed_for) in MAPPERS.items():
        fn = functions.get(name)
        required = needed_for != "optional" and any(present[p] for p in needed_for.split("|"))
        if fn is None:
            if required:
                problems.append(f"missing mapper {name}({', '.join(params)})")
            continue
        actual = tuple(inspect.signature(fn).parameters)
        if actual != params:
            problems.append(f"{name} must take ({', '.join(params)}), not ({', '.join(actual)})")
    if problems:
        raise PackageLoadError("mappers.py: " + "; ".join(problems))


# --- configuration ---------------------------------------------------------------------------


class _ConfigBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def config_model(manifest: Manifest) -> type[BaseModel]:
    """A Pydantic model for one package's instance config, built from its manifest."""
    fields: dict[str, Any] = {}
    for name, cred in manifest.credentials.items():
        item: Any = SecretStr if cred.secret else str
        if cred.pattern and not cred.secret:
            item = Annotated[str, Field(pattern=cred.pattern)]
        if cred.multiple:
            fields[name] = (tuple[item, ...], () if not cred.required else ...)
        else:
            fields[name] = (item, ... if cred.required else None)
    for name, setting in manifest.settings.items():
        kind = {"int": int, "str": str, "bool": bool}[setting.type]
        annotation: Any = Literal[setting.choices] if setting.choices else kind
        fields[name] = (
            annotation,
            Field(default=setting.default, ge=setting.minimum, le=setting.maximum),
        )
    fields["allow_live"] = (bool, False)
    fields["timeout_seconds"] = (float, Field(default=30.0, gt=0, le=120))

    guard = manifest.live_guard
    secrets = [n for n, c in manifest.credentials.items() if c.secret]

    def _check(self: BaseModel) -> BaseModel:
        for name in secrets:
            value = getattr(self, name)
            values = value if isinstance(value, tuple) else (value,)
            if any(isinstance(v, SecretStr) and not v.get_secret_value() for v in values):
                raise ValueError(f"{name} must not be empty")
        if guard is not None:
            value = _secret(getattr(self, guard.credential))  # secret keys too (sk_live_...)
            if guard.refuses(value) and not getattr(self, "allow_live"):  # noqa: B009
                raise ValueError(
                    f"{guard.credential} is not a test-mode credential and allow_live is false"
                )
        return self

    validators: dict[str, Any] = {"_check": model_validator(mode="after")(_check)}
    model: type[BaseModel] = create_model(
        f"{manifest.gateway.title()}PackageConfig",
        __base__=_ConfigBase,
        __validators__=validators,
        **fields,
    )
    return model


def _secret(value: object) -> str:
    return value.get_secret_value() if isinstance(value, SecretStr) else str(value or "")


# --- the gateway -----------------------------------------------------------------------------


class PackageGateway:
    def __init__(
        self,
        package: GatewayPackage,
        config: BaseModel,
        context: GatewayContext,
        client: PackageHttpClient | None = None,
    ) -> None:
        m = package.manifest
        self._package = package
        self._m = m
        self._config = config
        self._name = context.name
        self._clock = context.clock
        self._settings = {name: getattr(config, name) for name in m.settings}
        self._ctx = MapperContext(
            gateway=context.name,
            public={
                name: str(getattr(config, name))
                for name, c in m.credentials.items()
                if c.public and getattr(config, name) is not None
            },
        )
        self._client = client or PackageHttpClient(
            gateway=context.name,
            base_url=m.api.base_url,
            allowed_hosts=m.api.allowed_hosts,
            auth=http_auth(
                m.auth.scheme,
                username=str(getattr(config, m.auth.username)) if m.auth.username else "",
                password=_secret(getattr(config, m.auth.password)),
                username_header=m.auth.username_header or "",
                password_header=m.auth.password_header or "",
            ),
            headers=m.api.headers,
            errors=m.errors,
            timeout_seconds=float(getattr(config, "timeout_seconds")),  # noqa: B009
            transport=context.http_transport,
        )
        self._capabilities = self._derive_capabilities()

    def _derive_capabilities(self) -> frozenset[Capability]:
        ops = self._m.operations
        caps = {Capability(flag) for flag in self._m.flags}
        for op, cap in (
            (ops.create_payment, Capability.CREATE_PAYMENT),
            (ops.get_payment, Capability.GET_PAYMENT),
            (ops.capture, Capability.CAPTURE),
            (ops.refund, Capability.REFUND),
            (ops.get_refund, Capability.GET_REFUND),
            (self._m.checkout_return, Capability.CHECKOUT_RETURN),
        ):
            if op is not None:
                caps.add(cap)
        if ops.refund is not None and ops.refund.idempotency_header:
            caps.add(Capability.NATIVE_IDEMPOTENT_REFUND)
        if self._m.webhooks is not None and self._webhook_secrets():
            caps.add(Capability.WEBHOOKS)
        return frozenset(caps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    @property
    def package(self) -> GatewayPackage:
        return self._package

    async def aclose(self) -> None:
        await self._client.aclose()

    # Operations exist as methods on every PackageGateway; the registry only
    # exposes those whose capability is declared, and each one refuses if its
    # manifest operation is missing.

    async def create_payment(self, request: CreatePaymentRequest) -> PaymentIntent:
        op = self._need(self._m.operations.create_payment, Capability.CREATE_PAYMENT)
        self._check_amount(request.amount)
        if problem := self._call("check_create", request):
            raise InvalidRequestError(str(problem), gateway=self._name)
        body = self._call("create_payment_body", request, dict(self._settings))
        primary = await self._send(op, body=self._json_body(body, "create_payment_body"))
        return cast(PaymentIntent, self._call("intent", primary, {}, self._ctx))

    async def get_payment(self, gateway_intent_id: str) -> PaymentIntent:
        op = self._need(self._m.operations.get_payment, Capability.GET_PAYMENT)
        ids = {"intent_id": self._checked_id(gateway_intent_id, self._m.ids.intent)}
        names = list(op.related)
        primary, *related = await asyncio.gather(
            self._send(op.primary, ids=ids), *(self._send(op.related[n], ids=ids) for n in names)
        )
        return cast(
            PaymentIntent,
            self._call("intent", primary, dict(zip(names, related, strict=True)), self._ctx),
        )

    async def capture(self, request: CaptureRequest) -> PaymentIntent:
        """Capture the authorised attempt. Idempotent if the intent is already captured."""
        op = self._need(self._m.operations.capture, Capability.CAPTURE)
        intent = await self.get_payment(request.gateway_intent_id)
        authorized = [a for a in intent.attempts if a.status is AttemptStatus.AUTHORIZED]
        if request.attempt_id is not None:
            authorized = [a for a in authorized if a.gateway_attempt_id == request.attempt_id]
        if not authorized and intent.status in _CAPTURED_STATES:
            return intent
        attempt = self._single(authorized, "authorized attempt", request.gateway_intent_id)
        amount = request.amount or attempt.amount
        if amount != attempt.amount and Capability.PARTIAL_CAPTURE not in self._capabilities:
            raise UnsupportedCapabilityError(self._name, Capability.PARTIAL_CAPTURE)
        if amount.currency != attempt.amount.currency or amount > attempt.amount:
            raise InvalidAmountError(f"capture {amount} must be <= authorized {attempt.amount}")
        body = self._call("capture_body", attempt, amount)
        ids = self._attempt_ids(intent.gateway_intent_id, attempt)
        await self._send(op, ids=ids, body=self._json_body(body, "capture_body"))
        return await self.get_payment(request.gateway_intent_id)

    async def refund(self, request: RefundRequest) -> Refund:
        op = self._need(self._m.operations.refund, Capability.REFUND)
        intent = await self.get_payment(request.gateway_intent_id)
        captured = [a for a in intent.attempts if a.status is AttemptStatus.CAPTURED]
        attempt = self._single(captured, "captured attempt", request.gateway_intent_id)
        refundable = attempt.amount - attempt.amount_refunded
        amount = request.amount or refundable
        if amount.currency != refundable.currency:
            raise InvalidAmountError(f"refund currency {amount.currency} != {refundable.currency}")
        if amount.is_zero or amount > refundable:
            raise InvalidAmountError(f"refund {amount} must be > 0 and <= refundable {refundable}")
        if amount != refundable and Capability.PARTIAL_REFUND not in self._capabilities:
            raise UnsupportedCapabilityError(self._name, Capability.PARTIAL_REFUND)
        self._check_amount(amount)
        body = self._call("refund_body", amount, request.reason)
        ids = self._attempt_ids(intent.gateway_intent_id, attempt)
        headers = {op.idempotency_header: request.idempotency_key} if op.idempotency_header else {}
        entity = await self._send(
            op, ids=ids, body=self._json_body(body, "refund_body"), headers=headers
        )
        return cast(Refund, self._call("refund", entity, self._ctx))

    async def get_refund(
        self, gateway_refund_id: str, *, gateway_intent_id: str | None = None
    ) -> Refund:
        op = self._need(self._m.operations.get_refund, Capability.GET_REFUND)
        ids = {"refund_id": self._checked_id(gateway_refund_id, self._m.ids.refund)}
        if "intent_id" in op.placeholders():
            if gateway_intent_id is None:
                raise InvalidRequestError(
                    f"{self._m.gateway} looks refunds up by intent: pass gateway_intent_id",
                    gateway=self._name,
                )
            ids["intent_id"] = self._checked_id(gateway_intent_id, self._m.ids.intent)
        entity = await self._send(op, ids=ids)
        return cast(Refund, self._call("refund", entity, self._ctx))

    # --- signed inputs ----------------------------------------------------------------------

    def verify_and_parse(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookEvent:
        hooks = self._m.webhooks
        secrets = self._webhook_secrets()
        if hooks is None or not secrets:
            raise UnsupportedCapabilityError(self._name, Capability.WEBHOOKS)
        lower = {k.lower(): v for k, v in headers.items()}
        signature = lower.get(hooks.signature_header.lower())
        message = raw_body
        if hooks.timestamp_header is not None:
            stamp = lower.get(hooks.timestamp_header.lower(), "")
            if not self._fresh(stamp, hooks.max_age_seconds):
                raise InvalidSignatureError(
                    "webhook timestamp missing or too old", gateway=self._name
                )
            message = stamp.encode() + raw_body
        if not signature or not verify_signature(hooks.scheme, message, signature, secrets):
            raise InvalidSignatureError("webhook signature did not verify", gateway=self._name)
        # Only parse after the signature check passed.
        event_id = lower.get(hooks.event_id_header.lower())
        if not event_id:
            raise InvalidRequestError(
                f"missing {hooks.event_id_header.lower()} header", gateway=self._name
            )
        try:
            body = json.loads(raw_body)
        except ValueError as exc:
            raise InvalidRequestError("webhook body is not JSON", gateway=self._name) from exc
        if not isinstance(body, dict):
            raise InvalidRequestError("webhook body is not a JSON object", gateway=self._name)
        return cast(WebhookEvent, self._call("webhook_event", body, event_id, self._ctx))

    def verify_checkout_return(self, params: Mapping[str, str]) -> str:
        cr = self._m.checkout_return
        if cr is None:
            raise UnsupportedCapabilityError(self._name, Capability.CHECKOUT_RETURN)
        names = cr.params()
        values = {n: params.get(n, "") for n in names}
        signature = params.get(cr.signature_param, "")
        key = _secret(getattr(self._config, cr.key_credential))
        if not (all(values.values()) and signature) or not verify_signature(
            cr.scheme, cr.message_template.format_map(values).encode(), signature, [key]
        ):
            raise InvalidSignatureError("checkout signature did not verify", gateway=self._name)
        return values[cr.intent_param]

    # --- helpers ----------------------------------------------------------------------------

    def _need[T](self, op: T | None, capability: Capability) -> T:
        if op is None:
            raise UnsupportedCapabilityError(self._name, capability)
        return op

    def _webhook_secrets(self) -> list[str]:
        hooks = self._m.webhooks
        if hooks is None:
            return []
        value = getattr(self._config, hooks.secrets_credential)
        values = value if isinstance(value, tuple) else (value,) if value else ()
        return [_secret(s) for s in values]

    def _fresh(self, stamp: str, max_age: int) -> bool:
        """A signed timestamp within ``max_age`` of now (seconds or milliseconds)."""
        if not stamp.isdigit():
            return False
        seconds = int(stamp) / (1000 if len(stamp) > 11 else 1)
        return abs(self._clock().timestamp() - seconds) <= max_age

    def _call(self, mapper: str, *args: object) -> Any:
        fn = self._package.mappers.get(mapper)
        if fn is None:  # only optional mappers can be absent (checked at load)
            return None
        try:
            return fn(*args)
        except GatewayError as exc:
            exc.gateway = self._name
            raise
        except Exception as exc:
            raise UnknownGatewayError(
                f"mapper {mapper} failed: {type(exc).__name__}: {exc}", gateway=self._name
            ) from exc

    def _json_body(self, body: object, mapper: str) -> Json:
        if not isinstance(body, dict):
            raise UnknownGatewayError(f"mapper {mapper} must return a dict", gateway=self._name)
        return body

    def _check_amount(self, amount: Money) -> None:
        if problem := self._call("check_amount", amount):
            raise InvalidAmountError(str(problem))

    def _attempt_ids(self, intent_id: str, attempt: PaymentAttempt) -> dict[str, str]:
        return {
            "intent_id": self._checked_id(intent_id, self._m.ids.intent),
            "attempt_id": self._checked_id(attempt.gateway_attempt_id, self._m.ids.attempt),
        }

    def _checked_id(self, value: str, pattern: str) -> str:
        # Ids go into URL paths: they must match the native format, and are escaped anyway.
        if not re.fullmatch(pattern, value):
            raise InvalidRequestError(f"not a {self._m.gateway} id: {value!r}", gateway=self._name)
        return quote(value, safe="")

    async def _send(
        self,
        request: Request,
        *,
        ids: Mapping[str, str] | None = None,
        body: Json | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Json:
        path = request.path.format_map(dict(ids or {}))
        return await self._client.request(
            request.method, path, body=body if request.method != "GET" else None, headers=headers
        )

    def _single(self, attempts: list[PaymentAttempt], what: str, intent_id: str) -> PaymentAttempt:
        if len(attempts) != 1:
            raise InvalidRequestError(
                f"intent {intent_id} has {len(attempts)} {what}s; expected exactly one"
                + (" (pass attempt_id)" if len(attempts) > 1 else ""),
                gateway=self._name,
            )
        return attempts[0]


class PackageProvider:
    """Factory provider for one loaded package (``GatewayFactory.register_provider``).

    Refuses packages that weren't approved (gate G8). ``allow_unsigned`` is for
    tests and for trying out a package the agent just generated, never for serving.
    """

    def __init__(
        self,
        package: GatewayPackage,
        *,
        type_name: str | None = None,
        allow_unsigned: bool = False,
    ) -> None:
        if package.approval is None and not allow_unsigned:
            m = package.manifest
            raise PackageLoadError(
                f"{m.gateway} v{m.version} is not approved; load it with "
                "signing.load_signed_package (sign it with `python -m payweave.packages sign`)"
            )
        self._package = package
        self._type_name = type_name or package.manifest.gateway
        self._config_model = config_model(package.manifest)

    @property
    def type_name(self) -> str:
        return self._type_name

    @property
    def config_model(self) -> type[BaseModel]:
        return self._config_model

    def build(self, config: BaseModel, context: GatewayContext) -> PackageGateway:
        return PackageGateway(self._package, config, context)
