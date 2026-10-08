"""Gateway Package manifest: the declarative half of a package (PLAN.md §5).

Strict Pydantic: unknown keys are rejected, every cross-reference (credential
names, path placeholders, hosts) is checked, and anything security-relevant is
a choice from a closed list. The JSON Schema of ``Manifest`` is what the agent
is asked to fill in.
"""

from __future__ import annotations

import re
import string
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from payweave.packages.schemes import AuthScheme, SignatureScheme

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
HeaderName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,99}$")]
PLACEHOLDERS = frozenset({"intent_id", "attempt_id", "refund_id"})

ErrorCategory = Literal[
    "invalid_request",
    "authentication_failed",
    "duplicate_request",
    "card_declined",
    "insufficient_funds",
    "fraud_suspected",
    "action_required",
    "rate_limited",
]
Flag = Literal["manual_capture", "partial_capture", "partial_refund", "multiple_refunds"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Credential(_Strict):
    secret: bool = False  # held as SecretStr; never reaches mappers, logs or errors
    public: bool = False  # may be handed to mappers (e.g. a key id for browser checkout)
    multiple: bool = False  # a list, e.g. webhook secrets during rotation
    required: bool = True
    pattern: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.secret and self.public:
            raise ValueError("a credential can't be both secret and public")
        if self.pattern is not None:
            re.compile(self.pattern)
        return self


class Setting(_Strict):
    type: Literal["int", "str", "bool"]
    default: int | str | bool
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        kind = {"int": int, "str": str, "bool": bool}[self.type]
        if type(self.default) is not kind:
            raise ValueError(f"default {self.default!r} is not {self.type}")
        if (self.minimum is not None or self.maximum is not None) and self.type != "int":
            raise ValueError("minimum/maximum apply to int settings only")
        if self.choices is not None and (self.type != "str" or self.default not in self.choices):
            raise ValueError("choices apply to str settings and must include the default")
        return self


class Api(_Strict):
    base_url: str
    # Hosts a human confirmed from the docs (spec card). Requests go nowhere else.
    allowed_hosts: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        url = urlsplit(self.base_url)
        if url.scheme != "https" or not url.hostname:
            raise ValueError("base_url must be an https URL")
        if url.hostname not in self.allowed_hosts:
            raise ValueError(f"base_url host {url.hostname!r} is not in allowed_hosts")
        if url.query or url.fragment or url.username or url.password:
            raise ValueError("base_url must not carry a query, fragment or credentials")
        if not self.base_url.endswith("/"):
            raise ValueError("base_url must end with '/'")
        return self


class Auth(_Strict):
    scheme: AuthScheme
    username: Name | None = None  # credential names
    password: Name  # the secret: basic password or bearer token

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.scheme == "basic") != (self.username is not None):
            raise ValueError("basic auth needs a username credential; bearer must not have one")
        return self


class LiveGuard(_Strict):
    """Refuse live credentials unless the instance sets ``allow_live``."""

    credential: Name
    live_pattern: str

    @field_validator("live_pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        re.compile(value)
        return value


class Ids(_Strict):
    """Formats of native ids. Ids are put into URL paths, so they must match first."""

    intent: str
    attempt: str
    refund: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, pattern in self:
            if not (pattern.startswith("^") and pattern.endswith("$")):
                raise ValueError(f"ids.{name} must be anchored with ^ and $")
            compiled = re.compile(pattern)
            # Probe with the pattern's literal prefix (order_, pay_...) so the
            # unsafe part is what gets tested.
            prefix = re.match(r"\^([A-Za-z0-9_-]*)", pattern).group(1)  # type: ignore[union-attr]
            for tail in ("x/y", "../x", "x?y", "x#y", "x%2F", "x y", "x\\y", ""):
                probe = prefix + tail
                if compiled.fullmatch(probe):
                    raise ValueError(f"ids.{name} accepts the unsafe id {probe!r}")
        return self


class Request(_Strict):
    method: Literal["GET", "POST"]
    path: str  # relative to base_url; placeholders: {intent_id}, {attempt_id}, {refund_id}

    @field_validator("path")
    @classmethod
    def _check_path(cls, path: str) -> str:
        if path.startswith("/") or "://" in path or ".." in path or "?" in path:
            raise ValueError("path must be relative to base_url, without '..' or a query")
        names = {name for _, name, _, _ in string.Formatter().parse(path) if name is not None}
        if not names <= PLACEHOLDERS:
            raise ValueError(f"unknown placeholders {sorted(names - PLACEHOLDERS)}")
        return path

    def placeholders(self) -> frozenset[str]:
        return frozenset(n for _, n, _, _ in string.Formatter().parse(self.path) if n)


class GetPayment(_Strict):
    primary: Request  # the intent entity
    related: dict[Name, Request] = Field(default_factory=dict)  # e.g. its attempts


class RefundOperation(Request):
    idempotency_header: HeaderName | None = None  # gateway-native refund idempotency


class Operations(_Strict):
    create_payment: Request | None = None
    get_payment: GetPayment | None = None
    capture: Request | None = None
    refund: RefundOperation | None = None
    get_refund: Request | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = {
            "create_payment": frozenset(),
            "capture": frozenset({"attempt_id"}),
            "refund": frozenset({"attempt_id"}),
            "get_refund": frozenset({"refund_id"}),
        }
        for op, allowed in expected.items():
            req = getattr(self, op)
            if req is not None and not req.placeholders() <= allowed:
                raise ValueError(f"operations.{op} may only use {sorted(allowed) or 'no'} ids")
        if self.get_payment is not None:
            for req in (self.get_payment.primary, *self.get_payment.related.values()):
                if req.method != "GET" or not req.placeholders() <= {"intent_id"}:
                    raise ValueError("get_payment requests must be GETs using only {intent_id}")
        if self.get_refund is not None and self.get_refund.method != "GET":
            raise ValueError("get_refund must be a GET")
        if (self.capture or self.refund) and self.get_payment is None:
            raise ValueError("capture and refund need get_payment (they act on fetched attempts)")
        return self


class ErrorRule(_Strict):
    """Classify a 4xx response by its machine code or message. First match wins."""

    category: ErrorCategory
    code: str | None = None  # exact match on the extracted code
    message_contains: str | None = None  # case-insensitive substring

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.code is None) == (self.message_contains is None):
            raise ValueError("a rule needs exactly one of code / message_contains")
        return self


class Errors(_Strict):
    """Where to read error details in a failed response body (dotted paths, first non-empty)."""

    message_paths: tuple[str, ...] = ()
    code_paths: tuple[str, ...] = ()
    rules: tuple[ErrorRule, ...] = ()


class Webhooks(_Strict):
    scheme: SignatureScheme
    signature_header: HeaderName
    event_id_header: HeaderName
    secrets_credential: Name


class CheckoutReturn(_Strict):
    """Verify what the browser brings back from the gateway's checkout."""

    scheme: SignatureScheme
    message_template: str  # e.g. "{razorpay_order_id}|{razorpay_payment_id}"
    key_credential: Name
    signature_param: Name
    intent_param: Name  # the param holding the intent id to return

    @field_validator("message_template")
    @classmethod
    def _plain_fields(cls, template: str) -> str:
        for _, name, spec, conversion in string.Formatter().parse(template):
            if name is not None and (
                not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) or spec or conversion
            ):
                raise ValueError(f"template fields must be plain parameter names, not {name!r}")
        return template

    def params(self) -> frozenset[str]:
        names = string.Formatter().parse(self.message_template)
        return frozenset(n for _, n, _, _ in names if n)


class Manifest(_Strict):
    schema_version: Literal[1]
    gateway: Name
    version: Annotated[str, Field(pattern=r"^[0-9]+(\.[0-9]+)*$")]
    api: Api
    auth: Auth
    credentials: dict[Name, Credential]
    settings: dict[Name, Setting] = Field(default_factory=dict)
    live_guard: LiveGuard | None = None
    ids: Ids
    operations: Operations
    flags: tuple[Flag, ...] = ()
    errors: Errors = Field(default_factory=Errors)
    webhooks: Webhooks | None = None
    checkout_return: CheckoutReturn | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:  # noqa: PLR0912 - one check per cross-reference
        def credential(name: str, where: str) -> Credential:
            if name not in self.credentials:
                raise ValueError(f"{where} names unknown credential {name!r}")
            return self.credentials[name]

        if not credential(self.auth.password, "auth.password").secret:
            raise ValueError("auth.password must be a secret credential")
        if self.auth.username is not None:
            credential(self.auth.username, "auth.username")
        if self.live_guard is not None:
            credential(self.live_guard.credential, "live_guard.credential")
        if self.webhooks is not None:
            c = credential(self.webhooks.secrets_credential, "webhooks.secrets_credential")
            if not (c.secret and c.multiple and not c.required):
                raise ValueError("webhook secrets must be an optional, multiple, secret credential")
        if self.checkout_return is not None:
            cr = self.checkout_return
            if not credential(cr.key_credential, "checkout_return.key_credential").secret:
                raise ValueError("checkout_return.key_credential must be secret")
            if cr.intent_param not in cr.params():
                raise ValueError("checkout_return.intent_param must be part of the signed message")
        reserved = {"allow_live", "timeout_seconds"}
        overlap = (set(self.credentials) & set(self.settings)) | (
            (set(self.credentials) | set(self.settings)) & reserved
        )
        if overlap:
            raise ValueError(f"names used twice or reserved: {sorted(overlap)}")
        ops = self.operations
        if "manual_capture" in self.flags and ops.capture is None:
            raise ValueError("manual_capture needs a capture operation")
        if "partial_capture" in self.flags and "manual_capture" not in self.flags:
            raise ValueError("partial_capture needs manual_capture")
        if {"partial_refund", "multiple_refunds"} & set(self.flags) and ops.refund is None:
            raise ValueError("refund flags need a refund operation")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("flags must not repeat")
        return self
