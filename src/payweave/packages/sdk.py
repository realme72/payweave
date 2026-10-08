"""The only PayWeave module a package's ``mappers.py`` may import.

Mappers are pure functions: native JSON in, canonical models out (or request
bodies out). Everything they can use is re-exported here, so the allowlist
check is simple and this file is the whole surface an AI-written mapper sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from payweave.core.errors import UnknownGatewayError
from payweave.core.models import (
    CaptureMode,
    ClientCheckout,
    CreatePaymentRequest,
    FailureCategory,
    FailureInfo,
    PaymentAttempt,
    PaymentIntent,
    Refund,
    WebhookEvent,
    WebhookEventType,
)
from payweave.core.money import Money, currency_exponent
from payweave.core.status import AttemptStatus, IntentStatus, RefundStatus

type Json = Mapping[str, Any]

__all__ = [
    "UTC",
    "AttemptStatus",
    "CaptureMode",
    "ClientCheckout",
    "CreatePaymentRequest",
    "FailureCategory",
    "FailureInfo",
    "IntentStatus",
    "Json",
    "MapperContext",
    "MappingError",
    "Money",
    "PaymentAttempt",
    "PaymentIntent",
    "Refund",
    "RefundStatus",
    "WebhookEvent",
    "WebhookEventType",
    "currency_exponent",
    "from_unix",
    "money",
]


class MappingError(UnknownGatewayError):
    """The gateway returned something the mapping doesn't cover. Fail closed."""

    def __init__(self, message: str, *, gateway: str = "package") -> None:
        super().__init__(message, gateway=gateway)


@dataclass(frozen=True, slots=True)
class MapperContext:
    """What a mapper may know besides its input: the instance name and *public* config.

    Secrets are never passed to mappers. ``public`` holds credentials the
    manifest marks public (e.g. a key id the browser checkout needs).
    """

    gateway: str
    public: Mapping[str, str] = field(default_factory=dict)


def from_unix(ts: object) -> datetime:
    if not isinstance(ts, int) or isinstance(ts, bool):
        raise MappingError(f"expected a unix timestamp, got {ts!r}")
    return datetime.fromtimestamp(ts, tz=UTC)


def money(amount: object, currency: object) -> Money:
    """Minor-unit integer + currency code from native JSON; anything else fails closed."""
    if not isinstance(amount, int) or isinstance(amount, bool) or not isinstance(currency, str):
        raise MappingError(f"bad amount/currency: {amount!r} {currency!r}")
    return Money.of(amount, currency)
