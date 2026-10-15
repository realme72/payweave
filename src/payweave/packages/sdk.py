"""The only PayWeave module a package's ``mappers.py`` may import.

Mappers are pure functions: native JSON in, canonical models out (or request
bodies out). Everything they can use is re-exported here, so the allowlist
check is simple and this file is the whole surface an AI-written mapper sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from payweave.core.errors import InvalidAmountError, UnknownGatewayError
from payweave.core.models import (
    CaptureMode,
    ClientCheckout,
    CreatePaymentRequest,
    Customer,
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
    "Customer",
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
    "from_iso",
    "from_unix",
    "major_amount",
    "money",
    "money_major",
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


def from_iso(ts: object) -> datetime:
    """An ISO 8601 timestamp with a UTC offset (e.g. 2026-10-15T13:05:20+05:30)."""
    if not isinstance(ts, str):
        raise MappingError(f"expected an ISO timestamp, got {ts!r}")
    try:
        value = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise MappingError(f"not an ISO timestamp: {ts!r}") from exc
    if value.tzinfo is None:
        raise MappingError(f"timestamp without a UTC offset: {ts!r}")
    return value


def money_major(amount: object, currency: object) -> Money:
    """A major-unit amount from native JSON (10.01 rupees -> 1001 paise), exactly.

    JSON numbers arrive as int or float; the float's shortest repr is what was sent.
    More decimals than the currency has fails closed.
    """
    if isinstance(amount, bool) or not isinstance(amount, int | float | str):
        raise MappingError(f"bad amount: {amount!r}")
    if not isinstance(currency, str):
        raise MappingError(f"bad currency: {currency!r}")
    try:
        return Money.from_major(
            Decimal(repr(amount) if isinstance(amount, float) else str(amount)), currency
        )
    except (InvalidAmountError, InvalidOperation) as exc:
        raise MappingError(f"bad amount {amount!r} {currency}: {exc}") from exc


def major_amount(value: Money) -> float:
    """For request bodies that take decimal amounts: 1001 paise -> 10.01.

    The float is exact for the 0-3 decimal places currencies use."""
    return float(value.to_major())


def money(amount: object, currency: object) -> Money:
    """Minor-unit integer + currency code from native JSON; anything else fails closed."""
    if not isinstance(amount, int) or isinstance(amount, bool) or not isinstance(currency, str):
        raise MappingError(f"bad amount/currency: {amount!r} {currency!r}")
    return Money.of(amount, currency)
