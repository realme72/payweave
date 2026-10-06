"""Money and currencies.

Amounts are always integers in the currency's minor unit (paise, cents, fils).
Floats are rejected everywhere. Conversions to and from major units go through
``Decimal`` and must be exact, so nothing is rounded silently.

Exponents follow ISO 4217. Some gateways use a different convention for a few
currencies (e.g. treating a zero-decimal currency as two-decimal); that
translation belongs in the gateway's mapper, never here.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt

from payweave.core.errors import CurrencyMismatchError, InvalidAmountError, UnknownCurrencyError

# ISO 4217 code -> minor-unit exponent. Fail closed: unknown codes are rejected.
CURRENCY_EXPONENTS: MappingProxyType[str, int] = MappingProxyType(
    {
        # 0 decimals
        "CLP": 0, "ISK": 0, "JPY": 0, "KRW": 0, "PYG": 0, "UGX": 0, "VND": 0, "XAF": 0, "XOF": 0,
        # 2 decimals
        "AED": 2, "AUD": 2, "BDT": 2, "BRL": 2, "CAD": 2, "CHF": 2, "CNY": 2, "CZK": 2, "DKK": 2,
        "EGP": 2, "EUR": 2, "GBP": 2, "HKD": 2, "HUF": 2, "IDR": 2, "ILS": 2, "INR": 2, "KES": 2,
        "LKR": 2, "MXN": 2, "MYR": 2, "NGN": 2, "NOK": 2, "NPR": 2, "NZD": 2, "PHP": 2, "PKR": 2,
        "PLN": 2, "QAR": 2, "RON": 2, "SAR": 2, "SEK": 2, "SGD": 2, "THB": 2, "TRY": 2, "TWD": 2,
        "USD": 2, "ZAR": 2,
        # 3 decimals
        "BHD": 3, "IQD": 3, "JOD": 3, "KWD": 3, "LYD": 3, "OMR": 3, "TND": 3,
    }
)  # fmt: skip


def _validate_currency(code: str) -> str:
    if code not in CURRENCY_EXPONENTS:
        raise UnknownCurrencyError(code)
    return code


CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$"), AfterValidator(_validate_currency)]


def currency_exponent(code: str) -> int:
    return CURRENCY_EXPONENTS[_validate_currency(code)]


class Money(BaseModel):
    """A non-negative amount in minor units of one currency. Immutable."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    amount_minor: StrictInt = Field(ge=0)
    currency: CurrencyCode

    # --- construction -----------------------------------------------------

    @classmethod
    def of(cls, amount_minor: int, currency: str) -> Self:
        return cls(amount_minor=amount_minor, currency=currency)

    @classmethod
    def zero(cls, currency: str) -> Self:
        return cls(amount_minor=0, currency=currency)

    @classmethod
    def from_major(cls, amount: Decimal | str, currency: str) -> Self:
        """Build from a major-unit value, e.g. ``("499.99", "INR")`` -> 49999 paise.

        Raises ``InvalidAmountError`` if the value has more decimals than the
        currency allows. Floats are not accepted.
        """
        if not isinstance(amount, Decimal | str):
            raise InvalidAmountError(f"major amount must be Decimal or str, got {type(amount)!r}")
        try:
            value = Decimal(amount)
        except InvalidOperation as exc:
            raise InvalidAmountError(f"not a decimal number: {amount!r}") from exc
        if not value.is_finite():
            raise InvalidAmountError(f"amount must be finite: {amount!r}")
        scaled = value.scaleb(currency_exponent(currency))
        if scaled != scaled.to_integral_value():
            raise InvalidAmountError(f"{amount!r} has more decimal places than {currency} allows")
        return cls(amount_minor=int(scaled), currency=currency)

    # --- conversion -------------------------------------------------------

    def to_major(self) -> Decimal:
        return Decimal(self.amount_minor).scaleb(-currency_exponent(self.currency))

    # --- arithmetic (same currency only) -----------------------------------

    def _check_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(self.currency, other.currency)

    def __add__(self, other: Money) -> Money:
        self._check_same_currency(other)
        return Money(amount_minor=self.amount_minor + other.amount_minor, currency=self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check_same_currency(other)
        if other.amount_minor > self.amount_minor:
            raise InvalidAmountError("subtraction would produce a negative amount")
        return Money(amount_minor=self.amount_minor - other.amount_minor, currency=self.currency)

    def __lt__(self, other: Money) -> bool:
        self._check_same_currency(other)
        return self.amount_minor < other.amount_minor

    def __le__(self, other: Money) -> bool:
        self._check_same_currency(other)
        return self.amount_minor <= other.amount_minor

    def __gt__(self, other: Money) -> bool:
        self._check_same_currency(other)
        return self.amount_minor > other.amount_minor

    def __ge__(self, other: Money) -> bool:
        self._check_same_currency(other)
        return self.amount_minor >= other.amount_minor

    @property
    def is_zero(self) -> bool:
        return self.amount_minor == 0

    def __str__(self) -> str:
        return f"{self.to_major()} {self.currency}"
