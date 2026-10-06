from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from payweave.core.errors import CurrencyMismatchError, InvalidAmountError
from payweave.core.money import CURRENCY_EXPONENTS, Money, currency_exponent

currencies = st.sampled_from(sorted(CURRENCY_EXPONENTS))
amounts = st.integers(min_value=0, max_value=10**15)


@pytest.mark.parametrize(
    ("major", "currency", "minor"),
    [("499.99", "INR", 49999), ("5", "JPY", 5), ("1.234", "KWD", 1234), ("0.10", "USD", 10)],
)
def test_from_major_exact(major: str, currency: str, minor: int) -> None:
    assert Money.from_major(major, currency).amount_minor == minor


@pytest.mark.parametrize(
    ("major", "currency"), [("1.234", "INR"), ("0.5", "JPY"), ("1.0001", "KWD"), ("NaN", "INR")]
)
def test_from_major_never_rounds(major: str, currency: str) -> None:
    with pytest.raises(InvalidAmountError):
        Money.from_major(major, currency)


def test_floats_are_rejected_everywhere() -> None:
    with pytest.raises(InvalidAmountError):
        Money.from_major(1.5, "INR")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Money.of(1.5, "INR")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Money.of(True, "INR")


@pytest.mark.parametrize("currency", ["XYZ", "inr", "RUPEE", ""])
def test_unknown_currency_rejected(currency: str) -> None:
    with pytest.raises(ValidationError):
        Money.of(1, currency)


def test_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        Money.of(-1, "INR")
    with pytest.raises(InvalidAmountError):
        Money.of(1, "INR") - Money.of(2, "INR")


def test_mixed_currency_arithmetic_and_comparison_rejected() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money.of(1, "INR") + Money.of(1, "USD")
    with pytest.raises(CurrencyMismatchError):
        _ = Money.of(1, "INR") < Money.of(1, "USD")


@given(amounts, currencies)
def test_major_minor_round_trip(minor: int, currency: str) -> None:
    money = Money.of(minor, currency)
    assert Money.from_major(money.to_major(), currency) == money
    assert isinstance(money.to_major(), Decimal)


@given(amounts, amounts, currencies)
def test_add_sub_inverse(a: int, b: int, currency: str) -> None:
    x, y = Money.of(a, currency), Money.of(b, currency)
    assert (x + y) - y == x
    assert (x + y).amount_minor == a + b


def test_exponents_match_iso() -> None:
    assert (currency_exponent("JPY"), currency_exponent("INR"), currency_exponent("KWD")) == (
        0,
        2,
        3,
    )


def test_money_is_immutable() -> None:
    money = Money.of(1, "INR")
    with pytest.raises(ValidationError):
        money.amount_minor = 2  # type: ignore[misc]
