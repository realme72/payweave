"""PayWeave error categories.

Two families:

* ``DomainError``: our own invariants were violated (bad amount, illegal status
  transition, unknown gateway). These are programming or input errors, never
  retried.
* ``GatewayError``: something went wrong talking to a gateway. Every gateway
  maps its own codes onto exactly one of these categories.

Two flags on each ``GatewayError`` matter for safe retries:

* ``retryable``: the call may be repeated **with the same idempotency key**.
* ``outcome_unknown``: money may or may not have moved (e.g. a timeout after the
  request was sent). The caller must check the payment status (or rely on the
  gateway's native idempotency) before doing anything else. Never retry such a
  call blindly with a new key.
"""

from __future__ import annotations

from typing import ClassVar


class PayWeaveError(Exception):
    """Base for every error PayWeave raises."""


# --- domain errors ----------------------------------------------------------


class DomainError(PayWeaveError):
    """One of our own invariants was violated."""


class InvalidAmountError(DomainError, ValueError):
    pass


class UnknownCurrencyError(DomainError, ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(f"unknown or unsupported ISO 4217 currency: {code!r}")
        self.code = code


class CurrencyMismatchError(DomainError, ValueError):
    def __init__(self, left: str, right: str) -> None:
        super().__init__(f"currency mismatch: {left} vs {right}")
        self.left = left
        self.right = right


class IllegalTransitionError(DomainError):
    def __init__(self, current: str, target: str) -> None:
        super().__init__(f"illegal status transition: {current} -> {target}")
        self.current = current
        self.target = target


class GatewayNotFoundError(DomainError):
    def __init__(self, name: str) -> None:
        super().__init__(f"no gateway registered under {name!r}")
        self.name = name


class DuplicateGatewayError(DomainError):
    def __init__(self, name: str) -> None:
        super().__init__(f"a gateway named {name!r} is already registered")
        self.name = name


class CapabilityMismatchError(DomainError):
    """A gateway's declared capabilities don't match the interfaces it implements."""


class UnsupportedCapabilityError(DomainError):
    def __init__(self, gateway: str, capability: str) -> None:
        super().__init__(f"gateway {gateway!r} does not support {str(capability)!r}")
        self.gateway = gateway
        self.capability = capability


class IdempotencyConflictError(DomainError):
    """The same idempotency key was reused with a different request."""

    def __init__(self, key: str) -> None:
        super().__init__(f"idempotency key {key!r} was already used for a different request")
        self.key = key


class IdempotencyInProgressError(DomainError):
    """A request with this idempotency key is still running (or its outcome is unknown)."""

    def __init__(self, key: str) -> None:
        super().__init__(f"a request with idempotency key {key!r} is still in progress")
        self.key = key


# --- gateway errors ---------------------------------------------------------


class GatewayError(PayWeaveError):
    """A failure reported by (or while talking to) a payment gateway."""

    retryable: ClassVar[bool] = False
    outcome_unknown: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        gateway: str,
        gateway_code: str | None = None,
        decline_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.gateway = gateway
        self.gateway_code = gateway_code
        self.decline_reason = decline_reason

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(gateway={self.gateway!r}, "
            f"gateway_code={self.gateway_code!r}, message={self.message!r})"
        )


class InvalidRequestError(GatewayError):
    """The gateway rejected the request as malformed. A mapping bug: fix, don't retry."""


class AuthenticationFailedError(GatewayError):
    """Bad or missing credentials."""


class CardDeclinedError(GatewayError):
    """The issuer declined. ``decline_reason`` carries the gateway's reason when given."""


class InsufficientFundsError(CardDeclinedError):
    pass


class FraudSuspectedError(CardDeclinedError):
    pass


class ActionRequiredError(GatewayError):
    """The operation needs customer action (3DS, OTP, UPI approval) before it can finish."""


class DuplicateRequestError(GatewayError):
    """The gateway saw this idempotency key already with different parameters."""


class InvalidSignatureError(GatewayError):
    """A webhook's signature didn't verify. Never process such an event."""


class RateLimitedError(GatewayError):
    retryable = True


class GatewayUnavailableError(GatewayError):
    """The gateway refused the request before processing it (e.g. 503). Safe to retry."""

    retryable = True


class GatewayTimeoutError(GatewayError):
    """No response after the request was sent: the money may have moved."""

    retryable = True
    outcome_unknown = True


class UnknownGatewayError(GatewayError):
    """A gateway error we couldn't classify. Treated as outcome-unknown to stay safe."""

    outcome_unknown = True
