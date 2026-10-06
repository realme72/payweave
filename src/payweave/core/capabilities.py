"""Gateway capability interfaces (Strategy pattern, split per Interface Segregation).

A gateway implements only the interfaces for the operations it supports and
declares them in ``capabilities``. The registry checks at registration time
that every declared operation is actually implemented. A gateway may implement
more than it declares, e.g. to switch webhooks off when no secret is configured.

Some capabilities are *behaviour flags* with no extra methods (e.g. partial
refunds): they tell callers and the contract suite what to expect from
methods the gateway already has.

Authorization is not a separate method: ``create_payment`` with
``CaptureMode.MANUAL`` authorizes only, mirroring how most gateways work.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from payweave.core.models import (
    CaptureRequest,
    CreatePaymentRequest,
    PaymentIntent,
    Refund,
    RefundRequest,
    VoidRequest,
    WebhookEvent,
)


class Capability(StrEnum):
    # operations (each maps to an interface below)
    CREATE_PAYMENT = "create_payment"
    CAPTURE = "capture"
    VOID = "void"
    REFUND = "refund"
    GET_PAYMENT = "get_payment"
    GET_REFUND = "get_refund"
    WEBHOOKS = "webhooks"
    CHECKOUT_RETURN = "checkout_return"  # verify the signed result the client brings back
    # behaviour flags (no extra methods)
    MANUAL_CAPTURE = "manual_capture"  # create_payment honours CaptureMode.MANUAL
    PARTIAL_CAPTURE = "partial_capture"
    PARTIAL_REFUND = "partial_refund"
    MULTIPLE_REFUNDS = "multiple_refunds"
    # the gateway itself dedupes this operation on our idempotency key
    NATIVE_IDEMPOTENT_CREATE = "native_idempotent_create"
    NATIVE_IDEMPOTENT_CAPTURE = "native_idempotent_capture"
    NATIVE_IDEMPOTENT_REFUND = "native_idempotent_refund"


@runtime_checkable
class Gateway(Protocol):
    """What every gateway has: a unique name and its declared capabilities."""

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[Capability]: ...


@runtime_checkable
class PaymentCreator(Gateway, Protocol):
    async def create_payment(self, request: CreatePaymentRequest) -> PaymentIntent: ...


@runtime_checkable
class Capturer(Gateway, Protocol):
    async def capture(self, request: CaptureRequest) -> PaymentIntent: ...


@runtime_checkable
class Voider(Gateway, Protocol):
    async def void(self, request: VoidRequest) -> PaymentIntent: ...


@runtime_checkable
class Refunder(Gateway, Protocol):
    async def refund(self, request: RefundRequest) -> Refund: ...


@runtime_checkable
class PaymentReader(Gateway, Protocol):
    async def get_payment(self, gateway_intent_id: str) -> PaymentIntent: ...


@runtime_checkable
class RefundReader(Gateway, Protocol):
    async def get_refund(self, gateway_refund_id: str) -> Refund: ...


@runtime_checkable
class WebhookHandler(Gateway, Protocol):
    def verify_and_parse(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookEvent:
        """Verify the signature over the *raw* body, then parse it.

        Must raise ``InvalidSignatureError`` before any parsing if verification
        fails. Header lookup must be case-insensitive.
        """
        ...


@runtime_checkable
class CheckoutReturnVerifier(Gateway, Protocol):
    def verify_checkout_return(self, params: Mapping[str, str]) -> str:
        """Verify the signed result the customer's browser brings back from checkout.

        Returns the ``gateway_intent_id`` it belongs to, or raises
        ``InvalidSignatureError``. A valid signature proves who sent the data,
        not the final payment state: callers must still fetch the intent.
        """
        ...


# Operation capability -> the interface that provides it.
INTERFACE_FOR: MappingProxyType[Capability, type[Gateway]] = MappingProxyType(
    {
        Capability.CREATE_PAYMENT: PaymentCreator,
        Capability.CAPTURE: Capturer,
        Capability.VOID: Voider,
        Capability.REFUND: Refunder,
        Capability.GET_PAYMENT: PaymentReader,
        Capability.GET_REFUND: RefundReader,
        Capability.WEBHOOKS: WebhookHandler,
        Capability.CHECKOUT_RETURN: CheckoutReturnVerifier,
    }
)
