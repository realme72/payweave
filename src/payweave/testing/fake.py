"""FakeGateway: an in-memory gateway implementing every operation.

Two jobs:

* the reference the contract suite is checked against first: if a contract test
  fails on the fake, the test is wrong, not the gateway;
* a stand-in for app code and demos that need a gateway without a network.

It follows the canonical semantics directly (it has no native model to map),
written independently of any real gateway's mapping code.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, SecretStr

from payweave.core.capabilities import Capability, Gateway
from payweave.core.errors import (
    DuplicateRequestError,
    InvalidAmountError,
    InvalidRequestError,
    InvalidSignatureError,
    UnsupportedCapabilityError,
)
from payweave.core.factory import GatewayContext
from payweave.core.models import (
    CaptureMode,
    CaptureRequest,
    ClientCheckout,
    CreatePaymentRequest,
    FailureCategory,
    FailureInfo,
    PaymentAttempt,
    PaymentIntent,
    Refund,
    RefundRequest,
    VoidRequest,
    WebhookEvent,
    WebhookEventType,
)
from payweave.core.money import Money
from payweave.core.status import AttemptStatus, IntentStatus, RefundStatus

SIGNATURE_HEADER = "x-fake-signature"
EVENT_ID_HEADER = "x-fake-event-id"

CAPABILITIES = frozenset(
    {
        Capability.CREATE_PAYMENT,
        Capability.GET_PAYMENT,
        Capability.CAPTURE,
        Capability.VOID,
        Capability.REFUND,
        Capability.GET_REFUND,
        Capability.WEBHOOKS,
        Capability.CHECKOUT_RETURN,
        Capability.MANUAL_CAPTURE,
        Capability.PARTIAL_REFUND,
        Capability.MULTIPLE_REFUNDS,
        Capability.NATIVE_IDEMPOTENT_REFUND,
    }
)  # PARTIAL_CAPTURE deliberately absent: the suite must cover "unsupported" too.


def sign(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


@dataclass
class _Intent:
    request: CreatePaymentRequest
    created_at: datetime
    attempts: list[PaymentAttempt] = field(default_factory=list)


class FakeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    secret: SecretStr = SecretStr("fake_secret")  # signs webhooks and checkout returns


class FakeGateway:
    def __init__(
        self,
        name: str = "fake",
        *,
        secret: str = "fake_secret",  # noqa: S107 - a test double, not a credential
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._name = name
        self._secret = secret
        self._clock = clock
        self._ids = itertools.count(1)
        self._intents: dict[str, _Intent] = {}
        self._refunds: dict[str, tuple[str, Refund]] = {}  # id -> (intent id, refund)
        self._refund_keys: dict[str, tuple[str, Refund]] = {}  # key -> (fingerprint, refund)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> frozenset[Capability]:
        return CAPABILITIES

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids):08d}"

    def _error(self, message: str) -> InvalidRequestError:
        return InvalidRequestError(message, gateway=self._name)

    # --- simulation hooks (what the customer does in checkout) ---------------

    def customer_pays(self, intent_id: str, *, succeed: bool = True) -> str:
        """Add one attempt, as if the customer completed checkout. Returns its id."""
        intent = self._get(intent_id)
        if succeed:
            automatic = intent.request.capture_mode is CaptureMode.AUTOMATIC
            status = AttemptStatus.CAPTURED if automatic else AttemptStatus.AUTHORIZED
            failure = None
        else:
            status = AttemptStatus.FAILED
            failure = FailureInfo(category=FailureCategory.DECLINED, gateway_code="declined")
        attempt = PaymentAttempt(
            gateway_attempt_id=self._new_id("fake_pa"),
            status=status,
            amount=intent.request.amount,
            amount_refunded=Money.zero(intent.request.amount.currency),
            method="card",
            failure=failure,
            created_at=self._clock(),
        )
        intent.attempts.append(attempt)
        return attempt.gateway_attempt_id

    def checkout_signature(self, intent_id: str, attempt_id: str) -> str:
        return sign(self._secret, f"{intent_id}|{attempt_id}".encode())

    def webhook_signature(self, body: bytes) -> str:
        return sign(self._secret, body)

    # --- operations ----------------------------------------------------------

    async def create_payment(self, request: CreatePaymentRequest) -> PaymentIntent:
        intent_id = self._new_id("fake_pi")
        self._intents[intent_id] = _Intent(request, self._clock())
        return self._view(intent_id)

    async def get_payment(self, gateway_intent_id: str) -> PaymentIntent:
        return self._view(gateway_intent_id)

    async def capture(self, request: CaptureRequest) -> PaymentIntent:
        intent = self._get(request.gateway_intent_id)
        authorized = [a for a in intent.attempts if a.status is AttemptStatus.AUTHORIZED]
        if request.attempt_id is not None:
            authorized = [a for a in authorized if a.gateway_attempt_id == request.attempt_id]
        if not authorized and any(a.status is AttemptStatus.CAPTURED for a in intent.attempts):
            return self._view(request.gateway_intent_id)  # already captured: idempotent
        if len(authorized) != 1:
            raise self._error(f"{len(authorized)} authorized attempts; expected exactly one")
        attempt = authorized[0]
        if request.amount is not None and request.amount != attempt.amount:
            raise UnsupportedCapabilityError(self._name, Capability.PARTIAL_CAPTURE)
        self._replace(intent, attempt, status=AttemptStatus.CAPTURED)
        return self._view(request.gateway_intent_id)

    async def void(self, request: VoidRequest) -> PaymentIntent:
        intent = self._get(request.gateway_intent_id)
        authorized = [a for a in intent.attempts if a.status is AttemptStatus.AUTHORIZED]
        if not authorized and any(a.status is AttemptStatus.VOIDED for a in intent.attempts):
            return self._view(request.gateway_intent_id)
        if len(authorized) != 1:
            raise self._error(f"{len(authorized)} authorized attempts; expected exactly one")
        self._replace(intent, authorized[0], status=AttemptStatus.VOIDED)
        return self._view(request.gateway_intent_id)

    async def refund(self, request: RefundRequest) -> Refund:
        fingerprint = request.model_dump_json(exclude={"idempotency_key"})
        if (seen := self._refund_keys.get(request.idempotency_key)) is not None:
            if seen[0] != fingerprint:
                raise DuplicateRequestError(
                    "idempotency key reused with a different request", gateway=self._name
                )
            return seen[1]
        intent = self._get(request.gateway_intent_id)
        captured = [a for a in intent.attempts if a.status is AttemptStatus.CAPTURED]
        if len(captured) != 1:
            raise self._error(f"{len(captured)} captured attempts; expected exactly one")
        attempt = captured[0]
        refundable = attempt.amount - attempt.amount_refunded
        amount = request.amount or refundable
        if amount.currency != refundable.currency:
            raise InvalidAmountError(f"refund currency {amount.currency} != {refundable.currency}")
        if amount.is_zero or amount > refundable:
            raise InvalidAmountError(f"refund {amount} must be > 0 and <= refundable {refundable}")
        self._replace(intent, attempt, amount_refunded=attempt.amount_refunded + amount)
        refund = Refund(
            gateway=self._name,
            gateway_refund_id=self._new_id("fake_rf"),
            gateway_attempt_id=attempt.gateway_attempt_id,
            status=RefundStatus.SUCCEEDED,
            amount=amount,
            created_at=self._clock(),
        )
        self._refunds[refund.gateway_refund_id] = (request.gateway_intent_id, refund)
        self._refund_keys[request.idempotency_key] = (fingerprint, refund)
        return refund

    async def get_refund(
        self, gateway_refund_id: str, *, gateway_intent_id: str | None = None
    ) -> Refund:
        try:
            return self._refunds[gateway_refund_id][1]
        except KeyError:
            raise self._error(f"no refund {gateway_refund_id!r}") from None

    def verify_and_parse(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookEvent:
        lower = {k.lower(): v for k, v in headers.items()}
        signature = lower.get(SIGNATURE_HEADER, "")
        if not hmac.compare_digest(signature, self.webhook_signature(raw_body)):
            raise InvalidSignatureError("webhook signature did not verify", gateway=self._name)
        body = json.loads(raw_body)
        return WebhookEvent(
            gateway=self._name,
            event_id=lower.get(EVENT_ID_HEADER) or body["id"],
            event_type=WebhookEventType.INTENT_UPDATED,
            gateway_event_type=body["type"],
            occurred_at=self._clock(),
            gateway_intent_id=body["intent_id"],
        )

    def verify_checkout_return(self, params: Mapping[str, str]) -> str:
        intent_id = params.get("intent_id", "")
        attempt_id = params.get("attempt_id", "")
        signature = params.get("signature", "")
        expected = self.checkout_signature(intent_id, attempt_id)
        if not (intent_id and attempt_id and hmac.compare_digest(signature, expected)):
            raise InvalidSignatureError("checkout signature did not verify", gateway=self._name)
        return intent_id

    # --- state -----------------------------------------------------------------

    def _get(self, intent_id: str) -> _Intent:
        try:
            return self._intents[intent_id]
        except KeyError:
            raise self._error(f"no intent {intent_id!r}") from None

    @staticmethod
    def _replace(intent: _Intent, attempt: PaymentAttempt, **changes: object) -> None:
        # Re-validate (not model_copy) so an illegal change fails here, loudly.
        index = intent.attempts.index(attempt)
        intent.attempts[index] = PaymentAttempt.model_validate({**dict(attempt), **changes})

    def _view(self, intent_id: str) -> PaymentIntent:
        intent = self._get(intent_id)
        req, attempts = intent.request, intent.attempts
        zero = Money.zero(req.amount.currency)
        captured = sum((a.amount for a in attempts if a.status is AttemptStatus.CAPTURED), zero)
        refunded = sum((a.amount_refunded for a in attempts), zero)
        next_action = None
        if not captured.is_zero:
            if refunded.is_zero:
                status = IntentStatus.CAPTURED
            elif refunded < captured:
                status = IntentStatus.PARTIALLY_REFUNDED
            else:
                status = IntentStatus.REFUNDED
        elif any(a.status is AttemptStatus.AUTHORIZED for a in attempts):
            status = IntentStatus.AUTHORIZED
        elif any(a.status is AttemptStatus.VOIDED for a in attempts):
            status = IntentStatus.VOIDED
        else:
            status = IntentStatus.REQUIRES_ACTION
            next_action = ClientCheckout(gateway=self._name, params={"intent_id": intent_id})
        return PaymentIntent(
            gateway=self._name,
            gateway_intent_id=intent_id,
            merchant_reference=req.merchant_reference,
            status=status,
            capture_mode=req.capture_mode,
            amount=req.amount,
            amount_captured=captured,
            amount_refunded=refunded,
            attempts=tuple(attempts),
            next_action=next_action,
            created_at=intent.created_at,
            metadata=req.metadata,
        )


class FakeProvider:
    """Lets the factory build FakeGateways: ``GatewayFactory([FakeProvider()])``."""

    type_name = "fake"
    config_model = FakeConfig

    def build(self, config: FakeConfig, context: GatewayContext) -> Gateway:
        return FakeGateway(
            context.name, secret=config.secret.get_secret_value(), clock=context.clock
        )
