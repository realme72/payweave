"""Gateway Spec Card: the facts every later mapping step relies on (PLAN.md §6.2).

Every value carries citations: a chunk ``ref`` plus a quote copied from that
chunk. ``null`` means "the docs don't say", which is a valid answer and better
than a guess. Choices are closed lists of what our hand-written libraries
support; ``other`` means a human must extend a library first (PLAN.md §5).

The card is split into sections so each is extracted in a small, separate
conversation: free tiers cap tokens per minute, and short contexts are also
where smaller models are most accurate.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str = Field(description="The `ref` of a search result, e.g. razorpay:errors.md#3")
    quote: str = Field(
        min_length=8,
        max_length=400,
        description="Text copied exactly from that result that states the fact",
    )


class Cited[T](BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: T | None = Field(description="null if the documentation doesn't say")
    citations: list[Citation] = Field(
        description="At least one when value is not null; [] when value is null"
    )

    @model_validator(mode="after")
    def _known_values_are_cited(self) -> Self:
        if self.value is not None and not self.citations:
            raise ValueError("a non-null value needs at least one citation")
        return self


# Names and URLs exactly as the docs write them: one token, no prose. Without
# this, a description that happens to be in the quote ("Type of the error.")
# would pass the verbatim check in place of the field name (`code`).
Identifier = Annotated[str, Field(pattern=r"^\S+$", max_length=200)]


class Section(BaseModel):
    """Base for card sections. ``VERBATIM`` fields must appear in a cited quote."""

    model_config = ConfigDict(extra="forbid")

    TITLE: ClassVar[str]
    GOAL: ClassVar[str]
    VERBATIM: ClassVar[frozenset[str]] = frozenset()

    def cited_fields(self) -> dict[str, Cited[Any]]:
        return {name: value for name, value in self if isinstance(value, Cited)}


class Basics(Section):
    TITLE = "basics"
    GOAL = "how requests are sent and authenticated, and how amounts are expressed"
    VERBATIM = frozenset({"api_base_url"})

    api_base_url: Cited[Identifier] = Field(
        description="Base URL of the server-side REST API, up to and including any version "
        "segment, e.g. https://api.example.com/v2"
    )
    auth_scheme: Cited[Literal["basic", "bearer", "oauth2_client_credentials", "other"]] = Field(
        description="How server-side API requests authenticate"
    )
    amount_unit: Cited[Literal["minor_unit", "major_unit"]] = Field(
        description="minor_unit: amounts are integers in the smallest currency unit "
        "(paise, cents); major_unit: decimal amounts like 10.50"
    )


class PaymentFlow(Section):
    TITLE = "payment_flow"
    GOAL = "how a payment is created, completed by the customer, and captured"
    VERBATIM = frozenset({"intent_object"})

    intent_object: Cited[Identifier] = Field(
        description="Name of the object the merchant's server creates before the customer pays, "
        "e.g. order, payment_intent, session"
    )
    customer_completion: Cited[Literal["server_side", "hosted_checkout", "client_sdk"]] = Field(
        description="Who completes the payment: our server alone (server_side), a hosted page or "
        "checkout form provided by the gateway (hosted_checkout), or the gateway's mobile/JS SDK "
        "without a hosted form (client_sdk)"
    )
    manual_capture: Cited[bool] = Field(
        description="true if a payment can be authorized now and captured later with an API call"
    )
    partial_capture: Cited[bool] = Field(
        description="true if less than the authorized amount can be captured"
    )
    void_supported: Cited[bool] = Field(
        description="true if there is an API call to cancel/void an authorized, uncaptured payment"
    )


class Refunds(Section):
    TITLE = "refunds"
    GOAL = "how refunds work and whether the gateway dedupes repeated refund requests"
    VERBATIM = frozenset({"idempotency_header"})

    partial_refunds: Cited[bool] = Field(
        description="true if a refund can be for less than the captured amount"
    )
    multiple_refunds: Cited[bool] = Field(
        description="true if one payment can be refunded more than once"
    )
    idempotency_header: Cited[Identifier] = Field(
        description="Exact name of the HTTP header that makes refund creation idempotent, "
        "or null if there is none"
    )


class Webhooks(Section):
    TITLE = "webhooks"
    GOAL = "how webhook deliveries are signed, identified and ordered"
    VERBATIM = frozenset({"signature_header", "event_id_header"})

    signature_header: Cited[Identifier] = Field(
        description="Exact name of the HTTP header carrying the webhook signature"
    )
    signature_algorithm: Cited[
        Literal[
            "hmac_sha256_hex",
            "hmac_sha256_base64",
            "timestamped_hmac_sha256",
            "asymmetric_signature",
            "other",
        ]
    ] = Field(
        description="How the signature is computed over the raw request body; "
        "hmac_sha256_hex if the docs say HMAC SHA256 and don't name another encoding"
    )
    signing_secret: Cited[Literal["webhook_secret", "api_key_secret"]] = Field(
        description="webhook_secret: a secret set when creating the webhook; "
        "api_key_secret: the API key's secret"
    )
    event_id_header: Cited[Identifier] = Field(
        description="Exact name of the HTTP header with a unique id per event, used to drop "
        "duplicates; null if there is none"
    )
    ordered_delivery: Cited[bool] = Field(
        description="true if the docs guarantee events arrive in the order they happened"
    )
    duplicate_deliveries: Cited[bool] = Field(
        description="true if the same event can be delivered more than once"
    )


class Errors(Section):
    TITLE = "errors"
    GOAL = "the shape of API error responses"
    VERBATIM = frozenset({"error_code_field", "reason_field"})

    error_code_field: Cited[Identifier] = Field(
        description="Name of the field in an error response holding the broad error code"
    )
    reason_field: Cited[Identifier] = Field(
        description="Name of the field holding the specific machine-readable reason "
        "(e.g. insufficient_funds), or null if there is none"
    )


SECTIONS: tuple[type[Section], ...] = (Basics, PaymentFlow, Refunds, Webhooks, Errors)


class SpecCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway: str
    basics: Basics | None = None
    payment_flow: PaymentFlow | None = None
    refunds: Refunds | None = None
    webhooks: Webhooks | None = None
    errors: Errors | None = None


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema with every ``$ref`` inlined: smaller models handle flat schemas better."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    result: dict[str, Any] = inline(schema)
    return result
