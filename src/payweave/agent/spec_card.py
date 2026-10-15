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
    HINTS: ClassVar[str] = ""  # section-specific guidance for the first message
    # "An API call exists" fields: true must quote the endpoint (method + path).
    ENDPOINT_EVIDENCE: ClassVar[frozenset[str]] = frozenset()
    # Dotted JSON paths (error.description): every segment must appear in a cited quote.
    PATHS: ClassVar[frozenset[str]] = frozenset()
    # Values that must be quoted as a JSON string from a sample response ("...").
    JSON_STRINGS: ClassVar[frozenset[str]] = frozenset()
    # Extract only when an earlier fact has this value ("section.field", value).
    ONLY_IF: ClassVar[tuple[str, str] | None] = None
    # Show the gateway's endpoint index (one passage per API page) in the first message.
    ENDPOINT_INDEX: ClassVar[bool] = False
    # Facts from earlier sections this one builds on ("section.field"), shown in its prompt.
    USES: ClassVar[tuple[str, ...]] = ()

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
    auth_scheme: Cited[
        Literal["basic", "bearer", "api_key_headers", "oauth2_client_credentials", "other"]
    ] = Field(
        description="How server-side API requests authenticate: basic (key id and secret as "
        "HTTP Basic auth), bearer (a secret token in Authorization), api_key_headers (the key "
        "id and the secret each in their own custom header)"
    )
    amount_unit: Cited[Literal["minor_unit", "major_unit"]] = Field(
        description="minor_unit: amounts are integers in the smallest currency unit "
        "(paise, cents); major_unit: decimal amounts like 10.50"
    )


class PaymentFlow(Section):
    TITLE = "payment_flow"
    GOAL = "what the merchant's server creates before the customer pays, and who completes it"
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


# Split from PaymentFlow: these facts live in different docs (capture vs checkout),
# and two of three are usually "no", which takes searching for a rule that says so.
class Capture(Section):
    TITLE = "capture"
    GOAL = "whether payments can be authorized now and captured later, partially, or voided"
    ENDPOINT_EVIDENCE = frozenset({"manual_capture", "void_supported"})
    HINTS = (
        "For true/false fields, a documented rule that forbids something is evidence for "
        "false: cite the rule (e.g. 'the capture amount must equal the authorized amount' "
        "means partial capture is false). Not finding a mention is not evidence: use null. "
        "manual_capture and void_supported are true only if you can quote the endpoint "
        "(method and path) that does it."
    )

    manual_capture: Cited[bool] = Field(
        description="true if a payment can be authorized now and captured later with an API "
        "call; cite that endpoint (method and path)"
    )
    partial_capture: Cited[bool] = Field(
        description="true if a capture call can take less than one payment's authorized "
        "amount. Not the same as several separate payments against one order "
        "(often called partial payments)"
    )
    void_supported: Cited[bool] = Field(
        description="true only if there is an API endpoint to cancel/void an authorized, "
        "uncaptured payment; cite it (method and path). A payment *status* named voided is not "
        "an endpoint"
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
    VERBATIM = frozenset({"signature_header", "event_id_header", "timestamp_header"})

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
    # Added later; the defaults keep earlier cards loadable.
    timestamp_header: Cited[Identifier] = Field(
        default_factory=lambda: Cited(value=None, citations=[]),
        description="Header with the delivery timestamp that is part of the signed message "
        "(timestamped signatures); null if there is none",
    )
    signature_encoding: Cited[Literal["hex", "base64"]] = Field(
        default_factory=lambda: Cited(value=None, citations=[]),
        description="How the HMAC digest is written in the signature header",
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


# --- integration facts: what the manifest compiler needs (agent/manifest_compiler.py) -------

Endpoint = Annotated[str, Field(pattern=r"^(GET|POST|PUT|PATCH|DELETE) /\S*$", max_length=200)]
DottedPath = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")]


class Endpoints(Section):
    TITLE = "endpoints"
    GOAL = "the REST endpoints to create, read, capture and refund payments"
    USES = ("basics.api_base_url", "payment_flow.intent_object")
    ENDPOINT_INDEX = True
    VERBATIM = frozenset(
        {"create_intent", "get_intent", "list_attempts", "capture", "refund", "get_refund"}
    )
    HINTS = (
        "Each value is the HTTP method, a space, and the path exactly as the endpoint's "
        "reference page writes it, including any version prefix and its id placeholder "
        "(e.g. 'POST /v1/things/:id/cancel'). Quote the line that shows it."
    )

    create_intent: Cited[Endpoint] = Field(
        description="Create the object the merchant's server makes before the customer pays "
        "(the intent_object of the payment_flow section)"
    )
    get_intent: Cited[Endpoint] = Field(description="Fetch that object by id")
    list_attempts: Cited[Endpoint] = Field(
        description="List the payments (attempts) made against one such object, by its id; "
        "null if fetching the object already includes them"
    )
    capture: Cited[Endpoint] = Field(
        description="Capture an authorized payment: by payment id, or by the intent's id "
        "if the gateway captures through that object"
    )
    refund: Cited[Endpoint] = Field(
        description="Create a refund for a payment: by payment id, or by the intent's id "
        "if the gateway refunds through that object"
    )
    get_refund: Cited[Endpoint] = Field(description="Fetch a refund by refund id")


class IdFormats(Section):
    TITLE = "ids"
    GOAL = "what the gateway's ids look like, from sample responses"
    USES = ("payment_flow.intent_object",)
    VERBATIM = frozenset({"intent_id_example", "attempt_id_example", "refund_id_example"})
    HINTS = "Copy each id exactly from a sample response or request in the docs."

    intent_id_example: Cited[Identifier] = Field(
        description="A sample id of the intent object (the thing created before paying)"
    )
    attempt_id_example: Cited[Identifier] = Field(
        description="A sample id of a payment (one attempt by the customer)"
    )
    refund_id_example: Cited[Identifier] = Field(description="A sample id of a refund")


class ApiKeys(Section):
    TITLE = "api_keys"
    GOAL = "how test-mode and live-mode API keys can be told apart"
    VERBATIM = frozenset({"test_key_prefix", "live_key_prefix"})

    test_key_prefix: Cited[Identifier] = Field(
        description="The fixed start of a test-mode key, e.g. tk_test_; null if keys don't show it"
    )
    live_key_prefix: Cited[Identifier] = Field(
        description="The fixed start of a live-mode key, e.g. tk_live_; null if keys don't show it"
    )
    prefixed_key: Cited[Literal["key_id", "secret_key"]] = Field(
        description="Which key carries that prefix: key_id (a public/id part used as the "
        "username) or secret_key (the secret itself)"
    )


class AuthHeaders(Section):
    TITLE = "auth_headers"
    GOAL = "the custom headers that carry the API credentials, and any required version header"
    USES = ("basics.auth_scheme",)
    ONLY_IF = ("basics.auth_scheme", "api_key_headers")
    VERBATIM = frozenset({"key_id_header", "secret_header", "version_header", "api_version"})
    HINTS = (
        "Only for api_key_headers auth (null otherwise). Copy header names and the version "
        "value exactly as a request sample or the API reference writes them."
    )

    key_id_header: Cited[Identifier] = Field(
        description="Header that carries the key id / client id / app id"
    )
    secret_header: Cited[Identifier] = Field(description="Header that carries the secret key")
    version_header: Cited[Identifier] = Field(
        description="Header every request must send to pick the API version; null if none"
    )
    api_version: Cited[Identifier] = Field(
        description="The API version value to send in version_header (the current one the "
        "docs use); null if there is no version header"
    )


class ErrorShape(Section):
    TITLE = "error_shape"
    GOAL = (
        "where the body of a failed API request (an HTTP 4xx response) puts its message and "
        "codes, and the duplicate-request error"
    )
    PATHS = frozenset({"message_path", "code_path", "reason_path"})
    VERBATIM = frozenset({"idempotency_conflict_message"})
    JSON_STRINGS = frozenset({"idempotency_conflict_message"})
    HINTS = (
        "Paths are dotted JSON paths into the body the API returns when a request fails "
        '(HTTP 4xx), e.g. error.message for {"error": {"message": ...}}. Quote a sample '
        "error response that shows them. Not the failure fields stored on a payment or "
        "refund object (those describe a declined payment, not a failed request)."
    )

    message_path: Cited[DottedPath] = Field(description="Path to the human-readable message")
    code_path: Cited[DottedPath] = Field(description="Path to the broad error code")
    reason_path: Cited[DottedPath] = Field(
        description="Path to the specific machine-readable reason, or null if there is none"
    )
    idempotency_conflict_message: Cited[str] = Field(
        description="The error message the API returns when an idempotency key is reused with "
        "a different request, copied from a sample error response body (a JSON string value, "
        "quote it with its quotes), not from the docs' explanation of the error; null if no "
        "sample shows it"
    )


class CheckoutReturn(Section):
    TITLE = "checkout_return"
    GOAL = (
        "what the customer's browser sends back to the merchant after paying in the gateway's "
        "checkout, and how its signature is verified"
    )
    VERBATIM = frozenset({"intent_param", "attempt_param", "signature_param"})
    USES = ("payment_flow.intent_object",)

    intent_param: Cited[Identifier] = Field(
        description="Name of the returned parameter holding the intent object's id"
    )
    attempt_param: Cited[Identifier] = Field(
        description="Name of the returned parameter holding the payment id"
    )
    signature_param: Cited[Identifier] = Field(
        description="Name of the returned parameter holding the signature"
    )
    signed_message: Cited[Literal["intent_id|payment_id", "payment_id|intent_id", "other"]] = Field(
        description="What is signed: the intent id, a | separator, and the payment id (in "
        "the order the docs give); other for anything else"
    )
    signature_scheme: Cited[Literal["hmac_sha256_hex", "other"]] = Field(
        description="hmac_sha256_hex if it's an HMAC SHA256 hex digest"
    )
    signing_key: Cited[Literal["api_key_secret", "other"]] = Field(
        description="api_key_secret if it's signed with the API key's secret"
    )


SPEC_SECTIONS: tuple[type[Section], ...] = (
    Basics,
    PaymentFlow,
    Capture,
    Refunds,
    Webhooks,
    Errors,
)
INTEGRATION_SECTIONS: tuple[type[Section], ...] = (
    Endpoints,
    IdFormats,
    ApiKeys,
    AuthHeaders,
    ErrorShape,
    CheckoutReturn,
)
SECTIONS = SPEC_SECTIONS + INTEGRATION_SECTIONS


class SpecCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway: str
    basics: Basics | None = None
    payment_flow: PaymentFlow | None = None
    capture: Capture | None = None
    refunds: Refunds | None = None
    webhooks: Webhooks | None = None
    errors: Errors | None = None
    endpoints: Endpoints | None = None
    ids: IdFormats | None = None
    api_keys: ApiKeys | None = None
    auth_headers: AuthHeaders | None = None
    error_shape: ErrorShape | None = None
    checkout_return: CheckoutReturn | None = None


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
