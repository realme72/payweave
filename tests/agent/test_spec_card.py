from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from payweave.agent.citations import check_section, normalize
from payweave.agent.extract import extract_card, extract_section, fit_context
from payweave.agent.reference import Verdict, score
from payweave.agent.spec_card import Citation, Cited, Errors, SpecCard, Webhooks, tool_schema
from payweave.knowledge.chunker import ChunkKind
from payweave.llm import (
    LLMBadRequestError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    Usage,
)
from payweave.llm.retry import complete_with_retry

SIG = "razorpay:webhooks.md#0"
DUP = "razorpay:webhooks.md#1"
ERR = "razorpay:errors.md#0"
CHUNKS = {
    SIG: "Razorpay > Webhooks > Validate\nThe `X-Razorpay-Signature` header holds an HMAC SHA256 "
    "of the **raw request body**,\n  computed with the webhook secret.",
    DUP: "Razorpay > Webhooks > Idempotency\nYou may receive the same webhook more than once. "
    "Use the `x-razorpay-event-id` header to drop duplicates. Order is not guaranteed.",
    ERR: 'Razorpay > Errors\n```json\n{"error": {"code": "BAD_REQUEST_ERROR", '
    '"reason": "input_validation_failed"}}\n```',
}


@dataclass(frozen=True)
class FakeHit:
    ref: str
    content: str


class FakeDocs:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        gateway: str | None = None,
        kinds: Sequence[ChunkKind] | None = None,
        limit: int = 5,
    ) -> list[FakeHit]:
        self.queries.append(query)
        return [FakeHit(ref, text) for ref, text in CHUNKS.items()][:limit]

    def chunks_by_ref(self, refs: Sequence[str]) -> Mapping[str, str]:
        return {r: CHUNKS[r] for r in refs if r in CHUNKS}


class ScriptedLLM:
    """Replays canned responses and records every request."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def tool(name: str, args: dict[str, Any], call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        "scripted",
        "scripted-1",
        "",
        (ToolCall(call_id, name, args),),
        StopReason.TOOL_USE,
        Usage(100, 20),
    )


def text(content: str) -> LLMResponse:
    return LLMResponse("scripted", "scripted-1", content, (), StopReason.END, Usage(50, 10))


def fact(value: Any, ref: str = SIG, quote: str = "") -> dict[str, Any]:
    return {"value": value, "citations": [{"ref": ref, "quote": quote}] if quote else []}


WEBHOOKS_OK = {
    "signature_header": fact("X-Razorpay-Signature", quote="The `X-Razorpay-Signature` header"),
    "signature_algorithm": fact("hmac_sha256_hex", quote="an HMAC SHA256 of the raw request body"),
    "signing_secret": fact("webhook_secret", quote="computed with the webhook secret"),
    "event_id_header": fact(
        "x-razorpay-event-id", DUP, "Use the `x-razorpay-event-id` header to drop duplicates"
    ),
    "ordered_delivery": fact(False, DUP, "Order is not guaranteed"),
    "duplicate_deliveries": fact(True, DUP, "receive the same webhook more than once"),
}


# --- card model -----------------------------------------------------------------------------


def test_known_values_need_a_citation_but_null_does_not() -> None:
    with pytest.raises(ValidationError, match="needs at least one citation"):
        Cited[str](value="x", citations=[])
    assert Cited[str](value=None, citations=[]).value is None


def test_identifier_fields_reject_prose() -> None:
    data = {**WEBHOOKS_OK, "signature_header": fact("The signature header.", quote="The `X-Razo")}
    with pytest.raises(ValidationError, match=r"signature_header\.value"):
        Webhooks.model_validate(data)


def test_tool_schema_is_flat_and_closed() -> None:
    schema = tool_schema(Webhooks)
    assert "$ref" not in str(schema)
    assert "$defs" not in schema
    assert schema["additionalProperties"] is False
    algorithm = schema["properties"]["signature_algorithm"]["properties"]["value"]
    assert "hmac_sha256_hex" in str(algorithm)


# --- citation check -------------------------------------------------------------------------


def test_normalize_ignores_markdown_and_whitespace() -> None:
    assert normalize("The **raw  request\n body**") == "the raw request body"


def test_valid_citations_pass() -> None:
    assert check_section(Webhooks.model_validate(WEBHOOKS_OK), CHUNKS) == []


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"ref": "razorpay:made-up.md#4"}, "doesn't exist"),
        ({"quote": "The `X-Razorpay-Signature` header is optional"}, "quote isn't in"),
    ],
)
def test_bad_citations_are_rejected(change: dict[str, str], problem: str) -> None:
    data = {**WEBHOOKS_OK}
    data["signing_secret"] = {
        "value": "webhook_secret",
        "citations": [{**WEBHOOKS_OK["signing_secret"]["citations"][0], **change}],
    }
    [found] = check_section(Webhooks.model_validate(data), CHUNKS)
    assert found.startswith("signing_secret:")
    assert problem in found


def test_identifier_values_must_appear_in_a_quote() -> None:
    data = {**WEBHOOKS_OK, "signature_header": fact("X-Signature", quote="an HMAC SHA256")}
    [found] = check_section(Webhooks.model_validate(data), CHUNKS)
    assert "'X-Signature' doesn't appear" in found


# --- extraction loop ------------------------------------------------------------------------


async def test_rejected_submission_is_repaired_then_accepted() -> None:
    wrong_quote = {**WEBHOOKS_OK, "ordered_delivery": fact(False, DUP, "Events arrive in order")}
    llm = ScriptedLLM(
        tool("search_docs", {"query": "webhook signature header"}),
        tool("submit_webhooks", wrong_quote, "c2"),
        tool("submit_webhooks", WEBHOOKS_OK, "c3"),
    )
    docs = FakeDocs()
    result = await extract_section(llm, docs, "razorpay", Webhooks)

    assert isinstance(result.section, Webhooks)
    assert result.section.signature_header.value == "X-Razorpay-Signature"
    assert (result.turns, result.searches, result.problems) == (3, ["webhook signature header"], [])
    assert (result.input_tokens, result.output_tokens) == (300, 60)
    history = llm.requests[-1].messages
    assert '<passage ref="razorpay:webhooks.md#0">' in history[2].content  # search result
    assert "ordered_delivery: the quote isn't in" in history[4].content  # rejection
    assert {t.name for t in llm.requests[0].tools} == {"search_docs", "submit_webhooks"}


async def test_invalid_submission_reports_schema_errors() -> None:
    llm = ScriptedLLM(
        tool("submit_errors", {"error_code_field": {"value": "code"}}),
        tool(
            "submit_errors",
            {
                "error_code_field": fact("code", ERR, '"code": "BAD_REQUEST_ERROR"'),
                "reason_field": fact("reason", ERR, '"reason": "input_validation_failed"'),
            },
            "c2",
        ),
    )
    result = await extract_section(llm, FakeDocs(), "razorpay", Errors)
    assert result.section is not None
    rejection = llm.requests[1].messages[-1].content
    assert "error_code_field.citations: Field required" in (rejection)
    assert "reason_field: Field required" in rejection


async def test_text_reply_is_nudged_and_turns_are_capped() -> None:
    llm = ScriptedLLM(text("The header is X-Razorpay-Signature."), text("Sure."))
    result = await extract_section(llm, FakeDocs(), "razorpay", Webhooks, max_turns=2)
    assert result.section is None
    assert result.problems == ["no accepted submission in 2 turns"]
    assert "Don't reply in text" in llm.requests[1].messages[-1].content


async def test_reply_cut_off_by_max_tokens_gets_more_room() -> None:
    cut = LLMResponse("s", "m", "", (), StopReason.MAX_TOKENS, Usage(50, 1500))
    llm = ScriptedLLM(cut, cut, cut, tool("submit_webhooks", WEBHOOKS_OK))
    result = await extract_section(llm, FakeDocs(), "razorpay", Webhooks)
    assert result.section is not None
    assert [r.max_tokens for r in llm.requests] == [1500, 3000, 6000, 6000]


async def test_invalid_tool_json_is_sent_back() -> None:
    bad = ToolCall("c1", "search_docs", {}, "{query:", "invalid JSON: Expecting property name")
    llm = ScriptedLLM(
        LLMResponse("s", "m", "", (bad,), StopReason.TOOL_USE),
        tool("submit_webhooks", WEBHOOKS_OK, "c2"),
    )
    docs = FakeDocs()
    result = await extract_section(llm, docs, "razorpay", Webhooks)
    assert result.section is not None
    assert docs.queries == []
    assert "Send valid JSON" in llm.requests[1].messages[-1].content


async def test_card_keeps_going_after_a_failed_section_and_keeps_work_on_daily_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("payweave.llm.retry._sleep", no_sleep)
    tpm = LLMRateLimitError("TPM", provider="s", retry_after=1, daily_exhausted=False)
    llm = ScriptedLLM(
        *[tpm] * 4,  # webhooks: every retry rate-limited
        tool("submit_errors", {"error_code_field": fact(None), "reason_field": fact(None)}),
    )
    run = await extract_card(llm, FakeDocs(), "razorpay", sections=(Webhooks, Errors))
    assert run.card.webhooks is None
    assert run.card.errors is not None
    assert "TPM" in run.sections[0].problems[0]
    assert run.sections[0].turns == 1  # the section's own progress is kept

    daily = LLMRateLimitError("TPD", provider="s", retry_after=None, daily_exhausted=True)
    llm = ScriptedLLM(tool("submit_webhooks", WEBHOOKS_OK), daily)
    run = await extract_card(llm, FakeDocs(), "razorpay", sections=(Webhooks, Errors, Errors))
    assert run.card.webhooks is not None  # accepted before the budget ran out: kept
    assert [s.section is None for s in run.sections] == [False, True, True]
    assert run.sections[2].problems[0].startswith("skipped: daily budget exhausted")
    assert len(llm.requests) == 2  # nothing sent after the daily cap


async def test_daily_cap_mid_section_keeps_that_sections_counts() -> None:
    daily = LLMRateLimitError("TPD", provider="s", retry_after=None, daily_exhausted=True)
    llm = ScriptedLLM(tool("search_docs", {"query": "webhook signature header"}), daily)
    run = await extract_card(llm, FakeDocs(), "razorpay", sections=(Webhooks, Errors))
    first = run.sections[0]
    assert first.out_of_budget
    assert (first.turns, len(first.searches)) == (2, 1)
    assert first.input_tokens > 0
    assert run.sections[1].problems[0].startswith("skipped: daily budget exhausted")


def test_fit_context_drops_oldest_search_results_first() -> None:
    old = Message("tool", '<passage ref="a">' + "x" * 500, tool_call_id="1")
    newer = Message("tool", '<passage ref="b">' + "y" * 500, tool_call_id="2")
    rejection = Message("tool", "Rejected. Fix these: ...", tool_call_id="3")
    latest = Message("tool", '<passage ref="c">' + "z" * 500, tool_call_id="4")
    history = [Message("user", "q"), old, newer, rejection, latest]

    history.insert(4, Message("assistant", "", ()))  # the turn that asked for `latest`
    assert fit_context(history, 10_000) == history
    trimmed = fit_context(history, 1_200)
    assert trimmed[1].content.startswith("[older search results removed")
    assert trimmed[1].tool_call_id == "1"  # still answers its tool call
    assert trimmed[2:] == history[2:]
    tight = fit_context(history, 100)
    kept = [m.content.startswith("<passage") for m in tight]
    assert kept == [False, False, False, False, False, True]


def test_fit_context_trims_latest_results_that_alone_are_too_big() -> None:
    asked = Message("assistant", "", ())
    big = [Message("tool", '<passage ref="c">' + "z" * 5_000, tool_call_id=str(i)) for i in (1, 2)]
    fitted = fit_context([Message("user", "q"), asked, *big], 4_000)
    assert all(m.content.startswith('<passage ref="c">zzz') for m in fitted[2:])
    assert all("search more specifically" in m.content for m in fitted[2:])
    assert sum(len(m.content) for m in fitted) < 4_500


async def test_request_too_large_is_retried_with_less_context() -> None:
    from payweave.agent.extract import complete_fitted  # noqa: PLC0415

    too_big = LLMBadRequestError("HTTP 413: Request too large", provider="s", status=413)
    llm = ScriptedLLM(too_big, text("ok"))
    asked = Message("assistant", "", ())
    history = [
        Message("user", "q"),
        asked,
        Message("tool", "<passage>" + "z" * 9_000, tool_call_id="1"),
    ]
    logged: list[str] = []
    out = await complete_fitted(
        llm, history, LLMRequest(messages=()), budget_chars=8_000, log=logged.append
    )
    assert out.content == "ok"
    assert len(llm.requests[1].messages[-1].content) < len(llm.requests[0].messages[-1].content)
    assert logged == ["  request too large; retrying with 4000 characters of context"]


def test_fit_context_keeps_every_result_of_the_latest_turn() -> None:
    results = [
        Message("tool", '<passage ref="r">' + "x" * 900, tool_call_id=str(i)) for i in range(3)
    ]
    history = [Message("user", "q"), Message("assistant", "", ()), *results]
    fitted = fit_context(history, 100)  # parallel searches: all three are current
    assert [m.tool_call_id for m in fitted[2:]] == ["0", "1", "2"]  # none dropped,
    assert all(m.content.startswith('<passage ref="r">xxx') for m in fitted[2:])  # only cut
    assert fit_context(history, 10_000) == history


async def test_last_turns_force_a_submission() -> None:
    llm = ScriptedLLM(
        tool("search_docs", {"query": "a"}),
        tool("search_docs", {"query": "b"}, "c2"),
        tool("submit_webhooks", WEBHOOKS_OK, "c3"),
    )
    result = await extract_section(llm, FakeDocs(), "razorpay", Webhooks, max_turns=4)
    assert result.section is not None
    assert [r.require_tool for r in llm.requests] == [None, None, "submit_webhooks"]
    assert [t.name for t in llm.requests[2].tools] == ["submit_webhooks"]  # nothing else to call
    assert "Out of searches" in llm.requests[2].messages[-1].content


async def test_rejected_forced_tool_call_falls_back_to_json() -> None:
    import json  # noqa: PLC0415

    rejected = LLMBadRequestError("tool call validation failed", provider="s", status=400)
    llm = ScriptedLLM(
        rejected,  # forced turn: the model called a tool it wasn't offered
        text("Here you go:\n```json\n" + json.dumps(WEBHOOKS_OK) + "\n```"),
    )
    result = await extract_section(llm, FakeDocs(), "razorpay", Webhooks, max_turns=1)
    assert isinstance(result.section, Webhooks)
    assert result.section.event_id_header.value == "x-razorpay-event-id"
    fallback = llm.requests[1]
    assert (fallback.tools, fallback.require_tool) == ((), None)
    [ask] = fallback.messages  # flattened: no tool-call history to imitate
    assert ask.role == "user"
    assert "Reply with only a JSON object" in ask.content


async def test_searches_are_capped_per_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("payweave.agent.extract.MAX_SEARCHES", 1)
    llm = ScriptedLLM(
        tool("search_docs", {"query": "a"}),
        tool("search_docs", {"query": "b"}, "c2"),
        tool("submit_webhooks", WEBHOOKS_OK, "c3"),
    )
    docs = FakeDocs()
    result = await extract_section(llm, docs, "razorpay", Webhooks)
    assert docs.queries == ["a"]
    assert "Search limit reached" in llm.requests[2].messages[-2].content
    assert llm.requests[2].require_tool == "submit_webhooks"
    assert result.section is not None


# --- retry ----------------------------------------------------------------------------------


async def test_retry_waits_out_per_minute_limits_only() -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    tpm = LLMRateLimitError("TPM", provider="s", retry_after=7, daily_exhausted=False)
    llm = ScriptedLLM(tpm, text("ok"))
    request = LLMRequest(messages=())
    response = await complete_with_retry(llm, request, sleep=fake_sleep)
    assert (response.content, waits) == ("ok", [7.5])

    tpd = LLMRateLimitError("TPD", provider="s", retry_after=None, daily_exhausted=True)
    with pytest.raises(LLMRateLimitError):
        await complete_with_retry(ScriptedLLM(tpd), request, sleep=fake_sleep)
    assert waits == [7.5]


# --- scoring --------------------------------------------------------------------------------


def test_score_against_reference() -> None:
    card = SpecCard(
        gateway="razorpay",
        webhooks=Webhooks.model_validate(
            {
                **WEBHOOKS_OK,
                "ordered_delivery": fact(None),
                "duplicate_deliveries": fact(False, DUP, "receive the same webhook more than once"),
            }
        ),
    )
    reference: dict[str, dict[str, Any]] = {
        "webhooks": {
            "signature_header": "x-razorpay-signature",
            "ordered_delivery": False,
            "duplicate_deliveries": True,
        },
        "errors": {"reason_field": "reason"},
    }
    verdicts = [(s.field, s.verdict) for s in score(card, reference)]
    assert verdicts == [
        ("signature_header", Verdict.CORRECT),
        ("ordered_delivery", Verdict.ABSTAINED),
        ("duplicate_deliveries", Verdict.WRONG),
        ("reason_field", Verdict.MISSING),
    ]


def test_scoring_accepts_listed_readings_and_loose_placeholders() -> None:
    from payweave.agent.reference import _same  # noqa: PLC0415

    assert _same(["error_code", "type"], "TYPE")
    assert not _same(["error_code", "type"], "code")
    assert _same("GET /orders/{order_id}/refunds/{refund_id}", "GET /orders/:id/refunds/:rid")
    assert not _same("GET /orders/{order_id}", "GET /orders/{order_id}/payments")


def test_citation_model_rejects_trivial_quotes() -> None:
    with pytest.raises(ValidationError):
        Citation(ref=SIG, quote="HMAC")


def test_operation_claims_need_an_endpoint_quote() -> None:
    from payweave.agent.citations import check_section  # noqa: PLC0415
    from payweave.agent.spec_card import Capture  # noqa: PLC0415

    ref = "razorpay:api__payments__capture.md#1"
    chunk = (
        "# Capture a Payment\n**POST** `/v1/payments/:id/capture`\n"
        "A capture was attempted on a payment that has already moved out of the `authorized` "
        "state. It is either already captured or has been voided."
    )

    def card(void: Any, quote: str) -> Capture:
        cite = [{"ref": ref, "quote": quote}]
        return Capture.model_validate(
            {
                "manual_capture": {
                    "value": True,
                    "citations": [{"ref": ref, "quote": "POST /v1/payments/:id/capture"}],
                },
                "partial_capture": {"value": None, "citations": []},
                "void_supported": {"value": void, "citations": cite if void is not None else []},
            }
        )

    mentioned = card(True, "It is either already captured or has been voided.")
    assert any(
        "void_supported: true needs a quote of the endpoint" in p
        for p in check_section(mentioned, {ref: chunk})
    )
    assert check_section(card(None, ""), {ref: chunk}) == []  # manual_capture's endpoint passes
    # false claims aren't held to it: they cite a rule, not an endpoint
    assert (
        check_section(
            card(False, "It is either already captured or has been voided."), {ref: chunk}
        )
        == []
    )


async def test_dependent_sections_see_earlier_facts() -> None:
    from payweave.agent.spec_card import Endpoints  # noqa: PLC0415

    llm = ScriptedLLM(text("thinking"), *[text("still thinking")] * 10)
    await extract_card(
        llm,
        FakeDocs(),
        "razorpay",
        sections=(Endpoints,),
        known={"payment_flow.intent_object": "order", "refunds.partial_refunds": True},
    )
    first = llm.requests[0].messages[0].content
    assert "payment_flow.intent_object = 'order'" in first
    assert "partial_refunds" not in first  # only what the section USES


def test_json_string_facts_must_come_from_a_sample_body() -> None:
    from payweave.agent.citations import check_section  # noqa: PLC0415
    from payweave.agent.spec_card import ErrorShape  # noqa: PLC0415

    ref = "razorpay:api__refunds__normal-refunds-idempotent.md#3"
    chunk = (
        '{"error": {"code": "BAD_REQUEST_ERROR", "description": "Different request with the '
        'same idempotency key has already been processed.", "reason": "NA"}}\n'
        "* description: Another refund request with different parameters has been processed "
        "using the same idempotency key."
    )

    def shape(message: str, quote: str) -> ErrorShape:
        path = {
            "value": "error.code",
            "citations": [{"ref": ref, "quote": '{"error": {"code": "BAD_REQ'}],
        }
        return ErrorShape.model_validate(
            {
                "message_path": {"value": None, "citations": []},
                "code_path": path,
                "reason_path": {"value": None, "citations": []},
                "idempotency_conflict_message": {
                    "value": message,
                    "citations": [{"ref": ref, "quote": quote}],
                },
            }
        )

    explained = (
        "Another refund request with different parameters has been processed using the same "
        "idempotency key."
    )
    bad = check_section(shape(explained, "* description: " + explained), {ref: chunk})
    assert any("inside its JSON quotes" in p for p in bad)
    sample = "Different request with the same idempotency key has already been processed."
    assert check_section(shape(sample, f'"description": "{sample}"'), {ref: chunk}) == []


async def test_last_chance_nulls_failing_fields_and_keeps_the_rest() -> None:
    bad_header = {**WEBHOOKS_OK, "event_id_header": fact("x-made-up-id", DUP, "made up quote here")}
    # Not the last turn: the whole submission is rejected so the model can fix it.
    llm = ScriptedLLM(tool("submit_webhooks", bad_header), tool("submit_webhooks", WEBHOOKS_OK))
    result = await extract_section(llm, FakeDocs(), "razorpay", Webhooks, max_turns=4)
    assert isinstance(result.section, Webhooks)
    assert result.section.event_id_header.value == "x-razorpay-event-id"
    assert result.nulled == []

    # Last chance: the failing field becomes null, the checked facts are kept.
    llm = ScriptedLLM(tool("submit_webhooks", bad_header))
    result = await extract_section(llm, FakeDocs(), "razorpay", Webhooks, max_turns=1)
    assert isinstance(result.section, Webhooks)
    assert result.section.event_id_header.value is None
    assert result.section.signature_header.value == "X-Razorpay-Signature"
    assert result.nulled
    assert result.nulled[0].startswith("nulled event_id_header")


def test_recheck_nulls_facts_that_fail_todays_rules() -> None:
    from payweave.agent.citations import recheck_card  # noqa: PLC0415

    stale = {**WEBHOOKS_OK, "event_id_header": fact("x-made-up-id", DUP, "made up quote here")}
    card = SpecCard.model_validate({"gateway": "razorpay", "webhooks": stale})
    checked, nulled = recheck_card(card, lambda refs: {r: CHUNKS[r] for r in refs if r in CHUNKS})
    assert checked.webhooks is not None
    assert checked.webhooks.event_id_header.value is None
    assert checked.webhooks.signature_header.value == "X-Razorpay-Signature"
    assert nulled[0].startswith("webhooks.event_id_header:")


def test_path_segments_must_be_json_keys() -> None:
    from payweave.agent.spec_card import ErrorShape  # noqa: PLC0415

    ref = "razorpay:errors.md#9"
    chunk = '{"error": {"code": "BAD_REQUEST_ERROR", "description": "x"}}'

    def shape(quote: str) -> ErrorShape:
        return ErrorShape.model_validate(
            {
                "message_path": {"value": None, "citations": []},
                "code_path": {"value": "error.code", "citations": [{"ref": ref, "quote": quote}]},
                "reason_path": {"value": None, "citations": []},
                "idempotency_conflict_message": {"value": None, "citations": []},
            }
        )

    assert check_section(shape('{"error": {"code": "BAD_REQUEST_ERROR"'), {ref: chunk}) == []
    inner_only = check_section(shape('"code": "BAD_REQUEST_ERROR"'), {ref: chunk})
    assert inner_only
    assert "error not a JSON key" in inner_only[0]
