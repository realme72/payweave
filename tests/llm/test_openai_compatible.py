import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import respx

from payweave.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMRequest,
    LLMUnavailableError,
    Message,
    OpenAICompatibleProvider,
    StopReason,
    Tool,
    ToolCall,
)

BASE = "https://api.groq.com/openai/v1"
FAKE_KEY = "gsk_test_not_a_real_key"
SEARCH = Tool(
    "search_docs",
    "Search gateway docs",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def groq() -> AsyncIterator[OpenAICompatibleProvider]:
    provider = OpenAICompatibleProvider.from_preset("groq", env={"GROQ_API_KEY": FAKE_KEY})
    yield provider
    await provider.aclose()


def completion(message: dict, finish: str = "stop") -> dict:  # type: ignore[type-arg]
    return {
        "model": "qwen/qwen3.8-27b",
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }


async def test_plain_completion(router: respx.MockRouter, groq: OpenAICompatibleProvider) -> None:
    route = router.post("/chat/completions").respond(json=completion({"content": "Hello"}))
    response = await groq.complete(
        LLMRequest(messages=(Message("user", "Hi"),), system="Be brief", max_tokens=50)
    )
    assert (response.content, response.stop_reason) == ("Hello", StopReason.END)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 30)
    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Hi"},
    ]
    assert (sent["max_tokens"], sent["temperature"]) == (50, 0.0)
    assert route.calls.last.request.headers["authorization"] == f"Bearer {FAKE_KEY}"


async def test_tool_call_round_trip(
    router: respx.MockRouter, groq: OpenAICompatibleProvider
) -> None:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_docs", "arguments": '{"query": "refund minimum"}'},
    }
    route = router.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200, json=completion({"content": None, "tool_calls": [call]}, "tool_calls")
            ),
            httpx.Response(200, json=completion({"content": "At least INR 1.00."})),
        ]
    )
    first = await groq.complete(
        LLMRequest(messages=(Message("user", "min refund?"),), tools=(SEARCH,))
    )
    assert first.stop_reason is StopReason.TOOL_USE
    [tool_call] = first.tool_calls
    assert (tool_call.name, tool_call.arguments) == ("search_docs", {"query": "refund minimum"})
    sent = json.loads(route.calls.last.request.content)
    assert sent["tools"][0]["function"]["name"] == "search_docs"

    second = await groq.complete(
        LLMRequest(
            messages=(
                Message("user", "min refund?"),
                Message("assistant", tool_calls=first.tool_calls),
                Message("tool", "chunk: at least INR 1.00", tool_call_id="call_1"),
            ),
            tools=(SEARCH,),
        )
    )
    assert second.content == "At least INR 1.00."
    history = json.loads(route.calls.last.request.content)["messages"]
    assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "refund minimum"}'
    assert history[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "chunk: at least INR 1.00",
    }


async def test_invalid_tool_json_is_reported_not_raised(
    router: respx.MockRouter, groq: OpenAICompatibleProvider
) -> None:
    bad = {
        "id": "c",
        "type": "function",
        "function": {"name": "search_docs", "arguments": "{query:"},
    }
    router.post("/chat/completions").respond(json=completion({"tool_calls": [bad]}, "stop"))
    response = await groq.complete(LLMRequest(messages=(Message("user", "x"),), tools=(SEARCH,)))
    assert response.stop_reason is StopReason.TOOL_USE  # "stop" + tool calls is normalised
    [call] = response.tool_calls
    assert call.arguments == {}
    assert call.raw_arguments == "{query:"
    assert call.arguments_error is not None


async def test_truncated_output_is_flagged(
    router: respx.MockRouter, groq: OpenAICompatibleProvider
) -> None:
    router.post("/chat/completions").respond(json=completion({"content": "partial"}, "length"))
    response = await groq.complete(LLMRequest(messages=(Message("user", "x"),)))
    assert response.stop_reason is StopReason.MAX_TOKENS


@pytest.mark.parametrize(
    ("status", "body", "headers", "error", "retryable", "daily"),
    [
        (
            429,
            "Rate limit reached on tokens per minute (TPM)",
            {"retry-after": "7"},
            LLMRateLimitError,
            True,
            False,
        ),
        (
            429,
            "Rate limit reached on tokens per day (TPD): Limit 200000",
            {},
            LLMRateLimitError,
            False,
            True,
        ),
        (401, "Invalid API Key", {}, LLMAuthError, False, None),
        (400, "model not found", {}, LLMBadRequestError, False, None),
        (503, "over capacity", {}, LLMUnavailableError, True, None),
    ],
)
async def test_errors_are_classified_and_keep_the_providers_words(  # noqa: PLR0917
    router: respx.MockRouter,
    groq: OpenAICompatibleProvider,
    status: int,
    body: str,
    headers: dict[str, str],
    error: type[Exception],
    retryable: bool,
    daily: bool | None,
) -> None:
    router.post("/chat/completions").respond(
        status, json={"error": {"message": body}}, headers=headers
    )
    with pytest.raises(error) as exc:
        await groq.complete(LLMRequest(messages=(Message("user", "x"),)))
    assert body in str(exc.value)
    assert FAKE_KEY not in str(exc.value)
    assert exc.value.retryable is retryable  # type: ignore[attr-defined]
    if daily is not None:
        assert exc.value.daily_exhausted is daily  # type: ignore[attr-defined]
    if headers:
        assert exc.value.retry_after == 7.0  # type: ignore[attr-defined]


async def test_network_failure_is_unavailable(
    router: respx.MockRouter, groq: OpenAICompatibleProvider
) -> None:
    router.post("/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LLMUnavailableError):
        await groq.complete(LLMRequest(messages=(Message("user", "x"),)))


async def test_list_models(router: respx.MockRouter, groq: OpenAICompatibleProvider) -> None:
    router.get("/models").respond(json={"data": [{"id": "b-model"}, {"id": "a-model"}]})
    assert await groq.list_models() == ["a-model", "b-model"]


def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(LLMAuthError, match="GROQ_API_KEY"):
        OpenAICompatibleProvider.from_preset("groq", env={})
    local = OpenAICompatibleProvider.from_preset("ollama", env={})  # no key needed
    assert local.name == "ollama"


def test_model_override_from_env() -> None:
    provider = OpenAICompatibleProvider.from_preset(
        "groq", env={"GROQ_API_KEY": FAKE_KEY, "GROQ_MODEL": "openai/gpt-oss-120b"}
    )
    assert provider.model == "openai/gpt-oss-120b"


def test_tool_call_dataclass_defaults() -> None:
    assert ToolCall("i", "n", {}).arguments_error is None
