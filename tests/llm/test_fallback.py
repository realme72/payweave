import pytest

from payweave.llm import LLMRateLimitError, LLMRequest, LLMResponse, Message, StopReason, Usage
from payweave.llm.fallback import FallbackProvider, provider_for


class Fake:
    def __init__(self, model: str, *outcomes: LLMResponse | Exception) -> None:
        self.name = "fake"
        self.model = model
        self._outcomes = list(outcomes)
        self.calls = 0
        self.closed = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        nxt = self._outcomes.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def aclose(self) -> None:
        self.closed = True


def ok(model: str) -> LLMResponse:
    return LLMResponse("fake", model, "hi", (), StopReason.END, Usage(1, 1))


def limited(*, daily: bool) -> LLMRateLimitError:
    return LLMRateLimitError("quota", provider="fake", retry_after=None, daily_exhausted=daily)


REQUEST = LLMRequest(messages=(Message("user", "q"),))


async def test_moves_to_the_next_model_when_the_daily_quota_is_gone() -> None:
    a, b = Fake("a", ok("a"), limited(daily=True)), Fake("b", ok("b"))
    logs: list[str] = []
    llm = FallbackProvider([a, b], log=logs.append)  # type: ignore[list-item]
    assert (await llm.complete(REQUEST)).model == "a"
    assert (await llm.complete(REQUEST)).model == "b"
    assert llm.model == "b"
    assert logs == ["  fake: daily quota used up; switching to b"]
    await llm.aclose()
    assert a.closed
    assert b.closed


async def test_per_minute_limits_and_the_last_model_raise() -> None:
    llm = FallbackProvider([Fake("a", limited(daily=False)), Fake("b")])  # type: ignore[list-item]
    with pytest.raises(LLMRateLimitError):
        await llm.complete(REQUEST)  # waiting it out is complete_with_retry's job
    last = FallbackProvider([Fake("a", limited(daily=True))])  # type: ignore[list-item]
    with pytest.raises(LLMRateLimitError):
        await last.complete(REQUEST)


def test_model_list_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GEMINI_MODEL", "a, b")
    llm = provider_for("gemini", None)
    assert isinstance(llm, FallbackProvider)
    assert llm.model == "a"
    assert provider_for("gemini", "c").model == "c"  # --model wins
