"""Try models in order, moving on when one's daily quota is gone.

Free tiers cap requests per model per day (Gemini: 20), and a full run can need more
than one model's worth. ``--model a,b,c`` builds this: the run continues on ``b`` when
``a`` says its daily quota is exhausted, instead of dropping the rest of the work.
Per-minute limits are not a reason to switch; complete_with_retry waits those out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from payweave.llm.base import LLMRateLimitError, LLMRequest, LLMResponse
from payweave.llm.openai_compatible import OpenAICompatibleProvider


class FallbackProvider:
    def __init__(
        self,
        providers: Sequence[OpenAICompatibleProvider],
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider")
        self._providers = list(providers)
        self._current = 0
        self._log = log

    @property
    def name(self) -> str:
        return self._providers[self._current].name

    @property
    def model(self) -> str:
        return self._providers[self._current].model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        while True:
            try:
                return await self._providers[self._current].complete(request)
            except LLMRateLimitError as exc:
                if not exc.daily_exhausted or self._current + 1 >= len(self._providers):
                    raise
                self._current += 1
                self._log(f"  {exc.provider}: daily quota used up; switching to {self.model}")

    async def aclose(self) -> None:
        for p in self._providers:
            await p.aclose()


def provider_for(preset: str, model: str | None) -> OpenAICompatibleProvider | FallbackProvider:
    """One provider, or a FallbackProvider for a comma-separated model list."""
    models = [m.strip() for m in (model or "").split(",") if m.strip()]
    if len(models) <= 1:
        return OpenAICompatibleProvider.from_preset(preset, model=models[0] if models else None)
    return FallbackProvider([OpenAICompatibleProvider.from_preset(preset, model=m) for m in models])
