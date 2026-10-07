"""Retry policy for LLM calls.

Free tiers rate-limit per minute (Groq: 8,000 tokens/min per model), so a
multi-step run hits 429s routinely. Those are waited out using the provider's
own ``retry-after``. A *daily* cap can't be waited out, so it's raised at once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from payweave.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse

DEFAULT_WAIT_SECONDS = 10.0
MAX_WAIT_SECONDS = 90.0
_sleep = asyncio.sleep  # module-level so tests can replace it


async def complete_with_retry(
    llm: LLMProvider,
    request: LLMRequest,
    *,
    attempts: int = 4,
    on_wait: Callable[[float, LLMError], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> LLMResponse:
    for attempt in range(1, attempts + 1):
        try:
            return await llm.complete(request)
        except LLMError as exc:
            if not exc.retryable or attempt == attempts:
                raise
            wait = min(getattr(exc, "retry_after", None) or DEFAULT_WAIT_SECONDS, MAX_WAIT_SECONDS)
            if on_wait:
                on_wait(wait, exc)
            await (sleep or _sleep)(wait + 0.5)  # retry-after is rounded down; don't arrive early
    raise AssertionError("unreachable")  # pragma: no cover
