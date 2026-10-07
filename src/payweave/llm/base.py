"""Vendor-neutral LLM interface (Dependency Inversion: the agent depends on this).

Correctness never rests on the model: whatever it writes must pass the gates
(PLAN.md §7). So providers are swappable on cost and speed alone, and the
agent can run on free tiers (Groq, Gemini, local Ollama) or on Claude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

# --- request / response -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tool:
    """A function the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # The model sometimes emits invalid JSON; keep it so the agent can say so.
    raw_arguments: str = ""
    arguments_error: str | None = None


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant turns that called tools
    tool_call_id: str | None = None  # tool turns: which call this answers


class StopReason(StrEnum):
    END = "end"  # the model finished
    TOOL_USE = "tool_use"  # it wants tool results
    MAX_TOKENS = "max_tokens"  # cut off: the output is incomplete
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[Message, ...]
    system: str = ""
    tools: tuple[Tool, ...] = ()
    max_tokens: int = 2048  # some providers count this as used, so keep it tight
    temperature: float = 0.0
    model: str | None = None  # None = the provider's configured model


@dataclass(frozen=True, slots=True)
class LLMResponse:
    provider: str
    model: str
    content: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


# --- errors ------------------------------------------------------------------------------------


class LLMError(Exception):
    """Base for provider failures. Messages never include API keys."""

    retryable = False

    def __init__(self, message: str, *, provider: str, status: int | None = None) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status


class LLMAuthError(LLMError):
    pass


class LLMBadRequestError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):
    """429. ``daily_exhausted`` means waiting minutes won't help: stop the run."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retry_after: float | None,
        daily_exhausted: bool,
    ) -> None:
        super().__init__(message, provider=provider, status=429)
        self.retry_after = retry_after
        self.daily_exhausted = daily_exhausted
        self.retryable = not daily_exhausted
