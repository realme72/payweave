"""One adapter for every provider that speaks the OpenAI chat-completions format.

Groq, Google Gemini (OpenAI-compatible endpoint), local Ollama and OpenRouter
all do, so the free options share this code. Presets hold their base URLs and
key variables; model names change often, so check ``list_models()`` rather
than trusting a hard-coded list.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from opentelemetry import trace
from pydantic import SecretStr

from payweave.llm.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
    Message,
    StopReason,
    ToolCall,
    Usage,
)


@dataclass(frozen=True, slots=True)
class Preset:
    base_url: str
    key_env: str | None  # None: no key needed (local)
    default_model: str


PRESETS: dict[str, Preset] = {
    "groq": Preset("https://api.groq.com/openai/v1/", "GROQ_API_KEY", "qwen/qwen3.8-27b"),
    "gemini": Preset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
    "ollama": Preset("http://localhost:11434/v1/", None, "qwen2.5-coder:7b"),
    "openrouter": Preset("https://openrouter.ai/api/v1/", "OPENROUTER_API_KEY", "openrouter/auto"),
}

_STOP = {
    "stop": StopReason.END,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.CONTENT_FILTER,
}
_DAILY = re.compile(r"per day|\bTPD\b|\bRPD\b|daily", re.IGNORECASE)
_tracer = trace.get_tracer("payweave.llm")


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None = None,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._name = name
        self._model = model
        headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout_seconds, transport=transport
        )

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        model: str | None = None,
        env: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> OpenAICompatibleProvider:
        env = os.environ if env is None else env
        p = PRESETS[preset]
        key = None
        if p.key_env:
            if not env.get(p.key_env):
                raise LLMAuthError(f"set {p.key_env} in .env", provider=preset)
            key = SecretStr(env[p.key_env])
        return cls(
            preset,
            base_url=p.base_url,
            model=model or env.get(f"{preset.upper()}_MODEL") or p.default_model,
            api_key=key,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        return sorted(m["id"] for m in (await self._send("GET", "models"))["data"])

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._model
        body: dict[str, Any] = {
            "model": model,
            "messages": _messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        if request.require_tool:
            body["tool_choice"] = {"type": "function", "function": {"name": request.require_tool}}
        # GenAI semantic conventions, so traces read the same in any backend.
        with _tracer.start_as_current_span(f"chat {model}") as span:
            span.set_attributes(
                {
                    "gen_ai.system": self._name,
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": model,
                    "gen_ai.request.max_tokens": request.max_tokens,
                }
            )
            data = await self._send("POST", "chat/completions", body)
            response = _parse(data, provider=self._name, model=model)
            span.set_attributes(
                {
                    "gen_ai.response.model": response.model,
                    "gen_ai.response.finish_reasons": [str(response.stop_reason)],
                    "gen_ai.usage.input_tokens": response.usage.input_tokens,
                    "gen_ai.usage.output_tokens": response.usage.output_tokens,
                }
            )
        return response

    async def _send(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            r = await self._client.request(method, path, json=body)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("request timed out", provider=self._name) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(f"cannot reach provider: {exc}", provider=self._name) from exc
        if r.is_success:
            return r.json()
        detail = _error_detail(r)  # always keep the provider's own words (it names the limit)
        if r.status_code == 429:
            retry_after = r.headers.get("retry-after")
            raise LLMRateLimitError(
                detail,
                provider=self._name,
                retry_after=float(retry_after) if retry_after else None,
                daily_exhausted=bool(_DAILY.search(detail)),
            )
        if r.status_code in (401, 403):
            raise LLMAuthError(detail, provider=self._name, status=r.status_code)
        if r.status_code >= 500:
            raise LLMUnavailableError(detail, provider=self._name, status=r.status_code)
        raise LLMBadRequestError(detail, provider=self._name, status=r.status_code)


def _messages(request: LLMRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if request.system:
        out.append({"role": "system", "content": request.system})
    for m in request.messages:
        out.append(_message(m))
    return out


def _message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    msg: dict[str, Any] = {"role": m.role, "content": m.content or None}
    if m.tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.raw_arguments or json.dumps(c.arguments),
                },
            }
            for c in m.tool_calls
        ]
    return msg


def _parse(data: dict[str, Any], *, provider: str, model: str) -> LLMResponse:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("response has no choices", provider=provider) from exc
    calls = tuple(_tool_call(c) for c in message.get("tool_calls") or [])
    stop = _STOP.get(choice.get("finish_reason") or "", StopReason.OTHER)
    if calls and stop is StopReason.END:  # some providers say "stop" even with tool calls
        stop = StopReason.TOOL_USE
    usage = data.get("usage") or {}
    return LLMResponse(
        provider=provider,
        model=data.get("model") or model,
        content=message.get("content") or "",
        tool_calls=calls,
        stop_reason=stop,
        usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
    )


def _tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function") or {}
    text = function.get("arguments") or "{}"
    try:
        arguments = json.loads(text) if isinstance(text, str) else dict(text)
        error = None if isinstance(arguments, dict) else "arguments are not a JSON object"
    except json.JSONDecodeError as exc:
        arguments, error = {}, f"invalid JSON: {exc.msg}"
    return ToolCall(
        id=raw.get("id") or "",
        name=function.get("name") or "",
        arguments=arguments if isinstance(arguments, dict) else {},
        raw_arguments=text if isinstance(text, str) else json.dumps(text),
        arguments_error=error,
    )


def _error_detail(r: httpx.Response) -> str:
    try:
        err = r.json().get("error", {})
        message = err.get("message") if isinstance(err, dict) else str(err)
    except (ValueError, AttributeError):
        message = None
    return f"HTTP {r.status_code}: {message or r.text[:300]}"
