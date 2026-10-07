"""Live check of an LLM provider with our knowledge base as its only tool.

    set -a; . ./.env; set +a
    uv run --extra knowledge python scripts/llm_smoke.py groq "What is the minimum refund?"
    uv run --extra knowledge python scripts/llm_smoke.py groq --models   # current model list

The model decides what to search, reads the chunks, and must answer citing the
chunk refs it used. Prints token usage so free-tier budgets stay visible.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from payweave.knowledge.chunker import ChunkKind
from payweave.knowledge.embedder import FastEmbedEmbedder
from payweave.knowledge.store import KnowledgeStore
from payweave.llm import LLMRequest, Message, OpenAICompatibleProvider, StopReason, Tool
from payweave.llm.retry import complete_with_retry

SEARCH = Tool(
    name="search_docs",
    description=(
        "Search the payment gateway's documentation. Returns passages with a `ref` to cite. "
        "Use `kinds` to narrow: request_params, response_fields, path_params, error, code, guide."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words"},
            "kinds": {"type": "array", "items": {"type": "string", "enum": list(ChunkKind)}},
        },
        "required": ["query"],
    },
)
SYSTEM = (
    "You answer questions about the Razorpay API using only the search_docs tool. "
    "Search before answering; search again with different words if results don't answer it. "
    "Answer briefly and cite every fact with its ref in square brackets, e.g. "
    "[razorpay:api__refunds__create-normal.md#12]. If the docs don't say, say so."
)
MAX_TURNS = 5


async def main(provider_name: str, question: str | None, list_models: bool) -> None:
    llm = OpenAICompatibleProvider.from_preset(provider_name)
    try:
        if list_models:
            print("\n".join(await llm.list_models()))
            return
        assert question  # noqa: S101 - argparse guarantees it
        with KnowledgeStore(FastEmbedEmbedder()) as kb:
            messages: list[Message] = [Message("user", question)]
            used_in = used_out = 0
            for turn in range(1, MAX_TURNS + 1):
                response = await complete_with_retry(
                    llm,
                    LLMRequest(
                        messages=tuple(messages), system=SYSTEM, tools=(SEARCH,), max_tokens=700
                    ),
                    on_wait=lambda wait, _exc: print(f"(rate limited, retrying in {wait:.0f}s)"),
                )
                used_in += response.usage.input_tokens
                used_out += response.usage.output_tokens
                if response.stop_reason is not StopReason.TOOL_USE:
                    print(f"\n{response.content}\n")
                    if response.stop_reason is StopReason.MAX_TOKENS:
                        print("(answer cut off at max_tokens)")
                    break
                messages.append(Message("assistant", response.content, response.tool_calls))
                for call in response.tool_calls:
                    if call.arguments_error:
                        result = f"error: {call.arguments_error}; send valid JSON arguments"
                    else:
                        kinds = [
                            ChunkKind(k) for k in call.arguments.get("kinds", []) if k in ChunkKind
                        ]
                        hits = kb.search(
                            call.arguments.get("query", ""),
                            gateway="razorpay",
                            kinds=kinds,
                            limit=4,
                        )
                        print(
                            f"turn {turn}: search_docs({json.dumps(call.arguments)})"
                            f" -> {[h.ref for h in hits]}"
                        )
                        result = (
                            "\n\n".join(f"ref: {h.ref}\n{h.content}" for h in hits) or "no results"
                        )
                    messages.append(Message("tool", result, tool_call_id=call.id))
            else:
                print(f"(stopped after {MAX_TURNS} turns without a final answer)")
            print(f"{llm.name} / {llm.model}: {used_in} input + {used_out} output tokens")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=["groq", "gemini", "ollama", "openrouter"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--models", action="store_true", help="list the provider's current models")
    args = parser.parse_args()
    if not args.question and not args.models:
        parser.error("give a question or --models")
    asyncio.run(main(args.provider, args.question, args.models))
