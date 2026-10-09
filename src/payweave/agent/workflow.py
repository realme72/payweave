"""The integration workflow (PLAN.md §6.3) as a LangGraph graph.

    extract_card -> confirm_card [person] -> compile_manifest -> write_draft -> END
                         └ rejected -> END

* ``extract_card`` fills only the card sections still missing (earlier runs
  can be passed in), so a run interrupted by a rate limit costs nothing twice.
* ``confirm_card`` pauses the graph (``interrupt``) until a person approves
  the card. The checkpointer (Postgres in the CLI) keeps the paused run, so
  approval can come later, from another process.
* ``compile_manifest`` is deterministic (manifest_compiler). Gaps don't stop
  the draft from being written, but they mark the run ``needs_person``.

Mapper generation, gates and the repair loop are the next nodes (Phase 4).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from payweave.agent.extract import DocSource, extract_card, known_facts
from payweave.agent.manifest_compiler import compile_manifest
from payweave.agent.spec_card import SECTIONS, SpecCard
from payweave.llm import LLMProvider

type Status = Literal[
    "running", "card_rejected", "needs_person", "drafted", "promotable", "not_promotable"
]
MAPPER_FACT_SECTIONS = ("basics", "payment_flow", "capture", "refunds")


class BuildState(TypedDict, total=False):
    gateway: str
    observed_test_key_prefix: str | None  # from the configured sandbox key, never the key
    card: dict[str, Any]  # SpecCard JSON; sections may be null
    extraction: list[dict[str, Any]]  # per-section outcome of each extraction pass
    card_review: dict[str, Any]  # {"approved": bool, "by": str, "note": str}
    compiled: dict[str, Any]
    draft_dir: str
    mappers: dict[str, Any]  # outcome of writing + gating mappers.py
    status: Status


@dataclass
class Deps:
    drafts: Path
    llm: LLMProvider | None = None  # None: only the sections already in the card are used
    docs: DocSource | None = None
    log: Callable[[str], None] = field(default=lambda _: None)
    settings: dict[str, Any] | None = None  # [settings] a person chose for this gateway
    repair_rounds: int = 2
    gates: Callable[[Path], Any] | None = None  # tests stub the gate run


def card_summary(card: dict[str, Any]) -> list[str]:
    """One line per fact, with its first quote: what the person confirms."""
    lines: list[str] = []
    for cls in SECTIONS:
        section = card.get(cls.TITLE)
        if section is None:
            lines.append(f"{cls.TITLE}: (missing)")
            continue
        for name, fact in section.items():
            quote = fact["citations"][0]["quote"] if fact["citations"] else ""
            shown = f'   "{quote[:90]}"' if quote else ""
            lines.append(f"{cls.TITLE}.{name} = {fact['value']!r}{shown}")
    return lines


async def _write_mappers(deps: Deps, state: BuildState) -> BuildState:
    """Write mappers.py for the draft, then gate and repair it (mappers_gen.build_package)."""
    import tomllib  # noqa: PLC0415

    import tomli_w  # noqa: PLC0415

    from payweave.agent.mappers_gen import build_package  # noqa: PLC0415

    if deps.llm is None or deps.docs is None:  # after_draft routes here only with both
        return {"status": "drafted"}
    draft = Path(state["draft_dir"])
    if deps.settings is not None:
        manifest = tomllib.loads((draft / "manifest.toml").read_text())
        manifest["settings"] = deps.settings
        (draft / "manifest.toml").write_text(tomli_w.dumps(manifest))
    facts = {
        k: v
        for k, v in known_facts(state["card"]).items()
        if k.split(".")[0] in MAPPER_FACT_SECTIONS
    }
    out = await build_package(
        deps.llm, deps.docs, state["gateway"], draft, facts=facts,
        repair_rounds=deps.repair_rounds, log=deps.log, gates=deps.gates,
    )  # fmt: skip
    summary = {
        "stopped": out.stopped,
        "gate_runs": len(out.reports),
        "failed_gates": out.reports[-1].get("failed_gates") if out.reports else None,
        "mappers": [
            {"name": r.name, "accepted": r.code is not None, "turns": r.turns,
             "tokens": {"input": r.input_tokens, "output": r.output_tokens}}
            for r in out.results
        ],
    }  # fmt: skip
    return {"mappers": summary,
            "status": "promotable" if out.promotable else "not_promotable"}  # fmt: skip


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[BuildState, None, BuildState, BuildState]:
    async def extract(state: BuildState) -> BuildState:
        card = dict(state.get("card") or {"gateway": state["gateway"]})
        missing = [cls for cls in SECTIONS if card.get(cls.TITLE) is None]
        if not missing or deps.llm is None or deps.docs is None:
            return {"card": card, "status": "running"}
        run = await extract_card(
            deps.llm,
            deps.docs,
            state["gateway"],
            sections=missing,
            known=known_facts(card),
            log=deps.log,
        )
        found = run.card.model_dump(mode="json")
        card |= {cls.TITLE: found[cls.TITLE] for cls in missing if found[cls.TITLE] is not None}
        outcome = [
            {
                "section": s.name,
                "accepted": s.section is not None,
                "problems": s.problems,
                "turns": s.turns,
                "searches": len(s.searches),
                "tokens": {"input": s.input_tokens, "output": s.output_tokens},
            }
            for s in run.sections
        ]
        return {
            "card": card,
            "extraction": [*state.get("extraction", []), *outcome],
            "status": "running",
        }

    def confirm(state: BuildState) -> BuildState:
        decision = interrupt({"gateway": state["gateway"], "card": card_summary(state["card"])})
        return {"card_review": dict(decision)}

    def after_review(state: BuildState) -> str:
        return "compile_manifest" if state["card_review"].get("approved") else "rejected"

    def rejected(state: BuildState) -> BuildState:
        return {"status": "card_rejected"}

    def compile_(state: BuildState) -> BuildState:
        out = compile_manifest(
            SpecCard.model_validate(state["card"]),
            observed_test_key_prefix=state.get("observed_test_key_prefix"),
        )
        return {
            "compiled": {
                "manifest": out.manifest,
                "provenance": out.provenance,
                "gaps": out.gaps,
                "notes": out.notes,
                "problems": out.problems,
                "toml": out.toml(),
            }
        }

    def write_draft(state: BuildState) -> BuildState:
        compiled = state["compiled"]
        draft = deps.drafts
        draft.mkdir(parents=True, exist_ok=True)
        (draft / "manifest.toml").write_text(compiled["toml"])
        (draft / "provenance.json").write_text(
            json.dumps(
                {"card_review": state["card_review"], "fields": compiled["provenance"]}, indent=2
            )
            + "\n"
        )
        (draft / "card.json").write_text(json.dumps(state["card"], indent=2) + "\n")
        ok = not compiled["gaps"] and not compiled["problems"]
        return {"draft_dir": str(draft), "status": "drafted" if ok else "needs_person"}

    def after_draft(state: BuildState) -> str:
        if state["status"] != "drafted" or deps.llm is None or deps.docs is None:
            return END
        return "write_mappers"

    graph = StateGraph(BuildState)
    graph.add_node("extract_card", extract)
    graph.add_node("confirm_card", confirm)
    graph.add_node("rejected", rejected)
    graph.add_node("compile_manifest", compile_)
    graph.add_node("write_draft", write_draft)
    graph.add_edge(START, "extract_card")
    graph.add_edge("extract_card", "confirm_card")
    graph.add_conditional_edges("confirm_card", after_review, ["compile_manifest", "rejected"])
    graph.add_edge("rejected", END)
    graph.add_edge("compile_manifest", "write_draft")

    async def write_mappers(state: BuildState) -> BuildState:
        return await _write_mappers(deps, state)

    graph.add_node("write_mappers", write_mappers)
    graph.add_conditional_edges("write_draft", after_draft, ["write_mappers", END])
    graph.add_edge("write_mappers", END)
    return graph.compile(checkpointer=checkpointer)
