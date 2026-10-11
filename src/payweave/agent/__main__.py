"""Integration agent CLI.

    set -a; . ./.env; set +a
    uv run --extra knowledge python -m payweave.agent spec-card razorpay
    uv run --extra knowledge python -m payweave.agent spec-card razorpay --section webhooks
    uv run --extra knowledge python -m payweave.agent score .runs/spec-cards/<file>.json
    uv run --extra knowledge python -m payweave.agent manifest razorpay <run>.json <run>.json ...

    # The workflow (LangGraph, checkpointed in Postgres): extract what's missing,
    # pause for a person to confirm the card, then compile the draft package.
    uv run --extra agent --extra knowledge python -m payweave.agent build razorpay \
        [--card <run>.json ...]
    uv run --extra agent python -m payweave.agent review-card <thread-id>   # interactive

The card and a per-section log are written to .runs/spec-cards/ (git-ignored),
then scored against eval/spec_cards/<gateway>.toml when that exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from payweave.agent.extract import card_json, extract_card, known_facts
from payweave.agent.reference import FieldScore, Verdict, load_reference, score
from payweave.agent.spec_card import SECTIONS, SpecCard
from payweave.llm import PRESETS
from payweave.llm.fallback import provider_for

RUNS = Path(".runs/spec-cards")
REFERENCES = Path("eval/spec_cards")
DRAFTS = Path(".runs/packages")
PACKAGES = Path("packages")


def print_scores(scores: list[FieldScore]) -> None:
    marks = {
        Verdict.CORRECT: "ok ",
        Verdict.WRONG: "XX ",
        Verdict.ABSTAINED: "-- ",
        Verdict.MISSING: "?? ",
    }
    for s in scores:
        detail = "" if s.verdict is Verdict.CORRECT else f"   (expected {s.expected!r})"
        print(f"  {marks[s.verdict]} {s.section}.{s.field} = {s.got!r}{detail}")
    counts = {v: sum(s.verdict is v for s in scores) for v in Verdict}
    print(
        f"\n{counts[Verdict.CORRECT]}/{len(scores)} correct, {counts[Verdict.WRONG]} wrong, "
        f"{counts[Verdict.ABSTAINED]} abstained, {counts[Verdict.MISSING]} missing"
    )


def score_file(card: SpecCard, sections: list[str] | None = None) -> None:
    """Score against the reference; ``sections`` limits it to the ones that were run."""
    reference = REFERENCES / f"{card.gateway}.toml"
    if reference.exists():
        expected = load_reference(reference)
        if sections:
            expected = {k: v for k, v in expected.items() if k in sections}
        print(f"\nScore against {reference}:")
        print_scores(score(card, expected))


async def run_spec_card(
    gateway: str, provider: str, model: str | None, only: list[str], known_from: list[Path]
) -> None:
    # Imported here so `score` works without the knowledge extra installed.
    from payweave.knowledge.embedder import FastEmbedEmbedder  # noqa: PLC0415
    from payweave.knowledge.store import KnowledgeStore  # noqa: PLC0415

    sections = [s for s in SECTIONS if not only or s.TITLE in only]
    llm = provider_for(provider, model)
    try:
        with KnowledgeStore(FastEmbedEmbedder()) as kb:
            known = known_facts(merge_cards(gateway, known_from)) if known_from else None
            run = await extract_card(llm, kb, gateway, sections=sections, log=print, known=known)
    finally:
        await llm.aclose()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{gateway}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(card_json(run))
    print(
        f"\n{llm.name} / {llm.model}: {run.input_tokens} input + {run.output_tokens} output tokens"
    )
    for s in run.sections:
        status = "accepted" if s.section else f"NOT accepted: {s.problems}"
        print(f"  {s.name}: {status} ({s.turns} turns, {len(s.searches)} searches)")
    print(f"written to {out}")
    score_file(run.card, [s.name for s in run.sections])


def checked_card(gateway: str, paths: list[Path]) -> SpecCard:
    """Merge saved runs, then re-check every citation under today's rules (needs the KB)."""
    from payweave.agent.citations import recheck_card  # noqa: PLC0415
    from payweave.knowledge.embedder import FastEmbedEmbedder  # noqa: PLC0415
    from payweave.knowledge.store import KnowledgeStore  # noqa: PLC0415

    with KnowledgeStore(FastEmbedEmbedder()) as kb:
        card, nulled = recheck_card(merge_cards(gateway, paths), kb.chunks_by_ref)
    if nulled:
        print("re-check under current rules nulled:")
        print("\n".join(f"  - {n}" for n in nulled))
    return card


def merge_cards(gateway: str, paths: list[Path]) -> SpecCard:
    """Accepted sections from several runs; a later file wins for the same section."""
    merged: dict[str, Any] = {"gateway": gateway}
    for path in paths:
        card = json.loads(path.read_text())["card"]
        if card["gateway"] != gateway:
            raise SystemExit(f"{path} is a {card['gateway']} card, not {gateway}")
        merged |= {k: v for k, v in card.items() if k != "gateway" and v is not None}
    return SpecCard.model_validate(merged)


def observed_prefix(env_name: str | None) -> str | None:
    """The test-key prefix of a configured key (never the key itself)."""
    import os  # noqa: PLC0415

    from payweave.agent.manifest_compiler import test_key_prefix  # noqa: PLC0415

    if not env_name:
        return None
    value = os.environ.get(env_name, "")
    prefix = test_key_prefix(value) if value else None
    if prefix is None:
        print(f"note: ${env_name} is unset or has no test-key prefix; not used for live_guard")
    return prefix


def compile_draft(gateway: str, paths: list[Path], test_key_env: str | None = None) -> int:
    from payweave.agent.manifest_compiler import compile_manifest  # noqa: PLC0415

    card = checked_card(gateway, paths)
    out = compile_manifest(card, observed_test_key_prefix=observed_prefix(test_key_env))
    draft = DRAFTS / gateway / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    draft.mkdir(parents=True)
    (draft / "manifest.toml").write_text(out.toml())
    (draft / "provenance.json").write_text(
        json.dumps({"cards": [str(p) for p in paths], "fields": out.provenance}, indent=2) + "\n"
    )
    (draft / "card.json").write_text(card.model_dump_json(indent=2) + "\n")
    print(f"{gateway}: {'compiled' if out.ok else 'INCOMPLETE'} -> {draft / 'manifest.toml'}")
    for title, items in (
        ("gaps (need a person)", out.gaps),
        ("invalid", out.problems),
        ("notes", out.notes),
    ):
        if items:
            print(f"\n{title}:")
            print("\n".join(f"  - {i}" for i in items))
    compile_report(gateway, out.manifest)
    return 0 if out.ok else 1


def _print_compiled(state: dict[str, Any]) -> None:
    compiled = state.get("compiled") or {}
    print(f"\nstatus: {state.get('status')}   draft: {state.get('draft_dir', '-')}")
    for title, key in (
        ("gaps (need a person)", "gaps"),
        ("invalid", "problems"),
        ("notes", "notes"),
    ):
        if compiled.get(key):
            print(f"{title}:")
            print("\n".join(f"  - {i}" for i in compiled[key]))


async def _graph(  # noqa: PLR0917
    gateway: str,
    thread: str,
    saver: Any,
    llm: Any = None,
    docs: Any = None,
    settings: dict[str, Any] | None = None,
) -> Any:
    from payweave.agent.workflow import Deps, build_graph  # noqa: PLC0415

    deps = Deps(drafts=DRAFTS / gateway / thread, llm=llm, docs=docs, log=print, settings=settings)
    return build_graph(deps, saver)


def _settings(path: Path | None) -> dict[str, Any] | None:
    import tomllib  # noqa: PLC0415

    return None if path is None else tomllib.loads(path.read_text()).get("settings", {})


async def run_build(
    gateway: str, cards: list[Path], provider: str, model: str | None, test_key_env: str | None
) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    from payweave.knowledge.store import DEFAULT_DSN  # noqa: PLC0415

    card = checked_card(gateway, cards).model_dump(mode="json") if cards else None
    missing = [s.TITLE for s in SECTIONS if not card or card.get(s.TITLE) is None]
    thread = f"{gateway}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    config: Any = {"configurable": {"thread_id": thread}}
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_DSN) as saver:
        await saver.setup()
        start = {
            "gateway": gateway,
            "card": card or {"gateway": gateway},
            "observed_test_key_prefix": observed_prefix(test_key_env),
        }
        if missing:
            from payweave.knowledge.embedder import FastEmbedEmbedder  # noqa: PLC0415
            from payweave.knowledge.store import KnowledgeStore  # noqa: PLC0415

            print(f"extracting {len(missing)} missing section(s): {', '.join(missing)}")
            llm = provider_for(provider, model)
            try:
                with KnowledgeStore(FastEmbedEmbedder()) as kb:
                    app = await _graph(gateway, thread, saver, llm, kb)
                    await app.ainvoke(start, config)
            finally:
                await llm.aclose()
        else:
            app = await _graph(gateway, thread, saver)
            await app.ainvoke(start, config)
    print(f"\nthread {thread} is waiting for a person to confirm the card:")
    print(f"  uv run --extra agent python -m payweave.agent review-card {thread}")


async def review_card(
    thread: str, provider: str, model: str | None, settings_from: Path | None
) -> int:
    import getpass  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415
    from langgraph.types import Command  # noqa: PLC0415

    from payweave.agent.workflow import card_summary  # noqa: PLC0415
    from payweave.knowledge.store import DEFAULT_DSN  # noqa: PLC0415

    config: Any = {"configurable": {"thread_id": thread}}
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_DSN) as saver:
        gateway = thread.rsplit("-", 1)[0]
        app = await _graph(gateway, thread, saver)
        state = await app.aget_state(config)
        if not state.values:
            print(f"no run {thread!r}", file=sys.stderr)
            return 1
        if state.next != ("confirm_card",):
            _print_compiled(state.values)
            return 0
        print("\n".join(card_summary(state.values["card"])))
        if not sys.stdin.isatty():
            print(
                "\nreview-card needs an interactive terminal: a person confirms.", file=sys.stderr
            )
            return 2
        answer = input("\nApprove this card? Type 'yes' to approve, anything else rejects: ")
        note = input("Note (optional): ")
        decision = {"approved": answer.strip() == "yes", "by": getpass.getuser(), "note": note}
        # After approval the run continues into writing + gating mappers.py.
        from payweave.knowledge.embedder import FastEmbedEmbedder  # noqa: PLC0415
        from payweave.knowledge.store import KnowledgeStore  # noqa: PLC0415

        llm = provider_for(provider, model)
        try:
            with KnowledgeStore(FastEmbedEmbedder()) as kb:
                app = await _graph(gateway, thread, saver, llm, kb, _settings(settings_from))
                result = await app.ainvoke(Command(resume=decision), config)
        finally:
            await llm.aclose()
    _print_compiled(result)
    if result.get("draft_dir"):
        compile_report(gateway, result["compiled"]["manifest"])
    if result.get("mappers"):
        print(f"\nmappers: {json.dumps(result['mappers'], indent=2)}")
    return 0 if result.get("status") == "promotable" else 1


def compile_report(gateway: str, manifest: dict[str, Any]) -> None:
    from payweave.agent.manifest_compiler import compare  # noqa: PLC0415

    reference = PACKAGES / gateway / "v1" / "manifest.toml"
    if reference.exists():
        import tomllib  # noqa: PLC0415

        diff = compare(manifest, tomllib.loads(reference.read_text()))
        total = sum(len(v) for v in diff.values())
        print(f"\nvs {reference} (settings ignored): {len(diff['same'])}/{total} fields identical")
        for kind in ("different", "missing", "extra"):
            for item in diff[kind]:
                print(f"  {kind:<9} {item}")


MAPPER_FACT_SECTIONS = ("basics", "payment_flow", "capture", "refunds")


async def run_mappers(
    draft: Path, provider: str, model: str | None, rounds: int, settings_from: Path | None
) -> int:
    import tomllib  # noqa: PLC0415

    import tomli_w  # noqa: PLC0415

    from payweave.agent.extract import known_facts  # noqa: PLC0415
    from payweave.agent.mappers_gen import build_package  # noqa: PLC0415
    from payweave.knowledge.embedder import FastEmbedEmbedder  # noqa: PLC0415
    from payweave.knowledge.store import KnowledgeStore  # noqa: PLC0415

    manifest = tomllib.loads((draft / "manifest.toml").read_text())
    if settings_from is not None:  # a person's choice, recorded as such
        manifest["settings"] = tomllib.loads(settings_from.read_text()).get("settings", {})
        (draft / "manifest.toml").write_text(tomli_w.dumps(manifest))
        print(f"settings copied from {settings_from} ({len(manifest['settings'])})")
    card = json.loads((draft / "card.json").read_text())
    facts = {k: v for k, v in known_facts(card).items() if k.split(".")[0] in MAPPER_FACT_SECTIONS}
    llm = provider_for(provider, model)
    try:
        with KnowledgeStore(FastEmbedEmbedder()) as kb:
            out = await build_package(
                llm, kb, manifest["gateway"], draft, facts=facts, repair_rounds=rounds, log=print
            )
    finally:
        await llm.aclose()
    tokens_in = sum(r.input_tokens for r in out.results)
    tokens_out = sum(r.output_tokens for r in out.results)
    print(f"\n{llm.name} / {llm.model}: {tokens_in} input + {tokens_out} output tokens")
    for r in out.results:
        state = "accepted" if r.code else f"NOT accepted: {r.problems[:2]}"
        print(f"  {r.name}: {state} ({r.turns} turns, {len(r.searches)} searches)")
    if out.stopped:
        print(f"stopped: {out.stopped} (re-run the same command to resume)")
    if out.reports:
        last = out.reports[-1]
        for name, g in last["gates"].items():
            print(f"  {name} {g['status']:<8} {g['summary']}")
        print(("promotable" if out.promotable else "NOT promotable") + f": {draft}")
    return 0 if out.promotable else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="payweave.agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p_card = sub.add_parser("spec-card", help="extract a gateway's Spec Card from its docs")
    p_card.add_argument("gateway")
    p_card.add_argument("--provider", choices=sorted(PRESETS), default="groq")
    p_card.add_argument(
        "--model",
        help="override the provider's default model; a,b,c moves on when one's daily quota is used",
    )
    p_card.add_argument(
        "--section", action="append", choices=[s.TITLE for s in SECTIONS], default=[]
    )
    p_card.add_argument(
        "--known",
        action="append",
        type=Path,
        default=[],
        help="earlier runs whose facts later sections build on (Section.USES)",
    )
    p_score = sub.add_parser("score", help="score a saved run against its reference")
    p_score.add_argument("run", type=Path)
    p_man = sub.add_parser("manifest", help="compile spec-card runs into a draft manifest.toml")
    p_man.add_argument("gateway")
    p_man.add_argument("cards", nargs="+", type=Path)
    p_man.add_argument("--test-key-env", help="env var holding the sandbox key id (prefix only)")
    p_build = sub.add_parser("build", help="run the integration workflow (pauses for review)")
    p_build.add_argument("gateway")
    p_build.add_argument(
        "--card",
        action="append",
        type=Path,
        default=[],
        help="earlier spec-card runs to start from",
    )
    p_build.add_argument("--provider", choices=sorted(PRESETS), default="groq")
    p_build.add_argument("--model")
    p_build.add_argument("--test-key-env", help="env var holding the sandbox key id (prefix only)")
    p_review = sub.add_parser("review-card", help="confirm or reject a paused run's card")
    p_review.add_argument("thread")
    p_review.add_argument("--provider", choices=sorted(PRESETS), default="groq")
    p_review.add_argument("--model")
    p_review.add_argument("--settings-from", type=Path, help="a manifest whose [settings] to use")
    p_map = sub.add_parser("mappers", help="write mappers.py for a compiled draft, then gate it")
    p_map.add_argument("draft", type=Path)
    p_map.add_argument("--provider", choices=sorted(PRESETS), default="groq")
    p_map.add_argument("--model")
    p_map.add_argument("--rounds", type=int, default=2, help="repair rounds after the gates")
    p_map.add_argument("--settings-from", type=Path, help="a manifest whose [settings] to use")
    args = parser.parse_args()
    if args.command == "mappers":
        raise SystemExit(
            asyncio.run(
                run_mappers(args.draft, args.provider, args.model, args.rounds, args.settings_from)
            )
        )
    if args.command == "build":
        asyncio.run(
            run_build(args.gateway, args.card, args.provider, args.model, args.test_key_env)
        )
        return
    if args.command == "review-card":
        raise SystemExit(
            asyncio.run(review_card(args.thread, args.provider, args.model, args.settings_from))
        )
    if args.command == "manifest":
        raise SystemExit(compile_draft(args.gateway, args.cards, args.test_key_env))
    if args.command == "spec-card":
        asyncio.run(
            run_spec_card(args.gateway, args.provider, args.model, args.section, args.known)
        )
    else:
        score_file(SpecCard.model_validate(json.loads(args.run.read_text())["card"]))


if __name__ == "__main__":
    main()
