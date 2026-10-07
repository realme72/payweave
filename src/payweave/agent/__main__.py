"""Integration agent CLI.

    set -a; . ./.env; set +a
    uv run --extra knowledge python -m payweave.agent spec-card razorpay
    uv run --extra knowledge python -m payweave.agent spec-card razorpay --section webhooks
    uv run --extra knowledge python -m payweave.agent score .runs/spec-cards/<file>.json

The card and a per-section log are written to .runs/spec-cards/ (git-ignored),
then scored against eval/spec_cards/<gateway>.toml when that exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from payweave.agent.extract import card_json, extract_card
from payweave.agent.reference import FieldScore, Verdict, load_reference, score
from payweave.agent.spec_card import SECTIONS, SpecCard
from payweave.llm import PRESETS, OpenAICompatibleProvider

RUNS = Path(".runs/spec-cards")
REFERENCES = Path("eval/spec_cards")


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


async def run_spec_card(gateway: str, provider: str, model: str | None, only: list[str]) -> None:
    # Imported here so `score` works without the knowledge extra installed.
    from payweave.knowledge.embedder import FastEmbedEmbedder  # noqa: PLC0415
    from payweave.knowledge.store import KnowledgeStore  # noqa: PLC0415

    sections = [s for s in SECTIONS if not only or s.TITLE in only]
    llm = OpenAICompatibleProvider.from_preset(provider, model=model)
    try:
        with KnowledgeStore(FastEmbedEmbedder()) as kb:
            run = await extract_card(llm, kb, gateway, sections=sections, log=print)
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="payweave.agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p_card = sub.add_parser("spec-card", help="extract a gateway's Spec Card from its docs")
    p_card.add_argument("gateway")
    p_card.add_argument("--provider", choices=sorted(PRESETS), default="groq")
    p_card.add_argument("--model", help="override the provider's default model")
    p_card.add_argument(
        "--section", action="append", choices=[s.TITLE for s in SECTIONS], default=[]
    )
    p_score = sub.add_parser("score", help="score a saved run against its reference")
    p_score.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "spec-card":
        asyncio.run(run_spec_card(args.gateway, args.provider, args.model, args.section))
    else:
        score_file(SpecCard.model_validate(json.loads(args.run.read_text())["card"]))


if __name__ == "__main__":
    main()
