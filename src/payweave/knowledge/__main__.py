"""Knowledge base CLI.

uv run --extra knowledge python -m payweave.knowledge ingest razorpay
uv run --extra knowledge python -m payweave.knowledge search "minimum refund amount"
uv run --extra knowledge python -m payweave.knowledge eval eval/retrieval/razorpay.toml
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from payweave.knowledge.chunker import ChunkKind
from payweave.knowledge.embedder import FastEmbedEmbedder
from payweave.knowledge.evaluate import evaluate, load_cases, print_report
from payweave.knowledge.ingest import ingest_gateway
from payweave.knowledge.store import KnowledgeStore, SearchMode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="payweave.knowledge")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="chunk, embed and store a gateway's doc snapshot")
    p_ingest.add_argument("gateway")
    p_search = sub.add_parser("search", help="search the knowledge base")
    p_search.add_argument("query")
    p_search.add_argument("--gateway")
    p_search.add_argument("--mode", choices=[m.value for m in SearchMode], default="vector")
    p_search.add_argument("--kind", action="append", choices=[k.value for k in ChunkKind])
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--full", action="store_true", help="print whole chunks")
    p_eval = sub.add_parser("eval", help="measure retrieval quality on a question set")
    p_eval.add_argument("cases", type=Path)
    p_eval.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    embedder = FastEmbedEmbedder()
    with KnowledgeStore(embedder) as store:
        if args.command == "ingest":
            report = ingest_gateway(store, args.gateway, embedder)
            print(
                f"{args.gateway}: {report.documents} documents ({report.updated} updated, "
                f"{report.removed} removed), {report.chunks} chunks"
            )
            print(store.stats())
        elif args.command == "search":
            hits = store.search(
                args.query,
                gateway=args.gateway,
                mode=SearchMode(args.mode),
                kinds=[ChunkKind(k) for k in args.kind or []],
                limit=args.k,
            )
            for i, hit in enumerate(hits, 1):
                ranks = ", ".join(f"{m} #{r}" for m, r in hit.ranks.items())
                print(f"{i}. {hit.ref}  [{hit.kind}]  ({ranks})")
                body = hit.content if args.full else textwrap.shorten(hit.content, 300)
                print(textwrap.indent(body, "   "), "\n")
        else:
            cases = load_cases(args.cases)
            results = {m: evaluate(store, cases, mode=m, k=args.k) for m in SearchMode}
            print_report(results, k=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
