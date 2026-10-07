"""Retrieval evaluation: does the right passage come back for real questions?

A case names the document and a phrase the answering chunk must contain, so a
case stays valid when chunking changes (chunk ids don't). Metrics:

* recall@k: share of questions with a relevant chunk in the top k
* MRR: mean of 1/rank of the first relevant chunk (0 if not in the top 10)
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from payweave.knowledge.store import KnowledgeStore, SearchHit, SearchMode

MRR_DEPTH = 10


@dataclass(frozen=True, slots=True)
class Case:
    question: str
    gateway: str
    docs: tuple[str, ...]  # any of these documents may hold the answer
    contains: str  # the answering chunk must contain this (case/space-insensitive)


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: Case
    rank: int | None  # 1-based rank of the first relevant hit, None if not in the top MRR_DEPTH
    top: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalResult:
    mode: SearchMode
    k: int
    results: tuple[CaseResult, ...]

    @property
    def recall_at_k(self) -> float:
        return sum(r.rank is not None and r.rank <= self.k for r in self.results) / len(
            self.results
        )

    @property
    def recall_at_1(self) -> float:
        return sum(r.rank == 1 for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cases(path: Path) -> list[Case]:
    data = tomllib.loads(path.read_text())
    gateway = data["gateway"]
    cases = []
    for raw in data["case"]:
        docs = raw["doc"] if isinstance(raw["doc"], list) else [raw["doc"]]
        cases.append(Case(raw["question"], gateway, tuple(docs), raw["contains"]))
    return cases


def is_relevant(case: Case, hit: SearchHit) -> bool:
    document = hit.ref.split(":", 1)[1].split("#", 1)[0]
    return document in case.docs and _norm(case.contains) in _norm(hit.content)


def evaluate(
    store: KnowledgeStore, cases: list[Case], *, mode: SearchMode, k: int = 5
) -> EvalResult:
    results = []
    for case in cases:
        hits = store.search(case.question, gateway=case.gateway, mode=mode, limit=MRR_DEPTH)
        rank = next((i for i, hit in enumerate(hits, 1) if is_relevant(case, hit)), None)
        results.append(CaseResult(case, rank, tuple(h.ref for h in hits[:k])))
    return EvalResult(mode, k, tuple(results))


def print_report(results: dict[SearchMode, EvalResult], *, k: int) -> None:
    print(f"{'mode':8} {'recall@1':>9} {f'recall@{k}':>9} {'MRR':>6}")
    for mode, result in results.items():
        print(f"{mode:8} {result.recall_at_1:9.0%} {result.recall_at_k:9.0%} {result.mrr:6.3f}")
    default = results.get(SearchMode.VECTOR)  # the store's default mode
    if default:
        misses = [r for r in default.results if r.rank is None or r.rank > k]
        if misses:
            print(f"\nvector misses (not in top {k}):")
            for r in misses:
                print(f"  - {r.case.question}  [rank {r.rank or '>10'}; want {r.case.docs[0]}]")
                print(f"      got: {', '.join(x.split(':', 1)[1] for x in r.top[:3])}")
