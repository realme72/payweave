"""Score an extracted Spec Card against a hand-checked answer key (eval/spec_cards/*.toml).

Each field is ``correct``, ``wrong``, ``abstained`` (the model said null) or
``missing`` (its section was never accepted). Wrong is the costly outcome:
abstaining sends a human to look, a wrong value silently misleads.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from payweave.agent.spec_card import SpecCard


class Verdict(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"
    ABSTAINED = "abstained"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class FieldScore:
    section: str
    field: str
    expected: Any
    got: Any
    verdict: Verdict


def load_reference(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _same(expected: Any, got: Any) -> bool:
    if isinstance(expected, str) and isinstance(got, str):
        return expected.strip().rstrip("/").lower() == got.strip().rstrip("/").lower()
    return bool(expected == got)


def score(card: SpecCard, reference: dict[str, dict[str, Any]]) -> list[FieldScore]:
    scores: list[FieldScore] = []
    for section_name, fields in reference.items():
        section = getattr(card, section_name)
        for field_name, expected in fields.items():
            if section is None:
                scores.append(FieldScore(section_name, field_name, expected, None, Verdict.MISSING))
                continue
            got = getattr(section, field_name).value
            verdict = (
                Verdict.ABSTAINED
                if got is None
                else Verdict.CORRECT
                if _same(expected, got)
                else Verdict.WRONG
            )
            scores.append(FieldScore(section_name, field_name, expected, got, verdict))
    return scores
