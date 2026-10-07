"""Deterministic citation check (the code half of gate G3, PLAN.md §7).

A citation passes only if its ref exists and its quote really is in that
chunk. For identifier fields (header and field names) the value itself must
also appear in a cited quote, so a model can't cite a real passage and still
misspell what it says. Whether a quote *supports* a yes/no or choice field
can't be decided by string matching; that's for the human reviewer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from payweave.agent.spec_card import Section

_MARKUP = re.compile(r"[*`\\]|\[|\]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop Markdown emphasis/code marks and backslashes, collapse whitespace."""
    return _SPACE.sub(" ", _MARKUP.sub("", text)).strip().lower()


def check_section(section: Section, chunks: Mapping[str, str]) -> list[str]:
    """Return problems, one line each; empty means every citation holds."""
    problems: list[str] = []
    normalized = {ref: normalize(text) for ref, text in chunks.items()}
    for name, fact in section.cited_fields().items():
        quotes: list[str] = []
        for citation in fact.citations:
            body = normalized.get(citation.ref)
            if body is None:
                problems.append(f"{name}: ref {citation.ref!r} doesn't exist")
            elif normalize(citation.quote) not in body:
                problems.append(f"{name}: the quote isn't in {citation.ref}; copy the text exactly")
            else:
                quotes.append(normalize(citation.quote))
        if (
            name in section.VERBATIM
            and isinstance(fact.value, str)
            and quotes
            and not any(normalize(fact.value).rstrip("/") in q for q in quotes)
        ):
            problems.append(f"{name}: {fact.value!r} doesn't appear in any cited quote")
    return problems
