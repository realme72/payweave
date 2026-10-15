"""Deterministic citation check (the code half of gate G3, PLAN.md §7).

A citation passes only if its ref exists and its quote really is in that
chunk. For identifier fields (header and field names) the value itself must
also appear in a cited quote, so a model can't cite a real passage and still
misspell what it says. Fields claiming an API operation exists must, when
true, quote that operation's endpoint (method + path): a sentence that merely
*mentions* the concept ("has been voided") doesn't prove there's a call for it.
Whether a quote otherwise *supports* a yes/no or choice field can't be decided
by string matching; that's for the human reviewer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

from payweave.agent.spec_card import Section, SpecCard

_MARKUP = re.compile(r"[*`\\]|\[|\]")
_SPACE = re.compile(r"\s+")
# After normalize(): "**POST** `/v1/payments/:id/capture`" or "-X POST https://api.../capture".
_ENDPOINT = re.compile(r"\b(get|post|put|patch|delete) +(https?://|/)\S+")


def normalize(text: str) -> str:
    """Lowercase, drop Markdown emphasis/code marks and backslashes, collapse whitespace."""
    return _SPACE.sub(" ", _MARKUP.sub("", text)).strip().lower()


def _is_key(segment: str, quote: str) -> bool:
    """A key in a JSON sample (`"code": ...`) or an OpenAPI YAML example (`code: ...`)."""
    seg = re.escape(segment.lower())
    return bool(re.search(rf'"{seg}"\s*:|(?:^|\s){seg}:(?:\s|$)', quote))


OBSERVED = "observed:"


def check_section(section: Section, chunks: Mapping[str, str]) -> list[str]:
    """Return problems, one line each; empty means every citation holds."""
    problems: list[str] = []
    normalized = {ref: normalize(text) for ref, text in chunks.items()}
    for name, fact in section.cited_fields().items():
        quotes: list[str] = []
        for citation in fact.citations:
            body = normalized.get(citation.ref)
            if citation.ref.startswith(OBSERVED):
                # Seen in the gateway's own sandbox (verify/error_probe.py), not quoted from
                # its docs: there is no page to look the quote up in.
                quotes.append(normalize(citation.quote))
            elif body is None:
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
        if (
            name in section.JSON_STRINGS
            and isinstance(fact.value, str)
            and quotes
            and not any('"' + normalize(fact.value) in q for q in quotes)
        ):
            problems.append(
                f"{name}: quote it as it appears in a sample response body, inside its JSON "
                'quotes (e.g. "message": "..."), not from an explanation of the error'
            )
        if name in section.PATHS and isinstance(fact.value, str) and quotes:
            # As a key (`"error": {`), not just somewhere ("BAD_REQUEST_ERROR" isn't `error`).
            missing = [
                seg for seg in fact.value.split(".") if not any(_is_key(seg, q) for q in quotes)
            ]
            if missing:
                problems.append(
                    f"{name}: {', '.join(missing)} not a JSON or YAML key in any cited quote; "
                    "quote the sample body including its enclosing keys"
                )
        if (
            name in section.ENDPOINT_EVIDENCE
            and fact.value is True
            and quotes
            and not any(_ENDPOINT.search(q) for q in quotes)
        ):
            problems.append(
                f"{name}: true needs a quote of the endpoint that does it (method and path, "
                "e.g. 'POST /v1/things/:id/cancel'); if there is no such endpoint, "
                "it isn't true"
            )
    return problems


def recheck_card(
    card: SpecCard, chunks_by_ref: Callable[[Iterable[str]], Mapping[str, str]]
) -> tuple[SpecCard, list[str]]:
    """Re-run today's checks on a saved card; fields that no longer pass become null.

    Cards from earlier runs were accepted under the rules of their day. Before a
    card is compiled, every citation is checked again against the stored docs.
    """
    data = card.model_dump(mode="json")
    nulled: list[str] = []
    for name, section in card:
        if not isinstance(section, Section):
            continue
        refs = {c.ref for fact in section.cited_fields().values() for c in fact.citations}
        for problem in check_section(section, chunks_by_ref(sorted(refs))):
            field = problem.split(":", 1)[0]
            data[name][field] = {"value": None, "citations": []}
            nulled.append(f"{name}.{problem}")
    return SpecCard.model_validate(data), nulled
