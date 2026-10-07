"""Structure-aware chunking of gateway docs (Markdown snapshots).

Decisions (see the chunking notes in PLAN.md §6.1):

* **Boundaries follow structure.** Split at headings; never inside a code block,
  a parameter, an error entry or a table row.
* **Size.** Each chunk's *embedded text* (header + body) is at most
  ``max_tokens`` model tokens: BGE-small silently ignores anything past 512.
  Prose is packed up to ``target_tokens``; pieces under ``min_tokens`` merge
  into their neighbour.
* **Context header.** Every chunk starts with where it lives, e.g.
  ``Razorpay > Capture a Payment > Request body parameters`` plus the endpoint,
  so the three identical "### Parameters" headings on an API page stay distinct.
* **Noise.** SDK samples (Java, Python, Go, PHP, Ruby, Node.js, .NET) repeat the
  curl example; only curl, JSON and other non-SDK blocks are kept.
* **Kinds.** ``path_params``, ``request_params``, ``response_fields``, ``error``,
  ``code`` and ``guide``. Each error is its own chunk.
* **Small-to-big.** Chunks point to their parent ``Section``; the LLM is given
  the whole section, search only sees the chunk.

Token counting is injected (``count_tokens``), so tests don't need the model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum

type TokenCounter = Callable[[str], int]

CHUNKER_VERSION = "1"  # bump when chunking changes, so documents get re-ingested

SDK_LANGUAGES = frozenset({"java", "python", "go", "php", "ruby", "csharp", "dotnet"})
SDK_LABELS = frozenset({"node.js", ".net"})  # javascript is kept unless it's the Node SDK


class ChunkKind(StrEnum):
    PATH_PARAMS = "path_params"
    REQUEST_PARAMS = "request_params"
    RESPONSE_FIELDS = "response_fields"
    ERROR = "error"
    CODE = "code"
    GUIDE = "guide"


@dataclass(frozen=True, slots=True)
class Section:
    """A heading and everything under it (up to the next heading): the 'big' unit."""

    ordinal: int
    heading_path: tuple[str, ...]  # e.g. ("Capture a Payment", "Request body parameters")
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """The 'small' unit we embed and search."""

    section_ordinal: int
    ordinal: int
    kind: ChunkKind
    heading_path: tuple[str, ...]
    header: str  # context header, embedded together with the body
    body: str
    token_count: int

    @property
    def text(self) -> str:
        return f"{self.header}\n\n{self.body}"


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str
    description: str | None
    endpoint: str | None  # e.g. "POST /v1/payments/:id/capture"
    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]
    content_hash: str


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 350
    max_tokens: int = 512
    min_tokens: int = 50
    drop_sdk_samples: bool = True
    context_header: bool = True


# --- parsing ---------------------------------------------------------------------------

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_ENDPOINT = re.compile(r"^\*\*(GET|POST|PUT|PATCH|DELETE)\*\*\s+`([^`]+)`")
_FENCE = re.compile(r"^\s*```(.*)$")
_PARAM = re.compile(r"^`([A-Za-z0-9_.\[\]]+)`(?:\s+_(mandatory|optional)_)?\s*$")
_ERROR_FIELD = re.compile(r"^\* code:")
_TABLE_SEPARATOR = re.compile(r"^---\s*$")


@dataclass
class _RawSection:
    heading_path: list[str]
    lines: list[str] = field(default_factory=list)


def _unescape(text: str) -> str:
    return text.replace("\\{", "{").replace("\\}", "}")


def _is_sdk_fence(info: str) -> bool:
    lang, _, label = info.partition(":")
    return lang.strip().lower() in SDK_LANGUAGES or label.strip().lower() in SDK_LABELS


def _normalize_fences(lines: list[str]) -> list[str]:
    """Make every code block explicitly opened and closed, and un-indent it.

    Razorpay's tabbed samples open the next tab's fence (```java: Java) without
    closing the previous one; only the last tab is closed. Some fences are
    indented inside list items.
    """
    out: list[str] = []
    in_fence, indent = False, 0
    for line in lines:
        fence = _FENCE.match(line)
        if fence and in_fence and fence.group(1).strip():  # a new tab: close, then open
            out += ["```", "```" + fence.group(1).strip()]
            indent = len(line) - len(line.lstrip())
        elif fence and in_fence:
            out.append("```")
            in_fence = False
        elif fence:
            out.append("```" + fence.group(1).strip())
            in_fence, indent = True, len(line) - len(line.lstrip())
        elif in_fence:
            out.append(
                line if not indent else line[indent:] if line[:indent].isspace() else line.lstrip()
            )
        elif len(line) - len(line.lstrip()) >= 8:  # rendering artefact, not list nesting
            out.append(line.lstrip())
        else:
            out.append(line)
    if in_fence:
        out.append("```")
    return out


def _strip_sdk_samples(lines: list[str]) -> list[str]:
    """Drop SDK code blocks (needs normalized fences)."""
    out: list[str] = []
    in_fence = skipping = False
    for line in lines:
        fence = _FENCE.match(line)
        if fence and not in_fence:
            in_fence, skipping = True, _is_sdk_fence(fence.group(1))
        elif fence:
            in_fence = False
            if skipping:
                skipping = False
                continue
        if not skipping:
            out.append(line)
    return out


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def _raw_sections(lines: list[str]) -> list[_RawSection]:
    sections = [_RawSection(heading_path=[])]
    path: list[tuple[int, str]] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
            sections.append(_RawSection(heading_path=[t for _, t in path]))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if any(line.strip() for line in s.lines)]


# --- blocks: the atoms a chunk is packed from ------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Block:
    text: str
    atomic: bool  # True: never split (unless it alone exceeds the limit)
    standalone: bool = False  # True: always its own chunk (errors, table rows)


def _blocks(lines: list[str]) -> Iterator[_Block]:
    """Split a section body into paragraphs, code blocks, parameters, errors and rows."""
    text_lines = lines
    i, n = 0, len(text_lines)
    table_header: str | None = None
    while i < n:
        line = text_lines[i]
        if not line.strip():
            i += 1
            continue
        if _FENCE.match(line):  # code block: up to the closing fence
            j = i + 1
            while j < n and not _FENCE.match(text_lines[j]):
                j += 1
            yield _Block("\n".join(text_lines[i : j + 1]).strip(), atomic=True)
            i = j + 1
            continue
        if _PARAM.match(line.strip()):  # `name` _mandatory_ + its ": ..." description
            j = i + 1
            while (
                j < n
                and not _PARAM.match(text_lines[j].strip())
                and not _FENCE.match(text_lines[j])
            ):
                j += 1
            yield _Block("\n".join(text_lines[i:j]).strip(), atomic=True)
            i = j
            continue
        if i + 1 < n and _ERROR_FIELD.match(text_lines[i + 1].strip()):  # error entry
            j = i + 1
            while (
                j < n
                and text_lines[j].strip()
                and not (j + 1 < n and _ERROR_FIELD.match(text_lines[j + 1].strip()))
            ):
                j += 1
            yield _Block("\n".join(text_lines[i:j]).strip(), atomic=True, standalone=True)
            i = j
            continue
        if " | " in line and i + 1 < n and _TABLE_SEPARATOR.match(text_lines[i + 1]):
            if table_header is None:  # first row of a "a | b | c" + '---' table
                table_header = re.sub(r"\s*\[[^\]]*\]", "", line).strip()
                i += 2
                continue
            columns = [c.strip() for c in table_header.split("|")]
            cells = [c.strip() for c in line.split("|")]
            row = "\n".join(f"{c}: {v}" for c, v in zip(columns, cells, strict=False))
            yield _Block(row, atomic=True, standalone=True)
            i += 2
            continue
        table_header = None  # prose ends any table; the next table has its own header
        j = i + 1  # prose paragraph: until a blank line or a structural line
        while j < n and text_lines[j].strip() and not _FENCE.match(text_lines[j]):
            j += 1
        yield _Block("\n".join(text_lines[i:j]).strip(), atomic=False)
        i = j


# --- labelling ------------------------------------------------------------------------------


def _param_names(lines: list[str]) -> tuple[list[str], bool]:
    names, marked = [], False
    for line in lines:
        match = _PARAM.match(line.strip())
        if match:
            names.append(match.group(1))
            marked = marked or match.group(2) is not None
    return names, marked


def _label(
    heading_path: list[str], lines: list[str], endpoint: str | None
) -> tuple[str, ChunkKind]:
    """Turn a generic heading (e.g. 'Parameters') into a specific one, and pick the kind."""
    title = heading_path[-1] if heading_path else ""
    lowered = title.lower()
    names, marked = _param_names(lines)
    if names and "parameters" in lowered:
        placeholders = set(re.findall(r":(\w+)", endpoint or ""))
        if marked and set(names) <= placeholders | {"id"} and endpoint:
            return "Path parameters", ChunkKind.PATH_PARAMS
        if marked:
            return "Request body parameters", ChunkKind.REQUEST_PARAMS
        return "Response parameters", ChunkKind.RESPONSE_FIELDS
    if lowered == "errors" or any(_ERROR_FIELD.match(ln.strip()) for ln in lines):
        return title, ChunkKind.ERROR
    return title, ChunkKind.GUIDE


# --- packing ---------------------------------------------------------------------------------


def _split_oversized(text: str, budget: int, count: TokenCounter) -> list[str]:
    """Last resort for a single block over budget: split by lines, then sentences."""
    units = text.splitlines() if "\n" in text else re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n".join([*current, unit])
        if current and count(candidate) > budget:
            pieces.append("\n".join(current))
            current = [current[-1]] if not text.lstrip().startswith("```") else []  # overlap
            if count("\n".join([*current, unit])) > budget:
                current = []
        current.append(unit)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _pack(
    blocks: list[_Block], budget: int, target: int, minimum: int, count: TokenCounter
) -> list[tuple[str, bool]]:
    """Greedy packing into (body, standalone) pieces."""
    pieces: list[tuple[str, bool]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append(("\n\n".join(current), False))
            current.clear()

    for block in blocks:
        oversized = count(block.text) > budget
        if block.standalone or oversized:
            flush()
            parts = _split_oversized(block.text, budget, count) if oversized else [block.text]
            pieces.extend((part, block.standalone) for part in parts)
            continue
        if current and count("\n\n".join([*current, block.text])) > target:
            flush()
        current.append(block.text)
    flush()

    merged: list[tuple[str, bool]] = []  # fold tiny prose into the previous prose piece
    for body, standalone in pieces:
        if (
            merged
            and not standalone
            and not merged[-1][1]
            and count(body) < minimum
            and count(f"{merged[-1][0]}\n\n{body}") <= budget
        ):
            merged[-1] = (f"{merged[-1][0]}\n\n{body}", False)
        else:
            merged.append((body, standalone))
    return merged


# --- public API --------------------------------------------------------------------------------


def chunk_markdown(
    text: str,
    *,
    gateway: str,
    count_tokens: TokenCounter,
    config: ChunkingConfig | None = None,
) -> ParsedDocument:
    config = config or ChunkingConfig()
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    meta, body = _frontmatter(_unescape(text))
    lines = _normalize_fences(body.splitlines())
    if config.drop_sdk_samples:
        lines = _strip_sdk_samples(lines)

    endpoint = None
    for line in lines:
        match = _ENDPOINT.match(line.strip())
        if match:
            endpoint = f"{match.group(1)} {match.group(2)}"
            break

    raw = _raw_sections(lines)
    title = meta.get("title") or next((s.heading_path[0] for s in raw if s.heading_path), gateway)
    sections: list[Section] = []
    chunks: list[Chunk] = []
    for raw_section in raw:
        path = raw_section.heading_path or [title]
        label, kind = _label(path, raw_section.lines, endpoint)
        heading_path = (*path[:-1], label) if path else (label,)
        section_text = "\n".join(raw_section.lines).strip()
        section = Section(len(sections), tuple(heading_path), section_text)
        sections.append(section)

        header = f"{gateway.capitalize()} > " + " > ".join(dict.fromkeys(heading_path))
        if endpoint:
            header += f"\n{endpoint}"

        if not config.context_header:
            header = ""
        budget = config.max_tokens - count_tokens(header) - 2
        blocks = list(_blocks(raw_section.lines))
        packed = _pack(blocks, budget, config.target_tokens, config.min_tokens, count_tokens)
        for body_text, _standalone in packed:
            chunk_kind = kind
            if kind is ChunkKind.GUIDE and body_text.lstrip().startswith("```"):
                chunk_kind = ChunkKind.CODE
            elif kind is ChunkKind.GUIDE and "error" in title.lower():
                chunk_kind = ChunkKind.ERROR
            text_tokens = count_tokens(f"{header}\n\n{body_text}")
            chunks.append(
                Chunk(
                    section_ordinal=section.ordinal,
                    ordinal=len(chunks),
                    kind=chunk_kind,
                    heading_path=section.heading_path,
                    header=header,
                    body=body_text,
                    token_count=text_tokens,
                )
            )
    return ParsedDocument(
        title=title,
        description=meta.get("description"),
        endpoint=endpoint,
        sections=tuple(sections),
        chunks=tuple(chunks),
        content_hash=content_hash,
    )
