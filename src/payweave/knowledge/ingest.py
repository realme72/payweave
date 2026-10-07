"""Load a gateway's doc snapshot (docs/sources/<gateway>/) into the knowledge store."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from payweave.knowledge.chunker import CHUNKER_VERSION, ChunkingConfig, chunk_markdown
from payweave.knowledge.embedder import Embedder
from payweave.knowledge.store import DocumentSource, KnowledgeStore

SOURCES = Path(__file__).resolve().parents[3] / "docs" / "sources"


@dataclass(frozen=True, slots=True)
class IngestReport:
    documents: int
    updated: int
    removed: int
    chunks: int


def load_sources(
    gateway: str, embedder: Embedder, *, root: Path = SOURCES, config: ChunkingConfig | None = None
) -> list[DocumentSource]:
    directory = root / gateway
    urls: dict[str, str] = {}
    snapshot = directory / "SNAPSHOT.tsv"
    if snapshot.exists():
        with snapshot.open() as f:
            urls = {row[0]: row[1] for row in csv.reader(f, delimiter="\t") if len(row) >= 2}
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no snapshot docs in {directory}; run `make docs-fetch` first")
    return [
        DocumentSource(
            gateway=gateway,
            path=path.name,
            url=urls.get(path.name),
            parsed=chunk_markdown(
                path.read_text(), gateway=gateway, count_tokens=embedder.count_tokens, config=config
            ),
        )
        for path in paths
    ]


def pipeline_id(embedder: Embedder, config: ChunkingConfig | None = None) -> str:
    return f"chunker-{CHUNKER_VERSION}:{config or ChunkingConfig()}/{embedder.model_name}"


def ingest_gateway(store: KnowledgeStore, gateway: str, embedder: Embedder) -> IngestReport:
    sources = load_sources(gateway, embedder)
    pipeline = pipeline_id(embedder)
    updated = sum(store.ingest(source, pipeline=pipeline) for source in sources)
    removed = store.remove_missing(gateway, [s.path for s in sources])
    return IngestReport(
        documents=len(sources),
        updated=updated,
        removed=removed,
        chunks=sum(len(s.parsed.chunks) for s in sources),
    )
