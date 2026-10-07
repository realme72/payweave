"""Postgres knowledge store: documents -> sections (big) -> chunks (small, embedded).

Search modes:

* ``keyword``: full-text search. English stemming over the chunk, OR-ed query
  terms, ranked by ``ts_rank_cd``; identifier-like terms (``X-Refund-Idempotency``,
  ``error_reason``) are also matched unstemmed via the ``simple`` config.
* ``vector``: cosine distance between the query embedding and chunk embeddings
  (pgvector, HNSW index).
* ``hybrid``: both, merged with Reciprocal Rank Fusion (RRF).

**Default is ``vector``: it measured best** on eval/retrieval/razorpay.toml
(docs/knowledge/RETRIEVAL.md). Keyword search on these docs adds noise for
natural-language questions; keep it for deliberate exact-identifier lookups.

Every search can be filtered by gateway and chunk kind. Each hit carries a
stable ``ref`` (``razorpay:api__refunds__create-normal.md#12``) used for
citations, and its parent section text for the LLM (small-to-big).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from payweave.knowledge.chunker import ChunkKind, ParsedDocument
from payweave.knowledge.embedder import Embedder

DEFAULT_DSN = os.environ.get("PAYWEAVE_DATABASE_URL", "postgresql:///payweave")
RRF_K = 60  # standard constant: dampens the weight of top ranks
CANDIDATES = 40  # per method, before fusion

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_documents (
    id              bigserial PRIMARY KEY,
    gateway         text NOT NULL,
    path            text NOT NULL,          -- snapshot file, e.g. api__refunds__create-normal.md
    url             text,
    title           text NOT NULL,
    endpoint        text,                   -- e.g. POST /v1/payments/:id/refund
    content_hash    text NOT NULL,          -- sha256 of the snapshot
    pipeline        text NOT NULL,          -- chunker version + embedding model
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (gateway, path)
);

CREATE TABLE IF NOT EXISTS kb_sections (
    id              bigserial PRIMARY KEY,
    document_id     bigint NOT NULL REFERENCES kb_documents ON DELETE CASCADE,
    ordinal         int NOT NULL,
    heading_path    text[] NOT NULL,
    content         text NOT NULL,
    UNIQUE (document_id, ordinal)
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id              bigserial PRIMARY KEY,
    ref             text NOT NULL UNIQUE,   -- stable citation id
    document_id     bigint NOT NULL REFERENCES kb_documents ON DELETE CASCADE,
    section_id      bigint NOT NULL REFERENCES kb_sections ON DELETE CASCADE,
    gateway         text NOT NULL,
    kind            text NOT NULL,
    heading_path    text[] NOT NULL,
    content         text NOT NULL,          -- context header + body: exactly what was embedded
    token_count     int NOT NULL,
    embedding       vector(384) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', content) || to_tsvector('simple', content)) STORED
);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
    ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS kb_chunks_filter_idx ON kb_chunks (gateway, kind);
"""


class SearchMode(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SearchHit:
    ref: str
    gateway: str
    kind: str
    heading_path: tuple[str, ...]
    content: str
    url: str | None
    section: str  # parent section text, for the LLM
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # method -> 1-based rank


@dataclass(frozen=True, slots=True)
class DocumentSource:
    gateway: str
    path: str
    url: str | None
    parsed: ParsedDocument


_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_\-.][A-Za-z0-9]+)+")


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]], k: int = RRF_K, weights: dict[str, float] | None = None
) -> list[tuple[int, float, dict[str, int]]]:
    """Merge ranked id lists: score(id) = sum over lists of weight / (k + rank)."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for method, ids in rankings.items():
        weight = (weights or {}).get(method, 1.0)
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            ranks.setdefault(chunk_id, {})[method] = rank
    ordered = sorted(scores, key=lambda cid: (-scores[cid], min(ranks[cid].values())))
    return [(cid, scores[cid], ranks[cid]) for cid in ordered]


class KnowledgeStore:
    def __init__(self, embedder: Embedder, dsn: str = DEFAULT_DSN) -> None:
        if embedder.dimensions != 384:
            raise ValueError("the kb_chunks.embedding column is vector(384)")
        self._embedder = embedder
        self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        self._conn.execute(SCHEMA)
        register_vector(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- ingestion ---------------------------------------------------------------------

    def ingest(self, source: DocumentSource, *, pipeline: str) -> bool:
        """Insert or replace one document. Returns False if it was already up to date."""
        doc = source.parsed
        existing = self._conn.execute(
            "SELECT content_hash, pipeline FROM kb_documents WHERE gateway = %s AND path = %s",
            (source.gateway, source.path),
        ).fetchone()
        if existing and (existing["content_hash"], existing["pipeline"]) == (
            doc.content_hash,
            pipeline,
        ):
            return False
        vectors = self._embedder.embed_documents([c.text for c in doc.chunks])
        with self._conn.transaction():
            self._conn.execute(
                "DELETE FROM kb_documents WHERE gateway = %s AND path = %s",
                (source.gateway, source.path),
            )
            doc_id = self._conn.execute(
                """INSERT INTO kb_documents
                       (gateway, path, url, title, endpoint, content_hash, pipeline)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    source.gateway,
                    source.path,
                    source.url,
                    doc.title,
                    doc.endpoint,
                    doc.content_hash,
                    pipeline,
                ),
            ).fetchone()["id"]  # type: ignore[index]
            section_ids: dict[int, int] = {}
            for section in doc.sections:
                section_ids[section.ordinal] = self._conn.execute(
                    """INSERT INTO kb_sections (document_id, ordinal, heading_path, content)
                       VALUES (%s, %s, %s, %s) RETURNING id""",
                    (doc_id, section.ordinal, list(section.heading_path), section.text),
                ).fetchone()["id"]  # type: ignore[index]
            with self._conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO kb_chunks (ref, document_id, section_id, gateway, kind,
                           heading_path, content, token_count, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (
                            f"{source.gateway}:{source.path}#{c.ordinal}",
                            doc_id,
                            section_ids[c.section_ordinal],
                            source.gateway,
                            str(c.kind),
                            list(c.heading_path),
                            c.text,
                            c.token_count,
                            vector,
                        )
                        for c, vector in zip(doc.chunks, vectors, strict=True)
                    ],
                )
        return True

    def remove_missing(self, gateway: str, keep_paths: Iterable[str]) -> int:
        """Delete documents of ``gateway`` that are no longer in the snapshot."""
        cur = self._conn.execute(
            "DELETE FROM kb_documents WHERE gateway = %s AND NOT (path = ANY(%s))",
            (gateway, list(keep_paths)),
        )
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        return (
            self._conn.execute(
                """SELECT count(DISTINCT d.id) AS documents, count(c.id) AS chunks,
                      round(avg(c.token_count)) AS avg_tokens, max(c.token_count) AS max_tokens
               FROM kb_documents d LEFT JOIN kb_chunks c ON c.document_id = d.id"""
            ).fetchone()
            or {}
        )

    # --- search ----------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        gateway: str | None = None,
        kinds: Sequence[ChunkKind] | None = None,
        mode: SearchMode = SearchMode.VECTOR,
        limit: int = 5,
        weights: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        rankings: dict[str, list[int]] = {}
        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            rankings["keyword"] = self._keyword_ids(query, gateway, kinds, CANDIDATES)
        if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            rankings["vector"] = self._vector_ids(query, gateway, kinds, CANDIDATES)
        fused = reciprocal_rank_fusion(rankings, weights=weights)[:limit]
        return self._hits(fused)

    def _filters(
        self, gateway: str | None, kinds: Sequence[ChunkKind] | None
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if gateway:
            clauses.append("c.gateway = %s")
            params.append(gateway)
        if kinds:
            clauses.append("c.kind = ANY(%s)")
            params.append([str(k) for k in kinds])
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def _keyword_ids(
        self, query: str, gateway: str | None, kinds: Sequence[ChunkKind] | None, limit: int
    ) -> list[int]:
        where, params = self._filters(gateway, kinds)
        identifiers = " ".join(_IDENTIFIER.findall(query))
        rows = self._conn.execute(
            f"""WITH q AS (
                    SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery
                           || coalesce(nullif(plainto_tsquery('simple', %s)::text, ''), '')::tsquery
                           AS query
                )
                SELECT c.id FROM kb_chunks c, q
                WHERE c.tsv @@ q.query {where}
                ORDER BY ts_rank_cd(c.tsv, q.query, 1) DESC, c.id  -- 1: damp long chunks
                LIMIT %s""",  # noqa: S608 - only fixed clauses are interpolated
            [query, identifiers, *params, limit],
        ).fetchall()
        return [r["id"] for r in rows]

    def _vector_ids(
        self, query: str, gateway: str | None, kinds: Sequence[ChunkKind] | None, limit: int
    ) -> list[int]:
        where, params = self._filters(gateway, kinds)
        vector = self._embedder.embed_query(query)
        rows = self._conn.execute(
            f"""SELECT c.id FROM kb_chunks c
                WHERE true {where}
                ORDER BY c.embedding <=> %s::vector, c.id
                LIMIT %s""",  # noqa: S608 - only fixed clauses are interpolated
            [*params, vector, limit],
        ).fetchall()
        return [r["id"] for r in rows]

    def _hits(self, fused: list[tuple[int, float, dict[str, int]]]) -> list[SearchHit]:
        if not fused:
            return []
        rows = self._conn.execute(
            """SELECT c.id, c.ref, c.gateway, c.kind, c.heading_path, c.content, d.url,
                      s.content AS section
               FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id
                                JOIN kb_sections s ON s.id = c.section_id
               WHERE c.id = ANY(%s)""",
            ([cid for cid, _, _ in fused],),
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [
            SearchHit(
                ref=by_id[cid]["ref"],
                gateway=by_id[cid]["gateway"],
                kind=by_id[cid]["kind"],
                heading_path=tuple(by_id[cid]["heading_path"]),
                content=by_id[cid]["content"],
                url=by_id[cid]["url"],
                section=by_id[cid]["section"],
                score=score,
                ranks=ranks,
            )
            for cid, score, ranks in fused
        ]
