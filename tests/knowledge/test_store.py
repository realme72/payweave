"""Knowledge store against a real Postgres + pgvector (skipped if unavailable).

Uses a tiny deterministic bag-of-words embedder, so these tests check the
plumbing (ingest, filters, fusion, small-to-big), not embedding quality.
That is what eval/retrieval/ measures.
"""

import hashlib
import math
import os
import re
from collections.abc import Iterator, Sequence

import psycopg
import pytest

from payweave.knowledge.chunker import ChunkKind, chunk_markdown
from payweave.knowledge.evaluate import Case, CaseResult, EvalResult, is_relevant
from payweave.knowledge.store import (
    DocumentSource,
    KnowledgeStore,
    SearchHit,
    SearchMode,
    reciprocal_rank_fusion,
)

DSN = os.environ.get("PAYWEAVE_TEST_DATABASE_URL", "postgresql:///payweave_test")


class BagOfWordsEmbedder:
    model_name = "test-bag-of-words"
    dimensions = 384
    max_tokens = 512

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * 384
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0  # noqa: S324
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _database_available() -> bool:
    try:
        with psycopg.connect("postgresql:///postgres", autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'payweave_test'"
            ).fetchone()
            if not exists:
                conn.execute("CREATE DATABASE payweave_test")
    except psycopg.Error:
        return False
    return True


pytestmark = pytest.mark.skipif(not _database_available(), reason="no local Postgres")

REFUNDS = """---
title: Create a Normal Refund
---

# Create a Normal Refund

**POST** `/v1/payments/:id/refund`

Use this endpoint to refund a captured payment back to the customer.

### Parameters

`amount` _optional_
: `integer` The amount to be refunded, in the smallest currency unit.

### Errors

The amount must be at least INR 1.00.
* code: 400
* description: The refund amount is below the minimum.
* solution: Refund at least one rupee.
"""

WEBHOOKS = """---
title: Validate Webhooks
---

# Validate Webhooks

Razorpay signs each webhook body with HMAC SHA256 using your webhook secret,
sent in the X-Razorpay-Signature header.
"""


def source(gateway: str, path: str, text: str) -> DocumentSource:
    parsed = chunk_markdown(text, gateway=gateway, count_tokens=lambda t: len(t.split()))
    return DocumentSource(gateway, path, f"https://example.test/{path}", parsed)


@pytest.fixture
def store() -> Iterator[KnowledgeStore]:
    with KnowledgeStore(BagOfWordsEmbedder(), dsn=DSN) as kb:
        kb._conn.execute("TRUNCATE kb_documents CASCADE")
        kb.ingest(source("razorpay", "refunds.md", REFUNDS), pipeline="p1")
        kb.ingest(source("razorpay", "webhooks.md", WEBHOOKS), pipeline="p1")
        kb.ingest(
            source("stripe", "refunds.md", REFUNDS.replace("INR 1.00", "USD 0.50")), pipeline="p1"
        )
        yield kb


def test_ingest_is_idempotent_and_replaces_changed_documents(store: KnowledgeStore) -> None:
    assert store.ingest(source("razorpay", "refunds.md", REFUNDS), pipeline="p1") is False
    assert store.ingest(source("razorpay", "refunds.md", REFUNDS), pipeline="p2") is True
    changed = REFUNDS.replace("at least INR 1.00", "at least INR 2.00")
    assert store.ingest(source("razorpay", "refunds.md", changed), pipeline="p2") is True
    [hit] = store.search("minimum refund amount", gateway="razorpay", kinds=[ChunkKind.ERROR])
    assert "INR 2.00" in hit.content
    assert store.stats()["documents"] == 3


def test_remove_missing_documents(store: KnowledgeStore) -> None:
    assert store.remove_missing("razorpay", ["refunds.md"]) == 1
    refs = [h.ref for h in store.search("HMAC signature webhook", gateway="razorpay", limit=10)]
    assert refs
    assert not any("webhooks.md" in ref for ref in refs)
    assert store.stats()["documents"] == 2  # stripe untouched


@pytest.mark.parametrize("mode", list(SearchMode))
def test_every_mode_finds_the_webhook_signature_chunk(
    store: KnowledgeStore, mode: SearchMode
) -> None:
    hits = store.search("X-Razorpay-Signature HMAC webhook", gateway="razorpay", mode=mode, limit=3)
    assert hits[0].ref == "razorpay:webhooks.md#0"
    assert hits[0].url == "https://example.test/webhooks.md"


def test_filters_by_gateway_and_kind(store: KnowledgeStore) -> None:
    hits = store.search("refund minimum amount", gateway="stripe", limit=10)
    assert hits
    assert {h.gateway for h in hits} == {"stripe"}
    errors = store.search("refund minimum amount", gateway="razorpay", kinds=[ChunkKind.ERROR])
    assert [h.kind for h in errors] == ["error"]
    assert "INR 1.00" in errors[0].content


def test_hits_carry_their_parent_section_and_context_header(store: KnowledgeStore) -> None:
    [hit] = store.search(
        "amount to be refunded", gateway="razorpay", kinds=[ChunkKind.REQUEST_PARAMS]
    )
    assert hit.heading_path == ("Create a Normal Refund", "Request body parameters")
    assert hit.content.startswith("Razorpay > Create a Normal Refund > Request body parameters")
    assert "`amount` _optional_" in hit.section  # the whole section, for the LLM


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion({"keyword": [1, 2, 3], "vector": [3, 1, 4]})
    assert [cid for cid, _, _ in fused][:2] == [1, 3]  # high in both lists wins
    assert fused[0][2] == {"keyword": 1, "vector": 2}
    weighted = reciprocal_rank_fusion({"keyword": [9], "vector": [8]}, weights={"keyword": 0.5})
    assert [cid for cid, _, _ in weighted] == [8, 9]


def test_relevance_and_metrics() -> None:
    case = Case("q", "razorpay", ("refunds.md",), "at least  INR 1.00")
    hit = SearchHit(
        "razorpay:refunds.md#3", "razorpay", "error", (), "Must be AT LEAST inr 1.00", None, "", 0.0
    )
    other = SearchHit(
        "razorpay:webhooks.md#0", "razorpay", "guide", (), "at least INR 1.00", None, "", 0.0
    )
    assert is_relevant(case, hit)
    assert not is_relevant(case, other)  # right text, wrong document
    result = EvalResult(
        SearchMode.VECTOR,
        5,
        (CaseResult(case, 1, ()), CaseResult(case, 4, ()), CaseResult(case, None, ())),
    )
    assert (result.recall_at_1, result.recall_at_k) == (1 / 3, 2 / 3)
    assert result.mrr == pytest.approx((1 + 1 / 4) / 3)
