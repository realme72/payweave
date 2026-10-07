# Knowledge base: retrieval results

How well the knowledge base finds the right passage in Razorpay's docs, what we
tried, and what we kept. Re-run with `make kb-ingest && make kb-eval`.

## Setup

- **Corpus:** 35 Razorpay doc pages (`docs/sources/razorpay/`) → **801 chunks**,
  avg 140 tokens, max 508 (cap 512: BGE models silently ignore anything longer).
- **Embeddings:** `nomic-ai/nomic-embed-text-v1.5` (768 dims; `search_query:` /
  `search_document:` prefixes), local via fastembed. Chosen by the model comparison
  below; the method comparison was run with `BAAI/bge-small-en-v1.5`.
- **Store:** Postgres 18 + pgvector 0.8 (HNSW, cosine) and Postgres full-text search.
- **Eval set:** `eval/retrieval/razorpay.toml`, 55 developer-style questions, each
  with the document(s) that answer it and a phrase the answering chunk must contain.
  Every case was checked to be answerable by at least one chunk.
- **Metrics:** recall@1 and recall@5 (right passage ranked first / in the top 5),
  MRR (mean of 1/rank, over the top 10).

## Results (2026-10-07)

| Method | recall@1 | recall@5 | MRR | latency |
|---|---|---|---|---|
| keyword (full-text) | 33% | 53% | 0.420 | 4 ms |
| **vector** (BGE-small) | **64%** | **89%** | **0.735** | **9 ms** |
| hybrid, RRF equal weights | 51% | 78% | 0.620 | 15 ms |
| hybrid, keyword weight 0.3 | 62% | 82% | 0.697 | 18 ms |
| hybrid top 20 → MiniLM-L6 reranker | 60% | 84% | 0.701 | 633 ms |
| hybrid top 20 → bge-reranker-base | 64% | 87% | 0.732 | 3,209 ms |

**Target was recall@5 ≥ 90%.** BGE-small reached 89% (49/55); switching the
embedding model (next section) reached **91% (50/55)**.

### Embedding models (vector search, same chunks and questions)

| Model | dims | recall@1 | recall@5 | MRR | ingest | query |
|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 64% | 89% | 0.735 | 32 s | 8 ms |
| BAAI/bge-base-en-v1.5 | 768 | 60% | 89% | 0.721 | 139 s | 21 ms |
| BAAI/bge-large-en-v1.5 | 1024 | 62% | 89% | 0.747 | 301 s | 47 ms |
| snowflake-arctic-embed-m | 768 | 62% | 89% | 0.727 | 88 s | 14 ms |
| mxbai-embed-large-v1 | 1024 | 58% | 91% | 0.729 | 252 s | 37 ms |
| **nomic-embed-text-v1.5** (default) | 768 | **64%** | **91%** | **0.751** | 114 s | 18 ms |
| thenlper/gte-base | 768 | 56% | 89% | 0.708 | 91 s | 14 ms |

Every model lands at 89–91%: size barely matters here. Nomic is best or tied on
all three metrics, but its lead is one question, i.e. within noise. The plateau
says the remaining misses are about how those passages are chunked, not about
embedding quality.

### Chunking experiments (vector search)

| Variant | recall@1 | recall@5 |
|---|---|---|
| **baseline** (context header, 350-token target) | **64%** | **89%** |
| no context header | 36% | 75% |
| + page description in the header | 67% | 87% |
| + `_mandatory_` → "(required)" | 62% | 87% |
| both of the above | 67% | 85% |
| both, 250-token target | 62% | 82% |

With 55 questions, one question is ~2%, so only large gaps are meaningful.

## What we learned

1. **The context header is the biggest single win** (+14 points recall@5, +28
   recall@1). Without it, the three identical `### Parameters` sections on every
   API page are indistinguishable.
2. **Hybrid search did not help here.** Keyword search matches noise on these docs:
   every chunk says "payment", "order" and "amount". Gating keyword search to
   questions that contain identifiers (`razorpay_signature`) changed nothing net:
   BGE already handles identifiers well. Keyword/hybrid stay available for
   deliberate exact-identifier lookups.
3. **Rerankers didn't pay off:** the right passage is already in the top 40 for
   54/55 questions, but local cross-encoders reordered no better than the
   embeddings, at 60–350× the latency.
4. **Labels need auditing too.** Four "misses" were correct answers on pages the
   case didn't list (e.g. the receipt's 40-character limit is stated in an error
   entry). Those cases were fixed, with a comment on each in the TOML.

## Remaining misses (vector, BGE-small; Nomic recovers one of them)

- "How do I authenticate API requests?": the answer is a curl `-u key:secret`
  line in a general page; code-heavy chunks from other pages outrank it.
- "Which fields are required to create an order?": short error chunks
  ("The **field name** is required") outrank the request-parameter chunk.
- "set a custom capture timeout when creating the order" / "what happens to
  payments I never capture": the dashboard-settings page outranks the API page.
- "payment attempts against one order", "partial payments": rank 8–9.

## Next levers (not done yet)

- **The agent won't search like this eval.** It will filter by chunk kind
  (`request_params` for "required fields"), run several queries, and use exact
  endpoint lookup; measure retrieval *through the agent* in Phase 4.
- **Contextual retrieval:** have Claude write one sentence of context per chunk
  before embedding (a one-off cost per doc snapshot). Needs the Claude API key.
- A larger eval set (and a second gateway) before tuning further: 55 questions
  can't separate variants that differ by 1–2 questions.
