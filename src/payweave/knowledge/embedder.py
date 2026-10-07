"""Embedders: text -> vector. Local by default; the interface allows swapping.

All models run locally on the CPU via fastembed: free, and docs never leave
this machine. ``MODELS`` lists the ones we've measured (docs/knowledge/RETRIEVAL.md).
Anthropic has no embeddings API; a hosted model (e.g. Voyage) would be another
``Embedder`` implementation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # measured best: docs/knowledge/RETRIEVAL.md
# BGE's recommended prefix for *queries* (documents are embedded as-is).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True, slots=True)
class ModelSpec:
    dimensions: int
    query_prefix: str = ""  # each model's own recommended prefixes
    document_prefix: str = ""
    max_tokens: int = 512


# Models compared in docs/knowledge/RETRIEVAL.md. Prefixes from each model card.
MODELS: dict[str, ModelSpec] = {
    "BAAI/bge-small-en-v1.5": ModelSpec(384, BGE_QUERY_INSTRUCTION),
    "BAAI/bge-base-en-v1.5": ModelSpec(768, BGE_QUERY_INSTRUCTION),
    "BAAI/bge-large-en-v1.5": ModelSpec(1024, BGE_QUERY_INSTRUCTION),
    "snowflake/snowflake-arctic-embed-m": ModelSpec(768, BGE_QUERY_INSTRUCTION),
    "mixedbread-ai/mxbai-embed-large-v1": ModelSpec(1024, BGE_QUERY_INSTRUCTION),
    "nomic-ai/nomic-embed-text-v1.5": ModelSpec(768, "search_query: ", "search_document: "),
    "thenlper/gte-base": ModelSpec(768),
}
CACHE_DIR = Path(os.environ.get("PAYWEAVE_MODEL_CACHE", Path.home() / ".cache" / "payweave"))


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Normalised embeddings, so cosine distance == 1 - dot product."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if model_name not in MODELS:
            raise ValueError(f"unknown embedding model {model_name!r}; known: {sorted(MODELS)}")
        self._model_name = model_name
        self._spec = MODELS[model_name]

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]  # fastembed is untyped
        from fastembed import TextEmbedding  # noqa: PLC0415 - heavy import, only when used

        return TextEmbedding(self._model_name, cache_dir=str(CACHE_DIR / "fastembed"))

    @cached_property
    def _tokenizer(self):  # type: ignore[no-untyped-def]
        # A copy without truncation: we must see the real length to stay under the limit.
        tokenizer = self._model.model.tokenizer
        counter = type(tokenizer).from_str(tokenizer.to_str())
        counter.no_truncation()
        counter.no_padding()
        return counter

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._spec.dimensions

    @property
    def max_tokens(self) -> int:
        return self._spec.max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self._spec.document_prefix + t for t in texts]
        return [_unit(v) for v in self._model.embed(prefixed, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return _unit(next(iter(self._model.embed([self._spec.query_prefix + text]))))


def _unit(vector: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vector))
    return [float(x) / norm for x in vector] if norm else [float(x) for x in vector]
