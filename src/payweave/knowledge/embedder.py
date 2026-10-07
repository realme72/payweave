"""Embedders: text -> vector. Local by default; the interface allows swapping.

BGE-small-en-v1.5 (384 dims, ~67 MB quantised ONNX via fastembed) runs on the
CPU, is free, and keeps docs on this machine. Anthropic has no embeddings API;
a hosted model (e.g. Voyage) would be another ``Embedder`` implementation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE's recommended prefix for *queries* (documents are embedded as-is).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
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

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        query_instruction: str | None = BGE_QUERY_INSTRUCTION,
    ) -> None:
        self._model_name = model_name
        self._query_instruction = query_instruction or ""

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
        return 384

    @property
    def max_tokens(self) -> int:
        return 512

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_unit(v) for v in self._model.embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return _unit(next(iter(self._model.embed([self._query_instruction + text]))))


def _unit(vector: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vector))
    return [float(x) / norm for x in vector] if norm else [float(x) for x in vector]
