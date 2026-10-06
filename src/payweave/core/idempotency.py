"""Idempotency: the same key + same request always yields the same result.

Flow for a money-moving call::

    outcome = await store.begin(key, fingerprint(request))
    if isinstance(outcome, Replay):      return outcome.response   # already done
    try:   result = await gateway.call(...)
    except GatewayError as e:
        if e.outcome_unknown: leave IN_PROGRESS -> reconcile later; re-raise
        else:                 await store.release(key)  # nothing happened; may retry
    else:  await store.complete(key, serialized result)

* same key + different request  -> ``IdempotencyConflictError``
* same key while still running  -> ``IdempotencyInProgressError``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from payweave.core.errors import IdempotencyConflictError, IdempotencyInProgressError


def fingerprint(request: BaseModel) -> str:
    """Stable SHA-256 of a request's canonical JSON (sorted keys, no whitespace)."""
    payload = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class IdempotencyState(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class Started:
    """First time we've seen this key: the caller should perform the operation."""


@dataclass(frozen=True, slots=True)
class Replay:
    """The operation already completed: return the stored response."""

    response: bytes


type BeginOutcome = Started | Replay


class IdempotencyStore(Protocol):
    async def begin(self, key: str, request_fingerprint: str) -> BeginOutcome:
        """Atomically claim ``key``.

        Raises ``IdempotencyConflictError`` if the key was used with a different
        fingerprint, ``IdempotencyInProgressError`` if it's still in progress.
        """
        ...

    async def complete(self, key: str, response: bytes) -> None:
        """Store the final response. Only valid for a key that is IN_PROGRESS."""
        ...

    async def release(self, key: str) -> None:
        """Forget an IN_PROGRESS key whose operation definitely did not happen."""
        ...


@dataclass(slots=True)
class _Entry:
    fingerprint: str
    state: IdempotencyState
    response: bytes | None = None


class InMemoryIdempotencyStore:
    """Single-process store for tests and local runs; the Postgres store comes later.

    Each method runs to completion without awaiting, so on one event loop a
    ``begin`` can't interleave with another ``begin``: the claim is atomic.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    async def begin(self, key: str, request_fingerprint: str) -> BeginOutcome:
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = _Entry(request_fingerprint, IdempotencyState.IN_PROGRESS)
            return Started()
        if entry.fingerprint != request_fingerprint:
            raise IdempotencyConflictError(key)
        if entry.state is IdempotencyState.IN_PROGRESS or entry.response is None:
            raise IdempotencyInProgressError(key)
        return Replay(entry.response)

    async def complete(self, key: str, response: bytes) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.state is not IdempotencyState.IN_PROGRESS:
            raise LookupError(f"idempotency key {key!r} is not in progress")
        entry.state, entry.response = IdempotencyState.COMPLETED, response

    async def release(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry is not None and entry.state is IdempotencyState.COMPLETED:
            raise LookupError(f"idempotency key {key!r} already completed; it can't be released")
        self._entries.pop(key, None)
