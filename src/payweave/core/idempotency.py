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
