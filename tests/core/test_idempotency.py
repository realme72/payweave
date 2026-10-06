"""Behaviour every IdempotencyStore must have. Add the Postgres store to STORES later."""

import asyncio
from collections.abc import Callable

import pytest

from payweave.core.errors import IdempotencyConflictError, IdempotencyInProgressError
from payweave.core.idempotency import (
    IdempotencyStore,
    InMemoryIdempotencyStore,
    Replay,
    Started,
    fingerprint,
)
from payweave.core.models import RefundRequest
from payweave.core.money import Money

STORES: dict[str, Callable[[], IdempotencyStore]] = {"memory": InMemoryIdempotencyStore}
IDEM = "idem_0123456789abcdef"


@pytest.fixture(params=list(STORES))
def store(request: pytest.FixtureRequest) -> IdempotencyStore:
    return STORES[request.param]()


async def test_first_begin_starts_then_complete_replays(store: IdempotencyStore) -> None:
    assert await store.begin(IDEM, "fp") == Started()
    await store.complete(IDEM, b"result")
    assert await store.begin(IDEM, "fp") == Replay(b"result")
    assert await store.begin(IDEM, "fp") == Replay(b"result")  # stable on every replay


async def test_same_key_different_request_conflicts(store: IdempotencyStore) -> None:
    await store.begin(IDEM, "fp")
    with pytest.raises(IdempotencyConflictError):
        await store.begin(IDEM, "other")
    await store.complete(IDEM, b"result")
    with pytest.raises(IdempotencyConflictError):
        await store.begin(IDEM, "other")


async def test_concurrent_begins_only_one_starts(store: IdempotencyStore) -> None:
    async def attempt() -> object:
        try:
            return await store.begin(IDEM, "fp")
        except IdempotencyInProgressError as exc:
            return exc

    outcomes = await asyncio.gather(*(attempt() for _ in range(20)))
    assert sum(isinstance(o, Started) for o in outcomes) == 1
    assert sum(isinstance(o, IdempotencyInProgressError) for o in outcomes) == 19


async def test_release_allows_a_clean_retry(store: IdempotencyStore) -> None:
    await store.begin(IDEM, "fp")
    await store.release(IDEM)  # the call definitely did not happen
    assert await store.begin(IDEM, "other") == Started()  # key is free again
    await store.release("never_seen_key_0001")  # releasing an unknown key is a no-op


async def test_completed_key_cannot_be_released_or_recompleted(store: IdempotencyStore) -> None:
    await store.begin(IDEM, "fp")
    await store.complete(IDEM, b"result")
    with pytest.raises(LookupError):
        await store.release(IDEM)
    with pytest.raises(LookupError):
        await store.complete(IDEM, b"different")
    with pytest.raises(LookupError):
        await store.complete("never_seen_key_0001", b"x")
    assert await store.begin(IDEM, "fp") == Replay(b"result")


def test_fingerprint_is_stable_and_sensitive() -> None:
    def req(amount: int, reason: str | None = None) -> RefundRequest:
        return RefundRequest(
            gateway_intent_id="order_1",
            idempotency_key=IDEM,
            amount=Money.of(amount, "INR"),
            reason=reason,
        )

    assert fingerprint(req(100)) == fingerprint(req(100))
    assert fingerprint(req(100)) != fingerprint(req(101))
    assert fingerprint(req(100)) != fingerprint(req(100, "duplicate"))
