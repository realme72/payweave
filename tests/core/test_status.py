from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from payweave.core.status import (
    ATTEMPT_EDGES,
    INTENT_EDGES,
    REFUND_EDGES,
    AttemptStatus,
    IntentStatus,
    RefundStatus,
    can_advance,
    is_terminal,
    reachable,
)

ALL_GRAPHS = [INTENT_EDGES, ATTEMPT_EDGES, REFUND_EDGES]


@pytest.mark.parametrize("edges", ALL_GRAPHS)
def test_graph_is_total(edges: dict) -> None:  # type: ignore[type-arg]
    enum: Any = type(next(iter(edges)))
    assert set(edges) == set(enum)
    assert all(t in edges for targets in edges.values() for t in targets)


@pytest.mark.parametrize("edges", ALL_GRAPHS)
def test_graph_is_acyclic(edges: dict) -> None:  # type: ignore[type-arg]
    # Forward-only updates rely on this: no status can reach itself.
    for status in edges:
        assert status not in reachable(status)


@given(st.sampled_from(IntentStatus), st.sampled_from(IntentStatus))
def test_intent_never_moves_both_ways(a: IntentStatus, b: IntentStatus) -> None:
    assert not (can_advance(a, b) and can_advance(b, a))


def test_steps_can_be_skipped() -> None:
    assert can_advance(IntentStatus.REQUIRES_ACTION, IntentStatus.REFUNDED)
    assert can_advance(AttemptStatus.PENDING, AttemptStatus.VOIDED)


def test_late_authorisation_allowed_for_attempts() -> None:
    assert can_advance(AttemptStatus.FAILED, AttemptStatus.AUTHORIZED)
    assert can_advance(AttemptStatus.FAILED, AttemptStatus.CAPTURED)


def test_expired_intent_can_still_be_revived_by_late_auth() -> None:
    assert can_advance(IntentStatus.EXPIRED, IntentStatus.AUTHORIZED)
    assert not is_terminal(IntentStatus.EXPIRED)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (IntentStatus.CAPTURED, IntentStatus.AUTHORIZED),
        (IntentStatus.REFUNDED, IntentStatus.CAPTURED),
        (IntentStatus.AUTHORIZED, IntentStatus.REQUIRES_ACTION),
        (IntentStatus.VOIDED, IntentStatus.CAPTURED),
        (AttemptStatus.CAPTURED, AttemptStatus.FAILED),
        (RefundStatus.SUCCEEDED, RefundStatus.PENDING),
    ],
)
def test_backwards_moves_rejected(current: object, target: object) -> None:
    assert not can_advance(current, target)  # type: ignore[type-var]


def test_mixed_enum_types_never_match() -> None:
    # "failed" exists in both enums and StrEnum compares equal to str.
    assert not can_advance(AttemptStatus.PENDING, RefundStatus.FAILED)  # type: ignore[type-var]


def test_terminal_states() -> None:
    terminal = {s for s in IntentStatus if is_terminal(s)}
    assert terminal == {IntentStatus.REFUNDED, IntentStatus.VOIDED, IntentStatus.CANCELED}
