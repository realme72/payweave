"""Status state machines for payment intents, payment attempts and refunds.

Every graph is a DAG, which gives us one rule for all updates:

    a status may only ever move to a status *reachable* from it.

Updates often skip steps (a fetch can jump from REQUIRES_ACTION straight to
CAPTURED), so legality is reachability, not a single edge. Because the graphs
have no cycles, a stale or duplicated webhook can never move anything
backwards: the backwards move isn't reachable.

* **Intent**: what the merchant wants to collect (one per merchant order).
  Its status is derived from its attempts and changes only on money milestones.
  A failed attempt does not fail the intent: the customer may retry.
* **Attempt**: one try by the customer (a card, a UPI payment...). An attempt
  can go FAILED -> AUTHORIZED: some gateways (e.g. Razorpay "late
  authorisation") learn of a bank approval after first reporting a failure.
* **Refund**: pending -> succeeded | failed.

EXPIRED is set by our own timeout policy, not by a gateway, and is not final:
a late authorisation can still arrive after we expire an intent.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from functools import cache
from types import MappingProxyType
from typing import cast


class IntentStatus(StrEnum):
    CREATED = "created"
    REQUIRES_ACTION = "requires_action"  # customer must act: checkout, 3DS, UPI approval
    AUTHORIZED = "authorized"  # funds held, waiting for capture
    CAPTURED = "captured"
    PARTIALLY_REFUNDED = "partially_refunded"
    REFUNDED = "refunded"
    VOIDED = "voided"  # authorisation released without capture (incl. auto-expiry)
    CANCELED = "canceled"  # cancelled before any funds were held
    EXPIRED = "expired"  # our timeout policy; a late authorisation may still revive it


class AttemptStatus(StrEnum):
    PENDING = "pending"  # submitted, waiting for the bank
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    VOIDED = "voided"  # authorised, then released/auto-refunded without capture


class RefundStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


type Status = IntentStatus | AttemptStatus | RefundStatus

_I, _A, _R = IntentStatus, AttemptStatus, RefundStatus

INTENT_EDGES: Mapping[IntentStatus, frozenset[IntentStatus]] = MappingProxyType(
    {
        _I.CREATED: frozenset(
            {_I.REQUIRES_ACTION, _I.AUTHORIZED, _I.CAPTURED, _I.CANCELED, _I.EXPIRED}
        ),
        _I.REQUIRES_ACTION: frozenset({_I.AUTHORIZED, _I.CAPTURED, _I.CANCELED, _I.EXPIRED}),
        _I.EXPIRED: frozenset({_I.AUTHORIZED, _I.CAPTURED}),
        _I.AUTHORIZED: frozenset({_I.CAPTURED, _I.VOIDED}),
        _I.CAPTURED: frozenset({_I.PARTIALLY_REFUNDED, _I.REFUNDED}),
        _I.PARTIALLY_REFUNDED: frozenset({_I.REFUNDED}),
        _I.REFUNDED: frozenset(),
        _I.VOIDED: frozenset(),
        _I.CANCELED: frozenset(),
    }
)  # fmt: skip

ATTEMPT_EDGES: Mapping[AttemptStatus, frozenset[AttemptStatus]] = MappingProxyType(
    {
        _A.PENDING: frozenset({_A.AUTHORIZED, _A.CAPTURED, _A.FAILED}),
        _A.FAILED: frozenset({_A.AUTHORIZED, _A.CAPTURED}),  # late authorisation
        _A.AUTHORIZED: frozenset({_A.CAPTURED, _A.VOIDED}),
        _A.CAPTURED: frozenset(),
        _A.VOIDED: frozenset(),
    }
)

REFUND_EDGES: Mapping[RefundStatus, frozenset[RefundStatus]] = MappingProxyType(
    {
        _R.PENDING: frozenset({_R.SUCCEEDED, _R.FAILED}),
        _R.SUCCEEDED: frozenset(),
        _R.FAILED: frozenset(),
    }
)


def _edges(status: Status) -> Mapping[Status, frozenset[Status]]:
    if isinstance(status, IntentStatus):
        return cast(Mapping[Status, frozenset[Status]], INTENT_EDGES)
    if isinstance(status, AttemptStatus):
        return cast(Mapping[Status, frozenset[Status]], ATTEMPT_EDGES)
    return cast(Mapping[Status, frozenset[Status]], REFUND_EDGES)


@cache
def reachable(status: Status) -> frozenset[Status]:
    """All statuses reachable from ``status`` in one or more steps."""
    edges = _edges(status)
    seen: set[Status] = set()
    stack = list(edges[status])
    while stack:
        nxt = stack.pop()
        if nxt not in seen:
            seen.add(nxt)
            stack.extend(edges[nxt])
    return frozenset(seen)


def can_advance[S: (IntentStatus, AttemptStatus, RefundStatus)](current: S, target: S) -> bool:
    """True if ``target`` is reachable from ``current`` (moving forward, any number of steps)."""
    # StrEnum members equal plain strings, and "failed"/"pending" exist in more
    # than one enum, so compare types explicitly instead of trusting ==.
    if type(current) is not type(target):
        return False
    return target in reachable(current)


def is_terminal(status: Status) -> bool:
    return not _edges(status)[status]
