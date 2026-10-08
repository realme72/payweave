"""Per gateway: the hand-written reference (for G7) and the offline simulator (for G5 and G7).

A gateway without an entry can still be verified, but G5 and G7 report
``not_run`` for it, and the package can't be promoted on those gates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import respx

from payweave.core.factory import GatewayProvider
from payweave.gateways.razorpay import provider as hand_written_provider
from payweave.testing import GatewayHarness
from payweave.testing.razorpay import RazorpayHarness

type Ids = Callable[[str], str]


@dataclass(frozen=True)
class Reference:
    provider: GatewayProvider[Any]  # the hand-written implementation
    harness: Callable[[respx.MockRouter, GatewayProvider[Any], Ids | None], GatewayHarness]


def _razorpay(router: respx.MockRouter, provider: GatewayProvider[Any], ids: Ids | None) -> Any:
    return (
        RazorpayHarness(router, provider)
        if ids is None
        else RazorpayHarness(router, provider, ids=ids)
    )


REFERENCES: dict[str, Reference] = {"razorpay": Reference(hand_written_provider, _razorpay)}
