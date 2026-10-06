"""What the contract suite needs from each gateway under test.

The suite drives the gateway only through PayWeave's canonical interfaces. The
harness supplies the parts no merchant API can do: the *customer* paying in
checkout, and the gateway *sending* signed webhooks and checkout returns.

For the fake these are direct calls; for a real gateway they are a simulated
gateway server (offline) that behaves like the documented and observed API.
"""

from __future__ import annotations

from typing import Protocol

from payweave.core.capabilities import Gateway


class GatewayHarness(Protocol):
    @property
    def gateway(self) -> Gateway: ...

    @property
    def unknown_intent_id(self) -> str:
        """A well-formed intent id that doesn't exist."""
        ...

    @property
    def unknown_refund_id(self) -> str: ...

    async def customer_pays(self, gateway_intent_id: str, *, succeed: bool = True) -> str:
        """One checkout attempt by the customer. Returns the attempt id."""
        ...

    def signed_webhook(self, gateway_intent_id: str) -> tuple[bytes, dict[str, str]]:
        """A correctly signed 'payment updated' notification (raw body, headers)."""
        ...

    def checkout_return(self, gateway_intent_id: str, attempt_id: str) -> dict[str, str]:
        """The signed params the customer's browser brings back after paying."""
        ...
