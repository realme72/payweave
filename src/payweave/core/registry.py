"""Gateway registry: register gateways once, resolve them by name and capability.

Adding a gateway means registering it, never editing this module (Open/Closed).
Registration fails fast if a gateway declares a capability it doesn't
implement, so a misconfigured gateway can never be resolved. Resolving checks
the *declared* capabilities, so an implemented-but-disabled feature (e.g.
webhooks without a configured secret) can't be used either.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

from payweave.core.capabilities import FLAG_REQUIRES, INTERFACE_FOR, Capability, Gateway
from payweave.core.errors import (
    CapabilityMismatchError,
    DuplicateGatewayError,
    GatewayNotFoundError,
    UnsupportedCapabilityError,
)

_CAPABILITY_FOR: dict[type[Gateway], Capability] = {v: k for k, v in INTERFACE_FOR.items()}


def validate_gateway(gateway: object) -> None:
    """Raise ``CapabilityMismatchError`` if the gateway declares something it can't do."""
    if not isinstance(gateway, Gateway):
        raise CapabilityMismatchError(f"{gateway!r} does not expose name and capabilities")
    name = gateway.name
    if not name or name != name.strip().lower() or " " in name:
        raise CapabilityMismatchError(f"gateway name must be a lowercase slug, got {name!r}")

    declared = gateway.capabilities
    if not isinstance(declared, frozenset) or not all(isinstance(c, Capability) for c in declared):
        raise CapabilityMismatchError(f"{name}: capabilities must be a frozenset[Capability]")

    for capability, interface in INTERFACE_FOR.items():
        if capability in declared and not isinstance(gateway, interface):
            raise CapabilityMismatchError(
                f"{name}: declares {capability} but doesn't implement {interface.__name__}"
            )

    for flag, required in FLAG_REQUIRES.items():
        if flag in declared and required not in declared:
            raise CapabilityMismatchError(f"{name}: {flag} requires {required}")


class GatewayRegistry:
    def __init__(self) -> None:
        self._gateways: dict[str, Gateway] = {}

    def register(self, gateway: Gateway) -> None:
        validate_gateway(gateway)
        if gateway.name in self._gateways:
            raise DuplicateGatewayError(gateway.name)
        self._gateways[gateway.name] = gateway

    def unregister(self, name: str) -> None:
        if self._gateways.pop(name, None) is None:
            raise GatewayNotFoundError(name)

    def get[T: Gateway](self, name: str, interface: type[T]) -> T:
        """Resolve ``name`` as ``interface``, e.g. ``registry.get("stripe", Refunder)``."""
        gateway = self._gateways.get(name)
        if gateway is None:
            raise GatewayNotFoundError(name)
        capability = _CAPABILITY_FOR.get(interface)
        if capability is None:
            raise TypeError(f"{interface!r} is not a gateway capability interface")
        if capability not in gateway.capabilities:
            raise UnsupportedCapabilityError(name, capability)
        return cast(T, gateway)

    def supports(self, name: str, capability: Capability) -> bool:
        gateway = self._gateways.get(name)
        if gateway is None:
            raise GatewayNotFoundError(name)
        return capability in gateway.capabilities

    def names(self) -> list[str]:
        return sorted(self._gateways)

    def with_capability(self, capability: Capability) -> list[str]:
        return sorted(n for n, g in self._gateways.items() if capability in g.capabilities)

    def __contains__(self, name: object) -> bool:
        return name in self._gateways

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._gateways)
