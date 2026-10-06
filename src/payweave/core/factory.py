"""Gateway factory: turn configuration into ready-to-use gateway instances.

Patterns: **Abstract Factory** (one provider per gateway type) plus **Registry**
(providers keyed by type name, instances keyed by instance name). Adding a gateway
never touches this module:

1. Write a provider: a ``type_name``, a Pydantic ``config_model`` and ``build()``.
2. Expose it as a ``payweave.gateways`` entry point (or call ``register_provider``).

The factory then validates its config (secrets stay in ``SecretStr`` and never
appear in error messages), builds it, checks that it really implements what it
declares, and registers it.

One gateway *type* can back several *instances*, e.g. two Razorpay accounts::

    factory = GatewayFactory.with_installed_providers()
    registry = factory.build_registry([
        GatewaySpec(name="razorpay", type="razorpay", config=factory.config_from_env("razorpay")),
        GatewaySpec(name="razorpay_sg", type="razorpay", config={...}),
    ])
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import entry_points
from typing import Any, Protocol, get_origin

import httpx
from pydantic import BaseModel, ValidationError

from payweave.core.capabilities import Gateway
from payweave.core.errors import DomainError
from payweave.core.registry import GatewayRegistry, validate_gateway
from payweave.observability.tracing import traced

ENTRY_POINT_GROUP = "payweave.gateways"


class UnknownGatewayTypeError(DomainError):
    def __init__(self, type_name: str, known: Iterable[str]) -> None:
        super().__init__(f"no provider for gateway type {type_name!r}; known: {sorted(known)}")
        self.type_name = type_name


class DuplicateProviderError(DomainError):
    def __init__(self, type_name: str) -> None:
        super().__init__(f"a provider for gateway type {type_name!r} is already registered")
        self.type_name = type_name


class InvalidGatewayConfigError(DomainError):
    """Config failed validation. Lists field names and reasons, never the values."""

    def __init__(self, type_name: str, problems: list[str]) -> None:
        super().__init__(f"invalid config for {type_name!r}: " + "; ".join(problems))
        self.type_name = type_name
        self.problems = problems


@dataclass(frozen=True, slots=True)
class GatewayContext:
    """Shared dependencies handed to every gateway a factory builds.

    Gateways must take time and HTTP from here (not call ``datetime.now`` or
    open their own transport) so tests can control both.
    """

    name: str
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    http_transport: httpx.AsyncBaseTransport | None = None  # None = real network


class GatewayProvider[C: BaseModel](Protocol):
    """Knows how to build one gateway type."""

    @property
    def type_name(self) -> str: ...

    @property
    def config_model(self) -> type[C]: ...

    def build(self, config: C, context: GatewayContext) -> Gateway: ...


@dataclass(frozen=True, slots=True)
class GatewaySpec:
    """Declarative description of one gateway instance (e.g. from a TOML file)."""

    name: str
    type: str
    config: Mapping[str, object] | BaseModel


class GatewayFactory:
    def __init__(self, providers: Iterable[GatewayProvider[Any]] = ()) -> None:
        self._providers: dict[str, GatewayProvider[Any]] = {}
        for provider in providers:
            self.register_provider(provider)

    @classmethod
    def with_installed_providers(cls) -> GatewayFactory:
        """A factory with every provider exposed via the ``payweave.gateways`` entry point."""
        factory = cls()
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            factory.register_provider(ep.load())
        return factory

    # --- providers ------------------------------------------------------------

    def register_provider(self, provider: GatewayProvider[Any]) -> None:
        if provider.type_name in self._providers:
            raise DuplicateProviderError(provider.type_name)
        self._providers[provider.type_name] = provider

    def provider_types(self) -> list[str]:
        return sorted(self._providers)

    def _provider(self, type_name: str) -> GatewayProvider[Any]:
        try:
            return self._providers[type_name]
        except KeyError:
            raise UnknownGatewayTypeError(type_name, self._providers) from None

    # --- configuration --------------------------------------------------------

    def parse_config(self, type_name: str, config: Mapping[str, object] | BaseModel) -> BaseModel:
        provider = self._provider(type_name)
        model: type[BaseModel] = provider.config_model
        if isinstance(config, model):
            return config
        if isinstance(config, BaseModel):
            raise InvalidGatewayConfigError(type_name, [f"expected {model.__name__}"])
        try:
            return model.model_validate(dict(config))
        except ValidationError as exc:
            # Rebuild the message ourselves: pydantic's includes input values,
            # which would leak secrets into logs.
            problems = [
                f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()
            ]
            raise InvalidGatewayConfigError(type_name, problems) from None

    def config_from_env(
        self, type_name: str, *, prefix: str | None = None, env: Mapping[str, str] | None = None
    ) -> dict[str, object]:
        """Collect config from ``<PREFIX><FIELD>`` variables (default prefix ``<TYPE>_``).

        Tuple/list fields are comma-separated. Unset variables are left out, so
        model defaults apply.
        """
        env = os.environ if env is None else env
        prefix = f"{type_name.upper()}_" if prefix is None else prefix
        model = self._provider(type_name).config_model
        values: dict[str, object] = {}
        for field_name, info in model.model_fields.items():
            raw = env.get(f"{prefix}{field_name.upper()}")
            if raw is None:
                continue
            if get_origin(info.annotation) in (tuple, list):
                values[field_name] = [part.strip() for part in raw.split(",") if part.strip()]
            else:
                values[field_name] = raw
        return values

    # --- building -------------------------------------------------------------

    def create(
        self,
        type_name: str,
        config: Mapping[str, object] | BaseModel,
        *,
        name: str | None = None,
        context: GatewayContext | None = None,
    ) -> Gateway:
        instance_name = name or type_name
        if context is None:
            context = GatewayContext(name=instance_name)
        elif context.name != instance_name:
            raise InvalidGatewayConfigError(
                type_name, ["context.name must match the instance name"]
            )
        gateway = self._provider(type_name).build(self.parse_config(type_name, config), context)
        validate_gateway(gateway)
        if gateway.name != instance_name:
            raise InvalidGatewayConfigError(
                type_name, [f"provider built {gateway.name!r}, expected {instance_name!r}"]
            )
        return gateway

    def build_registry(
        self,
        specs: Iterable[GatewaySpec],
        *,
        context_for: Callable[[str], GatewayContext] | None = None,
        decorate: Callable[[Gateway], Gateway] | None = traced,
    ) -> GatewayRegistry:
        """Create and register every spec. ``decorate`` wraps each gateway
        (Decorator pattern); by default OpenTelemetry tracing, a no-op unless an
        SDK is configured. Pass ``None`` to register the raw gateways."""
        registry = GatewayRegistry()
        for spec in specs:
            context = context_for(spec.name) if context_for else None
            gateway = self.create(spec.type, spec.config, name=spec.name, context=context)
            registry.register(decorate(gateway) if decorate else gateway)
        return registry
