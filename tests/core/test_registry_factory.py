from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from payweave.core.capabilities import Capability, PaymentReader, Refunder
from payweave.core.errors import (
    CapabilityMismatchError,
    DuplicateGatewayError,
    GatewayNotFoundError,
    UnsupportedCapabilityError,
)
from payweave.core.factory import (
    DuplicateProviderError,
    GatewayContext,
    GatewayFactory,
    GatewaySpec,
    InvalidGatewayConfigError,
    UnknownGatewayTypeError,
)
from payweave.core.models import PaymentIntent


class DummyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr
    regions: tuple[str, ...] = ()


class DummyGateway:
    def __init__(self, name: str, caps: frozenset[Capability]) -> None:
        self._name, self._caps = name, caps

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._caps

    async def get_payment(self, gateway_intent_id: str) -> PaymentIntent:
        raise NotImplementedError


class DummyProvider:
    type_name = "dummy"
    config_model = DummyConfig

    def __init__(self, caps: frozenset[Capability] = frozenset({Capability.GET_PAYMENT})) -> None:
        self.caps = caps
        self.built: list[tuple[DummyConfig, GatewayContext]] = []

    def build(self, config: DummyConfig, context: GatewayContext) -> DummyGateway:
        self.built.append((config, context))
        return DummyGateway(context.name, self.caps)


def test_create_validates_and_resolves_by_capability() -> None:
    factory = GatewayFactory([DummyProvider()])
    registry = factory.build_registry(
        [GatewaySpec(name="dummy", type="dummy", config={"api_key": "s"})]
    )
    assert registry.get("dummy", PaymentReader).name == "dummy"
    with pytest.raises(UnsupportedCapabilityError):
        registry.get("dummy", Refunder)
    with pytest.raises(GatewayNotFoundError):
        registry.get("nope", PaymentReader)


def test_one_type_many_instances() -> None:
    provider = DummyProvider()
    factory = GatewayFactory([provider])
    registry = factory.build_registry(
        [
            GatewaySpec(name="dummy_in", type="dummy", config={"api_key": "a"}),
            GatewaySpec(name="dummy_sg", type="dummy", config={"api_key": "b"}),
        ]
    )
    assert registry.names() == ["dummy_in", "dummy_sg"]
    assert [c.name for _, c in provider.built] == ["dummy_in", "dummy_sg"]


def test_duplicate_instance_and_provider_rejected() -> None:
    factory = GatewayFactory([DummyProvider()])
    with pytest.raises(DuplicateProviderError):
        factory.register_provider(DummyProvider())
    spec = GatewaySpec(name="dummy", type="dummy", config={"api_key": "s"})
    with pytest.raises(DuplicateGatewayError):
        factory.build_registry([spec, spec])


def test_replace_swaps_on_purpose_and_returns_old() -> None:
    factory = GatewayFactory([DummyProvider()])
    registry = factory.build_registry(
        [GatewaySpec(name="dummy", type="dummy", config={"api_key": "old"})]
    )
    old = registry.get("dummy", PaymentReader)
    new = factory.create("dummy", {"api_key": "rotated"})
    assert registry.replace(new) is old
    assert registry.get("dummy", PaymentReader) is new
    with pytest.raises(GatewayNotFoundError):  # replace never adds: a typo can't sneak in
        registry.replace(factory.create("dummy", {"api_key": "x"}, name="dumy"))


def test_unknown_type() -> None:
    with pytest.raises(UnknownGatewayTypeError):
        GatewayFactory().create("nope", {})


def test_invalid_config_never_leaks_values() -> None:
    factory = GatewayFactory([DummyProvider()])
    with pytest.raises(InvalidGatewayConfigError) as exc:
        factory.create("dummy", {"api_key": "super-secret-value", "unexpected": "also-secret"})
    assert "unexpected" in str(exc.value)
    assert "secret" not in str(exc.value)


def test_declaring_unimplemented_capability_is_rejected() -> None:
    factory = GatewayFactory(
        [DummyProvider(frozenset({Capability.GET_PAYMENT, Capability.REFUND}))]
    )
    with pytest.raises(CapabilityMismatchError):
        factory.create("dummy", {"api_key": "s"})


def test_flag_without_its_operation_is_rejected() -> None:
    caps = frozenset({Capability.GET_PAYMENT, Capability.PARTIAL_REFUND})
    with pytest.raises(CapabilityMismatchError):
        GatewayFactory([DummyProvider(caps)]).create("dummy", {"api_key": "s"})


def test_config_from_env() -> None:
    factory = GatewayFactory([DummyProvider()])
    env = {"DUMMY_API_KEY": "k", "DUMMY_REGIONS": "in, sg", "OTHER": "x"}
    config: dict[str, Any] = factory.config_from_env("dummy", env=env)
    assert config == {"api_key": "k", "regions": ["in", "sg"]}
    gateway = factory.create("dummy", config)
    assert gateway.name == "dummy"


def test_installed_providers_include_razorpay() -> None:
    assert "razorpay" in GatewayFactory.with_installed_providers().provider_types()
