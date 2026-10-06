"""Factory provider for Razorpay (registered via the ``payweave.gateways`` entry point)."""

from __future__ import annotations

from payweave.core.factory import GatewayContext
from payweave.gateways.razorpay.config import RazorpayConfig
from payweave.gateways.razorpay.gateway import RazorpayGateway


class RazorpayProvider:
    type_name = "razorpay"
    config_model = RazorpayConfig

    def build(self, config: RazorpayConfig, context: GatewayContext) -> RazorpayGateway:
        return RazorpayGateway(config, context)


provider = RazorpayProvider()
