from pathlib import Path

from payweave.agent.mappers_gen import route_findings
from payweave.packages.runtime import load_package
from payweave.verify.sandbox_smoke import sandbox_config, sandbox_customer

RAZORPAY = Path(__file__).parents[2] / "packages" / "razorpay" / "v1"


def test_config_comes_from_gateway_prefixed_variables() -> None:
    package = load_package(RAZORPAY)
    assert sandbox_config(package, {}) is None  # nothing configured: G6 doesn't run
    env = {
        "RAZORPAY_KEY_ID": "rzp_test_x",
        "RAZORPAY_KEY_SECRET": "s",
        "RAZORPAY_WEBHOOK_SECRETS": "a,b",
    }
    assert sandbox_config(package, env) == {
        "key_id": "rzp_test_x",
        "key_secret": "s",
        "webhook_secrets": ["a", "b"],
    }


def test_customer_only_from_the_environment() -> None:
    assert sandbox_customer({}) is None  # never made up
    customer = sandbox_customer({"PAYWEAVE_TEST_CUSTOMER_ID": "c1"})
    assert customer is not None
    assert (customer.id, customer.phone) == ("c1", None)


def test_sandbox_findings_go_to_the_responsible_mapper() -> None:
    report = {
        "gates": {
            "G6": {
                "findings": [
                    "step create: InvalidRequestError: order_currency : is missing",
                    "step get: UnknownGatewayError: mapper intent failed: KeyError 'x'",
                    "step get unknown refund: expected InvalidRequestError, got X",
                    "step get: an unpaid intent should be REQUIRES_ACTION",
                ]
            }
        }
    }
    routed = route_findings(report)
    assert list(routed) == ["create_payment_body", "intent", "refund"]
    assert len(routed["intent"]) == 2
