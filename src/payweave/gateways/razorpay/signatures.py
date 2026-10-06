"""Razorpay HMAC signatures. Constant-time comparisons only.

* Webhooks: ``X-Razorpay-Signature`` = hex HMAC-SHA256 of the **raw** request
  body, keyed with the webhook secret (webhooks__validate-test.md).
* Checkout return: hex HMAC-SHA256 of ``order_id + "|" + payment_id``, keyed
  with the API key secret (standard integration-steps.md).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable


def hmac_sha256_hex(key: str, message: bytes) -> str:
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


def _matches(expected: str, received: str) -> bool:
    return hmac.compare_digest(expected.encode(), received.strip().lower().encode())


def verify_webhook_signature(raw_body: bytes, signature: str, secrets: Iterable[str]) -> bool:
    # Check every secret (no early exit) so timing doesn't reveal which one matched.
    results = [_matches(hmac_sha256_hex(s, raw_body), signature) for s in secrets]
    return any(results)


def verify_checkout_signature(
    order_id: str, payment_id: str, signature: str, key_secret: str
) -> bool:
    message = f"{order_id}|{payment_id}".encode()
    return _matches(hmac_sha256_hex(key_secret, message), signature)
