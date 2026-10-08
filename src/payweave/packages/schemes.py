"""Hand-written auth and signature schemes a manifest can *choose* (PLAN.md §5).

A package names a scheme; it can never define one. A gateway needing a scheme
that isn't here stops for a human to add it. Comparisons are constant-time.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from typing import Literal

import httpx

type AuthScheme = Literal["basic", "bearer"]
type SignatureScheme = Literal["hmac_sha256_hex"]


def http_auth(scheme: AuthScheme, *, username: str = "", password: str = "") -> httpx.Auth:
    if scheme == "basic":
        return httpx.BasicAuth(username, password)
    return _Bearer(password)


class _Bearer(httpx.Auth):
    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request: httpx.Request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def _hmac_sha256_hex(key: str, message: bytes) -> str:
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


def verify_signature(
    scheme: SignatureScheme, message: bytes, signature: str, keys: Iterable[str]
) -> bool:
    """True if ``signature`` matches ``message`` under any key (all keys are always tried)."""
    if scheme != "hmac_sha256_hex":  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown signature scheme {scheme!r}")
    received = signature.strip().lower().encode()
    results = [hmac.compare_digest(_hmac_sha256_hex(k, message).encode(), received) for k in keys]
    return any(results)
