"""Hand-written auth and signature schemes a manifest can *choose* (PLAN.md §5).

A package names a scheme; it can never define one. A gateway needing a scheme
that isn't here stops for a human to add it. Comparisons are constant-time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Iterable
from typing import Literal

import httpx

# api_key_headers: the key id and the secret each go in a named header (Cashfree:
# x-client-id / x-client-secret).
type AuthScheme = Literal["basic", "bearer", "api_key_headers"]
# timestamped_*: the signed message is the timestamp header's value followed directly by
# the raw body (Cashfree; checked against its official SDK, not only its pseudocode).
type SignatureScheme = Literal["hmac_sha256_hex", "timestamped_hmac_sha256_base64"]


def http_auth(
    scheme: AuthScheme,
    *,
    username: str = "",
    password: str = "",
    username_header: str = "",
    password_header: str = "",
) -> httpx.Auth:
    if scheme == "basic":
        return httpx.BasicAuth(username, password)
    if scheme == "api_key_headers":
        return _Headers({username_header: username, password_header: password})
    return _Bearer(password)


class _Bearer(httpx.Auth):
    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request: httpx.Request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


class _Headers(httpx.Auth):
    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = headers

    def auth_flow(self, request: httpx.Request):  # type: ignore[no-untyped-def]
        request.headers.update(self._headers)
        yield request


def _hmac_sha256_hex(key: str, message: bytes) -> str:
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


def _hmac_sha256_base64(key: str, message: bytes) -> str:
    return base64.b64encode(hmac.new(key.encode(), message, hashlib.sha256).digest()).decode()


def verify_signature(
    scheme: SignatureScheme, message: bytes, signature: str, keys: Iterable[str]
) -> bool:
    """True if ``signature`` matches ``message`` under any key (all keys are always tried).

    For timestamped schemes ``message`` is already timestamp + raw body.
    """
    if scheme == "hmac_sha256_hex":
        digest, received = _hmac_sha256_hex, signature.strip().lower().encode()
    elif scheme == "timestamped_hmac_sha256_base64":
        digest, received = _hmac_sha256_base64, signature.strip().encode()  # case matters
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown signature scheme {scheme!r}")
    results = [hmac.compare_digest(digest(k, message).encode(), received) for k in keys]
    return any(results)


def is_timestamped(scheme: SignatureScheme) -> bool:
    return scheme.startswith("timestamped_")
