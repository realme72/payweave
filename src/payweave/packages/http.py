"""Hand-written HTTP client for packages. Never generated.

Status handling is fixed here, not in manifests, because it decides whether
money may have moved:

* 401                         -> AuthenticationFailedError
* 429                         -> RateLimitedError (retryable)
* 5xx / timeout on GET        -> GatewayUnavailableError (reads are safe to retry)
* 5xx / timeout on POST       -> outcome unknown: the request may have been processed
* connection never made       -> GatewayUnavailableError (nothing was sent)
* other 4xx                   -> the manifest's error rules, else InvalidRequestError

Requests go only to the manifest's base URL (an allowlisted https host), with
redirects off, so credentials can't be sent anywhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from payweave.core.errors import (
    ActionRequiredError,
    AuthenticationFailedError,
    CardDeclinedError,
    DuplicateRequestError,
    FraudSuspectedError,
    GatewayError,
    GatewayTimeoutError,
    GatewayUnavailableError,
    InsufficientFundsError,
    InvalidRequestError,
    RateLimitedError,
    UnknownGatewayError,
)
from payweave.packages.manifest import ErrorCategory, Errors

type Json = dict[str, Any]

ERROR_CLASSES: Mapping[ErrorCategory, type[GatewayError]] = {
    "invalid_request": InvalidRequestError,
    "authentication_failed": AuthenticationFailedError,
    "duplicate_request": DuplicateRequestError,
    "card_declined": CardDeclinedError,
    "insufficient_funds": InsufficientFundsError,
    "fraud_suspected": FraudSuspectedError,
    "action_required": ActionRequiredError,
    "rate_limited": RateLimitedError,
}


def _path_value(data: object, dotted: str) -> str | None:
    for part in dotted.split("."):
        if not isinstance(data, Mapping):
            return None
        data = data.get(part)
    return str(data) if data not in (None, "", {}, []) else None


class PackageHttpClient:
    def __init__(
        self,
        *,
        gateway: str,
        base_url: str,
        allowed_hosts: tuple[str, ...],
        auth: httpx.Auth,
        errors: Errors,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if urlsplit(base_url).hostname not in allowed_hosts:  # the manifest checks this too
            raise ValueError("base_url host is not allowlisted")
        self._gateway = gateway
        self._errors = errors
        self._http = httpx.AsyncClient(
            base_url=base_url,
            auth=auth,
            timeout=timeout_seconds,
            transport=transport,
            headers={**(headers or {}), "Accept": "application/json", "User-Agent": "payweave"},
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Json | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Json:
        is_write = method != "GET"
        try:
            response = await self._http.request(method, path, json=body, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise GatewayUnavailableError("could not connect", gateway=self._gateway) from exc
        except httpx.TransportError as exc:
            if is_write:
                raise GatewayTimeoutError(
                    f"no response to {method} {path}; outcome unknown", gateway=self._gateway
                ) from exc
            raise GatewayUnavailableError(f"{method} {path} failed", gateway=self._gateway) from exc
        if response.is_success:
            try:
                data = response.json()
            except ValueError:
                data = None
            if not isinstance(data, dict):
                raise UnknownGatewayError("response is not a JSON object", gateway=self._gateway)
            return data
        raise self._map_error(response, is_write=is_write)

    def _map_error(self, response: httpx.Response, *, is_write: bool) -> GatewayError:
        status = response.status_code
        try:
            payload: object = response.json()
        except ValueError:
            payload = {}
        message = next(
            (v for p in self._errors.message_paths if (v := _path_value(payload, p))), None
        ) or str(response.reason_phrase or "error")
        code = next((v for p in self._errors.code_paths if (v := _path_value(payload, p))), None)
        kwargs: dict[str, Any] = {"gateway": self._gateway, "gateway_code": code}

        if status == 401:
            return AuthenticationFailedError(message, **kwargs)
        if status == 429:
            return RateLimitedError(message, **kwargs)
        if status >= 500:
            if is_write:
                return UnknownGatewayError(f"{status}: {message}; outcome unknown", **kwargs)
            return GatewayUnavailableError(f"{status}: {message}", **kwargs)
        for rule in self._errors.rules:
            if (rule.code is not None and rule.code == code) or (
                rule.message_contains is not None
                and rule.message_contains.lower() in message.lower()
            ):
                return ERROR_CLASSES[rule.category](message, **kwargs)
        return InvalidRequestError(message, **kwargs)
