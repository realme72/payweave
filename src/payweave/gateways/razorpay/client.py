"""Thin async HTTP client for the Razorpay API, with error mapping.

Error rules (errors.md, api__payments__capture.md, refunds idempotency doc):

* 401                    -> AuthenticationFailedError
* 429                    -> RateLimitedError (retryable)
* 400, "same idempotency key" -> DuplicateRequestError
* other 4xx              -> InvalidRequestError (a bug or bad input: never retry)
* 5xx / timeout on GET   -> GatewayUnavailableError (reads are always safe to retry)
* 5xx / timeout on POST  -> outcome unknown: the request may have been processed
* connection never made  -> GatewayUnavailableError (nothing was sent)

Credentials are sent only to the fixed API host over HTTPS.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from payweave.core.errors import (
    AuthenticationFailedError,
    DuplicateRequestError,
    GatewayError,
    GatewayTimeoutError,
    GatewayUnavailableError,
    InvalidRequestError,
    RateLimitedError,
    UnknownGatewayError,
)
from payweave.gateways.razorpay.config import API_BASE_URL, RazorpayConfig

type Json = dict[str, Any]


class RazorpayClient:
    def __init__(
        self,
        config: RazorpayConfig,
        *,
        gateway_name: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._gateway = gateway_name
        self._http = httpx.AsyncClient(
            base_url=API_BASE_URL,
            auth=httpx.BasicAuth(config.key_id, config.key_secret.get_secret_value()),
            timeout=config.timeout_seconds,
            transport=transport,
            headers={"Accept": "application/json", "User-Agent": "payweave"},
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get(self, path: str) -> Json:
        return await self._request("GET", path)

    async def post(
        self, path: str, body: Json, *, headers: Mapping[str, str] | None = None
    ) -> Json:
        return await self._request("POST", path, body=body, headers=headers)

    async def _request(
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
            return self._json(response)
        raise self._map_error(response, is_write=is_write)

    def _json(self, response: httpx.Response) -> Json:
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise UnknownGatewayError("response is not a JSON object", gateway=self._gateway)
        return data

    def _map_error(self, response: httpx.Response, *, is_write: bool) -> GatewayError:
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        error = error if isinstance(error, dict) else {}
        description = str(error.get("description") or response.reason_phrase or "error")
        code = error.get("reason") or error.get("code")
        kwargs: dict[str, Any] = {"gateway": self._gateway, "gateway_code": code}

        if status == 401:
            return AuthenticationFailedError(description, **kwargs)
        if status == 429:
            return RateLimitedError(description, **kwargs)
        if status >= 500:
            if is_write:
                return UnknownGatewayError(f"{status}: {description}; outcome unknown", **kwargs)
            return GatewayUnavailableError(f"{status}: {description}", **kwargs)
        if "same idempotency key" in description.lower():
            return DuplicateRequestError(description, **kwargs)
        return InvalidRequestError(description, **kwargs)
