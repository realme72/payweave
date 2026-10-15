"""Learn where a gateway's error bodies keep their message and codes, from its sandbox.

Reading the docs can't always tell the broad code from the specific one (Cashfree's
4xx body has both ``type`` and ``code``). The sandbox can: make a few requests that
must fail without side effects (fetch an intent that doesn't exist, create one with
an empty body, fetch a refund that doesn't exist) and compare the bodies.

* message: the field whose values read as sentences (contain spaces, and no links:
  Cashfree's ``help`` is a sentence pointing at its docs, not the error)
* code (broad): a code-like field that is the same across different failures
* reason (specific): a code-like field that differs between them

It answers only when the evidence is clear: at least two failures with different
reasons; otherwise that path stays None (and the card's value stands).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from payweave.packages.manifest import Manifest
from payweave.packages.schemes import http_auth

_CODE_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
PROBE_INTENT_ID = "payweaveprobe0000000000"
PROBE_REFUND_ID = "payweaveprobe0000000000"


@dataclass(frozen=True, slots=True)
class ErrorShapeObserved:
    message_path: str | None = None
    code_path: str | None = None
    reason_path: str | None = None
    bodies: int = 0  # how many JSON 4xx bodies the inference used
    evidence: dict[str, list[str]] = field(default_factory=dict)  # path -> values seen


def _leaves(data: object, prefix: str = "") -> dict[str, str]:
    """Dotted path -> string value, for every string leaf of a JSON object."""
    out: dict[str, str] = {}
    if isinstance(data, Mapping):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, str):
                out[path] = value
            elif isinstance(value, Mapping):
                out |= _leaves(value, path)
    return out


def infer_error_shape(bodies: Sequence[object]) -> ErrorShapeObserved:
    leaves = [_leaves(b) for b in bodies if isinstance(b, Mapping)]
    leaves = [x for x in leaves if x]
    if not leaves:
        return ErrorShapeObserved()
    common = set.intersection(*(set(x) for x in leaves))
    evidence = {p: [x[p] for x in leaves] for p in sorted(common)}
    messages = [
        p for p, vs in evidence.items() if all(" " in v.strip() and "://" not in v for v in vs)
    ]
    codes = [p for p, vs in evidence.items() if all(_CODE_LIKE.match(v) for v in vs)]
    constant = [p for p in codes if len(set(evidence[p])) == 1]
    varying = [p for p in codes if len(set(evidence[p])) == len(leaves)]
    clear = len(leaves) >= 2  # one failure can't show what varies
    return ErrorShapeObserved(
        message_path=messages[0] if len(messages) == 1 else None,
        code_path=constant[0] if clear and len(constant) == 1 and varying else None,
        reason_path=varying[0] if clear and len(varying) == 1 else None,
        bodies=len(leaves),
        evidence=evidence,
    )


OBSERVED_REF = "observed:sandbox"


def _sample(path: str, value: str) -> str:
    """The observed value as the JSON it came in: 'error.code' -> {"error": {"code": "X"}}."""
    node: object = value
    for key in reversed(path.split(".")):
        node = {key: node}
    return json.dumps(node)


def observed_card(seen: ErrorShapeObserved) -> dict[str, dict[str, Any]]:
    """Card sections from what the sandbox showed, cited as ``observed:sandbox``.

    Only settled paths become facts; the quote is the value as the response sent it, so
    the usual quote checks still hold. These are evidence from the gateway itself, not a
    reading of its docs, and the score report says so.
    """

    def fact(path: str | None, value: str | None = None) -> dict[str, Any] | None:
        if path is None:
            return None
        sample = _sample(path, seen.evidence.get(path, [""])[0])
        return {"value": value or path, "citations": [{"ref": OBSERVED_REF, "quote": sample}]}

    shape = {
        "message_path": fact(seen.message_path),
        "code_path": fact(seen.code_path),
        "reason_path": fact(seen.reason_path),
    }
    leaf = {p: p.rsplit(".", 1)[-1] if p else None for p in (seen.code_path, seen.reason_path)}
    errors = {
        "error_code_field": fact(seen.code_path, leaf[seen.code_path]),
        "reason_field": fact(seen.reason_path, leaf[seen.reason_path]),
    }
    return {
        name: {k: v for k, v in section.items() if v is not None}
        for name, section in (("error_shape", shape), ("errors", errors))
    }


async def probe_error_shape(
    manifest: Manifest,
    credentials: Mapping[str, str],
    *,
    base_url: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[ErrorShapeObserved, list[str]]:
    """Send the failing requests the manifest allows; returns the inference and a log.

    ``base_url`` is the sandbox's, passed by a person. Test keys only: the manifest's
    live guard is applied first. Nothing here should create anything: the create body
    is empty and the ids don't exist.
    """
    m = manifest
    if not base_url.startswith("https://"):
        raise ValueError("the sandbox base URL must be https")
    guard = m.live_guard
    if guard is None or guard.refuses(credentials.get(guard.credential, "")):
        raise ValueError("refusing to probe: no live guard, or the key isn't a test key")
    auth = http_auth(
        m.auth.scheme,
        username=credentials.get(m.auth.username or "", ""),
        password=credentials.get(m.auth.password, ""),
        username_header=m.auth.username_header or "",
        password_header=m.auth.password_header or "",
    )
    ops = m.operations
    ids = {"intent_id": PROBE_INTENT_ID, "refund_id": PROBE_REFUND_ID, "attempt_id": "x"}
    probes: list[tuple[str, str, Any]] = []
    if ops.get_payment is not None:
        probes.append(("GET", ops.get_payment.primary.path.format_map(ids), None))
    if ops.create_payment is not None:
        probes.append((ops.create_payment.method, ops.create_payment.path, {}))
    if ops.get_refund is not None:
        probes.append(("GET", ops.get_refund.path.format_map(ids), None))
    bodies: list[object] = []
    log: list[str] = []
    async with httpx.AsyncClient(
        base_url=base_url,
        auth=auth,
        headers=dict(m.api.headers),
        timeout=30,
        transport=transport,
        follow_redirects=False,
    ) as client:
        for method, path, body in probes:
            r = await client.request(method, path, json=body)
            log.append(f"{method} {path} -> HTTP {r.status_code}")
            if 400 <= r.status_code < 500 and r.status_code not in (401, 403):
                try:
                    bodies.append(r.json())
                except ValueError:
                    log.append("  (not JSON)")
    return infer_error_shape(bodies), log
