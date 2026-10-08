"""Gate G7: run the same scenarios on a reference gateway and a candidate, compare everything.

Each scenario runs twice against a fresh, deterministic simulator: once through
the reference (hand-written) gateway, once through the candidate (a Gateway
Package). Every step records the canonical result *and* the HTTP requests it
sent, so two implementations only count as equal if they return the same
models, raise the same errors and put the same bytes on the wire.

Error *messages* are not compared (wording may differ); the error type, gateway
code, decline reason and retry flags are.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import respx
from pydantic import BaseModel

from payweave.core.errors import GatewayError, PayWeaveError
from payweave.testing import GatewayHarness

type Trace = list[dict[str, Any]]
type HarnessFactory = Callable[[respx.MockRouter], GatewayHarness]
MAX_DIFFERENCES = 20


def canonical(value: object) -> Any:  # noqa: PLR0911 - one return per kind
    """A JSON-ready, order-stable form of a gateway result."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, frozenset | set):
        return sorted(canonical(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def describe_request(request: httpx.Request) -> dict[str, Any]:
    """What went on the wire, minus credentials (reports are committed)."""
    try:
        body: Any = json.loads(request.content) if request.content else None
    except ValueError:
        body = {"non_json_bytes": len(request.content)}
    scheme = request.headers.get("authorization", "").split(" ", 1)[0]
    extra = {
        k.lower(): v
        for k, v in request.headers.items()
        if k.lower().startswith("x-") or k.lower() == "idempotency-key"
    }
    return {
        "method": request.method,
        "url": str(request.url.copy_with(query=None)),
        "query": dict(request.url.params),
        "auth": scheme or None,
        "headers": extra,
        "json": body,
    }


class Recorder:
    """Runs scenario steps and records what each one returned and sent."""

    def __init__(self, router: respx.MockRouter) -> None:
        self._router = router
        self.trace: Trace = []

    async def step(self, name: str, action: Callable[[], Any]) -> Any:
        before = len(self._router.calls)
        value: Any = None
        try:
            value = action()
            if inspect.isawaitable(value):
                value = await value
            result: dict[str, Any] = {"ok": canonical(value)}
        except PayWeaveError as exc:
            result = {"error": type(exc).__name__}
            if isinstance(exc, GatewayError):
                result |= {
                    "gateway_code": exc.gateway_code,
                    "decline_reason": exc.decline_reason,
                    "retryable": exc.retryable,
                    "outcome_unknown": exc.outcome_unknown,
                }
            value = None
        except Exception as exc:  # a crash is recorded, then fails the gate
            result = {"crash": f"{type(exc).__name__}: {exc}"}
            value = None
        calls = list(self._router.calls)[before:]
        self.trace.append(
            {
                "step": name,
                "result": result,
                "requests": [describe_request(c.request) for c in calls],
            }
        )
        return value

    def note(self, name: str, value: object) -> None:
        self.trace.append({"step": name, "result": {"ok": canonical(value)}, "requests": []})


type ScenarioFn = Callable[[Recorder, GatewayHarness], Awaitable[None]]


@dataclass(frozen=True)
class Scenario:
    name: str
    run: ScenarioFn


@dataclass
class ScenarioResult:
    name: str
    same: bool
    steps: int
    requests: int
    differences: list[str] = field(default_factory=list)
    crashes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.same and not self.crashes


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(f"{where}: only in {'candidate' if key in b else 'reference'}")
            else:
                out += differences(a[key], b[key], where)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items in reference, {len(b)} in candidate"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: reference {a!r} != candidate {b!r}"]


async def record(scenario: Scenario, make_harness: HarnessFactory) -> Trace:
    with respx.mock(assert_all_called=False) as router:
        harness = make_harness(router)
        recorder = Recorder(router)
        try:
            await scenario.run(recorder, harness)
        finally:
            close = getattr(harness, "aclose", None)
            if close is not None:
                await close()
    return recorder.trace


def _label(trace: Trace, i: int) -> str:
    return trace[i]["step"] if i < len(trace) else f"#{i}"


def compare(scenario: str, reference: Trace, candidate: Trace) -> ScenarioResult:
    found: list[str] = []
    for i in range(max(len(reference), len(candidate))):
        if i >= len(reference) or i >= len(candidate):
            found.append(
                f"step {_label(candidate if i >= len(reference) else reference, i)}: "
                "missing on one side"
            )
            continue
        for d in differences(reference[i], candidate[i]):
            found.append(f"step {reference[i]['step']}: {d}")
    crashes = [
        f"{side} step {s['step']}: {s['result']['crash']}"
        for side, trace in (("reference", reference), ("candidate", candidate))
        for s in trace
        if "crash" in s["result"]
    ]
    return ScenarioResult(
        name=scenario,
        same=not found,
        steps=len(candidate),
        requests=sum(len(s["requests"]) for s in candidate),
        differences=found[:MAX_DIFFERENCES],
        crashes=crashes,
    )


@dataclass
class DifferentialReport:
    scenarios: list[ScenarioResult]
    seconds: float

    @property
    def passed(self) -> bool:
        return bool(self.scenarios) and all(s.passed for s in self.scenarios)

    def as_json(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "scenarios": len(self.scenarios),
            "identical": sum(s.passed for s in self.scenarios),
            "steps": sum(s.steps for s in self.scenarios),
            "requests_compared": sum(s.requests for s in self.scenarios),
            "seconds": round(self.seconds, 3),
            "results": [
                {"scenario": s.name, "identical": s.passed, "steps": s.steps}
                | ({"differences": s.differences} if s.differences else {})
                | ({"crashes": s.crashes} if s.crashes else {})
                for s in self.scenarios
            ],
        }


async def run_differential(
    scenarios: Sequence[Scenario], reference: HarnessFactory, candidate: HarnessFactory
) -> DifferentialReport:
    start = time.perf_counter()
    results = [
        compare(s.name, await record(s, reference), await record(s, candidate)) for s in scenarios
    ]
    return DifferentialReport(results, time.perf_counter() - start)
