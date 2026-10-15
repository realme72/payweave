"""Run gates G1-G7 on a package directory and build its ``report.json`` (PLAN.md §5, §7).

The report is evidence for the person signing (G8). It names the package hash
it was produced for, so a report for older bytes is visibly stale, and every
gate that didn't run says why instead of silently counting as passed.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from payweave.packages.runtime import (
    GatewayPackage,
    PackageLoadError,
    PackageProvider,
    load_package,
)
from payweave.testing.razorpay import sequential_ids
from payweave.verify.differential import run_differential
from payweave.verify.references import REFERENCES
from payweave.verify.sandbox_smoke import sandbox_config, sandbox_customer
from payweave.verify.sandbox_smoke import smoke as sandbox_smoke
from payweave.verify.scenarios import SCENARIOS

REPORT_FILE = "report.json"
CONTRACT_SUITE = Path("tests/contract")
UNDER_TEST_ENV = "PAYWEAVE_PACKAGE_UNDER_TEST"
REQUIRED = ("G1", "G2", "G5", "G7")  # must pass to promote; the rest must not fail

type Status = Literal["pass", "fail", "not_run"]


def gate(status: Status, summary: str, **details: Any) -> dict[str, Any]:
    return {"status": status, "summary": summary, **details}


def _tail(text: str, lines: int = 15) -> list[str]:
    return text.strip().splitlines()[-lines:]


def g1_types(package_dir: Path) -> dict[str, Any]:
    """Schema and signatures were checked by loading; here mappers.py is type-checked strictly."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-error-summary", "mappers.py"],
        cwd=package_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return gate(
            "fail",
            "mypy --strict found problems in mappers.py",
            output=_tail(proc.stdout + proc.stderr),
        )
    return gate(
        "pass", "manifest schema + mapper signatures valid; mappers.py passes mypy --strict"
    )


def g5_contract(package_dir: Path) -> dict[str, Any]:
    if not CONTRACT_SUITE.exists():
        return gate("not_run", f"contract suite not found at {CONTRACT_SUITE} (run from the repo)")
    env = os.environ | {UNDER_TEST_ENV: str(package_dir.resolve())}
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [
            sys.executable,
            "-m",
            "pytest",
            str(CONTRACT_SUITE),
            "-p",
            "no:cacheprovider",
            "-k",
            "under_test",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error)", proc.stdout)}
    if proc.returncode != 0 or not counts.get("passed"):
        return gate("fail", "contract suite failed", counts=counts, output=_tail(proc.stdout))
    return gate(
        "pass",
        f"{counts['passed']} contract tests passed, "
        f"{counts.get('skipped', 0)} skipped (capability not declared)",
        counts=counts,
    )


async def g7_differential(package: GatewayPackage) -> dict[str, Any]:
    gateway = package.manifest.gateway
    ref = REFERENCES.get(gateway)
    if ref is None:
        return gate("not_run", f"no hand-written reference for {gateway}")
    candidate = PackageProvider(package, allow_unsigned=True)
    result = await run_differential(
        SCENARIOS,
        reference=lambda router: ref.harness(router, ref.provider, sequential_ids()),
        candidate=lambda router: ref.harness(router, candidate, sequential_ids()),
    )
    details = result.as_json()
    status = details.pop("status")
    summary = (
        f"{details['identical']}/{details['scenarios']} scenarios identical to the hand-written "
        f"gateway ({details['steps']} steps, {details['requests_compared']} HTTP requests compared)"
    )
    return gate(status, summary, **details)


async def g6_sandbox(package: GatewayPackage | None) -> dict[str, Any]:
    """The automatic sandbox smoke run when its keys are configured (verify/sandbox_smoke.py).

    Paying at checkout still needs a person (tests/gateways/<gateway>/test_sandbox.py).
    """
    if package is None:
        return gate("not_run", "package did not load")
    config = sandbox_config(package)
    if config is None:
        upper = package.manifest.gateway.upper()
        return gate(
            "not_run",
            f"set {upper}_<CREDENTIAL> sandbox keys (and PAYWEAVE_TEST_CUSTOMER_*) to run "
            "the sandbox smoke test",
        )
    try:
        passed, findings = await sandbox_smoke(package, config, sandbox_customer())
    except ValueError as exc:  # config refused, e.g. by the live guard
        return gate("fail", f"sandbox config refused: {exc}")
    if findings:
        return gate(
            "fail", f"sandbox: {len(findings)} problem(s)", passed=passed, findings=findings
        )
    return gate("pass", f"sandbox: {', '.join(passed)} behaved as expected", passed=passed)


async def build_report(package_dir: Path, *, contract: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    gates: dict[str, dict[str, Any]] = {}
    try:
        package = load_package(package_dir)
    except PackageLoadError as exc:
        package = None
        gates["G1"] = gate("fail", f"package does not load: {exc}")
    if package is not None:
        gates["G1"] = g1_types(package_dir)
        gates["G2"] = gate(
            "pass",
            "mapper AST allowlist + restricted builtins: no findings; hosts limited to "
            + ", ".join(package.manifest.api.allowed_hosts),
            not_yet_run=["bandit", "semgrep"],
        )
    has_provenance = (package_dir / "provenance.json").exists()
    gates["G3"] = gate(
        "not_run",
        "citation gate not wired to packages yet"
        if has_provenance
        else "no provenance.json (hand-written port, nothing to cite)",
    )
    gates["G4"] = gate("not_run", "no OpenAPI spec for this gateway")
    if package is not None:
        if contract and package.manifest.gateway in REFERENCES:
            gates["G5"] = g5_contract(package_dir)
        else:
            gates["G5"] = gate(
                "not_run",
                "skipped (--no-contract)"
                if contract is False
                else f"no simulator for {package.manifest.gateway}",
            )
    gates["G6"] = await g6_sandbox(package)
    if package is not None:
        gates["G7"] = await g7_differential(package)
    for name in REQUIRED:
        gates.setdefault(name, gate("not_run", "package did not load"))

    failed = [n for n, g in gates.items() if g["status"] == "fail"]
    promotable = not failed and all(gates[n]["status"] == "pass" for n in REQUIRED)
    return {
        "schema_version": 1,
        "package": None
        if package is None
        else {
            "gateway": package.manifest.gateway,
            "version": package.manifest.version,
            "sha256": package.sha256,
        },
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "generator": {"payweave": version("payweave"), "python": platform.python_version()},
        "seconds": round(time.perf_counter() - started, 2),
        "promotable": promotable,
        "required_gates": list(REQUIRED),
        "failed_gates": failed,
        "gates": dict(sorted(gates.items())),
    }


def write_report(package_dir: Path, report: dict[str, Any]) -> Path:
    path = package_dir / REPORT_FILE
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def read_report(package_dir: Path) -> dict[str, Any] | None:
    path = package_dir / REPORT_FILE
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def report_problem(package: GatewayPackage, report: dict[str, Any] | None) -> str | None:
    """Why this report can't back an approval of ``package`` (None if it can)."""
    if report is None:
        return f"no {REPORT_FILE}; run `python -m payweave.verify <dir>` first"
    if (report.get("package") or {}).get("sha256") != package.sha256:
        return f"{REPORT_FILE} is for different bytes; re-run `python -m payweave.verify <dir>`"
    if not report.get("promotable"):
        failed = report.get("failed_gates") or [
            n for n in report.get("required_gates", []) if report["gates"][n]["status"] != "pass"
        ]
        return f"{REPORT_FILE} says not promotable (gates: {', '.join(failed)})"
    return None
