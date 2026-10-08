"""Run the verification gates on a Gateway Package and write its report.json.

    uv run python -m payweave.verify packages/razorpay/v1
    uv run python -m payweave.verify packages/razorpay/v1 --no-contract   # skip the G5 subprocess

Exit status 0 only if the package is promotable (every required gate passed).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from payweave.verify.report import build_report, write_report

MARKS = {"pass": "ok  ", "fail": "FAIL", "not_run": "--  "}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m payweave.verify")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-contract", action="store_true", help="skip G5 (runs pytest)")
    args = parser.parse_args(argv)

    report = asyncio.run(build_report(args.directory, contract=not args.no_contract))
    path = write_report(args.directory, report)
    pkg = report["package"]
    if pkg:
        print(f"{pkg['gateway']} v{pkg['version']}  sha256 {pkg['sha256']}")
    for name, g in report["gates"].items():
        print(f"  {MARKS[g['status']]} {name}  {g['summary']}")
        for line in g.get("output", []):
            print(f"         {line}")
        for result in g.get("results", []):
            for problem in result.get("differences", []) + result.get("crashes", []):
                print(f"         {result['scenario']}: {problem}")
    verdict = "promotable" if report["promotable"] else "NOT promotable"
    print(f"\n{verdict} (required: {', '.join(report['required_gates'])}); wrote {path}")
    return 0 if report["promotable"] else 1


if __name__ == "__main__":
    sys.exit(main())
