import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from payweave.packages.runtime import PackageProvider, load_package, load_package_from_sources
from payweave.testing.razorpay import KEY_SECRET, sequential_ids
from payweave.verify import report as report_mod
from payweave.verify.differential import (
    DifferentialReport,
    Recorder,
    Scenario,
    differences,
    record,
    run_differential,
)
from payweave.verify.references import REFERENCES
from payweave.verify.scenarios import SCENARIOS

RAZORPAY = Path(__file__).parents[2] / "packages" / "razorpay" / "v1"
MANIFEST = (RAZORPAY / "manifest.toml").read_text()
MAPPERS = (RAZORPAY / "mappers.py").read_text()
REF = REFERENCES["razorpay"]


async def differential(mappers: str = MAPPERS, manifest: str = MANIFEST) -> DifferentialReport:
    candidate = PackageProvider(load_package_from_sources(manifest, mappers), allow_unsigned=True)
    return await run_differential(
        SCENARIOS,
        reference=lambda router: REF.harness(router, REF.provider, sequential_ids()),
        candidate=lambda router: REF.harness(router, candidate, sequential_ids()),
    )


def failing(result: DifferentialReport) -> dict[str, list[str]]:
    return {s.name: s.differences + s.crashes for s in result.scenarios if not s.passed}


# --- G7 -------------------------------------------------------------------------------------


async def test_razorpay_package_is_identical_to_hand_written() -> None:
    result = await differential()
    assert failing(result) == {}
    assert result.passed
    summary = result.as_json()
    assert summary["scenarios"] == len(SCENARIOS)
    assert summary["requests_compared"] > 50


@pytest.mark.parametrize(
    ("old", "new", "scenario", "detail"),
    [
        # A changed request body: same results, different bytes on the wire.
        (
            'body: dict[str, Any] = {"amount": amount.amount_minor}',
            'body: dict[str, Any] = {"amount": amount.amount_minor, "speed": "optimum"}',
            "refunds",
            "json.speed: only in candidate",
        ),
        # A wrong status mapping.
        (
            'case "authorized":\n            return AttemptStatus.AUTHORIZED',
            'case "authorized":\n            return AttemptStatus.PENDING',
            "manual_capture",
            "result.ok.attempts[0].status",
        ),
        # A mapper that crashes on one path only.
        (
            "def refund(entity: Json, ctx: MapperContext) -> Refund:\n",
            "def refund(entity: Json, ctx: MapperContext) -> Refund:\n    entity = {}\n",
            "refunds",
            "result.error: only in candidate",
        ),
    ],
)
async def test_differential_catches_behaviour_changes(
    old: str, new: str, scenario: str, detail: str
) -> None:
    assert old in MAPPERS
    result = await differential(MAPPERS.replace(old, new, 1))
    problems = failing(result)
    assert scenario in problems
    assert any(detail in p for p in problems[scenario]), problems[scenario]
    assert not result.passed


async def test_capability_differences_show_up() -> None:
    manifest = MANIFEST.replace('"partial_refund", ', "")
    assert manifest != MANIFEST
    problems = failing(await differential(manifest=manifest))
    assert "capabilities" in problems


async def test_traces_never_contain_credentials() -> None:
    for scenario in SCENARIOS:
        trace = await record(
            scenario, lambda router: REF.harness(router, REF.provider, sequential_ids())
        )
        text = json.dumps(trace)
        assert KEY_SECRET not in text
        assert "Basic " not in text  # only the scheme name is kept


async def test_identical_crashes_still_fail() -> None:
    async def boom(r: Recorder, _h: Any) -> None:
        await r.step("explode", lambda: 1 / 0)

    result = await run_differential(
        [Scenario("boom", boom)],
        reference=lambda router: REF.harness(router, REF.provider, sequential_ids()),
        candidate=lambda router: REF.harness(router, REF.provider, sequential_ids()),
    )
    assert result.scenarios[0].same
    assert not result.passed
    assert "ZeroDivisionError" in result.scenarios[0].crashes[0]


def test_differences() -> None:
    assert differences({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}) == []
    assert differences({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}) == [
        "a[1].b: reference 2 != candidate 3"
    ]
    assert differences({"a": 1}, {"b": 1}) == ["a: only in reference", "b: only in candidate"]
    assert differences([1], [1, 2]) == [": 1 items in reference, 2 in candidate"]


# --- report.json ----------------------------------------------------------------------------


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    target = tmp_path / "razorpay" / "v1"
    target.mkdir(parents=True)
    for name in ("manifest.toml", "mappers.py"):
        shutil.copy(RAZORPAY / name, target / name)
    return target


@pytest.fixture
def fast_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    # G1's mypy and G5's pytest run as subprocesses; covered by `make gates`.
    monkeypatch.setattr(report_mod, "g1_types", lambda _d: report_mod.gate("pass", "stubbed"))
    monkeypatch.setattr(report_mod, "g5_contract", lambda _d: report_mod.gate("pass", "stubbed"))


@pytest.mark.usefixtures("fast_gates")
async def test_report_for_a_good_package(package_dir: Path) -> None:
    report = await report_mod.build_report(package_dir)
    assert report["promotable"]
    assert report["package"]["sha256"] == load_package(package_dir).sha256
    assert {n: g["status"] for n, g in report["gates"].items()} == {
        "G1": "pass",
        "G2": "pass",
        "G3": "not_run",
        "G4": "not_run",
        "G5": "pass",
        "G6": "not_run",
        "G7": "pass",
    }
    report_mod.write_report(package_dir, report)
    assert (
        report_mod.report_problem(load_package(package_dir), report_mod.read_report(package_dir))
        is None
    )


@pytest.mark.usefixtures("fast_gates")
async def test_report_blocks_a_divergent_package(package_dir: Path) -> None:
    mappers = package_dir / "mappers.py"
    mappers.write_text(
        mappers.read_text().replace("AttemptStatus.AUTHORIZED", "AttemptStatus.PENDING", 1)
    )
    report = await report_mod.build_report(package_dir)
    assert not report["promotable"]
    assert report["failed_gates"] == ["G7"]
    problem = report_mod.report_problem(load_package(package_dir), report)
    assert problem is not None
    assert "G7" in problem


async def test_report_when_package_does_not_load(package_dir: Path) -> None:
    (package_dir / "mappers.py").write_text("import os\n")
    report = await report_mod.build_report(package_dir, contract=False)
    assert report["package"] is None
    assert report["failed_gates"] == ["G1"]
    assert report["gates"]["G7"]["status"] == "not_run"


def test_report_problem_detects_missing_and_stale_reports(package_dir: Path) -> None:
    package = load_package(package_dir)
    assert "no report.json" in (report_mod.report_problem(package, None) or "")
    stale = {"package": {"sha256": "0" * 64}, "promotable": True}
    assert "different bytes" in (report_mod.report_problem(package, stale) or "")


async def test_no_reference_means_not_run(
    package_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report_mod, "REFERENCES", {})
    monkeypatch.setattr(report_mod, "g1_types", lambda _d: report_mod.gate("pass", "stubbed"))
    report = await report_mod.build_report(package_dir)
    assert report["gates"]["G5"]["status"] == "not_run"
    assert report["gates"]["G7"]["status"] == "not_run"
    assert not report["promotable"]


@pytest.mark.usefixtures("fast_gates")
def test_cli_writes_report_and_exits_by_verdict(
    package_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from payweave.verify.__main__ import main  # noqa: PLC0415

    assert main([str(package_dir)]) == 0
    assert "promotable" in capsys.readouterr().out
    assert report_mod.read_report(package_dir) is not None
    mappers = package_dir / "mappers.py"
    mappers.write_text(
        mappers.read_text().replace("AttemptStatus.AUTHORIZED", "AttemptStatus.PENDING", 1)
    )
    assert main([str(package_dir)]) == 1
    out = capsys.readouterr().out
    assert "FAIL G7" in out
    assert "manual_capture: step get authorized" in out
